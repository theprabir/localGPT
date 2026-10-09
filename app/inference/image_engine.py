"""Image engine abstraction (Phase 2).

AGENTS.md §18 requires an engine-independent interface:

    class ImageEngine:
        def generate(self, prompt, ...): ...
        def img2img(self, image_path, prompt, ...): ...
        def inpaint(self, image_path, mask_path, prompt, ...): ...

The concrete backend here is **stable-diffusion.cpp** (`sd-cli`), an external
C++ process — the same architecture LocalGPT already uses for the VLM via
llama-server. No Python ML framework (torch/OpenVINO) is loaded into the app.

This is an explicitly user-authorized deviation from the "FastSD CPU" line in
AGENTS.md: FastSD cannot load GGUF weights at all, while stable-diffusion.cpp
runs SD1.x GGUF/safetensors, SD-Turbo and LCM natively, so AGENTS.md's model
choice can still be honoured through the same runtime. Recorded in
CHANGELOG.md and docs/image-generation.md.

Never fake an operation: a request the configured model cannot honour must
raise :class:`UnsupportedOperation` (AGENTS.md §18).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import threading
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger("localgpt.inference.image_engine")

# Progress lines look like:   |#####             | 7/12 - 517.88s/it
_PROGRESS_RE = re.compile(r"(\d+)/(\d+)\s+-\s+[\d.]+s/it")


class ImageEngineError(RuntimeError):
    """A generation attempt failed. ``user_message`` is safe to show in the UI."""


class UnsupportedOperation(ImageEngineError):
    """The configured engine/model cannot perform the requested operation."""


class GenerationCancelled(ImageEngineError):
    """The job was cancelled by the user before it finished."""


class ImageEngine(ABC):
    """Backend-independent image engine interface (AGENTS.md §18)."""

    name = "abstract"

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        """Return ``(ready, reason_if_not)`` without starting heavy work."""

    @abstractmethod
    def capabilities(self) -> dict[str, bool]:
        """Which operations this engine+model combination supports."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        output_path: str,
        *,
        width: int = 512,
        height: int = 512,
        steps: int = 2,
        negative_prompt: str = "",
        seed: int = -1,
        cancel: threading.Event | None = None,
        on_progress=None,
    ) -> str:
        """Text-to-image. Returns the path actually written."""

    @abstractmethod
    def img2img(
        self,
        image_path: str,
        prompt: str,
        output_path: str,
        *,
        strength: float = 0.55,
        width: int = 512,
        height: int = 512,
        steps: int = 2,
        negative_prompt: str = "",
        seed: int = -1,
        cancel: threading.Event | None = None,
        on_progress=None,
    ) -> str:
        """Image-to-image. The original image is never modified in place."""

    def inpaint(
        self,
        image_path: str,
        mask_path: str,
        prompt: str,
        output_path: str,
        *,
        strength: float = 0.55,
        width: int = 512,
        height: int = 512,
        steps: int = 2,
        negative_prompt: str = "",
        seed: int = -1,
        cancel: threading.Event | None = None,
        on_progress=None,
    ) -> str:
        raise UnsupportedOperation(
            "Inpainting is not supported by the configured image engine."
        )


class UnavailableImageEngine(ImageEngine):
    """Used when no engine could be configured; fails loudly, never fakes output."""

    name = "unavailable"

    def __init__(self, reason: str = "The image engine is not configured."):
        self.reason = reason

    def available(self) -> tuple[bool, str]:
        return False, self.reason

    def capabilities(self) -> dict[str, bool]:
        return {"txt2img": False, "img2img": False, "inpaint": False}

    def _fail(self) -> None:
        raise ImageEngineError(self.reason)

    def generate(self, prompt, output_path, **kwargs) -> str:
        self._fail()

    def img2img(self, image_path, prompt, output_path, **kwargs) -> str:
        self._fail()


class SDCppEngine(ImageEngine):
    """stable-diffusion.cpp (``sd-cli``) backend.

    Runs one subprocess per job. That matches AGENTS.md §15/§17: the image
    engine lives outside the FastAPI process, uses ~1.5-2 GB only while a job
    runs, and releases everything when the process exits.
    """

    name = "sd_cpp"

    DEFAULT_MODEL_CANDIDATES = (
        "models/stable-diffusion-v1-5/stable-diffusion-v1-5-pruned-emaonly-Q4_0.gguf",
    )

    def __init__(self, settings=None):
        from app.config import get_settings

        self.settings = settings or get_settings()
        self.cli_path = self._resolve_cli()
        self.model_path = self._resolve_model()
        self._proc: subprocess.Popen | None = None
        self._proc_lock = threading.Lock()

    # -- discovery ---------------------------------------------------------

    def _resolve_cli(self) -> str:
        configured = (getattr(self.settings, "sd_cli_path", "") or "").strip()
        if configured:
            return configured

        root = Path(__file__).resolve().parents[2]
        names = ("sd-cli.exe", "sd-cli") if os.name == "nt" else ("sd-cli",)
        for name in names:
            candidate = root / "tools" / "sd.cpp" / name
            if candidate.exists():
                return str(candidate)

        found = shutil.which("sd-cli")
        if found:
            return found
        return ""  # reported through available()

    def _resolve_model(self) -> str:
        configured = (getattr(self.settings, "image_model_path", "") or "").strip()
        if configured:
            return configured

        root = Path(__file__).resolve().parents[2]
        for rel in self.DEFAULT_MODEL_CANDIDATES:
            candidate = root / rel
            if candidate.exists():
                return str(candidate)

        # Last resort: any SD GGUF the operator dropped into models/.
        models_dir = root / "models"
        if models_dir.is_dir():
            for pattern in ("**/stable-diffusion*.gguf", "**/*sd*.gguf", "**/*.gguf"):
                matches = sorted(models_dir.glob(pattern))
                if matches:
                    return str(matches[0])
        return ""

    def available(self) -> tuple[bool, str]:
        if not self.cli_path:
            return (
                False,
                "The image engine binary was not found. Download stable-diffusion.cpp "
                "into tools/sd.cpp (see docs/setup.md) and restart LocalGPT.",
            )
        if not Path(self.cli_path).exists():
            return False, f"The image engine binary is missing at {self.cli_path}."
        if not self.model_path:
            return (
                False,
                "No image model was found. Place a Stable Diffusion GGUF under "
                "models/ or set LOCALGPT_IMAGE_MODEL_PATH (see docs/models.md).",
            )
        if not Path(self.model_path).exists():
            return False, f"The image model was not found at {self.model_path}."
        return True, ""

    def capabilities(self) -> dict[str, bool]:
        ready, _ = self.available()
        return {"txt2img": ready, "img2img": ready, "inpaint": ready}

    # -- invocation --------------------------------------------------------

    def _build_command(
        self,
        output_path: str,
        *,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        negative_prompt: str,
        seed: int,
        init_image: str | None = None,
        mask_image: str | None = None,
        strength: float | None = None,
    ) -> list[str]:
        cmd = [
            self.cli_path,
            "-m", self.model_path,
            "-p", prompt,
            "-n", negative_prompt or "",
            "-W", str(int(width)),
            "-H", str(int(height)),
            "--steps", str(int(steps)),
            "--seed", str(int(seed)),
            "-o", output_path,
        ]
        if init_image:
            cmd += ["-i", str(init_image)]
        if mask_image:
            cmd += ["--mask", str(mask_image)]
        if strength is not None:
            cmd += ["--strength", f"{float(strength):.3f}"]
        threads = int(getattr(self.settings, "image_threads", 0) or 0)
        if threads > 0:
            cmd += ["-t", str(threads)]
        return cmd

    def _run(
        self,
        cmd: list[str],
        output_path: str,
        cancel: threading.Event | None,
        on_progress=None,
    ) -> str:
        ready, reason = self.available()
        if not ready:
            raise ImageEngineError(reason)

        timeout = float(getattr(self.settings, "image_job_timeout_seconds", 3600))
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        logger.info("image engine start: %s", " ".join(_redact(cmd)))
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            logger.exception("image engine failed to start")
            raise ImageEngineError(
                "The image engine could not start. Check the model configuration "
                "and system resources."
            ) from exc

        with self._proc_lock:
            self._proc = proc

        lines: list[str] = []
        reader = threading.Thread(
            target=_pump_output,
            args=(proc, lines, on_progress),
            daemon=True,
        )
        reader.start()

        deadline_hits = 0
        try:
            while True:
                if cancel is not None and cancel.is_set():
                    _terminate(proc)
                    raise GenerationCancelled("Image generation was cancelled.")
                try:
                    proc.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    deadline_hits += 1
                    if deadline_hits * 0.5 > timeout:
                        _terminate(proc)
                        raise ImageEngineError(
                            "Image generation timed out. Try a smaller image size "
                            "or fewer steps."
                        ) from None
        finally:
            with self._proc_lock:
                self._proc = None

        output = "".join(lines[-80:])
        if proc.returncode != 0:
            logger.error("image engine exited %s\n%s", proc.returncode, output)
            raise ImageEngineError(
                "Image generation failed. The image engine could not complete the "
                "job. Check the model configuration and system resources."
            )
        if not out.exists():
            logger.error("image engine reported success but wrote no file")
            raise ImageEngineError("Image generation produced no output file.")
        logger.info("image engine finished: %s", out)
        return str(out)

    def cancel(self) -> None:
        """Terminate the in-flight subprocess, if any (job cancellation)."""
        with self._proc_lock:
            proc = self._proc
        if proc is not None:
            _terminate(proc)

    # -- operations --------------------------------------------------------

    def generate(
        self,
        prompt: str,
        output_path: str,
        *,
        width: int = 512,
        height: int = 512,
        steps: int = 2,
        negative_prompt: str = "",
        seed: int = -1,
        cancel: threading.Event | None = None,
        on_progress=None,
    ) -> str:
        cmd = self._build_command(
            output_path,
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            negative_prompt=negative_prompt,
            seed=seed,
        )
        return self._run(cmd, output_path, cancel, on_progress)

    def img2img(
        self,
        image_path: str,
        prompt: str,
        output_path: str,
        *,
        strength: float = 0.55,
        width: int = 512,
        height: int = 512,
        steps: int = 2,
        negative_prompt: str = "",
        seed: int = -1,
        cancel: threading.Event | None = None,
        on_progress=None,
    ) -> str:
        if not Path(image_path).exists():
            raise ImageEngineError("The source image for editing was not found.")
        cmd = self._build_command(
            output_path,
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            negative_prompt=negative_prompt,
            seed=seed,
            init_image=image_path,
            strength=strength,
        )
        return self._run(cmd, output_path, cancel, on_progress)

    def inpaint(
        self,
        image_path: str,
        mask_path: str,
        prompt: str,
        output_path: str,
        *,
        strength: float = 0.55,
        width: int = 512,
        height: int = 512,
        steps: int = 2,
        negative_prompt: str = "",
        seed: int = -1,
        cancel: threading.Event | None = None,
        on_progress=None,
    ) -> str:
        if not Path(image_path).exists():
            raise ImageEngineError("The source image for editing was not found.")
        if not Path(mask_path).exists():
            raise ImageEngineError("The mask image for inpainting was not found.")
        cmd = self._build_command(
            output_path,
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            negative_prompt=negative_prompt,
            seed=seed,
            init_image=image_path,
            mask_image=mask_path,
            strength=strength,
        )
        return self._run(cmd, output_path, cancel, on_progress)


def _pump_output(proc: subprocess.Popen, lines: list[str], on_progress=None) -> None:
    """Read engine output, keep the tail for diagnostics, report step progress.

    sd-cli writes carriage-return separated progress bars, so split on both
    ``\\r`` and ``\\n``. Progress parsing is best effort: on failure the UI just
    shows an indeterminate state.
    """
    if proc.stdout is None:
        return
    buffer = ""
    try:
        for chunk in iter(lambda: proc.stdout.read(256), ""):
            if not chunk:
                break
            buffer += chunk
            parts = re.split(r"[\r\n]+", buffer)
            buffer = parts.pop()
            for part in parts:
                if part.strip():
                    lines.append(part + "\n")
                    match = _PROGRESS_RE.search(part)
                    if match and on_progress is not None:
                        try:
                            on_progress(int(match.group(1)), int(match.group(2)))
                        except Exception:  # progress must never break a job
                            logger.debug("progress callback failed", exc_info=True)
    except Exception:
        logger.debug("output pump stopped early", exc_info=True)
    finally:
        if buffer.strip():
            lines.append(buffer + "\n")


def _terminate(proc: subprocess.Popen) -> None:
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception:
        logger.debug("failed to terminate image engine process", exc_info=True)


def _redact(cmd: list[str]) -> list[str]:
    """Log commands without dumping multi-thousand-char prompts."""
    out: list[str] = []
    hide_next = False
    for part in cmd:
        if hide_next:
            out.append("<text>")
            hide_next = False
            continue
        if part in ("-p", "-n"):
            out.append(part)
            hide_next = True
            continue
        out.append(part)
    return out


def build_image_engine(settings=None) -> ImageEngine:
    """Factory honouring ``LOCALGPT_IMAGE_ENGINE``; never raises."""
    from app.config import get_settings

    settings = settings or get_settings()
    engine_name = (getattr(settings, "image_engine", "") or "sd_cpp").strip().lower()
    if engine_name in ("none", "off", "disabled", ""):
        return UnavailableImageEngine("Image generation is disabled in settings.")
    if engine_name != "sd_cpp":
        logger.warning("Unknown image engine %r; falling back to sd_cpp", engine_name)
    engine = SDCppEngine(settings=settings)
    ready, reason = engine.available()
    if not ready:
        logger.warning("Image engine not ready: %s", reason)
        return UnavailableImageEngine(reason)
    return engine
