#!/usr/bin/env python
"""
LocalGPT one-time model downloader.

Run this once before starting LocalGPT:

    python model_download.py            # downloads the default model (smolvlm2-2.2b)
    python model_download.py --model qwen3-vl-2b
    python model_download.py --dry-run  # show what would be downloaded

What it does:
1. Downloads the selected multimodal model GGUF + vision projector (mmproj)
   from Hugging Face into models/<model-name>/.
2. Copies .env.example to .env and pre-fills VLM_MODEL and VLM_MODEL_PATH
   with the downloaded model details.

The download is resumable: if interrupted, run the script again and it
continues from where it stopped. Files that already exist with the correct
size are skipped.

Only the Python standard library is used (urllib), so no extra dependencies
are required before `pip install -r requirements.txt`.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
ENV_FILE = PROJECT_ROOT / ".env"

# Base URL for direct file downloads from Hugging Face.
HF_RESOLVE_URL = "https://huggingface.co/{repo}/resolve/main/{filename}"

# Model registry: LocalGPT model key -> Hugging Face repo + files.
# Quantization choices favor the target hardware (i3 3rd gen, 8-10 GB RAM):
# Q8_0 keeps near-full quality at roughly half the size of f16.
MODELS: dict[str, dict] = {
    "smolvlm2-2.2b": {
        "repo": "ggml-org/SmolVLM2-2.2B-Instruct-GGUF",
        "friendly_name": "SmolVLM2 2.2B Instruct",
        "files": [
            "SmolVLM2-2.2B-Instruct-Q8_0.gguf",
            "mmproj-SmolVLM2-2.2B-Instruct-f16.gguf",
        ],
    },
    "qwen3-vl-2b": {
        "repo": "ggml-org/Qwen3-VL-2B-Instruct-GGUF",
        "friendly_name": "Qwen3-VL 2B Instruct",
        "files": [
            "Qwen3-VL-2B-Instruct-Q8_0.gguf",
            "mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf",
        ],
    },
}

CHUNK_SIZE = 1024 * 1024  # 1 MiB


def human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"


def get_remote_size(url: str) -> int | None:
    """Ask the server for the file size via a HEAD-style ranged GET."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length else None
    except (urllib.error.URLError, ValueError):
        return None


def download_file(url: str, dest: Path, expected_size: int | None = None) -> None:
    """Download url to dest with resume support and progress output."""
    if dest.exists() and expected_size is not None and dest.stat().st_size == expected_size:
        print(f"  already downloaded, skipping: {dest.name} ({human_size(expected_size)})")
        return

    part_file = dest.with_suffix(dest.suffix + ".part")
    headers = {}
    existing = 0
    if part_file.exists():
        existing = part_file.stat().st_size
        if expected_size is not None and existing >= expected_size:
            # Corrupt/stale partial; start over.
            part_file.unlink()
            existing = 0
        elif existing > 0:
            headers["Range"] = f"bytes={existing}-"
            print(f"  resuming from {human_size(existing)}...")

    req = urllib.request.Request(url, headers=headers)
    mode = "ab" if existing > 0 else "wb"

    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(part_file, mode) as f:
            total = resp.headers.get("Content-Length")
            total = int(total) + existing if total else expected_size
            downloaded = existing
            last_print = 0.0
            while True:
                chunk = resp.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                # Print progress at most twice per second.
                if now - last_print > 0.5:
                    last_print = now
                    if total:
                        pct = downloaded / total * 100
                        speed = (downloaded - existing) / max(now - started, 0.001)
                        print(
                            f"\r  {pct:5.1f}%  {human_size(downloaded)} / {human_size(total)}"
                            f"  ({human_size(speed)}/s)   ",
                            end="",
                            flush=True,
                        )
                    else:
                        print(f"\r  {human_size(downloaded)}   ", end="", flush=True)
        print()  # finish the progress line
    except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
        print(f"\n  download interrupted: {exc}")
        print(f"  partial file kept at {part_file.name}; run this script again to resume.")
        raise SystemExit(1) from exc

    if expected_size is not None and part_file.stat().st_size != expected_size:
        print(f"  size mismatch for {dest.name}; removing partial file. Try again.")
        part_file.unlink()
        raise SystemExit(1)

    part_file.replace(dest)
    print(f"  saved: {dest}")


def download_model(model_key: str, dry_run: bool = False) -> dict:
    """Download all files for a model. Returns info dict for .env generation."""
    info = MODELS[model_key]
    repo = info["repo"]
    model_dir = MODELS_DIR / model_key

    print(f"\nModel: {info['friendly_name']} ({model_key})")
    print(f"Source: https://huggingface.co/{repo}")
    print(f"Destination: {model_dir}")

    if dry_run:
        for filename in info["files"]:
            url = HF_RESOLVE_URL.format(repo=repo, filename=filename)
            size = get_remote_size(url)
            print(f"  would download: {filename}" + (f" ({human_size(size)})" if size else ""))
        return info

    model_dir.mkdir(parents=True, exist_ok=True)

    for filename in info["files"]:
        url = HF_RESOLVE_URL.format(repo=repo, filename=filename)
        print(f"\nDownloading {filename}")
        size = get_remote_size(url)
        download_file(url, model_dir / filename, expected_size=size)

    return info


def create_env_file(model_info: dict, model_key: str, llama_server_url: str) -> None:
    """Create .env from .env.example, pre-filling VLM_MODEL and VLM_MODEL_PATH."""
    if not ENV_EXAMPLE.exists():
        print(f"ERROR: {ENV_EXAMPLE.name} not found next to model_download.py")
        raise SystemExit(1)

    env_lines: list[str] = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    main_gguf = next(f for f in model_info["files"] if not f.startswith("mmproj-"))

    updates = {
        "VLM_MODEL": model_key,
        "VLM_MODEL_PATH": str(MODELS_DIR / model_key / main_gguf),
        "LLAMA_SERVER_URL": llama_server_url,
    }

    seen: set[str] = set()
    out: list[str] = []
    for line in env_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                out.append(f"{key}={updates[key]}")
                seen.add(key)
                continue
        out.append(line)

    # Append any keys that were not present in the example file.
    for key, value in updates.items():
        if key not in seen:
            out.append(f"{key}={value}")

    if ENV_FILE.exists():
        backup = ENV_FILE.with_suffix(".env.bak")
        shutil.copy2(ENV_FILE, backup)
        print(f"\nExisting .env backed up to {backup.name}")

    ENV_FILE.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Created {ENV_FILE.name} with:")
    for key in updates:
        print(f"  {key}={updates[key]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download LocalGPT models and set up .env (first-run setup)."
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODELS),
        default="smolvlm2-2.2b",
        help="Model to download (default: smolvlm2-2.2b)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading",
    )
    parser.add_argument(
        "--llama-server-url",
        default="http://127.0.0.1:8080",
        help="LLAMA_SERVER_URL to write into .env (default: http://127.0.0.1:8080)",
    )
    parser.add_argument(
        "--skip-env",
        action="store_true",
        help="Only download model files; do not create/update .env",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("LocalGPT model downloader")
    print("=" * 60)
    print(
        "\nNote: models are multi-GB files. On slower connections this can "
        "take a while.\nProgress is resumable if the download is interrupted."
    )

    info = download_model(args.model, dry_run=args.dry_run)

    if not args.dry_run and not args.skip_env:
        create_env_file(info, args.model, args.llama_server_url)

        main_gguf = next(f for f in info["files"] if not f.startswith("mmproj-"))
        mmproj = next((f for f in info["files"] if f.startswith("mmproj-")), None)

        print("\n" + "=" * 60)
        print("Setup complete! Next steps:")
        print("=" * 60)
        print()
        print("1. Start llama-server with the downloaded model:")
        print()
        if mmproj:
            print(f'   llama-server -m "models/{args.model}/{main_gguf}" \\')
            print(f'       --mmproj "models/{args.model}/{mmproj}" --port 8080')
        else:
            print(f'   llama-server -m "models/{args.model}/{main_gguf}" --port 8080')
        print()
        print("   (The model path is also written in .env as VLM_MODEL_PATH.)")
        print()
        print("2. Run LocalGPT:")
        print()
        print("   python run_localgpt.py")
        print()
        print("3. Open http://127.0.0.1:8000 in your browser.")


if __name__ == "__main__":
    main()
