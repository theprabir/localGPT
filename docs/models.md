# Models

## Multimodal

- Primary: SmolVLM2 2.2B (ggml-org/SmolVLM2-2.2B-Instruct-GGUF)
- Alternative: Qwen3-VL 2B (ggml-org/Qwen3-VL-2B-Instruct-GGUF)

Selection is controlled by `VLM_MODEL`. Both are downloaded by `python model_download.py` (see docs/setup.md).

Use quantized GGUF-compatible variants where available. LocalGPT expects llama-server to be running with the selected model and its vision projector (mmproj) file.

## Image generation

The image model target is SD Turbo / LCM-style lightweight Stable Diffusion. Image generation is implemented in Phase 2.

## Runtime

llama-server provides the OpenAI-compatible API used by LocalGPT for VLM inference. LocalGPT does not load the VLM into the FastAPI process.
