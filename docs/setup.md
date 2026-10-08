# Setup

## Prerequisites

- Python 3.11+
- llama-server or llama.cpp built for CPU

## Install dependencies

```
python -m pip install -r requirements.txt
```

## Download models (first run)

Run the one-time model downloader:

```
python model_download.py
```

This downloads the SmolVLM2 2.2B GGUF + vision projector into `models/` and creates `.env` from `.env.example` with `VLM_MODEL` and `VLM_MODEL_PATH` pre-filled.

Options:

- `--model qwen3-vl-2b` — download the alternative model
- `--dry-run` — preview sizes without downloading
- `--skip-env` — download only; do not touch `.env`

Downloads are resumable if interrupted.

## Start llama-server

Start llama-server with the downloaded model before running LocalGPT. LocalGPT will attempt to contact it at `LLAMA_SERVER_URL`.

Example:

```
llama-server -m models/smolvlm2-2.2b/SmolVLM2-2.2B-Instruct-Q8_0.gguf \\
    --mmproj models/smolvlm2-2.2b/mmproj-SmolVLM2-2.2B-Instruct-f16.gguf \\
    --port 8080
```

## Run

```
python run_localgpt.py
```

Then open `http://127.0.0.1:8000`.

## First run

The UI will show the chat interface. If the VLM is unavailable, the UI and API will surface a friendly error.
