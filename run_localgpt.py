#!/usr/bin/env python
"""
LocalGPT launcher.

Usage:
    python run_localgpt.py
    python run_localgpt.py --help

For production use, start llama-server first and point LOCALGPT_LLAMA_SERVER_URL at it.
"""
from app.main import run

if __name__ == "__main__":
    run()
