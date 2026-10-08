# Troubleshooting

## VLM unavailable

- Ensure llama-server is running.
- Verify `LLAMA_SERVER_URL` is correct.
- Confirm the model loaded by llama-server matches `VLM_MODEL`.

## Upload errors

- Check that the file is a supported image type.
- Verify upload size limits.

## Performance

- CPU-only image generation can be slow. Do not expect instant results.
- Reduce context size and image dimensions for weaker hardware.
