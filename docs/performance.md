# Performance

LocalGPT is optimized for CPU-only operation on modest hardware.

## Expectations

- Text chat depends on llama-server performance and the selected quantized model.
- Image generation on CPU is intentionally lightweight and can be slow.

## Optimization priorities

- Low memory consumption
- Streaming
- Lazy loading
- Small default image sizes and low step counts
- Disk-backed assets and thumbnails

## Monitoring

Use `GET /system/status` and `GET /api/models` to inspect memory and CPU usage.
