# Image Editing

Image editing is a Phase 2 feature. Supported operations target:

- Image-to-image
- Inpainting
- Local modifications

The implementation keeps editing abstractions independent of the exact image-generation backend and may raise `UnsupportedOperation` when a requested operation is not supported by the configured lightweight model.
