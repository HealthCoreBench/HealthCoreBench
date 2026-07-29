"""Media handling: encoding to OpenAI-compatible data URIs and provenance metadata.

The framework records exactly what media was sent to the model and how it was
preprocessed (dimensions, bytes, hash, detail setting), but never stores the full base64
in per-sample result logs by default — only stable references.
"""

from healthcorebench.media.encoder import (
    EncodedImage,
    encode_image,
    PROCESSOR_NAME,
    PROCESSOR_VERSION,
)
from healthcorebench.media.metadata import ImageMetadata, extract_image_metadata

__all__ = [
    "EncodedImage",
    "encode_image",
    "PROCESSOR_NAME",
    "PROCESSOR_VERSION",
    "ImageMetadata",
    "extract_image_metadata",
]
