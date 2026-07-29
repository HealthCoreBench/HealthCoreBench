"""Image metadata extraction for provenance records.

Captures the properties that actually affect model behaviour and token usage so they can be
tracked: dimensions, byte size, MIME type, content hash. Extraction never raises for a
missing optional field — unknown values are ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class ImageMetadata:
    """Provenance for a single image reference logged with a sample."""

    media_id: str
    source_path: str | None
    source_uri: str | None
    media_hash: str | None
    mime_type: str | None
    original_width: int | None
    original_height: int | None
    processed_width: int | None
    processed_height: int | None
    original_bytes: int | None
    processed_bytes: int | None
    image_detail_setting: str | None
    max_pixels: int | None
    processor_name: str | None
    processor_version: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def extract_image_metadata(image_obj) -> dict:
    """Extract width/height/mode from a PIL image, tolerating failures.

    Returns a dict with ``width``, ``height``, ``mode`` (any of which may be ``None``).
    """
    result = {"width": None, "height": None, "mode": None}
    try:
        size = getattr(image_obj, "size", None)
        if size and len(size) == 2:
            result["width"], result["height"] = int(size[0]), int(size[1])
        result["mode"] = getattr(image_obj, "mode", None)
    except Exception:
        pass
    return result
