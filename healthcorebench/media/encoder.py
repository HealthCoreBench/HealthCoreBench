"""Encode images into OpenAI-compatible data URIs, capturing full provenance.

Supports local file paths, PIL images, pre-existing base64 / data URIs, and (optionally)
HTTP(S) URLs. Every encode returns both the payload to send and an ``ImageMetadata``
describing the original and processed media, so logs can record exactly what was sent
without embedding the full base64.

Preprocessing performed here is deliberately explicit and versioned (``PROCESSOR_VERSION``)
because any change to it affects model output and token usage:

* EXIF orientation is applied (``ImageOps.exif_transpose``).
* Palette / RGBA / grayscale images are converted to RGB (alpha flattened onto white).
* Images may be downscaled so the encoded bytes fit ``max_size_mb`` and pixels fit
  ``max_pixels``. The algorithm is deterministic.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from healthcorebench.clients.errors import ClientError, ErrorType
from healthcorebench.media.metadata import ImageMetadata
from healthcorebench.utils.hashing import hash_bytes

PROCESSOR_NAME = "healthcorebench.media.encoder"
PROCESSOR_VERSION = "1.0"

_DATA_URI_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", re.DOTALL)


@dataclass
class EncodedImage:
    """Result of encoding one image."""

    data_uri: str
    metadata: ImageMetadata


def _pil():
    try:
        from PIL import Image, ImageOps  # noqa
        return Image, ImageOps
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ClientError(
            ErrorType.UNSUPPORTED_MEDIA,
            f"Pillow is required for image encoding: {exc}",
            retryable=False,
        )


def _path_exists(value: str) -> bool:
    """Return whether a string is a usable local path without treating base64 as one."""
    try:
        return Path(value).exists()
    except OSError:
        # Very long base64 strings can exceed OS filename limits; they should proceed to the
        # base64 decoder rather than leaking an OS error from path handling.
        return False


def encode_image(
    source,
    *,
    media_id: str,
    image_format: str = "png",
    image_detail: str = "auto",
    max_pixels: int | None = None,
    max_size_mb: float | None = 5.0,
    allow_url: bool = False,
) -> EncodedImage:
    """Encode ``source`` into a data URI plus provenance metadata.

    ``source`` may be a local path (str/Path), a PIL image, a base64 string, a ``data:``
    URI, or an ``http(s)://`` URL (only when ``allow_url`` is true — then it is passed
    through by reference, not fetched).
    """
    # --- HTTP URL: pass through by reference (do not fetch server-side) ---
    if isinstance(source, str) and source.startswith(("http://", "https://")):
        if not allow_url:
            raise ClientError(
                ErrorType.UNSUPPORTED_MEDIA,
                "HTTP image URLs are disabled (set media.allow_image_urls to enable).",
                retryable=False,
            )
        meta = ImageMetadata(
            media_id=media_id, source_path=None, source_uri=source, media_hash=None,
            mime_type=None, original_width=None, original_height=None,
            processed_width=None, processed_height=None, original_bytes=None,
            processed_bytes=None, image_detail_setting=image_detail, max_pixels=max_pixels,
            processor_name=PROCESSOR_NAME, processor_version=PROCESSOR_VERSION,
        )
        return EncodedImage(data_uri=source, metadata=meta)

    Image, ImageOps = _pil()

    # --- load source into a PIL image + capture original bytes/dims ---
    source_path: str | None = None
    original_bytes: int | None = None
    if isinstance(source, str) and source.startswith("data:"):
        m = _DATA_URI_RE.fullmatch(source)
        if not m:
            raise ClientError(ErrorType.MEDIA_DECODE_ERROR, "Malformed data URI.", retryable=False)
        try:
            raw = base64.b64decode(m.group("data"), validate=True)
            img = Image.open(BytesIO(raw))
            img.load()
            original_bytes = len(raw)
        except Exception as exc:
            raise ClientError(
                ErrorType.MEDIA_DECODE_ERROR,
                f"Failed to decode data URI image: {exc}",
                retryable=False,
            )
    elif isinstance(source, Path) or (isinstance(source, str) and _path_exists(source)):
        source_path = str(source)
        p = Path(source_path)
        if not p.exists():
            raise ClientError(ErrorType.MEDIA_NOT_FOUND, f"Image not found: {source_path}", retryable=False)
        try:
            original_bytes = p.stat().st_size
            img = Image.open(p)
            img.load()
        except Exception as exc:
            raise ClientError(ErrorType.MEDIA_DECODE_ERROR, f"Failed to decode {source_path}: {exc}", retryable=False)
    elif isinstance(source, str):
        # A string that is not an existing path can still be a bare base64 image. Decode
        # strictly so arbitrary filenames are never silently accepted as image bytes.
        try:
            raw = base64.b64decode(source, validate=True)
            img = Image.open(BytesIO(raw))
            img.load()
            original_bytes = len(raw)
        except Exception as exc:
            raise ClientError(
                ErrorType.MEDIA_NOT_FOUND,
                "Image path does not exist and source is not valid base64 image data.",
                retryable=False,
            ) from exc
    elif hasattr(source, "size") and hasattr(source, "mode"):
        img = source  # already a PIL image
    else:
        raise ClientError(ErrorType.UNSUPPORTED_MEDIA, f"Unsupported image source type: {type(source)}", retryable=False)

    original_width, original_height = img.size

    # --- deterministic preprocessing ---
    try:
        img = ImageOps.exif_transpose(img)  # honour EXIF orientation
    except Exception:
        pass
    if img.mode not in ("RGB",):
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img).convert("RGB")
        else:
            img = img.convert("RGB")

    if max_pixels is not None and img.size[0] * img.size[1] > max_pixels:
        img = _downscale_to_pixels(img, max_pixels, Image)

    fmt = image_format.upper()
    if fmt == "JPG":
        fmt = "JPEG"
    encoded = _encode_bytes(img, fmt)
    if max_size_mb is not None:
        img, encoded = _shrink_to_size(img, fmt, max_size_mb, Image, encoded)

    mime_type = f"image/{'jpeg' if fmt == 'JPEG' else fmt.lower()}"
    data_uri = f"data:{mime_type};base64,{base64.b64encode(encoded).decode('ascii')}"

    meta = ImageMetadata(
        media_id=media_id,
        source_path=source_path,
        source_uri=None,
        media_hash=hash_bytes(encoded),
        mime_type=mime_type,
        original_width=original_width,
        original_height=original_height,
        processed_width=img.size[0],
        processed_height=img.size[1],
        original_bytes=original_bytes,
        processed_bytes=len(encoded),
        image_detail_setting=image_detail,
        max_pixels=max_pixels,
        processor_name=PROCESSOR_NAME,
        processor_version=PROCESSOR_VERSION,
    )
    return EncodedImage(data_uri=data_uri, metadata=meta)


def _encode_bytes(img, fmt: str) -> bytes:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _downscale_to_pixels(img, max_pixels: int, Image):
    w, h = img.size
    scale = (max_pixels / float(w * h)) ** 0.5
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.LANCZOS)


def _shrink_to_size(img, fmt: str, max_size_mb: float, Image, encoded: bytes):
    """Iteratively downscale until the encoded payload fits ``max_size_mb``.

    Returns ``(image, encoded_bytes)`` reflecting the final (possibly downscaled) image so
    the caller records the true processed dimensions.
    """
    limit = max_size_mb * 1024 * 1024
    current = img
    data = encoded
    guard = 0
    while len(data) > limit and guard < 20:
        w, h = current.size
        current = current.resize((max(1, int(w * 0.9)), max(1, int(h * 0.9))), Image.LANCZOS)
        data = _encode_bytes(current, fmt)
        guard += 1
    if len(data) > limit:
        raise ClientError(
            ErrorType.MEDIA_TOO_LARGE,
            f"Image could not be reduced below the configured {max_size_mb} MiB limit.",
            retryable=False,
        )
    return current, data


def _metadata_from_bytes(raw: bytes, *, media_id, source_path, mime_type, image_detail, max_pixels) -> ImageMetadata:
    width = height = None
    try:
        from PIL import Image
        im = Image.open(BytesIO(raw))
        width, height = im.size
    except Exception:
        pass
    return ImageMetadata(
        media_id=media_id, source_path=source_path, source_uri=None,
        media_hash=hash_bytes(raw), mime_type=mime_type,
        original_width=width, original_height=height,
        processed_width=width, processed_height=height,
        original_bytes=len(raw), processed_bytes=len(raw),
        image_detail_setting=image_detail, max_pixels=max_pixels,
        processor_name=PROCESSOR_NAME, processor_version=PROCESSOR_VERSION,
    )
