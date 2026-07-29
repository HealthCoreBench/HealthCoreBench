"""Regression coverage for image source decoding and preprocessing limits."""

from __future__ import annotations

import base64
from io import BytesIO

import pytest
from PIL import Image

from healthcorebench.clients.errors import ClientError, ErrorType
from healthcorebench.media.encoder import encode_image


def _png_base64(size: tuple[int, int] = (24, 20)) -> str:
    image = Image.new("RGB", size, color=(12, 34, 56))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_bare_base64_image_is_decoded_and_preprocessed():
    encoded = encode_image(
        _png_base64(), media_id="image-1", image_format="png", max_pixels=100,
    )

    assert encoded.data_uri.startswith("data:image/png;base64,")
    assert encoded.metadata.source_path is None
    assert encoded.metadata.original_width == 24
    assert encoded.metadata.original_height == 20
    assert encoded.metadata.processed_width * encoded.metadata.processed_height <= 100


def test_data_uri_is_reencoded_and_respects_pixel_limit():
    data_uri = "data:image/png;base64," + _png_base64((30, 30))

    encoded = encode_image(
        data_uri, media_id="image-2", image_format="jpeg", max_pixels=100,
    )

    assert encoded.data_uri.startswith("data:image/jpeg;base64,")
    assert encoded.metadata.source_path is None
    assert encoded.metadata.original_bytes is not None
    assert encoded.metadata.processed_width * encoded.metadata.processed_height <= 100


def test_data_uri_rejects_invalid_base64():
    with pytest.raises(ClientError) as exc_info:
        encode_image("data:image/png;base64,not valid base64!", media_id="image-3")

    assert exc_info.value.error_type == ErrorType.MEDIA_DECODE_ERROR


def test_image_that_cannot_fit_size_limit_fails_explicitly():
    with pytest.raises(ClientError) as exc_info:
        encode_image(
            _png_base64((1, 1)), media_id="image-4", max_size_mb=0.000001,
        )

    assert exc_info.value.error_type == ErrorType.MEDIA_TOO_LARGE


def test_unknown_string_is_not_misreported_as_base64_decode_success():
    with pytest.raises(ClientError) as exc_info:
        encode_image("definitely-not-an-image-file", media_id="image-5")

    assert exc_info.value.error_type == ErrorType.MEDIA_NOT_FOUND
