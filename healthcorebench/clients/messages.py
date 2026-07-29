"""Build OpenAI wire messages from adapter-produced logical messages.

Adapters produce *logical* messages using a small, media-agnostic vocabulary so they never
touch encoding or the client:

    {"role": "system", "content": "..."}
    {"role": "user", "content": "..."}                      # plain text
    {"role": "user", "content": [                           # ordered mixed content
        {"type": "text", "text": "..."},
        {"type": "image", "source": <path|PIL|base64|url>, "media_id": "img_0"},
    ]}

This module converts those into:

* ``wire_messages`` — the exact ``messages`` payload sent to the API (images as data URIs);
* ``logged_messages`` — the same structure but with media replaced by ``image_ref``
  entries (media_id + hash), safe and compact to store in results.jsonl;
* ``image_infos`` — provenance for every image, in original order.

Image ordering relative to text is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass

from healthcorebench.clients.errors import ClientError, ErrorType
from healthcorebench.media.encoder import encode_image
from healthcorebench.schemas.sample import ImageInfo


@dataclass
class BuiltMessages:
    wire_messages: list[dict]
    logged_messages: list[dict]
    image_infos: list[ImageInfo]
    video_infos: list[dict]


def build_messages(
    logical_messages: list[dict],
    *,
    image_detail: str = "auto",
    image_format: str = "png",
    max_pixels: int | None = None,
    max_image_size_mb: float | None = 5.0,
    max_images: int | None = None,
    allow_image_urls: bool = False,
    max_video_frames: int = 32,
    video_frame_sampling_strategy: str = "uniform",
) -> BuiltMessages:
    """Convert logical messages to wire + logged messages plus image provenance."""
    wire: list[dict] = []
    logged: list[dict] = []
    infos: list[ImageInfo] = []
    video_infos: list[dict] = []
    image_count = 0

    for msg in logical_messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        if isinstance(content, str):
            wire.append({"role": role, "content": content})
            logged.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            raise ClientError(
                ErrorType.PROMPT_BUILD_ERROR,
                f"Unsupported message content type: {type(content)}",
                retryable=False,
            )

        wire_parts: list[dict] = []
        logged_parts: list[dict] = []
        for part in content:
            ptype = part.get("type")
            if ptype == "text":
                text = part.get("text", "")
                wire_parts.append({"type": "text", "text": text})
                logged_parts.append({"type": "text", "text": text})
            elif ptype in ("image", "image_url"):
                if max_images is not None and image_count >= max_images:
                    raise ClientError(
                        ErrorType.UNSUPPORTED_MEDIA,
                        f"Sample exceeds max_images={max_images}.",
                        retryable=False,
                    )
                source = part.get("source") if "source" in part else part.get("image_url", {}).get("url")
                media_id = part.get("media_id") or f"img_{image_count}"
                enc = encode_image(
                    source,
                    media_id=media_id,
                    image_detail=image_detail,
                    image_format=image_format,
                    max_pixels=max_pixels,
                    max_size_mb=max_image_size_mb,
                    allow_url=allow_image_urls,
                )
                wire_parts.append({
                    "type": "image_url",
                    "image_url": {"url": enc.data_uri, "detail": image_detail},
                })
                info = ImageInfo(**enc.metadata.to_dict())
                infos.append(info)
                logged_parts.append({
                    "type": "image_ref",
                    "media_id": media_id,
                    "media_hash": info.media_hash,
                })
                image_count += 1
            elif ptype == "video":
                from healthcorebench.media.video import sample_video

                source = part.get("source")
                media_id = part.get("media_id") or f"video_{len(video_infos)}"
                remaining = None if max_images is None else max_images - image_count
                if remaining is not None and remaining <= 0:
                    raise ClientError(
                        ErrorType.UNSUPPORTED_MEDIA,
                        f"No image slots remain for video {media_id}; max_images={max_images}.",
                        retryable=False,
                    )
                sampled = sample_video(
                    source,
                    max_frames=(max_video_frames if remaining is None
                                else min(max_video_frames, remaining)),
                    strategy=video_frame_sampling_strategy,
                )
                frame_refs = []
                for frame_number, (source_index, frame) in enumerate(
                    zip(sampled.sampled_indices, sampled.frames)
                ):
                    if max_images is not None and image_count >= max_images:
                        raise ClientError(
                            ErrorType.UNSUPPORTED_MEDIA,
                            f"Sample exceeds max_images={max_images} after video frame sampling.",
                            retryable=False,
                        )
                    frame_id = f"{media_id}_frame_{frame_number}"
                    enc = encode_image(
                        frame,
                        media_id=frame_id,
                        image_detail=image_detail,
                        image_format=image_format,
                        max_pixels=max_pixels,
                        max_size_mb=max_image_size_mb,
                        allow_url=False,
                    )
                    wire_parts.append({
                        "type": "image_url",
                        "image_url": {"url": enc.data_uri, "detail": image_detail},
                    })
                    info = ImageInfo(**enc.metadata.to_dict())
                    infos.append(info)
                    frame_refs.append({
                        "media_id": frame_id,
                        "source_frame_index": source_index,
                        "media_hash": info.media_hash,
                    })
                    image_count += 1
                video_info = {
                    "media_id": media_id,
                    "media_hash": sampled.media_hash,
                    "duration_seconds": sampled.duration_seconds,
                    "total_frames": sampled.total_frames,
                    "sampled_frame_indices": sampled.sampled_indices,
                    "frame_sampling_strategy": sampled.strategy,
                    "frames": frame_refs,
                }
                video_infos.append(video_info)
                logged_parts.append({
                    "type": "video_ref",
                    **video_info,
                })
            else:
                raise ClientError(
                    ErrorType.PROMPT_BUILD_ERROR,
                    f"Unknown content part type: {ptype}",
                    retryable=False,
                )

        wire.append({"role": role, "content": wire_parts})
        logged.append({"role": role, "content": logged_parts})

    return BuiltMessages(
        wire_messages=wire,
        logged_messages=logged,
        image_infos=infos,
        video_infos=video_infos,
    )
