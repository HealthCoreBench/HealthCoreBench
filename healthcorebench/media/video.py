"""Lazy, deterministic video frame sampling for multimodal requests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from healthcorebench.clients.errors import ClientError, ErrorType
from healthcorebench.utils.hashing import hash_file


@dataclass
class SampledVideo:
    frames: list
    duration_seconds: float | None
    total_frames: int | None
    sampled_indices: list[int]
    source_path: str
    media_hash: str
    strategy: str


def _uniform_indices(total_frames: int, max_frames: int) -> list[int]:
    if total_frames <= 0:
        return []
    count = min(total_frames, max_frames)
    if count == 1:
        return [0]
    return sorted({round(index * (total_frames - 1) / (count - 1)) for index in range(count)})


def sample_video(source, *, max_frames: int, strategy: str = "uniform") -> SampledVideo:
    """Decode only uniformly selected frames using PyAV.

    PyAV is optional because text/image-only installations should remain lightweight. A video
    task fails explicitly as unsupported media when the dependency is absent.
    """
    if strategy != "uniform":
        raise ClientError(
            ErrorType.UNSUPPORTED_MEDIA,
            f"Unsupported video_frame_sampling_strategy={strategy!r}; use 'uniform'.",
            retryable=False,
        )
    path = Path(source)
    if not path.is_file():
        raise ClientError(ErrorType.MEDIA_NOT_FOUND, f"Video not found: {path}", retryable=False)
    try:
        import av
    except ImportError as exc:
        raise ClientError(
            ErrorType.UNSUPPORTED_MEDIA,
            "Video input requires the optional 'av' dependency (install healthcorebench[video]).",
            retryable=False,
        ) from exc

    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            total = int(stream.frames or 0)
            if total <= 0:
                # Some containers omit frame count; decode once and retain bounded samples.
                decoded = [frame.to_image() for frame in container.decode(stream)]
                total = len(decoded)
                indices = _uniform_indices(total, max_frames)
                frames = [decoded[index] for index in indices]
            else:
                indices = _uniform_indices(total, max_frames)
                wanted = set(indices)
                frames = []
                for index, frame in enumerate(container.decode(stream)):
                    if index in wanted:
                        frames.append(frame.to_image())
                    if indices and index >= indices[-1]:
                        break
            duration = None
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = float(container.duration / av.time_base)
    except ClientError:
        raise
    except Exception as exc:
        raise ClientError(
            ErrorType.MEDIA_DECODE_ERROR,
            f"Failed to decode video {path}: {exc}",
            retryable=False,
        ) from exc
    if not frames:
        raise ClientError(
            ErrorType.MEDIA_DECODE_ERROR,
            f"Video contains no decodable frames: {path}",
            retryable=False,
        )
    return SampledVideo(
        frames=frames,
        duration_seconds=duration,
        total_frames=total,
        sampled_indices=indices,
        source_path=str(path),
        media_hash=hash_file(path),
        strategy=strategy,
    )
