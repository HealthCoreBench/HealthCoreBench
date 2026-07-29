"""Normalized evaluation sample schema (one line per selected sample in samples.jsonl).

``sample_id`` is the cross-run alignment key (stable, independent of file order and prompt
template). ``sample_index`` only encodes deterministic within-split ordering. The three
hashes are kept distinct: ``source_record_hash`` (raw record), ``input_hash`` (normalized
model input), ``reference_hash`` (normalized reference answer).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ConfigDict

from healthcorebench.version import SCHEMA_VERSION


class ImageInfo(BaseModel):
    model_config = ConfigDict(extra="allow")
    media_id: str
    source_path: str | None = None
    source_uri: str | None = None
    media_hash: str | None = None
    mime_type: str | None = None
    original_width: int | None = None
    original_height: int | None = None
    processed_width: int | None = None
    processed_height: int | None = None
    original_bytes: int | None = None
    processed_bytes: int | None = None
    image_detail_setting: str | None = None
    max_pixels: int | None = None
    processor_name: str | None = None
    processor_version: str | None = None


class MediaInfo(BaseModel):
    model_config = ConfigDict(extra="allow")
    num_images: int = 0
    images: list[ImageInfo] = Field(default_factory=list)
    video_duration_seconds: float | None = None
    num_video_frames: int | None = None
    frame_sampling_strategy: str | None = None
    audio_duration_seconds: float | None = None


class EvaluationSample(BaseModel):
    """Framework-standard sample. Persisted to ``samples.jsonl``."""

    model_config = ConfigDict(extra="allow")

    schema_version: str = SCHEMA_VERSION
    sample_id: str
    source_sample_id: str | None = None
    sample_index: int

    benchmark_name: str
    benchmark_version: str | None = None
    benchmark_split: str
    source_benchmark_entry: str | None = None
    source_file: str | None = None
    source_record_index: int | None = None
    source_record_hash: str | None = None

    input_hash: str | None = None
    reference_hash: str | None = None
    media_hashes: list[str] = Field(default_factory=list)

    input_type: str = "text"          # text | multimodal
    task_type: str | None = None      # multiple_choice | classification | open_ended | ...
    component: str | None = None      # Language | Multimodal
    capability: str | None = None     # Knowledge | Reasoning | Multimodal | Safety
    specialty: str | None = None
    language: str | None = None
    modality: str | None = None       # Text | Image | ...
    difficulty: str | None = None     # Easy | Medium | Hard
    answer_format: str | None = None  # single_choice | multi_choice | free_text | ...
    evaluation_metric: str | None = None
    sample_weight: float = Field(default=1.0, gt=0)

    source_content: dict[str, Any] = Field(default_factory=dict)
    reference_answer: Any | None = None
    reference_answer_normalized: Any | None = None
    # Additional accepted forms of the answer (aliases / alternates). Metrics score the best
    # match over reference_answer + these — used where one gold answer has several valid phrasings.
    reference_aliases: list[str] | None = None
    media: MediaInfo | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Request-time sources may contain absolute paths, PIL objects, or embedded base64. Adapters
    # need them to build a request, but they must never be serialized into samples.jsonl.
    runtime_media: list[dict[str, Any]] = Field(default_factory=list, exclude=True, repr=False)
