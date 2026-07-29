"""Pydantic schemas defining every persisted on-disk contract.

Each output object carries a ``schema_version``. Records validate types, enums, UTC
timestamps, and non-negative token/latency fields. Unknown provider-specific fields
are preserved under ``provider_metadata`` rather than dropped. JSONL must never contain
NaN / Infinity.
"""

from healthcorebench.schemas.config import (
    ExperimentConfig,
    BenchmarkConfig,
    ModelConfig,
    GenerationConfig,
    RuntimeConfig,
    MediaConfig,
    OutputConfig,
    EvaluationConfig,
    JudgeConfig,
    HardwareConfig,
    RunConfig,
)
from healthcorebench.schemas.sample import EvaluationSample, MediaInfo, ImageInfo
from healthcorebench.schemas.request import AttemptRecord, UsageInfo
from healthcorebench.schemas.result import ResultRecord
from healthcorebench.schemas.judgment import JudgmentRecord
from healthcorebench.schemas.manifest import Manifest
from healthcorebench.schemas.summary import Summary

__all__ = [
    "ExperimentConfig",
    "BenchmarkConfig",
    "ModelConfig",
    "GenerationConfig",
    "RuntimeConfig",
    "MediaConfig",
    "OutputConfig",
    "EvaluationConfig",
    "JudgeConfig",
    "HardwareConfig",
    "RunConfig",
    "EvaluationSample",
    "MediaInfo",
    "ImageInfo",
    "AttemptRecord",
    "UsageInfo",
    "ResultRecord",
    "JudgmentRecord",
    "Manifest",
    "Summary",
]
