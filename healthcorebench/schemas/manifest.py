"""Run manifest schema (manifest.json).

Records the full run configuration, environment, versions, benchmark source-file
integrity, model identity and run status. Written with ``run_status="running"`` at start
and atomically updated to a terminal status at the end. Never overwritten wholesale on a
resume — the existing manifest is loaded and updated.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ConfigDict

from healthcorebench.version import SCHEMA_VERSION


class SourceFileEntry(BaseModel):
    model_config = ConfigDict(extra="allow")
    relative_path: str
    file_name: str
    file_size_bytes: int | None = None
    sha256: str | None = None
    modified_time: str | None = None
    record_count: int | None = None


class BenchmarkManifest(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    registry_key: str | None = None
    version: str | None = None
    declared_benchmark_version: str | None = None
    effective_benchmark_revision: str | None = None
    split: str = "test"
    requested_split: str | None = None
    resolved_split: str | None = None
    benchmark_directory: str | None = None
    benchmark_path_mode: str = "registry_fixed"
    benchmark_path_override_used: bool = False
    declared_num_samples: int | None = None
    selected_num_samples: int = 0
    # Source records the adapter deliberately excluded (wrong question type, unparseable
    # options, missing media). Without these two fields the exclusions are invisible and
    # a partially-covered benchmark reads as fully covered.
    num_source_records_dropped: int | None = None
    source_record_drop_reasons: dict[str, int] = Field(default_factory=dict)
    adapter_name: str | None = None
    adapter_version: str | None = None
    source_files: list[SourceFileEntry] = Field(default_factory=list)
    source_files_combined_hash: str | None = None
    resolved_source_files: list[str] = Field(default_factory=list)
    sample_selection: dict = Field(default_factory=dict)


class ModelManifest(BaseModel):
    model_config = ConfigDict(extra="allow")
    provider: str | None = None
    base_url_redacted: str | None = None
    api_key_env: str | None = None
    requested_model_name: str | None = None
    served_model_name: str | None = None
    returned_model_names: list[str] = Field(default_factory=list)
    actual_model_version: str | None = None
    actual_model_version_status: str | None = None
    model_identity_source: str | None = None
    checkpoint_path: str | None = None
    checkpoint_revision: str | None = None
    checkpoint_hash: str | None = None
    system_fingerprints: list[str] = Field(default_factory=list)
    model_role: str = "evaluation"


class Manifest(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str = SCHEMA_VERSION
    experiment_id: str
    run_id: str
    run_name: str
    run_status: str = "running"  # running|completed|completed_with_errors|interrupted|failed
    config_hash: str | None = None
    full_config: dict[str, Any] = Field(default_factory=dict)

    benchmark: BenchmarkManifest
    model: ModelManifest
    generation: dict = Field(default_factory=dict)
    prompt: dict = Field(default_factory=dict)
    runtime: dict = Field(default_factory=dict)
    software: dict = Field(default_factory=dict)
    hardware: dict = Field(default_factory=dict)
    resume: dict = Field(default_factory=lambda: {"is_resumed": False, "resume_count": 0, "previous_stop_reason": None})
    # Values that must remain invariant when appending to an existing run directory.
    # Kept separate from ``full_config`` so it never needs to carry secrets.
    execution_identity: dict = Field(default_factory=dict)
