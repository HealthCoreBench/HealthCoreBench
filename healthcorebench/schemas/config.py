"""Run configuration schema (parsed from YAML).

A single run corresponds to exactly one model, one benchmark and one split. Batch matrices
of models/benchmarks are the job of an external orchestrator that creates multiple runs.

API keys can be supplied directly (``api_key``) or through an environment variable
(``api_key_env``). Direct keys are represented as ``SecretStr`` and excluded from every
model dump, so they are available at runtime but never persisted to manifests or logs.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict, SecretStr, model_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentConfig(_Base):
    experiment_id: str
    run_name: str
    model_role: str = "evaluation"


class BenchmarkConfig(_Base):
    name: str
    version: str | None = None
    split: str = "test"
    max_samples: int | None = Field(default=None, gt=0)
    sample_ids: list[str] | None = None
    selection_method: Literal["head", "stratified"] = "stratified"
    # Debug-only override of the fixed registry directory. Disabled by default; when set,
    # the run is flagged non-standard in the manifest and must not feed a leaderboard.
    debug_data_path_override: str | None = None


class ModelConfig(_Base):
    provider: str = "local_vllm"
    base_url: str
    api_key: SecretStr | None = Field(default=None, exclude=True)
    api_key_env: str = "OPENAI_API_KEY"
    requested_model_name: str
    served_model_name: str | None = None
    checkpoint_path: str | None = None
    checkpoint_revision: str | None = None
    checkpoint_hash: str | None = None
    role: str = "evaluation"
    default_headers: dict[str, str] | None = None


class GenerationConfig(_Base):
    temperature: float | None = 0.0
    top_p: float | None = Field(default=1.0, ge=0.0, le=1.0)
    # Generous cap so long chain-of-thought / long-form answers (reports, summaries, diagnoses)
    # are not truncated mid-answer and mis-scored. Override per-model if the server limits lower.
    max_tokens: int | None = Field(default=32768, gt=0)
    adaptive_max_tokens: bool = True
    max_tokens_candidates: list[int] = Field(
        default_factory=lambda: [8192, 4096, 2048, 1024, 512, 256, 128, 64]
    )
    reduce_max_tokens_on_context_error: bool = True
    reduce_max_tokens_on_timeout: bool = True
    length_finish_policy: Literal["score", "mark_incomplete"] = "mark_incomplete"
    seed: int | None = 42
    n: int = Field(default=1, ge=1)
    stop: Any | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    response_format: Any | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)
    # Long-context adapters use these values before persistence/request construction. The
    # default remains strict; enable head_tail explicitly for models shorter than the data.
    context_overflow_policy: Literal["error", "head_tail"] = "error"
    context_token_reserve: int = Field(default=512, ge=0)

    @model_validator(mode="after")
    def _validate_token_candidates(self):
        if any(candidate <= 0 for candidate in self.max_tokens_candidates):
            raise ValueError("generation.max_tokens_candidates must contain only positive integers")
        return self


class RuntimeConfig(_Base):
    concurrency: int = Field(default=16, ge=1)
    request_timeout_seconds: float = Field(default=180.0, gt=0)
    max_retries: int = Field(default=5, ge=0)
    same_budget_error_retries: int = Field(default=2, ge=0)
    # Kept for backward compatibility with existing non-adaptive timeout configs.
    same_budget_timeout_retries: int = Field(default=1, ge=0)
    # How far the adaptive token ladder is walked for a *transient* failure (timeout, 5xx,
    # rate limit). A context error walks the whole ladder because a smaller output budget is
    # the actual remedy; a sick server is not fixed by cutting max_tokens, and walking every
    # tier for it turned 10 samples into 176 requests in a measured run.
    transient_error_ladder_steps: int = Field(default=2, ge=0)
    retry_backoff_initial_seconds: float = Field(default=2.0, gt=0)
    retry_backoff_max_seconds: float = Field(default=60.0, gt=0)
    requests_per_minute: float | None = Field(default=None, gt=0)
    tokens_per_minute: float | None = Field(default=None, gt=0)
    resume: bool = True
    retry_failed: bool = False
    flush_every_record: bool = True
    fsync: bool = False


class MediaConfig(_Base):
    image_detail: str = "auto"
    # Includes sampled video frames. MTBBench cases contain up to 44 source images.
    max_images: int = Field(default=64, ge=0)
    max_pixels: int | None = Field(default=None, gt=0)
    max_image_size_mb: float | None = Field(default=5.0, gt=0)
    image_format: str = "png"
    allow_image_urls: bool = False
    video_frame_sampling_strategy: str = "uniform"
    max_video_frames: int = Field(default=32, ge=1)


class OutputConfig(_Base):
    root_dir: str = "runs"
    save_source_content: bool = True
    save_formatted_prompt: bool = True
    save_request_payload: bool = False
    save_raw_response_object: bool = True
    # Corpora declared ``store_full_input_allowed=False`` (or ``store_reference_allowed=False``)
    # in benchmarks/data_licenses.py normally have those fields blanked in the run directory:
    # ``source_content`` in samples.jsonl and ``formatted_prompt`` in results.jsonl, so a run
    # directory that gets copied or attached to a report does not carry the records with it.
    # Scoring is unaffected either way — the redaction only ever touched the persisted copy.
    #
    # Set this to true to assert, as the data controller, that this machine's run directories are
    # an appropriate place for those records. Each affected task is named in the log and recorded
    # in the run's own events, and the declarations in data_licenses.py are unchanged.
    #
    # ``save_source_content`` and ``save_formatted_prompt`` are independent: they are ordinary
    # size/privacy switches, and setting this does not turn them back on.
    allow_restricted_data_in_artifacts: bool = False


class JudgeConfig(_Base):
    """Independent config for an LLM judge; uses the same client but separate identity."""

    provider: str = "openai"
    base_url: str
    api_key: SecretStr | None = Field(default=None, exclude=True)
    api_key_env: str = "OPENAI_API_KEY"
    requested_model_name: str
    temperature: float | None = 0.0
    max_tokens: int | None = Field(default=8192, gt=0)
    # For reasoning-capable judge models (e.g. gpt-5.5), turn reasoning off: the judge only
    # emits a verdict, and reasoning tokens would blow the budget / truncate the output.
    # gpt-5.5 supports 'none' | 'low' | 'medium' | 'high' | 'xhigh' (NOT 'minimal').
    reasoning_effort: str | None = "none"
    concurrency: int = Field(default=8, ge=1)
    prompt_version: str = "2.0"
    request_timeout_seconds: float = Field(default=180.0, gt=0)
    max_retries: int = Field(default=5, ge=0)


class EvaluationConfig(_Base):
    # None = auto-select the rule-based evaluator from the benchmark's declared metric
    # (multiple_choice / multiple_answer / classification / numeric_tolerance / likert_credit /
    # text_f1_em / rouge / bleu). Set explicitly only to override. Benchmarks whose metric is
    # "llm_judge" auto-enable the judge path when use_llm_judge is left at its default.
    evaluator: str | None = None
    use_llm_judge: bool | None = None   # None = auto (on for llm_judge benchmarks, else off)
    judge: JudgeConfig | None = None
    # Secondary metrics reported *alongside* the primary score (they never replace it). None =
    # use the benchmark's per-task defaults (e.g. ROUGE-L on long-form QA, token-F1 on diagnosis);
    # set to [] to force no extras, or an explicit list of evaluator names to override.
    extra_evaluators: list[str] | None = None
    # Corpora declared ``redistribution_allowed=False`` in benchmarks/data_licenses.py (the
    # PhysioNet credentialed ones) are normally refused when the judge endpoint is not local,
    # because scoring them would send restricted records to a third party.
    #
    # Set this to true to assert, as the data controller, that you hold the authorization those
    # corpora require for the endpoint you configured. The refusal becomes a warning: the run
    # proceeds, every affected task is named in the log, and the acknowledgement is recorded in
    # the manifest so a run's own artifacts say it was made. The declarations themselves are
    # unchanged, so revoking this is a one-line edit rather than a re-derivation.
    #
    # It does not affect anything else: a local judge never needed it, and rule-based tasks do
    # not leave the machine at all.
    allow_restricted_data_to_remote_judge: bool = False


class HardwareConfig(_Base):
    """User-supplied hardware facts (framework cannot infer remote GPUs)."""

    execution_mode: str = "remote_api"
    gpu_type: str | None = None
    num_gpus: int | None = None
    precision: str | None = None
    tensor_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None
    dtype: str | None = None
    quantization: str | None = None
    max_model_len: int | None = Field(default=None, gt=0)
    vllm_version: str | None = None


class RunConfig(_Base):
    """Top-level run configuration."""

    experiment: ExperimentConfig
    benchmark: BenchmarkConfig
    model: ModelConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    media: MediaConfig = Field(default_factory=MediaConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)

    def config_hash_payload(self) -> dict:
        """Return the subset of config that defines run identity for resume validation.

        Excludes purely operational knobs (concurrency, timeouts, resume flags) that can
        change between resume attempts without invalidating already-collected data.
        Includes everything that would change the samples, prompts or model outputs.
        """
        judge = self.evaluation.judge
        judge_identity = None
        if judge is not None and self.evaluation.use_llm_judge:
            judge_identity = judge.model_dump(exclude={
                "api_key_env",
                "concurrency",
                "request_timeout_seconds",
                "max_retries",
            })
        return {
            "benchmark": self.benchmark.model_dump(),
            "model": {
                **self.model.model_dump(exclude={"default_headers"}),
                # Header values can select provider routing / feature flags. Keep non-secret
                # values in resume identity while making credential rotation resume-safe.
                "default_headers": _headers_for_identity(self.model.default_headers),
            },
            "generation": self.generation.model_dump(),
            # max_model_len changes long-context sample normalization, so it is part of run
            # identity even though the rest of the hardware inventory is informational.
            "context_window": {"max_model_len": self.hardware.max_model_len},
            "media": self.media.model_dump(),
            "evaluation": {
                "evaluator": self.evaluation.evaluator,
                "use_llm_judge": self.evaluation.use_llm_judge,
                "judge": judge_identity,
                "extra_evaluators": self.evaluation.extra_evaluators,
            },
        }


def _headers_for_identity(headers: dict[str, str] | None) -> dict[str, str] | None:
    if headers is None:
        return None
    out: dict[str, str] = {}
    for key, value in headers.items():
        out[key] = "***REDACTED***" if _is_sensitive_identity_key(key) else value
    return out


def _is_sensitive_identity_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    exact = {
        "api_key", "apikey", "authorization", "access_token", "auth_token",
        "token", "api_token", "bearer_token", "session_token", "refresh_token", "id_token", "bearer",
        "secret", "client_secret", "password", "credential", "credentials",
    }
    suffixes = (
        "_api_key", "_token", "_secret", "_password", "_credential",
        "_credentials",
    )
    return normalized in exact or normalized.endswith(suffixes)
