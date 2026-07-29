"""Run setup and top-level orchestration (the ``run`` command's engine).

Ties everything together deterministically:

1. Resolve the adapter from the registry and discover + hash fixed source files.
2. Load raw samples, normalize, and deterministically select (max_samples / sample_ids)
   without writing anything.
3. Validate resume config + source hashes, then write ``samples.jsonl`` (skipping any
   already recorded on resume).
4. Build the manifest (run_status=running), including model identity via ``/v1/models``.
5. Run inference + scoring via the async Runner, with a scoring callback that applies the
   adapter's parser + the configured evaluator.
6. Finalize the manifest status and write the summary.

Data reading (adapter) and model requests (client) stay decoupled: the adapter never sees
the client, the client never reads benchmark files.
"""

from __future__ import annotations

import asyncio
import ipaddress
import uuid
import warnings
from pathlib import Path
from urllib.parse import urlsplit

from healthcorebench.benchmarks import get_adapter, get_entry
from healthcorebench.benchmarks.context_window import UnbudgetedContextWarning
from healthcorebench.clients.openai_client import OpenAICompatibleClient
from healthcorebench.config import (
    RunConfig, config_hash, redact_base_url, redact_config_for_persistence,
    resolve_api_key, get_project_root,
)
from healthcorebench.evaluators import get_evaluator
from healthcorebench.evaluators.llm_judge import LLMJudgeEvaluator
from healthcorebench.runtime.executor import Executor
from healthcorebench.runtime.rate_limiter import RateLimiter
from healthcorebench.runtime.recorder import Recorder
from healthcorebench.runtime.resume import ResumeIndex
from healthcorebench.runtime.retry import RetryPolicy
from healthcorebench.runtime.runner import Runner
from healthcorebench.schemas.manifest import Manifest, BenchmarkManifest, ModelManifest, SourceFileEntry
from healthcorebench.utils.git import collect_git_info
from healthcorebench.utils.environment import collect_environment_info
from healthcorebench.utils.hashing import hash_json
from healthcorebench.utils.timestamps import utc_now_iso, duration_seconds
from healthcorebench.version import FRAMEWORK_VERSION, DEFAULT_PARSER_VERSION


def make_run_id() -> str:
    stamp = utc_now_iso().replace("-", "").replace(":", "").split(".")[0] + "Z"
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


class RunSetupError(Exception):
    pass


class RestrictedDataDisclosureWarning(UserWarning):
    """A corpus whose licence restricts it is being handled outside that restriction.

    Raised where the operator has authorized it explicitly — sending the records to a
    third-party judge (``evaluation.allow_restricted_data_to_remote_judge``) or writing them
    into the run directory (``output.allow_restricted_data_in_artifacts``).

    Its own category so an operator can promote it back to an error with ``-W error::...`` on a
    deployment where the authorization does not hold, without touching the config.
    """


def _is_local_endpoint(base_url: str) -> bool:
    """Whether ``base_url`` stays inside the operator's own network.

    Loopback, link-local, private ranges and ``.local``/``.internal`` names count as local;
    anything else is treated as a third party for data-protection purposes.
    """
    host = (urlsplit(base_url).hostname or "").strip("[]").lower()
    if not host:
        return False
    if host in {"localhost"} or host.endswith((".localhost", ".local", ".internal")):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(address.is_loopback or address.is_private or address.is_link_local
                or address.is_unspecified)


def _selection_dimensions(sample) -> tuple[str, ...]:
    metadata = sample.metadata or {}
    dimensions = [
        f"source:{sample.source_file or '-'}",
        f"language:{sample.language or '-'}",
    ]
    if sample.specialty is not None:
        dimensions.append(f"specialty:{sample.specialty}")
    patient_id = metadata.get("patient_id")
    if patient_id is not None:
        dimensions.append(f"patient:{patient_id}")
    if sample.answer_format in {
        "single_choice", "multi_choice", "label", "yes_no", "yes_no_maybe", "likert",
    }:
        dimensions.append(f"reference:{sample.reference_answer_normalized or sample.reference_answer}")
    return tuple(dimensions)


def _stratified_select(samples: list, limit: int) -> list:
    """Greedily balance source, language, case, and closed-task label dimensions."""
    remaining = list(enumerate(samples))
    counts: dict[str, int] = {}
    selected = []
    while remaining and len(selected) < limit:
        def key(item):
            index, sample = item
            dimensions = _selection_dimensions(sample)
            dimension_counts = [counts.get(dimension, 0) for dimension in dimensions]
            return (sum(dimension_counts), max(dimension_counts, default=0), index)

        position = min(range(len(remaining)), key=lambda i: key(remaining[i]))
        _, sample = remaining.pop(position)
        selected.append(sample)
        for dimension in _selection_dimensions(sample):
            counts[dimension] = counts.get(dimension, 0) + 1
    return selected


class RunOrchestrator:
    def __init__(self, config: RunConfig, *, run_id: str | None = None, run_dir: str | None = None,
                 task_number: int = 1, task_total: int = 1):
        self.config = config
        self.adapter = get_adapter(config.benchmark.name, config)
        self.entry = get_entry(config.benchmark.name)
        self.run_id = run_id or make_run_id()
        self.task_number = task_number
        self.task_total = task_total
        if run_dir:
            self.run_dir = Path(run_dir)
        else:
            # Layout: <root>/<experiment_id>/<bench>/<task>. The benchmark name is the task key
            # ("MMLU/mcqa"), so it nests as bench/task — one folder per benchmark under the task
            # (experiment), each holding that bench-task's files + logs. Re-running the same
            # experiment+bench reuses this dir (resume; config-hash guard blocks a changed config).
            self.run_dir = (get_project_root() / config.output.root_dir
                            / config.experiment.experiment_id / config.benchmark.name)
        self.recorder = Recorder(self.run_dir, fsync=config.runtime.fsync)
        # Which artifact permissions have already been disclosed, so an acknowledged
        # override warns once per run rather than once per sample.
        self._artifact_permissions_disclosed: set[str] = set()

    # ------------------------------------------------------------------ #
    def prepare_samples(self) -> list[dict]:
        """Discover, normalize and select samples without mutating the run directory."""
        files = self.adapter.discover_source_files()
        self.adapter.validate_source_files(files)
        self._source_entries, self._combined_hash = self.adapter.source_file_manifest(files)
        # ``prepare_samples`` may run more than once per process (prospective manifest,
        # then the real one). Reset so exclusions are counted for this pass only.
        if hasattr(self.adapter, "_source_record_drops"):
            self.adapter._source_record_drops.clear()

        cfg = self.config.benchmark
        wanted = set(cfg.sample_ids or [])
        if cfg.sample_ids and len(wanted) != len(cfg.sample_ids):
            raise RunSetupError("benchmark.sample_ids contains duplicate IDs")

        selected = []
        stratified_bucket_sizes: dict[tuple[str, ...], int] = {}
        seen_ids: set[str] = set()
        found_ids: set[str] = set()
        for raw_index, raw in enumerate(self.adapter.load_raw_samples(files)):
            sample = self.adapter.normalize_sample(raw, raw_index)
            if sample.sample_id in seen_ids:
                raise RunSetupError(f"duplicate normalized sample_id: {sample.sample_id}")
            seen_ids.add(sample.sample_id)
            if wanted and sample.sample_id not in wanted:
                continue
            if not wanted and cfg.max_samples is not None and cfg.selection_method == "stratified":
                bucket = _selection_dimensions(sample)
                bucket_size = stratified_bucket_sizes.get(bucket, 0)
                # Retaining at most the final sample limit per stratum bounds memory while still
                # allowing any one stratum to fill the entire requested quick-test sample.
                if bucket_size < cfg.max_samples:
                    selected.append(sample)
                    stratified_bucket_sizes[bucket] = bucket_size + 1
            else:
                selected.append(sample)
            found_ids.add(sample.sample_id)
            if wanted and found_ids == wanted:
                break
            if (not wanted and cfg.max_samples is not None
                    and cfg.selection_method == "head" and len(selected) >= cfg.max_samples):
                break

        if wanted:
            missing = [sample_id for sample_id in cfg.sample_ids or []
                       if sample_id not in found_ids]
            if missing:
                raise RunSetupError(
                    f"benchmark.sample_ids requested {len(wanted)} IDs but {len(missing)} "
                    f"were not found: {', '.join(missing[:5])}"
                )
            if cfg.max_samples is not None:
                selected = selected[:cfg.max_samples]
        elif cfg.max_samples is not None and cfg.selection_method == "stratified":
            selected = _stratified_select(selected, cfg.max_samples)
        if not selected:
            raise RunSetupError("sample selection produced zero samples")

        resume_idx = ResumeIndex.from_run_dir(self.run_dir) if self.config.runtime.resume else ResumeIndex()
        prepared: list[dict] = []
        for s in selected:
            request_max_tokens = self.adapter.max_output_tokens(s)
            if request_max_tokens is not None:
                s.metadata["request_max_tokens"] = request_max_tokens
                s.metadata.setdefault("output_token_budget_policy", "configured_generation_max_tokens")
            d = s.model_dump()
            d["logical_messages"] = self.adapter.build_messages(s)
            prepared.append(d)
        self._resume_index = resume_idx
        self._selected_count = len(prepared)
        self._selected_samples_hash = hash_json([
            {
                "sample_id": sample["sample_id"],
                "input_hash": sample.get("input_hash"),
                "reference_hash": sample.get("reference_hash"),
                # Embedded HLM images and local media paths are request-time data. Their source
                # file hashes already participate in the benchmark revision; do not hash raw
                # base64/path payloads into the selected-sample identity.
                "logical_messages_hash": hash_json(self._logical_messages_for_hash(sample["logical_messages"])),
            }
            for sample in prepared
        ])
        return prepared

    @staticmethod
    def _logical_messages_for_hash(messages: list[dict]) -> list[dict]:
        def sanitize(value):
            if isinstance(value, list):
                return [sanitize(item) for item in value]
            if isinstance(value, dict):
                if value.get("type") in {"image", "image_url", "video"}:
                    return {"type": value.get("type"), "media_id": value.get("media_id")}
                return {key: sanitize(item) for key, item in value.items() if key != "source"}
            return value
        return sanitize(messages)

    def _artifact_permission(self, field: str) -> bool:
        """Whether ``field`` of a restricted corpus may be written into the run directory.

        Resolved here rather than at each use so ``samples.jsonl`` and ``results.jsonl`` cannot
        disagree, and so the operator's acknowledgement is disclosed once per run instead of once
        per sample. ``field`` is ``store_full_input_allowed`` or ``store_reference_allowed``.
        """
        if getattr(self.adapter, field):
            return True
        if not self.config.output.allow_restricted_data_in_artifacts:
            return False
        if field not in self._artifact_permissions_disclosed:
            self._artifact_permissions_disclosed.add(field)
            declaration = self._license_declaration()
            detail = (
                f"Benchmark '{self.config.benchmark.name}' declares {field}=False"
                + (f" ({declaration.license_name})" if declaration else "")
                + ", and output.allow_restricted_data_in_artifacts is set, so those records are "
                  "written into the run directory."
            )
            warnings.warn(detail, RestrictedDataDisclosureWarning, stacklevel=3)
            self.recorder.record_event("restricted_data_written_to_artifacts", {
                "benchmark": self.config.benchmark.name,
                "field": field,
                "license_name": declaration.license_name if declaration else None,
                "license_evidence": declaration.evidence if declaration else None,
                "acknowledged_by_config": "output.allow_restricted_data_in_artifacts",
            })
        return True

    def _record_samples(self, samples: list[dict]) -> None:
        """Persist selected samples only after resume identity has been validated."""
        for sample in samples:
            if not self._resume_index.sample_recorded(sample["sample_id"]):
                payload = {k: v for k, v in sample.items() if k != "logical_messages"}
                if (not self.config.output.save_source_content
                        or not self._artifact_permission("store_full_input_allowed")):
                    payload["source_content"] = {}
                if not self._artifact_permission("store_reference_allowed"):
                    payload["reference_answer"] = None
                    payload["reference_answer_normalized"] = None
                    payload["reference_aliases"] = None
                self.recorder.record_sample(payload)

    # ------------------------------------------------------------------ #
    def build_manifest(self, model_identity: dict) -> Manifest:
        cfg = self.config
        bench = BenchmarkManifest(
            name=cfg.benchmark.name,
            registry_key=cfg.benchmark.name,
            version=self.adapter.benchmark_version,
            declared_benchmark_version=cfg.benchmark.version or self.adapter.benchmark_version,
            effective_benchmark_revision=f"sha256:{self._combined_hash}" if self._combined_hash and not self._combined_hash.startswith("sha256:") else self._combined_hash,
            split=cfg.benchmark.split,
            requested_split=cfg.benchmark.split,
            resolved_split=cfg.benchmark.split,
            benchmark_directory=self.entry.benchmark_dir,
            benchmark_path_mode="registry_fixed" if not cfg.benchmark.debug_data_path_override else "debug_override",
            benchmark_path_override_used=bool(cfg.benchmark.debug_data_path_override),
            selected_num_samples=self._selected_count,
            num_source_records_dropped=getattr(self.adapter, "num_source_records_dropped", None),
            source_record_drop_reasons=getattr(self.adapter, "source_record_drop_reasons", {}) or {},
            adapter_name=type(self.adapter).__name__,
            adapter_version=self.adapter.adapter_version,
            source_files=[SourceFileEntry(**e) for e in self._source_entries],
            source_files_combined_hash=self._combined_hash,
            resolved_source_files=[e["relative_path"] for e in self._source_entries],
            sample_selection={
                "selection_method": (
                    "sample_ids" if cfg.benchmark.sample_ids
                    else cfg.benchmark.selection_method if cfg.benchmark.max_samples
                    else "all"
                ),
                "max_samples": cfg.benchmark.max_samples,
                "selection_seed": None,
                "selected_samples_hash": self._selected_samples_hash,
            },
        )
        model = ModelManifest(
            provider=cfg.model.provider,
            base_url_redacted=redact_base_url(cfg.model.base_url),
            api_key_env=cfg.model.api_key_env,
            requested_model_name=cfg.model.requested_model_name,
            served_model_name=cfg.model.served_model_name,
            returned_model_names=model_identity.get("returned_model_names", []),
            actual_model_version=model_identity.get("actual_model_version"),
            actual_model_version_status=model_identity.get("actual_model_version_status"),
            model_identity_source=model_identity.get("model_identity_source"),
            checkpoint_path=cfg.model.checkpoint_path,
            checkpoint_revision=cfg.model.checkpoint_revision,
            checkpoint_hash=cfg.model.checkpoint_hash,
            model_role=cfg.model.role,
        )
        software = collect_environment_info()
        software.update(collect_git_info(get_project_root()))
        prompt_meta = {
            "template_name": self.adapter.prompt_template_name,
            "template_version": self.adapter.prompt_template_version,
        }
        return Manifest(
            experiment_id=cfg.experiment.experiment_id, run_id=self.run_id,
            run_name=cfg.experiment.run_name, run_status="running",
            config_hash=config_hash(cfg), full_config=redact_config_for_persistence(cfg),
            benchmark=bench, model=model,
            generation=cfg.generation.model_dump(), prompt=prompt_meta,
            runtime=cfg.runtime.model_dump(), software=software,
            hardware=cfg.hardware.model_dump(),
            execution_identity={
                "framework_version": FRAMEWORK_VERSION,
                "parser_version": DEFAULT_PARSER_VERSION,
                "adapter_name": type(self.adapter).__name__,
                "adapter_version": self.adapter.adapter_version,
                "prompt_template_name": self.adapter.prompt_template_name,
                "prompt_template_version": self.adapter.prompt_template_version,
                "selected_samples_hash": self._selected_samples_hash,
            },
        )

    # ------------------------------------------------------------------ #
    def _resolve_evaluation(self, cfg, samples: list[dict]) -> None:
        """Fill in evaluation.evaluator / use_llm_judge / extra_evaluators from the benchmark
        when left unset.

        Users specify only the benchmark; the correct scorer follows from the samples' declared
        ``evaluation_metric``/``answer_format``. An explicit config value is always respected.
        """
        from healthcorebench.evaluators import select_evaluator_name, default_extra_evaluators

        metric = fmt = None
        profile = None
        if samples:
            s0 = samples[0]
            metric = s0.get("evaluation_metric")
            fmt = s0.get("answer_format")
            profile = (s0.get("metadata") or {}).get("task_profile")

        # VLM free-text tasks enable their structured judge automatically when a judge endpoint
        # is configured. Without one, deterministic overlap metrics still run instead of making
        # an otherwise valid VLM config fail before inference.
        judge_explicitly_enabled = cfg.evaluation.use_llm_judge is True
        if cfg.evaluation.use_llm_judge is None:
            vlm_judge_profile = profile in {
                "short_open", "generation", "report", "video_open", "agentic_open",
                "document_qa", "document_parse", "document_complex_qa",
            }
            cfg.evaluation.use_llm_judge = (
                metric == "llm_judge"
                or s0.get("component") == "Multimodal" and vlm_judge_profile
                and cfg.evaluation.judge is not None
            )
        self._judge_as_primary = (
            judge_explicitly_enabled
            or metric == "llm_judge"
            or profile in {
                "generation", "report", "video_open", "agentic_open",
                "document_complex_qa",
            }
        )

        # evaluator: None => auto-pick the rule-based evaluator (unused when judging).
        if cfg.evaluation.evaluator is None:
            auto = select_evaluator_name(metric, fmt)
            if auto is None and not cfg.evaluation.use_llm_judge:
                # Defaulting to an MCQA parser here scored non-choice tasks against whatever
                # letter the parser happened to find. Refuse instead: the metric the benchmark
                # declares has no rule-based scorer, so either a judge or an explicit
                # evaluator must be chosen deliberately.
                raise RunSetupError(
                    f"No rule-based evaluator exists for benchmark '{cfg.benchmark.name}' "
                    f"(evaluation_metric={metric!r}, answer_format={fmt!r}) and "
                    "evaluation.use_llm_judge is false. Set evaluation.evaluator explicitly, "
                    "or enable evaluation.use_llm_judge with an evaluation.judge endpoint."
                )
            cfg.evaluation.evaluator = auto

        # extra_evaluators: None => the benchmark's per-task defaults (secondary metrics).
        if cfg.evaluation.extra_evaluators is None:
            cfg.evaluation.extra_evaluators = default_extra_evaluators(cfg.benchmark.name)

    def make_scoring_callback(self, judge_evaluator=None):
        """Build the scoring callback.

        The primary scorer is either a rule-based evaluator or (when ``use_llm_judge``) the LLM
        judge; it is tagged ``primary_metric`` so the summary always treats it as the headline
        score. Any ``evaluation.extra_evaluators`` are rule-based secondary metrics run in
        addition — their judgments are recorded alongside but never override the primary.
        """
        evaluator = get_evaluator(self.config.evaluation.evaluator) if self.config.evaluation.evaluator else None
        # Drop any extra that duplicates the primary evaluator: judgments dedup by
        # (result_id, evaluator_name), so an untagged duplicate would overwrite the primary.
        judge_as_primary = bool(getattr(self, "_judge_as_primary", True))
        extra_names = list(self.config.evaluation.extra_evaluators or [])
        if judge_evaluator is not None and judge_as_primary and self.config.evaluation.evaluator:
            extra_names.insert(0, self.config.evaluation.evaluator)
        extras = [get_evaluator(name) for name in dict.fromkeys(extra_names)
                  if judge_evaluator is not None or name != self.config.evaluation.evaluator]
        adapter = self.adapter

        def _parse_onto(result, sample_dict):
            # adapter.parse_response needs an EvaluationSample-like object; use a dict shim.
            from types import SimpleNamespace
            sample_obj = SimpleNamespace(**sample_dict)
            parsed = adapter.parse_response(sample_obj, result.raw_response or "")
            result.parsed_answer = parsed
            result.parser_name = type(adapter).__name__
            result.parser_version = DEFAULT_PARSER_VERSION
            result.normalized_answer = parsed
            result.parsing_status = "success" if parsed is not None else "error"
            result.parse_timestamp = utc_now_iso()
            return result.model_dump()

        def _tag_primary(judgment):
            # mark the headline judgment so summarize prefers it over rule-based secondaries.
            try:
                judgment.provider_metadata["primary_metric"] = True
            except Exception:
                pass
            return judgment

        def _run_extras(result_dict, sample_dict, only_evaluators=None):
            return [e.evaluate(result_dict, sample_dict) for e in extras
                    if only_evaluators is None or e.evaluator_name in only_evaluators]

        if judge_evaluator is None:
            if evaluator is None:
                raise RunSetupError(
                    "No evaluator resolved: benchmark expects an LLM judge but evaluation.judge "
                    "is not configured (set evaluation.judge, or an explicit evaluation.evaluator)."
                )
            def score_fn(result, sample_dict: dict, only_evaluators=None):
                result_dict = _parse_onto(result, sample_dict)
                judgments = []
                if only_evaluators is None or evaluator.evaluator_name in only_evaluators:
                    judgments.append(_tag_primary(evaluator.evaluate(result_dict, sample_dict)))
                judgments.extend(_run_extras(result_dict, sample_dict, only_evaluators))
                return judgments
            score_fn.parse_result = lambda result, sample: _parse_onto(result, sample)
            return score_fn

        async def score_fn_judge(result, sample_dict: dict, only_evaluators=None):
            result_dict = _parse_onto(result, sample_dict)
            judgments = []
            if not judge_as_primary and evaluator is not None and (
                only_evaluators is None or evaluator.evaluator_name in only_evaluators
            ):
                judgments.append(_tag_primary(evaluator.evaluate(result_dict, sample_dict)))
            if only_evaluators is None or judge_evaluator.evaluator_name in only_evaluators:
                judgment = await judge_evaluator.evaluate_async(result_dict, sample_dict)
                self._check_judge_health(judgment)
                if judge_as_primary:
                    judgment = _tag_primary(judgment)
                judgments.append(judgment)
            judgments.extend(_run_extras(result_dict, sample_dict, only_evaluators))
            return judgments
        score_fn_judge.parse_result = lambda result, sample: _parse_onto(result, sample)
        return score_fn_judge

    # ------------------------------------------------------------------ #
    # judge circuit breaker
    # ------------------------------------------------------------------ #
    # A judge endpoint that is unauthenticated, rate-limited or down fails every sample
    # identically. One measured run scored 25 judge tasks entirely against a 401, left 10 of
    # them at score null, and still reported run_status completed with 56 tasks. Break the run
    # after this many *consecutive* infrastructure failures: with 8 concurrent judge requests
    # it cannot be reached by chance (a 23.3% independent failure rate — the worst measured —
    # gives 0.233**20 ≈ 1e-13) while a wholesale outage trips it within the first task.
    JUDGE_INFRASTRUCTURE_FAILURE_LIMIT = 20

    def _check_judge_health(self, judgment) -> None:
        from healthcorebench.clients.errors import (
            INFRASTRUCTURE_ERROR_TYPES, classify_error_message,
        )
        from healthcorebench.runtime.runner import ScoringUnavailableError

        if judgment.evaluation_status == "success":
            self._judge_infrastructure_failures = 0
            return
        error_type = classify_error_message(judgment.evaluation_error)
        if error_type not in INFRASTRUCTURE_ERROR_TYPES:
            # An unparseable or empty verdict is a per-sample problem, not a dead endpoint.
            return
        self._judge_infrastructure_failures = getattr(
            self, "_judge_infrastructure_failures", 0
        ) + 1
        if self._judge_infrastructure_failures < self.JUDGE_INFRASTRUCTURE_FAILURE_LIMIT:
            return
        self.recorder.record_event("judge_circuit_break", {
            "consecutive_failures": self._judge_infrastructure_failures,
            "error_type": error_type.value,
            "error_message": (judgment.evaluation_error or "")[:500],
        })
        raise ScoringUnavailableError(
            f"LLM judge failed with {error_type.value} on "
            f"{self._judge_infrastructure_failures} consecutive samples "
            f"({(judgment.evaluation_error or '')[:200]}). Aborting rather than reporting "
            "null-scored tasks. Fix the judge endpoint/credentials and resume the run."
        )

    # ------------------------------------------------------------------ #
    async def run_async(self) -> dict:
        from healthcorebench.runtime.run_lock import RunDirectoryLease

        with RunDirectoryLease(self.run_dir) as lease:
            self._run_lease = lease.info
            return await self._run_async_locked()

    async def _run_async_locked(self) -> dict:
        cfg = self.config
        client = None
        judge_client = None
        run_started = False
        start = utc_now_iso()
        try:
            samples = self.prepare_samples()
            self._resolve_evaluation(cfg, samples)
            self._check_context_budget_configured(cfg)
            self._check_data_protection(cfg)

            # Validate run identity before writing samples, events, or a new manifest.
            # Model identity is intentionally not part of either resume guard.
            prospective_manifest = self.build_manifest({})
            self._handle_resume(prospective_manifest)
            prospective_manifest.runtime["run_lease"] = self._run_lease
            self.recorder.write_manifest(prospective_manifest)
            run_started = True
            self._record_samples(samples)

            # Print the evaluation plan before any model call.
            self._print_plan(cfg)

            api_key = resolve_api_key(cfg.model.api_key_env, cfg.model.api_key)
            client = OpenAICompatibleClient(
                base_url=cfg.model.base_url, api_key=api_key,
                requested_model_name=cfg.model.requested_model_name,
                timeout=cfg.runtime.request_timeout_seconds,
                default_headers=cfg.model.default_headers,
            )

            model_identity = await self._probe_identity(client)
            manifest = self.build_manifest(model_identity)
            manifest.resume = prospective_manifest.resume
            manifest.runtime["run_lease"] = self._run_lease
            self.recorder.write_manifest(manifest)
            self.recorder.record_event("run_start", {
                "run_id": self.run_id, "num_samples": len(samples),
                "resume_session_id": self._run_lease["session_id"],
            })
            if not getattr(self, "_context_budget_enforced", True):
                # Recorded only after the resume guard has accepted this run, so a refused
                # resume still leaves the existing logs byte-identical.
                self.recorder.record_event("context_budget_not_enforced", {
                    "benchmark": cfg.benchmark.name,
                    "context_overflow_policy": cfg.generation.context_overflow_policy,
                    "context_token_reserve": cfg.generation.context_token_reserve,
                })

            rate_limiter = None
            if cfg.runtime.requests_per_minute or cfg.runtime.tokens_per_minute:
                rate_limiter = RateLimiter(
                    cfg.runtime.requests_per_minute, cfg.runtime.tokens_per_minute,
                )

            executor = Executor(
                client=client, run_id=self.run_id, provider=cfg.model.provider,
                generation=cfg.generation, media=cfg.media, output=cfg.output,
                retry_policy=RetryPolicy(
                    cfg.runtime.max_retries,
                    cfg.runtime.retry_backoff_initial_seconds,
                    cfg.runtime.retry_backoff_max_seconds,
                ),
                recorder=self.recorder, rate_limiter=rate_limiter,
                same_budget_error_retries=cfg.runtime.same_budget_error_retries,
                same_budget_timeout_retries=cfg.runtime.same_budget_timeout_retries,
                transient_error_ladder_steps=cfg.runtime.transient_error_ladder_steps,
                max_model_len=cfg.hardware.max_model_len,
                store_full_input_allowed=self._artifact_permission(
                    "store_full_input_allowed"),
            )
            judge_evaluator = None
            if cfg.evaluation.use_llm_judge:
                jcfg = cfg.evaluation.judge
                if jcfg is None:
                    raise RunSetupError(
                        "evaluation.use_llm_judge is true but evaluation.judge is not configured."
                    )
                judge_client = OpenAICompatibleClient(
                    base_url=jcfg.base_url,
                    api_key=resolve_api_key(jcfg.api_key_env, jcfg.api_key),
                    requested_model_name=jcfg.requested_model_name,
                    timeout=jcfg.request_timeout_seconds,
                )
                judge_evaluator = LLMJudgeEvaluator(
                    client=judge_client, judge_model=jcfg.requested_model_name,
                    prompt_version=jcfg.prompt_version, temperature=jcfg.temperature,
                    max_tokens=jcfg.max_tokens, reasoning_effort=jcfg.reasoning_effort,
                    recorder=self.recorder, run_id=self.run_id, provider=jcfg.provider,
                    max_retries=jcfg.max_retries, concurrency=jcfg.concurrency,
                )

            primary_evaluator = ("llm_judge" if judge_evaluator is not None
                                 and getattr(self, "_judge_as_primary", True) else
                                 get_evaluator(cfg.evaluation.evaluator).evaluator_name)
            expected_evaluators = [primary_evaluator]
            if judge_evaluator is not None and cfg.evaluation.evaluator:
                expected_evaluators.append(get_evaluator(cfg.evaluation.evaluator).evaluator_name)
            expected_evaluators.extend(
                get_evaluator(name).evaluator_name
                for name in (cfg.evaluation.extra_evaluators or [])
                if name != cfg.evaluation.evaluator
            )

            runner = Runner(
                executor=executor, recorder=self.recorder,
                concurrency=cfg.runtime.concurrency,
                n_repeats=cfg.generation.n, resume_index=self._resume_index,
                retry_failed=cfg.runtime.retry_failed,
                score_fn=self.make_scoring_callback(judge_evaluator=judge_evaluator),
                expected_evaluators=expected_evaluators,
                length_finish_policy=cfg.generation.length_finish_policy,
            )

            report = await runner.run(samples, progress_desc=cfg.benchmark.name)
            end = utc_now_iso()

            # Recompute from persisted logs before choosing the terminal state.
            from healthcorebench.aggregation.summarize import summarize_run
            summary = summarize_run(self.run_dir)
            expected_results = len(samples) * cfg.generation.n
            coverage_ok = summary.counts.num_attempted == expected_results
            has_errors = any((
                summary.counts.num_failed,
                summary.counts.num_evaluation_errors,
                summary.counts.num_parsing_errors,
                summary.counts.num_max_length,
                summary.counts.num_missing_scoring,
            ))
            status = "interrupted" if report["interrupted"] else (
                "failed" if not coverage_ok else
                ("completed_with_errors" if has_errors else "completed")
            )
            self.recorder.write_summary(summary)
            self.recorder.update_manifest_status(status, extra={
                "runtime_end_time": end,
                "runtime_start_time": start,
                "wall_time_seconds": duration_seconds(start, end),
                "logical_result_coverage": {
                    "actual": summary.counts.num_attempted,
                    "expected": expected_results,
                },
            })
            self.recorder.record_event(
                "run_end", {"status": status, "counts": summary.counts.model_dump()},
            )

            from healthcorebench.runtime import reporting
            reporting.print_task_complete(
                task_key=cfg.benchmark.name,
                status=status,
                task_number=self.task_number,
                task_total=self.task_total,
            )

            return {"run_dir": str(self.run_dir), "status": status, "report": report,
                    "summary_metrics": summary.metrics.model_dump()}
        except asyncio.CancelledError:
            if run_started:
                self._record_terminal_failure(
                    "interrupted", start, "run_cancelled", "run cancelled",
                )
            raise
        except BaseException as exc:
            if run_started:
                self._record_terminal_failure(
                    "failed", start, "run_crash", str(exc)[:500],
                )
            raise
        finally:
            clients = [candidate for candidate in (judge_client, client) if candidate is not None]
            if clients:
                await asyncio.gather(
                    *(candidate.aclose() for candidate in clients), return_exceptions=True,
                )

    def _check_context_budget_configured(self, cfg) -> None:
        """Reject context budgets that cannot produce a valid request, before making any.

        With ``hardware.max_model_len`` unset nothing is trimmed, so records longer than the
        served window fail outright: a measured EHRBench/risk run lost its 2,494 longest
        records this way and published 0.622 over the remaining 5,227 (the same data with
        max_model_len=16384 scored 0.669 over all 7,721). A smaller max_tokens cannot rescue an
        over-long prompt, so those failures are unrecoverable.

        Two of these misconfigurations are refused outright rather than warned about, because
        no run started from them can produce a usable number:

        * ``context_overflow_policy=head_tail`` with no window. Trimming was asked for by
          name and is silently not happening; every over-long record fails instead.
        * An output budget that leaves no room for a prompt. ``max_tokens`` is subtracted from
          the window before the prompt gets any, so ``max_tokens=262144`` against a 16k window
          makes *every* request invalid — the observed 95.7% invalid-request rate. Only fatal
          when no rung of the adaptive ladder fits either; if a lower rung fits, the run is
          merely wasting one request per sample, which is a warning.
        """
        gen = cfg.generation
        window = cfg.hardware.max_model_len
        if window is None:
            if gen.context_overflow_policy == "head_tail":
                raise RunSetupError(
                    f"generation.context_overflow_policy=head_tail for '{cfg.benchmark.name}' "
                    "but hardware.max_model_len is not set. Trimming has no window to trim to, "
                    "so it is silently disabled and every over-long record fails instead of "
                    "being shortened — the opposite of what head_tail asks for. Set "
                    "hardware.max_model_len to the served context length."
                )
            warnings.warn(
                f"hardware.max_model_len is not set for '{cfg.benchmark.name}': prompt trimming "
                "is disabled, generation.context_overflow_policy "
                f"({gen.context_overflow_policy}) and generation.context_token_reserve "
                f"({gen.context_token_reserve}) have no effect, and any record longer "
                "than the served context window will fail instead of being trimmed — biasing the "
                "score towards shorter records. Set hardware.max_model_len to the served window.",
                UnbudgetedContextWarning,
                stacklevel=2,
            )
            self._context_budget_enforced = False
            return

        # Mirror the arithmetic in ``context_window.fit_context_to_window``, minus the
        # per-sample fixed prompt: what is left here is the most any sample could ever get,
        # so a non-positive result is fatal for all of them, not just the long ones.
        reserve = gen.context_token_reserve
        requested = gen.max_tokens or 0
        # Lowest rung the executor could settle on (``_max_token_budgets`` keeps only
        # candidates at or below the requested budget).
        rungs = [requested, *(gen.max_tokens_candidates if gen.adaptive_max_tokens else ())]
        floor = min(rung for rung in rungs if rung <= requested) if requested else 0
        if window - floor - reserve <= 0:
            raise RunSetupError(
                f"No usable context budget for '{cfg.benchmark.name}': hardware.max_model_len "
                f"({window}) minus the smallest reachable generation.max_tokens ({floor}) minus "
                f"generation.context_token_reserve ({reserve}) leaves "
                f"{window - floor - reserve} tokens for the prompt. Every request would be "
                "rejected by the server. Lower generation.max_tokens or raise "
                "hardware.max_model_len to the served context length."
            )
        if window - requested - reserve <= 0:
            warnings.warn(
                f"generation.max_tokens ({requested}) does not fit in hardware.max_model_len "
                f"({window}) alongside generation.context_token_reserve ({reserve}) for "
                f"'{cfg.benchmark.name}': the first attempt at every sample is invalid and only "
                f"succeeds after the adaptive ladder drops to {floor} or below. Set "
                "generation.max_tokens to a budget the served window can actually hold.",
                UnbudgetedContextWarning,
                stacklevel=2,
            )

    def _check_data_protection(self, cfg) -> None:
        """Guard shipping a non-redistributable benchmark's text to a remote judge.

        Adapters for restricted corpora (MIMIC / PhysioNet and similar) declare
        ``redistribution_allowed = False``; sending those records to a third-party judge
        endpoint is a redistribution the data use agreement does not permit by default.

        The framework cannot know what authorization the operator holds, so it does not get the
        last word: ``evaluation.allow_restricted_data_to_remote_judge`` lets the data controller
        assert that authorization exists. The check then still runs and still names the corpus and
        the endpoint — as a warning and a recorded event rather than a refusal — so the decision
        stays visible in the run's own artifacts instead of being erased.
        """
        if self.adapter.redistribution_allowed or not cfg.evaluation.use_llm_judge:
            return
        judge = cfg.evaluation.judge
        if judge is None or _is_local_endpoint(judge.base_url):
            return

        declaration = self._license_declaration()
        detail = (
            f"Benchmark '{cfg.benchmark.name}' declares redistribution_allowed=False"
            + (f" ({declaration.license_name}; evidence: {declaration.evidence})"
               if declaration else "")
            + f", and evaluation.judge.base_url ({redact_base_url(judge.base_url)}) is not a "
              "local endpoint, so scoring sends the restricted records to a third party."
        )
        if not cfg.evaluation.allow_restricted_data_to_remote_judge:
            raise RunSetupError(
                detail + " Use a locally served judge, an explicit rule-based "
                "evaluation.evaluator, or set evaluation."
                "allow_restricted_data_to_remote_judge: true to assert that you hold the "
                "authorization this corpus requires for that endpoint."
            )
        warnings.warn(
            detail + " Proceeding because evaluation.allow_restricted_data_to_remote_judge is "
            "set: the operator has asserted authorization for this endpoint.",
            RestrictedDataDisclosureWarning,
            stacklevel=2,
        )
        self.recorder.record_event("restricted_data_sent_to_remote_judge", {
            "benchmark": cfg.benchmark.name,
            "judge_base_url": redact_base_url(judge.base_url),
            "license_name": declaration.license_name if declaration else None,
            "license_evidence": declaration.evidence if declaration else None,
            "acknowledged_by_config": "evaluation.allow_restricted_data_to_remote_judge",
        })

    def _license_declaration(self):
        from healthcorebench.benchmarks.data_licenses import license_for

        return license_for(getattr(self.entry, "benchmark_dir", "") or "")

    def _record_terminal_failure(self, status: str, start: str,
                                 event_type: str, message: str) -> None:
        end = utc_now_iso()
        try:
            self.recorder.update_manifest_status(status, extra={
                "runtime_end_time": end,
                "runtime_start_time": start,
                "wall_time_seconds": duration_seconds(start, end),
                "fatal_error": message,
            })
            self.recorder.record_event(event_type, {"status": status, "error": message})
        except Exception:
            # Preserve the original failure if the storage layer itself is unavailable.
            pass

    def _print_plan(self, cfg) -> None:
        from healthcorebench.runtime import reporting
        judge_model = cfg.evaluation.judge.requested_model_name if (
            cfg.evaluation.use_llm_judge and cfg.evaluation.judge) else None
        reporting.print_task_plan(
            task_key=cfg.benchmark.name,
            bench_name=self.entry.benchmark_name,
            task=self.entry.task,
            num_samples=self._selected_count,
            evaluator=cfg.evaluation.evaluator,
            use_llm_judge=bool(cfg.evaluation.use_llm_judge),
            judge_model=judge_model,
            model_name=cfg.model.requested_model_name,
            base_url_redacted=redact_base_url(cfg.model.base_url),
            run_dir=self.run_dir,
            extra_evaluators=cfg.evaluation.extra_evaluators,
            task_number=self.task_number,
            task_total=self.task_total,
        )

    def run(self) -> dict:
        return asyncio.run(self.run_async())

    # ------------------------------------------------------------------ #
    async def _probe_identity(self, client) -> dict:
        served = await client.list_models()
        identity = {
            "returned_model_names": served,
            "model_identity_source": "v1/models" if served else "config",
            "actual_model_version": None,
            "actual_model_version_status": "not_exposed_by_provider",
        }
        return identity

    def _handle_resume(self, manifest: Manifest) -> None:
        existing = self.recorder.read_manifest()
        if not existing:
            return
        existing_run_id = existing.get("run_id")
        if existing_run_id:
            self.run_id = existing_run_id
            manifest.run_id = existing_run_id
        # Config identity guard.
        if existing.get("config_hash") and existing["config_hash"] != manifest.config_hash:
            raise RunSetupError(
                "Refusing to resume: run config changed since the existing run "
                f"({existing.get('config_hash')} != {manifest.config_hash}). Start a new run."
            )
        prev_hash = (existing.get("benchmark") or {}).get("source_files_combined_hash")
        if prev_hash and prev_hash != self._combined_hash:
            raise RunSetupError(
                "Refusing to resume: benchmark source files changed "
                f"({prev_hash} != {self._combined_hash}). Start a new run."
            )
        existing_identity = existing.get("execution_identity")
        if not existing_identity:
            raise RunSetupError(
                "Refusing to resume: the existing manifest has no execution identity. "
                "Start a new run rather than mixing legacy outputs with current code."
            )
        if existing_identity != manifest.execution_identity:
            raise RunSetupError(
                "Refusing to resume: execution identity changed (adapter, parser, prompt, "
                "framework, or selected samples). Start a new run."
            )
        # carry forward resume bookkeeping
        prev_resume = existing.get("resume") or {}
        manifest.resume = {
            "is_resumed": True,
            "resume_count": int(prev_resume.get("resume_count", 0)) + 1,
            "previous_stop_reason": existing.get("run_status"),
            "resume_session_id": self._run_lease["session_id"],
        }
