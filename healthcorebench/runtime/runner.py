"""Async run orchestrator.

Responsibilities:

* Drive the pipeline in order: samples (already written by the run setup) → inference
  (executor) → scoring (evaluators) → per-sample flush.
* Bound concurrency with a fixed worker pool; each sample runs independently and flushes on
  completion. Results may be written out of order; ``sample_index`` restores order later.
* Skip already-completed work on resume (successful inference, existing judgments).
* Handle SIGINT/SIGTERM: stop scheduling new samples, let in-flight ones finish, mark the
  manifest ``interrupted``.
* Never let one sample's failure abort the whole benchmark.

The runner is deliberately decoupled from *what* a benchmark is: it takes a list of
prepared sample dicts (with a ``logical_messages`` field) and a scoring callback.
"""

from __future__ import annotations

import asyncio
import inspect
import signal

from healthcorebench.runtime.executor import Executor
from healthcorebench.runtime.resume import ResumeIndex
from healthcorebench.schemas.result import ResultRecord
from healthcorebench.utils.timestamps import utc_now_iso


class ScoringUnavailableError(RuntimeError):
    """The scorer's endpoint is unusable, so no further sample can be scored.

    Raised by a scoring callback (see ``run_setup.make_scoring_callback``) rather than by one
    sample's failure: it aborts the run instead of letting every remaining task complete with
    a null score.
    """


class Runner:
    def __init__(
        self,
        *,
        executor: Executor,
        recorder,
        concurrency: int,
        n_repeats: int = 1,
        resume_index: ResumeIndex | None = None,
        retry_failed: bool = False,
        score_fn=None,   # callable(result_record_dict, sample_dict) -> list[JudgmentRecord]
        expected_evaluators: list[str] | None = None,
        length_finish_policy: str = "mark_incomplete",
    ) -> None:
        self.executor = executor
        self.recorder = recorder
        self.concurrency = max(1, concurrency)
        self.n_repeats = max(1, n_repeats)
        self.resume_index = resume_index or ResumeIndex()
        self.retry_failed = retry_failed
        self.score_fn = score_fn
        self.expected_evaluators = list(dict.fromkeys(expected_evaluators or []))
        self.length_finish_policy = length_finish_policy
        self._stop = asyncio.Event()
        self._interrupted = False

    def _install_signal_handlers(self, loop):
        def _handler():
            self._interrupted = True
            self._stop.set()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _handler)
            except (NotImplementedError, ValueError):
                pass  # e.g. not on main thread / unsupported platform

    async def run(self, samples: list[dict], *, progress_desc: str | None = None) -> dict:
        """Execute inference + scoring across ``samples``. Returns a small run report.

        When ``progress_desc`` is set, a tqdm progress bar (written to stderr) advances once
        per sample-repeat as it terminates — success, failure, skip or crash all tick it — so
        silence never looks like progress. The bar is best-effort: if tqdm is unavailable the
        run proceeds without one.
        """
        loop = asyncio.get_running_loop()
        self._install_signal_handlers(loop)

        counts = {"attempted": 0, "succeeded": 0, "failed": 0, "skipped": 0,
                  "scored": 0, "evaluation_failed": 0, "parsing_failed": 0,
                  "max_length": 0, "missing_scoring": 0}

        bar = self._make_bar(progress_desc, len(samples) * self.n_repeats)

        def _tick() -> None:
            if bar is None:
                return
            bar.set_postfix(ok=counts["succeeded"], fail=counts["failed"],
                            skip=counts["skipped"], refresh=False)
            bar.update(1)

        async def handle(
            sample: dict,
            repeat: int,
            logical_messages: list[dict] | None = None,
        ) -> ResultRecord | None:
            try:
                if self._stop.is_set():
                    counts["skipped"] += 1
                    return None
                sid = sample["sample_id"]
                # --- resume: skip completed inference ---
                if self.resume_index.has_success(sid, repeat):
                    counts["skipped"] += 1
                    backfill = await self._maybe_backfill_scoring(sample, repeat)
                    counts["scored"] += backfill["scored"]
                    counts["evaluation_failed"] += backfill["evaluation_failed"]
                    counts["missing_scoring"] += backfill["missing_scoring"]
                    stored = self.resume_index.result_for(sid, repeat)
                    return ResultRecord.model_validate(stored) if stored else None
                if self.resume_index.has_failure(sid, repeat) and not self.retry_failed:
                    counts["skipped"] += 1
                    stored = self.resume_index.result_for(sid, repeat)
                    return ResultRecord.model_validate(stored) if stored else None

                counts["attempted"] += 1
                logical = logical_messages if logical_messages is not None else sample["logical_messages"]
                try:
                    result = await self.executor.execute(sample, logical, repeat)
                except Exception as exc:
                    # An unexpected executor bug cannot be represented as a trustworthy model
                    # result. Make it fatal instead of silently completing with missing coverage.
                    self.recorder.record_event(
                        "sample_crash", {"sample_id": sid, "error": str(exc)[:500]},
                    )
                    raise RuntimeError(f"executor crashed for sample {sid}") from exc
                if result.status == "success":
                    counts["succeeded"] += 1
                    if result.finish_reason == "length":
                        counts["max_length"] += 1
                    if self._scoring_ineligible(result):
                        parse_result = getattr(self.score_fn, "parse_result", None)
                        if parse_result is not None:
                            parse_result(result, sample)
                        result.evaluation_status = "skipped"
                        result.evaluation_skip_reason = "max_length"
                        n, judgment_count = 0, 0
                    else:
                        n, judgment_count = await self._score_and_record(result, sample)
                    counts["scored"] += n
                    if result.parsing_status == "error":
                        counts["parsing_failed"] += 1
                    if result.evaluation_status == "error":
                        counts["evaluation_failed"] += 1
                    if judgment_count == 0:
                        counts["missing_scoring"] += 1
                else:
                    counts["failed"] += 1
                # A persistence failure is fatal and propagates out of the worker pool.
                self.recorder.record_result(result)
                return result
            finally:
                _tick()

        async def run_case_sequentially(case_samples: list[dict]) -> None:
            """Run one case in stage order, carrying model responses into later turns."""
            for repeat in range(self.n_repeats):
                history: list[dict] = []
                for sample in case_samples:
                    if self._stop.is_set():
                        return
                    current_messages = [*history, *sample["logical_messages"]]
                    result = await handle(sample, repeat, current_messages)
                    # A later LiveClin stage is meaningful only when the previous turn actually
                    # returned an answer. Transport/build failures stop this case; they are not
                    # converted into incorrect judgments for the unrequested later stages.
                    if result is None or result.status != "success" or result.raw_response is None:
                        return
                    history.extend(sample["logical_messages"])
                    history.append({"role": "assistant", "content": result.raw_response})

        queue: asyncio.Queue = asyncio.Queue(maxsize=max(self.concurrency * 4, 1))
        sentinel = object()

        async def worker() -> None:
            while True:
                item = await queue.get()
                try:
                    if item is sentinel:
                        return
                    if self._uses_case_sequential_protocol(samples):
                        await run_case_sequentially(item)
                    else:
                        sample, repeat = item
                        await handle(sample, repeat)
                finally:
                    queue.task_done()

        try:
            async with asyncio.TaskGroup() as group:
                for _ in range(self.concurrency):
                    group.create_task(worker())
                if self._uses_case_sequential_protocol(samples):
                    for case_samples in self._case_groups(samples):
                        if self._stop.is_set():
                            break
                        await queue.put(case_samples)
                else:
                    for sample in samples:
                        for repeat in range(self.n_repeats):
                            if self._stop.is_set():
                                break
                            await queue.put((sample, repeat))
                        if self._stop.is_set():
                            break
                for _ in range(self.concurrency):
                    await queue.put(sentinel)
        except* ScoringUnavailableError as unavailable:
            # Surface the breaker itself rather than a task group of identical copies.
            raise unavailable.exceptions[0] from None
        finally:
            if bar is not None:
                bar.close()

        return {"counts": counts, "interrupted": self._interrupted}

    @staticmethod
    def _uses_case_sequential_protocol(samples: list[dict]) -> bool:
        """Return whether the selected samples opt into case-level multi-turn execution."""
        return bool(samples) and all(
            (sample.get("metadata") or {}).get("conversation_mode") == "case_sequential"
            for sample in samples
        )

    @staticmethod
    def _case_groups(samples: list[dict]) -> list[list[dict]]:
        """Group opt-in samples by source case and preserve their stage order."""
        groups: dict[tuple, list[dict]] = {}
        for sample in samples:
            metadata = sample.get("metadata") or {}
            case_id = metadata.get("case_id")
            if case_id in (None, ""):
                key = (sample.get("sample_id"),)
            else:
                # Include the source row so a malformed/reused case label cannot merge unrelated
                # source records into one conversation.
                key = (
                    sample.get("source_file"),
                    sample.get("source_record_index"),
                    str(case_id),
                )
            groups.setdefault(key, []).append(sample)

        def stage_key(sample: dict) -> tuple[int, int]:
            value = (sample.get("metadata") or {}).get("stage_index")
            try:
                return 0, int(value)
            except (TypeError, ValueError):
                return 1, int(sample.get("sample_index", 0))

        return [sorted(case_samples, key=stage_key) for case_samples in groups.values()]

    @staticmethod
    def _make_bar(desc: str | None, total: int):
        if not desc:
            return None
        try:
            from tqdm.auto import tqdm
        except Exception:
            return None
        return tqdm(total=total, desc=desc, unit="sample", dynamic_ncols=True, leave=True)

    # ------------------------------------------------------------------ #
    def _scoring_ineligible(self, result: ResultRecord) -> bool:
        return (result.finish_reason == "length"
                and self.length_finish_policy == "mark_incomplete")

    async def _maybe_backfill_scoring(self, sample: dict, repeat: int) -> dict[str, int]:
        """If inference succeeded previously but a judgment is missing, score it now."""
        outcome = {"scored": 0, "evaluation_failed": 0, "missing_scoring": 0}
        if self.score_fn is None:
            return outcome
        result_id = self.resume_index.result_id_for(sample["sample_id"], repeat)
        if result_id is None:
            return outcome
        missing = {name for name in self.expected_evaluators
                   if not self.resume_index.is_judged(result_id, name)}
        if not missing:
            return outcome
        stored = self.resume_index.result_for(sample["sample_id"], repeat)
        if not stored:
            outcome["missing_scoring"] = len(missing)
            return outcome
        result = ResultRecord.model_validate(stored)
        if self._scoring_ineligible(result):
            result.evaluation_status = "skipped"
            result.evaluation_skip_reason = "max_length"
            self.recorder.record_result(result)
            self.resume_index.results_by_id[result_id] = result.model_dump()
            outcome["missing_scoring"] = len(missing)
            return outcome
        try:
            judgments = self.score_fn(result, sample, only_evaluators=missing)
            if inspect.isawaitable(judgments):
                judgments = await judgments
        except ScoringUnavailableError:
            raise
        except Exception as exc:
            self.recorder.record_event("scoring_backfill_crash", {
                "result_id": result_id, "error": str(exc)[:500],
            })
            outcome["evaluation_failed"] = 1
            outcome["missing_scoring"] = len(missing)
            return outcome
        primary = None
        for judgment in judgments or []:
            if judgment.evaluator_name not in missing:
                continue
            self.recorder.record_judgment(judgment)
            self.resume_index.judged.setdefault(result_id, set()).add(judgment.evaluator_name)
            if judgment.evaluation_status == "success":
                outcome["scored"] += 1
            else:
                outcome["evaluation_failed"] += 1
            if (judgment.provider_metadata or {}).get("primary_metric"):
                primary = judgment
        if primary is not None:
            result.evaluation_status = (
                "success" if primary.evaluation_status == "success" else "error"
            )
        self.recorder.record_result(result)
        self.resume_index.results_by_id[result_id] = result.model_dump()
        outcome["missing_scoring"] = sum(
            1 for name in missing if not self.resume_index.is_judged(result_id, name)
        )
        return outcome

    async def _score_and_record(self, result: ResultRecord, sample: dict) -> tuple[int, int]:
        if self.score_fn is None:
            return 0, 0
        try:
            judgments = self.score_fn(result, sample)
            if inspect.isawaitable(judgments):  # async score_fn (e.g. LLM judge)
                judgments = await judgments
        except ScoringUnavailableError:
            # Not one sample's problem: the scorer is gone. Let it abort the run.
            raise
        except Exception as exc:
            self.recorder.record_event("scoring_crash", {"result_id": result.result_id, "error": str(exc)[:500]})
            result.evaluation_status = "error"
            return 0, 0
        n = 0
        primary = None
        for j in judgments or []:
            self.recorder.record_judgment(j)
            if (getattr(j, "provider_metadata", None) or {}).get("primary_metric"):
                primary = j
            if j.evaluation_status == "success":
                n += 1
        if primary is None and judgments:
            primary = judgments[0]
        if primary is None:
            result.evaluation_status = "error"
        elif primary.evaluation_status == "success":
            result.evaluation_status = "success"
        else:
            result.evaluation_status = "error"
        return n, len(judgments or [])
