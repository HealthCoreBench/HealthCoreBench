"""PrinciplismQA case-level and rubric-level open-ended adapters.

Fixed data: ``67_PrinciplismQA/data/`` ships the two halves of one grading protocol.

``open-ended-qa.json`` — 677 clinical ethics cases::

    {"id": int, "title": str, "tags": [str], "case": str, "case_rewrite": str,
     "ethical_issues": [{"qid": int, "question": str, "keypoints": [str]}, ...]}

``open-ended-rubric-principles.json`` — the same 1,466 questions, one record each, joined to a
case by ``id`` and uniquely identified by ``qid``::

    {"id": int (case id), "qid": int, "question": str, "principles": [str],
     "keypoint_competencies": [{"keypoint": str, "competency": str}, ...]}

The ``keypoints`` are the grading rubric: the benchmark's own scorer
(``scripts/open_ended_eval.py``) asks a judge for one 1.0 / 0.5 / 0.0 verdict *per keypoint* and
averages them, and never asks for holistic similarity to a model answer. Both adapters used to
hand the judge ``json.dumps(...)`` of the raw structure as ``reference_answer`` and pass no
``metadata`` at all, so the keypoints arrived only as a JSON literal in the reference slot — the
judge was told to score for equivalence against brackets and escapes, and the rubric it is
prompted with was empty. The keypoints are therefore now rendered as text and also published in
metadata under ``judge_rubric``/``keypoints``, which ``LLMJudgeEvaluator`` folds into its rubric
section.

The two adapters keep different units on purpose: ``open`` scores one holistic discussion per
case (677 samples, all of that case's keypoints in scope), while ``rubric`` scores each ethical
question separately (1,466 samples), which is the unit the upstream script uses.
"""
import json
from pathlib import Path
from typing import Any, Iterable
from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.schemas.sample import EvaluationSample

# Tells the judge to grade by keypoint coverage instead of the default "equivalent to the
# reference" test: the reference is a checklist of several required points (1-10 per question,
# median 4), so a good answer covers them in its own words rather than restating one sentence.
_JUDGE_RUBRIC = (
    "The reference is a checklist of required keypoints, not a model answer to be paraphrased. "
    "Judge each keypoint as covered (1), partially or imprecisely covered (0.5), or missing or "
    "contradicted (0), then score the answer as the mean of those keypoint verdicts. Extra "
    "clinically sound content is not penalised; unsafe or unsupported claims are."
)


def _numbered(items: list[str]) -> str:
    """Render a checklist as numbered lines, the shape the upstream judge prompt uses."""
    return "\n".join(f"{n}. {text}" for n, text in enumerate(items, 1))


class _Base(BaseBenchmarkAdapter):
    benchmark_name = "PrinciplismQA"
    benchmark_version = "1.0"
    adapter_version = "1.1"
    filename = ""
    mode = ""

    def discover_source_files(self) -> list[Path]:
        return [self.get_benchmark_directory() / "data" / self.filename]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        for i, r in enumerate(json.loads(f.read_text(encoding="utf-8"))):
            yield {"r": r, "rel": rel, "i": i}

    def _question_and_reference(self, r: dict) -> tuple[str, str, dict]:
        """The prompt, the reference text, and the rubric metadata for one source record."""
        if self.mode == "case":
            issues = r.get("ethical_issues") or []
            # Every keypoint of every issue in the case is in scope, because the prompt asks for
            # one discussion of the whole case rather than one answer per issue.
            keypoints = [str(k) for issue in issues for k in (issue.get("keypoints") or [])]
            questions = [str(issue.get("question") or "") for issue in issues]
            q = (f"Clinical ethics case:\n{r.get('case_rewrite') or r.get('case')}\n\n"
                 "Discuss the ethical issues and recommended actions.")
            reference = _numbered(keypoints)
            metadata = {"judge_rubric": _JUDGE_RUBRIC, "keypoints": keypoints,
                        "ethical_questions": questions, "num_keypoints": len(keypoints),
                        "tags": r.get("tags"), "title": r.get("title")}
            return q, reference, metadata
        competencies = r.get("keypoint_competencies") or []
        keypoints = [str(k.get("keypoint") or "") for k in competencies]
        q = str(r.get("question"))
        reference = _numbered(keypoints)
        # ``principles`` and ``competency`` are the axes PrinciplismQA reports by; they are not
        # part of the expected answer, so they stay in metadata rather than in the reference.
        metadata = {"judge_rubric": _JUDGE_RUBRIC, "keypoints": keypoints,
                    "keypoint_competencies": competencies, "principles": r.get("principles"),
                    "num_keypoints": len(keypoints), "case_id": r.get("id")}
        return q, reference, metadata

    def normalize_sample(self, raw: dict, sample_index: int) -> EvaluationSample:
        r, rel = raw["r"], raw["rel"]
        q, ref, metadata = self._question_and_reference(r)
        # The rubric file repeats a case ``id`` for each of its questions (677 ids over 1,466
        # records), so ``qid`` — the key the upstream script joins on — is the addressable id there.
        sid = str(r.get("qid") if self.mode != "case" and r.get("qid") is not None
                  else r.get("id", raw["i"]))
        return EvaluationSample(
            sample_id=self.make_sample_id(source_file_rel=rel, source_sample_id=sid,
                                          content_hash=self.input_hash(q)),
            source_sample_id=sid, sample_index=sample_index,
            benchmark_name=self.benchmark_name, benchmark_version=self.benchmark_version,
            benchmark_split=self.split, source_file=rel, source_record_index=raw["i"],
            source_record_hash=self.input_hash(r), input_hash=self.input_hash(q),
            reference_hash=self.reference_hash(ref), task_type="open_ended",
            component="Language", capability="Reasoning", language="en", modality="Text",
            answer_format="free_text", evaluation_metric="llm_judge",
            source_content={"question": q}, reference_answer=ref, reference_answer_normalized=ref,
            metadata=metadata,
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        return [{"role": "user", "content": sample.source_content["question"]}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return (raw_response or "").strip()


class PrinciplismQAOpenAdapter(_Base):
    filename = "open-ended-qa.json"
    mode = "case"


class PrinciplismQARubricAdapter(_Base):
    filename = "open-ended-rubric-principles.json"
    mode = "rubric"
