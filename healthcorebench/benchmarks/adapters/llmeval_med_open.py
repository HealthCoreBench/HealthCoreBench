"""LLMEval-Med adapter (Chinese medical free-text evaluation).

Fixed data: ``35_LLMEval-Med/llmeval_med_test.json`` — a JSON list of records::

    {"category1": str, "category2": str, "scene": str, "round": int, "problem": str,
     "groupCode": str, "sanswer": str (standard reference answer), "difficulty": str,
     "checklist": str (scoring rubric)}

Task: open-ended free-text answering, Chinese, across knowledge / reasoning / language /
safety-ethics / text-generation categories. The reference ``sanswer`` (with the ``checklist``
rubric carried in metadata) is scored by an LLM judge.

Multi-turn: a record with ``round`` > 1 is a follow-up turn of the record(s) immediately above
it in the file (same ``groupCode``, ``round`` incremented by one). Measured on the fixed data:
571 conversations, 96 of the 643 scorable records are follow-ups, every one of them preceded by
its complete history. Those follow-ups are unanswerable in isolation ("那有个肿瘤病人的信息也帮我
写成一份病历" only makes sense after the previous request), so the earlier turns are replayed as
real ``user``/``assistant`` messages with the dataset's own gold answers as the assistant side.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.schemas.sample import EvaluationSample


def _text(v) -> str:
    """Stripped string, treating None and NaN as empty.

    The source JSON (pandas-exported) uses NaN for missing ``problem`` / ``sanswer``. NaN is
    truthy, so a naive ``str(x or "")`` would turn it into the literal ``"nan"`` — a garbage
    reference answer that then survives the not-empty filter. Detect NaN via ``v != v``.
    """
    if v is None or (isinstance(v, float) and v != v):
        return ""
    return str(v).strip()


class LLMEvalMedOpenAdapter(BaseBenchmarkAdapter):
    benchmark_name = "LLMEval-Med"
    benchmark_version = "1.0"
    adapter_version = "1.1"
    prompt_template_name = "open_ended"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"LLMEval-Med provides only 'test'; requested '{self.split}'.")
        return [directory / "llmeval_med_test.json"]

    @staticmethod
    def _conversations(records: list[dict]) -> list[list[tuple[int, dict]]]:
        """Group the flat record list into conversations.

        The file carries no conversation id of its own: ``groupCode`` labels the topic and is
        reused by unrelated conversations (324 distinct codes for 667 records, 185 of them used by
        more than one conversation), so it cannot be grouped on directly. What does hold is the
        file order — a conversation is a contiguous run whose ``groupCode`` stays the same while
        ``round`` counts up by one. That yields 571 conversations, every one of them starting at
        ``round`` 1, i.e. no follow-up is left without its history.
        """
        conversations: list[list[tuple[int, dict]]] = []
        current: list[tuple[int, dict]] = []
        for index, record in enumerate(records):
            previous = current[-1][1] if current else None
            if (previous is not None
                    and record.get("groupCode") == previous.get("groupCode")
                    and record.get("round") == (previous.get("round") or 0) + 1):
                current.append((index, record))
                continue
            if current:
                conversations.append(current)
            current = [(index, record)]
        if current:
            conversations.append(current)
        return conversations

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        for conversation_index, conversation in enumerate(self._conversations(records)):
            # Prior turns of this conversation, replayed for every later turn. The assistant side
            # is the dataset's own ``sanswer`` rather than a model reply: each turn is scored
            # against its own reference, so feeding the gold history keeps a late turn's score
            # independent of how the model happened to answer the earlier ones.
            history: list[dict] = []
            for i, rec in conversation:
                problem = _text(rec.get("problem"))
                answer = _text(rec.get("sanswer"))
                if not problem:
                    self.drop_source_record("empty_question")
                    continue
                if not answer:
                    # 24/667 rows ship ``sanswer: NaN`` (mostly 医疗文本生成); they keep a
                    # checklist but have no reference text for the judge to score against.
                    # Reported so the 643 scored items are not read as the full 667-row file.
                    self.drop_source_record("missing_reference_answer")
                    # Not appended to ``history`` either: a turn with no gold answer cannot
                    # supply an assistant message. None occur mid-conversation in the fixed data.
                    continue
                yield {"record": rec, "problem": problem, "reference": answer,
                       "history": list(history), "conversation_index": conversation_index,
                       "source_file_rel": rel, "source_record_index": i}
                history.append({"problem": problem, "answer": answer})

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]
        problem = raw_sample["problem"]
        reference = raw_sample["reference"]
        history: list[dict] = raw_sample["history"]

        # ``groupCode`` alone is not a sample id: it names the topic and is shared by every turn of
        # a conversation *and* by unrelated conversations, so it mapped up to five records onto one
        # id. Qualify it with the conversation ordinal and the turn number to make it addressable.
        group_code = rec.get("groupCode")
        source_id = (f"{group_code}#{raw_sample['conversation_index']}:{rec.get('round')}"
                     if group_code is not None else f"{rel}:{rec_index}")
        # The history is part of the prompt, so it is part of the sample's identity: two turns
        # asking the same follow-up after different openings are different inputs.
        content_hash = self.input_hash({"q": problem, "history": history})
        sample_id = self.make_sample_id(source_file_rel=rel, source_sample_id=source_id, content_hash=content_hash)

        return EvaluationSample(
            sample_id=sample_id,
            source_sample_id=source_id,
            sample_index=sample_index,
            benchmark_name=self.benchmark_name,
            benchmark_version=self.benchmark_version,
            benchmark_split=self.split,
            source_benchmark_entry=rel,
            source_file=rel,
            source_record_index=rec_index,
            source_record_hash=self.input_hash(rec),
            input_hash=self.input_hash({"problem": problem, "history": history}),
            reference_hash=self.reference_hash(reference),
            input_type="text",
            task_type="open_ended",
            component="Language",
            capability="Reasoning",
            specialty=rec.get("category1"),
            difficulty={"难": "hard", "中": "medium", "易": "easy"}.get(rec.get("difficulty")),
            language="zh",
            modality="Text",
            answer_format="free_text",
            evaluation_metric="llm_judge",
            source_content={"problem": problem, "history": history},
            reference_answer=reference,
            reference_answer_normalized=reference,
            metadata={"category1": rec.get("category1"), "category2": rec.get("category2"),
                      "scene": rec.get("scene"), "difficulty": rec.get("difficulty"),
                      "checklist": rec.get("checklist"),
                      "group_code": group_code, "round": rec.get("round"),
                      "num_history_turns": len(history)},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        # Replay the earlier turns before the question being scored. Sending only ``problem`` made
        # the 96 follow-up turns unanswerable — "这个作用的强弱的表示方法是什么" has no referent
        # without the turn that introduced 皂苷溶血作用 — and the judge then scored the model on a
        # question it was never shown.
        messages: list[dict] = []
        for turn in sample.source_content.get("history") or []:
            messages.append({"role": "user", "content": turn["problem"]})
            messages.append({"role": "assistant", "content": turn["answer"]})
        messages.append({"role": "user", "content": sample.source_content["problem"]})
        return messages

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return (raw_response or "").strip()
