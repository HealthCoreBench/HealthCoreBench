"""IOR-Bench adapter (Chinese intelligent-triage / department routing).

Fixed data: ``66_IOR-Bench/IOR-Static.json`` — a JSON list of records::

    {"性别": str, "年龄": float, "对话内容": str (Python-literal list of {'病人':..,'医生':..} turns),
     "lable": str (correct department, Chinese), "对话轮数": int}

Task: department classification (triage). Given the patient-doctor dialogue, predict the correct
department out of the label set observed in the data (35 Chinese departments). ``对话内容`` is a
Python-literal string and is parsed into a readable transcript. Scored with the ``classification``
evaluator (exact department-name match). The dynamic ``IOR-Dynamic.json`` variant is a separate
interactive task and not covered here.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.benchmarks.answer_parsing import parse_label
from healthcorebench.schemas.sample import EvaluationSample


def _render_dialogue(raw: Any) -> str:
    """Parse the Python-literal dialogue list into a readable transcript."""
    turns = raw
    if isinstance(raw, str):
        try:
            turns = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return str(raw).strip()
    if not isinstance(turns, list):
        return str(raw).strip()
    lines = []
    for t in turns:
        if isinstance(t, dict):
            for role, text in t.items():
                lines.append(f"{role}：{str(text).strip()}")
    return "\n".join(lines)


class IORBenchTriageAdapter(BaseBenchmarkAdapter):
    benchmark_name = "IOR-Bench"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "triage"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"IOR-Bench (static) provides only 'test'; requested '{self.split}'.")
        return [directory / "IOR-Static.json"]

    def _load_records(self, f: Path) -> list[dict]:
        with open(f, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        records = self._load_records(f)
        # the label set is the set of departments present in the data.
        labels = sorted({str(r["lable"]).strip() for r in records if str(r.get("lable") or "").strip()})
        for i, rec in enumerate(records):
            label = str(rec.get("lable") or "").strip()
            dialogue = _render_dialogue(rec.get("对话内容"))
            if not label or not dialogue:
                continue
            yield {"record": rec, "dialogue": dialogue, "label": label, "labels": labels,
                   "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]
        dialogue = raw_sample["dialogue"]
        label = raw_sample["label"]
        labels = raw_sample["labels"]

        gender = str(rec.get("性别") or "").strip()
        age = rec.get("年龄")

        source_id = f"{rel}:{rec_index}"
        content_hash = self.input_hash({"d": dialogue})
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
            input_hash=self.input_hash({"dialogue": dialogue}),
            reference_hash=self.reference_hash(label),
            input_type="text",
            task_type="classification",
            component="Language",
            capability="Reasoning",
            specialty=label,
            language="zh",
            modality="Text",
            answer_format="label",
            evaluation_metric="accuracy",
            source_content={"dialogue": dialogue, "gender": gender, "age": age, "labels": labels},
            reference_answer=label,
            reference_answer_normalized=label,
            metadata={"labels": labels, "rounds": rec.get("对话轮数")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        labels = "、".join(c["labels"])
        head = f"患者性别：{c.get('gender')}，年龄：{c.get('age')}\n" if c.get("gender") else ""
        prompt = (
            f"{head}以下是患者与医生的分诊对话：\n{c['dialogue']}\n\n"
            f"请从以下科室中选择最合适的一个（只输出科室名称）：\n{labels}"
        )
        return [{"role": "user", "content": prompt}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_label(raw_response, sample.metadata.get("labels") or [])
