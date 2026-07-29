"""MedChain adapter (Chinese clinical case → preliminary diagnosis, free-text).

Fixed data: ``55_MedChain/datasets/filtered_data_test_set.json`` — a dict of cases; each case::

    {"tags": "{'科室': [...], '病种': [...]}" (python-literal),
     "【病例摘要】": "[...]", "【病案介绍】": "{'既往史':..,'查体':..,'现病史':..,'主诉':..,'图像':..,'影像报告':..}",
     "【诊治过程】": "{'初步诊断': [...], '诊治经过': [...]}", "【分析总结】": ..., "【治疗项目】": ...}

Task: open-ended preliminary diagnosis, Chinese. MedChain is designed as a multi-stage clinical
agent workflow; this adapter uses the static case text (主诉/现病史/查体/既往史 — the image and
imaging-report fields are dropped, keeping it text-only) and asks for the preliminary diagnosis.
The reference is ``【诊治过程】.初步诊断`` (LLM judge). Only cases carrying a 初步诊断 are used.
All literal fields are Python-literal strings parsed with ``ast.literal_eval``.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.adapters.hle_med_exact import extract_short_answer
from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.schemas.sample import EvaluationSample

# The reference is the case's 初步诊断 — a short diagnosis string (measured over 1,000 cases:
# median 18 characters, p90 51, 92% under 60), sometimes listing two or three diagnoses. The
# prompt asked an open question, so the secondary token-F1 scored a whole Chinese case discussion
# against it and, without a marker, ``extract_short_answer``'s last-line fallback landed on the
# closing caveat paragraph rather than the diagnosis.
#
# Reasoning is kept (the LLM judge is the primary metric and reads the full reply); only a
# locatable final line is requested. ``extract_short_answer``'s marker regex accepts English
# labels and ASCII ":"/"=" only, so the Chinese instruction has to prescribe that literal ASCII
# marker — and a Chinese-IME full-width "：" is normalized back in ``parse_response``.
_FINAL_ANSWER_INSTRUCTION = (
    "可以先写出推理过程，但回复的最后一行必须严格采用以下格式（Answer 和冒号一律使用英文半角）：\n"
    "Answer: <最可能的初步诊断，只写诊断名称；若有多个诊断，在同一行内用「；」分隔>"
)

# Only the marker's separator is normalized — the rest of the reply is left untouched.
_FULLWIDTH_ANSWER_COLON = re.compile(
    r"(?im)^(\s*(?:\*\*|__)?\s*(?:final\s+answer|answer)\s*(?:\*\*|__)?\s*)："
)

# text fields from 病案介绍 to include (order matters for readability); image fields excluded.
_INTRO_FIELDS = ["主诉", "现病史", "既往史", "查体"]


def _literal(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def _join(value: Any) -> str:
    v = _literal(value)
    if isinstance(v, list):
        return "；".join(str(x).strip() for x in v if str(x).strip() and not str(x).strip().endswith((".png", ".jpg", ".jpeg")))
    return str(v or "").strip()


class MedChainDiagnosisAdapter(BaseBenchmarkAdapter):
    benchmark_name = "MedChain"
    benchmark_version = "1.0"
    adapter_version = "1.1"
    prompt_template_name = "open_ended"
    prompt_template_version = "1.1"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"MedChain provides only 'test'; requested '{self.split}'.")
        return [directory / "datasets" / "filtered_data_test_set.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for key, case in data.items():
            proc = _literal(case.get("【诊治过程】"))
            dx = proc.get("初步诊断") if isinstance(proc, dict) else None
            dx_text = "；".join(str(x).strip() for x in dx if str(x).strip()) if isinstance(dx, list) else str(dx or "").strip()
            if not dx_text:
                continue
            intro = _literal(case.get("【病案介绍】"))
            parts = []
            summary = _join(case.get("【病例摘要】"))
            if summary:
                parts.append(f"病例摘要：{summary}")
            if isinstance(intro, dict):
                for fld in _INTRO_FIELDS:
                    val = _join(intro.get(fld))
                    if val:
                        parts.append(f"{fld}：{val}")
            case_text = "\n".join(parts).strip()
            if not case_text:
                continue
            yield {"case_key": key, "case_text": case_text, "reference": dx_text, "tags": case.get("tags"),
                   "source_file_rel": rel}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        key = raw_sample["case_key"]
        case_text = raw_sample["case_text"]
        reference = raw_sample["reference"]
        rel = raw_sample["source_file_rel"]

        tags = _literal(raw_sample.get("tags"))
        dept = None
        if isinstance(tags, dict) and isinstance(tags.get("科室"), list) and tags["科室"]:
            dept = "/".join(str(x) for x in tags["科室"])

        content_hash = self.input_hash({"c": case_text})
        sample_id = self.make_sample_id(source_file_rel=rel, source_sample_id=str(key), content_hash=content_hash)

        return EvaluationSample(
            sample_id=sample_id,
            source_sample_id=str(key),
            sample_index=sample_index,
            benchmark_name=self.benchmark_name,
            benchmark_version=self.benchmark_version,
            benchmark_split=self.split,
            source_benchmark_entry=rel,
            source_file=rel,
            source_record_index=None,
            source_record_hash=self.input_hash(case_text),
            input_hash=self.input_hash({"case_text": case_text}),
            reference_hash=self.reference_hash(reference),
            input_type="text",
            task_type="open_ended",
            component="Language",
            capability="Reasoning",
            specialty=dept,
            language="zh",
            modality="Text",
            answer_format="free_text",
            evaluation_metric="llm_judge",
            source_content={"case_text": case_text},
            reference_answer=reference,
            reference_answer_normalized=reference,
            metadata={"department": dept, "case_key": key},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        prompt = (
            f"{sample.source_content['case_text']}\n\n"
            "根据以上病例信息，请给出最可能的初步诊断。\n\n"
            f"{_FINAL_ANSWER_INSTRUCTION}"
        )
        return [{"role": "user", "content": prompt}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return extract_short_answer(_FULLWIDTH_ANSWER_COLON.sub(r"\1:", raw_response or ""))
