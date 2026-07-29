"""MedQA-MCMLE adapter (Chinese mainland medical licensing exam).

Fixed data: ``70_MedQA-MCMLE/MedQA-MCMLE.jsonl`` — one JSON object per line, same schema
as MedQA-USMLE (``question``, ``options`` dict, ``answer_idx``), but Chinese. Reuses the
USMLE adapter's loading/normalization, overriding the source file, language and prompt
language.
"""

from __future__ import annotations

from healthcorebench.benchmarks.adapters.medqa_usmle_mcqa import MedQAUSMLEAdapter


class MedQAMCMLEAdapter(MedQAUSMLEAdapter):
    benchmark_name = "MedQA_MCMLE"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    _source_file = "MedQA-MCMLE.jsonl"
    _lang = "zh"
