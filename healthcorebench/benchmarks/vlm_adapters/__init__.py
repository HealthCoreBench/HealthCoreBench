"""Adapters for benchmarks under ``benchmarks/medical_vlm_benchmarks``."""

from healthcorebench.benchmarks.vlm_adapters.catalog import VLM_TASK_SPECS
from healthcorebench.benchmarks.vlm_adapters.generic import MedicalVLMAdapter

__all__ = ["MedicalVLMAdapter", "VLM_TASK_SPECS"]
