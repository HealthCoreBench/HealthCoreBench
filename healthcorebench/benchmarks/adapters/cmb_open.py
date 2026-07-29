"""CMB clinical case open-ended QA adapter."""
import json
from pathlib import Path
from typing import Any, Iterable
from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.schemas.sample import EvaluationSample

class CMBOpenAdapter(BaseBenchmarkAdapter):
    benchmark_name = "CMB"
    benchmark_version = "1.0"
    def discover_source_files(self) -> list[Path]:
        return [self.get_benchmark_directory() / "CMB-Clin" / "CMB-Clin-qa.json"]
    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f=files[0]; rel=self.rel_path(f); data=json.loads(f.read_text(encoding="utf-8"))
        for ci,case in enumerate(data):
            for qi,qa in enumerate(case.get("QA_pairs") or []):
                if qa.get("question") and qa.get("answer"):
                    yield {"case":case,"qa":qa,"rel":rel,"idx":ci,"qi":qi}
    def normalize_sample(self, raw: dict, sample_index: int) -> EvaluationSample:
        case,qa=raw["case"],raw["qa"]; q=str(qa["question"]); ref=str(qa["answer"]); rel=raw["rel"]
        sid=f"{case.get('id',raw['idx'])}:{raw['qi']}"; inp={"case":case.get("description"),"question":q}
        return EvaluationSample(sample_id=self.make_sample_id(source_file_rel=rel,source_sample_id=sid,content_hash=self.input_hash(inp)),source_sample_id=sid,sample_index=sample_index,benchmark_name=self.benchmark_name,benchmark_version=self.benchmark_version,benchmark_split=self.split,source_file=rel,source_record_index=raw["idx"],source_record_hash=self.input_hash({"case":case,"qa":qa}),input_hash=self.input_hash(inp),reference_hash=self.reference_hash(ref),task_type="open_ended",component="Language",capability="Reasoning",language="zh",modality="Text",answer_format="free_text",evaluation_metric="llm_judge",source_content=inp,reference_answer=ref,reference_answer_normalized=ref)
    def build_messages(self,sample:EvaluationSample)->list[dict]:
        c=sample.source_content; return [{"role":"user","content":f"病例：\n{c['case']}\n\n问题：{c['question']}"}]
    def parse_response(self,sample:EvaluationSample,raw_response:str)->Any:return (raw_response or "").strip()
