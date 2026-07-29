"""GlobalDentBench case-based open question adapter."""
import json
from pathlib import Path
from typing import Any, Iterable
from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.schemas.sample import EvaluationSample
class GlobalDentBenchCBQAdapter(BaseBenchmarkAdapter):
    benchmark_name="GlobalDentBench"; benchmark_version="1.0"
    def discover_source_files(self)->list[Path]:return [self.get_benchmark_directory()/"GlobalDentBench-OA.json"]
    def load_raw_samples(self,files:list[Path])->Iterable[dict]:
        f=files[0]; rel=self.rel_path(f)
        for i,r in enumerate(json.loads(f.read_text(encoding="utf-8"))["CBQ"]): yield {"r":r,"rel":rel,"i":i}
    def normalize_sample(self,raw:dict,sample_index:int)->EvaluationSample:
        r,rel=raw["r"],raw["rel"]; q=str((r.get("seed_question") or {}).get("question")); ref=json.dumps(r.get("key_points"),ensure_ascii=False); sid=str(r.get("id",raw["i"]))
        return EvaluationSample(sample_id=self.make_sample_id(source_file_rel=rel,source_sample_id=sid,content_hash=self.input_hash(q)),source_sample_id=sid,sample_index=sample_index,benchmark_name=self.benchmark_name,benchmark_version=self.benchmark_version,benchmark_split=self.split,source_file=rel,source_record_index=raw["i"],source_record_hash=self.input_hash(r),input_hash=self.input_hash(q),reference_hash=self.reference_hash(ref),task_type="open_ended",component="Language",capability="Reasoning",language="en",modality="Text",answer_format="free_text",evaluation_metric="llm_judge",source_content={"question":q},reference_answer=ref,reference_answer_normalized=ref)
    def build_messages(self,sample:EvaluationSample)->list[dict]:return [{"role":"user","content":sample.source_content["question"]}]
    def parse_response(self,sample:EvaluationSample,raw_response:str)->Any:return (raw_response or "").strip()
