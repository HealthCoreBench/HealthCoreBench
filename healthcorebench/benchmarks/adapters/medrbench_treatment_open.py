"""MedR-Bench treatment planning adapter."""
import json
from pathlib import Path
from typing import Any,Iterable
from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.schemas.sample import EvaluationSample
class MedRBenchTreatmentAdapter(BaseBenchmarkAdapter):
    benchmark_name="MedR-Bench";benchmark_version="1.0"
    def discover_source_files(self)->list[Path]:return [self.get_benchmark_directory()/"data"/"MedRBench"/"treatment_496_cases_with_rare_disease_165.json"]
    def load_raw_samples(self,files:list[Path])->Iterable[dict]:
        f=files[0];rel=self.rel_path(f)
        for key,r in json.loads(f.read_text(encoding="utf-8")).items():
            g=r.get("generate_case") or {}
            if g.get("case_summary") and g.get("treatment_plan_results"):yield {"key":key,"r":r,"g":g,"rel":rel}
    def normalize_sample(self,raw:dict,sample_index:int)->EvaluationSample:
        g,rel=raw["g"],raw["rel"];q=str(g["case_summary"]);ref=str(g["treatment_plan_results"]);sid=raw["key"]
        return EvaluationSample(sample_id=self.make_sample_id(source_file_rel=rel,source_sample_id=sid,content_hash=self.input_hash(q)),source_sample_id=sid,sample_index=sample_index,benchmark_name=self.benchmark_name,benchmark_version=self.benchmark_version,benchmark_split=self.split,source_file=rel,source_record_hash=self.input_hash(raw["r"]),input_hash=self.input_hash(q),reference_hash=self.reference_hash(ref),task_type="open_ended",component="Language",capability="Reasoning",language="en",modality="Text",answer_format="free_text",evaluation_metric="llm_judge",source_content={"case":q},reference_answer=ref,reference_answer_normalized=ref)
    def build_messages(self,sample:EvaluationSample)->list[dict]:return [{"role":"user","content":f"Clinical case:\n{sample.source_content['case']}\n\nProvide the treatment plan."}]
    def parse_response(self,sample:EvaluationSample,raw_response:str)->Any:return (raw_response or "").strip()
