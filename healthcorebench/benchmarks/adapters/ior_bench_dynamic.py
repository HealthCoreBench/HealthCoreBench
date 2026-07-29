"""IOR-Bench dynamic-case department routing adapter."""
import json
from pathlib import Path
from typing import Any, Iterable
from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.answer_parsing import parse_label
from healthcorebench.schemas.sample import EvaluationSample

class IORBenchDynamicAdapter(BaseBenchmarkAdapter):
    benchmark_name="IOR-Bench"; benchmark_version="1.0"
    def discover_source_files(self)->list[Path]: return [self.get_benchmark_directory()/"IOR-Dynamic.json"]
    def load_raw_samples(self,files:list[Path])->Iterable[dict]:
        f=files[0]; rel=self.rel_path(f); rows=json.loads(f.read_text(encoding="utf-8")); labels=sorted({str(x.get("hospital-1")) for x in rows})
        for i,r in enumerate(rows):
            if r.get("hospital-1"): yield {"r":r,"rel":rel,"i":i,"labels":labels}
    def normalize_sample(self,raw:dict,sample_index:int)->EvaluationSample:
        r,rel=raw["r"],raw["rel"]; label=str(r["hospital-1"]); inp={k:r.get(k) for k in ("性别","年龄","主诉","现病史","既往史","家族史")}
        sid=f"{rel}:{raw['i']}"
        return EvaluationSample(sample_id=self.make_sample_id(source_file_rel=rel,source_sample_id=sid,content_hash=self.input_hash(inp)),source_sample_id=sid,sample_index=sample_index,benchmark_name=self.benchmark_name,benchmark_version=self.benchmark_version,benchmark_split=self.split,source_file=rel,source_record_index=raw["i"],source_record_hash=self.input_hash(r),input_hash=self.input_hash(inp),reference_hash=self.reference_hash(label),task_type="classification",component="Language",capability="Reasoning",language="zh",modality="Text",answer_format="label",evaluation_metric="accuracy",source_content={**inp,"labels":raw["labels"]},reference_answer=label,reference_answer_normalized=label,metadata={"labels":raw["labels"],"secondary_department":r.get("hospital-2")})
    def build_messages(self,sample:EvaluationSample)->list[dict]:
        c=sample.source_content; body="\n".join(f"{k}：{v}" for k,v in c.items() if k!="labels" and v); return [{"role":"user","content":f"{body}\n\n请从以下科室选择首选科室，只输出科室名：{'、'.join(c['labels'])}"}]
    def parse_response(self,sample:EvaluationSample,raw_response:str)->Any:return parse_label(raw_response,sample.metadata.get("labels") or [])
