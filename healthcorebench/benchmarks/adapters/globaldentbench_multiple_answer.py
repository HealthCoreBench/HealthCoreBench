"""GlobalDentBench multiple-answer MCQ adapter."""
import json,re
from pathlib import Path
from typing import Any,Iterable
from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.prompts import multiple_answer_prompt
from healthcorebench.benchmarks.answer_parsing import parse_multiple_choice_letters
from healthcorebench.schemas.sample import EvaluationSample
class GlobalDentBenchMultipleAnswerAdapter(BaseBenchmarkAdapter):
    benchmark_name="GlobalDentBench"; benchmark_version="1.0"
    def discover_source_files(self)->list[Path]:return [self.get_benchmark_directory()/"GlobalDentBench-OA.json"]
    def load_raw_samples(self,files:list[Path])->Iterable[dict]:
        f=files[0];rel=self.rel_path(f)
        for i,r in enumerate(json.loads(f.read_text(encoding="utf-8"))["MCQ"]):
            ans=re.findall(r"[A-Z]",str(r.get("answer","")).upper()); opts=r.get("options") or {}
            if len(ans)>1 and set(ans)<=set(opts):yield {"r":r,"rel":rel,"i":i,"ans":ans}
    def normalize_sample(self,raw:dict,sample_index:int)->EvaluationSample:
        r,rel=raw["r"],raw["rel"];opts={str(k):str(v) for k,v in r["options"].items()};letters=sorted(opts);ref=",".join(sorted(set(raw["ans"])));sid=str(r.get("id",raw["i"]));block="\n".join(f"{x}. {opts[x]}" for x in letters)
        return EvaluationSample(sample_id=self.make_sample_id(source_file_rel=rel,source_sample_id=sid,content_hash=self.input_hash({"q":r["question"],"o":opts})),source_sample_id=sid,sample_index=sample_index,benchmark_name=self.benchmark_name,benchmark_version=self.benchmark_version,benchmark_split=self.split,source_file=rel,source_record_index=raw["i"],source_record_hash=self.input_hash(r),input_hash=self.input_hash({"q":r["question"],"b":block}),reference_hash=self.reference_hash(ref),task_type="multiple_choice",component="Language",capability="Knowledge",language="en",modality="Text",answer_format="multi_choice",evaluation_metric="set_match",source_content={"question":r["question"],"options":opts,"letters":letters},reference_answer=ref,reference_answer_normalized=ref,metadata={"letters":letters})
    def build_messages(self,sample:EvaluationSample)->list[dict]:
        c=sample.source_content;block="\n".join(f"{x}. {c['options'][x]}" for x in c["letters"]);return [{"role":"user","content":multiple_answer_prompt(c["question"],block,lang="en")}]
    def parse_response(self,sample:EvaluationSample,raw_response:str)->Any:return parse_multiple_choice_letters(raw_response,sample.metadata.get("letters") or [])
