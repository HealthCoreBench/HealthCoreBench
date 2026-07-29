"""Benchmark registry: maps benchmark names to fixed local directories + adapters.

The registry is the single source of truth for where a benchmark's data lives. Directories
are resolved as ``project_root / benchmark_dir`` so the CWD is irrelevant. Adapters are
referenced by dotted path and imported lazily, so importing the registry never drags in
every adapter's dependencies.

Every benchmark present under ``benchmarks/medical_llm_benchmarks/`` is registered. Benchmarks
whose concrete parser is not implemented yet have ``adapter_dotted=None`` — requesting them
raises ``BenchmarkFormatNotImplementedError`` with a clear message rather than guessing a
format.

Two annotations keep known data problems attached to the entry instead of to a changelog:
``overlap_note`` records that a task's items are a copy of another registered task's, and
``disabled_reason`` says why ``enabled=False``. Disabled tasks stay registered and runnable by
explicit key, but are skipped by bare-name/``ALL`` expansion and are absent from the shipped
configs.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from healthcorebench.benchmarks.errors import (
    BenchmarkNotRegisteredError,
    BenchmarkFormatNotImplementedError,
)
from healthcorebench.config import get_project_root

# Roots, relative to project root, holding the fixed benchmark data directories.
BENCHMARK_ROOT = "benchmarks/medical_llm_benchmarks"
VLM_BENCHMARK_ROOT = "benchmarks/medical_vlm_benchmarks"


class BenchmarkRegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    benchmark_name: str           # bare benchmark name, e.g. "MMLU"
    task: str | None = None       # task suffix, e.g. "mcqa"; None for not-yet-implemented placeholders
    benchmark_dir: str            # relative to project root
    adapter_dotted: str | None    # "module:ClassName" or None if not implemented
    component: str | None = None
    enabled: bool = True
    # Known content overlap with *another registry key*. Free text, by convention starting with
    # the other key(s), e.g. "MedQA_USMLE/mcqa: same 1,273 test items." Set on the redundant task,
    # not on the canonical one, so a reader of a score table can tell why two numbers move
    # together. An entry may carry an overlap note and still be enabled (partial overlap).
    overlap_note: str | None = None
    # Why ``enabled`` is False. Required in spirit whenever ``enabled=False``: a task is only ever
    # disabled because its scores would be invalid or redundant, and that has to be inspectable.
    disabled_reason: str | None = None

    @property
    def key(self) -> str:
        """Registry key: ``<benchmark>/<task>`` when a task is set, else the bare name."""
        return f"{self.benchmark_name}/{self.task}" if self.task else self.benchmark_name

    def directory(self) -> Path:
        return get_project_root() / self.benchmark_dir


# Concrete adapters implemented this iteration.
_IMPLEMENTED = {
    "MMLU/mcqa": ("1_MMLU", "healthcorebench.benchmarks.adapters.mmlu_mcqa:MMLUAdapter", "Language"),
    "PubMedQA/classification": ("2_PubMedQA", "healthcorebench.benchmarks.adapters.pubmedqa_classification:PubMedQAAdapter", "Language"),
    "MedMCQA/mcqa": ("3_MedMCQA", "healthcorebench.benchmarks.adapters.medmcqa_mcqa:MedMCQAAdapter", "Language"),
    "MedQA_USMLE/mcqa": ("69_MedQA-USMLE", "healthcorebench.benchmarks.adapters.medqa_usmle_mcqa:MedQAUSMLEAdapter", "Language"),
    "MedQA_MCMLE/mcqa": ("70_MedQA-MCMLE", "healthcorebench.benchmarks.adapters.medqa_mcmle_mcqa:MedQAMCMLEAdapter", "Language"),
    "ReDis-QA/mcqa": ("14_ReDis-QA", "healthcorebench.benchmarks.adapters.redis_qa_mcqa:ReDisQAAdapter", "Language"),
    "CMMLU/mcqa": ("20_CMMLU", "healthcorebench.benchmarks.adapters.cmmlu_mcqa:CMMLUAdapter", "Language"),
    "Meta-MedQA/mcqa": ("24_Meta-MedQA", "healthcorebench.benchmarks.adapters.meta_medqa_mcqa:MetaMedQAAdapter", "Language"),
    "MMLU-Pro_Health/mcqa": ("61_MMLU-Pro_Health", "healthcorebench.benchmarks.adapters.mmlu_pro_health_mcqa:MMLUProHealthAdapter", "Language"),
    "DiagnosisArena/mcqa": ("51_DiagnosisArena", "healthcorebench.benchmarks.adapters.diagnosisarena_mcqa:DiagnosisArenaAdapter", "Language"),
    "KorMedMCQA/mcqa": ("38_KorMedMCQA", "healthcorebench.benchmarks.adapters.kormedmcqa_mcqa:KorMedMCQAAdapter", "Language"),
    "mARC/mcqa": ("63_mARC", "healthcorebench.benchmarks.adapters.marc_mcqa:MARCAdapter", "Language"),
    "MedExQA/mcqa": ("25_MedExQA", "healthcorebench.benchmarks.adapters.medexqa_mcqa:MedExQAAdapter", "Language"),
    "EHRNoteQA/mcqa": ("46_EHRNoteQA", "healthcorebench.benchmarks.adapters.ehrnoteqa_mcqa:EHRNoteQAAdapter", "Language"),
    "Medbullets/mcqa": ("4_Medbullets", "healthcorebench.benchmarks.adapters.medbullets_mcqa:MedbulletsAdapter", "Language"),
    "SuperGPQA/mcqa": ("6_SuperGPQA", "healthcorebench.benchmarks.adapters.supergpqa_mcqa:SuperGPQAAdapter", "Language"),
    "MedXpertQA_Text/mcqa": ("5_MedXpertQA_Text", "healthcorebench.benchmarks.adapters.medxpertqa_text_mcqa:MedXpertQATextAdapter", "Language"),
    "JEMD/mcqa": ("64_JEMD", "healthcorebench.benchmarks.adapters.jemd_mcqa:JEMDAdapter", "Language"),
    "GlobalDentBench/mcqa": ("71_GlobalDentBench", "healthcorebench.benchmarks.adapters.globaldentbench_mcqa:GlobalDentBenchMCQAAdapter", "Language"),
    "GlobalDentBench/cbq": ("71_GlobalDentBench", "healthcorebench.benchmarks.adapters.globaldentbench_cbq:GlobalDentBenchCBQAdapter", "Language"),
    "GlobalDentBench/multiple_answer": ("71_GlobalDentBench", "healthcorebench.benchmarks.adapters.globaldentbench_multiple_answer:GlobalDentBenchMultipleAnswerAdapter", "Language"),
    "MMedBench/mcqa": ("16_MMedBench", "healthcorebench.benchmarks.adapters.mmedbench_mcqa:MMedBenchAdapter", "Language"),
    "MMedBench/multiple_answer": ("16_MMedBench", "healthcorebench.benchmarks.adapters.mmedbench_multiple_answer:MMedBenchMultipleAnswerAdapter", "Language"),
    "HEAD-QA_v2/mcqa": ("19_HEAD-QA_v2", "healthcorebench.benchmarks.adapters.head_qa_mcqa:HEADQAAdapter", "Language"),
    "HEAD-QA_v2/mcqa_es": ("19_HEAD-QA_v2", "healthcorebench.benchmarks.adapters.head_qa_mcqa:HEADQAAdapter", "Language"),
    "HEAD-QA_v2/mcqa_gl": ("19_HEAD-QA_v2", "healthcorebench.benchmarks.adapters.head_qa_mcqa:HEADQAAdapter", "Language"),
    "HEAD-QA_v2/mcqa_it": ("19_HEAD-QA_v2", "healthcorebench.benchmarks.adapters.head_qa_mcqa:HEADQAAdapter", "Language"),
    "HEAD-QA_v2/mcqa_ru": ("19_HEAD-QA_v2", "healthcorebench.benchmarks.adapters.head_qa_mcqa:HEADQAAdapter", "Language"),
    "MedExpQA/mcqa": ("42_MedExpQA", "healthcorebench.benchmarks.adapters.medexpqa_mcqa:MedExpQAAdapter", "Language"),
    "GPQA/mcqa": ("17_GPQA", "healthcorebench.benchmarks.adapters.gpqa_mcqa:GPQAAdapter", "Language"),
    "MedicationQA/open": ("26_MedicationQA", "healthcorebench.benchmarks.adapters.medicationqa_open:MedicationQAOpenAdapter", "Language"),
    "MeQSum/summarization": ("37_MeQSum", "healthcorebench.benchmarks.adapters.meqsum_summarization:MeQSumSummarizationAdapter", "Language"),
    "webMedQA/open": ("58_webMedQA", "healthcorebench.benchmarks.adapters.webmedqa_open:WebMedQAOpenAdapter", "Language"),
    "RJUA-QA/open": ("57_RJUA-QA", "healthcorebench.benchmarks.adapters.rjua_qa_open:RJUAQAOpenAdapter", "Language"),
    "CareQA/mcqa": ("10_CareQA", "healthcorebench.benchmarks.adapters.careqa_mcqa:CareQAAdapter", "Language"),
    "PubHealthBench/mcqa": ("62_PubHealthBench", "healthcorebench.benchmarks.adapters.pubhealthbench_mcqa:PubHealthBenchAdapter", "Language"),
    # One task per distractor difficulty rather than one pooled task. The three levels ask about
    # the same concepts (99.9% overlapping correct-answer sets), so the pooled accuracy scored
    # every concept three times and buried the axis the benchmark exists to vary.
    "MedConceptsQA/mcqa_easy": ("21_MedConceptsQA", "healthcorebench.benchmarks.adapters.medconceptsqa_mcqa:MedConceptsQAEasyAdapter", "Language"),
    "MedConceptsQA/mcqa_medium": ("21_MedConceptsQA", "healthcorebench.benchmarks.adapters.medconceptsqa_mcqa:MedConceptsQAMediumAdapter", "Language"),
    "MedConceptsQA/mcqa_hard": ("21_MedConceptsQA", "healthcorebench.benchmarks.adapters.medconceptsqa_mcqa:MedConceptsQAHardAdapter", "Language"),
    "AfriMedQA_v2/mcqa": ("7_AfriMedQA_v2", "healthcorebench.benchmarks.adapters.afrimedqa_mcqa:AfriMedQAAdapter", "Language"),
    "AfriMedQA_v2/multiple_answer": ("7_AfriMedQA_v2", "healthcorebench.benchmarks.adapters.afrimedqa_multiple_answer:AfriMedQAMultipleAnswerAdapter", "Language"),
    "AfriMedQA_v2/open": ("7_AfriMedQA_v2", "healthcorebench.benchmarks.adapters.afrimedqa_open:AfriMedQAOpenAdapter", "Language"),
    "CMExam/mcqa": ("56_CMExam", "healthcorebench.benchmarks.adapters.cmexam_mcqa:CMExamAdapter", "Language"),
    "CMExam/multiple_answer": ("56_CMExam", "healthcorebench.benchmarks.adapters.cmexam_mcqa:CMExamMultipleAnswerAdapter", "Language"),
    "FrenchMedMCQA/mcqa": ("18_FrenchMedMCQA", "healthcorebench.benchmarks.adapters.frenchmedmcqa_mcqa:FrenchMedMCQAAdapter", "Language"),
    "MedNLI/nli": ("12_MedNLI", "healthcorebench.benchmarks.adapters.mednli_classification:MedNLIAdapter", "Language"),
    "Swedish_Medical_LLM_Benchmark/mcqa": ("49_Swedish_Medical_LLM_Benchmark", "healthcorebench.benchmarks.adapters.swedish_medqa_mcqa:SwedishMedQAAdapter", "Language"),
    "Swedish_Medical_LLM_Benchmark/specialist_mcqa": ("49_Swedish_Medical_LLM_Benchmark", "healthcorebench.benchmarks.adapters.swedish_tasks:SwedishSpecialistMCQAAdapter", "Language"),
    "Swedish_Medical_LLM_Benchmark/clinical_case_mcqa": ("49_Swedish_Medical_LLM_Benchmark", "healthcorebench.benchmarks.adapters.swedish_tasks:SwedishClinicalCaseMCQAAdapter", "Language"),
    "Swedish_Medical_LLM_Benchmark/theory_mcqa": ("49_Swedish_Medical_LLM_Benchmark", "healthcorebench.benchmarks.adapters.swedish_tasks:SwedishTheoryExamAdapter", "Language"),
    "MedArabiQ/mcqa": ("48_MedArabiQ", "healthcorebench.benchmarks.adapters.medarabiq_mcqa:MedArabiQAdapter", "Language"),
    "MedArabiQ/bias_mcqa": ("48_MedArabiQ", "healthcorebench.benchmarks.adapters.medarabiq_tasks:MedArabiQBiasMCQAAdapter", "Language"),
    "MedArabiQ/fill_choice": ("48_MedArabiQ", "healthcorebench.benchmarks.adapters.medarabiq_tasks:MedArabiQFillChoiceAdapter", "Language"),
    "MedArabiQ/fill_open": ("48_MedArabiQ", "healthcorebench.benchmarks.adapters.medarabiq_tasks:MedArabiQFillOpenAdapter", "Language"),
    "MedArabiQ/patient_qa": ("48_MedArabiQ", "healthcorebench.benchmarks.adapters.medarabiq_tasks:MedArabiQPatientQAAdapter", "Language"),
    "MedArabiQ/patient_qa_llm": ("48_MedArabiQ", "healthcorebench.benchmarks.adapters.medarabiq_tasks:MedArabiQPatientLLMQAAdapter", "Language"),
    "MedArabiQ/patient_qa_gec": ("48_MedArabiQ", "healthcorebench.benchmarks.adapters.medarabiq_tasks:MedArabiQPatientGECQAAdapter", "Language"),
    "JMedBench/mcqa": ("40_JMedBench", "healthcorebench.benchmarks.adapters.jmedbench_mcqa:JMedBenchAdapter", "Language"),
    "JMedBench/crade": ("40_JMedBench", "healthcorebench.benchmarks.adapters.jmedbench_mcqa:JMedBenchAdapter", "Language"),
    "JMedBench/medmcqa_jp": ("40_JMedBench", "healthcorebench.benchmarks.adapters.jmedbench_mcqa:JMedBenchAdapter", "Language"),
    "JMedBench/rrtnm": ("40_JMedBench", "healthcorebench.benchmarks.adapters.jmedbench_mcqa:JMedBenchAdapter", "Language"),
    "JMedBench/smdis": ("40_JMedBench", "healthcorebench.benchmarks.adapters.jmedbench_mcqa:JMedBenchAdapter", "Language"),
    "CMB/mcqa": ("8_CMB", "healthcorebench.benchmarks.adapters.cmb_mcqa:CMBAdapter", "Language"),
    "CMB/multiple_answer": ("8_CMB", "healthcorebench.benchmarks.adapters.cmb_mcqa:CMBMultipleAnswerAdapter", "Language"),
    "CMB/open": ("8_CMB", "healthcorebench.benchmarks.adapters.cmb_open:CMBOpenAdapter", "Language"),
    "PrinciplismQA/mcqa": ("67_PrinciplismQA", "healthcorebench.benchmarks.adapters.principlismqa_mcqa:PrinciplismQAAdapter", "Language"),
    "PrinciplismQA/open": ("67_PrinciplismQA", "healthcorebench.benchmarks.adapters.principlismqa_open:PrinciplismQAOpenAdapter", "Language"),
    "PrinciplismQA/rubric": ("67_PrinciplismQA", "healthcorebench.benchmarks.adapters.principlismqa_open:PrinciplismQARubricAdapter", "Language"),
    "IgakuQA/mcqa": ("39_IgakuQA", "healthcorebench.benchmarks.adapters.igakuqa_mcqa:IgakuQAAdapter", "Language"),
    "IgakuQA/multiple_answer": ("39_IgakuQA", "healthcorebench.benchmarks.adapters.igakuqa_mcqa:IgakuQAMultipleAnswerAdapter", "Language"),
    "BioASQ/yesno": ("43_BioASQ", "healthcorebench.benchmarks.adapters.bioasq_yesno:BioASQYesNoAdapter", "Language"),
    "BioASQ/summary": ("43_BioASQ", "healthcorebench.benchmarks.adapters.bioasq_long_open:BioASQSummaryAdapter", "Language"),
    "BioASQ/list": ("43_BioASQ", "healthcorebench.benchmarks.adapters.bioasq_long_open:BioASQListAdapter", "Language"),
    "MedQuAD/open": ("54_MedQuAD", "healthcorebench.benchmarks.adapters.medquad_open:MedQuADOpenAdapter", "Language"),
    "CareQA/open": ("10_CareQA", "healthcorebench.benchmarks.adapters.careqa_open:CareQAOpenAdapter", "Language"),
    "MedThink-Bench/open": ("53_MedThink-Bench", "healthcorebench.benchmarks.adapters.medthink_bench_open:MedThinkBenchOpenAdapter", "Language"),
    "MedCaseReasoning/open": ("31_MedCaseReasoning", "healthcorebench.benchmarks.adapters.medcasereasoning_open:MedCaseReasoningOpenAdapter", "Language"),
    "LLMEval-Med/open": ("35_LLMEval-Med", "healthcorebench.benchmarks.adapters.llmeval_med_open:LLMEvalMedOpenAdapter", "Language"),
    "ClinicBench/mcqa": ("44_ClinicBench", "healthcorebench.benchmarks.adapters.clinicbench_pharmacology_mcqa:ClinicBenchPharmacologyAdapter", "Language"),
    "ClinicBench/patient_education": ("44_ClinicBench", "healthcorebench.benchmarks.adapters.clinicbench_open_tasks:ClinicBenchPatientEducationAdapter", "Language"),
    "ClinicBench/treatment": ("44_ClinicBench", "healthcorebench.benchmarks.adapters.clinicbench_open_tasks:ClinicBenchTreatmentAdapter", "Language"),
    "ClinicBench/hospitalization": ("44_ClinicBench", "healthcorebench.benchmarks.adapters.clinicbench_open_tasks:ClinicBenchHospitalizationAdapter", "Language"),
    "ClinicBench/drug_interaction": ("44_ClinicBench", "healthcorebench.benchmarks.adapters.clinicbench_open_tasks:ClinicBenchDrugInteractionAdapter", "Language"),
    "EHRBench/risk": ("47_EHRBench", "healthcorebench.benchmarks.adapters.ehrbench_risk_yesno:EHRBenchRiskYesNoAdapter", "Language"),
    "EHRBench/decision": ("47_EHRBench", "healthcorebench.benchmarks.adapters.ehrbench_decision_open:EHRBenchDecisionAdapter", "Language"),
    "Med-HALT/reasoning": ("60_Med-HALT", "healthcorebench.benchmarks.adapters.medhalt_reasoning_mcqa:MedHALTReasoningAdapter", "Language"),
    "Med-HALT/reasoning_nota": ("60_Med-HALT", "healthcorebench.benchmarks.adapters.medhalt_reasoning_mcqa:MedHALTReasoningAdapter", "Language"),
    "GeneTuring/open": ("33_GeneTuring", "healthcorebench.benchmarks.adapters.geneturing_open:GeneTuringOpenAdapter", "Language"),
    "BioASQ/factoid": ("43_BioASQ", "healthcorebench.benchmarks.adapters.bioasq_factoid_open:BioASQFactoidOpenAdapter", "Language"),
    "MedHallu/detection": ("23_MedHallu", "healthcorebench.benchmarks.adapters.medhallu_detection:MedHalluDetectionAdapter", "Language"),
    "LiveQA/open": ("9_LiveQA", "healthcorebench.benchmarks.adapters.liveqa_open:LiveQAOpenAdapter", "Language"),
    "ACI-Bench_HF/summarization": ("29_ACI-Bench_HF", "healthcorebench.benchmarks.adapters.aci_bench_summarization:ACIBenchSummarizationAdapter", "Language"),
    "ClinicalBench/mortality": ("65_ClinicalBench", "healthcorebench.benchmarks.adapters.clinicalbench_classification:ClinicalBenchAdapter", "Language"),
    "ClinicalBench/readmission": ("65_ClinicalBench", "healthcorebench.benchmarks.adapters.clinicalbench_classification:ClinicalBenchAdapter", "Language"),
    "ClinicalBench/length_of_stay": ("65_ClinicalBench", "healthcorebench.benchmarks.adapters.clinicalbench_classification:ClinicalBenchAdapter", "Language"),
    "MEDEC/detection": ("28_MEDEC", "healthcorebench.benchmarks.adapters.medec_detection:MEDECDetectionAdapter", "Language"),
    "MEDEC/correction": ("28_MEDEC", "healthcorebench.benchmarks.adapters.medec_correction:MEDECCorrectionAdapter", "Language"),
    "MedS-Bench/task1": ("36_MedS-Bench", "healthcorebench.benchmarks.adapters.medsbench_open:MedSBenchOpenAdapter", "Language"),
    "MedS-Bench/task2": ("36_MedS-Bench", "healthcorebench.benchmarks.adapters.medsbench_open:MedSBenchOpenAdapter", "Language"),
    "MedS-Bench/task3": ("36_MedS-Bench", "healthcorebench.benchmarks.adapters.medsbench_open:MedSBenchOpenAdapter", "Language"),
    "MedS-Bench/task12": ("36_MedS-Bench", "healthcorebench.benchmarks.adapters.medsbench_open:MedSBenchOpenAdapter", "Language"),
    "MedS-Bench/task16": ("36_MedS-Bench", "healthcorebench.benchmarks.adapters.medsbench_open:MedSBenchOpenAdapter", "Language"),
    "MedS-Bench/task18": ("36_MedS-Bench", "healthcorebench.benchmarks.adapters.medsbench_open:MedSBenchOpenAdapter", "Language"),
    "MedS-Bench/task29": ("36_MedS-Bench", "healthcorebench.benchmarks.adapters.medsbench_open:MedSBenchOpenAdapter", "Language"),
    "MedS-Bench/task46": ("36_MedS-Bench", "healthcorebench.benchmarks.adapters.medsbench_open:MedSBenchOpenAdapter", "Language"),
    "MedS-Bench/task50": ("36_MedS-Bench", "healthcorebench.benchmarks.adapters.medsbench_open:MedSBenchOpenAdapter", "Language"),
    "MedS-Bench/task57": ("36_MedS-Bench", "healthcorebench.benchmarks.adapters.medsbench_mcqa:MedSBenchMCQAAdapter", "Language"),
    "MedS-Bench/task58": ("36_MedS-Bench", "healthcorebench.benchmarks.adapters.medsbench_mcqa:MedSBenchMCQAAdapter", "Language"),
    "MedS-Bench/task59": ("36_MedS-Bench", "healthcorebench.benchmarks.adapters.medsbench_mcqa:MedSBenchMCQAAdapter", "Language"),
    "MedS-Bench/task60": ("36_MedS-Bench", "healthcorebench.benchmarks.adapters.medsbench_mcqa:MedSBenchMCQAAdapter", "Language"),
    "MedS-Bench/task61": ("36_MedS-Bench", "healthcorebench.benchmarks.adapters.medsbench_mcqa:MedSBenchMCQAAdapter", "Language"),
    "MedS-Bench/task74": ("36_MedS-Bench", "healthcorebench.benchmarks.adapters.medsbench_open:MedSBenchOpenAdapter", "Language"),
    "MedS-Bench/task100": ("36_MedS-Bench", "healthcorebench.benchmarks.adapters.medsbench_open:MedSBenchOpenAdapter", "Language"),
    "MedS-Bench/task106": ("36_MedS-Bench", "healthcorebench.benchmarks.adapters.medsbench_open:MedSBenchOpenAdapter", "Language"),
    "MedS-Bench/task122": ("36_MedS-Bench", "healthcorebench.benchmarks.adapters.medsbench_open:MedSBenchOpenAdapter", "Language"),
    "MedS-Bench/task123": ("36_MedS-Bench", "healthcorebench.benchmarks.adapters.medsbench_open:MedSBenchOpenAdapter", "Language"),
    "MedS-Bench/task125": ("36_MedS-Bench", "healthcorebench.benchmarks.adapters.medsbench_open:MedSBenchOpenAdapter", "Language"),
    "MedS-Bench/task126": ("36_MedS-Bench", "healthcorebench.benchmarks.adapters.medsbench_open:MedSBenchOpenAdapter", "Language"),
    "MedS-Bench/task127": ("36_MedS-Bench", "healthcorebench.benchmarks.adapters.medsbench_open:MedSBenchOpenAdapter", "Language"),
    "MedS-Bench/task128": ("36_MedS-Bench", "healthcorebench.benchmarks.adapters.medsbench_open:MedSBenchOpenAdapter", "Language"),
    "MedS-Bench/task129": ("36_MedS-Bench", "healthcorebench.benchmarks.adapters.medsbench_mcqa:MedSBenchMCQAAdapter", "Language"),
    "MedS-Bench/task130": ("36_MedS-Bench", "healthcorebench.benchmarks.adapters.medsbench_open:MedSBenchOpenAdapter", "Language"),
    "MedS-Bench/task131": ("36_MedS-Bench", "healthcorebench.benchmarks.adapters.medsbench_open:MedSBenchOpenAdapter", "Language"),
    "MedCalc-Bench/calculation": ("59_MedCalc-Bench", "healthcorebench.benchmarks.adapters.medcalc_bench_calculation:MedCalcBenchAdapter", "Language"),
    "IOR-Bench/triage": ("66_IOR-Bench", "healthcorebench.benchmarks.adapters.ior_bench_triage:IORBenchTriageAdapter", "Language"),
    "IOR-Bench/dynamic": ("66_IOR-Bench", "healthcorebench.benchmarks.adapters.ior_bench_dynamic:IORBenchDynamicAdapter", "Language"),
    "MIMIC-CDM/diagnosis": ("45_MIMIC-CDM", "healthcorebench.benchmarks.adapters.mimic_cdm_diagnosis:MIMICCDMDiagnosisAdapter", "Language"),
    "MedSafetyBench/safety": ("41_MedSafetyBench", "healthcorebench.benchmarks.adapters.medsafetybench_safety:MedSafetyBenchAdapter", "Language"),
    "MedQA-CS/open": ("34_MedQA-CS", "healthcorebench.benchmarks.adapters.medqa_cs_open:MedQACSOpenAdapter", "Language"),
    "VivaBench/diagnosis": ("68_VivaBench", "healthcorebench.benchmarks.adapters.vivabench_diagnosis_open:VivaBenchDiagnosisAdapter", "Language"),
    "LongHealth/mcqa": ("22_LongHealth", "healthcorebench.benchmarks.adapters.longhealth_mcqa:LongHealthMCQAAdapter", "Language"),
    "RareBench/diagnosis": ("15_RareBench", "healthcorebench.benchmarks.adapters.rarebench_diagnosis_open:RareBenchDiagnosisAdapter", "Language"),
    "HLE_med/mcqa": ("11_HLE_med", "healthcorebench.benchmarks.adapters.hle_med_mcqa:HLEMedMCQAAdapter", "Language"),
    "HLE_med/exact": ("11_HLE_med", "healthcorebench.benchmarks.adapters.hle_med_exact:HLEMedExactAdapter", "Language"),
    "MediQ/mcqa": ("13_MediQ", "healthcorebench.benchmarks.adapters.mediq_mcqa:MediQMCQAAdapter", "Language"),
    "AgentClinic/diagnosis": ("30_AgentClinic", "healthcorebench.benchmarks.adapters.agentclinic_diagnosis_open:AgentClinicDiagnosisAdapter", "Language"),
    "BioHopR/single": ("50_BioHopR", "healthcorebench.benchmarks.adapters.biohopr_single_open:BioHopRSingleOpenAdapter", "Language"),
    "BioHopR/multi": ("50_BioHopR", "healthcorebench.benchmarks.adapters.biohopr_multi_open:BioHopRMultiOpenAdapter", "Language"),
    "MedBrowseComp/open": ("52_MedBrowseComp", "healthcorebench.benchmarks.adapters.medbrowsecomp_open:MedBrowseCompOpenAdapter", "Language"),
    "SCTPublic/likert": ("32_SCTPublic", "healthcorebench.benchmarks.adapters.sctpublic_likert:SCTPublicLikertAdapter", "Language"),
    "MedChain/diagnosis": ("55_MedChain", "healthcorebench.benchmarks.adapters.medchain_diagnosis_open:MedChainDiagnosisAdapter", "Language"),
    "MedR-Bench/diagnosis": ("27_MedR-Bench", "healthcorebench.benchmarks.adapters.medrbench_diagnosis_open:MedRBenchDiagnosisAdapter", "Language"),
    "MedR-Bench/treatment": ("27_MedR-Bench", "healthcorebench.benchmarks.adapters.medrbench_treatment_open:MedRBenchTreatmentAdapter", "Language"),
}

# Known content overlap between registry keys, recorded on the *redundant* task. Several
# benchmarks in this suite repackage another one's items (translations, re-releases, task
# collections such as MedS-Bench), and a macro average over both silently double-weights those
# items. Keeping the relation in the registry means it travels with the score instead of living
# in someone's memory. An overlap note does not by itself disable a task — see ``_DISABLED``.
_OVERLAP_NOTES = {
    "JMedBench/medmcqa_jp": (
        "MedMCQA/mcqa: a machine translation of exactly the same 4,183 test records "
        "(sample_id sets are identical), so it measures MedMCQA in Japanese, not new content."
    ),
    "MedS-Bench/task57": (
        "MedQA_USMLE/mcqa: the same 1,273 MedQA (USMLE) test items, repackaged."
    ),
    "MedS-Bench/task58": (
        "MedQA_MCMLE/mcqa: the same 3,426 MedQA (MCMLE) test items, repackaged."
    ),
    "MedS-Bench/task59": (
        "IgakuQA/mcqa: 199 items sampled from the same Japanese national exam set."
    ),
    "MedS-Bench/task60": (
        "FrenchMedMCQA/mcqa: the same 622 test items; MedS-Bench additionally flattens them to "
        "single-choice, while FrenchMedMCQA keeps the multi-answer items as such."
    ),
    "MedS-Bench/task129": (
        "HEAD-QA_v2/mcqa_es: the same Spanish HEAD-QA test set, and a strictly worse copy — it "
        "keeps all 2,742 records including the 67 that only make sense with the exam figure, "
        "which the text-only task drops."
    ),
    # MMedBench stays enabled: 3,158 of its 8,178 single-answer items are genuinely its own
    # (Russian, Spanish, Japanese), so disabling it would lose that coverage. But the majority is
    # not new content, and a reader comparing MMedBench against MedQA_USMLE needs to know they
    # are largely the same questions rather than two independent measurements.
    "MMedBench/mcqa": (
        "MedQA_USMLE/mcqa (1,273 items), MedQA_MCMLE/mcqa (3,426) and FrenchMedMCQA/mcqa (321): "
        "5,020 of 8,178 items (61%) are verbatim the same questions, because MMedBench's English, "
        "Chinese and French splits are those benchmarks. Only the Russian, Spanish and Japanese "
        "splits (3,158 items) are content this suite does not already score elsewhere."
    ),
    "MMedBench/multiple_answer": (
        "FrenchMedMCQA/mcqa: 301 of 340 items (89%) are the same French multi-answer questions. "
        "MMedBench's other splits contribute almost no multi-answer items."
    ),
}

# Tasks that stay registered (so they remain inspectable and runnable by explicit key) but are
# kept out of the shipped configs and of bare-name/ALL expansion, because their scores would be
# invalid or pure duplicate coverage. Every entry must say why.
_DISABLED = {
    "EHRNoteQA/mcqa": (
        "Unanswerable as shipped: every question asks about a patient's MIMIC-IV discharge notes "
        "('based on the discharge summary above, ...'), but ehrnoteqa_test.jsonl holds only the "
        "question and its five choices — the notes need credentialed PhysioNet access and are "
        "nowhere under benchmarks/. Scoring it measures prior-guessing over 5 options, not "
        "clinical reading. To re-enable: add the note text to the source file and render it in "
        "ehrnoteqa_mcqa.py, budgeting the prompt with context_window.fit_context_to_window."
    ),
    "JMedBench/medmcqa_jp": (
        "100% duplicate of MedMCQA/mcqa (see overlap_note). It used to be the only JMedBench "
        "subset evaluated, which made the suite's contribution to the macro average a second "
        "copy of MedMCQA; JMedBench/mcqa now scores jmmlu_medical instead."
    ),
    "MedS-Bench/task57": "Duplicate of MedQA_USMLE/mcqa (see overlap_note).",
    "MedS-Bench/task58": "Duplicate of MedQA_MCMLE/mcqa (see overlap_note).",
    "MedS-Bench/task59": "Duplicate of IgakuQA/mcqa (see overlap_note).",
    "MedS-Bench/task60": "Duplicate of FrenchMedMCQA/mcqa (see overlap_note).",
    "MedS-Bench/task129": "Duplicate of, and lower quality than, HEAD-QA_v2/mcqa_es (see overlap_note).",
}

# All benchmark directories present in the data tree. Name -> directory. Names use the
# canonical benchmark name (directory number prefix stripped, dashes normalized).
_ALL_DIRS = {
    "MMLU": "1_MMLU", "PubMedQA": "2_PubMedQA", "MedMCQA": "3_MedMCQA",
    "Medbullets": "4_Medbullets", "MedXpertQA_Text": "5_MedXpertQA_Text",
    "SuperGPQA": "6_SuperGPQA", "AfriMedQA_v2": "7_AfriMedQA_v2", "CMB": "8_CMB",
    "LiveQA": "9_LiveQA", "CareQA": "10_CareQA",
    "HLE_med": "11_HLE_med", "MedNLI": "12_MedNLI", "MediQ": "13_MediQ", "ReDis-QA": "14_ReDis-QA",
    "RareBench": "15_RareBench", "MMedBench": "16_MMedBench", "GPQA": "17_GPQA",
    "FrenchMedMCQA": "18_FrenchMedMCQA", "HEAD-QA_v2": "19_HEAD-QA_v2", "CMMLU": "20_CMMLU",
    "MedConceptsQA": "21_MedConceptsQA", "LongHealth": "22_LongHealth", "MedHallu": "23_MedHallu",
    "Meta-MedQA": "24_Meta-MedQA", "MedExQA": "25_MedExQA", "MedicationQA": "26_MedicationQA",
    "MedR-Bench": "27_MedR-Bench", "MEDEC": "28_MEDEC", "ACI-Bench_HF": "29_ACI-Bench_HF",
    "AgentClinic": "30_AgentClinic",
    "MedCaseReasoning": "31_MedCaseReasoning", "SCTPublic": "32_SCTPublic",
    "GeneTuring": "33_GeneTuring", "MedQA-CS": "34_MedQA-CS", "LLMEval-Med": "35_LLMEval-Med",
    "MedS-Bench": "36_MedS-Bench", "MeQSum": "37_MeQSum", "KorMedMCQA": "38_KorMedMCQA",
    "IgakuQA": "39_IgakuQA", "JMedBench": "40_JMedBench", "MedSafetyBench": "41_MedSafetyBench",
    "MedExpQA": "42_MedExpQA",
    "BioASQ": "43_BioASQ", "ClinicBench": "44_ClinicBench", "MIMIC-CDM": "45_MIMIC-CDM",
    "EHRNoteQA": "46_EHRNoteQA", "EHRBench": "47_EHRBench",
    "MedArabiQ": "48_MedArabiQ",
    "Swedish_Medical_LLM_Benchmark": "49_Swedish_Medical_LLM_Benchmark", "BioHopR": "50_BioHopR",
    "DiagnosisArena": "51_DiagnosisArena", "MedBrowseComp": "52_MedBrowseComp",
    "MedThink-Bench": "53_MedThink-Bench", "MedQuAD": "54_MedQuAD", "MedChain": "55_MedChain",
    "CMExam": "56_CMExam", "RJUA-QA": "57_RJUA-QA", "webMedQA": "58_webMedQA",
    "MedCalc-Bench": "59_MedCalc-Bench", "Med-HALT": "60_Med-HALT",
    "MMLU-Pro_Health": "61_MMLU-Pro_Health", "PubHealthBench": "62_PubHealthBench",
    "mARC": "63_mARC", "JEMD": "64_JEMD", "ClinicalBench": "65_ClinicalBench",
    "IOR-Bench": "66_IOR-Bench", "PrinciplismQA": "67_PrinciplismQA",
    "VivaBench": "68_VivaBench", "MedQA_USMLE": "69_MedQA-USMLE",
    "MedQA_MCMLE": "70_MedQA-MCMLE", "GlobalDentBench": "71_GlobalDentBench",
}


def _build_registry() -> dict[str, BenchmarkRegistryEntry]:
    """Build the registry keyed by task key (``<benchmark>/<task>``).

    Every implemented ``(benchmark, task)`` gets its own entry. A benchmark directory that
    has no implemented task yet gets a single placeholder entry keyed by its bare name, with
    ``adapter_dotted=None`` (requesting it raises ``BenchmarkFormatNotImplementedError``).
    Directories always come from ``_ALL_DIRS`` so the data path is the single source of truth.
    """
    registry: dict[str, BenchmarkRegistryEntry] = {}

    # 1) implemented (benchmark, task) entries.
    benchmarks_with_task: set[str] = set()
    unknown_annotations = (set(_OVERLAP_NOTES) | set(_DISABLED)) - set(_IMPLEMENTED)
    if unknown_annotations:
        raise BenchmarkNotRegisteredError(
            f"_OVERLAP_NOTES/_DISABLED annotate unregistered keys: {sorted(unknown_annotations)}."
        )
    for task_key, (_, dotted, component) in _IMPLEMENTED.items():
        bench, task = task_key.split("/", 1)
        if bench not in _ALL_DIRS:
            raise BenchmarkNotRegisteredError(
                f"_IMPLEMENTED references unknown benchmark '{bench}' (from '{task_key}')."
            )
        rel_dir = _ALL_DIRS[bench]
        registry[task_key] = BenchmarkRegistryEntry(
            benchmark_name=bench, task=task, benchmark_dir=f"{BENCHMARK_ROOT}/{rel_dir}",
            adapter_dotted=dotted, component=component,
            enabled=task_key not in _DISABLED,
            overlap_note=_OVERLAP_NOTES.get(task_key),
            disabled_reason=_DISABLED.get(task_key),
        )
        benchmarks_with_task.add(bench)

    # 2) placeholder entries for benchmarks with no implemented task yet.
    for name, rel_dir in _ALL_DIRS.items():
        if name in benchmarks_with_task:
            continue
        registry[name] = BenchmarkRegistryEntry(
            benchmark_name=name, task=None, benchmark_dir=f"{BENCHMARK_ROOT}/{rel_dir}",
            adapter_dotted=None, component=None, enabled=True,
        )

    # 3) multimodal tasks use one profile-driven adapter while keeping their own fixed dirs.
    # ``ALL`` remains language-only in the CLI; multimodal expansion is exposed as ``ALL_VLM``.
    from healthcorebench.benchmarks.vlm_adapters.catalog import VLM_TASK_SPECS

    for task_key, spec in VLM_TASK_SPECS.items():
        if task_key in registry:
            raise BenchmarkNotRegisteredError(f"Duplicate language/VLM registry key: {task_key}")
        bench, task = task_key.split("/", 1)
        registry[task_key] = BenchmarkRegistryEntry(
            benchmark_name=bench,
            task=task,
            benchmark_dir=f"{VLM_BENCHMARK_ROOT}/{spec.directory}",
            adapter_dotted=(
                "healthcorebench.benchmarks.vlm_adapters.generic:MedicalVLMAdapter"
            ),
            component="Multimodal",
            enabled=True,
        )
    return registry


_REGISTRY: dict[str, BenchmarkRegistryEntry] | None = None


def get_registry() -> dict[str, BenchmarkRegistryEntry]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def resolve_benchmark_keys(name: str) -> list[str]:
    """Resolve a user-supplied benchmark identifier to its task-key(s).

    Users think in benchmarks ("MMLU", "CareQA"), not internal task suffixes. This maps:
      - a full task key ("CareQA/open")  -> [that key]
      - a bare benchmark name ("MMLU")   -> every implemented *enabled* task key of that benchmark,
                                             sorted (e.g. "CareQA" -> ["CareQA/mcqa","CareQA/open"]).
    A bare name with several tasks returns them all, so a plain ``--benchmark CareQA`` run
    evaluates each task. Tasks with ``enabled=False`` are skipped here — they are duplicates or
    unscorable (see ``disabled_reason``) and must not sneak into a sweep — but naming one
    explicitly still resolves, so they stay available for a deliberate one-off run.
    Raises ``BenchmarkNotRegisteredError`` for an unknown identifier and
    ``BenchmarkFormatNotImplementedError`` for a registered-but-unimplemented placeholder.
    """
    reg = get_registry()
    # exact task-key match (contains a "/").
    if name in reg:
        entry = reg[name]
        if entry.adapter_dotted is None:
            raise BenchmarkFormatNotImplementedError(
                f"Benchmark '{name}' is registered (data at {entry.benchmark_dir}) but its "
                f"concrete adapter/parser has not been implemented yet."
            )
        return [name]
    # bare benchmark name: gather all implemented, enabled task keys for it.
    keys = sorted(k for k, e in reg.items()
                  if e.benchmark_name == name and e.adapter_dotted is not None and e.enabled)
    if keys:
        return keys
    # bare name whose every task is disabled: say why instead of "not registered".
    disabled = sorted(k for k, e in reg.items()
                      if e.benchmark_name == name and e.adapter_dotted is not None)
    if disabled:
        reasons = "; ".join(f"{k}: {reg[k].disabled_reason}" for k in disabled)
        raise BenchmarkFormatNotImplementedError(
            f"Every task of benchmark '{name}' is disabled ({reasons}). Name a task key "
            f"explicitly ({disabled}) to run one anyway."
        )
    # bare name that exists only as an unimplemented placeholder.
    if any(e.benchmark_name == name for e in reg.values()):
        raise BenchmarkFormatNotImplementedError(
            f"Benchmark '{name}' is registered but has no implemented task/parser yet."
        )
    benches = sorted({e.benchmark_name for e in reg.values()})
    raise BenchmarkNotRegisteredError(
        f"Benchmark '{name}' is not registered. Known benchmarks: {benches}"
    )


def get_entry(name: str) -> BenchmarkRegistryEntry:
    reg = get_registry()
    if name in reg:
        return reg[name]
    # allow a bare benchmark name when it resolves to exactly one implemented task.
    keys = resolve_benchmark_keys(name)
    if len(keys) == 1:
        return reg[keys[0]]
    raise BenchmarkNotRegisteredError(
        f"Benchmark '{name}' maps to multiple tasks {keys}; specify one, "
        f"or use resolve_benchmark_keys() to run them all."
    )


def list_benchmarks() -> list[dict]:
    """Return a summary of all benchmarks (name, dir, implemented?)."""
    out = []
    for key, e in sorted(get_registry().items()):
        out.append({
            "key": key,
            "benchmark_name": e.benchmark_name,
            "task": e.task,
            "benchmark_dir": e.benchmark_dir,
            "implemented": e.adapter_dotted is not None,
            "component": e.component,
            "enabled": e.enabled,
            "disabled_reason": e.disabled_reason,
            "overlap_note": e.overlap_note,
        })
    return out


def get_adapter(name: str, config=None):
    """Instantiate the adapter for ``name``, or raise if its parser is not implemented."""
    entry = get_entry(name)
    if entry.adapter_dotted is None:
        raise BenchmarkFormatNotImplementedError(
            f"Benchmark '{name}' is registered (data at {entry.benchmark_dir}) but its "
            f"concrete adapter/parser has not been implemented yet."
        )
    module_path, class_name = entry.adapter_dotted.split(":")
    module = importlib.import_module(module_path)
    adapter_cls = getattr(module, class_name)
    return adapter_cls(entry=entry, config=config)
