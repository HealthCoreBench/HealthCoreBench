"""Declarative inventory of the fixed medical VLM benchmark tasks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VLMTaskSpec:
    directory: str
    files: tuple[str, ...]
    profile: str


VLM_TASK_SPECS: dict[str, VLMTaskSpec] = {
    "VQA-RAD/closed": VLMTaskSpec("1_VQA-RAD", ("vqa_rad_test.json",), "closed"),
    "VQA-RAD/open": VLMTaskSpec("1_VQA-RAD", ("vqa_rad_test.json",), "short_open"),
    "SLAKE/closed": VLMTaskSpec("2_SLAKE", ("slake_test.json",), "closed"),
    "SLAKE/open": VLMTaskSpec("2_SLAKE", ("slake_test.json",), "short_open"),
    "PathVQA/closed": VLMTaskSpec("3_PathVQA", ("pathvqa_test.json",), "closed"),
    "PathVQA/open": VLMTaskSpec("3_PathVQA", ("pathvqa_test.json",), "short_open"),
    "PMC-VQA/mcqa": VLMTaskSpec("4_PMC-VQA", ("pmc_vqa_test_clean.json",), "closed"),
    "OmniMedVQA/mcqa": VLMTaskSpec(
        "5_OmniMedVQA", ("omnimedvqa_openaccess.json",), "closed"
    ),
    "MedXpertQA/mcqa": VLMTaskSpec(
        "6_MedXpertQA", ("medxpertqa_mm_test.json",), "closed"
    ),
    "MIMIC-CXR/multilabel": VLMTaskSpec(
        "7_MIMIC-CXR", ("mimic_cxr_test.json",), "multilabel"
    ),
    "WorldMedQA-V/mcqa": VLMTaskSpec(
        "8_WorldMedQA-V",
        (
            "worldmedqa_v_brazil_english.json",
            "worldmedqa_v_brazil_local.json",
            "worldmedqa_v_israel_english.json",
            "worldmedqa_v_israel_local.json",
            "worldmedqa_v_japan_english.json",
            "worldmedqa_v_japan_local.json",
            "worldmedqa_v_spain_english.json",
            "worldmedqa_v_spain_local.json",
        ),
        "closed",
    ),
    "IU-Xray/report_generation": VLMTaskSpec(
        "9_IU-Xray", ("iu_xray_test.json",), "report"
    ),
    "SurgeryVideoQA/open": VLMTaskSpec(
        "10_SurgeryVideoQA", ("surgeryvideoqa_test.json",), "video_open"
    ),
    "MedFrameQA/mcqa": VLMTaskSpec(
        "11_MedFrameQA", ("medframeqa_test.json",), "closed"
    ),
    "BiMed-MBench/open_en": VLMTaskSpec(
        "12_BiMed-MBench", ("bimed_mbench_english_test.json",), "generation"
    ),
    "BiMed-MBench/open_ar": VLMTaskSpec(
        "12_BiMed-MBench", ("bimed_mbench_arabic_test.json",), "generation"
    ),
    "DrVD-Bench/independent_mcqa": VLMTaskSpec(
        "13_DrVD-Bench", ("independent_qa_test.json",), "closed"
    ),
    "DrVD-Bench/joint_reasoning": VLMTaskSpec(
        "13_DrVD-Bench", ("joint_qa_test.json",), "multistage"
    ),
    "DrVD-Bench/report_generation": VLMTaskSpec(
        "13_DrVD-Bench", ("report_generation_test.json",), "report"
    ),
    "DrVD-Bench/visual_evidence": VLMTaskSpec(
        "13_DrVD-Bench", ("visual_evidence_qa_test.json",), "closed"
    ),
    "KorMedMCQA-V/mcqa": VLMTaskSpec(
        "14_KorMedMCQA-V", ("kormedmcqa_v_test.json",), "closed"
    ),
    "Kvasir-VQA/open": VLMTaskSpec("15_Kvasir-VQA", ("kvasir_vqa.json",), "short_open"),
    "VQA-Med-2019/open": VLMTaskSpec(
        "16_VQA-Med-2019", ("vqamed2019_test.json",), "short_open"
    ),
    "VQA-Med-2020/open": VLMTaskSpec(
        "17_VQA-Med-2020", ("vqamed2020_test.json",), "short_open"
    ),
    "VQA-Med-2021/open": VLMTaskSpec(
        "18_VQA-Med-2021", ("vqamed2021_test.json",), "short_open"
    ),
    "ROCOv2/caption": VLMTaskSpec("19_ROCOv2", ("rocov2_test.json",), "generation"),
    "ROCOv2/concepts": VLMTaskSpec("19_ROCOv2", ("rocov2_test.json",), "multilabel"),
    "Medical-CXR-VQA/open": VLMTaskSpec(
        "20_Medical-CXR-VQA", ("medical_cxr_vqa_test.json",), "short_open"
    ),
    "Quilt-VQA/closed": VLMTaskSpec(
        "21_Quilt-VQA", ("quiltvqa_test_w_ans.json",), "closed"
    ),
    "Quilt-VQA/open": VLMTaskSpec(
        "21_Quilt-VQA", ("quiltvqa_test_w_ans.json",), "short_open"
    ),
    "PathMMU/mcqa": VLMTaskSpec("22_PathMMU", ("pathmmu_test.json",), "closed"),
    "MMMU-Health-Medicine/mcqa": VLMTaskSpec(
        "23_MMMU-Health-Medicine", ("mmmu_health_medicine_test.json",), "closed"
    ),
    "MMMU-Health-Medicine/open": VLMTaskSpec(
        "23_MMMU-Health-Medicine", ("mmmu_health_medicine_test.json",), "short_open"
    ),
    "PathText/caption": VLMTaskSpec("24_PathText", ("pathtext.json",), "generation"),
    "PathText/report_generation": VLMTaskSpec(
        "24_PathText", ("pathtext.json", "reports/*.txt"), "report"
    ),
    "MIMICEchoQA/mcqa": VLMTaskSpec(
        "25_MIMICEchoQA", ("mimicechoqa_test.json",), "closed"
    ),
    "MTBBench/hancock_mcqa": VLMTaskSpec(
        "26_MTBBench",
        (
            "mtbbench_hancock_test.json",
            "data/hancock/cases/**/*.txt",
            "data/hancock/cases/**/*.json",
        ),
        "closed",
    ),
    "MTBBench/msk_mcqa": VLMTaskSpec(
        "26_MTBBench",
        (
            "mtbbench_msk_test.json",
            "data/msk_bench/cases/**/*.txt",
            "data/msk_bench/cases/**/*.csv",
        ),
        "closed",
    ),
    "MX-CXR/grounding": VLMTaskSpec("27_MX-CXR", ("ms_cxr_test.json",), "grounding"),
    "3MDBench/diagnosis": VLMTaskSpec(
        "28_3MDBench", ("3mdbench_test.json",), "fixed_diagnosis"
    ),
    "MedBookVQA/mcqa": VLMTaskSpec(
        "29_MedBookVQA", ("medbookvqa_test.json",), "closed"
    ),
    "MIMIC-Ext/verify": VLMTaskSpec(
        "30_MIMIC-Ext-MIMIC-CXR-VQA", ("mimic_ext_mimic_cxr_vqa_test.json",), "closed"
    ),
    "MIMIC-Ext/query": VLMTaskSpec(
        "30_MIMIC-Ext-MIMIC-CXR-VQA",
        ("mimic_ext_mimic_cxr_vqa_test.json",),
        "multilabel",
    ),
    "MIMIC-Ext/choose": VLMTaskSpec(
        "30_MIMIC-Ext-MIMIC-CXR-VQA",
        ("mimic_ext_mimic_cxr_vqa_test.json",),
        "fixed_text",
    ),
    "OmniBrainBench/closed": VLMTaskSpec(
        "31_OmniBrainBench", ("omnibrainbench_closed_ended_test.json",), "closed"
    ),
    "OmniBrainBench/open": VLMTaskSpec(
        "31_OmniBrainBench", ("omnibrainbench_open_ended_test.json",), "generation"
    ),
    "HLM/mcqa": VLMTaskSpec("32_HLM", ("hle_med_test_multimodal.json",), "closed"),
    "HLM/exact": VLMTaskSpec("32_HLM", ("hle_med_test_multimodal.json",), "short_open"),
    "AgentClinic-VLM/mcqa": VLMTaskSpec(
        "33_AgentClinic", ("agentclinic_nejm_extended.jsonl",), "closed"
    ),
    "LiveClin/mcqa": VLMTaskSpec(
        "34_LiveClin", ("liveclin_2025_H1_test.json",), "multistage_closed"
    ),
    "MedDocBench/ltr_simple_qa": VLMTaskSpec(
        "35_MedDocBench", ("meddocbench_test.json",), "document_qa"
    ),
    "MedDocBench/ltr_full_parsing": VLMTaskSpec(
        "35_MedDocBench", ("meddocbench_test.json",), "document_parse"
    ),
    "MedDocBench/ltr_abnormality_qa": VLMTaskSpec(
        "35_MedDocBench", ("meddocbench_test.json",), "document_parse"
    ),
    "MedDocBench/gmd_simple_qa": VLMTaskSpec(
        "35_MedDocBench", ("meddocbench_test.json",), "document_qa"
    ),
    "MedDocBench/gmd_complex_qa": VLMTaskSpec(
        "35_MedDocBench", ("meddocbench_test.json",), "document_complex_qa"
    ),
    "GMAI-MMBench/mcqa": VLMTaskSpec(
        "36_GMAI-MMBench", ("GMAI_mm_bench_VAL.json",), "closed"
    ),
}


VLM_BENCHMARK_DIRS = {
    key.split("/", 1)[0]: spec.directory for key, spec in VLM_TASK_SPECS.items()
}
