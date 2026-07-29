"""Real-data smoke coverage for every registered medical VLM task."""

from __future__ import annotations

import json
import gc

from healthcorebench.benchmarks.registry import get_adapter, get_registry
from healthcorebench.benchmarks.vlm_adapters.catalog import VLM_TASK_SPECS
from healthcorebench.clients.messages import build_messages
from healthcorebench.evaluators import get_evaluator, select_evaluator_name
from healthcorebench.evaluators.llm_judge import LLMJudgeEvaluator


def _first_sample(task_key: str):
    adapter = get_adapter(task_key)
    files = adapter.discover_source_files()
    adapter.validate_source_files(files)
    raw = next(iter(adapter.load_raw_samples(files)))
    return adapter, adapter.normalize_sample(raw, 0)


def _prompt_text(adapter, sample) -> str:
    messages = adapter.build_messages(sample)
    return next(part["text"] for part in messages[0]["content"] if part["type"] == "text")


def test_vlm_registry_covers_every_catalog_task_and_directory() -> None:
    registry = get_registry()
    assert len(VLM_TASK_SPECS) == 56
    assert len({spec.directory for spec in VLM_TASK_SPECS.values()}) == 36
    assert {
        key for key, entry in registry.items() if entry.component == "Multimodal"
    } == set(VLM_TASK_SPECS)


def test_every_vlm_profile_has_an_explicit_task_shape() -> None:
    expected = {
        "closed": {("multiple_choice", "single_choice", "accuracy"),
                   ("classification", "yes_no", "accuracy"),
                   ("classification", "label", "accuracy")},
        "short_open": {("open_ended", "short_answer", "vlm_text_overlap")},
        "multilabel": {("multiple_label", "label_set", "multilabel")},
        "fixed_text": {("multiple_label", "label_set", "multilabel")},
        "report": {("open_ended", "free_text", "vlm_text_overlap")},
        # Video QA answers are short factual spans, not narratives: generic.py groups
        # video_open with short_open/document_qa, so it must declare short_answer here too.
        "video_open": {("open_ended", "short_answer", "vlm_text_overlap")},
        "generation": {("open_ended", "free_text", "vlm_text_overlap")},
        "multistage": {("multistage_choice", "ordered_choices", "multistage_choice")},
        "grounding": {("visual_grounding", "grounding_json", "grounding")},
        "fixed_diagnosis": {("classification", "label", "accuracy")},
        "document_qa": {("open_ended", "short_answer", "vlm_text_overlap")},
        "document_parse": {("document_understanding", "free_text", "document_fields")},
        "document_complex_qa": {("document_understanding", "free_text", "vlm_text_overlap")},
        "multistage_closed": {("multiple_choice", "single_choice", "accuracy")},
    }
    observed = {}
    for task_key, spec in VLM_TASK_SPECS.items():
        adapter, sample = _first_sample(task_key)
        observed.setdefault(spec.profile, set()).add(
            (sample.task_type, sample.answer_format, sample.evaluation_metric)
        )

    assert observed == expected


def test_every_vlm_task_loads_normalizes_and_scores_real_data() -> None:
    # Each large source JSON is loaded only once in this process.
    for task_key in VLM_TASK_SPECS:
        adapter, sample = _first_sample(task_key)
        dumped = sample.model_dump()
        messages = adapter.build_messages(sample)
        assert sample.component == "Multimodal", task_key
        assert sample.reference_answer is not None, task_key
        assert len(messages) == 1 and messages[0]["role"] == "user", task_key
        assert all(message["role"] not in {"system", "assistant"} for message in messages), task_key
        assert _prompt_text(adapter, sample).strip(), task_key
        assert "runtime_media" not in dumped, task_key
        assert "data:image" not in json.dumps(dumped, ensure_ascii=False), task_key

        evaluator_name = select_evaluator_name(sample.evaluation_metric, sample.answer_format)
        assert evaluator_name is not None, task_key
        evaluator = get_evaluator(evaluator_name)
        prediction = sample.reference_answer
        if sample.answer_format == "single_choice":
            prediction = f"The final answer is {sample.reference_answer}."
        elif sample.answer_format == "ordered_choices":
            prediction = ",".join(sample.reference_answer)
        elif sample.answer_format in {"label_set", "grounding_json"}:
            prediction = json.dumps(sample.reference_answer)
        parsed = adapter.parse_response(sample, str(prediction))
        judgment = evaluator.evaluate(
            {
                "run_id": "run", "result_id": "result", "sample_id": sample.sample_id,
                "status": "success", "parsed_answer": parsed,
            },
            sample.model_dump(),
        )
        assert judgment.evaluation_status == "success", (task_key, judgment.evaluation_error)
        assert judgment.normalized_score is not None, task_key
        del adapter, sample, dumped, messages, evaluator, judgment
        gc.collect()


def test_hlm_embedded_image_is_absent_from_persistence_hash_and_judge_prompt() -> None:
    from healthcorebench.runtime.run_setup import RunOrchestrator

    adapter, sample = _first_sample("HLM/exact")
    messages = adapter.build_messages(sample)
    persisted = sample.model_dump()
    hashed = RunOrchestrator._logical_messages_for_hash(messages)
    judge = LLMJudgeEvaluator(client=object(), judge_model="judge")
    judge_messages = judge._build_messages(
        {"reference_answer": sample.reference_answer, "raw_response": sample.reference_answer},
        {**persisted, "logical_messages": messages},
    )
    combined = json.dumps({"persisted": persisted, "hashed": hashed, "judge": judge_messages})
    assert "base64" not in combined
    assert "data:image" not in combined
    assert str(sample.runtime_media[0]["source"]) not in combined


def test_video_sampling_is_lazy_and_logs_references_only() -> None:
    _, sample = _first_sample("MIMICEchoQA/mcqa")
    assert sample.runtime_media[0]["kind"] == "video"
    messages = [{"role": "user", "content": [
        {"type": "video", "source": sample.runtime_media[0]["source"], "media_id": "video_0"},
        {"type": "text", "text": "Question"},
    ]}]
    built = build_messages(messages, max_images=2, max_video_frames=2)
    assert len(built.image_infos) == 2
    assert len(built.video_infos) == 1
    assert [part["type"] for part in built.logged_messages[0]["content"]] == ["video_ref", "text"]
    assert "base64" not in json.dumps(built.logged_messages)
    assert str(sample.runtime_media[0]["source"]) not in json.dumps(built.logged_messages)


def test_real_schema_edge_cases_keep_valid_references_and_options() -> None:
    slake = get_adapter("SLAKE/closed")
    slake_files = slake.discover_source_files()
    slake_raw = next(
        raw for raw in slake.load_raw_samples(slake_files)
        if str(raw["record"].get("answer")).casefold() == "lung"
    )
    slake_sample = slake.normalize_sample(slake_raw, 0)
    assert slake_sample.answer_format == "label"
    assert slake_sample.reference_answer == "lung"

    quilt = get_adapter("Quilt-VQA/closed")
    quilt_raw = next(iter(quilt.load_raw_samples(quilt.discover_source_files())))
    quilt_sample = quilt.normalize_sample(quilt_raw, 0)
    assert quilt_sample.reference_answer in {"yes", "no"}
    assert quilt_sample.reference_answer == quilt_raw["record"]["yes_no_answer"]

    medframe = get_adapter("MedFrameQA/mcqa")
    medframe_raw = next(
        raw for raw in medframe.load_raw_samples(medframe.discover_source_files())
        if (raw["record"].get("options") or [""])[0].startswith("E. coli")
    )
    medframe_sample = medframe.normalize_sample(medframe_raw, 0)
    assert [item["label"] for item in medframe_sample.source_content["options"]] == list("ABCDEFG")
    assert medframe_sample.reference_answer == "A"

    omnibrain = get_adapter("OmniBrainBench/closed")
    omnibrain_raw = next(
        raw for raw in omnibrain.load_raw_samples(omnibrain.discover_source_files())
        if raw["record"].get("label") == "Glioblastoma"
        and raw["record"].get("answer") == "B"
    )
    assert omnibrain.normalize_sample(omnibrain_raw, 0).reference_answer == "B"


def test_slake_closed_is_fixed_label_classification_not_mcqa() -> None:
    adapter = get_adapter("SLAKE/closed")
    raw_rows = list(adapter.load_raw_samples(adapter.discover_source_files()))
    samples = [adapter.normalize_sample(raw, index) for index, raw in enumerate(raw_rows)]

    assert len(samples) == 836
    assert {raw["record"]["answer_type"] for raw in raw_rows} == {"CLOSED"}
    assert all(not sample.source_content["options"] for sample in samples)
    assert {(sample.task_type, sample.answer_format, sample.evaluation_metric) for sample in samples} == {
        ("classification", "label", "accuracy")
    }
    assert {sample.reference_answer for sample in samples} >= {"yes", "no", "lung", "t1", "t2"}
    sample = samples[0]
    assert adapter.parse_response(sample, "The final answer is yes.") == "yes"
    assert adapter.parse_response(sample, "The final answer is spleen.") == "spleen"
    assert adapter.parse_response(sample, "最终答案是肺。") == "lung"


def test_slake_open_is_short_answer_and_disjoint_from_closed() -> None:
    closed = get_adapter("SLAKE/closed")
    opened = get_adapter("SLAKE/open")
    closed_ids = {
        closed.normalize_sample(raw, index).source_sample_id
        for index, raw in enumerate(closed.load_raw_samples(closed.discover_source_files()))
    }
    open_rows = list(opened.load_raw_samples(opened.discover_source_files()))
    open_samples = [opened.normalize_sample(raw, index) for index, raw in enumerate(open_rows)]

    assert len(open_samples) == 1258
    assert {raw["record"]["answer_type"] for raw in open_rows} == {"OPEN"}
    assert not closed_ids & {sample.source_sample_id for sample in open_samples}
    assert {(sample.task_type, sample.answer_format, sample.evaluation_metric) for sample in open_samples} == {
        ("open_ended", "short_answer", "vlm_text_overlap")
    }


def test_fixed_diagnosis_and_complex_document_qa_use_correct_metrics() -> None:
    _, diagnosis = _first_sample("3MDBench/diagnosis")
    _, document = _first_sample("MedDocBench/gmd_complex_qa")
    _, abnormalities = _first_sample("MedDocBench/ltr_abnormality_qa")

    assert (diagnosis.task_type, diagnosis.answer_format, diagnosis.evaluation_metric) == (
        "classification", "label", "accuracy"
    )
    assert (document.task_type, document.answer_format, document.evaluation_metric) == (
        "document_understanding", "free_text", "vlm_text_overlap"
    )
    assert (
        abnormalities.task_type,
        abnormalities.answer_format,
        abnormalities.evaluation_metric,
    ) == ("document_understanding", "free_text", "document_fields")


def test_vlm_judge_is_primary_only_for_semantic_generation_profiles(tmp_path) -> None:
    from healthcorebench.config import load_config
    from healthcorebench.runtime.run_setup import RunOrchestrator

    expected = {
        "SLAKE/open": ("vlm_text_overlap", True, False),
        "IU-Xray/report_generation": ("vlm_text_overlap", True, True),
        "MedDocBench/ltr_full_parsing": ("document_fields", True, False),
        "MedDocBench/ltr_abnormality_qa": ("document_fields", True, False),
        "MedDocBench/gmd_complex_qa": ("vlm_text_overlap", True, True),
        "3MDBench/diagnosis": ("classification", False, False),
    }
    for task_key, (evaluator, use_judge, judge_primary) in expected.items():
        config = load_config(
            "configs/run_all_benchmarks_multimodal.yaml",
            {"benchmark.name": task_key, "benchmark.max_samples": 1},
        )
        orchestrator = RunOrchestrator(
            config, run_dir=str(tmp_path / task_key.replace("/", "_"))
        )
        samples = orchestrator.prepare_samples()
        orchestrator._resolve_evaluation(config, samples)

        assert config.evaluation.evaluator == evaluator
        assert config.evaluation.use_llm_judge is use_judge
        assert orchestrator._judge_as_primary is judge_primary


def test_empty_mimic_ext_label_set_is_valid_but_empty_output_is_parse_failure() -> None:
    adapter = get_adapter("MIMIC-Ext/query")
    raw = next(
        raw for raw in adapter.load_raw_samples(adapter.discover_source_files())
        if raw["record"].get("answer") == []
    )
    sample = adapter.normalize_sample(raw, 0)
    evaluator = get_evaluator("multilabel")

    assert sample.reference_answer == []
    assert adapter.parse_response(sample, "[]") == []
    explicit_empty = evaluator.evaluate(
        {"run_id": "run", "result_id": "r1", "sample_id": sample.sample_id,
         "status": "success", "parsed_answer": []},
        sample.model_dump(),
    )
    missing = evaluator.evaluate(
        {"run_id": "run", "result_id": "r2", "sample_id": sample.sample_id,
         "status": "success", "parsed_answer": None},
        sample.model_dump(),
    )
    # An empty gold label set is valid source data, but it is not scorable: set F1 has no
    # positive reference to measure against, so the sample leaves the score denominator rather
    # than handing every prediction a free 1.0. The parse-failure signal is still recorded.
    assert explicit_empty.normalized_score is None
    assert explicit_empty.is_correct is None
    assert explicit_empty.parsed_judgment["unscorable_reason"] == "empty_reference"
    assert explicit_empty.parsed_judgment["parse_failed"] is False
    assert missing.normalized_score is None
    assert missing.parsed_judgment["parse_failed"] is True


def test_mimic_ext_label_universes_follow_each_tasks_answer_space() -> None:
    _, query = _first_sample("MIMIC-Ext/query")
    _, choose = _first_sample("MIMIC-Ext/choose")

    query_labels = set(query.metadata["label_universe"])
    choose_labels = set(choose.metadata["label_universe"])
    assert set(query.reference_answer) <= query_labels
    assert set(choose.reference_answer) <= choose_labels
    assert not {
        "anatomicalfinding", "disease", "technicalassessment", "tubesandlines",
    } & (query_labels | choose_labels)
    assert set(choose.metadata["candidate_labels"]) <= choose_labels


def test_multilabel_parser_preserves_unsupported_extra_labels() -> None:
    adapter = get_adapter("MIMIC-CXR/multilabel")
    raw = next(
        raw for index, raw in enumerate(adapter.load_raw_samples(adapter.discover_source_files()))
        if adapter.normalize_sample(raw, index).reference_answer
    )
    sample = adapter.normalize_sample(raw, 0)
    prediction = f'{sample.reference_answer[0]}, definitely_not_a_real_finding'
    parsed = adapter.parse_response(sample, prediction)
    judgment = get_evaluator("multilabel").evaluate(
        {"run_id": "run", "result_id": "result", "sample_id": sample.sample_id,
         "status": "success", "parsed_answer": parsed},
        sample.model_dump(),
    )

    assert "definitely_not_a_real_finding" in parsed
    assert judgment.parsed_judgment["predicted_labels"] != sample.reference_answer
    assert judgment.parsed_judgment["precision_sample"] < 1.0


def test_vlm_structured_parsers_use_post_think_answer_and_normalize_grounding_box() -> None:
    multilabel = get_adapter("MIMIC-CXR/multilabel")
    _, multilabel_sample = _first_sample("MIMIC-CXR/multilabel")
    assert multilabel.parse_response(
        multilabel_sample,
        'Possible labels were discussed.\n</think>\n```json\n["Edema"]\n```',
    ) == ["Edema"]

    grounding = get_adapter("MX-CXR/grounding")
    _, grounding_sample = _first_sample("MX-CXR/grounding")
    parsed = grounding.parse_response(
        grounding_sample,
        'Coordinates 1, 2, 3, 4 were considered.\n</think>\n'
        '```json\n[{"bbox_2d":[10,20,30,40],"label":"opacity",'
        '"category":"Pneumonia"}]\n```',
    )
    assert parsed == [{
        "bbox_xyxy": [10, 20, 30, 40],
        "label": "opacity",
        "category": "Pneumonia",
    }]


def test_vlm_free_text_parser_preserves_raw_log_but_scores_only_final_answer() -> None:
    adapter = get_adapter("VQA-RAD/open")
    _, sample = _first_sample("VQA-RAD/open")
    raw = "The differential contains several possibilities.\n</think>\nPneumonia"

    assert adapter.parse_response(sample, raw) == "Pneumonia"


def test_mx_cxr_full_split_has_stable_unique_sample_ids() -> None:
    adapter = get_adapter("MX-CXR/grounding")
    raw_rows = list(adapter.load_raw_samples(adapter.discover_source_files()))
    samples = [adapter.normalize_sample(raw, index) for index, raw in enumerate(raw_rows)]

    assert len(samples) == 167
    assert len({sample.sample_id for sample in samples}) == len(samples)
    assert sum(len(sample.reference_answer) for sample in samples) == 216

    by_source = {}
    for sample in samples:
        by_source.setdefault(sample.source_sample_id, []).append(sample)
    repeated_image = next(group for group in by_source.values() if len(group) > 1)
    questions = {sample.source_content["question"] for sample in repeated_image}
    assert len(questions) == len(repeated_image)
    for sample in repeated_image:
        for annotation in sample.reference_answer:
            assert annotation["label"] in sample.source_content["question"]
            assert all(0 <= coordinate <= 1000 for coordinate in annotation["bbox_xyxy"])
        assert sample.metadata["coordinate_format"] == "xyxy_normalized_0_1000"
        assert "[0,0,1000,1000] spans the entire image" in _prompt_text(adapter, sample)


def test_structured_vlm_parsers_accept_json_code_fences() -> None:
    multilabel, multilabel_sample = _first_sample("ROCOv2/concepts")
    grounding, grounding_sample = _first_sample("MX-CXR/grounding")

    assert multilabel.parse_response(multilabel_sample, '```json\n["C001"]\n```') == ["C001"]
    parsed = grounding.parse_response(
        grounding_sample,
        '```json\n[{"label":"opacity","bbox_xyxy":[0,0,1000,1000]}]\n```',
    )
    assert parsed == [{"label": "opacity", "bbox_xyxy": [0, 0, 1000, 1000]}]


def test_multilabel_prompts_declare_exact_task_label_space_without_missing_sentinels() -> None:
    mimic, mimic_sample = _first_sample("MIMIC-CXR/multilabel")
    roco, roco_sample = _first_sample("ROCOv2/concepts")
    choose, choose_sample = _first_sample("MIMIC-Ext/choose")

    assert "Allowed labels (use only these exact strings):" in _prompt_text(mimic, mimic_sample)
    assert "Support Devices" in _prompt_text(mimic, mimic_sample)
    assert "nan" not in roco_sample.metadata["label_universe"]
    assert "nan" not in roco_sample.reference_answer
    assert "C0040405" in _prompt_text(roco, roco_sample)
    choose_prompt = _prompt_text(choose, choose_sample)
    assert all(label in choose_prompt for label in choose_sample.metadata["candidate_labels"])


def test_mtbbench_includes_non_image_clinical_evidence_in_prompts() -> None:
    msk, msk_sample = _first_sample("MTBBench/msk_mcqa")
    hancock, hancock_sample = _first_sample("MTBBench/hancock_mcqa")

    assert msk_sample.runtime_media == []
    assert msk_sample.metadata["clinical_evidence_files"]
    assert msk_sample.metadata["case_id"]
    assert msk_sample.metadata["stage_name"].startswith("block_")
    assert "Clinical evidence files:" in _prompt_text(msk, msk_sample)
    assert "timeline" in _prompt_text(msk, msk_sample)
    assert hancock_sample.runtime_media
    assert "Clinical evidence files:" not in _prompt_text(hancock, hancock_sample)


def test_mtbbench_source_manifest_includes_prompt_evidence_files() -> None:
    for task_key, expected_suffix in (
        ("MTBBench/msk_mcqa", ".txt"),
        ("MTBBench/hancock_mcqa", ".json"),
    ):
        adapter = get_adapter(task_key)
        files = adapter.discover_source_files()
        assert any(path.suffix == expected_suffix for path in files)
        adapter.validate_source_files(files)


def test_task_specific_prompts_preserve_questions_and_required_output_shapes() -> None:
    iu_adapter, iu_sample = _first_sample("IU-Xray/report_generation")
    assert "radiology report" in _prompt_text(iu_adapter, iu_sample)

    medbook = get_adapter("MedBookVQA/mcqa")
    medbook_raw = next(iter(medbook.load_raw_samples(medbook.discover_source_files())))
    medbook_sample = medbook.normalize_sample(medbook_raw, 0)
    medbook_prompt = _prompt_text(medbook, medbook_sample)
    assert medbook_sample.source_content["question"] == medbook_raw["record"]["Question"].strip()
    assert medbook_sample.source_content["question"] in medbook_prompt

    drv, drv_sample = _first_sample("DrVD-Bench/joint_reasoning")
    drv_prompt = _prompt_text(drv, drv_sample)
    assert drv_sample.answer_format == "ordered_choices"
    assert drv_sample.source_content["options"] == []
    assert "2. Which organ appears to be abnormal" in drv_prompt
    assert "four selected letters in order" in drv_prompt
    assert "(A, B, C, or D)" not in drv_prompt
    assert "E. calcification" in drv_prompt
    assert "H. pulmonary edema" in drv_prompt
    assert drv_sample.metadata["stage_letters"] == [
        list("ABCD"), list("ABCD"), list("ABCDEFGH"), list("ABCDEFGH")
    ]
    assert drv.parse_response(drv_sample, "C,C,H,F") == ["C", "C", "H", "F"]
    assert drv.parse_response(
        drv_sample,
        "C,C,H,F\nThe four stages support this diagnosis.",
    ) == ["C", "C", "H", "F"]
    assert drv.parse_response(
        drv_sample,
        "I considered A and B. Final answer: C,C,H,F\nNo further explanation.",
    ) == ["C", "C", "H", "F"]
    assert drv.parse_response(
        drv_sample,
        "I considered alternatives.\nC,C,H,F",
    ) == ["C", "C", "H", "F"]
    assert drv.parse_response(
        drv_sample,
        "I considered alternatives.\nA,B,C,D\nFurther discussion only.",
    ) is None
    assert drv.parse_response(drv_sample, "C,C,H,F because these fit") is None
    assert drv.parse_response(drv_sample, "E,C,H,F") is None
    assert "Return only the final option letter." not in drv_prompt

    omnibrain = get_adapter("OmniBrainBench/open")
    omnibrain_raw = next(
        raw for raw in omnibrain.load_raw_samples(omnibrain.discover_source_files())
        if raw["record"].get("options")
    )
    omnibrain_sample = omnibrain.normalize_sample(omnibrain_raw, 0)
    omnibrain_prompt = _prompt_text(omnibrain, omnibrain_sample)
    assert omnibrain_sample.answer_format == "free_text"
    assert "do not reply with only an option letter" in omnibrain_prompt
    assert "Return only the final option letter." not in omnibrain_prompt


def test_image_only_mmmu_questions_receive_an_explicit_task_instruction() -> None:
    adapter = get_adapter("MMMU-Health-Medicine/mcqa")
    raw = next(
        raw for raw in adapter.load_raw_samples(adapter.discover_source_files())
        if raw["record"].get("question", "").strip() == "<image 1>"
    )
    sample = adapter.normalize_sample(raw, 0)
    assert sample.source_content["question"] == (
        "Based on the provided medical image(s), select the most appropriate answer."
    )
    assert sample.source_content["question"] in _prompt_text(adapter, sample)


def test_multilingual_prompts_use_normalized_languages_and_localized_instructions() -> None:
    slake = get_adapter("SLAKE/closed")
    slake_raw = next(
        raw for raw in slake.load_raw_samples(slake.discover_source_files())
        if raw["record"].get("q_lang") == "zh"
    )
    slake_sample = slake.normalize_sample(slake_raw, 0)
    assert slake_sample.language == "zh"
    assert "只输出简洁的最终答案" in _prompt_text(slake, slake_sample)

    world = get_adapter("WorldMedQA-V/mcqa")
    expected_world_languages = {
        "brazil": ("pt", "Opções de resposta:"),
        "israel": ("he", "אפשרויות תשובה:"),
        "japan": ("ja", "選択肢："),
        "spain": ("es", "Opciones de respuesta:"),
    }
    raw_by_country = {}
    for raw in world.load_raw_samples(world.discover_source_files()):
        record = raw["record"]
        if record.get("language") == "local":
            raw_by_country.setdefault(record.get("country"), raw)
        if set(raw_by_country) == set(expected_world_languages):
            break
    for country, (language, marker) in expected_world_languages.items():
        sample = world.normalize_sample(raw_by_country[country], 0)
        assert sample.language == language
        assert marker in _prompt_text(world, sample)

    korean, korean_sample = _first_sample("KorMedMCQA-V/mcqa")
    assert korean_sample.language == "ko"
    assert "선택지:" in _prompt_text(korean, korean_sample)

    meddoc, meddoc_sample = _first_sample("MedDocBench/ltr_simple_qa")
    assert meddoc_sample.language == "zh"
    assert "请使用中文作答" in _prompt_text(meddoc, meddoc_sample)

    arabic, arabic_sample = _first_sample("BiMed-MBench/open_ar")
    assert arabic_sample.language == "ar"
    assert "أجب باللغة العربية" in _prompt_text(arabic, arabic_sample)
