"""Shared adapter for the fixed medical VLM benchmark collection.

The VLM datasets use many small schema variations but the request lifecycle is identical:
load a fixed local record, normalize text/options/references/media, and build one ordered
multimodal message. Dataset-specific branches below are deliberately limited to places where
the source semantics differ (multi-stage cases, label sets, grounding, and document parsing).
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.answer_parsing import (
    final_answer_region,
    parse_multiple_choice_letter,
    parse_label,
    parse_yes_no_maybe,
)
from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.context_window import fit_context_to_window
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.benchmarks.vlm_adapters.catalog import VLM_TASK_SPECS, VLMTaskSpec
from healthcorebench.schemas.sample import EvaluationSample, MediaInfo

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LABELED_OPTION = re.compile(r"^\s*([A-Z])\s*[\.\):\-]\s*(.+?)\s*$", re.DOTALL)
_CHOICE_LINE = re.compile(r"(?m)^\s*\(?([A-Z])\)?[\.\):]\s*(.+?)\s*$")
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
_GROUNDING_COORDINATE_SCALE = 1000.0
_DOCUMENT_SPLITS = {
    "MedDocBench/ltr_simple_qa": "LTR_simpleQA",
    "MedDocBench/ltr_full_parsing": "LTR_fullparsing",
    "MedDocBench/ltr_abnormality_qa": "LTR_abnormalityQA",
    "MedDocBench/gmd_simple_qa": "GMD_simpleQA",
    "MedDocBench/gmd_complex_qa": "GMD_complexQA",
}
_LANGUAGE_ALIASES = {
    "ar": "ar", "arabic": "ar",
    "en": "en", "english": "en",
    "es": "es", "spanish": "es",
    "he": "he", "hebrew": "he",
    "ja": "ja", "japanese": "ja",
    "ko": "ko", "korean": "ko",
    "pt": "pt", "portuguese": "pt",
    "zh": "zh", "chinese": "zh", "zh-cn": "zh", "zh_cn": "zh",
}
_WORLD_LOCAL_LANGUAGES = {
    "brazil": "pt",
    "israel": "he",
    "japan": "ja",
    "spain": "es",
}
_SLAKE_CLOSED_LABELS = (
    "axial plane", "colon", "coronal plane", "ct", "esophagus", "heart", "hyperdense",
    "hypodense", "kidney", "left", "liver", "lung", "mri", "no", "rectum", "right",
    "right kidney", "small bowel", "spleen", "t1", "t2", "top", "white", "x-ray", "yes",
)
_SLAKE_CLOSED_ALIASES = {
    "true": "yes", "是的": "yes", "是": "yes", "包含": "yes", "有": "yes",
    "可以": "yes", "存在": "yes", "健康": "yes",
    "false": "no", "不是": "no", "否": "no", "不包含": "no", "没有": "no",
    "不可以": "no", "不正常": "no",
    "x光": "x-ray", "核磁共振": "mri", "横断面": "axial plane", "冠状面": "coronal plane",
    "低密度": "hypodense", "白色": "white", "右侧": "right", "左侧": "left",
    "心脏": "heart", "直肠": "rectum", "结肠": "colon", "肝脏": "liver",
    "肺": "lung", "肾脏": "kidney", "脾脏": "spleen",
}
_THREEMD_DIAGNOSIS_LABELS = (
    "abscess", "acne", "actinic keratosis", "allergic contact dermatitis", "caries",
    "chalazion", "chickenpox", "chronic lichen", "conjunctivitis", "contact dermatitis",
    "dental calculus", "eczema", "gingivitis", "herpes", "hives", "ingrown nail",
    "keratosis pilaris", "lichen planus", "molluscum contagiosum", "mycosis",
    "nail dystrophy", "onycholysis", "onychomycosis", "periodontitis", "psoriasis",
    "rosacea", "seborrheic dermatitis", "seborrheic keratosis", "shingles", "stomatitis",
    "stye", "tonsillitis", "vitiligo", "warts",
)
_PROMPT_TEXT = {
    "en": {
        "answer_choices": "Answer choices:",
        "final_letter": "Return only the final option letter.",
        "json_labels": "Return only a JSON array of labels.",
        "yes_no": "Return only yes or no.",
        "concise_label": "Return only the concise final answer, with no explanation.",
        "short_answer": "Return only the short final answer (a few words at most), with no explanation.",
        "same_language": "",
        "open_with_options": (
            "Provide the complete answer requested by the question; do not reply with only "
            "an option letter."
        ),
    },
    "zh": {
        "answer_choices": "候选答案：",
        "final_letter": "只输出最终答案对应的大写字母，不要解释。",
        "json_labels": "只输出由标签组成的 JSON 数组。",
        "yes_no": "只输出“是”或“否”。",
        "concise_label": "只输出简洁的最终答案，不要解释。",
        "short_answer": "只输出简短的最终答案（最多几个词），不要解释。",
        "same_language": "请使用中文作答。",
        "open_with_options": "请完整回答问题并提供题目要求的说明，不要只输出选项字母。",
    },
    "ar": {
        "answer_choices": "خيارات الإجابة:",
        "final_letter": "أخرج حرف الخيار النهائي فقط دون شرح.",
        "json_labels": "أخرج فقط مصفوفة JSON من التسميات.",
        "yes_no": "أخرج فقط نعم أو لا.",
        "concise_label": "أخرج الإجابة النهائية المختصرة فقط دون شرح.",
        "short_answer": "أخرج الإجابة النهائية القصيرة فقط (بضع كلمات على الأكثر) دون شرح.",
        "same_language": "أجب باللغة العربية.",
        "open_with_options": "أجب عن السؤال إجابة كاملة ولا تكتفِ بحرف الخيار.",
    },
    "ko": {
        "answer_choices": "선택지:",
        "final_letter": "최종 선택지의 대문자 알파벳만 출력하고 설명은 출력하지 마십시오.",
        "json_labels": "레이블로 이루어진 JSON 배열만 출력하십시오.",
        "yes_no": "예 또는 아니요만 출력하십시오.",
        "concise_label": "설명 없이 간결한 최종 답변만 출력하십시오.",
        "short_answer": "짧은 최종 답변만 출력하고(최대 몇 단어) 설명은 출력하지 마십시오.",
        "same_language": "한국어로 답하십시오.",
        "open_with_options": "선택지 문자만 출력하지 말고 질문에서 요구한 내용을 완전하게 답하십시오.",
    },
    "pt": {
        "answer_choices": "Opções de resposta:",
        "final_letter": "Retorne apenas a letra da opção final, sem explicação.",
        "json_labels": "Retorne apenas uma matriz JSON de rótulos.",
        "yes_no": "Retorne apenas sim ou não.",
        "concise_label": "Retorne apenas a resposta final concisa, sem explicação.",
        "short_answer": "Retorne apenas a resposta final curta (no máximo algumas palavras), sem explicação.",
        "same_language": "Responda em português.",
        "open_with_options": "Responda completamente ao que foi solicitado; não retorne apenas a letra da opção.",
    },
    "he": {
        "answer_choices": "אפשרויות תשובה:",
        "final_letter": "החזר רק את האות של האפשרות הסופית, ללא הסבר.",
        "json_labels": "החזר רק מערך JSON של תוויות.",
        "yes_no": "החזר רק כן או לא.",
        "concise_label": "החזר רק תשובה סופית קצרה, ללא הסבר.",
        "short_answer": "החזר רק את התשובה הסופית הקצרה (מספר מילים לכל היותר), ללא הסבר.",
        "same_language": "ענה בעברית.",
        "open_with_options": "ענה באופן מלא על השאלה ואל תחזיר רק את אות האפשרות.",
    },
    "ja": {
        "answer_choices": "選択肢：",
        "final_letter": "最終回答の選択肢を示す大文字のみを出力し、説明は加えないでください。",
        "json_labels": "ラベルの JSON 配列のみを出力してください。",
        "yes_no": "「はい」または「いいえ」のみを出力してください。",
        "concise_label": "説明を加えず、簡潔な最終回答のみを出力してください。",
        "short_answer": "簡潔な最終回答のみ（数語以内）を出力し、説明は加えないでください。",
        "same_language": "日本語で回答してください。",
        "open_with_options": "選択肢の文字だけではなく、質問で求められた内容を完全に回答してください。",
    },
    "es": {
        "answer_choices": "Opciones de respuesta:",
        "final_letter": "Devuelva únicamente la letra de la opción final, sin explicación.",
        "json_labels": "Devuelva únicamente una matriz JSON de etiquetas.",
        "yes_no": "Devuelva únicamente sí o no.",
        "concise_label": "Devuelva únicamente la respuesta final concisa, sin explicación.",
        "short_answer": "Devuelva únicamente la respuesta final corta (unas pocas palabras como máximo), sin explicación.",
        "same_language": "Responda en español.",
        "open_with_options": "Responda completamente a lo solicitado; no devuelva solo la letra de la opción.",
    },
}


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_label(value: Any) -> str:
    label = _clean_text(value)
    return "" if label.casefold() in {"nan", "none", "null"} else label


def _first(record: dict, *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _letter(value: Any) -> str | None:
    text = _clean_text(value).upper()
    match = re.match(r"^\(?([A-Z])\)?(?:\s*[\.\):\-]|$)", text)
    return match.group(1) if match else None


def _mimic_cxr_labels(record: dict) -> dict:
    """Label map for one MIMIC-CXR record, always a dict.

    ``manual_labeled`` rows carry ``manual_labels``, everything else falls back to the
    automatic CheXpert labels. Both keys can be present-but-null, so the guard has to apply to
    whichever branch is taken rather than only to the fallback.
    """
    labels = (
        record.get("manual_labels") if record.get("manual_labeled")
        else record.get("chexpert_labels")
    )
    return labels or {}


def _split_hle_question(question: str) -> tuple[str, list[tuple[str, str]]]:
    parts = re.split(r"\n\s*Answer Choices?\s*:\s*\n", question, maxsplit=1, flags=re.I)
    if len(parts) != 2:
        return question.strip(), []
    options: list[tuple[str, str]] = []
    for line in parts[1].splitlines():
        match = _LABELED_OPTION.match(line)
        if match:
            options.append((match.group(1), match.group(2).strip()))
    return parts[0].strip(), options


def _options(record: dict, task_key: str, question: str) -> list[tuple[str, str]]:
    if task_key == "DrVD-Bench/joint_reasoning":
        # The source joint_prompt already contains four separately labelled option groups.
        # Flattening them into one MCQA list destroys both the stages and their label scopes.
        return []
    if task_key == "AgentClinic-VLM/mcqa":
        return [
            (_LETTERS[index], _clean_text(item.get("text")))
            for index, item in enumerate(record.get("answers") or [])
            if index < len(_LETTERS)
        ]
    if task_key == "MedBookVQA/mcqa":
        answer_letter = _letter(record.get("correct_choice"))
        mapped = {
            _letter(letter): _clean_text(text)
            for letter, text in zip(record.get("other_choices") or [], record.get("Distractors") or [])
            if _letter(letter)
        }
        if answer_letter:
            mapped[answer_letter] = _clean_text(record.get("Answer"))
        return [(letter, mapped[letter]) for letter in _LETTERS if mapped.get(letter)]

    value = record.get("options")
    if isinstance(value, dict):
        return [(str(key).upper(), _clean_text(text)) for key, text in value.items()]
    if isinstance(value, list) and value:
        string_matches = [
            _LABELED_OPTION.match(_clean_text(item)) if isinstance(item, str) else None
            for item in value
        ]
        explicit_labels = [match.group(1) for match in string_matches if match]
        use_explicit_labels = (
            len(explicit_labels) == len(value)
            and explicit_labels == list(_LETTERS[:len(value)])
        )
        out: list[tuple[str, str]] = []
        for index, item in enumerate(value):
            if isinstance(item, dict):
                label = _clean_text(_first(item, "label", "key", "option") or _LETTERS[index]).upper()
                text = _clean_text(_first(item, "text", "value", "answer"))
            else:
                match = string_matches[index] if use_explicit_labels else None
                label, text = ((match.group(1), match.group(2)) if match
                               else (_LETTERS[index], _clean_text(item)))
            out.append((label, text))
        return out

    prefixed = []
    for letter in _LETTERS:
        value = _first(record, f"option_{letter}", f"choice_{letter}", letter)
        if value is None:
            continue
        text = _clean_text(value)
        text = re.sub(rf"^\s*{letter}\s*[:\.\)]\s*", "", text, flags=re.I)
        prefixed.append((letter, text))
    if prefixed:
        return prefixed

    _, hle_options = _split_hle_question(question)
    if hle_options:
        return hle_options
    return [(match.group(1), match.group(2).strip()) for match in _CHOICE_LINE.finditer(question)]


def _closed_reference(record: dict, task_key: str, options: list[tuple[str, str]]) -> str | None:
    if task_key == "AgentClinic-VLM/mcqa":
        for index, answer in enumerate(record.get("answers") or []):
            if answer.get("correct") is True:
                return _LETTERS[index]
    for key in (
        "answer_label", "correct_answer", "correct_option", "answer_letter",
        "correct_choice", "answer", "label",
    ):
        found = _letter(record.get(key))
        if found:
            return found
    target = _clean_text(_first(record, "gt_answer", "Answer", "answer"))
    for label, text in options:
        if target.casefold() == text.casefold():
            return label
    return None


def _record_question(record: dict, task_key: str) -> str:
    if task_key == "DrVD-Bench/joint_reasoning":
        question = _clean_text(record.get("joint_prompt"))
        return re.sub(
            r"For each question, choose one capital letter\s*\(A, B, C, or D\)\s*\.",
            "For each question, choose one capital letter from the options provided for "
            "that question.",
            question,
            flags=re.IGNORECASE,
        )
    if task_key == "IU-Xray/report_generation":
        return "Generate a complete radiology report for the provided chest X-ray images."
    if task_key == "PathText/caption":
        return "Generate a concise medical caption for the image."
    if task_key == "PathText/report_generation":
        return "Generate the pathology report associated with the image."
    if task_key == "MIMIC-CXR/multilabel":
        return "List all positive findings visible on this chest radiograph."
    if task_key == "ROCOv2/caption":
        return "Generate a concise medical caption for the image."
    if task_key == "ROCOv2/concepts":
        return "List the relevant UMLS concept identifiers (CUI) visible in the image."
    if task_key == "MX-CXR/grounding":
        # MS-CXR is phrase grounding: the report phrase is the query and has to stay in the
        # prompt. The disease category is a gold field, so it is asked for, never supplied —
        # embedding it made disease_category_accuracy a pure echo of the reference.
        targets = []
        for annotation in record.get("annotations") or []:
            target = _clean_text(annotation.get("label_text"))
            if target and target not in targets:
                targets.append(target)
        target_block = "\n".join(
            f'{index}. "{label}"' for index, label in enumerate(targets, start=1)
        )
        return (
            "Locate every region corresponding to the following target finding phrase(s):\n"
            f"{target_block}\n\nReturn JSON as an array of objects with keys "
            '"label" (the target finding phrase the region belongs to), "category" (the '
            'finding category you infer for that region), and "bbox_xyxy" using coordinates '
            "normalized to the range 0 to 1000, where [0,0,1000,1000] spans the entire image. "
            "Return one object for each corresponding region; a target phrase may have "
            "multiple regions."
        )
    if task_key == "MedDocBench/ltr_abnormality_qa":
        # The source question names the required content in Chinese prose ("检验项目名称、结果、
        # 参考范围和异常状态") while the gold answer is JSON keyed in English. Spell the schema
        # out instead of renaming the reference's keys: the reference is the scored gold and
        # stays byte-identical to the source record. The unit is part of every gold ``result``.
        return (
            f"{_clean_text(record.get('question'))}\n\n"
            "输出格式：只输出一个 JSON 数组，数组中每个对象必须使用以下英文键名："
            '"entryname"（检验项目名称）、"result"（结果，包含检验单上打印的单位）、'
            '"reference"（参考范围）、"status"（异常状态）。'
        )
    if task_key == "3MDBench/diagnosis":
        complaint = _clean_text(record.get("general_complaint"))
        details = _clean_text(record.get("complaints"))
        return f"{complaint}\n\n{details}\n\nWhat is the most likely diagnosis?".strip()
    if task_key.startswith("MTBBench/"):
        context = _clean_text(record.get("context"))
        question = _clean_text(record.get("question"))
        return f"Clinical context:\n{context}\n\nQuestion:\n{question}" if context else question
    return _clean_text(_first(record, "question", "Question", "text", "joint_prompt"))


def _normalized_language(record: dict, task_key: str) -> str:
    if task_key == "KorMedMCQA-V/mcqa":
        return "ko"
    if task_key.startswith("MedDocBench/"):
        return "zh"
    if task_key.endswith("open_ar"):
        return "ar"

    raw = _clean_text(_first(record, "q_lang", "language")).casefold()
    if task_key == "WorldMedQA-V/mcqa" and raw == "local":
        return _WORLD_LOCAL_LANGUAGES.get(_clean_text(record.get("country")).casefold(), "en")
    return _LANGUAGE_ALIASES.get(raw, "en")


def _prompt_text(language: str) -> dict[str, str]:
    return _PROMPT_TEXT.get(language, _PROMPT_TEXT["en"])


def _question_stem(question: str, options: list[tuple[str, str]]) -> str:
    if not options:
        return question
    question = re.split(r"\n\s*Answer Choices?\s*:", question, maxsplit=1, flags=re.I)[0]
    lines = question.splitlines()
    first_option = next(
        (index for index, line in enumerate(lines) if _LABELED_OPTION.match(line)), None
    )
    if first_option is not None and sum(
        _LABELED_OPTION.match(line) is not None for line in lines[first_option:]
    ) >= 2:
        question = "\n".join(lines[:first_option])
    return re.sub(r"<image\s+\d+>", "", question, flags=re.I).strip()


def _reference(record: dict, task_key: str, directory: Path) -> tuple[Any, list[str] | None]:
    aliases = record.get("answer_variants")
    aliases = [_clean_text(value) for value in aliases] if isinstance(aliases, list) else None
    if task_key == "IU-Xray/report_generation":
        return _clean_text(record.get("report")), aliases
    if task_key == "PathText/caption":
        return _clean_text(record.get("caption")), aliases
    if task_key == "PathText/report_generation":
        report_path = directory / _clean_text(record.get("report"))
        return report_path.read_text(encoding="utf-8", errors="replace").strip(), aliases
    if task_key == "ROCOv2/caption":
        return _clean_text(record.get("caption")), aliases
    if task_key == "ROCOv2/concepts":
        return sorted({
            _clean_label(value) for value in record.get("cui") or [] if _clean_label(value)
        }), None
    if task_key == "MIMIC-CXR/multilabel":
        labels = _mimic_cxr_labels(record)
        return sorted(label for label, value in labels.items() if value == 1 or value == 1.0), None
    if task_key == "MX-CXR/grounding":
        width = float(record.get("image_width") or 0)
        height = float(record.get("image_height") or 0)
        if width <= 0 or height <= 0:
            raise ValueError("MX-CXR records require positive image_width and image_height")
        boxes = []
        for annotation in record.get("annotations") or []:
            x, y, width, height = annotation.get("bbox_xywh") or [0, 0, 0, 0]
            boxes.append({
                "label": _clean_text(_first(annotation, "label_text", "category")),
                "category": _clean_text(annotation.get("category")),
                "bbox_xyxy": [
                    float(x) / float(record["image_width"]) * _GROUNDING_COORDINATE_SCALE,
                    float(y) / float(record["image_height"]) * _GROUNDING_COORDINATE_SCALE,
                    float(x + width) / float(record["image_width"]) * _GROUNDING_COORDINATE_SCALE,
                    float(y + height) / float(record["image_height"]) * _GROUNDING_COORDINATE_SCALE,
                ],
            })
        return boxes, None
    if task_key == "DrVD-Bench/joint_reasoning":
        return [
            _letter(record.get("modality_answer")), _letter(record.get("organ_answer")),
            _letter(record.get("lesion_answer")), _letter(record.get("diagnosis_answer")),
        ], None
    if task_key == "3MDBench/diagnosis":
        return _clean_text(record.get("diagnosis")).casefold(), aliases
    if task_key.startswith("MedDocBench/"):
        return _clean_text(record.get("answer")).replace("\\n", "\n"), aliases
    value = _first(record, "gpt4_answer", "gt_answer", "Answer", "answer", "caption", "report")
    if task_key.startswith("MIMIC-Ext/"):
        # An empty source list explicitly means that no finding/object satisfies the query.
        return list(record.get("answer") or []), aliases
    if isinstance(value, list):
        return [_clean_text(item) for item in value], aliases
    return _clean_text(value), aliases


def _canonical_slake_closed_label(value: Any) -> str:
    label = _clean_text(value).casefold()
    return _SLAKE_CLOSED_ALIASES.get(label, label)


def _media_values(record: dict, task_key: str) -> list[Any]:
    if task_key == "LiveClin/mcqa":
        return [item.get("image") or item.get("file") for item in record.get("image_details") or []]
    if task_key == "MIMICEchoQA/mcqa":
        candidates = _first(record, "videos", "video")
        return candidates if isinstance(candidates, list) else ([candidates] if candidates else [])
    if task_key.startswith("MTBBench/"):
        return list(record.get("file_paths") or [])
    candidates = _first(record, "images", "image_paths", "image", "image_path", "videos", "video")
    if candidates is None:
        return []
    values = candidates if isinstance(candidates, list) else [candidates]
    return [value for value in values if value not in (None, "")]


def _runtime_media(directory: Path, values: list[Any]) -> list[dict[str, Any]]:
    media = []
    for value in values:
        if isinstance(value, dict):
            value = _first(value, "image", "file", "path", "url")
        text = _clean_text(value)
        if not text:
            continue
        if text.startswith("data:image/"):
            media.append({"kind": "image", "source": text, "media_id": f"img_{len(media)}"})
            continue
        path = Path(text)
        if not path.is_absolute():
            path = directory / path
        suffix = path.suffix.lower()
        if suffix in _IMAGE_EXTENSIONS:
            media.append({"kind": "image", "source": path, "media_id": f"img_{len(media)}"})
        elif suffix in _VIDEO_EXTENSIONS:
            media.append({"kind": "video", "source": path, "media_id": f"video_{len(media)}"})
    return media


def _declared_video_metadata(record: dict) -> tuple[float | None, int | None]:
    """Duration/frame count the dataset declares for its clip (SurgeryVideoQA ships both)."""
    metadata = record.get("video_metadata")
    if not isinstance(metadata, dict):
        return None, None
    original = metadata.get("original") if isinstance(metadata.get("original"), dict) else metadata
    duration, frames = original.get("duration"), original.get("frame_count")
    return (
        float(duration) if isinstance(duration, (int, float)) else None,
        int(frames) if isinstance(frames, (int, float)) and frames > 0 else None,
    )


@lru_cache(maxsize=None)
def _probed_video_metadata(path_text: str) -> tuple[float | None, int | None]:
    """Read duration/frame count from a container header without decoding any frame.

    Provenance must not add a failure mode of its own: when ``av`` is absent or the container
    is unreadable the fields stay unknown and the request-time sampler reports the real error.
    """
    try:
        import av
    except ImportError:
        return None, None
    try:
        with av.open(path_text) as container:
            stream = container.streams.video[0]
            total_frames = int(stream.frames or 0) or None
            if stream.duration is not None and stream.time_base is not None:
                return float(stream.duration * stream.time_base), total_frames
            if container.duration is not None:
                return float(container.duration / av.time_base), total_frames
            return None, total_frames
    except Exception:
        return None, None


def _media_info(record: dict, media: list[dict[str, Any]], config: Any) -> MediaInfo:
    """Describe the media a request carries, including the video frame-sampling plan.

    Videos are decoded lazily at request time, so what is recorded here is the plan the
    executor follows: ``max_video_frames`` bounded by the remaining image slots, capped by the
    frames the source actually holds. ``media_provenance`` in results.jsonl still carries the
    per-frame indices that were really sent.
    """
    num_images = sum(item["kind"] == "image" for item in media)
    videos = [item for item in media if item["kind"] == "video"]
    if not videos:
        return MediaInfo(num_images=num_images)
    media_config = getattr(config, "media", None)
    max_frames = getattr(media_config, "max_video_frames", 32)
    max_images = getattr(media_config, "max_images", 64)
    strategy = getattr(media_config, "video_frame_sampling_strategy", "uniform")
    # Per-record metadata describes one clip, so only trust it when the record has one video.
    declared = _declared_video_metadata(record) if len(videos) == 1 else (None, None)
    durations, sampled_frames, used_slots = [], 0, num_images
    for item in videos:
        duration, total_frames = declared
        if duration is None or total_frames is None:
            probed_duration, probed_frames = _probed_video_metadata(str(item["source"]))
            duration = duration if duration is not None else probed_duration
            total_frames = total_frames or probed_frames
        budget = min(max_frames, max(0, max_images - used_slots)) if max_images is not None else max_frames
        frames = min(total_frames, budget) if total_frames else budget
        used_slots += frames
        sampled_frames += frames
        if duration is not None:
            durations.append(duration)
    return MediaInfo(
        num_images=num_images,
        num_videos=len(videos),
        num_video_frames=sampled_frames,
        video_duration_seconds=sum(durations) if durations else None,
        frame_sampling_strategy=strategy,
    )


def _matches_task(record: dict, task_key: str) -> bool:
    if task_key in _DOCUMENT_SPLITS:
        return record.get("dataset") == _DOCUMENT_SPLITS[task_key]
    if task_key.startswith("MIMIC-Ext/"):
        return record.get("semantic_type") == task_key.rsplit("/", 1)[1]
    if task_key == "HLM/mcqa":
        return record.get("answer_type") == "multipleChoice"
    if task_key == "HLM/exact":
        return record.get("answer_type") == "exactMatch"
    if task_key == "MMMU-Health-Medicine/mcqa":
        return record.get("question_type") == "multiple-choice"
    if task_key == "MMMU-Health-Medicine/open":
        return record.get("question_type") == "open"
    suffix = task_key.rsplit("/", 1)[-1]
    if suffix in {"closed", "open"} and task_key.split("/", 1)[0] in {"VQA-RAD", "SLAKE", "PathVQA", "Quilt-VQA"}:
        answer_type = _clean_text(record.get("answer_type")).lower()
        return answer_type == suffix
    return True


def _profile_fields(profile: str, task_key: str, options: list[tuple[str, str]]) -> tuple[str, str, str]:
    if profile in {"closed", "multistage_closed"}:
        if not options:
            if task_key == "SLAKE/closed":
                return "classification", "label", "accuracy"
            return "classification", "yes_no", "accuracy"
        return "multiple_choice", "single_choice", "accuracy"
    if profile == "fixed_text":
        return "multiple_label", "label_set", "multilabel"
    if profile == "fixed_diagnosis":
        return "classification", "label", "accuracy"
    if profile == "multistage":
        return "multistage_choice", "ordered_choices", "multistage_choice"
    if profile == "multilabel":
        return "multiple_label", "label_set", "multilabel"
    if profile == "grounding":
        return "visual_grounding", "grounding_json", "grounding"
    if profile == "document_parse":
        return "document_understanding", "free_text", "document_fields"
    # ``video_open`` is SurgeryVideoQA, whose references are 1-6 tokens ("Lingual nerve.") and are
    # scored by token overlap; it belongs with the other short-answer tasks, not with free text.
    if profile in {"short_open", "document_qa", "video_open"}:
        return "open_ended", "short_answer", "vlm_text_overlap"
    if profile == "document_complex_qa":
        return "document_understanding", "free_text", "vlm_text_overlap"
    return "open_ended", "free_text", "vlm_text_overlap"


def _parse_label_set(text: str, labels: list[str]) -> list[str] | None:
    stripped = final_answer_region(text or "")
    if not stripped:
        return None
    try:
        parsed = json.loads(_strip_json_fence(stripped))
        if isinstance(parsed, dict):
            parsed = parsed.get("labels") or parsed.get("findings") or parsed.get("concepts")
        if isinstance(parsed, list):
            return sorted({_clean_text(item) for item in parsed if _clean_text(item)})
    except json.JSONDecodeError:
        pass
    # Do not project arbitrary prose onto the current sample's known labels. That used to
    # silently discard an unsupported extra label ("atelectasis and pneumonia" became only
    # ["atelectasis"]), inflating precision and exact-set scores. Free-form lists retain every
    # item; canonicalization against a dataset-level universe happens in the evaluator.
    return sorted({part.strip(" .") for part in re.split(r"[,;\n]", stripped) if part.strip(" .")})


def _strip_json_fence(text: str) -> str:
    match = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text, flags=re.I | re.S)
    return match.group(1).strip() if match else text.strip()


def _read_mtb_evidence(directory: Path, record: dict) -> tuple[str, list[str]]:
    """Load the non-image clinical files referenced by an MTBBench question."""
    sections = []
    relative_files = []
    for value in record.get("file_paths") or []:
        relative = _clean_text(value)
        path = directory / relative
        if path.suffix.lower() not in {".txt", ".csv", ".json"}:
            continue
        relative_files.append(relative)
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            sections.append(f"[{relative}]\n{text}")
    return "\n\n".join(sections), relative_files


def _source_label_universe(task_key: str, directory: Path) -> list[str]:
    """Return a task-level label vocabulary without using the current sample's gold answer."""
    return list(_cached_source_label_universe(task_key, str(directory.resolve())))


@lru_cache(maxsize=None)
def _cached_source_label_universe(task_key: str, directory_text: str) -> tuple[str, ...]:
    directory = Path(directory_text)
    if task_key == "ROCOv2/concepts":
        records = json.loads((directory / "rocov2_test.json").read_text(encoding="utf-8"))
        return tuple(sorted({
            _clean_label(label)
            for record in records
            for label in record.get("cui") or []
            if _clean_label(label)
        }))
    if task_key == "MIMIC-CXR/multilabel":
        records = json.loads((directory / "mimic_cxr_test.json").read_text(encoding="utf-8"))
        return tuple(sorted({
            _clean_text(label)
            for record in records
            for label in _mimic_cxr_labels(record)
            if _clean_text(label)
        }))
    if task_key == "MX-CXR/grounding":
        records = json.loads((directory / "ms_cxr_test.json").read_text(encoding="utf-8"))
        return tuple(sorted({
            _clean_text(annotation.get("category"))
            for record in records
            for annotation in record.get("annotations") or []
            if _clean_text(annotation.get("category"))
        }))
    if task_key.startswith("MIMIC-Ext/"):
        records = json.loads(
            (directory / "mimic_ext_mimic_cxr_vqa_test.json").read_text(encoding="utf-8")
        )
        semantic_type = task_key.rsplit("/", 1)[1]
        labels = set()
        for record in records:
            if record.get("semantic_type") != semantic_type:
                continue
            labels.update(_clean_text(value) for value in record.get("answer") or [])
            # CHOOSE candidates may be findings (attribute) or anatomical sites (object).
            # Category names, gender, and view position are prompt metadata unless a source
            # answer actually uses them, so adding every template field would dilute Hamming
            # loss and macro metrics with labels that can never be predicted for this task.
            for key, group in (record.get("template_arguments") or {}).items():
                if semantic_type != "choose" or key not in {"attribute", "object"}:
                    continue
                if isinstance(group, dict):
                    labels.update(_clean_text(value) for value in group.values())
        return tuple(sorted(label for label in labels if label))
    return ()


def _parse_grounding(text: str) -> list[dict] | None:
    answer = final_answer_region(text or "")
    try:
        payload = json.loads(_strip_json_fence(answer))
        if isinstance(payload, dict):
            payload = payload.get("predictions") or payload.get("boxes") or [payload]
        if isinstance(payload, list):
            normalized = []
            for item in payload:
                if not isinstance(item, dict):
                    continue
                item = dict(item)
                if "bbox_xyxy" not in item and "bbox_2d" in item:
                    item["bbox_xyxy"] = item.pop("bbox_2d")
                normalized.append(item)
            return normalized
    except (json.JSONDecodeError, TypeError):
        pass
    # Never synthesize a box from loose digits. The first four numbers in an answer region are
    # usually enumeration or measurements ("1. Analyze the Request ... 4. Format the Output"
    # yielded [1,2,3,4]), so that fallback reported a fabricated prediction with IoU 0 instead
    # of an honest parse failure — and it masked real fenced JSON answers.
    return None


def _parse_ordered_choices(
    text: str,
    valid_letters_by_stage: list[list[str]],
) -> list[str] | None:
    """Parse one ordered option per stage without sorting or de-duplicating choices."""
    text = final_answer_region(text or "")
    if not text or not valid_letters_by_stage:
        return None

    stage_count = len(valid_letters_by_stage)

    def parse_structured_region(region: str) -> list[str] | None:
        region = region.strip().strip("`*_# ")
        # A final full stop or enclosing list punctuation is formatting, not prose.
        region = region.strip().strip("[]{}() ").rstrip(".。")
        matches = list(re.finditer(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])", region.upper()))
        if len(matches) != stage_count:
            return None
        residual_parts = []
        cursor = 0
        for match in matches:
            residual_parts.append(region[cursor:match.start()])
            cursor = match.end()
        residual_parts.append(region[cursor:])
        residual = "".join(residual_parts)
        # Permit list separators and numbered-stage formatting, but reject explanation text.
        if re.sub(r"[\s\d.,;:/|\[\]{}()*_#>`\-→]+", "", residual):
            return None
        choices = [match.group(1).upper() for match in matches]
        if any(
            choice not in {letter.upper() for letter in valid_letters_by_stage[index]}
            for index, choice in enumerate(choices)
        ):
            return None
        return choices

    boxed = re.findall(r"\\boxed\{([^}]*)\}", text, flags=re.IGNORECASE)
    for region in reversed(boxed):
        parsed = parse_structured_region(region)
        if parsed:
            return parsed

    marker = re.compile(
        r"(?:final\s+answer|answer|selected\s+letters?|selections?)\s*(?:is|are|[:=\-])\s*",
        re.IGNORECASE,
    )
    for match in reversed(list(marker.finditer(text))):
        parsed = parse_structured_region(text[match.end():].splitlines()[0])
        if parsed:
            return parsed

    lines = [line for line in text.splitlines() if line.strip()]
    # Models commonly put the requested structured answer before an explanation, or after it
    # on the final line. Do not scan arbitrary middle lines that may only discuss alternatives.
    candidate_lines = lines[:1]
    if len(lines) > 1:
        candidate_lines.append(lines[-1])
    for region in candidate_lines or [text]:
        parsed = parse_structured_region(region)
        if parsed:
            return parsed
    return None


class MedicalVLMAdapter(BaseBenchmarkAdapter):
    """One registered class parameterized by the declarative VLM task catalog."""

    benchmark_version = "1.0"
    adapter_version = "1.5"
    prompt_template_name = "medical_vlm"
    prompt_template_version = "1.5"

    def __init__(self, entry, config=None) -> None:
        super().__init__(entry=entry, config=config)
        self.task_key = entry.key
        self.spec: VLMTaskSpec = VLM_TASK_SPECS[self.task_key]
        self.benchmark_name = entry.benchmark_name

    def discover_source_files(self) -> list[Path]:
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(
                f"{self.task_key} provides only the fixed 'test' split; requested '{self.split}'."
            )
        directory = self.get_benchmark_directory()
        files: list[Path] = []
        for pattern in self.spec.files:
            matched = sorted(directory.glob(pattern)) if any(char in pattern for char in "*?[") else [directory / pattern]
            files.extend(path for path in matched if path not in files)
        return files

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        directory = self.get_benchmark_directory()
        # ``files`` also contains auxiliary sources whose bytes enter prompts or references
        # (MTBBench clinical records and PathText reports). Only the catalog's top-level JSON
        # or JSONL patterns contain benchmark rows.
        data_files = []
        for pattern in self.spec.files:
            if any(char in pattern for char in "*?["):
                continue
            path = directory / pattern
            if path.suffix.lower() in {".json", ".jsonl"}:
                data_files.append(path)
        for path in data_files:
            rel = self.rel_path(path)
            with open(path, "r", encoding="utf-8") as handle:
                if path.suffix.lower() == ".jsonl":
                    records = (json.loads(line) for line in handle if line.strip())
                else:
                    records = self._json_records(handle)
                for index, record in enumerate(records):
                    if not isinstance(record, dict) or not _matches_task(record, self.task_key):
                        continue
                    if self.task_key == "DrVD-Bench/visual_evidence":
                        question = _record_question(record, self.task_key)
                        options = _options(record, self.task_key, question)
                        # One source row has a gold answer absent from all answer choices.
                        if _closed_reference(record, self.task_key, options) is None:
                            continue
                    if self.task_key == "PathText/report_generation":
                        report_path = self.get_benchmark_directory() / _clean_text(
                            record.get("report")
                        )
                        if not report_path.read_text(
                            encoding="utf-8", errors="replace"
                        ).strip():
                            continue
                    if self.task_key == "ROCOv2/concepts":
                        # 105 source rows annotate the concept list as ``["nan"]``, i.e. no
                        # concept at all. The reference collapses to an empty label set that no
                        # non-empty prediction can score above zero, so the sample is
                        # unanswerable rather than hard.
                        if not [
                            value for value in record.get("cui") or [] if _clean_label(value)
                        ]:
                            continue
                    if self.task_key.startswith("MTBBench/"):
                        question = _record_question(record, self.task_key)
                        options = _options(record, self.task_key, question)
                        if _closed_reference(record, self.task_key, options) not in {
                            label for label, _ in options
                        }:
                            continue
                    if self.task_key == "LiveClin/mcqa":
                        policy = ((record.get("exam_creation") or {}).get("final_policy") or {})
                        scenario = _clean_text(policy.get("scenario"))
                        scenario_images = list(policy.get("scenario_image_details") or [])
                        scenario_tables = list(policy.get("scenario_table_details") or [])
                        for stage_index, mcq in enumerate(policy.get("mcqs") or []):
                            expanded = dict(mcq)
                            expanded["scenario"] = scenario
                            expanded["image_details"] = [
                                *scenario_images, *(mcq.get("image_details") or []),
                            ]
                            expanded["table_details"] = [
                                *scenario_tables, *(mcq.get("table_details") or []),
                            ]
                            expanded["case_metadata"] = {
                                key: record.get(key) for key in ("pmc", "title", "Level1", "Level2", "Rarity", "ICD-10")
                            }
                            yield {
                                "record": expanded, "source_file_rel": rel,
                                "source_record_index": index, "sub_index": stage_index,
                            }
                    else:
                        yield {"record": record, "source_file_rel": rel, "source_record_index": index}

    @staticmethod
    def _json_records(handle) -> Iterable[dict]:
        """Stream top-level JSON arrays when ijson is installed; keep a stdlib fallback."""
        try:
            import ijson
        except ImportError:
            payload = json.load(handle)
            yield from payload if isinstance(payload, list) else payload.values()
            return
        yield from ijson.items(handle, "item", use_float=True)

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        record = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]
        directory = self.get_benchmark_directory()
        question = _record_question(record, self.task_key)
        if self.task_key == "HLM/mcqa":
            question, _ = _split_hle_question(question)
        options = _options(record, self.task_key, _record_question(record, self.task_key))
        question = _question_stem(question, options)
        context_meta: dict[str, Any] = {}
        mtb_evidence_files: list[str] = []
        if self.task_key.startswith("MTBBench/"):
            evidence, mtb_evidence_files = _read_mtb_evidence(directory, record)
            if evidence:
                prompt = _prompt_text("en")
                block = "\n".join(f"{label}. {text}" for label, text in options)
                fixed_prompt = (
                    f"Clinical context:\n{_clean_text(record.get('context'))}\n\n"
                    f"Question:\n{_question_stem(_clean_text(record.get('question')), options)}\n\n"
                    f"{prompt['answer_choices']}\n{block}\n\n{prompt['final_letter']}"
                )
                generation = getattr(self.config, "generation", None)
                evidence, context_meta = fit_context_to_window(
                    evidence,
                    fixed_prompt=fixed_prompt,
                    max_model_len=getattr(
                        getattr(self.config, "hardware", None), "max_model_len", None
                    ),
                    max_output_tokens=getattr(generation, "max_tokens", None),
                    reserve_tokens=getattr(generation, "context_token_reserve", 512),
                    policy=getattr(generation, "context_overflow_policy", "error"),
                    ascii_chars_per_token=2,
                )
                question = f"Clinical evidence files:\n{evidence}\n\n{question}"
        if self.task_key == "MMMU-Health-Medicine/mcqa" and not question:
            question = "Based on the provided medical image(s), select the most appropriate answer."
        reference, aliases = _reference(record, self.task_key, directory)
        if self.spec.profile in {"closed", "multistage_closed"}:
            reference = _closed_reference(record, self.task_key, options)
            if not options:
                raw_ref = _first(
                    record, "yes_no_answer", "answer", "gt_answer", "correct_answer"
                )
                normalized_binary = parse_yes_no_maybe(_clean_text(raw_ref))
                reference = normalized_binary or _clean_text(raw_ref).casefold()
        if self.task_key == "SLAKE/closed":
            reference = _canonical_slake_closed_label(record.get("answer"))
        if self.task_key == "MIMIC-Ext/verify":
            raw_ref = reference[0] if isinstance(reference, list) and reference else reference
            reference = parse_yes_no_maybe(_clean_text(raw_ref))

        if self.task_key == "LiveClin/mcqa":
            scenario = _clean_text(record.get("scenario"))
            tables = "\n\n".join(
                f"{_clean_text(item.get('caption_prefix') or item.get('caption'))}:\n"
                f"{_clean_text(item.get('content'))}"
                for item in record.get("table_details") or []
                if _clean_text(item.get("content"))
            )
            if tables:
                scenario = f"{scenario}\n\nReference tables:\n{tables}".strip()
            question = f"Clinical scenario:\n{scenario}\n\n{question}" if scenario else question

        task_type, answer_format, metric = _profile_fields(self.spec.profile, self.task_key, options)
        media = _runtime_media(directory, _media_values(record, self.task_key))
        media_kinds = {item["kind"] for item in media}
        source_id = _clean_text(_first(
            record, "id", "question_id", "messages_id", "unique_id", "_id", "idx", "image_id", "qid", "No", "index",
        )) or f"{rel}:{rec_index}"
        if "sub_index" in raw_sample:
            source_id = f"{source_id}:stage:{raw_sample['sub_index']}"
        input_payload = {"question": question, "options": options, "media_count": len(media)}
        sample_id = self.make_sample_id(
            source_file_rel=rel, source_sample_id=source_id,
            content_hash=self.input_hash(input_payload),
        )

        metadata: dict[str, Any] = {
            "task_profile": self.spec.profile,
            "letters": [label for label, _ in options],
            "option_text": {label: text for label, text in options},
            **context_meta,
        }
        if mtb_evidence_files:
            metadata["clinical_evidence_files"] = mtb_evidence_files
        for key in (
            "language", "q_lang", "subject", "specialty", "modality", "modality_type",
            "category", "question_type", "content_type", "task_label", "clinical_phase",
            "stage", "case_metadata", "dataset", "dept_name", "doc_type", "image_width", "image_height",
        ):
            if record.get(key) not in (None, "", [], {}):
                metadata[key] = record.get(key)
        if self.task_key == "MIMIC-CXR/multilabel":
            labels = _mimic_cxr_labels(record)
            evaluated_labels = [
                label for label, value in labels.items() if value in (0, 0.0, 1, 1.0)
            ]
            metadata.update({
                "label_universe": _source_label_universe(self.task_key, directory),
                "evaluated_labels": sorted(evaluated_labels),
                "negative_labels": sorted(label for label, value in labels.items() if value == 0),
                "uncertain_labels": sorted(label for label, value in labels.items() if value == -1),
                "null_labels": sorted(label for label, value in labels.items() if value is None),
                "uncertain_label_policy": "excluded",
                "null_label_policy": "excluded",
            })
        elif metric == "multilabel":
            metadata["label_universe"] = _source_label_universe(self.task_key, directory)
            if self.task_key == "MIMIC-Ext/choose":
                arguments = record.get("template_arguments") or {}
                candidate_labels = {
                    _clean_text(value)
                    for key in ("attribute", "object")
                    for value in (arguments.get(key) or {}).values()
                    if _clean_text(value)
                }
                metadata["candidate_labels"] = sorted(candidate_labels)
        if self.task_key == "SLAKE/closed":
            # SLAKE's official CLOSED split is not MCQA: it has no options and contains
            # bilingual yes/no-like and fixed categorical labels.
            metadata.update({
                "labels": list(_SLAKE_CLOSED_LABELS),
                "label_aliases": _SLAKE_CLOSED_ALIASES,
                "source_answer_type": record.get("answer_type"),
                "closed_task_semantics": "fixed_label_without_options",
            })
        if self.task_key == "3MDBench/diagnosis":
            metadata["labels"] = list(_THREEMD_DIAGNOSIS_LABELS)
        if self.task_key == "DrVD-Bench/joint_reasoning":
            stage_names = ["modality", "organ", "lesion", "diagnosis"]
            metadata.update({
                "case_id": source_id,
                "stage_names": stage_names,
                "stage_letters": [
                    [
                        match.group(1)
                        for value in record.get(f"{stage_name}_options") or []
                        if (match := _LABELED_OPTION.match(_clean_text(value)))
                    ]
                    for stage_name in stage_names
                ],
            })
        if self.task_key.startswith("MTBBench/"):
            metadata.update({
                "case_id": _clean_text(record.get("patient_id")) or source_id,
                "stage_name": f"block_{record.get('block_idx', 0)}",
                "stage_index": record.get("block_idx"),
                "cohort": record.get("cohort"),
            })
        if self.task_key == "LiveClin/mcqa":
            metadata.update({
                "case_id": (record.get("case_metadata") or {}).get("pmc"),
                "stage_name": record.get("stage"),
                "stage_index": raw_sample.get("sub_index"),
            })
        if self.task_key == "MX-CXR/grounding":
            metadata.update({
                "coordinate_format": "xyxy_normalized_0_1000",
                "coordinate_scale": int(_GROUNDING_COORDINATE_SCALE),
                "source_image_width": record.get("image_width"),
                "source_image_height": record.get("image_height"),
                "category_universe": _source_label_universe(self.task_key, directory),
            })

        language = _normalized_language(record, self.task_key)
        modality = "Video" if "video" in media_kinds else ("Image" if "image" in media_kinds else "Text")
        return EvaluationSample(
            sample_id=sample_id,
            source_sample_id=source_id,
            sample_index=sample_index,
            benchmark_name=self.benchmark_name,
            benchmark_version=self.benchmark_version,
            benchmark_split=self.split,
            source_benchmark_entry=rel,
            source_file=rel,
            source_record_index=rec_index,
            source_record_hash=self.input_hash(record),
            input_hash=self.input_hash(input_payload),
            reference_hash=self.reference_hash(reference),
            input_type="multimodal" if media else "text",
            task_type=task_type,
            component="Multimodal",
            capability="Multimodal",
            specialty=_clean_text(_first(record, "specialty", "subject", "department", "body_system")) or None,
            language=language,
            modality=modality,
            difficulty=_clean_text(_first(record, "difficulty", "topic_difficulty")) or None,
            answer_format=answer_format,
            evaluation_metric=metric,
            source_content={
                "question": question,
                "options": [{"label": label, "text": text} for label, text in options],
            },
            reference_answer=reference,
            reference_answer_normalized=reference,
            reference_aliases=aliases,
            media=_media_info(record, media, self.config),
            metadata=metadata,
            runtime_media=media,
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        content: list[dict] = []
        for item in sample.runtime_media:
            content.append({
                "type": item["kind"], "source": item["source"], "media_id": item["media_id"],
            })
        question = sample.source_content["question"]
        options = sample.source_content.get("options") or []
        prompt = _prompt_text(sample.language)
        if options:
            block = "\n".join(f"{item['label']}. {item['text']}" for item in options)
            question = f"{question}\n\n{prompt['answer_choices']}\n{block}"
            if sample.answer_format == "single_choice":
                question += f"\n\n{prompt['final_letter']}"
            else:
                question += f"\n\n{prompt['open_with_options']}"
        elif sample.evaluation_metric == "multilabel":
            allowed_labels = (
                sample.metadata.get("candidate_labels")
                if self.task_key == "MIMIC-Ext/choose"
                else sample.metadata.get("label_universe")
            ) or []
            if allowed_labels:
                question += (
                    "\n\nAllowed labels (use only these exact strings):\n"
                    f"{json.dumps(allowed_labels, ensure_ascii=False)}"
                )
            question += f"\n\n{prompt['json_labels']}"
        elif sample.answer_format == "grounding_json":
            # A dataset-level closed category set, never the current sample's gold category.
            categories = sample.metadata.get("category_universe") or []
            if categories:
                question += (
                    "\n\nAllowed finding categories (use only these exact strings):\n"
                    f"{json.dumps(categories, ensure_ascii=False)}"
                )
        elif sample.answer_format == "yes_no":
            question += f"\n\n{prompt['yes_no']}"
        elif sample.answer_format == "label":
            question += f"\n\n{prompt['concise_label']}"
        elif sample.answer_format == "short_answer":
            # These references are 1-3 tokens, so an unconstrained answer collapses token
            # precision and can never reach exact match. The brevity instruction is required,
            # not cosmetic.
            question += f"\n\n{prompt['short_answer']}"
            if sample.language != "en" and prompt["same_language"]:
                question += f"\n\n{prompt['same_language']}"
        elif sample.language != "en" and prompt["same_language"]:
            question += f"\n\n{prompt['same_language']}"
        content.append({"type": "text", "text": question})
        return [{"role": "user", "content": content}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        if sample.answer_format == "single_choice":
            return parse_multiple_choice_letter(raw_response, sample.metadata.get("letters") or list(_LETTERS))
        if sample.answer_format == "yes_no":
            return parse_yes_no_maybe(raw_response)
        if sample.answer_format == "label":
            labels = sample.metadata.get("labels") or []
            return parse_label(
                raw_response,
                [str(label) for label in labels],
                aliases=sample.metadata.get("label_aliases"),
            )
        if sample.answer_format == "ordered_choices":
            return _parse_ordered_choices(
                raw_response,
                sample.metadata.get("stage_letters")
                or [list(_LETTERS) for _ in sample.reference_answer],
            )
        if sample.answer_format == "label_set":
            return _parse_label_set(raw_response, sample.metadata.get("label_universe") or [])
        if sample.answer_format == "grounding_json":
            return _parse_grounding(raw_response)
        return final_answer_region(raw_response or "")
