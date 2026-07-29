"""Profile-driven adapter for all 20 heterogeneous MedS-Bench tasks.

Known source-data limitations that the adapter surfaces rather than hides:

* ``task12_pubmedqa_classification.json`` — all 650 instances carry ``output == ["no"]``. The
  task declares ``accuracy``, so a constant-"no" predictor scores 1.000 and the number carries
  no information about the model. The data cannot be repaired here, so every sample records
  ``degenerate_label_distribution`` plus the observed ``reference_label_distribution`` in its
  metadata (see ``_label_distribution``) for downstream reporting to surface. The same audit
  runs for the other small closed-label classification tasks (task16 0.504 / task123 0.552 /
  task131 0.502 majority share) so the majority baseline is always visible next to the score.
* ``task100_ebms_answer_vertification.json`` — 304 instances cover only 110 unique inputs; one
  input repeats 36 times with 36 different accepted answers. Instances are grouped by input so
  each question is emitted once with every accepted answer in ``reference_aliases``.
* the ``set`` tasks ship a comma-joined *string* (not a per-entity list) as ``output``, and
  entity names legitimately contain commas inside brackets. Splitting is bracket-aware per span
  and identical for gold and prediction; task106's closed label set is matched as whole phrases.
* a handful of ``set`` golds are truncated in the source itself — task1 ships
  ``"... 13 age-matched typically developing ( TD"`` and ``"15 ) , ASD, 18 ) , and controls, 21"``,
  task3 ships ``"airway resistance changes ( ΔsRAW"``. Their brackets cannot be balanced by any
  splitting rule, so the 3 task1 / 2 task3 items that stay bracket-unbalanced are a property of
  the data, not of the parser; the item *boundaries* around them are already correct.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.answer_parsing import (
    final_answer_region,
    parse_label,
    parse_multiple_choice_letter,
    parse_yes_no_maybe,
)
from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkDataNotFoundError
from healthcorebench.schemas.sample import EvaluationSample


@dataclass(frozen=True)
class MedSTaskSpec:
    relative_file: str
    task_type: str
    answer_format: str
    evaluation_metric: str
    parser: str
    capability: str = "Reasoning"


MEDS_TASK_SPECS: dict[str, MedSTaskSpec] = {
    "task1": MedSTaskSpec("Information_extraction/task1_participant_extraction.json", "information_extraction", "multi_label", "multilabel", "set"),
    "task2": MedSTaskSpec("Information_extraction/task2_intervention_extraction.json", "information_extraction", "multi_label", "multilabel", "set"),
    "task3": MedSTaskSpec("Information_extraction/task3_outcome_extraction.json", "information_extraction", "multi_label", "multilabel", "set"),
    "task12": MedSTaskSpec("Fact_verication/task12_pubmedqa_classification.json", "classification", "yes_no", "accuracy", "yes_no", "Knowledge"),
    "task16": MedSTaskSpec("Fact_verication/task16_test_healthfact_classification.json", "classification", "label", "accuracy", "label", "Knowledge"),
    "task18": MedSTaskSpec("Explanation/task18_test_healthfact_sentence_generation.json", "explanation", "free_text", "llm_judge", "text"),
    "task29": MedSTaskSpec("Information_extraction/task29_drug_dose_extraction.json", "information_extraction", "short_answer", "any_of_match", "text"),
    "task46": MedSTaskSpec("Explanation/task46_do_entity_explanation.json", "explanation", "free_text", "llm_judge", "text", "Knowledge"),
    "task50": MedSTaskSpec("Explanation/task50_biolord_explanation.json", "explanation", "free_text", "llm_judge", "text", "Knowledge"),
    "task74": MedSTaskSpec("Information_extraction/task74_pmc_patient_case_report_basic_information_extraction.json", "information_extraction", "structured_text", "document_fields", "text"),
    "task100": MedSTaskSpec("Fact_verication/task100_ebms_answer_vertification.json", "fact_verification", "free_text", "llm_judge", "text"),
    "task106": MedSTaskSpec("Text_classification/task106_hoc_text_classification.json", "multilabel_classification", "multi_label", "multilabel", "set", "Knowledge"),
    "task122": MedSTaskSpec("MCQA/task122_medmcqa_test_set.json", "multiple_choice", "single_choice", "accuracy", "choice", "Knowledge"),
    "task123": MedSTaskSpec("MCQA/task123_pubmedqa_test_set.json", "classification", "yes_no_maybe", "accuracy", "yes_no_maybe", "Knowledge"),
    "task125": MedSTaskSpec("NER/task125_test_bc4chem_named_enetity_recognition.json", "named_entity_recognition", "multi_label", "multilabel", "set", "Knowledge"),
    "task126": MedSTaskSpec("NER/task126_test_bc5chem_named_enetity_recognition.json", "named_entity_recognition", "multi_label", "multilabel", "set", "Knowledge"),
    "task127": MedSTaskSpec("NER/task127_test_bc5disease_named_enetity_recognition.json", "named_entity_recognition", "multi_label", "multilabel", "set", "Knowledge"),
    "task128": MedSTaskSpec("NER/task128_test_species800_named_enetity_recognition.json", "named_entity_recognition", "multi_label", "multilabel", "set", "Knowledge"),
    "task130": MedSTaskSpec("Diagnosis/task130_DDXPlus_text_classification_test.json", "classification", "label", "accuracy", "diagnosis"),
    "task131": MedSTaskSpec("Treatment_planning/task131_SEER_text_classification_test.json", "classification", "label", "accuracy", "treatment"),
}

_NULL_SET_ANSWERS = {
    "not found",
    "there is no related enetity",
    "there is no related entity",
}
# Tasks whose gold answer is one input repeated with several accepted outputs. Grouping by input
# turns 304 task100 instances into 110 questions, each carrying every accepted answer.
_GROUPED_BY_INPUT_TASKS = frozenset({"task100"})
# Closed-label classification tasks whose source is small enough to audit with a second pass.
# task122 (3980 distinct golds) is not a closed label set and task130's source is 169 MB, so
# neither is scanned; their samples simply carry no distribution metadata.
_LABEL_DISTRIBUTION_TASKS = frozenset({"task12", "task16", "task123", "task131"})
_HOC_LABELS = [
    "Sustaining proliferative signaling",
    "Evading growth suppressors",
    "Resisting cell death",
    "Enabling replicative immortality",
    "Inducing angiogenesis",
    "Activating invasion and metastasis",
    "Genomic instability and mutation",
    "Tumor promoting inflammation",
    "Cellular energetics",
    "Avoiding immune destruction",
]
_DIAGNOSIS_LABELS = [
    "Acute COPD exacerbation / infection", "Acute dystonic reactions", "Acute laryngitis",
    "Acute otitis media", "Acute pulmonary edema", "Acute rhinosinusitis",
    "Allergic sinusitis", "Anaphylaxis", "Anemia", "Atrial fibrillation", "Boerhaave",
    "Bronchiectasis", "Bronchiolitis", "Bronchitis", "Bronchospasm / acute asthma exacerbation",
    "Chagas", "Chronic rhinosinusitis", "Cluster headache", "Croup", "Ebola", "Epiglottitis",
    "GERD", "Guillain-Barré syndrome", "HIV (initial infection)", "Influenza",
    "Inguinal hernia", "Larygospasm", "Localized edema", "Myasthenia gravis", "Myocarditis",
    "PSVT", "Pancreatic neoplasm", "Panic attack", "Pericarditis", "Pneumonia",
    "Possible NSTEMI / STEMI", "Pulmonary embolism", "Pulmonary neoplasm", "SLE",
    "Sarcoidosis", "Scombroid food poisoning", "Spontaneous pneumothorax",
    "Spontaneous rib fracture", "Tuberculosis", "URTI", "Unstable angina",
    "Viral pharyngitis", "Whooping cough",
]
_TREATMENT_LABELS = [
    "Intraoperative rad with other rad before/after surgery",
    "Intraoperative radiation",
    "No radiation and/or cancer-directed surgery",
    "Radiation after surgery",
    "Radiation before and after surgery",
    "Radiation prior to surgery",
    "Surgery both before and after radiation",
]


def _definition(data: dict) -> str:
    value = data.get("Definition")
    if isinstance(value, list):
        return " ".join(str(item) for item in value).strip()
    return str(value or "").strip()


def _outputs(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in values if str(item).strip()]


def _strip_reference_prefix(value: str, parser: str) -> str:
    patterns = {
        "diagnosis": r"^the\s+diagnosis\s+result\s+is\s*[:]?\s*",
        "treatment": r"^the\s+treatment\s+planning\s+is\s*[:]?\s*",
    }
    if parser in patterns:
        value = re.sub(patterns[parser], "", value, flags=re.IGNORECASE)
    return value.strip().rstrip(".").strip()


def _match_form(value: str) -> str:
    """Lower-cased alphanumeric-only form used for label/sentinel comparison."""
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _is_null_set_answer(text: str) -> bool:
    """True when the answer *states* the empty set anywhere in the response.

    The sentinel is a whole clause ("There is no related entity.", "Not found") rather than the
    whole response: models routinely wrap the correct negative in prose ("The sentence does not
    provide specific names of chemicals. Therefore, the output is: There is no related entity."),
    which the previous whole-text equality check turned into two bogus entities. Matching a
    complete clause keeps a partial negative ("there is no related entity for benzene, but
    toluene is present") out of the sentinel path.
    """
    for clause in re.split(r"[.!?;:\n]+", text):
        if _match_form(clause) in _NULL_SET_ANSWERS:
            return True
    return False


def _delimited_segments(text: str) -> list[tuple[str, str]]:
    """Split on ``,``/``;``, keeping each piece paired with the delimiter that followed it."""
    tokens = re.split(r"([,;])", text)
    return [
        (tokens[index], tokens[index + 1] if index + 1 < len(tokens) else "")
        for index in range(0, len(tokens), 2)
    ]


def _bracket_depth_after(segment: str, depth: int) -> int:
    """Running bracket depth after reading ``segment``, floored at zero for stray closers."""
    for char in segment:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
    return depth


def _split_set_text(text: str) -> list[str]:
    """Split a comma/semicolon separated answer, keeping bracketed spans intact.

    Chemical and outcome names contain commas inside brackets ("( 22E , 24R ) - 6 beta -
    methoxyergosta - 7 , 22 - diene - 3 beta , 5 alpha - diol"), so a delimiter is only a
    separator at bracket depth zero. Gold and prediction go through the same function.

    Bracket depth is decided per span rather than once for the whole answer. A single truncated
    annotation ("... ( imidazo [ 1 , 5 - a ] pyridin - 1 - yl") used to switch depth tracking off
    for the entire string, so the well-formed names listed before it were shredded at their
    internal commas too. Now a span whose brackets never close is split at its own delimiters —
    the right reading for a truncated annotation — while every balanced span around it stays
    whole.
    """
    parts: list[str] = []
    pending: list[tuple[str, str]] = []
    depth = 0
    for segment, delimiter in _delimited_segments(text):
        pending.append((segment, delimiter))
        depth = _bracket_depth_after(segment, depth)
        if depth > 0:
            continue  # an unclosed bracket makes this delimiter part of one name
        parts.append("".join(s + d for s, d in pending[:-1]) + pending[-1][0])
        pending, depth = [], 0
    # A bracket left open at the end of the answer marks a truncated annotation rather than a
    # grouped name, so its delimiters separate items after all.
    parts.extend(segment for segment, _ in pending)
    return [item for item in (_clean_set_item(part) for part in parts) if item]


def _clean_set_item(part: str) -> str:
    """Strip list decoration a model adds around one item (bullets, emphasis, enumeration).

    Task definitions present their label sets as numbered lists ("1. Sustaining proliferative
    signaling, 2. Evading growth suppressors, ..."), so models answer in kind. The enumeration
    prefix must be removed or every returned label is unmatchable. The pattern requires the digit
    to be glued to its delimiter and followed by whitespace so chemical names split into "3" /
    "4 - dihydroxybenzoic acid" are left alone.
    """
    item = part.strip()
    item = re.sub(r"^[-*•]+\s+", "", item)
    item = re.sub(r"^(?:\*\*|__)|(?:\*\*|__)$", "", item).strip()
    item = re.sub(r"^(?:\d{1,2}[.)、]|\(\d{1,2}\))\s+(?=\S)", "", item)
    return item.strip().rstrip(".").strip()


def _labels_in_text(text: str, labels: list[str]) -> list[str]:
    """Return every closed-set label that appears in ``text`` as a whole phrase."""
    haystack = _match_form(text)
    found = []
    for label in labels:
        needle = _match_form(label)
        if needle and re.search(
            r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])", haystack
        ):
            found.append(label)
    return found


def _set_items(value: Any, labels: list[str] | None = None) -> list[str]:
    """Normalize a set-valued answer (gold or prediction) into canonical items."""
    text = ", ".join(_outputs(value)).strip()
    if not text:
        return []
    if labels:
        # A closed label set makes phrase matching exact: the answer may enumerate, bullet, or
        # embed the labels in prose, and none of those forms survive a comma split.
        return _labels_in_text(text, labels)
    if _is_null_set_answer(text):
        return []
    return _split_set_text(text)


class MedSBenchOpenAdapter(BaseBenchmarkAdapter):
    benchmark_name = "MedS-Bench"
    benchmark_version = "1.0"
    adapter_version = "2.0"
    prompt_template_name = "meds_task_profile"
    prompt_template_version = "2.0"

    @property
    def task_key(self) -> str:
        task = self.entry.task
        if task not in MEDS_TASK_SPECS:
            raise BenchmarkDataNotFoundError(
                f"Unknown MedS-Bench task '{task}'; expected one of {sorted(MEDS_TASK_SPECS)}."
            )
        return task

    @property
    def spec(self) -> MedSTaskSpec:
        return MEDS_TASK_SPECS[self.task_key]

    def discover_source_files(self) -> list[Path]:
        return [self.get_benchmark_directory() / self.spec.relative_file]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        source = files[0]
        rel = self.rel_path(source)
        definition, instances = self._stream_source(source)
        labels = self._label_universe()
        distribution = self._label_distribution(source)
        if self.task_key in _GROUPED_BY_INPUT_TASKS:
            instances = self._group_by_input(instances)
        for index, instance in enumerate(instances):
            input_text = str(instance.get("input") or "").strip()
            references = _outputs(instance.get("output"))
            if input_text and references:
                yield {
                    "instance": instance,
                    "definition": definition,
                    "input": input_text,
                    "references": references,
                    "labels": labels,
                    "label_distribution": distribution,
                    "source_file_rel": rel,
                    "source_record_index": index,
                }

    @staticmethod
    def _group_by_input(instances: Iterable[dict]) -> list[dict]:
        """Collapse instances sharing an input, keeping every accepted output.

        ``task100`` ships 304 instances over 110 unique questions (one repeated 36 times with 36
        different valid answers). Emitting them separately asked the judge for equivalence to one
        arbitrary accepted answer and weighted that question 36x in the mean.
        """
        grouped: dict[str, dict] = {}
        for instance in instances:
            key = str(instance.get("input") or "").strip()
            outputs = _outputs(instance.get("output"))
            entry = grouped.get(key)
            if entry is None:
                grouped[key] = {**instance, "output": list(outputs),
                                "source_instance_count": 1}
                continue
            entry["source_instance_count"] += 1
            for value in outputs:
                if value not in entry["output"]:
                    entry["output"].append(value)
        return list(grouped.values())

    def _label_distribution(self, source: Path) -> dict | None:
        """Audit the reference-label distribution for the small closed-label tasks.

        Recorded on every sample so a reader can compare the reported accuracy with the majority
        baseline. ``task12`` is fully degenerate (650/650 instances answer "no"), so its accuracy
        is uninformative and the ``degenerate_label_distribution`` flag says so explicitly.
        """
        if self.task_key not in _LABEL_DISTRIBUTION_TASKS:
            return None
        counts: dict[str, int] = {}
        _, instances = self._stream_source(source)
        for instance in instances:
            references = _outputs(instance.get("output"))
            if not references:
                continue
            label = _strip_reference_prefix(references[0], self.spec.parser).lower()
            counts[label] = counts.get(label, 0) + 1
        total = sum(counts.values())
        if not total:
            return None
        majority = max(counts.values()) / total
        return {
            "reference_label_distribution": dict(sorted(counts.items())),
            "reference_label_count": total,
            "majority_label_share": round(majority, 6),
            # A single-valued gold column makes the declared accuracy metric unable to separate
            # any model from a constant predictor.
            "degenerate_label_distribution": len(counts) < 2,
        }

    @staticmethod
    def _stream_source(source: Path) -> tuple[str, Iterable[dict]]:
        """Read the task definition once and stream large instance arrays when possible."""
        try:
            import ijson
        except ImportError:
            with source.open(encoding="utf-8") as handle:
                data = json.load(handle)
            return _definition(data), iter(data.get("Instances") or [])

        with source.open("rb") as handle:
            raw_definition = next(ijson.items(handle, "Definition"), None)
        definition = _definition({"Definition": raw_definition})

        def _instances():
            with source.open("rb") as handle:
                yield from ijson.items(handle, "Instances.item", use_float=True)

        return definition, _instances()

    def _label_universe(self) -> list[str] | None:
        if self.task_key == "task16":
            return ["0", "1", "2"]
        if self.task_key == "task106":
            return list(_HOC_LABELS)
        if self.spec.parser == "diagnosis":
            return list(_DIAGNOSIS_LABELS)
        if self.spec.parser == "treatment":
            return list(_TREATMENT_LABELS)
        return None

    def _reference(self, raw_sample: dict) -> Any:
        references = raw_sample["references"]
        if self.spec.parser == "set":
            return _set_items(references, raw_sample.get("labels"))
        if self.spec.parser in {"diagnosis", "treatment"}:
            return _strip_reference_prefix(references[0], self.spec.parser)
        if self.spec.parser == "choice":
            match = re.search(r"\b(?:answer\s+is\s+)?([A-D])\s*[:.)]", references[0], re.I)
            return match.group(1).upper() if match else None
        if self.spec.parser in {"yes_no", "yes_no_maybe"}:
            return references[0].lower()
        if self.spec.evaluation_metric == "any_of_match":
            return references
        return references[0]

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        instance = raw_sample["instance"]
        definition = raw_sample["definition"]
        input_text = raw_sample["input"]
        references = raw_sample["references"]
        reference = self._reference(raw_sample)
        if reference is None:
            raise ValueError(f"Unable to parse reference for {self.task_key} sample {sample_index}")
        rel = raw_sample["source_file_rel"]
        record_index = raw_sample["source_record_index"]
        source_id = str(instance.get("id") or f"{rel}:{record_index}")
        labels = raw_sample["labels"]
        metadata = {"task": self.task_key, "source_task_file": self.spec.relative_file}
        if labels:
            metadata["labels"] = labels
            metadata["label_universe"] = labels
        if raw_sample.get("label_distribution"):
            metadata.update(raw_sample["label_distribution"])

        reference_answer = reference
        reference_normalized = reference
        aliases = None
        if self.spec.evaluation_metric == "any_of_match":
            reference_answer = reference[0]
        elif self.task_key in _GROUPED_BY_INPUT_TASKS and len(references) > 1:
            # Every accepted answer for this question, so the judge is asked for equivalence to
            # any of them rather than to one arbitrarily chosen accepted answer.
            aliases = references
            metadata["accepted_answers"] = references
            metadata["source_instance_count"] = instance.get("source_instance_count")
        return EvaluationSample(
            sample_id=self.make_sample_id(
                source_file_rel=rel,
                source_sample_id=source_id,
                content_hash=self.input_hash({"definition": definition, "input": input_text}),
            ),
            source_sample_id=source_id,
            sample_index=sample_index,
            benchmark_name=self.benchmark_name,
            benchmark_version=self.benchmark_version,
            benchmark_split=self.split,
            source_benchmark_entry=rel,
            source_file=rel,
            source_record_index=record_index,
            source_record_hash=self.input_hash(instance),
            input_hash=self.input_hash({"definition": definition, "input": input_text}),
            reference_hash=self.reference_hash(reference),
            input_type="text",
            task_type=self.spec.task_type,
            component="Language",
            capability=self.spec.capability,
            specialty=self.task_key,
            language="en",
            modality="Text",
            answer_format=self.spec.answer_format,
            evaluation_metric=self.spec.evaluation_metric,
            source_content={"definition": definition, "input": input_text},
            reference_answer=reference_answer,
            reference_answer_normalized=reference_normalized,
            reference_aliases=aliases,
            metadata=metadata,
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        content = sample.source_content
        prompt = "\n\n".join(part for part in (content["definition"], content["input"]) if part)
        suffix = {
            "choice": "Return only the option letter.",
            "yes_no": "Return only yes or no.",
            "yes_no_maybe": "Return only yes, no, or maybe.",
            "label": "Return only the requested class label.",
            "diagnosis": "Return only one diagnosis label from the task definition.",
            "treatment": "Return only one treatment label from the task definition.",
            "set": "Return only the requested comma-separated entities or labels.",
        }.get(self.spec.parser)
        if self.spec.evaluation_metric == "any_of_match":
            suffix = "Return only one final extracted answer without explanation."
        if self.spec.parser == "set" and self._label_universe():
            # The definition presents the labels as a numbered list, so say explicitly that the
            # answer is scored against those labels and that the numbers are not part of them.
            suffix = ("Return only the applicable labels from the list above, copied verbatim and "
                      "separated by commas. Do not include their numbers, and add no explanation.")
        if suffix:
            prompt = f"{prompt}\n\n{suffix}"
        return [{"role": "user", "content": prompt}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        text = (raw_response or "").strip()
        if not text:
            return None
        parser = self.spec.parser
        if parser == "choice":
            return parse_multiple_choice_letter(text, list("ABCD"))
        if parser in {"yes_no", "yes_no_maybe"}:
            value = parse_yes_no_maybe(text)
            return value if parser == "yes_no_maybe" or value in {"yes", "no"} else None
        if parser in {"label", "diagnosis", "treatment"}:
            return parse_label(text, (sample.metadata or {}).get("labels") or [])
        if parser == "set":
            return _set_items(
                final_answer_region(text), (sample.metadata or {}).get("labels")
            )
        return text
