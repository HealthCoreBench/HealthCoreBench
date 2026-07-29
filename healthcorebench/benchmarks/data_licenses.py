"""Per-corpus data-license declarations, keyed by benchmark directory.

``BaseBenchmarkAdapter`` exposes three data-protection flags (``redistribution_allowed``,
``store_full_input_allowed``, ``store_reference_allowed``) that the runtime honours:

* ``redistribution_allowed=False`` makes ``run_setup._check_data_protection`` refuse to run
  the task when an LLM judge is configured against a non-local ``base_url`` -- scoring would
  ship the restricted records to a third party.
* ``store_full_input_allowed=False`` blanks ``source_content`` in ``samples.jsonl`` and
  suppresses ``formatted_prompt`` in ``results.jsonl``, so a run directory that gets copied
  or attached to a report does not carry the source records with it.
* ``store_reference_allowed=False`` additionally blanks the gold answer.

Those flags default to ``True`` on the base class, and no adapter overrode them, so both
gates were dead code: every corpus, including the PhysioNet credentialed ones, was treated
as freely redistributable. This table is what makes them live.

It is keyed on the *directory* (``47_EHRBench``, ``7_MIMIC-CXR``) rather than the task key,
because a licence attaches to a corpus and not to the individual tasks derived from it --
``EHRBench/decision`` and ``EHRBench/risk`` cannot have different licences. Directory
basenames are unique across ``medical_llm_benchmarks`` and ``medical_vlm_benchmarks``.

Corpora absent from the table keep the permissive defaults. The table is therefore a list of
*determinations that were actually made*, not a claim about the other ~95 corpora; adding an
entry is how a determination gets recorded, including the ones that came out permissive.

``store_reference_allowed`` stays ``True`` throughout. The restricted material in these
corpora is the input (clinical notes, radiology reports, chest films); the references are
short labels or single clinical actions. Blanking them would make a run's artifacts
impossible to re-score offline while protecting nothing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataLicense:
    """A licence determination for one benchmark corpus.

    ``evidence`` records what the determination was read off, so a reviewer can re-check it
    without re-deriving the provenance: a file under the corpus directory where one exists,
    otherwise the field in the shipped data that identifies the source.
    """

    license_name: str
    evidence: str
    redistribution_allowed: bool = True
    store_full_input_allowed: bool = True
    store_reference_allowed: bool = True
    note: str = ""


_PHYSIONET = "PhysioNet Credentialed Health Data License 1.5.0"

# Clause 3 of the PhysioNet credentialed licence -- "The LICENSEE will not share access to
# PhysioNet restricted data with anyone else" -- is what closes the remote-judge path for the
# corpora below. Clause 4 ("maintain the physical and electronic security of PhysioNet
# restricted data") is why their source records are kept out of the persisted artifacts.
_CREDENTIALED = dict(
    redistribution_allowed=False,
    store_full_input_allowed=False,
    store_reference_allowed=True,
)

DATA_LICENSES: dict[str, DataLicense] = {
    # -- PhysioNet credentialed (MIMIC-derived) --------------------------------------- #
    "7_MIMIC-CXR": DataLicense(
        license_name=_PHYSIONET,
        evidence="benchmarks/medical_vlm_benchmarks/7_MIMIC-CXR/LICENSE.txt",
        **_CREDENTIALED,
    ),
    "12_MedNLI": DataLicense(
        license_name=_PHYSIONET,
        # No licence file ships with the corpus. mli_test_v1.jsonl carries verbatim MIMIC-III
        # note sentences ("In the ED, initial VS revealed T 98.9, HR 73, ..."); MedNLI is
        # distributed through PhysioNet under the credentialed licence.
        evidence="mli_test_v1.jsonl contains verbatim MIMIC-III clinical note sentences",
        **_CREDENTIALED,
    ),
    "20_Medical-CXR-VQA": DataLicense(
        license_name=_PHYSIONET,
        evidence="benchmarks/medical_vlm_benchmarks/20_Medical-CXR-VQA/README.md (MIMIC-CXR)",
        **_CREDENTIALED,
    ),
    "25_MIMICEchoQA": DataLicense(
        license_name=_PHYSIONET,
        evidence=(
            "benchmarks/medical_vlm_benchmarks/25_MIMICEchoQA/README.md "
            "(MIMIC-IV, PhysioNet credentialed, doi:10.13026/ef48-v217)"
        ),
        **_CREDENTIALED,
    ),
    "27_MX-CXR": DataLicense(
        license_name=_PHYSIONET,
        evidence="benchmarks/medical_vlm_benchmarks/27_MX-CXR/README.md (MS-CXR over MIMIC-CXR)",
        **_CREDENTIALED,
        note="MS-CXR Local Alignment v1.1.0 annotations over MIMIC-CXR images.",
    ),
    "30_MIMIC-Ext-MIMIC-CXR-VQA": DataLicense(
        license_name=_PHYSIONET,
        evidence="corpus is the MIMIC-Ext CXR-VQA release; images/ are MIMIC-CXR studies",
        **_CREDENTIALED,
    ),
    "45_MIMIC-CDM": DataLicense(
        license_name=_PHYSIONET,
        # mimic_cdm_test.json records are keyed on hadm_id and carry Patient History,
        # Laboratory Tests, Microbiology, Radiology and Discharge Diagnosis verbatim.
        evidence="mimic_cdm_test.json records carry hadm_id and full MIMIC-IV admission text",
        **_CREDENTIALED,
    ),
    "46_EHRNoteQA": DataLicense(
        license_name=_PHYSIONET,
        evidence="ehrnoteqa_test.jsonl questions are grounded in MIMIC-IV discharge notes",
        **_CREDENTIALED,
        note="EHRNoteQA/mcqa is disabled in the registry; the declaration still applies.",
    ),
    "47_EHRBench": DataLicense(
        license_name=_PHYSIONET,
        evidence="benchmarks/medical_llm_benchmarks/47_EHRBench/README.md (MIMIC-IV, PhysioNet)",
        **_CREDENTIALED,
    ),

    # -- determinations that came out permissive -------------------------------------- #
    # Recorded so the next reader does not have to re-derive them, and so a later change of
    # mind is a visible diff rather than a fresh judgement call.
    "48_MedArabiQ": DataLicense(
        license_name="NYU internal non-commercial research licence",
        evidence="benchmarks/medical_llm_benchmarks/48_MedArabiQ/LICENSE",
        note=(
            "Grants use 'solely for your internal non-commercial research and evaluation' and "
            "withholds rights to 'sublicense or further distribute'. That bars re-hosting the "
            "corpus, which is not what these flags gate; the records are public benchmark "
            "items, not restricted patient data, so redistribution_allowed stays True and the "
            "four judge-scored MedArabiQ tasks keep running. The non-commercial term is an "
            "obligation on the operator, not something the pipeline can enforce."
        ),
    ),
    "34_MedQA-CS": DataLicense(
        license_name="CC BY-NC 4.0",
        evidence="benchmarks/medical_llm_benchmarks/34_MedQA-CS/LICENSE.md",
        note=(
            "CC BY-NC permits sharing with attribution; the NonCommercial term restricts how "
            "the operator may use the results, not whether the pipeline may score the corpus."
        ),
    ),
}


def license_for(benchmark_dir: str) -> DataLicense | None:
    """Return the declaration for a benchmark directory, or ``None`` if none was recorded.

    ``benchmark_dir`` may be a full registry path (``benchmarks/medical_llm_benchmarks/
    47_EHRBench``) or a bare directory name; only the final component is matched.
    """
    if not benchmark_dir:
        return None
    return DATA_LICENSES.get(benchmark_dir.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1])
