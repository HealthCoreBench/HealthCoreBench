"""Documentation claims that are checkable against the registry and the filesystem.

Two kinds of drift had accumulated silently, both of which mislead a reader into looking in
the wrong place or quoting the wrong scope:

* 52 adapter docstrings named a benchmark directory that no longer existed, because the
  corpora were renumbered after the docstrings were written -- MIMIC-CDM was cited as
  number 49 when its directory is 45, MedQA-MCMLE as 105 when it is 70. (Written without
  the literal ``<number>_<Name>`` form on purpose: this file is one of the files scanned.)
* The README's headline counted 71/36 as *tasks* where they are *benchmarks*, off by the
  factor that multi-subset benchmarks contribute.

Neither is caught by any behavioural test: the code works, the prose is just wrong.
"""

from __future__ import annotations

import collections
import re
from pathlib import Path

import pytest

from healthcorebench.benchmarks.registry import get_registry
from healthcorebench.config import get_project_root

# A benchmark directory reference: a number, an underscore, then the corpus name. (Spelled as a
# pattern rather than with examples because this file is one of the files scanned.)
_DIRECTORY_TOKEN = re.compile(r"\b\d{1,3}_[A-Za-z][\w.\-]*")
_ROOTS = ("medical_llm_benchmarks", "medical_vlm_benchmarks")


def _present() -> set[str]:
    root = get_project_root() / "benchmarks"
    return {
        path.name
        for sub in _ROOTS
        if (root / sub).exists()
        for path in (root / sub).iterdir()
        if path.is_dir()
    }


def _documentation_files() -> list[Path]:
    root = get_project_root()
    files = sorted(root.joinpath("healthcorebench").rglob("*.py"))
    files += sorted(root.joinpath("tests").glob("*.py"))
    files += [path for path in (root / "README.md", root / "README_zh.md",
                                root / "benchmarks" / "README.md") if path.exists()]
    return files


def test_every_referenced_benchmark_directory_exists():
    """A renumbering must not leave prose pointing at a directory that is gone.

    This also covers ``data_licenses.py``, where a stale key is worse than a stale comment:
    the declaration silently stops applying while still reading as protective.
    """
    present = _present()
    if not present:
        pytest.skip("benchmark data directories unavailable")
    root = get_project_root()
    stale = [
        f"{path.relative_to(root)}:{lineno}  {token}"
        for path in _documentation_files()
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="ignore")
                                      .splitlines(), 1)
        for token in _DIRECTORY_TOKEN.findall(line)
        if token not in present
    ]
    assert not stale, "references to non-existent benchmark directories:\n" + "\n".join(stale)


def test_registry_directories_all_exist():
    present = _present()
    if not present:
        pytest.skip("benchmark data directories unavailable")
    missing = {key: entry.benchmark_dir for key, entry in get_registry().items()
               if Path(entry.benchmark_dir).name not in present}
    assert not missing, missing


# Anchored on the noun that follows the number rather than on the bolding, so that a count is
# only compared against the category it actually claims -- "Python 3.11" is not a task count.
_COUNT_CLAIMS = {
    ("Language", "benchmark"): (r"(\d{1,3}) language benchmarks", r"(\d{1,3}) 个语言 benchmark"),
    ("Multimodal", "benchmark"): (r"(\d{1,3}) multimodal benchmarks", r"(\d{1,3}) 个多模态 benchmark"),
    ("Language", "task"): (r"(\d{1,3}) language tasks", r"(\d{1,3}) 个语言任务"),
    ("Multimodal", "task"): (r"(\d{1,3}) multimodal tasks", r"(\d{1,3}) 个多模态任务"),
}


@pytest.mark.parametrize("readme", ["README.md", "README_zh.md"])
def test_readme_counts_match_the_registry(readme: str):
    """Every catalogue count the READMEs quote has to be the number the registry holds.

    The headline said "71 language tasks and 36 multimodal tasks" while 71/36 are the
    *benchmark* counts and the task counts are 141/56 -- the kind of error that survives
    indefinitely because both numbers are real, just of the wrong thing.
    """
    path = get_project_root() / readme
    if not path.exists():
        pytest.skip(f"{readme} absent")
    registry = get_registry()
    actual = {
        ("Language", "task"): sum(1 for e in registry.values() if e.component == "Language"),
        ("Multimodal", "task"): sum(1 for e in registry.values() if e.component == "Multimodal"),
    }
    per_benchmark = {e.benchmark_name: e.component for e in registry.values()}
    counts = collections.Counter(per_benchmark.values())
    actual[("Language", "benchmark")] = counts["Language"]
    actual[("Multimodal", "benchmark")] = counts["Multimodal"]

    # Only bolded spans: both READMEs bold the catalogue counts and leave incidental numbers
    # plain, which is what separates "**141 个语言任务**" from the adjacent sentence explaining
    # that 7 of them are disabled.
    bolded = "\n".join(re.findall(r"\*\*(.+?)\*\*", path.read_text(encoding="utf-8"), flags=re.S))
    for category, patterns in _COUNT_CLAIMS.items():
        stated = {int(number) for pattern in patterns for number in re.findall(pattern, bolded)}
        assert stated, f"{readme} no longer states a {category[0]} {category[1]} count."
        assert stated == {actual[category]}, (
            f"{readme} states {sorted(stated)} for {category[0]} {category[1]}s; "
            f"the registry holds {actual[category]}."
        )

    # Quoted in words ("Seven"/"7 个"), so the number is checked rather than the sentence.
    disabled = sum(1 for entry in registry.values() if not entry.enabled)
    assert disabled == 7, (
        f"{disabled} tasks are now disabled, but both READMEs describe seven. Update the prose."
    )


def test_benchmark_directory_basenames_are_unique():
    """``data_licenses.py`` keys on the basename alone; a collision would silently merge two
    corpora's declarations."""
    root = get_project_root() / "benchmarks"
    if not all((root / sub).exists() for sub in _ROOTS):
        pytest.skip("benchmark data directories unavailable")
    names = [path.name for sub in _ROOTS for path in (root / sub).iterdir() if path.is_dir()]
    duplicates = [name for name, count in collections.Counter(names).items() if count > 1]
    assert not duplicates, duplicates
