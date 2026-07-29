"""Best-effort git provenance capture.

Never fatal: if git is unavailable or the project is not a repository, every field is
``None`` / ``False`` and evaluation proceeds. This information is recorded in the manifest
so a run can be traced back to the exact code state.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def _run(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except Exception:
        return None


def collect_git_info(project_root: str | Path) -> dict:
    """Return git commit and dirty status for the project, tolerating any failure."""
    root = Path(project_root)
    commit = _run(["git", "rev-parse", "HEAD"], root)
    info: dict = {
        "git_commit": commit,
        "git_dirty": None,
        "git_branch": None,
    }
    if commit is None:
        # Not a git repo (or git missing): leave fields null rather than guessing.
        info["git_dirty"] = False if _run(["git", "rev-parse", "--is-inside-work-tree"], root) is None else info["git_dirty"]
        return info

    status = _run(["git", "status", "--porcelain"], root)
    info["git_dirty"] = bool(status) if status is not None else None
    info["git_branch"] = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], root)
    return info
