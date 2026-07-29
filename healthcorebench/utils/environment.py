"""Best-effort software/environment version capture.

Never fatal: a missing optional dependency yields ``None`` for that field rather than
aborting the run. Recorded in the manifest so results can be reproduced.
"""

from __future__ import annotations

import platform
import sys
from importlib import metadata as importlib_metadata

from healthcorebench.version import FRAMEWORK_VERSION


def _pkg_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return None


def collect_environment_info() -> dict:
    """Collect Python / SDK / platform versions, tolerating any lookup failure."""
    return {
        "framework_version": FRAMEWORK_VERSION,
        "python_version": platform.python_version(),
        "openai_sdk_version": _pkg_version("openai"),
        "pydantic_version": _pkg_version("pydantic"),
        "datasets_version": _pkg_version("datasets"),
        "pillow_version": _pkg_version("pillow") or _pkg_version("Pillow"),
        "pyarrow_version": _pkg_version("pyarrow"),
        # vLLM is only present when the framework happens to run on the serving host; it is
        # recorded when available but is otherwise null (the model runs out-of-process).
        "vllm_version": _pkg_version("vllm"),
        "platform": platform.platform(),
        "platform_machine": platform.machine(),
        "executable": sys.executable,
    }
