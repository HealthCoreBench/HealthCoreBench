"""Offline tools that operate purely on a run directory's JSONL logs.

None of these call the evaluated model. reparse re-extracts answers from stored raw
responses; rescore re-runs evaluators over stored results; validate-run checks integrity;
export-parquet converts JSONL to Parquet; migrate-legacy converts old-format results.
"""

from healthcorebench.tools.reparse import reparse_run
from healthcorebench.tools.rescore import rescore_run
from healthcorebench.tools.validate import validate_run
from healthcorebench.tools.export_parquet import export_parquet
from healthcorebench.tools.migrate_legacy import migrate_legacy

__all__ = ["reparse_run", "rescore_run", "validate_run", "export_parquet", "migrate_legacy"]
