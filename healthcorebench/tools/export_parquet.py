"""Export a run's JSONL logs to Parquet for large-scale analysis.

Each JSONL file becomes a Parquet file under ``<run_dir>/parquet/``. Nested/among-record
schema variation is handled by JSON-encoding complex columns (dict/list) to strings so
Parquet's columnar typing stays stable.
"""

from __future__ import annotations

import json
from pathlib import Path

from healthcorebench.utils.jsonl import read_jsonl


def export_parquet(run_dir: str | Path, out_dir: str | Path | None = None) -> dict:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir / "parquet"
    out_dir.mkdir(parents=True, exist_ok=True)

    import pyarrow as pa
    import pyarrow.parquet as pq

    written = {}
    for name in ("samples", "results", "judgments", "attempts"):
        src = run_dir / f"{name}.jsonl"
        if not src.exists():
            continue
        rows = read_jsonl(src)
        if not rows:
            continue
        flat = [_flatten(r) for r in rows]
        table = pa.Table.from_pylist(flat)
        dest = out_dir / f"{name}.parquet"
        pq.write_table(table, dest)
        written[name] = {"path": str(dest), "rows": len(rows)}
    return {"written": written, "out_dir": str(out_dir)}


def _flatten(record: dict) -> dict:
    """JSON-encode complex nested values so column types are stable across records."""
    out = {}
    for k, v in record.items():
        if isinstance(v, (dict, list)):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = v
    return out
