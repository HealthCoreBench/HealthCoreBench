"""Single-writer lease for a run directory."""

from __future__ import annotations

import fcntl
import json
import os
import socket
import uuid
from pathlib import Path

from healthcorebench.utils.timestamps import utc_now_iso


class RunDirectoryLockedError(RuntimeError):
    pass


class RunDirectoryLease:
    """Hold an advisory process lock for the complete lifetime of one run/resume session."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / ".healthcorebench.lock"
        self._file = None
        self.info = {
            "session_id": f"session_{uuid.uuid4().hex[:16]}",
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "acquired_at": utc_now_iso(),
        }

    def __enter__(self) -> "RunDirectoryLease":
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a+", encoding="utf-8")
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._file.seek(0)
            owner = self._file.read().strip() or "unknown owner"
            self._file.close()
            self._file = None
            raise RunDirectoryLockedError(
                f"Run directory already has an active writer: {self.run_dir} ({owner})"
            ) from exc
        self._file.seek(0)
        self._file.truncate()
        json.dump(self.info, self._file, ensure_ascii=True, sort_keys=True)
        self._file.write("\n")
        self._file.flush()
        os.fsync(self._file.fileno())
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._file is None:
            return
        fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        self._file.close()
        self._file = None
