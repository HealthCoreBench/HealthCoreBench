"""Dependency-free streaming reader for a top-level JSON array."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Any


def iter_json_array(path: str | Path, *, chunk_size: int = 1 << 20) -> Iterator[Any]:
    """Yield items from a top-level JSON array without materializing the whole file."""
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    eof = False
    started = False
    max_item_buffer = max(chunk_size * 64, 64 << 20)

    with open(path, "r", encoding="utf-8") as stream:
        while True:
            if position >= len(buffer) and not eof:
                buffer = stream.read(chunk_size)
                position = 0
                eof = buffer == ""
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started:
                if position >= len(buffer) and not eof:
                    continue
                if position >= len(buffer) or buffer[position] != "[":
                    raise ValueError(f"expected a top-level JSON array in {path}")
                position += 1
                started = True

            while True:
                while position < len(buffer) and (buffer[position].isspace()
                                                   or buffer[position] == ","):
                    position += 1
                if position < len(buffer) and buffer[position] == "]":
                    return
                if position >= len(buffer):
                    break
                try:
                    item, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    break
                yield item
                position = end

            if eof:
                remaining = buffer[position:].strip()
                raise ValueError(f"truncated or malformed JSON array in {path}: {remaining[:80]}")
            buffer = buffer[position:] + stream.read(chunk_size)
            position = 0
            if len(buffer) > max_item_buffer:
                raise ValueError(f"one JSON array item in {path} exceeds the streaming limit")
