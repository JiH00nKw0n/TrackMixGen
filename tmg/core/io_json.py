"""JSON and JSON-lines files as every stage writes and reads them (standard library only).

``write_json`` is atomic (temp file then rename) so a reader never sees a half-written marker; ``read_jsonl``
skips blank lines, which is what every appender in the package produces (one ``json.dumps`` line per row).
"""
from __future__ import annotations

import json
from pathlib import Path


def write_json(path: Path, data) -> Path:
    """Write ``data`` as pretty JSON atomically (temp file then rename) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)
    return path


def read_json(path: Path):
    """Load a JSON file."""
    return json.loads(Path(path).read_text())


def read_jsonl(path: Path) -> list[dict]:
    """Every non-blank line of a JSON-lines file as a dict; an empty list when the file does not exist."""
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    """Append one row as a JSON line (the file is created when missing)."""
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
