from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


class PredictionStore:
    """Append-only JSONL of per-sample records; lets interrupted runs resume."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:  # truncated last line after a crash
                        break
        return out

    def done_ids(self) -> set[str]:
        return {r["id"] for r in self.records()}

    def append(self, record: dict[str, Any]) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def iter_metrics(root: Path) -> Iterator[dict]:
    for p in sorted(root.rglob("metrics.json")):
        yield json.loads(p.read_text())
