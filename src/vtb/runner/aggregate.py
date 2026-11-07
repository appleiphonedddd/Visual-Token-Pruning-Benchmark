"""Collect metrics.json files into a paper-style table (scores per dataset, Acc., Rel.).

Acc. averages each dataset's score on a 0..100 scale (``score_pct``, e.g. MME
perception / 20); Rel. is the mean over datasets of score / vanilla score for the
same model, both as in CDPruner Table 1. ``prefill_ms`` is averaged over
datasets. Skipped runs are listed separately.
"""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from pathlib import Path

from vtb.runner.io import iter_metrics


def build_table(root: Path) -> tuple[list[str], list[list[str]], list[str]]:
    rows: dict[tuple, dict[str, float]] = defaultdict(dict)
    pct: dict[tuple, dict[str, float]] = defaultdict(dict)  # 0..100 scale, used for Acc.
    eff: dict[tuple, list[float]] = defaultdict(list)
    datasets: list[str] = []
    skipped: list[str] = []
    for m in iter_metrics(root):
        key = (m["model"], m["method"], m["budget"])
        if m.get("status") != "ok":
            skipped.append(f"{m['model']} / {m['dataset']} / {m['method']}: {m.get('reason', m.get('status'))}")
            continue
        rows[key][m["dataset"]] = m["score"]
        pct[key][m["dataset"]] = m.get("score_pct", m["score"])
        eff[key].append(m.get("efficiency", {}).get("avg_prefill_ms", float("nan")))
        if m["dataset"] not in datasets:
            datasets.append(m["dataset"])

    def sort_key(k):
        model, method, budget = k
        return (model, budget is not None, -(budget or 0), method != "vanilla", method)

    header = ["model", "method", "budget", *datasets, "Acc.", "Rel.", "prefill_ms"]
    table = []
    for key in sorted(rows, key=sort_key):
        model, method, budget = key
        scores = rows[key]
        base = next((v for (mo, me, _), v in rows.items() if mo == model and me == "vanilla"), None)
        vals = [scores.get(d) for d in datasets]
        present = list(pct[key].values())
        acc = sum(present) / len(present) if present else float("nan")
        rels = [scores[d] / base[d] for d in datasets if base and d in scores and base.get(d)]
        rel = 100 * sum(rels) / len(rels) if rels else float("nan")
        table.append([
            model, method, "all" if budget is None else str(budget),
            *["-" if v is None else f"{v:.1f}" for v in vals],
            f"{acc:.1f}", f"{rel:.1f}%", f"{sum(eff[key]) / len(eff[key]):.1f}",
        ])
    return header, table, skipped


def to_markdown(header: list[str], table: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in table]
    return "\n".join(lines)


def to_csv(header: list[str], table: list[list[str]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(table)
    return buf.getvalue()
