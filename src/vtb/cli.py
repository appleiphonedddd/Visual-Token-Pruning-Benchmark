"""Command line entry point.

    vtb run configs/experiments/smoke.yaml [dotlist overrides, e.g. dataset_overrides.limit=20]
    vtb aggregate outputs/smoke [--csv table.csv]
    vtb score outputs/smoke/<model>/<dataset>/<run>   # recompute metrics from predictions.jsonl
    vtb list
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vtb.core.registry import DATASETS, EVALUATORS, METHODS, MODELS


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="vtb")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="run an experiment YAML")
    p.add_argument("config")
    p.add_argument("overrides", nargs="*")

    p = sub.add_parser("aggregate", help="summarise metrics.json files into a table")
    p.add_argument("root")
    p.add_argument("--csv")

    p = sub.add_parser("score", help="re-score one run directory")
    p.add_argument("run_dir")

    sub.add_parser("list", help="list registered components")

    args = ap.parse_args(argv)
    if args.cmd == "run":
        from vtb.core.config import load_experiment
        from vtb.runner.experiment import Experiment

        Experiment(load_experiment(args.config, args.overrides)).run()
    elif args.cmd == "aggregate":
        from vtb.runner.aggregate import build_table, to_csv, to_markdown

        header, table, skipped = build_table(Path(args.root))
        print(to_markdown(header, table))
        if skipped:
            print("\nSkipped:\n" + "\n".join(f"  - {s}" for s in skipped))
        if args.csv:
            Path(args.csv).write_text(to_csv(header, table))
    elif args.cmd == "score":
        from vtb.runner.experiment import score_run
        from vtb.runner.io import write_json

        run_dir = Path(args.run_dir)
        metrics = json.loads((run_dir / "metrics.json").read_text())
        ds_cfg = metrics["config"]["dataset"]
        evaluator_name = DATASETS.get(ds_cfg["type"]).evaluator
        metrics["scores"] = score_run(run_dir, evaluator_name)
        metrics.update(EVALUATORS.build(evaluator_name).headline(metrics["scores"], ds_cfg.get("primary_metric")))
        write_json(run_dir / "metrics.json", metrics)
        print(json.dumps(metrics["scores"], indent=2))
    elif args.cmd == "list":
        for title, reg in (("models", MODELS), ("methods", METHODS), ("datasets", DATASETS), ("evaluators", EVALUATORS)):
            print(f"{title}: {', '.join(reg.names())}")


if __name__ == "__main__":
    main()
