"""`rexbench run|aggregate|stats` — the one-command entry point AUDIT.md's resubmission
goal asked for: every result in the paper regenerable from one config + one command."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from rexbench.config.load import load_config
from rexbench.core.determinism import build_manifest, resolve_device
from rexbench.core.runner import run_experiment
from rexbench.stats.aggregate import aggregate_across_seeds, wide_pivot
from rexbench.stats.significance import friedman_test, kendall_w, quade_test


def _run_id(experiment_name: str) -> str:
    return f"{experiment_name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def cmd_run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    run_dir = Path(config.experiment.output_dir) / _run_id(config.experiment.name)
    run_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(args.config, run_dir / "config.yaml")

    collector = run_experiment(config)
    collector.write(run_dir)

    device = resolve_device(config.determinism.device)
    manifest = build_manifest(config.model_dump(), config.determinism.seeds[0], device)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print(f"Run complete: {run_dir}")
    print(f"  results:  {len(collector.results)} rows")
    print(f"  failures: {len(collector.failures)} rows")
    print(f"  trials:   {len(collector.trials)} rows")
    if args.with_stats:
        cmd_stats(argparse.Namespace(run=str(run_dir), metric=None))


def cmd_aggregate(args: argparse.Namespace) -> None:
    run_dir = Path(args.run)
    results_df = pd.read_parquet(run_dir / "results.parquet")
    agg = aggregate_across_seeds(results_df)
    agg.to_csv(run_dir / "aggregated.csv", index=False)
    print(f"Wrote {run_dir / 'aggregated.csv'} ({len(agg)} rows)")


def cmd_stats(args: argparse.Namespace) -> None:
    run_dir = Path(args.run)
    results_df = pd.read_parquet(run_dir / "results.parquet")
    agg = aggregate_across_seeds(results_df)

    metrics = [args.metric] if args.metric else sorted(agg["metric"].unique())
    report = {}
    for metric in metrics:
        for k in sorted(agg.loc[agg["metric"] == metric, "k"].dropna().unique()) or [None]:
            pivot = wide_pivot(agg, metric, k)
            if pivot.shape[0] < 2 or pivot.shape[1] < 2:
                continue
            key = f"{metric}@{k}" if k is not None else metric
            report[key] = {
                "friedman": friedman_test(pivot),
                "quade": quade_test(pivot),
                "kendall_w": kendall_w(pivot),
            }
    out_path = run_dir / "statistics.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Wrote {out_path} ({len(report)} metric/k combinations tested)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rexbench")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run every dataset x model x explainer x seed combination in a config")
    run_p.add_argument("--config", required=True)
    run_p.add_argument("--with-stats", action="store_true")
    run_p.set_defaults(func=cmd_run)

    agg_p = sub.add_parser("aggregate", help="Mean +/- std across seeds for a completed run")
    agg_p.add_argument("--run", required=True)
    agg_p.set_defaults(func=cmd_aggregate)

    stats_p = sub.add_parser("stats", help="Friedman/Quade/Kendall's W for a completed run")
    stats_p.add_argument("--run", required=True)
    stats_p.add_argument("--metric", default=None)
    stats_p.set_defaults(func=cmd_stats)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
