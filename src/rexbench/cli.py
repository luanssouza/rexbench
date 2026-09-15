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
from rexbench.stats.aggregate import aggregate_across_seeds, wide_pivot
from rexbench.stats.significance import complete_cases, friedman_test, kendall_w, quade_test


def _run_id(experiment_name: str) -> str:
    return f"{experiment_name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def cmd_run(args: argparse.Namespace) -> None:
    from rexbench.core.runner import run_experiment  # lazy: needs torch/RecBole/EBPR/PGPR,
    # `split`/`aggregate`/`stats` don't and shouldn't require that whole environment just to
    # import this module.

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


def cmd_split(args: argparse.Namespace) -> None:
    """Materialize (or verify) every dataset's train/val/test split to
    `dataset.split.store_dir`, without touching any model -- run this locally, then copy
    the resulting directories (e.g. via rsync/scp) to another machine (a Lightning AI
    Studio) so `rexbench run` there loads the identical split instead of recomputing it.
    Only needs pandas/pydantic -- not torch/RecBole/EBPR/PGPR -- so this works even in an
    environment that doesn't have the full merged environment installed."""
    from rexbench.core.dataset import build_dataset_bundle  # lazy: keeps `rexbench run`'s
    # heavier import chain out of the (already lightweight) aggregate/stats path above.

    config = load_config(args.config)
    for dataset_config in config.datasets:
        store_dir = dataset_config.split.store_dir
        if store_dir is None:
            print(f"{dataset_config.name}: split.store_dir not set, skipping (in-memory split, recomputed on every run)")
            continue
        already_persisted = (Path(store_dir) / "split_meta.json").exists()
        bundle = build_dataset_bundle(dataset_config)
        verb = "verified existing" if already_persisted else "wrote new"
        print(
            f"{dataset_config.name}: {verb} split at {store_dir} "
            f"(train={len(bundle.train)}, val={len(bundle.val)}, test={len(bundle.test)} rows, "
            f"{bundle.num_users} users, {bundle.num_items} items)"
        )


_PGPR_KG_REQUIRED = ["entities", "relations", "mappings", "train.txt", "test.txt"]


def cmd_stage_pgpr_kg(args: argparse.Namespace) -> None:
    """Copies PGPR's own pre-existing KG dataset format (entities/, relations/, mappings/,
    train.txt, test.txt -- see data/README.md's PGPR row) from wherever you already have it
    on this machine into rexbench's own data/raw/pgpr_kg/<dataset>, so it's gitignored and
    travels alongside data/raw/ and data/splits/ the same way when you rsync to another
    machine (e.g. a Lightning AI Studio).

    Unlike `rexbench split`, this isn't a rexbench-computed artifact -- PGPR doesn't use
    DatasetBundle's split/sample mechanism at all (a disclosed scope limitation, see
    REGISTRY.md), it trains on its own pre-existing, pre-split data verbatim. This command
    is a plain validated copy, not a recomputation, so there's nothing to "reuse vs.
    recompute" here the way there is for rexbench split -- it either finds a complete source
    directory and copies it, or refuses and tells you what's missing."""
    source = Path(args.source)
    missing = [name for name in _PGPR_KG_REQUIRED if not (source / name).exists()]
    if missing:
        raise SystemExit(
            f"{source} is missing expected PGPR KG files/dirs: {missing} "
            f"(see data/README.md's PGPR row for the full expected layout) -- refusing to "
            f"stage an incomplete copy."
        )
    dest = Path("data") / "raw" / "pgpr_kg" / args.dataset
    if dest.exists():
        print(f"{dest} already exists, leaving it as-is (delete it first to re-stage from {source})")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest)
    print(f"Staged PGPR KG data for {args.dataset!r}: {source} -> {dest}")


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
            # Friedman/Quade/Kendall's W need a fully populated dataset x model matrix --
            # check the shape *after* dropping datasets any model is missing (e.g. PGPR,
            # which only applies to ml100k/ml1m), not before, so a metric/k combination
            # that only looks big enough before accounting for that isn't silently reported
            # with NaN statistics (see stats/significance.py's complete_cases).
            if complete_cases(pivot).shape[0] < 2 or pivot.shape[1] < 2:
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

    split_p = sub.add_parser(
        "split", help="Materialize (or verify) every dataset's split to disk, for reuse across machines"
    )
    split_p.add_argument("--config", required=True)
    split_p.set_defaults(func=cmd_split)

    stage_pgpr_p = sub.add_parser(
        "stage-pgpr-kg", help="Copy PGPR's own pre-existing KG dataset into data/raw/pgpr_kg/<dataset>"
    )
    stage_pgpr_p.add_argument("--source", required=True, help="Existing dir with entities/relations/mappings/train.txt/test.txt")
    stage_pgpr_p.add_argument("--dataset", required=True, help="Dataset name, e.g. ml100k or ml1m")
    stage_pgpr_p.set_defaults(func=cmd_stage_pgpr_kg)

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
