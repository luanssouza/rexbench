"""Long-format results/failures tables — AUDIT.md's OUTPUT requirement: one row per
measurement, plus a separate first-class failures table so a non-ok combo never just
disappears from the run."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

RESULTS_COLUMNS = ["dataset", "model", "explainer", "metric", "seed", "k", "value", "status"]
FAILURES_COLUMNS = [
    "dataset", "model", "explainer", "seed", "status", "exception_type", "message",
    "traceback", "measurements", "requirement", "reason",
]


@dataclass
class ResultsCollector:
    results: list[dict] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)

    def add_result(self, dataset: str, model: str, explainer: str | None, metric: str, seed: int, k: Any, value: float) -> None:
        self.results.append({
            "dataset": dataset, "model": model, "explainer": explainer, "metric": metric,
            "seed": seed, "k": k, "value": value, "status": "ok",
        })

    def add_failure(
        self, dataset: str, model: str, explainer: str | None, seed: int, status: str,
        exception_type: str | None = None, message: str | None = None, traceback: str | None = None,
        measurements: dict | None = None, requirement: dict | None = None, reason: str | None = None,
    ) -> None:
        self.failures.append({
            "dataset": dataset, "model": model, "explainer": explainer, "seed": seed,
            "status": status, "exception_type": exception_type, "message": message,
            "traceback": traceback,
            "measurements": json.dumps(measurements) if measurements else None,
            "requirement": json.dumps(requirement) if requirement else None,
            "reason": reason,
        })

    def results_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.results, columns=RESULTS_COLUMNS)

    def failures_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.failures, columns=FAILURES_COLUMNS)

    def write(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        results_df, failures_df = self.results_df(), self.failures_df()
        results_df.to_parquet(run_dir / "results.parquet", index=False)
        results_df.to_csv(run_dir / "results.csv", index=False)
        failures_df.to_parquet(run_dir / "failures.parquet", index=False)
        failures_df.to_csv(run_dir / "failures.csv", index=False)
