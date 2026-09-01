"""Mean +/- std across seeds, from the long-format results table."""
from __future__ import annotations

import pandas as pd


def aggregate_across_seeds(results_df: pd.DataFrame) -> pd.DataFrame:
    ok = results_df[results_df["status"] == "ok"]
    group_cols = ["dataset", "model", "explainer", "metric", "k"]
    agg = ok.groupby(group_cols, dropna=False)["value"].agg(["mean", "std", "count"]).reset_index()
    agg = agg.rename(columns={"mean": "value_mean", "std": "value_std", "count": "n_seeds"})
    return agg


def wide_pivot(aggregated_df: pd.DataFrame, metric: str, k=None) -> pd.DataFrame:
    """dataset x model table of value_mean for one metric (optionally filtered to one k) —
    the shape stats.significance's Friedman/Quade/Nemenyi functions expect."""
    subset = aggregated_df[aggregated_df["metric"] == metric]
    if k is not None:
        subset = subset[subset["k"] == k]
    return subset.pivot_table(values="value_mean", index="dataset", columns="model")
