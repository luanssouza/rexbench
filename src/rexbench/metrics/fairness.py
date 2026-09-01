"""Recommendation-side fairness metrics.

variance() and entropy() are ported verbatim from rexfair.rexfair.evaluation.metrics —
AUDIT.md 1.3/4.9 found these formulas correct (variance is O(n^2) pairwise, not textbook
variance, by original design; entropy is not normalized by max entropy, also by original
design — both preserved exactly).

gini() is FIXED per AUDIT.md section 5.4. The original (rexfair/rexfair/evaluation/
metrics.py:784-805) collapsed three distinct degenerate cases into a single out-of-range
-1.0 sentinel:
    if denominator == 0 or numerator == 0:
        return -1.0
That sentinel is mathematically outside gini's documented [0,1] range and was found reaching
a published LaTeX table unmasked (AUDIT.md 5.2). This version distinguishes the cases:
  - fewer than 2 users                    -> NaN  (truly undefined, insufficient data)
  - all scores identical AND sum > 0      -> 0.0  (perfect equality IS zero inequality)
  - sum(scores) == 0 (all-zero, n>=2)     -> NaN  (genuine 0/0, not zero and not computable)
  - otherwise                             -> numerator / denominator, as before
"""
from __future__ import annotations

import math
from typing import Dict

import pandas as pd

from rexbench.metrics.validate import validated_range

U_COL, I_COL, R_COL = "user_id", "item_id", "score"


def variance(scores_per_user: Dict) -> float:
    keys = list(scores_per_user.keys())
    n = len(keys)
    total = sum(
        (scores_per_user[kx] - scores_per_user[ky]) ** 2
        for kx in keys
        for ky in keys
        if kx != ky
    )
    return total / (n ** 2)


@validated_range(0.0, 1.0, allow_nan=True)
def gini(scores_per_user: Dict) -> float:
    keys = list(scores_per_user.keys())
    if len(keys) < 2:
        return float("nan")

    values = [scores_per_user[k] for k in keys]
    total = sum(values)
    if all(v == values[0] for v in values):
        return 0.0 if total > 0 else float("nan")

    numerator = sum(abs(scores_per_user[kx] - scores_per_user[ky]) for kx in keys for ky in keys if kx != ky)
    denominator = 2 * len(keys) * total
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def _item_appearance_probabilities(predictions_df: pd.DataFrame):
    items = {i: 0 for i in sorted(predictions_df[I_COL].unique())}
    rec_lists = predictions_df.groupby(U_COL)
    n_lists = len(rec_lists)
    for _, group in rec_lists:
        for item_id in group[I_COL]:
            items[item_id] += 1
    return [(i, cnt / n_lists) for i, cnt in items.items()]


def entropy(predictions_df: pd.DataFrame, top_k: int) -> float:
    topk_df = (
        predictions_df
        .sort_values([U_COL, R_COL], ascending=[True, False])
        .drop_duplicates(subset=[U_COL, I_COL], keep="first")
        .groupby(U_COL)
        .head(top_k)
    )
    probs = _item_appearance_probabilities(topk_df)
    return -sum(p * math.log(p) for _, p in probs)


@validated_range(0.0, 1.0)
def tail_coverage_rec(predictions_df: pd.DataFrame, tail_items) -> float:
    tail_count = predictions_df[I_COL].isin(tail_items).sum()
    return tail_count / predictions_df.shape[0]
