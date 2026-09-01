"""Friedman, Quade, Nemenyi post-hoc, Kendall's W, and the critical-difference value.

friedman_test mirrors the predecessor pipeline's friedman_test (AUDIT.md found it
correct: pivot by model x dataset, scipy.stats.friedmanchisquare, alpha=0.05). quade_test,
critical_difference, and nemenyi_pairwise_pvalues are ported verbatim from
notebooks/results_analysis.ipynb (AUDIT.md 1.4/OUTPUT: these existed only as ad hoc
notebook code, promoted here to tested library functions with the same formulas).
kendall_w is new: the standard textbook relationship to the already-computed Friedman
statistic (W = chi2 / (n_blocks * (n_groups - 1))), not a separate implementation.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def friedman_test(pivot: pd.DataFrame, alpha: float = 0.05) -> dict:
    groups = [pivot[col].dropna().values for col in pivot.columns]
    stat, p_value = stats.friedmanchisquare(*groups)
    return {"statistic": float(stat), "p_value": float(p_value), "significant": bool(p_value < alpha)}


def quade_test(pivot: pd.DataFrame, alpha: float = 0.05) -> dict:
    n_blocks, n_groups = pivot.shape
    within_block_ranks = pivot.rank(axis=1)
    block_ranges = pivot.max(axis=1) - pivot.min(axis=1)
    range_ranks = block_ranges.rank()

    s = within_block_ranks.sub((n_groups + 1) / 2).mul(range_ranks, axis=0)
    group_sums = s.sum(axis=0)
    a2 = (s ** 2).values.sum()
    b = (group_sums ** 2).sum() / n_blocks

    stat = (n_blocks - 1) * b / (a2 - b)
    df1, df2 = n_groups - 1, (n_blocks - 1) * (n_groups - 1)
    p_value = 1 - stats.f.cdf(stat, df1, df2)
    return {"statistic": float(stat), "df1": df1, "df2": df2, "p_value": float(p_value), "significant": bool(p_value < alpha)}


def critical_difference(n_models: int, n_blocks: int, alpha: float = 0.05) -> float:
    q_alpha = stats.studentized_range.ppf(1 - alpha, n_models, np.inf) / np.sqrt(2)
    return float(q_alpha * np.sqrt(n_models * (n_models + 1) / (6 * n_blocks)))


def nemenyi_pairwise_pvalues(avg_ranks: pd.Series, n_blocks: int) -> pd.DataFrame:
    n_models = len(avg_ranks)
    se = np.sqrt(n_models * (n_models + 1) / (6 * n_blocks))
    models = avg_ranks.index.tolist()

    pvalues = pd.DataFrame(1.0, index=models, columns=models)
    for mi, mj in combinations(models, 2):
        z = abs(avg_ranks[mi] - avg_ranks[mj]) / se
        p_value = 1 - stats.studentized_range.cdf(z * np.sqrt(2), n_models, np.inf)
        pvalues.loc[mi, mj] = pvalues.loc[mj, mi] = p_value
    return pvalues


def kendall_w(pivot: pd.DataFrame) -> float:
    n_blocks, n_groups = pivot.shape
    if n_groups < 2:
        return float("nan")
    groups = [pivot[col].dropna().values for col in pivot.columns]
    stat, _ = stats.friedmanchisquare(*groups)
    return float(stat / (n_blocks * (n_groups - 1)))


def average_ranks(pivot: pd.DataFrame, higher_is_better: bool = True) -> pd.Series:
    """Per-model average rank across datasets — the input nemenyi_pairwise_pvalues and
    critical_difference expect, and what a critical-difference diagram plots. Set
    higher_is_better=False for metrics like gini/variance/entropy where lower is fairer."""
    return pivot.rank(axis=1, ascending=not higher_is_better).mean(axis=0)
