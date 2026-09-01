"""Explanation-quality metrics, ported verbatim from the predecessor pipeline's evaluation
code (AUDIT.md 1.3 found these formulas correct).

Expects an explanations DataFrame with a real Python ``set`` per row in the
``explanation_set`` column (built directly by explainers/*.py in memory) plus
``tail_count``/``exp_count`` columns from ``annotate_tail_counts`` — this replaces the
predecessor pipeline's original ``read_explanation_csv`` CSV round-trip (which parsed set
literals with ``ast.literal_eval``); the metric formulas themselves are unchanged, only the
I/O path that used to go through a headerless CSV is gone since rexbench builds these
DataFrames in memory.
"""
from __future__ import annotations

import math
import random
from itertools import combinations
from typing import List, Tuple

import pandas as pd

from rexbench.metrics.validate import validated_range

EXP_COL = "explanation_set"


def annotate_tail_counts(df: pd.DataFrame, tail_items, exp_col: str = EXP_COL) -> pd.DataFrame:
    out = df.copy()
    out["tail_count"] = out[exp_col].apply(lambda x: sum(1 for i in x if i in tail_items))
    out["exp_count"] = out[exp_col].apply(len)
    return out


@validated_range(0.0, 1.0)
def model_fidelity(df: pd.DataFrame, exp_col: str = EXP_COL) -> float:
    explained = df[exp_col].apply(lambda x: 1 if x else 0).sum()
    return explained / df.shape[0]


@validated_range(0.0, 1.0, allow_nan=True)
def tail_coverage(df: pd.DataFrame, tail_count_col: str = "tail_count", exp_count_col: str = "exp_count") -> float:
    tail_sum = df[tail_count_col].sum()
    exp_sum = df[exp_count_col].sum()
    if exp_sum == 0:
        return float("nan")  # AUDIT.md 1.2/1.3: original silently produced NaN via 0/0 here too
    return tail_sum / exp_sum


def _explanation_item_appearance_probabilities(df: pd.DataFrame, exp_col: str = EXP_COL) -> List[Tuple]:
    counts: dict = {}
    n_rows = df.shape[0]
    for exp_set in df[exp_col]:
        for item_id in exp_set:
            counts[item_id] = counts.get(item_id, 0) + 1
    return [(i, cnt / n_rows) for i, cnt in counts.items()]


def explanation_diversity(df: pd.DataFrame, exp_col: str = EXP_COL) -> float:
    probs = _explanation_item_appearance_probabilities(df, exp_col)
    return -sum(p * math.log(p) for _, p in probs if p > 0)


def explanation_arp(df: pd.DataFrame, item_popularity: pd.Series, exp_col: str = EXP_COL) -> float:
    pops = [item_popularity.get(item_id, 0) for exp_set in df[exp_col] for item_id in exp_set]
    return sum(pops) / len(pops) if pops else 0.0


@validated_range(0.0, 1.0)
def explanation_coverage(df: pd.DataFrame, n_items: int, exp_col: str = EXP_COL) -> float:
    covered: set = set()
    for exp_set in df[exp_col]:
        covered.update(exp_set)
    return len(covered) / n_items if n_items else 0.0


def _mean_pairwise_jaccard_similarity(sets: List[set], n_samples: int, rng: random.Random):
    n = len(sets)
    if n < 2:
        return None
    total_pairs = n * (n - 1) // 2
    pairs = combinations(range(n), 2) if total_pairs <= n_samples else (rng.sample(range(n), 2) for _ in range(n_samples))
    sims = []
    for i, j in pairs:
        union = sets[i] | sets[j]
        if union:
            sims.append(len(sets[i] & sets[j]) / len(union))
    return sum(sims) / len(sims) if sims else None


@validated_range(0.0, 1.0, allow_nan=True)
def explanation_similarity(
    df: pd.DataFrame, group_col: str | None = None, exp_col: str = EXP_COL,
    n_samples: int = 2000, seed: int = 42,
) -> float:
    rng = random.Random(seed)
    if group_col is None:
        sets = [s for s in df[exp_col] if s]
        result = _mean_pairwise_jaccard_similarity(sets, n_samples, rng)
        return result if result is not None else float("nan")

    group_means = []
    for _, group in df.groupby(group_col):
        sets = [s for s in group[exp_col] if s]
        mean_sim = _mean_pairwise_jaccard_similarity(sets, n_samples, rng)
        if mean_sim is not None:
            group_means.append(mean_sim)
    return sum(group_means) / len(group_means) if group_means else float("nan")


@validated_range(0.0, 1.0, allow_nan=True)
def explanation_personalization(
    df: pd.DataFrame, exp_col: str = EXP_COL, n_samples: int = 2000, seed: int = 42,
) -> float:
    similarity = explanation_similarity(df, None, exp_col, n_samples, seed)
    return similarity if math.isnan(similarity) else 1 - similarity
