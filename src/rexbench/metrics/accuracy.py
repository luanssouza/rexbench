"""Recommendation accuracy and popularity-bias metrics.

Ported verbatim from the predecessor pipeline's evaluation code (AUDIT.md 1.3 found these
formulas correct — only gini() needed a fix, handled separately in fairness.py). Column parameters
are pinned to rexbench's canonical DatasetBundle column names instead of being passed in by
callers, since every predictions_df/rel_dict in rexbench always uses the same schema.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

from rexbench.metrics.validate import validated_range

U_COL, I_COL, R_COL = "user_id", "item_id", "score"


def dcg_at_k(r: Sequence, k: int, method: int = 1) -> float:
    r = np.asarray(r)[:k]
    if not r.size:
        return 0.0
    if method == 0:
        return float(r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1))))
    if method == 1:
        return float(np.sum(r / np.log2(np.arange(2, r.size + 2))))
    raise ValueError("method must be 0 or 1.")


@validated_range(0.0, 1.0)
def ndcg_at_k(r: Sequence, k: int, method: int = 0) -> float:
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k, method) / dcg_max


def _topk_per_user(predictions_df: pd.DataFrame, uid, top_k: int, groups) -> list:
    if uid not in groups.groups:
        return []
    return (
        groups.get_group(uid)
        .sort_values(R_COL, ascending=False)
        .drop_duplicates(subset=[U_COL, I_COL], keep="first")
        .head(top_k)[I_COL]
        .tolist()
    )


def dataset_ndcg_k(predictions_df: pd.DataFrame, user_test_dict: Dict, top_k: int = 10) -> Dict:
    """Per-user NDCG@k. Users whose top-k list has fewer than top_k entries are skipped
    (matches the predecessor pipeline's original behavior — flagged in AUDIT.md as silently shrinking the
    evaluated population on sparse data/models, preserved here rather than changed)."""
    groups = predictions_df.groupby(U_COL)
    out: Dict = {}
    for uid, rel_set in user_test_dict.items():
        topk = _topk_per_user(predictions_df, uid, top_k, groups)
        if len(topk) < top_k:
            continue
        hit_list = [1 if int(pid) in rel_set else 0 for pid in topk]
        out[uid] = ndcg_at_k(hit_list, top_k)
    return out


def ap_at_k(topk_items: Sequence, rel_set, top_k: int) -> float:
    m = len(rel_set)
    if m == 0:
        return 0.0
    norm = min(m, top_k)
    hits = 0
    score = 0.0
    for rank, item in enumerate(topk_items[:top_k], start=1):
        if int(item) in rel_set:
            hits += 1
            score += hits / rank
    return score / norm


@validated_range(0.0, 1.0)
def map_at_k(predictions_df: pd.DataFrame, user_test_dict: Dict, top_k: int = 10) -> float:
    groups = predictions_df.groupby(U_COL)
    ap_scores = []
    for uid, rel_items in user_test_dict.items():
        topk = _topk_per_user(predictions_df, uid, top_k, groups)
        if len(topk) < top_k:
            continue
        ap_scores.append(ap_at_k(topk, set(rel_items), top_k))
    return sum(ap_scores) / len(ap_scores) if ap_scores else 0.0


def average_recommendation_popularity(
    predictions_df: pd.DataFrame, item_popularity: pd.Series, top_k: int = 10
) -> float:
    """Raw ARP (Elliot-framework definition) — not range-bounded, by construction (AUDIT.md
    1.3 flagged the raw form as reported in interaction-count units, not [0,1])."""
    topk_df = (
        predictions_df
        .sort_values([U_COL, R_COL], ascending=[True, False])
        .drop_duplicates(subset=[U_COL, I_COL], keep="first")
        .groupby(U_COL)
        .head(top_k)
    )
    user_arps = []
    for _, group in topk_df.groupby(U_COL):
        pops = [item_popularity.get(i, 0) for i in group[I_COL]]
        if pops:
            user_arps.append(sum(pops) / len(pops))
    return sum(user_arps) / len(user_arps) if user_arps else 0.0


@validated_range(0.0, 1.0)
def average_recommendation_popularity_normalized(
    predictions_df: pd.DataFrame, item_popularity: pd.Series, top_k: int = 10
) -> float:
    """New metric (AUDIT.md 1.3): min-max normalizes raw ARP against the dataset's actual
    popularity range [min(item_popularity), max(item_popularity)], so it lands in [0,1] and
    is comparable across datasets with very different interaction-count scales."""
    raw = average_recommendation_popularity(predictions_df, item_popularity, top_k)
    lo, hi = float(item_popularity.min()), float(item_popularity.max())
    if hi == lo:
        return 0.0
    return (raw - lo) / (hi - lo)
