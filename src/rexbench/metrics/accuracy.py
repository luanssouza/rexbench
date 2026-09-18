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
def ndcg_at_k(r: Sequence, k: int, method: int = 0, n_relevant: int | None = None) -> float:
    """NDCG@k. Jarvelin & Kekalainen (2002), "Cumulated Gain-based Evaluation of IR
    Techniques", ACM TOIS 20(4):422-446. https://doi.org/10.1145/582415.582418

    FIXED: the ideal ranking is now `min(n_relevant, k)` relevant items, per the definition
    above. It used to be `sorted(r, reverse=True)` — the hits actually *found*, pushed to the
    top — which normalises away recall entirely: a user with 20 relevant items who got 2 of
    them, both at ranks 1-2, scored **1.0** instead of 0.359 (measured). Any list whose hits
    happen to sit at the top scored a perfect 1.0 no matter how many relevant items were
    missed, so the metric could not distinguish a model that finds everything from one that
    finds almost nothing. gini/variance are computed over these per-user values and inherited
    the distortion. `map_at_k` below always normalised by `min(len(rel_set), k)`, so NDCG and
    MAP were not even consistent with each other.

    n_relevant=None keeps the old hits-only normalisation, for callers that genuinely have no
    ground-truth size (no caller in rexbench does — dataset_ndcg_k always passes it).
    """
    if n_relevant is None:
        ideal = sorted(r, reverse=True)
    else:
        n_ideal = min(int(n_relevant), k)
        ideal = [1] * n_ideal + [0] * max(0, k - n_ideal)
    dcg_max = dcg_at_k(ideal, k, method)
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
    """Per-user NDCG@k over EVERY user in the ground truth.

    FIXED: users whose list had fewer than top_k entries used to be skipped. That made the
    evaluated population model-dependent, which breaks the premise that every algorithm is
    compared on the same scenario: models that always return k items (EBPR, RecBole) were
    scored on all users, while a model that returns short lists for some users (PGPR, whose
    candidates come from predicted paths) had exactly those users silently dropped from its
    average — and dropping users is not neutral, since the ones a path-based model fails to
    reach are precisely its hard cases. Demonstrated: a model that found the relevant item
    for all 5 users but returned short lists for 3 of them was averaged over 2 users.

    Now a short or empty list simply scores lower, which is the honest outcome. AUDIT.md
    flagged the old behaviour; it was preserved until this pass and is now corrected.
    """
    groups = predictions_df.groupby(U_COL)
    out: Dict = {}
    for uid, rel_set in user_test_dict.items():
        topk = _topk_per_user(predictions_df, uid, top_k, groups)
        hit_list = [1 if int(pid) in rel_set else 0 for pid in topk]
        out[uid] = ndcg_at_k(hit_list, top_k, n_relevant=len(rel_set))
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
    """Mean Average Precision@k. Manning, Raghavan & Schutze (2008), "Introduction to
    Information Retrieval", Cambridge University Press, section 8.4.
    https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html
    """
    groups = predictions_df.groupby(U_COL)
    ap_scores = []
    for uid, rel_items in user_test_dict.items():
        topk = _topk_per_user(predictions_df, uid, top_k, groups)
        # No skip: same evaluated population as dataset_ndcg_k above, for the same reason.
        ap_scores.append(ap_at_k(topk, set(rel_items), top_k))
    return sum(ap_scores) / len(ap_scores) if ap_scores else 0.0


def average_recommendation_popularity(
    predictions_df: pd.DataFrame, item_popularity: pd.Series, top_k: int = 10
) -> float:
    """Average Recommendation Popularity: mean training-set popularity of the items in each
    user's top-k list, averaged over users. Not range-bounded, by construction (AUDIT.md 1.3
    flagged the raw form as reported in interaction-count units, not [0,1]).

    Abdollahpouri & Burke (2019), "Reducing Popularity Bias in Recommendation Over Time",
    arXiv:1906.11711. https://arxiv.org/abs/1906.11711
    (ARP's attribution to this paper follows Klimashevskaia, Jannach, Elahi & Trattner
    (2024), "A Survey on Popularity Bias in Recommender Systems", UMUAI, Table 4.
    https://doi.org/10.1007/s11257-024-09406-0)
    """
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
    """New metric (AUDIT.md 1.3), rexbench-specific — no external source: min-max normalizes raw ARP against the dataset's actual
    popularity range [min(item_popularity), max(item_popularity)], so it lands in [0,1] and
    is comparable across datasets with very different interaction-count scales."""
    raw = average_recommendation_popularity(predictions_df, item_popularity, top_k)
    lo, hi = float(item_popularity.min()), float(item_popularity.max())
    if hi == lo:
        return 0.0
    return (raw - lo) / (hi - lo)
