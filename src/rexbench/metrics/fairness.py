"""Recommendation-side fairness metrics.

variance() and entropy() are ported verbatim from the predecessor pipeline's evaluation
code — AUDIT.md 1.3/4.9 found these formulas correct (variance is O(n^2) pairwise, not
textbook variance, by original design; entropy is not normalized by max entropy, also by
original design — both preserved exactly).

gini() is FIXED per AUDIT.md section 5.4. The original (in the predecessor pipeline's
evaluation code) collapsed three distinct degenerate cases into a single out-of-range
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

import numpy as np
import pandas as pd

from rexbench.metrics.validate import validated_range

U_COL, I_COL, R_COL = "user_id", "item_id", "score"


def _topk_frame(predictions_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Per-user top-k truncation, score-sorted. Shared by every k-dependent metric in this
    module so they all score exactly the same lists."""
    return (
        predictions_df
        .sort_values([U_COL, R_COL], ascending=[True, False])
        .drop_duplicates(subset=[U_COL, I_COL], keep="first")
        .groupby(U_COL)
        .head(top_k)
    )



def variance(scores_per_user: Dict) -> float:
    """USER-side: mean squared pairwise difference of per-user NDCG. No external source —
    this is the predecessor pipeline's own O(n^2) pairwise form, not textbook variance
    (AUDIT.md 1.3/4.9 confirmed the formula and it is preserved exactly)."""
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
    """USER-side: standard Gini coefficient applied to the per-user NDCG distribution — how
    unequally recommendation QUALITY is spread across users. Not to be confused with
    item_exposure_gini() below, which applies the same coefficient to per-item exposure.
    The predecessor pipeline applied the textbook Gini formula here without citing a
    recommender-systems source; see item_exposure_gini() for the recsys literature on
    Gini as an exposure-concentration measure. Degenerate-case handling is rexbench's own
    fix, per the module docstring."""
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
    """Shannon entropy of the item-appearance distribution across all top-k lists — higher
    means exposure is spread over more items. NOT normalized by max entropy (log |I|), by
    the predecessor pipeline's original design, so values are not comparable across datasets
    with different catalog sizes; preserved exactly (AUDIT.md 1.3/4.9).

    Shannon (1948), "A Mathematical Theory of Communication", Bell System Technical Journal
    27(3):379-423. https://doi.org/10.1002/j.1538-7305.1948.tb01338.x
    Used as an entropy-diversity measure for recommendations by Adomavicius & Kwon (2012),
    IEEE TKDE 24(5):896-911. https://doi.org/10.1109/TKDE.2011.15
    """
    topk_df = _topk_frame(predictions_df, top_k)
    probs = _item_appearance_probabilities(topk_df)
    return -sum(p * math.log(p) for _, p in probs)


@validated_range(0.0, 1.0)
def tail_coverage_rec(predictions_df: pd.DataFrame, tail_items, top_k: int) -> float:
    """APLT (Average Percentage of Long Tail items): for each user, the share of their top-k
    list that is long-tail, averaged over users.

        APLT = 1/|Ut| * sum_u |{i : i in (Lu intersect Gamma)}| / |Lu|

    Abdollahpouri, Burke & Mobasher (2019), "Managing Popularity Bias in Recommender Systems
    with Personalized Re-ranking", FLAIRS-32, pp. 413-418. https://arxiv.org/abs/1901.07555

    FIXED (was a real bug, not a stylistic deviation). The predecessor pipeline's version
    took `tail_count / len(predictions_df)` over the WHOLE prediction frame, which was wrong
    twice over:
      - it never truncated to top_k, so tail_coverage@5 and tail_coverage@10 came out
        identical in every run — the k column in results.parquet was meaningless for this
        metric;
      - it pooled every user's rows into one ratio instead of averaging per-user
        percentages, so users with longer lists were silently weighted more heavily than
        users with shorter ones. (These coincide only when every user has an equal-length
        list, which is exactly what the missing truncation failed to guarantee.)
    Both are corrected here to match the formula above. Numbers for this metric are NOT
    comparable to any run produced before this fix.
    """
    topk_df = _topk_frame(predictions_df, top_k)
    if topk_df.empty:
        return 0.0
    is_tail = topk_df[I_COL].isin(tail_items)
    per_user = is_tail.groupby(topk_df[U_COL]).mean()
    return float(per_user.mean())


# ---------------------------------------------------------------------------
# Item-side (provider-side) fairness metrics.
#
# Everything above this line measures either inequality ACROSS USERS (variance,
# gini, both computed over per-user NDCG) or a single aggregate property of the
# recommendation pool (entropy, tail_coverage_rec). What was missing was the
# item side: how exposure is distributed over the CATALOG. All four below are
# computable from `predictions_df` plus `num_items`/`tail_items`, which
# DatasetBundle already carries — no new data plumbing.
# ---------------------------------------------------------------------------


def _exposure_counts(predictions_df: pd.DataFrame, num_items: int, top_k: int) -> pd.Series:
    """Per-item recommendation count over the whole top-k pool, reindexed over the FULL
    catalog so never-recommended items appear as explicit zeros. Those zeros are the whole
    point for item_coverage/item_exposure_gini — dropping them would measure inequality
    only among items lucky enough to be recommended at least once."""
    topk_df = _topk_frame(predictions_df, top_k)
    counts = topk_df.groupby(I_COL).size()
    return counts.reindex(range(num_items), fill_value=0)


@validated_range(0.0, 1.0)
def item_coverage(predictions_df: pd.DataFrame, num_items: int, top_k: int) -> float:
    """Catalog coverage / aggregate diversity: the fraction of the item catalog that appears
    in at least one user's top-k list.

    Adomavicius & Kwon (2012), "Improving Aggregate Recommendation Diversity Using
    Ranking-Based Techniques", IEEE TKDE 24(5):896-911.
    https://doi.org/10.1109/TKDE.2011.15
    """
    if num_items <= 0:
        return 0.0
    distinct = _topk_frame(predictions_df, top_k)[I_COL].nunique()
    return distinct / num_items


@validated_range(0.0, 1.0, allow_nan=True)
def item_exposure_gini(predictions_df: pd.DataFrame, num_items: int, top_k: int) -> float:
    """Gini coefficient of the per-item exposure distribution over the full catalog.
    0.0 = every item recommended equally often, ->1.0 = exposure concentrated on a few
    items. This is the ITEM-side counterpart to gini() above, which measures inequality of
    NDCG across USERS — they answer different questions and are usually reported together.

    Uses the sorted (Lorenz-curve) formulation, O(n log n), rather than the O(n^2) pairwise
    form gini() uses: catalogs run to hundreds of thousands of items (lastfm1k), where the
    pairwise form is not tractable. The two formulations are mathematically equivalent.

    Fleder & Hosanagar (2009), "Blockbuster Culture's Next Rise or Fall: The Impact of
    Recommender Systems on Sales Diversity", Management Science 55(5):697-712.
    https://doi.org/10.1287/mnsc.1080.0974
    Also used as an item-exposure measure by Adomavicius & Kwon (2012), above.
    """
    counts = _exposure_counts(predictions_df, num_items, top_k)
    n = len(counts)
    if n < 2:
        return float("nan")
    values = counts.to_numpy(dtype=float)
    total = values.sum()
    if total == 0:
        return float("nan")  # nothing was recommended at all — undefined, not "equal"
    values.sort()
    index = np.arange(1, n + 1, dtype=float)
    return float((2.0 * (index * values).sum()) / (n * total) - (n + 1.0) / n)


@validated_range(0.0, 1.0)
def tail_item_coverage(predictions_df: pd.DataFrame, tail_items, top_k: int) -> float:
    """Fraction of DISTINCT long-tail items that get recommended to at least one user.

    This is the complement tail_coverage_rec() cannot provide: that one measures the share
    of recommendation SLOTS filled by tail items, which stays high even when every user is
    shown the same handful of tail items. This one catches exactly that case.

    Abdollahpouri, Burke & Mobasher (2019), "Managing Popularity Bias in Recommender
    Systems with Personalized Re-ranking", FLAIRS-32, pp. 413-418.
    https://arxiv.org/abs/1901.07555

    DEVIATION, disclosed: that paper names this concern when introducing ACLT ("One problem
    with APLT is that it could be high even if all users get the same set of long tail
    items"), but its printed ACLT formula is an average COUNT of long-tail items per list
    (1/|Ut| * sum_u sum_{i in Lu} 1(i in Gamma)), which is just APLT rescaled by list length
    and so does not actually measure distinct coverage. This implements the stated intent —
    distinct tail items covered, normalized to [0,1] — not the printed formula. Cite it as
    tail-restricted catalog coverage (Adomavicius & Kwon, above) rather than as ACLT
    verbatim.
    """
    tail = pd.Index(tail_items)
    if len(tail) == 0:
        return 0.0
    topk_df = _topk_frame(predictions_df, top_k)
    covered = topk_df.loc[topk_df[I_COL].isin(tail), I_COL].nunique()
    return covered / len(tail)


@validated_range(0.0, 1.0)
def tail_exposure_share(predictions_df: pd.DataFrame, tail_items, top_k: int) -> float:
    """Share of total position-discounted EXPOSURE that goes to long-tail items.

    Every other item-side metric here treats rank 1 and rank 10 as equally valuable. They
    are not: attention falls off sharply with rank, so a recommender can post a respectable
    tail share while parking every tail item at the bottom of the list. Exposure weights
    each slot by 1/log2(1 + rank) (the standard position-bias/DCG discount) before taking
    the tail's share, so burying tail items is penalized.

    Singh & Joachims (2018), "Fairness of Exposure in Rankings", KDD '18.
    https://arxiv.org/abs/1802.07281
    Biega, Gummadi & Weikum (2018), "Equity of Attention: Amortizing Individual Fairness in
    Rankings", SIGIR '18, pp. 405-414. https://doi.org/10.1145/3209978.3210063
    """
    tail = pd.Index(tail_items)
    topk_df = _topk_frame(predictions_df, top_k)
    if topk_df.empty:
        return 0.0
    ranks = topk_df.groupby(U_COL).cumcount() + 1  # already score-sorted by _topk_frame
    weights = 1.0 / np.log2(1.0 + ranks.to_numpy(dtype=float))
    total = weights.sum()
    if total == 0:
        return 0.0
    if len(tail) == 0:
        return 0.0
    is_tail = topk_df[I_COL].isin(tail).to_numpy()
    return float(weights[is_tail].sum() / total)
