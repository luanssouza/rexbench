"""Single named-metric dispatch, shared by core/runner.py's main results computation and
core/hpo.py's trial evaluation — one place that knows how to compute "metric X at k" for a
recommendations DataFrame, rather than two independently-maintained copies.
"""
from __future__ import annotations

import pandas as pd

from rexbench.metrics import accuracy, fairness


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


# Names accepted here match what shows up in results.parquet's `metric` column and in
# HPOConfig.metric — "tail_coverage" is accepted as an alias for "tail_coverage_rec" since
# that's the name users declare in metrics.fairness config, even though the stored result
# row (core/runner.py) uses "tail_coverage_rec" to disambiguate from the explanation-quality
# metric of the same short name (AUDIT.md flagged this exact ambiguity).
_NDCG_FAMILY = {"ndcg", "gini", "variance"}
_ITEM_SIDE = {"item_coverage", "item_exposure_gini", "tail_item_coverage", "tail_exposure_share"}


def compute_metric_value(metric_name: str, recs: pd.DataFrame, ground_truth: dict, dataset, k: int) -> float:
    if metric_name in _NDCG_FAMILY:
        ndcg_dict = accuracy.dataset_ndcg_k(recs, ground_truth, top_k=k)
        if metric_name == "ndcg":
            return mean(ndcg_dict.values())
        if metric_name == "gini":
            return fairness.gini(ndcg_dict)
        return fairness.variance(ndcg_dict)
    if metric_name == "map":
        return accuracy.map_at_k(recs, ground_truth, top_k=k)
    if metric_name == "entropy":
        return fairness.entropy(recs, top_k=k)
    if metric_name == "arp":
        return accuracy.average_recommendation_popularity(recs, dataset.popularity, top_k=k)
    if metric_name == "arp_normalized":
        return accuracy.average_recommendation_popularity_normalized(recs, dataset.popularity, top_k=k)
    if metric_name in ("tail_coverage", "tail_coverage_rec"):
        return fairness.tail_coverage_rec(recs, dataset.tail_items, k)
    # Item-side (provider-side) metrics — see fairness.py's section header. These need
    # dataset.num_items / dataset.tail_items and are genuinely k-dependent, unlike
    # tail_coverage_rec above.
    if metric_name == "item_coverage":
        return fairness.item_coverage(recs, dataset.num_items, k)
    if metric_name == "item_exposure_gini":
        return fairness.item_exposure_gini(recs, dataset.num_items, k)
    if metric_name == "tail_item_coverage":
        return fairness.tail_item_coverage(recs, dataset.tail_items, k)
    if metric_name == "tail_exposure_share":
        return fairness.tail_exposure_share(recs, dataset.tail_items, k)
    raise KeyError(
        f"unknown recommendation metric {metric_name!r}; known: "
        f"{sorted(_NDCG_FAMILY | _ITEM_SIDE | {'map', 'entropy', 'arp', 'arp_normalized', 'tail_coverage_rec'})}"
    )
