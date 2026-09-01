"""The seed x dataset x model x explainer loop and failure classification.

AUDIT.md's FAILURE HANDLING requirement, made concrete: every attempted combination
produces a row in `results` (status=ok) or `failures` (status in {crashed, degenerate,
resource, skipped, precondition-unmet}) — nothing is silently dropped, and a metric that
returns an out-of-range value (rexbench.metrics.validate.MetricRangeError) is caught and
recorded as `degenerate` rather than allowed to reach results.parquet — this is the
mechanism that would have caught AUDIT.md section 5's Gini -1.0 sentinel as a classified
failure instead of a silent out-of-range number.
"""
from __future__ import annotations

import traceback
from typing import Any

import pandas as pd

from rexbench.config.schema import ExperimentConfig
from rexbench.core.dataset import DatasetBundle, build_dataset_bundle
from rexbench.core.determinism import resolve_device, seed_all
from rexbench.core.registry import build_explainer, build_model
from rexbench.core.results_io import ResultsCollector
from rexbench.metrics import accuracy, explanation_quality, fairness
from rexbench.metrics.validate import MetricRangeError

_RESOURCE_SIGNATURES = ("out of memory", "cuda oom", "cannot allocate memory")


def _classify_exception(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(sig in text for sig in _RESOURCE_SIGNATURES) or type(exc).__name__ in (
        "OutOfMemoryError", "MemoryError",
    ):
        return "resource"
    return "crashed"


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _compute_recommendation_metrics(
    collector: ResultsCollector, dataset: DatasetBundle, dataset_name: str, model_name: str,
    seed: int, recs: pd.DataFrame, requested_accuracy: list[str], requested_fairness: list[str],
) -> None:
    user_test_dict = dataset.user_test_dict()
    for k in dataset.topk:
        try:
            ndcg_dict = accuracy.dataset_ndcg_k(recs, user_test_dict, top_k=k)
            if "ndcg" in requested_accuracy:
                collector.add_result(dataset_name, model_name, None, "ndcg", seed, k, _mean(ndcg_dict.values()))
            if "map" in requested_accuracy:
                map_value = accuracy.map_at_k(recs, user_test_dict, top_k=k)
                collector.add_result(dataset_name, model_name, None, "map", seed, k, map_value)
            if "gini" in requested_fairness:
                collector.add_result(dataset_name, model_name, None, "gini", seed, k, fairness.gini(ndcg_dict))
            if "variance" in requested_fairness:
                collector.add_result(dataset_name, model_name, None, "variance", seed, k, fairness.variance(ndcg_dict))
            if "entropy" in requested_fairness:
                collector.add_result(dataset_name, model_name, None, "entropy", seed, k, fairness.entropy(recs, top_k=k))
            if "arp" in requested_fairness or "arp_normalized" in requested_fairness:
                if "arp" in requested_fairness:
                    collector.add_result(
                        dataset_name, model_name, None, "arp", seed, k,
                        accuracy.average_recommendation_popularity(recs, dataset.popularity, top_k=k),
                    )
                if "arp_normalized" in requested_fairness:
                    collector.add_result(
                        dataset_name, model_name, None, "arp_normalized", seed, k,
                        accuracy.average_recommendation_popularity_normalized(recs, dataset.popularity, top_k=k),
                    )
            if "tail_coverage" in requested_fairness:  # tail_coverage_rec, recommendation-list tail exposure
                collector.add_result(
                    dataset_name, model_name, None, "tail_coverage_rec", seed, k,
                    fairness.tail_coverage_rec(recs, dataset.tail_items),
                )
        except MetricRangeError as exc:
            collector.add_failure(
                dataset_name, model_name, None, seed, "degenerate",
                exception_type=type(exc).__name__, message=str(exc),
            )
        except Exception as exc:  # noqa: BLE001 — deliberately broad, see module docstring
            collector.add_failure(
                dataset_name, model_name, None, seed, _classify_exception(exc),
                exception_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc(),
            )


def _compute_explanation_metrics(
    collector: ResultsCollector, dataset: DatasetBundle, dataset_name: str, model_name: str,
    explainer_name: str, seed: int, explained: pd.DataFrame, requested: list[str],
) -> None:
    try:
        annotated = explanation_quality.annotate_tail_counts(explained, dataset.tail_items)
        if "fidelity" in requested:
            collector.add_result(dataset_name, model_name, explainer_name, "fidelity", seed, None, explanation_quality.model_fidelity(annotated))
        if "tail_coverage" in requested:
            collector.add_result(dataset_name, model_name, explainer_name, "tail_coverage", seed, None, explanation_quality.tail_coverage(annotated))
        if "diversity" in requested:
            collector.add_result(dataset_name, model_name, explainer_name, "diversity", seed, None, explanation_quality.explanation_diversity(annotated))
        if "explanation_arp" in requested:
            collector.add_result(dataset_name, model_name, explainer_name, "explanation_arp", seed, None, explanation_quality.explanation_arp(annotated, dataset.popularity))
        if "coverage" in requested:
            collector.add_result(dataset_name, model_name, explainer_name, "coverage", seed, None, explanation_quality.explanation_coverage(annotated, dataset.num_items))
        if "personalization" in requested:
            collector.add_result(dataset_name, model_name, explainer_name, "personalization", seed, None, explanation_quality.explanation_personalization(annotated))
    except MetricRangeError as exc:
        collector.add_failure(
            dataset_name, model_name, explainer_name, seed, "degenerate",
            exception_type=type(exc).__name__, message=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        collector.add_failure(
            dataset_name, model_name, explainer_name, seed, _classify_exception(exc),
            exception_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc(),
        )


def run_experiment(config: ExperimentConfig) -> ResultsCollector:
    collector = ResultsCollector()
    device = resolve_device(config.determinism.device)

    dataset_bundles: dict[str, DatasetBundle] = {
        ds.name: build_dataset_bundle(ds) for ds in config.datasets
    }

    for seed in config.determinism.seeds:
        for dataset_config in config.datasets:
            dataset = dataset_bundles[dataset_config.name]
            for model_config in config.models:
                if not model_config.applies_to_dataset(dataset_config.name):
                    continue

                model = build_model(model_config)
                report = model.check_preconditions(dataset)
                if not report.satisfied:
                    collector.add_failure(
                        dataset_config.name, model_config.name, None, seed, "precondition-unmet",
                        measurements=report.measurements, requirement=report.requirement, reason=report.reason,
                    )
                    continue

                seed_all(seed)
                try:
                    model.fit(dataset, seed, device)
                    users = list(dataset.user_test_dict().keys())
                    recs = model.recommend(users, k=max(dataset.topk))
                except Exception as exc:  # noqa: BLE001
                    collector.add_failure(
                        dataset_config.name, model_config.name, None, seed, _classify_exception(exc),
                        exception_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc(),
                    )
                    continue

                _compute_recommendation_metrics(
                    collector, dataset, dataset_config.name, model_config.name, seed, recs,
                    config.metrics.accuracy, config.metrics.fairness,
                )

                for explainer_config in config.explainers:
                    if model_config.name not in explainer_config.applies_to_models:
                        continue
                    explainer = build_explainer(explainer_config)
                    if not explainer.compatible_with(model):
                        continue
                    try:
                        explained = explainer.explain_batch(model, dataset, recs)
                    except Exception as exc:  # noqa: BLE001
                        collector.add_failure(
                            dataset_config.name, model_config.name, explainer_config.name, seed,
                            _classify_exception(exc), exception_type=type(exc).__name__,
                            message=str(exc), traceback=traceback.format_exc(),
                        )
                        continue
                    _compute_explanation_metrics(
                        collector, dataset, dataset_config.name, model_config.name,
                        explainer_config.name, seed, explained, config.metrics.explanation,
                    )

    return collector
