"""The dataset x model x (HPO once) x seed x explainer loop and failure classification.

AUDIT.md's FAILURE HANDLING requirement, made concrete: every attempted combination
produces a row in `results` (status=ok) or `failures` (status in {crashed, degenerate,
resource, skipped, precondition-unmet}) — nothing is silently dropped, and a metric that
returns an out-of-range value (rexbench.metrics.validate.MetricRangeError) is caught and
recorded as `degenerate` rather than allowed to reach results.parquet — this is the
mechanism that would have caught AUDIT.md section 5's Gini -1.0 sentinel as a classified
failure instead of a silent out-of-range number. HPO trials (core/hpo.py) get the same
treatment via the `trials` table.

Loop order is `dataset -> model -> seed`, not `seed -> dataset -> model`: HPO (when a model
declares one via ModelConfig.hpo_for) runs exactly once per (dataset, model), before the
seed loop, since re-tuning per seed would multiply cost by n_trials x n_seeds for no benefit
— the seed loop exists to measure variance of the *final chosen* hyperparameters, not to
re-search them. dataset_bundles are built once up front regardless of loop order, so this
doesn't change dataset-loading behavior.
"""
from __future__ import annotations

import traceback

import pandas as pd

from rexbench.config.schema import ExperimentConfig, ModelConfig
from rexbench.core.dataset import DatasetBundle, build_dataset_bundle
from rexbench.core.determinism import resolve_device, seed_all
from rexbench.core.hpo import run_hpo
from rexbench.core.registry import build_explainer, build_model
from rexbench.core.results_io import ResultsCollector
from rexbench.metrics import explanation_quality
from rexbench.metrics.dispatch import compute_metric_value
from rexbench.metrics.validate import MetricRangeError

_RESOURCE_SIGNATURES = ("out of memory", "cuda oom", "cannot allocate memory")

_ACCURACY_METRICS = ("ndcg", "map")
_FAIRNESS_METRICS = (
    "gini", "variance", "entropy", "arp", "arp_normalized", "tail_coverage",
    # item-side (provider-side) — metrics/fairness.py
    "item_coverage", "item_exposure_gini", "tail_item_coverage", "tail_exposure_share",
)


def _classify_exception(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(sig in text for sig in _RESOURCE_SIGNATURES) or type(exc).__name__ in (
        "OutOfMemoryError", "MemoryError",
    ):
        return "resource"
    return "crashed"


def _compute_recommendation_metrics(
    collector: ResultsCollector, dataset: DatasetBundle, dataset_name: str, model_name: str,
    seed: int, recs: pd.DataFrame, requested_accuracy: list[str], requested_fairness: list[str],
) -> None:
    user_test_dict = dataset.user_test_dict()
    metric_names = [m for m in _ACCURACY_METRICS if m in requested_accuracy]
    metric_names += [m for m in _FAIRNESS_METRICS if m != "tail_coverage" and m in requested_fairness]
    # results.parquet disambiguates this from the explanation-quality metric of the same
    # short config name (AUDIT.md flagged the ambiguity) by storing it as tail_coverage_rec.
    stored_name = {m: m for m in metric_names}
    if "tail_coverage" in requested_fairness:
        metric_names.append("tail_coverage")
        stored_name["tail_coverage"] = "tail_coverage_rec"

    for k in dataset.topk:
        for metric_name in metric_names:
            try:
                value = compute_metric_value(metric_name, recs, user_test_dict, dataset, k)
            except MetricRangeError as exc:
                collector.add_failure(
                    dataset_name, model_name, None, seed, "degenerate",
                    exception_type=type(exc).__name__, message=str(exc),
                )
                continue
            except Exception as exc:  # noqa: BLE001 — deliberately broad, see module docstring
                collector.add_failure(
                    dataset_name, model_name, None, seed, _classify_exception(exc),
                    exception_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc(),
                )
                continue
            collector.add_result(dataset_name, model_name, None, stored_name[metric_name], seed, k, value)


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


def _resolve_hyperparameters(
    collector: ResultsCollector, model_config: ModelConfig, dataset: DatasetBundle, device: str,
) -> ModelConfig:
    """Runs HPO once for this (model, dataset) if configured, logs every trial, and returns
    a ModelConfig with the winning hyperparameters baked in as the top-level value (clearing
    any stale per-dataset override for this dataset so it doesn't get re-applied on top) —
    every seed's build_model() call downstream resolves hyperparameters_for(dataset.name) to
    the tuned values transparently, with no ModelAdapter interface change needed."""
    hpo = model_config.hpo_for(dataset.name)
    if hpo is None:
        return model_config

    best_hp, trial_records = run_hpo(model_config, dataset, hpo, device)
    for trial in trial_records:
        collector.add_trial(
            dataset.name, model_config.name, trial.trial_id, trial.hyperparameters,
            hpo.metric, hpo.k, trial.value, trial.status, trial.exception_type, trial.message,
        )
    return model_config.model_copy(update={
        "hyperparameters": best_hp,
        "dataset_overrides": {
            name: ov for name, ov in model_config.dataset_overrides.items() if name != dataset.name
        },
    })


def run_experiment(config: ExperimentConfig) -> ResultsCollector:
    collector = ResultsCollector()
    device = resolve_device(config.determinism.device)

    dataset_bundles: dict[str, DatasetBundle] = {
        ds.name: build_dataset_bundle(ds) for ds in config.datasets
    }

    for dataset_config in config.datasets:
        dataset = dataset_bundles[dataset_config.name]
        for model_config in config.models:
            if not model_config.applies_to_dataset(dataset_config.name):
                continue

            probe = build_model(model_config)
            report = probe.check_preconditions(dataset)
            if not report.satisfied:
                for seed in config.determinism.seeds:
                    collector.add_failure(
                        dataset_config.name, model_config.name, None, seed, "precondition-unmet",
                        measurements=report.measurements, requirement=report.requirement, reason=report.reason,
                    )
                continue

            resolved_config = _resolve_hyperparameters(collector, model_config, dataset, device)

            for seed in config.determinism.seeds:
                model = build_model(resolved_config)
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
