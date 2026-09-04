"""Grid/random hyperparameter search.

Runs once per (model, dataset) — not once per experiment seed (core/runner.py calls this
before its seed loop) — evaluated against the validation split only
(DatasetBundle.user_val_dict()), never test, so tuning can't leak into the numbers actually
reported. No external search library: grid enumerates the cartesian product, random search
samples with its own dedicated RNG (core/hpo_search.py). Deliberately not wired to RecBole's
own HyperTuning or to hyperopt/ray[tune] — AUDIT.md already found Ray Tune caused a hard
version-mismatch crash in the ProtoMF baseline, and this pipeline's merged environment
(numpy<2.0, torch>=2.0 exactly) is fragile enough already without adding another pinned
dependency.
"""
from __future__ import annotations

import traceback

from rexbench.config.schema import HPOConfig, ModelConfig
from rexbench.core.dataset import DatasetBundle
from rexbench.core.determinism import seed_all
from rexbench.core.hpo_search import (
    TrialRecord, enumerate_grid_trials, is_better, sample_random_trials, worst_value,
)
from rexbench.core.registry import build_model
from rexbench.metrics.dispatch import compute_metric_value
from rexbench.metrics.validate import MetricRangeError


def run_hpo(
    model_config: ModelConfig, dataset: DatasetBundle, hpo: HPOConfig, device: str,
) -> tuple[dict, list[TrialRecord]]:
    """Returns (best_hyperparameters, all_trial_records). best_hyperparameters is always a
    full, usable hyperparameters dict — trial dimensions merged on top of the model's own
    base hyperparameters_for(dataset.name), so dimensions not in search_space keep their
    configured value. If every trial fails, falls back to the base hyperparameters
    unchanged (never returns an empty/partial dict)."""
    base_hp = model_config.hyperparameters_for(dataset.name)

    if hpo.strategy == "grid":
        trial_overrides = enumerate_grid_trials(hpo.search_space)
    else:
        trial_overrides = sample_random_trials(hpo.search_space, hpo.n_trials, hpo.seed)

    val_users = list(dataset.user_val_dict().keys())
    records: list[TrialRecord] = []
    best_value = worst_value(hpo.direction)
    best_hp = dict(base_hp)

    for trial_id, overrides in enumerate(trial_overrides):
        trial_hp = {**base_hp, **overrides}
        trial_config = model_config.model_copy(update={
            "hyperparameters": trial_hp,
            "dataset_overrides": {
                name: ov for name, ov in model_config.dataset_overrides.items() if name != dataset.name
            },
        })
        try:
            seed_all(hpo.seed)  # same seed every trial — isolates the hyperparameter effect
            model = build_model(trial_config)
            model.fit(dataset, hpo.seed, device)
            recs = model.recommend(val_users, k=hpo.k)
            value = compute_metric_value(hpo.metric, recs, dataset.user_val_dict(), dataset, hpo.k)
        except MetricRangeError as exc:
            records.append(TrialRecord(trial_id, trial_hp, "degenerate", None, type(exc).__name__, str(exc)))
            continue
        except Exception as exc:  # noqa: BLE001 — a bad trial must not abort the search
            records.append(TrialRecord(
                trial_id, trial_hp, "crashed", None, type(exc).__name__,
                f"{exc}\n{traceback.format_exc()}",
            ))
            continue

        records.append(TrialRecord(trial_id, trial_hp, "ok", value))
        if is_better(value, best_value, hpo.direction):
            best_value = value
            best_hp = trial_hp

    return best_hp, records
