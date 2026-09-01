"""Association-Rules post-hoc explainer.

Reuses rexfair.rexfair.explanation.explainers.post_hoc_ar (recoxplainer.explain.
ARPostHocExplainer under the hood) unmodified. AUDIT.md 1.2 found this explainer produces
model_fidelity=0.0 on datasets with >99.9% sparsity (amazon/rentrunway/electronics) — that
is expected, disclosed behavior of the method on those datasets, not a bug; it will surface
here as a legitimate, low (not degenerate/error) metric value.
"""
from __future__ import annotations

import pandas as pd

import rexbench.vendor  # noqa: F401
from rexfair.explanation.explainers import post_hoc_ar

from rexbench.core.dataset import DatasetBundle
from rexbench.explainers._bridge import explanations_to_canonical, load_recoxplainer_bridge
from rexbench.explainers.base import ExplainerAdapter
from rexbench.models.base import ModelAdapter
from rexbench.models.recbole_adapter import RecBoleModelAdapter


class ARExplainerAdapter(ExplainerAdapter):
    name = "AR"

    def __init__(self, hyperparameters: dict):
        self.hyperparameters = hyperparameters

    def compatible_with(self, model: ModelAdapter) -> bool:
        return isinstance(model, RecBoleModelAdapter)

    def explain_batch(
        self, model: ModelAdapter, dataset: DatasetBundle, recommendations: pd.DataFrame
    ) -> pd.DataFrame:
        assert isinstance(model, RecBoleModelAdapter)
        custom_model, recs, data = load_recoxplainer_bridge(model.checkpoint_path(), dataset)
        explained = post_hoc_ar(custom_model, recs, data, **self.hyperparameters)
        return explanations_to_canonical(explained, data)
