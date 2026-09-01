"""Item-based kNN post-hoc explainer.

The predecessor pipeline's post_hoc_knn wrapper is a two-line call to
recoxplainer.explain.KNNPostHocExplainer that does not expose its `knn` parameter, so this
adapter calls KNNPostHocExplainer directly (the actual explainer class, unmodified) to make
`knn` config-driven, mirroring that wrapper's own call pattern exactly.
"""
from __future__ import annotations

import pandas as pd

import rexbench.vendor  # noqa: F401
from recoxplainer.explain import KNNPostHocExplainer

from rexbench.core.dataset import DatasetBundle
from rexbench.explainers._bridge import explanations_to_canonical, load_recoxplainer_bridge
from rexbench.explainers.base import ExplainerAdapter
from rexbench.models.base import ModelAdapter
from rexbench.models.recbole_adapter import RecBoleModelAdapter


class KNNExplainerAdapter(ExplainerAdapter):
    name = "KNN"

    def __init__(self, hyperparameters: dict):
        self.hyperparameters = hyperparameters

    def compatible_with(self, model: ModelAdapter) -> bool:
        return isinstance(model, RecBoleModelAdapter)

    def explain_batch(
        self, model: ModelAdapter, dataset: DatasetBundle, recommendations: pd.DataFrame
    ) -> pd.DataFrame:
        assert isinstance(model, RecBoleModelAdapter)
        custom_model, recs, data = load_recoxplainer_bridge(model.checkpoint_path(), dataset)
        explainer = KNNPostHocExplainer(custom_model, recs.copy(), data, **self.hyperparameters)
        explained = explainer.explain_recommendations()
        return explanations_to_canonical(explained, data)
