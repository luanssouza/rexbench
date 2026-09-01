from __future__ import annotations

from rexbench.config.schema import ExplainerConfig, ModelConfig
from rexbench.explainers.base import ExplainerAdapter
from rexbench.explainers.recoxplainer_ar import ARExplainerAdapter
from rexbench.explainers.recoxplainer_knn import KNNExplainerAdapter
from rexbench.models.base import ModelAdapter
from rexbench.models.ebpr_adapter import EBPRModelAdapter
from rexbench.models.pgpr_adapter import PGPRModelAdapter
from rexbench.models.recbole_adapter import RecBoleModelAdapter

_MODEL_ADAPTERS = {
    "recbole": RecBoleModelAdapter,
    "ebpr": EBPRModelAdapter,
    "pgpr": PGPRModelAdapter,
}

_EXPLAINER_ADAPTERS = {
    "recoxplainer_ar": ARExplainerAdapter,
    "recoxplainer_knn": KNNExplainerAdapter,
}


def build_model(config: ModelConfig) -> ModelAdapter:
    cls = _MODEL_ADAPTERS.get(config.adapter)
    if cls is None:
        raise KeyError(f"no adapter registered for {config.adapter!r}; known: {sorted(_MODEL_ADAPTERS)}")
    return cls(config)


def build_explainer(config: ExplainerConfig) -> ExplainerAdapter:
    cls = _EXPLAINER_ADAPTERS.get(config.adapter)
    if cls is None:
        raise KeyError(f"no explainer adapter registered for {config.adapter!r}; known: {sorted(_EXPLAINER_ADAPTERS)}")
    return cls(config.hyperparameters)
