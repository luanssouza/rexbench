from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from rexbench.core.dataset import DatasetBundle
from rexbench.models.base import ModelAdapter


class ExplainerAdapter(ABC):
    """Model-agnostic, post-hoc explainer (AR, KNN) applied on top of any compatible
    ModelAdapter's recommendations — distinct from a model's own native explain() (see
    models/base.py's ModelAdapter.explain docstring for why these are separate)."""

    name: str

    def compatible_with(self, model: ModelAdapter) -> bool:
        return model.family == "recbole"

    @abstractmethod
    def explain_batch(
        self, model: ModelAdapter, dataset: DatasetBundle, recommendations: pd.DataFrame
    ) -> pd.DataFrame:
        """Returns recommendations with an added `explanation_set` column (a Python set of
        canonical item ids per row) — the schema metrics/explanation_quality.py expects."""
