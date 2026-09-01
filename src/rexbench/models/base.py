from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import pandas as pd

from rexbench.config.schema import ModelConfig
from rexbench.core.dataset import DatasetBundle


@dataclass
class PreconditionReport:
    satisfied: bool
    measurements: dict[str, float] = field(default_factory=dict)
    requirement: dict[str, float] = field(default_factory=dict)
    reason: str | None = None

    @staticmethod
    def ok() -> "PreconditionReport":
        return PreconditionReport(satisfied=True)


@dataclass
class Explanation:
    kind: str  # e.g. "pgpr_path"
    payload: Any


class ModelAdapter(ABC):
    """Wraps an existing, audited implementation. fit()/recommend() must call into the
    original training/scoring code — see AUDIT.md and each adapter's module docstring for
    exactly which functions are called and which lines, if any, carry a disclosed patch."""

    name: str
    family: str  # "recbole" | "ebpr" | "pgpr"

    def __init__(self, config: ModelConfig):
        self.config = config

    def check_preconditions(self, dataset: DatasetBundle) -> PreconditionReport:
        return PreconditionReport.ok()

    @abstractmethod
    def fit(self, dataset: DatasetBundle, seed: int, device: str) -> None: ...

    @abstractmethod
    def recommend(self, users: Sequence[int], k: int) -> pd.DataFrame:
        """Returns a DataFrame with columns [user_id, item_id, rank, score]."""

    def explain(self, user_id: int, item_id: int) -> Explanation | None:
        """Native, model-intrinsic explanation (e.g. PGPR's reasoning path). None means this
        model has no built-in explanation mechanism — pair it with an ExplainerAdapter
        (AR/KNN) instead, which is model-agnostic and post-hoc."""
        return None
