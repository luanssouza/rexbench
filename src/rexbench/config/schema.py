"""Pydantic schema for a rexbench experiment config.

Every experimental variable that AUDIT.md found implicitly fixed by RecBole/library
defaults (seed, split ratios, negative sampling, k, tail fraction, per-model
hyperparameters) is a required or explicitly-defaulted field here, not a bare
Python default buried in adapter code.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

KgStatus = Literal["available", "needs_construction", "unsupported"]
Device = Literal["auto", "cpu", "cuda", "mps"]
SplitStrategy = Literal["random", "temporal"]
FailureStatus = Literal["crashed", "degenerate", "resource", "skipped", "precondition-unmet"]


class Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExperimentMeta(Frozen):
    name: str
    output_dir: str = "rexbench/outputs"


class DeterminismConfig(Frozen):
    seeds: list[int] = Field(min_length=1)
    reproducibility: bool = True
    device: Device = "auto"


class SplitConfig(Frozen):
    strategy: SplitStrategy = "random"
    val_size: float = 0.1
    test_size: float = 0.2
    random_state: int = 200
    min_interactions: int = 5  # only used by strategy="temporal"


class NegativeSamplingConfig(Frozen):
    distribution: Literal["uniform", "popularity"] = "uniform"
    sample_num: int = 1


class KgConfig(Frozen):
    status: KgStatus
    source_dir: str | None = None


class DatasetConfig(Frozen):
    name: str
    loader: str
    raw_path: str | dict[str, str]
    split: SplitConfig = SplitConfig()
    negative_sampling: NegativeSamplingConfig = NegativeSamplingConfig()
    topk: list[int] = Field(default_factory=lambda: [5, 10])
    tail_fraction: float = 0.20
    kg: KgConfig = KgConfig(status="unsupported")

    @model_validator(mode="after")
    def _validate_kg(self) -> "DatasetConfig":
        if self.kg.status == "available" and not self.kg.source_dir:
            raise ValueError(f"dataset {self.name!r}: kg.status=available requires kg.source_dir")
        return self


class PreconditionConfig(Frozen):
    min_fraction_users_with_2plus: float | None = None
    min_kg_triples: int | None = None


class HPOParamSpec(Frozen):
    """One hyperparameter's search dimension. `type="choice"` needs `values`; the 3 numeric
    types need `low`/`high`. `int_uniform` samples an int (inclusive), `uniform` a float,
    `loguniform` a float sampled in log-space (for scale-free params like learning rate)."""
    type: Literal["choice", "int_uniform", "uniform", "loguniform"]
    values: list | None = None
    low: float | None = None
    high: float | None = None

    @model_validator(mode="after")
    def _validate_bounds(self) -> "HPOParamSpec":
        if self.type == "choice":
            if not self.values:
                raise ValueError("HPOParamSpec type='choice' requires a non-empty values list")
        else:
            if self.low is None or self.high is None:
                raise ValueError(f"HPOParamSpec type={self.type!r} requires low and high")
            if self.low >= self.high:
                raise ValueError(f"HPOParamSpec: low ({self.low}) must be < high ({self.high})")
            if self.type == "loguniform" and self.low <= 0:
                raise ValueError("HPOParamSpec type='loguniform' requires low > 0")
        return self


class HPOConfig(Frozen):
    """Config-declared hyperparameter search. Runs once per (model, dataset) — not once per
    experiment seed, see core/runner.py — evaluated against the validation split only, never
    test, to avoid leaking the tuning signal into the numbers actually reported."""
    strategy: Literal["grid", "random"] = "random"
    n_trials: int = 20  # ignored for strategy="grid" (runs the full cartesian product)
    metric: str = "ndcg"
    k: int = 10
    direction: Literal["maximize", "minimize"] = "maximize"
    seed: int = 2020  # search-only: every trial's fit() reuses this SAME seed, isolating the
                       # hyperparameter effect from seed noise; a separate RNG derived from
                       # it (not model training) decides which combinations get sampled
    search_space: dict[str, HPOParamSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_grid_is_discrete(self) -> "HPOConfig":
        if self.strategy == "grid":
            non_choice = [name for name, spec in self.search_space.items() if spec.type != "choice"]
            if non_choice:
                raise ValueError(
                    f"strategy='grid' requires every search_space dimension to be "
                    f"type='choice' (no natural grid over a continuous range); "
                    f"non-choice dimensions: {non_choice}"
                )
        return self


class DatasetOverride(Frozen):
    """Per-(model, dataset) override — e.g. a lower EBPR sparsity precondition and a
    smaller neighborhood specifically for known-sparse datasets, without duplicating the
    model under a second name (which would split one model's results across two rows in
    results.parquet). Unset fields fall back to the model's own top-level value; set fields
    replace it entirely (hyperparameters replaces the whole dict, not a per-key merge —
    keeps "what ran with what config" unambiguous from the override block alone)."""
    hyperparameters: dict | None = None
    precondition: PreconditionConfig | None = None
    hpo: HPOConfig | None = None


class ModelConfig(Frozen):
    name: str
    tier: Literal[1, 2]
    adapter: Literal["recbole", "ebpr", "pgpr"]
    variant: str | None = None  # e.g. "EBPR" | "UBPR" | "UEBPR" for the ebpr adapter
    recbole_model: str | None = None  # e.g. "BPR" | "SLIMElastic" for the recbole adapter
    applies_to: Literal["all_datasets"] | list[str] = "all_datasets"
    hyperparameters: dict = Field(default_factory=dict)
    precondition: PreconditionConfig = PreconditionConfig()
    hpo: HPOConfig | None = None
    dataset_overrides: dict[str, DatasetOverride] = Field(default_factory=dict)

    def applies_to_dataset(self, dataset_name: str) -> bool:
        return self.applies_to == "all_datasets" or dataset_name in self.applies_to

    def hyperparameters_for(self, dataset_name: str) -> dict:
        override = self.dataset_overrides.get(dataset_name)
        if override is not None and override.hyperparameters is not None:
            return override.hyperparameters
        return self.hyperparameters

    def precondition_for(self, dataset_name: str) -> PreconditionConfig:
        override = self.dataset_overrides.get(dataset_name)
        if override is not None and override.precondition is not None:
            return override.precondition
        return self.precondition

    def hpo_for(self, dataset_name: str) -> HPOConfig | None:
        override = self.dataset_overrides.get(dataset_name)
        if override is not None and override.hpo is not None:
            return override.hpo
        return self.hpo

    @model_validator(mode="after")
    def _validate_adapter_fields(self) -> "ModelConfig":
        if self.adapter == "ebpr" and not self.variant:
            raise ValueError(f"model {self.name!r}: adapter=ebpr requires variant")
        if self.adapter == "recbole" and not self.recbole_model:
            raise ValueError(f"model {self.name!r}: adapter=recbole requires recbole_model")
        return self


class ExplainerConfig(Frozen):
    name: str
    adapter: Literal["recoxplainer_ar", "recoxplainer_knn"]
    applies_to_models: list[str]
    hyperparameters: dict = Field(default_factory=dict)


class MetricsConfig(Frozen):
    # accuracy/fairness operate on recommendation lists (the predecessor pipeline's
    # compute_metrics groups NDCG/MAP/Variance/Gini/Entropy/ARP together — AUDIT.md 1.3).
    # `explanation_arp` is a distinct metric (mean popularity of items inside *explanation*
    # sets, not recommendation lists) — kept separate from `arp` to avoid the ambiguity the
    # audit flagged between the two similarly-named metrics.
    accuracy: list[str] = Field(default_factory=lambda: ["ndcg", "map"])
    fairness: list[str] = Field(
        default_factory=lambda: ["gini", "variance", "entropy", "arp", "arp_normalized"]
    )
    explanation: list[str] = Field(
        default_factory=lambda: [
            "fidelity", "tail_coverage", "diversity", "explanation_arp",
            "coverage", "personalization",
        ]
    )


class StatisticsConfig(Frozen):
    tests: list[Literal["friedman", "quade", "nemenyi", "kendall_w"]] = Field(
        default_factory=lambda: ["friedman", "quade", "nemenyi", "kendall_w"]
    )
    alpha: float = 0.05


class ExperimentConfig(Frozen):
    experiment: ExperimentMeta
    determinism: DeterminismConfig
    datasets: list[DatasetConfig] = Field(min_length=1)
    models: list[ModelConfig] = Field(min_length=1)
    explainers: list[ExplainerConfig] = Field(default_factory=list)
    metrics: MetricsConfig = MetricsConfig()
    statistics: StatisticsConfig = StatisticsConfig()

    @model_validator(mode="after")
    def _validate_references(self) -> "ExperimentConfig":
        dataset_names = {d.name for d in self.datasets}
        model_names = {m.name for m in self.models}
        for m in self.models:
            if m.applies_to != "all_datasets":
                unknown = set(m.applies_to) - dataset_names
                if unknown:
                    raise ValueError(f"model {m.name!r} applies_to unknown datasets: {unknown}")
            unknown_overrides = set(m.dataset_overrides) - dataset_names
            if unknown_overrides:
                raise ValueError(f"model {m.name!r} dataset_overrides unknown datasets: {unknown_overrides}")
            not_applicable = {
                d for d in m.dataset_overrides if not m.applies_to_dataset(d)
            }
            if not_applicable:
                raise ValueError(
                    f"model {m.name!r} dataset_overrides for datasets it doesn't apply_to: {not_applicable}"
                )
        for e in self.explainers:
            unknown = set(e.applies_to_models) - model_names
            if unknown:
                raise ValueError(f"explainer {e.name!r} applies_to_models unknown models: {unknown}")
        return self

    def dataset(self, name: str) -> DatasetConfig:
        for d in self.datasets:
            if d.name == name:
                return d
        raise KeyError(name)
