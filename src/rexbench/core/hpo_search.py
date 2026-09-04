"""Pure trial-sampling logic for core/hpo.py, split into its own module so it stays
importable without torch/RecBole/EBPR/PGPR (core/hpo.py's `run_hpo` needs core.registry,
which pulls in every model adapter — this module deliberately doesn't)."""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass

from rexbench.config.schema import HPOParamSpec


@dataclass
class TrialRecord:
    trial_id: int
    hyperparameters: dict
    status: str  # "ok" | "crashed" | "degenerate"
    value: float | None
    exception_type: str | None = None
    message: str | None = None


def enumerate_grid_trials(search_space: dict[str, HPOParamSpec]) -> list[dict]:
    names = list(search_space.keys())
    value_lists = [search_space[name].values for name in names]
    return [dict(zip(names, combo)) for combo in itertools.product(*value_lists)]


def _sample_one(spec: HPOParamSpec, rng: random.Random):
    if spec.type == "choice":
        return rng.choice(spec.values)
    if spec.type == "int_uniform":
        return rng.randint(int(spec.low), int(spec.high))
    if spec.type == "uniform":
        return rng.uniform(spec.low, spec.high)
    if spec.type == "loguniform":
        log_low, log_high = math.log(spec.low), math.log(spec.high)
        return math.exp(rng.uniform(log_low, log_high))
    raise ValueError(f"unknown HPOParamSpec type {spec.type!r}")


def sample_random_trials(search_space: dict[str, HPOParamSpec], n_trials: int, seed: int) -> list[dict]:
    rng = random.Random(seed)  # dedicated to trial sampling, independent of model training
    return [
        {name: _sample_one(spec, rng) for name, spec in search_space.items()}
        for _ in range(n_trials)
    ]


def worst_value(direction: str) -> float:
    return float("-inf") if direction == "maximize" else float("inf")


def is_better(candidate: float, current_best: float, direction: str) -> bool:
    return candidate > current_best if direction == "maximize" else candidate < current_best
