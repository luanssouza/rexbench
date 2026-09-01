"""Deliberately does NOT eagerly import core.runner here (it pulls in every model adapter,
including torch-dependent ones) — that would make even `from rexbench.core.dataset import
...` require torch/recbole to be installed. Import `rexbench.core.runner.run_experiment`
directly where needed (cli.py already does)."""
from rexbench.core.dataset import DatasetBundle, build_dataset_bundle
from rexbench.core.determinism import build_manifest, resolve_device, seed_all
from rexbench.core.results_io import ResultsCollector

__all__ = [
    "DatasetBundle", "build_dataset_bundle", "seed_all", "resolve_device", "build_manifest",
    "ResultsCollector",
]
