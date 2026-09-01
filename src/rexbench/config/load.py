from pathlib import Path

import yaml

from rexbench.config.schema import ExperimentConfig


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return ExperimentConfig.model_validate(raw)
