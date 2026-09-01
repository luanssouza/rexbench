import yaml

from rexbench.config.schema import ExperimentConfig


def _minimal_config_dict():
    return {
        "experiment": {"name": "t"},
        "determinism": {"seeds": [1]},
        "datasets": [
            {"name": "ml100k", "loader": "ml100k", "raw_path": "x.data"},
        ],
        "models": [
            {"name": "BPR", "tier": 2, "adapter": "recbole", "recbole_model": "BPR"},
        ],
    }


def test_minimal_config_round_trips():
    cfg = ExperimentConfig.model_validate(_minimal_config_dict())
    assert cfg.experiment.name == "t"
    assert cfg.datasets[0].split.random_state == 200  # declared default, not a bare constant elsewhere
    assert cfg.datasets[0].topk == [5, 10]


def test_ebpr_requires_variant():
    d = _minimal_config_dict()
    d["models"] = [{"name": "EBPR", "tier": 1, "adapter": "ebpr"}]
    try:
        ExperimentConfig.model_validate(d)
        assert False, "expected validation error"
    except Exception:
        pass


def test_applies_to_unknown_dataset_rejected():
    d = _minimal_config_dict()
    d["models"][0]["applies_to"] = ["not_a_real_dataset"]
    try:
        ExperimentConfig.model_validate(d)
        assert False, "expected validation error"
    except Exception:
        pass


def test_kg_available_requires_source_dir():
    d = _minimal_config_dict()
    d["datasets"][0]["kg"] = {"status": "available"}
    try:
        ExperimentConfig.model_validate(d)
        assert False, "expected validation error"
    except Exception:
        pass


def test_real_configs_load():
    for path in ("configs/tier1.yaml", "configs/full.yaml"):
        raw = yaml.safe_load(open(path))
        cfg = ExperimentConfig.model_validate(raw)
        assert len(cfg.datasets) == 8
        assert any(m.name == "EBPR" for m in cfg.models)
