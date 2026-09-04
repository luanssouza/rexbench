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


def test_dataset_override_unknown_dataset_rejected():
    d = _minimal_config_dict()
    d["models"][0] = {
        "name": "EBPR", "tier": 1, "adapter": "ebpr", "variant": "EBPR",
        "dataset_overrides": {"not_a_real_dataset": {"precondition": {"min_fraction_users_with_2plus": 0.05}}},
    }
    try:
        ExperimentConfig.model_validate(d)
        assert False, "expected validation error"
    except Exception:
        pass


def test_dataset_override_outside_applies_to_rejected():
    d = _minimal_config_dict()
    d["datasets"].append({"name": "ml1m", "loader": "ml1m", "raw_path": "y.dat"})
    d["models"][0] = {
        "name": "EBPR", "tier": 1, "adapter": "ebpr", "variant": "EBPR",
        "applies_to": ["ml100k"],  # ml1m deliberately excluded
        "dataset_overrides": {"ml1m": {"precondition": {"min_fraction_users_with_2plus": 0.05}}},
    }
    try:
        ExperimentConfig.model_validate(d)
        assert False, "expected validation error"
    except Exception:
        pass


def test_hyperparameters_and_precondition_for_fall_back_and_override():
    d = _minimal_config_dict()
    d["datasets"].append({"name": "ml1m", "loader": "ml1m", "raw_path": "y.dat"})
    d["models"][0] = {
        "name": "EBPR", "tier": 1, "adapter": "ebpr", "variant": "EBPR",
        "applies_to": "all_datasets",
        "hyperparameters": {"neighborhood": 20},
        "precondition": {"min_fraction_users_with_2plus": 0.5},
        "dataset_overrides": {
            "ml1m": {
                "hyperparameters": {"neighborhood": 5},
                "precondition": {"min_fraction_users_with_2plus": 0.05},
            }
        },
    }
    cfg = ExperimentConfig.model_validate(d)
    model = cfg.models[0]
    # ml100k has no override -> falls back to the model's own top-level values
    assert model.hyperparameters_for("ml100k")["neighborhood"] == 20
    assert model.precondition_for("ml100k").min_fraction_users_with_2plus == 0.5
    # ml1m has an override -> replaced entirely
    assert model.hyperparameters_for("ml1m")["neighborhood"] == 5
    assert model.precondition_for("ml1m").min_fraction_users_with_2plus == 0.05


def test_real_configs_sparse_ebpr_overrides_resolve():
    for path in ("configs/tier1.yaml", "configs/full.yaml"):
        raw = yaml.safe_load(open(path))
        cfg = ExperimentConfig.model_validate(raw)
        for name in ("EBPR", "UBPR", "UEBPR"):
            model = next(m for m in cfg.models if m.name == name)
            for sparse_dataset in ("electronics", "rentrunway", "amazon_digital_music"):
                assert model.precondition_for(sparse_dataset).min_fraction_users_with_2plus == 0.05
                assert model.hyperparameters_for(sparse_dataset)["neighborhood"] == 5
            # an unaffected dataset still gets the model's default, unchanged
            assert model.precondition_for("ml100k").min_fraction_users_with_2plus == 0.5
            assert model.hyperparameters_for("ml100k")["neighborhood"] == 20
