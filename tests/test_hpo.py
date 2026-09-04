import pytest

from rexbench.config.schema import ExperimentConfig, HPOConfig, HPOParamSpec
from rexbench.core.hpo_search import enumerate_grid_trials, sample_random_trials


def test_enumerate_grid_trials_is_full_cartesian_product():
    search_space = {
        "num_latent": HPOParamSpec(type="choice", values=[4, 8]),
        "optimizer": HPOParamSpec(type="choice", values=["adam", "sgd"]),
    }
    trials = enumerate_grid_trials(search_space)
    assert len(trials) == 4
    combos = {(t["num_latent"], t["optimizer"]) for t in trials}
    assert combos == {(4, "adam"), (4, "sgd"), (8, "adam"), (8, "sgd")}


def test_sample_random_trials_reproducible_given_same_seed():
    search_space = {
        "lr": HPOParamSpec(type="loguniform", low=1e-4, high=1e-1),
        "num_latent": HPOParamSpec(type="int_uniform", low=4, high=64),
        "dropout": HPOParamSpec(type="uniform", low=0.0, high=0.5),
        "optimizer": HPOParamSpec(type="choice", values=["adam", "sgd"]),
    }
    trials_a = sample_random_trials(search_space, n_trials=10, seed=42)
    trials_b = sample_random_trials(search_space, n_trials=10, seed=42)
    assert trials_a == trials_b
    assert len(trials_a) == 10
    for t in trials_a:
        assert 1e-4 <= t["lr"] <= 1e-1
        assert 4 <= t["num_latent"] <= 64
        assert 0.0 <= t["dropout"] <= 0.5
        assert t["optimizer"] in ("adam", "sgd")


def test_sample_random_trials_different_seeds_differ():
    search_space = {"lr": HPOParamSpec(type="uniform", low=0.0, high=1.0)}
    trials_a = sample_random_trials(search_space, n_trials=5, seed=1)
    trials_b = sample_random_trials(search_space, n_trials=5, seed=2)
    assert trials_a != trials_b


def test_grid_strategy_rejects_non_choice_dimension():
    with pytest.raises(Exception):
        HPOConfig(
            strategy="grid",
            search_space={"lr": HPOParamSpec(type="uniform", low=0.0, high=1.0)},
        )


def test_grid_strategy_accepts_choice_only():
    cfg = HPOConfig(
        strategy="grid",
        search_space={"num_latent": HPOParamSpec(type="choice", values=[4, 8])},
    )
    assert cfg.strategy == "grid"


def test_choice_requires_values():
    with pytest.raises(Exception):
        HPOParamSpec(type="choice")


def test_numeric_type_requires_low_high():
    with pytest.raises(Exception):
        HPOParamSpec(type="uniform")


def test_loguniform_requires_positive_low():
    with pytest.raises(Exception):
        HPOParamSpec(type="loguniform", low=0.0, high=1.0)


def _minimal_config_dict_with_hpo():
    return {
        "experiment": {"name": "t"},
        "determinism": {"seeds": [1]},
        "datasets": [
            {"name": "ml100k", "loader": "ml100k", "raw_path": "x.data"},
            {"name": "ml1m", "loader": "ml1m", "raw_path": "y.dat"},
        ],
        "models": [
            {
                "name": "EBPR", "tier": 1, "adapter": "ebpr", "variant": "EBPR",
                "applies_to": "all_datasets",
                "hyperparameters": {"neighborhood": 20, "lr": 0.001},
                "hpo": {
                    "strategy": "random", "n_trials": 5, "metric": "ndcg", "k": 10,
                    "search_space": {"neighborhood": {"type": "choice", "values": [3, 5, 10]}},
                },
                "dataset_overrides": {
                    "ml1m": {"hpo": {
                        "strategy": "grid", "metric": "ndcg", "k": 10,
                        "search_space": {"neighborhood": {"type": "choice", "values": [1, 2]}},
                    }},
                },
            },
        ],
    }


def test_model_config_hpo_for_falls_back_and_overrides():
    cfg = ExperimentConfig.model_validate(_minimal_config_dict_with_hpo())
    model = cfg.models[0]
    assert model.hpo_for("ml100k").strategy == "random"
    assert model.hpo_for("ml1m").strategy == "grid"


def test_dataset_override_hpo_unknown_dataset_rejected():
    d = _minimal_config_dict_with_hpo()
    d["models"][0]["dataset_overrides"]["not_a_real_dataset"] = d["models"][0]["dataset_overrides"]["ml1m"]
    with pytest.raises(Exception):
        ExperimentConfig.model_validate(d)
