import pytest
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
    assert cfg.datasets[0].sample.max_users is None  # no sampling unless explicitly configured


def test_dataset_sample_config_resolves():
    d = _minimal_config_dict()
    d["datasets"][0]["sample"] = {"max_users": 100, "seed": 7}
    cfg = ExperimentConfig.model_validate(d)
    assert cfg.datasets[0].sample.max_users == 100
    assert cfg.datasets[0].sample.seed == 7


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


# --------------------------------------------------------------- shipped configs all load

def _load(path):
    return ExperimentConfig.model_validate(yaml.safe_load(open(path)))


ALL_CONFIGS = [
    "configs/tier1.yaml", "configs/tier1_local.yaml", "configs/full.yaml",
    "configs/hpo_example.yaml", "configs/smoke_test.yaml", "configs/smoke_test_local.yaml",
    "configs/stage1_tier1.yaml", "configs/stage1_tier1_local.yaml", "configs/stage1_full.yaml",
]

STAGE1_DEFERRED = {"ambar", "amazon_digital_music"}


@pytest.mark.parametrize("path", ALL_CONFIGS)
def test_every_shipped_config_loads(path):
    config = _load(path)
    assert config.datasets and config.models


@pytest.mark.parametrize("path", ALL_CONFIGS)
def test_split_store_dirs_are_namespaced_by_scale(path):
    """Full-scale splits under data/splits/full/, sampled ones under data/splits/smoke_test/
    — so a sampled split can never be mistaken for a full-scale one at the same path."""
    for dataset in _load(path).datasets:
        store_dir = dataset.split.store_dir
        assert store_dir is not None, f"{path}:{dataset.name} has no store_dir"
        sampled = dataset.sample.max_users is not None
        expected = "data/splits/smoke_test/" if sampled else "data/splits/full/"
        assert store_dir.startswith(expected), f"{path}:{dataset.name} -> {store_dir}"


@pytest.mark.parametrize(
    "path", ["configs/stage1_tier1.yaml", "configs/stage1_tier1_local.yaml", "configs/stage1_full.yaml"]
)
def test_stage1_configs_exclude_the_deferred_datasets(path):
    config = _load(path)
    names = {d.name for d in config.datasets}
    assert not (names & STAGE1_DEFERRED), f"{path} still contains {names & STAGE1_DEFERRED}"
    assert len(names) == 6
    # a dataset_override pointing at a dropped dataset would be a dangling reference
    for model in config.models:
        assert not (set(model.dataset_overrides) & STAGE1_DEFERRED)
        if isinstance(model.applies_to, list):
            assert not (set(model.applies_to) & STAGE1_DEFERRED)


def test_stage1_shares_store_dir_with_the_full_config():
    """Splits materialized by stage 1 must be reused verbatim by the 8-dataset configs
    later, not recomputed — same store_dir is what guarantees that."""
    stage1 = {d.name: d.split.store_dir for d in _load("configs/stage1_tier1.yaml").datasets}
    tier1 = {d.name: d.split.store_dir for d in _load("configs/tier1.yaml").datasets}
    for name, store_dir in stage1.items():
        assert tier1[name] == store_dir


# ------------------------------------------- the EBPR family is complete in the "all" configs

EBPR_ENGINE_VARIANTS = {"BPR", "UBPR", "EBPR", "pUEBPR", "UEBPR"}


@pytest.mark.parametrize("path", ["configs/full.yaml", "configs/stage1_full.yaml"])
def test_all_five_ebpr_engine_variants_are_configured(path):
    """external/ebpr/Code/engine_EBPR.py asserts config['model'] is one of five variants.
    The "run everything" configs should exercise all five, not the three that were
    originally wired (README's model table already claimed BPR was Tier 1)."""
    config = _load(path)
    variants = {m.variant for m in config.models if m.adapter == "ebpr"}
    assert variants == EBPR_ENGINE_VARIANTS


@pytest.mark.parametrize("path", ["configs/full.yaml", "configs/stage1_full.yaml"])
def test_ebpr_bpr_does_not_collide_with_recbole_bpr(path):
    """Two different codebases both implement BPR; they must land in results.parquet under
    distinct `model` names or their rows merge silently."""
    config = _load(path)
    names = [m.name for m in config.models]
    assert len(names) == len(set(names)), f"duplicate model names in {path}"
    by_name = {m.name: m for m in config.models}
    assert by_name["BPR_ebpr"].adapter == "ebpr"
    assert by_name["BPR"].adapter == "recbole"
