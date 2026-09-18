"""`eval_every`: how often EBPR runs the validation pass that selects the best model.

Engine.evaluate() is a full leave-one-out pass with 100 sampled negatives per user and
dominates EBPR's runtime, so this is the knob that makes a 50-epoch run affordable. The
schedule must never drop the final epoch, or the fully-trained model stops being a
candidate for "best" and save_implicit never writes its checkpoint.
"""
import pytest


def _evaluated_epochs(num_epoch, eval_every):
    """Mirrors the guard in EBPRModelAdapter.fit()'s epoch loop."""
    last = num_epoch - 1
    return [e for e in range(num_epoch) if not (e % eval_every) or e == last]


def test_eval_every_1_evaluates_every_epoch():
    assert _evaluated_epochs(50, 1) == list(range(50))


@pytest.mark.parametrize("eval_every", [1, 2, 3, 5, 7, 10, 50, 99])
def test_final_epoch_is_always_evaluated(eval_every):
    for num_epoch in (1, 2, 3, 10, 50):
        evaluated = _evaluated_epochs(num_epoch, eval_every)
        assert num_epoch - 1 in evaluated, (num_epoch, eval_every)


@pytest.mark.parametrize("eval_every", [1, 2, 5, 10])
def test_first_epoch_is_always_evaluated(eval_every):
    assert 0 in _evaluated_epochs(50, eval_every)


def test_eval_every_5_over_50_epochs_cuts_passes_from_50_to_11():
    evaluated = _evaluated_epochs(50, 5)
    assert evaluated == [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
    assert len(evaluated) == 11


def test_single_epoch_run_still_evaluates_once():
    assert _evaluated_epochs(1, 10) == [0]


def test_evaluated_epochs_never_exceed_num_epoch():
    for num_epoch in (1, 5, 50):
        for eval_every in (1, 3, 7):
            assert all(0 <= e < num_epoch for e in _evaluated_epochs(num_epoch, eval_every))


def test_adapter_default_preserves_original_behaviour():
    """Unless a config opts in, eval_every must be 1 so existing results are unchanged."""
    from rexbench.config.schema import ModelConfig
    from rexbench.models.ebpr_adapter import EBPRModelAdapter

    cfg = ModelConfig(name="EBPR", tier=1, adapter="ebpr", variant="EBPR")
    adapter = EBPRModelAdapter(cfg)

    class _Bundle:
        name, num_users, num_items, topk = "d", 10, 10, [10]

    assert adapter._build_config(_Bundle(), seed=1, device="cpu")["eval_every"] == 1


def test_adapter_reads_eval_every_from_hyperparameters():
    from rexbench.config.schema import ModelConfig
    from rexbench.models.ebpr_adapter import EBPRModelAdapter

    cfg = ModelConfig(name="EBPR", tier=1, adapter="ebpr", variant="EBPR",
                      hyperparameters={"eval_every": 5})
    adapter = EBPRModelAdapter(cfg)

    class _Bundle:
        name, num_users, num_items, topk = "d", 10, 10, [10]

    assert adapter._build_config(_Bundle(), seed=1, device="cpu")["eval_every"] == 5


def test_eval_every_is_floored_at_one():
    from rexbench.config.schema import ModelConfig
    from rexbench.models.ebpr_adapter import EBPRModelAdapter

    class _Bundle:
        name, num_users, num_items, topk = "d", 10, 10, [10]

    for bad in (0, -3):
        cfg = ModelConfig(name="EBPR", tier=1, adapter="ebpr", variant="EBPR",
                          hyperparameters={"eval_every": bad})
        built = EBPRModelAdapter(cfg)._build_config(_Bundle(), seed=1, device="cpu")
        assert built["eval_every"] == 1, bad
