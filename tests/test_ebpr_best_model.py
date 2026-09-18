"""EBPR must report the BEST epoch's model, not the last one.

The original code did `best_model = self.model`, storing a reference to the live model that
training keeps mutating in place — so "best" silently became "last", and the checkpoint
written under the best epoch's filename actually held the final epoch's weights.
"""
import sys

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

sys.path.insert(0, "external/ebpr")
from Code.engine_EBPR import Engine  # noqa: E402


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))


class _FakeEngine:
    save_implicit = Engine.save_implicit

    def __init__(self):
        self.model = _Tiny()
        self.config = {
            "model": "EBPR", "dataset": "d", "batch_size": 1, "optimizer": "adam",
            "lr": 0.1, "num_latent": 1, "l2_regularization": 0, "top_k": 10,
            "model_dir_implicit": "/dev/null",
        }


def _run(ndcgs):
    engine = _FakeEngine()
    best_model, best_performance = "", [0] * 8
    for epoch, ndcg in enumerate(ndcgs):
        with torch.no_grad():
            engine.model.w.fill_(float(epoch))  # weights change every epoch
        best_model, best_performance = engine.save_implicit(
            epoch, ndcg, 0, 0, 0, 0, 0, 0, len(ndcgs), best_model, best_performance,
            save_models=False,
        )
    return engine, best_model, best_performance


def test_best_model_is_a_snapshot_not_a_live_reference():
    engine, best_model, best_performance = _run([0.10, 0.90, 0.30, 0.20])
    assert best_performance[7] == 1, "epoch 1 had the highest NDCG"
    assert float(best_model.w.detach()) == 1.0, "best_model must hold epoch 1's weights"
    assert float(engine.model.w.detach()) == 3.0, "the live model is still at the last epoch"
    assert best_model is not engine.model


def test_best_model_tracks_a_later_improvement():
    _, best_model, best_performance = _run([0.5, 0.1, 0.1, 0.9])
    assert best_performance[7] == 3
    assert float(best_model.w.detach()) == 3.0


def test_best_model_keeps_first_epoch_when_nothing_improves():
    _, best_model, best_performance = _run([0.9, 0.5, 0.4, 0.3])
    assert best_performance[7] == 0
    assert float(best_model.w.detach()) == 0.0
