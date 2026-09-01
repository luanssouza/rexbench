from rexbench.config.schema import ModelConfig, PreconditionConfig
from rexbench.core.dataset import InteractionStats
from rexbench.models.ebpr_adapter import EBPRModelAdapter


class _FakeBundle:
    def __init__(self, fraction_users_lt_2: float):
        self.name = "fake"
        self.interaction_stats = InteractionStats(
            mean_interactions_per_user=1.0, median_interactions_per_user=1.0,
            fraction_users_lt_2=fraction_users_lt_2, num_users=10, num_items=10,
            num_interactions=10,
        )


def _ebpr_adapter(min_fraction_users_with_2plus: float) -> EBPRModelAdapter:
    config = ModelConfig(
        name="EBPR", tier=1, adapter="ebpr", variant="EBPR",
        precondition=PreconditionConfig(min_fraction_users_with_2plus=min_fraction_users_with_2plus),
    )
    return EBPRModelAdapter(config)


def test_ebpr_precondition_fails_on_sparse_data():
    # 90% of users have <2 interactions -> only 10% have >=2, well below a 0.5 requirement
    adapter = _ebpr_adapter(min_fraction_users_with_2plus=0.5)
    report = adapter.check_preconditions(_FakeBundle(fraction_users_lt_2=0.9))
    assert not report.satisfied
    assert report.measurements["fraction_users_with_2plus_interactions"] == 0.1
    assert report.requirement == {"min_fraction_users_with_2plus": 0.5}
    assert report.reason is not None


def test_ebpr_precondition_passes_on_dense_data():
    adapter = _ebpr_adapter(min_fraction_users_with_2plus=0.5)
    report = adapter.check_preconditions(_FakeBundle(fraction_users_lt_2=0.1))
    assert report.satisfied
    assert report.reason is None


def test_ebpr_precondition_skipped_when_not_configured():
    adapter = _ebpr_adapter(min_fraction_users_with_2plus=None)
    report = adapter.check_preconditions(_FakeBundle(fraction_users_lt_2=0.99))
    assert report.satisfied
