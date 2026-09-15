import numpy as np
import pandas as pd
import pytest

from rexbench.stats.significance import complete_cases, friedman_test, kendall_w, quade_test


def _balanced_pivot() -> pd.DataFrame:
    # 5 datasets x 3 models, every cell populated, values chosen so the models clearly
    # differ (avoids a degenerate all-equal-ranks edge case in quade_test's a2 - b term).
    rng = np.random.default_rng(0)
    data = {
        "ModelA": rng.normal(0.30, 0.01, 5),
        "ModelB": rng.normal(0.25, 0.01, 5),
        "ModelC": rng.normal(0.20, 0.01, 5),
    }
    return pd.DataFrame(data, index=[f"ds{i}" for i in range(5)])


def _ragged_pivot() -> pd.DataFrame:
    # Same as _balanced_pivot but with a PGPR-like column only populated for 2 of 5
    # datasets -- this is exactly the shape that used to crash friedman_test.
    pivot = _balanced_pivot()
    pivot["PGPR"] = [0.5, 0.4, np.nan, np.nan, np.nan]
    return pivot


def test_complete_cases_drops_rows_with_any_missing_model():
    pivot = _ragged_pivot()
    result = complete_cases(pivot)
    assert list(result.index) == ["ds0", "ds1"]
    assert not result.isna().any().any()


def test_complete_cases_noop_on_fully_populated_pivot():
    pivot = _balanced_pivot()
    result = complete_cases(pivot)
    pd.testing.assert_frame_equal(result, pivot)


def test_friedman_test_runs_on_balanced_pivot():
    result = friedman_test(_balanced_pivot())
    assert not np.isnan(result["statistic"])
    assert not np.isnan(result["p_value"])


def test_friedman_test_does_not_crash_on_ragged_pivot():
    # Regression test: independently dropna-ing each column used to produce
    # differently-sized groups here, raising "Array shapes are incompatible for
    # broadcasting" from scipy.stats.friedmanchisquare.
    result = friedman_test(_ragged_pivot())
    assert not np.isnan(result["statistic"])
    assert not np.isnan(result["p_value"])


def test_friedman_test_reports_insufficient_data_when_too_few_complete_rows():
    pivot = _balanced_pivot()
    pivot["PGPR"] = [0.5, np.nan, np.nan, np.nan, np.nan]  # only 1 complete row left
    result = friedman_test(pivot)
    assert np.isnan(result["statistic"])
    assert np.isnan(result["p_value"])
    assert result["significant"] is False


def test_quade_test_does_not_silently_return_nan_on_ragged_pivot():
    # Regression test: quade_test used to operate on the raw (ragged) pivot with no
    # complete-case filtering, letting NaN propagate through its own arithmetic into a
    # silent nan/nan/nan result instead of either failing loudly or excluding the
    # incomplete rows.
    result = quade_test(_ragged_pivot())
    assert not np.isnan(result["statistic"])
    assert not np.isnan(result["p_value"])


def test_kendall_w_does_not_crash_on_ragged_pivot():
    w = kendall_w(_ragged_pivot())
    assert not np.isnan(w)


def test_kendall_w_nan_when_too_few_complete_rows():
    pivot = _balanced_pivot()
    pivot["PGPR"] = [0.5, np.nan, np.nan, np.nan, np.nan]
    assert np.isnan(kendall_w(pivot))
