import gzip

import pandas as pd
import pytest

from rexbench.models.pgpr_adapter import _materialize_pgpr_dataset_dir, _write_pgpr_split_file


def _df(rows):
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


def test_write_pgpr_split_file_translates_canonical_ids_to_raw_ids(tmp_path):
    df = _df([[0, 0, 5, 1000], [1, 2, 3, 1001]])
    user_id_map = {0: 10, 1: 20}
    item_id_map = {0: 100, 2: 300}

    _write_pgpr_split_file(tmp_path, "train", df, user_id_map, item_id_map)

    lines = (tmp_path / "train.txt").read_text().splitlines()
    assert lines == ["10 100 5 1000", "20 300 3 1001"]
    with gzip.open(tmp_path / "train.txt.gz", "rt") as f:
        assert f.read().splitlines() == lines


def test_write_pgpr_split_file_handles_empty_dataframe(tmp_path):
    df = _df([])
    _write_pgpr_split_file(tmp_path, "test", df, {}, {})
    assert (tmp_path / "test.txt").read_text() == ""
    with gzip.open(tmp_path / "test.txt.gz", "rt") as f:
        assert f.read() == ""


def test_materialize_pgpr_dataset_dir_symlinks_static_entries_but_writes_fresh_split(tmp_path):
    source_dir = tmp_path / "source"
    (source_dir / "entities").mkdir(parents=True)
    (source_dir / "entities" / "user.txt.gz").write_text("stub")
    (source_dir / "train.txt").write_text("stale data that must be overwritten")
    (source_dir / "test.txt").write_text("stale data that must be overwritten")

    runtime_dir = tmp_path / "runtime"
    train_df = _df([[0, 0, 5, 1000]])
    test_df = _df([[0, 1, 4, 1001]])
    user_id_map = {0: 1}
    item_id_map = {0: 10, 1: 11}

    _materialize_pgpr_dataset_dir(runtime_dir, source_dir, train_df, test_df, user_id_map, item_id_map)

    assert (runtime_dir / "entities").is_symlink()
    assert (runtime_dir / "entities").resolve() == (source_dir / "entities").resolve()
    assert not (runtime_dir / "train.txt").is_symlink()
    assert (runtime_dir / "train.txt").read_text().strip() == "1 10 5 1000"
    assert (runtime_dir / "test.txt").read_text().strip() == "1 11 4 1001"
    # the staged source's own train.txt/test.txt must be untouched
    assert source_dir.joinpath("train.txt").read_text() == "stale data that must be overwritten"


def test_materialize_pgpr_dataset_dir_refreshes_split_on_second_call(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    runtime_dir = tmp_path / "runtime"
    user_id_map = {0: 1}
    item_id_map = {0: 10}

    _materialize_pgpr_dataset_dir(
        runtime_dir, source_dir, _df([[0, 0, 5, 1000]]), _df([]), user_id_map, item_id_map,
    )
    assert (runtime_dir / "train.txt").read_text().strip() == "1 10 5 1000"

    # a later run against a different (e.g. re-sampled) dataset must overwrite, not append
    _materialize_pgpr_dataset_dir(
        runtime_dir, source_dir, _df([[0, 0, 3, 2000]]), _df([]), user_id_map, item_id_map,
    )
    assert (runtime_dir / "train.txt").read_text().strip() == "1 10 3 2000"
