import gzip

from rexbench.models.pgpr_adapter import _count_kg_triples


def test_count_kg_triples_sums_tokens_across_relation_files(tmp_path):
    relations = tmp_path / "relations"
    relations.mkdir()
    (relations / "directed_by.txt").write_text("1 2 3\n\n4\n")
    (relations / "starring.txt").write_text("5 6\n7 8 9 10\n")
    # 3 + 0 + 1 (directed_by) + 2 + 4 (starring) = 10 tokens total
    assert _count_kg_triples(tmp_path) == 10


def test_count_kg_triples_prefers_either_txt_or_gz_without_double_counting(tmp_path):
    relations = tmp_path / "relations"
    relations.mkdir()
    (relations / "directed_by.txt").write_text("1 2 3\n")
    with gzip.open(relations / "directed_by.txt.gz", "wt") as f:
        f.write("1 2 3\n")
    # same relation shipped both plain and gzipped -- counted once, not twice
    assert _count_kg_triples(tmp_path) == 3


def test_count_kg_triples_reads_gz_only_relation_file(tmp_path):
    relations = tmp_path / "relations"
    relations.mkdir()
    with gzip.open(relations / "starring.txt.gz", "wt") as f:
        f.write("5 6 7\n")
    assert _count_kg_triples(tmp_path) == 3


def test_count_kg_triples_falls_back_to_kg_final_when_no_relations_dir(tmp_path):
    (tmp_path / "kg_final.txt").write_text("a\tb\tc\nd\te\tf\n")
    assert _count_kg_triples(tmp_path) == 2


def test_count_kg_triples_zero_when_nothing_present(tmp_path):
    assert _count_kg_triples(tmp_path) == 0
