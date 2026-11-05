import pandas as pd
import pytest

from scdori import read_gene_sets


@pytest.fixture
def gmt(tmp_path):
    path = tmp_path / "sets.gmt"
    path.write_text(
        "DESCARTES_ORGANOGENESIS_HEART\thttp://x\tMef2c\tTnnt2\tMef2c\n"
        "ZHANG_UTERUS_STROMA\thttp://y\tCol1a1\n"
        "Regulation of cell death WP1\t\tGata1\tTal1\n"
    )
    return path


def test_gmt_is_long_and_without_repeats(gmt):
    sets = read_gene_sets(gmt)
    assert list(sets.columns) == ["geneset", "genesymbol"]
    heart = sets[sets["geneset"] == "DESCARTES_ORGANOGENESIS_HEART"]
    assert heart["genesymbol"].tolist() == ["Mef2c", "Tnnt2"]


def test_gmt_set_names_keep_their_spaces(gmt):
    sets = read_gene_sets(gmt)
    assert sets[sets["geneset"] == "Regulation of cell death WP1"]["genesymbol"].tolist() == ["Gata1", "Tal1"]
    assert not sets["genesymbol"].isin(["cell", "death", "WP1", "http://x"]).any()


def test_name_filter_keeps_matching_sets(gmt):
    assert set(read_gene_sets(gmt, name_contains="ORGANOGENESIS")["geneset"]) == {"DESCARTES_ORGANOGENESIS_HEART"}


def test_name_filter_is_plain_text_not_a_pattern(gmt):
    with pytest.raises(ValueError, match="No gene set"):
        read_gene_sets(gmt, name_contains="DESCARTES.*")


def test_name_filter_without_a_match_is_an_error(gmt):
    with pytest.raises(ValueError, match="No gene set"):
        read_gene_sets(gmt, name_contains="HEPATOCYTE")


def test_table_input_keeps_two_columns(tmp_path):
    path = tmp_path / "sets.tsv"
    pd.DataFrame(
        {"geneset": ["a", "a", "b"], "genesymbol": ["x", "x", "y"], "collection": ["c", "c", "c"]}
    ).to_csv(path, sep="\t", index=False)
    sets = read_gene_sets(path)
    assert sets.to_dict("list") == {"geneset": ["a", "b"], "genesymbol": ["x", "y"]}


def test_table_without_the_columns_is_an_error(tmp_path):
    path = tmp_path / "sets.tsv"
    path.write_text("source\ttarget\na\tx\n")
    with pytest.raises(ValueError, match="geneset and genesymbol"):
        read_gene_sets(path)
