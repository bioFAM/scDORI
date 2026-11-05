import json
import importlib
import re
from pathlib import Path

import pytest

import scdori

API_DOCS = Path(__file__).resolve().parents[1] / "docs" / "api"
SUBPACKAGES = {"pp", "pl", "_core"}
NOTEBOOK_FUNCTIONS = {
    "exploration": [
        "apply_label_colors",
        "choose_example_topics",
        "example_groups",
        "example_tf",
        "linked_peak_table",
        "normalized_counts_for_plot",
        "read_gene_sets",
        "regions_for_link",
        "top_binding_tfs",
        "top_groups_for_topics",
        "top_targets",
        "top_topics_for_group",
        "top_topics_for_tf",
    ],
    "interpretation": ["compare_tf_rankings", "tf_importance_scores"],
    "tf_display": ["filter_tf_display", "topic_expression_support"],
}
SUBPACKAGE_NAMES = [
    "pl.plot_phase1_diagnostics",
    "pl.plot_phase1_overview",
    "pl.plot_tf_score_views",
    "pl.plot_topic_programs",
    "pl.topic_status_table",
    "pp.retain_requested_peaks",
]


def _autosummary_lines(page):
    lines = []
    for block in re.findall(r"\.\. autosummary::\n(.*?)```", page.read_text(), re.DOTALL):
        lines += [line.strip() for line in block.splitlines() if line.strip() and ":toctree:" not in line]
    return lines


@pytest.mark.parametrize(
    "module, name", [(module, name) for module, names in NOTEBOOK_FUNCTIONS.items() for name in names]
)
def test_notebook_functions_are_public_and_documented(module, name):
    public = getattr(scdori, name)
    assert public is getattr(importlib.import_module(f"scdori._core.{module}"), name)
    assert public.__doc__


@pytest.mark.parametrize("page", ["index.md", "core.md", "plotting.md", "preprocessing.md"])
def test_api_page_lists_one_name_per_line_without_repeats(page):
    lines = _autosummary_lines(API_DOCS / page)
    assert lines
    assert all(len(line.split()) == 1 for line in lines)
    if page != "index.md":
        assert len(lines) == len(set(lines))


@pytest.mark.parametrize("page", ["index.md", None])
def test_every_public_name_is_on_the_api_pages(page):
    pages = [API_DOCS / page] if page else [API_DOCS / name for name in ("core.md", "plotting.md", "preprocessing.md")]
    listed = {line for path in pages for line in _autosummary_lines(path)}
    assert set(scdori.__all__) - SUBPACKAGES - listed == set()
    assert set(SUBPACKAGE_NAMES) - listed == set()


@pytest.mark.parametrize("name", SUBPACKAGE_NAMES)
def test_documented_subpackage_names_resolve(name):
    package, attribute = name.split(".")
    assert callable(getattr(getattr(scdori, package), attribute))


def test_release_notebooks_use_only_public_imports():
    """A notebook is what users copy, so it must not depend on private modules."""
    folder = Path(__file__).parents[1] / "docs/notebooks"
    notebooks = sorted(folder.glob("*.ipynb"))
    assert [p.stem for p in notebooks] == ["downstream", "preprocessing", "training"]
    for path in notebooks:
        for cell in json.loads(path.read_text())["cells"]:
            if cell["cell_type"] == "code":
                assert "_core" not in "".join(cell["source"]), (path.name, cell["id"])
