import json
from pathlib import Path

import pytest

from scdori.configuration import check_notebook_inputs, load_notebook_config, save_notebook_config


def test_shared_file_discovery_paths_and_independent_objects(tmp_path, monkeypatch):
    monkeypatch.delenv("SCDORI_CONFIG_FILE", raising=False)
    source = tmp_path / "scdori_config.json"
    source.write_text(
        json.dumps(
            {
                "preprocessing": {"data_dir": "my_data", "output_subdir_name": "prepared", "genes_user": ["GeneA"]},
                "training": {"num_topics": 12},
                "downstream": {"geneset_file": "sets.tsv", "geneset_name_contains": "ORGANOGENESIS"},
            }
        )
    )
    nested = tmp_path / "docs/notebooks"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    settings = load_notebook_config()
    assert settings.source == source
    assert settings.training.data_dir == tmp_path / "my_data"
    assert settings.training.output_subdir == "prepared"
    assert settings.downstream.geneset_file == str(tmp_path / "sets.tsv")
    assert settings.downstream.geneset_name_contains == "ORGANOGENESIS"
    settings.preprocessing.genes_user.append("Other")
    assert load_notebook_config().preprocessing.genes_user == ["GeneA"]


def test_legacy_config_fallback_preserves_user_settings(tmp_path, monkeypatch):
    from scdori._core.config import trainConfig
    from scdori.pp.config import ppConfig

    monkeypatch.delenv("SCDORI_CONFIG_FILE", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(trainConfig, "data_dir", Path("existing_data"))
    monkeypatch.setattr(trainConfig, "num_topics", 23)
    before = list(ppConfig.genes_user)
    settings = load_notebook_config()
    assert settings.source is None
    assert settings.training.num_topics == 23
    assert settings.training.data_dir == tmp_path / "existing_data"
    settings.preprocessing.genes_user.append("Other")
    assert ppConfig.genes_user == before


@pytest.mark.parametrize(
    "contents",
    [
        {"training": {"num_topic": 12}},
        {"training": {"freeze_phase1_in_grn": "false"}},
        {"training": {"num_topics": True}},
        {"unknown_section": {}},
    ],
)
def test_config_rejects_typos_and_wrong_types(tmp_path, contents):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(contents))
    with pytest.raises(ValueError):
        load_notebook_config(path)


def test_preflight_lists_missing_files_and_checks_checkpoint_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"training": {"data_dir": "data", "weights_folder_grn": "phase2"}}))
    settings = load_notebook_config(path)
    with pytest.raises(FileNotFoundError, match="rna_processed.h5ad"):
        check_notebook_inputs(settings, "training")
    data = tmp_path / "data/generated"
    data.mkdir(parents=True)
    for name in [
        "rna_processed.h5ad",
        "atac_processed.h5ad",
        "gene_peak_distance_exp.npy",
        "insilico_chipseq_act.npy",
        "insilico_chipseq_rep.npy",
    ]:
        (data / name).touch()  # preflight checks paths only; loaders check contents
    check_notebook_inputs(settings, "training")
    save_notebook_config(settings, "training", tmp_path / "phase2")
    (tmp_path / "phase2/best_scdori_best_eval.pth").touch()
    (tmp_path / "phase2/grn_topics.json").write_text("{}")  # path preflight; loader validates contents
    check_notebook_inputs(settings, "downstream")
    # A saved older/all-topic fit takes precedence over newly changed training defaults.
    snapshot_path = tmp_path / "phase2/notebook_config.json"
    saved = json.loads(snapshot_path.read_text())
    saved["training"].pop("grn_topic_selection")
    snapshot_path.write_text(json.dumps(saved))
    (tmp_path / "phase2/grn_topics.json").unlink()
    assert settings.training.grn_topic_selection == "live"
    check_notebook_inputs(settings, "downstream")
    settings.training.num_topics += 1
    with pytest.raises(ValueError, match="differ from the trained model"):
        check_notebook_inputs(settings, "downstream")


def test_explicit_missing_config_never_silently_uses_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("SCDORI_CONFIG_FILE", str(tmp_path / "missing.json"))
    with pytest.raises(FileNotFoundError):
        load_notebook_config()


def test_distributed_example_accepts_legacy_class_attributes():
    example = Path(__file__).resolve().parents[1] / "configs/scdori.example.json"
    settings = load_notebook_config(example)
    assert settings.preprocessing.motif_database == "cisbp"
    assert settings.preprocessing.genes_user == []
    assert settings.training.freeze_phase1_in_grn is True
    assert settings.training.cache_frozen_grn is True
    assert settings.training.tf_expression_scaling == "mean_within_across"
    assert settings.training.tf_expression_clamp == 0.2


def test_expression_policy_roundtrip_and_preflight_before_loading_data(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"training": {"tf_expression_scaling": "mean_within_across"}}))
    settings = load_notebook_config(path)
    save_notebook_config(settings, "training", tmp_path / "run")
    saved = json.loads((tmp_path / "run/notebook_config.json").read_text())
    assert saved["training"]["tf_expression_scaling"] == "mean_within_across"
    settings.training.tf_expression_scaling = "typo"
    with pytest.raises(ValueError, match="tf_expression_scaling"):
        check_notebook_inputs(settings, "training")
    settings.training.tf_expression_scaling = "within_topic"
    settings.training.tf_expression_clamp = 1.1
    with pytest.raises(ValueError, match="tf_expression_clamp"):
        check_notebook_inputs(settings, "training")


@pytest.mark.parametrize("training, expected", [
    ({}, ("mean_within_across", 0.2)),
    ({"tf_expression_scaling": "within_topic", "tf_expression_clamp": 0.1}, ("within_topic", 0.1)),
])
def test_expression_defaults_and_explicit_legacy_settings_survive_save(tmp_path, training, expected):
    path = tmp_path / "scdori_config.json"
    path.write_text(json.dumps({"training": training}))
    settings = load_notebook_config(path)
    assert (settings.training.tf_expression_scaling, settings.training.tf_expression_clamp) == expected
    save_notebook_config(settings, "training", tmp_path / "run")
    saved = json.loads((tmp_path / "run/notebook_config.json").read_text())["training"]
    assert (saved["tf_expression_scaling"], saved["tf_expression_clamp"]) == expected


def test_inspection_defaults_paths_and_checkpoint_feature_order(tmp_path):
    from scdori.configuration import check_training_input_order

    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"training": {"phase1_checkpoint": "run/phase1.pth", "phase1_split_file": "run/split.npz"}})
    )
    settings = load_notebook_config(path)
    assert settings.training.notebook_run_phase2 is True
    assert settings.training.run_phase1_diagnostics is True
    assert settings.training.grn_topic_selection == "live"
    assert settings.training.phase1_checkpoint == str(tmp_path / "run/phase1.pth")
    assert settings.training.phase1_split_file == str(tmp_path / "run/split.npz")
    (tmp_path / "run").mkdir()
    record = {"genes": ["a", "b"], "cells": ["c", "d"]}
    (tmp_path / "run/training_input_order.json").write_text(json.dumps(record))
    check_training_input_order(tmp_path / "run/phase1.pth", record)
    with pytest.raises(ValueError, match="genes"):
        check_training_input_order(tmp_path / "run/phase1.pth", {**record, "genes": ["b", "a"]})


def test_saved_config_records_thread_settings(tmp_path, monkeypatch):
    settings = load_notebook_config(None)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    save_notebook_config(settings, "preprocessing", tmp_path)
    threads = json.loads((tmp_path / "notebook_config.json").read_text())["threads"]
    assert threads["OMP_NUM_THREADS"] == "8" and threads["MKL_NUM_THREADS"] is None
    assert threads["cpu_count"] >= 1


def test_example_config_points_at_motif_files_that_ship_with_the_repository():
    root = Path(__file__).parents[1]
    settings = json.loads((root / "configs/scdori.example.json").read_text())["preprocessing"]
    for species in ("mouse", "human"):
        assert (root / settings["motif_directory"] / f"{settings['motif_database']}_{species}.meme").is_file()
