# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for the hddflyzer.io persistence layer."""

import json

import numpy as np
import pandas as pd
import pytest

from hddflyzer.config import settings
from hddflyzer.io import (
    align_tanimoto_with_npclassifier,
    load_df,
    load_features_table,
    load_npclassifier_success,
    load_selected_features,
    load_tanimoto,
    resolve_features_csv,
    update_manifest,
)


@pytest.fixture
def results_root(monkeypatch, tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    monkeypatch.setattr(settings, "RESULTS_DIR", str(root))
    return root


def _normalized_paths(paths):
    return [path.replace("\\", "/") for path in paths]


def _workflow_stages(manifest):
    return {
        stage["stage"]: stage
        for stage in manifest["workflow_contract"]["stages"]
    }


def test_load_features_table_uses_dataset_first_path(results_root):
    out_dir = results_root / "aocd" / "features" / "full"
    out_dir.mkdir(parents=True)
    path = out_dir / "features.csv"
    pd.DataFrame({"identifier": ["A1"], "MW": [100.0]}).to_csv(path, index=False)

    df, loaded_path = load_features_table("aocd")

    assert loaded_path == str(path)
    assert df["identifier"].tolist() == ["A1"]


def test_load_df_rejects_pickle_by_default(tmp_path):
    path = tmp_path / "features.pkl"
    pd.DataFrame({"identifier": ["A1"], "MW": [100.0]}).to_pickle(path)

    with pytest.raises(ValueError, match="Pickle loading is disabled"):
        load_df(str(path))


def test_load_df_allows_pickle_with_explicit_opt_in(tmp_path):
    path = tmp_path / "features.pkl"
    pd.DataFrame({"identifier": ["A1"], "MW": [100.0]}).to_pickle(path)

    df = load_df(str(path), allow_pickle=True)

    assert df["identifier"].tolist() == ["A1"]


def test_resolve_features_csv_returns_sorted_first_match(results_root):
    out_dir = results_root / "aocd" / "features" / "full"
    out_dir.mkdir(parents=True)
    b = out_dir / "features_b.csv"
    a = out_dir / "features.csv"
    b.write_text("identifier,MW\nB,2\n", encoding="utf-8")
    a.write_text("identifier,MW\nA,1\n", encoding="utf-8")

    assert resolve_features_csv("aocd") == str(a)


def test_load_selected_features_skips_total_header(results_root):
    out_dir = results_root / "aocd" / "features" / "pruning"
    out_dir.mkdir(parents=True)
    (out_dir / "selected_features.txt").write_text(
        "Selected features\nTotal: 2\nMW\nTPSA\nMW\n",
        encoding="utf-8",
    )

    assert load_selected_features("aocd", excluded=["TPSA"]) == ["MW"]


def test_load_tanimoto_reads_matrix_and_ids(results_root):
    out_dir = results_root / "aocd" / "chemistry" / "tanimoto"
    out_dir.mkdir(parents=True)
    np.savez_compressed(
        out_dir / "tanimoto_matrix.npz",
        matrix=np.array([[1.0, 0.2], [0.2, 1.0]], dtype=np.float32),
    )
    pd.DataFrame({"id": ["A1", "A2"]}).to_csv(
        out_dir / "tanimoto_ids.csv", index=False)

    sim, ids, mat_path, ids_path = load_tanimoto("aocd")

    assert ids == ["A1", "A2"]
    assert sim.shape == (2, 2)
    assert mat_path.endswith("tanimoto_matrix.npz")
    assert ids_path.endswith("tanimoto_ids.csv")


def test_load_tanimoto_disables_numpy_pickle(monkeypatch, results_root):
    out_dir = results_root / "aocd" / "chemistry" / "tanimoto"
    out_dir.mkdir(parents=True)
    np.savez_compressed(
        out_dir / "tanimoto_matrix.npz",
        matrix=np.array([[1.0]], dtype=np.float32),
    )
    pd.DataFrame({"id": ["A1"]}).to_csv(
        out_dir / "tanimoto_ids.csv", index=False)
    real_load = np.load
    seen = {}

    def checked_load(*args, **kwargs):
        seen["allow_pickle"] = kwargs.get("allow_pickle")
        return real_load(*args, **kwargs)

    monkeypatch.setattr("hddflyzer.io.readers.np.load", checked_load)

    load_tanimoto("aocd")

    assert seen["allow_pickle"] is False


def test_npclassifier_loader_filters_success_and_alignment(results_root):
    out_dir = results_root / "aocd" / "annotations" / "npclassifier"
    out_dir.mkdir(parents=True)
    pd.DataFrame({
        "identifier": ["A1", "A2", "A3"],
        "Status": ["Success", "Failed", "Success"],
        "Pathway": ["P1", "", "P3"],
    }).to_csv(out_dir / "npclassifier.csv", index=False)

    npc, path = load_npclassifier_success("aocd")
    sim = np.eye(3, dtype=np.float32)
    sim_sub, ids_sub, npc_aln = align_tanimoto_with_npclassifier(
        sim, ["A1", "A2", "A3"], npc)

    assert path.endswith("npclassifier.csv")
    assert npc["identifier"].tolist() == ["A1", "A3"]
    assert ids_sub == ["A1", "A3"]
    assert sim_sub.shape == (2, 2)
    assert npc_aln["Pathway"].tolist() == ["P1", "P3"]


def test_update_manifest_appends_relative_files(results_root):
    output = results_root / "aocd" / "registry" / "molecules.csv"
    output.parent.mkdir(parents=True)
    output.write_text("identifier,SMILES\nA1,CCO\n", encoding="utf-8")

    manifest_path = update_manifest(
        "aocd",
        "data.prepare",
        [str(output)],
        metadata={"n": 1},
    )

    manifest = json.loads((results_root / "aocd" / "manifest.json").read_text())
    files = _normalized_paths(manifest["operations"][0]["files"])
    current_outputs = _normalized_paths(manifest["current_outputs"])

    assert manifest_path.endswith("manifest.json")
    assert manifest["tag"] == "aocd"
    assert manifest["operations"][0]["operation"] == "data.prepare"
    assert files == ["registry/molecules.csv"]
    assert manifest["operations"][0]["metadata"] == {"n": 1}
    assert current_outputs == [
        "registry/molecules.csv",
        "workflow_summary.md",
    ]
    assert manifest["stale_operation_files"] == []
    assert manifest["workflow_contract"]["collection"] == {"tag": "aocd"}
    assert manifest["workflow_contract"]["canonical_workflow"] == [
        "registry",
        "chem",
        "dimred",
        "viz",
        "metadata/results",
    ]
    assert manifest["workflow_contract"]["parameters_index"] == (
        "operations[].metadata")
    assert manifest["workflow_contract"]["outputs_index"] == "current_outputs"
    assert manifest["output_categories"] == {
        "registry/data": ["registry/molecules.csv"],
        "metadata": ["workflow_summary.md"],
    }

    stages = _workflow_stages(manifest)
    assert stages["registry"]["ran"] is True
    assert stages["registry"]["operations"] == ["data.prepare"]
    assert stages["chem"]["ran"] is False

    summary = (results_root / "aocd" / "workflow_summary.md").read_text(
        encoding="utf-8")
    assert "# HDDFLYZER Workflow Summary: aocd" in summary
    assert "registry -> chem -> dimred -> viz -> metadata/results" in summary
    assert "- registry: ran; operations: data.prepare" in summary
    assert "- chem: not detected; operations: none" in summary
    assert "### registry/data" in summary
    assert "- `registry/molecules.csv`" in summary
    assert "### metadata" in summary
    assert "- `workflow_summary.md`" in summary
    assert "- `manifest.json`" in summary
    assert "Reconstruction details live in manifest operations" in summary


def test_update_manifest_rejects_external_files(results_root, tmp_path):
    external = tmp_path / "outside.csv"
    external.write_text("identifier,SMILES\nA1,CCO\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside results/aocd"):
        update_manifest(
            "aocd",
            "data.prepare",
            [str(external)],
            metadata={"n": 1},
        )


def test_update_manifest_rejects_path_like_tag(results_root):
    output = results_root / "aocd" / "registry" / "molecules.csv"
    output.parent.mkdir(parents=True)
    output.write_text("identifier,SMILES\nA1,CCO\n", encoding="utf-8")

    with pytest.raises(ValueError):
        update_manifest("ao/cd", "data.prepare", [str(output)])


def test_update_manifest_appends_multiple_operations(results_root):
    first = results_root / "aocd" / "registry" / "molecules.csv"
    second = results_root / "aocd" / "features" / "full" / "features.csv"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("identifier,SMILES\nA1,CCO\n", encoding="utf-8")
    second.write_text("identifier,MW\nA1,100\n", encoding="utf-8")

    update_manifest("aocd", "data.prepare", [str(first)], metadata={"n": 1})
    update_manifest("aocd", "chem.features", [str(second)], metadata={"n": 1})

    manifest = json.loads((results_root / "aocd" / "manifest.json").read_text())

    assert [op["operation"] for op in manifest["operations"]] == [
        "data.prepare",
        "chem.features",
    ]
    assert set(_normalized_paths(manifest["current_outputs"])) == {
        "registry/molecules.csv",
        "features/full/features.csv",
        "workflow_summary.md",
    }
    assert manifest["output_categories"] == {
        "registry/data": ["registry/molecules.csv"],
        "chem": ["features/full/features.csv"],
        "metadata": ["workflow_summary.md"],
    }

    stages = _workflow_stages(manifest)
    assert stages["registry"]["operations"] == ["data.prepare"]
    assert stages["chem"]["operations"] == ["chem.features"]

    summary = (results_root / "aocd" / "workflow_summary.md").read_text(
        encoding="utf-8")
    assert "- chem: ran; operations: chem.features" in summary
    assert "### chem" in summary
    assert "- `features/full/features.csv`" in summary
