# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for semantic result artifact loading."""

import json

import numpy as np
import pandas as pd
import pytest

from hddflyzer.results import (
    KIND_DESCRIPTOR_TABLE,
    KIND_METADATA,
    KIND_MOLECULE_REGISTRY,
    KIND_PROJECTION_COORDINATES,
    KIND_TANIMOTO_MATRIX,
    KIND_WORKFLOW_SUMMARY,
    LoadedArtifact,
    ResultArtifact,
    load_artifact,
)


def _artifact(path, kind, metadata=None):
    return ResultArtifact(
        path=path,
        relative_path=path.name,
        category="test",
        kind=kind,
        operation="test.operation",
        metadata=metadata or {"source": "manifest"},
    )


def test_load_descriptor_table_as_dataframe(tmp_path):
    path = tmp_path / "features.csv"
    pd.DataFrame({"identifier": ["A1"], "QED": [0.8]}).to_csv(path, index=False)

    loaded = load_artifact(_artifact(path, KIND_DESCRIPTOR_TABLE))

    assert isinstance(loaded, LoadedArtifact)
    assert loaded.data["identifier"].tolist() == ["A1"]
    assert loaded.data["QED"].tolist() == [0.8]
    assert loaded.metadata == {"source": "manifest"}


def test_load_descriptor_table_rejects_pickle_by_default(tmp_path):
    path = tmp_path / "features.pkl"
    pd.DataFrame({"identifier": ["A1"], "QED": [0.8]}).to_pickle(path)

    with pytest.raises(ValueError, match="Pickle artifact loading is disabled"):
        load_artifact(_artifact(path, KIND_DESCRIPTOR_TABLE))


def test_load_descriptor_table_allows_pickle_with_explicit_opt_in(tmp_path):
    path = tmp_path / "features.pkl"
    pd.DataFrame({"identifier": ["A1"], "QED": [0.8]}).to_pickle(path)

    loaded = load_artifact(
        _artifact(path, KIND_DESCRIPTOR_TABLE),
        allow_pickle=True,
    )

    assert loaded.data["identifier"].tolist() == ["A1"]


def test_load_descriptor_table_rejects_empty_table(tmp_path):
    path = tmp_path / "features.csv"
    pd.DataFrame(columns=["identifier", "QED"]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="descriptor_table artifact"):
        load_artifact(_artifact(path, KIND_DESCRIPTOR_TABLE))


def test_load_projection_coordinates_as_dataframe(tmp_path):
    path = tmp_path / "pca_coordinates.csv"
    pd.DataFrame({"identifier": ["A1"], "PC1": [0.1], "PC2": [0.2]}).to_csv(
        path, index=False)

    loaded = load_artifact(_artifact(path, KIND_PROJECTION_COORDINATES))

    assert loaded.data["PC1"].tolist() == [0.1]
    assert loaded.artifact.kind == KIND_PROJECTION_COORDINATES


def test_load_projection_coordinates_rejects_missing_numeric_coordinates(tmp_path):
    path = tmp_path / "pca_coordinates.csv"
    pd.DataFrame({
        "identifier": ["A1"],
        "label": ["not_numeric"],
    }).to_csv(path, index=False)

    with pytest.raises(ValueError, match="two numeric columns"):
        load_artifact(_artifact(path, KIND_PROJECTION_COORDINATES))


def test_load_molecule_registry_as_dataframe(tmp_path):
    path = tmp_path / "molecules.csv"
    pd.DataFrame({
        "identifier": ["A1"],
        "canonical_smiles": ["CCO"],
    }).to_csv(path, index=False)

    loaded = load_artifact(_artifact(path, KIND_MOLECULE_REGISTRY))

    assert loaded.data["canonical_smiles"].tolist() == ["CCO"]


def test_load_molecule_registry_rejects_missing_id_or_smiles(tmp_path):
    path = tmp_path / "molecules.csv"
    pd.DataFrame({"name": ["compound"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="identifier or SMILES column"):
        load_artifact(_artifact(path, KIND_MOLECULE_REGISTRY))


def test_load_metadata_json_as_dict(tmp_path):
    path = tmp_path / "features_metadata.json"
    path.write_text(json.dumps({"n_features": 10}), encoding="utf-8")

    loaded = load_artifact(_artifact(path, KIND_METADATA))

    assert loaded.data == {"n_features": 10}


def test_load_workflow_summary_as_string(tmp_path):
    path = tmp_path / "workflow_summary.md"
    path.write_text("# Summary\n\nregistry -> chem\n", encoding="utf-8")

    loaded = load_artifact(_artifact(path, KIND_WORKFLOW_SUMMARY))

    assert isinstance(loaded.data, str)
    assert "registry -> chem" in loaded.data


def test_load_artifact_raises_for_missing_file(tmp_path):
    path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_artifact(_artifact(path, KIND_DESCRIPTOR_TABLE))


def test_load_tanimoto_matrix_as_array_with_ids_metadata(tmp_path):
    path = tmp_path / "tanimoto_matrix.npz"
    np.savez_compressed(
        path,
        matrix=np.array([[1.0, 0.25], [0.25, 1.0]], dtype=np.float32),
    )
    pd.DataFrame({"id": ["A1", "A2"]}).to_csv(
        tmp_path / "tanimoto_ids.csv", index=False)

    loaded = load_artifact(_artifact(path, KIND_TANIMOTO_MATRIX))

    assert isinstance(loaded.data, np.ndarray)
    assert loaded.data.shape == (2, 2)
    assert loaded.data.tolist() == [[1.0, 0.25], [0.25, 1.0]]
    assert loaded.metadata["ids"] == ["A1", "A2"]
    assert loaded.metadata["matrix_shape"] == [2, 2]
    assert loaded.metadata["ids_path"].endswith("tanimoto_ids.csv")


def test_load_tanimoto_matrix_disables_numpy_pickle(monkeypatch, tmp_path):
    path = tmp_path / "tanimoto_matrix.npz"
    np.savez_compressed(
        path,
        matrix=np.array([[1.0]], dtype=np.float32),
    )
    pd.DataFrame({"id": ["A1"]}).to_csv(
        tmp_path / "tanimoto_ids.csv", index=False)
    real_load = np.load
    seen = {}

    def checked_load(*args, **kwargs):
        seen["allow_pickle"] = kwargs.get("allow_pickle")
        return real_load(*args, **kwargs)

    monkeypatch.setattr("hddflyzer.results.artifacts.np.load", checked_load)

    load_artifact(_artifact(path, KIND_TANIMOTO_MATRIX))

    assert seen["allow_pickle"] is False


def test_load_tanimoto_matrix_rejects_invalid_archive(tmp_path):
    path = tmp_path / "tanimoto_matrix.npz"
    np.savez_compressed(path, values=np.array([1.0], dtype=np.float32))

    with pytest.raises(ValueError, match="missing 'matrix'"):
        load_artifact(_artifact(path, KIND_TANIMOTO_MATRIX))


def test_load_tanimoto_matrix_rejects_id_mismatch(tmp_path):
    path = tmp_path / "tanimoto_matrix.npz"
    np.savez_compressed(path, matrix=np.eye(2, dtype=np.float32))
    pd.DataFrame({"id": ["A1"]}).to_csv(
        tmp_path / "tanimoto_ids.csv", index=False)

    with pytest.raises(ValueError, match="size mismatch"):
        load_artifact(_artifact(path, KIND_TANIMOTO_MATRIX))


def test_load_artifact_raises_for_invalid_metadata_json(tmp_path):
    path = tmp_path / "bad_metadata.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid metadata JSON"):
        load_artifact(_artifact(path, KIND_METADATA))
