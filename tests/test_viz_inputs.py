# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for visualization input resolution from WorkflowRun."""

import pytest
import pandas as pd

from hddflyzer.config import settings
from hddflyzer.results import (
    KIND_DESCRIPTOR_TABLE,
    KIND_TANIMOTO_MATRIX,
    LoadedArtifact,
    ResultArtifact,
    WorkflowRun,
    load_artifact,
)
from hddflyzer.viz.correlations import HDDF_PAIRS, plot_hddf_scatters
from hddflyzer.viz.inputs import VizInputs, resolve_viz_inputs


def _workflow_run(tmp_path, outputs=None, categories=None):
    root = tmp_path / "results" / "aocd"
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "tag": "aocd",
        "operations": [],
        "current_outputs": outputs or [],
        "output_categories": categories or {},
        "workflow_contract": {
            "collection": {"tag": "aocd"},
            "stages": [],
        },
    }
    return WorkflowRun(
        tag="aocd",
        manifest_path=root / "manifest.json",
        manifest=manifest,
    )


def _write_hddf_features(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted(set(col for pair in HDDF_PAIRS for col in pair))
    data = {"identifier": ["A1", "A2", "A3"]}
    for idx, column in enumerate(columns):
        data[column] = [idx + 1.0, idx + 2.0, idx + 4.0]
    pd.DataFrame(data).to_csv(path, index=False)


def test_resolve_viz_inputs_returns_existing_category_paths(tmp_path):
    path = tmp_path / "results" / "aocd" / "dimred" / "pca"
    path.mkdir(parents=True)
    coords = path / "pca_coordinates.csv"
    coords.write_text("identifier,PC1,PC2\nA1,0.1,0.2\n", encoding="utf-8")
    run = _workflow_run(
        tmp_path,
        outputs=["dimred/pca/pca_coordinates.csv"],
        categories={"dimred": ["dimred/pca/pca_coordinates.csv"]},
    )

    inputs = resolve_viz_inputs(run, category="dimred", required="pca")

    assert isinstance(inputs, VizInputs)
    assert inputs.category == "dimred"
    assert inputs.root == tmp_path / "results" / "aocd"
    assert inputs.paths == (coords,)
    assert inputs.as_dict()["paths"] == [str(coords)]


def test_resolve_viz_inputs_rejects_empty_category(tmp_path):
    run = _workflow_run(tmp_path, categories={"chem": []})

    with pytest.raises(ValueError, match="No registered outputs"):
        resolve_viz_inputs(run, category="chem")


def test_resolve_viz_inputs_rejects_registered_missing_file(tmp_path):
    run = _workflow_run(
        tmp_path,
        outputs=["dimred/umap/umap_coordinates.csv"],
        categories={"dimred": ["dimred/umap/umap_coordinates.csv"]},
    )

    with pytest.raises(FileNotFoundError, match="Registered visualization input"):
        resolve_viz_inputs(run, category="dimred")


def test_resolve_viz_inputs_rejects_absent_required_output(tmp_path):
    path = tmp_path / "results" / "aocd" / "chemistry" / "tanimoto"
    path.mkdir(parents=True)
    matrix = path / "tanimoto_matrix.npz"
    matrix.write_bytes(b"fake npz")
    run = _workflow_run(
        tmp_path,
        outputs=["chemistry/tanimoto/tanimoto_matrix.npz"],
        categories={"chem": ["chemistry/tanimoto/tanimoto_matrix.npz"]},
    )

    with pytest.raises(FileNotFoundError, match="No registered output matching"):
        resolve_viz_inputs(run, category="chem", required="reference_features")


def test_resolve_viz_inputs_resolves_descriptor_table_by_kind(tmp_path):
    features = tmp_path / "results" / "aocd" / "features" / "full" / "features.csv"
    _write_hddf_features(features)
    run = _workflow_run(
        tmp_path,
        outputs=["features/full/features.csv"],
        categories={"chem": ["features/full/features.csv"]},
    )

    inputs = resolve_viz_inputs(run, kind=KIND_DESCRIPTOR_TABLE)

    assert inputs.category == "chem"
    assert inputs.paths == (features,)


def test_resolve_viz_inputs_resolves_by_kind_and_category(tmp_path):
    features = tmp_path / "results" / "aocd" / "features" / "full" / "features.csv"
    _write_hddf_features(features)
    run = _workflow_run(
        tmp_path,
        outputs=["features/full/features.csv"],
        categories={"chem": ["features/full/features.csv"]},
    )

    inputs = resolve_viz_inputs(
        run,
        category="chem",
        kind=KIND_DESCRIPTOR_TABLE,
    )

    assert inputs.category == "chem"
    assert inputs.paths == (features,)


def test_resolve_viz_inputs_rejects_missing_kind(tmp_path):
    run = _workflow_run(
        tmp_path,
        outputs=["features/full/features.csv"],
        categories={"chem": ["features/full/features.csv"]},
    )

    with pytest.raises(ValueError, match="No registered artifacts"):
        resolve_viz_inputs(run, category="chem", kind=KIND_TANIMOTO_MATRIX)


def test_plot_hddf_scatters_keeps_tag_interface(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "RESULTS_DIR", str(tmp_path / "results"))
    features = (
        tmp_path / "results" / "aocd" / "features" / "full" / "features.csv"
    )
    _write_hddf_features(features)

    assert plot_hddf_scatters("aocd") is True
    assert (
        tmp_path
        / "results"
        / "aocd"
        / "figures"
        / "correlations"
        / "hddf_corr_scatters_trendline.png"
    ).exists()


def test_plot_hddf_scatters_accepts_viz_inputs(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "RESULTS_DIR", str(tmp_path / "results"))
    features = (
        tmp_path / "results" / "aocd" / "features" / "full" / "features.csv"
    )
    _write_hddf_features(features)
    run = _workflow_run(
        tmp_path,
        outputs=["features/full/features.csv"],
        categories={"chem": ["features/full/features.csv"]},
    )
    inputs = resolve_viz_inputs(
        run, category="chem", kind=KIND_DESCRIPTOR_TABLE)

    assert plot_hddf_scatters(inputs) is True
    assert (
        tmp_path
        / "results"
        / "aocd"
        / "figures"
        / "correlations"
        / "hddf_corr_scatters_trendline.png"
    ).exists()


def test_plot_hddf_scatters_accepts_loaded_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "RESULTS_DIR", str(tmp_path / "results"))
    features = (
        tmp_path / "results" / "aocd" / "features" / "full" / "features.csv"
    )
    _write_hddf_features(features)
    artifact = ResultArtifact(
        path=features,
        relative_path="features/full/features.csv",
        category="chem",
        kind=KIND_DESCRIPTOR_TABLE,
        operation="chem.features",
        metadata={"source": "test"},
    )
    loaded = load_artifact(artifact)

    assert plot_hddf_scatters(loaded) is True
    assert (
        tmp_path
        / "results"
        / "aocd"
        / "figures"
        / "correlations"
        / "hddf_corr_scatters_trendline.png"
    ).exists()


def test_plot_hddf_scatters_reports_incompatible_loaded_artifact(
    tmp_path,
    capsys,
):
    path = tmp_path / "results" / "aocd" / "chemistry" / "tanimoto"
    path.mkdir(parents=True)
    matrix = path / "tanimoto_matrix.npz"
    matrix.write_bytes(b"fake npz")
    loaded = LoadedArtifact(
        artifact=ResultArtifact(
            path=matrix,
            relative_path="chemistry/tanimoto/tanimoto_matrix.npz",
            category="chem",
            kind=KIND_TANIMOTO_MATRIX,
        ),
        data=object(),
        metadata={},
    )

    assert plot_hddf_scatters(loaded) is False
    assert "descriptor_table" in capsys.readouterr().out


def test_plot_hddf_scatters_reports_viz_inputs_without_features_csv(
    tmp_path,
    capsys,
):
    path = tmp_path / "results" / "aocd" / "dimred" / "pca"
    path.mkdir(parents=True)
    coords = path / "pca_coordinates.csv"
    coords.write_text("identifier,PC1,PC2\nA1,0.1,0.2\n", encoding="utf-8")
    inputs = VizInputs(
        category="dimred",
        root=tmp_path / "results" / "aocd",
        paths=(coords,),
    )

    assert plot_hddf_scatters(inputs) is False
    assert "VizInputs does not contain a features CSV" in capsys.readouterr().out
