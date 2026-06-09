# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for hddflyzer.results workflow reconstruction."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from hddflyzer.results import (
    KIND_DESCRIPTOR_TABLE,
    KIND_FIGURE,
    KIND_MOLECULE_REGISTRY,
    KIND_PROJECTION_COORDINATES,
    KIND_TANIMOTO_MATRIX,
    KIND_UNKNOWN,
    KIND_WORKFLOW_SUMMARY,
    LoadedArtifact,
    ResultArtifact,
    WorkflowRun,
    classify_artifact,
    load_workflow_run,
)
from hddflyzer.science import DescriptorSpace, ProjectionSpace, SimilaritySpace


def _write_manifest(tmp_path, manifest):
    run_dir = tmp_path / "results" / "aocd"
    run_dir.mkdir(parents=True)
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _manifest():
    return {
        "tag": "aocd",
        "operations": [
            {
                "operation": "data.prepare",
                "files": ["registry/molecules.csv"],
                "metadata": {"source_file": "examples/aocd.csv"},
            },
            {
                "operation": "chem.tanimoto",
                "files": [
                    "features/full/features.csv",
                    "chemistry/tanimoto/tanimoto_matrix.npz",
                    "chemistry/tanimoto/tanimoto_metadata.json",
                ],
                "metadata": {
                    "n_compounds": 2,
                    "fingerprint": {
                        "type": "Morgan",
                        "radius": 2,
                        "n_bits": 1024,
                    },
                },
            },
            {
                "operation": "dimred.pca",
                "files": ["dimred/pca/pca_coordinates.csv"],
                "metadata": {"analysis": "pca"},
            },
            {
                "operation": "viz.correlations",
                "files": ["figures/correlations/hddf_corr_scatters_trendline.png"],
                "metadata": {"plot": "hddf_scatters"},
            },
        ],
        "current_outputs": [
            "registry/molecules.csv",
            "features/full/features.csv",
            "chemistry/tanimoto/tanimoto_matrix.npz",
            "dimred/pca/pca_coordinates.csv",
            "figures/correlations/hddf_corr_scatters_trendline.png",
            "workflow_summary.md",
            "misc/readme.txt",
        ],
        "output_categories": {
            "registry/data": ["registry/molecules.csv"],
            "chem": [
                "features/full/features.csv",
                "chemistry/tanimoto/tanimoto_matrix.npz",
            ],
            "dimred": ["dimred/pca/pca_coordinates.csv"],
            "viz/figures": [
                "figures/correlations/hddf_corr_scatters_trendline.png"
            ],
            "metadata": ["workflow_summary.md"],
        },
        "workflow_contract": {
            "collection": {"tag": "aocd"},
            "canonical_workflow": [
                "registry",
                "chem",
                "dimred",
                "viz",
                "metadata/results",
            ],
            "stages": [
                {
                    "stage": "registry",
                    "ran": True,
                    "operations": ["data.prepare"],
                },
                {
                    "stage": "chem",
                    "ran": True,
                    "operations": ["chem.tanimoto"],
                },
                {
                    "stage": "dimred",
                    "ran": True,
                    "operations": ["dimred.pca"],
                },
                {
                    "stage": "viz",
                    "ran": True,
                    "operations": ["viz.correlations"],
                },
            ],
        },
    }


def test_load_workflow_run_reconstructs_manifest_view(tmp_path):
    manifest_path = _write_manifest(tmp_path, _manifest())

    run = load_workflow_run("aocd", results_dir=tmp_path / "results")

    assert isinstance(run, WorkflowRun)
    assert run.tag == "aocd"
    assert run.manifest_path == manifest_path
    assert run.operations[0]["operation"] == "data.prepare"
    assert run.current_outputs == [
        "registry/molecules.csv",
        "features/full/features.csv",
        "chemistry/tanimoto/tanimoto_matrix.npz",
        "dimred/pca/pca_coordinates.csv",
        "figures/correlations/hddf_corr_scatters_trendline.png",
        "workflow_summary.md",
        "misc/readme.txt",
    ]
    assert run.output_categories["chem"] == [
        "features/full/features.csv",
        "chemistry/tanimoto/tanimoto_matrix.npz"
    ]
    assert run.workflow_contract["collection"] == {"tag": "aocd"}


def test_workflow_run_filters_outputs_and_operations(tmp_path):
    _write_manifest(tmp_path, _manifest())
    run = load_workflow_run("aocd", results_dir=tmp_path / "results")

    assert run.outputs() == run.current_outputs
    assert run.outputs(category="chem") == [
        "features/full/features.csv",
        "chemistry/tanimoto/tanimoto_matrix.npz"
    ]
    assert run.outputs(category="missing") == []
    assert [
        op["operation"] for op in run.operations_by_stage("chem")
    ] == ["chem.tanimoto"]
    assert [
        op["operation"] for op in run.operations_by_stage("dimred")
    ] == ["dimred.pca"]


def test_workflow_run_returns_operation_metadata_and_summaries(tmp_path):
    _write_manifest(tmp_path, _manifest())
    run = load_workflow_run("aocd", results_dir=tmp_path / "results")

    metadata = run.operation_metadata("chem.tanimoto")
    assert metadata["fingerprint"]["type"] == "Morgan"
    assert run.operation_metadata("missing.operation") is None

    summary = run.summary()
    assert summary["tag"] == "aocd"
    assert summary["n_operations"] == 4
    assert summary["n_current_outputs"] == 7
    assert summary["output_categories"]["dimred"] == [
        "dimred/pca/pca_coordinates.csv"
    ]

    as_dict = run.to_dict()
    assert as_dict["tag"] == "aocd"
    assert as_dict["manifest"]["tag"] == "aocd"


def test_classify_artifact_recognizes_supported_result_kinds():
    assert classify_artifact("registry/molecules.csv") == KIND_MOLECULE_REGISTRY
    assert classify_artifact("features/full/features.csv") == KIND_DESCRIPTOR_TABLE
    assert (
        classify_artifact("chemistry/tanimoto/tanimoto_matrix.npz")
        == KIND_TANIMOTO_MATRIX
    )
    assert (
        classify_artifact("dimred/pca/pca_coordinates.csv", "dimred.pca")
        == KIND_PROJECTION_COORDINATES
    )
    assert (
        classify_artifact("figures/dimred/pca/pca_plot.png")
        == KIND_FIGURE
    )
    assert classify_artifact("workflow_summary.md") == KIND_WORKFLOW_SUMMARY
    assert classify_artifact("misc/readme.txt") == KIND_UNKNOWN


def test_workflow_run_returns_semantic_artifacts(tmp_path):
    manifest_path = _write_manifest(tmp_path, _manifest())
    run = load_workflow_run("aocd", results_dir=tmp_path / "results")

    artifacts = run.artifacts()

    assert all(isinstance(artifact, ResultArtifact) for artifact in artifacts)
    assert [artifact.kind for artifact in artifacts] == [
        KIND_MOLECULE_REGISTRY,
        KIND_DESCRIPTOR_TABLE,
        KIND_TANIMOTO_MATRIX,
        KIND_PROJECTION_COORDINATES,
        KIND_FIGURE,
        KIND_WORKFLOW_SUMMARY,
        KIND_UNKNOWN,
    ]
    tanimoto_artifact = run.artifacts(kind=KIND_TANIMOTO_MATRIX)[0]
    assert tanimoto_artifact.path == (
        manifest_path.parent / "chemistry/tanimoto/tanimoto_matrix.npz"
    )
    assert tanimoto_artifact.category == "chem"
    assert tanimoto_artifact.operation == "chem.tanimoto"
    assert tanimoto_artifact.metadata["fingerprint"]["type"] == "Morgan"


def test_workflow_run_filters_artifacts_by_kind_category_and_operation(tmp_path):
    _write_manifest(tmp_path, _manifest())
    run = load_workflow_run("aocd", results_dir=tmp_path / "results")

    assert [
        artifact.relative_path
        for artifact in run.artifacts(kind=KIND_DESCRIPTOR_TABLE)
    ] == ["features/full/features.csv"]
    assert [
        artifact.relative_path
        for artifact in run.artifacts(category="dimred")
    ] == ["dimred/pca/pca_coordinates.csv"]
    assert [
        artifact.relative_path
        for artifact in run.artifacts(operation="viz.correlations")
    ] == ["figures/correlations/hddf_corr_scatters_trendline.png"]


def test_workflow_run_returns_single_artifact(tmp_path):
    _write_manifest(tmp_path, _manifest())
    run = load_workflow_run("aocd", results_dir=tmp_path / "results")

    artifact = run.artifact(kind=KIND_DESCRIPTOR_TABLE, category="chem")

    assert isinstance(artifact, ResultArtifact)
    assert artifact.relative_path == "features/full/features.csv"


def test_workflow_run_single_artifact_rejects_zero_matches(tmp_path):
    _write_manifest(tmp_path, _manifest())
    run = load_workflow_run("aocd", results_dir=tmp_path / "results")

    with pytest.raises(FileNotFoundError, match="No artifact found"):
        run.artifact(kind=KIND_DESCRIPTOR_TABLE, category="dimred")


def test_workflow_run_single_artifact_rejects_multiple_matches(tmp_path):
    _write_manifest(tmp_path, _manifest())
    run = load_workflow_run("aocd", results_dir=tmp_path / "results")

    with pytest.raises(ValueError, match="Multiple artifacts found"):
        run.artifact(category="chem")


def test_workflow_run_loads_single_artifact(tmp_path):
    _write_manifest(tmp_path, _manifest())
    features = (
        tmp_path / "results" / "aocd" / "features" / "full" / "features.csv"
    )
    features.parent.mkdir(parents=True)
    pd.DataFrame({"identifier": ["A1"], "QED": [0.8]}).to_csv(
        features, index=False)
    run = load_workflow_run("aocd", results_dir=tmp_path / "results")

    loaded = run.load_artifact(kind=KIND_DESCRIPTOR_TABLE, category="chem")

    assert isinstance(loaded, LoadedArtifact)
    assert loaded.artifact.relative_path == "features/full/features.csv"
    assert loaded.data["identifier"].tolist() == ["A1"]


def test_workflow_run_science_accessors_return_spaces(tmp_path):
    _write_manifest(tmp_path, _manifest())
    run_root = tmp_path / "results" / "aocd"
    features = run_root / "features" / "full" / "features.csv"
    features.parent.mkdir(parents=True)
    pd.DataFrame({
        "identifier": ["A1", "A2"],
        "QED": [0.8, 0.6],
    }).to_csv(features, index=False)
    tanimoto = run_root / "chemistry" / "tanimoto" / "tanimoto_matrix.npz"
    tanimoto.parent.mkdir(parents=True)
    np.savez_compressed(
        tanimoto,
        matrix=np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32),
    )
    pd.DataFrame({"id": ["A1", "A2"]}).to_csv(
        tanimoto.parent / "tanimoto_ids.csv",
        index=False,
    )
    projection = run_root / "dimred" / "pca" / "pca_coordinates.csv"
    projection.parent.mkdir(parents=True)
    pd.DataFrame({
        "identifier": ["A1", "A2"],
        "PC1": [0.1, 0.2],
        "PC2": [0.3, 0.4],
    }).to_csv(projection, index=False)
    run = load_workflow_run("aocd", results_dir=tmp_path / "results")

    descriptors = run.descriptor_space(
        category="chem",
        operation="chem.tanimoto",
    )
    similarity = run.similarity_space(category="chem")
    projection_space = run.projection_space(
        category="dimred",
        operation="dimred.pca",
    )

    assert isinstance(descriptors, DescriptorSpace)
    assert descriptors.feature_names == ("QED",)
    assert descriptors.n_molecules == 2
    assert isinstance(similarity, SimilaritySpace)
    assert similarity.ids == ("A1", "A2")
    assert similarity.n_molecules == 2
    assert isinstance(projection_space, ProjectionSpace)
    assert projection_space.coordinate_columns == ("PC1", "PC2")
    assert projection_space.n_molecules == 2


def test_workflow_run_science_accessors_filter_required_outputs(tmp_path):
    manifest = _manifest()
    manifest["operations"].append({
        "operation": "chem.curate_features",
        "files": ["features/curated/features_ml.csv"],
        "metadata": {"source": "curation"},
    })
    manifest["current_outputs"].append("features/curated/features_ml.csv")
    manifest["output_categories"]["chem"].append(
        "features/curated/features_ml.csv")
    _write_manifest(tmp_path, manifest)
    run_root = tmp_path / "results" / "aocd"
    full = run_root / "features" / "full" / "features.csv"
    full.parent.mkdir(parents=True)
    pd.DataFrame({"identifier": ["A1"], "QED": [0.8]}).to_csv(
        full, index=False)
    curated = run_root / "features" / "curated" / "features_ml.csv"
    curated.parent.mkdir(parents=True)
    pd.DataFrame({"identifier": ["A1"], "MolWt": [46.0]}).to_csv(
        curated, index=False)
    run = load_workflow_run("aocd", results_dir=tmp_path / "results")

    with pytest.raises(ValueError, match="Multiple artifacts found"):
        run.descriptor_space(category="chem")

    descriptors = run.descriptor_space(
        category="chem",
        required="features/full/features.csv",
    )

    assert descriptors.artifact.relative_path == "features/full/features.csv"
    assert descriptors.feature_names == ("QED",)


def test_workflow_run_science_accessors_raise_for_missing_kind(tmp_path):
    _write_manifest(tmp_path, _manifest())
    run = load_workflow_run("aocd", results_dir=tmp_path / "results")

    with pytest.raises(FileNotFoundError, match="No artifact found"):
        run.descriptor_space(category="dimred")
    with pytest.raises(FileNotFoundError, match="No artifact found"):
        run.projection_space(category="chem", required="features/full")


def test_workflow_run_science_accessors_report_kind_mismatch(monkeypatch, tmp_path):
    _write_manifest(tmp_path, _manifest())
    run = load_workflow_run("aocd", results_dir=tmp_path / "results")
    wrong_loaded = LoadedArtifact(
        artifact=ResultArtifact(
            path=tmp_path / "metadata.json",
            relative_path="metadata.json",
            category="metadata",
            kind=KIND_WORKFLOW_SUMMARY,
        ),
        data="# Summary",
        metadata={},
    )
    monkeypatch.setattr(WorkflowRun, "load_artifact", lambda *args, **kwargs: wrong_loaded)

    with pytest.raises(ValueError, match="DescriptorSpace requires"):
        run.descriptor_space(category="chem")


def test_load_workflow_run_raises_when_manifest_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_workflow_run("aocd", results_dir=tmp_path / "results")


def test_load_workflow_run_rejects_traversal_tag(tmp_path):
    with pytest.raises(ValueError):
        load_workflow_run("../aocd", results_dir=tmp_path / "results")


def test_load_workflow_run_rejects_empty_tag(tmp_path):
    with pytest.raises(ValueError):
        load_workflow_run("  ", results_dir=tmp_path / "results")


def test_load_workflow_run_rejects_absolute_tag(tmp_path):
    with pytest.raises(ValueError):
        load_workflow_run(str(tmp_path / "aocd"), results_dir=tmp_path / "results")


def test_load_workflow_run_rejects_path_separator_tag(tmp_path):
    with pytest.raises(ValueError):
        load_workflow_run("ao/cd", results_dir=tmp_path / "results")


def test_workflow_run_rejects_manifest_output_traversal(tmp_path):
    manifest = _manifest()
    manifest["current_outputs"] = ["../outside.csv"]
    manifest["output_categories"] = {"chem": ["../outside.csv"]}
    _write_manifest(tmp_path, manifest)
    run = load_workflow_run("aocd", results_dir=tmp_path / "results")

    with pytest.raises(ValueError, match="contains traversal"):
        run.artifacts()


def test_workflow_run_rejects_absolute_manifest_output(tmp_path):
    outside = tmp_path / "outside.csv"
    manifest = _manifest()
    manifest["current_outputs"] = [str(outside)]
    manifest["output_categories"] = {"chem": [str(outside)]}
    _write_manifest(tmp_path, manifest)
    run = load_workflow_run("aocd", results_dir=tmp_path / "results")

    with pytest.raises(ValueError, match="must be relative"):
        run.artifacts()


def test_workflow_run_rejects_symlink_artifact_escape(tmp_path):
    manifest = _manifest()
    manifest["current_outputs"] = ["features/full/features.csv"]
    manifest["output_categories"] = {"chem": ["features/full/features.csv"]}
    _write_manifest(tmp_path, manifest)
    outside = tmp_path / "outside.csv"
    outside.write_text("identifier,QED\nA1,0.8\n", encoding="utf-8")
    link = (
        tmp_path
        / "results"
        / "aocd"
        / "features"
        / "full"
        / "features.csv"
    )
    link.parent.mkdir(parents=True)
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError) as e:
        pytest.skip(f"Symlink creation unavailable: {e}")
    run = load_workflow_run("aocd", results_dir=tmp_path / "results")

    with pytest.raises(ValueError, match="escapes run directory"):
        run.artifacts()


def test_load_workflow_run_raises_for_invalid_json(tmp_path):
    run_dir = tmp_path / "results" / "aocd"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid manifest JSON"):
        load_workflow_run("aocd", results_dir=tmp_path / "results")


def test_load_workflow_run_raises_for_invalid_structure(tmp_path):
    _write_manifest(tmp_path, {"tag": "aocd", "operations": []})

    with pytest.raises(ValueError, match="current_outputs"):
        load_workflow_run("aocd", results_dir=tmp_path / "results")
