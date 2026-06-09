# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for structural metrics between scientific spaces."""

import numpy as np
import pandas as pd
import pytest

from hddflyzer.results import ResultArtifact
from hddflyzer.science import (
    DescriptorGroupComparisonResult,
    DescriptorProjectionCorrelationResult,
    DescriptorSpace,
    NeighborhoodPreservationResult,
    ProjectionSpace,
    SimilaritySpace,
    SpaceMetricResult,
    compare_descriptor_groups,
    descriptor_projection_correlations,
    projection_neighborhood_preservation,
    similarity_projection_correlation,
    similarity_projection_neighbor_overlap,
)


def _artifact(kind):
    return ResultArtifact(
        path=f"{kind}.csv",
        relative_path=f"{kind}.csv",
        category="test",
        kind=kind,
    )


def _similarity(ids=("A1", "A2", "A3")):
    matrix = np.array([
        [1.0, 0.9, 0.1],
        [0.9, 1.0, 0.2],
        [0.1, 0.2, 1.0],
    ], dtype=np.float32)
    order = ["A1", "A2", "A3"]
    indices = [order.index(identifier) for identifier in ids]
    return SimilaritySpace(
        artifact=_artifact("tanimoto_matrix"),
        matrix=matrix[np.ix_(indices, indices)].copy(),
        ids=tuple(ids),
        metadata={"ids": list(ids)},
        n_molecules=len(ids),
    )


def _projection(ids=("A1", "A2", "A3")):
    coords = {
        "A1": (0.0, 0.0),
        "A2": (0.1, 0.0),
        "A3": (5.0, 5.0),
    }
    return ProjectionSpace(
        artifact=_artifact("projection_coordinates"),
        data=pd.DataFrame({
            "identifier": list(ids),
            "PC1": [coords[identifier][0] for identifier in ids],
            "PC2": [coords[identifier][1] for identifier in ids],
        }),
        metadata={},
        coordinate_columns=("PC1", "PC2"),
        n_molecules=len(ids),
    )


def _descriptor_projection(ids=("A1", "A2", "A3")):
    coords = {
        "A1": (0.0, 0.0),
        "A2": (1.0, 0.0),
        "A3": (2.0, 0.0),
    }
    return ProjectionSpace(
        artifact=_artifact("projection_coordinates"),
        data=pd.DataFrame({
            "identifier": list(ids),
            "PC1": [coords[identifier][0] for identifier in ids],
            "PC2": [coords[identifier][1] for identifier in ids],
        }),
        metadata={},
        coordinate_columns=("PC1", "PC2"),
        n_molecules=len(ids),
    )


def _descriptors(ids=("A1", "A2", "A3")):
    values = {
        "A1": (0.0, 3.0),
        "A2": (1.0, 2.0),
        "A3": (2.0, 1.0),
    }
    return DescriptorSpace(
        artifact=_artifact("descriptor_table"),
        data=pd.DataFrame({
            "identifier": list(ids),
            "linear": [values[identifier][0] for identifier in ids],
            "inverse": [values[identifier][1] for identifier in ids],
            "label": ["x" for _ in ids],
        }),
        metadata={},
        n_molecules=len(ids),
        feature_names=("linear", "inverse", "label"),
    )


def _group_descriptors():
    return DescriptorSpace(
        artifact=_artifact("descriptor_table"),
        data=pd.DataFrame({
            "identifier": ["A1", "A2", "A3", "A4", "A5"],
            "class_label": ["alpha", "alpha", "beta", "beta", "tiny"],
            "strong": [10.0, 12.0, 1.0, 3.0, 100.0],
            "weak": [5.0, 5.5, 5.0, 5.5, 5.0],
            "note": ["x", "y", "z", "w", "q"],
        }),
        metadata={},
        n_molecules=5,
        feature_names=("strong", "weak", "note"),
    )


def test_compare_descriptor_groups_with_label_column():
    result = compare_descriptor_groups(
        _group_descriptors(),
        "class_label",
        min_group_size=2,
    )

    assert isinstance(result, DescriptorGroupComparisonResult)
    assert result.metadata["n_molecules"] == 5
    assert result.metadata["groups"] == ("alpha", "beta")
    assert result.metadata["feature_names"] == ("strong", "weak")
    assert result.metadata["min_group_size"] == 2
    assert list(result.data.columns) == [
        "group",
        "feature",
        "n",
        "mean",
        "median",
        "std",
        "delta_from_global_mean",
        "abs_delta_from_global_mean",
    ]
    alpha_strong = result.data[
        (result.data["group"] == "alpha")
        & (result.data["feature"] == "strong")
    ].iloc[0]
    assert alpha_strong["n"] == 2
    assert alpha_strong["mean"] == pytest.approx(11.0)


def test_compare_descriptor_groups_with_label_sequence():
    result = compare_descriptor_groups(
        _group_descriptors(),
        ["alpha", "alpha", "beta", "beta", "tiny"],
        min_group_size=2,
    )

    assert result.metadata["groups"] == ("alpha", "beta")
    assert set(result.data["group"]) == {"alpha", "beta"}


def test_compare_descriptor_groups_excludes_small_groups():
    result = compare_descriptor_groups(
        _group_descriptors(),
        "class_label",
        min_group_size=3,
    )

    assert result.data.empty
    assert list(result.data.columns) == [
        "group",
        "feature",
        "n",
        "mean",
        "median",
        "std",
        "delta_from_global_mean",
        "abs_delta_from_global_mean",
    ]
    assert result.metadata["groups"] == ()


def test_compare_descriptor_groups_ranks_by_absolute_delta():
    result = compare_descriptor_groups(
        _group_descriptors(),
        "class_label",
        min_group_size=2,
    )

    assert result.data.iloc[0]["feature"] == "strong"
    assert result.data.iloc[0]["abs_delta_from_global_mean"] >= (
        result.data.iloc[-1]["abs_delta_from_global_mean"]
    )


def test_descriptor_group_top_differences_returns_copy():
    result = compare_descriptor_groups(
        _group_descriptors(),
        "class_label",
        min_group_size=2,
    )

    top = result.top_differences(2)
    top.loc[:, "group"] = "changed"

    assert len(top) == 2
    assert "changed" not in result.data["group"].tolist()


def test_descriptor_group_top_differences_rejects_invalid_n():
    result = compare_descriptor_groups(
        _group_descriptors(),
        "class_label",
        min_group_size=2,
    )

    with pytest.raises(ValueError, match="positive integer"):
        result.top_differences(0)


def test_compare_descriptor_groups_ignores_non_numeric_columns():
    result = compare_descriptor_groups(
        _group_descriptors(),
        "class_label",
        min_group_size=2,
    )

    assert "note" not in result.metadata["feature_names"]
    assert "note" not in result.data["feature"].tolist()


def test_compare_descriptor_groups_rejects_no_numeric_features():
    descriptors = DescriptorSpace(
        artifact=_artifact("descriptor_table"),
        data=pd.DataFrame({
            "identifier": ["A1", "A2"],
            "class_label": ["alpha", "alpha"],
            "note": ["x", "y"],
        }),
        metadata={},
        n_molecules=2,
        feature_names=("note",),
    )

    with pytest.raises(ValueError, match="no numeric descriptor features"):
        compare_descriptor_groups(descriptors, "class_label")


def test_compare_descriptor_groups_rejects_bad_labels():
    descriptors = _group_descriptors()

    with pytest.raises(ValueError, match="Label column not found"):
        compare_descriptor_groups(descriptors, "missing_label")
    with pytest.raises(ValueError, match="sequence length"):
        compare_descriptor_groups(descriptors, ["a", "b"])


def test_compare_descriptor_groups_rejects_invalid_min_group_size():
    with pytest.raises(ValueError, match="positive integer"):
        compare_descriptor_groups(
            _group_descriptors(),
            "class_label",
            min_group_size=0,
        )


def test_compare_descriptor_groups_does_not_mutate_original_space():
    descriptors = _group_descriptors()
    original = descriptors.data.copy()

    compare_descriptor_groups(descriptors, "class_label")

    assert descriptors.data.equals(original)


def test_descriptor_projection_correlations_aligned_spaces():
    result = descriptor_projection_correlations(
        _descriptors(),
        _descriptor_projection(),
    )

    assert isinstance(result, DescriptorProjectionCorrelationResult)
    assert result.metadata["n_molecules"] == 3
    assert result.metadata["feature_names"] == ("linear", "inverse")
    assert result.metadata["coordinate_columns"] == ("PC1", "PC2")
    assert list(result.data.columns) == [
        "feature",
        "coordinate",
        "correlation",
        "abs_correlation",
    ]
    linear_pc1 = result.data[
        (result.data["feature"] == "linear")
        & (result.data["coordinate"] == "PC1")
    ].iloc[0]
    assert linear_pc1["correlation"] > 0.99


def test_descriptor_projection_correlations_aligns_internally():
    descriptors = _descriptors(ids=("A3", "A1", "A2"))
    projection = _descriptor_projection(ids=("A1", "A2", "A3"))

    result = descriptor_projection_correlations(descriptors, projection)

    linear_pc1 = result.data[
        (result.data["feature"] == "linear")
        & (result.data["coordinate"] == "PC1")
    ].iloc[0]
    assert linear_pc1["correlation"] > 0.99


def test_descriptor_projection_correlations_ranks_by_absolute_magnitude():
    descriptors = DescriptorSpace(
        artifact=_artifact("descriptor_table"),
        data=pd.DataFrame({
            "identifier": ["A1", "A2", "A3", "A4"],
            "strong": [0.0, 1.0, 2.0, 3.0],
            "weak": [1.0, 1.1, 0.9, 1.0],
        }),
        metadata={},
        n_molecules=4,
        feature_names=("strong", "weak"),
    )
    projection = ProjectionSpace(
        artifact=_artifact("projection_coordinates"),
        data=pd.DataFrame({
            "identifier": ["A1", "A2", "A3", "A4"],
            "PC1": [0.0, 1.0, 2.0, 3.0],
            "PC2": [0.0, 0.0, 0.0, 0.1],
        }),
        metadata={},
        coordinate_columns=("PC1", "PC2"),
        n_molecules=4,
    )

    result = descriptor_projection_correlations(descriptors, projection)

    assert result.data.iloc[0]["feature"] == "strong"
    assert result.data.iloc[0]["coordinate"] == "PC1"
    assert result.data.iloc[0]["abs_correlation"] == pytest.approx(1.0)


def test_descriptor_projection_top_features_returns_copy():
    result = descriptor_projection_correlations(_descriptors(), _projection())

    top = result.top_features(2)
    top.loc[:, "feature"] = "changed"

    assert len(top) == 2
    assert "changed" not in result.data["feature"].tolist()


def test_descriptor_projection_top_features_rejects_invalid_n():
    result = descriptor_projection_correlations(_descriptors(), _projection())

    with pytest.raises(ValueError, match="positive integer"):
        result.top_features(0)


def test_descriptor_projection_ignores_non_numeric_columns():
    result = descriptor_projection_correlations(_descriptors(), _projection())

    assert "label" not in result.metadata["feature_names"]
    assert "label" not in result.data["feature"].tolist()


def test_descriptor_projection_rejects_no_numeric_features():
    descriptors = DescriptorSpace(
        artifact=_artifact("descriptor_table"),
        data=pd.DataFrame({
            "identifier": ["A1", "A2"],
            "label": ["x", "y"],
        }),
        metadata={},
        n_molecules=2,
        feature_names=("label",),
    )

    with pytest.raises(ValueError, match="no numeric descriptor features"):
        descriptor_projection_correlations(
            descriptors,
            _projection(ids=("A1", "A2")),
        )


def test_descriptor_projection_rejects_missing_ids_or_no_shared_ids():
    missing_ids = DescriptorSpace(
        artifact=_artifact("descriptor_table"),
        data=pd.DataFrame({"linear": [0.0, 1.0]}),
        metadata={},
        n_molecules=2,
        feature_names=("linear",),
    )
    no_shared = DescriptorSpace(
        artifact=_artifact("descriptor_table"),
        data=pd.DataFrame({"identifier": ["B1", "B2"], "linear": [0.0, 1.0]}),
        metadata={},
        n_molecules=2,
        feature_names=("linear",),
    )

    with pytest.raises(ValueError, match="non-empty molecule_ids"):
        descriptor_projection_correlations(
            missing_ids,
            _projection(ids=("A1", "A2")),
        )
    with pytest.raises(ValueError, match="No shared molecule IDs"):
        descriptor_projection_correlations(
            no_shared,
            _projection(ids=("A1", "A2")),
        )


def test_descriptor_projection_rejects_projection_without_coordinates():
    projection = ProjectionSpace(
        artifact=_artifact("projection_coordinates"),
        data=pd.DataFrame({"identifier": ["A1", "A2"], "PC1": [0.0, 1.0]}),
        metadata={},
        coordinate_columns=("PC1",),
        n_molecules=2,
    )

    with pytest.raises(ValueError, match="at least two coordinates"):
        descriptor_projection_correlations(
            _descriptors(ids=("A1", "A2")),
            projection,
        )


def test_descriptor_projection_correlations_do_not_mutate_original_spaces():
    descriptors = _descriptors(ids=("A3", "A1", "A2"))
    projection = _projection(ids=("A1", "A2", "A3"))
    original_descriptors = descriptors.data.copy()
    original_projection = projection.data.copy()

    descriptor_projection_correlations(descriptors, projection)

    assert descriptors.data.equals(original_descriptors)
    assert projection.data.equals(original_projection)


def test_similarity_projection_correlation_aligned_spaces():
    result = similarity_projection_correlation(_similarity(), _projection())

    assert isinstance(result, SpaceMetricResult)
    assert result.name == "similarity_projection_correlation"
    assert result.value > 0.9
    assert result.metadata["n_molecules"] == 3
    assert result.metadata["n_pairs"] == 3


def test_similarity_projection_neighbor_overlap_aligned_spaces():
    result = similarity_projection_neighbor_overlap(
        _similarity(),
        _projection(),
        k=1,
    )

    assert isinstance(result, SpaceMetricResult)
    assert result.name == "similarity_projection_neighbor_overlap"
    assert result.value == pytest.approx(1.0)
    assert result.metadata["k"] == 1


def test_projection_neighborhood_preservation_aligned_spaces():
    result = projection_neighborhood_preservation(
        _similarity(),
        _projection(),
        k=1,
    )

    assert isinstance(result, NeighborhoodPreservationResult)
    assert result.metadata == {
        "k": 1,
        "n_molecules": 3,
        "coordinate_columns": ("PC1", "PC2"),
    }
    assert list(result.data.columns) == [
        "molecule_id",
        "overlap_count",
        "overlap_fraction",
        "similarity_neighbors",
        "projection_neighbors",
        "missing_similarity_neighbors",
        "extra_projection_neighbors",
    ]
    assert result.data["molecule_id"].tolist() == ["A1", "A2", "A3"]
    assert result.data["overlap_count"].tolist() == [1, 1, 1]
    assert result.data["overlap_fraction"].tolist() == [1.0, 1.0, 1.0]


def test_projection_neighborhood_preservation_aligns_internally():
    result = projection_neighborhood_preservation(
        _similarity(ids=("A3", "A1", "A2")),
        _projection(ids=("A1", "A2", "A3")),
        k=1,
    )

    assert result.data["molecule_id"].tolist() == ["A3", "A1", "A2"]
    assert result.data["overlap_fraction"].tolist() == [1.0, 1.0, 1.0]


def test_projection_neighborhood_preservation_reports_missing_and_extra_neighbors():
    similarity = SimilaritySpace(
        artifact=_artifact("tanimoto_matrix"),
        matrix=np.array([
            [1.0, 0.1, 0.9],
            [0.1, 1.0, 0.2],
            [0.9, 0.2, 1.0],
        ], dtype=np.float32),
        ids=("A1", "A2", "A3"),
        metadata={},
        n_molecules=3,
    )
    projection = _projection(ids=("A1", "A2", "A3"))

    result = projection_neighborhood_preservation(similarity, projection, k=1)
    row = result.data[result.data["molecule_id"] == "A1"].iloc[0]

    assert row["overlap_count"] == 0
    assert row["overlap_fraction"] == 0.0
    assert row["similarity_neighbors"] == ["A3"]
    assert row["projection_neighbors"] == ["A2"]
    assert row["missing_similarity_neighbors"] == ["A3"]
    assert row["extra_projection_neighbors"] == ["A2"]


def test_neighborhood_preservation_worst_preserved_returns_copy():
    similarity = SimilaritySpace(
        artifact=_artifact("tanimoto_matrix"),
        matrix=np.array([
            [1.0, 0.1, 0.9],
            [0.1, 1.0, 0.2],
            [0.9, 0.2, 1.0],
        ], dtype=np.float32),
        ids=("A1", "A2", "A3"),
        metadata={},
        n_molecules=3,
    )
    result = projection_neighborhood_preservation(
        similarity,
        _projection(),
        k=1,
    )

    worst = result.worst_preserved(1)
    worst.loc[:, "molecule_id"] = "changed"

    assert len(worst) == 1
    assert worst.iloc[0]["overlap_fraction"] == 0.0
    assert "changed" not in result.data["molecule_id"].tolist()


def test_neighborhood_preservation_worst_preserved_rejects_invalid_n():
    result = projection_neighborhood_preservation(
        _similarity(),
        _projection(),
        k=1,
    )

    with pytest.raises(ValueError, match="positive integer"):
        result.worst_preserved(0)


def test_metrics_align_internally_when_id_order_differs():
    similarity = _similarity(ids=("A3", "A1", "A2"))
    projection = _projection(ids=("A1", "A2", "A3"))

    correlation = similarity_projection_correlation(similarity, projection)
    overlap = similarity_projection_neighbor_overlap(
        similarity,
        projection,
        k=1,
    )

    assert correlation.metadata["n_molecules"] == 3
    assert correlation.value > 0.9
    assert overlap.value == pytest.approx(1.0)


def test_metrics_reject_empty_ids():
    similarity = SimilaritySpace(
        artifact=_artifact("tanimoto_matrix"),
        matrix=np.eye(2, dtype=np.float32),
        ids=(),
        metadata={},
        n_molecules=2,
    )

    with pytest.raises(ValueError, match="non-empty molecule_ids"):
        similarity_projection_correlation(similarity, _projection(ids=("A1", "A2")))
    with pytest.raises(ValueError, match="non-empty molecule_ids"):
        projection_neighborhood_preservation(
            similarity,
            _projection(ids=("A1", "A2")),
            k=1,
        )


def test_neighborhood_preservation_rejects_no_shared_ids():
    projection = ProjectionSpace(
        artifact=_artifact("projection_coordinates"),
        data=pd.DataFrame({
            "identifier": ["B1", "B2"],
            "PC1": [0.0, 1.0],
            "PC2": [0.0, 1.0],
        }),
        metadata={},
        coordinate_columns=("PC1", "PC2"),
        n_molecules=2,
    )

    with pytest.raises(ValueError, match="No shared molecule IDs"):
        projection_neighborhood_preservation(
            _similarity(ids=("A1", "A2")),
            projection,
            k=1,
        )


def test_neighbor_overlap_rejects_invalid_k():
    with pytest.raises(ValueError, match="positive integer"):
        similarity_projection_neighbor_overlap(_similarity(), _projection(), k=0)
    with pytest.raises(ValueError, match="smaller than the number of molecules"):
        similarity_projection_neighbor_overlap(_similarity(), _projection(), k=3)


def test_neighborhood_preservation_rejects_invalid_k():
    with pytest.raises(ValueError, match="positive integer"):
        projection_neighborhood_preservation(_similarity(), _projection(), k=0)
    with pytest.raises(ValueError, match="smaller than the number of molecules"):
        projection_neighborhood_preservation(_similarity(), _projection(), k=3)


def test_metrics_reject_projection_without_enough_coordinates():
    projection = ProjectionSpace(
        artifact=_artifact("projection_coordinates"),
        data=pd.DataFrame({"identifier": ["A1", "A2"], "PC1": [0.0, 1.0]}),
        metadata={},
        coordinate_columns=("PC1",),
        n_molecules=2,
    )

    with pytest.raises(ValueError, match="at least two coordinates"):
        similarity_projection_correlation(_similarity(ids=("A1", "A2")), projection)
    with pytest.raises(ValueError, match="at least two coordinates"):
        projection_neighborhood_preservation(
            _similarity(ids=("A1", "A2")),
            projection,
            k=1,
        )


def test_metrics_do_not_mutate_original_spaces():
    similarity = _similarity(ids=("A3", "A1", "A2"))
    projection = _projection(ids=("A1", "A2", "A3"))
    original_matrix = similarity.matrix.copy()
    original_projection = projection.data.copy()

    similarity_projection_neighbor_overlap(similarity, projection, k=1)
    projection_neighborhood_preservation(similarity, projection, k=1)

    assert np.array_equal(similarity.matrix, original_matrix)
    assert projection.data.equals(original_projection)
