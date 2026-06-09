# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for scientific views over loaded artifacts."""

import numpy as np
import pandas as pd
import pytest

from hddflyzer.results import (
    KIND_DESCRIPTOR_TABLE,
    KIND_METADATA,
    KIND_PROJECTION_COORDINATES,
    KIND_TANIMOTO_MATRIX,
    LoadedArtifact,
    ResultArtifact,
)
from hddflyzer.science import (
    DescriptorSpace,
    ProjectionSpace,
    SimilaritySpace,
    align_spaces,
    has_aligned_molecule_ids,
    shared_molecule_ids,
    to_descriptor_space,
    to_projection_space,
    to_similarity_space,
)


def _loaded(kind, data, metadata=None):
    artifact = ResultArtifact(
        path="artifact.csv",
        relative_path="artifact.csv",
        category="test",
        kind=kind,
        operation="test.operation",
        metadata={"source": "manifest"},
    )
    return LoadedArtifact(
        artifact=artifact,
        data=data,
        metadata=metadata or {"source": "loader"},
    )


def test_to_descriptor_space_wraps_descriptor_table():
    df = pd.DataFrame({
        "identifier": ["A1", "A2"],
        "SMILES": ["CCO", "CCC"],
        "QED": [0.8, 0.6],
        "MolWt": [46.0, 44.0],
    })

    space = to_descriptor_space(_loaded(KIND_DESCRIPTOR_TABLE, df))

    assert isinstance(space, DescriptorSpace)
    assert space.data is df
    assert space.n_molecules == 2
    assert space.feature_names == ("QED", "MolWt")
    assert space.molecule_ids == ("A1", "A2")
    assert space.metadata == {"source": "loader"}


def test_to_similarity_space_wraps_tanimoto_matrix():
    matrix = np.array([[1.0, 0.4], [0.4, 1.0]], dtype=np.float32)
    loaded = _loaded(
        KIND_TANIMOTO_MATRIX,
        matrix,
        metadata={"ids": ["A1", "A2"], "matrix_shape": [2, 2]},
    )

    space = to_similarity_space(loaded)

    assert isinstance(space, SimilaritySpace)
    assert space.matrix is matrix
    assert space.ids == ("A1", "A2")
    assert space.molecule_ids == ("A1", "A2")
    assert space.n_molecules == 2


def test_to_projection_space_wraps_projection_coordinates():
    df = pd.DataFrame({
        "identifier": ["A1", "A2"],
        "PC1": [0.1, 0.2],
        "PC2": [0.3, 0.4],
        "label": ["a", "b"],
    })

    space = to_projection_space(_loaded(KIND_PROJECTION_COORDINATES, df))

    assert isinstance(space, ProjectionSpace)
    assert space.data is df
    assert space.coordinate_columns == ("PC1", "PC2")
    assert space.molecule_ids == ("A1", "A2")
    assert space.n_molecules == 2


def test_descriptor_space_derives_molecule_ids_from_alternate_id_column():
    df = pd.DataFrame({
        "compound_id": ["C1", "C2"],
        "QED": [0.8, 0.6],
    })

    space = to_descriptor_space(_loaded(KIND_DESCRIPTOR_TABLE, df))

    assert space.molecule_ids == ("C1", "C2")


def test_projection_space_derives_molecule_ids_from_metadata_when_missing_column():
    df = pd.DataFrame({
        "PC1": [0.1, 0.2],
        "PC2": [0.3, 0.4],
    })

    space = to_projection_space(
        _loaded(
            KIND_PROJECTION_COORDINATES,
            df,
            metadata={"ids": ["A1", "A2"]},
        )
    )

    assert space.molecule_ids == ("A1", "A2")


def test_space_converters_reject_wrong_kind():
    loaded = _loaded(KIND_METADATA, {"n": 1})

    with pytest.raises(ValueError, match="DescriptorSpace requires"):
        to_descriptor_space(loaded)
    with pytest.raises(ValueError, match="SimilaritySpace requires"):
        to_similarity_space(loaded)
    with pytest.raises(ValueError, match="ProjectionSpace requires"):
        to_projection_space(loaded)


def test_to_similarity_space_requires_aligned_ids():
    matrix = np.eye(2, dtype=np.float32)
    loaded = _loaded(KIND_TANIMOTO_MATRIX, matrix, metadata={"ids": ["A1"]})

    with pytest.raises(ValueError, match="one identifier per matrix row"):
        to_similarity_space(loaded)


def test_to_projection_space_requires_numeric_coordinates():
    df = pd.DataFrame({"identifier": ["A1"], "label": ["x"]})
    loaded = _loaded(KIND_PROJECTION_COORDINATES, df)

    with pytest.raises(ValueError, match="two numeric coordinate columns"):
        to_projection_space(loaded)


def test_spaces_return_empty_molecule_ids_when_missing():
    descriptor_df = pd.DataFrame({"QED": [0.8], "MolWt": [46.0]})
    projection_df = pd.DataFrame({"PC1": [0.1], "PC2": [0.2]})

    descriptor = to_descriptor_space(
        _loaded(KIND_DESCRIPTOR_TABLE, descriptor_df))
    projection = to_projection_space(
        _loaded(KIND_PROJECTION_COORDINATES, projection_df))

    assert descriptor.molecule_ids == ()
    assert projection.molecule_ids == ()


def test_shared_molecule_ids_preserves_first_space_order():
    descriptor = to_descriptor_space(_loaded(
        KIND_DESCRIPTOR_TABLE,
        pd.DataFrame({
            "identifier": ["A3", "A1", "A2"],
            "QED": [0.4, 0.8, 0.6],
        }),
    ))
    projection = to_projection_space(_loaded(
        KIND_PROJECTION_COORDINATES,
        pd.DataFrame({
            "identifier": ["A1", "A2", "A4"],
            "PC1": [0.1, 0.2, 0.3],
            "PC2": [0.4, 0.5, 0.6],
        }),
    ))

    assert shared_molecule_ids(descriptor, projection) == ("A1", "A2")


def test_has_aligned_molecule_ids_requires_non_empty_same_order():
    descriptor = to_descriptor_space(_loaded(
        KIND_DESCRIPTOR_TABLE,
        pd.DataFrame({"identifier": ["A1", "A2"], "QED": [0.8, 0.6]}),
    ))
    projection = to_projection_space(_loaded(
        KIND_PROJECTION_COORDINATES,
        pd.DataFrame({
            "identifier": ["A1", "A2"],
            "PC1": [0.1, 0.2],
            "PC2": [0.3, 0.4],
        }),
    ))
    reordered = to_projection_space(_loaded(
        KIND_PROJECTION_COORDINATES,
        pd.DataFrame({
            "identifier": ["A2", "A1"],
            "PC1": [0.2, 0.1],
            "PC2": [0.4, 0.3],
        }),
    ))
    missing = to_descriptor_space(_loaded(
        KIND_DESCRIPTOR_TABLE,
        pd.DataFrame({"QED": [0.8, 0.6]}),
    ))

    assert has_aligned_molecule_ids(descriptor, projection) is True
    assert has_aligned_molecule_ids(descriptor, reordered) is False
    assert has_aligned_molecule_ids(descriptor, missing) is False


def test_align_spaces_aligns_descriptor_and_projection():
    descriptor = to_descriptor_space(_loaded(
        KIND_DESCRIPTOR_TABLE,
        pd.DataFrame({
            "identifier": ["A3", "A1", "A2"],
            "QED": [0.4, 0.8, 0.6],
        }),
    ))
    projection = to_projection_space(_loaded(
        KIND_PROJECTION_COORDINATES,
        pd.DataFrame({
            "identifier": ["A1", "A2", "A4"],
            "PC1": [0.1, 0.2, 0.4],
            "PC2": [0.3, 0.4, 0.8],
        }),
    ))

    aligned_descriptor, aligned_projection = align_spaces(descriptor, projection)

    assert isinstance(aligned_descriptor, DescriptorSpace)
    assert isinstance(aligned_projection, ProjectionSpace)
    assert aligned_descriptor.molecule_ids == ("A1", "A2")
    assert aligned_projection.molecule_ids == ("A1", "A2")
    assert aligned_descriptor.data["QED"].tolist() == [0.8, 0.6]
    assert aligned_projection.data["PC1"].tolist() == [0.1, 0.2]


def test_align_spaces_aligns_similarity_rows_and_columns():
    descriptor = to_descriptor_space(_loaded(
        KIND_DESCRIPTOR_TABLE,
        pd.DataFrame({
            "identifier": ["A3", "A1"],
            "QED": [0.4, 0.8],
        }),
    ))
    matrix = np.array([
        [1.0, 0.1, 0.2],
        [0.1, 1.0, 0.3],
        [0.2, 0.3, 1.0],
    ], dtype=np.float32)
    similarity = to_similarity_space(_loaded(
        KIND_TANIMOTO_MATRIX,
        matrix,
        metadata={"ids": ["A1", "A2", "A3"]},
    ))

    aligned_descriptor, aligned_similarity = align_spaces(
        descriptor,
        similarity,
    )

    assert aligned_descriptor.molecule_ids == ("A3", "A1")
    assert aligned_similarity.molecule_ids == ("A3", "A1")
    assert aligned_similarity.ids == ("A3", "A1")
    assert np.allclose(
        aligned_similarity.matrix,
        np.array([[1.0, 0.2], [0.2, 1.0]], dtype=np.float32),
    )


def test_align_spaces_preserves_order_of_first_space_for_three_spaces():
    descriptor = to_descriptor_space(_loaded(
        KIND_DESCRIPTOR_TABLE,
        pd.DataFrame({
            "identifier": ["A3", "A1", "A2"],
            "QED": [0.4, 0.8, 0.6],
        }),
    ))
    projection = to_projection_space(_loaded(
        KIND_PROJECTION_COORDINATES,
        pd.DataFrame({
            "identifier": ["A2", "A3", "A1"],
            "PC1": [0.2, 0.3, 0.1],
            "PC2": [0.4, 0.6, 0.2],
        }),
    ))
    similarity = to_similarity_space(_loaded(
        KIND_TANIMOTO_MATRIX,
        np.eye(3, dtype=np.float32),
        metadata={"ids": ["A1", "A2", "A3"]},
    ))

    aligned = align_spaces(descriptor, projection, similarity)

    assert [space.molecule_ids for space in aligned] == [
        ("A3", "A1", "A2"),
        ("A3", "A1", "A2"),
        ("A3", "A1", "A2"),
    ]


def test_align_spaces_does_not_mutate_original_spaces():
    descriptor = to_descriptor_space(_loaded(
        KIND_DESCRIPTOR_TABLE,
        pd.DataFrame({
            "identifier": ["A1", "A2"],
            "QED": [0.8, 0.6],
        }),
    ))
    projection = to_projection_space(_loaded(
        KIND_PROJECTION_COORDINATES,
        pd.DataFrame({
            "identifier": ["A2"],
            "PC1": [0.2],
            "PC2": [0.4],
        }),
    ))
    original_descriptor_df = descriptor.data.copy()
    original_projection_df = projection.data.copy()

    aligned_descriptor, aligned_projection = align_spaces(descriptor, projection)

    assert descriptor.data.equals(original_descriptor_df)
    assert projection.data.equals(original_projection_df)
    assert aligned_descriptor is not descriptor
    assert aligned_projection is not projection
    assert aligned_descriptor.data is not descriptor.data
    assert aligned_projection.data is not projection.data


def test_align_spaces_rejects_empty_molecule_ids():
    descriptor = to_descriptor_space(_loaded(
        KIND_DESCRIPTOR_TABLE,
        pd.DataFrame({"QED": [0.8]}),
    ))
    projection = to_projection_space(_loaded(
        KIND_PROJECTION_COORDINATES,
        pd.DataFrame({"identifier": ["A1"], "PC1": [0.1], "PC2": [0.2]}),
    ))

    with pytest.raises(ValueError, match="non-empty molecule_ids"):
        align_spaces(descriptor, projection)


def test_align_spaces_rejects_no_shared_ids():
    descriptor = to_descriptor_space(_loaded(
        KIND_DESCRIPTOR_TABLE,
        pd.DataFrame({"identifier": ["A1"], "QED": [0.8]}),
    ))
    projection = to_projection_space(_loaded(
        KIND_PROJECTION_COORDINATES,
        pd.DataFrame({"identifier": ["B1"], "PC1": [0.1], "PC2": [0.2]}),
    ))

    with pytest.raises(ValueError, match="No shared molecule IDs"):
        align_spaces(descriptor, projection)


def test_align_spaces_rejects_unsupported_type():
    descriptor = to_descriptor_space(_loaded(
        KIND_DESCRIPTOR_TABLE,
        pd.DataFrame({"identifier": ["A1"], "QED": [0.8]}),
    ))

    with pytest.raises(ValueError, match="Unsupported scientific space type"):
        align_spaces(descriptor, object())


def test_align_spaces_requires_two_or_more_spaces():
    descriptor = to_descriptor_space(_loaded(
        KIND_DESCRIPTOR_TABLE,
        pd.DataFrame({"identifier": ["A1"], "QED": [0.8]}),
    ))

    with pytest.raises(ValueError, match="two or more spaces"):
        align_spaces(descriptor)
