# SPDX-License-Identifier: LGPL-3.0-or-later

"""Lightweight scientific views over loaded result artifacts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hddflyzer.results import (
    KIND_DESCRIPTOR_TABLE,
    KIND_PROJECTION_COORDINATES,
    KIND_TANIMOTO_MATRIX,
    LoadedArtifact,
    ResultArtifact,
)


IDENTITY_COLUMNS = {
    "identifier",
    "id",
    "compound_id",
    "molecule_id",
    "smiles",
    "canonical_smiles",
    "valid_smiles",
    "mol_parse_status",
    "source_file",
    "source_row",
}

MOLECULE_ID_COLUMNS = (
    "identifier",
    "id",
    "compound_id",
    "molecule_id",
)


@dataclass(frozen=True)
class DescriptorSpace:
    """Descriptor table interpreted as a molecular descriptor space.

    Attributes
    ----------
    artifact : ResultArtifact
        Source descriptor-table artifact.
    data : pandas.DataFrame
        Loaded descriptor table.
    metadata : dict
        Loader and operation metadata.
    n_molecules : int
        Number of rows in ``data``.
    feature_names : tuple of str
        Descriptor feature columns, excluding common identity columns.
    """

    artifact: ResultArtifact
    data: pd.DataFrame
    metadata: dict
    n_molecules: int
    feature_names: tuple[str, ...]

    @property
    def molecule_ids(self) -> tuple[str, ...]:
        """tuple of str: Molecular identifiers when present, else empty."""
        return _molecule_ids_from_table_or_metadata(self.data, self.metadata)


@dataclass(frozen=True)
class SimilaritySpace:
    """Pairwise molecular similarity matrix with aligned identifiers.

    Attributes
    ----------
    artifact : ResultArtifact
        Source Tanimoto matrix artifact.
    matrix : numpy.ndarray
        Pairwise similarity matrix.
    ids : tuple of str
        Identifiers aligned to matrix rows and columns.
    metadata : dict
        Loader and operation metadata.
    n_molecules : int
        Number of molecules in the similarity matrix.
    """

    artifact: ResultArtifact
    matrix: np.ndarray
    ids: tuple[str, ...]
    metadata: dict
    n_molecules: int

    @property
    def molecule_ids(self) -> tuple[str, ...]:
        """tuple of str: Molecular identifiers aligned to the matrix."""
        return self.ids


@dataclass(frozen=True)
class ProjectionSpace:
    """Dimensionality-reduction coordinates for a molecular collection.

    Attributes
    ----------
    artifact : ResultArtifact
        Source projection-coordinate artifact.
    data : pandas.DataFrame
        Loaded coordinate table.
    metadata : dict
        Loader and operation metadata.
    coordinate_columns : tuple of str
        Numeric coordinate columns used as the projection axes.
    n_molecules : int
        Number of rows in ``data``.
    """

    artifact: ResultArtifact
    data: pd.DataFrame
    metadata: dict
    coordinate_columns: tuple[str, ...]
    n_molecules: int

    @property
    def molecule_ids(self) -> tuple[str, ...]:
        """tuple of str: Molecular identifiers when present, else empty."""
        return _molecule_ids_from_table_or_metadata(self.data, self.metadata)


def to_descriptor_space(loaded: LoadedArtifact) -> DescriptorSpace:
    """Convert a loaded descriptor table artifact into a descriptor space.

    Parameters
    ----------
    loaded : LoadedArtifact
        Loaded artifact with kind ``"descriptor_table"``.

    Returns
    -------
    DescriptorSpace
        Scientific view over the loaded descriptor table.

    Raises
    ------
    ValueError
        If the artifact kind or data type is incompatible.

    Notes
    -----
    This converter wraps existing loaded data and does not recalculate
    descriptors.
    """
    _require_kind(loaded, KIND_DESCRIPTOR_TABLE, "DescriptorSpace")
    if not isinstance(loaded.data, pd.DataFrame):
        raise ValueError("DescriptorSpace requires pandas DataFrame data.")

    feature_names = tuple(
        str(column)
        for column in loaded.data.columns
        if str(column).lower() not in IDENTITY_COLUMNS
    )
    return DescriptorSpace(
        artifact=loaded.artifact,
        data=loaded.data,
        metadata=dict(loaded.metadata),
        n_molecules=len(loaded.data),
        feature_names=feature_names,
    )


def to_similarity_space(loaded: LoadedArtifact) -> SimilaritySpace:
    """Convert a loaded Tanimoto matrix artifact into a similarity space.

    Parameters
    ----------
    loaded : LoadedArtifact
        Loaded artifact with kind ``"tanimoto_matrix"``.

    Returns
    -------
    SimilaritySpace
        Scientific view over the loaded similarity matrix.

    Raises
    ------
    ValueError
        If the artifact kind, data type, or ID alignment is incompatible.

    Notes
    -----
    This converter wraps an existing matrix and does not recalculate
    fingerprints or similarity.
    """
    _require_kind(loaded, KIND_TANIMOTO_MATRIX, "SimilaritySpace")
    if not isinstance(loaded.data, np.ndarray):
        raise ValueError("SimilaritySpace requires numpy ndarray data.")

    ids = tuple(str(identifier) for identifier in loaded.metadata.get("ids", []))
    if len(ids) != loaded.data.shape[0]:
        raise ValueError(
            "SimilaritySpace requires one identifier per matrix row."
        )
    return SimilaritySpace(
        artifact=loaded.artifact,
        matrix=loaded.data,
        ids=ids,
        metadata=dict(loaded.metadata),
        n_molecules=loaded.data.shape[0],
    )


def to_projection_space(loaded: LoadedArtifact) -> ProjectionSpace:
    """Convert loaded projection coordinates into a projection space.

    Parameters
    ----------
    loaded : LoadedArtifact
        Loaded artifact with kind ``"projection_coordinates"``.

    Returns
    -------
    ProjectionSpace
        Scientific view over existing projection coordinates.

    Raises
    ------
    ValueError
        If the artifact kind, data type, or coordinate columns are
        incompatible.

    Notes
    -----
    This converter does not recalculate PCA, t-SNE, UMAP, or other
    projections.
    """
    _require_kind(loaded, KIND_PROJECTION_COORDINATES, "ProjectionSpace")
    if not isinstance(loaded.data, pd.DataFrame):
        raise ValueError("ProjectionSpace requires pandas DataFrame data.")

    coordinate_columns = tuple(
        str(column)
        for column in loaded.data.select_dtypes(include="number").columns
    )
    if len(coordinate_columns) < 2:
        raise ValueError(
            "ProjectionSpace requires at least two numeric coordinate columns."
        )
    return ProjectionSpace(
        artifact=loaded.artifact,
        data=loaded.data,
        metadata=dict(loaded.metadata),
        coordinate_columns=coordinate_columns,
        n_molecules=len(loaded.data),
    )


def shared_molecule_ids(space_a, space_b) -> tuple[str, ...]:
    """Return shared molecule IDs, preserving first-space order.

    Parameters
    ----------
    space_a, space_b : object
        Objects exposing a ``molecule_ids`` attribute.

    Returns
    -------
    tuple of str
        IDs present in both spaces, ordered as in ``space_a``.
    """
    ids_a = tuple(getattr(space_a, "molecule_ids", ()))
    ids_b = set(getattr(space_b, "molecule_ids", ()))
    return tuple(identifier for identifier in ids_a if identifier in ids_b)


def has_aligned_molecule_ids(space_a, space_b) -> bool:
    """Return whether two spaces have the same non-empty molecule ID order.

    Parameters
    ----------
    space_a, space_b : object
        Objects exposing a ``molecule_ids`` attribute.

    Returns
    -------
    bool
        ``True`` only when both ID sequences are non-empty and identical.
    """
    ids_a = tuple(getattr(space_a, "molecule_ids", ()))
    ids_b = tuple(getattr(space_b, "molecule_ids", ()))
    return bool(ids_a) and ids_a == ids_b


def align_spaces(*spaces):
    """Return spaces filtered and reordered to shared molecule IDs.

    Parameters
    ----------
    *spaces : DescriptorSpace, ProjectionSpace, or SimilaritySpace
        Two or more scientific spaces to align.

    Returns
    -------
    tuple
        New space instances of the same types, filtered to shared IDs and
        ordered according to the first space.

    Raises
    ------
    ValueError
        If fewer than two spaces are provided, any space type is unsupported,
        any space has empty ``molecule_ids``, or no IDs are shared.

    Notes
    -----
    Alignment uses existing data only. It does not recalculate descriptors,
    similarity matrices, or projections, and it does not mutate the input
    spaces.
    """
    if len(spaces) < 2:
        raise ValueError("align_spaces requires two or more spaces.")
    for space in spaces:
        _require_supported_space(space)
        if not space.molecule_ids:
            raise ValueError("All spaces must have non-empty molecule_ids.")

    shared = _shared_ids_for_spaces(spaces)
    if not shared:
        raise ValueError("No shared molecule IDs across spaces.")
    return tuple(_align_one_space(space, shared) for space in spaces)


def _molecule_ids_from_table_or_metadata(
    df: pd.DataFrame,
    metadata: dict,
) -> tuple[str, ...]:
    """Derive molecule IDs from common columns or loader metadata."""
    lower_to_column = {str(column).lower(): column for column in df.columns}
    for name in MOLECULE_ID_COLUMNS:
        column = lower_to_column.get(name)
        if column is not None:
            return tuple(df[column].astype(str).tolist())

    ids = metadata.get("ids") if isinstance(metadata, dict) else None
    if ids is None:
        return ()
    return tuple(str(identifier) for identifier in ids)


def _shared_ids_for_spaces(spaces: tuple) -> tuple[str, ...]:
    """Return IDs shared by all spaces, preserving first-space order."""
    first_ids = tuple(spaces[0].molecule_ids)
    other_sets = [set(space.molecule_ids) for space in spaces[1:]]
    return tuple(
        identifier
        for identifier in first_ids
        if all(identifier in ids for ids in other_sets)
    )


def _align_one_space(space, molecule_ids: tuple[str, ...]):
    """Return one aligned copy of a supported space."""
    if isinstance(space, DescriptorSpace):
        data = _align_table(space.data, space.molecule_ids, molecule_ids)
        return DescriptorSpace(
            artifact=space.artifact,
            data=data,
            metadata=dict(space.metadata),
            n_molecules=len(data),
            feature_names=space.feature_names,
        )
    if isinstance(space, ProjectionSpace):
        data = _align_table(space.data, space.molecule_ids, molecule_ids)
        return ProjectionSpace(
            artifact=space.artifact,
            data=data,
            metadata=dict(space.metadata),
            coordinate_columns=space.coordinate_columns,
            n_molecules=len(data),
        )
    if isinstance(space, SimilaritySpace):
        id_to_index = {identifier: idx for idx, identifier in enumerate(space.ids)}
        indices = [id_to_index[identifier] for identifier in molecule_ids]
        matrix = space.matrix[np.ix_(indices, indices)].copy()
        return SimilaritySpace(
            artifact=space.artifact,
            matrix=matrix,
            ids=tuple(molecule_ids),
            metadata=dict(space.metadata),
            n_molecules=len(molecule_ids),
        )
    _require_supported_space(space)


def _align_table(
    df: pd.DataFrame,
    current_ids: tuple[str, ...],
    molecule_ids: tuple[str, ...],
) -> pd.DataFrame:
    """Filter and reorder a table by molecule IDs."""
    id_to_index = {identifier: idx for idx, identifier in enumerate(current_ids)}
    indices = [id_to_index[identifier] for identifier in molecule_ids]
    return df.iloc[indices].reset_index(drop=True).copy()


def _require_supported_space(space) -> None:
    """Raise for unsupported space-like objects."""
    if not isinstance(space, (DescriptorSpace, ProjectionSpace, SimilaritySpace)):
        raise ValueError(f"Unsupported scientific space type: {type(space).__name__}")


def _require_kind(
    loaded: LoadedArtifact,
    expected_kind: str,
    target_name: str,
) -> None:
    """Raise a clear error when an artifact cannot form a science view."""
    if loaded.artifact.kind != expected_kind:
        raise ValueError(
            f"{target_name} requires artifact kind {expected_kind!r}; "
            f"got {loaded.artifact.kind!r}."
        )
