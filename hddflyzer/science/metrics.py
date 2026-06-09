# SPDX-License-Identifier: LGPL-3.0-or-later

"""Structural metrics between aligned scientific spaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .spaces import DescriptorSpace, ProjectionSpace, SimilaritySpace, align_spaces


DESCRIPTOR_GROUP_COLUMNS = [
    "group",
    "feature",
    "n",
    "mean",
    "median",
    "std",
    "delta_from_global_mean",
    "abs_delta_from_global_mean",
]


@dataclass(frozen=True)
class SpaceMetricResult:
    """Scalar result of a structural comparison between scientific spaces.

    Attributes
    ----------
    name : str
        Metric name.
    value : float
        Scalar metric value. Some metrics may return ``nan`` when a correlation
        is undefined.
    metadata : dict
        Metric metadata such as molecule counts, pair counts, or coordinate
        columns.
    """

    name: str
    value: float
    metadata: dict


@dataclass(frozen=True)
class DescriptorProjectionCorrelationResult:
    """Ranked descriptor/projection coordinate correlations.

    Attributes
    ----------
    data : pandas.DataFrame
        Ranked table with descriptor-coordinate correlation rows.
    metadata : dict
        Result metadata including molecule count, feature names, and coordinate
        columns.
    """

    data: pd.DataFrame
    metadata: dict

    def top_features(self, n: int = 10) -> pd.DataFrame:
        """Return the top ranked descriptor-coordinate rows.

        Parameters
        ----------
        n : int, default=10
            Number of rows to return.

        Returns
        -------
        pandas.DataFrame
            Copy of the top ``n`` rows.

        Raises
        ------
        ValueError
            If ``n`` is not a positive integer.
        """
        if not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer.")
        return self.data.head(n).copy()


@dataclass(frozen=True)
class DescriptorGroupComparisonResult:
    """Ranked descriptor differences for explicit groups.

    Attributes
    ----------
    data : pandas.DataFrame
        Group-feature summary table ranked by absolute deviation from the
        global descriptor mean.
    metadata : dict
        Result metadata including molecule count, retained groups, feature
        names, and minimum group size.
    """

    data: pd.DataFrame
    metadata: dict

    def top_differences(self, n: int = 10) -> pd.DataFrame:
        """Return the top group-feature differences.

        Parameters
        ----------
        n : int, default=10
            Number of rows to return.

        Returns
        -------
        pandas.DataFrame
            Copy of the top ``n`` rows.

        Raises
        ------
        ValueError
            If ``n`` is not a positive integer.
        """
        if not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer.")
        return self.data.head(n).copy()


@dataclass(frozen=True)
class NeighborhoodPreservationResult:
    """Per-molecule neighborhood preservation diagnostics.

    Attributes
    ----------
    data : pandas.DataFrame
        Per-molecule table with neighbor overlap counts, fractions, and
        neighbor ID lists.
    metadata : dict
        Result metadata including molecule count, ``k``, and coordinate
        columns.
    """

    data: pd.DataFrame
    metadata: dict

    def worst_preserved(self, n: int = 10) -> pd.DataFrame:
        """Return molecules with the lowest neighborhood overlap.

        Parameters
        ----------
        n : int, default=10
            Number of rows to return.

        Returns
        -------
        pandas.DataFrame
            Copy of the lowest-overlap rows, sorted by overlap fraction and
            molecule ID.

        Raises
        ------
        ValueError
            If ``n`` is not a positive integer.
        """
        if not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer.")
        return self.data.sort_values(
            ["overlap_fraction", "molecule_id"],
            ascending=[True, True],
        ).head(n).reset_index(drop=True).copy()


def compare_descriptor_groups(
    descriptors: DescriptorSpace,
    labels,
    *,
    min_group_size: int = 2,
) -> DescriptorGroupComparisonResult:
    """Compare numeric descriptors across explicit groups.

    Parameters
    ----------
    descriptors : DescriptorSpace
        Descriptor space containing numeric descriptor columns.
    labels : str or sequence
        Group labels. A string is interpreted as a column name in
        ``descriptors.data``. A sequence must be aligned with
        ``descriptors.molecule_ids`` and have length ``descriptors.n_molecules``.
    min_group_size : int, default=2
        Minimum number of molecules required for a group to be included.

    Returns
    -------
    DescriptorGroupComparisonResult
        Ranked group-feature summary table and metadata.

    Raises
    ------
    ValueError
        If ``descriptors`` is not a ``DescriptorSpace``, no numeric descriptor
        features exist, labels are invalid, or ``min_group_size`` is invalid.

    Notes
    -----
    This function compares groups explicitly provided by the user or by an
    existing column. It does not perform clustering, enrichment, or automatic
    chemical interpretation.
    """
    if not isinstance(descriptors, DescriptorSpace):
        raise ValueError("Expected DescriptorSpace for descriptors.")
    if not isinstance(min_group_size, int) or min_group_size < 1:
        raise ValueError("min_group_size must be a positive integer.")

    feature_names = _numeric_descriptor_features(descriptors)
    if not feature_names:
        raise ValueError("DescriptorSpace has no numeric descriptor features.")

    label_values = _resolve_group_labels(descriptors, labels)
    df = descriptors.data.loc[:, list(feature_names)].copy()
    df["_group"] = label_values
    global_means = df.loc[:, list(feature_names)].mean(numeric_only=True)

    rows = []
    groups = []
    for group, group_df in df.groupby("_group", sort=True, dropna=True):
        if len(group_df) < min_group_size:
            continue
        group_name = str(group)
        groups.append(group_name)
        for feature in feature_names:
            values = group_df[feature]
            mean = float(values.mean())
            delta = mean - float(global_means[feature])
            rows.append({
                "group": group_name,
                "feature": feature,
                "n": int(values.count()),
                "mean": mean,
                "median": float(values.median()),
                "std": float(values.std(ddof=1)) if values.count() > 1 else float("nan"),
                "delta_from_global_mean": delta,
                "abs_delta_from_global_mean": abs(delta),
            })

    data = pd.DataFrame(rows, columns=DESCRIPTOR_GROUP_COLUMNS)
    if not data.empty:
        data = data.sort_values(
            ["abs_delta_from_global_mean", "group", "feature"],
            ascending=[False, True, True],
            na_position="last",
        ).reset_index(drop=True)
    return DescriptorGroupComparisonResult(
        data=data,
        metadata={
            "n_molecules": descriptors.n_molecules,
            "groups": tuple(groups),
            "feature_names": tuple(feature_names),
            "min_group_size": min_group_size,
        },
    )


def descriptor_projection_correlations(
    descriptors: DescriptorSpace,
    projection: ProjectionSpace,
) -> DescriptorProjectionCorrelationResult:
    """Correlate numeric descriptors with projection coordinates.

    Parameters
    ----------
    descriptors : DescriptorSpace
        Descriptor space containing numeric descriptor columns.
    projection : ProjectionSpace
        Projection space with at least two coordinate columns.

    Returns
    -------
    DescriptorProjectionCorrelationResult
        Ranked descriptor-coordinate correlations.

    Raises
    ------
    ValueError
        If inputs have wrong types, molecule IDs cannot be aligned, no numeric
        descriptor features exist, or the projection lacks sufficient
        coordinates.

    Notes
    -----
    Inputs are aligned by molecule ID before calculation. The function uses
    existing descriptor values and projection coordinates only; it does not
    recalculate descriptors or projections and does not make automatic chemical
    interpretations.
    """
    if not isinstance(descriptors, DescriptorSpace):
        raise ValueError("Expected DescriptorSpace for descriptors.")
    if not isinstance(projection, ProjectionSpace):
        raise ValueError("Expected ProjectionSpace for projection.")
    descriptors, projection = align_spaces(descriptors, projection)

    feature_names = _numeric_descriptor_features(descriptors)
    if not feature_names:
        raise ValueError("DescriptorSpace has no numeric descriptor features.")
    coords = _projection_coordinates(projection)
    coordinate_columns = tuple(projection.coordinate_columns)

    rows = []
    for feature in feature_names:
        feature_values = descriptors.data[feature].to_numpy(dtype=float)
        for idx, coordinate in enumerate(coordinate_columns):
            coord_values = coords[:, idx]
            correlation = _safe_correlation(feature_values, coord_values)
            rows.append({
                "feature": feature,
                "coordinate": coordinate,
                "correlation": correlation,
                "abs_correlation": abs(correlation),
            })

    data = pd.DataFrame(rows).sort_values(
        ["abs_correlation", "feature", "coordinate"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    return DescriptorProjectionCorrelationResult(
        data=data,
        metadata={
            "n_molecules": descriptors.n_molecules,
            "feature_names": tuple(feature_names),
            "coordinate_columns": coordinate_columns,
        },
    )


def projection_neighborhood_preservation(
    similarity: SimilaritySpace,
    projection: ProjectionSpace,
    k: int = 10,
) -> NeighborhoodPreservationResult:
    """Evaluate local neighbor preservation for each molecule.

    Parameters
    ----------
    similarity : SimilaritySpace
        Similarity space containing an existing pairwise similarity matrix.
    projection : ProjectionSpace
        Projection space containing existing coordinates.
    k : int, default=10
        Number of neighbors to compare for each molecule.

    Returns
    -------
    NeighborhoodPreservationResult
        Per-molecule overlap diagnostics and metadata.

    Raises
    ------
    ValueError
        If inputs have wrong types, IDs cannot be aligned, ``k`` is invalid, or
        the projection lacks sufficient coordinates.

    Notes
    -----
    This diagnostic compares neighbors from existing similarity and projection
    spaces. It does not recalculate fingerprints, similarity, projections, or
    clusters.
    """
    similarity, projection = _aligned_similarity_projection(
        similarity,
        projection,
    )
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")
    n = similarity.n_molecules
    if k >= n:
        raise ValueError("k must be smaller than the number of molecules.")

    coords = _projection_coordinates(projection)
    distances = _pairwise_distances(coords)
    rows = []
    ids = tuple(similarity.molecule_ids)
    for idx, molecule_id in enumerate(ids):
        sim_neighbor_idx = _ordered_similarity_neighbors(
            similarity.matrix,
            idx,
            k,
        )
        proj_neighbor_idx = _ordered_projection_neighbors(distances, idx, k)
        sim_neighbors = [ids[j] for j in sim_neighbor_idx]
        proj_neighbors = [ids[j] for j in proj_neighbor_idx]
        sim_set = set(sim_neighbors)
        proj_set = set(proj_neighbors)
        overlap = sim_set & proj_set
        rows.append({
            "molecule_id": molecule_id,
            "overlap_count": len(overlap),
            "overlap_fraction": len(overlap) / k,
            "similarity_neighbors": sim_neighbors,
            "projection_neighbors": proj_neighbors,
            "missing_similarity_neighbors": [
                neighbor for neighbor in sim_neighbors if neighbor not in proj_set
            ],
            "extra_projection_neighbors": [
                neighbor for neighbor in proj_neighbors if neighbor not in sim_set
            ],
        })

    return NeighborhoodPreservationResult(
        data=pd.DataFrame(rows),
        metadata={
            "k": k,
            "n_molecules": n,
            "coordinate_columns": projection.coordinate_columns,
        },
    )


def similarity_projection_correlation(
    similarity: SimilaritySpace,
    projection: ProjectionSpace,
) -> SpaceMetricResult:
    """Correlate pairwise similarity with projected-space proximity.

    Parameters
    ----------
    similarity : SimilaritySpace
        Similarity space containing an existing pairwise similarity matrix.
    projection : ProjectionSpace
        Projection space containing existing coordinates.

    Returns
    -------
    SpaceMetricResult
        Scalar correlation result and metadata.

    Raises
    ------
    ValueError
        If inputs have wrong types, molecule IDs cannot be aligned, fewer than
        two aligned molecules are available, or projection coordinates are
        insufficient.

    Notes
    -----
    The function aligns inputs by molecule ID and uses existing artifacts only.
    It does not recalculate similarity or projections.
    """
    similarity, projection = _aligned_similarity_projection(
        similarity,
        projection,
    )
    coords = _projection_coordinates(projection)
    distances = _pairwise_distances(coords)
    sim_values = _upper_triangle_values(similarity.matrix)
    proximity_values = _upper_triangle_values(-distances)
    if sim_values.size == 0:
        raise ValueError("At least two aligned molecules are required.")
    if np.std(sim_values) == 0.0 or np.std(proximity_values) == 0.0:
        value = float("nan")
    else:
        value = float(np.corrcoef(sim_values, proximity_values)[0, 1])
    return SpaceMetricResult(
        name="similarity_projection_correlation",
        value=value,
        metadata={
            "n_molecules": similarity.n_molecules,
            "n_pairs": int(sim_values.size),
            "coordinate_columns": projection.coordinate_columns,
        },
    )


def similarity_projection_neighbor_overlap(
    similarity: SimilaritySpace,
    projection: ProjectionSpace,
    k: int = 10,
) -> SpaceMetricResult:
    """Return mean overlap between similarity and projection neighbors.

    Parameters
    ----------
    similarity : SimilaritySpace
        Similarity space containing an existing pairwise similarity matrix.
    projection : ProjectionSpace
        Projection space containing existing coordinates.
    k : int, default=10
        Number of neighbors to compare for each molecule.

    Returns
    -------
    SpaceMetricResult
        Mean neighbor-overlap fraction and metadata.

    Raises
    ------
    ValueError
        If inputs have wrong types, molecule IDs cannot be aligned, ``k`` is
        invalid, or projection coordinates are insufficient.

    Notes
    -----
    The function uses existing similarity and projection artifacts only. It
    does not perform clustering or automatic chemical interpretation.
    """
    similarity, projection = _aligned_similarity_projection(
        similarity,
        projection,
    )
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")
    n = similarity.n_molecules
    if k >= n:
        raise ValueError("k must be smaller than the number of molecules.")

    coords = _projection_coordinates(projection)
    distances = _pairwise_distances(coords)
    overlaps = []
    for idx in range(n):
        sim_neighbors = _top_similarity_neighbors(similarity.matrix, idx, k)
        proj_neighbors = _nearest_projection_neighbors(distances, idx, k)
        overlaps.append(len(sim_neighbors & proj_neighbors) / k)

    return SpaceMetricResult(
        name="similarity_projection_neighbor_overlap",
        value=float(np.mean(overlaps)),
        metadata={
            "k": k,
            "n_molecules": n,
            "coordinate_columns": projection.coordinate_columns,
        },
    )


def _numeric_descriptor_features(descriptors: DescriptorSpace) -> tuple[str, ...]:
    """Return numeric descriptor feature columns."""
    numeric_columns = set(descriptors.data.select_dtypes(include="number").columns)
    return tuple(
        feature
        for feature in descriptors.feature_names
        if feature in numeric_columns
    )


def _resolve_group_labels(descriptors: DescriptorSpace, labels) -> pd.Series:
    """Resolve group labels from a column name or aligned sequence."""
    if isinstance(labels, str):
        if labels not in descriptors.data.columns:
            raise ValueError(f"Label column not found: {labels}")
        return descriptors.data[labels].astype(str)

    try:
        values = list(labels)
    except TypeError as e:
        raise ValueError("labels must be a column name or sequence.") from e
    if len(values) != descriptors.n_molecules:
        raise ValueError(
            "labels sequence length must match DescriptorSpace.n_molecules."
        )
    return pd.Series([str(value) for value in values], index=descriptors.data.index)


def _safe_correlation(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Return Pearson correlation, or NaN for constant/invalid vectors."""
    valid = np.isfinite(values_a) & np.isfinite(values_b)
    if valid.sum() < 2:
        return float("nan")
    a = values_a[valid]
    b = values_b[valid]
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _aligned_similarity_projection(
    similarity: SimilaritySpace,
    projection: ProjectionSpace,
) -> tuple[SimilaritySpace, ProjectionSpace]:
    """Align and type-check a similarity/projection pair."""
    if not isinstance(similarity, SimilaritySpace):
        raise ValueError("Expected SimilaritySpace for similarity.")
    if not isinstance(projection, ProjectionSpace):
        raise ValueError("Expected ProjectionSpace for projection.")
    aligned_similarity, aligned_projection = align_spaces(similarity, projection)
    return aligned_similarity, aligned_projection


def _projection_coordinates(projection: ProjectionSpace) -> np.ndarray:
    """Return numeric projection coordinates from a ProjectionSpace."""
    columns = tuple(projection.coordinate_columns)
    if len(columns) < 2:
        raise ValueError("ProjectionSpace must have at least two coordinates.")
    missing = [column for column in columns if column not in projection.data.columns]
    if missing:
        raise ValueError(f"Projection coordinate columns missing: {missing}")
    coords = projection.data.loc[:, list(columns)].to_numpy(dtype=float)
    if coords.shape[1] < 2:
        raise ValueError("ProjectionSpace must have at least two coordinates.")
    return coords


def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """Return Euclidean pairwise distances for rows of coords."""
    delta = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


def _upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    """Return upper-triangle values excluding the diagonal."""
    rows, cols = np.triu_indices(matrix.shape[0], k=1)
    return np.asarray(matrix[rows, cols], dtype=float)


def _top_similarity_neighbors(
    matrix: np.ndarray,
    idx: int,
    k: int,
) -> set[int]:
    """Return indices of the top-k similarity neighbors excluding self."""
    candidates = [j for j in range(matrix.shape[0]) if j != idx]
    ordered = sorted(candidates, key=lambda j: (-matrix[idx, j], j))
    return set(ordered[:k])


def _nearest_projection_neighbors(
    distances: np.ndarray,
    idx: int,
    k: int,
) -> set[int]:
    """Return indices of the nearest projected neighbors excluding self."""
    candidates = [j for j in range(distances.shape[0]) if j != idx]
    ordered = sorted(candidates, key=lambda j: (distances[idx, j], j))
    return set(ordered[:k])


def _ordered_similarity_neighbors(
    matrix: np.ndarray,
    idx: int,
    k: int,
) -> list[int]:
    """Return ordered top-k similarity neighbor indices excluding self."""
    candidates = [j for j in range(matrix.shape[0]) if j != idx]
    return sorted(candidates, key=lambda j: (-matrix[idx, j], j))[:k]


def _ordered_projection_neighbors(
    distances: np.ndarray,
    idx: int,
    k: int,
) -> list[int]:
    """Return ordered nearest projected neighbor indices excluding self."""
    candidates = [j for j in range(distances.shape[0]) if j != idx]
    return sorted(candidates, key=lambda j: (distances[idx, j], j))[:k]
