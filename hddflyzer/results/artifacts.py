# SPDX-License-Identifier: LGPL-3.0-or-later

"""Semantic result artifact views derived from manifest outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np


KIND_MOLECULE_REGISTRY = "molecule_registry"
KIND_DESCRIPTOR_TABLE = "descriptor_table"
KIND_TANIMOTO_MATRIX = "tanimoto_matrix"
KIND_PROJECTION_COORDINATES = "projection_coordinates"
KIND_FIGURE = "figure"
KIND_METADATA = "metadata"
KIND_WORKFLOW_SUMMARY = "workflow_summary"
KIND_UNKNOWN = "unknown"


@dataclass(frozen=True)
class ResultArtifact:
    """A result file with workflow and scientific semantics.

    Attributes
    ----------
    path : pathlib.Path
        Absolute or resolved filesystem path for the artifact.
    relative_path : str
        Manifest-relative path inside ``results/<tag>/``.
    category : str
        Workflow area, such as ``"chem"``, ``"dimred"``, or
        ``"viz/figures"``.
    kind : str
        Semantic artifact kind, for example ``"descriptor_table"`` or
        ``"projection_coordinates"``.
    operation : str or None, default=None
        Manifest operation that produced the artifact, when known.
    metadata : dict or None, default=None
        Operation metadata associated with the artifact, when available.
    """

    path: Path
    relative_path: str
    category: str
    kind: str
    operation: str | None = None
    metadata: dict | None = None


@dataclass(frozen=True)
class LoadedArtifact:
    """Loaded data plus its semantic result artifact.

    Attributes
    ----------
    artifact : ResultArtifact
        Artifact that was loaded.
    data : Any
        Loaded Python object. The type depends on ``artifact.kind``.
    metadata : dict
        Loader metadata and artifact metadata useful for traceability.
    """

    artifact: ResultArtifact
    data: Any
    metadata: dict


def load_artifact(
    artifact: ResultArtifact,
    allow_pickle: bool = False,
) -> LoadedArtifact:
    """Load a supported result artifact into Python data.

    Parameters
    ----------
    artifact : ResultArtifact
        Semantic artifact to load.
    allow_pickle : bool, default=False
        Whether pickle-backed table artifacts may be loaded. Pickle is disabled
        by default and should be enabled only for trusted local files.

    Returns
    -------
    LoadedArtifact
        Loaded data, metadata, and the source artifact.

    Raises
    ------
    FileNotFoundError
        If the artifact file or required companion file is missing.
    ValueError
        If the artifact kind is unsupported, the file format is invalid, pickle
        loading is not allowed, or the loaded data violates its minimum
        contract.

    Notes
    -----
    Supported loaded kinds include descriptor tables, projection coordinates,
    molecule registries, metadata JSON, workflow summaries, and Tanimoto
    matrices. Tanimoto matrices are loaded with ``numpy.load(...,
    allow_pickle=False)``.
    """
    if not artifact.path.exists():
        raise FileNotFoundError(f"Artifact file not found: {artifact.path}")

    metadata = dict(artifact.metadata or {})
    if artifact.kind in {
        KIND_DESCRIPTOR_TABLE,
        KIND_PROJECTION_COORDINATES,
        KIND_MOLECULE_REGISTRY,
    }:
        data = _load_table(artifact.path, allow_pickle=allow_pickle)
        _validate_table_artifact(data, artifact)
    elif artifact.kind == KIND_METADATA:
        data = _load_json(artifact.path)
    elif artifact.kind == KIND_WORKFLOW_SUMMARY:
        data = artifact.path.read_text(encoding="utf-8")
    elif artifact.kind == KIND_TANIMOTO_MATRIX:
        data, ids, ids_path = _load_tanimoto_matrix(artifact.path)
        metadata.update({
            "ids": ids,
            "ids_path": str(ids_path),
            "matrix_shape": list(data.shape),
        })
    else:
        raise ValueError(f"Unsupported artifact kind: {artifact.kind}")

    return LoadedArtifact(artifact=artifact, data=data, metadata=metadata)


def classify_artifact(relative_path: str, operation: str | None = None) -> str:
    """Classify a manifest output path into a semantic artifact kind.

    Parameters
    ----------
    relative_path : str
        Manifest-relative output path.
    operation : str, optional
        Operation name that produced the output, when known.

    Returns
    -------
    str
        Semantic artifact kind. Unknown paths return ``"unknown"``.
    """
    path = relative_path.replace("\\", "/")
    name = Path(path).name.lower()
    suffix = Path(path).suffix.lower()
    operation = operation or ""

    if path == "workflow_summary.md":
        return KIND_WORKFLOW_SUMMARY
    if path == "registry/molecules.csv":
        return KIND_MOLECULE_REGISTRY
    if name == "tanimoto_matrix.npz":
        return KIND_TANIMOTO_MATRIX
    if suffix in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
        return KIND_FIGURE
    if suffix == ".json" or name.endswith("_metadata.json"):
        return KIND_METADATA
    if (
        path.startswith("features/")
        and suffix in {".csv", ".pkl"}
        and "metadata" not in name
    ):
        return KIND_DESCRIPTOR_TABLE
    if (
        operation.startswith("dimred.")
        and suffix == ".csv"
        and (
            "coordinates" in name
            or "embedding" in name
            or "pca" in path
            or "tsne" in path
            or "umap" in path
        )
    ):
        return KIND_PROJECTION_COORDINATES
    if path.startswith("dimred/") and suffix == ".csv":
        return KIND_PROJECTION_COORDINATES
    return KIND_UNKNOWN


def _load_table(path: Path, allow_pickle: bool = False) -> pd.DataFrame:
    """Load a tabular artifact from CSV or pickle."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".pkl":
        if not allow_pickle:
            raise ValueError(
                "Pickle artifact loading is disabled by default. "
                "Pass allow_pickle=True only for trusted local files."
            )
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported table artifact format: {path}")


def _validate_table_artifact(df: pd.DataFrame, artifact: ResultArtifact) -> None:
    """Validate minimal contracts for loaded tabular artifacts."""
    if artifact.kind == KIND_DESCRIPTOR_TABLE:
        _require_non_empty_table(df, "descriptor_table")
    elif artifact.kind == KIND_PROJECTION_COORDINATES:
        _require_non_empty_table(df, "projection_coordinates")
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) < 2:
            raise ValueError(
                "projection_coordinates artifact must contain at least "
                "two numeric columns."
            )
    elif artifact.kind == KIND_MOLECULE_REGISTRY:
        _require_non_empty_table(df, "molecule_registry")
        lower_cols = {str(col).lower() for col in df.columns}
        id_cols = {"identifier", "id", "compound_id", "molecule_id"}
        smiles_cols = {"smiles", "canonical_smiles"}
        if not (lower_cols & id_cols or lower_cols & smiles_cols):
            raise ValueError(
                "molecule_registry artifact must contain an identifier "
                "or SMILES column."
            )


def _require_non_empty_table(df: pd.DataFrame, kind: str) -> None:
    """Require a DataFrame with rows and columns."""
    if df.empty or len(df.columns) == 0:
        raise ValueError(f"{kind} artifact must be a non-empty table.")


def _load_json(path: Path) -> dict:
    """Load a JSON metadata artifact."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except JSONDecodeError as e:
        raise ValueError(f"Invalid metadata JSON: {path}") from e
    if not isinstance(data, dict):
        raise ValueError(f"Metadata JSON must be an object: {path}")
    return data


def _load_tanimoto_matrix(path: Path) -> tuple[np.ndarray, list[str], Path]:
    """Load the canonical Tanimoto matrix and aligned IDs."""
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Unsupported Tanimoto matrix format: {path}")
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as e:
        raise ValueError(f"Invalid Tanimoto matrix file: {path}") from e
    if "matrix" not in data:
        raise ValueError(f"Tanimoto matrix archive missing 'matrix': {path}")

    matrix = np.clip(np.asarray(data["matrix"], dtype=np.float32), 0.0, 1.0)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Tanimoto matrix must be square: {path}")
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 1.0)

    ids_path = path.with_name("tanimoto_ids.csv")
    if not ids_path.exists():
        legacy_ids_path = path.with_name("ids.csv")
        if legacy_ids_path.exists():
            ids_path = legacy_ids_path
        else:
            raise FileNotFoundError(f"Tanimoto IDs file not found: {ids_path}")

    ids_df = pd.read_csv(ids_path)
    id_col = "id" if "id" in ids_df.columns else "identifier"
    if id_col not in ids_df.columns:
        raise ValueError(f"Tanimoto IDs file missing id column: {ids_path}")
    ids = ids_df[id_col].astype(str).tolist()
    if len(ids) != matrix.shape[0]:
        raise ValueError(
            f"IDs ({len(ids)}) and matrix ({matrix.shape[0]}) size mismatch."
        )
    return matrix, ids, ids_path
