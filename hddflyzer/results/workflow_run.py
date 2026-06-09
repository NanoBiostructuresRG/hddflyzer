# SPDX-License-Identifier: LGPL-3.0-or-later

"""Reconstruct completed HDDFlyzer runs from manifest.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Iterable

from hddflyzer.config import settings
from hddflyzer.utils.naming import validate_tag

from .artifacts import (
    KIND_DESCRIPTOR_TABLE,
    KIND_PROJECTION_COORDINATES,
    KIND_TANIMOTO_MATRIX,
    LoadedArtifact,
    ResultArtifact,
    classify_artifact,
)


STAGE_PREFIXES = {
    "registry": ("data.",),
    "registry/data": ("data.",),
    "data": ("data.",),
    "chem": ("annotate.", "chem."),
    "dimred": ("dimred.",),
    "viz": ("viz.",),
    "viz/figures": ("viz.",),
}


@dataclass(frozen=True)
class WorkflowRun:
    """Queryable view of a completed HDDFlyzer run.

    ``WorkflowRun`` reconstructs a completed run from an existing
    ``results/<tag>/manifest.json`` file. It does not execute pipeline stages,
    recalculate outputs, or create new state.

    Attributes
    ----------
    tag : str
        Dataset tag recorded in the manifest.
    manifest_path : pathlib.Path
        Path to the reconstructed ``manifest.json`` file.
    manifest : dict
        Parsed manifest content.
    """

    tag: str
    manifest_path: Path
    manifest: dict[str, Any]

    @property
    def operations(self) -> list[dict[str, Any]]:
        """list of dict: Operation records from the manifest."""
        return list(self.manifest.get("operations", []))

    @property
    def current_outputs(self) -> list[str]:
        """list of str: Currently registered manifest output paths."""
        return list(self.manifest.get("current_outputs", []))

    @property
    def output_categories(self) -> dict[str, list[str]]:
        """dict: Output paths grouped by workflow area."""
        return {
            category: list(outputs)
            for category, outputs in self.manifest.get(
                "output_categories", {}
            ).items()
        }

    @property
    def workflow_contract(self) -> dict[str, Any]:
        """dict: Workflow contract recorded in the manifest."""
        return dict(self.manifest.get("workflow_contract", {}))

    def outputs(self, category: str | None = None) -> list[str]:
        """Return output paths from the reconstructed manifest.

        Parameters
        ----------
        category : str, optional
            Workflow category to select. When omitted, all current outputs are
            returned.

        Returns
        -------
        list of str
            Manifest-relative output paths.
        """
        if category is None:
            return self.current_outputs
        return list(self.output_categories.get(category, []))

    def operations_by_stage(self, stage: str) -> list[dict[str, Any]]:
        """Return operation records associated with a workflow stage.

        Parameters
        ----------
        stage : str
            Workflow stage or area, such as ``"chem"`` or ``"dimred"``.

        Returns
        -------
        list of dict
            Matching operation records in manifest order.
        """
        operation_names = set()
        for stage_info in self.workflow_contract.get("stages", []):
            if stage_info.get("stage") == stage:
                operation_names.update(stage_info.get("operations", []))

        prefixes = STAGE_PREFIXES.get(stage, (f"{stage}.",))
        matches = []
        for operation in self.operations:
            name = operation.get("operation", "")
            if name in operation_names or name.startswith(prefixes):
                matches.append(operation)
        return matches

    def operation_metadata(self, operation_name: str) -> dict[str, Any] | None:
        """Return metadata for the latest recorded operation.

        Parameters
        ----------
        operation_name : str
            Operation name, for example ``"chem.features"``.

        Returns
        -------
        dict or None
            Operation metadata when present.
        """
        for operation in reversed(self.operations):
            if operation.get("operation") == operation_name:
                metadata = operation.get("metadata")
                return dict(metadata) if isinstance(metadata, dict) else metadata
        return None

    def artifacts(
        self,
        kind: str | None = None,
        category: str | None = None,
        operation: str | None = None,
        required: str | Iterable[str] | None = None,
    ) -> list[ResultArtifact]:
        """Return semantic result artifacts derived from manifest outputs.

        Parameters
        ----------
        kind : str, optional
            Semantic artifact kind to select.
        category : str, optional
            Output category to select.
        operation : str, optional
            Producing operation to select.
        required : str or iterable of str, optional
            Required path fragment or fragments used to disambiguate outputs.

        Returns
        -------
        list of ResultArtifact
            Matching artifacts with resolved paths, categories, kinds, and
            operation metadata.

        Raises
        ------
        ValueError
            If a manifest output path is absolute, contains traversal, or would
            resolve outside the run directory.
        """
        artifacts = []
        output_categories = self.output_categories
        operation_index = _operation_index(self.operations)
        for relative_path in self.current_outputs:
            artifact_path = _resolve_artifact_path(
                self.manifest_path.parent,
                relative_path,
            )
            display_path = _display_path(relative_path)
            artifact_category = _category_for_output(
                relative_path, output_categories)
            operation_record = operation_index.get(relative_path)
            operation_name = (
                operation_record.get("operation")
                if operation_record is not None
                else None
            )
            metadata = (
                operation_record.get("metadata")
                if operation_record is not None
                else None
            )
            artifact = ResultArtifact(
                path=artifact_path,
                relative_path=display_path,
                category=artifact_category,
                kind=classify_artifact(relative_path, operation_name),
                operation=operation_name,
                metadata=dict(metadata) if isinstance(metadata, dict) else metadata,
            )
            if kind is not None and artifact.kind != kind:
                continue
            if category is not None and artifact.category != category:
                continue
            if operation is not None and artifact.operation != operation:
                continue
            if required is not None and not _matches_required(
                artifact.relative_path,
                required,
            ):
                continue
            artifacts.append(artifact)
        return artifacts

    def artifact(
        self,
        kind: str | None = None,
        category: str | None = None,
        operation: str | None = None,
        required: str | Iterable[str] | None = None,
    ) -> ResultArtifact:
        """Return exactly one artifact matching the requested filters.

        Parameters
        ----------
        kind, category, operation, required
            Filters passed to ``artifacts``.

        Returns
        -------
        ResultArtifact
            The single matching artifact.

        Raises
        ------
        FileNotFoundError
            If no artifact matches the filters.
        ValueError
            If multiple artifacts match the filters.
        """
        matches = self.artifacts(
            kind=kind,
            category=category,
            operation=operation,
            required=required,
        )
        if not matches:
            raise FileNotFoundError(
                "No artifact found"
                f"{_format_filters(kind, category, operation, required)}."
            )
        if len(matches) > 1:
            candidates = ", ".join(artifact.relative_path for artifact in matches)
            raise ValueError(
                "Multiple artifacts found"
                f"{_format_filters(kind, category, operation, required)}: "
                f"{candidates}"
            )
        return matches[0]

    def load_artifact(
        self,
        kind: str | None = None,
        category: str | None = None,
        operation: str | None = None,
        required: str | Iterable[str] | None = None,
        allow_pickle: bool = False,
    ) -> LoadedArtifact:
        """Select and load exactly one artifact.

        Parameters
        ----------
        kind, category, operation, required
            Filters passed to ``artifact``.
        allow_pickle : bool, default=False
            Whether pickle-backed table artifacts may be loaded. Enable only
            for trusted local files.

        Returns
        -------
        LoadedArtifact
            Loaded data, metadata, and source artifact.

        Raises
        ------
        FileNotFoundError
            If no artifact matches or a required file is missing.
        ValueError
            If multiple artifacts match, loading is unsupported, pickle loading
            is blocked, or loaded data is invalid.
        """
        from .artifacts import load_artifact

        return load_artifact(
            self.artifact(
                kind=kind,
                category=category,
                operation=operation,
                required=required,
            ),
            allow_pickle=allow_pickle,
        )

    def descriptor_space(
        self,
        category: str | None = None,
        operation: str | None = None,
        required: str | Iterable[str] | None = None,
        allow_pickle: bool = False,
    ) -> "DescriptorSpace":
        """Load a descriptor-table artifact as a scientific descriptor space.

        Parameters
        ----------
        category, operation, required
            Artifact filters used to select one descriptor table.
        allow_pickle : bool, default=False
            Whether pickle-backed descriptor tables may be loaded.

        Returns
        -------
        DescriptorSpace
            Descriptor-space view over an existing loaded artifact.

        Notes
        -----
        This method does not recalculate descriptors.
        """
        from hddflyzer.science import to_descriptor_space

        loaded = self.load_artifact(
            kind=KIND_DESCRIPTOR_TABLE,
            category=category,
            operation=operation,
            required=required,
            allow_pickle=allow_pickle,
        )
        return to_descriptor_space(loaded)

    def similarity_space(
        self,
        category: str | None = None,
        operation: str | None = None,
        required: str | Iterable[str] | None = None,
        allow_pickle: bool = False,
    ) -> "SimilaritySpace":
        """Load a Tanimoto matrix artifact as a scientific similarity space.

        Parameters
        ----------
        category, operation, required
            Artifact filters used to select one Tanimoto matrix.
        allow_pickle : bool, default=False
            Present for API consistency. Tanimoto matrices are loaded from
            ``.npz`` with pickle disabled.

        Returns
        -------
        SimilaritySpace
            Similarity-space view over an existing Tanimoto matrix.

        Notes
        -----
        This method does not recalculate fingerprints or similarity.
        """
        from hddflyzer.science import to_similarity_space

        loaded = self.load_artifact(
            kind=KIND_TANIMOTO_MATRIX,
            category=category,
            operation=operation,
            required=required,
            allow_pickle=allow_pickle,
        )
        return to_similarity_space(loaded)

    def projection_space(
        self,
        category: str | None = None,
        operation: str | None = None,
        required: str | Iterable[str] | None = None,
        allow_pickle: bool = False,
    ) -> "ProjectionSpace":
        """Load projection coordinates as a scientific projection space.

        Parameters
        ----------
        category, operation, required
            Artifact filters used to select one projection-coordinate table.
        allow_pickle : bool, default=False
            Whether pickle-backed projection tables may be loaded.

        Returns
        -------
        ProjectionSpace
            Projection-space view over existing dimensionality-reduction
            coordinates.

        Notes
        -----
        This method does not recalculate PCA, t-SNE, UMAP, or other
        projections.
        """
        from hddflyzer.science import to_projection_space

        loaded = self.load_artifact(
            kind=KIND_PROJECTION_COORDINATES,
            category=category,
            operation=operation,
            required=required,
            allow_pickle=allow_pickle,
        )
        return to_projection_space(loaded)

    def summary(self) -> dict[str, Any]:
        """Return a compact programmatic summary of the reconstructed run.

        Returns
        -------
        dict
            Summary containing tag, manifest path, operation count, current
            output count, output categories, and workflow contract.
        """
        return {
            "tag": self.tag,
            "manifest_path": str(self.manifest_path),
            "n_operations": len(self.operations),
            "n_current_outputs": len(self.current_outputs),
            "output_categories": self.output_categories,
            "workflow_contract": self.workflow_contract,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of this reconstructed run.

        Returns
        -------
        dict
            Dictionary containing tag, manifest path, and manifest content.
        """
        return {
            "tag": self.tag,
            "manifest_path": str(self.manifest_path),
            "manifest": dict(self.manifest),
        }


def _display_path(path: str) -> str:
    """Return manifest paths with platform-neutral separators."""
    return path.replace("\\", "/")


def _operation_index(operations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map relative output paths to the latest operation that registered them."""
    index = {}
    for operation in operations:
        for relative_path in operation.get("files", []):
            index[_display_path(relative_path)] = operation
    return index


def _category_for_output(
    relative_path: str,
    output_categories: dict[str, list[str]],
) -> str:
    """Return the manifest output category for a relative path."""
    target = _display_path(relative_path)
    for category, outputs in output_categories.items():
        if target in {_display_path(output) for output in outputs}:
            return category
    return "unknown"


def _format_filters(
    kind: str | None,
    category: str | None,
    operation: str | None,
    required: str | Iterable[str] | None = None,
) -> str:
    """Format artifact filters for user-facing errors."""
    filters = []
    if kind is not None:
        filters.append(f"kind={kind!r}")
    if category is not None:
        filters.append(f"category={category!r}")
    if operation is not None:
        filters.append(f"operation={operation!r}")
    if required is not None:
        filters.append(f"required={required!r}")
    return f" for {', '.join(filters)}" if filters else ""


def _matches_required(
    relative_path: str,
    required: str | Iterable[str],
) -> bool:
    """Return True when a manifest path matches required fragments."""
    path = _display_path(relative_path)
    if isinstance(required, str):
        fragments = (required,)
    else:
        fragments = tuple(required)
    return any(_display_path(fragment) in path for fragment in fragments)


def _resolve_artifact_path(run_root: Path, relative_path: str) -> Path:
    """Resolve a manifest output path while enforcing run containment."""
    display_path = _display_path(relative_path)
    candidate = Path(display_path)
    if candidate.is_absolute():
        raise ValueError(f"Manifest output path must be relative: {relative_path}")
    if ".." in candidate.parts:
        raise ValueError(f"Manifest output path contains traversal: {relative_path}")
    root = run_root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as e:
        raise ValueError(
            f"Manifest output path escapes run directory: {relative_path}"
        ) from e
    return resolved


def load_workflow_run(
    tag: str,
    results_dir: Path | str | None = None,
) -> WorkflowRun:
    """Load a completed run from ``results/<tag>/manifest.json``.

    Parameters
    ----------
    tag : str
        Run tag to reconstruct. Tags are validated and must not contain path
        separators, traversal, or absolute paths.
    results_dir : pathlib.Path or str, optional
        Root results directory. Defaults to ``hddflyzer.config.settings.RESULTS_DIR``.

    Returns
    -------
    WorkflowRun
        Reconstructed run backed by the parsed manifest.

    Raises
    ------
    FileNotFoundError
        If ``manifest.json`` does not exist.
    ValueError
        If the tag is invalid, the resolved manifest escapes ``results_dir``,
        or the manifest JSON/structure is invalid.
    """
    tag = validate_tag(tag)
    root = Path(results_dir) if results_dir is not None else Path(settings.RESULTS_DIR)
    root = root.resolve()
    manifest_path = (root / tag / "manifest.json").resolve()
    try:
        manifest_path.relative_to(root)
    except ValueError as e:
        raise ValueError(f"Manifest path escapes results directory: {tag}") from e
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except JSONDecodeError as e:
        raise ValueError(f"Invalid manifest JSON: {manifest_path}") from e

    _validate_manifest(manifest, manifest_path)
    return WorkflowRun(
        tag=str(manifest.get("tag", tag)),
        manifest_path=manifest_path,
        manifest=manifest,
    )


def _validate_manifest(manifest: Any, manifest_path: Path) -> None:
    """Validate the minimal manifest structure required for reconstruction."""
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must be a JSON object: {manifest_path}")
    if not isinstance(manifest.get("tag"), str):
        raise ValueError(f"Manifest missing string tag: {manifest_path}")
    if not isinstance(manifest.get("operations"), list):
        raise ValueError(f"Manifest missing operations list: {manifest_path}")
    if not isinstance(manifest.get("current_outputs"), list):
        raise ValueError(
            f"Manifest missing current_outputs list: {manifest_path}")
    if not isinstance(manifest.get("workflow_contract"), dict):
        raise ValueError(
            f"Manifest missing workflow_contract object: {manifest_path}")
    output_categories = manifest.get("output_categories", {})
    if not isinstance(output_categories, dict):
        raise ValueError(
            f"Manifest output_categories must be an object: {manifest_path}")
