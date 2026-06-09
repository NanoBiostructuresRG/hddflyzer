# SPDX-License-Identifier: LGPL-3.0-or-later

"""Visualization input resolution from reconstructed workflow runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from hddflyzer.results import WorkflowRun


@dataclass(frozen=True)
class VizInputs:
    """Resolved file inputs for visualization code.

    Attributes
    ----------
    category : str
        Workflow category used to resolve inputs.
    root : pathlib.Path
        Root directory of the reconstructed run.
    paths : tuple of pathlib.Path
        Existing input paths selected from the run manifest.
    """

    category: str
    root: Path
    paths: tuple[Path, ...]

    def as_dict(self) -> dict:
        """Return a serializable representation of the resolved inputs.

        Returns
        -------
        dict
            Dictionary with ``category``, ``root``, and string paths.
        """
        return {
            "category": self.category,
            "root": str(self.root),
            "paths": [str(path) for path in self.paths],
        }


def resolve_viz_inputs(
    run: WorkflowRun,
    category: str | None = None,
    required: str | Iterable[str] | None = None,
    kind: str | None = None,
) -> VizInputs:
    """Resolve visualization input paths from a reconstructed workflow run.

    Parameters
    ----------
    run : WorkflowRun
        Reconstructed run whose manifest contains registered outputs.
    category : str, optional
        Output category to select, such as ``"chem"`` or ``"dimred"``.
    required : str or iterable of str, optional
        Required path fragment or fragments used to select specific files.
    kind : str, optional
        Semantic artifact kind to select through ``run.artifacts``.

    Returns
    -------
    VizInputs
        Existing paths suitable for visualization functions.

    Raises
    ------
    ValueError
        If no outputs or artifacts match the requested category/kind.
    FileNotFoundError
        If ``required`` filters match nothing or a registered input path is
        missing on disk.

    Notes
    -----
    This function resolves inputs from an existing ``WorkflowRun``. It does not
    create files, run pipeline stages, or generate plots.
    """
    if kind is not None:
        artifacts = run.artifacts(kind=kind, category=category)
        if not artifacts:
            detail = f"kind: {kind}"
            if category is not None:
                detail += f", category: {category}"
            raise ValueError(f"No registered artifacts for visualization {detail}")
        outputs = [artifact.relative_path for artifact in artifacts]
        resolved_category = category or artifacts[0].category
    elif category is not None:
        outputs = run.outputs(category=category)
        resolved_category = category
    else:
        outputs = run.outputs()
        resolved_category = "all"

    if not outputs:
        raise ValueError(
            f"No registered outputs for visualization category: {resolved_category}")

    selected = _filter_required(outputs, required)
    if required is not None and not selected:
        raise FileNotFoundError(
            f"No registered output matching {required!r} in category: {resolved_category}")

    root = run.manifest_path.parent
    paths = tuple(root / output for output in selected)
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Registered visualization input missing: {missing_list}")

    return VizInputs(category=resolved_category, root=root, paths=paths)


def _filter_required(
    outputs: list[str],
    required: str | Iterable[str] | None,
) -> list[str]:
    """Return outputs matching required path fragments."""
    if required is None:
        return list(outputs)
    if isinstance(required, str):
        fragments = (required,)
    else:
        fragments = tuple(required)
    return [
        output
        for output in outputs
        if any(fragment in output for fragment in fragments)
    ]
