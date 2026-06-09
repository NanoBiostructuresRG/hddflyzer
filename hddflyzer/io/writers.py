# SPDX-License-Identifier: LGPL-3.0-or-later

"""Canonical writers for HDDFlyzer pipeline artifacts."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

from hddflyzer.config import get_dataset_path
from hddflyzer.utils.naming import validate_tag


CANONICAL_WORKFLOW = ["registry", "chem", "dimred", "viz", "metadata/results"]
WORKFLOW_SUMMARY_FILENAME = "workflow_summary.md"

WORKFLOW_STAGE_PREFIXES = {
    "registry": ("data.",),
    "chem": ("annotate.", "chem."),
    "dimred": ("dimred.",),
    "viz": ("viz.",),
}

OUTPUT_CATEGORY_NAMES = [
    "registry/data",
    "chem",
    "dimred",
    "viz/figures",
    "metadata",
]


def write_json(path: str, data) -> str:
    """Write JSON with stable formatting and return the output path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def _manifest_file_state(tag: str, operations: list) -> dict:
    """Return current files and stale operation references for a dataset."""
    root = get_dataset_path(tag)
    current_outputs = []
    if os.path.isdir(root):
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                full = os.path.join(dirpath, filename)
                rel = os.path.relpath(full, root)
                if rel == "manifest.json":
                    continue
                current_outputs.append(rel)

    stale = []
    for op in operations:
        operation = op.get("operation")
        for rel in op.get("files", []):
            if rel and not os.path.exists(os.path.join(root, rel)):
                stale.append({"operation": operation, "file": rel})

    return {
        "current_outputs": sorted(current_outputs),
        "stale_operation_files": stale,
    }


def _display_path(path: str) -> str:
    """Return a platform-neutral relative path for user-facing metadata."""
    return path.replace("\\", "/")


def _output_category(path: str) -> str:
    """Classify an output path into a workflow area."""
    path = _display_path(path)
    name = os.path.basename(path)
    if path == WORKFLOW_SUMMARY_FILENAME or name.endswith("_metadata.json"):
        return "metadata"
    if path.startswith(("registry/", "data/")):
        return "registry/data"
    if path.startswith(("annotations/", "chemistry/", "features/")):
        return "chem"
    if path.startswith("dimred/"):
        return "dimred"
    if path.startswith(("figures/", "plots/", "similarity/")):
        return "viz/figures"
    return "metadata"


def _categorize_outputs(outputs: list) -> dict:
    """Group output paths by workflow area using existing relative paths."""
    categories = {name: [] for name in OUTPUT_CATEGORY_NAMES}
    for output in outputs:
        display_output = _display_path(output)
        categories[_output_category(display_output)].append(display_output)
    return {
        name: sorted(paths)
        for name, paths in categories.items()
        if paths
    }


def _workflow_contract(tag: str, operations: list) -> dict:
    """Return the canonical workflow contract represented by a manifest."""
    operation_names = [
        op.get("operation")
        for op in operations
        if op.get("operation")
    ]
    stages = []
    for stage, prefixes in WORKFLOW_STAGE_PREFIXES.items():
        stage_operations = [
            name for name in operation_names if name.startswith(prefixes)
        ]
        stages.append({
            "stage": stage,
            "ran": bool(stage_operations),
            "operations": stage_operations,
        })

    return {
        "identity": "HDDFLYZER canonical workflow contract",
        "collection": {"tag": tag},
        "canonical_workflow": CANONICAL_WORKFLOW,
        "stages": stages,
        "operation_history": "operations",
        "parameters_index": "operations[].metadata",
        "outputs_index": "current_outputs",
    }


def _render_workflow_summary(manifest: dict) -> str:
    """Render a concise human-readable workflow summary from a manifest."""
    contract = manifest["workflow_contract"]
    lines = [
        f"# HDDFLYZER Workflow Summary: {contract['collection']['tag']}",
        "",
        "## Collection",
        "",
        f"- Tag: `{contract['collection']['tag']}`",
        "",
        "## Canonical Workflow",
        "",
        " -> ".join(contract["canonical_workflow"]),
        "",
        "## Detected Workflow Stages",
        "",
    ]

    for stage in contract["stages"]:
        status = "ran" if stage["ran"] else "not detected"
        operations = ", ".join(stage["operations"]) or "none"
        lines.append(
            f"- {stage['stage']}: {status}; operations: {operations}")

    lines.extend([
        "",
        "## Registered Current Outputs",
        "",
    ])
    output_categories = manifest.get("output_categories", {})
    if output_categories:
        for category in OUTPUT_CATEGORY_NAMES:
            outputs = output_categories.get(category, [])
            if not outputs:
                continue
            lines.extend([f"### {category}", ""])
            lines.extend(f"- `{output}`" for output in outputs)
            lines.append("")
    else:
        lines.append("- none")

    lines.extend([
        "",
        "## Manifest",
        "",
        "- `manifest.json`",
        "",
        "Reconstruction details live in manifest operations, operation "
        "metadata, and current_outputs.",
        "",
    ])
    return "\n".join(lines)


def _write_workflow_summary(tag: str, manifest: dict) -> str:
    """Write results/{tag}/workflow_summary.md from manifest data."""
    path = get_dataset_path(tag, WORKFLOW_SUMMARY_FILENAME)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(_render_workflow_summary(manifest))
    return path


def update_manifest(tag: str, operation: str, files: Iterable[str], metadata=None) -> str:
    """Update results/{tag}/manifest.json with history and current file state."""
    tag = validate_tag(tag)
    path = get_dataset_path(tag, "manifest.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {"tag": tag, "operations": []}

    record = {
        "operation": operation,
        "timestamp": datetime.now().isoformat(),
        "files": [_relative_output_path(tag, f) for f in files if f],
    }
    if metadata:
        record["metadata"] = metadata

    manifest["updated_at"] = record["timestamp"]
    manifest["operations"].append(record)
    manifest.update(_manifest_file_state(tag, manifest["operations"]))
    manifest["workflow_contract"] = _workflow_contract(
        tag, manifest["operations"])
    if WORKFLOW_SUMMARY_FILENAME not in manifest["current_outputs"]:
        manifest["current_outputs"].append(WORKFLOW_SUMMARY_FILENAME)
        manifest["current_outputs"].sort()
    manifest["output_categories"] = _categorize_outputs(
        manifest["current_outputs"])
    write_json(path, manifest)
    _write_workflow_summary(tag, manifest)
    return path


def _relative_output_path(tag: str, file_path: str) -> str:
    """Return a manifest-safe relative output path inside results/{tag}."""
    root = Path(get_dataset_path(tag)).resolve()
    candidate = Path(file_path)
    resolved = candidate.resolve()
    if not _is_relative_to(resolved, root) and not candidate.is_absolute():
        resolved = (root / candidate).resolve()
    if not _is_relative_to(resolved, root):
        raise ValueError(
            f"Manifest output is outside results/{tag}: {file_path}")
    return os.path.relpath(str(resolved), str(root))


def _is_relative_to(path: Path, root: Path) -> bool:
    """Return True when path is inside root."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
