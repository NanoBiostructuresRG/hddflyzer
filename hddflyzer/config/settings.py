# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Global path settings for HDDFlyzer.

All pipeline modules read RESULTS_DIR from here. The canonical output layout is
dataset-first:

    results/{tag}/{family}/{process}/...

Historical process-first calls such as get_path("tanimoto", tag) are still
accepted and routed to the canonical layout.
"""

import os

# ============================================================
# ROOT DIRECTORIES
# ============================================================

#: Root directory for all pipeline outputs.
#: Can be overridden via environment variable HDDFLYZER_RESULTS_DIR.
RESULTS_DIR: str = os.environ.get("HDDFLYZER_RESULTS_DIR", "results")

#: Root directory for local input/example CSV files.
DATA_DIR: str = os.environ.get("HDDFLYZER_DATA_DIR", "examples")

#: Legacy feature root kept for compatibility with older imports.
#: New code should use get_features_path(tag).
FEATURES_DIR: str = os.path.join(RESULTS_DIR, "features")

# Backwards-compatible alias for older imports. New code should use get_features_path.
ARTIFACTS_DIR: str = FEATURES_DIR


# ============================================================
# PIPELINE OUTPUT SUBDIRECTORIES
# ============================================================

_ROUTES = {
    "registry": ("registry",),
    "npclassifier": ("annotations", "npclassifier"),
    "tanimoto": ("chemistry", "tanimoto"),
    "features": ("features", "full"),
    "reference_features": ("features", "reference"),
    "features_curated": ("features", "curated"),
    "feature_pruning": ("features", "pruning"),
    "base_correlation": ("features", "correlations"),
    "correlation": ("features", "correlations"),
    "pca": ("dimred", "pca"),
    "pca_joint": ("dimred", "pca_joint"),
    "tsne": ("dimred", "tsne"),
    "umap": ("dimred", "umap"),
    "figures": ("figures",),
}


def get_dataset_path(tag: str, *parts: str) -> str:
    """Build a dataset-first results path: results/{tag}/..."""
    return os.path.join(RESULTS_DIR, tag, *parts)


def get_path(*parts: str) -> str:
    """
    Build a path under RESULTS_DIR.

    For known pipeline roots, historical process-first calls are routed to the
    dataset-first layout. Examples:

        get_path("npclassifier", "aocd")
        -> results/aocd/annotations/npclassifier

        get_path("tanimoto", "aocd", "samples")
        -> results/aocd/chemistry/tanimoto/samples
    """
    if len(parts) >= 2:
        key, tag, rest = parts[0], parts[1], parts[2:]
        route = _ROUTES.get(key)
        if route:
            return get_dataset_path(tag, *route, *rest)
    return os.path.join(RESULTS_DIR, *parts)


def get_features_path(tag: str) -> str:
    """Return feature table path for a tag: results/{tag}/features/full."""
    return get_path("features", tag)


def get_artifacts_path(tag: str) -> str:
    """Backward-compatible alias for get_features_path."""
    return get_features_path(tag)
