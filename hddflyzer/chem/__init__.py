# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Chemistry module for HDDFlyzer.

Submodules
----------
feature_engineering : Intrinsic descriptors and reference similarity features
feature_curation    : ML-ready curated descriptor tables
stats               : Pearson/Spearman correlations (BASE and HDDF modes)
pruning             : Remove redundant descriptors via correlation graph
npclassifier        : Optional NPClassifier annotation via external API
tanimoto            : Compute Morgan fingerprint Tanimoto matrices
tanimoto_sampling   : Stratified sampling from Tanimoto matrices

Note: feature_engineering and tanimoto require RDKit.
All other submodules are RDKit-independent.
"""

# RDKit-independent imports (always available)
from .stats import (
    HDDF_COLUMNS,
    HDDF_NAMES,
    NONCORR_THRESHOLD,
    run_base_stats,
    run_hddf_stats,
)

from .pruning import (
    DEFAULT_THRESHOLD,
    run as run_pruning,
)

from .npclassifier import (
    NPClassifierClient,
    run as run_npclassifier,
)

from .tanimoto_sampling import (
    DEFAULTS as TANIMOTO_SAMPLING_DEFAULTS,
    stratify,
    balanced_sample,
    run as run_tanimoto_sampling,
)

from .feature_curation import (
    curate_features,
    run as run_feature_curation,
)

_LAZY_EXPORTS = {
    "REFERENCE_MOLECULES": ("hddflyzer.chem.reference_catalog", "REFERENCE_MOLECULES"),
    "calculate_molecular_descriptors": (
        "hddflyzer.chem.feature_engineering",
        "calculate_molecular_descriptors",
    ),
    "calculate_fingerprint_similarities": (
        "hddflyzer.chem.feature_engineering",
        "calculate_fingerprint_similarities",
    ),
    "calculate_mcs_features": (
        "hddflyzer.chem.feature_engineering",
        "calculate_mcs_features",
    ),
    "run_feature_engineering": ("hddflyzer.chem.feature_engineering", "run"),
    "run_reference_features": (
        "hddflyzer.chem.feature_engineering",
        "run_reference_features",
    ),
    "build_morgan_fps": ("hddflyzer.chem.tanimoto", "build_morgan_fps"),
    "compute_tanimoto_matrix": ("hddflyzer.chem.tanimoto", "compute_tanimoto_matrix"),
    "stats_from_matrix": ("hddflyzer.chem.tanimoto", "stats_from_matrix"),
    "run_tanimoto": ("hddflyzer.chem.tanimoto", "run"),
}

__all__ = [
    "HDDF_COLUMNS",
    "HDDF_NAMES",
    "NONCORR_THRESHOLD",
    "run_base_stats",
    "run_hddf_stats",
    "DEFAULT_THRESHOLD",
    "run_pruning",
    "NPClassifierClient",
    "run_npclassifier",
    "TANIMOTO_SAMPLING_DEFAULTS",
    "stratify",
    "balanced_sample",
    "run_tanimoto_sampling",
    "curate_features",
    "run_feature_curation",
    *_LAZY_EXPORTS,
]


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
