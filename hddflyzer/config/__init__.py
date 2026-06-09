# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Static configuration for HDDFlyzer.

Provides the descriptor configuration (categories + human-readable names)
used by the stats and analysis modules.
"""

import json
import os
from typing import Dict, List, Tuple

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "descriptor_config.json")

# Excluded by default from BASE correlation/stats analysis.
# Fingerprints and MCS are dyadic reference features stored separately under
# features/reference/, not intrinsic molecular descriptors.
EXCLUDED_CATEGORIES: List[str] = ["Fingerprints", "MCS", "Hybrid", "HDDF"]

# Individual descriptors excluded regardless of category
EXCLUDED_DESCRIPTORS: List[str] = ["SMR_VSA2"]


def load_descriptor_config(
    path: str = _CONFIG_PATH,
    excluded_categories: List[str] = None,
    excluded_descriptors: List[str] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """
    Load descriptor definitions from the JSON config file.

    Builds BASE_COLUMNS as the union of all categories
    EXCEPT those listed in excluded_categories, then removes
    any individual descriptors in excluded_descriptors.

    Parameters
    ----------
    path : str
        Path to descriptor_config.json. Defaults to the bundled file.
    excluded_categories : list of str, optional
        Category names to skip. Defaults to EXCLUDED_CATEGORIES.
    excluded_descriptors : list of str, optional
        Individual descriptors to remove. Defaults to EXCLUDED_DESCRIPTORS.

    Returns
    -------
    columns : list of str
        Ordered list of valid descriptor names.
    names : dict
        Mapping from descriptor code to human-readable label.

    Raises
    ------
    FileNotFoundError
        If the config file is missing.
    ValueError
        If the config file is unreadable or malformed.
    """
    exc_cats = excluded_categories if excluded_categories is not None else EXCLUDED_CATEGORIES
    exc_desc = excluded_descriptors if excluded_descriptors is not None else EXCLUDED_DESCRIPTORS

    if not os.path.exists(path):
        print(f"[ERROR] Descriptor config not found: {path}")
        raise FileNotFoundError(f"Descriptor config not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"[ERROR] Cannot read descriptor config: {e}")
        raise ValueError(f"Cannot read descriptor config: {e}") from e

    categories = cfg.get("categories")
    names = cfg.get("names", {})

    if not categories or not isinstance(categories, dict):
        print("[ERROR] 'categories' key missing or malformed in descriptor config.")
        raise ValueError("'categories' key missing or malformed in descriptor config.")

    if not names or not isinstance(names, dict):
        print("[ERROR] 'names' key missing or malformed in descriptor config.")
        raise ValueError("'names' key missing or malformed in descriptor config.")

    columns: List[str] = []
    for cat_name, feats in categories.items():
        if cat_name in exc_cats:
            continue
        if not isinstance(feats, list):
            print(f"[WARN] Category '{cat_name}' is not a list — skipping.")
            continue
        columns.extend(feats)

    # Remove individually excluded descriptors
    columns = [c for c in columns if c not in exc_desc]

    if not columns:
        print("[ERROR] No valid columns after applying exclusions.")
        raise ValueError("No valid columns after applying exclusions.")

    # Warn about missing names
    missing = [c for c in columns if c not in names]
    if missing:
        print(f"[WARN] {len(missing)} descriptors have no human-readable name:")
        for c in missing:
            print(f"   - {c}")

    return columns, names


# Eagerly load at import time for convenience
BASE_COLUMNS, BASE_NAMES = load_descriptor_config()

# Settings
from hddflyzer.config.settings import (
    RESULTS_DIR,
    DATA_DIR,
    ARTIFACTS_DIR,
    FEATURES_DIR,
    get_dataset_path,
    get_path,
    get_artifacts_path,
    get_features_path,
)
