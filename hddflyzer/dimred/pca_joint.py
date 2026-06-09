# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Joint PCA for two collections in BASE and HDDF spaces.

Trains a single PCA on the union of two datasets (A + B), then separates
coordinates by dataset for comparison.

Usage
-----
    python -m hddflyzer.dimred.pca_joint <tag_a> <tag_b>
"""

import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from hddflyzer.config import get_path
from hddflyzer.io import update_manifest
from hddflyzer.utils.naming import sanitize_tag
from hddflyzer.utils.descriptors import get_zero_variance_descriptors
from hddflyzer.dimred.pca_utils import (
    coerce_numeric,
    intersect_columns,
    load_pca_features,
    normalize_column_list,
    normalize_columns,
)


def _load_meta(tag: str) -> Dict:
    path = get_path("pca", tag, "pca_metadata.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"PCA meta not found: {path}\n"
            f"Run PCA first: hddflyzer dimred pca {tag}"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_joint_pca(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    tag_a: str,
    tag_b: str,
    cols_space: List[str],
    n_components: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Train a single PCA on A + B using the given columns.

    Returns combined coordinates, per-dataset coordinates, and metadata.
    """
    df_a = normalize_columns(df_a)
    df_b = normalize_columns(df_b)
    cols_space = normalize_column_list(cols_space)

    cols_a = [c for c in cols_space if c in df_a.columns]
    cols_b = [c for c in cols_space if c in df_b.columns]
    common = intersect_columns(cols_a, cols_b)

    if len(common) < 2:
        raise ValueError(
            f"Fewer than 2 common columns in this space ({len(common)} found).")

    data_a = coerce_numeric(df_a[common]).dropna(how="all")
    data_b = coerce_numeric(df_b[common]).dropna(how="all")

    if len(data_a) < 2 or len(data_b) < 2:
        raise ValueError(
            f"Too few rows after NaN removal (A={len(data_a)}, B={len(data_b)}).")

    idx_a = data_a.index.tolist()
    idx_b = data_b.index.tolist()

    combined = pd.concat([data_a, data_b], axis=0, ignore_index=True)
    combined_imp = combined.fillna(combined.median())

    zero_vars = get_zero_variance_descriptors(combined_imp)
    used_cols = [c for c in common if c not in zero_vars]

    if len(used_cols) < 2:
        raise ValueError("Nearly all columns have zero variance in this space.")

    n_a = len(data_a)
    combined_final = combined_imp[used_cols].copy()
    scaled = StandardScaler().fit_transform(combined_final.values)

    max_pcs = min(n_components, combined_final.shape[0], combined_final.shape[1])
    pca = PCA(n_components=max_pcs, svd_solver="full")
    scores = pca.fit_transform(scaled)

    pc_cols = [f"PC{i + 1}" for i in range(max_pcs)]
    coords_a = pd.DataFrame(scores[:n_a], columns=pc_cols)
    coords_b = pd.DataFrame(scores[n_a:], columns=pc_cols)
    coords_a.insert(0, "dataset", tag_a)
    coords_b.insert(0, "dataset", tag_b)
    coords_a.insert(0, "row_id", [str(i) for i in idx_a])
    coords_b.insert(0, "row_id", [str(i) for i in idx_b])

    coords_joint = pd.concat([coords_a, coords_b], ignore_index=True)

    meta = {
        "tags": {"A": tag_a, "B": tag_b},
        "n_samples": {
            "A": int(len(data_a)),
            "B": int(len(data_b)),
            "total": int(len(data_a) + len(data_b)),
        },
        "used_columns": used_cols,
        "excluded_zero_variance": zero_vars,
        "explained_variance_ratio": list(map(
            float, np.round(pca.explained_variance_ratio_, 6))),
        "notes": "Joint PCA trained on A + B.",
    }

    return coords_joint, coords_a, coords_b, meta


def save_joint_outputs(
    out_dir: str,
    pair_tag: str,
    coords_joint: pd.DataFrame,
    coords_a: pd.DataFrame,
    coords_b: pd.DataFrame,
    meta: Dict,
) -> List[str]:
    """Save joint PCA outputs and return generated file paths."""
    os.makedirs(out_dir, exist_ok=True)
    joint_path = os.path.join(out_dir, f"joint_pca_coordinates_{pair_tag}.csv")
    a_path = os.path.join(out_dir, f"coords_{meta['tags']['A']}.csv")
    b_path = os.path.join(out_dir, f"coords_{meta['tags']['B']}.csv")
    meta_path = os.path.join(out_dir, "joint_pca_metadata.json")

    coords_joint.to_csv(joint_path, index=False)
    coords_a.to_csv(a_path, index=False)
    coords_b.to_csv(b_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"  Saved to: {out_dir}/")
    return [joint_path, a_path, b_path, meta_path]


def run(tag_a: str, tag_b: str) -> bool:
    """
    Run joint PCA for two collections in BASE and HDDF spaces.

    Reads  : results/{tag}/features/full/features_*.csv  (both tags)
             results/{tag}/dimred/pca/pca_metadata.json   (both tags)
    Writes : results/{tag_a}_vs_{tag_b}/dimred/pca_joint/BASE/
             results/{tag_a}_vs_{tag_b}/dimred/pca_joint/HDDF/
    """
    pair_tag = f"{tag_a}_vs_{tag_b}"
    root_out = get_path("pca_joint", pair_tag)

    try:
        df_a = load_pca_features(tag_a)
        df_b = load_pca_features(tag_b)
        meta_a = _load_meta(tag_a)
        meta_b = _load_meta(tag_b)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return False

    base_a = normalize_column_list(meta_a.get("base_valid", []))
    base_b = normalize_column_list(meta_b.get("base_valid", []))
    hddf_a = normalize_column_list(meta_a.get("hddf_cols", []))
    hddf_b = normalize_column_list(meta_b.get("hddf_cols", []))

    success = True
    generated_files = []

    for space, cols_a, cols_b in [
        ("BASE", base_a, base_b),
        ("HDDF", hddf_a, hddf_b),
    ]:
        print(f"\n==> Joint PCA - {space} space")
        out_dir = os.path.join(root_out, space)
        cols_space = intersect_columns(cols_a, cols_b)
        try:
            coords_joint, coords_a, coords_b, meta = run_joint_pca(
                df_a, df_b, tag_a, tag_b, cols_space)
            generated_files.extend(save_joint_outputs(
                out_dir, pair_tag, coords_joint, coords_a, coords_b, meta))
        except (ValueError, Exception) as e:
            print(f"[ERROR] {space} PCA failed: {e}")
            success = False

    if success:
        update_manifest(
            pair_tag,
            "dimred.pca_joint",
            generated_files,
            {"tag_a": tag_a, "tag_b": tag_b, "spaces": ["BASE", "HDDF"]},
        )
        print(f"\n[OK] Joint PCA complete: {root_out}/")
    return success


def main():
    if len(sys.argv) != 3:
        print("Usage: python -m hddflyzer.dimred.pca_joint <tag_a> <tag_b>")
        print("Example: python -m hddflyzer.dimred.pca_joint aocd dianatdb")
        sys.exit(1)

    tag_a = sanitize_tag(sys.argv[1])
    tag_b = sanitize_tag(sys.argv[2])
    if not run(tag_a, tag_b):
        sys.exit(1)


if __name__ == "__main__":
    main()
