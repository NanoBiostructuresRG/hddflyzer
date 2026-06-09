# SPDX-License-Identifier: LGPL-3.0-or-later

"""
PCA analysis for individual collections.

Computes PCA in two descriptor spaces:
  - BASE : all constitutional + topological + electronic + geometrical + hybrid
  - HDDF : QED, LeadLikeness_Score, Pharma_Complexity,
            Synthetic_Accessibility, Desirability_Profile

HDDF PCs are sign-aligned to BASE PCs for direct comparison.
The alignment is a sign flip only; no rotation or rescaling is applied.

Usage
-----
    python -m hddflyzer.dimred.pca <tag>
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

from hddflyzer.config import get_path
from hddflyzer.io import resolve_features_csv
from hddflyzer.utils.naming import sanitize_tag
from hddflyzer.io import update_manifest
from hddflyzer.utils.descriptors import (
    categorize_descriptors,
    get_zero_variance_descriptors,
)
from hddflyzer.dimred.pca_utils import coerce_numeric


# ============================================================
# ALIGNMENT
# ============================================================

def align_pc_signs(
    base_scores: np.ndarray,
    hddf_scores: np.ndarray,
) -> np.ndarray:
    """
    Flip sign of HDDF PCs so they correlate positively with BASE PCs.
    """
    aligned = hddf_scores.copy()
    for j in range(min(base_scores.shape[1], hddf_scores.shape[1])):
        r = np.corrcoef(base_scores[:, j], hddf_scores[:, j])[0, 1]
        if np.isfinite(r) and r < 0:
            aligned[:, j] *= -1.0
    return aligned


def pc_sign_alignment_diagnostics(
    base_scores: np.ndarray,
    hddf_scores: np.ndarray,
) -> dict:
    """Return sign flips and correlations used to align HDDF PCs to BASE PCs."""
    rows = []
    for j in range(min(base_scores.shape[1], hddf_scores.shape[1])):
        r_raw = float(np.corrcoef(base_scores[:, j], hddf_scores[:, j])[0, 1])
        sign = -1.0 if np.isfinite(r_raw) and r_raw < 0 else 1.0
        r_aligned = r_raw * sign if np.isfinite(r_raw) else np.nan
        rows.append({
            "component": f"PC{j + 1}",
            "raw_correlation_base_vs_hddf": round(r_raw, 6) if np.isfinite(r_raw) else None,
            "sign_multiplier": sign,
            "aligned_correlation_base_vs_hddf": round(r_aligned, 6) if np.isfinite(r_aligned) else None,
        })
    return {
        "method": "component-wise sign flip",
        "description": (
            "HDDF PCA scores are multiplied by -1 only when their same-numbered "
            "BASE PC has negative correlation. PCA signs are arbitrary; this "
            "makes visual comparisons easier without rotating or rescaling scores."
        ),
        "applies_rotation": False,
        "applies_rescaling": False,
        "components": rows,
    }


# ============================================================
# CORE COMPUTATION
# ============================================================

def compute_pca(
    df: pd.DataFrame,
    tag: str,
    output_dir: str,
    n_components: int = 2,
) -> bool:
    """
    Compute PCA for BASE and HDDF descriptor spaces.

    Parameters
    ----------
    df         : features DataFrame
    tag        : dataset tag
    output_dir : where to save outputs
    n_components : number of PCA components

    Outputs
    -------
    pca_coordinates.csv
    pca_metadata.json
    """
    df = coerce_numeric(df)

    zero_vars = get_zero_variance_descriptors(
        df,
        exclude_columns=["SMILES", "identifier", "row_id"],
    )
    cats = categorize_descriptors(df.columns.tolist(), zero_vars)

    hddf_cols  = cats["hddf"]
    base_valid = [c for c in cats["all_base"] if c not in zero_vars]

    if len(hddf_cols) < 2:
        print(f"[ERROR] Insufficient HDDF descriptors ({len(hddf_cols)}).")
        return False
    if len(base_valid) < 2:
        print("[ERROR] Insufficient BASE descriptors after filtering.")
        return False

    # Common index — no NaNs in BASE + HDDF + color variables
    color_cols = ["QED", "Desirability_Profile"]
    must_have  = sorted(set(base_valid + hddf_cols + color_cols))
    idx = df[must_have].dropna().index

    if len(idx) < 2:
        print("[ERROR] Too few samples after NaN removal.")
        return False

    print(f"[INFO] {len(idx)} samples | {len(base_valid)} BASE | {len(hddf_cols)} HDDF")

    Xb  = StandardScaler().fit_transform(df.loc[idx, base_valid].values)
    Xh  = StandardScaler().fit_transform(df.loc[idx, hddf_cols].values)
    qed  = df.loc[idx, "QED"].values
    adme = df.loc[idx, "Desirability_Profile"].values

    pca_base = PCA(n_components=n_components, svd_solver="full")
    pca_hddf = PCA(n_components=n_components, svd_solver="full")
    Xb_2d = pca_base.fit_transform(Xb)
    Xh_2d = pca_hddf.fit_transform(Xh)
    Xh_aligned = align_pc_signs(Xb_2d, Xh_2d)
    alignment = pc_sign_alignment_diagnostics(Xb_2d, Xh_2d)

    corr_pc1 = float(abs(pearsonr(Xb_2d[:, 0], Xh_aligned[:, 0])[0]))
    corr_pc2 = float(abs(pearsonr(Xb_2d[:, 1], Xh_aligned[:, 1])[0]))
    base_evr = list(map(float, np.round(pca_base.explained_variance_ratio_, 6)))
    hddf_evr = list(map(float, np.round(pca_hddf.explained_variance_ratio_, 6)))
    base_evr_pct = list(map(float, np.round(100 * pca_base.explained_variance_ratio_, 2)))
    hddf_evr_pct = list(map(float, np.round(100 * pca_hddf.explained_variance_ratio_, 2)))

    # Save coordinates
    os.makedirs(output_dir, exist_ok=True)
    coords = pd.DataFrame({
        "row_id":             idx.astype(str),
        "PC1_base":           Xb_2d[:, 0],
        "PC2_base":           Xb_2d[:, 1],
        "PC1_hddf_aligned":   Xh_aligned[:, 0],
        "PC2_hddf_aligned":   Xh_aligned[:, 1],
        "QED":                qed,
        "Desirability_Profile": adme,
    })
    coords_path = os.path.join(output_dir, "pca_coordinates.csv")
    coords.to_csv(coords_path, index=False)

    base_distance = np.sqrt(np.sum(Xb_2d ** 2, axis=1))
    hddf_distance = np.sqrt(np.sum(Xh_aligned ** 2, axis=1))

    def _top_distances(distances: np.ndarray) -> list:
        order = np.argsort(distances)[::-1][:10]
        return [
            {
                "row_id": str(idx[i]),
                "score_distance": float(round(distances[i], 6)),
                "QED": float(round(qed[i], 6)),
                "Desirability_Profile": float(round(adme[i], 6)),
            }
            for i in order
        ]

    # Save metadata
    meta = {
        "tag":        tag,
        "n_samples":  int(len(idx)),
        "input_table": "features/full/features.csv",
        "preprocessing": {
            "numeric_coercion": True,
            "scaler": "StandardScaler per descriptor space",
            "nan_policy": "rows with NaN in selected BASE, HDDF, QED, or Desirability_Profile columns are removed",
            "zero_variance_policy": "zero-variance descriptors are excluded before PCA",
        },
        "coordinate_columns": {
            "row_id": "row index from the input feature table",
            "PC1_base": "PC1 score from intrinsic BASE descriptor PCA",
            "PC2_base": "PC2 score from intrinsic BASE descriptor PCA",
            "PC1_hddf_aligned": "PC1 score from HDDF PCA after sign alignment to PC1_base",
            "PC2_hddf_aligned": "PC2 score from HDDF PCA after sign alignment to PC2_base",
            "QED": "Drug-likeness score copied for interpretation/coloring",
            "Desirability_Profile": "HDDF desirability score copied for interpretation/coloring",
        },
        "base_valid": base_valid,
        "hddf_cols":  hddf_cols,
        "categories": {
            k: [c for c in cats[k] if c in base_valid]
            for k in ["constitutional", "topological", "electronic", "hybrid_base"]
        },
        "explained_variance_ratio": {
            "base": base_evr,
            "hddf": hddf_evr,
        },
        "explained_variance_percent": {
            "base": base_evr_pct,
            "hddf": hddf_evr_pct,
        },
        "corr_abs_between_spaces": {
            "PC1": round(corr_pc1, 6),
            "PC2": round(corr_pc2, 6),
        },
        "alignment": alignment,
        "outlier_screen": {
            "method": "largest Euclidean distance from origin in the first two PCA dimensions",
            "interpretation": "large distances are candidates for chemical inspection, not automatic errors",
            "base_top10": _top_distances(base_distance),
            "hddf_aligned_top10": _top_distances(hddf_distance),
        },
        "pca_base_components": [
            list(map(float, row)) for row in pca_base.components_
        ],
        "notes": (
            "BASE and HDDF are separate PCA spaces. HDDF-aligned scores use only "
            "component-wise sign flips for visual comparability; PCA sign is "
            "arbitrary, and no rotation or rescaling is applied."
        ),
    }
    meta_path = os.path.join(output_dir, "pca_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    update_manifest(tag, "dimred.pca", [coords_path, meta_path], meta)

    print(f"[OK] PCA complete.")
    print(f"     Coordinates : {coords_path}")
    print(f"     Metadata    : {meta_path}")
    print(f"     BASE EVR    : {[round(float(v), 3) for v in pca_base.explained_variance_ratio_]}")
    print(f"     HDDF EVR    : {[round(float(v), 3) for v in pca_hddf.explained_variance_ratio_]}")
    print(f"     |r| PC1/PC2 : {corr_pc1:.3f} / {corr_pc2:.3f}")
    return True


# ============================================================
# PIPELINE
# ============================================================

def run(tag: str) -> bool:
    """
    Run PCA analysis for a tag.

    Reads  : results/{tag}/features/full/features_*.csv
    Writes : results/{tag}/dimred/pca/
    """
    try:
        input_file = resolve_features_csv(tag)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return False

    df = pd.read_csv(input_file)
    return compute_pca(df, tag, output_dir=get_path("pca", tag))


# ============================================================
# CLI
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m hddflyzer.dimred.pca <tag>")
        print("Example: python -m hddflyzer.dimred.pca aocd")
        sys.exit(1)

    tag = sanitize_tag(sys.argv[1])
    if not run(tag):
        sys.exit(1)


if __name__ == "__main__":
    main()
