# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Similarity visualizations.

tanimoto    : Histogram + boxplot of Tanimoto pairwise similarities,
              with optional hierarchical clustering and reordering.
              (plot_tanimoto_cheminfo.py)

fingerprints: Violin + boxplot comparing 7 fingerprint types.
              (plot_fingerprint_comparison.py)

Usage
-----
    python -m hddflyzer.viz.similarity tanimoto    <tag>
    python -m hddflyzer.viz.similarity fingerprints <tag>
"""

import os
import sys
import re
import time
import glob
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from typing import Optional

from hddflyzer.config import get_path
from hddflyzer.utils.naming import sanitize_tag
from hddflyzer.viz.colors import TAG_COLORS
from hddflyzer.io import load_tanimoto

# ============================================================
# CONFIG
# ============================================================

CLUSTER_DIST      = 0.4     # 1 - Tanimoto >= 0.60
DIVERSITY_THRESHOLD = 0.5
MAX_N_CLUSTERING  = 2_000
MAX_N_CSV         = 2_000

FINGERPRINT_COLS = [
    "morgan_tanimoto", "featmorgan_tanimoto", "atompair_tanimoto",
    "rdk_tanimoto",    "torsion_tanimoto",    "layered_tanimoto",
    "maccs_tanimoto",
]
FP_NAMES = {
    "morgan_tanimoto":     "Morgan",
    "featmorgan_tanimoto": "FeatMorgan",
    "atompair_tanimoto":   "AtomPair",
    "rdk_tanimoto":        "RDKFingerprint",
    "torsion_tanimoto":    "Torsion",
    "layered_tanimoto":    "Layered",
    "maccs_tanimoto":      "MACCS",
}


# ============================================================
# TANIMOTO — HELPERS
# ============================================================

def _load_tanimoto_inputs(tag: str):
    base     = get_path("tanimoto", re.sub(r"[^\w\-]", "_", tag))
    mat, ids, _, _ = load_tanimoto(tag)
    return base, mat, ids


def _similarity_figures_dir(tag: str) -> str:
    """Canonical directory for similarity visualization outputs."""
    return get_path("figures", tag, "similarity")


def _order_by_clustering(mat: np.ndarray, threshold: float):
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    dist      = np.asarray(1.0 - mat, dtype=np.float32)
    condensed = squareform(dist, checks=False)
    Z         = linkage(condensed, method="average", optimal_ordering=True)
    labels    = fcluster(Z, t=threshold, criterion="distance")

    order = []
    for cid in np.unique(labels):
        idx = np.where(labels == cid)[0]
        if len(idx) == 1:
            order.extend(idx.tolist())
        else:
            sim_mean = np.mean(mat[np.ix_(idx, idx)], axis=1)
            order.extend(idx[np.argsort(-sim_mean)].tolist())
    return Z, np.array(order), labels


def _identify_unique(mat: np.ndarray, ids, threshold: float):
    m = mat.copy()
    np.fill_diagonal(m, 0.0)
    max_sim   = m.max(axis=1)
    unique_idx = np.where(max_sim < threshold)[0]
    unique     = [(ids[i], float(max_sim[i])) for i in unique_idx]
    return sorted(unique, key=lambda x: x[1]), max_sim


# ============================================================
# TANIMOTO — PLOTS
# ============================================================

def _plot_histogram(vals: np.ndarray, out_path: str, tag: str) -> None:
    color = TAG_COLORS.get(tag.lower(), TAG_COLORS["default"])
    fig, axs = plt.subplots(2, 1, figsize=(10, 6),
                             gridspec_kw={"height_ratios": [3, 1]},
                             sharex=True)
    sns.set_style("whitegrid")

    axs[0].hist(vals, bins=50, edgecolor="black", color=color, alpha=0.7,
                label=f"{tag.upper()} (n={len(vals):,} pairs)")
    axs[0].axvline(x=DIVERSITY_THRESHOLD, color="red", linestyle="--", linewidth=1.0)
    axs[0].text(DIVERSITY_THRESHOLD + 0.01,
                axs[0].get_ylim()[1] * 0.8, "THRES", color="red", fontsize=12)
    axs[0].yaxis.set_major_locator(MaxNLocator(nbins=6))
    axs[0].set_ylabel("Number of pairs", fontsize=18, fontname="Arial")
    axs[0].tick_params(axis="y", labelsize=16)
    axs[0].set_xlim(0, 1.0)
    axs[0].grid(False)
    axs[0].legend(loc="upper right", fontsize=12, frameon=False)
    for sp in axs[0].spines.values():
        sp.set_color("black"); sp.set_linewidth(2.0)

    bp_props = dict(facecolor="lightgray", edgecolor="black", linewidth=1.0)
    axs[1].boxplot(vals, vert=False, widths=0.7, showfliers=False,
                   patch_artist=True,
                   boxprops=bp_props,
                   medianprops={"color": "black", "linewidth": 2.0},
                   whiskerprops={"color": "black", "linewidth": 1.0},
                   capprops={"color": "black", "linewidth": 1.0})

    if len(vals) < 10_000:
        axs[1].scatter(vals, np.random.normal(1, 0.04, size=len(vals)),
                       alpha=0.5, color=color, s=1.5, rasterized=True)

    axs[1].set_xlabel("Tanimoto similarity", fontsize=18, fontname="Arial")
    axs[1].tick_params(axis="x", labelsize=16)
    axs[1].tick_params(axis="y", left=False, labelleft=False)
    axs[1].set_xlim(0, 1.0)
    for sp in axs[1].spines.values():
        sp.set_color("black"); sp.set_linewidth(1.0)

    fig.subplots_adjust(left=0.12, right=0.98, top=0.97, bottom=0.12)
    plt.savefig(out_path, dpi=500)
    plt.close()


# ============================================================
# TANIMOTO — PIPELINE
# ============================================================

def plot_tanimoto(tag: str) -> bool:
    """
    Generate Tanimoto histogram.

    Reads  : results/{tag}/chemistry/tanimoto/tanimoto_matrix.npz
             results/{tag}/chemistry/tanimoto/tanimoto_ids.csv
    Writes : results/{tag}/figures/similarity/
               tanimoto_histogram.png
    """
    try:
        base, mat, ids = _load_tanimoto_inputs(tag)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return False

    n = mat.shape[0]
    print(f"[INFO] Matrix: {n}x{n}")
    t0 = time.time()

    iu, ju = np.triu_indices(n, k=1)
    vals   = mat[iu, ju]

    # Histogram
    out_dir = _similarity_figures_dir(tag)
    os.makedirs(out_dir, exist_ok=True)
    hist_path = os.path.join(out_dir, "tanimoto_histogram.png")
    _plot_histogram(vals, hist_path, tag)
    print(f"[OK] Histogram: {hist_path}")

    for stale_name in (
        "tanimoto_histogram.png",
        "tanimoto_summary.txt",
        "tanimoto_order.csv",
    ):
        stale_path = os.path.join(base, stale_name)
        if os.path.exists(stale_path):
            os.remove(stale_path)

    print(f"[OK] Done in {time.time() - t0:.1f}s")
    return True


# ============================================================
# FINGERPRINTS — PIPELINE
# ============================================================

def plot_fingerprints(tag: str) -> bool:
    """
    Violin + boxplot comparing 7 fingerprint Tanimoto types.

    Reads  : results/{tag}/features/reference/reference_features_*.csv
    Writes : results/{tag}/figures/similarity/fingerprint_results.png
    """
    pattern = os.path.join(get_path("reference_features", tag), "reference_features_*.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f"[ERROR] No reference-features CSV for tag '{tag}'")
        print(f"        Run: hddflyzer chem reference-features {tag}")
        return False

    df = pd.read_csv(files[0])
    if "rdk_tanimoto" not in df.columns and "rdkit_tanimoto" in df.columns:
        df = df.rename(columns={"rdkit_tanimoto": "rdk_tanimoto"})
    present = [c for c in FINGERPRINT_COLS if c in df.columns]
    if not present:
        print(f"[ERROR] No fingerprint columns found in dataset.")
        return False

    long_parts = [
        pd.DataFrame({
            "Fingerprint": FP_NAMES.get(col, col),
            "Tanimoto":    df[col].dropna().astype(float),
        })
        for col in present
    ]
    df_long = pd.concat(long_parts, ignore_index=True)

    order = (df_long.groupby("Fingerprint")["Tanimoto"]
             .mean()
             .sort_values(ascending=False)
             .index.tolist())

    sns.set_style("whitegrid")
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(10, 6))

    palette_color = TAG_COLORS.get(tag.lower(), sns.color_palette("Set2")[0])

    sns.violinplot(data=df_long, y="Fingerprint", x="Tanimoto",
                   order=order, inner=None, cut=0, ax=ax,
                   linewidth=0.7, color=palette_color)
    sns.boxplot(data=df_long, y="Fingerprint", x="Tanimoto",
                order=order, ax=ax,
                showcaps=True,
                boxprops={"facecolor": "white", "alpha": 0.5},
                whiskerprops={"linewidth": 1},
                medianprops={"linewidth": 2, "color": "black"},
                fliersize=1.5)

    ax.set_ylabel("Fingerprints", fontsize=18, fontname="Arial")
    ax.set_xlabel("Tanimoto score", fontsize=18, fontname="Arial")
    ax.set_xlim(-0.05, 1.05)
    for sp in ax.spines.values():
        sp.set_color("black"); sp.set_linewidth(2.0)

    plt.tight_layout()
    out_dir = _similarity_figures_dir(tag)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fingerprint_results.png")
    legacy_path = os.path.join(get_path("figures", tag), "fingerprint_results.png")
    if os.path.exists(legacy_path) and legacy_path != out_path:
        os.remove(legacy_path)
    plt.savefig(out_path, dpi=500, bbox_inches="tight")
    plt.close()
    print(f"[OK] Fingerprint plot: {out_path}")
    return True


# ============================================================
# CLI
# ============================================================

def main():
    if len(sys.argv) < 3:
        print("Usage: python -m hddflyzer.viz.similarity <mode> <tag>")
        print("  mode: tanimoto | fingerprints")
        print("  Examples:")
        print("    python -m hddflyzer.viz.similarity tanimoto     aocd")
        print("    python -m hddflyzer.viz.similarity fingerprints aocd")
        sys.exit(1)

    mode = sys.argv[1].strip().lower()
    tag  = sanitize_tag(sys.argv[2])

    if mode == "tanimoto":
        ok = plot_tanimoto(tag)
    elif mode in ("fingerprints", "fp"):
        ok = plot_fingerprints(tag)
    else:
        print(f"[ERROR] Unknown mode '{mode}'. Use: tanimoto | fingerprints")
        sys.exit(1)

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
