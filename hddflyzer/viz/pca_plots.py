# SPDX-License-Identifier: LGPL-3.0-or-later

"""
PCA visualization plots.

analysis    : 6 figures for a single collection
              (BASE/HDDF × QED/Desirability + overlay + correlation).
              (plot_pca_analysis.py)

collections : Joint PCA scatter for two collections in BASE and/or HDDF.
              Uses outputs from `hddflyzer dimred pca-joint`.

Usage
-----
    python -m hddflyzer.viz.pca_plots analysis    <tag>
    python -m hddflyzer.viz.pca_plots collections <tag_a> <tag_b> [--space BASE|HDDF|BOTH]
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import pearsonr
from typing import Optional

from hddflyzer.config import get_path
from hddflyzer.utils.naming import sanitize_tag
from hddflyzer.viz.colors import TAG_COLORS, anabook_palette_8

_HUSL_BASE = sns.color_palette("husl", 8)[3]  # base space color


# ============================================================
# SHARED STYLE
# ============================================================

def _style_ax(ax, xlabel="", ylabel="", tag=""):
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.grid(False)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    for sp in ax.spines.values():
        sp.set_color("black"); sp.set_linewidth(2.0)
    if tag:
        ax.text(0.95, 0.95, tag.upper(), transform=ax.transAxes,
                fontsize=18, fontfamily="Arial", va="top", ha="right")


def _colorbar(sc, ax, label: str):
    cbar = plt.colorbar(sc, ax=ax, ticks=np.arange(0, 1.1, 0.2))
    cbar.set_label(label, fontsize=18)
    cbar.ax.tick_params(labelsize=16)


# ============================================================
# SINGLE COLLECTION ANALYSIS
# ============================================================

def plot_analysis(tag: str) -> bool:
    """
    Generate 6 PCA figures for a single collection.

    Reads  : results/{tag}/dimred/pca/pca_coordinates.csv
             results/{tag}/dimred/pca/pca_metadata.json
    Writes : results/{tag}/figures/dimred/pca/
               pca_base_qed_{tag}.png
               pca_base_desirability_{tag}.png
               pca_hddf_qed_{tag}.png
               pca_hddf_desirability_{tag}.png
               pca_comparison_{tag}.png
               pca_correlation_{tag}.png
    """
    pca_dir = get_path("pca", tag)
    output_dir = get_path("figures", tag, "dimred", "pca")
    coords_path = os.path.join(pca_dir, "pca_coordinates.csv")
    meta_path = os.path.join(pca_dir, "pca_metadata.json")

    for p in (coords_path, meta_path):
        if not os.path.exists(p):
            print(f"[ERROR] File not found: {p}")
            print(f"  Run PCA first: python -m hddflyzer.dimred.pca {tag}")
            return False

    os.makedirs(output_dir, exist_ok=True)

    coords = pd.read_csv(coords_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    Xb    = coords[["PC1_base", "PC2_base"]].values
    Xh    = coords[["PC1_hddf_aligned", "PC2_hddf_aligned"]].values
    qed   = coords["QED"].values
    deseo = coords["Desirability_Profile"].values

    evr_base = meta["explained_variance_ratio"]["base"]
    evr_hddf = meta["explained_variance_ratio"]["hddf"]
    c1 = meta["corr_abs_between_spaces"]["PC1"]
    c2 = meta["corr_abs_between_spaces"]["PC2"]

    hddf_color = TAG_COLORS.get(tag.lower(), TAG_COLORS["default"])

    output_files = [
        f"pca_base_qed_{tag}.png",
        f"pca_base_desirability_{tag}.png",
        f"pca_hddf_qed_{tag}.png",
        f"pca_hddf_desirability_{tag}.png",
        f"pca_comparison_{tag}.png",
        f"pca_correlation_{tag}.png",
    ]
    for fname in output_files:
        legacy_path = os.path.join(pca_dir, fname)
        if os.path.exists(legacy_path):
            os.remove(legacy_path)

    scatter_configs = [
        (Xb, evr_base, "QED",                  qed,  "viridis", f"pca_base_qed_{tag}.png"),
        (Xb, evr_base, "Desirability Profile",  deseo,"viridis", f"pca_base_desirability_{tag}.png"),
        (Xh, evr_hddf, "QED",                  qed,  "viridis", f"pca_hddf_qed_{tag}.png"),
        (Xh, evr_hddf, "Desirability Profile",  deseo,"viridis", f"pca_hddf_desirability_{tag}.png"),
    ]

    for X, evr, cbar_label, color_vals, cmap, fname in scatter_configs:
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(X[:, 0], X[:, 1], c=color_vals,
                        alpha=0.6, s=26, cmap=cmap, vmin=0, vmax=1)
        _style_ax(ax,
                  xlabel=f"PC1 ({evr[0]:.1%})",
                  ylabel=f"PC2 ({evr[1]:.1%})",
                  tag=tag)
        _colorbar(sc, ax, cbar_label)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname), dpi=500, bbox_inches="tight")
        plt.close()

    # Overlay BASE vs HDDF
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(Xb[:, 0], Xb[:, 1], alpha=0.40, s=22,
               label=f"{tag.upper()}-base ({len(meta['base_valid'])})",
               color=_HUSL_BASE)
    ax.scatter(Xh[:, 0], Xh[:, 1], alpha=0.60, s=24,
               label=f"{tag.upper()}-HDDF ({len(meta['hddf_cols'])})",
               color=hddf_color)
    _style_ax(ax, xlabel="PC1", ylabel="PC2")
    ax.legend(frameon=False, fontsize=14, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"pca_comparison_{tag}.png"),
                dpi=500, bbox_inches="tight")
    plt.close()

    # PC correlation between spaces
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(Xb[:, 0], Xh[:, 0], alpha=0.40, s=22,
               label=f"PC1 (|r|={c1:.3f})", color=_HUSL_BASE)
    ax.scatter(Xb[:, 1], Xh[:, 1], alpha=0.60, s=24,
               label=f"PC2 (|r|={c2:.3f})", color=hddf_color)
    _style_ax(ax,
              xlabel=f"{tag.upper()}-BASE PCs",
              ylabel=f"{tag.upper()}-HDDF PCs")
    ax.legend(frameon=False, fontsize=14, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"pca_correlation_{tag}.png"),
                dpi=500, bbox_inches="tight")
    plt.close()

    print(f"[OK] PCA figures saved: {output_dir}/")
    return True


# ============================================================
# COLLECTIONS
# ============================================================

def _load_joint(pair_tag: str, space: str):
    base_dir    = get_path("pca_joint", pair_tag, space.upper())
    coords_path = os.path.join(base_dir, f"joint_pca_coordinates_{pair_tag}.csv")
    meta_path   = os.path.join(base_dir, "joint_pca_metadata.json")
    for p in (coords_path, meta_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"File not found: {p}\n"
                f"Run: hddflyzer dimred pca-joint <tag_a> <tag_b>")
    coords = pd.read_csv(coords_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return base_dir, coords, meta


def plot_collections_space(pair_tag: str, tag_a: str, tag_b: str,
                            space: str) -> bool:
    try:
        out_dir, coords, meta = _load_joint(pair_tag, space)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return False

    if not {"dataset", "PC1", "PC2"}.issubset(coords.columns):
        print("[ERROR] Joint CSV must contain 'dataset', 'PC1', 'PC2' columns.")
        return False

    # Normalise tag names to match what's in the CSV
    ds_vals = set(coords["dataset"].astype(str).unique())
    if tag_a not in ds_vals:
        tag_a = tag_a.lower()
    if tag_b not in ds_vals:
        tag_b = tag_b.lower()

    evr = meta.get("explained_variance_ratio", [])
    pc1_lab = f"PCA1-{space} ({evr[0]*100:.1f}%)" if len(evr) > 0 else f"PCA1-{space}"
    pc2_lab = f"PCA2-{space} ({evr[1]*100:.1f}%)" if len(evr) > 1 else f"PCA2-{space}"

    col_a = TAG_COLORS.get(tag_a.lower(), anabook_palette_8[1])
    col_b = TAG_COLORS.get(tag_b.lower(), anabook_palette_8[3])

    df_a = coords[coords["dataset"] == tag_a]
    df_b = coords[coords["dataset"] == tag_b]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(df_a["PC1"], df_a["PC2"], s=24, alpha=0.60,
               label=tag_a.upper(), color=col_a)
    ax.scatter(df_b["PC1"], df_b["PC2"], s=24, alpha=0.60,
               label=tag_b.upper(), color=col_b)
    ax.set_xlabel(pc1_lab, fontsize=22)
    ax.set_ylabel(pc2_lab, fontsize=22)
    ax.tick_params(labelsize=18)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%d"))
    ax.legend(frameon=False, fontsize=14, loc="upper right")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_color("black"); sp.set_linewidth(2.0)

    plt.tight_layout()
    fname = os.path.join(out_dir,
                         f"pca_scatter_{space.lower()}_{pair_tag}.png")
    plt.savefig(fname, dpi=500, bbox_inches="tight")
    plt.close()
    print(f"[OK] {space} scatter: {fname}")
    return True


def plot_collections(tag_a: str, tag_b: str,
                     space: str = "BOTH") -> bool:
    pair_tag = f"{tag_a}_vs_{tag_b}"
    spaces   = ["BASE", "HDDF"] if space.upper() == "BOTH" else [space.upper()]
    ok = True
    for sp in spaces:
        ok = plot_collections_space(pair_tag, tag_a, tag_b, sp) and ok
    return ok


# ============================================================
# CLI
# ============================================================

def main():
    if len(sys.argv) < 3:
        print("Usage: python -m hddflyzer.viz.pca_plots <mode> <tag(s)> [--space BASE|HDDF|BOTH]")
        print("  mode: analysis | collections")
        print("  Examples:")
        print("    python -m hddflyzer.viz.pca_plots analysis aocd")
        print("    python -m hddflyzer.viz.pca_plots collections aocd dianatdb")
        print("    python -m hddflyzer.viz.pca_plots collections aocd dianatdb --space HDDF")
        sys.exit(1)

    mode = sys.argv[1].strip().lower()

    if mode == "analysis":
        tag = sanitize_tag(sys.argv[2])
        ok  = plot_analysis(tag)

    elif mode == "collections":
        if len(sys.argv) < 4:
            print("[ERROR] collections mode requires two tags.")
            sys.exit(1)
        tag_a = sanitize_tag(sys.argv[2])
        tag_b = sanitize_tag(sys.argv[3])
        space = "BOTH"
        if "--space" in sys.argv:
            idx   = sys.argv.index("--space")
            space = sys.argv[idx + 1].upper() if idx + 1 < len(sys.argv) else "BOTH"
        ok = plot_collections(tag_a, tag_b, space)

    else:
        print(f"[ERROR] Unknown mode '{mode}'. Use: analysis | collections")
        sys.exit(1)

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
