# SPDX-License-Identifier: LGPL-3.0-or-later

"""
t-SNE visualization plots.

tanimoto : Scatter plots from the Tanimoto t-SNE embedding colored by
           NPClassifier annotations: Pathway, Superclass, or Class.

features     : Scatter plots colored by HDDF features (QED, etc.)
               for multiple perplexities.
               (plot_tsne_withfeatures.py)

pruning      : Scatter plots from t-SNE coordinates computed with
               feature-pruning-selected descriptors.

Usage
-----
    python -m hddflyzer.viz.tsne_plots tanimoto <tag> [pathway|superclass|class|all] [top_n]
    python -m hddflyzer.viz.tsne_plots features <tag> [color] [perplexities]
    python -m hddflyzer.viz.tsne_plots pruning  <tag> [color] [perplexities]

    python -m hddflyzer.viz.tsne_plots tanimoto aocd
    python -m hddflyzer.viz.tsne_plots tanimoto aocd pathway 15
    python -m hddflyzer.viz.tsne_plots features aocd QED 30,80
    python -m hddflyzer.viz.tsne_plots pruning aocd QED 30,80
"""

import os
import sys
import glob
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Optional

from hddflyzer.config import get_path
from hddflyzer.utils.naming import sanitize_tag
from hddflyzer.viz.colors import (
    anabook_palette_8, anabook_palette_16, TAG_COLORS,
)


# ============================================================
# SHARED COLOR UTILITIES
# ============================================================

def _get_discrete_palette(n: int) -> list:
    if n <= len(anabook_palette_8):
        return anabook_palette_8[:n]
    if n <= len(anabook_palette_16):
        return anabook_palette_16[:n]
    from itertools import cycle
    return [c for c, _ in zip(cycle(anabook_palette_16), range(n))]


def _continuous_cmap():
    return LinearSegmentedColormap.from_list(
        "anabook_cont", anabook_palette_16, N=256)


def _style_scatter(ax, xlabel="", ylabel="", tag=""):
    ax.set_xlabel(xlabel, fontsize=22, fontfamily="Arial")
    ax.set_ylabel(ylabel, fontsize=22, fontfamily="Arial")
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.grid(True, alpha=0.45, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("black"); sp.set_linewidth(2.0)
    if tag:
        ax.text(0.95, 0.95, tag.upper(), transform=ax.transAxes,
                fontsize=18, fontfamily="Arial", va="top", ha="right")


# ============================================================
# MODE 1: TANIMOTO EMBEDDING COLORED BY NPCLASSIFIER
# ============================================================

TANIMOTO_COLOR_CHOICES = {
    "1": "pathway",
    "2": "superclass",
    "3": "class",
    "4": "all",
    "p": "pathway",
    "s": "superclass",
    "c": "class",
    "a": "all",
    "pathway": "pathway",
    "superclass": "superclass",
    "class": "class",
    "all": "all",
}


def _load_tsne_tanimoto(tag: str) -> pd.DataFrame:
    path = get_path("tsne", tag, "tanimoto", "tsne_tanimoto_coordinates.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"t-SNE data not found: {path}\n"
            f"Run: python -m hddflyzer.dimred.tsne tanimoto {tag}")
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df)} compounds from {path}")
    return df


def _select_tanimoto_categories(category: str = None) -> List[str]:
    if category:
        choice = TANIMOTO_COLOR_CHOICES.get(category.strip().lower())
        if not choice:
            raise ValueError(
                "Unknown color option. Use: pathway, superclass, class, or all"
            )
    else:
        print("Available t-SNE Tanimoto color options:")
        print("  1. pathway")
        print("  2. superclass")
        print("  3. class")
        print("  4. all")
        choice = TANIMOTO_COLOR_CHOICES.get(
            input("Select color option (number or name): ").strip().lower()
        )
        if not choice:
            raise ValueError(
                "Unknown color option. Use: 1, 2, 3, 4, pathway, superclass, class, or all"
            )
    return ["pathway", "superclass", "class"] if choice == "all" else [choice]


def _top_categories(df: pd.DataFrame, col: str, top_n: int) -> pd.DataFrame:
    counts   = df[col].value_counts()
    top_cats = counts.head(top_n).index.tolist()
    df2      = df.copy()
    df2[col] = df2[col].apply(lambda x: x if x in top_cats else "Other")
    return df2


def plot_tsne_tanimoto_category(df: pd.DataFrame, category: str, tag: str,
                                out_dir: str, top_n: int = 15) -> Optional[str]:
    col = category.capitalize()
    if col not in df.columns:
        print(f"[WARN] Column '{col}' not found — skipping.")
        return None

    df_plot = _top_categories(df, col, top_n)
    counts  = df_plot[col].value_counts()
    cats    = counts.index.tolist()
    palette = _get_discrete_palette(len(cats))
    c_map   = dict(zip(cats, palette))

    fig     = plt.figure(figsize=(12, 6))
    gs      = GridSpec(1, 2, width_ratios=[1.0, 0.85], figure=fig)
    ax      = fig.add_subplot(gs[0])
    lax     = fig.add_subplot(gs[1])
    lax.axis("off")

    sns.scatterplot(data=df_plot, x="tsne_x", y="tsne_y",
                    hue=col, palette=c_map, s=40, alpha=0.7,
                    edgecolor="white", linewidth=0.3,
                    hue_order=cats, legend=False, ax=ax)
    _style_scatter(ax, "t-SNE 1", "t-SNE 2")

    handles = [
        Line2D([0], [0], marker="o", linestyle="",
               markersize=8, markerfacecolor=c_map[c],
               markeredgecolor="white", markeredgewidth=0.3,
               label=f"{c} ({counts.get(c, 0)})")
        for c in cats
    ]
    lax.legend(handles=handles,
               title=f"{tag.upper()}: {col} (count)",
               title_fontsize=14, fontsize=12,
               loc="upper left", frameon=False)

    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"{tag}_tsne_{category.lower()}.png")
    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.15,
                        top=0.95, wspace=0.15)
    plt.savefig(fname, dpi=500, facecolor="white")
    plt.close()
    print(f"[OK] {fname}")
    return fname


def plot_tsne_tanimoto(tag: str,
                       category: str = None,
                       top_n: int = 15) -> bool:
    try:
        df = _load_tsne_tanimoto(tag)
        requested = _select_tanimoto_categories(category)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        return False

    out_dir = get_path("figures", tag, "dimred", "tsne", "tanimoto")
    cats = [c for c in requested if c.capitalize() in df.columns]
    legacy_dir = get_path("tsne", tag, "tanimoto", "plots")
    if os.path.isdir(legacy_dir):
        shutil.rmtree(legacy_dir)

    ok = True
    for cat in cats:
        fname = plot_tsne_tanimoto_category(df, cat, tag, out_dir, top_n)
        ok = ok and (fname is not None)
    return ok


# ============================================================
# MODE 2: FEATURES
# ============================================================

def _find_tsne_features_file(tag: str,
                              perps: List[int] = None) -> str:
    base = get_path("tsne", tag, "features")
    if perps and len(perps) == 1:
        p = os.path.join(base, f"tsne_features_coordinates_perp{perps[0]}.csv")
        if os.path.exists(p):
            return p
    candidates = glob.glob(os.path.join(base, "tsne_features_coordinates*.csv"))
    if not candidates:
        candidates = glob.glob(os.path.join(base, "tsne_features_analysis*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No t-SNE features file in {base}\n"
            f"Run: python -m hddflyzer.dimred.tsne features {tag}")
    return max(candidates, key=os.path.getmtime)


def _find_tsne_pruning_file(tag: str) -> str:
    base = get_path("tsne", tag, "pruning")
    path = os.path.join(base, "tsne_pruning_coordinates.csv")
    if os.path.exists(path):
        return path
    raise FileNotFoundError(
        f"No t-SNE pruning file in {base}\n"
        f"Run: python -m hddflyzer.dimred.tsne pruning {tag}")


def _detect_perplexities(df: pd.DataFrame) -> List[int]:
    perps = []
    for c in df.columns:
        if c.startswith("tSNE_1_perp"):
            try:
                p = int(c.replace("tSNE_1_perp", ""))
                if f"tSNE_2_perp{p}" in df.columns:
                    perps.append(p)
            except ValueError:
                pass
    return sorted(set(perps))


def plot_tsne_features_single(df: pd.DataFrame, tag: str, perp: int,
                               color_col: str, out_dir: str,
                               size: int = 26, alpha: float = 0.8) -> Optional[str]:
    x_col = f"tSNE_1_perp{perp}"
    y_col = f"tSNE_2_perp{perp}"
    if x_col not in df.columns or y_col not in df.columns:
        print(f"[WARN] Columns {x_col}/{y_col} not found.")
        return None
    if color_col not in df.columns:
        print(f"[WARN] Color column '{color_col}' not found.")
        return None

    data     = df[[x_col, y_col, color_col]].copy()
    is_num   = pd.api.types.is_numeric_dtype(data[color_col])
    fig, ax  = plt.subplots(figsize=(8, 6))

    if is_num:
        vals  = pd.to_numeric(data[color_col], errors="coerce")
        valid = vals.notna()
        if valid.sum() == 0:
            plt.close()
            return None
        sc = ax.scatter(data.loc[valid, x_col], data.loc[valid, y_col],
                        c=vals[valid].astype(float),
                        cmap=sns.color_palette("viridis", as_cmap=True),
                        s=size, alpha=alpha,
                        edgecolor="white", linewidth=0.3,
                        vmin=0.0, vmax=1.0)
        cbar = plt.colorbar(sc, ax=ax, ticks=np.arange(0, 1.1, 0.2))
        cbar.set_label(color_col, fontsize=22)
        cbar.ax.tick_params(labelsize=16)
    else:
        cats    = data[color_col].astype(str).fillna("NA")
        uniq    = cats.unique().tolist()
        pal     = _get_discrete_palette(len(uniq))
        c_map   = dict(zip(uniq, pal))
        ax.scatter(data[x_col], data[y_col],
                   c=cats.map(c_map), s=size, alpha=alpha,
                   edgecolor="white", linewidth=0.3)
        ax.legend(handles=[
            Line2D([0], [0], marker="o", linestyle="",
                   markersize=6, markerfacecolor=c_map[c],
                   label=f"{c} ({(cats==c).sum()})")
            for c in uniq
        ], title=color_col, loc="best", frameon=False, fontsize=12)

    _style_scatter(ax,
                   xlabel=f"t-SNE 1 (perp={perp})",
                   ylabel="t-SNE 2", tag=tag)

    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"{tag}_tsne_perp{perp}_{color_col}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=500, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"[OK] {fname}")
    return fname


def plot_tsne_features(tag: str,
                        color_col: str = "QED",
                        perps: List[int] = None) -> bool:
    try:
        tsne_file = _find_tsne_features_file(tag, perps)
        df        = pd.read_csv(tsne_file)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return False

    if not perps:
        perps = _detect_perplexities(df)
        print(f"[INFO] Perplexities found: {perps}")

    out_dir = get_path("figures", tag, "dimred", "tsne", "features")
    legacy_dir = get_path("tsne", tag, "features", "plots")
    if os.path.isdir(legacy_dir):
        shutil.rmtree(legacy_dir)
    ok = True
    for p in perps:
        fname = plot_tsne_features_single(df, tag, p, color_col, out_dir)
        ok = ok and (fname is not None)
    return ok


def plot_tsne_pruning(tag: str,
                       color_col: str = "QED",
                       perps: List[int] = None) -> bool:
    try:
        tsne_file = _find_tsne_pruning_file(tag)
        df = pd.read_csv(tsne_file)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return False

    if not perps:
        perps = _detect_perplexities(df)
        print(f"[INFO] Perplexities found: {perps}")

    out_dir = get_path("figures", tag, "dimred", "tsne", "pruning")
    legacy_dir = get_path("tsne", tag, "pruning", "plots")
    if os.path.isdir(legacy_dir):
        shutil.rmtree(legacy_dir)
    ok = True
    for p in perps:
        fname = plot_tsne_features_single(df, tag, p, color_col, out_dir)
        ok = ok and (fname is not None)
    return ok


# ============================================================
# CLI
# ============================================================

def main():
    if len(sys.argv) < 3:
        print("Usage: python -m hddflyzer.viz.tsne_plots <mode> <tag> [...]")
        print("  mode: tanimoto | features | pruning")
        print("  Examples:")
        print("    python -m hddflyzer.viz.tsne_plots tanimoto aocd")
        print("    python -m hddflyzer.viz.tsne_plots tanimoto aocd pathway 15")
        print("    python -m hddflyzer.viz.tsne_plots features aocd QED 30,80")
        print("    python -m hddflyzer.viz.tsne_plots pruning aocd QED 30,80")
        sys.exit(1)

    mode = sys.argv[1].strip().lower()
    tag  = sanitize_tag(sys.argv[2])

    if mode == "tanimoto":
        cat   = sys.argv[3].lower() if len(sys.argv) > 3 else None
        top_n = int(sys.argv[4]) if len(sys.argv) > 4 else 15
        ok = plot_tsne_tanimoto(tag, cat, top_n)

    elif mode == "features":
        color = sys.argv[3] if len(sys.argv) > 3 else "QED"
        perps = (
            [int(p.strip()) for p in sys.argv[4].split(",")]
            if len(sys.argv) > 4 else None
        )
        ok = plot_tsne_features(tag, color, perps)

    elif mode == "pruning":
        color = sys.argv[3] if len(sys.argv) > 3 else "QED"
        perps = (
            [int(p.strip()) for p in sys.argv[4].split(",")]
            if len(sys.argv) > 4 else None
        )
        ok = plot_tsne_pruning(tag, color, perps)

    else:
        print(f"[ERROR] Unknown mode '{mode}'. Use: tanimoto | features | pruning")
        sys.exit(1)

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
