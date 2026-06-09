# SPDX-License-Identifier: LGPL-3.0-or-later

"""
UMAP visualization plots.

features : Scatter plots from umap/{tag}/features/ (combined/base/hddf kinds).
           (plot_umap_withfeatures.py)

tanimoto : Scatter plots from umap/{tag}/tanimoto/.

pruning  : Scatter plots from umap/{tag}/pruning/.
           (plot_umap_pruning.py)

Usage
-----
    python -m hddflyzer.viz.umap_plots features <tag> [color] [n_neighbors] [min_dist] [kinds]
    python -m hddflyzer.viz.umap_plots tanimoto <tag> [color] [n_neighbors] [min_dist]
    python -m hddflyzer.viz.umap_plots pruning  <tag> [color] [n_neighbors] [min_dist]

    python -m hddflyzer.viz.umap_plots features aocd QED
    python -m hddflyzer.viz.umap_plots tanimoto aocd Pathway
    python -m hddflyzer.viz.umap_plots features aocd QED 15,30 0.1,0.5 combined,base
    python -m hddflyzer.viz.umap_plots pruning  aocd Desirability_Profile
"""

import os
import sys
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
from typing import Dict, List, Optional, Set, Tuple

from hddflyzer.config import get_path
from hddflyzer.utils.naming import sanitize_tag
from hddflyzer.viz.colors import TAG_COLORS


# ============================================================
# SHARED COLOR UTILITIES
# ============================================================

def _continuous_cmap():
    return sns.color_palette("viridis", as_cmap=True)


def _discrete_colors(n: int) -> list:
    cmap = _continuous_cmap()
    return [cmap(0.5)] if n <= 1 else [cmap(i / (n - 1)) for i in range(n)]


def _style_ax(ax, xlabel="", ylabel="", tag="", subtitle=""):
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.grid(True, alpha=0.45, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    for sp in ax.spines.values():
        sp.set_color("black"); sp.set_linewidth(2.0)
    if tag:
        label = f"{tag.upper()}{' • ' + subtitle if subtitle else ''}"
        ax.text(0.95, 0.95, label, transform=ax.transAxes,
                fontsize=18, fontfamily="Arial", va="top", ha="right")


# ============================================================
# SHARED SCATTER
# ============================================================

def _scatter_plot(df: pd.DataFrame,
                  x_col: str, y_col: str,
                  color_col: str,
                  tag: str, subtitle: str,
                  out_path: str,
                  size: int = 26, alpha: float = 0.85) -> Optional[str]:

    if x_col not in df.columns or y_col not in df.columns:
        print(f"[WARN] Columns {x_col}/{y_col} not found.")
        return None
    if color_col not in df.columns:
        print(f"[WARN] Color column '{color_col}' not found.")
        return None

    data   = df[[x_col, y_col, color_col]].copy()
    is_num = pd.api.types.is_numeric_dtype(data[color_col])
    fig, ax = plt.subplots(figsize=(8, 6))

    if is_num:
        vals  = pd.to_numeric(data[color_col], errors="coerce")
        valid = vals.notna()
        if valid.sum() == 0:
            plt.close()
            return None
        vmin, vmax = (0.0, 1.0) if color_col in (
            "QED", "Desirability_Profile") else (
            vals[valid].min(), vals[valid].max())
        sc = ax.scatter(data.loc[valid, x_col], data.loc[valid, y_col],
                        c=vals[valid].astype(float),
                        cmap=_continuous_cmap(),
                        s=size, alpha=alpha,
                        edgecolor="white", linewidth=0.3,
                        vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(color_col, fontsize=22)
        cbar.ax.tick_params(labelsize=16)
    else:
        cats  = data[color_col].astype(str).fillna("NA")
        uniq  = sorted(cats.unique().tolist())
        pal   = _discrete_colors(len(uniq))
        c_map = dict(zip(uniq, pal))
        ax.scatter(data[x_col], data[y_col],
                   c=cats.map(c_map), s=size, alpha=alpha,
                   edgecolor="white", linewidth=0.3)
        ax.legend(handles=[
            Line2D([0], [0], marker="o", linestyle="",
                   markersize=6, markerfacecolor=c_map[c],
                   label=f"{c} ({(cats==c).sum()})")
            for c in uniq
        ], title=color_col, loc="best", frameon=False, fontsize=12)

    xlabel = x_col.replace("_", " ")
    _style_ax(ax, xlabel=xlabel, ylabel="UMAP 2",
              tag=tag, subtitle=subtitle)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=500, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"[OK] {os.path.basename(out_path)}")
    return out_path


# ============================================================
# MODE 1: FEATURES
# ============================================================

def _find_features_file(tag: str) -> str:
    base = get_path("umap", tag, "features")
    if not os.path.exists(base):
        raise FileNotFoundError(
            f"UMAP output not found: {base}\n"
            f"Run: python -m hddflyzer.dimred.umap features {tag}")
    preferred = os.path.join(base, "umap_features_coordinates.csv")
    if os.path.exists(preferred):
        return preferred
    candidates = glob.glob(os.path.join(base, "umap_features_coordinates*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No UMAP coordinates CSV in {base}")
    return max(candidates, key=os.path.getmtime)


def _detect_combos(df: pd.DataFrame) -> Dict[str, Set[Tuple]]:
    combos: Dict[str, Set[Tuple]] = {
        "combined": set(), "base": set(), "hddf": set()}
    regex = re.compile(
        r"UMAP[12]_(combined|base|hddf)_nn(\d+)_dist([0-9]*\.?[0-9]+)")
    for c in df.columns:
        m = regex.match(c)
        if m:
            combos[m.group(1)].add((int(m.group(2)), float(m.group(3))))
    return {k: sorted(v) for k, v in combos.items()}


def plot_umap_features(tag: str,
                        color_col: str = "QED",
                        nns: List[int] = None,
                        dists: List[float] = None,
                        kinds: List[str] = None) -> bool:
    kinds = kinds or ["combined"]
    try:
        path = _find_features_file(tag)
        df   = pd.read_csv(path)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return False

    combos = _detect_combos(df)
    pool   = [(nn, d) for k in ["combined", "base", "hddf"]
              for nn, d in combos.get(k, [])]
    nns   = nns   or sorted({nn for nn, _ in pool})
    dists = dists or sorted({d  for _, d  in pool})

    out_dir = get_path("figures", tag, "dimred", "umap", "features")
    saved = []
    for kind in kinds:
        for nn in nns:
            for dist in dists:
                x_col = f"UMAP1_{kind}_nn{nn}_dist{dist}"
                y_col = f"UMAP2_{kind}_nn{nn}_dist{dist}"
                fname = os.path.join(out_dir,
                    f"{tag}_umap_{kind}_nn{nn}_dist{dist}_{color_col}.png")
                f = _scatter_plot(df, x_col, y_col, color_col,
                                  tag, kind, fname)
                if f:
                    saved.append(f)

    print(f"\n[OK] {len(saved)} UMAP (features) figures saved.")
    return bool(saved)


# ============================================================
# MODE 2: TANIMOTO
# ============================================================

def _find_tanimoto_file(tag: str) -> str:
    base = get_path("umap", tag, "tanimoto")
    if not os.path.exists(base):
        raise FileNotFoundError(
            f"UMAP tanimoto output not found: {base}\n"
            f"Run: python -m hddflyzer.dimred.umap tanimoto {tag}")
    path = os.path.join(base, "umap_tanimoto_coordinates.csv")
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"Not found: {path}")


def _detect_tanimoto_combos(df: pd.DataFrame) -> Set[Tuple]:
    combos = set()
    regex = re.compile(r"UMAP[12]_tanimoto_nn(\d+)_dist([0-9]*\.?[0-9]+)")
    for c in df.columns:
        m = regex.match(c)
        if m:
            combos.add((int(m.group(1)), float(m.group(2))))
    return combos


def plot_umap_tanimoto(tag: str,
                        color_col: str = "Pathway",
                        nns: List[int] = None,
                        dists: List[float] = None) -> bool:
    try:
        path = _find_tanimoto_file(tag)
        df = pd.read_csv(path)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return False

    available = _detect_tanimoto_combos(df)
    nns = nns or sorted({nn for nn, _ in available})
    dists = dists or sorted({d for _, d in available})

    out_dir = get_path("figures", tag, "dimred", "umap", "tanimoto")
    saved = []
    for nn in nns:
        for dist in dists:
            x_col = f"UMAP1_tanimoto_nn{nn}_dist{dist}"
            y_col = f"UMAP2_tanimoto_nn{nn}_dist{dist}"
            fname = os.path.join(out_dir,
                f"{tag}_umap_tanimoto_nn{nn}_dist{dist}_{color_col}.png")
            f = _scatter_plot(df, x_col, y_col, color_col,
                              tag, "tanimoto", fname)
            if f:
                saved.append(f)

    print(f"\n[OK] {len(saved)} UMAP (tanimoto) figures saved.")
    return bool(saved)


# ============================================================
# MODE 3: PRUNING
# ============================================================

def _find_pruning_file(tag: str) -> str:
    base = get_path("umap", tag, "pruning")
    if not os.path.exists(base):
        raise FileNotFoundError(
            f"UMAP pruning output not found: {base}\n"
            f"Run: python -m hddflyzer.dimred.umap pruning {tag}")
    path = os.path.join(base, "umap_pruning_coordinates.csv")
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"Not found: {path}")


def _detect_pruning_combos(df: pd.DataFrame) -> Set[Tuple]:
    combos = set()
    regex  = re.compile(r"UMAP[12]_nn(\d+)_dist([0-9]*\.?[0-9]+)")
    for c in df.columns:
        m = regex.match(c)
        if m:
            combos.add((int(m.group(1)), float(m.group(2))))
    return combos


def plot_umap_pruning(tag: str,
                       color_col: str = "QED",
                       nns: List[int] = None,
                       dists: List[float] = None) -> bool:
    try:
        path = _find_pruning_file(tag)
        df   = pd.read_csv(path)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return False

    available = _detect_pruning_combos(df)
    nns   = nns   or sorted({nn  for nn, _  in available})
    dists = dists or sorted({d   for _,  d  in available})

    out_dir = get_path("figures", tag, "dimred", "umap", "pruning")
    saved = []
    for nn in nns:
        for dist in dists:
            x_col = f"UMAP1_nn{nn}_dist{dist}"
            y_col = f"UMAP2_nn{nn}_dist{dist}"
            fname = os.path.join(out_dir,
                f"{tag}_umap_nn{nn}_dist{dist}_{color_col}.png")
            f = _scatter_plot(df, x_col, y_col, color_col,
                              tag, "pruning", fname)
            if f:
                saved.append(f)

    print(f"\n[OK] {len(saved)} UMAP (pruning) figures saved.")
    return bool(saved)


# ============================================================
# CLI
# ============================================================

def _parse_ints(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def _parse_floats(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def _looks_int_list(s):
    return all(p.lstrip("-").isdigit()
               for p in s.split(",") if p.strip())

def _looks_float_list(s):
    try:
        [float(p) for p in s.split(",") if p.strip()]
        return True
    except ValueError:
        return False


def main():
    if len(sys.argv) < 3:
        print("Usage: python -m hddflyzer.viz.umap_plots <mode> <tag> [color] [nn] [dist] [kinds]")
        print("  mode: features | tanimoto | pruning")
        print("  Examples:")
        print("    python -m hddflyzer.viz.umap_plots features aocd")
        print("    python -m hddflyzer.viz.umap_plots tanimoto aocd Pathway")
        print("    python -m hddflyzer.viz.umap_plots features aocd QED 15,30 0.1,0.5 combined,base")
        print("    python -m hddflyzer.viz.umap_plots pruning  aocd Desirability_Profile")
        sys.exit(1)

    mode  = sys.argv[1].strip().lower()
    tag   = sanitize_tag(sys.argv[2])
    color = sys.argv[3] if len(sys.argv) > 3 else (
        "Pathway" if mode == "tanimoto" else "QED")

    tokens     = sys.argv[4:]
    valid_kinds = {"combined", "base", "hddf"}
    nns = dists = kinds = None

    for tok in tokens:
        if tok in valid_kinds or all(t in valid_kinds for t in tok.split(",")):
            kinds = [t.strip() for t in tok.split(",")]
        elif _looks_int_list(tok):
            nns = _parse_ints(tok)
        elif _looks_float_list(tok):
            dists = _parse_floats(tok)

    if mode == "features":
        ok = plot_umap_features(tag, color, nns, dists, kinds)
    elif mode == "tanimoto":
        ok = plot_umap_tanimoto(tag, color, nns, dists)
    elif mode == "pruning":
        ok = plot_umap_pruning(tag, color, nns, dists)
    else:
        print(f"[ERROR] Unknown mode '{mode}'. Use: features | tanimoto | pruning")
        sys.exit(1)

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
