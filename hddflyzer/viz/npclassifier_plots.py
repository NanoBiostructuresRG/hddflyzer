# SPDX-License-Identifier: LGPL-3.0-or-later

"""
NPClassifier distribution plots.

Generates user-selected Pathway, Class, and Superclass distribution plots.

Usage
-----
    python -m hddflyzer.viz.npclassifier_plots <tag>
"""

import os
import sys
import glob
import shutil
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

from hddflyzer.config import get_path
from hddflyzer.utils.naming import sanitize_tag
from hddflyzer.viz.colors import anabook_palette_8


# ============================================================
# DATA LOADING
# ============================================================

def _load_npc_data(tag: str) -> pd.DataFrame:
    clean = "".join(c for c in tag if c.isalnum() or c in "-_")
    tag_dir = get_path("npclassifier", clean)
    if not os.path.exists(tag_dir):
        raise FileNotFoundError(
            f"NPClassifier directory not found: {tag_dir}\n"
            f"Run: python -m hddflyzer.chem.npclassifier {tag}")
    files = glob.glob(os.path.join(tag_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV in {tag_dir}")
    df = pd.read_csv(files[0])
    df = df[df["Status"] == "Success"].copy()
    print(f"[INFO] {len(df)} successful classifications.")
    return df


# ============================================================
# PIE CHART HELPER
# ============================================================

def _pie_chart(counts: pd.Series,
               title_label: str,
               tag: str,
               ax: plt.Axes) -> None:
    """Draw a pie chart with collision-aware percentage labels."""
    colors = anabook_palette_8[:len(counts)]

    def autopct(pct):
        return f"{pct:.1f}%" if pct >= 3.0 else ""

    wedges, _, autotexts = ax.pie(
        counts.values, labels=None, autopct=autopct,
        startangle=90, colors=colors,
        pctdistance=0.6,
        textprops={"fontsize": 14},
    )

    # Adjust label positions to avoid collision
    for wedge, at in zip(wedges, autotexts):
        if not at.get_text():
            continue
        angle     = (wedge.theta2 + wedge.theta1) / 2
        angle_rad = np.radians(angle)
        seg_size  = wedge.theta2 - wedge.theta1
        dist = 1.4 if seg_size < 20 else (1.3 if seg_size < 40 else 1.2)
        x = dist * np.cos(angle_rad)
        y = dist * np.sin(angle_rad)
        at.set_position((x, y))
        ax.annotate("",
                    xy=(0.95 * np.cos(angle_rad), 0.95 * np.sin(angle_rad)),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="-", color="gray",
                                    lw=0.8, alpha=0.6))

    legend_labels = [f"{n} ({c})" for n, c in zip(counts.index, counts.values)]
    ax.legend(wedges, legend_labels,
              title=f"{tag.upper()}: {title_label} (count)",
              title_fontsize=14,
              loc="center left",
              bbox_to_anchor=(1.0, 0, 0.5, 1),
              frameon=False, fontsize=12)
    ax.set_xlim(-1.5, 2.0)
    ax.set_ylim(-1.2, 1.2)


# ============================================================
# INDIVIDUAL FIGURES
# ============================================================

def _fig_pathway(df: pd.DataFrame, tag: str) -> plt.Figure:
    counts = df["Pathway"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 8))
    _pie_chart(counts, "Pathway", tag, ax)
    plt.tight_layout()
    return fig


def _fig_superclass(df: pd.DataFrame, tag: str) -> plt.Figure:
    counts = df["Superclass"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, 8))
    _pie_chart(counts, "Superclass", tag, ax)
    plt.tight_layout()
    return fig


def _fig_class(df: pd.DataFrame, tag: str) -> plt.Figure:
    counts = df["Class"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, 8))
    _pie_chart(counts, "Class", tag, ax)
    plt.tight_layout()
    return fig


def _fig_hierarchical(df: pd.DataFrame, tag: str) -> plt.Figure:
    """Stacked bar: Pathway → Superclass."""
    top_paths = df["Pathway"].value_counts().head(8).index
    hier = (df[df["Pathway"].isin(top_paths)]
            .groupby(["Pathway", "Superclass"])
            .size().reset_index(name="count"))

    pathways = hier["Pathway"].unique()
    x_pos    = np.arange(len(pathways))
    fig, ax  = plt.subplots(figsize=(14, 10))

    bottom = np.zeros(len(pathways))
    colors = anabook_palette_8 * 4   # extend if needed
    for ci, sc in enumerate(hier["Superclass"].unique()):
        counts = [
            hier.loc[(hier["Pathway"] == p) & (hier["Superclass"] == sc),
                     "count"].sum()
            for p in pathways
        ]
        if sum(counts) > 0:
            ax.bar(x_pos, counts, 0.8, bottom=bottom,
                   label=sc, color=colors[ci % len(colors)])
            bottom += np.array(counts)

    ax.set_xlabel("Pathway")
    ax.set_ylabel("Number of compounds")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(pathways, rotation=45, ha="right")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    return fig


def _fig_summary(df: pd.DataFrame, tag: str) -> plt.Figure:
    data = {
        "Metric": [
            "Total compounds", "Unique pathways",
            "Unique superclasses", "Unique classes",
            "Most common pathway", "Most common superclass",
            "Most common class", "Glycosides %",
        ],
        "Value": [
            len(df),
            df["Pathway"].nunique(),
            df["Superclass"].nunique(),
            df["Class"].nunique(),
            df["Pathway"].mode()[0] if not df["Pathway"].mode().empty else "N/A",
            df["Superclass"].mode()[0] if not df["Superclass"].mode().empty else "N/A",
            df["Class"].mode()[0] if not df["Class"].mode().empty else "N/A",
            f"{df['Is_Glycoside'].sum() / len(df) * 100:.1f}%",
        ],
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    tbl = ax.table(cellText=pd.DataFrame(data).values,
                   colLabels=["Metric", "Value"],
                   cellLoc="center", loc="center",
                   bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)
    return fig


def _plot_registry():
    return {
        "pathway": ("pathway_distribution", _fig_pathway),
        "class": ("class_distribution", _fig_class),
        "superclass": ("superclass_distribution", _fig_superclass),
    }


def _resolve_plot_selection(selection: str = None) -> List[str]:
    """Resolve CLI/user plot selection to canonical plot keys."""
    options = list(_plot_registry())
    if selection is None or not str(selection).strip():
        return options

    sel = str(selection).strip().lower()
    aliases = {
        "1": "pathway",
        "p": "pathway",
        "pathways": "pathway",
        "2": "class",
        "c": "class",
        "classes": "class",
        "3": "superclass",
        "s": "superclass",
        "superclasses": "superclass",
        "4": "all",
        "a": "all",
    }
    sel = aliases.get(sel, sel)
    if sel == "all":
        return options
    if sel in options:
        return [sel]
    raise ValueError(f"Unknown plot option: {selection}")


def _prompt_plot_selection() -> List[str]:
    """Ask user which NPClassifier plot(s) to generate."""
    print("Available NPClassifier plots:")
    print("  1. pathway")
    print("  2. class")
    print("  3. superclass")
    print("  4. all")
    try:
        choice = input("\nSelect plot option (number or name): ").strip()
    except EOFError:
        choice = "all"
    return _resolve_plot_selection(choice)


def _remove_legacy_plots(out_dir: str, tag: str) -> None:
    """Remove old high-density NPClassifier plot outputs."""
    for stem in (
        "pathway_pie_chart",
        "hierarchical_distribution",
        "summary_statistics",
    ):
        path = os.path.join(out_dir, f"{tag}_{stem}.png")
        if os.path.exists(path):
            os.remove(path)


def _remove_legacy_plot_dir(tag: str) -> None:
    """Remove old annotation-local plot directory after figures migration."""
    clean = "".join(c for c in tag if c.isalnum() or c in "-_")
    legacy_dir = get_path("npclassifier", clean, "plots")
    if os.path.isdir(legacy_dir):
        shutil.rmtree(legacy_dir)


# ============================================================
# PIPELINE
# ============================================================

def run(tag: str, selection: str = None) -> bool:
    """
    Generate selected NPClassifier plots for a tag.

    Reads  : npclassifier/{tag}/*.csv
    Writes : npclassifier/{tag}/plots/
    """
    try:
        df = _load_npc_data(tag)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return False

    if df.empty:
        print("[ERROR] No successful classifications to plot.")
        return False

    clean   = "".join(c for c in tag if c.isalnum() or c in "-_")
    out_dir = get_path("figures", clean, "npclassifier")
    os.makedirs(out_dir, exist_ok=True)
    _remove_legacy_plots(out_dir, clean)
    _remove_legacy_plot_dir(clean)

    try:
        selected = _resolve_plot_selection(selection) if selection else _prompt_plot_selection()
    except ValueError as e:
        print(f"[ERROR] {e}")
        return False

    registry = _plot_registry()
    figures: List[Tuple[plt.Figure, str]] = [
        (registry[key][1](df, tag), registry[key][0])
        for key in selected
    ]

    for fig, name in figures:
        fname = os.path.join(out_dir, f"{clean}_{name}.png")
        fig.savefig(fname, dpi=500, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[OK] {os.path.basename(fname)}")

    print(f"\n[OK] {len(figures)} NPClassifier plot(s) saved: {out_dir}/")
    return True


# ============================================================
# CLI
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m hddflyzer.viz.npclassifier_plots <tag> [pathway|class|superclass|all]")
        print("Example: python -m hddflyzer.viz.npclassifier_plots aocd")
        sys.exit(1)

    tag = sanitize_tag(sys.argv[1])
    selection = sys.argv[2] if len(sys.argv) > 2 else None
    if not run(tag, selection=selection):
        sys.exit(1)


if __name__ == "__main__":
    main()
