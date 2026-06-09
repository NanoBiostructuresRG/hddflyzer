# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Physicochemical distribution plots.

Generates violin, split-violin, and KDE plots comparing
descriptor distributions across one or more dataset tags.

Usage
-----
    python -m hddflyzer.viz.distributions <tag1,tag2,...> ["FeatA,FeatB"]

    python -m hddflyzer.viz.distributions aocd
    python -m hddflyzer.viz.distributions aocd,dianatdb "MW,TPSA,MolLogP"
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional

from hddflyzer.config import get_features_path, get_path
from hddflyzer.io import load_df
from hddflyzer.utils.naming import sanitize_tag
from hddflyzer.viz.colors import anabook_palette_8, TAG_COLORS

# ============================================================
# CONFIG
# ============================================================

DEFAULT_FEATURES = [
    "MW", "TPSA", "MolLogP", "NumRotatableBonds",
    "NumHAcceptors", "NumHDonors", "RingCount", "FractionCSP3",
]

FEATURE_LABELS = {
    "MW":                  "Molecular weight (g/mol)",
    "TPSA":                r"Topological polar surface area ($\mathrm{\AA^2}$)",
    "FractionCSP3":        r"Fraction of sp$^3$ carbon atoms",
    "MolLogP":             "logP (octanol/water)",
    "LabuteASA":           r"Labute ASA ($\mathrm{\AA^2}$)",
    "Chi1":                "First-order connectivity",
    "BalabanJ":            "Balaban distance connectivity",
    "HallKierAlpha":       "Hall-Kier alpha value",
    "Chi1v":               "First-order valence connectivity",
    "PEOE_VSA1":           r"PEOE.VSA1 ($\mathrm{\AA^2}$)",
    "PEOE_VSA2":           r"PEOE.VSA2 ($\mathrm{\AA^2}$)",
    "SlogP_VSA1":          r"SlogP.VSA1 ($\mathrm{\AA^2}$)",
    "SlogP_VSA2":          r"SlogP.VSA2 ($\mathrm{\AA^2}$)",
    "SMR_VSA1":            r"SMR.VSA1 ($\mathrm{\AA^2}$)",
    "SMR_VSA2":            r"SMR.VSA2 ($\mathrm{\AA^2}$)",
    "PMI1":                r"PM1 (u$\cdot$Å$^2$)",
    "PMI2":                r"PM2 (u$\cdot$Å$^2$)",
    "QED":                 "Quantitative estimate of drug-likeness",
    "LeadLikeness_Score":  "Lead-likeness",
    "Pharma_Complexity":   "Pharma complexity",
    "Desirability_Profile":"Desirability score",
    "Synthetic_Accessibility": "Synthetic accessibility",
    "Zagreb1":             "First Zagreb index",
    "Ipc":                 "Information content index",
    "NumRotatableBonds":   "Rotatable bonds (count)",
    "NumHAcceptors":       "H-bond acceptors (count)",
    "NumHDonors":          "H-bond donors (count)",
}


def _nice_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))


# ============================================================
# DATA LOADING
# ============================================================

def _resolve_input(tag: str, allow_pickle: bool = False) -> str:
    base = get_features_path(tag)
    patterns = [
        os.path.join(base, "features.csv"),
        os.path.join(base, "features_*.csv"),
    ]
    if allow_pickle:
        patterns.extend([
            os.path.join(base, "features.pkl"),
            os.path.join(base, "features_*.pkl"),
        ])
    for pattern in patterns:
        files = sorted(glob.glob(pattern))
        if files:
            return files[0]
    raise FileNotFoundError(
        f"No features file found for tag '{tag}' in {base}")


def _clean_series(s: pd.Series,
                  clip_low: float = 0.001,
                  clip_high: float = 0.999) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = s.median()
    s   = s.fillna(med)
    lo, hi = s.quantile(clip_low), s.quantile(clip_high)
    if pd.notna(lo) and pd.notna(hi) and hi > lo:
        s = s.clip(lower=lo, upper=hi)
    return s


def build_long_df(
    tags: List[str],
    features: List[str],
    allow_pickle: bool = False,
) -> pd.DataFrame:
    rows = []
    for tag in tags:
        path = _resolve_input(tag, allow_pickle=allow_pickle)
        df   = load_df(path, allow_pickle=allow_pickle)
        for feat in features:
            if feat not in df.columns:
                print(f"[WARN] ({tag}) missing column: {feat}")
                continue
            s = _clean_series(df[feat])
            if s.dropna().empty:
                continue
            for v in s.values:
                rows.append({"Tag": tag, "Feature": feat, "Value": float(v)})
    if not rows:
        raise ValueError("No data to plot. Check tags and features.")
    return pd.DataFrame(rows)


# ============================================================
# STATS TABLE
# ============================================================

def save_stats(df_long: pd.DataFrame, out_dir: str) -> None:
    for feat, gdf in df_long.groupby("Feature"):
        stats = (
            gdf.groupby("Tag")["Value"]
            .agg(n="count", mean="mean", std="std", median="median",
                 q10=lambda x: x.quantile(0.10),
                 q90=lambda x: x.quantile(0.90),
                 min="min", max="max")
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        stats.to_csv(
            os.path.join(out_dir, f"dist_stats_{feat}.csv"), index=False)


# ============================================================
# VIOLIN PLOT
# ============================================================

def plot_violin(df_long: pd.DataFrame,
                feature: str,
                out_dir: str) -> Optional[str]:
    sub = df_long[df_long["Feature"] == feature].copy()
    if sub.empty:
        return None

    sns.set_style("whitegrid")
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(10, 6))

    order = (sub.groupby("Tag")["Value"]
               .median()
               .sort_values(ascending=False)
               .index.tolist())

    custom_palette = [
        TAG_COLORS.get(t.lower(), anabook_palette_8[i % len(anabook_palette_8)])
        for i, t in enumerate(order)
    ]

    sns.violinplot(data=sub, y="Tag", x="Value", order=order,
                   inner=None, cut=2, palette=custom_palette, ax=ax)
    sns.boxplot(data=sub, y="Tag", x="Value", order=order, ax=ax,
                showcaps=True,
                boxprops={"facecolor": "white", "alpha": 0.6},
                medianprops={"linewidth": 2, "color": "black"},
                whiskerprops={"linewidth": 1},
                fliersize=1.5)

    ax.set_xlabel(_nice_label(feature), fontsize=18, fontname="Arial")
    ax.set_ylabel("")
    ax.tick_params(labelsize=14)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_color("black"); sp.set_linewidth(2.0)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{feature}_violin.png")
    plt.savefig(path, dpi=500, bbox_inches="tight")
    plt.close()
    print(f"[OK] Violin {feature}: {path}")
    return path


# ============================================================
# SPLIT VIOLIN (2 tags)
# ============================================================

def plot_split_violin(df_long: pd.DataFrame,
                      feature: str,
                      out_dir: str) -> Optional[str]:
    sub  = df_long[df_long["Feature"] == feature].copy()
    tags = sub["Tag"].unique().tolist()
    if sub.empty or len(tags) != 2:
        return None

    t_a, t_b = tags[0], tags[1]
    col_a = TAG_COLORS.get(t_a.lower(), anabook_palette_8[1])
    col_b = TAG_COLORS.get(t_b.lower(), anabook_palette_8[3])

    sns.set_style("whitegrid")
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.violinplot(data=sub, x="Value", y="Tag",
                   hue="Tag", split=True,
                   palette={t_a: col_a, t_b: col_b},
                   inner="box", cut=0, ax=ax, linewidth=0.7)

    ax.set_xlabel(_nice_label(feature), fontsize=18, fontname="Arial")
    ax.set_ylabel("")
    ax.tick_params(labelsize=14)
    ax.grid(False)
    if ax.get_legend():
        ax.get_legend().remove()
    for sp in ax.spines.values():
        sp.set_color("black"); sp.set_linewidth(2.0)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{feature}_split_violin.png")
    plt.savefig(path, dpi=500, bbox_inches="tight")
    plt.close()
    print(f"[OK] Split violin {feature}: {path}")
    return path


# ============================================================
# KDE PLOT
# ============================================================

def plot_kde(df_long: pd.DataFrame,
             feature: str,
             out_dir: str,
             bw_adjust: float = 1.0,
             fill: bool = False,
             tag_order: List[str] = None) -> Optional[str]:
    sub  = df_long[df_long["Feature"] == feature].copy()
    if sub.empty:
        return None

    available = sub["Tag"].unique().tolist()
    order = [t for t in (tag_order or []) if t in available] or available

    line_styles = ["-", "--", "-.", ":"]
    colors = [
        TAG_COLORS.get(t.lower(), anabook_palette_8[i % len(anabook_palette_8)])
        for i, t in enumerate(order)
    ]

    sns.set_style("white")
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(9, 7))

    for i, tag in enumerate(order):
        s = sub.loc[sub["Tag"] == tag, "Value"].dropna().astype(float)
        if s.empty:
            continue
        sns.kdeplot(x=s.values, ax=ax,
                    color=colors[i],
                    linewidth=3.0,
                    fill=fill,
                    bw_adjust=bw_adjust,
                    common_norm=False,
                    label=tag.upper(),
                    linestyle=line_styles[i % len(line_styles)])

    ax.set_xlabel(_nice_label(feature), fontsize=26, fontname="Arial", labelpad=14)
    ax.set_ylabel("Probability density", fontsize=26, fontname="Arial", labelpad=14)
    ax.tick_params(axis="both", which="major", labelsize=22,
                   direction="out", length=8, width=2.0, color="black",
                   bottom=True, top=False, left=True, right=False, pad=8)
    ax.legend(title="Datasets", frameon=False)

    for sp in ax.spines.values():
        sp.set_color("black"); sp.set_linewidth(2.0)

    path = os.path.join(out_dir, f"{feature}_kde.png")
    plt.savefig(path, dpi=500, bbox_inches="tight")
    plt.close()
    print(f"[OK] KDE {feature}: {path}")
    return path


# ============================================================
# PIPELINE
# ============================================================

def run(tags: List[str],
        features: List[str] = None,
        bw_adjust: float = 1.0) -> bool:
    features  = features or DEFAULT_FEATURES
    tags_key  = "_".join(tags)
    out_dir   = get_path("figures", "compare", tags_key)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Tags     : {tags}")
    print(f"[INFO] Features : {features}")
    print(f"[INFO] Output   : {out_dir}")

    try:
        df_long = build_long_df(tags, features)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        return False

    save_stats(df_long, out_dir)

    for feat in features:
        plot_violin(df_long, feat, out_dir)
        plot_kde(df_long, feat, out_dir, bw_adjust=bw_adjust)
        if len(tags) == 2:
            plot_split_violin(df_long, feat, out_dir)

    print(f"[OK] Distributions complete: {out_dir}/")
    return True


# ============================================================
# CLI
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m hddflyzer.viz.distributions <tag1,tag2,...> [features]")
        print("Example: python -m hddflyzer.viz.distributions aocd,dianatdb \"MW,TPSA\"")
        sys.exit(1)

    tags = [sanitize_tag(t) for t in sys.argv[1].split(",") if t.strip()]
    features = (
        [f.strip() for f in sys.argv[2].split(",") if f.strip()]
        if len(sys.argv) > 2 else None
    )
    if not run(tags, features):
        sys.exit(1)


if __name__ == "__main__":
    main()
