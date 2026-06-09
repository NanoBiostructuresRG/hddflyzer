# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Correlation visualizations.

hddf_scatters : Scatter plots for all 10 HDDF descriptor pairs
                with Pearson/Spearman annotations and trendlines.

Usage
-----
    python -m hddflyzer.viz.correlations hddf <tag>
"""

import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from hddflyzer.config import get_path
from hddflyzer.io import resolve_features_csv
from hddflyzer.results import KIND_DESCRIPTOR_TABLE, LoadedArtifact
from hddflyzer.viz.colors import TAG_COLORS
from hddflyzer.viz.inputs import VizInputs
from hddflyzer.utils.naming import sanitize_tag


HDDF_PAIRS = [
    ("Pharma_Complexity", "Synthetic_Accessibility"),
    ("Pharma_Complexity", "Desirability_Profile"),
    ("Pharma_Complexity", "LeadLikeness_Score"),
    ("Pharma_Complexity", "QED"),
    ("Desirability_Profile", "Synthetic_Accessibility"),
    ("Synthetic_Accessibility", "LeadLikeness_Score"),
    ("Desirability_Profile", "LeadLikeness_Score"),
    ("QED", "LeadLikeness_Score"),
    ("QED", "Desirability_Profile"),
    ("QED", "Synthetic_Accessibility"),
]


def _resolve_hddf_source(source) -> tuple[str, str | None, pd.DataFrame | None]:
    """Return tag, optional features CSV path, and optional DataFrame."""
    if isinstance(source, LoadedArtifact):
        if source.artifact.kind != KIND_DESCRIPTOR_TABLE:
            raise ValueError(
                "LoadedArtifact must have kind='descriptor_table' for "
                "HDDF scatter plots."
            )
        if not isinstance(source.data, pd.DataFrame):
            raise ValueError(
                "LoadedArtifact data must be a pandas DataFrame for "
                "HDDF scatter plots."
            )
        return source.artifact.path.parent.parent.parent.name, None, source.data.copy()

    if isinstance(source, VizInputs):
        for path in source.paths:
            parts = {part.lower() for part in path.parts}
            if path.suffix.lower() == ".csv" and "features" in parts:
                return source.root.name, str(path), None
        raise FileNotFoundError(
            "VizInputs does not contain a features CSV for HDDF scatter plots."
        )

    tag = str(source)
    return tag, resolve_features_csv(tag), None


def plot_hddf_scatters(source) -> bool:
    """Generate scatter plots for HDDF descriptor pairs.

    Parameters
    ----------
    source : str, VizInputs, or LoadedArtifact
        Input source. A string is interpreted as a collection tag. ``VizInputs``
        should contain a features CSV. ``LoadedArtifact`` must have kind
        ``"descriptor_table"`` and hold a pandas ``DataFrame``.

    Returns
    -------
    bool
        ``True`` when the plot is written successfully, otherwise ``False``.

    Notes
    -----
    This function writes
    ``results/<tag>/figures/correlations/hddf_corr_scatters_trendline.png``.
    When a reconstructed input object is supplied, the function uses existing
    descriptor-table data and does not recalculate descriptors.
    """
    try:
        tag, input_file, loaded_df = _resolve_hddf_source(source)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        return False

    df = loaded_df if loaded_df is not None else pd.read_csv(input_file)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    all_cols = sorted(set(c for pair in HDDF_PAIRS for c in pair))
    present = [c for c in all_cols if c in df.columns]
    df = df.dropna(subset=present)

    if df.empty:
        print("[ERROR] No data after NaN removal.")
        return False

    point_color = TAG_COLORS.get(tag.lower(), TAG_COLORS["default"])
    output_dir = get_path("figures", tag, "correlations")
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 5, figsize=(18, 10))
    axes = axes.ravel()

    for idx, (x_col, y_col) in enumerate(HDDF_PAIRS):
        if idx >= len(axes):
            break
        ax = axes[idx]
        if x_col not in df.columns or y_col not in df.columns:
            ax.set_visible(False)
            continue

        x, y = df[x_col].values, df[y_col].values
        ax.scatter(x, y, alpha=0.6, s=24, color=point_color)

        if len(x) >= 2 and np.std(x) > 0:
            coef = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 100)
            ax.plot(xx, np.polyval(coef, xx), linewidth=2.0, color="black", alpha=0.6)

        r = np.corrcoef(x, y)[0, 1]
        rho = pd.Series(x).corr(pd.Series(y), method="spearman")
        ax.text(
            0.05,
            0.05,
            f"Pearson = {r:.3f}\nSpearman = {rho:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=12,
        )

        ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=18)
        ax.set_ylabel(y_col.replace("_", " ").title(), fontsize=18)
        ax.tick_params(labelsize=14)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_color("black")
            sp.set_linewidth(2.0)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "hddf_corr_scatters_trendline.png")
    plt.savefig(out_path, dpi=500, bbox_inches="tight")
    plt.close(fig)

    legacy_path = os.path.join(
        get_path("correlation", tag), "hddf_corr_scatters_trendline.png"
    )
    if os.path.exists(legacy_path):
        os.remove(legacy_path)

    print(f"[OK] HDDF scatters: {out_path}")
    return True


def main():
    if len(sys.argv) < 3:
        print("Usage: python -m hddflyzer.viz.correlations <mode> <tag>")
        print("  mode: hddf")
        print("  Example:")
        print("    python -m hddflyzer.viz.correlations hddf aocd")
        sys.exit(1)

    mode = sys.argv[1].strip().lower()
    tag = sanitize_tag(sys.argv[2])

    if mode != "hddf":
        print(f"[ERROR] Unknown mode '{mode}'. Use: hddf")
        sys.exit(1)

    if not plot_hddf_scatters(tag):
        sys.exit(1)


if __name__ == "__main__":
    main()
