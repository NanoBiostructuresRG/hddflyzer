# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Correlation statistics for molecular descriptors.

Provides two analysis modes:

base_stats
    Pearson + Spearman correlations for BASE descriptors.

hddf_stats
    Pearson + Spearman correlations for the five HDDF descriptors.

Each mode writes only two files:

    *_correlation_stats.csv
        Canonical long-form pairwise correlation table.

    *_correlation_metadata.json
        Compact metadata, strongest pairs, and mode-specific summary.
"""

import itertools
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from hddflyzer.config import (
    EXCLUDED_CATEGORIES,
    EXCLUDED_DESCRIPTORS,
    get_path,
    load_descriptor_config,
)
from hddflyzer.utils.descriptors import (
    get_zero_variance_descriptors,
    strength_label,
)
from hddflyzer.io import resolve_features_csv
from hddflyzer.utils.naming import sanitize_tag
from hddflyzer.io import update_manifest

NONCORR_THRESHOLD = 0.30

HDDF_COLUMNS = [
    "QED",
    "LeadLikeness_Score",
    "Pharma_Complexity",
    "Synthetic_Accessibility",
    "Desirability_Profile",
]
HDDF_NAMES = {
    "QED": "Drug-likeness (QED)",
    "LeadLikeness_Score": "Lead Optimization Potential",
    "Pharma_Complexity": "Pharmacophore Richness",
    "Synthetic_Accessibility": "Synthetic Accessibility",
    "Desirability_Profile": "Desirability Profile",
}


def calculate_correlations(
    df: pd.DataFrame,
    columns: List[str],
    names: Dict[str, str],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], pd.DataFrame, Dict]:
    """
    Calculate pairwise Pearson and Spearman correlations.

    The square Pearson/Spearman matrices are returned for API compatibility,
    but are not persisted by default. The long-form stats_df is the canonical
    output used by downstream modules.
    """
    n = len(df)
    k = len(columns)
    m = k * (k - 1) // 2
    alpha = 0.05
    alpha_bonf = alpha / m if m > 0 else alpha

    rows = []
    for a, b in itertools.combinations(columns, 2):
        x, y = df[a].values, df[b].values
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        try:
            r_p, p_p = pearsonr(x, y)
            r_s, p_s = spearmanr(x, y, nan_policy="omit")
            p_p_adj = min(float(p_p) * m, 1.0) if m > 0 else float(p_p)
            p_s_adj = min(float(p_s) * m, 1.0) if m > 0 else float(p_s)
            rows.append({
                "var_a": a,
                "var_b": b,
                "name_a": names.get(a, a),
                "name_b": names.get(b, b),
                "pearson_r": r_p,
                "pearson_p": p_p,
                "pearson_p_adj_bonferroni": p_p_adj,
                "spearman_r": r_s,
                "spearman_p": p_s,
                "spearman_p_adj_bonferroni": p_s_adj,
                "pearson_strength": strength_label(abs(r_p)),
                "spearman_strength": strength_label(abs(r_s)),
                "signif_pearson": p_p_adj < alpha,
                "signif_spearman": p_s_adj < alpha,
                "max_abs_corr": max(abs(r_p), abs(r_s)),
            })
        except Exception as e:
            print(f"[WARN] Correlation error {a} vs {b}: {e}")

    stats_df = pd.DataFrame(rows)
    meta = {
        "n": n,
        "pairs_calculated": len(rows),
        "pairs_total": m,
        "alpha": alpha,
        "p_adjustment": "bonferroni",
        "alpha_bonferroni": alpha_bonf,
        "significance_rule": "significant if p_adj_bonferroni < alpha",
        "columns_valid": columns,
        "names": names,
    }

    if stats_df.empty:
        return None, None, stats_df, meta

    try:
        pearson_corr = df[columns].corr(method="pearson")
        spearman_corr = df[columns].corr(method="spearman")
    except Exception as e:
        print(f"[WARN] Matrix correlation error: {e}")
        pearson_corr = spearman_corr = None

    return pearson_corr, spearman_corr, stats_df, meta


def _remove_legacy_outputs(output_dir: str, prefix: str) -> None:
    for fname in (
        "pearson_corr.csv",
        "spearman_corr.csv",
        f"{prefix}_correlation_stats.json",
        f"{prefix}_correlation_stats.txt",
        f"{prefix}_correlation_summary.json",
        "non_correlated_descriptors.txt",
    ):
        path = os.path.join(output_dir, fname)
        if os.path.exists(path):
            os.remove(path)


def save_reports(
    pearson_corr: Optional[pd.DataFrame],
    spearman_corr: Optional[pd.DataFrame],
    stats_df: pd.DataFrame,
    meta: Dict,
    output_dir: str,
    prefix: str,
    names: Dict[str, str],
    extra_summary: Dict = None,
) -> None:
    """Save one canonical CSV plus one compact JSON summary."""
    os.makedirs(output_dir, exist_ok=True)
    if stats_df.empty:
        print("[WARN] stats_df is empty - nothing to save.")
        return

    _remove_legacy_outputs(output_dir, prefix)

    stats_path = os.path.join(output_dir, f"{prefix}_correlation_stats.csv")
    stats_df.to_csv(stats_path, index=False)

    summary = {
        "meta": meta,
        "files": {"correlations": os.path.basename(stats_path)},
        "strongest_pairs_top20": (
            stats_df.sort_values("max_abs_corr", ascending=False)
            .head(20)
            .to_dict(orient="records")
        ),
    }
    if extra_summary:
        summary.update(extra_summary)

    summary_path = os.path.join(output_dir, f"{prefix}_correlation_metadata.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] Correlations saved: {stats_path}")
    print(f"[OK] Summary saved     : {summary_path}")
    tag = str(meta.get("tag", "") or "")
    if tag:
        update_manifest(
            tag,
            f"chem.stats.{prefix}",
            [stats_path, summary_path],
            {"prefix": prefix, "pairs_calculated": meta.get("pairs_calculated")},
        )


def find_weak_correlations(
    stats_df: pd.DataFrame,
    valid_columns: List[str],
    names: Dict[str, str],
) -> List[Dict]:
    """Return descriptors with all pairwise correlations below threshold."""
    if stats_df.empty:
        return []

    max_corr = {c: 0.0 for c in valid_columns}
    for _, row in stats_df.iterrows():
        a, b = row["var_a"], row["var_b"]
        val = max(abs(row["pearson_r"]), abs(row["spearman_r"]))
        if a in max_corr:
            max_corr[a] = max(max_corr[a], val)
        if b in max_corr:
            max_corr[b] = max(max_corr[b], val)

    non_corr = sorted(
        [
            {"descriptor": c, "name": names.get(c, c), "max_correlation": v}
            for c, v in max_corr.items()
            if v < NONCORR_THRESHOLD
        ],
        key=lambda x: x["max_correlation"],
    )
    print(f"[INFO] Non-correlated descriptors: {len(non_corr)}")
    return non_corr


def run_base_stats(tag: str) -> bool:
    """
    Run BASE descriptor correlation analysis for a tag.

    Reads  : results/{tag}/features/full/features_*.csv
    Writes : results/{tag}/features/correlations/base_correlation_stats.csv
             results/{tag}/features/correlations/base_correlation_metadata.json
    """
    input_file = resolve_features_csv(tag)
    output_dir = get_path("base_correlation", tag)

    base_columns, base_names = load_descriptor_config(
        excluded_categories=EXCLUDED_CATEGORIES,
        excluded_descriptors=EXCLUDED_DESCRIPTORS,
    )

    df = pd.read_csv(input_file)
    missing = [c for c in base_columns if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        return False

    sub = df[base_columns].copy()
    for c in base_columns:
        if not pd.api.types.is_numeric_dtype(sub[c]):
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(subset=base_columns)

    if len(sub) < 3:
        print("[ERROR] Too few rows after cleaning.")
        return False

    zero_vars = get_zero_variance_descriptors(sub)
    valid_columns = [c for c in base_columns if c not in zero_vars]
    print(f"[INFO] {len(sub)} rows | {len(valid_columns)} valid columns")

    pearson, spearman, stats_df, meta = calculate_correlations(
        sub, valid_columns, base_names
    )
    if stats_df.empty:
        print("[ERROR] No valid correlations computed.")
        return False

    meta["columns_original"] = base_columns
    meta["columns_excluded"] = list(set(base_columns) - set(valid_columns))
    meta["tag"] = tag

    non_corr = find_weak_correlations(stats_df, valid_columns, base_names)
    save_reports(
        pearson,
        spearman,
        stats_df,
        meta,
        output_dir,
        "base",
        base_names,
        extra_summary={
            "non_correlated_threshold": NONCORR_THRESHOLD,
            "non_correlated_descriptors": non_corr,
        },
    )
    return True


def run_hddf_stats(tag: str) -> bool:
    """
    Run HDDF descriptor correlation analysis for a tag.

    Reads  : results/{tag}/features/full/features_*.csv
    Writes : results/{tag}/features/correlations/hddf_correlation_stats.csv
             results/{tag}/features/correlations/hddf_correlation_metadata.json
    """
    input_file = resolve_features_csv(tag)
    output_dir = get_path("correlation", tag)

    df = pd.read_csv(input_file)
    missing = [c for c in HDDF_COLUMNS if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing HDDF columns: {missing}")
        return False

    sub = df[HDDF_COLUMNS].copy()
    for c in HDDF_COLUMNS:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(subset=HDDF_COLUMNS)
    print(f"[INFO] {len(sub)} rows for HDDF analysis")

    pearson, spearman, stats_df, meta = calculate_correlations(
        sub, HDDF_COLUMNS, HDDF_NAMES
    )
    if stats_df.empty:
        print("[ERROR] No valid HDDF correlations computed.")
        return False
    meta["tag"] = tag

    save_reports(pearson, spearman, stats_df, meta, output_dir, "hddf", HDDF_NAMES)
    return True


def main():
    if len(sys.argv) < 3:
        print("Usage: python -m hddflyzer.chem.stats <mode> <tag>")
        print("  mode: base | hddf")
        print("  Examples:")
        print("    python -m hddflyzer.chem.stats base aocd")
        print("    python -m hddflyzer.chem.stats hddf aocd")
        sys.exit(1)

    mode = sys.argv[1].strip().lower()
    tag = sanitize_tag(sys.argv[2])

    if mode == "base":
        ok = run_base_stats(tag)
    elif mode == "hddf":
        ok = run_hddf_stats(tag)
    else:
        print(f"[ERROR] Unknown mode '{mode}'. Use: base | hddf")
        sys.exit(1)

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
