# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Feature pruning based on high pairwise correlations.

Builds a redundancy graph from base_correlation stats and removes
correlated descriptors, keeping one representative per group.

Usage
-----
    python -m hddflyzer.chem.pruning <tag> [--threshold 0.85]
"""

import os
import sys
import json
import argparse
import pandas as pd
from typing import Dict, List, Set, Tuple

from hddflyzer.config import get_path
from hddflyzer.utils.naming import sanitize_tag
from hddflyzer.io import update_manifest

DEFAULT_THRESHOLD = 0.80
STATS_FILENAME    = "base_correlation_stats.csv"


# ============================================================
# GRAPH UTILITIES
# ============================================================

def build_redundant_pairs(stats_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    max_abs = stats_df[["pearson_r", "spearman_r"]].abs().max(axis=1)
    mask = max_abs >= threshold
    pairs = stats_df.loc[mask].copy()
    pairs["max_abs_corr"] = max_abs[mask]
    return pairs.sort_values("max_abs_corr", ascending=False).reset_index(drop=True)


def build_redundancy_graph(pairs: pd.DataFrame) -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = {}
    for _, row in pairs.iterrows():
        a, b = row["var_a"], row["var_b"]
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    return graph


def find_connected_components(graph: Dict[str, Set[str]]) -> List[Set[str]]:
    visited: Set[str] = set()
    components = []
    for node in graph:
        if node in visited:
            continue
        stack, component = [node], set()
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            component.add(cur)
            stack.extend(graph.get(cur, set()) - visited)
        components.append(component)
    return components


def choose_representative(group: Set[str], degrees: Dict[str, int]) -> str:
    members = sorted(group, key=lambda x: (-degrees.get(x, 0), x))
    return members[0]


def prune_from_correlation_stats(
    stats_df: pd.DataFrame,
    threshold: float,
) -> dict:
    """Prune descriptors from correlation stats and return core summary data."""
    pairs = build_redundant_pairs(stats_df, threshold)
    graph = build_redundancy_graph(pairs)
    comps = find_connected_components(graph)
    degrees = {n: len(nb) for n, nb in graph.items()}
    all_descs = set(stats_df["var_a"]) | set(stats_df["var_b"])

    groups_info = []
    representatives: Set[str] = set()
    removed: Set[str] = set()

    for gid, comp in enumerate(comps, 1):
        rep = next(iter(comp)) if len(comp) == 1 else choose_representative(comp, degrees)
        others = comp - {rep}
        representatives.add(rep)
        removed.update(others)
        groups_info.append({
            "group_id":       gid,
            "members":        sorted(list(comp)),
            "representative": rep,
            "size":           len(comp),
        })

    # Descriptors never in any redundant pair - keep automatically
    representatives.update(all_descs - set(graph.keys()))

    selected = sorted(representatives)
    removed = sorted(removed)

    return {
        "threshold": threshold,
        "n_all_descriptors": len(all_descs),
        "n_selected": len(selected),
        "n_removed": len(removed),
        "n_redundant_pairs": len(pairs),
        "n_redundancy_groups": len(comps),
        "selected_features": selected,
        "removed_features": removed,
        "groups": groups_info,
        "redundant_pairs": pairs.to_dict(orient="records"),
    }


# ============================================================
# MAIN LOGIC
# ============================================================

def run(tag: str, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """
    Perform feature pruning for a tag.

    Reads  : results/{tag}/features/correlations/base_correlation_stats.csv
    Writes : results/{tag}/features/pruning/
               selected_features.txt
               pruning_metadata.json
    """
    stats_path = get_path("base_correlation", tag, STATS_FILENAME)
    if not os.path.exists(stats_path):
        print(f"[ERROR] Stats file not found: {stats_path}")
        print("  Run base stats first: python -m hddflyzer.chem.stats base <tag>")
        return False

    stats_df = pd.read_csv(stats_path)
    required = {"var_a", "var_b", "pearson_r", "spearman_r"}
    if not required.issubset(stats_df.columns):
        print(f"[ERROR] Missing columns in {STATS_FILENAME}: {required - set(stats_df.columns)}")
        return False

    print(f"[INFO] Tag: {tag} | Threshold: |r| >= {threshold:.2f}")

    summary = {
        "tag": tag,
        **prune_from_correlation_stats(stats_df, threshold),
    }

    print(f"[INFO] Redundant pairs: {summary['n_redundant_pairs']}")
    print(f"[INFO] Redundancy groups: {summary['n_redundancy_groups']}")

    output_dir = get_path("feature_pruning", tag)
    os.makedirs(output_dir, exist_ok=True)

    for stale in ("redundant_pairs.csv", "pruning_groups.json", "removed_features.txt"):
        stale_path = os.path.join(output_dir, stale)
        if os.path.exists(stale_path):
            os.remove(stale_path)

    legacy_summary = os.path.join(output_dir, "pruning_summary.json")
    if os.path.exists(legacy_summary):
        os.remove(legacy_summary)
    metadata_path = os.path.join(output_dir, "pruning_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "selected_features.txt"), "w", encoding="utf-8") as f:
        f.write("SELECTED FEATURES (non-pruned)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Tag: {tag}\nThreshold: |r| >= {threshold:.2f}\n")
        f.write(f"Total: {summary['n_selected']}\n\n")
        for feat in summary["selected_features"]:
            f.write(f"{feat}\n")
    update_manifest(
        tag,
        "chem.pruning",
        [os.path.join(output_dir, "selected_features.txt"), metadata_path],
        {
            "threshold": threshold,
            "n_selected": summary["n_selected"],
            "n_removed": summary["n_removed"],
        },
    )

    print(f"\n[SUMMARY]")
    print(f"  All descriptors : {summary['n_all_descriptors']}")
    print(f"  Selected        : {summary['n_selected']}")
    print(f"  Removed         : {summary['n_removed']}")
    print(f"  Output          : {output_dir}/")
    return True


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Feature pruning based on high pairwise correlations.")
    parser.add_argument("tag", type=str, help="Dataset tag (e.g. aocd)")
    parser.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Redundancy threshold (default: {DEFAULT_THRESHOLD})")
    args = parser.parse_args()

    tag = sanitize_tag(args.tag)
    if not (0 < args.threshold <= 1):
        print("[ERROR] Threshold must be in (0, 1].")
        sys.exit(1)

    if not run(tag, args.threshold):
        sys.exit(1)


if __name__ == "__main__":
    main()
