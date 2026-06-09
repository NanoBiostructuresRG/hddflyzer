# SPDX-License-Identifier: LGPL-3.0-or-later

"""
t-SNE dimensionality reduction — two modes.

tanimoto mode   : t-SNE from a precomputed Tanimoto distance matrix,
                  aligned with NPClassifier metadata.
                  (get_tsne_cheminfo.py)

features mode   : t-SNE from BASE + HDDF descriptor vectors,
                  with multiple perplexities.
                  (get_tsne_withfeatures.py)

Usage
-----
    python -m hddflyzer.dimred.tsne tanimoto <tag>
    python -m hddflyzer.dimred.tsne features <tag> [perplexities]
    python -m hddflyzer.dimred.tsne pruning <tag> [perplexities]

    python -m hddflyzer.dimred.tsne tanimoto aocd
    python -m hddflyzer.dimred.tsne features aocd 5,30,80
    python -m hddflyzer.dimred.tsne pruning aocd
"""

import os
import sys
import json
from importlib import metadata as importlib_metadata
import numpy as np
import pandas as pd
import sklearn
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from typing import List, Optional

from hddflyzer.config import get_path
from hddflyzer.utils.naming import sanitize_tag
from hddflyzer.utils.descriptors import (
    categorize_descriptors,
    get_zero_variance_descriptors,
)
from hddflyzer.io import (
    EXCLUDED_SIMILARITY_FEATURES,
    align_tanimoto_with_npclassifier,
    load_features_table,
    load_npclassifier_success,
    load_selected_features,
    load_tanimoto,
    update_manifest,
)

TSNE_RANDOM_STATE = 42
TSNE_MAX_ITER = 1000
TSNE_LEARNING_RATE = "auto"
TSNE_INIT = "random"
TSNE_EARLY_EXAGGERATION = 12.0
TSNE_ANGLE = 0.5
TSNE_METHOD = "barnes_hut"
TSNE_N_ITER_WITHOUT_PROGRESS = 300
TSNE_MIN_GRAD_NORM = 1e-7


# ============================================================
# SHARED UTILITIES
# ============================================================

def _safe_perplexity(requested: int, n: int, minimum: int = 5) -> int:
    """Cap perplexity to a numerically safe value for sample size n."""
    max_safe = max(minimum, min(n - 1, (n - 1) // 3))
    return min(max_safe, max(minimum, requested))


def _run_tsne(
    X: np.ndarray,
    perplexity: int,
    random_state: int = TSNE_RANDOM_STATE,
    metric: str = "euclidean",
    max_iter: int = TSNE_MAX_ITER,
) -> np.ndarray:
    """Run 2D t-SNE and return coordinates array."""
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric=metric,
        random_state=random_state,
        max_iter=max_iter,
        learning_rate=TSNE_LEARNING_RATE,
        init=TSNE_INIT,
        early_exaggeration=TSNE_EARLY_EXAGGERATION,
        angle=TSNE_ANGLE,
        method=TSNE_METHOD,
        n_iter_without_progress=TSNE_N_ITER_WITHOUT_PROGRESS,
        min_grad_norm=TSNE_MIN_GRAD_NORM,
        verbose=0,
    )
    return tsne.fit_transform(X)


def _package_version(package: str) -> Optional[str]:
    """Return installed package version without importing heavy optional modules."""
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return None


def _library_versions() -> dict:
    """Versions relevant for t-SNE reproducibility."""
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": _package_version("scipy"),
    }


def _tsne_parameters(
    perplexity,
    random_state: int,
    metric: str,
) -> dict:
    """Canonical t-SNE parameters persisted in metadata."""
    return {
        "perplexity": perplexity,
        "random_state": random_state,
        "n_components": 2,
        "metric": metric,
        "max_iter": TSNE_MAX_ITER,
        "learning_rate": TSNE_LEARNING_RATE,
        "init": TSNE_INIT,
        "early_exaggeration": TSNE_EARLY_EXAGGERATION,
        "angle": TSNE_ANGLE,
        "method": TSNE_METHOD,
        "n_iter_without_progress": TSNE_N_ITER_WITHOUT_PROGRESS,
        "min_grad_norm": TSNE_MIN_GRAD_NORM,
    }


def _tsne_interpretation(perplexities_requested: List[int], capped: List[int]) -> dict:
    """Return reusable metadata explaining how to interpret t-SNE outputs."""
    return {
        "reproducibility": (
            "t-SNE is stochastic; coordinates are reproducible for a fixed "
            "input table, preprocessing, random_state, initialization, and "
            "software stack."
        ),
        "perplexity": (
            "Lower perplexity emphasizes local neighborhoods; higher perplexity "
            "uses broader neighborhoods and may reveal more global structure."
        ),
        "coordinate_scale": (
            "t-SNE coordinates do not preserve absolute distances or axes across "
            "different perplexities. Compare neighborhood patterns within the "
            "same perplexity, not raw coordinate magnitudes across perplexities."
        ),
        "requested_vs_used": [
            {"requested": int(req), "used": int(use)}
            for req, use in zip(perplexities_requested, capped)
        ],
    }


def _add_tsne_outlier_columns(
    coords_data: dict,
    perplexities_requested: List[int],
    z_threshold: float = 3.0,
) -> dict:
    """
    Add centroid-distance outlier screening columns for each t-SNE perplexity.

    This is a reproducible visual-screening aid, not a chemical outlier test.
    """
    summary = {}
    for p in perplexities_requested:
        x_key = f"tSNE_1_perp{p}"
        y_key = f"tSNE_2_perp{p}"
        if x_key not in coords_data or y_key not in coords_data:
            continue

        x = np.asarray(coords_data[x_key], dtype=float)
        y = np.asarray(coords_data[y_key], dtype=float)
        centroid_x = float(np.nanmean(x))
        centroid_y = float(np.nanmean(y))
        dist = np.sqrt((x - centroid_x) ** 2 + (y - centroid_y) ** 2)
        mean_dist = float(np.nanmean(dist))
        std_dist = float(np.nanstd(dist))
        z = (dist - mean_dist) / std_dist if std_dist > 0 else np.zeros_like(dist)
        outlier = z > z_threshold

        coords_data[f"centroid_x_perp{p}"] = np.full(len(x), centroid_x)
        coords_data[f"centroid_y_perp{p}"] = np.full(len(y), centroid_y)
        coords_data[f"dist_to_centroid_perp{p}"] = dist
        coords_data[f"z_score_perp{p}"] = z
        coords_data[f"outlier_perp{p}"] = outlier
        summary[str(p)] = {
            "centroid_x": round(centroid_x, 6),
            "centroid_y": round(centroid_y, 6),
            "mean_distance": round(mean_dist, 6),
            "std_distance": round(std_dist, 6),
            "z_threshold": z_threshold,
            "n_outliers": int(np.sum(outlier)),
        }
    return summary


def _load_features_dataset(tag: str) -> pd.DataFrame:
    """Load latest feature-engineering table for a tag."""
    df, path = load_features_table(tag)
    print(f"[INFO] Loaded features: {path}")
    print(f"[INFO] Loaded {len(df)} compounds, {len(df.columns)} columns")
    return df


# ============================================================
# MODE 1: TANIMOTO
# ============================================================


def run_tanimoto(
    tag: str,
    perplexity: int = 30,
    seed: int = TSNE_RANDOM_STATE,
) -> bool:
    """
    t-SNE from Tanimoto distance matrix + NPClassifier metadata.

    Reads  : tanimoto/{tag}/tanimoto_matrix.npz
             results/{tag}/chemistry/tanimoto/tanimoto_ids.csv
             results/{tag}/annotations/npclassifier/npclassifier.csv
    Writes : results/{tag}/dimred/tsne/tanimoto/
               tsne_tanimoto_coordinates.csv
               tsne_tanimoto_metadata.json
    """
    out_dir = get_path("tsne", tag, "tanimoto")
    os.makedirs(out_dir, exist_ok=True)

    try:
        sim, ids, _, _ = load_tanimoto(tag)
        npc_df, _ = load_npclassifier_success(tag)
        sim, ids, npc_aln = align_tanimoto_with_npclassifier(sim, ids, npc_df)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        return False

    n = sim.shape[0]
    if n <= 5:
        print(f"[ERROR] Too few samples ({n}) for t-SNE.")
        return False

    safe_p = _safe_perplexity(perplexity, n)
    if safe_p != perplexity:
        print(f"[INFO] Perplexity adjusted {perplexity} → {safe_p} (n={n})")

    dist = np.maximum(0.0, 1.0 - sim).astype(np.float32)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0

    print(f"[INFO] Running t-SNE (n={n}, perplexity={safe_p})...")
    coords = _run_tsne(dist, safe_p, seed, metric="precomputed")

    # Build output table
    df_out = pd.DataFrame({"tsne_x": coords[:, 0], "tsne_y": coords[:, 1]})
    for col in npc_aln.columns:
        df_out[col] = npc_aln[col].values

    coords_path = os.path.join(out_dir, "tsne_tanimoto_coordinates.csv")
    params_path = os.path.join(out_dir, "tsne_tanimoto_metadata.json")
    df_out.to_csv(coords_path, index=False)

    embedding_distance = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
    top_idx = np.argsort(embedding_distance)[::-1][:10]
    outlier_candidates = []
    for i in top_idx:
        row = df_out.iloc[i]
        outlier_candidates.append({
            "identifier": str(row.get("identifier", ids[i] if i < len(ids) else i)),
            "tsne_x": float(round(coords[i, 0], 6)),
            "tsne_y": float(round(coords[i, 1], 6)),
            "embedding_distance": float(round(embedding_distance[i], 6)),
            "Pathway": str(row.get("Pathway", "")),
            "Superclass": str(row.get("Superclass", "")),
            "Class": str(row.get("Class", "")),
        })

    classification_summary = {}
    for col in ("Pathway", "Superclass", "Class"):
        if col in df_out.columns:
            counts = df_out[col].fillna("Unknown").astype(str).value_counts()
            classification_summary[col] = {
                "n_unique": int(counts.shape[0]),
                "top_counts": {
                    str(k): int(v) for k, v in counts.head(20).items()
                },
            }

    for legacy_name in (
        "tsne_coordinates.csv",
        "tsne_coordinates.npy",
        "tsne_parameters.json",
        "tsne_tanimoto_parameters.json",
    ):
        legacy_path = os.path.join(out_dir, legacy_name)
        if os.path.exists(legacy_path):
            os.remove(legacy_path)

    params = {
        "perplexity":         safe_p,
        "perplexity_requested": int(perplexity),
        "tsne_parameters":    _tsne_parameters(safe_p, seed, "precomputed"),
        "library_versions":   _library_versions(),
        "n_compounds":        int(n),
        "input_tag":          tag,
        "source_tanimoto":    get_path("tanimoto", tag, "tanimoto_matrix.npz"),
        "source_npclassifier": get_path("npclassifier", tag),
        "interpretation":     _tsne_interpretation([perplexity], [safe_p]),
        "classification_summary": classification_summary,
        "outlier_screen": {
            "method": "largest Euclidean distance from origin in the 2D t-SNE embedding",
            "interpretation": (
                "Large embedding distances are candidates for visual/chemical "
                "inspection. They are not automatic errors because t-SNE does "
                "not preserve absolute global distances."
            ),
            "top10": outlier_candidates,
        },
    }
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    update_manifest(tag, "dimred.tsne.tanimoto", [coords_path, params_path], params)

    print(f"[OK] t-SNE (tanimoto) complete: {out_dir}/")
    print(f"     Coordinates : {coords_path}")
    print(f"     Parameters  : {params_path}")
    return True


# ============================================================
# MODE 2: FEATURES
# ============================================================

def run_features(
    tag: str,
    perplexities: List[int] = None,
    seed: int = TSNE_RANDOM_STATE,
    min_samples: int = 10,
) -> bool:
    """
    t-SNE from BASE + HDDF descriptor vectors.

    Reads  : results/{tag}/features/full/features_*.csv
    Writes : results/{tag}/dimred/tsne/features/
               tsne_features_coordinates[_perp{N}].csv
               tsne_features_metadata[_perp{N}].json

    Parameters
    ----------
    perplexities : list of ints to try (default: [5, 15, 30, 50, 100])
    """
    if perplexities is None:
        perplexities = [5, 15, 30, 50, 100]

    out_dir = get_path("tsne", tag, "features")
    os.makedirs(out_dir, exist_ok=True)

    try:
        df = _load_features_dataset(tag)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return False

    # Descriptor selection
    zero_vars = get_zero_variance_descriptors(
        df.select_dtypes(include=[np.number])
          .replace([np.inf, -np.inf], np.nan)
    )
    cats = categorize_descriptors(df.columns.tolist(), zero_vars)
    all_desc = cats["all_base"] + cats["hddf"]
    all_desc = [c for c in all_desc if c in df.columns]

    if len(all_desc) < 2:
        print("[ERROR] Insufficient descriptors after filtering.")
        return False

    X_df = df[all_desc].apply(pd.to_numeric, errors="coerce") \
                       .replace([np.inf, -np.inf], np.nan)
    idx  = X_df.dropna().index

    if len(idx) < min_samples:
        print(f"[ERROR] Too few samples ({len(idx)}) after NaN removal.")
        return False

    X_scaled = StandardScaler().fit_transform(X_df.loc[idx].values)
    n = X_scaled.shape[0]
    print(f"[INFO] {n} samples | {len(all_desc)} descriptors")

    # Cap and deduplicate perplexities
    capped  = [_safe_perplexity(p, n) for p in perplexities]
    unique_p = sorted(set(capped))
    for orig, cap in zip(perplexities, capped):
        if orig != cap:
            print(f"[INFO] Perplexity adjusted {orig} → {cap} (n={n})")

    # Run t-SNE (cache unique)
    cache = {}
    for p in unique_p:
        print(f"[INFO] Running t-SNE perplexity={p}...")
        cache[p] = _run_tsne(X_scaled, p, seed, metric="euclidean")

    # Build output DataFrame
    coords_data = {}
    for id_col in ("identifier", "ID", "id", "SMILES"):
        if id_col in df.columns:
            coords_data[id_col] = df.loc[idx, id_col].astype(str).values
    coords_data["row_id"] = idx.astype(str)

    for orig, cap in zip(perplexities, capped):
        arr = cache[cap]
        coords_data[f"tSNE_1_perp{orig}"] = arr[:, 0]
        coords_data[f"tSNE_2_perp{orig}"] = arr[:, 1]

    color_vars = cats["hddf"] + ["QED", "Desirability_Profile"]
    for cv in dict.fromkeys(color_vars):
        if cv in df.columns:
            coords_data[cv] = pd.to_numeric(
                df[cv], errors="coerce").reindex(idx).values

    coords_df = pd.DataFrame(coords_data)

    # Save — single perplexity gets a named file
    suffix = f"_perp{perplexities[0]}" if len(perplexities) == 1 else ""
    coords_path = os.path.join(out_dir, f"tsne_features_coordinates{suffix}.csv")
    meta_path   = os.path.join(out_dir, f"tsne_features_metadata{suffix}.json")

    coords_df.to_csv(coords_path, index=False)

    meta = {
        "tag": tag, "n_samples": n,
        "perplexities_requested": perplexities,
        "perplexities_capped":    capped,
        "perplexities_unique_run": unique_p,
        "descriptors_used": {
            "base":  cats["all_base"],
            "hddf":  cats["hddf"],
            "total_count": len(all_desc),
        },
        "preprocessing": {
            "zero_variance_removed": zero_vars,
            "samples_after_nan_removal": int(len(idx)),
            "scaling": "StandardScaler",
        },
        "tsne_parameters": _tsne_parameters(unique_p, seed, "euclidean"),
        "library_versions": _library_versions(),
        "interpretation": _tsne_interpretation(perplexities, capped),
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    update_manifest(tag, "dimred.tsne.features", [coords_path, meta_path], meta)

    print(f"[OK] t-SNE (features) complete.")
    print(f"     Coordinates : {coords_path}")
    print(f"     Metadata    : {meta_path}")
    return True


# ============================================================
# MODE 3: PRUNING
# ============================================================

def run_pruning(
    tag: str,
    perplexities: List[int] = None,
    seed: int = TSNE_RANDOM_STATE,
    min_samples: int = 10,
    outlier_z_threshold: float = 3.0,
) -> bool:
    """
    t-SNE from features selected by feature pruning.

    Reads  : results/{tag}/features/full/features_*.csv
             results/{tag}/features/pruning/selected_features.txt
    Writes : results/{tag}/dimred/tsne/pruning/
               tsne_pruning_coordinates.csv
               tsne_pruning_metadata.json

    Parameters
    ----------
    perplexities : list of ints to try (default: [5, 15, 30, 50, 100])
    """
    if perplexities is None:
        perplexities = [5, 15, 30, 50, 100]

    out_dir = get_path("tsne", tag, "pruning")
    os.makedirs(out_dir, exist_ok=True)

    try:
        df = _load_features_dataset(tag)
        selected = load_selected_features(tag, excluded=EXCLUDED_SIMILARITY_FEATURES)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return False

    selected = [c for c in selected if c in df.columns]
    if len(selected) < 2:
        print(f"[ERROR] Too few selected features present in dataset ({len(selected)}).")
        return False

    X_df = df[selected].apply(pd.to_numeric, errors="coerce") \
                       .replace([np.inf, -np.inf], np.nan)
    zero_vars = get_zero_variance_descriptors(X_df.dropna())
    selected = [c for c in selected if c not in zero_vars]
    X_df = X_df[selected]
    idx = X_df.dropna().index

    if len(selected) < 2:
        print("[ERROR] Insufficient selected features after zero-variance filtering.")
        return False
    if len(idx) < min_samples:
        print(f"[ERROR] Too few samples ({len(idx)}) after NaN removal.")
        return False

    X_scaled = StandardScaler().fit_transform(X_df.loc[idx].values)
    n = X_scaled.shape[0]
    print(f"[INFO] {n} samples | {len(selected)} pruned features")

    capped = [_safe_perplexity(p, n) for p in perplexities]
    unique_p = sorted(set(capped))
    for orig, cap in zip(perplexities, capped):
        if orig != cap:
            print(f"[INFO] Perplexity adjusted {orig} -> {cap} (n={n})")

    cache = {}
    for p in unique_p:
        print(f"[INFO] Running t-SNE pruning perplexity={p}...")
        cache[p] = _run_tsne(X_scaled, p, seed, metric="euclidean")

    coords_data = {}
    for id_col in ("identifier", "ID", "id", "SMILES"):
        if id_col in df.columns:
            coords_data[id_col] = df.loc[idx, id_col].astype(str).values
    coords_data["row_id"] = idx.astype(str)

    for orig, cap in zip(perplexities, capped):
        arr = cache[cap]
        coords_data[f"tSNE_1_perp{orig}"] = arr[:, 0]
        coords_data[f"tSNE_2_perp{orig}"] = arr[:, 1]

    outlier_summary = _add_tsne_outlier_columns(
        coords_data,
        perplexities,
        z_threshold=outlier_z_threshold,
    )

    color_vars = [
        "QED", "LeadLikeness_Score", "Pharma_Complexity",
        "Synthetic_Accessibility", "Desirability_Profile",
    ]
    for cv in dict.fromkeys(selected + color_vars):
        if cv in df.columns:
            coords_data[cv] = pd.to_numeric(
                df[cv], errors="coerce").reindex(idx).values

    coords_path = os.path.join(out_dir, "tsne_pruning_coordinates.csv")
    meta_path = os.path.join(out_dir, "tsne_pruning_metadata.json")
    pd.DataFrame(coords_data).to_csv(coords_path, index=False)

    meta = {
        "tag": tag,
        "n_samples": int(n),
        "perplexities_requested": perplexities,
        "perplexities_capped": capped,
        "perplexities_unique_run": unique_p,
        "features_used": selected,
        "features_excluded_zero_variance": zero_vars,
        "preprocessing": {
            "scaling": "StandardScaler",
            "nan_handling": "drop_rows_with_any_nan",
        },
        "tsne_parameters": _tsne_parameters(unique_p, seed, "euclidean"),
        "library_versions": _library_versions(),
        "interpretation": _tsne_interpretation(perplexities, capped),
        "outlier_screen": {
            "method": "distance from per-perplexity t-SNE centroid",
            "z_threshold": outlier_z_threshold,
            "columns_added": [
                "centroid_x_perpN",
                "centroid_y_perpN",
                "dist_to_centroid_perpN",
                "z_score_perpN",
                "outlier_perpN",
            ],
            "interpretation": (
                "Outlier flags are reproducible screening candidates within "
                "each perplexity. Because t-SNE does not preserve absolute "
                "global distances, these flags require chemical/visual review."
            ),
            "summary_by_perplexity": outlier_summary,
        },
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    update_manifest(tag, "dimred.tsne.pruning", [coords_path, meta_path], meta)

    print("[OK] t-SNE (pruning) complete.")
    print(f"     Coordinates : {coords_path}")
    print(f"     Metadata    : {meta_path}")
    return True


# ============================================================
# CLI
# ============================================================

def main():
    if len(sys.argv) < 3:
        print("Usage: python -m hddflyzer.dimred.tsne <mode> <tag> [perplexities]")
        print("  mode: tanimoto | features | pruning")
        print("  Examples:")
        print("    python -m hddflyzer.dimred.tsne tanimoto aocd")
        print("    python -m hddflyzer.dimred.tsne features aocd 5,30,80")
        print("    python -m hddflyzer.dimred.tsne pruning aocd")
        sys.exit(1)

    mode = sys.argv[1].strip().lower()
    tag  = sanitize_tag(sys.argv[2])

    if mode == "tanimoto":
        perp = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        ok = run_tanimoto(tag, perplexity=perp)

    elif mode == "features":
        if len(sys.argv) > 3:
            try:
                perps = [int(p.strip()) for p in sys.argv[3].split(",")]
            except ValueError:
                print(f"[WARN] Invalid perplexities '{sys.argv[3]}'. Using defaults.")
                perps = [5, 15, 30, 50, 100]
        else:
            perps = [5, 15, 30, 50, 100]
        ok = run_features(tag, perplexities=perps)

    elif mode == "pruning":
        if len(sys.argv) > 3:
            try:
                perps = [int(p.strip()) for p in sys.argv[3].split(",")]
            except ValueError:
                print(f"[WARN] Invalid perplexities '{sys.argv[3]}'. Using defaults.")
                perps = [5, 15, 30, 50, 100]
        else:
            perps = [5, 15, 30, 50, 100]
        ok = run_pruning(tag, perplexities=perps)

    else:
        print(f"[ERROR] Unknown mode '{mode}'. Use: tanimoto | features | pruning")
        sys.exit(1)

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
