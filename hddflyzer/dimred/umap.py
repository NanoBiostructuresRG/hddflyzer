# SPDX-License-Identifier: LGPL-3.0-or-later

"""
UMAP dimensionality reduction — two modes.

features mode : UMAP from BASE + HDDF descriptors.
                Optionally computes BASE and HDDF spaces separately
                with Procrustes alignment.
                (get_umap_withfeatures.py)

pruning mode  : UMAP from features selected by get_feature_pruning.
                (get_umap_pruning.py)

tanimoto mode : UMAP from a precomputed Tanimoto distance matrix,
                aligned with NPClassifier metadata.

Usage
-----
    python -m hddflyzer.dimred.umap features <tag> [n_neighbors] [min_dist]
    python -m hddflyzer.dimred.umap tanimoto <tag> [n_neighbors] [min_dist]
    python -m hddflyzer.dimred.umap pruning  <tag> [n_neighbors] [min_dist]

    python -m hddflyzer.dimred.umap features aocd 15,30,50 0.1,0.5
    python -m hddflyzer.dimred.umap tanimoto aocd
    python -m hddflyzer.dimred.umap pruning  aocd
"""

import os
import sys
import glob
import json
import warnings
import importlib.util
from importlib import metadata as importlib_metadata
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from hddflyzer.config import get_features_path, get_path
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

def _is_umap_available() -> bool:
    try:
        return importlib.util.find_spec("umap.umap_") is not None
    except ModuleNotFoundError:
        return False


_UMAP_AVAILABLE = _is_umap_available()

try:
    from scipy.linalg import orthogonal_procrustes
    from scipy.stats import pearsonr
    from numpy import isfinite
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_N_NEIGHBORS = [15, 30, 50]
DEFAULT_MIN_DIST    = [0.1, 0.5]
EXCLUDED_FOR_UMAP = sorted(EXCLUDED_SIMILARITY_FEATURES)
UMAP_RANDOM_STATE = 42
UMAP_N_EPOCHS = 500
UMAP_LEARNING_RATE = 1.0
UMAP_NEGATIVE_SAMPLE_RATE = 5
UMAP_INIT = "spectral"


# ============================================================
# SHARED UTILITIES
# ============================================================

def _load_dataset(tag: str) -> pd.DataFrame:
    df, path = load_features_table(tag)
    print(f"[INFO] Loading: {path}")
    print(f"[INFO] {len(df)} compounds, {len(df.columns)} columns")
    return df


def _safe_n_neighbors(nn: int, n_samples: int) -> int:
    return min(nn, n_samples - 1) if n_samples > 1 else 2


def _run_umap(
    X: np.ndarray,
    nn: int,
    min_dist: float,
    seed: int = UMAP_RANDOM_STATE,
    metric: str = "euclidean",
) -> np.ndarray:
    try:
        import umap.umap_ as umap_lib
    except ImportError as e:
        raise ImportError(
            "umap-learn is required. Install with: pip install umap-learn"
        ) from e
    reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=nn,
        min_dist=min_dist,
        random_state=seed,
        n_epochs=UMAP_N_EPOCHS,
        learning_rate=UMAP_LEARNING_RATE,
        negative_sample_rate=UMAP_NEGATIVE_SAMPLE_RATE,
        init=UMAP_INIT,
        metric=metric,
    )
    return reducer.fit_transform(X)


def _package_version(package: str) -> Optional[str]:
    """Return installed package version without importing heavy modules."""
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return None


def _library_versions() -> Dict[str, Optional[str]]:
    """Versions relevant for UMAP reproducibility."""
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "umap_learn": _package_version("umap-learn"),
        "numba": _package_version("numba"),
        "pynndescent": _package_version("pynndescent"),
        "scipy": _package_version("scipy"),
    }
    return versions


def _umap_parameters(seed: int, metric: str) -> Dict:
    """Canonical UMAP parameters persisted in metadata."""
    return {
        "random_state": seed,
        "n_epochs": UMAP_N_EPOCHS,
        "learning_rate": UMAP_LEARNING_RATE,
        "negative_sample_rate": UMAP_NEGATIVE_SAMPLE_RATE,
        "init": UMAP_INIT,
        "metric": metric,
        "n_components": 2,
    }


def _umap_interpretation(mode: str, compute_separate: bool = False) -> Dict:
    """Return reusable metadata explaining UMAP coordinate spaces."""
    spaces = {
        "combined": (
            "UMAP trained on BASE + HDDF descriptors together after scaling. "
            "Use this as the default descriptor-space embedding."
        ),
        "base": (
            "UMAP trained only on intrinsic BASE molecular descriptors."
        ),
        "hddf": (
            "UMAP trained only on HDDF scores, then Procrustes-aligned to the "
            "BASE embedding for visual comparison when separate spaces are enabled."
        ),
    }
    return {
        "mode": mode,
        "reproducibility": (
            "UMAP is stochastic; coordinates are reproducible for a fixed input "
            "table, preprocessing, random_state, n_neighbors, min_dist, metric, "
            "and software stack."
        ),
        "coordinate_scale": (
            "UMAP coordinates are embedding coordinates. Axes, origins, and raw "
            "coordinate magnitudes are not directly comparable across parameter "
            "settings or descriptor spaces."
        ),
        "n_neighbors": (
            "Lower n_neighbors emphasizes local neighborhoods; higher values "
            "preserve broader neighborhood structure."
        ),
        "min_dist": (
            "Lower min_dist allows tighter clusters; higher min_dist spreads "
            "nearby points more evenly."
        ),
        "spaces": spaces if compute_separate else {"combined": spaces["combined"]},
        "quality_metrics": (
            "No trustworthiness/continuity metrics are currently persisted. "
            "Use embeddings as exploratory projections, not as metric-preserving maps."
        ),
    }


def _align_procrustes(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Align target embedding to reference using Procrustes analysis."""
    if not _SCIPY_AVAILABLE:
        return target
    ref_c  = reference - reference.mean(axis=0)
    tgt_c  = target    - target.mean(axis=0)
    if ref_c.shape != tgt_c.shape:
        return target
    R, _   = orthogonal_procrustes(tgt_c, ref_c)
    return tgt_c @ R


def _id_columns(df: pd.DataFrame, idx) -> Dict:
    data = {}
    for col in ("identifier", "ID", "id", "SMILES"):
        if col in df.columns:
            data[col] = df.loc[idx, col].astype(str).values
    data["row_id"] = idx.astype(str)
    return data


def _clean_legacy_outputs(output_dir: str, patterns: List[str]) -> None:
    """Remove obsolete files from previous output layouts."""
    for pattern in patterns:
        for path in glob.glob(os.path.join(output_dir, pattern)):
            if os.path.isfile(path):
                os.remove(path)


# ============================================================
# MODE 1: FEATURES
# ============================================================

def run_features(
    tag: str,
    n_neighbors_list: List[int] = None,
    min_dist_list: List[float] = None,
    seed: int = UMAP_RANDOM_STATE,
    min_samples: int = 10,
    compute_separate: bool = True,
) -> bool:
    """
    UMAP from BASE + HDDF descriptors.

    Reads  : results/{tag}/features/full/features_*.csv
    Writes : results/{tag}/dimred/umap/features/
               umap_features_coordinates.csv
               umap_features_metadata.json
    """
    if not _UMAP_AVAILABLE:
        print("[ERROR] umap-learn is not installed.")
        return False

    n_neighbors_list = n_neighbors_list or DEFAULT_N_NEIGHBORS
    min_dist_list    = min_dist_list    or DEFAULT_MIN_DIST
    output_dir       = get_path("umap", tag, "features")
    os.makedirs(output_dir, exist_ok=True)

    try:
        df = _load_dataset(tag)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return False

    # Descriptor selection
    zero_vars = get_zero_variance_descriptors(
        df.select_dtypes(include=[np.number])
          .replace([np.inf, -np.inf], np.nan)
    )
    cats     = categorize_descriptors(df.columns.tolist(), zero_vars)
    all_desc = cats["all_base"] + cats["hddf"]
    all_desc = [c for c in all_desc if c in df.columns]

    if len(all_desc) < 2:
        print("[ERROR] Insufficient descriptors.")
        return False

    X_df = df[all_desc].apply(pd.to_numeric, errors="coerce") \
                       .replace([np.inf, -np.inf], np.nan)
    idx  = X_df.dropna().index
    n    = len(idx)

    if n < min_samples:
        print(f"[ERROR] Too few samples ({n}).")
        return False

    print(f"[INFO] {n} samples | {len(all_desc)} descriptors")
    X_scaled = StandardScaler().fit_transform(X_df.loc[idx].values)

    # Separate BASE / HDDF matrices
    base_cols  = [c for c in cats["all_base"] if c in all_desc]
    hddf_cols  = [c for c in cats["hddf"]     if c in all_desc]
    do_separate = (compute_separate
                   and len(base_cols) >= 2
                   and len(hddf_cols) >= 2)

    if do_separate:
        X_base = StandardScaler().fit_transform(
            df.loc[idx, base_cols].apply(pd.to_numeric, errors="coerce").values)
        X_hddf = StandardScaler().fit_transform(
            df.loc[idx, hddf_cols].apply(pd.to_numeric, errors="coerce").values)

    safe_nn = [_safe_n_neighbors(nn, n) for nn in n_neighbors_list]
    unique_nn = sorted(set(safe_nn))
    for orig, safe in zip(n_neighbors_list, safe_nn):
        if orig != safe:
            print(f"[INFO] n_neighbors adjusted {orig} → {safe}")

    # Cache results
    cache = {}
    param_combos = []
    for nn_safe in unique_nn:
        for min_dist in min_dist_list:
            key = (nn_safe, min_dist)
            print(f"[INFO] UMAP n_neighbors={nn_safe}, min_dist={min_dist}")
            entry = {"combined": _run_umap(X_scaled, nn_safe, min_dist, seed)}
            if do_separate:
                base_emb   = _run_umap(X_base, nn_safe, min_dist, seed)
                hddf_emb   = _run_umap(X_hddf, nn_safe, min_dist, seed)
                hddf_align = _align_procrustes(base_emb, hddf_emb)
                corr_1 = corr_2 = None
                if _SCIPY_AVAILABLE:
                    m1 = isfinite(base_emb[:, 0]) & isfinite(hddf_align[:, 0])
                    m2 = isfinite(base_emb[:, 1]) & isfinite(hddf_align[:, 1])
                    if m1.sum() > 2:
                        corr_1 = float(abs(pearsonr(
                            base_emb[m1, 0], hddf_align[m1, 0])[0]))
                    if m2.sum() > 2:
                        corr_2 = float(abs(pearsonr(
                            base_emb[m2, 1], hddf_align[m2, 1])[0]))
                entry.update({
                    "base": base_emb, "hddf": hddf_emb,
                    "hddf_aligned": hddf_align,
                    "correlations": (corr_1, corr_2),
                })
            cache[key] = entry

    # Build output DataFrame
    coords_data = _id_columns(df, idx)
    for nn_orig, nn_safe in zip(n_neighbors_list, safe_nn):
        for min_dist in min_dist_list:
            key = (nn_safe, min_dist)
            sfx = f"nn{nn_orig}_dist{min_dist}"
            e   = cache[key]
            coords_data[f"UMAP1_combined_{sfx}"] = e["combined"][:, 0]
            coords_data[f"UMAP2_combined_{sfx}"] = e["combined"][:, 1]
            if do_separate and "base" in e:
                coords_data[f"UMAP1_base_{sfx}"] = e["base"][:, 0]
                coords_data[f"UMAP2_base_{sfx}"] = e["base"][:, 1]
                coords_data[f"UMAP1_hddf_{sfx}"] = e["hddf_aligned"][:, 0]
                coords_data[f"UMAP2_hddf_{sfx}"] = e["hddf_aligned"][:, 1]

    for cv in cats["hddf"]:
        if cv in df.columns:
            coords_data[cv] = pd.to_numeric(
                df[cv], errors="coerce").reindex(idx).values

    coords_df = pd.DataFrame(coords_data)

    # Single combo → named file
    single = (len(n_neighbors_list) == 1 and len(min_dist_list) == 1)
    sfx    = f"_nn{n_neighbors_list[0]}_dist{min_dist_list[0]}" if single else ""
    coords_path = os.path.join(output_dir, f"umap_features_coordinates{sfx}.csv")
    meta_path   = os.path.join(output_dir, f"umap_features_metadata{sfx}.json")

    coords_df.to_csv(coords_path, index=False)
    _clean_legacy_outputs(output_dir, ["umap_coordinates*.csv", "umap_metadata*.json"])

    corr_info = {}
    for nn_orig, nn_safe in zip(n_neighbors_list, safe_nn):
        for md in min_dist_list:
            k   = (nn_safe, md)
            sfx = f"nn{nn_orig}_dist{md}"
            if do_separate and "correlations" in cache[k]:
                c1, c2 = cache[k]["correlations"]
                corr_info[sfx] = {
                    "correlation_UMAP1": c1,
                    "correlation_UMAP2": c2,
                }

    meta = {
        "tag": tag, "n_samples": int(n),
        "descriptors_used": {
            "base": base_cols if do_separate else cats["all_base"],
            "hddf": hddf_cols if do_separate else cats["hddf"],
            "zero_variance_removed": zero_vars,
        },
        "parameter_combinations": [
            {"n_neighbors_original": nn, "n_neighbors_used": s, "min_dist": md}
            for nn, s in zip(n_neighbors_list, safe_nn)
            for md in min_dist_list
        ],
        "preprocessing": {
            "scaling": "StandardScaler",
            "nan_handling": "drop_rows_with_any_nan",
            "base_hddf_separate": do_separate,
        },
        "umap_parameters": _umap_parameters(seed, "euclidean"),
        "library_versions": _library_versions(),
        "coordinate_spaces": _umap_interpretation(
            "features",
            compute_separate=do_separate,
        ),
        "base_hddf_correlations": corr_info,
        "timestamp": datetime.now().isoformat(),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    update_manifest(tag, "dimred.umap.features", [coords_path, meta_path], meta)

    print(f"[OK] UMAP (features) complete.")
    print(f"     Coordinates : {coords_path}")
    print(f"     Metadata    : {meta_path}")
    return True


# ============================================================
# MODE 2: TANIMOTO
# ============================================================

def run_tanimoto(
    tag: str,
    n_neighbors_list: List[int] = None,
    min_dist_list: List[float] = None,
    seed: int = UMAP_RANDOM_STATE,
    min_samples: int = 10,
) -> bool:
    """
    UMAP from Tanimoto distance matrix + NPClassifier metadata.

    Reads  : results/{tag}/chemistry/tanimoto/tanimoto_matrix.npz
             results/{tag}/chemistry/tanimoto/tanimoto_ids.csv
             results/{tag}/annotations/npclassifier/npclassifier.csv
    Writes : results/{tag}/dimred/umap/tanimoto/
               umap_tanimoto_coordinates.csv
               umap_tanimoto_metadata.json
    """
    if not _UMAP_AVAILABLE:
        print("[ERROR] umap-learn is not installed.")
        return False

    n_neighbors_list = n_neighbors_list or DEFAULT_N_NEIGHBORS
    min_dist_list = min_dist_list or DEFAULT_MIN_DIST
    output_dir = get_path("umap", tag, "tanimoto")
    os.makedirs(output_dir, exist_ok=True)

    try:
        sim, ids, _, _ = load_tanimoto(tag)
        npc_df, _ = load_npclassifier_success(tag)
        sim, ids, npc_aln = align_tanimoto_with_npclassifier(sim, ids, npc_df)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        return False

    n = sim.shape[0]
    if n < min_samples:
        print(f"[ERROR] Too few samples ({n}).")
        return False

    safe_nn = [_safe_n_neighbors(nn, n) for nn in n_neighbors_list]
    for orig, safe in zip(n_neighbors_list, safe_nn):
        if orig != safe:
            print(f"[INFO] n_neighbors adjusted {orig} -> {safe}")

    dist = np.maximum(0.0, 1.0 - sim).astype(np.float32)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0

    coords_data = {"identifier": ids}
    for col in npc_aln.columns:
        if col != "identifier":
            coords_data[col] = npc_aln[col].values

    param_combos = []
    for nn_orig, nn_safe in zip(n_neighbors_list, safe_nn):
        for min_dist in min_dist_list:
            print(f"[INFO] UMAP (tanimoto) n_neighbors={nn_safe}, min_dist={min_dist}")
            emb = _run_umap(dist, nn_safe, min_dist, seed, metric="precomputed")
            sfx = f"nn{nn_orig}_dist{min_dist}"
            coords_data[f"UMAP1_tanimoto_{sfx}"] = emb[:, 0]
            coords_data[f"UMAP2_tanimoto_{sfx}"] = emb[:, 1]
            param_combos.append({
                "n_neighbors_original": nn_orig,
                "n_neighbors_used": nn_safe,
                "min_dist": min_dist,
            })

    coords_path = os.path.join(output_dir, "umap_tanimoto_coordinates.csv")
    params_path = os.path.join(output_dir, "umap_tanimoto_metadata.json")
    legacy_params = os.path.join(output_dir, "umap_tanimoto_parameters.json")
    if os.path.exists(legacy_params):
        os.remove(legacy_params)
    pd.DataFrame(coords_data).to_csv(coords_path, index=False)
    _clean_legacy_outputs(output_dir, ["umap_coordinates*.csv", "umap_metadata*.json"])

    params = {
        "tag": tag,
        "n_samples": int(n),
        "parameter_combinations": param_combos,
        "umap_parameters": {
            **_umap_parameters(seed, "precomputed"),
        },
        "library_versions": _library_versions(),
        "coordinate_spaces": _umap_interpretation("tanimoto"),
        "source_tanimoto": get_path("tanimoto", tag, "tanimoto_matrix.npz"),
        "source_npclassifier": get_path("npclassifier", tag),
        "timestamp": datetime.now().isoformat(),
    }
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    update_manifest(tag, "dimred.umap.tanimoto", [coords_path, params_path], params)

    print(f"[OK] UMAP (tanimoto) complete.")
    print(f"     Coordinates : {coords_path}")
    print(f"     Parameters  : {params_path}")
    return True


# ============================================================
# MODE 3: PRUNING
# ============================================================

def run_pruning(
    tag: str,
    n_neighbors_list: List[int] = None,
    min_dist_list: List[float] = None,
    seed: int = UMAP_RANDOM_STATE,
    min_samples: int = 10,
) -> bool:
    """
    UMAP using only features selected by feature pruning.

    Reads  : results/{tag}/features/full/features_*.csv
             results/{tag}/features/pruning/selected_features.txt
    Writes : results/{tag}/dimred/umap/pruning/
               umap_pruning_coordinates.csv
               umap_pruning_metadata.json
    """
    if not _UMAP_AVAILABLE:
        print("[ERROR] umap-learn is not installed.")
        return False

    n_neighbors_list = n_neighbors_list or DEFAULT_N_NEIGHBORS
    min_dist_list    = min_dist_list    or DEFAULT_MIN_DIST
    output_dir       = get_path("umap", tag, "pruning")
    os.makedirs(output_dir, exist_ok=True)

    try:
        df       = _load_dataset(tag)
        features = load_selected_features(tag, excluded=EXCLUDED_FOR_UMAP)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return False

    if len(features) < 2:
        print(f"[ERROR] Too few selected features ({len(features)}).")
        return False

    base_desc = [c for c in features if c in df.columns]
    if len(base_desc) < 2:
        print(f"[ERROR] Too few features present in dataset ({len(base_desc)}).")
        return False

    X_df = df[base_desc].apply(pd.to_numeric, errors="coerce") \
                        .replace([np.inf, -np.inf], np.nan)
    zero_vars  = get_zero_variance_descriptors(X_df.dropna())
    base_desc  = [c for c in base_desc if c not in zero_vars]
    X_df       = X_df[base_desc]
    idx        = X_df.dropna().index
    n          = len(idx)

    if n < min_samples:
        print(f"[ERROR] Too few samples ({n}).")
        return False

    print(f"[INFO] {n} samples | {len(base_desc)} pruned features")
    X_scaled = StandardScaler().fit_transform(X_df.loc[idx].values)

    safe_nn = [_safe_n_neighbors(nn, n) for nn in n_neighbors_list]
    results = {}
    param_combos = []
    for nn_orig, nn_safe in zip(n_neighbors_list, safe_nn):
        for min_dist in min_dist_list:
            print(f"[INFO] UMAP n_neighbors={nn_safe}, min_dist={min_dist}")
            key = f"nn{nn_orig}_dist{min_dist}"
            results[key] = _run_umap(X_scaled, nn_safe, min_dist, seed)
            param_combos.append({
                "n_neighbors_original": nn_orig,
                "n_neighbors_used":     nn_safe,
                "min_dist":             min_dist,
            })

    coords_data = _id_columns(df, idx)
    for key, emb in results.items():
        coords_data[f"UMAP1_{key}"] = emb[:, 0]
        coords_data[f"UMAP2_{key}"] = emb[:, 1]

    hddf_vars = ["QED", "LeadLikeness_Score", "Pharma_Complexity",
                 "Synthetic_Accessibility", "Desirability_Profile"]
    for v in hddf_vars:
        if v in df.columns:
            coords_data[v] = pd.to_numeric(
                df[v], errors="coerce").reindex(idx).values

    coords_df   = pd.DataFrame(coords_data)
    coords_path = os.path.join(output_dir, "umap_pruning_coordinates.csv")
    meta_path   = os.path.join(output_dir, "umap_pruning_metadata.json")

    coords_df.to_csv(coords_path, index=False)
    _clean_legacy_outputs(output_dir, ["umap_coordinates*.csv", "umap_metadata*.json"])
    meta = {
        "tag": tag, "n_samples": int(n),
        "features_used": base_desc,
        "features_excluded_zero_variance": zero_vars,
        "parameter_combinations": param_combos,
        "preprocessing": {
            "scaling": "StandardScaler",
            "nan_handling": "drop_rows_with_any_nan",
        },
        "umap_parameters": _umap_parameters(seed, "euclidean"),
        "library_versions": _library_versions(),
        "coordinate_spaces": _umap_interpretation("pruning"),
        "timestamp": datetime.now().isoformat(),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    update_manifest(tag, "dimred.umap.pruning", [coords_path, meta_path], meta)

    print(f"[OK] UMAP (pruning) complete.")
    print(f"     Coordinates : {coords_path}")
    print(f"     Metadata    : {meta_path}")
    return True


# ============================================================
# CLI
# ============================================================

def _parse_int_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def _parse_float_list(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main():
    if len(sys.argv) < 3:
        print("Usage: python -m hddflyzer.dimred.umap <mode> <tag> [n_neighbors] [min_dist]")
        print("  mode: features | tanimoto | pruning")
        print("  Examples:")
        print("    python -m hddflyzer.dimred.umap features aocd")
        print("    python -m hddflyzer.dimred.umap features aocd 15,30,50 0.1,0.5")
        print("    python -m hddflyzer.dimred.umap tanimoto aocd")
        print("    python -m hddflyzer.dimred.umap pruning  aocd")
        sys.exit(1)

    mode = sys.argv[1].strip().lower()
    tag  = sanitize_tag(sys.argv[2])

    nn   = _parse_int_list(sys.argv[3])   if len(sys.argv) > 3 else None
    dist = _parse_float_list(sys.argv[4]) if len(sys.argv) > 4 else None

    if mode == "features":
        ok = run_features(tag, n_neighbors_list=nn, min_dist_list=dist)
    elif mode == "tanimoto":
        ok = run_tanimoto(tag, n_neighbors_list=nn, min_dist_list=dist)
    elif mode == "pruning":
        ok = run_pruning(tag, n_neighbors_list=nn, min_dist_list=dist)
    else:
        print(f"[ERROR] Unknown mode '{mode}'. Use: features | tanimoto | pruning")
        sys.exit(1)

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
