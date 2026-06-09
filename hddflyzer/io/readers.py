# SPDX-License-Identifier: LGPL-3.0-or-later

"""Canonical readers for HDDFlyzer pipeline artifacts."""

import os
import glob
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from hddflyzer.config import get_features_path, get_path

FEATURES_FILE = "features.csv"
FEATURES_PATTERN = "features_*.csv"
FEATURES_METADATA = "features_metadata.json"

REGISTRY_CSV = "molecules.csv"
NPCLASSIFIER_CSV = "npclassifier.csv"
NPCLASSIFIER_METADATA = "npclassifier_metadata.json"

TANIMOTO_MATRIX = "tanimoto_matrix.npz"
TANIMOTO_IDS = "tanimoto_ids.csv"
TANIMOTO_METADATA = "tanimoto_metadata.json"

SELECTED_FEATURES = "selected_features.txt"
PRUNING_METADATA = "pruning_metadata.json"

EXCLUDED_SIMILARITY_FEATURES = {
    "morgan_tanimoto", "featmorgan_tanimoto", "atompair_tanimoto",
    "rdk_tanimoto", "rdkit_tanimoto", "torsion_tanimoto", "layered_tanimoto",
    "maccs_tanimoto", "mcs_overlap", "mcs_tanimoto", "mcs_size",
}


def _latest(paths: Iterable[str]) -> str:
    paths = list(paths)
    if not paths:
        raise FileNotFoundError("No matching files found.")
    return max(paths, key=os.path.getmtime)


def load_features_table(tag: str) -> Tuple[pd.DataFrame, str]:
    """Load the latest canonical features CSV for a dataset tag."""
    canonical = os.path.join(get_features_path(tag), FEATURES_FILE)
    if os.path.exists(canonical):
        path = canonical
    else:
        pattern = os.path.join(get_features_path(tag), FEATURES_PATTERN)
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No features CSV found: {canonical}")
        path = _latest(files)
    df = pd.read_csv(path)
    return df, path


def resolve_features_csv(tag: str, base_dir: str = None) -> str:
    """
    Locate the canonical features CSV for a given tag.

    Returns the first sorted match, preserving historical behavior used by
    several analysis modules.
    """
    if base_dir is None:
        canonical = os.path.join(get_features_path(tag), FEATURES_FILE)
        if os.path.exists(canonical):
            return canonical
        pattern = os.path.join(get_features_path(tag), FEATURES_PATTERN)
    else:
        pattern = os.path.join(base_dir, tag, "features", "full", FEATURES_PATTERN)
    files = sorted(glob.glob(pattern))
    if not files and base_dir is None:
        legacy = os.path.join(get_features_path(tag), f"features_{tag}_*.csv")
        files = sorted(glob.glob(legacy))
    if not files:
        raise FileNotFoundError(
            f"No features CSV found for tag '{tag}'. "
            f"Expected pattern: {pattern}"
        )
    return files[0]


def resolve_features_csv_latest(tag: str, base_dir: str = None) -> str:
    """Locate the most recently modified canonical features CSV for a tag."""
    if base_dir is None:
        canonical = os.path.join(get_features_path(tag), FEATURES_FILE)
        if os.path.exists(canonical):
            return canonical
        pattern = os.path.join(get_features_path(tag), FEATURES_PATTERN)
    else:
        pattern = os.path.join(base_dir, tag, "features", "full", FEATURES_PATTERN)
    files = glob.glob(pattern)
    if not files and base_dir is None:
        legacy = os.path.join(get_features_path(tag), f"features_{tag}_*.csv")
        files = glob.glob(legacy)
    if not files:
        raise FileNotFoundError(
            f"No features CSV found for tag '{tag}'. "
            f"Expected pattern: {pattern}"
        )
    return max(files, key=os.path.getmtime)


def load_features(tag: str, base_dir: str = None) -> pd.DataFrame:
    """Load the canonical features DataFrame for a given tag."""
    path = resolve_features_csv(tag, base_dir)
    print(f"[INFO] Loading: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df)} rows, {len(df.columns)} columns.")
    return df


def load_df(path: str, allow_pickle: bool = False) -> pd.DataFrame:
    """Load a DataFrame from CSV or PKL based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".pkl":
        if not allow_pickle:
            raise ValueError(
                "Pickle loading is disabled by default. Pass "
                "allow_pickle=True only for trusted local files."
            )
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported file format: '{ext}'. Use .csv or .pkl.")


def load_selected_features(tag: str, excluded: Iterable[str] = None) -> List[str]:
    """Load selected feature names from the pruning output."""
    path = get_path("feature_pruning", tag, SELECTED_FEATURES)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Selected features not found: {path}\n"
            f"Run pruning first: hddflyzer chem pruning {tag}")

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    try:
        start = next(i for i, line in enumerate(lines)
                     if line.lower().startswith("total")) + 1
    except StopIteration:
        start = 0

    excluded = set(excluded or [])
    return [f for f in dict.fromkeys(lines[start:]) if f not in excluded]


def load_tanimoto(tag: str) -> Tuple[np.ndarray, List[str], str, str]:
    """Load the canonical Tanimoto matrix and aligned compound IDs."""
    base = get_path("tanimoto", tag)
    mat_path = os.path.join(base, TANIMOTO_MATRIX)
    ids_path = os.path.join(base, TANIMOTO_IDS)
    if not os.path.exists(ids_path):
        ids_path = os.path.join(base, "ids.csv")

    for path in (mat_path, ids_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    data = np.load(mat_path, allow_pickle=False)
    sim = np.clip(np.asarray(data["matrix"], dtype=np.float32), 0.0, 1.0)
    sim = (sim + sim.T) / 2.0
    np.fill_diagonal(sim, 1.0)

    ids_df = pd.read_csv(ids_path)
    id_col = "id" if "id" in ids_df.columns else "identifier"
    ids = ids_df[id_col].astype(str).tolist()
    if len(ids) != sim.shape[0]:
        raise ValueError(
            f"IDs ({len(ids)}) and matrix ({sim.shape[0]}) size mismatch.")

    return sim, ids, mat_path, ids_path


def load_npclassifier_success(tag: str) -> Tuple[pd.DataFrame, str]:
    """Load successful NPClassifier annotations."""
    base = get_path("npclassifier", tag)
    path = os.path.join(base, NPCLASSIFIER_CSV)
    if not os.path.exists(path):
        files = glob.glob(os.path.join(base, "npclassifier*.csv"))
        if not files:
            raise FileNotFoundError(f"No NPClassifier CSV in {base}")
        path = sorted(files)[0]

    df = pd.read_csv(path)
    if "Status" in df.columns:
        df = df[df["Status"] == "Success"].copy()
    df["identifier"] = df["identifier"].astype(str)
    return df, path


def align_tanimoto_with_npclassifier(sim, ids, npc_df):
    """Subset Tanimoto matrix/ids to compounds present in NPClassifier."""
    id_set = set(npc_df["identifier"])
    keep_idx = [i for i, cid in enumerate(ids) if cid in id_set]
    if not keep_idx:
        raise ValueError("No overlap between Tanimoto IDs and NPClassifier IDs.")

    sim_sub = sim[np.ix_(keep_idx, keep_idx)]
    ids_sub = [ids[i] for i in keep_idx]
    npc_map = {cid: i for i, cid in enumerate(npc_df["identifier"])}
    ordered = [npc_map[cid] for cid in ids_sub if cid in npc_map]
    npc_aln = npc_df.iloc[ordered].reset_index(drop=True)
    return sim_sub, ids_sub, npc_aln
