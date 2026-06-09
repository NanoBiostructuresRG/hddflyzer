# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Stratified sampling from a Tanimoto similarity matrix.

Stratifies compounds into three groups based on their maximum
pairwise similarity (unique / intermediate / analogue) and
draws a balanced sample from each stratum.

Sampling is a reduction step only when a stratum contains more compounds than
k_per_stratum. If all strata are smaller than or equal to k_per_stratum, every
compound is retained and downstream --sampled feature calculation is equivalent
to the full-registry feature calculation.

Usage
-----
    python -m hddflyzer.chem.tanimoto_sampling <tag>
    python -m hddflyzer.chem.tanimoto_sampling <tag> k_per_stratum=300
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Tuple

from hddflyzer.config import get_path
from hddflyzer.utils.naming import sanitize_tag
from hddflyzer.io import load_tanimoto, update_manifest

# ============================================================
# DEFAULT CONFIG
# ============================================================

DEFAULTS = {
    "base_dir":      None,
    "ids_file":      None,
    "unique_thr":    0.30,   # max_sim < unique_thr  → 'unique'
    "analogue_thr":  0.85,   # max_sim >= analogue_thr → 'analogue'
    "k_per_stratum": 300,
    "seed":          42,
}


# ============================================================
# UTILITIES
# ============================================================

def parse_overrides(tokens: list) -> dict:
    """Parse key=value tokens from CLI args."""
    out = {}
    for tok in tokens:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        k, v = k.strip(), v.strip()
        if k in ("unique_thr", "analogue_thr"):
            v = float(v)
        elif k in ("k_per_stratum", "seed"):
            v = int(v)
        out[k] = v
    return out


def load_matrix(npz_path: str) -> np.ndarray:
    """Load a dense NxN float32 matrix from a .npz file."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Matrix file not found: {npz_path}")
    z   = np.load(npz_path, allow_pickle=False)
    key = "matrix" if "matrix" in z.files else (z.files[0] if z.files else None)
    if key is None:
        raise ValueError(f"No arrays found in {npz_path}")
    mat = np.asarray(z[key], dtype=np.float32)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {mat.shape}")
    return mat


def max_sim_per_compound(mat: np.ndarray) -> np.ndarray:
    """For each compound i, return max_{j≠i} mat[i,j]."""
    m = mat.copy()
    np.fill_diagonal(m, 0.0)
    return m.max(axis=1)


def stratify(
    max_sim: np.ndarray,
    unique_thr: float,
    analogue_thr: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split compound indices into unique / intermediate / analogue."""
    unique       = np.where(max_sim < unique_thr)[0]
    intermediate = np.where((max_sim >= unique_thr) & (max_sim < analogue_thr))[0]
    analogue     = np.where(max_sim >= analogue_thr)[0]
    return unique, intermediate, analogue


def balanced_sample(indices: np.ndarray, k: int, rng) -> np.ndarray:
    """
    Sample up to k indices from an array without replacement.

    If the stratum has k or fewer compounds, all indices are returned. This
    preserves small strata and means sampling may intentionally keep the full
    dataset when k_per_stratum is larger than each stratum.
    """
    if indices.size == 0 or indices.size <= k:
        return indices
    return rng.choice(indices, size=k, replace=False)


def sample_from_matrix(
    mat: np.ndarray,
    ids,
    unique_thr: float,
    analogue_thr: float,
    k_per_stratum: int,
    seed: int,
) -> Tuple[pd.DataFrame, dict]:
    """Sample IDs from a Tanimoto matrix and return sampled rows plus metadata."""
    ids = np.asarray(ids, dtype=str)
    n = mat.shape[0]

    if ids.size != n:
        n = min(ids.size, n)
        ids = ids[:n]
        mat = mat[:n, :n]

    max_sim = max_sim_per_compound(mat)
    u_idx, mid_idx, a_idx = stratify(max_sim, unique_thr, analogue_thr)

    rng = np.random.default_rng(seed)
    u_s = balanced_sample(u_idx, k_per_stratum, rng)
    mid_s = balanced_sample(mid_idx, k_per_stratum, rng)
    a_s = balanced_sample(a_idx, k_per_stratum, rng)

    sampled_idx = np.unique(np.concatenate([u_s, mid_s, a_s]))
    sampled_ids = ids[sampled_idx]

    group = np.full(sampled_idx.shape, "intermediate", dtype=object)
    group[np.isin(sampled_idx, u_s)] = "unique"
    group[np.isin(sampled_idx, a_s)] = "analogue"

    sampled_df = pd.DataFrame({
        "id": sampled_ids,
        "index": sampled_idx,
        "group": group,
    })
    metadata = {
        "matrix_size": int(n),
        "unique_threshold": float(unique_thr),
        "analogue_threshold": float(analogue_thr),
        "k_per_stratum": int(k_per_stratum),
        "seed": int(seed),
        "strata_totals": {
            "unique": int(u_idx.size),
            "intermediate": int(mid_idx.size),
            "analogue": int(a_idx.size),
        },
        "sample_size": int(sampled_ids.size),
    }
    return sampled_df, metadata


# ============================================================
# PIPELINE
# ============================================================

def run(tag: str, **overrides) -> bool:
    """
    Run stratified Tanimoto sampling for a tag.

    Reads  : results/{tag}/chemistry/tanimoto/tanimoto_matrix.npz
             results/{tag}/chemistry/tanimoto/tanimoto_ids.csv
    Writes : results/{tag}/chemistry/tanimoto/samples/
               sampled_ids.csv
               sampling_metadata.json

    Parameters
    ----------
    tag      : Dataset tag.
    overrides: Any key from DEFAULTS to override
               (e.g. k_per_stratum=200).

    Notes
    -----
    sampled_ids.csv is a filter for downstream steps such as
    hddflyzer chem features <tag> --sampled. It is most useful when the sample
    is smaller than the registry. If sample_size equals matrix_size in
    sampling_metadata.json, --sampled and full feature calculation use the same
    molecules.
    """
    cfg = {**DEFAULTS, **overrides}

    base = os.path.join(cfg["base_dir"], tag) if cfg["base_dir"] else get_path("tanimoto", tag)
    out_dir   = os.path.join(base, "samples")
    os.makedirs(out_dir, exist_ok=True)
    stale_smiles = os.path.join(out_dir, "sampled_with_smiles.csv")
    if os.path.exists(stale_smiles):
        os.remove(stale_smiles)

    print(f"[INFO] Tag            : {tag}")
    print(f"[INFO] unique_thr     : {cfg['unique_thr']}")
    print(f"[INFO] analogue_thr   : {cfg['analogue_thr']}")
    print(f"[INFO] k_per_stratum  : {cfg['k_per_stratum']}")

    # Load matrix and IDs through the canonical I/O layer when possible.
    if cfg["base_dir"] or cfg["ids_file"]:
        mat_path = os.path.join(base, "tanimoto_matrix.npz")
        ids_path = cfg["ids_file"] or os.path.join(base, "tanimoto_ids.csv")
        print(f"[INFO] Loading matrix : {mat_path}")
        mat = load_matrix(mat_path)
        ids_df = pd.read_csv(ids_path)
        id_col = "id" if "id" in ids_df.columns else "identifier"
        if id_col not in ids_df.columns:
            raise ValueError("IDs file must contain an 'id' or 'identifier' column.")
        ids = ids_df[id_col].astype(str).values
    else:
        mat, ids_list, mat_path, ids_path = load_tanimoto(tag)
        ids = np.asarray(ids_list, dtype=str)
        print(f"[INFO] Loading matrix : {mat_path}")

    n = mat.shape[0]
    print(f"[INFO] Matrix size    : {n}x{n}")

    if ids.size != n:
        n = min(ids.size, n)
        ids = ids[:n]
        mat = mat[:n, :n]
        print(f"[WARN] ID/matrix size mismatch — truncated to {n}")

    sampled_df, metadata = sample_from_matrix(
        mat,
        ids,
        cfg["unique_thr"],
        cfg["analogue_thr"],
        cfg["k_per_stratum"],
        cfg["seed"],
    )
    strata = metadata["strata_totals"]
    print(f"[INFO] Strata — unique: {strata['unique']} | "
          f"intermediate: {strata['intermediate']} | analogue: {strata['analogue']}")

    # Save sampled IDs
    out_csv = os.path.join(out_dir, "sampled_ids.csv")
    sampled_df.to_csv(out_csv, index=False)
    print(f"[OK] Sampled IDs saved: {out_csv} (n={len(sampled_df)})")

    stale_summary = os.path.join(out_dir, "sampled_summary.txt")
    if os.path.exists(stale_summary):
        os.remove(stale_summary)

    out_json = os.path.join(out_dir, "sampling_metadata.json")
    metadata = {
        "tag": tag,
        **metadata,
        "files": {"sampled_ids": os.path.basename(out_csv)},
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    update_manifest(tag, "chem.sample", [out_csv, out_json], metadata)
    print(f"[OK] Metadata saved : {out_json}")

    return True


# ============================================================
# CLI
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m hddflyzer.chem.tanimoto_sampling <tag> [key=value ...]")
        print("Keys  : unique_thr, analogue_thr, k_per_stratum, seed")
        print("Example:")
        print("  python -m hddflyzer.chem.tanimoto_sampling aocd k_per_stratum=100")
        sys.exit(1)

    tag = sanitize_tag(sys.argv[1])
    overrides = parse_overrides(sys.argv[2:])

    try:
        if not run(tag, **overrides):
            sys.exit(1)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
