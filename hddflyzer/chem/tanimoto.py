# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Tanimoto similarity matrix computation using Morgan fingerprints.

Reads the local molecule registry,
computes Morgan fingerprints, builds the NxN Tanimoto matrix,
and saves compressed outputs.

Usage
-----
    python -m hddflyzer.chem.tanimoto <tag>
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import List, Tuple

from hddflyzer.config import get_path
from hddflyzer.data.registry import ensure_registry, load_registry
from hddflyzer.utils.columns import find_smiles_column
from hddflyzer.utils.naming import sanitize_tag
from hddflyzer.io import update_manifest

# RDKit is a hard dependency for this module
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False

RADIUS  = 2
NBITS   = 1024
DIVERSITY_THRESHOLD = 0.5
CLUSTER_DIST = 0.4


# ============================================================
# FINGERPRINTS
# ============================================================

def build_morgan_fps(smiles_list: List[str]) -> Tuple[list, List[int]]:
    """
    Build Morgan fingerprints for a list of SMILES.

    Returns
    -------
    fps           : list of RDKit fingerprint objects
    valid_indices : original indices of successfully parsed molecules
    """
    if not _RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for tanimoto computation.")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS, fpSize=NBITS)
    fps, valid_indices = [], []

    for i, smi in enumerate(smiles_list):
        s = str(smi).strip()
        if not s or s.lower() in ("nan", "none", "null", "na", ""):
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            fps.append(generator.GetFingerprint(mol))
            valid_indices.append(i)

    return fps, valid_indices


def compute_tanimoto_matrix(fps) -> np.ndarray:
    """
    Compute the symmetric NxN Tanimoto similarity matrix.
    Diagonal is 1.0.
    """
    n = len(fps)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        mat[i, i] = 1.0
        if i < n - 1:
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
            mat[i, i + 1:] = sims
            mat[i + 1:, i] = sims
    return mat


def stats_from_matrix(mat: np.ndarray) -> dict:
    """Compute summary statistics from the upper triangle."""
    if mat.size == 0 or len(mat) <= 1:
        return {"error": "Matrix too small"}
    vals = mat[np.triu_indices_from(mat, k=1)]
    m = mat.copy()
    np.fill_diagonal(m, 0.0)
    max_sim = m.max(axis=1)
    n_unique = int(np.sum(max_sim < DIVERSITY_THRESHOLD))
    n_clusters = _count_similarity_clusters(mat, CLUSTER_DIST)
    return {
        "mean":        round(float(np.mean(vals)), 6),
        "median":      round(float(np.median(vals)), 6),
        "min":         round(float(np.min(vals)), 6),
        "max":         round(float(np.max(vals)), 6),
        "p10":         round(float(np.percentile(vals, 10)), 6),
        "p90":         round(float(np.percentile(vals, 90)), 6),
        "n_pairs":     int(len(vals)),
        "n_compounds": int(len(mat)),
        "unique_threshold": DIVERSITY_THRESHOLD,
        "n_unique": n_unique,
        "unique_percent": round(n_unique / len(mat) * 100, 6),
        "cluster_distance_threshold": CLUSTER_DIST,
        "n_clusters": n_clusters,
    }


def _count_similarity_clusters(mat: np.ndarray, threshold: float) -> int:
    """Count hierarchical clusters using distance = 1 - Tanimoto."""
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
    except ImportError:
        return 0

    if mat.shape[0] <= 1:
        return int(mat.shape[0])
    dist = np.asarray(1.0 - mat, dtype=np.float32)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    condensed = squareform(dist, checks=False)
    labels = fcluster(linkage(condensed, method="average"), t=threshold, criterion="distance")
    return int(len(np.unique(labels)))


def _remove_if_exists(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


# ============================================================
# PIPELINE
# ============================================================

def run(tag: str) -> bool:
    """
    Compute the Tanimoto matrix for a tag.

    Reads  : results/{tag}/registry/molecules.csv
    Writes : results/{tag}/chemistry/tanimoto/
               tanimoto_matrix.npz
               tanimoto_metadata.json
               tanimoto_ids.csv
               invalid_smiles.csv   (if any)

    """
    if not _RDKIT_AVAILABLE:
        print("[ERROR] RDKit is not installed. Cannot compute Tanimoto matrix.")
        return False

    output_dir = get_path("tanimoto", tag)
    os.makedirs(output_dir, exist_ok=True)

    try:
        registry_path = ensure_registry(tag)
        df = load_registry(tag, valid_only=True)
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

    print(f"[INFO] Tag      : {tag}")
    print(f"[INFO] Registry : {registry_path}")
    print(f"[INFO] Output   : {output_dir}/")
    for stale in (
        "tanimoto_matrix.csv", "fingerprints_dense.npz", "fingerprints.pkl",
        "ids.csv", "tanimoto_stats.json", "tanimoto_summary.txt",
        "tanimoto_order.csv",
    ):
        _remove_if_exists(os.path.join(output_dir, stale))

    if df.empty:
        print("[ERROR] No rows to process.")
        return False

    id_col     = "identifier"
    smiles_col = "canonical_smiles" if "canonical_smiles" in df.columns else find_smiles_column(df)
    smiles_list = df[smiles_col].astype(str).tolist()
    ids_list    = df[id_col].astype(str).tolist()

    print(f"[INFO] Processing {len(smiles_list)} compounds...")

    # Fingerprints
    fps, valid_idx = build_morgan_fps(smiles_list)
    n_invalid = len(smiles_list) - len(fps)
    print(f"[INFO] Valid: {len(fps)} | Invalid: {n_invalid}")

    if not fps:
        print("[ERROR] No valid fingerprints generated.")
        return False

    # Valid and invalid IDs
    valid_ids = [ids_list[i] for i in valid_idx]
    pd.DataFrame({"id": valid_ids}).to_csv(
        os.path.join(output_dir, "tanimoto_ids.csv"), index=False)

    invalid_idx = [i for i in range(len(smiles_list)) if i not in valid_idx]
    if invalid_idx:
        pd.DataFrame([{"index": i, "id": ids_list[i], "smiles": smiles_list[i]}
                      for i in invalid_idx]).to_csv(
            os.path.join(output_dir, "invalid_smiles.csv"), index=False)
    else:
        _remove_if_exists(os.path.join(output_dir, "invalid_smiles.csv"))

    # Tanimoto matrix
    print("[INFO] Computing Tanimoto matrix...")
    mat = compute_tanimoto_matrix(fps)
    np.savez_compressed(os.path.join(output_dir, "tanimoto_matrix.npz"),
                        matrix=mat)

    # Statistics
    stats = stats_from_matrix(mat)
    metadata_path = os.path.join(output_dir, "tanimoto_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(stats, f, indent=2)
    manifest_metadata = {
        **stats,
        "tag": tag,
        "source_registry": registry_path,
        "id_column": id_col,
        "smiles_column": smiles_col,
        "n_input_compounds": int(len(smiles_list)),
        "n_valid_fingerprints": int(len(fps)),
        "n_invalid_smiles": int(n_invalid),
        "fingerprint": {
            "type": "Morgan",
            "radius": RADIUS,
            "n_bits": NBITS,
        },
    }
    update_manifest(
        tag,
        "chem.tanimoto",
        [
            os.path.join(output_dir, "tanimoto_matrix.npz"),
            os.path.join(output_dir, "tanimoto_ids.csv"),
            metadata_path,
        ],
        manifest_metadata,
    )

    print(f"\n[OK] Tanimoto matrix: {mat.shape}")
    print(f"     Mean similarity : {stats['mean']:.3f}")
    print(f"     Output          : {output_dir}/")
    return True


# ============================================================
# CLI
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m hddflyzer.chem.tanimoto <tag>")
        print("Example: python -m hddflyzer.chem.tanimoto aocd")
        sys.exit(1)

    tag = sanitize_tag(sys.argv[1])
    if not run(tag):
        sys.exit(1)


if __name__ == "__main__":
    main()
