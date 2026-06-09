# SPDX-License-Identifier: LGPL-3.0-or-later

"""Create ML-ready feature tables from intrinsic molecular descriptors."""

import json
import os
import sys
from typing import Dict, List, Tuple

import pandas as pd

from hddflyzer.config import get_path
from hddflyzer.io import load_features_table, update_manifest
from hddflyzer.utils.descriptors import HYBRID_BASE
from hddflyzer.utils.naming import sanitize_tag

try:
    from rdkit import Chem, rdBase
    _RDKIT_AVAILABLE = True
except ImportError:
    rdBase = None
    _RDKIT_AVAILABLE = False

ID_COLUMNS = ["identifier", "SMILES"]

REDUNDANT_COLUMNS = [
    "HeavyAtomMolWt",
    "NHOHCount",
    "NOCount",
]

HIGH_ORDER_CONNECTIVITY_COLUMNS = [
    "Chi4",
    "Chi4v",
]

DERIVED_COLUMNS = list(HYBRID_BASE)

ATOM_COUNT_COLUMNS = [
    "NumCarbonAtoms",
    "NumNitrogenAtoms",
    "NumOxygenAtoms",
]


def _remove_columns(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    present = [c for c in columns if c in df.columns and c not in ID_COLUMNS]
    return df.drop(columns=present), present


def _zero_variance_columns(df: pd.DataFrame) -> List[str]:
    numeric = df.select_dtypes(include="number")
    return [
        col for col in numeric.columns
        if col not in ID_COLUMNS and numeric[col].nunique(dropna=False) <= 1
    ]


def _atom_counts(smiles: object) -> Dict[str, int]:
    counts = {col: 0 for col in ATOM_COUNT_COLUMNS}
    if not _RDKIT_AVAILABLE:
        return counts
    mol = Chem.MolFromSmiles("" if pd.isna(smiles) else str(smiles))
    if mol is None:
        return counts
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num == 6:
            counts["NumCarbonAtoms"] += 1
        elif atomic_num == 7:
            counts["NumNitrogenAtoms"] += 1
        elif atomic_num == 8:
            counts["NumOxygenAtoms"] += 1
    return counts


def add_atom_counts(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Add simple atom-count descriptors when SMILES are available."""
    if "SMILES" not in df.columns:
        return df, []
    out = df.copy()
    counts = pd.DataFrame([_atom_counts(smi) for smi in out["SMILES"]])
    for col in ATOM_COUNT_COLUMNS:
        out[col] = counts[col].values
    return out, ATOM_COUNT_COLUMNS.copy()


def curate_features(
    df: pd.DataFrame,
    drop_redundant: bool = True,
    drop_derived: bool = True,
    drop_high_order_connectivity: bool = True,
    drop_zero_variance: bool = True,
    add_atoms: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """Return an ML-ready feature table and curation metadata."""
    curated = df.copy()
    removed = {}
    added = []

    if add_atoms:
        curated, added = add_atom_counts(curated)

    if drop_redundant:
        curated, removed["redundant"] = _remove_columns(curated, REDUNDANT_COLUMNS)

    if drop_derived:
        curated, removed["derived_ratios"] = _remove_columns(curated, DERIVED_COLUMNS)

    if drop_high_order_connectivity:
        curated, removed["high_order_connectivity"] = _remove_columns(
            curated, HIGH_ORDER_CONNECTIVITY_COLUMNS)

    if drop_zero_variance:
        zero_cols = _zero_variance_columns(curated)
        curated, removed["zero_variance"] = _remove_columns(curated, zero_cols)

    metadata = {
        "input_shape": {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
        },
        "output_shape": {
            "rows": int(len(curated)),
            "columns": int(len(curated.columns)),
        },
        "added_features": added,
        "removed_features": removed,
        "rules": {
            "drop_redundant": bool(drop_redundant),
            "drop_derived": bool(drop_derived),
            "drop_high_order_connectivity": bool(drop_high_order_connectivity),
            "drop_zero_variance": bool(drop_zero_variance),
            "add_atom_counts": bool(add_atoms),
        },
        "chemistry_backend": {
            "rdkit_available": bool(_RDKIT_AVAILABLE),
            "rdkit_version": rdBase.rdkitVersion if _RDKIT_AVAILABLE else None,
        },
        "descriptor_parameters": {
            "atom_counts": {
                "enabled": bool(add_atoms),
                "source": "RDKit MolFromSmiles atomic numbers",
                "elements": ["C", "N", "O"],
            },
            "morgan_fingerprint": {
                "applies": False,
                "reason": (
                    "features_ml.csv is curated from intrinsic molecular "
                    "descriptors and does not include reference or pairwise "
                    "fingerprint similarities."
                ),
            },
            "reference_similarity": {
                "applies": False,
                "reason": (
                    "Reference-dependent Tanimoto and MCS features are stored "
                    "separately under features/reference/."
                ),
            },
        },
        "notes": (
            "Curated table for ML workflows. The canonical chemistry table "
            "remains results/{tag}/features/full/features.csv."
        ),
    }
    return curated, metadata


def run(tag: str) -> bool:
    """Create results/{tag}/features/curated/features_ml.csv."""
    try:
        df, input_path = load_features_table(tag)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return False

    curated, metadata = curate_features(df)
    metadata["tag"] = tag
    metadata["source_features"] = input_path

    out_dir = get_path("features_curated", tag)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "features_ml.csv")
    meta_path = os.path.join(out_dir, "feature_curation_metadata.json")

    curated.to_csv(out_csv, index=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    update_manifest(tag, "chem.curate_features", [out_csv, meta_path], metadata)

    removed_total = sum(len(v) for v in metadata["removed_features"].values())
    print(f"[OK] Curated ML features: {out_csv}")
    print(f"     Input columns  : {metadata['input_shape']['columns']}")
    print(f"     Output columns : {metadata['output_shape']['columns']}")
    print(f"     Removed        : {removed_total}")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m hddflyzer.chem.feature_curation <tag>")
        print("Example: python -m hddflyzer.chem.feature_curation aocd")
        sys.exit(1)
    tag = sanitize_tag(sys.argv[1])
    if not run(tag):
        sys.exit(1)


if __name__ == "__main__":
    main()
