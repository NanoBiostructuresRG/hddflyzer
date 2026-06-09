# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Canonical molecule registry.

This module normalizes an input dataset into:

    results/{tag}/registry/molecules.csv

The registry is intentionally local and API-independent. It contains stable
identifiers, SMILES strings, optional canonical SMILES, validity flags, and
source provenance. Chemistry calculations and external annotations can then
share the same molecule base.
"""

import glob
import os
import sys
from typing import Optional, Tuple

import pandas as pd

from hddflyzer.config import DATA_DIR, get_path
from hddflyzer.utils.columns import find_id_column, find_smiles_column
from hddflyzer.utils.naming import sanitize_tag
from hddflyzer.io import update_manifest

try:
    from rdkit import Chem
    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False


def find_input_file(tag: str, data_dir: str = DATA_DIR) -> str:
    """Locate a CSV input file for tag inside data_dir."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: '{data_dir}'")

    pattern = os.path.join(data_dir, "*.csv")
    matches = [
        f for f in glob.glob(pattern)
        if tag.lower() in os.path.basename(f).lower()
    ]
    if not matches:
        available = [os.path.basename(f) for f in glob.glob(pattern)]
        raise FileNotFoundError(
            f"No CSV found in '{data_dir}/' containing tag '{tag}'.\n"
            f"Available: {available}"
        )
    if len(matches) > 1:
        print(f"[WARN] Multiple files match tag '{tag}'. Using: {os.path.basename(matches[0])}")
    return matches[0]


def resolve_registry_csv(tag: str) -> str:
    """Return the expected registry CSV path for a tag."""
    return get_path("registry", tag, "molecules.csv")


def _normalize_smiles(smiles: object) -> Tuple[str, str, bool, str]:
    raw = "" if pd.isna(smiles) else str(smiles).strip()
    if not raw or raw.lower() in ("nan", "none", "null", "na"):
        return raw, "", False, "empty_smiles"

    if not _RDKIT_AVAILABLE:
        return raw, raw, len(raw) > 3, "not_checked_rdkit_unavailable"

    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        return raw, "", False, "invalid_smiles"
    return raw, Chem.MolToSmiles(mol, canonical=True), True, "valid"


def build_registry_frame(df: pd.DataFrame, source_file: str = "") -> Tuple[pd.DataFrame, dict]:
    """Build a canonical registry DataFrame and core metadata from input rows."""
    smiles_col = find_smiles_column(df)
    try:
        id_col = find_id_column(df)
        identifiers = df[id_col].astype(str)
    except ValueError:
        id_col = None
        identifiers = pd.Series(
            [f"COMPD_{i + 1:04d}" for i in range(len(df))],
            index=df.index,
        )

    rows = []
    source_name = os.path.basename(source_file) if source_file else ""
    for idx, row in df.iterrows():
        raw, canonical, valid, status = _normalize_smiles(row[smiles_col])
        rows.append({
            "identifier": identifiers.loc[idx],
            "SMILES": raw,
            "canonical_smiles": canonical,
            "valid_smiles": bool(valid),
            "mol_parse_status": status,
            "source_file": source_name,
            "source_row": int(idx),
        })

    reg = pd.DataFrame(rows)
    before = len(reg)
    reg = reg.drop_duplicates(subset=["identifier"], keep="first").reset_index(drop=True)
    duplicates_removed = before - len(reg)
    validation = validate_registry_frame(reg)

    meta = {
        "source_id_column": id_col,
        "source_smiles_column": smiles_col,
        "n_source_rows": int(len(df)),
        "n_registry_rows": int(len(reg)),
        "n_valid_smiles": int(reg["valid_smiles"].sum()),
        "duplicates_removed_by_identifier": int(duplicates_removed),
        "validation": validation,
    }
    return reg, meta


def prepare_registry(
    tag: str,
    input_file: Optional[str] = None,
    data_dir: str = DATA_DIR,
    output_dir: Optional[str] = None,
) -> str:
    """
    Create the canonical molecule registry for a tag.

    Returns the molecules.csv path.
    """
    tag = sanitize_tag(tag)
    input_file = input_file or find_input_file(tag, data_dir)
    output_dir = output_dir or get_path("registry", tag)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_file)
    reg, core_meta = build_registry_frame(df, source_file=input_file)

    out_csv = os.path.join(output_dir, "molecules.csv")
    reg.to_csv(out_csv, index=False)

    meta = {
        "tag": tag,
        "source_file": input_file,
        "rdkit_available": _RDKIT_AVAILABLE,
        **core_meta,
    }
    legacy_metadata = os.path.join(output_dir, "registry_metadata.json")
    if os.path.exists(legacy_metadata):
        os.remove(legacy_metadata)
    update_manifest(tag, "data.prepare", [out_csv], meta)

    print(f"[OK] Registry saved: {out_csv}")
    print(f"     Molecules    : {len(reg)}")
    print(f"     Valid SMILES : {int(reg['valid_smiles'].sum())}")
    print(f"     Duplicates   : {core_meta['validation']['duplicated_identifiers']}")
    return out_csv


def ensure_registry(tag: str, data_dir: str = DATA_DIR) -> str:
    """Return registry path, creating it from data_dir when missing."""
    tag = sanitize_tag(tag)
    path = resolve_registry_csv(tag)
    if os.path.exists(path):
        return path
    print(f"[INFO] Registry not found for '{tag}'. Preparing from {data_dir}/")
    return prepare_registry(tag, data_dir=data_dir)


def load_registry(tag: str, valid_only: bool = True) -> pd.DataFrame:
    """Load the registry for tag."""
    path = resolve_registry_csv(tag)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Registry not found: {path}\n"
            f"Run: hddflyzer data prepare {tag}"
        )
    df = pd.read_csv(path)
    if valid_only and "valid_smiles" in df.columns:
        df = df[df["valid_smiles"] == True].copy()
    return df


def validate_registry_frame(df: pd.DataFrame) -> dict:
    """Return compact validation statistics for a registry DataFrame."""
    n = len(df)
    valid = int(df["valid_smiles"].sum()) if "valid_smiles" in df.columns else 0
    duplicated = int(df["identifier"].duplicated().sum()) if "identifier" in df.columns else 0
    return {
        "n_rows": int(n),
        "n_valid_smiles": int(valid),
        "duplicated_identifiers": int(duplicated),
        "ok": bool(duplicated == 0 and valid > 0),
    }


def run_prepare(tag: str, input_file: Optional[str] = None) -> bool:
    try:
        prepare_registry(tag, input_file=input_file)
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def main():
    if len(sys.argv) < 3:
        print("Usage: python -m hddflyzer.data.registry prepare <tag> [input_csv]")
        sys.exit(1)

    mode = sys.argv[1].strip().lower()
    tag = sanitize_tag(sys.argv[2])
    input_file = sys.argv[3] if len(sys.argv) > 3 else None

    if mode == "prepare":
        ok = run_prepare(tag, input_file=input_file)
    else:
        print(f"[ERROR] Unknown mode '{mode}'. Use: prepare")
        ok = False

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
