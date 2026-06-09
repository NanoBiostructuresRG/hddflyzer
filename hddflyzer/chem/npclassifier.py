# SPDX-License-Identifier: LGPL-3.0-or-later

"""
NPClassifier client for HDDFlyzer.

Classifies molecules via the NPClassifier REST API
(https://npclassifier.ucsd.edu) and saves results as CSV.

Usage
-----
    python -m hddflyzer.chem.npclassifier <tag>
"""

import os
import sys
import csv
import time
import json
import requests
import pandas as pd
from typing import Dict, List, Optional, Tuple

from hddflyzer.config import DATA_DIR, get_path
from hddflyzer.data.registry import ensure_registry, load_registry
from hddflyzer.utils.columns import find_smiles_column, find_id_column
from hddflyzer.utils.naming import sanitize_tag
from hddflyzer.io import update_manifest

API_URL = "https://npclassifier.ucsd.edu/classify"
REQUEST_DELAY = 0.5   # seconds between API calls


# ============================================================
# API CLIENT
# ============================================================

class NPClassifierClient:
    """Thin wrapper around the NPClassifier REST API."""

    def __init__(self, base_url: str = API_URL):
        self.base_url = base_url
        self.session  = requests.Session()
        self.session.headers.update({
            "User-Agent": "HDDFlyzer-NPClassifier/1.0",
            "Accept":     "application/json",
        })

    def classify(self, smiles: str) -> Optional[Dict]:
        """
        Classify a single SMILES string.

        Returns the API response dict (fingerprint fields removed),
        or None on failure.
        """
        try:
            resp = self.session.get(self.base_url,
                                    params={"smiles": smiles},
                                    timeout=30)
            resp.raise_for_status()
            result = resp.json()
            result.pop("fp1", None)
            result.pop("fp2", None)
            return result
        except requests.exceptions.RequestException as e:
            print(f"[WARN] API error for '{smiles[:40]}...': {e}")
            return None

    def classify_batch(
        self,
        smiles_list: List[str],
        delay: float = REQUEST_DELAY,
    ) -> List[Optional[Dict]]:
        """
        Classify a list of SMILES with progress output.

        Parameters
        ----------
        smiles_list : list of str
        delay : float
            Seconds between requests (be nice to the API).

        Returns
        -------
        list of dict or None
        """
        results = []
        print("Progress: ", end="", flush=True)
        total = len(smiles_list)

        for i, smiles in enumerate(smiles_list):
            results.append(self.classify(smiles))
            print("Y", end="", flush=True)

            if (i + 1) % 50 == 0:
                print(f" {i + 1}/{total}")
                if i + 1 < total:
                    print("Progress: ", end="", flush=True)

            if i < total - 1:
                time.sleep(delay)

        if total % 50 != 0:
            print(f" {total}/{total}")
        else:
            print()

        return results


# ============================================================
# I/O
# ============================================================

def find_input_file(tag: str, data_dir: str = "data") -> str:
    """
    Locate the input CSV for a tag inside the configured input directory.

    Searches for any CSV whose filename contains the tag
    (case-insensitive).
    """
    import glob
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: '{data_dir}'")

    pattern = os.path.join(data_dir, "*.csv")
    matches = [f for f in glob.glob(pattern)
               if tag.lower() in os.path.basename(f).lower()]

    if not matches:
        available = [os.path.basename(f) for f in glob.glob(pattern)]
        raise FileNotFoundError(
            f"No CSV found in '{data_dir}/' containing tag '{tag}'.\n"
            f"Available: {available}"
        )
    if len(matches) > 1:
        print(f"[WARN] Multiple files match tag '{tag}'. Using: {os.path.basename(matches[0])}")

    return matches[0]


def read_smiles_and_ids(file_path: str) -> Tuple[List[str], List[str]]:
    """
    Read SMILES and identifiers from a CSV file.

    Auto-detects SMILES column (prefers smiles_rdkit, then
    canonical_smiles, then any column containing 'smiles').
    Auto-detects identifier column.

    Returns
    -------
    smiles_list, identifiers_list : filtered lists (removes empty/nan SMILES)
    """
    df = pd.read_csv(file_path)

    # SMILES column priority
    for kw in ["smiles_rdkit", "canonical_smiles"]:
        cols = [c for c in df.columns if kw in c.lower()]
        if cols:
            smiles_col = cols[0]
            break
    else:
        smiles_col = find_smiles_column(df)

    try:
        id_col = find_id_column(df)
        ids = df[id_col].astype(str).tolist()
    except ValueError:
        ids = [f"COMPD_{i+1:04d}" for i in range(len(df))]

    smiles_raw = df[smiles_col].astype(str).tolist()
    print(f"[INFO] SMILES column: '{smiles_col}' | {len(smiles_raw)} rows")

    valid_smiles, valid_ids = [], []
    for smi, cid in zip(smiles_raw, ids):
        if smi and len(smi) > 3 and smi.lower() not in ("nan", "none", "na"):
            valid_smiles.append(smi)
            valid_ids.append(cid)

    print(f"[INFO] Valid SMILES: {len(valid_smiles)} / {len(smiles_raw)}")
    return valid_smiles, valid_ids


def save_results(
    results: List[Optional[Dict]],
    smiles_list: List[str],
    ids_list: List[str],
    tag: str,
    output_dir: str = None,
) -> str:
    """
    Save classification results to results/{tag}/annotations/npclassifier/npclassifier.csv.

    Returns the output file path.
    """
    if output_dir is None:
        output_dir = get_path("npclassifier", tag)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "npclassifier.csv")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["identifier", "SMILES", "Pathway",
                         "Superclass", "Class", "Is_Glycoside", "Status"])

        for cid, smi, result in zip(ids_list, smiles_list, results):
            if result:
                pathway    = (result.get("pathway_results") or ["Unknown"])[0]
                superclass = (result.get("superclass_results") or ["Unknown"])[0]
                cls        = (result.get("class_results") or ["Unknown"])[0]
                glycoside  = result.get("isglycoside", False)
                status     = "Success"
            else:
                pathway = superclass = cls = "Error"
                glycoside = False
                status = "Failed"

            writer.writerow([cid, smi, pathway, superclass, cls, glycoside, status])

    print(f"[OK] Results saved: {output_path}")
    return output_path


# ============================================================
# PIPELINE
# ============================================================

def run(tag: str, data_dir: str = DATA_DIR) -> bool:
    """
    Run the NPClassifier pipeline for a tag.

    Reads  : results/{tag}/registry/molecules.csv
    Writes : results/{tag}/annotations/npclassifier/npclassifier.csv

    Parameters
    ----------
    tag      : Dataset tag.
    data_dir : Directory containing input CSV files.
    """
    try:
        registry_path = ensure_registry(tag, data_dir=data_dir)
        registry = load_registry(tag, valid_only=True)
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

    print(f"[INFO] Registry: {registry_path}")
    smiles_col = "canonical_smiles" if "canonical_smiles" in registry.columns else "SMILES"
    registry = registry[registry[smiles_col].astype(str).str.len() > 0].copy()
    smiles_list = registry[smiles_col].astype(str).tolist()
    ids_list = registry["identifier"].astype(str).tolist()
    print(f"[INFO] Molecules from registry: {len(smiles_list)}")

    if not smiles_list:
        print("[ERROR] No valid SMILES found.")
        return False

    print(f"\n=== NPClassifier: {tag} ({len(smiles_list)} compounds) ===")
    t0 = time.time()

    client  = NPClassifierClient()
    results = client.classify_batch(smiles_list)

    elapsed = time.time() - t0
    success = sum(1 for r in results if r is not None)

    print(f"\n[SUMMARY]")
    print(f"  Tag           : {tag}")
    print(f"  Total         : {len(smiles_list)}")
    print(f"  Successful    : {success}")
    print(f"  Failed        : {len(smiles_list) - success}")
    print(f"  Success rate  : {success / len(smiles_list) * 100:.1f}%")
    print(f"  Elapsed       : {elapsed:.1f}s")

    output_path = save_results(results, smiles_list, ids_list, tag)
    metadata = {
        "tag": tag,
        "n_total": int(len(smiles_list)),
        "n_successful": int(success),
        "n_failed": int(len(smiles_list) - success),
        "success_rate": float(success / len(smiles_list)) if smiles_list else 0.0,
        "elapsed_seconds": float(elapsed),
        "api_url": API_URL,
        "files": {"annotations": os.path.basename(output_path)},
    }
    with open(os.path.join(os.path.dirname(output_path), "npclassifier_metadata.json"),
              "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    update_manifest(
        tag,
        "annotate.npclassifier",
        [output_path, os.path.join(os.path.dirname(output_path), "npclassifier_metadata.json")],
        metadata,
    )

    # Top pathways summary
    pathway_counts: Dict[str, int] = {}
    for r in results:
        if r:
            pw = (r.get("pathway_results") or ["Unknown"])[0]
            pathway_counts[pw] = pathway_counts.get(pw, 0) + 1

    if pathway_counts:
        print("\nTop pathways:")
        for pw, cnt in sorted(pathway_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"  {pw}: {cnt} ({cnt / success * 100:.1f}%)")

    return True


# ============================================================
# CLI
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m hddflyzer.chem.npclassifier <tag>")
        print("Example: python -m hddflyzer.chem.npclassifier aocd")
        sys.exit(1)

    tag = sanitize_tag(sys.argv[1])
    if not run(tag):
        sys.exit(1)


if __name__ == "__main__":
    main()
