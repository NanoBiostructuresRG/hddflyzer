# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Molecular feature engineering for HDDFlyzer.

Calculates intrinsic molecular descriptors per compound. Reference-dependent
Tanimoto and MCS features are exposed through a separate operation because
they describe molecule-reference pairs, not molecules alone.

Categories
----------
Constitutional  : 17 descriptors (1D)
Topological     : 25 descriptors (2D)
Electronic      : 10 descriptors (3D)
Geometrical     :  5 descriptors (3D)
Hybrid          :  6 descriptors (derived ratios)
HDDF            :  5 descriptors (QED, LeadLikeness, Pharma_Complexity,
                                  Synthetic_Accessibility, Desirability_Profile)
Usage
-----
    python -m hddflyzer.chem.feature_engineering <tag>
    python -m hddflyzer.chem.feature_engineering aocd
    python -m hddflyzer.chem.feature_engineering reference aocd --reference rosiglitazone
"""

import os
import sys
import warnings
import json
import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem, DataStructs, Descriptors
from rdkit.Chem import MACCSkeys, QED as rdQED
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdFingerprintGenerator
from typing import Dict, Optional, Tuple

from hddflyzer.config import get_features_path, get_path
from hddflyzer.data.registry import ensure_registry, load_registry
from hddflyzer.utils.columns import find_smiles_column, find_id_column
from hddflyzer.utils.naming import sanitize_tag
from hddflyzer.io import update_manifest
from hddflyzer.chem.reference_catalog import (
    REFERENCE_MOLECULES,
    USER_REFERENCE_TOKEN,
    list_reference_molecules,
    prompt_reference_name,
    prompt_reference_smiles,
    resolve_reference_name,
)

# Suppress RDKit noise
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


FINGERPRINT_PARAMETERS = {
    "morgan": {"radius": 2, "fpSize": 2048},
    "featmorgan": {"radius": 2, "fpSize": 2048},
    "atompair": {"fpSize": 2048},
    "rdk": {"implementation": "Chem.RDKFingerprint defaults"},
    "torsion": {"fpSize": 2048},
    "layered": {"implementation": "Chem.LayeredFingerprint defaults"},
    "maccs": {"implementation": "MACCSkeys.GenMACCSKeys defaults"},
}

REFERENCE_FLOAT_DECIMALS = 6


# ============================================================
# MOLECULAR DESCRIPTORS
# ============================================================

def calculate_molecular_descriptors(smiles: str) -> Optional[Dict]:
    """
    Calculate 78 molecular descriptors from a SMILES string.

    Returns None if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    d = {}

    try:
        # --- Constitutional (17) ---
        d["MW"]                  = Descriptors.MolWt(mol)
        d["MolLogP"]             = Descriptors.MolLogP(mol)
        d["NumHDonors"]          = Descriptors.NumHDonors(mol)
        d["NumRotatableBonds"]   = Descriptors.NumRotatableBonds(mol)
        d["FractionCSP3"]        = Descriptors.FractionCSP3(mol)
        d["RingCount"]           = Descriptors.RingCount(mol)
        d["HeavyAtomCount"]      = Descriptors.HeavyAtomCount(mol)
        d["HeavyAtomMolWt"]      = Descriptors.HeavyAtomMolWt(mol)
        d["NHOHCount"]           = Descriptors.NHOHCount(mol)
        d["NOCount"]             = Descriptors.NOCount(mol)
        d["NumHAcceptors"]       = Descriptors.NumHAcceptors(mol)
        d["NumHeteroatoms"]      = Descriptors.NumHeteroatoms(mol)
        d["NumValenceElectrons"] = Descriptors.NumValenceElectrons(mol)
        d["HallKierAlpha"]       = Descriptors.HallKierAlpha(mol)
        d["MolMR"]               = Descriptors.MolMR(mol)
        d["TPSA"]                = Descriptors.TPSA(mol)
        d["LabuteASA"]           = Descriptors.LabuteASA(mol)

        # --- Topological (25) ---
        d["BalabanJ"]  = Descriptors.BalabanJ(mol)
        d["BertzCT"]   = Descriptors.BertzCT(mol)
        d["Chi0"]  = Descriptors.Chi0(mol)
        d["Chi1"]  = Descriptors.Chi1(mol)
        d["Chi2"]  = Descriptors.Chi2n(mol)
        d["Chi3"]  = Descriptors.Chi3n(mol)
        d["Chi4"]  = Descriptors.Chi4n(mol)
        d["Chi0v"] = Descriptors.Chi0v(mol)
        d["Chi1v"] = Descriptors.Chi1v(mol)
        d["Chi2v"] = Descriptors.Chi2v(mol)
        d["Chi3v"] = Descriptors.Chi3v(mol)
        d["Chi4v"] = Descriptors.Chi4v(mol)
        d["Kappa1"] = Descriptors.Kappa1(mol)
        d["Kappa2"] = Descriptors.Kappa2(mol)
        d["Kappa3"] = Descriptors.Kappa3(mol)
        d["Ipc"]    = Descriptors.Ipc(mol)

        try:
            d["EccentricConnectivityIndex"] = Chem.GraphDescriptors.CalcEccentricConnectivityIndex(mol)
        except Exception:
            d["EccentricConnectivityIndex"] = 0.0
        try:
            d["Zagreb1"] = Chem.GraphDescriptors.CalcZagrebIndex1(mol)
            d["Zagreb2"] = Chem.GraphDescriptors.CalcZagrebIndex2(mol)
        except Exception:
            d["Zagreb1"] = d["Zagreb2"] = 0.0
        try:
            d["Platt"] = Chem.GraphDescriptors.CalcPlatt(mol)
        except Exception:
            d["Platt"] = 0.0
        try:
            d["NumRadicalElectrons"] = Descriptors.NumRadicalElectrons(mol)
            d["NumAliphaticRings"]   = Descriptors.NumAliphaticRings(mol)
            d["NumAromaticRings"]    = Descriptors.NumAromaticRings(mol)
            d["NumSaturatedRings"]   = Descriptors.NumSaturatedRings(mol)
            d["NumHeterocycles"]     = Descriptors.NumHeterocycles(mol)
        except Exception:
            pass

        # --- Electronic (10) ---
        d["MaxPartialCharge"]    = Descriptors.MaxPartialCharge(mol)
        d["MinPartialCharge"]    = Descriptors.MinPartialCharge(mol)
        d["MaxAbsPartialCharge"] = Descriptors.MaxAbsPartialCharge(mol)
        d["MinAbsPartialCharge"] = Descriptors.MinAbsPartialCharge(mol)
        try:
            d["PEOE_VSA1"]  = Descriptors.PEOE_VSA1(mol)
            d["PEOE_VSA2"]  = Descriptors.PEOE_VSA2(mol)
            d["SMR_VSA1"]   = Descriptors.SMR_VSA1(mol)
            d["SMR_VSA2"]   = Descriptors.SMR_VSA2(mol)
            d["SlogP_VSA1"] = Descriptors.SlogP_VSA1(mol)
            d["SlogP_VSA2"] = Descriptors.SlogP_VSA2(mol)
        except Exception:
            pass

        # --- Geometrical (5) ---
        try:
            d["PMI1"] = Descriptors.PMI1(mol)
            d["PMI2"] = Descriptors.PMI2(mol)
            d["PMI3"] = Descriptors.PMI3(mol)
            d["NPR1"] = Descriptors.NPR1(mol)
            d["NPR2"] = Descriptors.NPR2(mol)
        except Exception:
            d["PMI1"] = d["PMI2"] = d["PMI3"] = 0.0
            d["NPR1"] = d["NPR2"] = 0.0

        # --- Hybrid (6) ---
        d["MolLogP_MW_Ratio"]        = d["MolLogP"] / d["MW"] if d["MW"] > 0 else 0.0
        d["HDonor_Acceptor_Ratio"]   = d["NumHDonors"] / d["NumHAcceptors"] if d["NumHAcceptors"] > 0 else 0.0
        d["RotatableBonds_Fraction"] = d["NumRotatableBonds"] / d["HeavyAtomCount"] if d["HeavyAtomCount"] > 0 else 0.0
        d["PolarSurfaceArea_Fraction"] = d["TPSA"] / d["MW"] if d["MW"] > 0 else 0.0
        d["PolarAtom_Fraction"]      = d["NumHeteroatoms"] / d["HeavyAtomCount"] if d["HeavyAtomCount"] > 0 else 0.0
        d["MolDensity_Index"]        = d["MW"] / d["LabuteASA"] if d["LabuteASA"] > 0 else 0.0

        # --- HDDF (5) ---
        d["QED"] = rdQED.qed(mol)

        d["LeadLikeness_Score"] = _calculate_lead_score(d)
        d["Pharma_Complexity"]  = _calculate_pharma_complexity(d)
        d["Synthetic_Accessibility"] = _calculate_synthetic_accessibility(d)
        d["Desirability_Profile"] = _calculate_desirability_profile(d)

    except Exception as e:
        print(f"[WARN] Descriptor error for {smiles}: {e}")
        return None

    return d


# ============================================================
# HDDF SCORING FUNCTIONS
# ============================================================

def _calculate_lead_score(d: Dict) -> float:
    mw_score   = 1.0 if 250 <= d["MW"] <= 350 else (0.7 if 200 <= d["MW"] <= 400 else 0.3)
    logp_score = 1.0 if 1.0 <= d["MolLogP"] <= 3.0 else (0.7 if 0.5 <= d["MolLogP"] <= 3.5 else 0.3)
    hbd_score  = 1.0 if d["NumHDonors"] <= 3 else (0.7 if d["NumHDonors"] <= 4 else 0.3)
    rotb_score = 1.0 if d["NumRotatableBonds"] <= 5 else (0.7 if d["NumRotatableBonds"] <= 7 else 0.3)
    complexity = (
        d["FractionCSP3"] * 0.4
        + (min(d["RingCount"], 4) / 4.0) * 0.3
        + (1.0 - min(d["NumRotatableBonds"] / 10.0, 1.0)) * 0.3
    )
    return mw_score * 0.25 + logp_score * 0.25 + hbd_score * 0.20 + rotb_score * 0.15 + complexity * 0.15


def _calculate_pharma_complexity(d: Dict) -> float:
    return (
        d["NumHDonors"] * 0.3
        + d["FractionCSP3"] * 0.2
        + d["NumHAcceptors"] * 0.3
        + d.get("NumAromaticRings", 0) * 0.2
    )


def _calculate_synthetic_accessibility(d: Dict) -> float:
    return (
        (1.0 - min(d["RingCount"] / 6.0, 1.0)) * 0.25
        + (1.0 - min(d.get("NumChiralCenters", 0) / 4.0, 1.0)) * 0.30
        + (1.0 - min(d["NumRotatableBonds"] / 12.0, 1.0)) * 0.20
        + d["FractionCSP3"] * 0.15
        + (1.0 - min((d["HeavyAtomCount"] * 0.5) / 8.0, 1.0)) * 0.10
    )


def _calculate_desirability_profile(d: Dict) -> float:
    mw_score   = 1.0 - min(abs(d["MW"] - 350) / 300, 1.0)
    logp_score = 1.0 - min(abs(d["MolLogP"] - 2.5) / 4.0, 1.0)
    tpsa_score = 1.0 - min(abs(d["TPSA"] - 80) / 120, 1.0)
    hbd_score  = 1.0 - min(d["NumHDonors"] / 8.0, 1.0)
    hba_score  = 1.0 - min(d["NumHAcceptors"] / 12.0, 1.0)
    rotb_score = 1.0 - min(d["NumRotatableBonds"] / 10.0, 1.0)
    score = (
        mw_score   * 0.20
        + logp_score * 0.25
        + tpsa_score * 0.20
        + hbd_score  * 0.15
        + hba_score  * 0.10
        + rotb_score * 0.10
    )
    return max(0.0, min(1.0, score))


# ============================================================
# FINGERPRINT SIMILARITIES
# ============================================================

def calculate_fingerprint_similarities(smiles1: str, smiles2: str) -> Dict[str, float]:
    """
    Calculate 7 Tanimoto fingerprint similarities between two molecules.
    Returns zeros for any type that fails.
    """
    _zero = {f"{fp}_tanimoto": 0.0
             for fp in ["morgan", "featmorgan", "atompair",
                        "rdk", "torsion", "layered", "maccs"]}

    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return _zero

    sims = dict(_zero)

    try:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        sims["morgan_tanimoto"] = DataStructs.TanimotoSimilarity(
            gen.GetFingerprint(mol1), gen.GetFingerprint(mol2))
    except Exception as e:
        print(f"[WARN] Morgan fingerprint error: {e}")

    try:
        feat_inv = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
        gen_feat = rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=2048,
            atomInvariantsGenerator=feat_inv,
        )
    except AttributeError:
        gen_feat = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, useFeatures=True)
    try:
        sims["featmorgan_tanimoto"] = DataStructs.TanimotoSimilarity(
            gen_feat.GetFingerprint(mol1), gen_feat.GetFingerprint(mol2))
    except Exception as e:
        print(f"[WARN] Feature Morgan fingerprint error: {e}")

    try:
        gen_ap = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)
        sims["atompair_tanimoto"] = DataStructs.TanimotoSimilarity(
            gen_ap.GetFingerprint(mol1), gen_ap.GetFingerprint(mol2))
    except Exception:
        try:
            sims["atompair_tanimoto"] = DataStructs.TanimotoSimilarity(
                    Pairs.GetAtomPairFingerprintAsBitVect(mol1),
                    Pairs.GetAtomPairFingerprintAsBitVect(mol2))
        except Exception as e:
            print(f"[WARN] AtomPair fingerprint error: {e}")

    try:
        sims["rdk_tanimoto"] = DataStructs.TanimotoSimilarity(
            Chem.RDKFingerprint(mol1), Chem.RDKFingerprint(mol2))
    except Exception as e:
        print(f"[WARN] RDKFingerprint error: {e}")

    try:
        gen_tt = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048)
        sims["torsion_tanimoto"] = DataStructs.TanimotoSimilarity(
            gen_tt.GetFingerprint(mol1), gen_tt.GetFingerprint(mol2))
    except Exception:
        try:
            sims["torsion_tanimoto"] = DataStructs.TanimotoSimilarity(
                    Torsions.GetTopologicalTorsionFingerprintAsBitVect(mol1),
                    Torsions.GetTopologicalTorsionFingerprintAsBitVect(mol2))
        except Exception as e:
            print(f"[WARN] Torsion fingerprint error: {e}")

    try:
        sims["layered_tanimoto"] = DataStructs.TanimotoSimilarity(
            Chem.LayeredFingerprint(mol1), Chem.LayeredFingerprint(mol2))
    except Exception as e:
        print(f"[WARN] Layered fingerprint error: {e}")

    try:
        sims["maccs_tanimoto"] = DataStructs.TanimotoSimilarity(
            MACCSkeys.GenMACCSKeys(mol1), MACCSkeys.GenMACCSKeys(mol2))
    except Exception as e:
        print(f"[WARN] MACCS fingerprint error: {e}")

    return sims


# ============================================================
# MCS FEATURES
# ============================================================

def calculate_mcs_features(smiles1: str, smiles2: str) -> Dict:
    """
    Calculate auditable Maximum Common Substructure (MCS) features.

    size_A and size_B are the atom counts used to derive:
    mcs_tanimoto = mcs_size / (size_A + size_B - mcs_size)
    mcs_overlap  = mcs_size / min(size_A, size_B)
    """
    _zero = {
        "size_A": 0,
        "size_B": 0,
        "mcs_size": 0,
        "mcs_tanimoto": 0.0,
        "mcs_overlap": 0.0,
    }

    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return _zero
    n1, n2 = mol1.GetNumAtoms(), mol2.GetNumAtoms()

    try:
        result = rdFMCS.FindMCS([mol1, mol2], timeout=5)
        if not result.smartsString or result.numAtoms == 0:
            return {**_zero, "size_A": n1, "size_B": n2}

        n = result.numAtoms
        return {
            "size_A":       n1,
            "size_B":       n2,
            "mcs_size":     n,
            "mcs_tanimoto": n / (n1 + n2 - n) if (n1 + n2 - n) > 0 else 0.0,
            "mcs_overlap":  n / min(n1, n2) if min(n1, n2) > 0 else 0.0,
        }
    except Exception:
        return {**_zero, "size_A": n1, "size_B": n2}


# ============================================================
# PIPELINE
# ============================================================

def _load_feature_input(tag: str, sampled: bool = False) -> pd.DataFrame:
    """
    Load molecules from the registry.

    When sampled=True, restrict the registry to IDs listed in
    results/{tag}/chemistry/tanimoto/samples/sampled_ids.csv. This is useful for large
    datasets where Tanimoto sampling selects a smaller representative subset.
    If sampling selected every molecule, sampled=True is intentionally
    equivalent to the full registry and does not add analytical value.
    """
    ensure_registry(tag)
    compounds_df = load_registry(tag, valid_only=True).copy()

    if "canonical_smiles" in compounds_df.columns:
        canonical = compounds_df["canonical_smiles"].fillna("").astype(str)
        compounds_df["SMILES"] = canonical.where(
            canonical.str.len() > 0,
            compounds_df["SMILES"],
        )

    if sampled:
        sampled_path = get_path("tanimoto", tag, "samples", "sampled_ids.csv")
        if not os.path.exists(sampled_path):
            raise FileNotFoundError(
                f"Sampled IDs not found: {sampled_path}\n"
                f"Run: hddflyzer chem sampling {tag}"
            )
        sampled_df = pd.read_csv(sampled_path)
        if "id" not in sampled_df.columns:
            raise ValueError("sampled_ids.csv must contain an 'id' column.")
        wanted = set(sampled_df["id"].astype(str))
        compounds_df = compounds_df[
            compounds_df["identifier"].astype(str).isin(wanted)
        ].copy()

    return compounds_df


def _remove_legacy_feature_outputs(features_dir: str) -> None:
    """Remove old reference-mixed feature outputs from earlier layouts."""
    import glob

    for pattern in ("features_*.csv", "features_*.pkl"):
        for path in glob.glob(os.path.join(features_dir, pattern)):
            if os.path.basename(path) not in ("features.csv", "features.pkl"):
                os.remove(path)


def run(tag: str, sampled: bool = False, save_pickle: bool = False) -> bool:
    """
    Run intrinsic molecular feature engineering for a given tag.

    Reads from  : results/{tag}/registry/molecules.csv
                  optionally results/{tag}/chemistry/tanimoto/samples/sampled_ids.csv
    Writes to   : results/{tag}/features/full/features.csv
                  optionally results/{tag}/features/full/features.pkl

    Parameters
    ----------
    tag : str
        Dataset tag.
    sampled : bool
        If True, calculate features only for IDs selected by
        hddflyzer chem sample. Use this to reduce large datasets before
        expensive descriptor calculations. For small datasets where the sample
        contains all compounds, this produces the same molecule set as the
        default full-registry mode.
    save_pickle : bool
        If True, also write a pandas pickle. Disabled by default to reduce
        redundant storage; CSV is the canonical table.

    Returns
    -------
    bool
        True on success.
    """
    features_dir = get_features_path(tag)
    os.makedirs(features_dir, exist_ok=True)

    try:
        compounds_df = _load_feature_input(tag, sampled=sampled)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        return False

    print(f"[INFO] Tag: {tag}")
    print(f"[INFO] Input: registry{' + sampled_ids' if sampled else ''}")

    output_csv = os.path.join(features_dir, "features.csv")
    output_pkl = os.path.join(features_dir, "features.pkl")
    _remove_legacy_feature_outputs(features_dir)

    smiles_col = find_smiles_column(compounds_df)
    id_col = find_id_column(compounds_df)
    print(f"[INFO] {len(compounds_df)} compounds | SMILES: '{smiles_col}' | ID: '{id_col}'")

    # Process
    all_features = []
    print("Progress: ", end="", flush=True)

    for idx, (_, row) in enumerate(compounds_df.iterrows()):
        smiles = row[smiles_col]
        features = {
            "SMILES":     smiles,
            "identifier": row[id_col],
        }
        desc = calculate_molecular_descriptors(smiles)
        if desc:
            features.update(desc)
        all_features.append(features)

        print("Y", end="", flush=True)
        if (idx + 1) % 50 == 0:
            print(f" {idx + 1}/{len(compounds_df)}")
            if idx + 1 < len(compounds_df):
                print("Progress: ", end="", flush=True)

    if len(compounds_df) % 50 != 0:
        print(f" {len(compounds_df)}/{len(compounds_df)}")

    features_df = pd.DataFrame(all_features).fillna(0)
    features_df.to_csv(output_csv, index=False)
    if save_pickle:
        features_df.to_pickle(output_pkl)
    elif os.path.exists(output_pkl):
        os.remove(output_pkl)

    n_feat = len([c for c in features_df.columns if c not in ["SMILES", "identifier"]])
    files = {"csv": os.path.basename(output_csv)}
    if save_pickle:
        files["pickle"] = os.path.basename(output_pkl)
    metadata = {
        "tag": tag,
        "feature_scope": "intrinsic_molecular_descriptors",
        "sampled": bool(sampled),
        "n_compounds": int(len(features_df)),
        "n_features": int(n_feat),
        "files": files,
    }
    metadata_path = os.path.join(features_dir, "features_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    update_manifest(tag, "chem.features", [output_csv, metadata_path, output_pkl if save_pickle else None], metadata)

    print(f"\n[OK] {len(features_df)} compounds, {n_feat} features")
    print(f"     CSV : {output_csv}")
    if save_pickle:
        print(f"     PKL : {output_pkl}")
    return True


def run_reference_features(
    tag: str,
    reference_name: str = "rosiglitazone",
    reference_smiles: Optional[str] = None,
    sampled: bool = False,
) -> bool:
    """
    Calculate molecule-reference similarity features for a given tag.

    Each output row is a dyadic observation: one dataset molecule compared
    against one named reference molecule.
    """
    reference_name = resolve_reference_name(reference_name) or reference_name
    if reference_smiles is None and reference_name not in REFERENCE_MOLECULES:
        print(f"[ERROR] Unknown reference molecule: {reference_name}")
        list_reference_molecules()
        print("Or provide a custom reference with: --reference-smiles SMILES")
        return False
    if reference_smiles is not None and Chem.MolFromSmiles(reference_smiles) is None:
        print(f"[ERROR] Invalid reference SMILES: {reference_smiles}")
        return False

    try:
        compounds_df = _load_feature_input(tag, sampled=sampled)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        return False

    ref_smiles = reference_smiles or REFERENCE_MOLECULES[reference_name]
    out_dir = get_path("reference_features", tag)
    os.makedirs(out_dir, exist_ok=True)

    reference_slug = sanitize_tag(reference_name)
    output_csv = os.path.join(out_dir, f"reference_features_{reference_slug}.csv")
    metadata_path = os.path.join(
        out_dir, f"reference_features_{reference_slug}_metadata.json")

    smiles_col = find_smiles_column(compounds_df)
    id_col = find_id_column(compounds_df)

    print(f"[INFO] Tag: {tag}")
    print(f"[INFO] Reference: {reference_name}")
    print(f"[INFO] Input: registry{' + sampled_ids' if sampled else ''}")

    rows = []
    print("Progress: ", end="", flush=True)
    for idx, (_, row) in enumerate(compounds_df.iterrows()):
        smiles = row[smiles_col]
        features = {
            "molecule_identifier": row[id_col],
            "molecule_smiles": smiles,
            "reference_name": reference_name,
            "reference_smiles": ref_smiles,
        }
        features.update(calculate_fingerprint_similarities(smiles, ref_smiles))
        features.update(calculate_mcs_features(smiles, ref_smiles))
        rows.append(features)

        print("Y", end="", flush=True)
        if (idx + 1) % 50 == 0:
            print(f" {idx + 1}/{len(compounds_df)}")
            if idx + 1 < len(compounds_df):
                print("Progress: ", end="", flush=True)

    if len(compounds_df) % 50 != 0:
        print(f" {len(compounds_df)}/{len(compounds_df)}")

    ref_df = pd.DataFrame(rows).fillna(0).round(REFERENCE_FLOAT_DECIMALS)
    ref_df.to_csv(output_csv, index=False)

    identity_cols = (
        "molecule_identifier", "molecule_smiles",
        "reference_name", "reference_smiles",
    )
    support_cols = ("size_A", "size_B")
    n_feat = len([
        c for c in ref_df.columns
        if c not in identity_cols + support_cols
    ])
    metadata = {
        "tag": tag,
        "feature_scope": "dyadic_reference_similarity",
        "reference": reference_name,
        "reference_source": "custom_smiles" if reference_smiles else "built_in",
        "sampled": bool(sampled),
        "n_pairs": int(len(ref_df)),
        "n_features": int(n_feat),
        "float_decimals": REFERENCE_FLOAT_DECIMALS,
        "support_columns": list(support_cols),
        "chemistry_backend": {
            "rdkit_available": True,
            "rdkit_version": rdBase.rdkitVersion,
        },
        "descriptor_parameters": {
            "fingerprints": FINGERPRINT_PARAMETERS,
            "mcs": {
                "algorithm": "rdFMCS.FindMCS",
                "timeout_seconds": 5,
                "size_A": "number of atoms in the dataset molecule",
                "size_B": "number of atoms in the reference molecule",
                "mcs_size": "number of atoms in the maximum common substructure",
                "mcs_tanimoto_formula": "mcs_size / (size_A + size_B - mcs_size)",
                "mcs_overlap_formula": "mcs_size / min(size_A, size_B)",
            },
        },
        "files": {"csv": os.path.basename(output_csv)},
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    legacy_metadata_path = os.path.join(out_dir, "reference_features_metadata.json")
    if os.path.exists(legacy_metadata_path):
        os.remove(legacy_metadata_path)
    update_manifest(tag, "chem.reference_features", [output_csv, metadata_path], metadata)

    print(f"\n[OK] {len(ref_df)} molecule-reference pairs, {n_feat} features")
    print(f"     CSV : {output_csv}")
    return True


# ============================================================
# CLI
# ============================================================

def main():
    tag = sanitize_tag(sys.argv[1]) if len(sys.argv) > 1 else None
    if tag is None:
        print("Usage: python -m hddflyzer.chem.feature_engineering <tag> [--sampled] [--save-pkl]")
        print("       python -m hddflyzer.chem.feature_engineering reference <tag> [--reference NAME] [--sampled]")
        print("       python -m hddflyzer.chem.feature_engineering reference <tag> --list-references")
        print("Example: python -m hddflyzer.chem.feature_engineering aocd")
        sys.exit(1)

    mode = "features"
    if tag in ("reference", "reference-features", "similarity-features"):
        mode = "reference"
        if len(sys.argv) < 3:
            print("Usage: python -m hddflyzer.chem.feature_engineering reference <tag> [--reference NAME] [--sampled]")
            list_reference_molecules()
            sys.exit(1)
        tag = sanitize_tag(sys.argv[2])
        args = sys.argv[3:]
    else:
        args = sys.argv[2:]

    sampled = "--sampled" in args
    save_pickle = "--save-pkl" in args
    reference_name = None
    reference_smiles = None
    if "--list-references" in args or "list-references" in args:
        list_reference_molecules()
        return
    for i, tok in enumerate(args):
        if tok.startswith("reference="):
            reference_name = tok.split("=", 1)[1]
        elif tok.startswith("reference_smiles="):
            reference_smiles = tok.split("=", 1)[1]
        elif tok == "--reference":
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                reference_name = args[i + 1]
        elif tok == "--reference-smiles" and i + 1 < len(args):
            reference_smiles = args[i + 1]

    if mode == "reference":
        if reference_smiles and reference_name is None:
            reference_name = USER_REFERENCE_TOKEN
        elif reference_name is None:
            reference_name = prompt_reference_name()
            if reference_name is None:
                print("[ERROR] No valid reference molecule selected.")
                return
            if reference_name == USER_REFERENCE_TOKEN:
                reference_smiles = prompt_reference_smiles(reference_name)
                if reference_smiles is None:
                    print("[ERROR] No SMILES provided for user reference.")
                    return
        else:
            resolved = resolve_reference_name(reference_name)
            if resolved is None and reference_smiles is None:
                reference_smiles = prompt_reference_smiles(reference_name)
                if reference_smiles is None:
                    print("[ERROR] No SMILES provided for custom reference.")
                    return
            reference_name = resolved or reference_name
            if reference_name == USER_REFERENCE_TOKEN and reference_smiles is None:
                reference_smiles = prompt_reference_smiles(reference_name)
                if reference_smiles is None:
                    print("[ERROR] No SMILES provided for user reference.")
                    return
        run_reference_features(
            tag,
            reference_name=reference_name,
            reference_smiles=reference_smiles,
            sampled=sampled,
        )
    else:
        run(tag, sampled=sampled, save_pickle=save_pickle)


if __name__ == "__main__":
    main()
