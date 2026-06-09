# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Shared descriptor utilities for HDDFlyzer.

Centralizes descriptor categorization and variance detection,
which were duplicated across get_pca_analysis.py,
get_tsne_withfeatures.py, get_umap_withfeatures.py,
and get_umap_pruning.py.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


# ============================================================
# DESCRIPTOR DEFINITIONS
# ============================================================

#: Canonical list of constitutional descriptors (17)
CONSTITUTIONAL: List[str] = [
    "MW", "MolLogP", "NumHDonors", "NumRotatableBonds", "FractionCSP3",
    "RingCount", "HeavyAtomCount", "HeavyAtomMolWt", "NHOHCount", "NOCount",
    "NumHAcceptors", "NumHeteroatoms", "NumValenceElectrons", "HallKierAlpha",
    "MolMR", "TPSA", "LabuteASA",
]

#: Canonical list of topological descriptors (25)
TOPOLOGICAL: List[str] = [
    "BalabanJ", "BertzCT",
    "Chi0", "Chi1", "Chi2", "Chi3", "Chi4",
    "Chi0v", "Chi1v", "Chi2v", "Chi3v", "Chi4v",
    "Kappa1", "Kappa2", "Kappa3", "Ipc",
    "EccentricConnectivityIndex", "Zagreb1", "Zagreb2", "Platt",
    "NumRadicalElectrons", "NumAliphaticRings", "NumAromaticRings",
    "NumSaturatedRings", "NumHeterocycles",
]

#: Canonical list of electronic descriptors (10)
ELECTRONIC: List[str] = [
    "MaxPartialCharge", "MinPartialCharge",
    "MaxAbsPartialCharge", "MinAbsPartialCharge",
    "PEOE_VSA1", "PEOE_VSA2",
    "SMR_VSA1", "SMR_VSA2",
    "SlogP_VSA1", "SlogP_VSA2",
]

#: Canonical list of geometrical descriptors (5)
GEOMETRICAL: List[str] = [
    "PMI1", "PMI2", "PMI3", "NPR1", "NPR2",
]

#: Canonical list of hybrid base descriptors (6)
HYBRID_BASE: List[str] = [
    "MolLogP_MW_Ratio", "HDonor_Acceptor_Ratio", "RotatableBonds_Fraction",
    "PolarSurfaceArea_Fraction", "PolarAtom_Fraction", "MolDensity_Index",
]

#: Canonical list of HDDF descriptors (5)
HDDF: List[str] = [
    "QED", "LeadLikeness_Score", "Pharma_Complexity",
    "Synthetic_Accessibility", "Desirability_Profile",
]

#: All base descriptors (constitutional + topological + electronic +
#: geometrical + hybrid) — excludes HDDF
ALL_BASE: List[str] = (
    CONSTITUTIONAL + TOPOLOGICAL + ELECTRONIC + GEOMETRICAL + HYBRID_BASE
)

#: Fingerprint similarity descriptors (7)
FINGERPRINTS: List[str] = [
    "morgan_tanimoto", "featmorgan_tanimoto", "atompair_tanimoto",
    "rdk_tanimoto", "torsion_tanimoto", "layered_tanimoto", "maccs_tanimoto",
]

#: MCS descriptors (3)
MCS: List[str] = [
    "mcs_size", "mcs_tanimoto", "mcs_overlap",
]


# ============================================================
# CATEGORIZATION
# ============================================================

def categorize_descriptors(
    df_columns: List[str],
    zero_vars: List[str] = None,
) -> Dict[str, List[str]]:
    """
    Categorize descriptors present in a DataFrame's columns.

    Filters each canonical category to only include columns that:
      - Are present in df_columns
      - Are NOT in zero_vars (if provided)

    Parameters
    ----------
    df_columns : list of str
        Column names available in the dataset.
    zero_vars : list of str, optional
        Descriptors to exclude (e.g. zero-variance ones).

    Returns
    -------
    dict with keys:
        'constitutional', 'topological', 'electronic', 'geometrical',
        'hybrid_base', 'hddf', 'all_base', 'all_included',
        'excluded_zero_variance'
    """
    z = set(zero_vars or [])
    cols = set(df_columns)

    def _filter(lst: List[str]) -> List[str]:
        return [c for c in lst if c in cols and c not in z]

    constitutional = _filter(CONSTITUTIONAL)
    topological    = _filter(TOPOLOGICAL)
    electronic     = _filter(ELECTRONIC)
    geometrical    = _filter(GEOMETRICAL)
    hybrid_base    = _filter(HYBRID_BASE)
    hddf           = _filter(HDDF)

    all_base     = constitutional + topological + electronic + geometrical + hybrid_base
    all_included = all_base + hddf

    return {
        "constitutional":        constitutional,
        "topological":           topological,
        "electronic":            electronic,
        "geometrical":           geometrical,
        "hybrid_base":           hybrid_base,
        "hddf":                  hddf,
        "all_base":              all_base,
        "all_included":          all_included,
        "excluded_zero_variance": list(z),
    }


# ============================================================
# VARIANCE DETECTION
# ============================================================

def get_zero_variance_descriptors(
    df: pd.DataFrame,
    eps: float = 1e-10,
    exclude_columns: List[str] = None,
) -> List[str]:
    """
    Detect descriptors with zero or near-zero variance.

    A descriptor is considered constant if:
      - Its variance is <= eps, AND
      - It has at most 1 unique non-NaN value.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing only the columns to evaluate.
    eps : float
        Variance threshold. Default: 1e-10.
    exclude_columns : list of str, optional
        Columns to skip regardless of variance.

    Returns
    -------
    list of str
        Names of zero-variance descriptors.
    """
    excluded = set(exclude_columns or [])
    zero_variance = []

    for col in df.columns:
        if col in excluded:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        col_data = (
            df[col]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if len(col_data) == 0:
            zero_variance.append(col)
            continue

        if col_data.var() <= eps and col_data.nunique() <= 1:
            zero_variance.append(col)

    if zero_variance:
        print(f"[INFO] Zero-variance descriptors detected ({len(zero_variance)}):")
        for desc in sorted(zero_variance):
            val = df[desc].dropna().iloc[0] if df[desc].dropna().shape[0] > 0 else "NaN"
            print(f"   - {desc} (constant value: {val})")

    return zero_variance


# ============================================================
# CORRELATION STRENGTH
# ============================================================

def strength_label(r_abs: float) -> str:
    """
    Classify correlation strength from absolute correlation value.

    Parameters
    ----------
    r_abs : float
        Absolute value of Pearson or Spearman r.

    Returns
    -------
    str
        One of: 'FUERTE', 'MODERADA', 'DEBIL', 'MUY_DEBIL'
    """
    if r_abs >= 0.7:
        return "FUERTE"
    elif r_abs >= 0.5:
        return "MODERADA"
    elif r_abs >= 0.3:
        return "DEBIL"
    else:
        return "MUY_DEBIL"
