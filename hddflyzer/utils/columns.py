# SPDX-License-Identifier: LGPL-3.0-or-later

"""Column detection helpers for molecular input tables."""

import pandas as pd


def find_smiles_column(df: pd.DataFrame) -> str:
    """
    Auto-detect the SMILES column in a DataFrame.

    Searches for columns containing 'smiles' or 'canonical_smiles'
    case-insensitively and returns the first match.
    """
    keywords = ["smiles", "canonical_smiles"]
    for col in df.columns:
        if any(kw in col.lower() for kw in keywords):
            return col
    raise ValueError(
        f"No SMILES column found. Available columns: {list(df.columns)}"
    )


def find_id_column(df: pd.DataFrame) -> str:
    """
    Auto-detect the compound identifier column in a DataFrame.

    Searches for columns named 'identifier', 'id', 'compound_id',
    or 'molecule_id' case-insensitively.
    """
    keywords = ["identifier", "id", "compound_id", "molecule_id"]
    for col in df.columns:
        if col.lower() in keywords:
            return col
    raise ValueError(
        f"No identifier column found. Available columns: {list(df.columns)}"
    )
