# SPDX-License-Identifier: LGPL-3.0-or-later

"""Shared helpers for PCA workflows."""

import re
from typing import List

import pandas as pd

from hddflyzer.io import load_features_table


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: strip, lower, collapse spaces."""
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", c).strip().lower() for c in df.columns]
    return df


def normalize_column_list(columns: List[str]) -> List[str]:
    """Normalize a list of column names in the same way as normalize_columns."""
    return [re.sub(r"\s+", " ", c).strip().lower() for c in columns]


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with all columns coerced to numeric values."""
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def intersect_columns(a: List[str], b: List[str]) -> List[str]:
    """Return sorted intersection between two column-name lists."""
    return sorted(set(a) & set(b))


def load_pca_features(tag: str) -> pd.DataFrame:
    """Load the canonical feature table used by PCA workflows."""
    df, path = load_features_table(tag)
    print(f"[INFO] Loading: {path}")
    return df
