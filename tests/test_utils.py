# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for hddflyzer.utils."""

import numpy as np
import pandas as pd
import pytest

from hddflyzer.utils.descriptors import (
    ALL_BASE,
    CONSTITUTIONAL,
    ELECTRONIC,
    GEOMETRICAL,
    HDDF,
    HYBRID_BASE,
    TOPOLOGICAL,
    categorize_descriptors,
    get_zero_variance_descriptors,
    strength_label,
)
from hddflyzer.utils.columns import (
    find_id_column,
    find_smiles_column,
)
from hddflyzer.utils.naming import sanitize_tag, validate_tag


# ============================================================
# columns.py / naming.py
# ============================================================

class TestFindSmiles:
    def test_canonical_smiles(self):
        df = pd.DataFrame({"canonical_smiles": ["CCO"], "id": [1]})
        assert find_smiles_column(df) == "canonical_smiles"

    def test_smiles_col(self):
        df = pd.DataFrame({"SMILES": ["CCO"], "id": [1]})
        assert find_smiles_column(df) == "SMILES"

    def test_no_smiles_raises(self):
        df = pd.DataFrame({"compound": ["CCO"]})
        with pytest.raises(ValueError):
            find_smiles_column(df)


class TestFindId:
    def test_identifier(self):
        df = pd.DataFrame({"identifier": ["A1"], "smiles": ["CCO"]})
        assert find_id_column(df) == "identifier"

    def test_id(self):
        df = pd.DataFrame({"id": ["A1"], "smiles": ["CCO"]})
        assert find_id_column(df) == "id"

    def test_no_id_raises(self):
        df = pd.DataFrame({"name": ["aspirin"]})
        with pytest.raises(ValueError):
            find_id_column(df)


class TestSanitizeTag:
    def test_slash_replaced(self):
        assert sanitize_tag("ao/cd") == "ao_cd"

    def test_backslash_replaced(self):
        assert sanitize_tag("ao\\cd") == "ao_cd"

    def test_strip_whitespace(self):
        assert sanitize_tag("  aocd  ") == "aocd"


class TestValidateTag:
    def test_valid_tag_returned(self):
        assert validate_tag("aocd_2026") == "aocd_2026"

    def test_empty_tag_rejected(self):
        with pytest.raises(ValueError):
            validate_tag("  ")

    def test_traversal_rejected(self):
        with pytest.raises(ValueError):
            validate_tag("..")

    def test_path_separator_rejected(self):
        with pytest.raises(ValueError):
            validate_tag("ao/cd")


# ============================================================
# descriptors.py
# ============================================================

class TestCategorize:
    def test_all_base_present(self):
        cats = categorize_descriptors(ALL_BASE)
        assert len(cats["all_base"]) == len(ALL_BASE)

    def test_hddf_excluded_when_not_in_cols(self):
        cats = categorize_descriptors(["MW", "TPSA"])
        assert cats["hddf"] == []

    def test_hddf_included_when_in_cols(self):
        cats = categorize_descriptors(["QED", "MW"])
        assert "QED" in cats["hddf"]

    def test_zero_vars_excluded(self):
        cats = categorize_descriptors(["MW", "TPSA"], zero_vars=["TPSA"])
        assert "TPSA" not in cats["constitutional"]

    def test_all_included_combines_base_and_hddf(self):
        cats = categorize_descriptors(ALL_BASE + HDDF)
        assert set(cats["all_included"]) == set(ALL_BASE + HDDF)


class TestZeroVariance:
    def test_constant_column_detected(self):
        df = pd.DataFrame({"MW": [100.0] * 10, "TPSA": [50.0, 60.0] * 5})
        zero = get_zero_variance_descriptors(df)
        assert "MW" in zero
        assert "TPSA" not in zero

    def test_empty_column_detected(self):
        df = pd.DataFrame({"A": [np.nan] * 5})
        zero = get_zero_variance_descriptors(df)
        assert "A" in zero

    def test_normal_column_not_flagged(self):
        df = pd.DataFrame({"MW": np.linspace(100, 500, 20)})
        zero = get_zero_variance_descriptors(df)
        assert "MW" not in zero


class TestStrengthLabel:
    def test_fuerte(self):
        assert strength_label(0.8) == "FUERTE"
        assert strength_label(0.7) == "FUERTE"

    def test_moderada(self):
        assert strength_label(0.6) == "MODERADA"
        assert strength_label(0.5) == "MODERADA"

    def test_debil(self):
        assert strength_label(0.4) == "DEBIL"
        assert strength_label(0.3) == "DEBIL"

    def test_muy_debil(self):
        assert strength_label(0.2) == "MUY_DEBIL"
        assert strength_label(0.0) == "MUY_DEBIL"


class TestDescriptorLists:
    def test_constitutional_count(self):
        assert len(CONSTITUTIONAL) == 17

    def test_topological_count(self):
        assert len(TOPOLOGICAL) == 25

    def test_electronic_count(self):
        assert len(ELECTRONIC) == 10

    def test_geometrical_count(self):
        assert len(GEOMETRICAL) == 5

    def test_hybrid_count(self):
        assert len(HYBRID_BASE) == 6

    def test_hddf_count(self):
        assert len(HDDF) == 5

    def test_all_base_count(self):
        expected = 17 + 25 + 10 + 5 + 6
        assert len(ALL_BASE) == expected
