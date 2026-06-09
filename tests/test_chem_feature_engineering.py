# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for hddflyzer.chem.feature_engineering."""

import pytest

from hddflyzer.chem.feature_engineering import (
    calculate_fingerprint_similarities,
    calculate_mcs_features,
    prompt_reference_name,
    prompt_reference_smiles,
    resolve_reference_name,
)


class TestReferenceMCS:
    def test_reference_selection_accepts_number_or_name(self):
        assert resolve_reference_name("1") == "rosiglitazone"
        assert resolve_reference_name("rosiglitazone") == "rosiglitazone"
        assert resolve_reference_name("U") == "user_reference"
        assert resolve_reference_name("999") is None

    def test_reference_prompt_accepts_number(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "1")
        assert prompt_reference_name() == "rosiglitazone"

    def test_custom_reference_prompt_accepts_smiles(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "CCO")
        assert prompt_reference_smiles("my_reference") == "CCO"

    def test_fingerprint_names_are_explicit(self):
        result = calculate_fingerprint_similarities("CCO", "CCN")

        assert "rdk_tanimoto" in result
        assert "rdkit_tanimoto" not in result

    def test_mcs_support_columns_make_formulas_auditable(self):
        result = calculate_mcs_features("CCO", "CCN")

        assert {
            "size_A", "size_B", "mcs_size", "mcs_tanimoto", "mcs_overlap",
        }.issubset(result)

        if result["mcs_size"] > 0:
            expected_tanimoto = result["mcs_size"] / (
                result["size_A"] + result["size_B"] - result["mcs_size"]
            )
            expected_overlap = result["mcs_size"] / min(
                result["size_A"], result["size_B"])

            assert result["mcs_tanimoto"] == pytest.approx(expected_tanimoto)
            assert result["mcs_overlap"] == pytest.approx(expected_overlap)
