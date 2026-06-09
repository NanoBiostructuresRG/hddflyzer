# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for hddflyzer.chem.feature_curation."""

import pandas as pd

from hddflyzer.chem.feature_curation import curate_features


class TestFeatureCuration:
    def test_curate_features_removes_redundant_and_zero_variance(self):
        df = pd.DataFrame({
            "identifier": ["A1", "A2"],
            "SMILES": ["CCO", "CCCN"],
            "MW": [46.07, 45.08],
            "HeavyAtomMolWt": [40.0, 39.0],
            "NHOHCount": [1, 1],
            "NOCount": [1, 1],
            "NumRadicalElectrons": [0, 0],
            "MolLogP_MW_Ratio": [0.01, 0.02],
            "Chi4": [0.1, 0.2],
            "Chi4v": [0.2, 0.3],
            "QED": [0.4, 0.5],
        })

        curated, meta = curate_features(df)

        for col in [
            "HeavyAtomMolWt", "NHOHCount", "NOCount",
            "NumRadicalElectrons", "MolLogP_MW_Ratio", "Chi4", "Chi4v",
        ]:
            assert col not in curated.columns

        assert {"identifier", "SMILES", "MW", "QED"}.issubset(curated.columns)
        assert {"NumCarbonAtoms", "NumNitrogenAtoms", "NumOxygenAtoms"}.issubset(
            curated.columns)
        assert "HeavyAtomMolWt" in meta["removed_features"]["redundant"]
        assert "NumRadicalElectrons" in meta["removed_features"]["zero_variance"]
