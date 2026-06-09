# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for hddflyzer.chem.stats."""

import numpy as np
import pandas as pd

from hddflyzer.chem.stats import (
    HDDF_COLUMNS,
    HDDF_NAMES,
    NONCORR_THRESHOLD,
    calculate_correlations,
)


class TestHDDFConstants:
    def test_hddf_columns_count(self):
        assert len(HDDF_COLUMNS) == 5

    def test_hddf_names_complete(self):
        for col in HDDF_COLUMNS:
            assert col in HDDF_NAMES, f"Missing name for {col}"

    def test_noncorr_threshold_range(self):
        assert 0 < NONCORR_THRESHOLD < 1


class TestCalculateCorrelations:
    def _make_df(self):
        np.random.seed(42)
        n = 50
        x = np.random.randn(n)
        return pd.DataFrame({
            "QED":                    x,
            "LeadLikeness_Score":     x * 0.9 + np.random.randn(n) * 0.1,
            "Pharma_Complexity":     -x * 0.7 + np.random.randn(n) * 0.2,
            "Synthetic_Accessibility": x * 0.5 + np.random.randn(n) * 0.3,
            "Desirability_Profile":   x * 0.8 + np.random.randn(n) * 0.1,
        })

    def test_returns_correct_shapes(self):
        df = self._make_df()
        pc, sc, stats, meta = calculate_correlations(
            df, HDDF_COLUMNS, HDDF_NAMES)
        assert pc.shape  == (5, 5)
        assert sc.shape  == (5, 5)
        assert len(stats) == 10   # C(5,2) = 10 pairs

    def test_meta_keys_present(self):
        df = self._make_df()
        _, _, _, meta = calculate_correlations(df, HDDF_COLUMNS, HDDF_NAMES)
        for key in (
            "n", "pairs_calculated", "alpha", "alpha_bonferroni",
            "p_adjustment", "significance_rule",
        ):
            assert key in meta

    def test_significance_uses_bonferroni_adjusted_p_values(self):
        df = self._make_df()
        _, _, stats, meta = calculate_correlations(df, HDDF_COLUMNS, HDDF_NAMES)

        assert "pearson_p_adj_bonferroni" in stats.columns
        assert "spearman_p_adj_bonferroni" in stats.columns
        assert (
            stats["signif_pearson"]
            == (stats["pearson_p_adj_bonferroni"] < meta["alpha"])
        ).all()
        assert (
            stats["signif_spearman"]
            == (stats["spearman_p_adj_bonferroni"] < meta["alpha"])
        ).all()

    def test_pearson_diagonal_is_one(self):
        df = self._make_df()
        pc, _, _, _ = calculate_correlations(df, HDDF_COLUMNS, HDDF_NAMES)
        np.testing.assert_allclose(np.diag(pc.values), 1.0, atol=1e-6)
