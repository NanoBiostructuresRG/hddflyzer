# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for hddflyzer.dimred.pca_joint."""

import pandas as pd

from hddflyzer.dimred.pca_joint import run_joint_pca


class TestJointPCA:
    def test_joint_pca_returns_split_and_combined_coordinates(self):
        df_a = pd.DataFrame({
            "MW": [100.0, 110.0, 120.0],
            "TPSA": [20.0, 22.0, 24.0],
            "QED": [0.2, 0.3, 0.4],
        })
        df_b = pd.DataFrame({
            "MW": [200.0, 210.0, 220.0],
            "TPSA": [40.0, 42.0, 44.0],
            "QED": [0.5, 0.6, 0.7],
        })

        joint, coords_a, coords_b, meta = run_joint_pca(
            df_a, df_b, "aocd", "other", ["MW", "TPSA", "QED"])

        assert len(joint) == 6
        assert len(coords_a) == 3
        assert len(coords_b) == 3
        assert {"PC1", "PC2"}.issubset(joint.columns)
        assert meta["tags"] == {"A": "aocd", "B": "other"}
