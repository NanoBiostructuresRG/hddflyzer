# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for hddflyzer.dimred.pca."""

import numpy as np

from hddflyzer.dimred.pca import align_pc_signs, pc_sign_alignment_diagnostics


class TestAlignPCSigns:
    def test_negative_pc1_flipped(self):
        base = np.array([[1.0, 0.5], [-1.0, -0.5], [0.5, 0.25]])
        hddf = base * np.array([-1.0, 1.0])   # PC1 negated
        aligned = align_pc_signs(base, hddf)
        # After alignment, PC1 of hddf should correlate positively with base
        corr = np.corrcoef(base[:, 0], aligned[:, 0])[0, 1]
        assert corr > 0

    def test_positive_pc_unchanged(self):
        base = np.array([[1.0, 2.0], [-1.0, -2.0], [0.5, 1.0]])
        hddf = base.copy()
        aligned = align_pc_signs(base, hddf)
        np.testing.assert_array_equal(aligned, hddf)

    def test_shape_preserved(self):
        base = np.random.randn(50, 2)
        hddf = np.random.randn(50, 2)
        aligned = align_pc_signs(base, hddf)
        assert aligned.shape == hddf.shape

    def test_alignment_diagnostics_document_no_rotation_or_rescaling(self):
        base = np.array([[1.0, 0.5], [-1.0, -0.5], [0.5, 0.25]])
        hddf = base * np.array([-1.0, 1.0])
        meta = pc_sign_alignment_diagnostics(base, hddf)

        assert meta["method"] == "component-wise sign flip"
        assert meta["applies_rotation"] is False
        assert meta["applies_rescaling"] is False
        assert meta["components"][0]["sign_multiplier"] == -1.0
