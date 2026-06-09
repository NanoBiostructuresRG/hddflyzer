# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for hddflyzer.dimred.tsne."""

import numpy as np
import pandas as pd

import hddflyzer.dimred.tsne as tsne_module
from hddflyzer.dimred.tsne import (
    TSNE_ANGLE,
    TSNE_EARLY_EXAGGERATION,
    TSNE_INIT,
    TSNE_LEARNING_RATE,
    TSNE_MAX_ITER,
    TSNE_METHOD,
    TSNE_MIN_GRAD_NORM,
    TSNE_N_ITER_WITHOUT_PROGRESS,
    TSNE_RANDOM_STATE,
    _add_tsne_outlier_columns,
    _library_versions as _tsne_library_versions,
    _safe_perplexity,
    _tsne_interpretation,
    _tsne_parameters,
)


class TestSafePerplexity:
    def test_large_perp_capped(self):
        # n=50, max_safe = min(49, 49//3) = 16
        result = _safe_perplexity(200, n=50)
        assert result <= 16

    def test_small_perp_unchanged(self):
        result = _safe_perplexity(10, n=100)
        assert result == 10

    def test_minimum_floor(self):
        # Very small n -> should still return minimum=5
        result = _safe_perplexity(30, n=8, minimum=5)
        assert result == 5

    def test_exact_boundary(self):
        # n=31, max_safe = min(30, 10) = 10
        result = _safe_perplexity(10, n=31)
        assert result == 10

    def test_perp_less_than_max_safe(self):
        result = _safe_perplexity(5, n=1000)
        assert result == 5

    def test_interpretation_documents_reproducibility_and_scale(self):
        meta = _tsne_interpretation([5, 100], [5, 100])

        assert "stochastic" in meta["reproducibility"]
        assert "absolute distances" in meta["coordinate_scale"]
        assert meta["requested_vs_used"][0] == {"requested": 5, "used": 5}
        assert "Lower perplexity" in meta["perplexity"]

    def test_outlier_columns_added_per_perplexity(self):
        coords = {
            "tSNE_1_perp5": np.array([0.0, 1.0, 100.0]),
            "tSNE_2_perp5": np.array([0.0, 1.0, 100.0]),
        }
        summary = _add_tsne_outlier_columns(coords, [5], z_threshold=1.0)

        assert "centroid_x_perp5" in coords
        assert "dist_to_centroid_perp5" in coords
        assert "z_score_perp5" in coords
        assert "outlier_perp5" in coords
        assert summary["5"]["z_threshold"] == 1.0

    def test_reproducibility_metadata_includes_parameters_and_versions(self):
        params = _tsne_parameters(perplexity=30, random_state=42, metric="euclidean")
        versions = _tsne_library_versions()

        assert params["random_state"] == TSNE_RANDOM_STATE
        assert params["max_iter"] == TSNE_MAX_ITER
        assert params["learning_rate"] == TSNE_LEARNING_RATE
        assert params["init"] == TSNE_INIT
        assert params["early_exaggeration"] == TSNE_EARLY_EXAGGERATION
        assert params["angle"] == TSNE_ANGLE
        assert params["method"] == TSNE_METHOD
        assert params["n_iter_without_progress"] == TSNE_N_ITER_WITHOUT_PROGRESS
        assert params["min_grad_norm"] == TSNE_MIN_GRAD_NORM
        assert {"python", "numpy", "pandas", "scikit_learn"}.issubset(versions)


def test_run_tanimoto_returns_false_before_tsne_for_too_few_samples(
    tmp_path,
    monkeypatch,
):
    sim = np.eye(5, dtype=np.float32)
    ids = [f"C{i}" for i in range(5)]
    npc_df = pd.DataFrame({"identifier": ids})

    monkeypatch.setattr(tsne_module, "get_path", lambda *parts: str(tmp_path))
    monkeypatch.setattr(
        tsne_module,
        "load_tanimoto",
        lambda tag: (sim, ids, "matrix.npz", "ids.csv"),
    )
    monkeypatch.setattr(
        tsne_module,
        "load_npclassifier_success",
        lambda tag: (npc_df, "npclassifier.csv"),
    )
    monkeypatch.setattr(
        tsne_module,
        "align_tanimoto_with_npclassifier",
        lambda sim_arg, ids_arg, npc_arg: (sim_arg, ids_arg, npc_arg),
    )

    def fail_tsne(*args, **kwargs):
        raise AssertionError("_run_tsne should not be called")

    monkeypatch.setattr(tsne_module, "_run_tsne", fail_tsne)

    assert tsne_module.run_tanimoto("aocd") is False
