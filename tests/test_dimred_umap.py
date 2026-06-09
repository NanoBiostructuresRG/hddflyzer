# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for hddflyzer.dimred.umap."""

import hddflyzer.dimred.umap as umap_module
from hddflyzer.dimred.umap import (
    DEFAULT_MIN_DIST,
    DEFAULT_N_NEIGHBORS,
    UMAP_INIT,
    UMAP_LEARNING_RATE,
    UMAP_N_EPOCHS,
    UMAP_NEGATIVE_SAMPLE_RATE,
    UMAP_RANDOM_STATE,
    _library_versions,
    _umap_interpretation,
    _umap_parameters,
)


class TestUMAPDefaults:
    def test_n_neighbors_list(self):
        assert isinstance(DEFAULT_N_NEIGHBORS, list)
        assert all(isinstance(n, int) for n in DEFAULT_N_NEIGHBORS)
        assert all(n > 0 for n in DEFAULT_N_NEIGHBORS)

    def test_min_dist_list(self):
        assert isinstance(DEFAULT_MIN_DIST, list)
        assert all(isinstance(d, float) for d in DEFAULT_MIN_DIST)
        assert all(0 < d < 1 for d in DEFAULT_MIN_DIST)

    def test_default_values(self):
        assert DEFAULT_N_NEIGHBORS == [15, 30, 50]
        assert DEFAULT_MIN_DIST    == [0.1, 0.5]
        assert UMAP_RANDOM_STATE == 42
        assert UMAP_N_EPOCHS == 500
        assert UMAP_LEARNING_RATE == 1.0
        assert UMAP_NEGATIVE_SAMPLE_RATE == 5
        assert UMAP_INIT == "spectral"

    def test_interpretation_documents_spaces_and_scale(self):
        meta = _umap_interpretation("features", compute_separate=True)

        assert "stochastic" in meta["reproducibility"]
        assert "coordinate magnitudes" in meta["coordinate_scale"]
        assert {"combined", "base", "hddf"}.issubset(meta["spaces"])

    def test_reproducibility_metadata_includes_parameters_and_versions(self):
        params = _umap_parameters(seed=42, metric="precomputed")
        versions = _library_versions()

        for key in ("n_epochs", "learning_rate", "negative_sample_rate", "init"):
            assert key in params
        assert params["metric"] == "precomputed"
        assert {"python", "numpy", "pandas", "scikit_learn", "umap_learn"}.issubset(
            versions)


class TestUMAPUnavailable:
    def test_run_features_returns_false_before_loading_data(self, monkeypatch):
        def fail_load(tag):
            raise AssertionError("data should not be loaded")

        monkeypatch.setattr(umap_module, "_UMAP_AVAILABLE", False)
        monkeypatch.setattr(umap_module, "_load_dataset", fail_load)

        assert umap_module.run_features("aocd") is False

    def test_run_tanimoto_returns_false_before_loading_data(self, monkeypatch):
        def fail_load(tag):
            raise AssertionError("tanimoto should not be loaded")

        monkeypatch.setattr(umap_module, "_UMAP_AVAILABLE", False)
        monkeypatch.setattr(umap_module, "load_tanimoto", fail_load)

        assert umap_module.run_tanimoto("aocd") is False

    def test_run_pruning_returns_false_before_loading_data(self, monkeypatch):
        def fail_load(tag):
            raise AssertionError("data should not be loaded")

        monkeypatch.setattr(umap_module, "_UMAP_AVAILABLE", False)
        monkeypatch.setattr(umap_module, "_load_dataset", fail_load)

        assert umap_module.run_pruning("aocd") is False
