# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for hddflyzer.chem.tanimoto_sampling."""

import numpy as np

from hddflyzer.chem.tanimoto_sampling import (
    balanced_sample,
    parse_overrides,
    sample_from_matrix,
    stratify,
)


class TestParseOverrides:
    def test_float_parsed(self):
        result = parse_overrides(["unique_thr=0.25"])
        assert result["unique_thr"] == 0.25

    def test_int_parsed(self):
        result = parse_overrides(["k_per_stratum=200"])
        assert result["k_per_stratum"] == 200

    def test_invalid_token_ignored(self):
        result = parse_overrides(["notakeyvalue"])
        assert result == {}


class TestStratify:
    def setup_method(self):
        self.max_sim = np.array([0.10, 0.50, 0.90, 0.20, 0.70, 0.05])

    def test_unique_stratum(self):
        u, _, _ = stratify(self.max_sim, unique_thr=0.30, analogue_thr=0.85)
        assert set(u.tolist()) == {0, 3, 5}

    def test_analogue_stratum(self):
        _, _, a = stratify(self.max_sim, unique_thr=0.30, analogue_thr=0.85)
        assert set(a.tolist()) == {2}

    def test_intermediate_stratum(self):
        _, m, _ = stratify(self.max_sim, unique_thr=0.30, analogue_thr=0.85)
        assert set(m.tolist()) == {1, 4}

    def test_all_indices_covered(self):
        u, m, a = stratify(self.max_sim, unique_thr=0.30, analogue_thr=0.85)
        all_idx = set(u) | set(m) | set(a)
        assert all_idx == set(range(len(self.max_sim)))


class TestBalancedSample:
    def test_k_less_than_population(self):
        rng = np.random.default_rng(42)
        idx = np.arange(100)
        s   = balanced_sample(idx, k=20, rng=rng)
        assert len(s) == 20

    def test_k_greater_returns_all(self):
        rng = np.random.default_rng(42)
        idx = np.arange(10)
        s   = balanced_sample(idx, k=50, rng=rng)
        assert len(s) == 10

    def test_empty_returns_empty(self):
        rng = np.random.default_rng(42)
        s   = balanced_sample(np.array([]), k=10, rng=rng)
        assert len(s) == 0

    def test_no_duplicates(self):
        rng = np.random.default_rng(42)
        idx = np.arange(50)
        s   = balanced_sample(idx, k=30, rng=rng)
        assert len(s) == len(set(s.tolist()))


class TestSampleFromMatrix:
    def test_deterministic_output_with_same_seed(self):
        mat = np.full((20, 20), 0.5, dtype=np.float32)
        np.fill_diagonal(mat, 1.0)
        ids = [f"C{i}" for i in range(20)]

        first, first_meta = sample_from_matrix(
            mat, ids, unique_thr=0.30, analogue_thr=0.85,
            k_per_stratum=5, seed=42)
        second, second_meta = sample_from_matrix(
            mat, ids, unique_thr=0.30, analogue_thr=0.85,
            k_per_stratum=5, seed=42)

        assert first.equals(second)
        assert first_meta == second_meta

    def test_small_strata_are_kept_fully(self):
        mat = np.array([
            [1.0, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 1.0, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.5, 0.1, 0.1],
            [0.1, 0.1, 0.5, 1.0, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 1.0, 0.9],
            [0.1, 0.1, 0.1, 0.1, 0.9, 1.0],
        ], dtype=np.float32)
        ids = [f"C{i}" for i in range(6)]

        sampled, meta = sample_from_matrix(
            mat, ids, unique_thr=0.30, analogue_thr=0.85,
            k_per_stratum=10, seed=42)

        assert sampled["id"].tolist() == ids
        assert meta["sample_size"] == 6
        assert meta["strata_totals"] == {
            "unique": 2,
            "intermediate": 2,
            "analogue": 2,
        }

    def test_id_matrix_mismatch_truncates_as_before(self):
        mat = np.full((4, 4), 0.5, dtype=np.float32)
        np.fill_diagonal(mat, 1.0)
        ids = ["A", "B", "C"]

        sampled, meta = sample_from_matrix(
            mat, ids, unique_thr=0.30, analogue_thr=0.85,
            k_per_stratum=10, seed=42)

        assert sampled["id"].tolist() == ids
        assert sampled["index"].max() < len(ids)
        assert meta["matrix_size"] == 3
