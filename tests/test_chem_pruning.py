# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for hddflyzer.chem.pruning."""

import pandas as pd

from hddflyzer.chem.pruning import (
    build_redundant_pairs,
    build_redundancy_graph,
    choose_representative,
    find_connected_components,
    prune_from_correlation_stats,
)


class TestBuildRedundantPairs:
    def _make_stats(self):
        return pd.DataFrame({
            "var_a":      ["A", "A", "B", "C"],
            "var_b":      ["B", "C", "C", "D"],
            "pearson_r":  [0.90, 0.20, 0.85, 0.30],
            "spearman_r": [0.88, 0.15, 0.82, 0.25],
        })

    def test_correct_count(self):
        pairs = build_redundant_pairs(self._make_stats(), threshold=0.80)
        assert len(pairs) == 2

    def test_threshold_boundary(self):
        pairs = build_redundant_pairs(self._make_stats(), threshold=0.90)
        assert len(pairs) == 1

    def test_sorted_descending(self):
        pairs = build_redundant_pairs(self._make_stats(), threshold=0.80)
        corrs = pairs["max_abs_corr"].tolist()
        assert corrs == sorted(corrs, reverse=True)


class TestRedundancyGraph:
    def _make_pairs(self):
        return pd.DataFrame({
            "var_a": ["A", "B"], "var_b": ["B", "C"],
            "pearson_r": [0.9, 0.85], "spearman_r": [0.88, 0.82],
            "max_abs_corr": [0.9, 0.85],
        })

    def test_graph_nodes(self):
        g = build_redundancy_graph(self._make_pairs())
        assert set(g.keys()) == {"A", "B", "C"}

    def test_graph_edges(self):
        g = build_redundancy_graph(self._make_pairs())
        assert "B" in g["A"]
        assert "A" in g["B"]


class TestConnectedComponents:
    def test_single_component(self):
        g = {"A": {"B", "C"}, "B": {"A", "C"}, "C": {"A", "B"}}
        comps = find_connected_components(g)
        assert len(comps) == 1
        assert comps[0] == {"A", "B", "C"}

    def test_two_components(self):
        g = {"A": {"B"}, "B": {"A"}, "C": {"D"}, "D": {"C"}}
        comps = find_connected_components(g)
        assert len(comps) == 2


class TestChooseRepresentative:
    def test_highest_degree_chosen(self):
        # A has degree 3, B has degree 1 - A should be representative
        group   = {"A", "B"}
        degrees = {"A": 3, "B": 1}
        assert choose_representative(group, degrees) == "A"

    def test_tie_broken_alphabetically(self):
        group   = {"B", "A"}
        degrees = {"A": 2, "B": 2}
        assert choose_representative(group, degrees) == "A"


class TestPruneFromCorrelationStats:
    def _make_stats(self):
        return pd.DataFrame({
            "var_a": ["A", "A", "B", "D", "F"],
            "var_b": ["B", "C", "C", "E", "G"],
            "pearson_r": [0.90, 0.88, 0.20, 0.10, 0.30],
            "spearman_r": [0.10, 0.20, 0.86, 0.20, 0.40],
        })

    def test_redundant_pairs_convert_to_selected_and_removed(self):
        summary = prune_from_correlation_stats(self._make_stats(), threshold=0.80)

        assert summary["selected_features"] == ["A", "D", "E", "F", "G"]
        assert summary["removed_features"] == ["B", "C"]
        assert summary["groups"][0]["representative"] == "A"
        assert summary["groups"][0]["members"] == ["A", "B", "C"]

    def test_disconnected_non_redundant_descriptors_are_kept(self):
        summary = prune_from_correlation_stats(self._make_stats(), threshold=0.80)

        assert {"D", "E", "F", "G"}.issubset(summary["selected_features"])

    def test_representative_selection_is_deterministic(self):
        first = prune_from_correlation_stats(self._make_stats(), threshold=0.80)
        second = prune_from_correlation_stats(self._make_stats(), threshold=0.80)

        assert first["groups"] == second["groups"]
        assert first["selected_features"] == second["selected_features"]
        assert first["removed_features"] == second["removed_features"]

    def test_summary_counts_are_correct(self):
        summary = prune_from_correlation_stats(self._make_stats(), threshold=0.80)

        assert summary["threshold"] == 0.80
        assert summary["n_all_descriptors"] == 7
        assert summary["n_selected"] == 5
        assert summary["n_removed"] == 2
        assert summary["n_redundant_pairs"] == 3
        assert summary["n_redundancy_groups"] == 1
