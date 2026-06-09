# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for visualization helpers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from hddflyzer.config import settings
from hddflyzer.viz.distributions import _resolve_input
from hddflyzer.viz.npclassifier_plots import (
    _fig_class,
    _fig_pathway,
    _fig_superclass,
    _resolve_plot_selection,
)


def _npc_df():
    return pd.DataFrame({
        "Pathway": ["Terpenoids", "Terpenoids", "Alkaloids"],
        "Superclass": ["A", "A", "B"],
        "Class": ["C1", "C2", "C1"],
    })


def test_npclassifier_pathway_figure_can_be_built():
    fig = _fig_pathway(_npc_df(), "aocd")
    try:
        assert len(fig.axes) == 1
        assert fig.axes[0].get_legend() is not None
    finally:
        plt.close(fig)


def test_npclassifier_superclass_figure_can_be_built():
    fig = _fig_superclass(_npc_df(), "aocd")
    try:
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)


def test_npclassifier_class_figure_can_be_built():
    fig = _fig_class(_npc_df(), "aocd")
    try:
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)


def test_npclassifier_plot_selection_aliases():
    assert _resolve_plot_selection("1") == ["pathway"]
    assert _resolve_plot_selection("class") == ["class"]
    assert _resolve_plot_selection("3") == ["superclass"]
    assert _resolve_plot_selection("all") == ["pathway", "class", "superclass"]


def test_distribution_input_does_not_select_pickle_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "RESULTS_DIR", str(tmp_path / "results"))
    features = tmp_path / "results" / "aocd" / "features" / "full"
    features.mkdir(parents=True)
    pd.DataFrame({"identifier": ["A1"], "MW": [100.0]}).to_pickle(
        features / "features.pkl")

    with pytest.raises(FileNotFoundError):
        _resolve_input("aocd")

    assert _resolve_input("aocd", allow_pickle=True).endswith("features.pkl")
