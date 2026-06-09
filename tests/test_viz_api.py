# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for the public hddflyzer.viz package API."""

import pytest


def test_import_hddflyzer_viz_works():
    import hddflyzer.viz as viz

    assert viz is not None


def test_viz_public_imports_work():
    from hddflyzer.viz import (
        VizInputs,
        plot_hddf_scatters,
        resolve_viz_inputs,
    )

    assert VizInputs is not None
    assert callable(resolve_viz_inputs)
    assert callable(plot_hddf_scatters)


def test_viz_all_contains_expected_names():
    import hddflyzer.viz as viz

    expected = {
        "VizInputs",
        "resolve_viz_inputs",
        "plot_distributions",
        "plot_hddf_scatters",
        "plot_tanimoto",
        "plot_fingerprints",
        "plot_pca",
        "plot_pca_collections",
        "plot_tsne_tanimoto",
        "plot_tsne_features",
        "plot_tsne_pruning",
        "plot_umap_features",
        "plot_umap_tanimoto",
        "plot_umap_pruning",
        "plot_npclassifier",
    }

    assert set(viz.__all__) == expected


def test_unknown_viz_attribute_raises_attribute_error():
    import hddflyzer.viz as viz

    with pytest.raises(AttributeError):
        viz.not_a_real_plot


def test_lazy_viz_export_resolves():
    import hddflyzer.viz as viz

    assert callable(viz.plot_hddf_scatters)
    assert callable(viz.resolve_viz_inputs)
