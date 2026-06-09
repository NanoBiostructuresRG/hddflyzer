# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Visualization module for HDDFlyzer.

Submodules
----------
distributions      : Violin, split-violin, and KDE plots across tags
correlations       : HDDF scatter plots
similarity         : Tanimoto histogram and fingerprint comparison
pca_plots          : PCA figures (single collection and joint collections)
tsne_plots         : t-SNE plots (tanimoto, features, and pruning modes)
umap_plots         : UMAP scatter plots (features, tanimoto, and pruning modes)
npclassifier_plots : NPClassifier pie charts and hierarchical bar
"""

from importlib import import_module


_LAZY_EXPORTS = {
    "VizInputs": ("hddflyzer.viz.inputs", "VizInputs"),
    "resolve_viz_inputs": ("hddflyzer.viz.inputs", "resolve_viz_inputs"),
    "plot_distributions": ("hddflyzer.viz.distributions", "run"),
    "plot_hddf_scatters": ("hddflyzer.viz.correlations", "plot_hddf_scatters"),
    "plot_tanimoto": ("hddflyzer.viz.similarity", "plot_tanimoto"),
    "plot_fingerprints": ("hddflyzer.viz.similarity", "plot_fingerprints"),
    "plot_pca": ("hddflyzer.viz.pca_plots", "plot_analysis"),
    "plot_pca_collections": ("hddflyzer.viz.pca_plots", "plot_collections"),
    "plot_tsne_tanimoto": ("hddflyzer.viz.tsne_plots", "plot_tsne_tanimoto"),
    "plot_tsne_features": ("hddflyzer.viz.tsne_plots", "plot_tsne_features"),
    "plot_tsne_pruning": ("hddflyzer.viz.tsne_plots", "plot_tsne_pruning"),
    "plot_umap_features": ("hddflyzer.viz.umap_plots", "plot_umap_features"),
    "plot_umap_tanimoto": ("hddflyzer.viz.umap_plots", "plot_umap_tanimoto"),
    "plot_umap_pruning": ("hddflyzer.viz.umap_plots", "plot_umap_pruning"),
    "plot_npclassifier": ("hddflyzer.viz.npclassifier_plots", "run"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
