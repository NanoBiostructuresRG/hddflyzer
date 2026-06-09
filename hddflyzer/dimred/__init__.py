# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Dimensionality reduction module for HDDFlyzer.

Submodules
----------
pca             : PCA for individual collections (BASE and HDDF spaces)
pca_joint       : Joint PCA for two collections
tsne            : t-SNE from Tanimoto matrix, descriptors, or pruned features
umap            : UMAP from Tanimoto matrix, descriptors, or pruned features
"""

from .pca import (
    align_pc_signs,
    compute_pca,
    run as run_pca,
)

from .pca_joint import (
    run_joint_pca,
    run as run_pca_joint,
)

from .tsne import (
    run_tanimoto as run_tsne_tanimoto,
    run_features as run_tsne_features,
    run_pruning as run_tsne_pruning,
)

from .umap import (
    DEFAULT_N_NEIGHBORS,
    DEFAULT_MIN_DIST,
    run_features as run_umap_features,
    run_tanimoto as run_umap_tanimoto,
    run_pruning  as run_umap_pruning,
)
