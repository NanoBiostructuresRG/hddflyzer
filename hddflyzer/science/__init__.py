# SPDX-License-Identifier: LGPL-3.0-or-later

"""Scientific views over loaded HDDFlyzer result artifacts."""

from .metrics import (
    DescriptorGroupComparisonResult,
    DescriptorProjectionCorrelationResult,
    NeighborhoodPreservationResult,
    SpaceMetricResult,
    compare_descriptor_groups,
    descriptor_projection_correlations,
    projection_neighborhood_preservation,
    similarity_projection_correlation,
    similarity_projection_neighbor_overlap,
)
from .spaces import (
    DescriptorSpace,
    ProjectionSpace,
    SimilaritySpace,
    align_spaces,
    has_aligned_molecule_ids,
    shared_molecule_ids,
    to_descriptor_space,
    to_projection_space,
    to_similarity_space,
)

__all__ = [
    "DescriptorSpace",
    "DescriptorGroupComparisonResult",
    "DescriptorProjectionCorrelationResult",
    "NeighborhoodPreservationResult",
    "ProjectionSpace",
    "SimilaritySpace",
    "SpaceMetricResult",
    "align_spaces",
    "compare_descriptor_groups",
    "descriptor_projection_correlations",
    "projection_neighborhood_preservation",
    "has_aligned_molecule_ids",
    "shared_molecule_ids",
    "similarity_projection_correlation",
    "similarity_projection_neighbor_overlap",
    "to_descriptor_space",
    "to_projection_space",
    "to_similarity_space",
]
