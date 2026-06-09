# SPDX-License-Identifier: LGPL-3.0-or-later

"""Public API tests for hddflyzer.science."""

import hddflyzer.science as science


def test_science_public_exports_are_explicit_and_complete():
    expected = {
        "DescriptorGroupComparisonResult",
        "DescriptorProjectionCorrelationResult",
        "DescriptorSpace",
        "NeighborhoodPreservationResult",
        "ProjectionSpace",
        "SimilaritySpace",
        "SpaceMetricResult",
        "align_spaces",
        "compare_descriptor_groups",
        "descriptor_projection_correlations",
        "has_aligned_molecule_ids",
        "projection_neighborhood_preservation",
        "shared_molecule_ids",
        "similarity_projection_correlation",
        "similarity_projection_neighbor_overlap",
        "to_descriptor_space",
        "to_projection_space",
        "to_similarity_space",
    }

    assert set(science.__all__) == expected
    for name in expected:
        assert getattr(science, name) is not None


def test_science_public_exports_do_not_include_internal_helpers():
    assert all(not name.startswith("_") for name in science.__all__)
    assert "DESCRIPTOR_GROUP_COLUMNS" not in science.__all__
