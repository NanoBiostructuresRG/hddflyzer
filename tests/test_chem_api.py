# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for the public hddflyzer.chem package API."""

import pytest


def test_import_hddflyzer_chem_works():
    import hddflyzer.chem as chem

    assert chem is not None


def test_eager_public_names_are_available():
    import hddflyzer.chem as chem

    assert isinstance(chem.HDDF_COLUMNS, list)
    assert chem.DEFAULT_THRESHOLD == 0.80
    assert callable(chem.curate_features)


def test_chem_all_contains_eager_and_lazy_names():
    import hddflyzer.chem as chem

    expected = {
        "HDDF_COLUMNS",
        "DEFAULT_THRESHOLD",
        "curate_features",
        "stats_from_matrix",
        "calculate_mcs_features",
        "run_tanimoto",
    }

    assert expected.issubset(set(chem.__all__))


def test_unknown_attribute_raises_attribute_error():
    import hddflyzer.chem as chem

    with pytest.raises(AttributeError):
        chem.not_a_real_export


def test_lazy_rdkit_dependent_symbol_resolves():
    import hddflyzer.chem as chem

    assert callable(chem.stats_from_matrix)
