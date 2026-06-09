# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for hddflyzer.chem.reference_catalog."""

from hddflyzer.chem import feature_engineering
from hddflyzer.chem.reference_catalog import (
    REFERENCE_MOLECULES,
    USER_REFERENCE_TOKEN,
    resolve_reference_name,
)


def test_reference_molecules_available_from_catalog():
    assert "rosiglitazone" in REFERENCE_MOLECULES
    assert REFERENCE_MOLECULES["rosiglitazone"]


def test_resolve_reference_name_accepts_number_name_and_user_alias():
    assert resolve_reference_name("1") == "rosiglitazone"
    assert resolve_reference_name("rosiglitazone") == "rosiglitazone"
    assert resolve_reference_name("U") == USER_REFERENCE_TOKEN
    assert resolve_reference_name("999") is None


def test_feature_engineering_reexports_reference_catalog_imports():
    assert feature_engineering.REFERENCE_MOLECULES is REFERENCE_MOLECULES
    assert feature_engineering.resolve_reference_name("user") == USER_REFERENCE_TOKEN
