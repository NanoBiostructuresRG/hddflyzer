# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for hddflyzer.config."""

import json

import pytest

from hddflyzer.config import BASE_COLUMNS, BASE_NAMES, load_descriptor_config


class TestConfig:
    def test_base_columns_loaded(self):
        assert len(BASE_COLUMNS) > 0

    def test_base_names_loaded(self):
        assert len(BASE_NAMES) > 0

    def test_all_columns_have_names(self):
        # Every BASE_COLUMN should have a name entry.
        missing = [c for c in BASE_COLUMNS if c not in BASE_NAMES]
        assert missing == [], f"Missing names for: {missing}"

    def test_hddf_excluded_by_default(self):
        # HDDF category should not appear in BASE_COLUMNS.
        hddf = [
            "QED",
            "LeadLikeness_Score",
            "Pharma_Complexity",
            "Synthetic_Accessibility",
            "Desirability_Profile",
        ]
        for descriptor in hddf:
            assert descriptor not in BASE_COLUMNS, (
                f"{descriptor} should be excluded"
            )

    def test_reference_features_excluded_by_default(self):
        reference_features = [
            "morgan_tanimoto",
            "rdk_tanimoto",
            "mcs_size",
            "mcs_tanimoto",
            "mcs_overlap",
        ]
        for descriptor in reference_features:
            assert descriptor not in BASE_COLUMNS, (
                f"{descriptor} should be excluded"
            )

    def test_custom_exclusions(self):
        cols, _ = load_descriptor_config(excluded_categories=[])
        # Without exclusions, HDDF columns should be present.
        assert any(c in cols for c in ["QED", "LeadLikeness_Score"])

    def test_version_2_format(self):
        # Should load without error; format is validated inside loader.
        cols, names = load_descriptor_config()
        assert isinstance(cols, list)
        assert isinstance(names, dict)

    def test_missing_descriptor_config_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_descriptor_config(path=str(tmp_path / "missing.json"))

    def test_malformed_descriptor_config_raises_value_error(self, tmp_path):
        cfg = tmp_path / "descriptor_config.json"
        cfg.write_text("{not json", encoding="utf-8")

        with pytest.raises(ValueError):
            load_descriptor_config(path=str(cfg))

    def test_missing_categories_raises_value_error(self, tmp_path):
        cfg = tmp_path / "descriptor_config.json"
        cfg.write_text(json.dumps({"names": {"MW": "Molecular Weight"}}), encoding="utf-8")

        with pytest.raises(ValueError):
            load_descriptor_config(path=str(cfg))

    def test_missing_names_raises_value_error(self, tmp_path):
        cfg = tmp_path / "descriptor_config.json"
        cfg.write_text(json.dumps({"categories": {"Base": ["MW"]}}), encoding="utf-8")

        with pytest.raises(ValueError):
            load_descriptor_config(path=str(cfg))

    def test_empty_columns_after_exclusions_raises_value_error(self, tmp_path):
        cfg = tmp_path / "descriptor_config.json"
        cfg.write_text(
            json.dumps({
                "categories": {"Only": ["MW"]},
                "names": {"MW": "Molecular Weight"},
            }),
            encoding="utf-8",
        )

        with pytest.raises(ValueError):
            load_descriptor_config(
                path=str(cfg),
                excluded_categories=[],
                excluded_descriptors=["MW"],
            )

    def test_valid_descriptor_config_returns_columns_and_names(self, tmp_path):
        cfg = tmp_path / "descriptor_config.json"
        cfg.write_text(
            json.dumps({
                "categories": {
                    "Base": ["MW", "TPSA"],
                    "HDDF": ["QED"],
                },
                "names": {
                    "MW": "Molecular Weight",
                    "TPSA": "Topological Polar Surface Area",
                    "QED": "QED",
                },
            }),
            encoding="utf-8",
        )

        cols, names = load_descriptor_config(path=str(cfg))

        assert cols == ["MW", "TPSA"]
        assert names["MW"] == "Molecular Weight"
