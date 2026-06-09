# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for the hddflyzer.data package."""

import json

import numpy as np
import pandas as pd

from hddflyzer.config import settings
from hddflyzer.data.registry import (
    build_registry_frame,
    find_input_file,
    load_registry,
    prepare_registry,
)


def _normalized_paths(paths):
    return [str(path).replace("\\", "/") for path in paths]


class TestRegistry:
    def test_prepare_registry_creates_canonical_table(self, monkeypatch, tmp_path):
        src = tmp_path / "examples" / "valid_metadata_aocd.csv"
        out_dir = tmp_path / "results" / "aocd" / "registry"
        src.parent.mkdir(parents=True)
        src.write_text("identifier,SMILES\nA1,CCO\nA2,\n", encoding="utf-8")
        monkeypatch.setattr(settings, "RESULTS_DIR", str(tmp_path / "results"))

        prepare_registry("aocd", input_file=str(src), output_dir=str(out_dir))

        monkeypatch.setattr(
            "hddflyzer.data.registry.resolve_registry_csv",
            lambda tag: str(out_dir / "molecules.csv"),
        )
        df = load_registry("aocd", valid_only=False)

        assert {"identifier", "SMILES", "valid_smiles"}.issubset(df.columns)
        assert len(df) == 2
        assert df.loc[df["identifier"] == "A1", "valid_smiles"].iloc[0] in (
            True,
            np.True_,
        )
        assert not (out_dir / "registry_metadata.json").exists()

        manifest = json.loads(
            (tmp_path / "results" / "aocd" / "manifest.json").read_text()
        )
        op = manifest["operations"][0]

        assert _normalized_paths(op["files"]) == ["registry/molecules.csv"]
        assert op["metadata"]["validation"]["n_rows"] == 2
        assert op["metadata"]["validation"]["duplicated_identifiers"] == 0

    def test_find_input_file_uses_configured_examples_directory(self, tmp_path):
        src = tmp_path / "valid_metadata_aocd.csv"
        src.write_text("identifier,SMILES\nA1,CCO\n", encoding="utf-8")

        assert find_input_file("aocd", data_dir=str(tmp_path)) == str(src)

    def test_build_registry_frame_uses_existing_id_column(self):
        df = pd.DataFrame({"identifier": ["A1"], "SMILES": ["CCO"]})

        registry, meta = build_registry_frame(df, source_file="input.csv")

        assert registry["identifier"].tolist() == ["A1"]
        assert meta["source_id_column"] == "identifier"

    def test_build_registry_frame_generates_ids_when_missing(self):
        df = pd.DataFrame({"SMILES": ["CCO", "CCN"]})

        registry, meta = build_registry_frame(df, source_file="input.csv")

        assert registry["identifier"].tolist() == ["COMPD_0001", "COMPD_0002"]
        assert meta["source_id_column"] is None

    def test_build_registry_frame_removes_duplicate_identifiers(self):
        df = pd.DataFrame({
            "identifier": ["A1", "A1", "A2"],
            "SMILES": ["CCO", "CCN", "CCC"],
        })

        registry, meta = build_registry_frame(df, source_file="input.csv")

        assert registry["identifier"].tolist() == ["A1", "A2"]
        assert meta["duplicates_removed_by_identifier"] == 1
        assert meta["n_source_rows"] == 3
        assert meta["n_registry_rows"] == 2

    def test_build_registry_frame_columns_and_validation_metadata(self):
        df = pd.DataFrame({"identifier": ["A1"], "SMILES": ["CCO"]})

        registry, meta = build_registry_frame(df, source_file="input.csv")

        assert registry.columns.tolist() == [
            "identifier",
            "SMILES",
            "canonical_smiles",
            "valid_smiles",
            "mol_parse_status",
            "source_file",
            "source_row",
        ]
        assert meta["source_smiles_column"] == "SMILES"
        assert meta["n_valid_smiles"] == 1
        assert meta["validation"] == {
            "n_rows": 1,
            "n_valid_smiles": 1,
            "duplicated_identifiers": 0,
            "ok": True,
        }
