# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for hddflyzer.chem.tanimoto."""

import numpy as np
import pandas as pd

from hddflyzer.chem import tanimoto
from hddflyzer.chem.tanimoto import stats_from_matrix


class TestTanimotoMetadata:
    def test_stats_include_unique_and_cluster_metadata(self):
        mat = np.array([
            [1.0, 0.60, 0.10],
            [0.60, 1.0, 0.20],
            [0.10, 0.20, 1.0],
        ], dtype=np.float32)

        stats = stats_from_matrix(mat)

        assert stats["mean"] == round(float(np.mean([0.60, 0.10, 0.20])), 6)
        assert stats["median"] == 0.2
        assert stats["unique_threshold"] == 0.5
        assert stats["n_unique"] == 1
        assert stats["cluster_distance_threshold"] == 0.4
        assert "n_clusters" in stats


def test_run_removes_stale_invalid_smiles_when_current_run_has_no_invalid(
    tmp_path,
    monkeypatch,
):
    out_dir = tmp_path / "tanimoto"
    out_dir.mkdir()
    stale = out_dir / "invalid_smiles.csv"
    stale.write_text("index,id,smiles\n0,bad,notasmiles\n", encoding="utf-8")

    registry = pd.DataFrame({
        "identifier": ["A", "B"],
        "canonical_smiles": ["CCO", "CCN"],
    })

    monkeypatch.setattr(tanimoto, "_RDKIT_AVAILABLE", True)
    monkeypatch.setattr(tanimoto, "get_path", lambda kind, tag: str(out_dir))
    monkeypatch.setattr(tanimoto, "ensure_registry", lambda tag: "registry.csv")
    monkeypatch.setattr(tanimoto, "load_registry", lambda tag, valid_only=True: registry)
    monkeypatch.setattr(tanimoto, "build_morgan_fps", lambda smiles: (["fp1", "fp2"], [0, 1]))
    monkeypatch.setattr(
        tanimoto,
        "compute_tanimoto_matrix",
        lambda fps: np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32),
    )
    manifest_call = {}

    def fake_update_manifest(tag, operation, files, metadata):
        manifest_call.update({
            "tag": tag,
            "operation": operation,
            "files": files,
            "metadata": metadata,
        })

    monkeypatch.setattr(tanimoto, "update_manifest", fake_update_manifest)

    assert tanimoto.run("aocd") is True
    assert not stale.exists()
    assert manifest_call["operation"] == "chem.tanimoto"
    assert manifest_call["metadata"]["tag"] == "aocd"
    assert manifest_call["metadata"]["source_registry"] == "registry.csv"
    assert manifest_call["metadata"]["id_column"] == "identifier"
    assert manifest_call["metadata"]["smiles_column"] == "canonical_smiles"
    assert manifest_call["metadata"]["n_input_compounds"] == 2
    assert manifest_call["metadata"]["n_valid_fingerprints"] == 2
    assert manifest_call["metadata"]["n_invalid_smiles"] == 0
    assert manifest_call["metadata"]["fingerprint"] == {
        "type": "Morgan",
        "radius": tanimoto.RADIUS,
        "n_bits": tanimoto.NBITS,
    }
