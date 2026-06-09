# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for the central hddflyzer CLI helpers."""

import sys
import types

import pytest

import hddflyzer.cli as cli


def _fake_module(seen, raises=None):
    module = types.ModuleType("fake_module")

    def main():
        seen.append(sys.argv[:])
        if raises is not None:
            raise raises

    module.main = main
    return module


def test_run_submain_patches_and_restores_sys_argv(monkeypatch):
    seen = []
    original = ["outer", "command"]
    monkeypatch.setattr(sys, "argv", original[:])

    def fake_main():
        seen.append(sys.argv[:])

    cli._run_submain(fake_main, ["hddflyzer", "aocd"])

    assert seen == [["hddflyzer", "aocd"]]
    assert sys.argv == original


def test_run_submain_restores_sys_argv_after_system_exit(monkeypatch):
    original = ["outer", "command"]
    monkeypatch.setattr(sys, "argv", original[:])

    def fake_main():
        raise SystemExit(7)

    with pytest.raises(SystemExit) as exc:
        cli._run_submain(fake_main, ["hddflyzer", "aocd"])

    assert exc.value.code == 7
    assert sys.argv == original


def test_chem_router_delegates_with_standard_argv(monkeypatch):
    seen = []
    monkeypatch.setitem(
        sys.modules,
        "hddflyzer.chem.tanimoto",
        _fake_module(seen),
    )

    cli._chem(["tanimoto", "aocd"])

    assert seen == [["hddflyzer", "aocd"]]


def test_data_prepare_router_preserves_subcommand_in_argv(monkeypatch):
    seen = []
    monkeypatch.setitem(
        sys.modules,
        "hddflyzer.data.registry",
        _fake_module(seen),
    )

    cli._data(["prepare", "aocd", "input.csv"])

    assert seen == [["hddflyzer", "prepare", "aocd", "input.csv"]]


def test_reference_features_router_preserves_reference_mode(monkeypatch):
    seen = []
    monkeypatch.setitem(
        sys.modules,
        "hddflyzer.chem.feature_engineering",
        _fake_module(seen),
    )

    cli._chem(["reference-features", "aocd", "--reference", "custom"])

    assert seen == [["hddflyzer", "reference", "aocd", "--reference", "custom"]]
