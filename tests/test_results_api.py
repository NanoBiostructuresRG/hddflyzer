# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for the public hddflyzer.results package API."""


def test_results_public_imports_work():
    from hddflyzer.results import (
        LoadedArtifact,
        ResultArtifact,
        WorkflowRun,
        classify_artifact,
        load_artifact,
        load_workflow_run,
    )

    assert WorkflowRun is not None
    assert load_workflow_run is not None
    assert ResultArtifact is not None
    assert classify_artifact is not None
    assert LoadedArtifact is not None
    assert load_artifact is not None


def test_results_all_contains_minimal_public_api():
    import hddflyzer.results as results

    expected = {
        "WorkflowRun",
        "load_workflow_run",
        "ResultArtifact",
        "classify_artifact",
        "LoadedArtifact",
        "load_artifact",
    }

    assert expected.issubset(set(results.__all__))
