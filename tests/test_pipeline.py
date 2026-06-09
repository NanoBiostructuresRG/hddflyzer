# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for the hddflyzer.pipeline orchestration layer."""

import json
import os

import pytest

import hddflyzer.cli as cli
from hddflyzer.config import get_dataset_path, settings
from hddflyzer.io import update_manifest
from hddflyzer.pipeline.contracts import PipelineContext, StageResult
from hddflyzer.pipeline.orchestrator import (
    WorkflowExecution,
    execute_workflow,
    run_pipeline,
    run_workflow,
)
from hddflyzer.pipeline.stages import FunctionStage, default_stages
from hddflyzer.results import WorkflowRun


def _workflow_run():
    return WorkflowRun(
        tag="aocd",
        manifest_path="manifest.json",
        manifest={
            "tag": "aocd",
            "operations": [],
            "current_outputs": [],
            "workflow_contract": {},
        },
    )


def test_workflow_execution_ok_when_all_stages_succeed():
    execution = WorkflowExecution(
        tag="aocd",
        stage_results=[
            StageResult("data.prepare", True),
            StageResult("chem.features", True),
        ],
        run=_workflow_run(),
    )

    assert execution.ok is True
    assert execution.failed_stages == []


def test_workflow_execution_reports_failed_stages_and_allows_no_run():
    failed = StageResult("chem.features", False)
    execution = WorkflowExecution(
        tag="aocd",
        stage_results=[
            StageResult("data.prepare", True),
            failed,
        ],
        run=None,
    )

    assert execution.ok is False
    assert execution.failed_stages == [failed]
    assert execution.run is None


def test_workflow_execution_public_import():
    from hddflyzer.pipeline import WorkflowExecution

    assert WorkflowExecution is not None


def test_function_stage_wraps_existing_operation():
    seen = {}

    def operation(context):
        seen["tag"] = context.tag
        return True

    result = FunctionStage("demo.stage", operation).run(
        PipelineContext(tag="aocd"))

    assert result == StageResult("demo.stage", True, "completed")
    assert seen["tag"] == "aocd"


def test_default_stages_include_expected_order():
    names = [stage.name for stage in default_stages()]

    assert names[:4] == [
        "data.prepare",
        "annotate.npc",
        "chem.tanimoto",
        "chem.sample",
    ]
    assert "dimred.tsne.pruning" in names
    assert "dimred.umap.pruning" in names


def test_default_stages_can_skip_sample_and_dimred():
    names = [stage.name for stage in default_stages(
        include_sample=False,
        include_dimred=False,
    )]

    assert "chem.sample" not in names
    assert "dimred.tsne.features" not in names
    assert names[-1] == "dimred.pca"


def test_run_pipeline_passes_stage_options(monkeypatch):
    seen = {}

    class DummyStage:
        name = "data.prepare"

        def run(self, context):
            return StageResult(self.name, True)

    def fake_default_stages(include_sample=True, include_dimred=True):
        seen["include_sample"] = include_sample
        seen["include_dimred"] = include_dimred
        return [DummyStage()]

    monkeypatch.setattr(
        "hddflyzer.pipeline.orchestrator.default_stages",
        fake_default_stages,
    )

    results = run_pipeline(
        "aocd",
        include_sample=False,
        include_dimred=False,
    )

    assert [result.name for result in results] == ["data.prepare"]
    assert seen == {
        "include_sample": False,
        "include_dimred": False,
    }


def test_run_pipeline_executes_selected_stages(monkeypatch):
    calls = []

    class DummyStage:
        def __init__(self, name):
            self.name = name

        def run(self, context):
            calls.append((self.name, context.tag))
            return StageResult(self.name, True)

    monkeypatch.setattr(
        "hddflyzer.pipeline.orchestrator.default_stages",
        lambda include_sample=True, include_dimred=True: [
            DummyStage("data.prepare"),
            DummyStage("chem.features"),
        ],
    )

    results = run_pipeline(
        " ao/cd ",
        stage_names=["chem.features"],
    )

    assert [r.name for r in results] == ["chem.features"]
    assert calls == [("chem.features", "ao_cd")]


def test_run_pipeline_executes_multiple_selected_stages_in_pipeline_order(
    monkeypatch,
):
    calls = []

    class DummyStage:
        def __init__(self, name):
            self.name = name

        def run(self, context):
            calls.append(self.name)
            return StageResult(self.name, True)

    monkeypatch.setattr(
        "hddflyzer.pipeline.orchestrator.default_stages",
        lambda include_sample=True, include_dimred=True: [
            DummyStage("data.prepare"),
            DummyStage("chem.features"),
            DummyStage("dimred.pca"),
        ],
    )

    results = run_pipeline(
        "aocd",
        stage_names=["dimred.pca", "chem.features"],
    )

    assert [r.name for r in results] == ["chem.features", "dimred.pca"]
    assert calls == ["chem.features", "dimred.pca"]


def test_run_pipeline_stops_on_failure(monkeypatch):
    calls = []

    class DummyStage:
        def __init__(self, name, ok):
            self.name = name
            self.ok = ok

        def run(self, context):
            calls.append(self.name)
            return StageResult(self.name, self.ok)

    monkeypatch.setattr(
        "hddflyzer.pipeline.orchestrator.default_stages",
        lambda include_sample=True, include_dimred=True: [
            DummyStage("first", False),
            DummyStage("second", True),
        ],
    )

    results = run_pipeline("aocd")

    assert [r.name for r in results] == ["first"]
    assert calls == ["first"]


def test_run_pipeline_continues_on_failure_when_requested(monkeypatch):
    calls = []

    class DummyStage:
        def __init__(self, name, ok):
            self.name = name
            self.ok = ok

        def run(self, context):
            calls.append((self.name, context.continue_on_error))
            return StageResult(self.name, self.ok)

    monkeypatch.setattr(
        "hddflyzer.pipeline.orchestrator.default_stages",
        lambda include_sample=True, include_dimred=True: [
            DummyStage("first", False),
            DummyStage("second", True),
        ],
    )

    results = run_pipeline("aocd", continue_on_error=True)

    assert [r.name for r in results] == ["first", "second"]
    assert calls == [("first", True), ("second", True)]


def test_run_pipeline_rejects_unknown_stage(monkeypatch):
    monkeypatch.setattr(
        "hddflyzer.pipeline.orchestrator.default_stages",
        lambda include_sample=True, include_dimred=True: [],
    )

    with pytest.raises(ValueError):
        run_pipeline("aocd", stage_names=["missing.stage"])


def test_run_workflow_delegates_and_returns_workflow_run(monkeypatch):
    seen = {}
    expected = WorkflowRun(
        tag="aocd",
        manifest_path="manifest.json",
        manifest={
            "tag": "aocd",
            "operations": [],
            "current_outputs": [],
            "workflow_contract": {},
        },
    )

    def fake_run_pipeline(**kwargs):
        seen.update(kwargs)
        return [StageResult("chem.features", True)]

    monkeypatch.setattr(
        "hddflyzer.pipeline.orchestrator.run_pipeline",
        fake_run_pipeline,
    )
    monkeypatch.setattr(
        "hddflyzer.pipeline.orchestrator.load_workflow_run",
        lambda tag: expected,
    )

    run = run_workflow(
        "aocd",
        stage_names=["chem.features"],
        include_sample=False,
        include_dimred=False,
        save_pickle=True,
        continue_on_error=False,
    )

    assert run is expected
    assert seen == {
        "tag": "aocd",
        "stage_names": ["chem.features"],
        "include_sample": False,
        "include_dimred": False,
        "save_pickle": True,
        "continue_on_error": False,
    }


def test_run_workflow_raises_when_stage_fails_without_continue(monkeypatch):
    monkeypatch.setattr(
        "hddflyzer.pipeline.orchestrator.run_pipeline",
        lambda **kwargs: [StageResult("chem.features", False)],
    )

    with pytest.raises(RuntimeError, match="chem.features"):
        run_workflow("aocd")


def test_run_workflow_loads_run_when_continue_on_error_is_true(monkeypatch):
    expected = WorkflowRun(
        tag="aocd",
        manifest_path="manifest.json",
        manifest={
            "tag": "aocd",
            "operations": [],
            "current_outputs": [],
            "workflow_contract": {},
        },
    )
    monkeypatch.setattr(
        "hddflyzer.pipeline.orchestrator.run_pipeline",
        lambda **kwargs: [StageResult("chem.features", False)],
    )
    monkeypatch.setattr(
        "hddflyzer.pipeline.orchestrator.load_workflow_run",
        lambda tag: expected,
    )

    assert run_workflow("aocd", continue_on_error=True) is expected


def test_execute_workflow_delegates_and_returns_execution(monkeypatch):
    seen = {}
    expected_run = _workflow_run()
    stage_results = [StageResult("chem.features", True)]

    def fake_run_pipeline(**kwargs):
        seen.update(kwargs)
        return stage_results

    monkeypatch.setattr(
        "hddflyzer.pipeline.orchestrator.run_pipeline",
        fake_run_pipeline,
    )
    monkeypatch.setattr(
        "hddflyzer.pipeline.orchestrator.load_workflow_run",
        lambda tag: expected_run,
    )

    execution = execute_workflow(
        " ao/cd ",
        stage_names=["chem.features"],
        include_sample=False,
        include_dimred=False,
        save_pickle=True,
        continue_on_error=True,
    )

    assert isinstance(execution, WorkflowExecution)
    assert execution.tag == "ao_cd"
    assert execution.stage_results == stage_results
    assert execution.run is expected_run
    assert seen == {
        "tag": " ao/cd ",
        "stage_names": ["chem.features"],
        "include_sample": False,
        "include_dimred": False,
        "save_pickle": True,
        "continue_on_error": True,
    }


def test_execute_workflow_allows_missing_reconstructed_run(monkeypatch):
    stage_results = [StageResult("chem.features", False)]
    monkeypatch.setattr(
        "hddflyzer.pipeline.orchestrator.run_pipeline",
        lambda **kwargs: stage_results,
    )
    monkeypatch.setattr(
        "hddflyzer.pipeline.orchestrator.load_workflow_run",
        lambda tag: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )

    execution = execute_workflow("aocd")

    assert execution.stage_results == stage_results
    assert execution.run is None
    assert execution.ok is False


def test_pipeline_public_imports_still_work():
    from hddflyzer.pipeline import execute_workflow, run_pipeline, run_workflow

    assert callable(execute_workflow)
    assert callable(run_pipeline)
    assert callable(run_workflow)


def test_cli_pipeline_run_delegates_to_execute_workflow(monkeypatch, tmp_path, capsys):
    seen = {}

    def fake_execute_workflow(**kwargs):
        seen.update(kwargs)
        return WorkflowExecution(
            tag=kwargs["tag"],
            stage_results=[StageResult("chem.features", True)],
            run=None,
        )

    monkeypatch.setattr(settings, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(
        "hddflyzer.pipeline.execute_workflow",
        fake_execute_workflow,
    )

    cli._pipeline([
        "run", "aocd",
        "--stages", "chem.features,dimred.pca",
        "--skip-sample",
        "--skip-dimred",
        "--save-pkl",
        "--continue-on-error",
    ])

    assert seen == {
        "tag": "aocd",
        "stage_names": ["chem.features", "dimred.pca"],
        "include_sample": False,
        "include_dimred": False,
        "save_pickle": True,
        "continue_on_error": True,
    }

    output = capsys.readouterr().out
    assert "[INFO] Results directory: " in output
    assert "results" in output
    assert "aocd" in output
    assert "[INFO] Workflow summary: " in output
    assert "workflow_summary.md" in output
    assert "[INFO] Manifest: " in output
    assert "manifest.json" in output


def test_cli_pipeline_run_writes_workflow_contract_smoke(
    monkeypatch,
    tmp_path,
    capsys,
):
    results_root = tmp_path / "results"
    monkeypatch.setattr(settings, "RESULTS_DIR", str(results_root))

    class ManifestStage:
        name = "data.prepare"

        def run(self, context):
            output = get_dataset_path(
                context.tag, "registry", "molecules.csv")
            os.makedirs(os.path.dirname(output), exist_ok=True)
            with open(output, "w", encoding="utf-8") as f:
                f.write("identifier,SMILES\nA1,CCO\n")
            update_manifest(
                context.tag,
                self.name,
                [output],
                metadata={"n_molecules": 1},
            )
            return StageResult(self.name, True)

    monkeypatch.setattr(
        "hddflyzer.pipeline.orchestrator.default_stages",
        lambda include_sample=True, include_dimred=True: [ManifestStage()],
    )

    cli._pipeline(["run", "aocd"])

    manifest_path = results_root / "aocd" / "manifest.json"
    summary_path = results_root / "aocd" / "workflow_summary.md"
    assert manifest_path.exists()
    assert summary_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = summary_path.read_text(encoding="utf-8")
    output = capsys.readouterr().out

    assert "workflow_contract" in manifest
    assert manifest["workflow_contract"]["collection"] == {"tag": "aocd"}
    assert "registry -> chem -> dimred -> viz -> metadata/results" in summary
    assert f"[INFO] Results directory: {results_root / 'aocd'}" in output
    assert f"[INFO] Workflow summary: {summary_path}" in output
    assert f"[INFO] Manifest: {manifest_path}" in output


def test_cli_pipeline_run_exits_when_stage_fails(monkeypatch):
    monkeypatch.setattr(
        "hddflyzer.pipeline.execute_workflow",
        lambda **kwargs: WorkflowExecution(
            tag=kwargs["tag"],
            stage_results=[StageResult("chem.features", False)],
            run=None,
        ),
    )

    with pytest.raises(SystemExit) as exc:
        cli._pipeline(["run", "aocd"])

    assert exc.value.code == 1


def test_cli_pipeline_run_continue_on_error_still_exits_on_failed_stage(monkeypatch):
    seen = {}

    def fake_execute_workflow(**kwargs):
        seen.update(kwargs)
        return WorkflowExecution(
            tag=kwargs["tag"],
            stage_results=[
                StageResult("chem.features", False),
                StageResult("dimred.pca", True),
            ],
            run=None,
        )

    monkeypatch.setattr(
        "hddflyzer.pipeline.execute_workflow",
        fake_execute_workflow,
    )

    with pytest.raises(SystemExit) as exc:
        cli._pipeline(["run", "aocd", "--continue-on-error"])

    assert exc.value.code == 1
    assert seen["continue_on_error"] is True


def test_cli_pipeline_run_rejects_incomplete_option():
    with pytest.raises(SystemExit) as exc:
        cli._pipeline(["run", "aocd", "--stages"])

    assert exc.value.code == 1
