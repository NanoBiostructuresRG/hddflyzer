# SPDX-License-Identifier: LGPL-3.0-or-later

"""Explicit pipeline runner for HDDFlyzer real-data workflows."""

from dataclasses import dataclass
from typing import Iterable, List, Optional

from hddflyzer.utils.naming import sanitize_tag

from hddflyzer.results import WorkflowRun, load_workflow_run

from .contracts import PipelineContext, StageResult
from .stages import default_stages


@dataclass(frozen=True)
class WorkflowExecution:
    """Complete programmatic result of a workflow execution.

    Attributes
    ----------
    tag : str
        Sanitized dataset tag that was executed.
    stage_results : list of StageResult
        Per-stage execution results returned by ``run_pipeline``.
    run : WorkflowRun or None
        Reconstructed workflow run when ``manifest.json`` could be loaded.
        ``None`` means execution produced stage results but the run could not
        be reconstructed.
    """

    tag: str
    stage_results: list[StageResult]
    run: WorkflowRun | None

    @property
    def ok(self) -> bool:
        """bool: Whether every recorded stage succeeded."""
        return all(result.ok for result in self.stage_results)

    @property
    def failed_stages(self) -> list[StageResult]:
        """list of StageResult: Stage results with ``ok`` set to ``False``."""
        return [result for result in self.stage_results if not result.ok]


def run_pipeline(
    tag: str,
    stage_names: Optional[Iterable[str]] = None,
    include_sample: bool = True,
    include_dimred: bool = True,
    save_pickle: bool = False,
    continue_on_error: bool = False,
) -> List[StageResult]:
    """Run an ordered HDDFlyzer pipeline for a dataset tag.

    Parameters
    ----------
    tag : str
        Dataset tag to process.
    stage_names : iterable of str, optional
        Stage names to run. When omitted, the default stage sequence is used.
    include_sample : bool, default=True
        Whether to include the Tanimoto sampling stage in the default stage
        sequence.
    include_dimred : bool, default=True
        Whether to include dimensionality-reduction stages in the default stage
        sequence.
    save_pickle : bool, default=False
        Whether stages that support optional pickle output should write it.
    continue_on_error : bool, default=False
        Whether to continue executing stages after a failed stage.

    Returns
    -------
    list of StageResult
        Per-stage results in execution order.

    Raises
    ------
    ValueError
        If ``stage_names`` contains an unknown stage name.

    Notes
    -----
    This function performs the file-based workflow and writes the normal
    HDDFlyzer outputs under ``results/<tag>/``. It returns stage status only;
    use ``execute_workflow`` when both status and reconstructed results are
    needed.
    """
    tag = sanitize_tag(tag)
    requested = set(stage_names or [])
    context = PipelineContext(
        tag=tag,
        save_pickle=save_pickle,
        continue_on_error=continue_on_error,
    )
    stages = default_stages(
        include_sample=include_sample,
        include_dimred=include_dimred,
    )
    if requested:
        stages = [stage for stage in stages if stage.name in requested]
        missing = requested - {stage.name for stage in stages}
        if missing:
            raise ValueError(f"Unknown pipeline stages: {', '.join(sorted(missing))}")

    results = []
    print(f"[INFO] Running pipeline for tag='{tag}'")
    for stage in stages:
        print(f"[PIPELINE] {stage.name}")
        result = stage.run(context)
        results.append(result)
        if not result.ok and not context.continue_on_error:
            print(f"[ERROR] Pipeline stopped at stage: {stage.name}")
            break

    ok_count = sum(1 for result in results if result.ok)
    print(f"[DONE] {ok_count}/{len(results)} stages completed.")
    return results


def run_workflow(
    tag: str,
    stage_names: Optional[Iterable[str]] = None,
    include_sample: bool = True,
    include_dimred: bool = True,
    save_pickle: bool = False,
    continue_on_error: bool = False,
) -> WorkflowRun:
    """Run a pipeline and return the reconstructed workflow run.

    Parameters
    ----------
    tag : str
        Dataset tag to process.
    stage_names : iterable of str, optional
        Stage names to run. When omitted, the default stage sequence is used.
    include_sample : bool, default=True
        Whether to include the Tanimoto sampling stage.
    include_dimred : bool, default=True
        Whether to include dimensionality-reduction stages.
    save_pickle : bool, default=False
        Whether stages that support optional pickle output should write it.
    continue_on_error : bool, default=False
        Whether to continue after failed stages.

    Returns
    -------
    WorkflowRun
        Reconstructed run loaded from ``results/<tag>/manifest.json``.

    Raises
    ------
    RuntimeError
        If one or more stages fail and ``continue_on_error`` is ``False``.
    FileNotFoundError
        If the run manifest cannot be found after execution.
    ValueError
        If the run manifest is invalid or cannot be reconstructed.
    """
    results = run_pipeline(
        tag=tag,
        stage_names=stage_names,
        include_sample=include_sample,
        include_dimred=include_dimred,
        save_pickle=save_pickle,
        continue_on_error=continue_on_error,
    )
    failed = [result.name for result in results if not result.ok]
    if failed and not continue_on_error:
        raise RuntimeError(
            "Workflow failed in stages: " + ", ".join(failed)
        )
    return load_workflow_run(sanitize_tag(tag))


def execute_workflow(
    tag: str,
    stage_names: Optional[Iterable[str]] = None,
    include_sample: bool = True,
    include_dimred: bool = True,
    save_pickle: bool = False,
    continue_on_error: bool = False,
) -> WorkflowExecution:
    """Run a pipeline and return execution status plus reconstructed results.

    Parameters
    ----------
    tag : str
        Dataset tag to process.
    stage_names : iterable of str, optional
        Stage names to run. When omitted, the default stage sequence is used.
    include_sample : bool, default=True
        Whether to include the Tanimoto sampling stage.
    include_dimred : bool, default=True
        Whether to include dimensionality-reduction stages.
    save_pickle : bool, default=False
        Whether stages that support optional pickle output should write it.
    continue_on_error : bool, default=False
        Whether to continue after failed stages.

    Returns
    -------
    WorkflowExecution
        Execution object containing stage results, global success status, and
        the reconstructed ``WorkflowRun`` when available.

    Notes
    -----
    ``execute_workflow`` is the common programmatic contract used by the CLI.
    It does not hide stage failures; inspect ``execution.stage_results`` and
    ``execution.ok`` for status.
    """
    clean_tag = sanitize_tag(tag)
    stage_results = run_pipeline(
        tag=tag,
        stage_names=stage_names,
        include_sample=include_sample,
        include_dimred=include_dimred,
        save_pickle=save_pickle,
        continue_on_error=continue_on_error,
    )
    try:
        run = load_workflow_run(clean_tag)
    except (FileNotFoundError, ValueError):
        run = None
    return WorkflowExecution(
        tag=clean_tag,
        stage_results=stage_results,
        run=run,
    )
