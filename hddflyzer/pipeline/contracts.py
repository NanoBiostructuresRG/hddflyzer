# SPDX-License-Identifier: LGPL-3.0-or-later

"""Shared contracts for executable HDDFlyzer pipeline stages."""

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol


@dataclass
class PipelineContext:
    """Runtime options shared by pipeline stages.

    Attributes
    ----------
    tag : str
        Dataset tag for the run. Pipeline stages use this tag to resolve
        inputs and outputs under ``results/<tag>/``.
    save_pickle : bool, default=False
        Whether stages that support optional pickle output should write it.
    continue_on_error : bool, default=False
        Whether the pipeline runner should continue after a failed stage.
    options : dict
        Additional stage options. The core pipeline keeps this mapping generic
        so stages can receive small, stage-specific values without changing the
        shared context contract.
    """

    tag: str
    save_pickle: bool = False
    continue_on_error: bool = False
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageResult:
    """Result returned by a pipeline stage.

    Attributes
    ----------
    name : str
        Stage name, for example ``"chem.features"``.
    ok : bool
        ``True`` when the stage completed successfully.
    message : str, default=""
        Optional human-readable status or error message.
    """

    name: str
    ok: bool
    message: str = ""


class Stage(Protocol):
    """Executable pipeline stage interface.

    Stages are small objects with a ``name`` and a ``run`` method. They are
    consumed by ``run_pipeline`` and return ``StageResult`` instances.

    Attributes
    ----------
    name : str
        Unique stage name used for selection and reporting.
    """

    name: str

    def run(self, context: PipelineContext) -> StageResult:
        """Execute the stage with the given pipeline context.

        Parameters
        ----------
        context : PipelineContext
            Shared runtime options for the current workflow execution.

        Returns
        -------
        StageResult
            Result describing whether the stage succeeded.
        """
