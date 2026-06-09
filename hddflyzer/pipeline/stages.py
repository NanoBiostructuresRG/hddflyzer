# SPDX-License-Identifier: LGPL-3.0-or-later

"""Concrete pipeline stages wrapping HDDFlyzer domain operations."""

from dataclasses import dataclass
from typing import Callable

from .contracts import PipelineContext, StageResult


@dataclass(frozen=True)
class FunctionStage:
    """Small adapter that gives existing run functions a common interface."""

    name: str
    operation: Callable[[PipelineContext], bool]

    def run(self, context: PipelineContext) -> StageResult:
        ok = bool(self.operation(context))
        return StageResult(
            name=self.name,
            ok=ok,
            message="completed" if ok else "failed",
        )


def default_stages(include_sample: bool = True, include_dimred: bool = True):
    """Return the default real-data pipeline stages in execution order."""
    stages = [
        FunctionStage("data.prepare", _prepare_data),
        FunctionStage("annotate.npc", _annotate_npc),
        FunctionStage("chem.tanimoto", _tanimoto),
    ]
    if include_sample:
        stages.append(FunctionStage("chem.sample", _sample))

    stages.extend([
        FunctionStage("chem.features", _features),
        FunctionStage("chem.curate_features", _curate_features),
        FunctionStage("chem.stats.base", _stats_base),
        FunctionStage("chem.stats.hddf", _stats_hddf),
        FunctionStage("chem.pruning", _pruning),
        FunctionStage("dimred.pca", _pca),
    ])

    if include_dimred:
        stages.extend([
            FunctionStage("dimred.tsne.features", _tsne_features),
            FunctionStage("dimred.tsne.tanimoto", _tsne_tanimoto),
            FunctionStage("dimred.tsne.pruning", _tsne_pruning),
            FunctionStage("dimred.umap.features", _umap_features),
            FunctionStage("dimred.umap.tanimoto", _umap_tanimoto),
            FunctionStage("dimred.umap.pruning", _umap_pruning),
        ])

    return stages


def _prepare_data(context: PipelineContext) -> bool:
    from hddflyzer.data.registry import run_prepare
    return run_prepare(context.tag)


def _annotate_npc(context: PipelineContext) -> bool:
    from hddflyzer.chem.npclassifier import run
    return run(context.tag)


def _tanimoto(context: PipelineContext) -> bool:
    from hddflyzer.chem.tanimoto import run
    return run(context.tag)


def _sample(context: PipelineContext) -> bool:
    from hddflyzer.chem.tanimoto_sampling import run
    return run(context.tag)


def _features(context: PipelineContext) -> bool:
    from hddflyzer.chem.feature_engineering import run
    return run(
        context.tag,
        sampled=False,
        save_pickle=context.save_pickle,
    )


def _curate_features(context: PipelineContext) -> bool:
    from hddflyzer.chem.feature_curation import run
    return run(context.tag)


def _stats_base(context: PipelineContext) -> bool:
    from hddflyzer.chem.stats import run_base_stats
    return run_base_stats(context.tag)


def _stats_hddf(context: PipelineContext) -> bool:
    from hddflyzer.chem.stats import run_hddf_stats
    return run_hddf_stats(context.tag)


def _pruning(context: PipelineContext) -> bool:
    from hddflyzer.chem.pruning import run
    return run(context.tag)


def _pca(context: PipelineContext) -> bool:
    from hddflyzer.dimred.pca import run
    return run(context.tag)


def _tsne_features(context: PipelineContext) -> bool:
    from hddflyzer.dimred.tsne import run_features
    return run_features(context.tag)


def _tsne_tanimoto(context: PipelineContext) -> bool:
    from hddflyzer.dimred.tsne import run_tanimoto
    return run_tanimoto(context.tag)


def _tsne_pruning(context: PipelineContext) -> bool:
    from hddflyzer.dimred.tsne import run_pruning
    return run_pruning(context.tag)


def _umap_features(context: PipelineContext) -> bool:
    from hddflyzer.dimred.umap import run_features
    return run_features(context.tag)


def _umap_tanimoto(context: PipelineContext) -> bool:
    from hddflyzer.dimred.umap import run_tanimoto
    return run_tanimoto(context.tag)


def _umap_pruning(context: PipelineContext) -> bool:
    from hddflyzer.dimred.umap import run_pruning
    return run_pruning(context.tag)
