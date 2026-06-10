# API Overview and Reference

This page shows how to use HDDFlyzer programmatically from Python, then lists
the current public API surface generated from NumPy-style docstrings.

HDDFlyzer is local pre-release research software. The objects below represent
the supported Python surface at this stage; internal helpers are intentionally
not listed here.

---

## API Layers

| Layer | Purpose |
|---|---|
| Pipeline | Execute the workflow and inspect execution summaries. |
| Results | Reconstruct completed runs and select generated artifacts. |
| Science | Wrap loaded artifacts as descriptor, similarity, or projection spaces. |
| Visualization | Resolve visualization inputs and plot from reconstructed results. |

## Execute a Workflow from Python

The workflow engine can be called from Python at three levels of control:

```python
from hddflyzer.pipeline import execute_workflow, run_workflow, run_pipeline

# High-level: returns WorkflowExecution
execution = execute_workflow("aocd")

# Mid-level: returns reconstructed WorkflowRun
run = run_workflow("aocd")

# Low-level: returns list[StageResult]
results = run_pipeline("aocd")
```

## Reconstruct Completed Runs

Completed runs can be reconstructed from `results/<tag>/manifest.json`:

```python
from hddflyzer.results import load_workflow_run

run = load_workflow_run("aocd")

run.workflow_contract
run.outputs(category="chem")
run.outputs(category="dimred")
```

The reconstructed run exposes the workflow contract, current outputs, output
categories, artifact metadata, and semantic artifact selectors.

## Select and Load Artifacts

Artifacts can be selected by semantic kind:

```python
artifact = run.artifact(
    kind="descriptor_table",
    category="chem",
    operation="chem.features",
)

loaded = run.load_artifact(
    kind="descriptor_table",
    category="chem",
    operation="chem.features",
)

loaded.data
loaded.metadata
```

Use `required="path/fragment.csv"` when a kind/category query needs
disambiguation.

Supported artifact kinds include:

| Kind | Description |
|---|---|
| `molecule_registry` | Canonical molecule registry |
| `descriptor_table` | Molecular descriptor matrix |
| `tanimoto_matrix` | Pairwise Tanimoto similarity |
| `projection_coordinates` | PCA / t-SNE / UMAP coordinates |
| `figure` | Generated plot files |
| `metadata` | Operation metadata |
| `workflow_summary` | Human-readable run summary |
| `unknown` | Unclassified artifact |

## Scientific Views

Loaded artifacts can be wrapped as lightweight scientific spaces:

- `DescriptorSpace`
- `SimilaritySpace`
- `ProjectionSpace`

`WorkflowRun` can load scientific views over existing artifacts:

```python
descriptors = run.descriptor_space(
    category="chem",
    operation="chem.features",
)

similarity = run.similarity_space(category="chem")

projection = run.projection_space(
    category="dimred",
    operation="dimred.pca",
)
```

These views expose molecule identifiers when available:

```python
descriptors.molecule_ids
similarity.molecule_ids
projection.molecule_ids
```

These views operate on existing artifacts. They do not recalculate descriptors,
similarity matrices, or projections, and they do not create plots.

The science layer also provides molecule identity helpers, cross-space
alignment, structural metrics, descriptor-projection correlation, neighborhood
preservation, and group comparison for explicitly defined groups.

## Alignment and Science Helpers

```python
from hddflyzer.science import (
    align_spaces,
    compare_descriptor_groups,
    descriptor_projection_correlations,
    projection_neighborhood_preservation,
    similarity_projection_correlation,
    similarity_projection_neighbor_overlap,
)

descriptors_aligned, projection_aligned = align_spaces(descriptors, projection)

global_corr      = similarity_projection_correlation(similarity, projection)
neighbor_overlap = similarity_projection_neighbor_overlap(similarity, projection, k=10)
desc_ranking     = descriptor_projection_correlations(descriptors, projection)
local_preserv    = projection_neighborhood_preservation(similarity, projection, k=10)
group_diff       = compare_descriptor_groups(descriptors, labels="class_label")
```

!!! note "Science helpers operate on existing artifacts"
    These helpers operate on existing artifacts. They do not recalculate
    descriptors, similarity, or projections; do not generate plots; and do not
    perform automatic clustering, enrichment, or chemical interpretation.

## Visualization from Reconstructed Results

```python
from hddflyzer.viz import resolve_viz_inputs, plot_hddf_scatters

inputs = resolve_viz_inputs(
    run,
    kind="descriptor_table",
    category="chem",
    required="features/full/features.csv",
)

plot_hddf_scatters(inputs)
```

`plot_hddf_scatters()` also accepts a loaded descriptor-table artifact directly.

---

## Reference

### Pipeline

::: hddflyzer.pipeline.PipelineContext

---

::: hddflyzer.pipeline.StageResult

---

::: hddflyzer.pipeline.Stage

---

::: hddflyzer.pipeline.WorkflowExecution

---

::: hddflyzer.pipeline.run_pipeline

---

::: hddflyzer.pipeline.run_workflow

---

::: hddflyzer.pipeline.execute_workflow

---

### Results

::: hddflyzer.results.ResultArtifact

---

::: hddflyzer.results.LoadedArtifact

---

::: hddflyzer.results.classify_artifact

---

::: hddflyzer.results.load_artifact

---

::: hddflyzer.results.WorkflowRun

---

::: hddflyzer.results.load_workflow_run

---

### Science

::: hddflyzer.science.DescriptorSpace

---

::: hddflyzer.science.SimilaritySpace

---

::: hddflyzer.science.ProjectionSpace

---

::: hddflyzer.science.to_descriptor_space

---

::: hddflyzer.science.to_similarity_space

---

::: hddflyzer.science.to_projection_space

---

::: hddflyzer.science.shared_molecule_ids

---

::: hddflyzer.science.has_aligned_molecule_ids

---

::: hddflyzer.science.align_spaces

---

::: hddflyzer.science.SpaceMetricResult

---

::: hddflyzer.science.DescriptorProjectionCorrelationResult

---

::: hddflyzer.science.NeighborhoodPreservationResult

---

::: hddflyzer.science.DescriptorGroupComparisonResult

---

::: hddflyzer.science.similarity_projection_correlation

---

::: hddflyzer.science.similarity_projection_neighbor_overlap

---

::: hddflyzer.science.descriptor_projection_correlations

---

::: hddflyzer.science.projection_neighborhood_preservation

---

::: hddflyzer.science.compare_descriptor_groups

---

### Visualization

::: hddflyzer.viz.inputs.VizInputs

---

::: hddflyzer.viz.inputs.resolve_viz_inputs

---

::: hddflyzer.viz.correlations.plot_hddf_scatters
