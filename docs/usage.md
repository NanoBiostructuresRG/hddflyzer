# Usage

This page shows the practical HDDFlyzer workflow from the command line and from
Python. HDDFlyzer is currently local pre-release research software.

## Installation

```bash
pip install hddflyzer
```

> Python 3.11 or newer is required. RDKit is best installed from conda-forge
> before installing HDDFlyzer.

```bash
conda create -n hddflyzer_env python=3.11
conda activate hddflyzer_env
conda install -c conda-forge rdkit
pip install hddflyzer
```

UMAP support uses `umap-learn` when available and is listed as an optional
dependency.

## Quick Start

=== "CLI"

    ```bash
    # Prepare a molecule registry
    hddflyzer data prepare aocd

    # Run the canonical pipeline
    hddflyzer pipeline run aocd
    ```

=== "Python"

    ```python
    from hddflyzer.pipeline import execute_workflow

    execution = execute_workflow("aocd")

    if execution.ok:
        run = execution.run
    ```

=== "Reconstruct"

    ```python
    from hddflyzer.results import load_workflow_run

    run = load_workflow_run("aocd")
    run.workflow_contract
    run.outputs(category="dimred")
    ```

=== "Science"

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

## Input and Results

HDDFlyzer works with a collection tag, for example `aocd`.

Input CSV files are usually placed under `examples/`. All workflow outputs are
written under:

```text
results/<tag>/
```

Important result files include:

- `manifest.json`
- `workflow_summary.md`
- registry, chemistry, feature, dimensionality-reduction, and figure outputs
- operation metadata

## Command-Line Interface

The unified CLI entry point is:

```bash
hddflyzer <module> <subcommand> [args]
```

Common commands:

```bash
# Data preparation
hddflyzer data prepare aocd

# Pipeline control
hddflyzer pipeline run aocd
hddflyzer pipeline run aocd --skip-dimred
hddflyzer pipeline run aocd --stages chem.features,chem.pruning

# Module-level commands
hddflyzer chem tanimoto aocd
hddflyzer chem features aocd
hddflyzer chem pruning aocd
hddflyzer dimred pca aocd
hddflyzer viz pca analysis aocd
```

## Workflow Modules

<div class="hdf-grid hdf-grid--three">
  <article class="hdf-card hdf-card--compact">
    <h3>Data</h3>
    <p><code>hddflyzer data prepare</code> builds the canonical molecule registry from a local collection.</p>
  </article>

  <article class="hdf-card hdf-card--compact">
    <h3>Chemistry</h3>
    <p><code>hddflyzer chem</code> computes descriptors, Tanimoto similarity, and feature pruning.</p>
  </article>

  <article class="hdf-card hdf-card--compact">
    <h3>Dim. reduction &amp; viz</h3>
    <p><code>hddflyzer dimred</code> and <code>hddflyzer viz</code> run PCA, t-SNE, UMAP, and generate figures.</p>
  </article>
</div>

<table class="hdf-pipeline-table">
  <thead>
    <tr>
      <th>Module</th>
      <th>Subcommand</th>
      <th>Description</th>
      <th>Output category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>data</code></td>
      <td><code>prepare</code></td>
      <td>Build canonical molecule registry</td>
      <td><code>registry</code></td>
    </tr>
    <tr>
      <td><code>chem</code></td>
      <td><code>features</code></td>
      <td>Compute molecular descriptors</td>
      <td><code>chem</code></td>
    </tr>
    <tr>
      <td><code>chem</code></td>
      <td><code>tanimoto</code></td>
      <td>Compute Tanimoto similarity matrix</td>
      <td><code>chem</code></td>
    </tr>
    <tr>
      <td><code>chem</code></td>
      <td><code>pruning</code></td>
      <td>Prune low-variance and correlated features</td>
      <td><code>chem</code></td>
    </tr>
    <tr>
      <td><code>dimred</code></td>
      <td><code>pca</code></td>
      <td>PCA projection</td>
      <td><code>dimred</code></td>
    </tr>
    <tr>
      <td><code>dimred</code></td>
      <td><code>tsne</code></td>
      <td>t-SNE projection</td>
      <td><code>dimred</code></td>
    </tr>
    <tr>
      <td><code>dimred</code></td>
      <td><code>umap</code></td>
      <td>UMAP projection</td>
      <td><code>dimred</code></td>
    </tr>
    <tr>
      <td><code>viz</code></td>
      <td><code>pca analysis</code></td>
      <td>Generate PCA figures</td>
      <td><code>figures</code></td>
    </tr>
  </tbody>
</table>

## Python Execution API

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

## Reconstruct a Completed Run

Completed runs can be reconstructed from `results/<tag>/manifest.json`:

```python
from hddflyzer.results import load_workflow_run

run = load_workflow_run("aocd")

run.workflow_contract
run.outputs(category="chem")
run.outputs(category="dimred")
```

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

## Scientific Spaces

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

!!! note "Science helpers do not recalculate"
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

## Safety Notes

!!! warning "Security defaults"
    - Pickle loading is **blocked by default** in all public loaders. Use
      `allow_pickle=True` only with trusted local files.
    - Run tags reject empty values, path traversal, absolute paths, and path
      separators.
    - Reconstructed artifacts must remain inside `results/<tag>/`.
    - `update_manifest()` rejects files outside the run directory.
