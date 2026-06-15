# Usage

This page shows how to run HDDFlyzer from the command line and how to interpret
the result folder it creates.

## Installation

After `v0.1.5` is merged, tagged, and published through the manual PyPI
workflow:

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

Prepare a molecule registry and run the standard workflow for the `aocd`
collection:

```bash
hddflyzer data prepare aocd
hddflyzer pipeline run aocd
```

The `aocd` value is the dataset tag. HDDFlyzer uses that tag to locate input
data and to write outputs under `results/aocd/`.

## Input Data

HDDFlyzer starts from a local molecular collection stored as a CSV file. By
default, input collections are placed in:

```text
examples/
```

For a tag named `aocd`, a typical input file is:

```text
examples/valid_metadata_aocd.csv
```

When you run:

```bash
hddflyzer data prepare aocd
```

HDDFlyzer searches the input directory for a CSV file whose filename contains
the tag `aocd`. You can also pass an explicit CSV path:

```bash
hddflyzer data prepare aocd path/to/input.csv
```

The input table must contain a SMILES column. Columns whose names contain
`smiles` or `canonical_smiles` are detected automatically. A compound identifier
column is optional; HDDFlyzer detects `identifier`, `id`, `compound_id`, or
`molecule_id` when present, and otherwise creates identifiers automatically.

To use a different input directory, set `HDDFLYZER_DATA_DIR`:

```powershell
$env:HDDFLYZER_DATA_DIR = "C:\path\to\csvs"
hddflyzer data prepare aocd
```

```bash
HDDFLYZER_DATA_DIR=/path/to/csvs hddflyzer data prepare aocd
```

## Running the Workflow

The canonical workflow follows this shape:

```text
compound collection
  -> registry
  -> descriptors and similarity
  -> dimensionality reduction
  -> visualization
  -> manifest/results
```

Run the full workflow with:

```bash
hddflyzer pipeline run aocd
```

You can also run selected stages:

```bash
hddflyzer pipeline run aocd --skip-dimred
hddflyzer pipeline run aocd --stages chem.features,chem.pruning
```

## Understanding the Result Folder

All workflow outputs are written under:

```text
results/<tag>/
```

For `aocd`, this becomes:

```text
results/aocd/
```

The preparation step creates the canonical molecule registry:

```text
results/aocd/registry/molecules.csv
```

This registry records stable identifiers, raw and canonical SMILES, validity
flags, source provenance, and row-level input metadata. Downstream descriptor,
similarity, dimensionality-reduction, and visualization steps use this registry
as the shared molecule base.

Important result files include:

- `manifest.json`
- `workflow_summary.md`
- registry, chemistry, feature, dimensionality-reduction, and figure outputs
- operation metadata

Representative outputs include:

- canonical molecule registry;
- descriptor tables;
- Tanimoto similarity matrix;
- PCA, t-SNE, and UMAP projection coordinates;
- figures;
- result manifest and workflow summary.

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

## Common CLI Commands

```bash
# Data preparation
hddflyzer data prepare aocd
hddflyzer data prepare aocd path/to/input.csv

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

## Current Scope and Boundaries

HDDFlyzer is not currently:

- a docking workflow;
- a web dashboard;
- a cloud or server workflow;
- an automatic clustering system;
- an enrichment workflow;
- an automatic chemical interpretation engine;
- a published PyPI package or public release until the `v0.1.5` tag and manual
  publishing workflow have completed successfully.

## Safety Notes

!!! warning "Security defaults"
    - Pickle loading is **blocked by default** in all public loaders. Use
      `allow_pickle=True` only with trusted local files.
    - Run tags reject empty values, path traversal, absolute paths, and path
      separators.
    - Reconstructed artifacts must remain inside `results/<tag>/`.
    - `update_manifest()` rejects files outside the run directory.
