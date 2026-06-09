# HDDFlyzer

<section class="hdf-hero">
  <div class="hdf-hero__content">
    <p class="hdf-eyebrow">Molecular descriptor-space workflows</p>
    <div class="hdf-brand" aria-label="HDDFlyzer">
      <span class="hdf-dotmark" aria-hidden="true">
        <span></span><span></span><span></span>
        <span></span><span></span><span></span>
        <span></span><span></span><span></span>
      </span>
      <span class="hdf-wordmark">HDDFlyzer</span>
    </div>
    <p class="hdf-subtitle">Traceable, reproducible molecular descriptor-space workflows for CLI and Python.</p>

    <div class="hdf-actions">
      <a class="md-button md-button--primary" href="usage/#installation">Install</a>
      <a class="md-button" href="usage/#quick-start">Quick start</a>
      <a class="md-button" href="api/">API Reference</a>
      <a class="md-button" href="changelog/">Changelog</a>
    </div>

    <div class="hdf-badges">
      <a href="https://github.com/NanoBiostructuresRG/hddflyzer/actions/workflows/ci.yml"><img src="https://github.com/NanoBiostructuresRG/hddflyzer/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
      <a href="https://pypi.org/project/hddflyzer/"><img src="https://img.shields.io/pypi/v/hddflyzer.svg" alt="PyPI"></a>
      <a href="https://pypi.org/project/hddflyzer/"><img src="https://img.shields.io/pypi/pyversions/hddflyzer.svg" alt="Python versions"></a>
      <a href="https://github.com/NanoBiostructuresRG/hddflyzer/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-LGPL_v3-blue.svg" alt="License: LGPL v3"></a>
    </div>
  </div>
</section>


## Why It Exists

Exploratory cheminformatics workflows often produce many related outputs:
molecule registries, descriptors, similarity matrices, dimensionality-reduction
coordinates, figures, and metadata. Without a clear result contract, those files
can become hard to interpret or reuse.

HDDFlyzer keeps each collection under a dataset-specific folder:

```text
results/<tag>/
```

Each run records metadata in `manifest.json`, writes a concise
`workflow_summary.md`, and keeps a structured index of current outputs through
`workflow_contract`, `current_outputs`, and `output_categories`.

!!! tip "Design principle"
    HDDFlyzer emphasizes **traceability and reproducibility**, not automatic
    chemical interpretation. Every run is self-describing and reconstructible
    from its result folder.

## What You Provide and Receive

| You provide | HDDFlyzer returns |
|---|---|
| A local molecular collection (SDF, CSV, or SMILES file). | A structured `results/<tag>/` folder with all outputs. |
| A dataset tag and workflow parameters. | Descriptor tables, similarity matrices, and projection coordinates. |
| Optional group definitions for comparison. | Figures, metadata, a manifest, and a workflow summary. |

## Workflow Overview

<section class="hdf-panel">
  <div class="hdf-grid hdf-grid--three">
    <article class="hdf-card">
      <span class="hdf-card__icon">REG</span>
      <h3>Registry</h3>
      <p>Build a canonical molecule registry from local SDF, CSV, or SMILES collections.</p>
    </article>

    <article class="hdf-card">
      <span class="hdf-card__icon">DSC</span>
      <h3>Descriptors &amp; similarity</h3>
      <p>Compute molecular descriptors and Tanimoto similarity matrices with full provenance.</p>
    </article>

    <article class="hdf-card">
      <span class="hdf-card__icon">VIZ</span>
      <h3>Projection &amp; visualization</h3>
      <p>Reduce dimensionality with PCA, t-SNE, and UMAP, then generate publication-ready figures.</p>
    </article>
  </div>
</section>

## Canonical Workflow

The current workflow follows this shape:

```text
compound collection
  -> registry
  -> descriptors and similarity
  -> dimensionality reduction
  -> visualization
  -> manifest/results
  -> reconstruction and scientific views
```

Representative outputs include:

- canonical molecule registry;
- descriptor tables;
- Tanimoto similarity matrix;
- PCA, t-SNE, and UMAP projection coordinates;
- figures;
- operation metadata;
- result manifest and workflow summary.

## Two Entry Points

HDDFlyzer is designed around two complementary entry points:

- **CLI:** run workflows with the `hddflyzer` command.
- **Python API:** execute, reconstruct, load, and analyze completed runs from Python.

Both entry points point to the same traceable workflow engine.

## Reconstructible Results

Completed runs can be reconstructed from Python:

```python
from hddflyzer.results import load_workflow_run

run = load_workflow_run("aocd")
run.workflow_contract
run.outputs(category="dimred")
```

Result artifacts can be selected by semantic kind and loaded into Python:

```python
loaded = run.load_artifact(
    kind="descriptor_table",
    category="chem",
    operation="chem.features",
)

loaded.data
loaded.metadata
```

## Scientific Views

Loaded artifacts can be wrapped as lightweight scientific spaces:

- `DescriptorSpace`
- `SimilaritySpace`
- `ProjectionSpace`

These views operate on existing artifacts. They do not recalculate descriptors,
similarity matrices, or projections, and they do not create plots.

The science layer also provides molecule identity helpers, cross-space
alignment, structural metrics, descriptor-projection correlation, neighborhood
preservation, and group comparison for explicitly defined groups.

## Boundaries

HDDFlyzer is not currently:

- a docking workflow;
- a web dashboard;
- a cloud or server workflow;
- an automatic clustering system;
- an enrichment workflow;
- an automatic chemical interpretation engine;
- a published PyPI package or public release, unless verified separately.

## Documentation

- [Usage](usage.md) covers practical CLI and Python examples.
- [API Reference](api.md) documents the current public API boundary.
- [Changelog](changelog.md) lists pre-release change history.

## Citation

```text
Contreras-Torres, F. F. (2026). HDDFlyzer: High-Dimensional Descriptor-based Feature Space Analyzer. https://github.com/NanoBiostructuresRG/hddflyzer
```

## License

This project is licensed under the terms of the
[GNU Lesser General Public License v3.0 or later](https://github.com/NanoBiostructuresRG/hddflyzer/blob/main/LICENSE).
SPDX identifier: `LGPL-3.0-or-later`.
