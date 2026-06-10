# HDDFlyzer: High-Dimensional Descriptor-based Feature Space Analyzer

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-v0.1.1-blue.svg)](https://pypi.org/project/hddflyzer/)
[![PyPI](https://img.shields.io/pypi/v/hddflyzer.svg)](https://pypi.org/project/hddflyzer/)
[![Python](https://img.shields.io/pypi/pyversions/hddflyzer.svg)](https://pypi.org/project/hddflyzer/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-teal.svg)](https://nanobiostructuresrg.github.io/hddflyzer/)

---

## Description

**HDDFlyzer** is a CLI-first cheminformatics toolkit with a Python API for reproducible, traceable molecular descriptor-space workflows.
It generates descriptor, similarity, projection, and visualization outputs from compound collections, stores them with metadata, and supports later reconstruction, inspection, and analysis of completed runs.

**HDDFlyzer** can be used in two complementary ways:

- as a **command-line tool**, through the `hddflyzer` command;
- as a **Python library**, for workflow execution, run reconstruction, artifact
  inspection, and scientific analysis over existing outputs.

The CLI and Python API are two entry points into the same traceable workflow
engine.

---

## What Is HDDFlyzer?

**HDDFlyzer** organizes a cheminformatics analysis around a local molecule registry
and a dataset-first results folder. A typical workflow starts from a CSV of
compounds, creates a canonical registry, annotates molecules, computes Tanimoto
similarity and molecular descriptor tables, performs feature curation and
correlation-based pruning, projects molecular spaces with PCA, t-SNE, or UMAP,
and writes figures plus metadata-rich outputs.

The design goal is practical reproducibility: every major operation writes
stable files under `results/<tag>/`, and each step keeps enough metadata to
understand how the output was generated.

---

## Why Use It?

Cheminformatics workflows often produce many disconnected files: descriptor
tables, similarity matrices, dimensionality-reduction coordinates, figures, and
metadata. HDDFlyzer keeps these products organized under a dataset-specific
results folder and records how they were generated.

This makes the package useful for exploratory molecular-space analysis,
teaching, workflow prototyping, and reproducible local studies where the
relationship between input molecules, computational steps, and generated
artifacts must remain visible.

---

## Current Scope and Non-Goals

**HDDFlyzer** currently focuses on traceable molecular descriptor-space workflows
for local datasets. It supports registry preparation, natural-product class
annotation, similarity calculation, descriptor engineering, feature curation,
dimensionality reduction, visualization, run reconstruction, and scientific
views over existing outputs.

**HDDFlyzer** is not a docking tool, a complete drug-discovery platform, an
automatic enrichment workflow, or an automatic chemical interpretation system.
Its scientific helpers operate on already generated artifacts; they do not
rerun the pipeline or infer chemical conclusions automatically.

---

## Core Workflow

```text
molecule CSV
  -> canonical registry
  -> NP-class annotation
  -> Tanimoto similarity
  -> descriptor tables
  -> feature curation and pruning
  -> PCA/t-SNE/UMAP projections
  -> figures, metadata, manifest, and workflow summary
  -> reconstructed result views from Python
```

All outputs for a collection are stored under one dataset-specific directory,
for example:

```text
results/aocd/
```

---

## Installation

Python 3.11 or newer is recommended. RDKit is best installed from conda-forge.

```bash
conda create -n hddflyzer_env python=3.11
conda activate hddflyzer_env
conda install -c conda-forge rdkit
pip install -e ".[dev]"
```

UMAP support is included through `umap-learn` in the package dependencies.

---

## Input Data

Place local input collections in:

```text
examples/
```

For example:

```text
examples/valid_metadata_aocd.csv
```

Then prepare the registry with:

```bash
hddflyzer data prepare aocd
```

The default input directory is separate from `hddflyzer/data/`, which contains
package code. To use another input directory, set `HDDFLYZER_DATA_DIR`.

PowerShell:

```powershell
$env:HDDFLYZER_DATA_DIR = "C:\path\to\csvs"
hddflyzer data prepare aocd
```

Unix-like shells:

```bash
HDDFLYZER_DATA_DIR=/path/to/csvs hddflyzer data prepare aocd
```

---

## Quickstart

Prepare a molecule registry:

```bash
hddflyzer data prepare aocd
```

Run the standard workflow:

```bash
hddflyzer pipeline run aocd
```

Run a lighter workflow that stops before t-SNE and UMAP:

```bash
hddflyzer pipeline run aocd --skip-dimred
```

Execute only selected stages:

```bash
hddflyzer pipeline run aocd --stages chem.features,chem.pruning
```

Load the completed run from Python:

```python
from hddflyzer.results import load_workflow_run

run = load_workflow_run("aocd")
run.workflow_contract
run.outputs(category="dimred")
```

---

## CLI Workflow

Use the unified CLI:

```bash
hddflyzer <module> <subcommand> [args]
```

Recommended real-data sequence:

```bash
hddflyzer data prepare aocd
hddflyzer annotate npc aocd
hddflyzer chem tanimoto aocd
hddflyzer chem sample aocd
hddflyzer chem features aocd
hddflyzer chem curate-features aocd
hddflyzer chem reference-features aocd --reference
hddflyzer chem stats base aocd
hddflyzer chem stats hddf aocd
hddflyzer chem pruning aocd
```

Dimensionality-reduction commands include:

```bash
hddflyzer dimred pca aocd
hddflyzer dimred pca-joint aocd dianatdb
hddflyzer dimred tsne features aocd
hddflyzer dimred tsne tanimoto aocd
hddflyzer dimred tsne pruning aocd
hddflyzer dimred umap features aocd
hddflyzer dimred umap tanimoto aocd
hddflyzer dimred umap pruning aocd
```

Projection modes mean:

```text
features  = descriptor space from full feature table
tanimoto  = structural similarity space from Morgan fingerprints
pruning   = descriptor space after correlation-based feature pruning
```

Visualization commands include:

```bash
hddflyzer viz npc aocd
hddflyzer viz similarity tanimoto aocd
hddflyzer viz similarity fingerprints aocd
hddflyzer viz correlations hddf aocd
hddflyzer viz pca analysis aocd
hddflyzer viz tsne features aocd QED 30
hddflyzer viz umap features aocd QED
hddflyzer viz umap tanimoto aocd Pathway
```

Reference-dependent similarity descriptors are intentionally separate from the
intrinsic feature table. They describe molecule-reference pairs, not molecules
alone:

```bash
hddflyzer chem reference-features aocd --reference
```

Optional sampled feature calculation is available when Tanimoto sampling keeps
fewer compounds than the full registry:

```bash
hddflyzer chem features aocd --sampled
```

Optional pickle output is explicit and not the default:

```bash
hddflyzer chem features aocd --save-pkl
```

---

## Programmatic Workflow Execution

The same workflow engine can be called from Python:

```python
from hddflyzer.pipeline import execute_workflow

execution = execute_workflow("aocd")
execution.ok
execution.stage_results
run = execution.run
```

For lower-level control, `run_pipeline()` returns `list[StageResult]`, while
`run_workflow()` returns a reconstructed `WorkflowRun` or propagates an error if
the run cannot be reconstructed.

---

## Main Outputs

The project uses a dataset-first results layout:

```text
results/
  <tag>/
    manifest.json
    workflow_summary.md
    registry/
    annotations/
    chemistry/
    features/
    dimred/
    figures/
```

For `aocd`, representative outputs include:

```text
results/aocd/registry/molecules.csv
results/aocd/annotations/npclassifier/npclassifier.csv
results/aocd/chemistry/tanimoto/tanimoto_matrix.npz
results/aocd/chemistry/tanimoto/tanimoto_ids.csv
results/aocd/features/full/features.csv
results/aocd/features/curated/features_ml.csv
results/aocd/features/pruning/selected_features.txt
results/aocd/dimred/pca/pca_coordinates.csv
results/aocd/dimred/tsne/*/
results/aocd/dimred/umap/*/
results/aocd/manifest.json
results/aocd/workflow_summary.md
```

Each major step writes one interpretable data file plus one metadata file where
possible. Large matrices are stored in compressed `.npz`. Pickle output is
optional, not default.

`manifest.json` is the dataset-level index for a run. It records operation
history, generated files, stale file references, per-operation metadata,
`current_outputs`, `output_categories`, and a `workflow_contract` block that
makes the canonical `registry -> chem -> dimred -> viz -> metadata/results`
workflow explicit. `workflow_summary.md` presents the same contract as a concise
human-readable index.

---

## Reconstructing Previous Runs

The result API reconstructs completed runs from `results/<tag>/manifest.json`.
It does not create new state or recalculate scientific outputs.

```python
from hddflyzer.results import load_workflow_run, load_artifact

run = load_workflow_run("aocd")

artifact = run.artifact(kind="tanimoto_matrix", category="chem")
loaded = load_artifact(artifact)

loaded.artifact
loaded.data
loaded.metadata
```

Loaded result artifacts can also be selected by semantic kind:

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

Supported artifact kinds are:

```text
molecule_registry
descriptor_table
tanimoto_matrix
projection_coordinates
figure
metadata
workflow_summary
unknown
```

`run.artifacts(...)` returns all matching artifacts. `run.artifact(...)` returns
exactly one artifact or raises a clear error when none or multiple candidates
match. `run.load_artifact(...)` selects one artifact and loads it using the same
contracts as `load_artifact(...)`. The optional `required` filter can be used to
disambiguate artifacts by a path fragment, for example
`required="features/full/features.csv"`.

---

## Scientific Result Views

Loaded artifacts can be wrapped as lightweight scientific views:

```text
LoadedArtifact -> DescriptorSpace
LoadedArtifact -> SimilaritySpace
LoadedArtifact -> ProjectionSpace
```

These views do not execute the pipeline and do not recalculate descriptors,
similarity, or projections. They wrap artifacts that already exist in a
reconstructed run, so Python code can work with results in domain language.

```python
from hddflyzer.results import load_workflow_run

run = load_workflow_run("aocd")

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

Each scientific space exposes `molecule_ids` when identifiers are available.
Spaces can be compared or aligned by molecular identity:

```python
from hddflyzer.science import align_spaces, has_aligned_molecule_ids, shared_molecule_ids

shared = shared_molecule_ids(descriptors, projection)
aligned = has_aligned_molecule_ids(descriptors, projection)
descriptors_aligned, projection_aligned = align_spaces(descriptors, projection)
```

`hddflyzer.science` also provides small, pure analysis helpers over existing
spaces:

```python
from hddflyzer.science import (
    compare_descriptor_groups,
    descriptor_projection_correlations,
    projection_neighborhood_preservation,
    similarity_projection_correlation,
    similarity_projection_neighbor_overlap,
)

global_corr = similarity_projection_correlation(similarity, projection)
neighbor_overlap = similarity_projection_neighbor_overlap(similarity, projection, k=10)
descriptor_ranking = descriptor_projection_correlations(descriptors, projection)
local_preservation = projection_neighborhood_preservation(similarity, projection, k=10)
group_differences = compare_descriptor_groups(descriptors, labels="class_label")
```

These helpers use only already reconstructed artifacts. They do not generate
plots, do not perform automatic clustering, do not run enrichment, and do not
make automatic chemical interpretations.

---

## Visualization from Results

Visualization inputs can be resolved from reconstructed results:

```python
from hddflyzer.results import load_workflow_run
from hddflyzer.viz import resolve_viz_inputs, plot_hddf_scatters

run = load_workflow_run("aocd")
inputs = resolve_viz_inputs(
    run,
    kind="descriptor_table",
    category="chem",
    required="features/full/features.csv",
)
plot_hddf_scatters(inputs)
```

`plot_hddf_scatters()` also accepts a loaded descriptor-table artifact:

```python
loaded = run.load_artifact(
    kind="descriptor_table",
    category="chem",
    operation="chem.features",
)
plot_hddf_scatters(loaded)
```

---

## Safety Notes

**HDDFlyzer** is designed as a local Python library and CLI application with
defensive handling around reconstructed results:

- Pickle (`.pkl`) loading is blocked by default in public loaders.
- Pickle can be loaded only with `allow_pickle=True`, and should be used only
  for trusted local files produced in a controlled environment.
- Run tags are validated against empty values, path traversal, absolute paths,
  and path separators.
- Artifacts reconstructed from `manifest.json` must resolve inside the
  corresponding `results/<tag>/` run directory.
- `update_manifest()` rejects files outside `results/<tag>/`.

These checks reduce accidental unsafe file access and unsafe deserialization.
They are not a substitute for treating unknown files and manifests as untrusted
input.

---

## Project Architecture

**HDDFlyzer** separates command-line execution, workflow orchestration, result
reconstruction, scientific views, visualization, and low-level I/O.

```text
hddflyzer/
  cli.py          # command-line entry point
  config/         # global settings and canonical output paths
  pipeline/       # workflow execution contracts and orchestration
  results/        # WorkflowRun, ResultArtifact, LoadedArtifact
  science/        # scientific views, alignment, metrics, interpretation
  io/             # readers, writers, manifests, path safety
  data/           # canonical molecule registry logic
  chem/           # descriptors, similarity, sampling, pruning, references
  dimred/         # PCA, t-SNE, UMAP
  viz/            # plotting layer and visualization input resolution
  utils/          # shared helpers
```

The domain modules keep the scientific logic. The `io/` layer owns canonical
read/write helpers. The `pipeline/` layer exposes `run_pipeline()`,
`run_workflow()`, and `execute_workflow()` for programmatic execution. The
`results/` layer reconstructs completed runs from `manifest.json` through
`WorkflowRun`, `ResultArtifact`, and `LoadedArtifact`.

The central CLI lives in `hddflyzer/cli.py`. Subpackages may still expose
module-level `main()` functions for direct execution, but `hddflyzer` is the
user-facing entry point.

**HDDFlyzer** follows a simple separation of responsibilities:

```text
pure/helper functions  reusable logic for tests, notebooks, and scripts
run() functions        file-based workflow execution and manifest updates
main()/hddflyzer       command-line parsing and process exit behavior
```

---

## Descriptor Families

| Family | Description |
| --- | --- |
| Constitutional | Molecular weight, LogP, H-bonding, rings, rotatable bonds |
| Topological | Chi, Kappa, Balaban, Bertz, graph indices |
| Electronic | Partial charges and VSA descriptors |
| Geometrical | Currently limited; conformer-based 3D descriptors are planned |
| Hybrid | Derived ratios from base descriptors |
| HDDF | QED, lead-likeness, pharmacophore complexity, synthetic accessibility, desirability |
| Reference Similarity | Morgan, FeatMorgan, AtomPair, RDKFingerprint, Torsion, Layered, MACCS; stored under `features/reference/` |
| Reference MCS | Maximum common substructure size, molecule atom counts, Tanimoto, and overlap; stored under `features/reference/` |

---

## Design Principles

**HDDFlyzer** follows these output conventions:

```text
CSV  = canonical tabular results
JSON = metadata, parameters, provenance, compact summaries
NPZ  = compressed numeric matrices
PKL  = optional cache only
PNG  = visualization output
TXT  = only for small human-facing operational files such as selected_features.txt
```

The goal is low redundancy, reproducible outputs, stable paths, and readable
data products suitable for real dataset analysis.

---

## Future Directions

**HDDFlyzer** currently focuses on 1D/2D descriptor tables, similarity matrices,
dimensionality reduction, visualization, and reconstructed result analysis.
Future versions may add explicit conformer-generation and 3D descriptor
workflows, including reproducible conformer parameters, energy-minimization
metadata, and clear separation between 2D descriptor tables and conformer-based
3D descriptor tables.

---

## Development

Run tests:

```bash
pytest -q
```

The test suite currently covers core utilities, chemistry calculations,
dimensionality-reduction helpers, configuration, persistence, orchestration, CLI
routing, reconstructed results, scientific views, and visualization APIs.

---

## Status

**HDDFlyzer** is pre-stable research software. The command structure is stabilizing, but output
contracts may still evolve as the real-data pipeline matures.

---

## Contributing

Contributions are welcome. Please open an issue before submitting a pull request.
Follow the existing code style: NumPy-style docstrings, type hints, and SPDX license
headers in all source files.

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.
Please also read our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Citation

If you use **HDDFlyzer** in your research, please cite it using the metadata in
[CITATION.cff](CITATION.cff) or the format below:

```text
Contreras-Torres, F. F. (2026). HDDFlyzer: High-Dimensional Descriptor-based Feature Space Analyzer. https://github.com/NanoBiostructuresRG/hddflyzer
```

---

## Author

Developed by **Flavio F. Contreras-Torres** (Tecnologico de Monterrey)
Monterrey, Mexico - June 2026

---

## License

LGPL-3.0-or-later. See `LICENSE`, `COPYING`, and `COPYING.LESSER`.
