# ROADMAP

This document is the internal development roadmap for **HDDFlyzer**.

It was created by moving planning material out of the old changelog. The
changelog should record notable user-facing changes. This roadmap records
project direction, completed checkpoints, active documentation/hardening work,
and future readiness steps.

HDDFlyzer is currently treated as pre-stable research software. The
current priority is not adding more scientific features, but consolidating the
project identity, documentation, safety posture, installability, and external
readiness.

## Status Legend

| Marker | Meaning |
| --- | --- |
| ✅ Closed | Implemented, reviewed, or historically closed as a checkpoint. |
| 🟡 Active | Current documentation, hardening, or cleanup work. |
| ⬜ Planned | Future work, not yet started. |
| ⏸ Deferred | Intentionally postponed to avoid scope creep. |
| 🚫 Omitted | Considered historically, then skipped because it was unnecessary or superseded. |

## Strategic Direction

HDDFlyzer should mature into a traceable local cheminformatics workflow tool for
molecular descriptor-space analysis.

The project should emphasize:

- reproducible local workflows;
- dataset-specific results under `results/<tag>/`;
- traceable outputs through `manifest.json` and `workflow_summary.md`;
- reconstruction of completed runs from Python;
- semantic artifact selection and loading;
- scientific views over already generated artifacts;
- conservative safety defaults for local file loading;
- a clear CLI/API boundary.

The project should not currently present itself as:

- a complete drug-discovery platform;
- a docking workflow;
- an automatic chemical interpretation engine;
- an automatic clustering or enrichment tool;
- a dashboard or web application;
- a collection of unrelated scripts.

## Completed Foundation

### P1 — Functional Baseline ✅

**Goal:** Confirm that the project runs, tests pass, and the module structure is
understandable.

**Outcome:** The main workflow executes, baseline tests exist, and the broad
responsibilities of the package modules are understood.

### P2 — Internal Responsibilities ✅

**Goal:** Separate the responsibilities of CLI, I/O, configuration, scientific
logic, helpers, and pipeline code.

**Outcome:** The project direction moved away from mixed procedural scripts and
toward clearer layers with smaller responsibility boundaries.

### P3 — Execution Traceability ✅

**Goal:** Ensure each run leaves evidence of what was produced, where it was
written, and under which workflow contract.

**Outcome:** Completed runs now include dataset-level metadata such as
`manifest.json`, `workflow_summary.md`, `current_outputs`, `output_categories`,
operation metadata, and a workflow contract.

| Checkpoint | Status | Historical outcome |
| --- | --- | --- |
| P3.1 | ✅ Closed | Added a canonical workflow contract to the manifest layer. |
| P3.2 | ✅ Closed | Improved run-level traceability through operation metadata. |
| P3.3 | ✅ Closed | Strengthened the human-readable workflow summary. |
| P3.4 | ✅ Closed | Added/validated smoke-style coverage around workflow execution behavior. |
| P3.5 | ✅ Closed | Added `output_categories` while preserving `current_outputs` compatibility. |
| P3.6 | ✅ Closed | Reviewed metadata/output completeness for the traceability layer. |

## Results and Artifact Reconstruction

### P4 — Run Reconstruction ✅

**Goal:** Convert generated result folders into completed runs that can be
queried from Python.

**Outcome:** `WorkflowRun` and `load_workflow_run()` make it possible to
reconstruct a run from `results/<tag>/manifest.json`.

### P4.1 — WorkflowRun / Run Reconstruction ✅

**Outcome:** `WorkflowRun` and `load_workflow_run()` were created.

### P4.2 — Results to VizInputs ✅

**Outcome:** `resolve_viz_inputs()` and `VizInputs` were added, connecting
`WorkflowRun` with visualization inputs derived from result metadata.

### P4.3 — VizInputs to Real Plotting ✅

**Outcome:** `plot_hddf_scatters()` can consume `VizInputs` while preserving the
existing `plot_hddf_scatters("aocd")` behavior.

### P4 Summary — Traceable Reconstruction ✅

**Outcome:** The chain below became possible:

```text
manifest.json -> WorkflowRun -> resolve_viz_inputs() -> VizInputs -> plot_hddf_scatters()
```

### P5 — Artifact Semantics ✅

**Goal:** Stop treating outputs only as paths and start classifying them by
scientific meaning.

**Outcome:** Result artifacts can now be classified with semantic kinds such as
`descriptor_table`, `tanimoto_matrix`, and `projection_coordinates`.

### P5.1 — Result Artifact Semantics ✅

**Outcome:** `ResultArtifact`, `classify_artifact()`, and
`WorkflowRun.artifacts(...)` were created.

### P5.2 — Semantic Visualization Input Resolution ✅

**Outcome:** `resolve_viz_inputs()` can use artifact kinds instead of relying
only on path fragments.

### P5 Summary — Outputs as Scientific Artifacts ✅

**Outcome:** HDDFlyzer moved from path-based output handling toward semantic
artifact handling.

### P6 — Loadable Artifacts ✅

**Goal:** Convert semantic artifacts into loaded and validated Python objects.

**Outcome:** `load_artifact()` returns `LoadedArtifact(data, metadata, artifact)`
with minimum contracts for supported artifact types.

### P6.1 — `load_artifact()` ✅

**Outcome:** `LoadedArtifact` and `load_artifact()` were created.

### P6.2 — `tanimoto_matrix` Contract ✅

**Outcome:** Tanimoto matrices are loaded from `.npz` and validated for square
shape and aligned molecule IDs.

### P6.3 — Minimum Loaded Data Contracts ✅

**Outcome:** Contracts exist for descriptor tables, projection coordinates,
molecule registries, and Tanimoto matrices.

### P6.4 — Real Consumer Uses LoadedArtifact ✅

**Outcome:** `plot_hddf_scatters()` accepts `LoadedArtifact` in addition to a
string tag and `VizInputs`.

### P6.5 — Minimum Public API ✅

**Outcome:** Public imports from `hddflyzer.results` and `hddflyzer.viz` are
protected by explicit tests.

### P6 Summary — Loadable Artifact Chain ✅

**Outcome:** The result chain became:

```text
WorkflowRun -> ResultArtifact -> load_artifact() -> LoadedArtifact -> plotting/API use
```

### P7 — Results API ✅

**Goal:** Make the Python result API comfortable and safe enough to use without
manual path indexing.

**Outcome:** Users can select and load artifacts through `run.artifact(...)` and
`run.load_artifact(...)`.

| Checkpoint | Status | Historical outcome |
| --- | --- | --- |
| P7.1 | ✅ Closed | Added `run.artifact(...)` and `run.load_artifact(...)` for single-artifact selection and loading. |
| P7.2 | ✅ Closed | Diagnosed the existing programmatic workflow entrypoint and confirmed where orchestration lived. |
| P7.3 | ✅ Closed | Stabilized `run_workflow(...)` as a programmatic route that reconstructs and returns `WorkflowRun`. |
| P7.4 | ✅ Closed | Evaluated CLI migration to `run_workflow(...)`; did not force it because the CLI still needed stage-result information. |

## Programmatic Execution and CLI Boundary

### P8 — WorkflowExecution / Programmatic Execution Contract ✅

**Goal:** Make workflows executable from Python, not only from the CLI.

**Outcome:** Programmatic workflow execution was formalized with a contract that
preserves both reconstructed run access and stage-result visibility.

| Checkpoint | Status | Outcome |
| --- | --- | --- |
| P8.1 | ✅ Closed | `WorkflowExecution` was created. |
| P8.2 | ✅ Closed | `execute_workflow()` was created. |
| P8.3 | ✅ Closed | `run_pipeline()`, `run_workflow()`, and `execute_workflow()` were confirmed as distinct contracts. |
| P8.4 | ✅ Closed | The CLI was migrated to `execute_workflow()` while preserving exit codes and `--continue-on-error`. |

### P9 — Thin CLI / Interface-Engine Separation ✅

**Goal:** Turn the CLI into a façade over the programmatic engine.

**Outcome:** The CLI routes through the execution layer clearly. The final review
found no major remaining extraction required before closing the architecture
phase.

| Checkpoint | Status | Historical outcome |
| --- | --- | --- |
| P9.1 | ✅ Closed | Confirmed the CLI uses `execution.ok` / execution-level state rather than manually reinterpreting stage results. |
| P9.2 | ✅ Closed | Verified the `pipeline run` path already met the thin-facade contract; no implementation changes were required. |
| P9.3 | 🚫 Omitted | Not needed after P9.2 verification; there was no meaningful extraction left to perform. |
| P9.4 | ✅ Closed | Final validation confirmed equivalent behavior with a sufficiently thin CLI. |

## Security and Result-Loading Hardening

This block records the security-oriented work that supported the transition
from raw result files to reconstructed, loadable artifacts.

| Checkpoint | Status | Historical outcome |
| --- | --- | --- |
| S1 | ✅ Closed | Performed secure-coding audit around local result reconstruction, artifact paths, and file loading. |
| S2 | ✅ Closed | Hardened public artifact loading and path handling: pickle is blocked by default, tags are validated, artifact paths must stay inside `results/<tag>/`, and manifest updates reject out-of-run files. |
| S3 | ✅ Closed | Reviewed or added regression/security coverage for path containment, tag validation, and unsafe-loading boundaries. |
| S4 | ✅ Closed | Documented safety notes in the README, including pickle-loading limits and treatment of unknown files/manifests as untrusted input. |

## Scientific API Layer

The scientific layer was created to operate on existing reconstructed artifacts.
It should not recalculate descriptors, similarity matrices, or projections, and
it should not generate plots or automatic chemical conclusions.

### P10 — Scientific Result Views ✅

**Goal:** Create lightweight scientific views over `LoadedArtifact`.

**Outcome:** `DescriptorSpace`, `SimilaritySpace`, and `ProjectionSpace` were
introduced.

### P11 — WorkflowRun Scientific Accessors ✅

**Goal:** Connect reconstructed runs with scientific views.

**Outcome:** `WorkflowRun` gained accessors such as
`run.descriptor_space(...)`, `run.similarity_space(...)`, and
`run.projection_space(...)`.

### P12 — Scientific Identity Contract ✅

**Goal:** Make molecule identity explicit across spaces.

**Outcome:** Scientific spaces expose molecule IDs when available and can be
checked for shared molecular identity.

### P13 — Cross-Space Alignment ✅

**Goal:** Align descriptor, similarity, and projection spaces by molecule ID
without recalculating outputs.

**Outcome:** Spaces can be aligned and subsetted into comparable forms.

### P14 — Structural Metrics Between Spaces ✅

**Goal:** Compute simple, reproducible, non-interpretive metrics between
already aligned scientific spaces.

**Outcome:** Initial cross-space metrics were added without generating plots,
clusters, or chemical conclusions.

### P15 — Descriptor Projection Interpretation ✅

**Goal:** Identify descriptors that help explain projection geometry using
existing descriptor and projection spaces.

**Outcome:** Descriptor-projection interpretation helpers were added without
automatic chemical interpretation.

### P16 — Local Neighborhood Interpretation ✅

**Goal:** Move from global metrics to per-molecule local neighborhood
diagnostics.

**Outcome:** Local neighborhood preservation can be evaluated per molecule
without plots, clustering, or automatic conclusions.

### P17 — Group / Region Comparison ✅

**Goal:** Compare explicit user-provided groups inside a descriptor space using
existing data.

**Outcome:** Group-level descriptor comparison was added while keeping
interpretation explicit and user-controlled.

### P18 — Scientific Layer Closure ✅

**Goal:** Review the coherence of `hddflyzer.science`.

**Outcome:** Names, exports, result dataclasses, tests, and organization were
reviewed. The layer is considered coherent and minimal. Future splits may be
considered only if the scientific API grows further.

## Documentation and Architecture Review

### D1-D4 — Architectural README Alignment ✅

**Goal:** Bring README documentation into alignment with the emerging CLI,
pipeline, results, visualization, and architecture contracts.

**Outcome:** Earlier README/documentation passes clarified project identity,
module responsibilities, results layout, architecture, and CLI/API relationships
as the internal architecture stabilized.

### D-Science — Scientific API Documentation ✅

**Goal:** Document the scientific API clearly and compactly.

**Outcome:** The README documents scientific spaces, run accessors, molecular
identity helpers, alignment helpers, and science metrics while explicitly
stating that these APIs use existing artifacts and do not recalculate outputs or
generate automatic interpretations.

### R1 — Architecture Closure Review ✅

**Goal:** Review whether the current architecture is coherent enough to close
the architecture phase.

**Outcome:** Architecture was considered coherent. No critical corrections were
required before moving to documentation, hardening, packaging, and external
readiness.

## Active Documentation Cleanup

### D-Cleanup — Public Documentation and Internal Planning Cleanup ✅

**Goal:** Convert accumulated internal documentation into public-facing project
documentation and separate public changelog material from internal planning
history.

**Outcome:** The public-facing documentation and visual cleanup cycle was closed
on `dev/v0.1.1`.

Completed documentation work included:

- Home/docs identity refinements for HDDFlyzer.
- Usage/API page flow cleanup so practical CLI usage and Python API reference
  are separated.
- Changelog documentation wrapper review and rendered changelog validation.
- MkDocs theme, color, footer, admonition, and documentation asset updates.
- Addition of `CITATION.cff`.

## Packaging and External Readiness

### P19 — Packaging ✅

**Goal:** Convert HDDFlyzer into a cleanly installable package.

**Expected outcome:**

- `pyproject.toml` is present and coherent.
- HDDFlyzer packaging and local installability are validated for `v0.1.2`.
- The `hddflyzer` CLI entry point installs correctly.
- Global version flags are supported and covered:
  `hddflyzer --version` and `hddflyzer -V`.
- `pip install -e .`, import smoke checks, CLI smoke checks, and
  `python -m build` were validated locally.
- Wheel and sdist builds include `hddflyzer/config/descriptor_config.json`.

### P20 — Repository Hygiene ⬜

**Goal:** Keep the private GitHub repository ready for eventual public
publication.

**Expected outcome:**

- `.gitignore` excludes local caches, generated temporary files, and unsafe
  local-only artifacts.
- The policy for `results/` is clear.
- The policy for example input data is clear.
- No large, private, or accidental local files are tracked.
- The private `NanoBiostructuresRG/hddflyzer` repository remains clean enough
  to make public when release readiness allows it.

### P21 — Canonical Example Policy ⬜

**Goal:** Decide how the `aocd` example should be represented.

**Expected outcome:**

- Decide whether `results/aocd/` should be kept, reduced, regenerated, moved,
  or documented as local-only.
- Define what example data is small and safe enough to include.
- Document the expected example workflow without making the repository heavy.

### P22 — Local Validation Checklist ⬜

**Goal:** Define the minimum validation required before external publication.

**Expected checks:**

```powershell
python -m pytest tests -q
python -m py_compile hddflyzer
python -c "import hddflyzer; print('OK')"
hddflyzer --help
hddflyzer --version
hddflyzer -V
hddflyzer pipeline --help
```

Packaging validation:

```powershell
python -m pip install -e .
hddflyzer --help
python -m build
```

### P23 — Repository Publication Readiness ⬜

**Goal:** Prepare the existing private GitHub repository for public visibility.

**Expected outcome:**

- README is public-facing.
- CHANGELOG is changelog-shaped.
- ROADMAP is internal-planning-shaped.
- License files are present.
- Citation metadata is present.
- Release links and changelog references match existing GitHub pre-releases and
  releases.
- Example data and generated-result policies are clear before public exposure.

### P24 — Minimal CI ✅

**Goal:** Add basic continuous integration after the repository is public.

**Expected outcome:**

- `v0.1.3` adds minimal GitHub Actions CI for HDDFlyzer.
- Install dependencies on `ubuntu-latest` with Python 3.11.
- Run the test suite.
- Smoke-test package import.
- Smoke-test CLI availability.
- Build wheel and sdist artifacts locally.

Python-version matrices, docs deployment, package publishing, and more complex
CI jobs should be added only after the basic workflow is stable. PyPI remains
deferred.

### P25 — Pre-release and Release Readiness ⬜

**Goal:** Maintain defensible GitHub pre-releases/releases and prepare for
broader release readiness.

**Expected outcome:**

- `v0.1.0` remains documented as the first GitHub pre-release.
- `v0.1.1` is scoped as a documentation-focused GitHub pre-release.
- `v0.1.2` is scoped as a packaging/installability-focused GitHub
  pre-release.
- `v0.1.3` is scoped as a minimal-CI GitHub pre-release.
- Future releases summarize real current capabilities.
- Tags and GitHub Releases are created only after branch validation and review.
- PyPI remains deferred until after installability and minimal CI are reliable.

## Deferred Work

These directions are intentionally postponed until the project identity,
packaging, documentation, and validation are stronger:

- additional scientific metrics;
- automatic clustering;
- automatic enrichment;
- automatic chemical interpretation;
- dashboard or web application work;
- docking workflows;
- conformer generation and 3D descriptor workflows;
- broad CLI expansion;
- large notebook collections;
- PyPI publication before installability is validated.

## Current Recommended Order

1. Close `dev/v0.1.3` after minimal CI metadata and documentation are reviewed.
2. Push `dev/v0.1.3` and open PR to `main`.
3. Merge to `main` if review passes.
4. Tag and publish `v0.1.3` as a GitHub pre-release.
5. Defer PyPI readiness/publication to `v0.1.4` or later, after
   installability and minimal CI are validated.
