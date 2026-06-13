# Changelog

All notable changes to HDDFlyzer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.4] - 2026-06-13

### Added

- Added a GitHub Actions documentation workflow for MkDocs / GitHub Pages
  validation.

### Changed

- Updated the internal HDDFlyzer version to `0.1.4`.
- Separated documentation dependencies through the `[docs]` optional dependency
  group.
- Added a preventive MkDocs version bound with `mkdocs>=1.6,<2.0`.

### Documentation

- Validated the documentation site with `mkdocs build --strict`.
- Prepared the `v0.1.4` cycle for documentation web / GitHub Pages readiness
  without publishing PyPI in this cycle.

---

## [0.1.3] - 2026-06-11

### Added

- Added minimal GitHub Actions CI for HDDFlyzer with local validation steps for
  tests, import smoke, CLI smoke, and package build.

### Changed

- Updated the internal HDDFlyzer version to `0.1.3`.

---

## [0.1.2] - 2026-06-10

### Added

- Added global CLI version reporting with `hddflyzer --version`.
- Added the short CLI version flag `hddflyzer -V`.

### Changed

- Updated the internal HDDFlyzer version to `0.1.2`.

### Packaging

- Validated local editable installation with `python -m pip install -e .`.
- Validated local wheel and sdist builds with `python -m build`.
- Confirmed that `hddflyzer/config/descriptor_config.json` is included in both
  the wheel and sdist artifacts.
- Kept PyPI publication deferred to a later release cycle.

---

## [0.1.1] - 2026-06-10

### Changed

- Reworked the project documentation site around clearer page identities:
  `Usage` now focuses on running HDDFlyzer and understanding generated outputs,
  while `API Overview and Reference` separates guided Python examples from the
  auto-generated API reference.
- Refined the Home page narrative to better describe the motivation for
  HDDFlyzer, the dataset-first result model, descriptor-space provenance, and
  representative molecular features.
- Updated the documentation visual identity with a teal/deep-petroleum palette,
  warmer admonitions, a single-color header, balanced dark-mode contrast, and
  more consistent footer spacing.
- Improved the Changelog page rendering and presentation through MkDocs snippet
  support and focused changelog styling.

### Added

- Added Home-page examples for descriptor-space outputs using documentation
  assets from `docs/assets/`.
- Added footer social links for the HDDFlyzer GitHub repository and the
  NanoBiostructures Research Group website.
- Added a pre-stable documentation notice that states HDDFlyzer is currently in
  Alpha-stage development.

### Fixed

- Fixed the documentation Changelog page so it renders entries from
  `CHANGELOG.md` in the project documentation site.
- Corrected the API navigation label from `API Overview` to
  `API Reference`, then refined the page title to
  `API Overview and Reference` to match its combined guide/reference role.

---

## [0.1.0] - 2026-06-08

### Added

- Added a CLI-first workflow for local cheminformatics analysis with `hddflyzer`
  as the user-facing command-line entry point.
- Added a dataset-first results layout under `results/<tag>/` so each molecular
  collection keeps its generated registry, annotations, chemistry outputs,
  dimensionality-reduction outputs, figures, metadata, and summaries together.
- Added molecule-registry preparation from local input collections.
- Added natural-product class annotation outputs.
- Added Tanimoto similarity workflows, including similarity matrix output,
  molecule ID alignment files, sampling outputs, and metadata.
- Added molecular descriptor feature generation, feature curation, correlation
  analysis, and correlation-based pruning outputs.
- Added dimensionality-reduction workflows for PCA, t-SNE, and UMAP across
  descriptor, Tanimoto, and pruned-feature spaces where supported.
- Added visualization commands for natural-product classes, similarity outputs,
  descriptor correlations, PCA, t-SNE, and UMAP projections.
- Added `manifest.json` and `workflow_summary.md` as run-level indexes for
  generated outputs, operation metadata, current outputs, output categories, and
  workflow-contract reconstruction.
- Added the programmatic pipeline contracts `run_pipeline()`, `run_workflow()`,
  and `execute_workflow()`.
- Added `WorkflowExecution` for programmatic workflow execution summaries,
  including stage results, failure tracking, and optional reconstructed run
  access.
- Added the results API with `WorkflowRun`, `ResultArtifact`, `LoadedArtifact`,
  `load_workflow_run()`, and `load_artifact()`.
- Added semantic artifact classification for outputs such as molecule
  registries, descriptor tables, Tanimoto matrices, projection coordinates,
  figures, metadata files, and workflow summaries.
- Added artifact selection and loading helpers through `run.artifacts(...)`,
  `run.artifact(...)`, and `run.load_artifact(...)`.
- Added the optional `required` artifact filter to disambiguate matching outputs
  by path fragment.
- Added visualization input resolution through `VizInputs` and
  `resolve_viz_inputs()`.
- Added support for using reconstructed result objects as visualization inputs,
  including `plot_hddf_scatters(LoadedArtifact)` for loaded descriptor-table
  artifacts.
- Added scientific result views over existing outputs:
  `DescriptorSpace`, `SimilaritySpace`, and `ProjectionSpace`.
- Added `WorkflowRun` scientific accessors:
  `run.descriptor_space(...)`, `run.similarity_space(...)`, and
  `run.projection_space(...)`.
- Added molecular-identity utilities for comparing and aligning scientific
  spaces, including `shared_molecule_ids(...)`, `has_aligned_molecule_ids(...)`,
  and `align_spaces(...)`.
- Added scientific analysis helpers over reconstructed artifacts, including:
  `similarity_projection_correlation(...)`,
  `similarity_projection_neighbor_overlap(...)`,
  `descriptor_projection_correlations(...)`,
  `projection_neighborhood_preservation(...)`, and
  `compare_descriptor_groups(...)`.

### Changed

- Consolidated the CLI around the programmatic workflow engine so
  `hddflyzer pipeline run` uses `execute_workflow()` while preserving existing
  exit-code behavior.
- Preserved distinct execution contracts:
  - `run_pipeline(...)` returns `list[StageResult]`;
  - `run_workflow(...)` returns a reconstructed `WorkflowRun`;
  - `execute_workflow(...)` returns `WorkflowExecution`.
- Extended the run manifest with a workflow-contract block that describes the
  canonical `registry -> chem -> dimred -> viz -> metadata/results` workflow.
- Added grouped output categories while preserving the existing flat
  `current_outputs` list for compatibility.
- Refined the README to present HDDFlyzer as a local, traceable molecular
  descriptor-space workflow tool rather than a collection of unrelated scripts.
- Clarified that “HDDFlyzer” is a project name/brand: `HDD` refers to
  high-dimensional descriptors, while `Flyzer` is a stylized name for exploring
  and analyzing molecular descriptor spaces.

### Security

- Blocked pickle loading by default in public artifact and dataframe loaders.
- Required explicit `allow_pickle=True` for trusted local pickle files.
- Added run-tag validation to reject empty tags, absolute paths, path traversal,
  and path separators.
- Added containment checks so reconstructed artifacts must resolve inside their
  corresponding `results/<tag>/` directory.
- Added manifest-update checks that reject registered output files outside the
  expected run directory.
- Made matrix loading use non-pickle NumPy loading by default.

### Documentation

- Documented the traceable results model based on `results/<tag>/`,
  `manifest.json`, `workflow_summary.md`, artifact semantics, and reconstructed
  runs.
- Documented scientific result views and analysis helpers as operations over
  already generated artifacts.
- Documented that the scientific API does not recalculate descriptors,
  similarities, or projections, and does not perform automatic clustering,
  enrichment, or chemical interpretation.
- Documented local safety expectations for pickle files, path validation, and
  untrusted manifests.

### Tests

- Added and expanded tests for CLI routing, configuration, IO, manifest writing,
  pipeline execution contracts, results reconstruction, artifact loading,
  visualization inputs, public API exports, scientific spaces, alignment, and
  scientific metrics.
- Added tests for defensive path handling, tag validation, artifact containment,
  and pickle-blocking behavior.

### Notes

- HDDFlyzer is not yet a complete drug-discovery platform, docking workflow,
  automatic chemical-interpretation system, or automatic enrichment/clustering
  tool.
- Current workflows focus on local molecular descriptor spaces, similarity
  matrices, dimensionality reduction, visualization, result reconstruction, and
  lightweight scientific analysis over existing outputs.
- Future conformer-generation and 3D descriptor workflows are planned but are
  not part of the current workflow contract.

---

[0.1.4]: https://github.com/NanoBiostructuresRG/hddflyzer/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/NanoBiostructuresRG/hddflyzer/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/NanoBiostructuresRG/hddflyzer/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/NanoBiostructuresRG/hddflyzer/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/NanoBiostructuresRG/hddflyzer/releases/tag/v0.1.0
