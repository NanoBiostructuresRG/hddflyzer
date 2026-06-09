# SPDX-License-Identifier: LGPL-3.0-or-later

"""
HDDFlyzer CLI
=============

Usage
-----
    hddflyzer <module> <subcommand> [args...]

Modules
-------
    data     : Local molecule registry
    annotate : External annotation layers
    chem     : Chemistry pipeline
    dimred   : Dimensionality reduction
    viz      : Visualization
    pipeline : End-to-end orchestration

Examples
--------
    hddflyzer data prepare       aocd
    hddflyzer annotate npc       aocd

    hddflyzer chem npc          aocd
    hddflyzer chem tanimoto     aocd
    hddflyzer chem sampling     aocd k_per_stratum=300
    hddflyzer chem features     aocd
    hddflyzer chem curate-features aocd
    hddflyzer chem reference-features aocd --reference
    hddflyzer chem stats        base aocd
    hddflyzer chem stats        hddf aocd
    hddflyzer chem pruning      aocd
    hddflyzer chem pruning      aocd --threshold 0.85

    hddflyzer dimred pca        aocd
    hddflyzer dimred pca-joint  aocd dianatdb
    hddflyzer dimred tsne       tanimoto aocd
    hddflyzer dimred tsne       features aocd 5,30,80
    hddflyzer dimred tsne       pruning  aocd
    hddflyzer dimred umap       features aocd 15,30,50 0.1,0.5
    hddflyzer dimred umap       tanimoto aocd
    hddflyzer dimred umap       pruning  aocd

    hddflyzer pipeline run      aocd
    hddflyzer pipeline run      aocd --skip-dimred

    hddflyzer viz distributions aocd,dianatdb "MW,TPSA"
    hddflyzer viz correlations  hddf  aocd
    hddflyzer viz similarity    tanimoto aocd
    hddflyzer viz similarity    fingerprints aocd
    hddflyzer viz pca           analysis    aocd
    hddflyzer viz pca           collections aocd dianatdb
    hddflyzer viz tsne          tanimoto aocd
    hddflyzer viz tsne          features     aocd QED 30
    hddflyzer viz tsne          pruning      aocd QED 30
    hddflyzer viz umap          features aocd QED
    hddflyzer viz umap          tanimoto aocd Pathway
    hddflyzer viz umap          pruning  aocd
    hddflyzer viz npc           aocd
"""

import sys


# ============================================================
# ROUTERS
# ============================================================

def _run_submain(main_func, argv):
    original_argv = sys.argv[:]
    sys.argv = argv
    try:
        main_func()
    finally:
        sys.argv = original_argv


def _print_pipeline_outputs(tag):
    from hddflyzer.config import get_dataset_path

    results_dir = get_dataset_path(tag)
    print(f"[INFO] Results directory: {results_dir}")
    print(f"[INFO] Workflow summary: {get_dataset_path(tag, 'workflow_summary.md')}")
    print(f"[INFO] Manifest: {get_dataset_path(tag, 'manifest.json')}")


def _chem(args):
    if not args:
        _usage_chem(); return

    sub = args[0].lower()
    rest = args[1:]

    if sub == "npc":
        from hddflyzer.chem.npclassifier import main
        _run_submain(main, ["hddflyzer"] + rest)
    elif sub == "tanimoto":
        from hddflyzer.chem.tanimoto import main
        _run_submain(main, ["hddflyzer"] + rest)
    elif sub in ("sampling", "sample"):
        from hddflyzer.chem.tanimoto_sampling import main
        _run_submain(main, ["hddflyzer"] + rest)
    elif sub == "features":
        from hddflyzer.chem.feature_engineering import main
        _run_submain(main, ["hddflyzer"] + rest)
    elif sub in ("curate-features", "curate", "ml-features"):
        from hddflyzer.chem.feature_curation import main
        _run_submain(main, ["hddflyzer"] + rest)
    elif sub in ("reference-features", "similarity-features"):
        from hddflyzer.chem.feature_engineering import main
        _run_submain(main, ["hddflyzer", "reference"] + rest)
    elif sub == "stats":
        from hddflyzer.chem.stats import main
        _run_submain(main, ["hddflyzer"] + rest)
    elif sub in ("pruning", "prune"):
        from hddflyzer.chem.pruning import main
        _run_submain(main, ["hddflyzer"] + rest)
    else:
        print(f"[ERROR] Unknown chem subcommand: '{sub}'")
        _usage_chem()


def _data(args):
    if not args:
        _usage_data(); return

    sub = args[0].lower()
    rest = args[1:]

    if sub == "prepare":
        from hddflyzer.data.registry import main
        _run_submain(main, ["hddflyzer", sub] + rest)
    else:
        print(f"[ERROR] Unknown data subcommand: '{sub}'")
        _usage_data()


def _annotate(args):
    if not args:
        _usage_annotate(); return

    sub = args[0].lower()
    rest = args[1:]

    if sub == "npc":
        from hddflyzer.chem.npclassifier import main
        _run_submain(main, ["hddflyzer"] + rest)
    else:
        print(f"[ERROR] Unknown annotate subcommand: '{sub}'")
        _usage_annotate()


def _dimred(args):
    if not args:
        _usage_dimred(); return

    sub = args[0].lower()
    rest = args[1:]

    if sub == "pca":
        from hddflyzer.dimred.pca import main
        _run_submain(main, ["hddflyzer"] + rest)
    elif sub == "pca-joint":
        from hddflyzer.dimred.pca_joint import main
        _run_submain(main, ["hddflyzer"] + rest)
    elif sub == "tsne":
        from hddflyzer.dimred.tsne import main
        _run_submain(main, ["hddflyzer"] + rest)
    elif sub == "umap":
        from hddflyzer.dimred.umap import main
        _run_submain(main, ["hddflyzer"] + rest)
    else:
        print(f"[ERROR] Unknown dimred subcommand: '{sub}'")
        _usage_dimred()


def _viz(args):
    if not args:
        _usage_viz(); return

    sub = args[0].lower()
    rest = args[1:]

    if sub == "distributions":
        from hddflyzer.viz.distributions import main
        _run_submain(main, ["hddflyzer"] + rest)
    elif sub == "correlations":
        from hddflyzer.viz.correlations import main
        _run_submain(main, ["hddflyzer"] + rest)
    elif sub == "similarity":
        from hddflyzer.viz.similarity import main
        _run_submain(main, ["hddflyzer"] + rest)
    elif sub == "pca":
        from hddflyzer.viz.pca_plots import main
        _run_submain(main, ["hddflyzer"] + rest)
    elif sub == "tsne":
        from hddflyzer.viz.tsne_plots import main
        _run_submain(main, ["hddflyzer"] + rest)
    elif sub == "umap":
        from hddflyzer.viz.umap_plots import main
        _run_submain(main, ["hddflyzer"] + rest)
    elif sub == "npc":
        from hddflyzer.viz.npclassifier_plots import main
        _run_submain(main, ["hddflyzer"] + rest)
    else:
        print(f"[ERROR] Unknown viz subcommand: '{sub}'")
        _usage_viz()


def _pipeline(args):
    if not args:
        _usage_pipeline(); return

    sub = args[0].lower()
    rest = args[1:]

    if sub == "run":
        _pipeline_run(rest)
    else:
        print(f"[ERROR] Unknown pipeline subcommand: '{sub}'")
        _usage_pipeline()


def _pipeline_run(args):
    if not args or args[0] in ("-h", "--help"):
        _usage_pipeline()
        return

    tag = args[0]
    stage_names = None
    include_sample = True
    include_dimred = True
    save_pickle = False
    continue_on_error = False

    i = 1
    while i < len(args):
        arg = args[i]
        if arg == "--stages" and i + 1 < len(args):
            stage_names = [s.strip() for s in args[i + 1].split(",") if s.strip()]
            i += 2
        elif arg == "--skip-sample":
            include_sample = False
            i += 1
        elif arg == "--skip-dimred":
            include_dimred = False
            i += 1
        elif arg == "--save-pkl":
            save_pickle = True
            i += 1
        elif arg == "--continue-on-error":
            continue_on_error = True
            i += 1
        else:
            print(f"[ERROR] Unknown or incomplete option: {arg}")
            _usage_pipeline()
            sys.exit(1)

    from hddflyzer.pipeline import execute_workflow
    execution = execute_workflow(
        tag=tag,
        stage_names=stage_names,
        include_sample=include_sample,
        include_dimred=include_dimred,
        save_pickle=save_pickle,
        continue_on_error=continue_on_error,
    )
    _print_pipeline_outputs(tag)
    if not execution.ok:
        sys.exit(1)


# ============================================================
# USAGE MESSAGES
# ============================================================

def _usage_chem():
    print("""
hddflyzer chem <subcommand> [args]

Subcommands:
  npc          <tag>                       NPClassifier API annotation
  tanimoto     <tag>                       Tanimoto matrix
  sampling     <tag> [key=value ...]       Stratified sampling
  sample       <tag> [key=value ...]       Alias for sampling
  features     <tag> [--sampled] [--save-pkl] Intrinsic molecular descriptors
  curate-features <tag>                    ML-ready curated descriptor table
  curate       <tag>                       Alias for curate-features
  ml-features  <tag>                       Alias for curate-features
  reference-features <tag> --reference  Dyadic reference similarities
  similarity-features <tag> [--reference NAME] Alias for reference-features
  stats        base|hddf <tag>             Correlation statistics
  pruning      <tag> [--threshold 0.80]    Feature pruning
  prune        <tag> [--threshold 0.80]    Alias for pruning
""".strip())


def _usage_data():
    print("""
hddflyzer data <subcommand> [args]

Subcommands:
  prepare      <tag> [input_csv]           Build and validate local molecule registry
""".strip())


def _usage_annotate():
    print("""
hddflyzer annotate <subcommand> [args]

Subcommands:
  npc          <tag>                       NPClassifier external annotation
""".strip())


def _usage_dimred():
    print("""
hddflyzer dimred <subcommand> [args]

Subcommands:
  pca          <tag>                       PCA single collection
  pca-joint    <tag_a> <tag_b>             Joint PCA for two collections
  tsne         tanimoto|features|pruning <tag> t-SNE
  umap         features|tanimoto|pruning <tag> UMAP
""".strip())


def _usage_viz():
    print("""
hddflyzer viz <subcommand> [args]

Subcommands:
  distributions  <tag1,tag2> [features]   Violin / KDE plots
  correlations   hddf <tag>               Correlation plots
  similarity     tanimoto|fingerprints <tag>
  pca            analysis|collections     PCA figures
  tsne           tanimoto|features|pruning t-SNE figures
  umap           features|tanimoto|pruning <tag> UMAP figures
  npc            <tag>                    NPClassifier plots
""".strip())


def _usage_pipeline():
    print("""
hddflyzer pipeline <subcommand> [args]

Subcommands:
  run          <tag> [options]             End-to-end real-data workflow

Options for run:
  --stages a,b,c                           Run selected stage names only
  --skip-sample                            Skip stratified sampling
  --skip-dimred                            Skip t-SNE and UMAP stages
  --save-pkl                               Save optional feature pickle
  --continue-on-error                      Continue after failed stages
""".strip())


def _usage_main():
    print("""
HDDFlyzer — High-Dimensional Descriptor-based Pharmacological Analyzer

Usage:
  hddflyzer <module> <subcommand> [args...]

Modules:
  data     Local molecule registry (canonical molecule source)
  annotate External annotations (NPClassifier)
  chem     Chemistry pipeline (features, stats, pruning, similarity)
  dimred   Dimensionality reduction (PCA, t-SNE, UMAP)
  viz      Visualization (distributions, correlations, scatter plots)
  pipeline End-to-end orchestration over domain modules

Run 'hddflyzer <module>' for subcommand details.
""".strip())


# ============================================================
# MAIN
# ============================================================

def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        _usage_main()
        return

    module = args[0].lower()
    rest   = args[1:]

    if module == "data":
        _data(rest)
    elif module == "annotate":
        _annotate(rest)
    elif module == "chem":
        _chem(rest)
    elif module == "dimred":
        _dimred(rest)
    elif module == "viz":
        _viz(rest)
    elif module == "pipeline":
        _pipeline(rest)
    else:
        print(f"[ERROR] Unknown module: '{module}'")
        _usage_main()
        sys.exit(1)


if __name__ == "__main__":
    main()
