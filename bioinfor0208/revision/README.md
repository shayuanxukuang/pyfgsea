# Revision Asset Reproducibility Notes

This directory contains the source files used to assemble the Bioinformatics revision package.

## Core files

- `generate_revision_assets.py`: regenerates the revised main/supplementary figures and summary CSV tables.
- `build.ps1`: convenience build script that regenerates assets and then compiles the revision LaTeX files.
- `main_revised.tex`: revision-stage main manuscript source.
- `supplementary_revised.tex`: revision-stage supplementary source.
- `response_to_reviewers.tex`: response letter source.
- `sections/`: modular manuscript section files used by the revision sources.
- `data/*.csv`: lightweight exported summaries used for validation tables and figure audit traces.

## External inputs expected by `generate_revision_assets.py`

The script expects the following repository-level inputs to already exist:

- `data/gse155254_ery_only_pt.h5ad`
- `results/gse155254_hallmark_traj_ery_only.csv`

It also calls the local Python package, `Rscript`, and the R package `fgsea` for cross-implementation validation.

## Intended use

This directory is meant to preserve the revision workflow and figure-generation logic for transparency and reproducibility.
Temporary QA images, LaTeX intermediate files, and large raw archives are intentionally excluded from version control.
