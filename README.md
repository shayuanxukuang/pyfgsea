# pyfgsea-TED

Software and reproducibility companion for **TED**, an evidence-gated framework for dynamic pathway event interpretation in single-cell genomics.

TED starts from pathway, module or perturbation activity profiles and writes them as row-wise event objects. Each event row records the event type or mode, event-level evidence, block support, matched-state and negative-control behavior, and the evidence boundary supported by the analysis design. The goal is to make dynamic pathway interpretation reproducible: a changed pathway score becomes a structured event call with explicit support, controls and missing evidence.

## Release

The archived release for review and citation is:

- Git tag: `ted-gb-rc7`
- Commit: `3ffec1a1dcb4261303fc130b81ccd6b29a2fa34f`
- Zenodo DOI: [10.5281/zenodo.20378158](https://doi.org/10.5281/zenodo.20378158)
- License: MIT

The tables under `tables/` and the audit files in this repository refer to this archived release.

## What is in this repository

| Path | Purpose |
| --- | --- |
| `pyfgsea/` | Python package code for PyFgsea and TED event-analysis components. |
| `scripts/` | Reproducibility scripts for known-source validation, GATA1/GATA1s support, external baselines, benchmarks and figure generation. |
| `tests/` | Unit and validation tests used for the release snapshot. |
| `config/` | Event axes, negative-control axes, claim-boundary rules and preregistration cards. |
| `tables/` | Machine-readable event objects, benchmark summaries, claim-boundary outputs, validation summaries and release-audit tables. |
| `figures/` | Main figure PDFs/PNGs and their source-data TSV files. |
| `reproducibility/` | Minimal demo entry point and reviewer-facing reproducibility helpers. |
| `Dockerfile`, `Dockerfile.baselines`, `environment*.yml` | Runtime environments for TED analyses and direct external baseline execution. |

Journal submission files are managed separately from this software archive. Large public raw datasets are also kept at their original repositories; the manifests in `tables/` record accessions, file provenance and processed-output checks.

## Quick start

Create the Python environment:

```bash
conda env create -f environment.yml
conda activate ted-development
```

Run the smallest local check:

```bash
python reproducibility/run_minimal_demo.py
```

Run the release validation tests used most often for the known-source analyses:

```bash
python -m pytest \
  tests/test_scp1064_file_qc.py \
  tests/test_scp1064_cell_alignment.py \
  tests/test_scp1064_event_outcome_alignment.py \
  tests/test_scp1064_claim_boundary.py \
  tests/test_ted_known_source_validation.py
```

For the external baseline environment:

```bash
docker build -f Dockerfile.baselines -t ted-external-baselines .
docker run --rm ted-external-baselines
```

## Key validation outputs

TED was evaluated with a combination of same-input benchmarks, public known-source datasets and claim-boundary audits. The most useful starting points are:

| File | What it records |
| --- | --- |
| `tables/known_source_validation_summary.tsv` | Public known-source validation results for GSE153056, GSE93735 and SCP1064. |
| `tables/ted_dataset_level_claim_boundary.tsv` | Dataset-level evidence boundaries assigned by TED. |
| `tables/benchmark_audit_table.tsv` | Benchmark truth sources, scored units, uncertainty reporting and frozen/optimization status. |
| `tables/benchmark_non_circular_evaluation_table.tsv` | Separation of biological correctness metrics from reporting-completeness fields. |
| `tables/dynamic_pathway_event_table.tsv` | Standardized dynamic pathway-event grammar rows. |
| `tables/scp1064_lightweight_shuffle_summary.tsv` | Lightweight label-shuffle audit for SCP1064 outcome alignment. |
| `tables/gata1_cross_dataset_support_summary.tsv` | Independent GATA1/GATA1s directional-support summary. |
| `figures/figure2_known_source_validation.pdf` | Public known-source outcome and reversal validation figure. |
| `figures/figure4_gse271399_gata1_cross_dataset_support.pdf` | GSE271399 and independent GATA1/GATA1s support figure. |
| `figures/figure5_claim_upgrade_block_audit.pdf` | Claim-boundary upgrade/block audit figure. |

## Public known-source validation snapshot

| Dataset | Known source | Readout | TED boundary |
| --- | --- | --- | --- |
| GSE153056 | IFNG stimulation and PD-L1 regulator perturbation | PD-L1 protein outcome; event-protein Spearman correlation 0.7846 | `outcome_supported_event` |
| GSE93735 | LPS plus dexamethasone intervention | Dexamethasone-associated reversal; primary reversal fraction 0.3375 | `reversal_supported_event` |
| SCP1064 | CRISPR guide identity in Perturb-CITE-seq | RNA immune-evasion event aligned with CITE-seq protein readouts in 195,303 source-labeled cells | `outcome_supported_event` |

These examples test whether TED can assign stronger event boundaries when source, outcome or reversal evidence is present. Matched same-system rescue claims remain governed by the stricter claim-boundary rules in `config/ted_claim_boundary_rules.yml`.

## Direct external baselines

The direct external baseline suite includes wrappers for representative upstream tools, including tradeSeq, GSVA, AUCell and POT. These runs are used to check executable upstream outputs and to define how native outputs can be carried into the downstream TED-object comparison.

```bash
python scripts/run_direct_external_baseline_suite.py --quick
```

For the package-complete baseline image, use `Dockerfile.baselines`.

## Data access

The release uses public datasets from GEO, Single Cell Portal, STOmicsDB/CNGB and related public resources. Raw public archives are not mirrored here. The relevant accessions, download status, checksums and analysis roles are tracked in:

- `tables/availability_accession_audit.tsv`
- `tables/candidate_download_manifest.tsv`
- `RELEASE_MANIFEST.tsv`

## Citation

If this archive is used before the journal article is available, cite the Zenodo release:

> pyfgsea-TED release `ted-gb-rc7`, Zenodo DOI [10.5281/zenodo.20378158](https://doi.org/10.5281/zenodo.20378158).

After publication, please cite the article together with the archived release.
