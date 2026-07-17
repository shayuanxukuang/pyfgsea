# Trajectory Pathway Event Discovery (TED)

**Trajectory Pathway Event Discovery (TED)** is an artifact-aware protocol for structured interpretation of dynamic pathway events in single-cell genomics.

TED starts from pathway, module or perturbation activity profiles and writes row-wise event objects. Each event row records event mode, effect and uncertainty, block support, matched-state and negative-control behavior, identifiability, an E0--E2 within-study support code, and a V0--V4 external-evidence provenance code. Escort addresses upstream trajectory suitability; TED starts after a trajectory, time or state representation has been supplied. PyFgsea can generate upstream activity profiles, but TED is a separate downstream inference task and accepts activity matrices produced by other scoring or trajectory methods.

## Release status

The journal-neutral semantic release `ted-v1.0.0` contains the July 2026 E/V v2 schema, current-task baselines, repeated embryo holdouts, heavy-control outputs, scaling results and validation-demo records. Its version-specific archive DOI is [10.5281/zenodo.21403133](https://doi.org/10.5281/zenodo.21403133).

The historical May 2026 archive is:

- Historical Git tag: `ted-gb-rc7`
- Commit: `3ffec1a1dcb4261303fc130b81ccd6b29a2fa34f`
- Zenodo DOI: [10.5281/zenodo.20378158](https://doi.org/10.5281/zenodo.20378158)
- License: MIT

That DOI remains valid for the historical snapshot, but it does not contain the complete July 2026 materials. Cite `10.5281/zenodo.21403133` for the E/V v2 release.

### Analysis lock and final release

The analysis lock identifies the exact estimator code, thresholds, seeds, inputs and machine-readable output hashes used for evaluation. The final-release commit may add packaging, documentation, citation metadata and archive manifests, but it must not silently change locked estimators or thresholds. Release metadata must record both identifiers and a file-level difference audit. If analysis code or thresholds change, the affected analyses must be rerun and a new analysis lock recorded.

## What is in this repository

| Path | Purpose |
| --- | --- |
| `pyfgsea/` | Python package code for PyFgsea and TED event-analysis components. |
| `scripts/` | Reproducibility scripts for known-source validation, GATA1/GATA1s support, external baselines, benchmarks and figure generation. |
| `tests/` | Unit and validation tests used for the release snapshot. |
| `config/` | Event axes, negative-control axes, evidence-boundary rules and preregistration cards. Some historical filenames retain legacy `claim_boundary` names. |
| `tables/` | Machine-readable event objects, benchmark summaries, evidence-boundary outputs, validation summaries and release-audit tables. Some historical filenames retain legacy terminology for provenance. |
| `figures/` | Main figure PDFs/PNGs and their source-data TSV files. |
| `reproducibility/` | Minimal demo entry point and reviewer-facing reproducibility helpers. |
| `results/ted_v1_submission/` | Final figure source data, controlled-benchmark summaries, public-data audits and checksummed release outputs. |
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

## Key validation outputs in the historical snapshot

TED was evaluated with a combination of same-input benchmarks, public known-source datasets and evidence-boundary audits. The most useful starting points in the historical snapshot are:

| File | What it records |
| --- | --- |
| `tables/known_source_validation_summary.tsv` | Public known-source validation results for GSE153056, GSE93735 and SCP1064. |
| `tables/ted_dataset_level_claim_boundary.tsv` | Dataset-level evidence boundaries assigned by TED; legacy filename. |
| `tables/benchmark_audit_table.tsv` | Benchmark truth sources, scored units, uncertainty reporting and frozen/optimization status. |
| `tables/benchmark_non_circular_evaluation_table.tsv` | Separation of biological correctness metrics from reporting-completeness fields. |
| `tables/dynamic_pathway_event_table.tsv` | Standardized dynamic pathway-event grammar rows. |
| `tables/scp1064_lightweight_shuffle_summary.tsv` | Lightweight label-shuffle audit for SCP1064 outcome alignment. |
| `tables/gata1_cross_dataset_support_summary.tsv` | Independent GATA1/GATA1s directional-support summary. |
| `figures/figure2_known_source_validation.pdf` | Public known-source outcome and reversal validation figure. |
| `figures/figure4_gse271399_gata1_cross_dataset_support.pdf` | GSE271399 and independent GATA1/GATA1s support figure. |
| `figures/figure5_claim_upgrade_block_audit.pdf` | Evidence-promotion/block audit figure; legacy filename. |

## Current evidence descriptors

The July 2026 control audit supersedes the historical descriptors below for manuscript interpretation. These rows must be accompanied by their E/V v2 tables and full control outputs in `ted-v1.0.0`.

| Dataset | Readout | Current descriptor | Qualification |
| --- | --- | --- | --- |
| GSE153056 | IFNG/PD-L1 RNA event aligned with PD-L1 protein effects | E1--V1 | Formal locked assignment; a retrospective three-block audit met E2 eligibility but does not replace the locked endpoint. |
| GSE93735 | Partial dexamethasone reversal of an LPS-associated signal | E0--V2 | Two samples per group and no event-level q value; reversal provenance is descriptive. |
| SCP1064 | CRISPR-linked RNA events aligned with CITE-seq protein readouts | E1--V1 (partial) | Mandatory within-block guide-label shuffling passed only 1 of 4 axes, so E2 promotion is blocked. |
| GSE271399 | GATA1/T21 computational event with cross-dataset directional context | E1--V0 | Available design strata are not independent biological blocks, and no same-system matched full-length GATA1 rescue is available. |

Positive outcome alignment cannot override a failed mandatory control. Same-system matched rescue is represented separately from orthogonal outcomes or intervention-reversal provenance.

## Current benchmark status

The July 2026 packet split is retrospective: replicates 1--8 per event mode are development data, 9--12 are used for baseline tuning, and 13--16 form a 40-packet shifted audit whose labels were masked during prediction. Because the generator and TED rules pre-date this split, the audit is not an untouched final test and no truly post-freeze external dataset has yet been evaluated.

On the shifted audit, TED aggregate packet-class macro-F1 was 0.741 and controlled E-level macro-F1 was 0.761. Supervised models classified packet and artifact states more accurately. TED made no upward E errors in this audit, at the cost of a 0.225 false-demotion rate; each downgrade was linked to a recorded failed gate. TED is therefore evaluated as a conservative evidence-assignment protocol rather than an accuracy-maximizing classifier.

Across 100 repeated 20% ZSCAPE embryo holdouts, median event-set Jaccard was 0.654, direction agreement 1.000 and mode agreement 0.936. Across 50 balanced split halves, median event-set Jaccard was 0.207, showing that event discovery is sample-sensitive even when directions of common calls are stable.

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

Machine-readable citation metadata are provided in `CITATION.cff`. Cite the `ted-v1.0.0` archive as [10.5281/zenodo.21403133](https://doi.org/10.5281/zenodo.21403133). The historical DOI [10.5281/zenodo.20378158](https://doi.org/10.5281/zenodo.20378158) refers only to the May 2026 development snapshot.
