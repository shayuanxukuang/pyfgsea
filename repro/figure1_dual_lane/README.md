# Figure 1 dual-lane comparison

This directory recalculates the PyFgsea/R-fgsea comparisons used in Figure 1.
It produces pathway-level rows, agreement metrics, pathway overlap,
extreme-tail cases, runtime and memory measurements, a comparison figure, and a
run record with SHA-256 values.

No completed Figure 1 comparison is implied by the presence of these scripts.
Both lanes must run from verified artifacts before the outputs can be used in a
paper update.

## Reference lanes

| Lane | Installed PyFgsea | Module version | R | Bioconductor | fgsea | Algorithm revision |
| --- | --- | --- | --- | --- | --- | --- |
| `legacy` | 0.1.4 | 0.1.3 | 4.4.3 | 3.20 | 1.32.2 | no revision API |
| `current` | 0.2.0rc8 | 0.2.0rc8 | 4.6.0 | 3.23 | 1.38.0 | `fgsea-1.38-pr178-v1` |

The 0.1.4 distribution metadata and 0.1.3 module declaration are the recorded
historical packaging state. The legacy verifier checks both values. It accepts
only an official PyPI 0.1.4 wheel with a recorded SHA-256 value and the exact
`v0.1.4` Python sources. No source-reproducible native-core claim is made for
that historical wheel.

The lanes answer different questions. The legacy lane reproduces the paper's
fgsea 1.32.2 comparison. The current lane tests the repaired implementation
against fgsea 1.38.0. Do not pool or relabel them.

## Input contract

`prepare_inputs.py` copies four committed files into a new output directory:

- `publication_main/ranks.csv`;
- `publication_main/pathways.gmt`;
- `ties_predeclared/ranks.csv`;
- `ties_predeclared/pathways.gmt`.

`publication_main` reconstructs the historical
`generate_test_data(n_genes=12000, n_sets=100, seed=42)` call. The score column
was canonicalized once to 12 significant decimal digits because the final
binary64 bits of generated normal values can vary by platform. Gene identifiers
and pathway membership are unchanged.

`ties_predeclared` is a separately labelled quantized-score stress case. It is
not a paper input.

GitHub functional tests check the committed bytes and semantic invariants,
including generator behavior, ordering, pathway membership, and tie rounding.
They do not compare the inputs with a predeclared SHA-256 value. Run records
calculate SHA-256 values for the exact files used and require both lanes to use
the same materialized bytes.

## Fixed analysis parameters

Both PyFgsea calls use:

- `min_size=15`;
- `max_size=500`;
- `sample_size=101`;
- `seed=1`;
- `nperm_nes=1800`;
- `eps=1e-50`; and
- one thread.

Both R calls use `set.seed(314)`, `scoreType="std"`, and `nproc=1`.

The current PyFgsea lane additionally requires `mode="aligned"`,
`score_type="std"`, `bin_width=0`, `tie_policy="gene_id"`, and
`nperm_simple=1000`. The legacy call preserves its historical argument set and
effective score type.

## Requirements

Use one clean checkout for all scripts. The checkout must have:

- no tracked or untracked worktree changes;
- a full recorded commit and tree;
- an annotated release-candidate tag that peels to that commit; and
- no source checkout on the import path before the installed package.

The current package report must record the complete source-to-install sequence:

1. source tests and Rust tests passed;
2. an sdist was built from the recorded source;
3. the wheel was built from that verified sdist;
4. the wheel was installed in a clean environment;
5. the installed version and native core match the wheel; and
6. the installed-wheel tests passed.

The legacy package report is produced by `verify_legacy_artifact.py` from an
official PyPI 0.1.4 wheel. Each lane also requires the matching linux/amd64 OCI
reference build report and its image digest.

Run the two Python environments with the same Python implementation and
version, OS family, architecture, NumPy, pandas, and psutil versions. Peak RSS
sampling requires `psutil`.

## Run sequence

Set paths from the clean checkout and create the shared input bundle:

```bash
export PYFGSEA_EVIDENCE_REPO="${PWD}"
export EVIDENCE_ROOT="${PYFGSEA_EVIDENCE_REPO}/../pyfgsea-evidence"

python "${PYFGSEA_EVIDENCE_REPO}/repro/figure1_dual_lane/prepare_inputs.py" \
  --output-dir "${EVIDENCE_ROOT}/figure1-inputs"
```

Run each lane inside the linux/amd64 OCI image named by its reference build report,
or inside a recorded execution image derived directly from that digest without
changing `/opt/reference`. The execution image may add the common Python
runtime and the verified wheel.

### Legacy lane: PyFgsea 0.1.4 / fgsea 1.32.2

```bash
export FGSEA_REFERENCE_VERSION=1.32.2
export FGSEA_REFERENCE_ID=legacy-publication
export FGSEA_REFERENCE_IMAGE_DIGEST=sha256:BUILT_DIGEST_FROM_RECEIPT

cd "${EVIDENCE_ROOT}"
python "${PYFGSEA_EVIDENCE_REPO}/repro/figure1_dual_lane/run_lane.py" \
  --lane legacy \
  --input-manifest "${EVIDENCE_ROOT}/figure1-inputs/input_manifest.json" \
  --artifact-receipt "${EVIDENCE_ROOT}/legacy-artifact/receipt.json" \
  --reference-receipt "${EVIDENCE_ROOT}/legacy-oci/oci-receipt.json" \
  --output-dir "${EVIDENCE_ROOT}/figure1-legacy" \
  --expected-git-commit FULL_40_CHARACTER_RC_COMMIT \
  --expected-git-tag v0.2.0-rc8
```

### Current lane: PyFgsea 0.2.0rc8 / fgsea 1.38.0

```bash
export FGSEA_REFERENCE_VERSION=1.38.0
export FGSEA_REFERENCE_ID=current-conformance
export FGSEA_REFERENCE_IMAGE_DIGEST=sha256:BUILT_DIGEST_FROM_RECEIPT

cd "${EVIDENCE_ROOT}"
python "${PYFGSEA_EVIDENCE_REPO}/repro/figure1_dual_lane/run_lane.py" \
  --lane current \
  --input-manifest "${EVIDENCE_ROOT}/figure1-inputs/input_manifest.json" \
  --artifact-receipt "${EVIDENCE_ROOT}/current-artifact/receipt.json" \
  --reference-receipt "${EVIDENCE_ROOT}/current-oci/oci-receipt.json" \
  --output-dir "${EVIDENCE_ROOT}/figure1-current" \
  --expected-git-commit FULL_40_CHARACTER_RC_COMMIT \
  --expected-git-tag v0.2.0-rc8
```

### Compare the two reference runs

Run the comparison in a clean environment containing pandas, NumPy, and
Matplotlib:

```bash
cd "${EVIDENCE_ROOT}"
python "${PYFGSEA_EVIDENCE_REPO}/repro/figure1_dual_lane/compare_results.py" \
  --legacy-receipt "${EVIDENCE_ROOT}/figure1-legacy/lane_receipt.json" \
  --current-receipt "${EVIDENCE_ROOT}/figure1-current/lane_receipt.json" \
  --input-manifest "${EVIDENCE_ROOT}/figure1-inputs/input_manifest.json" \
  --output-dir "${EVIDENCE_ROOT}/figure1-compared" \
  --expected-git-commit FULL_40_CHARACTER_RC_COMMIT \
  --expected-git-tag v0.2.0-rc8
```

## Outputs

`compare_results.py` derives every metric from the written pathway-level table
and creates:

- `figure1_pathway_level_raw.tsv`;
- `figure1_agreement_metrics.tsv`;
- `figure1_pathway_overlap.tsv`;
- `figure1_extreme_tail_cases.tsv`;
- `figure1_legacy_current_change.tsv`;
- `figure1_runtime_memory.tsv`;
- `figure1_dual_lane_agreement.png`; and
- `adjudication_receipt.json` (the existing schema filename).

Metrics include maximum ES difference; NES Pearson and Spearman correlation,
RMSE, signed and absolute differences; p-value and `-log10(P)` agreement;
extreme-tail and tie cases; top-10 absolute-NES and strict BH-FDR `<0.05`
pathway overlap; runtime; peak RSS; and within-engine legacy/current changes.

The run record includes the commit, tree, annotated tag, scripts, input run
records, wheels, native cores, R environments, input files, commands, outputs,
and their SHA-256 values.

## Failure handling and limitations

The pipeline stops on a dirty or incorrectly tagged checkout, source-tree
import, artifact or native-core mismatch, failed installed-wheel tests, wrong
Python/R/Bioconductor/fgsea version, missing R output, changed SHA-256 record,
or incompatible lane parameters.

Unresolved and failure states remain explicit in the lane output and run record.
They must be reported rather than dropped from the comparison.

The reconstructed publication input is tied to the recovered generator, not to
an unavailable historical temporary file. The pipeline does not reproduce the
separate GSEApy or BlitzGSEA panels. Peak RSS is sampled every 20 ms and remains
hardware-specific. Legacy tie ordering was not cross-platform stable, so the
ties scenario is a same-recorded-environment sensitivity result, not a claim of
cross-platform tie equivalence.
