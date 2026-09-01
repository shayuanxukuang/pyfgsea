# Figure 1 dual-lane evidence pipeline

This directory defines, but does not claim to have executed, the formal
PyFgsea/R-fgsea comparison needed for the 0.2.0 manuscript audit.  It is
fail-closed: a missing input, unclean or untagged evidence checkout, source-tree
import, mismatched wheel/core hash, wrong Python/R/Bioconductor/fgsea version,
missing R output, or changed output hash stops the run.

The two mandatory lanes are:

| Lane | Installed PyFgsea distribution | Module declaration | R | Bioconductor | fgsea | Rust revision |
| --- | --- | --- | --- | --- | --- | --- |
| `legacy` | 0.1.4 | 0.1.3 (historical packaging mismatch) | 4.4.3 | 3.20 | 1.32.2 | no revision API (required) |
| `current` | 0.2.0rc3 | 0.2.0rc3 | 4.6.0 | 3.23 | 1.38.0 | `fgsea-1.38-pr178-v1` |

The historical mismatch in the legacy row is intentional.  The `v0.1.4` tag's
Python project metadata declares 0.1.4, while its module and Cargo crate still
declare 0.1.3.  The runner verifies both identities instead of patching history.
The legacy lane is an **official PyPI artifact lane**: the verifier requires one
of PyPI's frozen 0.1.4 wheel SHA-256 values and exact v0.1.4 Python sources.  It
is not described as a source-reproducible native-core build because no such
historical build receipt exists.

## Evidence design

`prepare_inputs.py` materializes two byte-hashed scenarios:

- `publication_main` transcribes the historical
  `generate_test_data(n_genes=12000, n_sets=100, seed=42)` call recovered from
  commit `5cedd4abbc8d399221c741256ec5f3839861686d`.
- `ties_predeclared` is a separately labelled quantized-score stress case.  It
  is never represented as an input from the published paper.

The four expected rank/GMT SHA-256 values are also embedded in the tagged suite,
so changing input bytes and merely rewriting the manifest does not pass a lane.

Both PyFgsea calls use `min_size=15`, `max_size=500`, `sample_size=101`, Python
seed 1, `nperm_nes=1800`, `eps=1e-50`, and one thread. Both R calls use
`set.seed(314)`, the default `scoreType="std"`, and `nproc=1`. The current lane
also explicitly requires aligned mode, exact pathway sizes (`bin_width=0`),
`score_type=std`, deterministic gene-ID tie ordering, and `nperm_simple=1000`.
The legacy call passes only the historical arguments, preserving its historical
defaults.

`run_lane.py` retains the separate engine tables and emits
`pathway_level_raw.tsv`.  `adjudicate.py` verifies both receipts, concatenates
the raw pathway rows, reads the resulting table back from disk, and derives all
metrics from that table.  There is no code path for replacing a calculated
metric with a manuscript value.

Derived artifacts cover:

- maximum ES difference;
- NES Pearson and Spearman correlation, RMSE, median/p95/maximum absolute
  difference, and mean signed difference;
- the same audit vector for raw p-values and the historical
  `-log10(max(p, 1e-300))` display transform;
- predeclared extreme-tail rows and the ties scenario;
- top-10 absolute-NES pathway overlap and strict BH-FDR `<0.05` pathway-set
  overlap, matching the supplementary-table contract;
- Python/R elapsed time and externally sampled peak RSS;
- within-engine legacy/current changes; and
- a two-lane agreement figure.

## Formal run sequence

Use the same clean, tagged 0.2.0 evidence checkout for the scripts, but install
each PyFgsea wheel into a different clean environment.  Run from outside the
checkout so the source package cannot shadow the wheel.  Obtain wheel and
native-core hashes from the artifact-verification receipts; do not copy hashes
from an unverified working build.

The two Python environments must use the same Python implementation/version,
OS family and machine architecture, NumPy, pandas, and psutil versions.
`psutil` is an explicit runner dependency for peak-RSS sampling.  The full
platform strings are recorded, while the adjudicator rejects a pair when the
listed comparability fields differ.

Starting at the clean evidence-checkout repository root, capture the repository
and evidence roots, then create the common input bundle once:

```bash
export PYFGSEA_EVIDENCE_REPO="${PWD}"
export EVIDENCE_ROOT="${PYFGSEA_EVIDENCE_REPO}/../pyfgsea-evidence"
python "${PYFGSEA_EVIDENCE_REPO}/repro/figure1_dual_lane/prepare_inputs.py" \
  --output-dir "${EVIDENCE_ROOT}/figure1-inputs"
```

First bind the downloaded official 0.1.4 wheel to the clean `v0.1.4` checkout
with `verify_legacy_artifact.py`. The current lane instead consumes the passed
receipt from `scripts/verify_pyfgsea_artifacts.py`.

Run each lane **inside the corresponding linux/amd64 OCI image named by its
passed reference receipt, or a recorded execution image derived directly from
that digest without modifying `/opt/reference`**. The reference images are
R-focused, so the latter may add the common Python runtime and verified wheel.
Inject the reference receipt's built digest as
`FGSEA_REFERENCE_IMAGE_DIGEST`; the runner verifies the receipt profile,
commit/tree, Dockerfile/base/built digests, and all six `/opt/reference` file
hashes. For the legacy container:

```bash
export PYFGSEA_EVIDENCE_REPO="${PYFGSEA_EVIDENCE_REPO:-${PWD}}"
export EVIDENCE_ROOT="${EVIDENCE_ROOT:-${PYFGSEA_EVIDENCE_REPO}/../pyfgsea-evidence}"
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
  --expected-git-tag v0.2.0-rc3
```

Run the analogous current command in the 0.2.0rc3 wheel environment with R
4.6.0/Bioconductor 3.23/fgsea 1.38.0:

```bash
export PYFGSEA_EVIDENCE_REPO="${PYFGSEA_EVIDENCE_REPO:-${PWD}}"
export EVIDENCE_ROOT="${EVIDENCE_ROOT:-${PYFGSEA_EVIDENCE_REPO}/../pyfgsea-evidence}"
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
  --expected-git-tag v0.2.0-rc3
```

Finally, in a clean analysis environment containing pandas, NumPy, and
Matplotlib, combine the two receipts:

```bash
export PYFGSEA_EVIDENCE_REPO="${PYFGSEA_EVIDENCE_REPO:-${PWD}}"
export EVIDENCE_ROOT="${EVIDENCE_ROOT:-${PYFGSEA_EVIDENCE_REPO}/../pyfgsea-evidence}"
cd "${EVIDENCE_ROOT}"
python "${PYFGSEA_EVIDENCE_REPO}/repro/figure1_dual_lane/adjudicate.py" \
  --legacy-receipt "${EVIDENCE_ROOT}/figure1-legacy/lane_receipt.json" \
  --current-receipt "${EVIDENCE_ROOT}/figure1-current/lane_receipt.json" \
  --input-manifest "${EVIDENCE_ROOT}/figure1-inputs/input_manifest.json" \
  --output-dir "${EVIDENCE_ROOT}/figure1-adjudicated" \
  --expected-git-commit FULL_40_CHARACTER_RC_COMMIT \
  --expected-git-tag v0.2.0-rc3
```

The adjudication receipt binds the lane receipts, raw rows, metrics, plot,
runtime/memory table, exact commands, scripts, Git commit/tree/tag, wheels,
native cores, R environments, inputs, and every output by SHA-256.

## Boundaries

- The original temporary rank/GMT files were not archived, so the publication
  input is a source-level reconstruction of the recovered generator, not a
  claim of byte identity to an unavailable historical temporary file.
- This pipeline adjudicates PyFgsea versus the declared R-fgsea reference.  It
  does not reproduce the separate GSEApy or BlitzGSEA panels.
- Peak RSS is sampled externally every 20 ms and is hardware-specific.  It is
  suitable for a recorded rerun, not a guarantee that a short allocation spike
  was observed.
- The legacy implementation's tie ordering was not cross-platform stable.
  `ties_predeclared` is therefore same-recorded-environment sensitivity evidence,
  not evidence of cross-platform tie equivalence.
- A successful run supplies evidence; it does not by itself decide whether the
  manuscript requires a compatibility note, Author Correction, or corrigendum.
