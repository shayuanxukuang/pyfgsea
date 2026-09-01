# fgsea 1.38 alignment boundary

Status: the 0.2.0 implementation, required source-level regression matrix, and an isolated R 4.6.0/Bioconductor 3.23/fgsea 1.38.0 comparison have passed. The Docker reference profiles, legacy 1.32.2 lane, artifact-level conformance matrix, Figure 1, formal Figure 2 binding, supplement, and manuscript statistics have not been accepted as repaired 0.2.0 results.

## Two reference lanes

The published comparison and the current conformance target are separate evidence lanes. They must not be merged into one unnamed “R/fgsea reference.”

| Lane | PyFgsea | R | Bioconductor | fgsea | Permitted claim |
| --- | --- | --- | --- | --- | --- |
| `legacy_publication` | 0.1.4 | 4.4.3 | 3.20 | 1.32.2 | Reproduction of the originally reported comparison only |
| `current_conformance` | 0.2.0 | 4.6.0 | 3.23 | 1.38.0 | Current alignment only after the new conformance gates and figure recalculation pass |

The lane definitions live in `reference_manifest.json`. The corresponding R reference images are `Dockerfile.reference-fgsea-1.32.2` and `Dockerfile.reference-fgsea-1.38.0`. Their Linux/amd64 `r-base` manifests are pinned by digest. The checked-in lockfiles are only bootstrap contracts for the direct reference packages: they do not lock transitive packages, the Dockerfiles do not restore them, and the apt repositories are not snapshot-pinned. Each image is designed to retain a resolved `renv.resolved.lock`, `sessionInfo()`, package-manager inventory, and verification output after installation, but neither image has yet been built. The profiles therefore must not yet be described as fully frozen environments; the exported OCI digest and embedded evidence must first be recorded.

Every R comparison is fail-closed. Set `FGSEA_REFERENCE_VERSION` to exactly `1.32.2` or `1.38.0`; `scripts/verify_fgsea_reference.R` rejects an absent or mismatched installation. The Docker profiles additionally assert the exact R and Bioconductor versions. A workstation package such as fgsea 1.36.0 is not a reference environment.

## Upstream basis for the current lane

The current lane is bounded to Bioconductor fgsea 1.38.0. Its relevant upstream change is [alserglab/fgsea PR 178](https://github.com/alserglab/fgsea/pull/178), merge commit `9a06694dfc7b54a0a698061a97db15945ede725c`, which introduced gene-set-hash ordering in the multilevel state and closed [issue 151](https://github.com/alserglab/fgsea/issues/151). This PR merge identifier is the alignment basis, not the source commit recorded in the Bioconductor tarball metadata. The official 1.38.0 tarball reports source commit `1fe4644`; the legacy 1.32.2 tarball reports `4620281`. The repository does not vendor either source snapshot; each official URL, SHA-256, and source commit is recorded in the manifest.

“Aligned” therefore means agreement with the declared 1.38.0 reference under matched score type, pathway universe, size filters, `sampleSize`, `eps`, seed policy, and explicit result mode. It does not mean bitwise identity across different RNG implementations, and it does not retroactively convert the published 1.32.2 comparison into a 1.38.0 validation.

## Required recalculation before a 0.2.0 publication claim

The following artifacts are invalidated by statistical-result changes and remain unaccepted. A candidate run is not an accepted artifact unless its parameter contract, clean source commit, inputs, and result hashes are bound in the manifest:

1. Figure 1 raw pathway values and all ES, NES, nominal-p, rank-overlap, and FDR-overlap summaries in both reference lanes. Metrics must be derived from the emitted raw table; manual metric overrides are prohibited.
   The versioned fail-closed runner is `repro/figure1_dual_lane/`. It reconstructs the publication generator input, labels a separate predeclared ties scenario, and verifies the historical 0.1.4 distribution/module version mismatch. The suite has passed static and input-hash tests but has not yet emitted accepted legacy/current lane receipts.
2. The six-regime equivalence audit and Supplementary Table S7/Figure S10, including unresolved/failure-state counts and `log2err` where available.
3. Deep-tail, tie-heavy, integer-weight, score-type, and upstream issue-151 regression cases. These cases now pass the local 0.2.0 test suite, but they remain part of the artifact-level conformance gate. A run that hangs, emits an unqualified zero p-value, or silently drops a pathway fails the gate.
4. Figure 2 from the processed GSE155254 input through rolling-window GSEA. The manuscript's 500-cell baseline conflicts with the existing 400-cell/20-step/500-permutation generation script. A non-overwriting 0.2.0 candidate was regenerated after the final local Rust-core rebuild with the explicit assumption `window_size=500`, `step=50`, `nperm_nes=2000`, `score_type=std`, exact sizes, and no cross-window NES cache. It produced 62 windows by 43 pathways, all with `status=resolved`; all p-values were finite and positive, all recorded artifact hashes matched, and an independent within-window BH calculation agreed within the predeclared `1e-14` absolute tolerance. It remains publication-unbound because the step/permutation choice requires author acceptance and the source tree was dirty. The pre-final-audit candidate is retained only as ignored local evidence; it is not part of this source revision. The versioned runner is `scripts/run_gse155254_0_2_0_figure2_candidate.py`, accepts explicit external H5AD/GMT paths, and enforces their frozen hashes. A formal receipt must be generated from a clean tagged checkout and attached to the release/archival record; no ignored `results/` path is cited as accepted evidence.
5. Window-size/step sensitivity, null calibration, Bland–Altman, thread scaling, and stateful/cache benchmarks. Missing source results must stop the build instead of producing synthetic or dummy evidence.
6. All manuscript values and wording that depend on those artifacts, including broad `fgseaMultilevel-aligned` language and performance claims affected by cache or binning changes.

Each accepted run must record the clean per-profile PyFgsea commit, PyFgsea source-tree hash, fgsea source commit, profile, R/Bioconductor/fgsea versions, exact parameters and seeds, primary dataset SHA-256, all input and result SHA-256 values, Docker base and built-image digests, platform, and resolved reference-lock hash. `scripts/update_reference_manifest.py` is the only supported path for binding those fields to the checked-in manifest; it rejects missing evidence fields and a dirty source tree. Its `--pyfgsea-source` argument must point to a separately verified clean 0.1.4 checkout for `legacy_publication` and a clean 0.2.0 checkout for `current_conformance`; the command verifies the profile version before binding.

## Executed local verification

After the final Rust score-direction and tie-semantics repair, the release extension was rebuilt and the following checks passed on 2026-09-01:

- 7 Rust unit tests, including exact-score ordering, hash tie-breaking, underflow, no-progress/mixing failure semantics, and zero-tail pseudocount behavior.
- 236 non-slow/non-external tests under `tests/`. The generic Python run skipped only the optional exact-R subprocess test and deselected 13 explicitly slow or external tests.
- Both tests in `tests/test_reference_fgsea.py` in a separate exact R 4.6.0/Bioconductor 3.23/fgsea 1.38.0 environment. Together with the generic run, all 53 tests across the eight required 0.2.0 regression files executed successfully.
- 7 lightweight tests for the Figure 1 dual-lane evidence pipeline, plus deterministic regeneration of all four predeclared input hashes.

These counts are source-level verification, not substitutes for the clean-image, legacy-lane, figure, supplement, and manuscript evidence gates above.

The read-only workflow `.github/workflows/pyfgsea-0.2.0-artifacts.yml` defines the next artifact gates: per-platform clean-commit sdist-to-wheel receipts and non-pushing Linux/amd64 OCI exports for both R reference profiles. All referenced Actions are commit-pinned. The workflow has not yet run, so no cross-platform artifact or OCI digest in that definition is accepted evidence.

## Local current-reference smoke result

An isolated, non-Docker R 4.6.0 library was created on 2026-09-01 and verified as Bioconductor 3.23 with fgsea 1.38.0. `tests/test_reference_fgsea.py` passed its exact-ES and Monte Carlo-aware NES, p-value, and `log2err` checks. The installer and fgsea binary hashes are recorded in the manifest. This is useful smoke evidence, but it is not a frozen-image receipt, does not validate the legacy lane, and does not bind any manuscript artifact to a clean PyFgsea commit.

## Current verification limitation

On the present host the Docker CLI is installed, but the Docker Desktop backend failed while `initializing Inference manager` (`dockerInference`): `The filename, directory name, or volume label syntax is incorrect.` No Docker global setting was changed. Consequently the Dockerfiles and source hashes were checked statically, but neither image build nor the image-generated resolved lock is claimed complete. The isolated current-reference smoke result above is the only executed R 1.38 validation and is explicitly not an artifact-binding run.
