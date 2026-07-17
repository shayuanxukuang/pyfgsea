# TED v1.0.0 release status

Status: **released as the immutable `ted-v1.0.0` archive**. This record is bound to the final tag and version-specific DOI.

## Identifiers

| Field | Current value | Interpretation |
| --- | --- | --- |
| Release version | `ted-v1.0.0` | Journal-neutral semantic release tag. |
| Historical archive | `ted-gb-rc7`; `3ffec1a1dcb4261303fc130b81ccd6b29a2fa34f`; `10.5281/zenodo.20378158` | May 2026 snapshot; not the archive for the July E/V v2 manuscript. |
| Analysis base commit | `2b047dc557604018b28d82fa3da9ab496e1955a4` | Base PyFgsea commit recorded by the retrospective lock; not sufficient on its own because July TED analysis files were not all tracked at that commit. |
| Analysis evidence manifest | `results/ted_submission_supplement/evidence_manifest.tsv` | File-level SHA256 record for the July analysis outputs and sources. |
| Complete analysis-lock commit | `54ec1269344b6a1392928324ae4a54c7d68d0260` | Contains the locked estimator, thresholds/configuration, schemas, analysis scripts and verified outputs. |
| Final-release commit | `ted-v1.0.0` | Immutable tag adding citation metadata, this verification record and the release-wide SHA256 manifest. |
| Zenodo version DOI | `10.5281/zenodo.21403133` | Version-specific DOI for the `ted-v1.0.0` archive. |

## Local verification completed on 2026-07-17

- The Rust-backed wheel built successfully from the dedicated release clone and was installed into an isolated Python environment.
- The release-contained test suite passed in two jobs: `175 passed` for the fast/integration group and `7 passed` for the slow group. Six tests that require large source matrices not distributed in the repository are marked `external_data` and are not represented as local passes.
- The validation demo ran from outside the source tree. Both the activity-table v1 and event-table v2 CLI validations returned `ok`.
- `docker build --no-cache -t ted:1.0.0 .` completed, followed by successful CLI, event-validation and validation-demo smoke tests inside the container.
- The R/Bioconductor baseline environment completed a no-cache build after moving Cargo output to a writable directory. The final command/work-directory layer was rebuilt and smoke-tested with pyfgsea `0.1.4`, R `4.5.3` and tradeSeq `1.24.0`.
- The release tree contains no credential-like strings, private key files or machine-specific absolute paths detected by the release scan.

## Required July archive contents

- E/V v2 schemas, validator, CLI integration and schema tests.
- Controlled-simulation calibration outputs, truth key, predictions, ambiguity/selective-coverage tables and run configuration.
- Retrospective development/tuning/shifted-audit packet partitions, all current-task baseline predictions, tuning records, confusion tables, paired bootstrap differences and leakage audit.
- ZSCAPE 100-repeat 20% embryo holdout, 50-repeat balanced split-half and threshold-sensitivity outputs; exact leave-one-embryo refits are retained as a secondary check.
- SCP1064 full heavy-shuffle results, estimability audit, leave-one-unit outputs and failed guide-label controls.
- Upstream-method sensitivity tables and native-baseline execution manifests.
- Module-level synthetic scaling results with all five repeats, hardware, median and IQR metadata.
- Independent deterministic E/V assignment module, reason-code audit trails, schemas and targeted regression tests.
- Current known-source validation tables, including the GSE153056 block audit and the descriptive GSE93735 result.
- Validation-demo inputs, E/V v2 outputs, validation records and an empty-environment execution log.
- Complete source/output SHA256 manifests, full test-suite log, clean Docker-build log and CI run URL.

## Release integrity rule

Compute and review the diff between the complete analysis-lock commit and the final-release commit. Any estimator, threshold, event-family, null, data-selection or result change requires a new analysis lock and rerunning affected analyses. Documentation-only and packaging-only changes may retain the analysis lock when they are enumerated in the release audit.

## Remote verification

The GitHub release and Zenodo record both resolve to `ted-v1.0.0`. Tag-bound GitHub Actions provide the remote Linux test, packaging, schema-validation, validation-demo and no-cache Docker build records. The Zenodo deposit contains the tag-derived archive; its checksum can be checked against the release asset and manifest.
