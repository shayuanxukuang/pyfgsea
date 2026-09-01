# fgsea reference alignment

PyFgsea uses two named R fgsea reference lanes. Each lane answers a different
question and must be reported separately.

| Lane | PyFgsea | R | Bioconductor | fgsea | Purpose |
| --- | --- | --- | --- | --- | --- |
| `legacy_publication` | 0.1.4 | 4.4.3 | 3.20 | 1.32.2 | Reproduce the comparison declared in the published paper |
| `current_conformance` | 0.2.0rc7 candidate; 0.2.0 target | 4.6.0 | 3.23 | 1.38.0 | Test the repaired implementation against the current reference |

Do not pool these lanes or describe them as one unspecified fgsea reference.
A result against fgsea 1.32.2 does not establish conformance with fgsea 1.38.0,
and a current-reference result does not rewrite the paper's historical
comparison.

## Aligned comparison

An aligned comparison uses the declared lane and matches:

- score type and result mode;
- ranked genes and pathway membership;
- minimum and maximum pathway sizes;
- exact or binned pathway-size handling;
- sample size, simple-permutation count, and epsilon;
- random seeds and thread count; and
- unresolved and failure-state handling.

Aligned comparison does not mean bitwise identity between different
random-number generators. Numerical results must be assessed with the declared
ES, NES, p-value, error, overlap, runtime, and memory metrics.

## Reference definitions

Machine-readable versions, source URLs, source SHA-256 values, commits,
base-image digests, and pending run-record fields are stored in
`reference_manifest.json`.

The container definitions are:

- `Dockerfile.reference-fgsea-1.32.2` for `legacy_publication`;
- `Dockerfile.reference-fgsea-1.38.0` for `current_conformance`.

Each container verifies its R, Bioconductor, and fgsea versions and writes its
resolved package versions and session details into the OCI image.

## Required records

A complete reference run requires:

1. a clean worktree at the recorded PyFgsea commit and tree;
2. an annotated release-candidate tag that peels to that commit;
3. source tests and Rust tests;
4. an sdist built from the recorded source;
5. a wheel built from that verified sdist;
6. clean-environment installation of the wheel;
7. installed version and native-core checks;
8. passing installed-wheel tests;
9. the exact R, Bioconductor, and fgsea environment;
10. the fgsea source and built OCI image digests;
11. exact parameters and command lines; and
12. SHA-256 values for inputs, artifacts, native cores, and outputs.

The run record must include unresolved and failure-state counts. Missing pathways
or missing R output cannot be treated as agreement.

The Figure 1 implementation is in
[`repro/figure1_dual_lane/`](../repro/figure1_dual_lane/README.md). Lightweight
tests validate its contracts, but they do not execute the two R reference
lanes.

## Current limitations

Figure 1, Figure 2, null-calibration, deep-tail, tie, Bland-Altman,
window-sensitivity, pathway-overlap, scaling, runtime, and memory results remain
separate result sets. Source-level or wheel-level tests alone do not
determine whether published numerical or biological conclusions change.
