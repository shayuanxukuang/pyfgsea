# fgsea 1.38 alignment boundary

Status: the clean PyFgsea `0.2.0rc5` candidate defines the reference and
artifact gates below. The clean candidate has not yet produced accepted
cross-platform artifacts, reference-image digests, Figure 1 receipts, a formal
Figure 2 receipt, or a manuscript-impact decision.

## Two non-interchangeable reference lanes

| Lane | PyFgsea | R | Bioconductor | fgsea | Permitted use |
| --- | --- | --- | --- | --- | --- |
| `legacy_publication` | 0.1.4 | 4.4.3 | 3.20 | 1.32.2 | Reproduce the comparison declared by the published paper |
| `current_conformance` | 0.2.0rc5 candidate; 0.2.0 target | 4.6.0 | 3.23 | 1.38.0 | Evaluate the repaired implementation against the current reference |

The exact URLs, source hashes, source commits, base-image digests and pending
receipt fields are recorded in `reference_manifest.json`. The corresponding
container definitions are `Dockerfile.reference-fgsea-1.32.2` and
`Dockerfile.reference-fgsea-1.38.0`.

“Aligned” means comparison under a declared lane with matched score type,
pathway universe, size filters, sample size, epsilon, seeds and result mode. It
does not mean bitwise identity between different random-number generators, and
it does not retroactively reinterpret the publication's 1.32.2 comparison as a
1.38.0 comparison.

## Evidence gates

An accepted lane receipt must bind all of the following:

- a clean PyFgsea commit and tree;
- an annotated release tag;
- sdist, wheel and installed native-core SHA-256 values;
- R, Bioconductor and fgsea versions;
- fgsea source and OCI image digests;
- input hashes, exact parameters, command lines and output hashes;
- unresolved/failure-state counts rather than silently omitted pathways.

The dual-lane Figure 1 pipeline is defined in `repro/figure1_dual_lane/`. Its
existence and lightweight tests are not substitutes for executing both lanes
inside their verified environments.

## Manuscript decision boundary

Figure 1, the accepted Figure 2 contract, null calibration, deep-tail and tie
cases, Bland–Altman analysis, window sensitivity, pathway-set overlap, thread
scaling, runtime and memory must be recalculated from verified artifacts. Only
then can the project distinguish among a software update with unchanged paper
conclusions, an author correction with changed values but unchanged biological
conclusions, or a corrigendum affecting key claims.

No source-level test result alone closes that decision.
