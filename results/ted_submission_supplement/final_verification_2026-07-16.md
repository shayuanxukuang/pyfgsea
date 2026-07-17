# TED final local verification - 2026-07-16

This record separates completed local checks from release gates that require an immutable public commit or an external service.

## Completed local checks

- Statistical/schema regression set: **71 passed, 10 warnings in 11.34 s**. The warnings are `anndata` index-coercion warnings, not test failures.
- Event-v2 schema: package and repository mirrors are byte-identical; SHA-256 `AD4382783B7A71ED9F8589A4F92A3C31E1910F86D3C6FA2F5E2BA04E82F09489`. `event_test_status` separates `not_run`, `run_not_supported` and `run_supported`; optional resampling stability and upstream-disagreement fields are semantically checked, and a disagreement flag forbids E2.
- Factorized benchmark: 675 dynamic packets fully crossed five biological modes, three artifact states, three identifiability states, three block counts and five V states. Six gate ablations, a 250-case reason-code challenge and seven invalid schema combinations completed.
- Adaptive-window benchmark: 36 balanced scenarios and 696 profile replicates. The full candidate-window search was rerun inside each permutation; 30 scenarios used 1,000 permutations and six critical scenarios used 5,000.
- ZSCAPE stability: 100 holdouts, 50 split halves and five retained-fraction points with 20 repeats each completed; per-event selection and effect-rank tables were written. Real-data upstream sensitivity completed 12 combinations on two datasets.
- Installed-wheel check: wheel SHA-256 `94EEF654D76D92D91E14234ABFE71C587404B8C460E8423F9EBAAC99C6BB5DB5`. A fresh Python 3.11.15 environment imported `pyfgsea 0.1.4` from isolated site-packages under `python -I`; the installed `ted` entry point passed activity-v1 and event-v2 validation.
- Main manuscript PDF: **27 pages**, SHA-256 `EEA1EC85C786C18601E78F9EE738991BBDFFC1EE5EFF1C19EC9322FCB06118CE`.
- Supplementary Information PDF: **25 pages**, SHA-256 `6FEBEA3E8D3E186E58806653FF2573A934FE96BB6FCDEC04A8928BBB6C8A6C21`.
- PDF audit: no LaTeX errors, undefined references, overfull boxes or Type 3 fonts were detected. The title page, revised Figures 3 and 4, factorized-benchmark pages and new result-table pages were rendered to PNG and visually inspected.

## Complete-suite result

The prior post-revision command covering the complete `tests/` directory remained CPU-active but exceeded the **3601-second** execution limit. It emitted neither a final pytest summary nor a failure traceback. This is recorded as **timeout/incomplete**, not pass or fail. A completed clean-environment run and remote CI on the immutable release commit remain mandatory before submission.

## Statistical interpretation

The primary procedure is now per-event max-over-window permutation p values followed by BH across events. Family-wide maxT is reported separately as FWER-adjusted and receives no second BH correction. In 300 clean-null profile replicates, FWER was 0.317 for naive selected-window p plus BH, 0.033 for the primary procedure and 0.080 for family-wide maxT. In 72 clean-signal replicates, FDP/power was 0.218/0.347, 0.016/0.167 and 0.034/0.375, respectively. Under 324 artifact-stress replicates, primary-procedure FDP rose to 0.114, making clear that multiplicity correction does not substitute for artifact gates.

## Unverified external release gates

- The neutral `ted-v1.0.0` tag and immutable final commit do not yet exist.
- A version-specific Zenodo DOI for the exact submitted release has not been minted.
- The current main and baseline containers have not both been rebuilt and executed from a clean context.
- Remote CI has not run on the immutable July 2026 release commit.
- Repository-remote metadata still differs from the manuscript's public project URL.

Until these gates are closed and the same tag, commit and DOI appear across README, manuscript, Supplementary Information, schemas and archive manifests, the package is a locally verified submission candidate rather than a submission-ready public release.
