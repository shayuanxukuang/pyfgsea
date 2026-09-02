# Figure 1 dual-lane comparison

This directory recalculates the PyFgsea/R-fgsea comparisons used in Figure 1.
The active GitHub workflow runs installed wheels and the matching R reference
versions, then derives all reported metrics from the pathway-level output.

## Reference lanes

| Lane | Installed PyFgsea | Module version | R | Bioconductor | fgsea |
| --- | --- | --- | --- | --- | --- |
| `legacy` | 0.1.4 | 0.1.3 | 4.4.3 | 3.20 | 1.32.2 |
| `current` | 0.2.0 | 0.2.0 | 4.6.0 | 3.23 | 1.38.0 |

The legacy module's internal `0.1.3` declaration is part of the published
0.1.4 packaging state. The lanes answer different questions: the legacy lane
reproduces the paper's comparison, while the current lane evaluates the
repaired implementation against current fgsea. They are not pooled.

## Inputs and parameters

Both lanes use the committed `publication_main` and `ties_predeclared` inputs.
The first reconstructs the paper's test-data generator. The second is a
separately labelled tie-sensitivity case and is not a paper input.

Both PyFgsea calls use `min_size=15`, `max_size=500`, `sample_size=101`,
`seed=1`, `nperm_nes=1800`, `eps=1e-50`, and one thread. Both R calls use
`set.seed(314)`, `scoreType="std"`, and `nproc=1`.

The current lane additionally uses `mode="aligned"`, `score_type="std"`,
`bin_width=0`, `tie_policy="gene_id"`, and `nperm_simple=1000`. The legacy lane
uses the arguments available in the published 0.1.4 API.

## Active workflow

`.github/workflows/figure1.yml` performs the following steps:

1. install the public 0.1.4 wheel and the verified 0.2.0 wheel in separate
   Python 3.11.9 environments;
2. start the fgsea 1.32.2 and 1.38.0 reference images and verify their reported
   R, Bioconductor, and fgsea versions;
3. execute both Python and R engines on both frozen scenarios;
4. require complete pathway sets, matching pathway sizes, finite statistics,
   valid p-values, and unchanged numerical parameters; and
5. calculate agreement, overlap, tail, runtime, and memory summaries from the
   raw rows.

`run_functional_lane.py` and `compare_functional.py` are the active runners.
Their pass/fail decisions use installed versions, imports, executable reference
sessions, result schemas, row completeness, parameters, and numerical checks.
SHA-256 values are written only as provenance fields and do not decide whether
the workflow passes.

To run the scripts manually, use the matching R image for each lane:

```bash
FGSEA_REFERENCE_VERSION=1.32.2 python run_functional_lane.py \
  --lane legacy --output-dir /new/path/figure1-legacy

FGSEA_REFERENCE_VERSION=1.38.0 python run_functional_lane.py \
  --lane current --output-dir /new/path/figure1-current

python compare_functional.py \
  --legacy-result /new/path/figure1-legacy/lane_result.json \
  --current-result /new/path/figure1-current/lane_result.json \
  --output-dir /new/path/figure1-compared
```

## Outputs

The comparison creates:

- `figure1_pathway_level_raw.tsv`;
- `figure1_agreement_metrics.tsv`;
- `figure1_pathway_overlap.tsv`;
- `figure1_extreme_tail_cases.tsv`;
- `figure1_legacy_current_change.tsv`;
- `figure1_runtime_memory.tsv`;
- `figure1_dual_lane_agreement.png`; and
- `comparison_result.json`.

The runtime table is descriptive. PyFgsea measures the `run_gsea` call, while
the R row includes the Rscript process and the internal fgsea measurement; it
is not an equal-scope speedup estimate. The ties scenario is a same-environment
sensitivity result, not a cross-platform equivalence claim.
