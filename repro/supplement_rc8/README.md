# Figure 1 supplementary summaries

`assemble_figure1.py` derives descriptive supplementary tables and a
Bland–Altman figure from a completed functional Figure 1 comparison. It reads:

- `comparison_result.json`;
- `figure1_pathway_level_raw.tsv`;
- `figure1_runtime_memory.tsv`.

Run it after the legacy and current installed-wheel lanes have completed:

```bash
python repro/supplement_rc8/assemble_figure1.py \
  --figure1-result-dir /path/to/figure1-compared \
  --output-dir /path/to/figure1-supplement
```

The assembler checks lane versions, functional execution flags, row counts,
probability ranges, recomputed differences, and timing scopes. It does not use
file hashes to accept or reject a result. SHA-256 values in the manifest are
provenance records only.

The outputs cover NES and transformed-P Bland–Altman summaries, top-10 and FDR
set overlap, the ten deepest tail cases per lane/scenario, and single-run timing
and memory descriptions. Bland–Altman limits are descriptive rather than
equivalence bounds. The R and Python timing scopes differ, so their ratio is not
an equal-scope speed estimate.

Null calibration is a separate computation because it must execute the
installed package over multiple real permutation replicates:

```bash
python repro/supplement_rc8/run_null_calibration.py \
  --output-dir /path/to/null-calibration
```

The default run uses 20 predetermined seeds. Each replicate independently
permutes the fixed Figure 1 scores across gene labels, preserves the score
distribution and pathway definitions, and calls the installed 0.2.0rc8 wheel.
It reports per-replicate and pooled descriptive P-value summaries plus QQ and
ECDF plots. It does not encode an equivalence margin, a statistical acceptance
threshold, or an inferential KS test over overlapping pathways. Unit-test
fixtures test validation logic only and are never written as formal results.
