# PyFgsea: GSEA in Python and Rust

PyFgsea runs preranked Gene Set Enrichment Analysis (GSEA) from Python with a
Rust numerical core. It supports single analyses and rolling-window pathway
analysis along an ordered trajectory.

The current stable release is `0.2.0`.

## Features

- Rust-backed enrichment-score and null-estimation kernels.
- `mode="aligned"` for comparisons with R `fgseaMultilevel` under an explicit
  parameter contract.
- Deterministic gene-ID tie ordering and exact pathway-size nulls.
- Explicit ES, NES, p-value, tail-error, unresolved, and failure diagnostics.
- Rolling-window GSEA for single-cell or other ordered trajectories.
- pandas-friendly input and output.

## Installation

PyFgsea requires Python 3.9 or newer. Building from source also requires a Rust
toolchain.

```bash
git clone https://github.com/shayuanxukuang/pyfgsea.git
cd pyfgsea
python -m pip install --upgrade pip maturin
python -m pip install .
```

Install trajectory and plotting dependencies only when needed:

```bash
python -m pip install ".[trajectory]"
```

For Rust development:

```bash
maturin develop --release --locked
```

## Quick start

```python
import pandas as pd
import pyfgsea

ranks = pd.DataFrame(
    {
        "gene_name": ["GeneA", "GeneB", "GeneC", "GeneD", "GeneE"],
        "score": [2.5, 1.8, 0.5, -0.2, -1.5],
    }
)

pathways = {
    "Pathway_1": ["GeneA", "GeneB"],
    "Pathway_2": ["GeneD", "GeneE"],
}

result = pyfgsea.run_gsea(
    data=ranks,
    gmt=pathways,
    gene_col="gene_name",
    score_col="score",
    min_size=1,
    max_size=500,
    nperm_nes=100,
)

print(result[["Pathway", "NES", "P-value", "padj", "ES", "status"]])
```

### Inputs

- A `DataFrame` must contain a gene column and a ranking-score column. Select
  them with `gene_col` and `score_col`.
- A `Series` uses its index as gene identifiers and its values as scores.
- `dedup_genes="max_abs"`, the default, keeps the duplicate entry with the
  largest absolute score.
- Pathways may be supplied as a mapping from pathway names to gene lists.

### Outputs

The main result columns are:

- `Pathway`, `Size`, `ES`, `NES`, `P-value`, and `padj`;
- `log2err`, `log_pval`, `n_levels`, and `pval_capped`;
- `status` and `termination_reason` for resolved, unresolved, and failed rows;
- `observed_pathway_size`, `null_curve_size`, `size_binned`, and
  `approximate`;
- `ranking_hash` and `algorithm_revision` for provenance.

Pathways with unresolved or failed estimates remain visible in the output.
They must not be silently removed before reporting or comparison.

## Aligned and fast modes

`mode="aligned"` is the default. It uses exact pathway sizes (`bin_width=0`)
and is the mode intended for the current R fgsea 1.38.0 conformance lane.

`mode="fast"` starts with an empirical precheck. A shallow tail may return the
simple estimate; a deeper tail proceeds to the multilevel compound ruler. Fast
results are marked `approximate=True`, and `termination_reason` records the
route. Do not report fast-mode output as aligned-mode output.

The deprecated `score_type="two_sided_abs"` and low-level empirical tail
helpers remain available for bounded compatibility work. They are approximate
and are not equivalent to R `fgseaMultilevel(scoreType="std")`.

## Rolling-window trajectory GSEA

PyFgsea can rank genes and run GSEA repeatedly across overlapping windows of
cells ordered by pseudotime. The default ranking statistic is:

```text
mean(expression in window) - mean(expression outside window)
```

Run the included synthetic example:

```bash
python examples/trajectory_demo.py \
  --adata repro/data/toy_trajectory.h5ad \
  --gmt repro/data/toy_pathways.gmt \
  --pseudotime-key dpt_pseudotime \
  --window-size 100 \
  --step 50 \
  --outdir results/
```

The example writes:

- `results/trajectory_demo.png`;
- `results/trajectory_gsea_table.tsv`.

Trajectory defaults are `window_size=500`, `step=50`, `min_size=15`,
`max_size=500`, `nperm_nes=2000`, `score_type="std"`, exact pathway sizes,
NES caching off, and `seed=42`. The Python API does not currently expose an
`n_threads` keyword.

## Reproducing the paper

The paper used PyFgsea 0.1.4 with R fgsea 1.32.2. Current comparisons use
PyFgsea 0.2.0 with R fgsea 1.38.0. Keep these two reference lanes separate.

See:

- [fgsea reference alignment](docs/fgsea-1.38-alignment.md);
- [0.2.0 release note](docs/releases/0.2.0.md);
- [Figure 1 dual-lane protocol](repro/figure1_dual_lane/README.md).

## Current limitations

- The formal Figure 1 comparison covers the publication input and a separate
  tie-heavy sensitivity case; it is not a universal equivalence claim.
- Figure 2 is a descriptive analysis of a processed, mixed-assignment subset;
  it is not a control-versus-mutant comparison.
- Fast mode and the legacy empirical helpers are approximate.
- Runtime and peak-memory values depend on the recorded hardware and software
  environment.

## Citation

If you use PyFgsea in academic work, cite:

> Wang K, Shi H. PyFgsea: a Rust-powered, fgseaMultilevel-aligned GSEA framework with rolling-window enrichment along single-cell trajectories. *Bioinformatics*. 2026;42(5):btag257. doi:[10.1093/bioinformatics/btag257](https://doi.org/10.1093/bioinformatics/btag257).

- Article: <https://doi.org/10.1093/bioinformatics/btag257>
- Source: <https://github.com/shayuanxukuang/pyfgsea>
- PyPI: <https://pypi.org/project/pyfgsea/>
- Zenodo concept DOI: <https://doi.org/10.5281/zenodo.19446445>

## License

MIT License. See [LICENSE](LICENSE).
