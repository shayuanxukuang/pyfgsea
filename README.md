# PyFgsea: GSEA in Python and Rust

**PyFgsea** is a Python-first library for Gene Set Enrichment Analysis (GSEA), powered by a Rust backend. Its aligned mode implements a multilevel estimator intended for comparison with the reference R `fgseaMultilevel` methodology.

The current development version is the PEP 440 prerelease `0.2.0rc3`. Once its release gates pass, the corresponding Git prerelease tag is `v0.2.0-rc3`; the Git tag contains a hyphen, while the Python distribution version does not.

## Key Features
- **Rust-backed execution**: Native enrichment-score and null-estimation kernels.
- **Numerical diagnostics**: ES, NES, tail-error and failure-state fields support explicit reference-lane auditing; formal RC3 conformance is still pending.
- **Trajectory analysis**: Rolling-window GSEA for single-cell trajectory analysis.
- **Python-first API**: Designed for seamless integration with pandas and scanpy workflows.

## Installation

### Prerequisites
- Python 3.9+
- Rust toolchain (stable)
- RC3 CI matrix: Linux/Windows with Python 3.9–3.13 (gate execution pending)

### Install from Source

**Recommended (Standard):**
```bash
git clone https://github.com/shayuanxukuang/pyfgsea.git
cd pyfgsea
pip install --upgrade pip maturin
pip install .           # Core GSEA installation
# OR
pip install -e .        # Editable/Development mode

# Add single-cell trajectory and plotting support when needed
pip install ".[trajectory]"
```

**For Rust Development:**
```bash
maturin develop --release
```

## Quick Start

Here is a minimal example to get you running in seconds.

```python
import pandas as pd
import pyfgsea

# 1. Prepare Data
# Option A: DataFrame (Customizable column names)
df = pd.DataFrame({
    'gene_name': ['GeneA', 'GeneB', 'GeneC', 'GeneD', 'GeneE'],
    'score': [2.5, 1.8, 0.5, -0.2, -1.5]
})

# Option B: Series (Index=Gene, Value=Score)
# scores = df.set_index('gene_name')['score']

# 2. Define Pathways (Dict of List)
pathways = {
    'Pathway_1': ['GeneA', 'GeneB'],
    'Pathway_2': ['GeneD', 'GeneE']
}

# 3. Run GSEA
# For DataFrame: specify gene_col and score_col
res = pyfgsea.run_gsea(
    data=df,
    gmt=pathways,
    gene_col='gene_name',
    score_col='score',
    min_size=1,     # Lower for toy example
    max_size=500,
    nperm_nes=100
)

# 4. View Results (column names are case-sensitive)
print(res[["Pathway", "NES", "P-value", "padj", "ES"]])
```

### Input Formats
- **DataFrame**: Must contain a gene column and a ranking score column. Specify via `gene_col` and `score_col`.
- **Series**: Index must be gene names, values must be ranking scores.
- **Deduplication**: Default strategy (`dedup_genes='max_abs'`) retains the gene entry with the highest absolute score.

### Output Columns
The principal columns returned by `run_gsea` are:
- `Pathway`: Pathway name
- `P-value`: Estimated P-value
- `padj`: Benjamini-Hochberg adjusted P-value
- `ES`: Enrichment Score
- `NES`: Normalized Enrichment Score
- `Size`: Size of the pathway after filtering
- `log2err`: P-value estimation error metric
- `n_levels`: Multilevel depth used for the result
- `pval_capped`: Whether p-value hit the eps floor

Additional diagnostic columns include `log_pval`, `observed_pathway_size`, `null_curve_size`, `size_binned`, `approximate`, `status`, `termination_reason`, acceptance-rate summaries, `ranking_hash`, and `algorithm_revision`.

### Aligned and fast modes

`mode="aligned"` is the default and requires exact pathway sizes
(`bin_width=0`). `mode="fast"` first runs an empirical precheck (default
`precheck_n=64`, `precheck_eps=0.005`): a shallow tail uses the simple estimate,
while a deeper tail proceeds to the multilevel compound ruler. The
`termination_reason` field records which route was taken. Fast-mode results are
marked approximate and must not be reported as aligned-mode results.

## Trajectory (rolling-window) GSEA along pseudotime

PyFgsea supports rolling-window preranked GSEA to track pathway activity changes along single-cell trajectories.
It reuses a stateful runner across windows.

<p align="center">
  <img src="https://raw.githubusercontent.com/shayuanxukuang/pyfgsea/v0.2.0-rc3/docs/assets/trajectory_demo.png" width="900" alt="Rolling-window trajectory demo">
</p>

### One-command demo

> **Note**: Install the trajectory extra first: `pip install "pyfgsea[trajectory]"`.
> This is a synthetic toy dataset for demonstration only (not biological data).

```bash
# from the repo root
python examples/trajectory_demo.py \
  --adata repro/data/toy_trajectory.h5ad \
  --pseudotime-key dpt_pseudotime \
  --outdir results/
```

### What the plot shows
- **Cells are ordered by pseudotime.**
- **A sliding window moves along this ordering** (window size = `window_size`, step = `step`).
- **Points denote window centers** (`pt_mid`), and curves are lightly smoothed for display.
- **Preranked GSEA is run per window** to obtain ES/NES and FDR.

### Ranking statistic
`mean(expression in window) − mean(expression outside window)` on log1p-normalized expression.

### Key parameters
- `window_size`: number of cells per window (larger = smoother, smaller = higher resolution)
- `step`: stride between windows (smaller = more windows)
- `min_size`, `max_size`: gene set size filters
- `eps`, `sample_size`: multilevel sampling controls
- Threading: the current Python API does not expose an `n_threads` keyword.
Defaults: `window_size=500`, `step=50`, `min_size=15`, `max_size=500`, `nperm_nes=2000`, `score_type="std"`, exact pathway sizes (`bin_width=0`), NES caching off (`use_nes_cache=False`), and `seed=42`.

### Outputs
- `results/trajectory_demo.png`: marker expression and pathway NES dynamics along pseudotime
- `results/trajectory_gsea_table.tsv`: per-window GSEA results (ES, NES, P-value, padj, window index)

### Notes
- Synthetic toy dataset for demonstration only (not biological data).
- Expected trend: Pathway_Up increases and Pathway_Down decreases along pseudotime.

## Reproducing Paper Results

The `repro/` directory contains the versioned reproduction definitions. PyFgsea 0.2.0 keeps two non-interchangeable reference lanes: the publication audit uses R fgsea 1.32.2, while current conformance uses R fgsea 1.38.0. See the [alignment boundary](https://github.com/shayuanxukuang/pyfgsea/blob/v0.2.0-rc3/docs/fgsea-1.38-alignment.md), [0.2.0 release note](https://github.com/shayuanxukuang/pyfgsea/blob/v0.2.0-rc3/docs/releases/0.2.0.md), and [dual-lane Figure 1 protocol](https://github.com/shayuanxukuang/pyfgsea/blob/v0.2.0-rc3/repro/figure1_dual_lane/README.md).

### Reproducibility (with/without R)

Some scripts can exercise the Python implementation without R. Such runs are partial checks, not reference-comparison or manuscript-artifact receipts. A formal comparison requires the pinned R/Bioconductor/fgsea environment and the artifact provenance described by the dual-lane protocol.

Recommended setup: `pip install -e ".[trajectory]"`. The core install keeps only
the numerical/dataframe runtime plus the lightweight command entry dependency;
single-cell and plotting packages are isolated in the `trajectory` extra.

**Core Commands:**
```bash
python repro/fig_ablation_tail.py
python repro/fig_supp_tail_consistency.py
```

### Exploratory unpinned R check (optional)

The generic command below is useful only for local exploration. It is not a
formal baseline because it does not pin fgsea 1.32.2 or 1.38.0. Formal
comparisons must use the two frozen reference-lane definitions linked above.

```r
# In R console:
install.packages("BiocManager")
BiocManager::install("fgsea")
```

> **Note**: If R is not found, any skipped comparison must be reported as incomplete; a Python-only result does not close either reference lane.

For full details, see the [reproduction guide](https://github.com/shayuanxukuang/pyfgsea/blob/v0.2.0-rc3/repro/README.md).

## Citation
If you use PyFgsea in academic work, please cite:

> Wang K, Shi H. PyFgsea: a Rust-powered, fgseaMultilevel-aligned GSEA framework with rolling-window enrichment along single-cell trajectories. *Bioinformatics*. 2026;42(5):btag257. doi:[10.1093/bioinformatics/btag257](https://doi.org/10.1093/bioinformatics/btag257).

- Article: <https://doi.org/10.1093/bioinformatics/btag257>
- Source code: <https://github.com/shayuanxukuang/pyfgsea>
- PyPI: <https://pypi.org/project/pyfgsea/>
- Zenodo archive: <https://doi.org/10.5281/zenodo.19446446>

## License
MIT License. See [LICENSE](LICENSE) for details.
