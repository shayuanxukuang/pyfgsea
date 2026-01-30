# PyFgsea: High-Performance GSEA in Python & Rust

**PyFgsea** is a hyper-optimized Python library for Gene Set Enrichment Analysis (GSEA), powered by a Rust backend. It implements the "Multilevel Split Monte Carlo" algorithm to achieve orders-of-magnitude speedups over existing Python implementations while maintaining strict statistical equivalence to the reference R-fgsea package.

## Key Features
- **Speed**: Up to 100x faster than pure Python implementations.
- **Precision**: Exact alignment with R-fgsea P-values (verified down to $10^{-16}$).
- **Scalability**: Efficient rolling-window GSEA for single-cell trajectory analysis.
- **Ease of Use**: Drop-in replacement for standard GSEA workflows.

## Installation

### Prerequisites
- Python 3.8+
- Rust toolchain (stable)

### Install from Source
```bash
git clone https://github.com/yourusername/pyfgsea.git
cd pyfgsea
pip install -r requirements.txt
maturin develop --release
```

## Reproducing Paper Results

We provide a complete suite of reproduction scripts in the `repro/` directory.

### Reproducibility (with/without R)

The core reproduction scripts can run in pure Python mode (skipping R benchmarks if R is missing) or full comparison mode.

**Core Commands:**
```bash
python repro/fig_ablation_tail.py
python repro/fig_supp_tail_consistency.py
```

> **Note**: To run the full R baseline comparison, ensure `Rscript` is in your PATH and the `fgsea` package is installed in R. If not found, these scripts will automatically skip the R comparison steps and only run the Python parts.

For full details, see [repro/README.md](repro/README.md).

## License
MIT License. See [LICENSE](LICENSE) for details.
