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

### Quick Start
To generate the key "Tail Consistency" figure (Figure Sx):
```bash
python repro/fig_supp_tail_consistency.py
```

For full details, see [repro/README.md](repro/README.md).

## License
MIT License. See [LICENSE](LICENSE) for details.
