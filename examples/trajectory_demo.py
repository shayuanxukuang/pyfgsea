"""Run a real rolling-window GSEA example with precomputed pseudotime."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad

from pyfgsea import plot_trajectory_heatmap, run_trajectory_gsea


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run trajectory GSEA from an AnnData file and a GMT file."
    )
    parser.add_argument("--adata", required=True, type=Path)
    parser.add_argument("--gmt", required=True, type=Path)
    parser.add_argument("--pseudotime-key", default="dpt_pseudotime")
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    parser.add_argument("--window-size", type=int, default=500)
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--min-size", type=int, default=15)
    parser.add_argument("--max-size", type=int, default=500)
    parser.add_argument("--sample-size", type=int, default=101)
    parser.add_argument("--nperm-nes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.adata.is_file():
        raise FileNotFoundError(args.adata)
    if not args.gmt.is_file():
        raise FileNotFoundError(args.gmt)

    adata = ad.read_h5ad(args.adata)
    if args.pseudotime_key not in adata.obs:
        raise ValueError(
            f"Pseudotime column is missing: {args.pseudotime_key}. "
            "Precompute it or call run_trajectory_gsea with an explicit DPT root."
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    table_path = args.outdir / "trajectory_gsea_table.tsv"
    figure_path = args.outdir / "trajectory_demo.png"
    result = run_trajectory_gsea(
        adata,
        gmt_path=str(args.gmt),
        pseudotime_key=args.pseudotime_key,
        window_size=args.window_size,
        step=args.step,
        min_size=args.min_size,
        max_size=args.max_size,
        sample_size=args.sample_size,
        nperm_nes=args.nperm_nes,
        seed=args.seed,
        score_type="std",
        bin_width=0,
        use_nes_cache=False,
    )
    if result.empty:
        raise RuntimeError("Trajectory GSEA returned no pathways")

    result.to_csv(table_path, sep="\t", index=False)
    plot_trajectory_heatmap(result, save_path=str(figure_path))
    print(f"Results: {table_path}")
    print(f"Figure: {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
