
import os
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Any, Union

from .io.anndata_io import load_adata
from .io.meta_merge import merge_metadata_safe
from .preprocess.pseudotime import ensure_pseudotime
from .gsea.runner import run_core
from .anchors.select import select_anchor_pair
from .anchors.switch import find_switch_point
from .plotting.overview import plot_overview_heatmap
from .plotting.fastproof import plot_fastproof
from .windows.binning import build_anchor_matrix_from_df

logger = logging.getLogger(__name__)

def run_pipeline(
    adata: Any,
    gmt_path: str,
    output_dir: str = "results/run1",
    pseudotime_key: str = "dpt_pseudotime",
    force_rerun: bool = False,
    window_size: int = 800,
    step: int = 50,
    min_size: int = 15
) -> pd.DataFrame:
    """
    Orchestrates the full GSEA trajectory analysis pipeline.
    
    Args:
        adata: AnnData object or path to .h5ad
        gmt_path: Path to GMT file
        output_dir: Directory to save results
        pseudotime_key: Column name in adata.obs for pseudotime
        force_rerun: If True, ignore cached results and re-run GSEA
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Data Prep
    adata = ensure_pseudotime(adata, key=pseudotime_key)
    
    # Step 2: Compute or Load GSEA
    gsea_csv = out_path / "gsea_results_core.csv"
    if gsea_csv.exists() and not force_rerun:
        logger.info(f"Loading cached results from {gsea_csv}")
        gsea_df = pd.read_csv(gsea_csv)
    else:
        logger.info("Running core GSEA analysis...")
        gsea_df = run_core(
            adata, 
            gmt_path, 
            out_csv=str(gsea_csv), 
            pseudotime_key=pseudotime_key,
            window_size=window_size,
            step=step,
            min_size=min_size
        )

    if gsea_df.empty:
        logger.warning("GSEA yielded no results. Check gene coverage or GMT file.")
        return gsea_df

    # Step 3: Analysis & Visualization
    _analyze_anchors_and_plot(gsea_df, out_path)

    return gsea_df

def _analyze_anchors_and_plot(df: pd.DataFrame, out_path: Path):
    """Internal helper to handle anchor selection and plotting."""
    early, late, score, stats = select_anchor_pair(
        df, 
        out_report=str(out_path / "anchor_report.csv")
    )
    
    if not (early and late):
        logger.warning("Skipping plots: No valid anchor pair found.")
        return

    logger.info(f"Best Pair: {early} vs {late} (Score={score:.3f})")
    logger.info(f"Stats: Corr={stats['Corr']:.3f}, Sep={stats['Sep']:.3f}, Range={stats['Range']:.3f}")
    
    # 4. Plots
    plot_overview_heatmap(df, str(out_path))
    plot_fastproof(df, early, late, str(out_path))
    
    # 5. Switch Point
    try:
        import numpy as np
        b_grid = np.linspace(0, 1, 61)
        mat, centers = build_anchor_matrix_from_df(df, [early, late], bins=b_grid, value_col="NES_smooth")
        pt_switch, _ = find_switch_point(mat.loc[early].values, mat.loc[late].values, centers)
        logger.info(f"Switch Point: {pt_switch:.3f}")
    except Exception as e:
        logger.error(f"Failed to calculate switch point: {e}")

# Maintain backward compatibility alias
run = run_pipeline
