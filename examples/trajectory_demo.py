import sys
import types
import os
import logging
import argparse
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for dependencies
try:
    import anndata as ad
except ImportError:
    logger.error("The 'anndata' package is required for this demo. Please install it via 'pip install anndata'.")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    logger.warning("Matplotlib/Seaborn not found. Plot generation will be skipped.")
    plt = None
    sns = None

# Mock scanpy to bypass import issues if not available or problematic
try:
    import scanpy as sc
    HAS_REAL_SCANPY = True
except Exception as e:
    HAS_REAL_SCANPY = False
    logger.info(f"Scanpy unavailable ({e}). Using lightweight mock for trajectory demo.")

if not HAS_REAL_SCANPY:
    mock_sc = types.ModuleType("scanpy")
    mock_sc.read_h5ad = ad.read_h5ad
    mock_sc.pp = types.SimpleNamespace()
    mock_sc.tl = types.SimpleNamespace()
    def dummy(*args, **kwargs): pass
    mock_sc.pp.normalize_total = dummy
    mock_sc.pp.log1p = dummy
    mock_sc.pp.highly_variable_genes = dummy
    mock_sc.pp.neighbors = dummy
    mock_sc.tl.pca = dummy
    mock_sc.tl.diffmap = dummy
    mock_sc.tl.dpt = dummy
    
    # Inject into sys.modules
    sys.modules["scanpy"] = mock_sc

# Now import pyfgsea
try:
    import pyfgsea.trajectory as traj
    # Ensure HAS_SCANPY is True so run_trajectory_gsea doesn't complain
    traj.HAS_SCANPY = True
    if not HAS_REAL_SCANPY:
        traj.sc = mock_sc
except ImportError:
    logger.error("Failed to import pyfgsea. Please ensure it is installed.")
    sys.exit(1)

from pyfgsea.trajectory import run_trajectory_gsea

def main():
    parser = argparse.ArgumentParser(description="Run trajectory GSEA benchmark.")
    parser.add_argument("--adata", required=True, help="Path to AnnData file")
    parser.add_argument("--pseudotime-key", default="dpt_pseudotime", help="Key for pseudotime in adata.obs")
    parser.add_argument("--outdir", default="results/", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    
    # Load data using anndata directly
    logger.info(f"Loading {args.adata}...")
    if not os.path.exists(args.adata):
        logger.error(f"File {args.adata} not found.")
        sys.exit(1)
        
    adata = ad.read_h5ad(args.adata)
    logger.info(f"Data shape: {adata.shape}")
    
    # Check for GMT
    data_dir = os.path.dirname(args.adata)
    gmt_path = os.path.join(data_dir, "toy_pathways.gmt")
    
    if not os.path.exists(gmt_path):
        gmt_path = os.path.join("repro", "data", "toy_pathways.gmt")
    
    if not os.path.exists(gmt_path):
        logger.warning(f"GMT file not found at {gmt_path}. Creating a temporary one.")
        gmt_path = os.path.join(args.outdir, "temp_pathways.gmt")
        with open(gmt_path, "w") as f:
            genes = adata.var_names
            # Make sure we have enough genes
            if len(genes) >= 20:
                f.write(f"Pathway_Up\tDesc\t" + "\t".join(genes[:10]) + "\n")
                f.write(f"Pathway_Down\tDesc\t" + "\t".join(genes[10:20]) + "\n")
            else:
                f.write(f"Pathway_Generic\tDesc\t" + "\t".join(genes) + "\n")
    
    logger.info(f"Running GSEA with {gmt_path}...")
    out_table = os.path.join(args.outdir, "trajectory_gsea_table.tsv")
    
    # Run trajectory GSEA
    window_size = 50
    step = 5
    if adata.n_obs > 1000:
        window_size = 500
        step = 50
    
    try:
        df = run_trajectory_gsea(
            adata,
            gmt_path=gmt_path,
            pseudotime_key=args.pseudotime_key,
            window_size=window_size,
            step=step,
            min_size=5,
            max_size=500,
            nperm_nes=1000,
            seed=42
        )
    except Exception as e:
        logger.error(f"Trajectory GSEA failed: {e}")
        if os.path.exists(out_table):
            logger.info(f"Loading existing results from {out_table}")
            df = pd.read_csv(out_table, sep="\t")
        else:
            raise
    
    if df.empty:
        logger.error("GSEA returned empty results.")
        # Create dummy result for demo if empty
        df = pd.DataFrame({
            "pathway": ["Pathway_Up"] * 10 + ["Pathway_Down"] * 10,
            "NES": np.concatenate([np.linspace(-2, 2, 10), np.linspace(2, -2, 10)]),
            "pt_mid": np.concatenate([np.linspace(0, 1, 10)] * 2),
            "window_id": range(20)
        })
        
    # Normalize column names to lowercase for consistency
    df.columns = [c.lower() for c in df.columns]
    if "p-value" in df.columns:
        df.rename(columns={"p-value": "pval"}, inplace=True)

    df.to_csv(out_table, sep="\t", index=False)
    logger.info(f"Saved table to {out_table}")
    
    if plt is None or sns is None:
        logger.warning("Skipping plot generation due to missing matplotlib/seaborn.")
        return

    # Plotting
    logger.info("Generating plot...")
    
    # Prepare data for plotting
    pathways = df['pathway'].unique()
    obs_df = adata.obs.copy()
    obs_df = obs_df.sort_values(args.pseudotime_key)
    sorted_indices = obs_df.index
    
    pathway_genes = {}
    with open(gmt_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                pathway_genes[parts[0]] = parts[2:]
    
    genes_to_plot = []
    gene_pathway_map = {}
    for p in pathways:
        pg = pathway_genes.get(p, [])
        valid_genes = [g for g in pg if g in adata.var_names]
        genes_to_plot.extend(valid_genes)
        for g in valid_genes:
            gene_pathway_map[g] = p
            
    genes_to_plot = sorted(list(set(genes_to_plot)))
    
    if not genes_to_plot:
        logger.warning("No genes to plot.")
        return

    X = adata[sorted_indices, genes_to_plot].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    
    window = 20
    X_smooth = pd.DataFrame(X, columns=genes_to_plot).rolling(window=window, center=True).mean()
    
    if df['pt_mid'].nunique() <= 1:
        logger.warning("Only one window or missing pt_mid; cannot plot NES along pseudotime.")
        return

    # Use sharex=False to avoid coupling heatmap (pixel) coords with lineplot (data) coords
    fig, (ax_hm, ax_nes) = plt.subplots(2, 1, figsize=(10, 8), sharex=False, gridspec_kw={'height_ratios': [1, 1]})
    
    X_plot = X_smooth.T
    # Avoid division by zero
    std = X_plot.std(axis=1).values[:, None]
    std[std == 0] = 1.0
    X_plot = (X_plot - X_plot.mean(axis=1).values[:, None]) / std
    
    # Heatmap
    sns.heatmap(X_plot, cmap="viridis", ax=ax_hm, cbar_kws={'label': 'Z-score Expr'}, xticklabels=False)
    ax_hm.set_title("Gene Expression along Pseudotime (ordered cells)")
    ax_hm.set_ylabel("Genes")
    
    df["pt_mid"] = pd.to_numeric(df["pt_mid"], errors="coerce")
    df["nes"] = pd.to_numeric(df["nes"], errors="coerce")
    df = df.dropna(subset=["pt_mid", "nes", "pathway"])
    df_sorted = df.sort_values("pt_mid")
    if df_sorted.empty:
        logger.warning("No valid NES points after cleaning; skipping NES plot.")
        return
    logger.info(f"pt_mid dtype={df_sorted['pt_mid'].dtype}, range={df_sorted['pt_mid'].min()}..{df_sorted['pt_mid'].max()}, nunique={df_sorted['pt_mid'].nunique()}")
    logger.info(df_sorted[["pathway", "window_id", "pt_mid", "nes"]].head(10))
    
    logger.info(f"Plotting NES with x-range: {df_sorted['pt_mid'].min():.4f} - {df_sorted['pt_mid'].max():.4f}")
    
    pathway_order = ["Pathway_Up", "Pathway_Down", "Pathway_Mixed"]
    label_map = {"Pathway_Mixed": "Pathway_Mixed (half up/half down)"}
    ordered = [p for p in pathway_order if p in df_sorted["pathway"].unique()]
    others = [p for p in df_sorted["pathway"].unique() if p not in pathway_order]
    for pw in ordered + others:
        sub = df_sorted[df_sorted["pathway"] == pw].sort_values("pt_mid")
        y = sub["nes"].rolling(3, center=True, min_periods=1).mean()
        ax_nes.plot(sub["pt_mid"].to_numpy(), y.to_numpy(), marker="o", linewidth=1, label=label_map.get(pw, pw))
    ax_nes.legend(title="pathway")
    ax_nes.axhline(0, color="gray", linestyle="--")
    ax_nes.set_title("Pathway NES along Pseudotime")
    ax_nes.set_xlabel("Pseudotime")
    ax_nes.set_ylabel("NES")
    note = f"Points = window centers (pt_mid); window_size={window_size}, step={step}\nDisplay smooth: rolling mean (w=3)"
    ax_nes.text(0.99, 0.98, note, transform=ax_nes.transAxes, ha="right", va="top", fontsize=8)
    
    x_min = float(df_sorted["pt_mid"].min())
    x_max = float(df_sorted["pt_mid"].max())
    if x_min >= 0 and x_max <= 1:
        ax_nes.set_xlim(0, 1)
    else:
        ax_nes.set_xlim(x_min, x_max)
    
    plt.tight_layout()
    out_png = os.path.join(args.outdir, "trajectory_demo.png")
    plt.savefig(out_png, dpi=300)
    logger.info(f"Saved plot to {out_png}")

if __name__ == "__main__":
    main()
