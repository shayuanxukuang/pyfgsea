import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import urllib.request
from pathlib import Path

# Ensure repo root is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from pyfgsea import run_trajectory_gsea

def ensure_gmt(gmt_path):
    """Ensures GMT file exists, downloads if necessary."""
    path = Path(gmt_path)
    if path.exists():
        return str(path)
    
    print(f"GMT not found at {path}. Attempting to download Hallmark gene sets...")
    path.parent.mkdir(parents=True, exist_ok=True)
    
    url = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=MSigDB_Hallmark_2020"
    try:
        urllib.request.urlretrieve(url, path)
        print(f"Downloaded to {path}")
        return str(path)
    except Exception as e:
        print(f"Failed to download GMT: {e}")
        return None

def run_myeloid_validation_supplement():
    print("Generating HSC->Myeloid Trajectory Validation (Supplement 4a)...")
    
    # Check GMT
    gmt_path = ensure_gmt("data/gmt/hallmark.gmt")
    if not gmt_path:
        print("Skipping due to missing GMT.")
        return

    # Load Paul15
    try:
        adata = sc.datasets.paul15()
    except Exception as e:
        print(f"Error loading Paul15: {e}")
        return

    adata.var_names = adata.var_names.str.upper()

    # Define Myeloid Lineage
    myeloid_clusters = ['9GMP', '10GMP', '14Mo', '15Mo', '16Neu', '17Neu'] 
    adata_sub = adata[adata.obs['paul15_clusters'].isin(myeloid_clusters)].copy()
    
    # Root Gene
    root_gene = "MPO"
    if root_gene not in adata_sub.var_names:
        if 'PRTN3' in adata_sub.var_names: root_gene = 'PRTN3'
        elif 'ELANE' in adata_sub.var_names: root_gene = 'ELANE'
        else:
            print("No suitable root gene found.")
            return

    print(f"Running Myeloid Trajectory GSEA (Root: {root_gene})...")
    
    out_csv = Path("results/validation/supp_myeloid_traj.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    df = run_trajectory_gsea(
        adata_sub,
        gmt_path=gmt_path,
        root_gene=root_gene,
        window_size=100,
        step=20,
        out_csv=str(out_csv),
        min_size=15,
        max_size=500,
        nperm_nes=100,
        seed=42
    )
    
    if df.empty:
        print("No GSEA results generated.")
        return

    # Filter Pathways
    target_paths = ["MSigDB_Hallmark_2020_Inflammatory_Response", "MSigDB_Hallmark_2020_TNFA_Signaling_via_NFkB"]
    # Adjust for standard Hallmark names if downloaded from other source
    # The Enrichr download uses "MSigDB_Hallmark_2020_..." prefix usually.
    # Let's check what's in the df
    # Standard Hallmark is "HALLMARK_..."
    
    # If Enrichr, names are "MSigDB_Hallmark_2020_..."
    # If standard MSigDB, "HALLMARK_..."
    
    # Let's try to match broadly
    df_plot = df[df["Pathway"].str.contains("INFLAMMATORY|TNFA", case=False)].copy()
    
    if df_plot.empty:
        print("Target myeloid pathways not found. Plotting top 2 instead.")
        top_paths = df["Pathway"].unique()[:2]
        df_plot = df[df["Pathway"].isin(top_paths)].copy()

    # Determine X-axis
    x_col = "Window"
    if "Window" not in df_plot.columns:
        if "window_id" in df_plot.columns:
            x_col = "window_id"
        else:
            df_plot["Window"] = range(len(df_plot))
            x_col = "Window"
            
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_plot, x=x_col, y="NES", hue="Pathway", marker="o", linewidth=2)
    
    plt.title("HSC -> Myeloid Trajectory (Paul15): Pathway Activity")
    plt.xlabel("Pseudotime Window")
    plt.ylabel("NES")
    plt.axhline(0, color='gray', linestyle='--')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    out_dir = Path("supplementary_figures")
    out_dir.mkdir(exist_ok=True)
    save_path = out_dir / "Supp_Fig4a_Myeloid_Trajectory.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")

if __name__ == "__main__":
    run_myeloid_validation_supplement()
