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

from pyfgsea import run_trajectory_gsea  # noqa: E402


def ensure_gmt(gmt_path):
    path = Path(gmt_path)
    if path.exists():
        return str(path)

    print(f"GMT not found. Downloading Hallmark gene sets to {path}...")
    path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=MSigDB_Hallmark_2020"
    try:
        urllib.request.urlretrieve(url, path)
        return str(path)
    except Exception as e:
        print(f"Download failed: {e}")
        return None


def run_window_sensitivity_supplement():
    print("Generating Window Sensitivity Analysis (Supplement 4b)...")

    gmt_path = ensure_gmt("data/gmt/hallmark.gmt")
    if not gmt_path:
        return

    try:
        adata = sc.datasets.paul15()
    except Exception as e:
        print(f"Error loading Paul15: {e}")
        return

    adata.var_names = adata.var_names.str.upper()

    # Erythroid Lineage
    ery_clusters = ["1Ery", "2Ery", "3Ery", "4Ery", "5Ery", "6Ery", "7MEP"]
    adata_sub = adata[adata.obs["paul15_clusters"].isin(ery_clusters)].copy()

    root_gene = "GATA2"
    if root_gene not in adata_sub.var_names:
        root_gene = "KIT"

    print(f"Running Sensitivity Analysis with root={root_gene}...")

    windows = [50, 100, 200]
    # Broad match for Heme Metabolism
    target_pattern = "HEME_METABOLISM"

    dfs = []

    for w in windows:
        print(f"  Running window_size={w}...")
        out_csv = Path(f"results/validation/supp_sens_w{w}.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        df = run_trajectory_gsea(
            adata_sub,
            gmt_path=gmt_path,
            root_gene=root_gene,
            window_size=w,
            step=20,
            out_csv=str(out_csv),
            min_size=15,
            max_size=500,
            nperm_nes=100,
            seed=42,
        )

        if not df.empty:
            sub = df[df["Pathway"].str.contains(target_pattern, case=False)].copy()
            if not sub.empty:
                # Normalize x-axis
                sub["Progress"] = np.linspace(0, 1, len(sub))
                sub["Window Size"] = str(w)
                dfs.append(sub)

    if not dfs:
        print("No sensitivity results found for Heme Metabolism.")
        return

    final_df = pd.concat(dfs)

    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=final_df,
        x="Progress",
        y="NES",
        hue="Window Size",
        palette="viridis",
        linewidth=2.5,
        alpha=0.8,
    )

    plt.title("Window Size Sensitivity: Heme Metabolism")
    plt.xlabel("Pseudotime Progress (0=Start, 1=End)")
    plt.ylabel("NES")
    plt.axhline(0, color="gray", linestyle="--")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    out_dir = Path("supplementary_figures")
    out_dir.mkdir(exist_ok=True)
    save_path = out_dir / "Supp_Fig4b_Window_Sensitivity.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    run_window_sensitivity_supplement()
