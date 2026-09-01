import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Ensure repo root is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

def ensure_gmt(gmt_path):
    """Require a pre-fetched GMT rather than silently changing reference bytes."""
    path = Path(gmt_path)
    if path.is_file():
        return str(path)
    raise FileNotFoundError(
        f"GMT evidence input is missing: {path}. Supply the pre-fetched, "
        "hash-verified file; this script does not download mutable inputs."
    )


def run_myeloid_validation_supplement():
    print("Generating HSC->Myeloid Trajectory Validation (Supplement 4a)...")

    # Check GMT
    gmt_path = ensure_gmt("data/gmt/hallmark.gmt")

    from pyfgsea import run_trajectory_gsea

    # Load Paul15
    try:
        adata = sc.datasets.paul15()
    except Exception as e:
        raise RuntimeError("Paul15 could not be loaded") from e

    adata.var_names = adata.var_names.str.upper()

    # Define Myeloid Lineage
    myeloid_clusters = ["9GMP", "10GMP", "14Mo", "15Mo", "16Neu", "17Neu"]
    adata_sub = adata[adata.obs["paul15_clusters"].isin(myeloid_clusters)].copy()

    # Root Gene
    root_gene = "MPO"
    if root_gene not in adata_sub.var_names:
        if "PRTN3" in adata_sub.var_names:
            root_gene = "PRTN3"
        elif "ELANE" in adata_sub.var_names:
            root_gene = "ELANE"
        else:
            raise ValueError("Paul15 subset contains none of MPO, PRTN3, or ELANE")

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
        seed=42,
    )

    if df.empty:
        raise RuntimeError("Trajectory GSEA produced no results")

    # Filter Pathways
    # target_paths = [
    #     "MSigDB_Hallmark_2020_Inflammatory_Response",
    #     "MSigDB_Hallmark_2020_TNFA_Signaling_via_NFkB",
    # ]
    # Adjust for standard Hallmark names if downloaded from other source
    # The Enrichr download uses "MSigDB_Hallmark_2020_..." prefix usually.
    # Let's check what's in the df
    # Standard Hallmark is "HALLMARK_..."

    # If Enrichr, names are "MSigDB_Hallmark_2020_..."
    # If standard MSigDB, "HALLMARK_..."

    # Let's try to match broadly
    df_plot = df[df["Pathway"].str.contains("INFLAMMATORY|TNFA", case=False)].copy()

    if df_plot.empty:
        raise ValueError(
            "Neither INFLAMMATORY nor TNFA target pathways occur in the result; "
            "refusing to substitute unrelated pathways"
        )

    # Determine X-axis
    x_col = "Window"
    if "Window" not in df_plot.columns:
        if "window_id" in df_plot.columns:
            x_col = "window_id"
        else:
            raise ValueError("Trajectory result has neither Window nor window_id")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_plot, x=x_col, y="NES", hue="Pathway", marker="o", linewidth=2)

    plt.title("HSC -> Myeloid Trajectory (Paul15): Pathway Activity")
    plt.xlabel("Pseudotime Window")
    plt.ylabel("NES")
    plt.axhline(0, color="gray", linestyle="--")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    out_dir = Path("supplementary_figures")
    out_dir.mkdir(exist_ok=True)
    save_path = out_dir / "Supp_Fig4a_Myeloid_Trajectory.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    run_myeloid_validation_supplement()
