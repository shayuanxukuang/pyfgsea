import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional


def plot_trajectory_heatmap(
    df: pd.DataFrame,
    pathways: Optional[List[str]] = None,
    n_top_pathways: int = 30,
    sort_by_peak: bool = True,
    cmap: str = "RdBu_r",
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None,
):
    """
    Plot a heatmap of NES values along pseudotime windows.

    Args:
        df: Result DataFrame from run_trajectory_gsea (must contain 'Pathway', 'window_id', 'NES').
        pathways: List of specific pathways to plot. If None, selects top varied pathways.
        n_top_pathways: Number of top pathways to select by variance if 'pathways' is None.
        sort_by_peak: Whether to sort pathways by their peak NES time.
        cmap: Colormap for the heatmap.
        figsize: Figure size.
        save_path: If provided, save the figure to this path.
    """
    if df.empty:
        print("Data is empty, cannot plot heatmap.")
        return

    # Fill NAs with 0 to allow heatmap rendering for sparse pathways
    nes_matrix = df.pivot_table(
        index="Pathway", columns="window_id", values="NES"
    ).fillna(0.0)

    # Filter pathways
    if pathways:
        valid_paths = [p for p in pathways if p in nes_matrix.index]
        if not valid_paths:
            print("None of the requested pathways found in data.")
            return
        nes_matrix = nes_matrix.loc[valid_paths]
    else:
        # Select top varied
        if len(nes_matrix) > n_top_pathways:
            variances = (
                nes_matrix.var(axis=1).sort_values(ascending=False).head(n_top_pathways)
            )
            nes_matrix = nes_matrix.loc[variances.index]

    # Sort by peak time
    if sort_by_peak:
        peak_time = nes_matrix.idxmax(axis=1)
        nes_matrix = nes_matrix.loc[peak_time.sort_values().index]

    # Clean pathway names (remove HALLMARK_ prefix for display if present)
    display_names = [
        p.replace("HALLMARK_", "").replace("KEGG_", "").replace("REACTOME_", "")
        for p in nes_matrix.index
    ]

    plt.figure(figsize=figsize)
    sns.heatmap(
        nes_matrix,
        cmap=cmap,
        center=0,
        robust=True,
        cbar_kws={"label": "NES"},
        yticklabels=display_names,
    )
    plt.title("Trajectory GSEA Heatmap")
    plt.xlabel("Pseudotime Window")
    plt.ylabel("Pathway")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved heatmap to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_pathway_dynamics(
    df: pd.DataFrame,
    pathways: List[str],
    figsize: tuple = (10, 4),
    save_path: Optional[str] = None,
):
    """
    Plot line charts for specific pathway NES dynamics.

    Args:
        df: Result DataFrame from run_trajectory_gsea.
        pathways: List of pathways to plot.
        figsize: Figure size.
        save_path: If provided, save the figure to this path.
    """
    if df.empty:
        print("Data is empty, cannot plot dynamics.")
        return

    plt.figure(figsize=figsize)

    plotted = False
    for pathway in pathways:
        if pathway in df["Pathway"].values:
            subset = df[df["Pathway"] == pathway].sort_values("window_id")
            label = (
                pathway.replace("HALLMARK_", "")
                .replace("KEGG_", "")
                .replace("REACTOME_", "")
            )
            plt.plot(subset["window_id"], subset["NES"], label=label, linewidth=2)
            plotted = True
        else:
            print(f"Warning: Pathway {pathway} not found in results.")

    if not plotted:
        print("No valid pathways to plot.")
        plt.close()
        return

    plt.axhline(0, color="black", linestyle="--", alpha=0.3)
    plt.legend()
    plt.title("Pathway Dynamics along Pseudotime")
    plt.xlabel("Pseudotime Window")
    plt.ylabel("NES")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved dynamics plot to {save_path}")
    else:
        plt.show()

    plt.close()
