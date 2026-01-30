import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Ensure repo root is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))


def plot_tail_consistency():
    """Generates Figure Sx: Deep tail consistency check vs R-fgsea inter-seed variance."""
    print("Generating tail consistency plot...")

    # Check for results
    result_dir = Path("results/ablation_tail")
    if not result_dir.exists():
        print(
            f"Error: Directory {result_dir} not found. Please run repro/fig_ablation_tail.py first."
        )
        # Optional: Generate dummy data if user just wants to see the plot style
        # But better to warn.
        return

    # Load Summary (Contains PyFgsea results and Pathway list)
    summary_path = result_dir / "tail_summary.csv"
    if not summary_path.exists():
        print(f"Error: {summary_path} not found.")
        return

    df_sum = pd.read_csv(summary_path)
    if "Pathway" not in df_sum.columns or "LogP_Py" not in df_sum.columns:
        print("Error: tail_summary.csv format invalid.")
        return

    pathways = df_sum["Pathway"].values
    py_vals = df_sum.set_index("Pathway")["LogP_Py"].to_dict()

    # Collect R Distributions
    r_data = {p: [] for p in pathways}
    r_seeds = range(42, 62)

    print(f"  Loading R results for {len(r_seeds)} seeds...")
    for s in r_seeds:
        fpath = result_dir / f"r_res_{s}.csv"
        if not fpath.exists():
            continue

        try:
            tmp = pd.read_csv(fpath)
            # Normalize column names
            col_map = {"pathway": "Pathway"}
            tmp = tmp.rename(columns=col_map).set_index("Pathway")

            for p in pathways:
                if p in tmp.index:
                    pval = tmp.loc[p, "pval"]
                    logp = -np.log10(pval + 1e-300)
                    r_data[p].append(logp)
        except Exception as e:
            print(f"  Error reading {fpath}: {e}")

    # Sort pathways by R mean
    p_means = {p: np.mean(v) if v else 0 for p, v in r_data.items()}
    sorted_paths = sorted(pathways, key=lambda x: p_means[x])

    # Prepare Plot Data
    x = range(len(sorted_paths))
    # r_means = [p_means[p] for p in sorted_paths]
    r_lows = [np.percentile(r_data[p], 2.5) if r_data[p] else 0 for p in sorted_paths]
    r_highs = [np.percentile(r_data[p], 97.5) if r_data[p] else 0 for p in sorted_paths]
    py_points = [py_vals.get(p, 0) for p in sorted_paths]

    # Plot
    plt.figure(figsize=(12, 6))

    # R Confidence Band
    plt.fill_between(
        x, r_lows, r_highs, color="gray", alpha=0.3, label="R fgsea (95% CI)"
    )

    # PyFgsea Points
    # Color outliers red (outside R band)
    colors = []
    for i, val in enumerate(py_points):
        if val < r_lows[i] or val > r_highs[i]:
            colors.append("red")
        else:
            colors.append("black")

    plt.scatter(x, py_points, c=colors, s=30, zorder=5, label="PyFgsea")

    # Labels
    plt.xticks(x, sorted_paths, rotation=90, fontsize=8)
    plt.ylabel("-log10(P-value)")
    plt.title("Tail Consistency: PyFgsea vs R fgsea (Deep Tail)")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "supp_tail_consistency.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    plot_tail_consistency()
