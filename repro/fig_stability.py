import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Ensure repo root is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from repro.data_utils import generate_test_data  # noqa: E402
import pyfgsea  # noqa: E402


def run_stability_benchmark():
    """Generates stability boxplot showing P-value variance across random seeds."""
    out_dir = Path("results/benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating stability test data...")
    # Use same seed to ensure reproducible "ground truth"
    df_rnk, gmt = generate_test_data(n_genes=12000, n_sets=100, seed=42)

    print("Running initial pass to pick pathways...")
    res_init = pyfgsea.run_gsea(
        df_rnk, gmt, gene_col="Gene", score_col="Score", seed=42
    )
    res_init = res_init.sort_values("P-value")

    # Pick top 5 significant pathways
    top_paths = res_init.head(5)["Pathway"].tolist()
    print(f"Selected pathways: {top_paths}")

    reps = 50
    data = []

    print(f"Running {reps} replicates...")
    for i in range(reps):
        seed = 1000 + i  # Different seeds
        res = pyfgsea.run_gsea(
            df_rnk, gmt, gene_col="Gene", score_col="Score", seed=seed
        )
        res_filt = res[res["Pathway"].isin(top_paths)]

        for _, row in res_filt.iterrows():
            pval = row["P-value"]

            if pd.isna(pval):
                continue

            # Handle 0 p-values for log plot
            logp = -np.log10(max(pval, 1e-100))

            data.append(
                {
                    "Pathway": row["Pathway"],
                    "Rep": i,
                    "P-value": pval,
                    "LogP": logp,
                    "NES": row["NES"],
                }
            )

    df_res = pd.DataFrame(data)

    # Plot Boxplot of LogP
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_res, x="Pathway", y="LogP", color="skyblue")
    sns.stripplot(
        data=df_res, x="Pathway", y="LogP", color="black", alpha=0.3, jitter=True
    )
    plt.title(f"Stability of P-values ({reps} Replicates)")
    plt.ylabel("-log10(P-value)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    out_path = Path("figures")
    out_path.mkdir(exist_ok=True)
    save_file = out_path / "fig_stability_boxplot.png"
    plt.savefig(save_file, dpi=300)
    print(f"Saved {save_file}")

    # Calculate stats
    stats = df_res.groupby("Pathway")["LogP"].agg(["mean", "std", "min", "max"])
    stats["CV"] = np.where(stats["mean"] > 0, stats["std"] / stats["mean"], np.nan)
    print("\nStability Statistics (-log10 P):")
    print(stats)
    stats.to_csv(out_dir / "stability_stats.csv")


if __name__ == "__main__":
    run_stability_benchmark()
