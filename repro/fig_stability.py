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
from repro.evidence_receipt import (  # noqa: E402
    capture_git_state,
    verify_pyfgsea_installation,
    write_evidence_receipt,
)
import pyfgsea  # noqa: E402


def run_stability_benchmark():
    """Generates stability boxplot showing P-value variance across random seeds."""
    initial_git_state = capture_git_state()
    verify_pyfgsea_installation()
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
    if len(top_paths) != 5:
        raise RuntimeError("Initial stability run did not return five pathways")
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
        if set(res_filt["Pathway"]) != set(top_paths):
            raise RuntimeError(f"Stability replicate {i} is missing selected pathways")

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
    expected_rows = reps * len(top_paths)
    if len(df_res) != expected_rows:
        raise RuntimeError(
            f"Stability evidence has {len(df_res)} rows; expected {expected_rows}"
        )

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
    plt.close()
    print(f"Saved {save_file}")

    # Calculate stats
    stats = df_res.groupby("Pathway")["LogP"].agg(["mean", "std", "min", "max"])
    stats["CV"] = np.where(stats["mean"] > 0, stats["std"] / stats["mean"], np.nan)
    print("\nStability Statistics (-log10 P):")
    print(stats)
    stats_path = out_dir / "stability_stats.csv"
    values_path = out_dir / "stability_values.csv"
    stats.to_csv(stats_path)
    df_res.to_csv(values_path, index=False)
    write_evidence_receipt(
        out_dir / "stability.receipt.json",
        script=Path(__file__),
        parameters={
            "n_genes": 12000,
            "n_sets": 100,
            "data_seed": 42,
            "selection_seed": 42,
            "replicates": reps,
            "replicate_seed_start": 1000,
            "selected_pathway_count": 5,
            "pvalue_floor_for_log10": 1e-100,
        },
        inputs={},
        outputs={
            "figure": save_file,
            "statistics": stats_path,
            "values": values_path,
        },
        git_state=initial_git_state,
        extra={"selected_pathways": top_paths},
    )


if __name__ == "__main__":
    run_stability_benchmark()
