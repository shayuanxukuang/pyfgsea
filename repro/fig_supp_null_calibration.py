import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Ensure repo root is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from pyfgsea import run_gsea  # noqa: E402


def run_multi_null_calibration_supplement():
    """
    Generates null calibration evidence using multiple replicates.
    Plots QQ and ECDF to demonstrate uniformity of P-values under the null hypothesis.
    """
    print("Generating Multi-Replicate Null Calibration (Supplement 5)...")

    n_reps = 20
    n_tests = 1000
    universe_size = 15000

    genes = [f"Gene_{i}" for i in range(universe_size)]

    plt.figure(figsize=(12, 5))

    # Subplot 1: QQ Plot Overlap
    ax1 = plt.subplot(1, 2, 1)

    # Subplot 2: ECDF
    ax2 = plt.subplot(1, 2, 2)

    all_pvals = []

    print(f"Running {n_reps} replicates...")
    for r in range(n_reps):
        seed = 42 + r
        rng = np.random.default_rng(seed)

        # Random ranks
        scores = rng.normal(0, 1, universe_size)
        df_rank = pd.DataFrame({"Gene": genes, "Score": scores})

        # Random pathways
        gmt = {}
        for i in range(n_tests):
            pathway_genes = rng.choice(genes, size=50, replace=False)
            gmt[f"Null_{r}_{i}"] = list(pathway_genes)

        # Run GSEA
        # Using batched=False (standard) to be conservative
        res = run_gsea(
            df_rank, gmt, use_batched=False, seed=seed, min_size=15, max_size=500
        )

        if not res.empty:
            p = res["P-value"].values
            # Clip small values to avoid inf in log plot
            p = np.clip(p, 1e-10, 1.0)
            all_pvals.extend(p)

            # Add trace to QQ
            p_sorted = np.sort(p)
            n = len(p)
            expected = np.arange(1, n + 1) / (n + 1)

            ax1.plot(
                -np.log10(expected),
                -np.log10(p_sorted),
                color="gray",
                alpha=0.3,
                linewidth=1,
            )

    # Convert all pvals to array
    all_pvals = np.array(all_pvals)

    # QQ Plot Final touches
    ax1.plot([0, 4], [0, 4], "r--", label="Ideal Uniform")
    ax1.set_xlabel("Expected -log10(p)")
    ax1.set_ylabel("Observed -log10(p)")
    ax1.set_title(f"QQ Plot: {n_reps} Replicates (Null)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ECDF Plot
    x = np.linspace(0, 1, 1000)
    ax2.plot(x, x, "r--", label="Theoretical Uniform")
    sns.ecdfplot(
        all_pvals, ax=ax2, color="blue", label=f"Empirical (N={len(all_pvals)})"
    )

    ax2.set_xlabel("P-value")
    ax2.set_ylabel("Cumulative Probability")
    ax2.set_title("P-value Distribution (Global)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    out_dir = Path("supplementary_figures")
    out_dir.mkdir(exist_ok=True)
    save_path = out_dir / "Supp_Fig5_Null_Calibration_Multi.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    run_multi_null_calibration_supplement()
