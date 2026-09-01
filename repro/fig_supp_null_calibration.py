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
from repro.evidence_receipt import (  # noqa: E402
    capture_git_state,
    verify_pyfgsea_installation,
    write_evidence_receipt,
)


def run_multi_null_calibration_supplement():
    """
    Generates null calibration evidence using multiple replicates.
    Plots QQ and ECDF to demonstrate uniformity of P-values under the null hypothesis.
    """
    initial_git_state = capture_git_state()
    verify_pyfgsea_installation()
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
    evidence_rows = []

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

        if res.empty:
            raise RuntimeError(f"Null calibration replicate {r} produced no results")
        if len(res) != n_tests:
            raise RuntimeError(
                f"Null calibration replicate {r} returned {len(res)} of {n_tests} pathways"
            )
        p = pd.to_numeric(res["P-value"], errors="coerce").to_numpy()
        if (not np.isfinite(p).all()) or (p < 0).any() or (p > 1).any():
            raise RuntimeError(f"Null calibration replicate {r} has invalid p-values")
        all_pvals.extend(p)
        evidence_rows.extend(
            {
                "replicate": r,
                "seed": seed,
                "pathway": pathway,
                "pvalue": float(pvalue),
            }
            for pathway, pvalue in zip(res["Pathway"], p)
        )

        # Clip only the plotted transform; retain exact p-values in evidence CSV.
        p_plot = np.clip(p, np.finfo(float).tiny, 1.0)
        p_sorted = np.sort(p_plot)
        n = len(p_plot)
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
    plt.close()
    values_path = out_dir / "Supp_Fig5_Null_Calibration_Multi.values.csv"
    pd.DataFrame(evidence_rows).to_csv(values_path, index=False)
    write_evidence_receipt(
        save_path.with_suffix(".receipt.json"),
        script=Path(__file__),
        parameters={
            "n_replicates": n_reps,
            "n_pathways_per_replicate": n_tests,
            "universe_size": universe_size,
            "pathway_size": 50,
            "seed_start": 42,
            "use_batched": False,
            "min_size": 15,
            "max_size": 500,
        },
        inputs={},
        outputs={"figure": save_path, "pvalues": values_path},
        git_state=initial_git_state,
    )
    print(f"Saved {save_path}")


if __name__ == "__main__":
    run_multi_null_calibration_supplement()
