import pandas as pd
import numpy as np
import subprocess
from pathlib import Path
from pyfgsea import run_gsea


def generate_tied_data(n_genes=10000, n_ties=500, seed=42):
    """Generate synthetic data with intentional score ties."""
    rng = np.random.default_rng(seed)
    genes = [f"Gene_{i}" for i in range(n_genes)]
    scores = rng.normal(0, 1, n_genes)

    # Introduce ties by duplicating scores
    tie_indices = rng.choice(n_genes, n_ties, replace=False)
    for i in range(0, n_ties, 2):
        if i + 1 < n_ties:
            scores[tie_indices[i + 1]] = scores[tie_indices[i]]

    df = pd.DataFrame({"Gene": genes, "Score": scores})

    # Generate random pathways
    gmt = {}
    for i in range(100):
        size = rng.integers(15, 100)
        path_genes = rng.choice(genes, size, replace=False)
        gmt[f"Path_{i}"] = list(path_genes)

    return df, gmt


def run_r_baseline(df, gmt, out_path, eps=0):
    """Run R-fgsea via subprocess for baseline comparison."""
    temp_dir = Path("temp_r_work")
    temp_dir.mkdir(exist_ok=True)

    ranks_file = temp_dir / "ranks.csv"
    gmt_file = temp_dir / "pathways.gmt"
    r_script_file = temp_dir / "run.R"

    df.to_csv(ranks_file, index=False)

    with open(gmt_file, "w") as f:
        for k, v in gmt.items():
            joined_v = '\t'.join(v)
            f.write(f"{k}\tNA\t{joined_v}\n")

    r_script = f"""
    suppressPackageStartupMessages(library(fgsea))
    suppressPackageStartupMessages(library(data.table))
    
    df <- fread("{ranks_file.as_posix()}")
    stats <- df$Score
    names(stats) <- df$Gene
    
    pathways <- gmtPathways("{gmt_file.as_posix()}")
    
    set.seed(42)
    # Using fgseaMultilevel
    res <- fgseaMultilevel(pathways, stats, sampleSize=101, eps={eps}, nproc=1)
    
    fwrite(res, "{Path(out_path).as_posix()}")
    """

    with open(r_script_file, "w") as f:
        f.write(r_script)

    try:
        subprocess.run(
            ["Rscript", str(r_script_file)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        print("Error running R script. Ensure R and fgsea are installed.")

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


def run_ablation_study():
    """
    Conducts ablation study to identify sources of divergence between PyFgsea and R-fgsea.
    Factors tested: Tie handling, EPS parameter, Sorting stability.
    """
    print("Running Root Cause Ablation Study...")
    out_dir = Path("results/ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup Data with Ties
    df_raw, gmt = generate_tied_data(n_genes=12000, n_ties=2000, seed=42)

    # 2. Run R Baseline (eps=0)
    print("  Running R Baseline (eps=0)...")
    r_out = out_dir / "r_baseline.csv"
    run_r_baseline(df_raw, gmt, r_out, eps=0)

    if not r_out.exists():
        print("Skipping ablation due to missing R baseline.")
        return

    df_r = pd.read_csv(r_out).set_index("pathway")

    results_summary = []

    # Define Configurations
    configs = [
        ("Default", {}, None),
        ("Stable Sort", {}, "stable"),
        ("Random Tie Break", {}, "random"),
        ("Eps Aligned (eps=0)", {"eps": 0.0}, None),
        ("Eps=1e-10", {"eps": 1e-10}, None),
        ("Stable + Eps=0", {"eps": 0.0}, "stable"),
        ("Stable + Eps=1e-10", {"eps": 1e-10}, "stable"),
    ]

    for name, kwargs, sort_mode in configs:
        print(f"  Testing: {name}...")

        # Prepare Data Sorting
        df_run = df_raw.copy()
        if sort_mode == "stable":
            df_run = df_run.sort_values(["Score", "Gene"], ascending=[False, True])
        elif sort_mode == "random":
            # Random shuffle then stable sort
            df_run = df_run.sample(frac=1, random_state=42).sort_values(
                "Score", kind="mergesort", ascending=False
            )
        else:
            df_run = df_run.sort_values("Score", ascending=False)

        # Run PyFgsea
        res = run_gsea(
            df_run, gmt, gene_col="Gene", score_col="Score", seed=42, **kwargs
        )

        # Compare with R
        common = df_r.index.intersection(res["Pathway"])
        r_sub = df_r.loc[common]
        py_sub = res.set_index("Pathway").loc[common]

        # Standardize P-value column name
        if "P-value" in py_sub.columns:
            py_sub = py_sub.rename(columns={"P-value": "pval"})

        nes_diff = (py_sub["NES"] - r_sub["NES"]).abs()

        # Avoid log(0)
        p_py = py_sub["pval"] + 1e-300
        p_r = r_sub["pval"] + 1e-300
        pval_log_diff = (-np.log10(p_py) - -np.log10(p_r)).abs()

        metrics = {
            "Config": name,
            "NES_RMSE": np.sqrt((nes_diff**2).mean()),
            "NES_MAE": nes_diff.mean(),
            "LogP_MAE": pval_log_diff.mean(),
            "Max_NES_Diff": nes_diff.max(),
            "N_Outliers_NES>0.1": (nes_diff > 0.1).sum(),
        }
        results_summary.append(metrics)

        # Diagnostics for best candidate config
        if name == "Stable + Eps=0":
            outliers = nes_diff[nes_diff > 0.05].index.tolist()
            if outliers:
                print(
                    f"    Found {len(outliers)} outliers in 'Stable + Eps=0': {outliers[:3]}..."
                )
                target = outliers[0]
                print(f"    Tracing outlier: {target}")
                print(
                    f"      R NES: {r_sub.loc[target, 'NES']:.4f}, Py NES: {py_sub.loc[target, 'NES']:.4f}"
                )
                print(
                    f"      R Pval: {r_sub.loc[target, 'pval']:.4e}, Py Pval: {py_sub.loc[target, 'pval']:.4e}"
                )

    # Save Summary
    df_sum = pd.DataFrame(results_summary)
    print("\n=== Ablation Results ===")
    print(df_sum.to_string(index=False))
    df_sum.to_csv(out_dir / "ablation_summary.csv", index=False)


if __name__ == "__main__":
    run_ablation_study()
