import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Ensure repo root is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from pyfgsea import run_gsea

def ks_uniform_dist(p_values):
    """Calculates KS distance to Uniform(0,1)."""
    p = np.clip(np.asarray(p_values, float), 1e-300, 1.0)
    p = np.sort(p)
    n = len(p)
    if n == 0: return np.nan
    ecdf = np.arange(1, n + 1) / n
    return np.max(np.abs(ecdf - p))

def main():
    n_tests = 10000
    seed = 42
    print(f"Running Null Calibration (n={n_tests})...")
    
    # 1. Generate null data
    universe_size = 20000
    genes = [f"Gene_{i}" for i in range(universe_size)]
    rng = np.random.default_rng(seed)
    scores = rng.normal(0, 1, universe_size)
    df_rank = pd.DataFrame({"Gene": genes, "Score": scores})
    
    # 2. Generate random pathways
    gmt = {}
    for i in range(n_tests):
        pathway_genes = rng.choice(genes, size=50, replace=False)
        gmt[f"NullPath_{i}"] = list(pathway_genes)
        
    # 3. Run Standard GSEA
    print("  Running Standard Mode...")
    res_std = run_gsea(df_rank, gmt, use_batched=False, seed=seed)
    p_std = res_std["P-value"].values
    
    # 4. Run Batched GSEA
    print("  Running Batched Mode...")
    res_bat = run_gsea(df_rank, gmt, use_batched=True, seed=seed, sample_size=1001)
    p_bat = res_bat["P-value"].values
    
    # 5. Metrics
    ks_std = ks_uniform_dist(p_std)
    ks_bat = ks_uniform_dist(p_bat)
    frac_std = (p_std < 0.05).mean()
    frac_bat = (p_bat < 0.05).mean()
    
    print(f"  Standard: KS={ks_std:.4f}, p<0.05={frac_std:.4f}")
    print(f"  Batched:  KS={ks_bat:.4f}, p<0.05={frac_bat:.4f}")
    
    # 6. Plotting
    plt.figure(figsize=(6, 6))
    expected = np.arange(1, n_tests + 1) / (n_tests + 1)
    
    plt.plot(-np.log10(expected), -np.log10(np.sort(p_std)), label=f'Standard (KS={ks_std:.3f})')
    plt.plot(-np.log10(expected), -np.log10(np.sort(p_bat)), label=f'Batched (KS={ks_bat:.3f})', linestyle='--')
    plt.plot([0, 4], [0, 4], 'k-', alpha=0.5)
    plt.xlabel("Expected -log10(p)")
    plt.ylabel("Observed -log10(p)")
    plt.title("QQ Plot: Null Calibration")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    out_path = fig_dir / "null_calibration_qq.png"
    plt.savefig(out_path, dpi=300)
    print(f"  Saved plot to {out_path}")

if __name__ == "__main__":
    main()
