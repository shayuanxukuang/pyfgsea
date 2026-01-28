
import pandas as pd
import numpy as np
import subprocess
from pathlib import Path
from pyfgsea import run_gsea

def generate_synthetic_data(n_genes=20000, n_sets=500, seed=42):
    """Generates synthetic data with embedded signals for tail analysis."""
    rng = np.random.default_rng(seed)
    genes = [f"Gene_{i}" for i in range(n_genes)]
    scores = rng.normal(0, 1, n_genes)
    
    # Sort for easier signal embedding
    sorted_idx = np.argsort(scores)
    top_genes = np.array(genes)[sorted_idx[-1000:]]
    bottom_genes = np.array(genes)[sorted_idx[:1000]]
    
    gmt = {}
    
    # Background null pathways
    for i in range(n_sets - 20):
        size = rng.integers(15, 200)
        path_genes = rng.choice(genes, size, replace=False)
        gmt[f"Null_{i}"] = list(path_genes)
        
    # Enriched pathways (Positive/Negative)
    for i in range(10):
        hits = rng.choice(top_genes, 30, replace=False)
        rest = rng.choice(genes, 20, replace=False)
        gmt[f"PosTail_{i}"] = list(hits) + list(rest)
        
    for i in range(10):
        hits = rng.choice(bottom_genes, 30, replace=False)
        rest = rng.choice(genes, 20, replace=False)
        gmt[f"NegTail_{i}"] = list(hits) + list(rest)
        
    return pd.DataFrame({"Gene": genes, "Score": scores}), gmt

def run_r_multiseed(df, gmt, seeds, out_dir):
    """Executes R-fgsea with multiple seeds for variance estimation."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    temp_dir = out_path / "temp_r_work"
    temp_dir.mkdir(exist_ok=True)
    
    ranks_file = temp_dir / "ranks.csv"
    gmt_file = temp_dir / "pathways.gmt"
    r_script_file = temp_dir / "run_multiseed.R"
    
    df.to_csv(ranks_file, index=False)
    with open(gmt_file, "w") as f:
        for k, v in gmt.items():
            joined = "\t".join(v)
            f.write(f"{k}\tNA\t{joined}\n")
            
    r_script = f"""
    suppressPackageStartupMessages(library(fgsea))
    suppressPackageStartupMessages(library(data.table))
    
    df <- fread("{ranks_file.as_posix()}")
    stats <- df$Score
    names(stats) <- df$Gene
    pathways <- gmtPathways("{gmt_file.as_posix()}")
    
    seeds <- c({",".join(map(str, seeds))})
    
    for (s in seeds) {{
        set.seed(s)
        # Use eps=0 for maximum precision in tail estimation
        res <- fgseaMultilevel(pathways, stats, sampleSize=101, eps=0, nproc=1)
        fwrite(res, paste0("{out_path.as_posix()}/r_res_", s, ".csv"))
    }}
    """
    
    with open(r_script_file, "w") as f:
        f.write(r_script)
        
    try:
        subprocess.run(["Rscript", str(r_script_file)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("Error running R script. Please check R installation.")
        
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    print("Running ablation study for tail precision...")
    out_dir = Path("results/ablation_tail")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Data generation
    df, gmt = generate_synthetic_data()
    
    # Run PyFgsea
    print("  Running PyFgsea...")
    res_py = run_gsea(df, gmt, seed=42, eps=0.0)
    if "P-value" in res_py.columns: 
        res_py = res_py.rename(columns={"P-value": "pval"})
    res_py = res_py.set_index("Pathway")
    
    # Run R baseline
    print("  Running R baseline...")
    run_r_multiseed(df, gmt, [42], out_dir)
    
    r_res_file = out_dir / "r_res_42.csv"
    if not r_res_file.exists():
        print("Error: R results not found. Skipping analysis.")
        return
        
    res_r = pd.read_csv(r_res_file).set_index("pathway")
    
    # Intersection
    common = res_py.index.intersection(res_r.index)
    res_py = res_py.loc[common]
    res_r = res_r.loc[common]
    
    # Focus on deep tail (p < 1e-4)
    tail_mask = res_r["pval"] < 1e-4
    tail_py = res_py[tail_mask]
    tail_r = res_r[tail_mask]
    
    print(f"\n[Analysis] Pathways with p < 1e-4: {len(tail_r)}")
    
    if len(tail_r) > 0:
        logp_py = -np.log10(tail_py["pval"] + 1e-300)
        logp_r = -np.log10(tail_r["pval"] + 1e-300)
        
        mae = (logp_py - logp_r).abs().mean()
        max_diff = (logp_py - logp_r).abs().max()
        
        print(f"  LogP MAE: {mae:.4f}")
        print(f"  Max LogP Diff: {max_diff:.4f}")
        
    # Multi-seed consistency check
    print("\n[Consistency] Running multi-seed validation (R-fgsea)...")
    seeds = range(42, 62)  # 20 seeds
    run_r_multiseed(df, gmt, seeds, out_dir)
    
    summary = pd.DataFrame({
        "Pathway": tail_r.index,
        "LogP_R": -np.log10(tail_r["pval"] + 1e-300),
        "LogP_Py": -np.log10(tail_py["pval"] + 1e-300),
        "NES_R": tail_r["NES"],
        "NES_Py": tail_py["NES"]
    })
    summary.to_csv(out_dir / "tail_summary.csv")
    print(f"  Saved summary to {out_dir / 'tail_summary.csv'}")

if __name__ == "__main__":
    main()
