import pandas as pd
import numpy as np
import subprocess
import sys
import inspect
from pathlib import Path
from pyfgsea import run_gsea

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repro.reference_environment import (  # noqa: E402
    RReferenceEnvironment,
    verify_r_fgsea_reference,
    write_reference_environment,
)
from repro.evidence_receipt import (  # noqa: E402
    capture_git_state,
    verify_pyfgsea_installation,
    write_evidence_receipt,
)


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


def run_r_multiseed(df, gmt, seeds, out_dir, reference: RReferenceEnvironment):
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
    {reference.r_guard}

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

    subprocess.run([reference.rscript, str(r_script_file)], check=True)

    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    initial_git_state = capture_git_state()
    pyfgsea_identity = verify_pyfgsea_installation()
    print("Running ablation study for tail precision...")
    out_dir = Path("results/ablation_tail")
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = verify_r_fgsea_reference()
    expected_reference = {"0.1.4": "1.32.2", "0.2.0": "1.38.0"}[
        str(pyfgsea_identity["version"])
    ]
    if reference.expected_fgsea_version != expected_reference:
        raise RuntimeError(
            "Python/R reference lane mismatch: "
            f"PyFgsea {pyfgsea_identity['version']} requires fgsea {expected_reference}, "
            f"not {reference.expected_fgsea_version}"
        )
    write_reference_environment(
        out_dir / "r_reference_environment.json",
        reference,
        {"script": "repro/fig_ablation_tail.py"},
    )

    # Data generation. Persist the exact synthetic inputs used by both engines
    # so a receipt can bind the comparison to bytes, not only to a seed claim.
    n_genes = 20000
    n_sets = 500
    data_seed = 42
    df, gmt = generate_synthetic_data(
        n_genes=n_genes, n_sets=n_sets, seed=data_seed
    )
    ranks_path = out_dir / "input_ranks.csv"
    pathways_path = out_dir / "input_pathways.gmt"
    df.to_csv(ranks_path, index=False)
    with pathways_path.open("w", encoding="utf-8", newline="\n") as handle:
        for name, genes in gmt.items():
            joined_genes = "\t".join(genes)
            handle.write(f"{name}\tNA\t{joined_genes}\n")

    # Run PyFgsea
    print("  Running PyFgsea...")
    py_seed = 42
    pyfgsea_overrides = {"seed": py_seed, "eps": 0.0}
    run_signature = inspect.signature(run_gsea)
    pyfgsea_effective_parameters = {}
    for name, parameter in run_signature.parameters.items():
        if name in {"data", "gmt"}:
            continue
        value = pyfgsea_overrides.get(name, parameter.default)
        if value is inspect.Parameter.empty:
            continue
        if value is None or isinstance(value, (str, int, float, bool)):
            pyfgsea_effective_parameters[name] = value
        else:
            pyfgsea_effective_parameters[name] = repr(value)
    res_py = run_gsea(df, gmt, **pyfgsea_overrides)
    if "P-value" in res_py.columns:
        res_py = res_py.rename(columns={"P-value": "pval"})
    required_py = {"Pathway", "pval", "NES"}
    missing_py = sorted(required_py.difference(res_py.columns))
    if missing_py:
        raise RuntimeError(f"PyFgsea output is missing columns: {missing_py}")
    if res_py.empty:
        raise RuntimeError("PyFgsea produced no ablation-tail results")
    py_results_path = out_dir / "pyfgsea_results.csv"
    res_py.to_csv(py_results_path, index=False)
    res_py = res_py.set_index("Pathway")

    # Run R baseline
    print("  Running R baseline...")
    run_r_multiseed(df, gmt, [42], out_dir, reference)

    r_res_file = out_dir / "r_res_42.csv"
    if not r_res_file.is_file():
        raise RuntimeError("R reference completed without producing r_res_42.csv")

    res_r = pd.read_csv(r_res_file).set_index("pathway")

    # Intersection
    common = res_py.index.intersection(res_r.index)
    if len(common) == 0:
        raise RuntimeError("PyFgsea and R fgsea have no pathways in common")
    res_py_cmp = res_py.loc[common]
    res_r_cmp = res_r.loc[common]

    # Focus on deep tail (p < 1e-4)
    tail_mask = res_r_cmp["pval"] < 1e-4
    tail_py = res_py_cmp[tail_mask]
    tail_r = res_r_cmp[tail_mask]

    print(f"\n[Analysis] Pathways with p < 1e-4: {len(tail_r)}")

    if len(tail_r) == 0:
        raise RuntimeError(
            "The deterministic ablation fixture produced no R pathways with p < 1e-4"
        )
    logp_py = -np.log10(tail_py["pval"] + 1e-300)
    logp_r = -np.log10(tail_r["pval"] + 1e-300)

    mae = (logp_py - logp_r).abs().mean()
    max_diff = (logp_py - logp_r).abs().max()

    print(f"  LogP MAE: {mae:.4f}")
    print(f"  Max LogP Diff: {max_diff:.4f}")

    # Multi-seed consistency check
    print("\n[Consistency] Running multi-seed validation (R-fgsea)...")
    seeds = list(range(42, 62))  # 20 seeds
    run_r_multiseed(df, gmt, seeds, out_dir, reference)

    r_seed_paths = {}
    for seed in seeds:
        seed_path = out_dir / f"r_res_{seed}.csv"
        if not seed_path.is_file():
            raise RuntimeError(f"R reference seed output is missing: {seed_path}")
        r_seed_paths[f"r_seed_{seed}"] = seed_path

    summary = pd.DataFrame(
        {
            "Pathway": tail_r.index.to_numpy(),
            "LogP_R": logp_r.to_numpy(),
            "LogP_Py": logp_py.to_numpy(),
            "NES_R": tail_r["NES"].to_numpy(),
            "NES_Py": tail_py["NES"].to_numpy(),
        },
    )
    summary_path = out_dir / "tail_summary.csv"
    summary.to_csv(summary_path, index=False)

    environment_path = out_dir / "r_reference_environment.json"
    write_evidence_receipt(
        out_dir / "tail_analysis.receipt.json",
        script=Path(__file__),
        parameters={
            "n_genes": n_genes,
            "n_sets": n_sets,
            "data_seed": data_seed,
            "pyfgsea_run_gsea": pyfgsea_effective_parameters,
            "pyfgsea_run_gsea_signature": str(run_signature),
            "r_seeds": seeds,
            "r_sample_size": 101,
            "r_eps": 0.0,
            "r_nproc": 1,
            "tail_threshold": 1e-4,
            "pvalue_floor_for_log10": 1e-300,
            "fgsea_reference_version": reference.expected_fgsea_version,
        },
        inputs={
            "ranks": ranks_path,
            "pathways": pathways_path,
            "r_reference_environment": environment_path,
        },
        outputs={
            "pyfgsea_results": py_results_path,
            "tail_summary": summary_path,
            **r_seed_paths,
        },
        git_state=initial_git_state,
        extra={"tail_logp_mae": float(mae), "tail_logp_max_abs_diff": float(max_diff)},
    )
    print(f"  Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
