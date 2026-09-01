import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Ensure repo root is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from repro.evidence_receipt import (  # noqa: E402
    capture_git_state,
    sha256_file,
    verify_pyfgsea_installation,
    write_evidence_receipt,
)


def plot_tail_consistency(
    result_dir: Path = Path("results/ablation_tail"),
    out_path: Path = Path("figures/supp_tail_consistency.png"),
):
    """Generates Figure Sx: Deep tail consistency check vs R-fgsea inter-seed variance."""
    initial_git_state = capture_git_state()
    pyfgsea_identity = verify_pyfgsea_installation()
    print("Generating tail consistency plot...")

    # Check for results
    result_dir = Path(result_dir)
    if not result_dir.is_dir():
        raise FileNotFoundError(
            f"Evidence directory is missing: {result_dir}. "
            "Run repro/fig_ablation_tail.py first."
        )

    # Load Summary (Contains PyFgsea results and Pathway list)
    summary_path = result_dir / "tail_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Tail summary is missing: {summary_path}")

    df_sum = pd.read_csv(summary_path)
    required_summary = {"Pathway", "LogP_Py"}
    missing_summary = sorted(required_summary.difference(df_sum.columns))
    if missing_summary:
        raise ValueError(f"{summary_path} is missing columns: {missing_summary}")
    if df_sum.empty:
        raise ValueError(f"{summary_path} contains no deep-tail pathways")
    if df_sum["Pathway"].duplicated().any():
        raise ValueError(f"{summary_path} contains duplicate pathways")
    if not np.isfinite(pd.to_numeric(df_sum["LogP_Py"], errors="coerce")).all():
        raise ValueError(f"{summary_path} contains invalid LogP_Py values")

    pathways = df_sum["Pathway"].values
    py_vals = df_sum.set_index("Pathway")["LogP_Py"].to_dict()

    # Collect R Distributions
    r_data = {p: [] for p in pathways}
    r_seeds = range(42, 62)

    print(f"  Loading R results for {len(r_seeds)} seeds...")
    r_paths = {}
    for s in r_seeds:
        fpath = result_dir / f"r_res_{s}.csv"
        if not fpath.is_file():
            raise FileNotFoundError(
                f"R seed result is missing: {fpath}; all seeds 42..61 are required"
            )
        r_paths[f"r_seed_{s}"] = fpath

        tmp = pd.read_csv(fpath).rename(columns={"pathway": "Pathway"})
        required_r = {"Pathway", "pval"}
        missing_r = sorted(required_r.difference(tmp.columns))
        if missing_r:
            raise ValueError(f"{fpath} is missing columns: {missing_r}")
        if tmp["Pathway"].duplicated().any():
            raise ValueError(f"{fpath} contains duplicate pathways")
        tmp = tmp.set_index("Pathway")
        missing_pathways = sorted(set(pathways).difference(tmp.index))
        if missing_pathways:
            raise ValueError(
                f"{fpath} is missing {len(missing_pathways)} summary pathways"
            )
        pvals = pd.to_numeric(tmp.loc[pathways, "pval"], errors="coerce")
        if (not np.isfinite(pvals).all()) or (pvals < 0).any() or (pvals > 1).any():
            raise ValueError(f"{fpath} contains invalid p-values")

        for pathway, pval in pvals.items():
            r_data[pathway].append(-np.log10(float(pval) + 1e-300))

    # Sort pathways by R mean
    expected_seed_count = len(r_seeds)
    incomplete = [p for p, values in r_data.items() if len(values) != expected_seed_count]
    if incomplete:
        raise RuntimeError(
            f"R tail distributions are incomplete for {len(incomplete)} pathways"
        )

    environment_path = result_dir / "r_reference_environment.json"
    if not environment_path.is_file():
        raise FileNotFoundError(
            f"Verified R reference sidecar is missing: {environment_path}"
        )
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    expected_version = environment.get("expected_fgsea_version")
    if expected_version not in {"1.32.2", "1.38.0"}:
        raise ValueError(
            f"{environment_path} does not declare a supported fgsea reference lane"
        )
    required_r_version = {"0.1.4": "1.32.2", "0.2.0": "1.38.0"}[
        str(pyfgsea_identity["version"])
    ]
    if expected_version != required_r_version:
        raise ValueError(
            "Python/R reference lane mismatch: "
            f"PyFgsea {pyfgsea_identity['version']} requires fgsea {required_r_version}, "
            f"but the sidecar declares {expected_version}"
        )

    upstream_receipt_path = result_dir / "tail_analysis.receipt.json"
    if not upstream_receipt_path.is_file():
        raise FileNotFoundError(
            f"Tail analysis receipt is missing: {upstream_receipt_path}"
        )
    upstream = json.loads(upstream_receipt_path.read_text(encoding="utf-8"))
    if upstream.get("git", {}).get("clean") is not True:
        raise ValueError("Tail analysis receipt was not captured from a clean Git tree")
    if upstream.get("pyfgsea", {}).get("version") != pyfgsea_identity["version"]:
        raise ValueError("Tail analysis receipt belongs to a different PyFgsea lane")
    upstream_outputs = upstream.get("outputs", {})
    bound_outputs = {"tail_summary": summary_path, **r_paths}
    for name, bound_path in bound_outputs.items():
        recorded_hash = upstream_outputs.get(name, {}).get("sha256")
        if recorded_hash != sha256_file(bound_path):
            raise ValueError(
                f"{bound_path} does not match {name} in the tail analysis receipt"
            )

    p_means = {p: np.mean(v) for p, v in r_data.items()}
    sorted_paths = sorted(pathways, key=lambda x: p_means[x])

    # Prepare Plot Data
    x = range(len(sorted_paths))
    # r_means = [p_means[p] for p in sorted_paths]
    r_lows = [np.percentile(r_data[p], 2.5) for p in sorted_paths]
    r_highs = [np.percentile(r_data[p], 97.5) for p in sorted_paths]
    py_points = [py_vals[p] for p in sorted_paths]

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

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()

    write_evidence_receipt(
        out_path.with_suffix(".receipt.json"),
        script=Path(__file__),
        parameters={
            "r_seeds": list(r_seeds),
            "interval_percentiles": [2.5, 97.5],
            "pvalue_floor_for_log10": 1e-300,
            "fgsea_reference_version": expected_version,
        },
        inputs={
            "tail_summary": summary_path,
            "r_reference_environment": environment_path,
            "tail_analysis_receipt": upstream_receipt_path,
            **r_paths,
        },
        outputs={"figure": out_path},
        git_state=initial_git_state,
    )
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("results/ablation_tail"))
    parser.add_argument(
        "--output", type=Path, default=Path("figures/supp_tail_consistency.png")
    )
    arguments = parser.parse_args()
    plot_tail_consistency(arguments.result_dir, arguments.output)
