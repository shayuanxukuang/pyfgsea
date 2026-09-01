"""Recompute the Figure 2 trajectory inputs without overwriting legacy evidence.

This runner makes one explicit candidate choice where the historical record is
inconsistent: the manuscript declares a 500-cell baseline, while the old
script used 400 cells / 20 cells / 500 NES permutations and described those
settings as provisional.  The candidate uses 500 / 50 / 2000 and the PyFgsea
0.2.0 aligned defaults.  Its run manifest always remains publication-unbound
until a clean source commit and accepted parameter contract are recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_TEXT = str(REPO_ROOT)
if _REPO_ROOT_TEXT in sys.path:
    sys.path.remove(_REPO_ROOT_TEXT)
sys.path.insert(0, _REPO_ROOT_TEXT)
DEFAULT_DATASET = REPO_ROOT / "data" / "gse155254_ery_only_pt.h5ad"
DEFAULT_GENE_SETS = REPO_ROOT / "hallmark_enrichr.gmt"
EXPECTED_DATASET_SHA256 = "9d9d1db60fe06037c5bfcf1a6ce06adfa74fe6ef715d910ef8b7e004d05cd21e"
EXPECTED_GENE_SETS_SHA256 = "92203149acdfa1e7d583fad5da99487244ce353b4d10d3193cd64329e334da66"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results"
    / "fgsea_0.2.0"
    / "figure2_candidate_w500_s50_nes2000_std_exact_size"
)
PARAMETERS = {
    "pseudotime_key": "dpt_pseudotime",
    "window_mode": "cell_count",
    "window_size": 500,
    "step": 50,
    "min_size": 15,
    "max_size": 500,
    "sample_size": 101,
    "seed": 42,
    "eps": 1e-50,
    "nperm_nes": 2000,
    "nperm_simple": 1000,
    "gsea_param": 1.0,
    "mode": "aligned",
    "score_type": "std",
    "tie_policy": "gene_id",
    "bin_width": 0,
    "calculate_nes": True,
    "use_nes_cache": False,
    "max_levels": None,
    "ranker": "mean_diff",
}
TARGET_PATHWAYS = ("heme Metabolism", "E2F Targets")
RESEARCH_CONTEXT = {
    "research_mode": "in-silico computational analysis",
    "input_provenance": (
        "processed single-cell data derived from public accession GSE155254; "
        "the exact H5AD and Hallmark GMT bytes are enforced by SHA-256"
    ),
    "computational_operation": "rolling-window trajectory GSEA recalculation",
    "intended_artifact": "candidate Figure 2 tables, plots, and provenance receipt",
    "claim_boundary": "publication-unbound computational candidate",
    "physical_experiment_requested": False,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def preflight(dataset: Path, gene_sets: Path) -> Dict[str, object]:
    expected_inputs = {
        "dataset": (dataset.resolve(), EXPECTED_DATASET_SHA256),
        "gene_sets": (gene_sets.resolve(), EXPECTED_GENE_SETS_SHA256),
    }
    inputs: Dict[str, object] = {}
    for name, (path, expected) in expected_inputs.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(
                f"Input hash mismatch for {path}: expected {expected}, found {observed}"
            )
        inputs[name] = {
            "path": display_path(path),
            "bytes": path.stat().st_size,
            "sha256": observed,
        }

    return {
        "research_context": RESEARCH_CONTEXT,
        "parameters": PARAMETERS,
        "inputs": inputs,
        "parameter_contract_status": "candidate-assumption-pending-author-acceptance",
        "parameter_contract_reason": (
            "The manuscript specifies a 500-cell baseline but not the exact step or "
            "NES permutation count; 50 and 2000 are the explicit candidate values."
        ),
    }


def bh_adjust(values: Iterable[float]) -> np.ndarray:
    pvalues = np.asarray(list(values), dtype=np.float64)
    adjusted = np.full(pvalues.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(pvalues)
    if not valid.any():
        return adjusted
    selected = pvalues[valid]
    order = np.argsort(selected, kind="mergesort")
    ranked = selected[order]
    scaled = ranked * selected.size / np.arange(1, selected.size + 1)
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    restored = np.empty_like(scaled)
    restored[order] = np.clip(scaled, 0.0, 1.0)
    adjusted[valid] = restored
    return adjusted


def pathway_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pathway, group in results.groupby("Pathway", sort=True):
        ordered = group.sort_values(["pt_mid", "window_id"], kind="mergesort")
        peak_index = ordered["NES"].abs().idxmax()
        peak = ordered.loc[peak_index]
        peak_nes = float(peak["NES"])
        rows.append(
            {
                "Pathway": pathway,
                "start_pt": float(ordered.iloc[0]["pt_mid"]),
                "start_NES": float(ordered.iloc[0]["NES"]),
                "end_pt": float(ordered.iloc[-1]["pt_mid"]),
                "end_NES": float(ordered.iloc[-1]["NES"]),
                "peak_pt": float(peak["pt_mid"]),
                "peak_NES": peak_nes,
                "max_abs_NES": abs(peak_nes),
                "peak_direction": "positive" if peak_nes > 0 else "negative",
                "min_padj": float(ordered["padj"].min()),
                "significant_windows_fdr_0_05": int((ordered["padj"] < 0.05).sum()),
                "n_windows": int(len(ordered)),
            }
        )
    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        ["max_abs_NES", "min_padj", "Pathway"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    summary.insert(0, "top_pathway_rank", np.arange(1, len(summary) + 1))
    return summary


def plot_target_curves(results: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    import matplotlib.pyplot as plt

    present = set(results["Pathway"])
    missing = [name for name in TARGET_PATHWAYS if name not in present]
    if missing:
        raise RuntimeError(f"Required Figure 2 pathways are missing: {missing}")

    fig, axes = plt.subplots(len(TARGET_PATHWAYS), 1, figsize=(8.2, 6.6), sharex=True)
    for axis, pathway in zip(np.atleast_1d(axes), TARGET_PATHWAYS):
        curve = results.loc[results["Pathway"] == pathway].sort_values("pt_mid")
        axis.axhline(0.0, color="#777777", linewidth=0.8)
        axis.plot(curve["pt_mid"], curve["NES"], color="#155e75", linewidth=2.0)
        significant = curve["padj"] < 0.05
        axis.scatter(
            curve.loc[significant, "pt_mid"],
            curve.loc[significant, "NES"],
            color="#c2410c",
            s=19,
            label="within-window BH FDR < 0.05",
            zorder=3,
        )
        axis.set_title(pathway)
        axis.set_ylabel("NES")
        axis.legend(loc="best", frameon=False, fontsize=8)
    axes[-1].set_xlabel("Pseudotime midpoint")
    fig.suptitle("PyFgsea 0.2.0 Figure 2 candidate (500-cell / 50-step)")
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def artifact_record(path: Path, final_dir: Path) -> Dict[str, object]:
    return {
        "path": (final_dir / path.name).relative_to(REPO_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def run(output_dir: Path, dataset: Path, gene_sets: Path) -> Path:
    from pyfgsea import __version__
    from pyfgsea import _core
    from pyfgsea.trajectory import run_trajectory_gsea

    if __version__ != "0.2.0":
        raise RuntimeError(f"This runner requires PyFgsea 0.2.0, found {__version__}")
    if _core.algorithm_revision() != "fgsea-1.38-pr178-v1":
        raise RuntimeError(
            "Unexpected statistical core revision: " + _core.algorithm_revision()
        )

    dataset = dataset.expanduser().resolve()
    gene_sets = gene_sets.expanduser().resolve()
    preflight_receipt = preflight(dataset, gene_sets)
    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to(REPO_ROOT)
    except ValueError as error:
        raise ValueError("--output-dir must remain inside the repository") from error
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing evidence directory: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    incomplete_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.incomplete-", dir=output_dir.parent)
    )

    import scanpy as sc

    adata = sc.read_h5ad(dataset)
    if adata.shape != (3576, 3000):
        raise RuntimeError(f"Unexpected GSE155254 shape: {adata.shape}")
    if PARAMETERS["pseudotime_key"] not in adata.obs:
        raise RuntimeError("The frozen processed input lacks dpt_pseudotime")

    results = run_trajectory_gsea(
        adata,
        gmt_path=str(gene_sets),
        lineage_col=None,
        lineage_keyword=None,
        root_gene=None,
        out_csv=None,
        **PARAMETERS,
    )
    if results.empty:
        raise RuntimeError("Trajectory recalculation returned no pathways")

    required = {
        "Pathway",
        "NES",
        "P-value",
        "padj",
        "status",
        "window_id",
        "pt_mid",
    }
    missing = sorted(required.difference(results.columns))
    if missing:
        raise RuntimeError(f"Trajectory result is missing required columns: {missing}")
    bad_status = results.loc[~results["status"].isin(["resolved", "eps_floor"])]
    if not bad_status.empty:
        counts = bad_status["status"].value_counts(dropna=False).to_dict()
        raise RuntimeError(f"Unresolved statistical states prevent artifact emission: {counts}")
    pvalues = results["P-value"].to_numpy(dtype=np.float64)
    if not np.isfinite(pvalues).all() or (pvalues <= 0).any():
        raise RuntimeError("Qualified results must have finite, strictly positive p-values")

    original_padj = results["padj"].to_numpy(dtype=np.float64).copy()
    for _, indices in results.groupby("window_id", sort=False).groups.items():
        results.loc[indices, "padj"] = bh_adjust(results.loc[indices, "P-value"])
    bh_matches_core = bool(
        np.allclose(
            original_padj,
            results["padj"].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-14,
            equal_nan=True,
        )
    )
    if not bh_matches_core:
        raise RuntimeError("Independent within-window BH adjustment disagrees with core output")

    results = results.sort_values(
        ["window_id", "padj", "P-value", "Pathway"], kind="mergesort"
    ).reset_index(drop=True)
    summary = pathway_summary(results)
    targets = summary.loc[summary["Pathway"].isin(TARGET_PATHWAYS)].copy()
    if len(targets) != len(TARGET_PATHWAYS):
        raise RuntimeError("Target pathway summary is incomplete")

    result_path = incomplete_dir / "trajectory_results.csv"
    summary_path = incomplete_dir / "pathway_summary.csv"
    target_path = incomplete_dir / "figure2_target_pathway_summary.csv"
    png_path = incomplete_dir / "figure2_candidate.png"
    pdf_path = incomplete_dir / "figure2_candidate.pdf"
    results.to_csv(result_path, index=False)
    summary.to_csv(summary_path, index=False)
    targets.to_csv(target_path, index=False)
    plot_target_curves(results, png_path, pdf_path)

    git_status = _git("status", "--porcelain", "--untracked-files=normal")
    clean_tree = not git_status.strip()
    artifacts = {
        path.name: artifact_record(path, output_dir)
        for path in (result_path, summary_path, target_path, png_path, pdf_path)
    }
    binding_blockers = [
        "parameter contract requires author acceptance",
        "manuscript and supplement have not been updated from these values",
    ]
    if not clean_tree:
        binding_blockers.append("PyFgsea source tree was not clean at execution time")
    manifest = {
        "schema_version": 1,
        "artifact_type": "pyfgsea-figure2-candidate",
        "publication_accepted": False,
        "binding_status": "candidate-unbound",
        "binding_blockers": binding_blockers,
        "research_context": RESEARCH_CONTEXT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pyfgsea_version": __version__,
        "algorithm_revision": _core.algorithm_revision(),
        "pyfgsea_commit": _git("rev-parse", "HEAD").strip(),
        "source_tree_clean": clean_tree,
        "source_status_sha256": hashlib.sha256(git_status.encode("utf-8")).hexdigest(),
        "python_version": platform.python_version(),
        "parameters": dict(PARAMETERS),
        "preflight": preflight_receipt,
        "dataset_shape": [int(adata.n_obs), int(adata.n_vars)],
        "bh_recalculation_matches_core": bh_matches_core,
        "status_counts": {
            str(key): int(value)
            for key, value in results["status"].value_counts().sort_index().items()
        },
        "n_windows": int(results["window_id"].nunique()),
        "n_pathways": int(results["Pathway"].nunique()),
        "artifacts": artifacts,
    }
    (incomplete_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    incomplete_dir.replace(output_dir)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Processed GSE155254 H5AD; the frozen SHA-256 is always enforced.",
    )
    parser.add_argument(
        "--gene-sets",
        type=Path,
        default=DEFAULT_GENE_SETS,
        help="Hallmark GMT; the frozen SHA-256 is always enforced.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify fixed inputs and print the candidate parameter receipt.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.preflight_only:
        print(json.dumps(preflight(args.dataset, args.gene_sets), indent=2, sort_keys=True))
        return 0
    output = run(args.output_dir, args.dataset, args.gene_sets)
    print(f"Candidate artifacts written without overwriting legacy evidence: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
