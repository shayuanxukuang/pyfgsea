"""End-to-end upstream sensitivity on two real single-cell datasets.

This audit uses PyFgsea plus three executable score-then-smooth alternatives on
real matrices.  Alternative rows are internal implementations; native GSVA or
AUCell package execution is not claimed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

import pyfgsea


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "ted_real_data_upstream_sensitivity"
SEED = 20260717
METHODS = ("pyfgsea_rolling", "rank_auc", "mean_zscore", "ssgsea_bridge")
DATASETS = (
    {
        "dataset_id": "GSE126085_real_time",
        "path": ROOT / "data" / "gse126085_stress_test.h5ad",
        "gmt": ROOT / "hallmark_enrichr.gmt",
        "trajectories": ("true_time", "dpt_pseudotime"),
        "reference_trajectory": "true_time",
        "stratify": "timepoint_label",
    },
    {
        "dataset_id": "Nestorowa_trajectory",
        "path": ROOT / "data" / "priority_h5ad" / "nestorowa_dpt.h5ad",
        "gmt": ROOT / "data" / "paul15_branch_gene_sets.gmt",
        "trajectories": ("dpt_pseudotime",),
        "reference_trajectory": "dpt_pseudotime",
        "stratify": "cell_type",
    },
)


def stratified_subsample(adata: ad.AnnData, key: str, maximum: int, seed: int) -> ad.AnnData:
    if adata.n_obs <= maximum:
        return adata.copy()
    rng = np.random.default_rng(seed)
    labels = adata.obs[key].astype(str).to_numpy()
    selected: list[int] = []
    for label in pd.unique(labels):
        indices = np.flatnonzero(labels == label)
        quota = max(1, int(round(maximum * len(indices) / adata.n_obs)))
        selected.extend(rng.choice(indices, size=min(quota, len(indices)), replace=False).tolist())
    if len(selected) > maximum:
        selected = rng.choice(selected, size=maximum, replace=False).tolist()
    return adata[np.sort(np.asarray(selected, dtype=int))].copy()


def prepare_dataset(spec: dict[str, object], maximum_cells: int, seed: int) -> tuple[ad.AnnData, dict[str, list[str]]]:
    adata = ad.read_h5ad(Path(spec["path"]))
    adata.var_names_make_unique()
    adata = stratified_subsample(adata, str(spec["stratify"]), maximum_cells, seed)
    if spec["dataset_id"] == "GSE126085_real_time":
        day = pd.to_numeric(adata.obs["timepoint_day"], errors="coerce").to_numpy(float)
        adata.obs["true_time"] = (day - np.nanmin(day)) / max(float(np.nanmax(day) - np.nanmin(day)), 1e-12)
    for trajectory in spec["trajectories"]:
        values = pd.to_numeric(adata.obs[str(trajectory)], errors="coerce")
        finite = values.notna()
        adata = adata[finite].copy()
        values = pd.to_numeric(adata.obs[str(trajectory)], errors="coerce")
        lo, hi = float(values.min()), float(values.max())
        adata.obs[str(trajectory)] = (values - lo) / max(hi - lo, 1e-12)
    gene_sets = pyfgsea.load_gmt(Path(spec["gmt"]))
    genes = set(map(str, adata.var_names))
    overlap = {
        name: [gene for gene in members if gene in genes]
        for name, members in gene_sets.items()
    }
    overlap = {name: members for name, members in overlap.items() if 5 <= len(members) <= 150}
    ranked = sorted(overlap, key=lambda name: (-len(overlap[name]), name))[:18]
    return adata, {name: overlap[name] for name in ranked}


def run_method(
    adata: ad.AnnData,
    gene_sets: dict[str, list[str]],
    method: str,
    trajectory: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float, str]:
    started = time.perf_counter()
    if method == "pyfgsea_rolling":
        result = pyfgsea.run_trajectory_gsea(
            adata,
            gene_sets,
            pseudotime_key=trajectory,
            window_size=max(80, min(240, adata.n_obs // 10)),
            step=max(40, min(120, adata.n_obs // 20)),
            min_size=5,
            max_size=150,
            nperm_nes=30,
            use_nes_cache=False,
            seed=seed,
        )
        implementation = "PyFgsea rolling-window core"
    else:
        bridge = {"rank_auc": "rank_auc", "mean_zscore": "mean_zscore", "ssgsea_bridge": "ssgsea"}[method]
        result = pyfgsea.run_score_then_smooth_baseline(
            adata,
            gene_sets,
            pseudotime_key=trajectory,
            method=bridge,
            window_size=max(80, min(240, adata.n_obs // 10)),
            step=max(40, min(120, adata.n_obs // 20)),
            min_size=5,
            max_size=150,
        )
        implementation = "internal score-then-smooth implementation; native external package not claimed"
    events = pyfgsea.summarize_events(result, min_consecutive=1)
    return result, events, time.perf_counter() - started, implementation


def event_rows(result: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    indexed = events.set_index("Pathway") if len(events) else pd.DataFrame()
    for pathway, group in result.groupby("Pathway", sort=False):
        group = group.sort_values("pt_mid")
        x = pd.to_numeric(group.pt_mid, errors="coerce").to_numpy(float)
        y = pd.to_numeric(group.NES, errors="coerce").to_numpy(float)
        area = float(np.trapz(y, x)) if len(x) > 1 else float(np.nanmean(y))
        label = str(indexed.loc[pathway, "event_label"]) if pathway in indexed.index else "no clear event"
        confidence = str(indexed.loc[pathway, "event_confidence_class"]) if pathway in indexed.index else "not supported"
        e_code = "E2" if "multi" in confidence.lower() or "switching" in confidence.lower() else "E1" if label != "no clear event" else "E0"
        summaries.append(
            {
                "pathway": str(pathway),
                "direction": "up" if area > 0 else "down" if area < 0 else "flat",
                "event_mode": label,
                "event_support_code": e_code,
                "called": label != "no clear event",
                "ambiguous": label == "no clear event" or "multi" in label.lower() or "switch" in label.lower(),
                "area": area,
            }
        )
    return pd.DataFrame(summaries)


def curve_correlation(result: pd.DataFrame, reference: pd.DataFrame) -> float:
    grid = np.linspace(0.0, 1.0, 101)
    values = []
    refs = []
    common = sorted(set(result.Pathway.astype(str)) & set(reference.Pathway.astype(str)))
    for pathway in common:
        left = result[result.Pathway.astype(str).eq(pathway)].sort_values("pt_mid")
        right = reference[reference.Pathway.astype(str).eq(pathway)].sort_values("pt_mid")
        values.extend(np.interp(grid, left.pt_mid.astype(float), left.NES.astype(float)))
        refs.extend(np.interp(grid, right.pt_mid.astype(float), right.NES.astype(float)))
    return float(pd.Series(values).corr(pd.Series(refs), method="spearman")) if values else np.nan


def compare(result: pd.DataFrame, calls: pd.DataFrame, reference_result: pd.DataFrame, reference_calls: pd.DataFrame) -> dict[str, object]:
    merged = reference_calls.merge(calls, on="pathway", suffixes=("_reference", "_candidate"), validate="one_to_one")
    left = set(reference_calls.loc[reference_calls.called, "pathway"])
    right = set(calls.loc[calls.called, "pathway"])
    union = left | right
    rank = {"E0": 0, "E1": 1, "E2": 2}
    return {
        "effect_curve_spearman": curve_correlation(result, reference_result),
        "event_family_overlap": len(left & right) / len(union) if union else 1.0,
        "direction_agreement": float((merged.direction_reference == merged.direction_candidate).mean()),
        "event_mode_agreement": float((merged.event_mode_reference == merged.event_mode_candidate).mean()),
        "e_code_agreement": float((merged.event_support_code_reference == merged.event_support_code_candidate).mean()),
        "false_e_promotion": float(np.mean([rank[c] > rank[r] for r, c in zip(merged.event_support_code_reference, merged.event_support_code_candidate, strict=True)])),
        "ambiguity_frequency": float(calls.ambiguous.mean()),
    }


def run(outdir: Path, maximum_cells: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    call_frames = []
    registry_rows = []
    for dataset_index, spec in enumerate(DATASETS):
        adata, gene_sets = prepare_dataset(spec, maximum_cells, seed + dataset_index)
        reference_result, reference_events, _, _ = run_method(
            adata, gene_sets, "pyfgsea_rolling", str(spec["reference_trajectory"]), seed + dataset_index
        )
        reference_calls = event_rows(reference_result, reference_events)
        for method in METHODS:
            for trajectory in spec["trajectories"]:
                result, events, runtime, implementation = run_method(
                    adata, gene_sets, method, str(trajectory), seed + dataset_index
                )
                calls = event_rows(result, events)
                metric_rows.append(
                    {
                        "dataset_id": spec["dataset_id"],
                        "scoring_method": method,
                        "trajectory_input": trajectory,
                        "n_cells": adata.n_obs,
                        "n_gene_sets": len(gene_sets),
                        "runtime_seconds": runtime,
                        "implementation": implementation,
                        **compare(result, calls, reference_result, reference_calls),
                    }
                )
                local = calls.copy()
                local["dataset_id"] = spec["dataset_id"]
                local["scoring_method"] = method
                local["trajectory_input"] = trajectory
                call_frames.append(local)
                registry_rows.append(
                    {
                        "dataset_id": spec["dataset_id"],
                        "scoring_method": method,
                        "trajectory_input": trajectory,
                        "implementation": implementation,
                        "native_external_package_executed": method == "pyfgsea_rolling",
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    calls = pd.concat(call_frames, ignore_index=True)
    agreements = []
    for (dataset_id, pathway), group in calls.groupby(["dataset_id", "pathway"], sort=False):
        joint = group.direction.astype(str) + "|" + group.event_mode.astype(str)
        modal = joint.mode().iloc[0]
        agreement = float((joint == modal).mean())
        disagreement = agreement < 0.75
        agreements.append(
            {
                "dataset_id": dataset_id,
                "pathway": pathway,
                "n_upstream_combinations": len(group),
                "upstream_method_agreement": agreement,
                "upstream_disagreement_flag": disagreement,
                "maximum_observed_e_code": max(group.event_support_code, key={"E0": 0, "E1": 1, "E2": 2}.get),
                "ted_disagreement_handling": "cap_at_E1_and_return_ambiguity_set" if disagreement else "eligible_subject_to_design_gates",
            }
        )
    return metrics, calls, pd.DataFrame(agreements), pd.DataFrame(registry_rows).drop_duplicates()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--maximum-cells", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    metrics, calls, agreements, registry = run(args.outdir, args.maximum_cells, args.seed)
    metrics.to_csv(args.outdir / "real_data_upstream_metrics.tsv", sep="\t", index=False)
    calls.to_csv(args.outdir / "real_data_event_calls.tsv", sep="\t", index=False)
    agreements.to_csv(args.outdir / "upstream_event_agreement.tsv", sep="\t", index=False)
    registry.to_csv(args.outdir / "upstream_method_registry.tsv", sep="\t", index=False)
    (args.outdir / "run_config.json").write_text(
        json.dumps({"seed": args.seed, "maximum_cells_per_dataset": args.maximum_cells, "n_combinations": len(metrics), "native_GSVA_AUCell_claimed": False}, indent=2),
        encoding="utf-8",
    )
    report = [
        "# TED real-data upstream sensitivity",
        "",
        f"Executed {len(metrics)} method/trajectory combinations on two real expression matrices.",
        f"Median direction agreement: {metrics.direction_agreement.median():.3f}.",
        f"Median event-family overlap: {metrics.event_family_overlap.median():.3f}.",
        f"Maximum false E promotion: {metrics.false_e_promotion.max():.3f}.",
        f"Events triggering the <0.75 upstream-agreement gate: {int(agreements.upstream_disagreement_flag.sum())}/{len(agreements)}.",
        "",
        "The alternative activity rows are internal executable implementations. Native GSVA/AUCell package performance is not claimed.",
    ]
    (args.outdir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = []
    for path in sorted(args.outdir.glob("*")):
        if path.is_file() and path.name != "manifest.tsv":
            manifest.append({"file": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    pd.DataFrame(manifest).to_csv(args.outdir / "manifest.tsv", sep="\t", index=False)
    print(metrics.to_string(index=False))
    print(agreements.groupby("dataset_id").upstream_disagreement_flag.agg(["sum", "count"]).to_string())


if __name__ == "__main__":
    main()
