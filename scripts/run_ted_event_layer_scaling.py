from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "ted_submission_supplement" / "event_layer_scaling"
SEED = 20260715


PROFILES = {
    "quick": {"pathways": 16, "blocks": 3, "windows": 20, "permutations": 20, "genes_per_pathway": 4},
    "full": {"pathways": 64, "blocks": 10, "windows": 50, "permutations": 100, "genes_per_pathway": 4},
}


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024**2)


def bh(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.clip(ranked, 0, 1)
    return out


def run_case(n_cells: int, profile: str, seed: int) -> dict[str, object]:
    spec = PROFILES[profile]
    rng = np.random.default_rng(seed)
    baseline_memory = rss_mb()
    peak_memory = baseline_memory
    started = time.perf_counter()
    pseudotime = rng.random(n_cells, dtype=np.float32)
    block = rng.integers(0, spec["blocks"], size=n_cells, dtype=np.int16)
    activity = np.empty((n_cells, spec["pathways"]), dtype=np.float32)
    for pathway in range(spec["pathways"]):
        genes = rng.standard_normal((n_cells, spec["genes_per_pathway"]), dtype=np.float32)
        score = genes.mean(axis=1)
        if pathway < max(1, spec["pathways"] // 8):
            score += (1.0 / (1.0 + np.exp(-10 * (pseudotime - 0.55)))).astype(np.float32)
        activity[:, pathway] = score
        del genes, score
        peak_memory = max(peak_memory, rss_mb())
    upstream_seconds = time.perf_counter() - started

    event_started = time.perf_counter()
    window = np.minimum((pseudotime * spec["windows"]).astype(np.int32), spec["windows"] - 1)
    group = block.astype(np.int32) * spec["windows"] + window
    n_groups = spec["blocks"] * spec["windows"]
    counts = np.bincount(group, minlength=n_groups).astype(np.float64)
    aggregate = np.empty((n_groups, spec["pathways"]), dtype=np.float64)
    for pathway in range(spec["pathways"]):
        aggregate[:, pathway] = np.bincount(group, weights=activity[:, pathway], minlength=n_groups) / np.maximum(counts, 1)
    aggregate = aggregate.reshape(spec["blocks"], spec["windows"], spec["pathways"])
    block_condition = np.arange(spec["blocks"]) % 2
    case = aggregate[block_condition == 1].mean(axis=0)
    control = aggregate[block_condition == 0].mean(axis=0)
    observed = np.max(np.abs(case - control), axis=0)
    null = np.empty((spec["permutations"], spec["pathways"]), dtype=np.float64)
    for permutation in range(spec["permutations"]):
        labels = rng.permutation(block_condition)
        if not (labels == 1).any() or not (labels == 0).any():
            null[permutation] = np.nan
            continue
        delta = aggregate[labels == 1].mean(axis=0) - aggregate[labels == 0].mean(axis=0)
        null[permutation] = np.max(np.abs(delta), axis=0)
    event_p = (1 + np.sum(null >= observed[None, :], axis=0)) / (spec["permutations"] + 1)
    event_q = bh(event_p)
    event_seconds = time.perf_counter() - event_started
    peak_memory = max(peak_memory, rss_mb())
    total_seconds = time.perf_counter() - started
    result = {
        "profile": profile,
        "cells": n_cells,
        "pathways": spec["pathways"],
        "blocks": spec["blocks"],
        "windows": spec["windows"],
        "permutations": spec["permutations"],
        "upstream_scoring_seconds": upstream_seconds,
        "ted_event_layer_seconds": event_seconds,
        "total_seconds": total_seconds,
        "peak_rss_mb": peak_memory,
        "incremental_peak_rss_mb": peak_memory - baseline_memory,
        "cells_per_second_event_layer": n_cells / max(event_seconds, 1e-12),
        "pathway_windows_per_second": spec["pathways"] * spec["windows"] / max(event_seconds, 1e-12),
        "n_event_q_le_0_05": int((event_q <= 0.05).sum()),
        "parallel_workers": 1,
        "seed": seed,
        "status": "completed",
    }
    del activity, aggregate, null, pseudotime, block, window, group, counts
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark upstream scoring and TED event-layer scaling to one million cells")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cells", default="10000,50000,100000,500000,1000000")
    parser.add_argument("--profiles", default="quick,full")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cells = [int(value) for value in args.cells.split(",") if value]
    profiles = [value for value in args.profiles.split(",") if value]
    rows = []
    for profile in profiles:
        if profile not in PROFILES:
            raise ValueError(f"Unknown profile: {profile}")
        for index, n_cells in enumerate(cells):
            for repeat in range(args.repeats):
                seed = SEED + index + 100 * profiles.index(profile) + 10_000 * repeat
                row = run_case(n_cells, profile, seed)
                row["repeat"] = repeat + 1
                rows.append(row)
                print(
                    f"completed profile={profile} cells={n_cells} repeat={repeat + 1}/{args.repeats} "
                    f"total={row['total_seconds']:.3f}s peak={row['peak_rss_mb']:.1f}MB",
                    flush=True,
                )
    long = pd.DataFrame(rows)
    long.to_csv(args.outdir / "ted_event_layer_scaling.tsv", sep="\t", index=False)
    def q25(values: pd.Series) -> float:
        return float(values.quantile(0.25))

    def q75(values: pd.Series) -> float:
        return float(values.quantile(0.75))

    summary = long.groupby(["profile", "cells"], as_index=False).agg(
        repeats=("repeat", "nunique"),
        median_upstream_seconds=("upstream_scoring_seconds", "median"),
        iqr_low_upstream_seconds=("upstream_scoring_seconds", q25),
        iqr_high_upstream_seconds=("upstream_scoring_seconds", q75),
        median_event_layer_seconds=("ted_event_layer_seconds", "median"),
        iqr_low_event_layer_seconds=("ted_event_layer_seconds", q25),
        iqr_high_event_layer_seconds=("ted_event_layer_seconds", q75),
        median_total_seconds=("total_seconds", "median"),
        iqr_low_total_seconds=("total_seconds", q25),
        iqr_high_total_seconds=("total_seconds", q75),
        median_peak_rss_mb=("peak_rss_mb", "median"),
        iqr_low_peak_rss_mb=("peak_rss_mb", q25),
        iqr_high_peak_rss_mb=("peak_rss_mb", q75),
    )
    summary.to_csv(args.outdir / "ted_event_layer_scaling_summary.tsv", sep="\t", index=False)
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "total_memory_gb": psutil.virtual_memory().total / (1024**3),
        "numpy": np.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", "unset"),
    }
    (args.outdir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    report = [
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: run + validate",
        "- Origin Date: 2026-07-15",
        "- Verification Status: ANALYZED",
        "- Version Label: ted_event_layer_scaling_v1",
        "",
        "# TED runtime and memory scaling",
        "",
        "Upstream module scoring and the downstream TED event layer are timed separately.",
        "The one-million-cell cases use predeclared synthetic modules, biological-block aggregation, window summaries, and block-label permutation; single cells are not treated as independent inferential units.",
        "This benchmark does not equate the Rust PyFgsea core speed with end-to-end TED speed.",
        "Parallel scaling is reported separately using benchmark_ted_performance.py because this synthetic event-layer kernel is deliberately single-worker.",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
    ]
    (args.outdir / "scaling_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = []
    for path in sorted(args.outdir.glob("*")):
        if path.is_file():
            manifest.append({"file": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    pd.DataFrame(manifest).to_csv(args.outdir / "manifest.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
