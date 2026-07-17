from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import pyfgsea


HERE = Path(__file__).resolve().parent
RUN_TED_BENCHMARK = HERE / "run_ted_benchmark.py"


def _load_legacy_benchmark():
    spec = importlib.util.spec_from_file_location("run_ted_benchmark", RUN_TED_BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


legacy = _load_legacy_benchmark()


TED_PROFILES = (
    "core",
    "trajectory",
    "rankers",
    "windows",
    "calibration",
    "bootstrap",
    "graph",
    "end_to_end",
)

EXTRA_METRIC_COLUMNS = [
    "repeat",
    "seed",
    "ranker_time",
    "gsea_time",
    "event_summary_time",
    "graph_window_construction_time",
]

REGRESSION_KEYS = [
    "benchmark_level",
    "profile",
    "case",
    "cells",
    "genes",
    "pathways",
    "windows_target",
]


def _tokens(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        out.extend(item.strip() for item in str(value).split(",") if item.strip())
    return out


def _normalize_profiles(values: Iterable[str]) -> list[str]:
    profiles = _tokens(values)
    if not profiles:
        return ["core", "trajectory", "rankers", "windows"]
    if "all" in profiles:
        return list(TED_PROFILES)
    unknown = sorted(set(profiles) - set(TED_PROFILES))
    if unknown:
        raise ValueError(f"Unknown TED benchmark profile(s): {', '.join(unknown)}")
    return profiles


def _normalize_sizes(values: Iterable[str]) -> list[str]:
    sizes = _tokens(values)
    if not sizes:
        return ["tiny"]
    aliases = {"ci": "tiny", "tiny_ci": "tiny", "small_smoke": "small"}
    sizes = [aliases.get(size, size) for size in sizes]
    if "all" in sizes:
        return list(legacy.PROFILES)
    unknown = sorted(set(sizes) - set(legacy.PROFILES))
    if unknown:
        raise ValueError(f"Unknown benchmark size(s): {', '.join(unknown)}")
    return sizes


def _augment_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in EXTRA_METRIC_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    out["ranker_time"] = np.where(
        out["benchmark_level"].eq("ranker"), out["wall_time"], out["ranker_time"]
    )
    out["gsea_time"] = np.where(
        out["benchmark_level"].eq("core_gsea"), out["wall_time"], out["gsea_time"]
    )
    out["event_summary_time"] = np.where(
        out["benchmark_level"].eq("event_summary"),
        out["wall_time"],
        out["event_summary_time"],
    )
    out["graph_window_construction_time"] = np.where(
        out["benchmark_level"].eq("graph_window"),
        out["wall_time"],
        out["graph_window_construction_time"],
    )
    return out


def _phase_probe_event_summary(size: str, seed: int, repeat: int) -> pd.DataFrame:
    spec = legacy.profile_for(size)
    adata, gene_sets = legacy.make_adata(spec, seed)
    kwargs = legacy._window_kwargs(spec)
    res = pyfgsea.run_trajectory_gsea(
        adata,
        gene_sets,
        pseudotime_key="dpt_pseudotime",
        ranker="detection_weighted",
        window_mode="cell_count",
        seed=seed,
        **kwargs,
    )
    row = legacy.measure_case(
        spec=spec,
        benchmark_level="event_summary",
        case="summarize_events",
        fn=lambda: pyfgsea.summarize_events(res, min_consecutive=1),
    )
    row["repeat"] = repeat
    row["seed"] = seed
    return _augment_metric_columns(pd.DataFrame([row]))


def _phase_probe_graph_window(size: str, seed: int, repeat: int) -> pd.DataFrame:
    spec = legacy.profile_for(size)
    adata, _gene_sets = legacy.make_adata(spec, seed, graph=True)
    pt = adata.obs["dpt_pseudotime"].to_numpy(dtype=float)
    order = np.argsort(pt)
    graph_kwargs = legacy._graph_window_kwargs(spec)
    holder = {}

    def build_index():
        holder["window_index"] = pyfgsea.build_window_index(
            adata,
            order=order,
            pt=pt,
            window_mode="graph_adaptive",
            graph_key="connectivities",
            branch_key="branch",
            fate_weights=adata.obs["fate_prob"].to_numpy(dtype=float),
            **graph_kwargs,
        )
        return holder["window_index"]

    row = legacy.measure_case(
        spec=spec,
        benchmark_level="graph_window",
        case="graph_adaptive_window_index",
        fn=build_index,
    )
    if row["status"] == "ok" and "window_index" in holder:
        window_index = holder["window_index"]
        row["actual_windows"] = int(len(window_index.windows))
        row["result_rows"] = int(len(window_index.windows))
        if not window_index.diagnostics.empty and "skipped" in window_index.diagnostics:
            row["failed_windows"] = int(window_index.diagnostics["skipped"].sum())
            row["diagnostic_warnings"] = int(window_index.diagnostics["skipped"].sum())
    row["repeat"] = repeat
    row["seed"] = seed
    return _augment_metric_columns(pd.DataFrame([row]))


def run_performance_benchmark(
    *,
    profiles: Iterable[str] = ("core", "trajectory", "rankers", "windows"),
    sizes: Iterable[str] = ("tiny",),
    repeats: int = 1,
    seed: int = 1,
    rankers: Iterable[str] = legacy.DEFAULT_RANKERS,
    include_heavy: bool = False,
    n_perm: Optional[int] = None,
    n_boot: Optional[int] = None,
    threads: Optional[int] = None,
    phase_probes: bool = True,
) -> pd.DataFrame:
    profiles = _normalize_profiles(profiles)
    sizes = _normalize_sizes(sizes)
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    frames = []
    for size in sizes:
        for repeat in range(repeats):
            repeat_seed = seed + repeat
            df = legacy.run_benchmark_suite(
                profile=size,
                suites=profiles,
                seed=repeat_seed,
                rankers=rankers,
                include_heavy=include_heavy,
                n_perm=n_perm,
                n_boot=n_boot,
                threads=threads,
            )
            df["repeat"] = repeat
            df["seed"] = repeat_seed
            frames.append(_augment_metric_columns(df))

            if phase_probes and ("trajectory" in profiles or "end_to_end" in profiles):
                frames.append(_phase_probe_event_summary(size, repeat_seed, repeat))
            if phase_probes and "graph" in profiles:
                frames.append(_phase_probe_graph_window(size, repeat_seed, repeat))

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    ordered = list(legacy.METRIC_COLUMNS) + [
        column for column in EXTRA_METRIC_COLUMNS if column not in legacy.METRIC_COLUMNS
    ]
    for column in ordered:
        if column not in out.columns:
            out[column] = np.nan
    return out[ordered]


def summarize_repeats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    numeric = [
        "wall_time",
        "peak_rss_mb",
        "cpu_time",
        "windows_per_second",
        "pathway_windows_per_second",
        "memory_per_1000_pathways",
        "event_fdr_time_per_perm",
        "bootstrap_time_per_resample",
        "ranker_time",
        "gsea_time",
        "event_summary_time",
        "graph_window_construction_time",
        "result_rows",
        "failed_windows",
        "diagnostic_warnings",
    ]
    grouped = df.groupby(REGRESSION_KEYS, dropna=False, as_index=False)
    rows = []
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(REGRESSION_KEYS, key))
        row["repeats"] = int(len(group))
        row["status_ok_fraction"] = float(group["status"].eq("ok").mean())
        for column in numeric:
            if column in group.columns:
                values = pd.to_numeric(group[column], errors="coerce")
            else:
                values = pd.Series([np.nan] * len(group), index=group.index)
            row[f"{column}_mean"] = float(values.mean()) if values.notna().any() else np.nan
            row[f"{column}_sd"] = (
                float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
            )
            row[f"{column}_median"] = (
                float(values.median()) if values.notna().any() else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _baseline_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "wall_time_mean" in df.columns:
        return df.copy()
    return summarize_repeats(_augment_metric_columns(df))


def compare_to_baseline(
    current: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    runtime_threshold: float = 1.25,
    memory_threshold: float = 1.25,
    throughput_threshold: float = 0.80,
) -> pd.DataFrame:
    cur = _baseline_summary(current)
    base = _baseline_summary(baseline)
    if cur.empty or base.empty:
        return pd.DataFrame()
    merged = cur.merge(base, on=REGRESSION_KEYS, suffixes=("_current", "_baseline"))
    if merged.empty:
        return merged

    merged["runtime_ratio"] = (
        merged["wall_time_mean_current"] / merged["wall_time_mean_baseline"]
    )
    merged["memory_ratio"] = (
        merged["peak_rss_mb_mean_current"] / merged["peak_rss_mb_mean_baseline"]
    )
    merged["throughput_ratio"] = (
        merged["pathway_windows_per_second_mean_current"]
        / merged["pathway_windows_per_second_mean_baseline"]
    )
    merged["runtime_regression"] = merged["runtime_ratio"] > runtime_threshold
    merged["memory_regression"] = merged["memory_ratio"] > memory_threshold
    merged["throughput_regression"] = (
        merged["throughput_ratio"].notna()
        & (merged["throughput_ratio"] < throughput_threshold)
    )
    merged["regression_status"] = np.where(
        merged[["runtime_regression", "memory_regression", "throughput_regression"]]
        .any(axis=1),
        "regression",
        "ok",
    )
    return merged


def write_outputs(
    df: pd.DataFrame,
    *,
    outdir: Path,
    config: dict,
    baseline: Optional[pd.DataFrame] = None,
    runtime_threshold: float = 1.25,
    memory_threshold: float = 1.25,
    throughput_threshold: float = 0.80,
) -> tuple[Path, Path, Optional[Path], pd.DataFrame]:
    outdir.mkdir(parents=True, exist_ok=True)
    long_path = outdir / "ted_performance_long.csv"
    summary_path = outdir / "ted_performance_summary.csv"
    config_path = outdir / "ted_performance_config.json"

    summary = summarize_repeats(df)
    df.to_csv(long_path, index=False)
    summary.to_csv(summary_path, index=False)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    regression_path = None
    regression = pd.DataFrame()
    if baseline is not None:
        regression = compare_to_baseline(
            df,
            baseline,
            runtime_threshold=runtime_threshold,
            memory_threshold=memory_threshold,
            throughput_threshold=throughput_threshold,
        )
        regression_path = outdir / "ted_performance_regression.csv"
        regression.to_csv(regression_path, index=False)

    return long_path, summary_path, regression_path, regression


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PyFgsea-TED performance benchmarks with repeats and regression gates."
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["core", "trajectory", "rankers", "windows"],
        help="TED layers: core trajectory rankers windows calibration bootstrap graph end_to_end all",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["tiny"],
        help="Benchmark sizes: tiny small medium large calibration graph all",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--outdir", default="results/ted_performance")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--rankers", default=",".join(legacy.DEFAULT_RANKERS))
    parser.add_argument("--include-heavy", action="store_true")
    parser.add_argument("--n-perm", type=int, default=None)
    parser.add_argument("--n-boot", type=int, default=None)
    parser.add_argument("--skip-phase-probes", action="store_true")
    parser.add_argument("--baseline", default=None, help="Previous release long or summary CSV")
    parser.add_argument("--runtime-threshold", type=float, default=1.25)
    parser.add_argument("--memory-threshold", type=float, default=1.25)
    parser.add_argument("--throughput-threshold", type=float, default=0.80)
    parser.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args()

    profiles = _normalize_profiles(args.profiles)
    sizes = _normalize_sizes(args.sizes)
    rankers = _tokens([args.rankers])
    start = time.time()
    df = run_performance_benchmark(
        profiles=profiles,
        sizes=sizes,
        repeats=args.repeats,
        seed=args.seed,
        rankers=rankers,
        include_heavy=args.include_heavy,
        n_perm=args.n_perm,
        n_boot=args.n_boot,
        threads=args.threads,
        phase_probes=not args.skip_phase_probes,
    )
    baseline = pd.read_csv(args.baseline) if args.baseline else None
    config = {
        "profiles": profiles,
        "sizes": sizes,
        "repeats": args.repeats,
        "seed": args.seed,
        "threads": args.threads,
        "rankers": rankers,
        "include_heavy": bool(args.include_heavy),
        "n_perm": args.n_perm,
        "n_boot": args.n_boot,
        "phase_probes": not args.skip_phase_probes,
        "baseline": args.baseline,
        "runtime_threshold": args.runtime_threshold,
        "memory_threshold": args.memory_threshold,
        "throughput_threshold": args.throughput_threshold,
        "elapsed_wall_time": time.time() - start,
    }
    long_path, summary_path, regression_path, regression = write_outputs(
        df,
        outdir=Path(args.outdir),
        config=config,
        baseline=baseline,
        runtime_threshold=args.runtime_threshold,
        memory_threshold=args.memory_threshold,
        throughput_threshold=args.throughput_threshold,
    )

    print(f"Wrote {long_path}")
    print(f"Wrote {summary_path}")
    if regression_path is not None:
        print(f"Wrote {regression_path}")
        n_regressions = int(regression["regression_status"].eq("regression").sum())
        print(f"Regression rows: {n_regressions}")
        if args.fail_on_regression and n_regressions:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
