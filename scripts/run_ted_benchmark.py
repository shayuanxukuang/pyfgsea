from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import json
import os
from pathlib import Path
import threading
import time
import tracemalloc
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

import pyfgsea


try:
    import anndata as ad
    from scipy import sparse
except Exception as exc:  # pragma: no cover - caught at runtime for clearer CLI errors
    ad = None
    sparse = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


METRIC_COLUMNS = [
    "benchmark_level",
    "profile",
    "case",
    "cells",
    "genes",
    "pathways",
    "windows_target",
    "actual_windows",
    "wall_time",
    "peak_rss_mb",
    "cpu_time",
    "windows_per_second",
    "pathway_windows_per_second",
    "memory_per_1000_pathways",
    "event_fdr_time_per_perm",
    "bootstrap_time_per_resample",
    "result_rows",
    "failed_windows",
    "diagnostic_warnings",
    "status",
    "error",
]


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    cells: int
    genes: int
    pathways: int
    windows: int
    density: float
    pathway_size: int
    nperm_nes: int
    sample_size: int
    n_perm: int
    n_boot: int

    @property
    def window_size(self) -> int:
        return max(20, int(round(self.cells * 0.08)))

    @property
    def step(self) -> int:
        if self.windows <= 1:
            return self.window_size
        return max(1, int(round((self.cells - self.window_size) / max(self.windows - 1, 1))))


PROFILES = {
    "tiny": BenchmarkSpec("tiny", 500, 1000, 50, 20, 0.04, 20, 8, 8, 2, 2),
    "small": BenchmarkSpec("small", 2000, 3000, 200, 50, 0.025, 25, 16, 12, 5, 5),
    "medium": BenchmarkSpec("medium", 10000, 10000, 1000, 100, 0.01, 30, 32, 16, 10, 10),
    "large": BenchmarkSpec("large", 50000, 20000, 5000, 200, 0.004, 30, 64, 16, 10, 10),
    "calibration": BenchmarkSpec("calibration", 5000, 5000, 200, 80, 0.015, 25, 16, 12, 20, 20),
    "graph": BenchmarkSpec("graph", 5000, 5000, 200, 80, 0.015, 25, 16, 12, 5, 5),
}


DEFAULT_SUITES = ("core", "trajectory", "rankers", "windows")
RELEASE_SUITES = (
    "core",
    "trajectory",
    "rankers",
    "windows",
    "calibration",
    "bootstrap",
    "graph",
    "end_to_end",
)
DEFAULT_RANKERS = (
    "mean_diff",
    "detection_weighted",
    "t_stat",
    "z_score",
    "cohens_d",
    "local_slope",
    "neighbor_contrast",
    "smooth_slope",
)


class ResourceMonitor:
    def __init__(self, interval: float = 0.02):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self._peak_rss = 0
        self._memory_metric = "rss"
        try:
            import psutil

            self._process = psutil.Process(os.getpid())
        except Exception:
            self._process = None
            self._memory_metric = "tracemalloc"

    def _sample(self):
        while not self._stop.is_set():
            self._peak_rss = max(self._peak_rss, self._rss_bytes())
            time.sleep(self.interval)

    def _rss_bytes(self) -> int:
        if self._process is None:
            return 0
        try:
            rss = self._process.memory_info().rss
            for child in self._process.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except Exception:
                    pass
            return int(rss)
        except Exception:
            return 0

    def start(self):
        gc.collect()
        tracemalloc.start()
        self._peak_rss = self._rss_bytes()
        self._stop.clear()
        if self._process is not None:
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        self._wall_start = time.perf_counter()
        self._cpu_start = time.process_time()

    def stop(self) -> dict[str, float | str]:
        wall = time.perf_counter() - self._wall_start
        cpu = time.process_time() - self._cpu_start
        current, peak_py = tracemalloc.get_traced_memory()
        del current
        tracemalloc.stop()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        peak = max(self._peak_rss, peak_py)
        return {
            "wall_time": wall,
            "cpu_time": cpu,
            "peak_rss_mb": peak / 1024 / 1024,
            "memory_metric": self._memory_metric,
        }


def profile_for(name: str) -> BenchmarkSpec:
    if name not in PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Choices: {', '.join(PROFILES)}")
    return PROFILES[name]


def _window_kwargs(spec: BenchmarkSpec) -> dict:
    return {
        "window_size": spec.window_size,
        "step": spec.step,
        "min_size": max(5, min(15, spec.pathway_size // 2)),
        "max_size": max(500, spec.pathway_size * 4),
        "nperm_nes": spec.nperm_nes,
        "sample_size": min(spec.sample_size, spec.pathway_size),
        "bin_width": None,
    }


def _graph_window_kwargs(spec: BenchmarkSpec) -> dict:
    return {
        "graph_radius": max(2, spec.window_size // 2),
        "target_span": 1.0 / max(spec.windows, 1),
        "span_step": 1.0 / max(spec.windows, 1),
        "min_cells": max(5, spec.window_size // 8),
        "max_cells": spec.window_size,
    }


def make_gene_sets(genes: list[str], n_pathways: int, pathway_size: int) -> dict[str, list[str]]:
    max_start = max(1, len(genes) - pathway_size)
    sets = {}
    for idx in range(n_pathways):
        start = (idx * max(3, pathway_size // 2)) % max_start
        sets[f"P{idx:05d}"] = genes[start : start + pathway_size]
    return sets


def make_scores(spec: BenchmarkSpec, seed: int) -> tuple[pd.Series, dict[str, list[str]]]:
    rng = np.random.default_rng(seed)
    genes = [f"Gene_{idx}" for idx in range(spec.genes)]
    scores = pd.Series(rng.normal(size=spec.genes), index=genes, name="score")
    return scores, make_gene_sets(genes, spec.pathways, spec.pathway_size)


def make_adata(spec: BenchmarkSpec, seed: int, graph: bool = False):
    if IMPORT_ERROR is not None:
        raise ImportError("TED benchmarks require anndata and scipy") from IMPORT_ERROR
    rng = np.random.default_rng(seed)
    genes = [f"Gene_{idx}" for idx in range(spec.genes)]
    data_rvs = lambda n: rng.gamma(shape=1.5, scale=1.0, size=n)
    X = sparse.random(
        spec.cells,
        spec.genes,
        density=spec.density,
        format="lil",
        random_state=seed,
        data_rvs=data_rvs,
    )
    pt = np.linspace(0.0, 1.0, spec.cells)
    signal = 4.0 * np.clip((pt - 0.2) / 0.8, 0.0, 1.0)
    X[:, : spec.pathway_size] = X[:, : spec.pathway_size] + signal[:, None]
    X = X.tocsr()

    obs = pd.DataFrame(index=[f"Cell_{idx}" for idx in range(spec.cells)])
    obs["dpt_pseudotime"] = pt
    obs["condition"] = np.where(np.arange(spec.cells) % 2 == 0, "control", "case")
    donor_id = np.arange(spec.cells) % 6
    obs["donor"] = np.where(
        obs["condition"].to_numpy() == "control",
        np.char.add("ctrl_", (donor_id % 3 + 1).astype(str)),
        np.char.add("case_", (donor_id % 3 + 1).astype(str)),
    )
    obs["branch"] = np.where(np.arange(spec.cells) % 2 == 0, "branch_a", "branch_b")
    obs["fate_prob"] = np.where(obs["branch"].to_numpy() == "branch_a", 0.9, 0.1)

    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
    if graph:
        rows = np.concatenate([np.arange(spec.cells - 1), np.arange(1, spec.cells)])
        cols = np.concatenate([np.arange(1, spec.cells), np.arange(spec.cells - 1)])
        values = np.ones(len(rows), dtype=float)
        adata.obsp["connectivities"] = sparse.csr_matrix(
            (values, (rows, cols)), shape=(spec.cells, spec.cells)
        )
    return adata, make_gene_sets(genes, spec.pathways, spec.pathway_size)


def _diagnostic_warnings(obj) -> int:
    count = 0
    if isinstance(obj, pd.DataFrame):
        if "calibration_warning" in obj.columns:
            count += int(obj["calibration_warning"].fillna("").astype(str).ne("").sum())
        diag = obj.attrs.get("graph_window_diagnostics")
        if isinstance(diag, pd.DataFrame) and "skipped" in diag.columns:
            count += int(diag["skipped"].sum())
    return count


def _failed_windows(obj) -> int:
    if isinstance(obj, pd.DataFrame):
        diag = obj.attrs.get("graph_window_diagnostics")
        if isinstance(diag, pd.DataFrame) and "skipped" in diag.columns:
            return int(diag["skipped"].sum())
    return 0


def _actual_windows(obj) -> int:
    if isinstance(obj, pd.DataFrame) and "window_id" in obj.columns:
        return int(obj["window_id"].nunique())
    return 0


def _result_rows(obj) -> int:
    if isinstance(obj, pd.DataFrame):
        return int(len(obj))
    if isinstance(obj, dict):
        return int(sum(len(value) for value in obj.values() if isinstance(value, pd.DataFrame)))
    if isinstance(obj, pyfgsea.TrajectoryEventResult):
        return int(sum(len(table) for table in obj.to_tables().values()))
    return 0


def measure_case(
    *,
    spec: BenchmarkSpec,
    benchmark_level: str,
    case: str,
    fn: Callable[[], object],
    n_perm: Optional[int] = None,
    n_boot: Optional[int] = None,
) -> dict:
    monitor = ResourceMonitor()
    status = "ok"
    error = ""
    output = None
    monitor.start()
    try:
        output = fn()
    except Exception as exc:
        status = "failed"
        error = repr(exc)
    metrics = monitor.stop()
    actual_windows = _actual_windows(output)
    result_rows = _result_rows(output)
    wall_time = float(metrics["wall_time"])
    peak_rss = float(metrics["peak_rss_mb"])
    pathways = spec.pathways
    row = {
        "benchmark_level": benchmark_level,
        "profile": spec.name,
        "case": case,
        "cells": spec.cells,
        "genes": spec.genes,
        "pathways": pathways,
        "windows_target": spec.windows,
        "actual_windows": actual_windows,
        "wall_time": wall_time,
        "peak_rss_mb": peak_rss,
        "cpu_time": float(metrics["cpu_time"]),
        "windows_per_second": actual_windows / wall_time if actual_windows and wall_time > 0 else np.nan,
        "pathway_windows_per_second": result_rows / wall_time if result_rows and wall_time > 0 else np.nan,
        "memory_per_1000_pathways": peak_rss / max(pathways / 1000.0, 1e-12),
        "event_fdr_time_per_perm": wall_time / n_perm if n_perm else np.nan,
        "bootstrap_time_per_resample": wall_time / n_boot if n_boot else np.nan,
        "result_rows": result_rows,
        "failed_windows": _failed_windows(output),
        "diagnostic_warnings": _diagnostic_warnings(output),
        "status": status,
        "error": error,
    }
    return row


def run_core_benchmark(spec: BenchmarkSpec, seed: int) -> list[dict]:
    scores, gene_sets = make_scores(spec, seed)
    return [
        measure_case(
            spec=spec,
            benchmark_level="core_gsea",
            case="run_gsea",
            fn=lambda: pyfgsea.run_gsea(
                scores,
                gene_sets,
                min_size=max(5, spec.pathway_size // 2),
                max_size=spec.pathway_size * 4,
                nperm_nes=spec.nperm_nes,
                sample_size=min(spec.sample_size, spec.pathway_size),
                seed=seed,
                bin_width=None,
            ),
        )
    ]


def run_trajectory_benchmark(spec: BenchmarkSpec, seed: int) -> list[dict]:
    adata, gene_sets = make_adata(spec, seed)
    kwargs = _window_kwargs(spec)
    return [
        measure_case(
            spec=spec,
            benchmark_level="trajectory_runner",
            case="mean_diff_cell_count",
            fn=lambda: pyfgsea.run_trajectory_gsea(
                adata,
                gene_sets,
                pseudotime_key="dpt_pseudotime",
                ranker="mean_diff",
                window_mode="cell_count",
                seed=seed,
                **kwargs,
            ),
        )
    ]


def run_ranker_benchmark(
    spec: BenchmarkSpec,
    seed: int,
    rankers: Iterable[str],
    include_heavy: bool = False,
) -> list[dict]:
    adata, gene_sets = make_adata(spec, seed)
    kwargs = _window_kwargs(spec)
    selected = list(rankers)
    if include_heavy and "wilcoxon" not in selected:
        selected.append("wilcoxon")
    rows = []
    for ranker in selected:
        rows.append(
            measure_case(
                spec=spec,
                benchmark_level="ranker",
                case=ranker,
                fn=lambda ranker=ranker: pyfgsea.run_trajectory_gsea(
                    adata,
                    gene_sets,
                    pseudotime_key="dpt_pseudotime",
                    ranker=ranker,
                    window_mode="cell_count",
                    seed=seed,
                    **kwargs,
                ),
            )
        )
    rows.append(
        measure_case(
            spec=spec,
            benchmark_level="ranker",
            case="branch_contrast",
            fn=lambda: pyfgsea.run_branch_gsea(
                adata,
                gene_sets,
                branch_key="branch",
                mode="branch_contrast",
                pseudotime_key="dpt_pseudotime",
                seed=seed,
                window_size=kwargs["window_size"],
                step=kwargs["step"],
                min_reference_cells=10,
                min_size=kwargs["min_size"],
                max_size=kwargs["max_size"],
                nperm_nes=kwargs["nperm_nes"],
                sample_size=kwargs["sample_size"],
                bin_width=None,
            ),
        )
    )
    return rows


def run_window_benchmark(spec: BenchmarkSpec, seed: int) -> list[dict]:
    adata, gene_sets = make_adata(spec, seed, graph=True)
    kwargs = _window_kwargs(spec)
    rows = []
    for mode in ("cell_count", "pseudotime_span", "adaptive"):
        extra = {}
        if mode != "cell_count":
            extra = {
                "target_span": 1.0 / max(spec.windows, 1),
                "span_step": 1.0 / max(spec.windows, 1),
                "min_cells": max(10, spec.window_size // 3),
                "max_cells": spec.window_size,
            }
        rows.append(
            measure_case(
                spec=spec,
                benchmark_level="window",
                case=mode,
                fn=lambda mode=mode, extra=extra: pyfgsea.run_trajectory_gsea(
                    adata,
                    gene_sets,
                    pseudotime_key="dpt_pseudotime",
                    ranker="mean_diff",
                    window_mode=mode,
                    seed=seed,
                    **kwargs,
                    **extra,
                ),
            )
        )
    rows.append(
        measure_case(
            spec=spec,
            benchmark_level="window",
            case="graph_adaptive",
            fn=lambda: pyfgsea.run_trajectory_gsea(
                adata,
                gene_sets,
                pseudotime_key="dpt_pseudotime",
                ranker="mean_diff",
                window_mode="graph_adaptive",
                graph_key="connectivities",
                cell_weight_key="fate_prob",
                experimental=True,
                seed=seed,
                **kwargs,
                **_graph_window_kwargs(spec),
            ),
        )
    )
    return rows


def run_calibration_benchmark(
    spec: BenchmarkSpec,
    seed: int,
    n_perm: Optional[int],
    threads: Optional[int] = None,
) -> list[dict]:
    n_perm = spec.n_perm if n_perm is None else n_perm
    adata, gene_sets = make_adata(spec, seed)
    kwargs = _window_kwargs(spec)
    return [
        measure_case(
            spec=spec,
            benchmark_level="calibration",
            case="event_fdr_pseudotime_within_replicate",
            n_perm=n_perm,
            fn=lambda: pyfgsea.estimate_event_fdr(
                adata=adata,
                gmt_path=gene_sets,
                pseudotime_key="dpt_pseudotime",
                null="pseudotime_within_replicate_permutation",
                replicate_key="donor",
                n_perm=n_perm,
                n_jobs=max(1, int(threads)) if threads is not None else 1,
                seed=seed,
                event_kwargs={"min_consecutive": 1},
                **kwargs,
            ),
        )
    ]


def run_bootstrap_benchmark(spec: BenchmarkSpec, seed: int, n_boot: Optional[int]) -> list[dict]:
    n_boot = spec.n_boot if n_boot is None else n_boot
    adata, gene_sets = make_adata(spec, seed)
    kwargs = _window_kwargs(spec)
    return [
        measure_case(
            spec=spec,
            benchmark_level="bootstrap",
            case="cell_bootstrap",
            n_boot=n_boot,
            fn=lambda: pyfgsea.bootstrap_trajectory_gsea(
                adata,
                gene_sets,
                pseudotime_key="dpt_pseudotime",
                n_boot=n_boot,
                resample="cells_within_windows",
                seed=seed,
                event_kwargs={"min_consecutive": 1},
                **kwargs,
            ),
        )
    ]


def run_graph_benchmark(spec: BenchmarkSpec, seed: int) -> list[dict]:
    adata, gene_sets = make_adata(spec, seed, graph=True)
    kwargs = _window_kwargs(spec)
    return [
        measure_case(
            spec=spec,
            benchmark_level="graph",
            case="graph_adaptive_detection_weighted",
            fn=lambda: pyfgsea.run_trajectory_gsea(
                adata,
                gene_sets,
                pseudotime_key="dpt_pseudotime",
                ranker="detection_weighted",
                window_mode="graph_adaptive",
                graph_key="connectivities",
                cell_weight_key="fate_prob",
                experimental=True,
                seed=seed,
                **kwargs,
                **_graph_window_kwargs(spec),
            ),
        )
    ]


def run_end_to_end_benchmark(
    spec: BenchmarkSpec,
    seed: int,
    n_perm: Optional[int],
    threads: Optional[int] = None,
) -> list[dict]:
    n_perm = spec.n_perm if n_perm is None else n_perm
    adata, gene_sets = make_adata(spec, seed)
    kwargs = _window_kwargs(spec)

    def run():
        res = pyfgsea.run_trajectory_gsea(
            adata,
            gene_sets,
            pseudotime_key="dpt_pseudotime",
            ranker="detection_weighted",
            window_mode="adaptive",
            target_span=1.0 / max(spec.windows, 1),
            span_step=1.0 / max(spec.windows, 1),
            min_cells=max(10, spec.window_size // 3),
            max_cells=spec.window_size,
            seed=seed,
            **kwargs,
        )
        events = pyfgsea.summarize_events(res, min_consecutive=1)
        event_fdr = pyfgsea.estimate_event_fdr(
            adata=adata,
            gmt_path=gene_sets,
            result=res,
            pseudotime_key="dpt_pseudotime",
            null="pseudotime_within_replicate_permutation",
            replicate_key="donor",
            n_perm=n_perm,
            n_jobs=max(1, int(threads)) if threads is not None else 1,
            seed=seed,
            event_kwargs={"min_consecutive": 1},
            **kwargs,
        )
        return pyfgsea.make_trajectory_event_result(
            adata=adata,
            gmt_path=gene_sets,
            results=res,
            events=events,
            event_fdr=event_fdr,
            seed=seed,
            replicate_key="donor",
        )

    return [
        measure_case(
            spec=spec,
            benchmark_level="end_to_end",
            case="trajectory_event_result",
            n_perm=n_perm,
            fn=run,
        )
    ]


def run_benchmark_suite(
    profile: str = "tiny",
    suites: Iterable[str] = DEFAULT_SUITES,
    seed: int = 1,
    rankers: Iterable[str] = DEFAULT_RANKERS,
    include_heavy: bool = False,
    n_perm: Optional[int] = None,
    n_boot: Optional[int] = None,
    threads: Optional[int] = None,
) -> pd.DataFrame:
    if threads is not None:
        os.environ["RAYON_NUM_THREADS"] = str(threads)
    spec = profile_for(profile)
    rows = []
    for suite in suites:
        if suite == "all":
            rows.extend(
                run_benchmark_suite(
                    profile=profile,
                    suites=RELEASE_SUITES,
                    seed=seed,
                    rankers=rankers,
                    include_heavy=include_heavy,
                    n_perm=n_perm,
                    n_boot=n_boot,
                    threads=threads,
                ).to_dict("records")
            )
        elif suite == "core":
            rows.extend(run_core_benchmark(spec, seed))
        elif suite == "trajectory":
            rows.extend(run_trajectory_benchmark(spec, seed))
        elif suite == "rankers":
            rows.extend(run_ranker_benchmark(spec, seed, rankers, include_heavy=include_heavy))
        elif suite == "windows":
            rows.extend(run_window_benchmark(spec, seed))
        elif suite == "calibration":
            rows.extend(run_calibration_benchmark(spec, seed, n_perm=n_perm, threads=threads))
        elif suite == "bootstrap":
            rows.extend(run_bootstrap_benchmark(spec, seed, n_boot=n_boot))
        elif suite == "graph":
            rows.extend(run_graph_benchmark(spec, seed))
        elif suite == "end_to_end":
            rows.extend(run_end_to_end_benchmark(spec, seed, n_perm=n_perm, threads=threads))
        else:
            raise ValueError(f"Unknown benchmark suite '{suite}'")
    df = pd.DataFrame(rows)
    for col in METRIC_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[METRIC_COLUMNS]


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run PyFgsea-TED performance benchmarks.")
    parser.add_argument("--profile", default="tiny", choices=sorted(PROFILES))
    parser.add_argument(
        "--suite",
        default=",".join(DEFAULT_SUITES),
        help="Comma-separated suites: core,trajectory,rankers,windows,calibration,bootstrap,graph,end_to_end,all",
    )
    parser.add_argument("--outdir", default="results/ted_benchmark")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--rankers", default=",".join(DEFAULT_RANKERS))
    parser.add_argument("--include-heavy", action="store_true", help="Include heavy rankers such as wilcoxon")
    parser.add_argument("--n-perm", type=int, default=None)
    parser.add_argument("--n-boot", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Also write JSON records")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    suites = parse_csv(args.suite)
    rankers = parse_csv(args.rankers)
    df = run_benchmark_suite(
        profile=args.profile,
        suites=suites,
        seed=args.seed,
        rankers=rankers,
        include_heavy=args.include_heavy,
        n_perm=args.n_perm,
        n_boot=args.n_boot,
        threads=args.threads,
    )
    stem = f"ted_benchmark_{args.profile}_{'_'.join(suites)}"
    csv_path = outdir / f"{stem}.csv"
    df.to_csv(csv_path, index=False)
    if args.json:
        json_path = outdir / f"{stem}.json"
        json_path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    print(df.to_string(index=False))
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
