"""Build descriptive Figure 1 supplements from the functional comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPECTED_COUNTS = {"publication_main": 100, "ties_predeclared": 60}
EXPECTED_LANES = {
    "legacy": {"pyfgsea": "0.1.4", "fgsea": "1.32.2"},
    "current": {"pyfgsea": "0.2.0", "fgsea": "1.38.0"},
}
LANES = tuple(EXPECTED_LANES)
SCENARIOS = tuple(EXPECTED_COUNTS)
REQUIRED_VALIDATION = {
    "installed_wheels_executed": True,
    "reference_versions_executed": True,
    "raw_rows_complete": True,
    "metrics_recomputed_from_raw_rows": True,
    "manual_metric_overrides": False,
    "hashes_are_pass_fail_checks": False,
}
REQUIRED_RAW_COLUMNS = {
    "lane",
    "scenario",
    "pathway",
    "py_es",
    "r_es",
    "es_difference",
    "py_nes",
    "r_nes",
    "nes_difference",
    "py_pval",
    "r_pval",
    "py_padj",
    "r_padj",
    "py_neg_log10_pval",
    "r_neg_log10_pval",
    "neg_log10_pval_difference",
}
REQUIRED_TIMING_COLUMNS = {
    "lane",
    "scenario",
    "engine",
    "measurement_scope",
    "elapsed_seconds",
    "peak_rss_bytes",
    "peak_increment_bytes",
}
EXPECTED_SCOPES = {
    "pyfgsea": "run_gsea_call_only",
    "r_fgsea": "Rscript_process_and_internal_fgsea",
}


class SupplementError(RuntimeError):
    """Raised when a functional Figure 1 result is incomplete or inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise SupplementError(f"required file was not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SupplementError(f"invalid JSON file: {path}") from error
    if not isinstance(value, Mapping):
        raise SupplementError(f"JSON root must be an object: {path}")
    return value


def _input_paths(result_dir: Path) -> dict[str, Path]:
    root = result_dir.expanduser().resolve()
    if not root.is_dir():
        raise SupplementError(f"Figure 1 result directory was not found: {root}")
    paths = {
        "root": root,
        "comparison": root / "comparison_result.json",
        "raw": root / "figure1_pathway_level_raw.tsv",
        "timing": root / "figure1_runtime_memory.tsv",
    }
    for name in ("comparison", "raw", "timing"):
        path = paths[name].resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise SupplementError(f"{name} input must remain inside {root}") from error
        if not path.is_file():
            raise SupplementError(f"required Figure 1 file was not found: {path}")
        paths[name] = path
    return paths


def _validate_comparison(result: Mapping[str, Any]) -> None:
    if (
        result.get("schema_version") != 1
        or result.get("kind") != "figure1_functional_comparison"
        or result.get("status") != "complete"
    ):
        raise SupplementError("Figure 1 comparison has the wrong identity or status")
    lanes = result.get("lanes")
    if not isinstance(lanes, Mapping) or any(
        lanes.get(lane) != expected for lane, expected in EXPECTED_LANES.items()
    ):
        raise SupplementError("Figure 1 comparison has the wrong lane versions")
    validation = result.get("validation")
    if not isinstance(validation, Mapping) or any(
        validation.get(key) is not expected
        for key, expected in REQUIRED_VALIDATION.items()
    ):
        raise SupplementError("Figure 1 comparison lacks required execution checks")


def _numeric(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    for column in columns:
        try:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise SupplementError(
                f"{label} column {column!r} is not numeric"
            ) from error
        if not np.isfinite(values).all():
            raise SupplementError(
                f"{label} column {column!r} contains non-finite values"
            )


def _read_raw(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    missing = sorted(REQUIRED_RAW_COLUMNS.difference(frame.columns))
    if missing:
        raise SupplementError(f"raw Figure 1 table is missing columns: {missing}")
    expected_rows = sum(EXPECTED_COUNTS.values()) * len(LANES)
    if len(frame) != expected_rows:
        raise SupplementError(
            f"raw Figure 1 table has {len(frame)} rows, not {expected_rows}"
        )
    if frame.duplicated(["lane", "scenario", "pathway"]).any():
        raise SupplementError("raw Figure 1 table contains duplicate pathway rows")
    counts = frame.groupby(["lane", "scenario"]).size().to_dict()
    expected = {
        (lane, scenario): count
        for lane in LANES
        for scenario, count in EXPECTED_COUNTS.items()
    }
    if counts != expected:
        raise SupplementError("raw Figure 1 table has the wrong lane/scenario lattice")
    numeric = sorted(REQUIRED_RAW_COLUMNS.difference({"lane", "scenario", "pathway"}))
    _numeric(frame, numeric, "raw Figure 1")
    for column in ("py_pval", "r_pval", "py_padj", "r_padj"):
        if not ((frame[column] >= 0.0) & (frame[column] <= 1.0)).all():
            raise SupplementError(f"raw Figure 1 {column} must be in [0, 1]")
    checks = {
        "es_difference": frame["py_es"] - frame["r_es"],
        "nes_difference": frame["py_nes"] - frame["r_nes"],
        "neg_log10_pval_difference": (
            frame["py_neg_log10_pval"] - frame["r_neg_log10_pval"]
        ),
    }
    for column, expected_values in checks.items():
        if not np.allclose(frame[column], expected_values, rtol=1e-13, atol=1e-14):
            raise SupplementError(f"raw Figure 1 {column} is inconsistent")
    return frame


def _read_timing(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    missing = sorted(REQUIRED_TIMING_COLUMNS.difference(frame.columns))
    if missing:
        raise SupplementError(f"Figure 1 timing table is missing columns: {missing}")
    expected = {
        (lane, scenario, engine)
        for lane in LANES
        for scenario in SCENARIOS
        for engine in EXPECTED_SCOPES
    }
    actual = set(zip(frame["lane"], frame["scenario"], frame["engine"]))
    if len(frame) != len(expected) or actual != expected:
        raise SupplementError("Figure 1 timing table has the wrong result lattice")
    if frame.duplicated(["lane", "scenario", "engine"]).any():
        raise SupplementError("Figure 1 timing table contains duplicates")
    _numeric(
        frame,
        ("elapsed_seconds", "peak_rss_bytes", "peak_increment_bytes"),
        "Figure 1 timing",
    )
    if (frame["elapsed_seconds"] <= 0.0).any():
        raise SupplementError("Figure 1 elapsed time must be positive")
    for engine, scope in EXPECTED_SCOPES.items():
        observed = set(frame.loc[frame["engine"] == engine, "measurement_scope"])
        if observed != {scope}:
            raise SupplementError(f"Figure 1 timing scope is wrong for {engine}")
    return frame


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    value = float(np.corrcoef(left, right)[0, 1])
    if not math.isfinite(value):
        raise SupplementError("agreement correlation is undefined")
    return value


def _bland_altman_rows(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = {
        "NES": ("py_nes", "r_nes"),
        "neg_log10_pvalue": ("py_neg_log10_pval", "r_neg_log10_pval"),
    }
    for (lane, scenario), group in raw.groupby(["lane", "scenario"], sort=True):
        for metric, (py_column, r_column) in metrics.items():
            py_values = group[py_column].to_numpy(dtype=float)
            r_values = group[r_column].to_numpy(dtype=float)
            difference = py_values - r_values
            bias = float(difference.mean())
            standard_deviation = float(difference.std(ddof=1))
            rows.append(
                {
                    "lane": lane,
                    "scenario": scenario,
                    "metric": metric,
                    "n": len(group),
                    "mean_bias_py_minus_r": bias,
                    "sd_difference": standard_deviation,
                    "lower_95_limit": bias - 1.96 * standard_deviation,
                    "upper_95_limit": bias + 1.96 * standard_deviation,
                    "rmse": float(np.sqrt(np.mean(np.square(difference)))),
                    "median_absolute_difference": float(np.median(np.abs(difference))),
                    "maximum_absolute_difference": float(np.max(np.abs(difference))),
                    "pearson": _correlation(py_values, r_values),
                    "spearman": _correlation(
                        pd.Series(py_values).rank(method="average").to_numpy(),
                        pd.Series(r_values).rank(method="average").to_numpy(),
                    ),
                }
            )
    return pd.DataFrame(rows)


def _overlap_rows(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (lane, scenario), group in raw.groupby(["lane", "scenario"], sort=True):
        py_top = set(
            group.assign(abs_nes=group["py_nes"].abs())
            .sort_values(["abs_nes", "pathway"], ascending=[False, True])
            .head(10)["pathway"]
        )
        r_top = set(
            group.assign(abs_nes=group["r_nes"].abs())
            .sort_values(["abs_nes", "pathway"], ascending=[False, True])
            .head(10)["pathway"]
        )
        py_fdr = set(group.loc[group["py_padj"] < 0.05, "pathway"])
        r_fdr = set(group.loc[group["r_padj"] < 0.05, "pathway"])
        for name, py_set, r_set in (
            ("top10_by_absolute_NES", py_top, r_top),
            ("FDR_below_0.05", py_fdr, r_fdr),
        ):
            union = py_set | r_set
            intersection = py_set & r_set
            denominator = max(len(py_set), len(r_set))
            rows.append(
                {
                    "lane": lane,
                    "scenario": scenario,
                    "set_definition": name,
                    "py_count": len(py_set),
                    "r_count": len(r_set),
                    "intersection_count": len(intersection),
                    "union_count": len(union),
                    "overlap_fraction_max_set": (
                        len(intersection) / denominator if denominator else 1.0
                    ),
                    "jaccard": len(intersection) / len(union) if union else 1.0,
                    "both_sets_empty": not union,
                }
            )
    return pd.DataFrame(rows)


def _tail_rows(raw: pd.DataFrame, count: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for (lane, scenario), group in raw.groupby(["lane", "scenario"], sort=True):
        ranked = group.assign(
            tail_depth=group[["py_neg_log10_pval", "r_neg_log10_pval"]].max(axis=1),
            absolute_tail_difference=group["neg_log10_pval_difference"].abs(),
        ).sort_values(
            ["tail_depth", "absolute_tail_difference", "pathway"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        chosen = ranked.head(count).copy()
        chosen["selection_rank"] = np.arange(1, len(chosen) + 1)
        selected.append(chosen)
        summaries.append(
            {
                "lane": lane,
                "scenario": scenario,
                "n_selected": len(chosen),
                "maximum_tail_depth": float(chosen["tail_depth"].max()),
                "median_absolute_neg_log10_pvalue_difference": float(
                    chosen["absolute_tail_difference"].median()
                ),
                "maximum_absolute_neg_log10_pvalue_difference": float(
                    chosen["absolute_tail_difference"].max()
                ),
                "median_absolute_nes_difference": float(
                    chosen["nes_difference"].abs().median()
                ),
                "maximum_absolute_nes_difference": float(
                    chosen["nes_difference"].abs().max()
                ),
            }
        )
    columns = [
        "lane",
        "scenario",
        "selection_rank",
        "pathway",
        "py_nes",
        "r_nes",
        "nes_difference",
        "py_pval",
        "r_pval",
        "py_neg_log10_pval",
        "r_neg_log10_pval",
        "neg_log10_pval_difference",
        "tail_depth",
        "absolute_tail_difference",
    ]
    return pd.concat(selected, ignore_index=True)[columns], pd.DataFrame(summaries)


def _runtime_rows(timing: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (lane, scenario), group in timing.groupby(["lane", "scenario"], sort=True):
        by_engine = group.set_index("engine")
        py_row = by_engine.loc["pyfgsea"]
        r_row = by_engine.loc["r_fgsea"]
        rows.append(
            {
                "lane": lane,
                "scenario": scenario,
                "py_elapsed_seconds": py_row["elapsed_seconds"],
                "r_elapsed_seconds": r_row["elapsed_seconds"],
                "descriptive_r_over_py_elapsed_ratio": (
                    r_row["elapsed_seconds"] / py_row["elapsed_seconds"]
                ),
                "py_peak_rss_bytes": py_row["peak_rss_bytes"],
                "r_peak_rss_bytes": r_row["peak_rss_bytes"],
                "py_peak_increment_bytes": py_row["peak_increment_bytes"],
                "r_peak_increment_bytes": r_row["peak_increment_bytes"],
                "py_measurement_scope": py_row["measurement_scope"],
                "r_measurement_scope": r_row["measurement_scope"],
                "interpretation": (
                    "single-run descriptive timing with different process scopes; "
                    "not an equal-scope performance estimate"
                ),
            }
        )
    return pd.DataFrame(rows)


def _render(raw: pd.DataFrame, summary: pd.DataFrame, png: Path, pdf: Path) -> None:
    with plt.rc_context({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25}):
        figure, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
        for axis, lane, scenario in zip(
            axes.flat,
            ("legacy", "current", "legacy", "current"),
            (
                "publication_main",
                "publication_main",
                "ties_predeclared",
                "ties_predeclared",
            ),
        ):
            group = raw.loc[(raw["lane"] == lane) & (raw["scenario"] == scenario)]
            mean = (group["py_nes"] + group["r_nes"]) / 2.0
            difference = group["py_nes"] - group["r_nes"]
            row = summary.loc[
                (summary["lane"] == lane)
                & (summary["scenario"] == scenario)
                & (summary["metric"] == "NES")
            ].iloc[0]
            axis.scatter(mean, difference, s=20, alpha=0.7, edgecolors="none")
            axis.axhline(row["mean_bias_py_minus_r"], color="#b91c1c", linewidth=1.1)
            axis.axhline(row["lower_95_limit"], color="#64748b", linestyle="--")
            axis.axhline(row["upper_95_limit"], color="#64748b", linestyle="--")
            axis.set_xlabel("Mean NES (PyFgsea and R fgsea)")
            axis.set_ylabel("NES difference (PyFgsea - R fgsea)")
            lane_label = (
                "0.1.4 / fgsea 1.32.2"
                if lane == "legacy"
                else "0.2.0 / fgsea 1.38.0"
            )
            scenario_label = (
                "publication input"
                if scenario == "publication_main"
                else "predeclared ties"
            )
            axis.set_title(f"{lane_label}\n{scenario_label}")
        figure.suptitle("Figure 1 NES difference distributions")
        figure.savefig(png, dpi=300, metadata={"Software": "PyFgsea"})
        figure.savefig(pdf, metadata={"Title": "PyFgsea Figure 1 NES differences"})
        plt.close(figure)


def _record(path: Path, root: Path, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "sha256_role": "provenance_only",
    }
    if rows is not None:
        record["rows"] = rows
    return record


def assemble(result_dir: Path, output_dir: Path) -> Path:
    paths = _input_paths(result_dir)
    output = output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if output == paths["root"] or paths["root"] in output.parents:
        raise SupplementError("output directory must not modify the Figure 1 result")
    output.parent.mkdir(parents=True, exist_ok=True)
    comparison = _load_json(paths["comparison"])
    _validate_comparison(comparison)
    raw = _read_raw(paths["raw"])
    timing = _read_timing(paths["timing"])
    bland_altman = _bland_altman_rows(raw)
    overlap = _overlap_rows(raw)
    tail, tail_summary = _tail_rows(raw)
    runtime = _runtime_rows(timing)

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        tables = {
            "figure1_bland_altman.tsv": bland_altman,
            "figure1_pathway_overlap.tsv": overlap,
            "figure1_extreme_tail_cases.tsv": tail,
            "figure1_extreme_tail_summary.tsv": tail_summary,
            "figure1_runtime_descriptive.tsv": runtime,
        }
        for name, frame in tables.items():
            frame.to_csv(staging / name, sep="\t", index=False, float_format="%.17g")
        _render(
            raw,
            bland_altman,
            staging / "figure1_bland_altman.png",
            staging / "figure1_bland_altman.pdf",
        )
        artifacts = {
            name: _record(staging / name, staging, len(frame))
            for name, frame in tables.items()
        }
        for name in ("figure1_bland_altman.png", "figure1_bland_altman.pdf"):
            artifacts[name] = _record(staging / name, staging)
        manifest = {
            "schema_version": 1,
            "kind": "figure1_supplement",
            "status": "complete",
            "source": {
                "result_dir": str(paths["root"]),
                "lanes": EXPECTED_LANES,
                "comparison": _record(paths["comparison"], paths["root"]),
                "raw": _record(paths["raw"], paths["root"], len(raw)),
                "timing": _record(paths["timing"], paths["root"], len(timing)),
            },
            "validation": {
                "functional_comparison_complete": True,
                "raw_values_recomputed": True,
                "timing_scopes_checked": True,
                "hashes_are_pass_fail_checks": False,
            },
            "methods": {
                "difference": "PyFgsea minus R fgsea",
                "bland_altman_limits": "mean difference plus or minus 1.96 sample SD",
                "equivalence_margin": None,
                "top_pathways": "10 largest absolute NES per engine",
                "fdr_threshold": 0.05,
                "tail_selection": (
                    "10 largest max(Py -log10P, R -log10P) per lane/scenario"
                ),
            },
            "limitations": [
                "Bland-Altman limits are descriptive and are not equivalence bounds.",
                "The ties scenario is a sensitivity analysis, not publication input.",
                "Extreme-tail rows are single-run comparisons, not multi-seed intervals.",
                "Runtime rows have different process scopes and do not estimate equal-scope speedup.",
            ],
            "artifacts": artifacts,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output / "manifest.json"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure1-result-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        manifest = assemble(args.figure1_result_dir, args.output_dir)
    except (SupplementError, FileExistsError) as error:
        print(f"Figure 1 supplement failed: {error}", file=sys.stderr)
        return 1
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
