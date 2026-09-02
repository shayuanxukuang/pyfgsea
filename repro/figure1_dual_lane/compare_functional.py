#!/usr/bin/env python3
"""Compare functional legacy/current Figure 1 lane results."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from .common import (
        GSEA_PARAMETERS,
        LANE_CONTRACTS,
        SCENARIOS,
        ensure_empty_output_dir,
    )
    from .compare_results import (
        RAW_REQUIRED_COLUMNS,
        _cross_lane_rows,
        _draw_figure,
        _extreme_rows,
        _metric_rows,
        _overlap_rows,
    )
except ImportError:  # pragma: no cover - direct script execution
    from common import (  # type: ignore
        GSEA_PARAMETERS,
        LANE_CONTRACTS,
        SCENARIOS,
        ensure_empty_output_dir,
    )
    from compare_results import (  # type: ignore
        RAW_REQUIRED_COLUMNS,
        _cross_lane_rows,
        _draw_figure,
        _extreme_rows,
        _metric_rows,
        _overlap_rows,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
EXPECTED_COUNTS = {"publication_main": 100, "ties_predeclared": 60}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path, root: Path, *, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.resolve().relative_to(root.resolve())),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "sha256_role": "provenance_only",
    }
    if rows is not None:
        record["rows"] = rows
    return record


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError(f"lane result was not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"lane result root is not an object: {path}")
    return value


def _finite(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{label} {column} contains non-finite values")


def _read_lane(
    result_path: Path, lane: str
) -> tuple[Mapping[str, Any], pd.DataFrame, pd.DataFrame]:
    result_path = result_path.expanduser().resolve()
    result = _read_json(result_path)
    if (
        result.get("schema_version") != 1
        or result.get("kind") != "figure1_functional_lane"
        or result.get("lane") != lane
        or result.get("status") != "complete"
    ):
        raise ValueError(f"{lane} lane result has the wrong identity or status")
    package = result.get("package")
    reference = result.get("reference")
    environment = result.get("environment")
    if not all(isinstance(item, Mapping) for item in (package, reference, environment)):
        raise ValueError(
            f"{lane} lane result lacks package/reference environment details"
        )
    contract = LANE_CONTRACTS[lane]
    expected_package = {
        "distribution_version": contract["pyfgsea_distribution_version"],
        "module_version": contract["pyfgsea_module_version"],
        "algorithm_revision": contract["algorithm_revision"],
    }
    if any(package.get(key) != value for key, value in expected_package.items()):
        raise ValueError(f"{lane} installed PyFgsea identity differs from its lane")
    expected_reference = {
        "r_version": contract["r_version"],
        "bioconductor_version": contract["bioconductor_version"],
        "fgsea_version": contract["fgsea_version"],
    }
    if any(reference.get(key) != value for key, value in expected_reference.items()):
        raise ValueError(f"{lane} R/fgsea identity differs from its lane")
    if result.get("gsea_parameters") != GSEA_PARAMETERS:
        raise ValueError(f"{lane} GSEA parameters differ from the comparison contract")

    raw_path = result_path.parent / "pathway_level_raw.tsv"
    timing_path = result_path.parent / "runtime_memory.tsv"
    if not raw_path.is_file() or not timing_path.is_file():
        raise ValueError(f"{lane} lane is missing raw or timing output")
    raw = pd.read_csv(raw_path, sep="\t")
    missing = sorted(RAW_REQUIRED_COLUMNS.difference(raw.columns))
    if missing:
        raise ValueError(f"{lane} raw table is missing columns: {missing}")
    counts = raw.groupby("scenario").size().to_dict()
    if set(raw["lane"]) != {lane} or counts != EXPECTED_COUNTS:
        raise ValueError(f"{lane} raw table has the wrong lane/scenario rows")
    if raw.duplicated(["scenario", "pathway"]).any():
        raise ValueError(f"{lane} raw table contains duplicate pathways")
    numeric = sorted(RAW_REQUIRED_COLUMNS.difference({"lane", "scenario", "pathway"}))
    _finite(raw, numeric, f"{lane} raw")
    for column in ("py_pval", "r_pval", "py_padj", "r_padj"):
        if not ((raw[column] >= 0.0) & (raw[column] <= 1.0)).all():
            raise ValueError(f"{lane} {column} is outside [0, 1]")
    recomputed = {
        "es_difference": raw["py_es"] - raw["r_es"],
        "nes_difference": raw["py_nes"] - raw["r_nes"],
        "neg_log10_pval_difference": (
            raw["py_neg_log10_pval"] - raw["r_neg_log10_pval"]
        ),
    }
    for column, values in recomputed.items():
        if not np.allclose(raw[column], values, rtol=1e-13, atol=1e-14):
            raise ValueError(f"{lane} raw {column} is inconsistent")

    timing = pd.read_csv(timing_path, sep="\t")
    required_timing = {
        "lane",
        "scenario",
        "engine",
        "measurement_scope",
        "elapsed_seconds",
        "peak_rss_bytes",
        "peak_increment_bytes",
    }
    if not required_timing.issubset(timing.columns):
        raise ValueError(f"{lane} timing table lacks required columns")
    expected_timing = {
        (scenario, engine)
        for scenario in SCENARIOS
        for engine in ("pyfgsea", "r_fgsea")
    }
    if set(zip(timing["scenario"], timing["engine"])) != expected_timing:
        raise ValueError(f"{lane} timing table has the wrong result lattice")
    if set(timing["lane"]) != {lane} or timing.duplicated(["scenario", "engine"]).any():
        raise ValueError(f"{lane} timing table has duplicate or wrong-lane rows")
    _finite(
        timing,
        ("elapsed_seconds", "peak_rss_bytes", "peak_increment_bytes"),
        f"{lane} timing",
    )
    if (timing["elapsed_seconds"] <= 0.0).any():
        raise ValueError(f"{lane} timing contains non-positive elapsed time")
    return result, raw, timing


def compare(legacy_result: Path, current_result: Path, output_dir: Path) -> Path:
    output = output_dir.expanduser().resolve()
    try:
        output.relative_to(REPO_ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("choose an output directory outside the repository")
    output = ensure_empty_output_dir(output)

    legacy, legacy_raw, legacy_timing = _read_lane(legacy_result, "legacy")
    current, current_raw, current_timing = _read_lane(current_result, "current")
    for key in (
        "python",
        "system",
        "machine",
        "implementation",
        "numpy",
        "pandas",
        "threads",
    ):
        if legacy["environment"].get(key) != current["environment"].get(key):
            raise ValueError(f"legacy/current execution environments differ for {key}")
    if legacy.get("input_invariants") != current.get("input_invariants"):
        raise ValueError("legacy/current input invariants differ")

    raw = pd.concat([legacy_raw, current_raw], ignore_index=True).sort_values(
        ["scenario", "lane", "pathway"], kind="mergesort"
    )
    raw_path = output / "figure1_pathway_level_raw.tsv"
    raw.to_csv(raw_path, sep="\t", index=False, float_format="%.17g")
    raw_source = _sha256(raw_path)
    raw = pd.read_csv(raw_path, sep="\t")

    metrics = _metric_rows(raw, raw_source)
    overlap = _overlap_rows(raw, raw_source)
    extreme = _extreme_rows(raw, raw_source, 10)
    cross_lane = _cross_lane_rows(raw, raw_source)
    timing = pd.concat([legacy_timing, current_timing], ignore_index=True).sort_values(
        ["scenario", "lane", "engine"], kind="mergesort"
    )
    tables = {
        "figure1_agreement_metrics.tsv": metrics,
        "figure1_pathway_overlap.tsv": overlap,
        "figure1_extreme_tail_cases.tsv": extreme,
        "figure1_legacy_current_change.tsv": cross_lane,
        "figure1_runtime_memory.tsv": timing,
    }
    for name, frame in tables.items():
        frame.to_csv(output / name, sep="\t", index=False, float_format="%.17g")
    figure_path = output / "figure1_dual_lane_agreement.png"
    _draw_figure(raw, metrics, figure_path)

    artifacts = {
        "figure1_pathway_level_raw.tsv": _record(raw_path, output, rows=len(raw))
    }
    for name, frame in tables.items():
        artifacts[name] = _record(output / name, output, rows=len(frame))
    artifacts[figure_path.name] = _record(figure_path, output)
    result = {
        "schema_version": 1,
        "kind": "figure1_functional_comparison",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "lanes": {
            "legacy": {
                "pyfgsea": LANE_CONTRACTS["legacy"]["pyfgsea_distribution_version"],
                "fgsea": LANE_CONTRACTS["legacy"]["fgsea_version"],
            },
            "current": {
                "pyfgsea": LANE_CONTRACTS["current"]["pyfgsea_distribution_version"],
                "fgsea": LANE_CONTRACTS["current"]["fgsea_version"],
            },
        },
        "gsea_parameters": dict(GSEA_PARAMETERS),
        "validation": {
            "installed_wheels_executed": True,
            "reference_versions_executed": True,
            "raw_rows_complete": True,
            "metrics_recomputed_from_raw_rows": True,
            "manual_metric_overrides": False,
            "hashes_are_pass_fail_checks": False,
        },
        "scope": {
            "publication_main": "frozen publication comparison input",
            "ties_predeclared": "same-environment tie sensitivity input",
            "runtime": "single-run descriptive timing with engine-specific scopes",
        },
        "artifacts": artifacts,
    }
    result_path = output / "comparison_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-result", required=True, type=Path)
    parser.add_argument("--current-result", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = compare(args.legacy_result, args.current_result, args.output_dir)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
