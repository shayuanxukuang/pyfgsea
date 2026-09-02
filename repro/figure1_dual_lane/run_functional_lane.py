#!/usr/bin/env python3
"""Run one Figure 1 lane using installed packages and numerical checks."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from .common import (
        EPS,
        GSEA_PARAMETERS,
        LANE_CONTRACTS,
        LOG10_FLOOR,
        SCENARIOS,
        ensure_empty_output_dir,
        write_json,
    )
    from .run_lane import (
        EXPECTED_SCENARIO_INVARIANTS,
        R_HELPER,
        _is_within,
        _load_rank_and_gmt,
        _measure_call,
        _measure_process,
        _read_r_environment,
        _run_python_engine,
        _validate_engine_table,
        _verify_scenario_invariants,
    )
except ImportError:  # pragma: no cover - direct script execution
    from common import (  # type: ignore
        EPS,
        GSEA_PARAMETERS,
        LANE_CONTRACTS,
        LOG10_FLOOR,
        SCENARIOS,
        ensure_empty_output_dir,
        write_json,
    )
    from run_lane import (  # type: ignore
        EXPECTED_SCENARIO_INVARIANTS,
        R_HELPER,
        _is_within,
        _load_rank_and_gmt,
        _measure_call,
        _measure_process,
        _read_r_environment,
        _run_python_engine,
        _validate_engine_table,
        _verify_scenario_invariants,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
FROZEN_INPUT_ROOT = SCRIPT_DIR / "frozen_inputs"
EXPECTED_INPUT_PARAMETERS = {
    "publication_main": {"n_genes": 12000},
    "ties_predeclared": {"n_genes": 4000},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance_record(
    path: Path, *, relative_to: Path | None = None
) -> dict[str, Any]:
    resolved = path.resolve()
    display = (
        resolved if relative_to is None else resolved.relative_to(relative_to.resolve())
    )
    return {
        "path": str(display),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
        "sha256_role": "provenance_only",
    }


def _installed_package(lane: str) -> tuple[Any, dict[str, Any]]:
    contract = LANE_CONTRACTS[lane]
    distribution_version = importlib.metadata.version("pyfgsea")
    if distribution_version != contract["pyfgsea_distribution_version"]:
        raise RuntimeError(
            f"installed PyFgsea version is {distribution_version}, expected "
            f"{contract['pyfgsea_distribution_version']}"
        )
    pyfgsea = importlib.import_module("pyfgsea")
    module_version = str(getattr(pyfgsea, "__version__", ""))
    if module_version != contract["pyfgsea_module_version"]:
        raise RuntimeError(
            f"imported PyFgsea version is {module_version}, expected "
            f"{contract['pyfgsea_module_version']}"
        )
    module_path = Path(pyfgsea.__file__).resolve()
    if _is_within(module_path, REPO_ROOT):
        raise RuntimeError(
            "PyFgsea was imported from the source checkout, not the wheel"
        )
    core = importlib.import_module("pyfgsea._core")
    core_path = Path(core.__file__).resolve()
    if not core_path.is_file():
        raise RuntimeError("installed PyFgsea native core was not found")

    revision: str | None = None
    if lane == "current":
        wrapper = importlib.import_module("pyfgsea.wrapper")
        revision_function = getattr(wrapper, "_algorithm_revision", None)
        if not callable(revision_function):
            raise RuntimeError(
                "installed current PyFgsea lacks algorithm revision metadata"
            )
        revision = str(revision_function())
        if revision != contract["algorithm_revision"]:
            raise RuntimeError(
                f"installed algorithm revision is {revision}, expected "
                f"{contract['algorithm_revision']}"
            )
    return pyfgsea, {
        "distribution_version": distribution_version,
        "module_version": module_version,
        "algorithm_revision": revision,
        "module_path": str(module_path),
        "native_core_path": str(core_path),
    }


def _input_manifest() -> tuple[dict[str, Any], dict[str, dict[str, Path]]]:
    manifest: dict[str, Any] = {"scenarios": {}}
    paths: dict[str, dict[str, Path]] = {}
    for scenario in SCENARIOS:
        scenario_root = (FROZEN_INPUT_ROOT / scenario).resolve()
        ranks = scenario_root / "ranks.csv"
        pathways = scenario_root / "pathways.gmt"
        if not ranks.is_file() or not pathways.is_file():
            raise RuntimeError(f"frozen Figure 1 inputs are incomplete for {scenario}")
        paths[scenario] = {"ranks": ranks, "pathways": pathways}
        manifest["scenarios"][scenario] = {
            "invariants": EXPECTED_SCENARIO_INVARIANTS[scenario],
            "parameters": EXPECTED_INPUT_PARAMETERS[scenario],
        }
    return manifest, paths


def _thread_environment() -> dict[str, str]:
    variables = (
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    )
    for variable in variables:
        os.environ[variable] = "1"
    return {variable: os.environ[variable] for variable in variables}


def run_lane(lane: str, output_dir: Path) -> Path:
    contract = LANE_CONTRACTS[lane]
    if (
        os.environ.get("FGSEA_REFERENCE_VERSION", "").strip()
        != contract["fgsea_version"]
    ):
        raise RuntimeError(
            f"FGSEA_REFERENCE_VERSION must be {contract['fgsea_version']} for {lane}"
        )
    threads = _thread_environment()
    output = output_dir.expanduser().resolve()
    if _is_within(output, REPO_ROOT):
        raise ValueError("choose an output directory outside the repository")
    output = ensure_empty_output_dir(output)
    pyfgsea, package_identity = _installed_package(lane)
    manifest, input_paths = _input_manifest()

    rscript = shutil.which("Rscript") or ""
    if not rscript:
        raise RuntimeError("Rscript is required for the reference lane")
    if not R_HELPER.is_file():
        raise RuntimeError(f"R helper is missing: {R_HELPER}")

    raw_frames: list[pd.DataFrame] = []
    timing_rows: list[dict[str, Any]] = []
    r_environments: dict[str, dict[str, str]] = {}
    python_arguments: Mapping[str, Any] | None = None
    outputs: dict[str, Path] = {}
    started_at = datetime.now(timezone.utc).isoformat()

    for scenario in SCENARIOS:
        ranks_path = input_paths[scenario]["ranks"]
        pathways_path = input_paths[scenario]["pathways"]
        ranks, pathways = _load_rank_and_gmt(ranks_path, pathways_path)
        _verify_scenario_invariants(scenario, ranks, pathways, manifest)

        (py_result, call_arguments), py_timing = _measure_call(
            lambda: _run_python_engine(pyfgsea, lane, ranks, pathways)
        )
        if python_arguments is None:
            python_arguments = call_arguments
        elif python_arguments != call_arguments:
            raise RuntimeError("Python call arguments changed between scenarios")
        _validate_engine_table(py_result, prefix="py")
        py_path = output / f"pyfgsea_{scenario}.tsv"
        py_result.to_csv(py_path, sep="\t", index=False, float_format="%.17g")
        outputs[f"pyfgsea_{scenario}"] = py_path
        timing_rows.append(
            {
                "lane": lane,
                "scenario": scenario,
                "engine": "pyfgsea",
                "measurement_scope": "run_gsea_call_only",
                "engine_elapsed_seconds": py_timing["elapsed_seconds"],
                **py_timing,
            }
        )

        r_result_path = output / f"r_fgsea_{scenario}.tsv"
        r_environment_path = output / f"r_environment_{scenario}.tsv"
        r_session_path = output / f"r_sessionInfo_{scenario}.txt"
        command = [
            rscript,
            "--vanilla",
            str(R_HELPER),
            str(ranks_path),
            str(pathways_path),
            str(r_result_path),
            str(r_environment_path),
            str(r_session_path),
            str(contract["fgsea_version"]),
            str(contract["r_version"]),
            str(contract["bioconductor_version"]),
            str(GSEA_PARAMETERS["r_seed"]),
            str(GSEA_PARAMETERS["min_size"]),
            str(GSEA_PARAMETERS["max_size"]),
            str(GSEA_PARAMETERS["sample_size"]),
        ]
        environment = os.environ.copy()
        environment["PYFGSEA_FIGURE1_EPS"] = f"{EPS:.17g}"
        completed, r_timing = _measure_process(command, environment=environment)
        (output / f"r_stdout_{scenario}.txt").write_text(
            completed.stdout, encoding="utf-8"
        )
        (output / f"r_stderr_{scenario}.txt").write_text(
            completed.stderr, encoding="utf-8"
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"R reference failed for {scenario}: {completed.stderr[-1000:]}"
            )
        r_environment = _read_r_environment(r_environment_path, contract)
        r_environments[scenario] = r_environment
        timing_rows.append(
            {
                "lane": lane,
                "scenario": scenario,
                "engine": "r_fgsea",
                "measurement_scope": "Rscript_process_and_internal_fgsea",
                "engine_elapsed_seconds": float(r_environment["elapsed_seconds"]),
                **r_timing,
            }
        )
        for path in (
            r_result_path,
            r_environment_path,
            r_session_path,
            output / f"r_stdout_{scenario}.txt",
            output / f"r_stderr_{scenario}.txt",
        ):
            if not path.is_file():
                raise RuntimeError(f"R reference did not produce {path.name}")
            outputs[path.stem] = path

        r_result = pd.read_csv(r_result_path, sep="\t").rename(
            columns={
                "ES": "r_es",
                "NES": "r_nes",
                "pval": "r_pval",
                "padj": "r_padj",
                "size": "r_size",
            }
        )
        _validate_engine_table(r_result, prefix="r")
        if set(py_result["pathway"]) != set(r_result["pathway"]):
            raise RuntimeError(f"Python/R pathway universe differs for {scenario}")
        merged = py_result.merge(
            r_result, on="pathway", how="inner", validate="one_to_one"
        )
        if not (merged["py_size"].astype(int) == merged["r_size"].astype(int)).all():
            raise RuntimeError(f"Python/R pathway sizes differ for {scenario}")
        expected_count = EXPECTED_SCENARIO_INVARIANTS[scenario]["pathway_count"]
        if len(merged) != expected_count:
            raise RuntimeError(
                f"{scenario} returned {len(merged)} pathways, expected {expected_count}"
            )
        merged.insert(0, "lane", lane)
        merged.insert(1, "scenario", scenario)
        merged["es_difference"] = merged["py_es"] - merged["r_es"]
        merged["nes_difference"] = merged["py_nes"] - merged["r_nes"]
        merged["py_neg_log10_pval"] = -np.log10(
            np.maximum(merged["py_pval"], LOG10_FLOOR)
        )
        merged["r_neg_log10_pval"] = -np.log10(
            np.maximum(merged["r_pval"], LOG10_FLOOR)
        )
        merged["neg_log10_pval_difference"] = (
            merged["py_neg_log10_pval"] - merged["r_neg_log10_pval"]
        )
        invariants = EXPECTED_SCENARIO_INVARIANTS[scenario]
        merged["input_tied_score_group_count"] = invariants["tied_score_group_count"]
        merged["input_tied_gene_count"] = invariants["tied_gene_count"]
        merged["input_maximum_tie_multiplicity"] = invariants[
            "maximum_tie_multiplicity"
        ]
        raw_frames.append(merged.sort_values("pathway", kind="mergesort"))

    raw = pd.concat(raw_frames, ignore_index=True)
    raw_path = output / "pathway_level_raw.tsv"
    raw.to_csv(raw_path, sep="\t", index=False, float_format="%.17g")
    timing_path = output / "runtime_memory.tsv"
    timing = pd.DataFrame(timing_rows)
    timing.to_csv(timing_path, sep="\t", index=False, float_format="%.17g")
    outputs["pathway_level_raw"] = raw_path
    outputs["runtime_memory"] = timing_path

    result = {
        "schema_version": 1,
        "kind": "figure1_functional_lane",
        "lane": lane,
        "status": "complete",
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "package": package_identity,
        "reference": {
            "r_version": contract["r_version"],
            "bioconductor_version": contract["bioconductor_version"],
            "fgsea_version": contract["fgsea_version"],
            "scenario_environments": r_environments,
        },
        "gsea_parameters": dict(GSEA_PARAMETERS),
        "python_call_arguments": python_arguments,
        "input_invariants": EXPECTED_SCENARIO_INVARIANTS,
        "environment": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "threads": threads,
        },
        "outputs": {
            name: {"path": path.name, "rows": len(raw) if path == raw_path else None}
            for name, path in sorted(outputs.items())
        },
        "provenance": {
            "hashes_are_pass_fail_checks": False,
            "inputs": {
                scenario: {
                    label: _provenance_record(path)
                    for label, path in sorted(input_paths[scenario].items())
                }
                for scenario in SCENARIOS
            },
            "outputs": {
                name: _provenance_record(path, relative_to=output)
                for name, path in sorted(outputs.items())
            },
        },
    }
    result_path = output / "lane_result.json"
    write_json(result_path, result)
    return result_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", required=True, choices=LANE_CONTRACTS)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_lane(args.lane, args.output_dir)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
