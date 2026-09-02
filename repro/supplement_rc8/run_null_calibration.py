"""Run multi-replicate null calibration with the installed PyFgsea wheel."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_INPUT_DIR = (
    SCRIPT_DIR.parent / "figure1_dual_lane" / "frozen_inputs" / "publication_main"
)
EXPECTED_DISTRIBUTION_VERSION = "0.2.0"
EXPECTED_MODULE_VERSION = "0.2.0"
EXPECTED_ALGORITHM_REVISION = "fgsea-1.38-pr178-v1"
EXPECTED_GENE_COUNT = 12000
EXPECTED_PATHWAY_COUNT = 100
DEFAULT_REPLICATES = 20
DEFAULT_BASE_SEED = 20260902
P_VALUE_THRESHOLDS = (0.01, 0.05, 0.10)
GSEA_PARAMETERS: dict[str, Any] = {
    "min_size": 15,
    "max_size": 500,
    "sample_size": 101,
    "nperm_nes": 1800,
    "gsea_param": 1.0,
    "eps": 1e-50,
    "score_type": "std",
    "use_batched": True,
    "bin_width": 0,
    "mode": "aligned",
    "tie_policy": "gene_id",
}


class NullCalibrationError(RuntimeError):
    """Raised when a null-calibration run is incomplete or invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path, root: Path, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.resolve().relative_to(root.resolve())),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "sha256_role": "provenance_only",
    }
    if rows is not None:
        record["rows"] = rows
    return record


def _parse_gmt(path: Path) -> dict[str, list[str]]:
    pathways: dict[str, list[str]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 3 or not fields[0].strip():
            raise NullCalibrationError(f"invalid GMT row at line {line_number}")
        name = fields[0]
        members = fields[2:]
        if name in pathways:
            raise NullCalibrationError(f"duplicate GMT pathway: {name}")
        if len(members) != len(set(members)) or any(not member for member in members):
            raise NullCalibrationError(f"invalid members in GMT pathway: {name}")
        pathways[name] = members
    return pathways


def _load_inputs(
    input_dir: Path,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Path]]:
    root = input_dir.expanduser().resolve()
    ranks_path = root / "ranks.csv"
    pathways_path = root / "pathways.gmt"
    if not ranks_path.is_file() or not pathways_path.is_file():
        raise NullCalibrationError("fixed null-calibration inputs are incomplete")
    ranks = pd.read_csv(ranks_path)
    if list(ranks.columns) != ["Gene", "Score"] or len(ranks) != EXPECTED_GENE_COUNT:
        raise NullCalibrationError("fixed ranks have the wrong columns or row count")
    if ranks["Gene"].isna().any() or ranks["Gene"].astype(str).duplicated().any():
        raise NullCalibrationError("fixed ranks require unique, non-missing genes")
    try:
        scores = pd.to_numeric(ranks["Score"], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise NullCalibrationError("fixed ranks contain non-numeric scores") from error
    if not np.isfinite(scores).all() or np.ptp(scores) <= 0.0:
        raise NullCalibrationError("fixed ranks require finite, non-constant scores")
    ranks = pd.DataFrame({"Gene": ranks["Gene"].astype(str), "Score": scores})
    pathways = _parse_gmt(pathways_path)
    if len(pathways) != EXPECTED_PATHWAY_COUNT:
        raise NullCalibrationError("fixed GMT has the wrong pathway count")
    genes = set(ranks["Gene"])
    for name, members in pathways.items():
        if not set(members).issubset(genes):
            raise NullCalibrationError(f"pathway contains genes outside ranks: {name}")
        if (
            not GSEA_PARAMETERS["min_size"]
            <= len(members)
            <= GSEA_PARAMETERS["max_size"]
        ):
            raise NullCalibrationError(
                f"pathway size is outside the run bounds: {name}"
            )
    return ranks, pathways, {"ranks": ranks_path, "pathways": pathways_path}


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


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


def _installed_package() -> tuple[Callable[..., pd.DataFrame], dict[str, Any]]:
    distribution_version = importlib.metadata.version("pyfgsea")
    if distribution_version != EXPECTED_DISTRIBUTION_VERSION:
        raise NullCalibrationError(
            f"installed PyFgsea version is {distribution_version}, expected "
            f"{EXPECTED_DISTRIBUTION_VERSION}"
        )
    module = importlib.import_module("pyfgsea")
    module_version = str(getattr(module, "__version__", ""))
    if module_version != EXPECTED_MODULE_VERSION:
        raise NullCalibrationError(
            f"imported PyFgsea version is {module_version}, expected {EXPECTED_MODULE_VERSION}"
        )
    module_path = Path(module.__file__).resolve()
    if _is_within(module_path, REPO_ROOT):
        raise NullCalibrationError(
            "PyFgsea was imported from source, not an installed wheel"
        )
    core = importlib.import_module("pyfgsea._core")
    core_path = Path(core.__file__).resolve()
    if not core_path.is_file():
        raise NullCalibrationError("installed PyFgsea native core was not found")
    wrapper = importlib.import_module("pyfgsea.wrapper")
    revision_function = getattr(wrapper, "_algorithm_revision", None)
    if not callable(revision_function):
        raise NullCalibrationError(
            "installed PyFgsea lacks algorithm revision metadata"
        )
    revision = str(revision_function())
    if revision != EXPECTED_ALGORITHM_REVISION:
        raise NullCalibrationError(
            f"installed algorithm revision is {revision}, expected "
            f"{EXPECTED_ALGORITHM_REVISION}"
        )
    run_gsea = getattr(module, "run_gsea", None)
    if not callable(run_gsea):
        raise NullCalibrationError("installed PyFgsea lacks run_gsea")
    return run_gsea, {
        "distribution_version": distribution_version,
        "module_version": module_version,
        "algorithm_revision": revision,
        "module_path": str(module_path),
        "native_core_path": str(core_path),
    }


def _validate_result(
    result: pd.DataFrame, expected_pathways: set[str], replicate: int
) -> pd.DataFrame:
    required = {"Pathway", "ES", "NES", "P-value", "padj", "status"}
    missing = sorted(required.difference(result.columns))
    if missing:
        raise NullCalibrationError(
            f"replicate {replicate} result is missing columns: {missing}"
        )
    if len(result) != len(expected_pathways) or result["Pathway"].duplicated().any():
        raise NullCalibrationError(f"replicate {replicate} has incomplete pathway rows")
    if set(result["Pathway"].astype(str)) != expected_pathways:
        raise NullCalibrationError(f"replicate {replicate} has the wrong pathways")
    checked = result.copy()
    for column in ("ES", "NES", "P-value", "padj"):
        try:
            checked[column] = pd.to_numeric(checked[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise NullCalibrationError(
                f"replicate {replicate} {column} is not numeric"
            ) from error
        if not np.isfinite(checked[column].to_numpy(dtype=float)).all():
            raise NullCalibrationError(
                f"replicate {replicate} {column} contains non-finite values"
            )
    if not ((checked["P-value"] > 0.0) & (checked["P-value"] <= 1.0)).all():
        raise NullCalibrationError(f"replicate {replicate} P-values are outside (0, 1]")
    if not ((checked["padj"] >= 0.0) & (checked["padj"] <= 1.0)).all():
        raise NullCalibrationError(
            f"replicate {replicate} adjusted P-values are outside [0, 1]"
        )
    if not set(checked["status"].astype(str)).issubset({"resolved", "eps_floor"}):
        raise NullCalibrationError(f"replicate {replicate} contains unresolved results")
    return checked


def _ks_uniform(values: np.ndarray) -> float:
    pvalues = np.sort(np.asarray(values, dtype=float))
    if pvalues.ndim != 1 or len(pvalues) == 0:
        raise NullCalibrationError("uniform-distance input must be a non-empty vector")
    if not np.isfinite(pvalues).all() or np.any((pvalues < 0.0) | (pvalues > 1.0)):
        raise NullCalibrationError("uniform-distance input must contain probabilities")
    n = len(pvalues)
    upper = np.arange(1, n + 1, dtype=float) / n
    lower = np.arange(0, n, dtype=float) / n
    return float(max(np.max(upper - pvalues), np.max(pvalues - lower)))


def _summary_row(label: int | str, values: np.ndarray) -> dict[str, Any]:
    return {
        "replicate": label,
        "n_pathways": len(values),
        "mean_pvalue": float(np.mean(values)),
        "median_pvalue": float(np.median(values)),
        "minimum_pvalue": float(np.min(values)),
        "maximum_pvalue": float(np.max(values)),
        "ks_distance_from_uniform": _ks_uniform(values),
        **{
            f"fraction_p_below_{threshold:.2f}": float(np.mean(values < threshold))
            for threshold in P_VALUE_THRESHOLDS
        },
    }


def _execute_calibration(
    ranks: pd.DataFrame,
    pathways: Mapping[str, Sequence[str]],
    *,
    replicates: int,
    base_seed: int,
    run_gsea: Callable[..., pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates < 2
    ):
        raise ValueError("replicates must be an integer of at least 2")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer")
    genes = ranks["Gene"].astype(str).to_numpy()
    scores = ranks["Score"].to_numpy(dtype=float)
    expected_pathways = set(pathways)
    raw_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for replicate in range(1, replicates + 1):
        seed = base_seed + replicate - 1
        rng = np.random.default_rng(seed)
        permuted = pd.DataFrame(
            {"Gene": genes, "Score": scores[rng.permutation(len(scores))]}
        )
        result = run_gsea(
            permuted,
            dict(pathways),
            gene_col="Gene",
            score_col="Score",
            seed=seed,
            **GSEA_PARAMETERS,
        )
        checked = _validate_result(result, expected_pathways, replicate)
        pvalues = checked["P-value"].to_numpy(dtype=float)
        summary_rows.append(_summary_row(replicate, pvalues))
        selected = checked[["Pathway", "ES", "NES", "P-value", "padj", "status"]].copy()
        selected.insert(0, "permutation_seed", seed)
        selected.insert(0, "replicate", replicate)
        selected = selected.rename(
            columns={
                "Pathway": "pathway",
                "ES": "es",
                "NES": "nes",
                "P-value": "pvalue",
            }
        )
        raw_frames.append(selected.sort_values("pathway", kind="mergesort"))
    raw = pd.concat(raw_frames, ignore_index=True)
    pooled = raw["pvalue"].to_numpy(dtype=float)
    summary_rows.append(_summary_row("pooled_descriptive", pooled))
    return raw, pd.DataFrame(summary_rows)


def _render(raw: pd.DataFrame, output: Path) -> None:
    with plt.rc_context({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25}):
        figure, (qq_axis, ecdf_axis) = plt.subplots(
            1, 2, figsize=(10, 4.5), constrained_layout=True
        )
        maximum = 0.0
        for _, group in raw.groupby("replicate", sort=True):
            observed = np.sort(group["pvalue"].to_numpy(dtype=float))
            expected = np.arange(1, len(observed) + 1) / (len(observed) + 1)
            expected_log = -np.log10(expected)
            observed_log = -np.log10(observed)
            maximum = max(maximum, float(expected_log.max()), float(observed_log.max()))
            qq_axis.plot(expected_log, observed_log, color="#64748b", alpha=0.3)
        qq_axis.plot([0.0, maximum], [0.0, maximum], color="#b91c1c", linestyle="--")
        qq_axis.set_xlabel("Expected -log10(P)")
        qq_axis.set_ylabel("Observed -log10(P)")
        qq_axis.set_title("Per-replicate null QQ curves")

        pooled = np.sort(raw["pvalue"].to_numpy(dtype=float))
        ecdf = np.arange(1, len(pooled) + 1) / len(pooled)
        ecdf_axis.plot(pooled, ecdf, color="#1d4ed8", label="Pooled empirical")
        ecdf_axis.plot(
            [0.0, 1.0], [0.0, 1.0], color="#b91c1c", linestyle="--", label="Uniform"
        )
        ecdf_axis.set_xlabel("P-value")
        ecdf_axis.set_ylabel("Empirical cumulative fraction")
        ecdf_axis.set_title("Pooled descriptive ECDF")
        ecdf_axis.legend()
        figure.suptitle("PyFgsea null calibration by score permutation")
        figure.savefig(output, dpi=300, metadata={"Software": "PyFgsea"})
        plt.close(figure)


def run_calibration(
    input_dir: Path,
    output_dir: Path,
    *,
    replicates: int = DEFAULT_REPLICATES,
    base_seed: int = DEFAULT_BASE_SEED,
) -> Path:
    output = output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if _is_within(output, REPO_ROOT):
        raise NullCalibrationError("choose an output directory outside the repository")
    output.parent.mkdir(parents=True, exist_ok=True)
    threads = _thread_environment()
    run_gsea, package = _installed_package()
    ranks, pathways, inputs = _load_inputs(input_dir)
    started_at = datetime.now(timezone.utc).isoformat()
    raw, summary = _execute_calibration(
        ranks,
        pathways,
        replicates=replicates,
        base_seed=base_seed,
        run_gsea=run_gsea,
    )

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        raw_path = staging / "null_calibration_pathway_results.tsv"
        summary_path = staging / "null_calibration_summary.tsv"
        figure_path = staging / "null_calibration_qq_ecdf.png"
        raw.to_csv(raw_path, sep="\t", index=False, float_format="%.17g")
        summary.to_csv(summary_path, sep="\t", index=False, float_format="%.17g")
        _render(raw, figure_path)
        artifacts = {
            raw_path.name: _record(raw_path, staging, len(raw)),
            summary_path.name: _record(summary_path, staging, len(summary)),
            figure_path.name: _record(figure_path, staging),
        }
        source_root = input_dir.expanduser().resolve()
        manifest = {
            "schema_version": 1,
            "kind": "pyfgsea_null_calibration",
            "status": "complete",
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "package": package,
            "environment": {
                "python": platform.python_version(),
                "system": platform.system(),
                "machine": platform.machine(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "threads": threads,
            },
            "design": {
                "input": "fixed Figure 1 publication benchmark",
                "null_operation": (
                    "independent permutation of the fixed scores across gene labels "
                    "for each replicate"
                ),
                "replicates": replicates,
                "base_seed": base_seed,
                "pathways_per_replicate": len(pathways),
                "gsea_parameters": GSEA_PARAMETERS,
                "reported_thresholds": list(P_VALUE_THRESHOLDS),
                "equivalence_margin": None,
                "acceptance_threshold": None,
            },
            "validation": {
                "installed_wheel_executed": True,
                "native_core_executed": True,
                "all_replicates_complete": True,
                "all_pathway_rows_complete": len(raw) == replicates * len(pathways),
                "all_pvalues_finite_and_in_range": True,
                "hashes_are_pass_fail_checks": False,
            },
            "interpretation": (
                "Descriptive null-calibration output; no equivalence or acceptance "
                "decision is encoded. Pathway overlap means pooled values are not "
                "treated as independent for an inferential KS test."
            ),
            "source": {
                label: _record(path, source_root) for label, path in inputs.items()
            },
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
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        manifest = run_calibration(
            args.input_dir,
            args.output_dir,
            replicates=args.replicates,
            base_seed=args.base_seed,
        )
    except (NullCalibrationError, FileExistsError, ValueError) as error:
        print(f"Null calibration failed: {error}", file=sys.stderr)
        return 1
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
