"""Run the fixed GSE155254 Figure 2 Panel-D table with installed PyFgsea."""

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import importlib.metadata
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMETER_PATH = REPO_ROOT / "repro" / "figure2_gse155254" / "figure2_parameters.json"
DEFAULT_EXPECTED_VERSION = "0.2.0rc8"
EXPECTED_ALGORITHM_REVISION = "fgsea-1.38-pr178-v1"
EXPECTED_DATASET_SHAPE = (3576, 3000)
EXPECTED_N_WINDOWS = 62
EXPECTED_N_PATHWAYS = 43
EXPECTED_N_ROWS = EXPECTED_N_WINDOWS * EXPECTED_N_PATHWAYS

RUN_PARAMETERS: dict[str, Any] = {
    "pseudotime_key": "dpt_pseudotime",
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
}
PARAMETER_CONTRACT = {
    **RUN_PARAMETERS,
    "pathway_size_policy": "exact",
}


class Figure2RunError(RuntimeError):
    """Raised when the fixed Panel-D table cannot be completed."""


def _load_installed_runtime(expected_version: str) -> tuple[Any, dict[str, str]]:
    loaded = sys.modules.get("pyfgsea")
    if loaded is not None:
        loaded_path = Path(getattr(loaded, "__file__", "")).resolve()
        if loaded_path.is_relative_to(REPO_ROOT):
            raise Figure2RunError(
                "PyFgsea was imported from the source checkout; run this script in a "
                "clean process with the release wheel installed"
            )

    original_path = list(sys.path)
    sys.path[:] = [
        entry for entry in sys.path if Path(entry or Path.cwd()).resolve() != REPO_ROOT
    ]
    try:
        package = importlib.import_module("pyfgsea")
        wrapper = importlib.import_module("pyfgsea.wrapper")
        distribution_version = importlib.metadata.version("pyfgsea")
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise Figure2RunError(
            "an installed PyFgsea distribution is required"
        ) from error
    finally:
        sys.path[:] = original_path

    package_version = str(package.__version__)
    if package_version != expected_version or distribution_version != expected_version:
        raise Figure2RunError(
            f"installed PyFgsea version is {distribution_version}, expected "
            f"{expected_version}"
        )
    core_path = Path(wrapper._ext.__file__).resolve()
    if not core_path.is_file() or not any(
        str(core_path).endswith(suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    ):
        raise Figure2RunError(
            "PyFgsea native core is not an installed extension module"
        )
    revision = str(wrapper._algorithm_revision())
    if revision != EXPECTED_ALGORITHM_REVISION:
        raise Figure2RunError(
            f"native algorithm revision is {revision}, expected "
            f"{EXPECTED_ALGORITHM_REVISION}"
        )
    return package, {
        "distribution_version": distribution_version,
        "package_version": package_version,
        "algorithm_revision": revision,
        "package_file": str(Path(package.__file__).resolve()),
        "native_core_file": str(core_path),
    }


def _validate_parameter_record() -> None:
    try:
        recorded = json.loads(PARAMETER_PATH.read_text(encoding="utf-8"))["parameters"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise Figure2RunError("Figure 2 parameter record could not be read") from error
    if recorded != PARAMETER_CONTRACT:
        raise Figure2RunError("Figure 2 parameter record differs from the fixed runner")


def _validate_dataset(adata: ad.AnnData) -> dict[str, Any]:
    if tuple(adata.shape) != EXPECTED_DATASET_SHAPE:
        raise Figure2RunError(
            f"dataset shape is {tuple(adata.shape)}, expected {EXPECTED_DATASET_SHAPE}"
        )
    required_obs = {"dpt_pseudotime", "sample_id", "condition"}
    missing = sorted(required_obs.difference(adata.obs.columns))
    if missing:
        raise Figure2RunError(f"dataset is missing obs columns: {', '.join(missing)}")
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise Figure2RunError("dataset cell and gene identifiers must be unique")
    pseudotime = pd.to_numeric(adata.obs["dpt_pseudotime"], errors="coerce").to_numpy()
    if not np.isfinite(pseudotime).all() or np.any(
        (pseudotime < 0.0) | (pseudotime > 1.0)
    ):
        raise Figure2RunError("dpt_pseudotime must be finite and in [0, 1]")
    named_layers = sorted(str(name) for name in adata.layers.keys() if name is not None)
    if adata.raw is not None or named_layers:
        raise Figure2RunError(
            "the frozen dataset must have no adata.raw or named layers"
        )
    condition_counts = {
        str(label): int(count)
        for label, count in adata.obs["condition"].astype(str).value_counts().items()
    }
    return {
        "shape": list(adata.shape),
        "pseudotime_key": "dpt_pseudotime",
        "expression_matrix": "scaled adata.X",
        "raw_present": False,
        "named_layers": [],
        "condition_label_counts": condition_counts,
        "condition_label_role": "legacy inferred label; not used for grouping",
        "analysis_scope": "all cells pooled; descriptive trajectory table",
    }


def _validate_gene_sets(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise Figure2RunError(f"gene-set file was not found at {path}")
    records = [
        line.split("\t") for line in path.read_text(encoding="utf-8").splitlines()
    ]
    if not records or any(len(fields) < 3 or not fields[0] for fields in records):
        raise Figure2RunError("gene-set file contains an invalid record")
    return {"input_gene_sets": len(records)}


def _validate_results(results: pd.DataFrame) -> dict[str, Any]:
    required = {
        "Pathway",
        "ES",
        "NES",
        "P-value",
        "padj",
        "status",
        "window_id",
        "observed_pathway_size",
        "null_curve_size",
        "size_binned",
        "algorithm_revision",
    }
    missing = sorted(required.difference(results.columns))
    if missing:
        raise Figure2RunError(f"result table is missing columns: {', '.join(missing)}")
    if len(results) != EXPECTED_N_ROWS:
        raise Figure2RunError(
            f"result table has {len(results)} rows, expected {EXPECTED_N_ROWS}"
        )
    checked = results.copy()
    for column in (
        "ES",
        "NES",
        "P-value",
        "padj",
        "window_id",
        "observed_pathway_size",
        "null_curve_size",
    ):
        checked[column] = pd.to_numeric(checked[column], errors="coerce")
        if not np.isfinite(checked[column].to_numpy(dtype=float)).all():
            raise Figure2RunError(f"result column {column} contains non-finite values")
    windows = checked["window_id"].to_numpy(dtype=float)
    if not np.equal(windows, np.floor(windows)).all():
        raise Figure2RunError("window_id values must be integers")
    checked["window_id"] = windows.astype(int)
    if set(checked["window_id"]) != set(range(EXPECTED_N_WINDOWS)):
        raise Figure2RunError("result table does not contain the expected windows")
    if checked.duplicated(["window_id", "Pathway"]).any():
        raise Figure2RunError("result table contains duplicate window/pathway rows")
    if not (
        checked.groupby("window_id")["Pathway"].nunique() == EXPECTED_N_PATHWAYS
    ).all():
        raise Figure2RunError("result table is not a complete window/pathway grid")
    if (
        checked["Pathway"].nunique() != EXPECTED_N_PATHWAYS
        or not (
            checked.groupby("Pathway")["window_id"].nunique() == EXPECTED_N_WINDOWS
        ).all()
    ):
        raise Figure2RunError("result table does not repeat one pathway set per window")
    if not (checked["status"].astype(str) == "resolved").all():
        raise Figure2RunError("result table contains unresolved rows")
    if (
        not np.equal(checked["observed_pathway_size"], checked["null_curve_size"]).all()
        or checked["size_binned"].astype(bool).any()
    ):
        raise Figure2RunError("result table does not use exact pathway sizes")
    if set(checked["algorithm_revision"].astype(str)) != {EXPECTED_ALGORITHM_REVISION}:
        raise Figure2RunError("result table contains an unexpected algorithm revision")
    if not ((checked["P-value"] > 0.0) & (checked["P-value"] <= 1.0)).all():
        raise Figure2RunError("P-value must be in (0, 1]")
    if not ((checked["padj"] >= 0.0) & (checked["padj"] <= 1.0)).all():
        raise Figure2RunError("padj must be in [0, 1]")
    return {
        "complete_grid": True,
        "n_rows": len(checked),
        "n_windows": EXPECTED_N_WINDOWS,
        "n_pathways": EXPECTED_N_PATHWAYS,
        "resolved_rows": len(checked),
        "pathway_size_policy": "exact",
    }


def run_figure2(
    dataset_path: Path,
    gene_sets_path: Path,
    output_dir: Path,
    *,
    expected_version: str = DEFAULT_EXPECTED_VERSION,
) -> dict[str, Any]:
    """Run and write the fixed Panel-D table; no figure is rendered."""

    _validate_parameter_record()
    package, runtime = _load_installed_runtime(expected_version)
    dataset = dataset_path.expanduser().resolve()
    gene_sets = gene_sets_path.expanduser().resolve()
    if not dataset.is_file():
        raise Figure2RunError(f"dataset was not found at {dataset}")
    try:
        adata = ad.read_h5ad(dataset)
    except Exception as error:
        raise Figure2RunError("dataset could not be read as H5AD") from error
    dataset_summary = _validate_dataset(adata)
    gene_set_summary = _validate_gene_sets(gene_sets)

    results = package.run_trajectory_gsea(
        adata,
        str(gene_sets),
        **RUN_PARAMETERS,
    )
    result_summary = _validate_results(results)

    output = output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    result_path = output / "trajectory_results.csv"
    results.to_csv(result_path, index=False, float_format="%.17g")
    summary: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "pyfgsea-figure2-panel-d-table",
        "status": "complete",
        "runtime": runtime,
        "parameters": PARAMETER_CONTRACT,
        "inputs": {
            "dataset": str(dataset),
            "gene_sets": str(gene_sets),
        },
        "dataset": dataset_summary,
        "gene_sets": gene_set_summary,
        "results": {**result_summary, "path": str(result_path)},
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--gene-sets", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--expected-version",
        default=DEFAULT_EXPECTED_VERSION,
        help=f"Installed PyFgsea version (default: {DEFAULT_EXPECTED_VERSION}).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        run_figure2(
            args.dataset,
            args.gene_sets,
            args.output_dir,
            expected_version=args.expected_version,
        )
    except (Figure2RunError, FileExistsError) as error:
        print(f"Figure 2 table run failed: {error}", file=sys.stderr)
        return 1
    print(f"Figure 2 table written to {args.output_dir.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
