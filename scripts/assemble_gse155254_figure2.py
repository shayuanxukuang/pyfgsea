"""Assemble GSE155254 Figure 2 from frozen inputs and a functional table run.

This script does not recompute UMAP coordinates or enrichment statistics. It
uses the frozen inputs and trajectory table in the supplied Figure 2 run, plus
the two public GEO control/gata307mut assignment tables.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve()
EXPECTED_ALGORITHM_REVISION = "fgsea-1.38-pr178-v1"
EXPECTED_DATASET_SHAPE = (3576, 3000)
EXPECTED_N_WINDOWS = 62
EXPECTED_N_PATHWAYS = 43
EXPECTED_N_ROWS = EXPECTED_N_WINDOWS * EXPECTED_N_PATHWAYS
BH_ATOL = 1e-14

EXPECTED_PARAMETERS: dict[str, Any] = {
    "window_size": 500,
    "step": 50,
    "nperm_nes": 2000,
    "score_type": "std",
    "pathway_size_policy": "exact",
    "bin_width": 0,
    "use_nes_cache": False,
}
TARGET_PATHWAYS = ("heme Metabolism", "E2F Targets")
MARKERS = ("CD34", "MKI67", "HBB")
MARKER_STATE_NAMES = {
    "CD34": "CD34-dominant",
    "MKI67": "MKI67-dominant",
    "HBB": "HBB-dominant",
}
STATE_COLORS = {
    "CD34-dominant": "#2563eb",
    "MKI67-dominant": "#ea580c",
    "HBB-dominant": "#b91c1c",
}
PATHWAY_COLORS = {
    "heme Metabolism": "#b91c1c",
    "E2F Targets": "#1d4ed8",
}
PUBLIC_ASSIGNMENT_LABELS = ("control", "gata307mut")
PUBLIC_ASSIGNMENT_SOURCES: dict[str, dict[str, str]] = {
    "GSM4698215_rep1": {
        "accession": "GSM4698215",
        "filename": "GSM4698215_g1mut_scRNA_rep1_donor_assignments.tsv.gz",
        "url": (
            "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4698nnn/"
            "GSM4698215/suppl/"
            "GSM4698215_g1mut_scRNA_rep1_donor_assignments.tsv.gz"
        ),
    },
    "GSM4698216_rep2": {
        "accession": "GSM4698216",
        "filename": "GSM4698216_g1mut_scRNA_rep2_donor_assignments.tsv.gz",
        "url": (
            "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4698nnn/"
            "GSM4698216/suppl/"
            "GSM4698216_g1mut_scRNA_rep2_donor_assignments.tsv.gz"
        ),
    },
}


class Figure2AssemblyError(RuntimeError):
    """Raised when the functional run cannot support Figure 2 assembly."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Figure2AssemblyError(f"{context} must be a JSON object")
    return value


def _load_json(path: Path, context: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise Figure2AssemblyError(f"{context} was not found at {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise Figure2AssemblyError(f"{context} is not valid JSON") from error
    return _require_mapping(value, context)


def _resolved_file(path: Path, parent: Path, context: str) -> Path:
    resolved = path.resolve()
    if not _is_within(resolved, parent) or not resolved.is_file():
        raise Figure2AssemblyError(
            f"{context} must be an existing file inside {parent.resolve()}"
        )
    return resolved


def _input_paths(
    figure2_run_dir: Path,
    dataset_path: Path,
    gene_sets_path: Path,
    assignment_dir: Path,
) -> dict[str, Path]:
    run_dir = figure2_run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise Figure2AssemblyError(f"Figure 2 run directory was not found at {run_dir}")
    assignments = assignment_dir.expanduser().resolve()
    if not assignments.is_dir():
        raise Figure2AssemblyError(
            f"Public assignment directory was not found at {assignments}"
        )
    paths = {
        "run_dir": run_dir,
        "assignment_dir": assignments,
        "run_summary": run_dir / "run_summary.json",
        "dataset": dataset_path.expanduser().resolve(),
        "gene_sets": gene_sets_path.expanduser().resolve(),
        "trajectory_results": run_dir / "trajectory_results.csv",
    }
    for name in ("run_summary", "trajectory_results"):
        paths[name] = _resolved_file(paths[name], run_dir, name.replace("_", " "))
    for name in ("dataset", "gene_sets"):
        if not paths[name].is_file():
            raise Figure2AssemblyError(
                f"{name.replace('_', ' ')} was not found at {paths[name]}"
            )
    for sample_id, source in PUBLIC_ASSIGNMENT_SOURCES.items():
        key = f"assignment_{sample_id}"
        paths[key] = _resolved_file(
            assignments / source["filename"],
            assignments,
            f"public assignment file for {sample_id}",
        )
    return paths


def _require_external_output(output_dir: Path, run_dir: Path) -> Path:
    output = output_dir.expanduser().resolve()
    if _is_within(output, REPO_ROOT):
        raise Figure2AssemblyError(
            "The output directory must be outside the assembly Git checkout"
        )
    if _is_within(output, run_dir):
        raise Figure2AssemblyError(
            "The output directory must not modify the functional Figure 2 run"
        )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _same_json_value(observed: Any, expected: Any) -> bool:
    return type(observed) is type(expected) and observed == expected


def _validate_run_summary(summary: Mapping[str, Any]) -> None:
    if summary.get("schema_version") != 1 or summary.get("status") != "complete":
        raise Figure2AssemblyError("run_summary does not report a complete table run")
    if summary.get("artifact_type") != "pyfgsea-figure2-panel-d-table":
        raise Figure2AssemblyError("run_summary has the wrong artifact type")
    dataset = _require_mapping(summary.get("dataset"), "run_summary dataset")
    if tuple(dataset.get("shape", ())) != EXPECTED_DATASET_SHAPE:
        raise Figure2AssemblyError(
            f"run_summary dataset shape is not {EXPECTED_DATASET_SHAPE}"
        )

    runtime = _require_mapping(summary.get("runtime"), "run_summary runtime")
    if (
        runtime.get("package_version") != runtime.get("distribution_version")
        or runtime.get("algorithm_revision") != EXPECTED_ALGORITHM_REVISION
    ):
        raise Figure2AssemblyError("run_summary has the wrong installed runtime")

    parameters = _require_mapping(summary.get("parameters"), "run_summary parameters")
    for name, expected in EXPECTED_PARAMETERS.items():
        if not _same_json_value(parameters.get(name), expected):
            raise Figure2AssemblyError(
                f"run_summary parameter {name!r} is not {expected!r}"
            )

    validation = _require_mapping(summary.get("results"), "run_summary results")
    expected_values = {
        "complete_grid": True,
        "n_windows": EXPECTED_N_WINDOWS,
        "n_pathways": EXPECTED_N_PATHWAYS,
        "n_rows": EXPECTED_N_ROWS,
        "resolved_rows": EXPECTED_N_ROWS,
        "pathway_size_policy": "exact",
    }
    if any(
        validation.get(key) != expected for key, expected in expected_values.items()
    ):
        raise Figure2AssemblyError(
            "run_summary does not report the complete resolved 62 x 43 grid"
        )


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise Figure2AssemblyError("BH requires a non-empty one-dimensional vector")
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    scale = values.size / np.arange(1, values.size + 1, dtype=float)
    adjusted_ranked = np.minimum.accumulate((ranked * scale)[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def _validate_results(results: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"Pathway", "NES", "P-value", "padj", "status", "window_id", "pt_mid"}
    missing = sorted(required.difference(results.columns))
    if missing:
        raise Figure2AssemblyError(
            f"trajectory_results.csv is missing columns: {', '.join(missing)}"
        )

    checked = results.copy()
    for column in ("NES", "P-value", "padj", "window_id", "pt_mid"):
        checked[column] = pd.to_numeric(checked[column], errors="coerce")
        if not np.isfinite(checked[column].to_numpy(dtype=float)).all():
            raise Figure2AssemblyError(
                f"trajectory_results.csv column {column!r} contains non-finite values"
            )
    if not ((checked["P-value"] > 0.0) & (checked["P-value"] <= 1.0)).all():
        raise Figure2AssemblyError("trajectory P-value must be in (0, 1]")
    if not ((checked["padj"] >= 0.0) & (checked["padj"] <= 1.0)).all():
        raise Figure2AssemblyError("trajectory padj must be in [0, 1]")

    window_values = checked["window_id"].to_numpy(dtype=float)
    if not np.equal(window_values, np.floor(window_values)).all():
        raise Figure2AssemblyError("window_id values must be integers")
    checked["window_id"] = window_values.astype(int)
    checked["Pathway"] = checked["Pathway"].astype(str)
    checked["status"] = checked["status"].astype(str)

    if len(checked) != EXPECTED_N_ROWS:
        raise Figure2AssemblyError(
            f"trajectory table has {len(checked)} rows, not {EXPECTED_N_ROWS}"
        )
    if set(checked["window_id"].unique()) != set(range(EXPECTED_N_WINDOWS)):
        raise Figure2AssemblyError(
            "trajectory table does not contain windows 0 through 61"
        )
    pathways = sorted(checked["Pathway"].unique())
    if len(pathways) != EXPECTED_N_PATHWAYS:
        raise Figure2AssemblyError(
            f"trajectory table has {len(pathways)} pathways, not {EXPECTED_N_PATHWAYS}"
        )
    duplicate_count = int(checked.duplicated(["window_id", "Pathway"]).sum())
    if duplicate_count:
        raise Figure2AssemblyError(
            "trajectory grid contains duplicate window/pathway rows"
        )
    per_window = checked.groupby("window_id")["Pathway"].nunique()
    per_pathway = checked.groupby("Pathway")["window_id"].nunique()
    if not (
        (per_window == EXPECTED_N_PATHWAYS).all()
        and (per_pathway == EXPECTED_N_WINDOWS).all()
    ):
        raise Figure2AssemblyError("trajectory table is not a complete 62 x 43 grid")
    midpoint_counts = checked.groupby("window_id")["pt_mid"].nunique(dropna=False)
    if not (midpoint_counts == 1).all():
        raise Figure2AssemblyError(
            "trajectory table has inconsistent pt_mid values within a window"
        )
    midpoints = (
        checked.groupby("window_id", sort=True)["pt_mid"].first().to_numpy(dtype=float)
    )
    if not (np.diff(midpoints) > 0.0).all():
        raise Figure2AssemblyError(
            "trajectory window midpoints must be strictly increasing"
        )
    if not (checked["status"] == "resolved").all():
        raise Figure2AssemblyError("trajectory table contains unresolved rows")

    max_bh_difference = 0.0
    for _, window in checked.groupby("window_id", sort=True):
        independent = _benjamini_hochberg(window["P-value"].to_numpy(dtype=float))
        observed = window["padj"].to_numpy(dtype=float)
        max_bh_difference = max(
            max_bh_difference, float(np.max(np.abs(independent - observed)))
        )
    if max_bh_difference > BH_ATOL:
        raise Figure2AssemblyError(
            "trajectory padj does not match independent within-window BH"
        )

    validation = {
        "complete_grid": True,
        "expected_grid": [EXPECTED_N_WINDOWS, EXPECTED_N_PATHWAYS],
        "n_windows": EXPECTED_N_WINDOWS,
        "n_pathways": EXPECTED_N_PATHWAYS,
        "n_rows": EXPECTED_N_ROWS,
        "resolved_rows": EXPECTED_N_ROWS,
        "duplicate_key_count": duplicate_count,
        "bh": {
            "scope": "within-window",
            "matches_core": True,
            "max_absolute_difference": max_bh_difference,
            "tolerance_absolute": BH_ATOL,
        },
    }
    return checked, validation


def _validate_gene_sets(path: Path) -> dict[str, Any]:
    names: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) < 3 or not fields[0]:
                raise Figure2AssemblyError(
                    f"gene-set file has an invalid record at line {line_number}"
                )
            names.append(fields[0])
    missing = [name for name in TARGET_PATHWAYS if name not in names]
    if missing:
        raise Figure2AssemblyError(
            f"gene-set file is missing target pathways: {', '.join(missing)}"
        )
    return {"n_gene_sets": len(names), "target_pathways_present": True}


def _gene_vector(adata: ad.AnnData, gene: str) -> np.ndarray:
    index = adata.var_names.get_loc(gene)
    values = adata.X[:, index]
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values, dtype=float).reshape(-1)


def _expression_matrix_scope(adata: ad.AnnData) -> dict[str, Any]:
    named_layers = sorted(str(name) for name in adata.layers.keys() if name is not None)
    if adata.raw is not None or named_layers:
        raise Figure2AssemblyError(
            "the frozen dataset is expected to have no adata.raw or named layers"
        )
    return {
        "matrix_used": "adata.X",
        "representation": "scaled_expression",
        "raw_present": False,
        "named_layers": [],
        "log1p_metadata_present": "log1p" in adata.uns,
        "interpretation": (
            "adata.X was overwritten by scaling during preparation; log1p metadata "
            "records an earlier preprocessing step and does not describe the matrix "
            "used for Figure 2"
        ),
    }


def _validate_adata(adata: ad.AnnData) -> dict[str, Any]:
    if tuple(adata.shape) != EXPECTED_DATASET_SHAPE:
        raise Figure2AssemblyError(
            f"frozen dataset shape is {tuple(adata.shape)}, not {EXPECTED_DATASET_SHAPE}"
        )
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise Figure2AssemblyError(
            "frozen dataset obs_names and var_names must be unique"
        )
    if "dpt_pseudotime" not in adata.obs:
        raise Figure2AssemblyError("frozen dataset is missing dpt_pseudotime")
    missing_obs = [name for name in ("sample_id", "condition") if name not in adata.obs]
    if missing_obs:
        raise Figure2AssemblyError(
            f"frozen dataset is missing obs columns: {', '.join(missing_obs)}"
        )
    if "X_umap" not in adata.obsm:
        raise Figure2AssemblyError("frozen dataset is missing precomputed X_umap")
    missing_markers = [gene for gene in MARKERS if gene not in adata.var_names]
    if missing_markers:
        raise Figure2AssemblyError(
            f"frozen dataset is missing markers: {', '.join(missing_markers)}"
        )

    cell_ids = adata.obs_names.astype(str).to_numpy()
    if not pd.Index(cell_ids).is_unique:
        raise Figure2AssemblyError(
            "cell identifiers are not unique after string conversion"
        )
    try:
        pseudotime = adata.obs["dpt_pseudotime"].to_numpy(dtype=float)
        umap = np.asarray(adata.obsm["X_umap"], dtype=float)
    except (TypeError, ValueError) as error:
        raise Figure2AssemblyError(
            "dpt_pseudotime and X_umap must be numeric"
        ) from error
    if pseudotime.shape != (EXPECTED_DATASET_SHAPE[0],):
        raise Figure2AssemblyError("dpt_pseudotime has the wrong shape")
    if umap.shape != (EXPECTED_DATASET_SHAPE[0], 2):
        raise Figure2AssemblyError("X_umap must have shape (3576, 2)")
    if not np.isfinite(pseudotime).all() or not np.isfinite(umap).all():
        raise Figure2AssemblyError("dpt_pseudotime and X_umap must be finite")
    if np.any((pseudotime < 0.0) | (pseudotime > 1.0)):
        raise Figure2AssemblyError("dpt_pseudotime must be in [0, 1]")

    marker_values = {gene: _gene_vector(adata, gene) for gene in MARKERS}
    for gene, values in marker_values.items():
        if values.shape != pseudotime.shape or not np.isfinite(values).all():
            raise Figure2AssemblyError(f"marker {gene} must be finite for every cell")
        if float(values.std(ddof=0)) == 0.0:
            raise Figure2AssemblyError(f"marker {gene} has zero population variance")
    sample_ids = adata.obs["sample_id"].astype(str).to_numpy()
    condition_labels = adata.obs["condition"].astype(str).to_numpy()
    return {
        "cell_id": cell_ids,
        "sample_id": sample_ids,
        "condition_label": condition_labels,
        "pseudotime": pseudotime,
        "umap": umap,
        "marker_values": marker_values,
        "expression_matrix": _expression_matrix_scope(adata),
    }


def _load_public_assignments(
    paths: Mapping[str, Path], data: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cell_ids = pd.Index(np.asarray(data["cell_id"], dtype=str), name="cell_id")
    sample_ids = np.asarray(data["sample_id"], dtype=str)
    frames: list[pd.DataFrame] = []
    source_summaries: list[dict[str, Any]] = []
    for sample_id, source in PUBLIC_ASSIGNMENT_SOURCES.items():
        assignment_path = paths[f"assignment_{sample_id}"]
        frame = pd.read_csv(
            assignment_path,
            sep="\t",
            compression="infer",
            dtype=str,
        )
        missing = sorted({"barcode", "assignment"}.difference(frame.columns))
        if missing:
            raise Figure2AssemblyError(
                f"public assignment file for {sample_id} is missing columns: "
                f"{', '.join(missing)}"
            )
        selected = frame.loc[:, ["barcode", "assignment"]].astype(str)
        unexpected = sorted(
            set(selected["assignment"]).difference(PUBLIC_ASSIGNMENT_LABELS)
        )
        if unexpected:
            raise Figure2AssemblyError(
                f"public assignment file for {sample_id} contains unexpected labels: "
                f"{', '.join(unexpected)}"
            )
        selected["cell_id"] = sample_id + ":" + selected["barcode"]
        subset = selected.loc[selected["cell_id"].isin(cell_ids)]
        source_summaries.append(
            {
                "sample_id": sample_id,
                "accession": source["accession"],
                "filename": source["filename"],
                "url": source["url"],
                "source_rows": len(selected),
                "erythroid_subset_matches": len(subset),
                "erythroid_subset_assignment_counts": {
                    label: int((subset["assignment"] == label).sum())
                    for label in PUBLIC_ASSIGNMENT_LABELS
                },
            }
        )
        frames.append(selected)

    combined = pd.concat(frames, ignore_index=True)
    if combined["cell_id"].duplicated().any():
        raise Figure2AssemblyError("public assignment cell identifiers are not unique")
    lookup = combined.set_index("cell_id")
    aligned = lookup.reindex(cell_ids)
    matched = aligned["assignment"].notna().to_numpy()
    assignments = aligned["assignment"].fillna("unmatched").to_numpy(dtype=str)
    accession_by_sample = {
        sample: source["accession"]
        for sample, source in PUBLIC_ASSIGNMENT_SOURCES.items()
    }
    accessions = np.asarray(
        [accession_by_sample.get(sample, "unavailable") for sample in sample_ids],
        dtype=str,
    )
    assignment_table = pd.DataFrame(
        {
            "cell_id": cell_ids.to_numpy(),
            "public_assignment": assignments,
            "public_assignment_match_status": np.where(matched, "matched", "unmatched"),
            "public_assignment_source_accession": accessions,
        }
    )
    assignment_counts = {
        label: int(np.count_nonzero(assignments == label))
        for label in (*PUBLIC_ASSIGNMENT_LABELS, "unmatched")
    }
    condition_counts = {
        str(label): int(count)
        for label, count in pd.Series(data["condition_label"])
        .value_counts(dropna=False)
        .sort_index()
        .items()
    }
    summary = {
        "source_files": source_summaries,
        "label_semantics": (
            "GEO names these files donor_assignments, but their recorded labels are "
            "control and gata307mut; no donor identity is inferred"
        ),
        "subset_assignment_counts": assignment_counts,
        "matched_cells": int(matched.sum()),
        "unmatched_cells": int((~matched).sum()),
        "total_cells": len(cell_ids),
        "pooled_in_figure": True,
        "frozen_condition_label": {
            "column": "condition",
            "counts": condition_counts,
            "status": (
                "legacy sample-prefix inference labelled the frozen subset uniformly "
                "as Disease; this label is not used for grouping or interpretation"
                if condition_counts == {"Disease": len(cell_ids)}
                else "legacy inferred label; not used for grouping or interpretation"
            ),
        },
    }
    return assignment_table, summary


def _population_zscore(values: Sequence[float] | np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise Figure2AssemblyError("population z-score requires a finite vector")
    standard_deviation = float(vector.std(ddof=0))
    if standard_deviation == 0.0:
        raise Figure2AssemblyError("population z-score requires nonzero variance")
    return (vector - float(vector.mean())) / standard_deviation


def _pseudotime_order(pseudotime: np.ndarray, cell_ids: np.ndarray) -> np.ndarray:
    values = np.asarray(pseudotime, dtype=float)
    ids = np.asarray(cell_ids, dtype=str)
    if values.ndim != 1 or ids.ndim != 1 or values.shape != ids.shape:
        raise Figure2AssemblyError("pseudotime and cell identifiers must align")
    if not np.isfinite(values).all() or not pd.Index(ids).is_unique:
        raise Figure2AssemblyError("pseudotime must be finite and cell IDs unique")
    return np.lexsort((ids, values))


def _smooth_scaled_expression(
    values: Sequence[float] | np.ndarray,
    *,
    window: int = 151,
    min_periods: int = 15,
) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise Figure2AssemblyError("smoothing requires a finite vector")
    if window <= 0 or min_periods <= 0 or min_periods > window:
        raise Figure2AssemblyError("smoothing window and min_periods are invalid")
    smoothed = (
        pd.Series(vector)
        .rolling(window=window, center=True, min_periods=min_periods)
        .mean()
        .bfill()
        .ffill()
        .to_numpy(dtype=float)
    )
    if not np.isfinite(smoothed).all():
        raise Figure2AssemblyError("smoothing did not produce a finite profile")
    return smoothed


def _build_cell_tables(
    adata: ad.AnnData, paths: Mapping[str, Path]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    data = _validate_adata(adata)
    assignment_table, assignment_summary = _load_public_assignments(paths, data)
    cell_ids = data["cell_id"]
    pseudotime = data["pseudotime"]
    umap = data["umap"]
    marker_values = data["marker_values"]
    marker_z = {gene: _population_zscore(marker_values[gene]) for gene in MARKERS}
    z_matrix = np.column_stack([marker_z[gene] for gene in MARKERS])
    winner_index = np.argmax(z_matrix, axis=1)
    state_names = np.asarray([MARKER_STATE_NAMES[gene] for gene in MARKERS])
    states = state_names[winner_index]
    sorted_z = np.sort(z_matrix, axis=1)
    margins = sorted_z[:, -1] - sorted_z[:, -2]

    cell_source = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "sample_id": data["sample_id"],
            "frozen_inferred_condition_label": data["condition_label"],
            "public_assignment": assignment_table["public_assignment"],
            "public_assignment_match_status": assignment_table[
                "public_assignment_match_status"
            ],
            "public_assignment_source_accession": assignment_table[
                "public_assignment_source_accession"
            ],
            "umap1": umap[:, 0],
            "umap2": umap[:, 1],
            "dpt_pseudotime": pseudotime,
            "CD34_scaled": marker_values["CD34"],
            "MKI67_scaled": marker_values["MKI67"],
            "HBB_scaled": marker_values["HBB"],
            "CD34_z": marker_z["CD34"],
            "MKI67_z": marker_z["MKI67"],
            "HBB_z": marker_z["HBB"],
            "marker_dominant_heuristic_state": states,
            "top_second_z_margin": margins,
        }
    )

    order = _pseudotime_order(pseudotime, cell_ids)
    ordered_hbb = marker_values["HBB"][order]
    ordered_mki67 = marker_values["MKI67"][order]
    marker_profiles = pd.DataFrame(
        {
            "pseudotime_rank": np.arange(1, len(order) + 1),
            "cell_id": cell_ids[order],
            "dpt_pseudotime": pseudotime[order],
            "HBB_scaled": ordered_hbb,
            "MKI67_scaled": ordered_mki67,
            "HBB_smoothed_scaled": _smooth_scaled_expression(ordered_hbb),
            "MKI67_smoothed_scaled": _smooth_scaled_expression(ordered_mki67),
        }
    )
    return (
        cell_source,
        marker_profiles,
        dict(data["expression_matrix"]),
        assignment_summary,
    )


def _select_target_pathways(results: pd.DataFrame) -> pd.DataFrame:
    selections: list[pd.DataFrame] = []
    for pathway in TARGET_PATHWAYS:
        selected = results.loc[results["Pathway"] == pathway].sort_values(
            ["pt_mid", "window_id"], kind="mergesort"
        )
        if len(selected) != EXPECTED_N_WINDOWS:
            raise Figure2AssemblyError(
                f"target pathway {pathway!r} does not contain 62 windows"
            )
        selections.append(selected)
    return pd.concat(selections, ignore_index=True)


def _render_figure(
    cell_source: pd.DataFrame,
    marker_profiles: pd.DataFrame,
    pathway_profiles: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
) -> None:
    style = {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#d1d5db",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "axes.edgecolor": "#64748b",
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.frameon": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 10))
        ax1, ax2, ax3, ax4 = axes.ravel()

        scatter = ax1.scatter(
            cell_source["umap1"],
            cell_source["umap2"],
            c=cell_source["dpt_pseudotime"],
            s=9,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            linewidths=0,
            alpha=0.9,
        )
        ax1.set_title("A. UMAP colored by pseudotime", loc="left", fontweight="bold")
        ax1.set_xlabel("UMAP1")
        ax1.set_ylabel("UMAP2")
        colorbar = fig.colorbar(scatter, ax=ax1, fraction=0.046, pad=0.04)
        colorbar.set_label("Pseudotime")

        state_column = "marker_dominant_heuristic_state"
        for state, color in STATE_COLORS.items():
            subset = cell_source.loc[cell_source[state_column] == state]
            ax2.scatter(
                subset["umap1"],
                subset["umap2"],
                s=9,
                color=color,
                linewidths=0,
                alpha=0.9,
                label=state,
            )
        ax2.set_title("B. Dominant-marker groups", loc="left", fontweight="bold")
        ax2.set_xlabel("UMAP1")
        ax2.set_ylabel("UMAP2")
        ax2.legend(title="Largest marker z-score", fontsize=9, loc="best")

        ax3.plot(
            marker_profiles["dpt_pseudotime"],
            marker_profiles["HBB_smoothed_scaled"],
            color="#b91c1c",
            linewidth=2.2,
            label="HBB",
        )
        ax3.plot(
            marker_profiles["dpt_pseudotime"],
            marker_profiles["MKI67_smoothed_scaled"],
            color="#1d4ed8",
            linewidth=2.2,
            label="MKI67",
        )
        ax3.set_title(
            "C. Marker-gene dynamics along pseudotime",
            loc="left",
            fontweight="bold",
        )
        ax3.set_xlabel("Pseudotime")
        ax3.set_ylabel("Smoothed scaled expression")
        ax3.legend()

        for pathway in TARGET_PATHWAYS:
            subset = pathway_profiles.loc[pathway_profiles["Pathway"] == pathway]
            color = PATHWAY_COLORS[pathway]
            ax4.plot(
                subset["pt_mid"],
                subset["NES"],
                linewidth=2.2,
                color=color,
                label=pathway,
            )
            significant = subset.loc[subset["padj"] < 0.05]
            ax4.scatter(
                significant["pt_mid"],
                significant["NES"],
                color=color,
                s=26,
                zorder=3,
            )
        ax4.axhline(0.0, linestyle="--", color="#64748b", linewidth=1.0)
        ax4.set_title("D. Rolling-window pathway NES", loc="left", fontweight="bold")
        ax4.set_xlabel("Pseudotime window midpoint")
        ax4.set_ylabel("NES")
        ax4.legend(title="Points: within-window BH FDR < 0.05")

        fig.suptitle(
            "GSE155254 erythroid-subset trajectory summary",
            y=0.98,
            fontsize=14,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
            metadata={"Software": "PyFgsea Figure 2 assembly"},
        )
        fig.savefig(
            pdf_path,
            bbox_inches="tight",
            metadata={
                "Title": "PyFgsea GSE155254 Figure 2",
                "Subject": "Descriptive computational trajectory assembly",
            },
        )
        plt.close(fig)


def _file_record(
    path: Path, published_path: Path, *, rows: int | None = None
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(published_path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if rows is not None:
        record["rows"] = rows
    return record


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("anndata", "h5py", "matplotlib", "numpy", "pandas", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _runner_git_state() -> dict[str, Any]:
    def run(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    branch = run("branch", "--show-current")
    if commit.returncode or status.returncode or branch.returncode:
        return {"available": False}
    return {
        "available": True,
        "commit": commit.stdout.strip().lower(),
        "branch": branch.stdout.strip(),
        "clean": not bool(status.stdout),
    }


def _input_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "sha256_role": "provenance_only",
    }


def _write_manifest(
    path: Path,
    *,
    paths: Mapping[str, Path],
    upstream_summary: Mapping[str, Any],
    result_validation: Mapping[str, Any],
    gene_set_validation: Mapping[str, Any],
    expression_matrix: Mapping[str, Any],
    assignment_summary: Mapping[str, Any],
    rendered_figure: bool,
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    assignment_inputs: dict[str, Any] = {}
    for sample_id, source in PUBLIC_ASSIGNMENT_SOURCES.items():
        record = _input_record(paths[f"assignment_{sample_id}"])
        record.update(
            {
                "accession": source["accession"],
                "filename": source["filename"],
                "url": source["url"],
            }
        )
        assignment_inputs[sample_id] = record
    assignment_counts = _require_mapping(
        assignment_summary.get("subset_assignment_counts"),
        "public assignment counts",
    )
    total_cells = int(assignment_summary["total_cells"])
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": (
            "pyfgsea-figure2-full-assembly"
            if rendered_figure
            else "pyfgsea-figure2-table-assembly"
        ),
        "assembly_status": "assembled",
        "panel_status": {
            "panels_a_c": "deterministic_reconstruction_from_frozen_h5ad",
            "panel_d": "installed_pyfgsea_functional_rerun",
            "rendered_figure": "generated" if rendered_figure else "not_requested",
        },
        "created_at_utc": _utc_now(),
        "upstream": {
            "figure2_run_dir": str(paths["run_dir"]),
            "status": upstream_summary["status"],
            "runtime": dict(upstream_summary["runtime"]),
            "parameters": dict(upstream_summary["parameters"]),
            "result_validation": dict(result_validation),
            "path_resolution": "explicit dataset, gene-set, and run-directory inputs",
            "inputs": {
                "run_summary": _input_record(paths["run_summary"]),
                "dataset": _input_record(paths["dataset"]),
                "gene_sets": _input_record(paths["gene_sets"]),
                "trajectory_results": _input_record(paths["trajectory_results"]),
                "public_assignments": assignment_inputs,
            },
            "gene_set_validation": dict(gene_set_validation),
        },
        "assembly": {
            "runner": {
                "path": str(RUNNER_PATH),
                "sha256": _sha256_file(RUNNER_PATH),
                "sha256_role": "provenance_only",
                "git": _runner_git_state(),
            },
            "environment": {
                "python": sys.version.splitlines()[0],
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "packages": _package_versions(),
                "matplotlib_backend": matplotlib.get_backend(),
            },
            "methods": {
                "panels_a_b_coordinates": (
                    "precomputed X_umap coordinates retained for the erythroid subset "
                    "from the original full-data embedding; no UMAP recomputation"
                ),
                "pseudotime": (
                    "precomputed obs['dpt_pseudotime'] from the erythroid-subset graph"
                ),
                "pseudotime_tie_order": "numpy.lexsort(cell_id, pseudotime)",
                "frozen_expression_matrix": dict(expression_matrix),
                "public_assignments": dict(assignment_summary),
                "panel_b": {
                    "markers": list(MARKERS),
                    "standardization": "per-marker population z-score (ddof=0)",
                    "assignment": "argmax marker z-score",
                    "interpretation": "marker-dominant heuristic; not a cell-type annotation",
                },
                "panel_c": {
                    "expression": "scaled adata.X values",
                    "smoothing": (
                        "151-cell centered rolling mean, min_periods=15, then bfill/ffill"
                    ),
                },
                "panel_d": {
                    "source": "functional upstream trajectory_results.csv; no GSEA recomputation",
                    "ranking_matrix": (
                        "scaled adata.X from the frozen H5AD; no raw or named "
                        "expression layer is available"
                    ),
                    "pathways": list(TARGET_PATHWAYS),
                    "significance_points": "padj < 0.05",
                    "multiple_testing_scope": "Benjamini-Hochberg within each window",
                },
            },
        },
        "artifacts": dict(artifacts),
        "claim_boundary": {
            "research_mode": "descriptive in-silico assembly of an existing public dataset",
            "panels_a_c": "descriptive cell and marker trajectories",
            "panel_b": "marker-dominant heuristic states, not cell-type annotations",
            "panel_d": "within-window FDR only; not trajectory-wide inference",
            "dataset_scope": (
                f"{total_cells} erythroid-subset cells pooled across public assignment "
                f"categories: {assignment_counts['control']} control, "
                f"{assignment_counts['gata307mut']} gata307mut, and "
                f"{assignment_counts['unmatched']} unmatched; Figure 2 is descriptive "
                "and is not a group comparison"
            ),
            "frozen_condition_label": dict(
                _require_mapping(
                    assignment_summary.get("frozen_condition_label"),
                    "frozen condition label",
                )
            ),
            "public_assignments": {
                "matched_cells": assignment_summary["matched_cells"],
                "unmatched_cells": assignment_summary["unmatched_cells"],
                "pooled_in_figure": True,
                "label_semantics": (
                    "control/gata307mut assignment from the public GEO files; not "
                    "donor identity"
                ),
            },
            "excluded_claims": [
                "control-versus-gata307mut inference",
                "donor-level inference",
                "disease-only inference",
                "causal or mechanistic inference",
            ],
        },
    }
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def assemble_figure2(
    figure2_run_dir: Path,
    dataset_path: Path,
    gene_sets_path: Path,
    assignment_dir: Path,
    output_dir: Path,
    *,
    render_figure: bool = True,
) -> dict[str, Any]:
    """Validate inputs and atomically publish the reconstructed Figure 2."""

    paths = _input_paths(
        figure2_run_dir,
        dataset_path,
        gene_sets_path,
        assignment_dir,
    )
    output = _require_external_output(output_dir, paths["run_dir"])
    upstream_summary = _load_json(paths["run_summary"], "run_summary.json")
    _validate_run_summary(upstream_summary)
    gene_set_validation = _validate_gene_sets(paths["gene_sets"])

    try:
        raw_results = pd.read_csv(paths["trajectory_results"])
    except Exception as error:
        raise Figure2AssemblyError(
            "trajectory_results.csv could not be read"
        ) from error
    results, result_validation = _validate_results(raw_results)
    pathway_profiles = _select_target_pathways(results)

    try:
        adata = ad.read_h5ad(paths["dataset"])
    except Exception as error:
        raise Figure2AssemblyError("frozen H5AD dataset could not be read") from error
    cell_source, marker_profiles, expression_matrix, assignment_summary = (
        _build_cell_tables(adata, paths)
    )

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}-", suffix="-staging", dir=str(output.parent)
        )
    )
    try:
        csv_options = {"index": False, "float_format": "%.17g"}
        cell_source.to_csv(staging / "figure2_cell_source.csv", **csv_options)
        marker_profiles.to_csv(staging / "figure2_marker_profiles.csv", **csv_options)
        pathway_profiles.to_csv(staging / "figure2_pathway_profiles.csv", **csv_options)
        if render_figure:
            _render_figure(
                cell_source,
                marker_profiles,
                pathway_profiles,
                staging / "figure2_full_rc8.png",
                staging / "figure2_full_rc8.pdf",
            )

        rows_by_name = {
            "figure2_cell_source.csv": len(cell_source),
            "figure2_marker_profiles.csv": len(marker_profiles),
            "figure2_pathway_profiles.csv": len(pathway_profiles),
        }
        artifact_names = [
            "figure2_cell_source.csv",
            "figure2_marker_profiles.csv",
            "figure2_pathway_profiles.csv",
        ]
        if render_figure:
            artifact_names[:0] = ["figure2_full_rc8.png", "figure2_full_rc8.pdf"]
        artifacts = {
            name: _file_record(
                staging / name,
                output / name,
                rows=rows_by_name.get(name),
            )
            for name in artifact_names
        }
        manifest = _write_manifest(
            staging / "assembly_manifest.json",
            paths=paths,
            upstream_summary=upstream_summary,
            result_validation=result_validation,
            gene_set_validation=gene_set_validation,
            expression_matrix=expression_matrix,
            assignment_summary=assignment_summary,
            rendered_figure=render_figure,
            artifacts=artifacts,
        )
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output}")
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct Figure 2 panels A-C from frozen inputs and assemble them "
            "with a completed functional Panel-D run."
        )
    )
    parser.add_argument(
        "--figure2-run-dir",
        required=True,
        type=Path,
        help="Completed functional Panel-D run directory.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Frozen GSE155254 erythroid-subset H5AD.",
    )
    parser.add_argument(
        "--gene-sets",
        required=True,
        type=Path,
        help="Gene-set GMT used for the Panel-D run.",
    )
    parser.add_argument(
        "--assignment-dir",
        required=True,
        type=Path,
        help="Directory containing the two public GEO assignment TSV files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New output directory outside the checkout and functional run.",
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Write source tables and the manifest without rendering image files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        assemble_figure2(
            args.figure2_run_dir,
            args.dataset,
            args.gene_sets,
            args.assignment_dir,
            args.output_dir,
            render_figure=not args.tables_only,
        )
    except (Figure2AssemblyError, FileExistsError) as error:
        print(f"Figure 2 assembly failed: {error}", file=sys.stderr)
        return 1
    print(f"Figure 2 assembly written to {args.output_dir.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
