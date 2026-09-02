"""Assemble the GSE155254 Figure 2 from frozen inputs and a verified RC8 run.

This script does not recompute UMAP coordinates or enrichment statistics. It
uses only the frozen inputs and trajectory table contained in the supplied
Figure 2 run directory.
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
EXPECTED_COMMIT = "ee5855bad7200655e580f44fdf4087bb1bad67b5"
EXPECTED_TAG = "v0.2.0-rc8"
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


class Figure2AssemblyError(RuntimeError):
    """Raised when the verified run cannot support Figure 2 assembly."""


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


def _input_paths(figure2_run_dir: Path) -> dict[str, Path]:
    run_dir = figure2_run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise Figure2AssemblyError(f"Figure 2 run directory was not found at {run_dir}")
    paths = {
        "run_dir": run_dir,
        "run_manifest": run_dir / "run_manifest.json",
        "dataset": run_dir / "frozen_inputs" / "gse155254_ery_only_pt.h5ad",
        "gene_sets": run_dir / "frozen_inputs" / "hallmark_enrichr.gmt",
        "trajectory_results": run_dir / "trajectory_results.csv",
    }
    for name in ("run_manifest", "dataset", "gene_sets", "trajectory_results"):
        paths[name] = _resolved_file(paths[name], run_dir, name.replace("_", " "))
    return paths


def _require_external_output(output_dir: Path, run_dir: Path) -> Path:
    output = output_dir.expanduser().resolve()
    if _is_within(output, REPO_ROOT):
        raise Figure2AssemblyError(
            "The output directory must be outside the assembly Git checkout"
        )
    if _is_within(output, run_dir):
        raise Figure2AssemblyError(
            "The output directory must not modify the verified Figure 2 run"
        )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _same_json_value(observed: Any, expected: Any) -> bool:
    return type(observed) is type(expected) and observed == expected


def _validate_release_state(value: Any, context: str) -> None:
    state = _require_mapping(value, context)
    if state.get("commit") != EXPECTED_COMMIT or state.get("clean") is not True:
        raise Figure2AssemblyError(
            f"{context} must identify the clean RC8 commit {EXPECTED_COMMIT}"
        )
    tag = _require_mapping(state.get("release_tag"), f"{context} release tag")
    expected_tag = {
        "name": EXPECTED_TAG,
        "annotated": True,
        "peeled_commit": EXPECTED_COMMIT,
    }
    if any(tag.get(key) != expected for key, expected in expected_tag.items()):
        raise Figure2AssemblyError(
            f"{context} must identify annotated tag {EXPECTED_TAG} at RC8"
        )


def _validate_run_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("verification_status") != "verified":
        raise Figure2AssemblyError("run_manifest verification_status is not verified")
    if manifest.get("schema_version") != 2:
        raise Figure2AssemblyError("run_manifest schema_version is not 2")
    if tuple(manifest.get("dataset_shape", ())) != EXPECTED_DATASET_SHAPE:
        raise Figure2AssemblyError(
            f"run_manifest dataset_shape is not {EXPECTED_DATASET_SHAPE}"
        )

    git = _require_mapping(manifest.get("git"), "run_manifest git")
    _validate_release_state(git.get("start"), "run_manifest git.start")
    _validate_release_state(git.get("end"), "run_manifest git.end")
    if git.get("unchanged") is not True:
        raise Figure2AssemblyError("run_manifest Git state changed during the run")

    parameters = _require_mapping(manifest.get("parameters"), "run_manifest parameters")
    for name, expected in EXPECTED_PARAMETERS.items():
        if not _same_json_value(parameters.get(name), expected):
            raise Figure2AssemblyError(
                f"run_manifest parameter {name!r} is not {expected!r}"
            )

    validation = _require_mapping(
        manifest.get("result_validation"), "run_manifest result_validation"
    )
    expected_values = {
        "complete_grid": True,
        "expected_grid": [EXPECTED_N_WINDOWS, EXPECTED_N_PATHWAYS],
        "n_windows": EXPECTED_N_WINDOWS,
        "n_pathways": EXPECTED_N_PATHWAYS,
        "n_rows": EXPECTED_N_ROWS,
        "resolved_rows": EXPECTED_N_ROWS,
        "status_counts": {"resolved": EXPECTED_N_ROWS},
    }
    if any(
        validation.get(key) != expected for key, expected in expected_values.items()
    ):
        raise Figure2AssemblyError(
            "run_manifest does not report the complete resolved 62 x 43 grid"
        )
    bh = _require_mapping(validation.get("bh"), "run_manifest BH validation")
    if (
        bh.get("matches_core") is not True
        or bh.get("scope") != "within-window"
        or not isinstance(bh.get("max_absolute_difference"), (int, float))
        or float(bh["max_absolute_difference"]) > BH_ATOL
    ):
        raise Figure2AssemblyError(
            "run_manifest does not report a matching within-window BH calculation"
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
    return {
        "cell_id": cell_ids,
        "pseudotime": pseudotime,
        "umap": umap,
        "marker_values": marker_values,
    }


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


def _build_cell_tables(adata: ad.AnnData) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = _validate_adata(adata)
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
    return cell_source, marker_profiles


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


def _input_record(path: Path, declared: Any = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "sha256_role": "provenance_only",
    }
    if isinstance(declared, Mapping) and isinstance(declared.get("sha256"), str):
        record["upstream_declared_sha256"] = declared["sha256"]
    return record


def _write_manifest(
    path: Path,
    *,
    paths: Mapping[str, Path],
    upstream_manifest: Mapping[str, Any],
    result_validation: Mapping[str, Any],
    gene_set_validation: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    frozen = _require_mapping(
        upstream_manifest.get("frozen_input_artifacts"),
        "run_manifest frozen_input_artifacts",
    )
    upstream_artifacts = _require_mapping(
        upstream_manifest.get("artifacts"), "run_manifest artifacts"
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "pyfgsea-figure2-full-assembly",
        "assembly_status": "assembled",
        "panel_status": {
            "panels_a_c": "deterministic_reconstruction_from_frozen_h5ad",
            "panel_d": "verified_rc8_rerun",
        },
        "created_at_utc": _utc_now(),
        "upstream": {
            "figure2_run_dir": str(paths["run_dir"]),
            "verification_status": upstream_manifest["verification_status"],
            "git_commit": EXPECTED_COMMIT,
            "git_tag": EXPECTED_TAG,
            "parameters": dict(upstream_manifest["parameters"]),
            "result_validation": dict(result_validation),
            "path_resolution": (
                "fixed run-relative filenames; upstream absolute paths were not followed"
            ),
            "inputs": {
                "run_manifest": _input_record(paths["run_manifest"]),
                "dataset": _input_record(paths["dataset"], frozen.get("dataset")),
                "gene_sets": _input_record(paths["gene_sets"], frozen.get("gene_sets")),
                "trajectory_results": _input_record(
                    paths["trajectory_results"],
                    upstream_artifacts.get("trajectory_results.csv"),
                ),
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
                    "source": "verified upstream trajectory_results.csv; no GSEA recomputation",
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
                "3576 Disease-labelled cells from two samples; donor assignments and "
                "a control group are unavailable"
            ),
            "excluded_claims": [
                "control-versus-disease inference",
                "disease-only inference",
                "causal or mechanistic inference",
            ],
        },
    }
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def assemble_figure2(figure2_run_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Validate inputs and atomically publish the reconstructed Figure 2."""

    paths = _input_paths(figure2_run_dir)
    output = _require_external_output(output_dir, paths["run_dir"])
    upstream_manifest = _load_json(paths["run_manifest"], "run_manifest.json")
    _validate_run_manifest(upstream_manifest)
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
    cell_source, marker_profiles = _build_cell_tables(adata)

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
        artifacts = {
            name: _file_record(
                staging / name,
                output / name,
                rows=rows_by_name.get(name),
            )
            for name in (
                "figure2_full_rc8.png",
                "figure2_full_rc8.pdf",
                "figure2_cell_source.csv",
                "figure2_marker_profiles.csv",
                "figure2_pathway_profiles.csv",
            )
        }
        manifest = _write_manifest(
            staging / "assembly_manifest.json",
            paths=paths,
            upstream_manifest=upstream_manifest,
            result_validation=result_validation,
            gene_set_validation=gene_set_validation,
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
            "with a verified RC8 panel-D run."
        )
    )
    parser.add_argument(
        "--figure2-run-dir",
        required=True,
        type=Path,
        help="Verified RC8 Figure 2 run directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New output directory outside the checkout and verified run.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        assemble_figure2(args.figure2_run_dir, args.output_dir)
    except (Figure2AssemblyError, FileExistsError) as error:
        print(f"Figure 2 assembly failed: {error}", file=sys.stderr)
        return 1
    print(f"Figure 2 assembly written to {args.output_dir.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
