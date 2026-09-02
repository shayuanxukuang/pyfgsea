from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "assemble_gse155254_figure2.py"
)
SPEC = importlib.util.spec_from_file_location("figure2_assembly", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
figure2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(figure2)


def _release_state() -> dict[str, object]:
    return {
        "clean": True,
        "commit": figure2.EXPECTED_COMMIT,
        "release_tag": {
            "name": figure2.EXPECTED_TAG,
            "annotated": True,
            "peeled_commit": figure2.EXPECTED_COMMIT,
        },
    }


def _manifest() -> dict[str, object]:
    return {
        "schema_version": 2,
        "verification_status": "verified",
        "dataset_shape": list(figure2.EXPECTED_DATASET_SHAPE),
        "git": {
            "start": _release_state(),
            "end": _release_state(),
            "unchanged": True,
        },
        "parameters": {
            **figure2.EXPECTED_PARAMETERS,
            "pseudotime_key": "dpt_pseudotime",
        },
        "result_validation": {
            "complete_grid": True,
            "expected_grid": [
                figure2.EXPECTED_N_WINDOWS,
                figure2.EXPECTED_N_PATHWAYS,
            ],
            "n_windows": figure2.EXPECTED_N_WINDOWS,
            "n_pathways": figure2.EXPECTED_N_PATHWAYS,
            "n_rows": figure2.EXPECTED_N_ROWS,
            "resolved_rows": figure2.EXPECTED_N_ROWS,
            "status_counts": {"resolved": figure2.EXPECTED_N_ROWS},
            "bh": {
                "matches_core": True,
                "max_absolute_difference": 0.0,
                "scope": "within-window",
            },
        },
        "frozen_input_artifacts": {
            "dataset": {"sha256": "declared-dataset-provenance"},
            "gene_sets": {"sha256": "declared-gmt-provenance"},
        },
        "artifacts": {
            "trajectory_results.csv": {"sha256": "declared-results-provenance"}
        },
    }


def _results() -> pd.DataFrame:
    pathways = [figure2.TARGET_PATHWAYS[0], figure2.TARGET_PATHWAYS[1]] + [
        f"Pathway {index:02d}" for index in range(figure2.EXPECTED_N_PATHWAYS - 2)
    ]
    rows: list[dict[str, object]] = []
    for window_id in range(figure2.EXPECTED_N_WINDOWS):
        for pathway_index, pathway in enumerate(pathways, start=1):
            rows.append(
                {
                    "Pathway": pathway,
                    "ES": np.sin((window_id + pathway_index) / 20.0),
                    "NES": np.sin((window_id + pathway_index) / 12.0),
                    "P-value": pathway_index / 1000.0,
                    "padj": figure2.EXPECTED_N_PATHWAYS / 1000.0,
                    "status": "resolved",
                    "window_id": window_id,
                    "pt_start": window_id / 100.0,
                    "pt_end": (window_id + 1) / 100.0,
                    "pt_mid": (window_id + 0.5) / 100.0,
                }
            )
    return pd.DataFrame(rows)


def _adata() -> ad.AnnData:
    n_obs, n_vars = figure2.EXPECTED_DATASET_SHAPE
    pseudotime = np.linspace(0.0, 1.0, n_obs)
    pseudotime[101] = pseudotime[100]
    cell_ids = [f"cell-{index:04d}" for index in range(n_obs)]
    cd34 = 2.0 - 4.0 * pseudotime
    mki67 = 2.5 - 14.0 * np.square(pseudotime - 0.5)
    hbb = -2.0 + 4.0 * pseudotime
    marker_matrix = np.column_stack([cd34, mki67, hbb])
    row_indices = np.repeat(np.arange(n_obs), 3)
    column_indices = np.tile(np.arange(3), n_obs)
    matrix = sparse.csr_matrix(
        (marker_matrix.ravel(), (row_indices, column_indices)),
        shape=(n_obs, n_vars),
    )
    var_names = list(figure2.MARKERS) + [
        f"Gene-{index:04d}" for index in range(3, n_vars)
    ]
    adata = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame({"dpt_pseudotime": pseudotime}, index=pd.Index(cell_ids)),
        var=pd.DataFrame(index=pd.Index(var_names)),
    )
    adata.obsm["X_umap"] = np.column_stack(
        [
            np.cos(2.0 * np.pi * pseudotime),
            np.sin(2.0 * np.pi * pseudotime),
        ]
    )
    return adata


def _write_verified_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "verified-run"
    frozen_inputs = run_dir / "frozen_inputs"
    frozen_inputs.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(_manifest(), indent=2), encoding="utf-8"
    )
    _results().to_csv(run_dir / "trajectory_results.csv", index=False)
    _adata().write_h5ad(frozen_inputs / "gse155254_ery_only_pt.h5ad")
    pathways = [figure2.TARGET_PATHWAYS[0], figure2.TARGET_PATHWAYS[1]] + [
        f"Pathway {index:02d}" for index in range(figure2.EXPECTED_N_PATHWAYS - 2)
    ]
    (frozen_inputs / "hallmark_enrichr.gmt").write_text(
        "".join(f"{pathway}\tfixture\tGENE1\tGENE2\n" for pathway in pathways),
        encoding="utf-8",
    )
    return run_dir


def test_complete_assembly_from_verified_run(tmp_path: Path) -> None:
    run_dir = _write_verified_run(tmp_path)
    output_dir = tmp_path / "assembled-figure2"

    manifest = figure2.assemble_figure2(run_dir, output_dir)

    expected_files = {
        "assembly_manifest.json",
        "figure2_cell_source.csv",
        "figure2_full_rc8.pdf",
        "figure2_full_rc8.png",
        "figure2_marker_profiles.csv",
        "figure2_pathway_profiles.csv",
    }
    assert {path.name for path in output_dir.iterdir()} == expected_files
    assert manifest["assembly_status"] == "assembled"
    assert manifest["panel_status"] == {
        "panels_a_c": "deterministic_reconstruction_from_frozen_h5ad",
        "panel_d": "verified_rc8_rerun",
    }
    assert manifest["upstream"]["verification_status"] == "verified"
    assert manifest["claim_boundary"]["panel_b"].endswith("not cell-type annotations")
    assert manifest["upstream"]["inputs"]["dataset"]["sha256_role"] == (
        "provenance_only"
    )

    cell_source = pd.read_csv(output_dir / "figure2_cell_source.csv")
    marker_profiles = pd.read_csv(output_dir / "figure2_marker_profiles.csv")
    pathway_profiles = pd.read_csv(output_dir / "figure2_pathway_profiles.csv")
    assert len(cell_source) == figure2.EXPECTED_DATASET_SHAPE[0]
    assert len(marker_profiles) == figure2.EXPECTED_DATASET_SHAPE[0]
    assert len(pathway_profiles) == 2 * figure2.EXPECTED_N_WINDOWS
    assert pathway_profiles.groupby("Pathway").size().to_dict() == {
        "E2F Targets": 62,
        "heme Metabolism": 62,
    }
    assert {
        "CD34_z",
        "MKI67_z",
        "HBB_z",
        "marker_dominant_heuristic_state",
        "top_second_z_margin",
    }.issubset(cell_source.columns)
    assert set(cell_source["marker_dominant_heuristic_state"]) == {
        "CD34-dominant",
        "MKI67-dominant",
        "HBB-dominant",
    }
    assert {
        "HBB_smoothed_scaled",
        "MKI67_smoothed_scaled",
    }.issubset(marker_profiles.columns)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda value: value.update(verification_status="candidate"), "not verified"),
        (
            lambda value: value["git"]["end"].update(commit="0" * 40),
            "clean RC8 commit",
        ),
        (
            lambda value: value["git"]["start"]["release_tag"].update(
                name="v0.2.0-rc7"
            ),
            "annotated tag",
        ),
        (
            lambda value: value["parameters"].update(window_size=400),
            "window_size",
        ),
        (
            lambda value: value.update(dataset_shape=[3575, 3000]),
            "dataset_shape",
        ),
    ],
)
def test_manifest_and_parameter_failures(change: object, message: str) -> None:
    manifest = copy.deepcopy(_manifest())
    change(manifest)
    with pytest.raises(figure2.Figure2AssemblyError, match=message):
        figure2._validate_run_manifest(manifest)


@pytest.mark.parametrize(
    "failure", ["missing", "unresolved", "bh", "midpoint_group", "midpoint_order"]
)
def test_grid_and_bh_failures(failure: str) -> None:
    results = _results()
    if failure == "missing":
        results = results.iloc[:-1].copy()
        message = "rows"
    elif failure == "unresolved":
        results.loc[0, "status"] = "eps_floor"
        message = "unresolved"
    elif failure == "bh":
        results.loc[0, "padj"] = 0.5
        message = "within-window BH"
    elif failure == "midpoint_group":
        results.loc[0, "pt_mid"] += 0.001
        message = "inconsistent pt_mid"
    else:
        results.loc[results["window_id"] == 20, "pt_mid"] = results.loc[
            results["window_id"] == 19, "pt_mid"
        ].iloc[0]
        message = "strictly increasing"
    with pytest.raises(figure2.Figure2AssemblyError, match=message):
        figure2._validate_results(results)


def test_missing_adata_fields_and_nonfinite_values() -> None:
    missing_pseudotime = _adata()
    del missing_pseudotime.obs["dpt_pseudotime"]
    with pytest.raises(figure2.Figure2AssemblyError, match="dpt_pseudotime"):
        figure2._validate_adata(missing_pseudotime)

    missing_umap = _adata()
    del missing_umap.obsm["X_umap"]
    with pytest.raises(figure2.Figure2AssemblyError, match="X_umap"):
        figure2._validate_adata(missing_umap)

    missing_marker = _adata()
    names = missing_marker.var_names.to_list()
    names[0] = "NOT_CD34"
    missing_marker.var_names = names
    with pytest.raises(figure2.Figure2AssemblyError, match="CD34"):
        figure2._validate_adata(missing_marker)

    nonfinite_pseudotime = _adata()
    nonfinite_pseudotime.obs.iloc[0, 0] = np.nan
    with pytest.raises(figure2.Figure2AssemblyError, match="must be finite"):
        figure2._validate_adata(nonfinite_pseudotime)

    nonfinite_marker = _adata()
    nonfinite_marker.X[0, 0] = np.nan
    with pytest.raises(figure2.Figure2AssemblyError, match="marker CD34"):
        figure2._validate_adata(nonfinite_marker)


def test_pseudotime_ties_use_cell_id_as_secondary_key() -> None:
    pseudotime = np.array([0.2, 0.1, 0.1, 0.2])
    cell_ids = np.array(["d", "b", "a", "c"])
    order = figure2._pseudotime_order(pseudotime, cell_ids)
    assert order.tolist() == [2, 1, 3, 0]


def test_population_zscore_and_centered_rolling_mean() -> None:
    values = np.array([1.0, 2.0, 3.0])
    expected_z = (values - values.mean()) / values.std(ddof=0)
    np.testing.assert_allclose(figure2._population_zscore(values), expected_z)

    smoothed = figure2._smooth_scaled_expression(
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        window=3,
        min_periods=1,
    )
    np.testing.assert_allclose(smoothed, [1.5, 2.0, 3.0, 4.0, 4.5])


def test_output_guards_and_failed_staging_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _write_verified_run(tmp_path)
    with pytest.raises(figure2.Figure2AssemblyError, match="must not modify"):
        figure2._require_external_output(run_dir / "nested", run_dir)

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        figure2._require_external_output(existing, run_dir)

    output = tmp_path / "failed-output"

    def fail_render(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("plot failed")

    monkeypatch.setattr(figure2, "_render_figure", fail_render)
    with pytest.raises(RuntimeError, match="plot failed"):
        figure2.assemble_figure2(run_dir, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".failed-output-*/"))
