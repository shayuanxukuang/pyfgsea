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


def _run_summary() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "pyfgsea-figure2-panel-d-table",
        "status": "complete",
        "runtime": {
            "package_version": "0.2.0",
            "distribution_version": "0.2.0",
            "algorithm_revision": figure2.EXPECTED_ALGORITHM_REVISION,
        },
        "dataset": {"shape": list(figure2.EXPECTED_DATASET_SHAPE)},
        "parameters": {
            **figure2.EXPECTED_PARAMETERS,
            "pseudotime_key": "dpt_pseudotime",
        },
        "results": {
            "complete_grid": True,
            "n_windows": figure2.EXPECTED_N_WINDOWS,
            "n_pathways": figure2.EXPECTED_N_PATHWAYS,
            "n_rows": figure2.EXPECTED_N_ROWS,
            "resolved_rows": figure2.EXPECTED_N_ROWS,
            "pathway_size_policy": "exact",
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
    sample_ids = np.where(
        np.arange(n_obs) < n_obs // 2,
        "GSM4698215_rep1",
        "GSM4698216_rep2",
    )
    barcodes = [f"CELL{index:04d}-1" for index in range(n_obs)]
    cell_ids = [
        f"{sample_id}:{barcode}" for sample_id, barcode in zip(sample_ids, barcodes)
    ]
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
        obs=pd.DataFrame(
            {
                "sample_id": sample_ids,
                "condition": "Disease",
                "dpt_pseudotime": pseudotime,
            },
            index=pd.Index(cell_ids),
        ),
        var=pd.DataFrame(index=pd.Index(var_names)),
    )
    adata.uns["log1p"] = {"base": None}
    adata.obsm["X_umap"] = np.column_stack(
        [
            np.cos(2.0 * np.pi * pseudotime),
            np.sin(2.0 * np.pi * pseudotime),
        ]
    )
    return adata


def _write_functional_run(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    run_dir = tmp_path / "functional-run"
    run_dir.mkdir()
    (run_dir / "run_summary.json").write_text(
        json.dumps(_run_summary(), indent=2), encoding="utf-8"
    )
    _results().to_csv(run_dir / "trajectory_results.csv", index=False)
    adata = _adata()
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    dataset = input_dir / "gse155254_ery_only_pt.h5ad"
    adata.write_h5ad(dataset)
    pathways = [figure2.TARGET_PATHWAYS[0], figure2.TARGET_PATHWAYS[1]] + [
        f"Pathway {index:02d}" for index in range(figure2.EXPECTED_N_PATHWAYS - 2)
    ]
    gene_sets = input_dir / "hallmark_enrichr.gmt"
    gene_sets.write_text(
        "".join(f"{pathway}\tfixture\tGENE1\tGENE2\n" for pathway in pathways),
        encoding="utf-8",
    )
    assignment_dir = tmp_path / "public-assignments"
    assignment_dir.mkdir()
    for sample_id, source in figure2.PUBLIC_ASSIGNMENT_SOURCES.items():
        barcodes = [
            cell_id.split(":", maxsplit=1)[1]
            for cell_id in adata.obs_names
            if cell_id.startswith(f"{sample_id}:")
        ][:2]
        pd.DataFrame(
            {
                "barcode": barcodes,
                "assignment": ["control", "gata307mut"],
            }
        ).to_csv(
            assignment_dir / source["filename"],
            sep="\t",
            index=False,
            compression="gzip",
        )
    return run_dir, dataset, gene_sets, assignment_dir


def test_complete_assembly_from_functional_run(tmp_path: Path) -> None:
    run_dir, dataset, gene_sets, assignment_dir = _write_functional_run(tmp_path)
    output_dir = tmp_path / "assembled-figure2"

    manifest = figure2.assemble_figure2(
        run_dir,
        dataset,
        gene_sets,
        assignment_dir,
        output_dir,
        render_figure=False,
    )

    expected_files = {
        "assembly_manifest.json",
        "figure2_cell_source.csv",
        "figure2_marker_profiles.csv",
        "figure2_pathway_profiles.csv",
    }
    assert {path.name for path in output_dir.iterdir()} == expected_files
    assert manifest["artifact_type"] == "pyfgsea-figure2-table-assembly"
    assert manifest["assembly_status"] == "assembled"
    assert manifest["panel_status"] == {
        "panels_a_c": "deterministic_reconstruction_from_frozen_h5ad",
        "panel_d": "installed_pyfgsea_functional_rerun",
        "rendered_figure": "not_requested",
    }
    assert manifest["upstream"]["status"] == "complete"
    assert manifest["claim_boundary"]["panel_b"].endswith("not cell-type annotations")
    assert manifest["upstream"]["inputs"]["dataset"]["sha256_role"] == (
        "provenance_only"
    )
    expression = manifest["assembly"]["methods"]["frozen_expression_matrix"]
    assert expression["matrix_used"] == "adata.X"
    assert expression["representation"] == "scaled_expression"
    assert expression["raw_present"] is False
    assert expression["named_layers"] == []
    assert expression["log1p_metadata_present"] is True
    assert "does not describe the matrix" in expression["interpretation"]
    assignments = manifest["assembly"]["methods"]["public_assignments"]
    assert assignments["subset_assignment_counts"] == {
        "control": 2,
        "gata307mut": 2,
        "unmatched": figure2.EXPECTED_DATASET_SHAPE[0] - 4,
    }
    assert assignments["pooled_in_figure"] is True
    assert manifest["claim_boundary"]["public_assignments"]["label_semantics"].endswith(
        "not donor identity"
    )
    assert "not a group comparison" in manifest["claim_boundary"]["dataset_scope"]
    assert set(manifest["upstream"]["inputs"]["public_assignments"]) == set(
        figure2.PUBLIC_ASSIGNMENT_SOURCES
    )
    assert all(
        record["sha256_role"] == "provenance_only"
        for record in manifest["upstream"]["inputs"]["public_assignments"].values()
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
        "sample_id",
        "frozen_inferred_condition_label",
        "public_assignment",
        "public_assignment_match_status",
        "public_assignment_source_accession",
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
    assert cell_source["public_assignment"].value_counts().to_dict() == {
        "unmatched": figure2.EXPECTED_DATASET_SHAPE[0] - 4,
        "control": 2,
        "gata307mut": 2,
    }
    assert set(cell_source["frozen_inferred_condition_label"]) == {"Disease"}
    assert {
        "HBB_smoothed_scaled",
        "MKI67_smoothed_scaled",
    }.issubset(marker_profiles.columns)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda value: value.update(status="incomplete"), "complete table run"),
        (
            lambda value: value.update(artifact_type="legacy-receipt"),
            "artifact type",
        ),
        (
            lambda value: value["runtime"].update(algorithm_revision="legacy"),
            "installed runtime",
        ),
        (
            lambda value: value["parameters"].update(window_size=400),
            "window_size",
        ),
        (
            lambda value: value["dataset"].update(shape=[3575, 3000]),
            "dataset shape",
        ),
    ],
)
def test_run_summary_and_parameter_failures(change: object, message: str) -> None:
    summary = copy.deepcopy(_run_summary())
    change(summary)
    with pytest.raises(figure2.Figure2AssemblyError, match=message):
        figure2._validate_run_summary(summary)


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
    nonfinite_pseudotime.obs.loc[
        nonfinite_pseudotime.obs_names[0], "dpt_pseudotime"
    ] = np.nan
    with pytest.raises(figure2.Figure2AssemblyError, match="must be finite"):
        figure2._validate_adata(nonfinite_pseudotime)

    nonfinite_marker = _adata()
    nonfinite_marker.X[0, 0] = np.nan
    with pytest.raises(figure2.Figure2AssemblyError, match="marker CD34"):
        figure2._validate_adata(nonfinite_marker)

    with_raw = _adata()
    with_raw.raw = with_raw
    with pytest.raises(figure2.Figure2AssemblyError, match="no adata.raw"):
        figure2._validate_adata(with_raw)

    with_layer = _adata()
    with_layer.layers["counts"] = with_layer.X.copy()
    with pytest.raises(figure2.Figure2AssemblyError, match="no adata.raw"):
        figure2._validate_adata(with_layer)


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
    run_dir, dataset, gene_sets, assignment_dir = _write_functional_run(tmp_path)
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
        figure2.assemble_figure2(
            run_dir,
            dataset,
            gene_sets,
            assignment_dir,
            output,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".failed-output-*/"))
