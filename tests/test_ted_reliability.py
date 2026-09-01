import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import anndata as ad
import pyfgsea
import pyfgsea.trajectory as traj
from pyfgsea.trajectory import (
    _axis_sum,
    _axis_sum_squares,
    _detection_count,
    _make_windows,
    _rank_gene_scores,
)


def _adata(X, genes=None, pseudotime=None):
    genes = genes or [f"G{i}" for i in range(X.shape[1])]
    pseudotime = np.linspace(0, 1, X.shape[0]) if pseudotime is None else pseudotime
    return ad.AnnData(
        X=X,
        obs=pd.DataFrame({"dpt_pseudotime": pseudotime}),
        var=pd.DataFrame(index=genes),
    )


def test_validate_inputs_reports_duplicate_missing_and_nan():
    a = _adata(
        np.ones((5, 3)),
        genes=["G0", "G0", "G2"],
        pseudotime=[0.0, 0.2, np.nan, 0.8, 1.0],
    )
    gmt = {"TooSmall": ["NOPE"], "Valid": ["G0", "G2"]}

    report = pyfgsea.validate_inputs(
        a,
        gmt,
        pseudotime_key="dpt_pseudotime",
        min_size=1,
        max_size=10,
        window_size=10,
    )
    checks = set(report.loc[report["level"].isin(["warning", "error"]), "check"])
    assert "gene_names_unique" in checks
    assert "pseudotime_finite" in checks
    assert "window_size" in checks

    missing = pyfgsea.validate_inputs(a, gmt, pseudotime_key="missing_pt")
    assert "pseudotime_key" in set(missing.loc[missing["level"] == "error", "check"])


def test_run_trajectory_dropna_and_duplicate_gene_policy():
    traj.HAS_SCANPY = True
    a = _adata(
        np.random.default_rng(0).normal(size=(30, 4)) + 2,
        genes=["G0", "G0", "G2", "G3"],
        pseudotime=np.linspace(0, 1, 30),
    )
    gmt = {"P": ["G0", "G2"]}

    with pytest.raises(ValueError, match="duplicated"):
        pyfgsea.run_trajectory_gsea(
            a,
            gmt,
            window_size=10,
            step=10,
            min_size=1,
            max_size=10,
            nperm_nes=10,
        )

    res = pyfgsea.run_trajectory_gsea(
        a,
        gmt,
        window_size=10,
        step=10,
        min_size=1,
        max_size=10,
        nperm_nes=10,
        make_var_names_unique=True,
    )
    assert not res.empty

    b = _adata(np.random.default_rng(1).normal(size=(30, 4)) + 2)
    b.obs.loc[b.obs.index[0], "dpt_pseudotime"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        pyfgsea.run_trajectory_gsea(
            b,
            {"P": ["G0", "G1"]},
            window_size=10,
            step=10,
            min_size=1,
            max_size=10,
            nperm_nes=10,
            dropna=False,
        )


def test_layer_source_is_used_for_trajectory_scores():
    traj.HAS_SCANPY = True
    X = np.zeros((40, 4))
    a = _adata(X)
    layer = np.zeros_like(X)
    layer[:20, 0] = 4.0
    layer[20:, 1] = 4.0
    a.layers["log1p"] = layer

    res = pyfgsea.run_trajectory_gsea(
        a,
        {"Early": ["G0"], "Late": ["G1"]},
        window_size=20,
        step=20,
        min_size=1,
        max_size=10,
        nperm_nes=10,
        layer="log1p",
        calculate_nes=False,
    )
    assert set(res["Pathway"]) == {"Early", "Late"}
    assert res.attrs["trajectory_params"]["expression_source"] == "layer:log1p"


def test_run_trajectory_uses_cell_weight_key():
    traj.HAS_SCANPY = True
    X = np.zeros((40, 4))
    X[:20, 0] = 5.0
    X[20:, 1] = 5.0
    a = _adata(X)
    a.obs["fate_weight"] = np.concatenate([np.ones(20), np.full(20, 0.25)])

    report = pyfgsea.validate_inputs(
        a,
        {"Early": ["G0"], "Late": ["G1"]},
        pseudotime_key="dpt_pseudotime",
        cell_weight_key="fate_weight",
        min_size=1,
        max_size=10,
        window_size=20,
    )
    assert "error" not in set(report["level"])

    res = pyfgsea.run_trajectory_gsea(
        a,
        {"Early": ["G0"], "Late": ["G1"]},
        window_size=20,
        step=20,
        min_size=1,
        max_size=10,
        nperm_nes=10,
        calculate_nes=False,
        cell_weight_key="fate_weight",
    )
    assert "weight_sum" in res.columns
    assert res.attrs["trajectory_params"]["cell_weight_key"] == "fate_weight"
    assert res.groupby("window_id")["weight_sum"].first().iloc[0] == 20.0


def test_run_trajectory_supports_smooth_slope_and_split_signed_sets():
    traj.HAS_SCANPY = True
    X = np.zeros((40, 4))
    X[:, 0] = np.linspace(0, 5, 40)
    X[:, 1] = np.linspace(5, 0, 40)
    X[:, 2] = 1.0
    X[:, 3] = np.linspace(0, 2, 40)
    a = _adata(X)
    gene_sets = {"SignedProgram": {"G0": 1.0, "G1": -1.0, "G2": 0.1}}

    res = pyfgsea.run_trajectory_gsea(
        a,
        gene_sets,
        ranker="gam_slope",
        gene_set_mode="split_signed",
        min_abs_gene_weight=0.5,
        smooth_slope_bandwidth=0.3,
        window_size=20,
        step=20,
        min_size=1,
        max_size=10,
        nperm_nes=10,
        sample_size=3,
        calculate_nes=False,
    )

    assert set(res["Pathway"]) == {
        "SignedProgram__positive",
        "SignedProgram__negative",
    }
    assert res.attrs["trajectory_params"]["ranker"] == "smooth_slope"
    assert res.attrs["trajectory_params"]["gene_set_mode"] == "split_signed"


def test_sparse_dense_rank_scores_are_consistent():
    X_dense = np.array(
        [
            [0.0, 1.0, 4.0],
            [0.0, 1.0, 3.0],
            [2.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )
    X_sparse = sparse.csr_matrix(X_dense)
    window = np.array([2, 3])

    for ranker in ["mean_diff", "t_stat", "cohens_d", "detection_weighted"]:
        kwargs_dense = {
            "sum_total": _axis_sum(X_dense),
            "n_all": X_dense.shape[0],
            "sum_sq_total": _axis_sum_squares(X_dense),
            "det_total": _detection_count(X_dense),
        }
        kwargs_sparse = {
            "sum_total": _axis_sum(X_sparse),
            "n_all": X_sparse.shape[0],
            "sum_sq_total": _axis_sum_squares(X_sparse),
            "det_total": _detection_count(X_sparse),
        }
        dense_scores = _rank_gene_scores(X_dense, window, ranker, **kwargs_dense)
        sparse_scores = _rank_gene_scores(X_sparse, window, ranker, **kwargs_sparse)
        np.testing.assert_allclose(dense_scores, sparse_scores)


def test_validate_trajectory_result_checks_events_and_leading_edge():
    result = pd.DataFrame(
        {
            "Pathway": ["P", "P", "P"],
            "window_id": [0, 1, 2],
            "pt_mid": [0.0, 0.5, 1.0],
            "pt_start": [0.0, 0.45, 0.95],
            "pt_end": [0.05, 0.55, 1.0],
            "ES": [0.1, 0.9, 0.2],
            "NES": [0.1, 2.0, 0.2],
            "P-value": [0.5, 0.001, 0.4],
            "padj": [0.5, 0.003, 0.4],
            "leading_edge": ["G0", "G0;G1", "G1"],
        }
    )
    events = pyfgsea.summarize_events(result, min_consecutive=1)
    ledge = pyfgsea.leading_edge_dynamics(result)
    report = pyfgsea.validate_trajectory_result(
        result,
        events=events,
        leading_edge=ledge,
        gmt_path={"P": ["G0", "G1"]},
    )
    assert "error" not in set(report["level"])


def test_pseudotime_reversal_maps_peak_time():
    base = pd.DataFrame(
        {
            "Pathway": ["P"] * 5,
            "pt_mid": [0.0, 0.25, 0.5, 0.75, 1.0],
            "pt_start": [0.0, 0.2, 0.45, 0.7, 0.95],
            "pt_end": [0.05, 0.3, 0.55, 0.8, 1.0],
            "NES": [0.0, 2.5, 1.0, 0.2, 0.0],
            "padj": [1.0, 0.001, 0.01, 1.0, 1.0],
        }
    )
    original = pyfgsea.summarize_events(base, min_consecutive=1)
    reversed_table = base.copy()
    reversed_table["pt_mid"] = 1 - reversed_table["pt_mid"]
    reversed_table["pt_start"] = 1 - base["pt_end"]
    reversed_table["pt_end"] = 1 - base["pt_start"]
    reversed_events = pyfgsea.summarize_events(reversed_table, min_consecutive=1)
    assert np.isclose(reversed_events.loc[0, "peak_time"], 1 - original.loc[0, "peak_time"])


def test_constant_expression_rankers_do_not_emit_inf_or_nan():
    X = np.ones((12, 5))
    window = np.arange(4)
    sum_total = _axis_sum(X)
    sum_sq_total = _axis_sum_squares(X)
    det_total = _detection_count(X)
    for ranker in ["mean_diff", "t_stat", "cohens_d", "detection_weighted", "local_slope"]:
        scores = _rank_gene_scores(
            X,
            window,
            ranker,
            sum_total=sum_total,
            n_all=X.shape[0],
            sum_sq_total=sum_sq_total,
            det_total=det_total,
            pt=np.linspace(0, 1, X.shape[0]),
        )
        assert np.isfinite(scores).all()
        assert np.allclose(scores, 0)


def test_window_invariants_for_adaptive_and_pseudotime_span():
    order = np.arange(50)
    pt = np.linspace(0, 1, 50)
    adaptive = _make_windows(
        order,
        window_size=10,
        step=5,
        pt=pt,
        window_mode="adaptive",
        min_cells=6,
        max_cells=12,
        target_span=0.1,
        span_step=0.1,
    )
    assert all(6 <= len(win) <= 12 for _, _, win in adaptive)

    span = _make_windows(
        order,
        window_size=10,
        step=5,
        pt=pt,
        window_mode="pseudotime_span",
        min_cells=3,
        max_cells=15,
        target_span=0.2,
        span_step=0.2,
    )
    assert all((pt[win].max() - pt[win].min()) <= 0.25 for _, _, win in span)


def test_permuted_condition_labels_remove_peak_shift_signal():
    events = pd.DataFrame(
        {
            "Pathway": ["P", "P"],
            "condition": ["control", "case"],
            "peak_time": [0.5, 0.5],
            "peak_NES": [2.0, 2.0],
            "trough_time": [0.0, 0.0],
            "trough_NES": [0.0, 0.0],
            "duration": [0.2, 0.2],
            "AUC": [1.0, 1.0],
            "window_fdr_min": [0.001, 0.001],
            "event_label": ["mid sustained activation", "mid sustained activation"],
        }
    )
    comparison = pyfgsea.compare_event_tables(
        events,
        group_col="condition",
        reference="control",
        query="case",
    )
    assert comparison.loc[0, "program_type"] == "shared_program"


def test_synthetic_truth_benchmark_smoke():
    traj.HAS_SCANPY = True
    bench = pyfgsea.run_synthetic_truth_benchmark(
        truth_types=["monotonic_up"],
        rankers=["mean_diff"],
        n_cells=80,
        n_genes=40,
        window_size=20,
        step=20,
        nperm_nes=10,
        sample_size=21,
        seed=7,
    )
    assert list(bench["truth_type"]) == ["monotonic_up"]
    assert list(bench["ranker"]) == ["mean_diff"]
    assert {"detected", "peak_time_error", "event_label_accuracy"}.issubset(bench.columns)
