import numpy as np
import pandas as pd

from pyfgsea.trajectory import (
    _axis_sum,
    _axis_sum_squares,
    _detection_count,
    _make_windows,
    _normalize_ranker,
    _prepare_gene_sets_for_mode,
    _rank_gene_scores,
)
from pyfgsea.trajectory_events import summarize_events
from pyfgsea.leading_edge import compute_leading_edges, leading_edge_dynamics
from pyfgsea.trajectory_grid import summarize_event_consensus
from pyfgsea.trajectory_compare import compare_event_tables


def test_ranker_aliases_and_basic_scores():
    X = np.array(
        [
            [0.0, 1.0, 4.0],
            [0.0, 1.0, 3.0],
            [2.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )
    window = np.array([2, 3])
    sum_total = _axis_sum(X)
    sum_sq_total = _axis_sum_squares(X)
    det_total = _detection_count(X)

    assert _normalize_ranker("logfc") == "mean_diff"
    mean_diff = _rank_gene_scores(X, window, "mean_diff", sum_total, X.shape[0])
    assert mean_diff[0] > 0
    assert mean_diff[2] < 0

    t_stat = _rank_gene_scores(
        X,
        window,
        "t_stat",
        sum_total,
        X.shape[0],
        sum_sq_total=sum_sq_total,
    )
    assert np.isfinite(t_stat).all()
    assert t_stat[0] > 0

    weighted = _rank_gene_scores(
        X,
        window,
        "detection_weighted",
        sum_total,
        X.shape[0],
        det_total=det_total,
    )
    assert np.isfinite(weighted).all()
    assert weighted[0] > 0


def test_local_slope_and_neighbor_contrast_rankers():
    X = np.array(
        [
            [0.0, 3.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 0.0],
        ]
    )
    pt = np.array([0.0, 0.33, 0.66, 1.0])
    window = np.array([0, 1, 2, 3])
    sum_total = _axis_sum(X)

    slope = _rank_gene_scores(
        X, window, "local_slope", sum_total, X.shape[0], pt=pt
    )
    assert slope[0] > 0
    assert slope[1] < 0

    neighbor = _rank_gene_scores(
        X,
        np.array([1, 2]),
        "neighbor_contrast",
        sum_total,
        X.shape[0],
        neighbor_indices=np.array([0, 3]),
    )
    assert np.isfinite(neighbor).all()


def test_smooth_slope_ranker_alias_and_direction():
    X = np.array(
        [
            [0.0, 3.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 0.0],
        ]
    )
    pt = np.array([0.0, 0.33, 0.66, 1.0])
    scores = _rank_gene_scores(
        X,
        np.array([1, 2]),
        _normalize_ranker("gam_slope"),
        _axis_sum(X),
        X.shape[0],
        pt=pt,
        smooth_center=0.5,
        smooth_bandwidth=0.5,
    )

    assert scores[0] > 0
    assert scores[1] < 0


def test_cell_weights_affect_window_rank_scores():
    X = np.array(
        [
            [10.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 10.0],
        ]
    )
    window = np.array([0, 1])
    weights = np.array([0.0, 10.0, 10.0, 0.0])
    unweighted = _rank_gene_scores(
        X,
        window,
        "mean_diff",
        _axis_sum(X),
        X.shape[0],
    )
    weighted = _rank_gene_scores(
        X,
        window,
        "mean_diff",
        sum_total=np.asarray(X.T @ weights).ravel(),
        n_all=X.shape[0],
        weights=weights,
        weight_total=float(weights.sum()),
    )

    assert unweighted[0] > 0
    assert unweighted[1] < 0
    np.testing.assert_allclose(weighted, np.zeros(2))


def test_split_signed_gene_sets_expand_positive_and_negative_arms():
    gene_sets = {
        "RegulonA": {"G0": 1.0, "G1": -1.0, "G2": 0.2},
        "Unsigned": ["G0", "G1"],
    }
    expanded = _prepare_gene_sets_for_mode(
        gene_sets,
        gene_set_mode="split_signed",
        min_abs_gene_weight=0.5,
    )

    assert expanded["RegulonA__positive"] == ["G0"]
    assert expanded["RegulonA__negative"] == ["G1"]
    assert expanded["Unsigned__positive"] == ["G0", "G1"]


def test_pseudotime_span_and_adaptive_windows():
    order = np.arange(10)
    pt = np.linspace(0.0, 1.0, 10)

    span_windows = _make_windows(
        order,
        window_size=4,
        step=2,
        pt=pt,
        window_mode="pseudotime_span",
        target_span=0.25,
        span_step=0.25,
    )
    assert span_windows
    assert all(len(win) >= 1 for _, _, win in span_windows)

    adaptive_windows = _make_windows(
        order,
        window_size=4,
        step=2,
        pt=pt,
        window_mode="adaptive",
        min_cells=3,
        max_cells=4,
        target_span=0.05,
        span_step=0.2,
    )
    assert adaptive_windows
    assert all(3 <= len(win) <= 4 for _, _, win in adaptive_windows)


def test_summarize_events_labels_transient_activation():
    result = pd.DataFrame(
        {
            "Pathway": ["E2F_TARGETS"] * 5,
            "pt_mid": [0.0, 0.25, 0.5, 0.75, 1.0],
            "pt_start": [0.0, 0.2, 0.45, 0.7, 0.95],
            "pt_end": [0.05, 0.3, 0.55, 0.8, 1.0],
            "NES": [0.0, 0.7, 2.0, 0.7, 0.1],
            "padj": [1.0, 0.01, 0.001, 0.01, 1.0],
        }
    )

    summary = summarize_events(result, min_consecutive=2)
    assert list(summary["Pathway"]) == ["E2F_TARGETS"]
    row = summary.iloc[0]
    assert row["activation_onset"] == 0.25
    assert row["peak_time"] == 0.5
    assert row["window_fdr_min"] == 0.001
    assert "transient activation" in row["event_label"]
    assert row["event_confidence_class"] == "multi_window_event"


def test_summarize_events_marks_single_window_pulse():
    result = pd.DataFrame(
        {
            "Pathway": ["Pulse"] * 3,
            "pt_mid": [0.0, 0.5, 1.0],
            "pt_start": [0.0, 0.5, 1.0],
            "pt_end": [0.0, 0.5, 1.0],
            "NES": [0.0, 3.0, 0.0],
            "padj": [1.0, 0.001, 1.0],
        }
    )

    row = summarize_events(result, min_consecutive=1).iloc[0]
    assert row["duration"] == 0.0
    assert row["event_confidence_class"] == "single_window_pulse"


def test_leading_edge_computation_and_dynamics():
    genes = np.array(["G0", "G1", "G2", "G3", "G4"])
    scores = np.array([5.0, 4.0, 3.0, 1.0, 0.0])
    leading = compute_leading_edges(
        genes,
        scores,
        pathway_names=["Pathway_A"],
        pathway_indices=[[0, 1, 3]],
    )
    assert leading["Pathway_A"] == "G0;G1"

    result = pd.DataFrame(
        {
            "Pathway": ["Pathway_A", "Pathway_A"],
            "window_id": [0, 1],
            "pt_mid": [0.2, 0.4],
            "NES": [2.0, 2.5],
            "leading_edge": ["G0;G1", "G1;G3"],
        }
    )
    dyn = leading_edge_dynamics(result, core_fraction=1.0)
    assert dyn.loc[0, "core_genes"] == "G1"
    assert np.isclose(dyn.loc[1, "turnover_score"], 2 / 3)


def test_event_consensus_recommendations():
    events = pd.DataFrame(
        {
            "Pathway": ["A", "A", "A", "B", "B", "B"],
            "grid_run": [1, 2, 3, 1, 2, 3],
            "peak_time": [0.50, 0.52, 0.48, 0.2, 0.8, 0.3],
            "peak_NES": [3.0, 2.8, 3.1, 2.0, -2.0, 0.5],
            "duration": [0.10, 0.12, 0.09, 0.2, 0.2, 0.2],
            "window_fdr_min": [0.001, 0.01, 0.02, 0.5, 0.001, 0.9],
        }
    )
    consensus = summarize_event_consensus(events)
    recs = dict(zip(consensus["Pathway"], consensus["recommendation"]))
    assert recs["A"] == "robust transient"
    assert recs["B"] == "unstable"


def test_multi_ranker_event_consensus_columns():
    import pyfgsea
    import pyfgsea.trajectory as traj

    traj.HAS_SCANPY = True
    adata, gene_sets, _truth = pyfgsea.make_synthetic_trajectory_truth(
        "monotonic_up",
        n_cells=80,
        n_genes=40,
        pathway_size=8,
        seed=19,
    )
    out = pyfgsea.run_trajectory_gsea_grid(
        adata,
        gene_sets,
        window_sizes=[20],
        step_sizes=[20],
        rankers=["mean_diff", "local_slope"],
        seeds=[1, 2],
        min_size=5,
        max_size=100,
        nperm_nes=8,
        sample_size=8,
        event_kwargs={"min_consecutive": 1},
    )

    assert out["results"]["grid_run"].nunique() == 4
    assert set(out["events"]["grid_ranker"]) == {"mean_diff", "local_slope"}
    assert {
        "ranker_support",
        "seed_support",
        "dominant_event_label",
        "event_label_consistency",
    }.issubset(out["consensus"].columns)
    assert int(out["consensus"]["n_runs"].iloc[0]) == 4


def test_run_ranker_consensus_wrapper_returns_event_agreement():
    import pyfgsea
    import pyfgsea.trajectory as traj

    traj.HAS_SCANPY = True
    adata, gene_sets, _truth = pyfgsea.make_synthetic_trajectory_truth(
        "monotonic_up",
        n_cells=80,
        n_genes=40,
        pathway_size=8,
        seed=31,
    )
    consensus = pyfgsea.run_ranker_consensus(
        adata,
        gene_sets,
        pseudotime_key="dpt_pseudotime",
        rankers=["mean_diff", "local_slope"],
        window_size=20,
        step=20,
        min_size=5,
        max_size=100,
        nperm_nes=8,
        sample_size=8,
        event_kwargs={"min_consecutive": 1},
    )

    assert {"consensus_label", "ranker_agreement", "recommendation"}.issubset(
        consensus.columns
    )
    assert "events" in consensus.attrs


def test_bootstrap_trajectory_gsea_smoke():
    import pyfgsea
    import pyfgsea.trajectory as traj

    traj.HAS_SCANPY = True
    adata, gene_sets, _truth = pyfgsea.make_synthetic_trajectory_truth(
        "monotonic_up",
        n_cells=60,
        n_genes=40,
        pathway_size=8,
        seed=32,
    )
    bands = pyfgsea.bootstrap_trajectory_gsea(
        adata,
        gene_sets,
        pseudotime_key="dpt_pseudotime",
        n_boot=2,
        seed=32,
        window_size=20,
        step=20,
        min_size=5,
        max_size=100,
        nperm_nes=8,
        sample_size=8,
        event_kwargs={"min_consecutive": 1},
    )

    assert {"NES_mean", "NES_lower", "NES_upper", "n_boot"}.issubset(bands.columns)
    assert "boot_results" in bands.attrs


def test_branch_contrast_gsea_smoke():
    import pyfgsea
    import pyfgsea.trajectory as traj

    traj.HAS_SCANPY = True
    adata, gene_sets, _truth = pyfgsea.make_synthetic_trajectory_truth(
        "branch_specific_activation",
        n_cells=100,
        n_genes=40,
        pathway_size=8,
        seed=23,
    )
    out = pyfgsea.run_branch_gsea(
        adata,
        gene_sets,
        branch_key="branch",
        branches=["branch_a", "branch_b"],
        ranker="branch_contrast",
        pseudotime_key="dpt_pseudotime",
        window_size=20,
        step=20,
        min_reference_cells=10,
        min_size=5,
        max_size=100,
        nperm_nes=8,
        sample_size=8,
        event_kwargs={"min_consecutive": 1},
    )

    assert not out["results"].empty
    assert set(out["results"]["ranker"]) == {"branch_contrast"}
    assert "n_reference_cells" in out["results"].columns
    assert "ranker" in out["comparisons"].columns or out["comparisons"].empty


def test_compare_event_tables_marks_peak_shift():
    events = pd.DataFrame(
        {
            "Pathway": ["E2F_TARGETS", "E2F_TARGETS"],
            "condition": ["control", "case"],
            "peak_time": [0.34, 0.28],
            "peak_NES": [4.0, 4.3],
            "trough_time": [0.8, 0.8],
            "trough_NES": [-0.2, -0.1],
            "duration": [0.09, 0.10],
            "AUC": [1.0, 2.2],
            "window_fdr_min": [0.001, 0.001],
            "event_label": ["mid transient activation", "early transient activation"],
        }
    )
    comparison = compare_event_tables(
        events,
        group_col="condition",
        reference="control",
        query="case",
        reference_label="control",
        query_label="case",
    )
    row = comparison.iloc[0]
    assert row["delta_peak_time"] == -0.06
    assert row["interpretation"] == "earlier activation"
    assert row["program_type"] == "divergence_program"


def test_aligned_trajectory_contrast_separates_speed_from_rewiring():
    import pyfgsea

    times = np.linspace(0.0, 1.0, 31)
    rows = []

    def pulse(x, center, width=0.08, amplitude=2.5):
        return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)

    for condition in ("control", "case"):
        for t in times:
            bio_t = t if condition == "control" else t**0.75
            profiles = {
                "STAGE_ANCHOR": pulse(bio_t, 0.35),
                "STAGE_ANCHOR_LATE": pulse(bio_t, 0.75),
                "SPEED_ONLY": pulse(bio_t, 0.68),
                "AMPLITUDE_DRIVER": pulse(bio_t, 0.45, amplitude=3.0)
                if condition == "control"
                else pulse(bio_t, 0.45, amplitude=1.0),
                "BACKGROUND": 0.15 * np.sin(2 * np.pi * bio_t),
            }
            for pathway, nes in profiles.items():
                rows.append(
                    {
                        "condition": condition,
                        "Pathway": pathway,
                        "pt_mid": t,
                        "NES": nes,
                        "padj": 0.001,
                    }
                )

    tables = pyfgsea.run_aligned_trajectory_contrast(
        pd.DataFrame(rows),
        condition_col="condition",
        condition_a="control",
        condition_b="case",
        anchor_pathways=["STAGE_ANCHOR", "STAGE_ANCHOR_LATE"],
        contrast_threshold=0.45,
        min_consecutive=1,
        n_permutations=5,
        seed=7,
    )

    assert {
        "trajectory_alignment_functions",
        "alignment_anchor_pathways",
        "aligned_pathway_score_process",
        "differential_event_table",
        "differential_event_fdr",
        "alignment_sensitivity_report",
    }.issubset(tables)
    metrics = (
        tables["differential_event_table"]
        .drop_duplicates("pathway")
        .set_index("pathway")
    )
    assert "AMPLITUDE_DRIVER" in metrics.index
    assert metrics.loc["AMPLITUDE_DRIVER", "effect_type"] == "amplitude_rewiring"

    aligned = tables["aligned_pathway_score_process"]
    speed = aligned[aligned["pathway"] == "SPEED_ONLY"]
    integrate = getattr(np, "trapezoid", None) or np.trapz
    raw_auc = integrate(speed["raw_D_A_minus_B"], x=speed["state_time"])
    aligned_auc = integrate(speed["D_A_minus_B"], x=speed["state_time"])
    assert abs(aligned_auc) < abs(raw_auc)
    assert "event_q" in tables["differential_event_fdr"].columns


def test_fate_predictive_events_report_prebranch_auc_and_drivers():
    import pyfgsea

    cell_pt = np.linspace(0.0, 1.0, 120)
    pre_signal = np.exp(-0.5 * ((cell_pt - 0.28) / 0.08) ** 2)
    future_fate = np.where(pre_signal > 0.55, "erythroid", "myeloid")
    obs = pd.DataFrame(
        {
            "dpt_pseudotime": cell_pt,
            "future_fate": future_fate,
        }
    )

    times = np.linspace(0.0, 1.0, 25)
    rows = []
    for t in times:
        profiles = {
            "ERY_PRE_EVENT": np.exp(-0.5 * ((t - 0.28) / 0.08) ** 2),
            "MY_PRE_EVENT": np.exp(-0.5 * ((t - 0.08) / 0.08) ** 2),
            "BACKGROUND": 0.05 * np.sin(2 * np.pi * t),
        }
        for pathway, score in profiles.items():
            rows.append(
                {
                    "Pathway": pathway,
                    "pt_mid": t,
                    "NES": score,
                    "leading_edge": "Gata1;Klf1" if pathway == "ERY_PRE_EVENT" else "Spi1",
                }
            )

    tables = pyfgsea.run_fate_predictive_events(
        pd.DataFrame(rows),
        obs=obs,
        pseudotime_key="dpt_pseudotime",
        fate_key="future_fate",
        split_time=0.45,
        event_threshold=0.25,
        n_permutations=5,
        seed=11,
        min_prebranch_cells=20,
    )

    assert {
        "prebranch_fate_predictive_events",
        "fate_prediction_model_performance",
        "prebranch_event_fdr",
        "fate_predictive_driver_genes",
        "fate_predictive_leading_edge",
    }.issubset(tables)
    events = tables["prebranch_fate_predictive_events"]
    row = events[
        (events["pathway"] == "ERY_PRE_EVENT")
        & (events["future_fate"] == "erythroid")
    ].iloc[0]
    assert row["cross_validated_AUC"] > 0.8
    assert row["FPES"] > 0
    assert "theta" in events.columns
    assert "macro_F1" in events.columns
    assert row["prebranch_fraction"] > 0.5
    assert "Gata1" in row["driver_genes"]
    assert {"driver_score", "pathway_or_module"}.issubset(tables["fate_predictive_driver_genes"].columns)
    assert "event_q" in tables["prebranch_event_fdr"].columns


def test_dynamic_gene_modules_discover_module_events_and_annotations():
    import anndata as ad
    import pyfgsea

    rng = np.random.default_rng(51)
    pt = np.linspace(0.0, 1.0, 100)
    genes = [f"G{i}" for i in range(30)]
    early = np.exp(-0.5 * ((pt - 0.25) / 0.12) ** 2)
    late = np.exp(-0.5 * ((pt - 0.75) / 0.12) ** 2)
    X = rng.normal(0.1, 0.03, size=(len(pt), len(genes)))
    X[:, :8] += early[:, None] * 3.0
    X[:, 8:16] += late[:, None] * 2.5
    X = np.maximum(X, 0.0)
    adata = ad.AnnData(X)
    adata.var_names = genes
    adata.obs["dpt_pseudotime"] = pt
    gene_sets = {
        "EARLY_PROGRAM": genes[:8],
        "LATE_PROGRAM": genes[8:16],
        "BACKGROUND": genes[16:24],
    }

    tables = pyfgsea.discover_dynamic_gene_modules(
        adata,
        gmt_path=gene_sets,
        pseudotime_key="dpt_pseudotime",
        n_modules=2,
        window_size=20,
        step=10,
        top_variable_genes=24,
        top_genes=8,
        event_threshold=0.5,
        min_consecutive=1,
        n_permutations=2,
        seed=51,
        max_iter=300,
    )

    assert {
        "dynamic_gene_modules",
        "module_time_profiles",
        "module_event_table",
        "module_event_fdr",
        "module_pathway_annotation",
        "module_driver_score",
        "module_leading_edge_drivers",
    }.issubset(tables)
    assert tables["dynamic_gene_modules"]["module"].nunique() == 2
    assert not tables["module_event_table"].empty
    assert {"event_q", "event_id"}.issubset(tables["module_event_fdr"].columns)
    assert {"driver_score", "event_id"}.issubset(tables["module_driver_score"].columns)
    assert {"EARLY_PROGRAM", "LATE_PROGRAM"} & set(
        tables["module_pathway_annotation"]["pathway"]
    )


def test_biological_discovery_score_penalizes_generic_events():
    import pyfgsea

    events = pd.DataFrame(
        {
            "event_id": ["generic", "prime"],
            "pathway": ["MOUSE_CELL_CYCLE", "ERYTHROID_PRIME_DRIVER"],
            "event_q": [0.001, 0.01],
            "cross_validated_AUC": [0.7, 0.9],
            "prebranch_fraction": [0.5, 0.8],
            "stability": [0.8, 0.8],
        }
    )
    drivers = pd.DataFrame(
        {
            "event_id": ["generic", "prime"],
            "gene": ["Mki67", "Gata1"],
            "driver_score": [1.0, 1.0],
        }
    )

    scored = pyfgsea.score_biological_discovery(events, driver_scores=drivers)

    table = scored.set_index("event_id")
    assert table.loc["prime", "N_novelty"] > table.loc["generic", "N_novelty"]
    assert table.loc["prime", "biological_discovery_score"] > table.loc["generic", "biological_discovery_score"]


def test_event_graph_reports_reversed_edges():
    import pyfgsea

    events = pd.DataFrame(
        {
            "condition": ["basal", "basal", "fetal", "fetal"],
            "pathway": ["A", "B", "A", "B"],
            "event_id": ["basal_A", "basal_B", "fetal_A", "fetal_B"],
            "peak_time": [0.2, 0.7, 0.8, 0.3],
        }
    )
    boot = pd.DataFrame(
        {
            "condition": ["basal", "basal", "fetal", "fetal"] * 3,
            "pathway": ["A", "B", "A", "B"] * 3,
            "event_id": ["basal_A", "basal_B", "fetal_A", "fetal_B"] * 3,
            "peak_time": [0.2, 0.7, 0.8, 0.3, 0.25, 0.75, 0.78, 0.32, 0.22, 0.74, 0.81, 0.31],
            "boot_id": np.repeat([0, 1, 2], 4),
        }
    )

    graph = pyfgsea.build_event_graph(
        events,
        bootstrap_events=boot,
        condition_col="condition",
        reference="basal",
        query="fetal",
        order_probability_threshold=0.9,
    )

    assert {
        "event_order_probability_matrix",
        "condition_event_graph_edges",
        "event_graph_rewiring",
        "event_graph_bootstrap_support",
    }.issubset(graph)
    rewiring = graph["event_graph_rewiring"]
    assert "edge_reversed" in set(rewiring["rewiring_type"])


def test_event_driver_score_prioritizes_specific_regulatory_driver():
    import pyfgsea

    events = pd.DataFrame(
        {
            "dataset": ["toy", "toy"],
            "condition": ["fetal", "basal"],
            "pathway": ["ERY_EVENT", "OTHER_EVENT"],
            "event_id": ["ery_1", "other_1"],
            "peak_NES": [3.0, 2.0],
            "event_q": [0.01, 0.2],
        }
    )
    leading = pd.DataFrame(
        {
            "event_id": ["ery_1", "ery_1", "other_1"],
            "pathway": ["ERY_EVENT", "ERY_EVENT", "OTHER_EVENT"],
            "gene": ["Gata1", "Alas2", "Alas2"],
            "bootstrap_probability": [0.9, 0.8, 0.7],
        }
    )

    tables = pyfgsea.score_event_drivers(
        events,
        leading,
        regulator_targets={"GATA1": {"ALAS2"}},
        top_genes_per_event=None,
    )

    assert {
        "event_driver_score",
        "event_regulator_activity",
        "event_driver_network",
        "driver_specificity_report",
    }.issubset(tables)
    drivers = tables["event_driver_score"]
    gata1 = drivers[(drivers["event_id"] == "ery_1") & (drivers["gene"] == "Gata1")].iloc[0]
    shared = drivers[(drivers["event_id"] == "ery_1") & (drivers["gene"] == "Alas2")].iloc[0]
    assert gata1["driver_score"] > 0
    assert gata1["specificity"] > shared["specificity"]
    assert "GATA1" in set(tables["event_regulator_activity"]["regulator"])


def test_cross_dataset_event_replication_matches_events_and_scores_meta():
    import pyfgsea

    events = pd.DataFrame(
        {
            "dataset": ["d1", "d2", "d2"],
            "pathway": ["ERY_EVENT", "ERY_EVENT_ALT", "MY_EVENT"],
            "event_id": ["d1_ery", "d2_ery", "d2_my"],
            "activation_onset": [0.2, 0.22, 0.75],
            "duration": [0.2, 0.22, 0.1],
            "peak_time": [0.3, 0.32, 0.8],
            "peak_NES": [2.5, 2.2, -2.0],
            "event_q": [0.01, 0.02, 0.5],
        }
    )
    gene_sets = {
        "ERY_EVENT": {"GATA1", "ALAS2", "KLF1"},
        "ERY_EVENT_ALT": {"GATA1", "ALAS2", "HBB"},
        "MY_EVENT": {"SPI1", "MPO"},
    }
    drivers = pd.DataFrame(
        {
            "event_id": ["d1_ery", "d1_ery", "d2_ery", "d2_my"],
            "gene": ["Gata1", "Alas2", "Gata1", "Spi1"],
        }
    )

    tables = pyfgsea.match_cross_dataset_events(
        events,
        driver_scores=drivers,
        gene_sets=gene_sets,
        match_threshold=0.45,
    )

    assert {
        "event_match_matrix",
        "cross_dataset_event_replication",
        "meta_event_score",
        "dataset_event_coverage",
    }.issubset(tables)
    replicated = tables["cross_dataset_event_replication"]
    assert ((replicated["event_id_1"] == "d1_ery") & (replicated["event_id_2"] == "d2_ery")).any()
    meta = tables["meta_event_score"].set_index("event_id")
    assert meta.loc["d1_ery", "replicated_dataset_count"] == 1
    assert meta.loc["d1_ery", "meta_event_score"] > 2.0


def test_phenotype_linked_ted_finds_event_burden_association():
    import pyfgsea

    rng = np.random.default_rng(88)
    samples = [f"s{i}" for i in range(12)]
    phenotype_values = np.linspace(-1.0, 1.0, len(samples))
    times = np.linspace(0.0, 1.0, 9)
    rows = []
    for sample, phenotype in zip(samples, phenotype_values):
        for t in times:
            signal_profile = np.exp(-0.5 * ((t - 0.45) / 0.12) ** 2)
            background_profile = np.sin(2 * np.pi * t)
            rows.append(
                {
                    "sample": sample,
                    "pathway": "SIGNAL_EVENT",
                    "pt_mid": t,
                    "NES": 2.0 * phenotype * signal_profile + rng.normal(0.0, 0.03),
                }
            )
            rows.append(
                {
                    "sample": sample,
                    "pathway": "BACKGROUND_EVENT",
                    "pt_mid": t,
                    "NES": rng.normal(0.0, 0.3) * background_profile,
                }
            )
    events = pd.DataFrame(
        {
            "event_id": ["signal", "background"],
            "pathway": ["SIGNAL_EVENT", "BACKGROUND_EVENT"],
            "activation_onset": [0.25, 0.25],
            "duration": [0.45, 0.45],
            "peak_time": [0.45, 0.45],
        }
    )
    phenotype = pd.DataFrame(
        {
            "sample": samples,
            "severity": phenotype_values + rng.normal(0.0, 0.02, len(samples)),
        }
    )

    tables = pyfgsea.associate_phenotype_events(
        pd.DataFrame(rows),
        events,
        phenotype,
        sample_col="sample",
        phenotype_col="severity",
        min_samples=8,
        cv_splits=3,
        q_threshold=0.1,
    )

    assert {
        "event_burden_score",
        "phenotype_event_association",
        "phenotype_prediction_performance",
        "phenotype_linked_event_report",
    }.issubset(tables)
    assoc = tables["phenotype_event_association"].set_index("event_id")
    assert assoc.loc["signal", "event_q"] < 0.05
    assert assoc.loc["signal", "beta"] > 0
    assert tables["phenotype_prediction_performance"]["cross_validated_correlation"].iloc[0] > 0.7
    report = tables["phenotype_linked_event_report"].set_index("event_id")
    assert report.loc["signal", "evidence_level"] == "phenotype_linked_event"


def test_replicate_aware_condition_comparison_smoke():
    import pyfgsea
    import pyfgsea.trajectory as traj

    traj.HAS_SCANPY = True
    adata, gene_sets, _truth = pyfgsea.make_synthetic_trajectory_truth(
        "condition_delayed_activation",
        n_cells=120,
        n_genes=40,
        pathway_size=8,
        seed=33,
    )
    idx = np.arange(adata.n_obs)
    condition = adata.obs["condition"].astype(str).to_numpy()
    sample = np.where(
        condition == "control",
        np.where((idx // 2) % 2 == 0, "ctrl_1", "ctrl_2"),
        np.where((idx // 2) % 2 == 0, "case_1", "case_2"),
    )
    adata.obs["sample"] = sample

    comparison = pyfgsea.compare_trajectory_gsea(
        adata,
        gene_sets,
        condition_key="condition",
        sample_key="sample",
        mode="replicate_aware",
        control="control",
        case="case",
        pseudotime_key="dpt_pseudotime",
        n_permutations=2,
        seed=33,
        window_size=15,
        step=15,
        min_size=5,
        max_size=100,
        nperm_nes=8,
        sample_size=8,
        event_kwargs={"min_consecutive": 1},
    )

    assert {"sample_consistency", "event_fdr", "event_type"}.issubset(
        comparison.columns
    )
    assert "events" in comparison.attrs
    assert set(comparison["calibration_status"]) == {
        "descriptive_only_low_replicate_count"
    }
    assert "n_replicates_control" in comparison.columns


def test_replicate_key_sample_balanced_calibration_smoke():
    import pyfgsea
    import pyfgsea.trajectory as traj

    traj.HAS_SCANPY = True
    adata, gene_sets, _truth = pyfgsea.make_synthetic_trajectory_truth(
        "condition_delayed_activation",
        n_cells=180,
        n_genes=40,
        pathway_size=8,
        seed=36,
    )
    condition = adata.obs["condition"].astype(str).to_numpy()
    ctrl_seen = np.cumsum(condition == "control") - 1
    case_seen = np.cumsum(condition == "case") - 1
    adata.obs["donor"] = np.where(
        condition == "control",
        np.char.add("ctrl_", ((ctrl_seen % 3) + 1).astype(str)),
        np.char.add("case_", ((case_seen % 3) + 1).astype(str)),
    )

    comparison = pyfgsea.compare_trajectory_gsea(
        adata,
        gene_sets,
        condition_key="condition",
        replicate_key="donor",
        mode="replicate_aware",
        control="control",
        case="case",
        pseudotime_key="dpt_pseudotime",
        ranker="detection_weighted",
        n_permutations=2,
        n_bootstrap=5,
        seed=36,
        window_size=24,
        step=24,
        min_cells_per_replicate=2,
        min_replicates_per_condition=3,
        min_size=5,
        max_size=100,
        nperm_nes=8,
        sample_size=8,
        event_kwargs={"min_consecutive": 1},
    )

    assert {
        "n_replicates_control",
        "n_replicates_case",
        "min_cells_per_replicate",
        "replicate_support",
        "sample_consistency",
        "delta_duration",
        "replicate_aware_p",
        "replicate_aware_q",
        "calibration_method",
        "calibration_status",
    }.issubset(comparison.columns)
    assert "sample_balanced" in comparison.attrs["results"]["ranker"].iloc[0]
    assert comparison.attrs["replicate_aware"]["replicate_key"] == "donor"


def test_matched_window_condition_comparison_reports_balance():
    import pyfgsea
    import pyfgsea.trajectory as traj

    traj.HAS_SCANPY = True
    adata, gene_sets, _truth = pyfgsea.make_synthetic_trajectory_truth(
        "condition_delayed_activation",
        n_cells=120,
        n_genes=40,
        pathway_size=8,
        seed=38,
    )

    comparison = pyfgsea.compare_trajectory_gsea(
        adata,
        gene_sets,
        condition_key="condition",
        mode="matched_window",
        control="control",
        case="case",
        pseudotime_key="dpt_pseudotime",
        ranker="detection_weighted",
        balance="weights",
        seed=38,
        window_size=24,
        step=24,
        min_size=5,
        max_size=100,
        nperm_nes=8,
        sample_size=8,
        max_window_merge=1,
        n_counts_balance_weight=0.5,
        event_kwargs={"min_consecutive": 1},
    )

    assert {
        "balance_pass_rate",
        "median_balance_score",
        "core_balance_pass_rate",
        "median_core_balance_score",
        "event_balance_coverage",
        "event_core_balance_coverage",
        "event_median_balance_score",
        "event_median_core_balance_score",
        "balanced_event_windows",
        "core_balanced_event_windows",
        "n_counts_sensitivity_flag",
        "sign_consistency",
        "eligible_condition_event",
    }.issubset(comparison.columns)
    assert set(comparison["n_counts_sensitivity_flag"]).issubset(
        {"balanced", "core_balanced_ncounts_shift", "not_comparable"}
    )
    assert "matched_window" in comparison.attrs
    assert comparison.attrs["matched_window"]["n_counts_balance_weight"] == 0.5
    assert "diagnostics" in comparison.attrs
    assert "balance_summary" in comparison.attrs
    assert {"pt_smd_before", "pt_smd_after", "balance_pass", "core_balance_pass", "balance_score", "core_balance_score", "window_merge_level"}.issubset(
        comparison.attrs["diagnostics"].columns
    )
    assert {
        "pt_smd_after_pass",
        "detection_rate_smd_after_pass",
        "n_counts_smd_after_pass",
        "n_genes_smd_after_pass",
        "effective_n_control_pass",
        "effective_n_case_pass",
        "overall_balance_pass",
        "median_balance_score",
        "core_balance_pass",
        "median_core_balance_score",
    }.issubset(set(comparison.attrs["balance_summary"]["metric"]))


def test_pseudobulk_condition_gsea_smoke():
    import pyfgsea
    import pyfgsea.trajectory as traj

    traj.HAS_SCANPY = True
    adata, gene_sets, _truth = pyfgsea.make_synthetic_trajectory_truth(
        "condition_delayed_activation",
        n_cells=120,
        n_genes=40,
        pathway_size=8,
        seed=34,
    )
    idx = np.arange(adata.n_obs)
    condition = adata.obs["condition"].astype(str).to_numpy()
    adata.obs["sample"] = np.where(
        condition == "control",
        np.where((idx // 2) % 2 == 0, "ctrl_1", "ctrl_2"),
        np.where((idx // 2) % 2 == 0, "case_1", "case_2"),
    )

    comparison = pyfgsea.compare_trajectory_gsea(
        adata,
        gene_sets,
        condition_key="condition",
        sample_key="sample",
        mode="pseudobulk",
        control="control",
        case="case",
        pseudotime_key="dpt_pseudotime",
        n_permutations=2,
        seed=34,
        window_size=20,
        step=20,
        min_cells_per_sample=2,
        min_samples_per_condition=2,
        min_size=5,
        max_size=100,
        nperm_nes=8,
        sample_size=8,
        event_kwargs={"min_consecutive": 1},
    )

    assert {"event_fdr", "event_type", "delta_AUC"}.issubset(comparison.columns)
    assert "calibrated_events" in comparison.attrs
    assert "diagnostics" in comparison.attrs
    assert "n_control_samples" in comparison.attrs["results"].columns


def test_mixed_effect_condition_calibration_smoke():
    import pyfgsea
    import pyfgsea.trajectory as traj

    traj.HAS_SCANPY = True
    adata, gene_sets, _truth = pyfgsea.make_synthetic_trajectory_truth(
        "condition_delayed_activation",
        n_cells=160,
        n_genes=40,
        pathway_size=8,
        seed=35,
    )
    idx = np.arange(adata.n_obs)
    condition = adata.obs["condition"].astype(str).to_numpy()
    adata.obs["sample"] = np.where(
        condition == "control",
        np.where((idx // 2) % 2 == 0, "ctrl_1", "ctrl_2"),
        np.where((idx // 2) % 2 == 0, "case_1", "case_2"),
    )

    comparison = pyfgsea.compare_trajectory_gsea(
        adata,
        gene_sets,
        condition_key="condition",
        sample_key="sample",
        mode="mixed_effect",
        control="control",
        case="case",
        pseudotime_key="dpt_pseudotime",
        n_permutations=0,
        seed=35,
        window_size=20,
        step=20,
        min_size=5,
        max_size=100,
        nperm_nes=8,
        sample_size=8,
        event_kwargs={"min_consecutive": 1},
    )

    assert {"mixed_event_p", "mixed_event_fdr", "mixed_effect_method"}.issubset(
        comparison.columns
    )
    assert {"event_p", "event_fdr"}.issubset(comparison.columns)
    assert "mixed_effect" in comparison.attrs


def test_score_then_smooth_baseline_and_event_comparison():
    import pyfgsea
    import pyfgsea.trajectory as traj

    traj.HAS_SCANPY = True
    adata, gene_sets, _truth = pyfgsea.make_synthetic_trajectory_truth(
        "monotonic_up",
        n_cells=80,
        n_genes=40,
        pathway_size=8,
        seed=37,
    )
    ted = pyfgsea.run_trajectory_gsea(
        adata,
        gene_sets,
        pseudotime_key="dpt_pseudotime",
        window_size=20,
        step=20,
        min_size=5,
        max_size=100,
        nperm_nes=8,
        sample_size=8,
    )
    ted_events = pyfgsea.summarize_events(ted, min_consecutive=1)

    baseline = pyfgsea.run_score_then_smooth_baseline(
        adata,
        gene_sets,
        pseudotime_key="dpt_pseudotime",
        method="rank_auc",
        smoother="rolling",
        window_size=20,
        step=20,
        min_size=5,
        max_size=100,
    )
    baseline_events = pyfgsea.summarize_events(baseline, min_consecutive=1)
    comparison = pyfgsea.compare_event_tables(
        ted_events,
        baseline_events,
        left_name="PyFgsea-TED",
        right_name="rank_auc_smooth",
    )

    assert {"activity_score", "activity_z", "NES", "window_midpoint"}.issubset(
        baseline.columns
    )
    assert not baseline_events.empty
    assert {
        "event_label_agreement",
        "peak_time_delta",
        "AUC_correlation",
        "top_event_overlap",
        "false_positive_under_random_sets",
        "runtime",
    }.issubset(comparison.columns)


def _make_y_shaped_adata(mixed_graph=False, seed=41):
    import anndata as ad
    from scipy import sparse

    rng = np.random.default_rng(seed)
    n_trunk = 24
    n_branch = 48
    n_obs = n_trunk + 2 * n_branch
    n_genes = 50
    genes = [f"G{i}" for i in range(n_genes)]

    pt_trunk = np.linspace(0.0, 0.42, n_trunk)
    pt_branch = np.linspace(0.45, 1.0, n_branch)
    pt = np.concatenate([pt_trunk, pt_branch, pt_branch])
    lineage = np.array(
        ["trunk"] * n_trunk + ["branch_a"] * n_branch + ["branch_b"] * n_branch
    )

    X = rng.poisson(0.25, size=(n_obs, n_genes)).astype(float)
    a_idx = np.arange(n_trunk, n_trunk + n_branch)
    b_idx = np.arange(n_trunk + n_branch, n_obs)
    late_a = ((pt[a_idx] - 0.45) / 0.55).clip(0, 1)
    late_b = ((pt[b_idx] - 0.45) / 0.55).clip(0, 1)
    X[np.ix_(a_idx, np.arange(0, 8))] += 5.0 * late_a[:, None] + 1.0
    X[np.ix_(b_idx, np.arange(8, 16))] += 5.0 * late_b[:, None] + 1.0

    rows = []
    cols = []

    def add_edge(i, j):
        rows.extend([i, j])
        cols.extend([j, i])

    for i in range(n_trunk - 1):
        add_edge(i, i + 1)
    add_edge(n_trunk - 1, n_trunk)
    add_edge(n_trunk - 1, n_trunk + n_branch)
    for offset in range(n_branch - 1):
        add_edge(n_trunk + offset, n_trunk + offset + 1)
        add_edge(n_trunk + n_branch + offset, n_trunk + n_branch + offset + 1)
    if mixed_graph:
        for offset in range(n_branch):
            add_edge(n_trunk + offset, n_trunk + n_branch + offset)
            if offset + 1 < n_branch:
                add_edge(n_trunk + offset, n_trunk + n_branch + offset + 1)

    data = np.ones(len(rows), dtype=float)
    graph = sparse.csr_matrix((data, (rows, cols)), shape=(n_obs, n_obs))
    adata = ad.AnnData(X)
    adata.var_names = genes
    adata.obs["dpt_pseudotime"] = pt
    adata.obs["lineage"] = lineage
    adata.obs["branch_a_fate"] = np.where(lineage == "branch_a", 1.0, 0.05)
    adata.obs.loc[lineage == "trunk", "branch_a_fate"] = 0.4
    adata.obsp["connectivities"] = graph
    gene_sets = {
        "BRANCH_A_SIGNAL": genes[:8],
        "BRANCH_B_SIGNAL": genes[8:16],
        "BACKGROUND": genes[16:24],
    }
    return adata, gene_sets


def test_graph_adaptive_windows_find_branch_specific_event():
    import pyfgsea
    import pyfgsea.trajectory as traj

    traj.HAS_SCANPY = True
    adata, gene_sets = _make_y_shaped_adata(seed=43)

    res = pyfgsea.run_trajectory_gsea(
        adata,
        gene_sets,
        pseudotime_key="dpt_pseudotime",
        ranker="mean_diff",
        window_mode="graph_adaptive",
        graph_key="connectivities",
        graph_radius=3,
        target_span=0.14,
        span_step=0.08,
        min_cells=6,
        max_cells=32,
        branch_key="lineage",
        min_branch_purity=0.75,
        cell_weight_key="branch_a_fate",
        experimental=True,
        min_size=5,
        max_size=100,
        nperm_nes=8,
        sample_size=8,
        bin_width=None,
    )

    assert not res.empty
    assert {
        "anchor_pseudotime",
        "effective_n_cells",
        "pseudotime_span",
        "graph_radius",
        "mean_graph_distance",
        "branch_purity",
        "weight_sum",
        "weight_entropy",
        "fate_weight_mean",
    }.issubset(res.columns)
    diagnostics = res.attrs["graph_window_diagnostics"]
    kept = diagnostics[~diagnostics["skipped"]]
    assert not kept.empty
    assert kept["branch_purity"].min() >= 0.75
    assert res.attrs["trajectory_params"]["window_mode"] == "graph_adaptive"

    events = summarize_events(res, min_consecutive=1)
    assert "BRANCH_A_SIGNAL" in set(events["Pathway"])
    assert res.loc[res["Pathway"] == "BRANCH_A_SIGNAL", "NES"].max() > 0


def test_graph_adaptive_window_diagnostics_are_topology_aware():
    import pyfgsea
    import pyfgsea.trajectory as traj

    traj.HAS_SCANPY = True
    adata, gene_sets = _make_y_shaped_adata(seed=45)

    linear = pyfgsea.run_trajectory_gsea(
        adata,
        gene_sets,
        pseudotime_key="dpt_pseudotime",
        window_mode="pseudotime_span",
        target_span=0.18,
        span_step=0.12,
        min_cells=8,
        max_cells=45,
        min_size=5,
        max_size=100,
        nperm_nes=8,
        sample_size=8,
        bin_width=None,
    )
    graph = pyfgsea.run_trajectory_gsea(
        adata,
        gene_sets,
        pseudotime_key="dpt_pseudotime",
        window_mode="graph_adaptive",
        graph_key="connectivities",
        graph_radius=3,
        target_span=0.14,
        span_step=0.1,
        min_cells=6,
        max_cells=32,
        branch_key="lineage",
        min_branch_purity=0.75,
        experimental=True,
        min_size=5,
        max_size=100,
        nperm_nes=8,
        sample_size=8,
        bin_width=None,
    )

    assert not linear.empty
    assert "branch_purity" not in linear.columns
    assert not graph.empty
    assert graph["branch_purity"].min() >= 0.75


def test_graph_adaptive_skips_low_purity_windows_when_graph_is_mixed():
    import pyfgsea
    import pyfgsea.trajectory as traj

    traj.HAS_SCANPY = True
    adata, gene_sets = _make_y_shaped_adata(mixed_graph=True, seed=47)

    res = pyfgsea.run_trajectory_gsea(
        adata,
        gene_sets,
        pseudotime_key="dpt_pseudotime",
        window_mode="graph_adaptive",
        graph_key="connectivities",
        graph_radius=4,
        target_span=0.2,
        span_step=0.12,
        min_cells=8,
        max_cells=60,
        branch_key="lineage",
        min_branch_purity=0.95,
        experimental=True,
        min_size=5,
        max_size=100,
        nperm_nes=8,
        sample_size=8,
        bin_width=None,
    )

    diagnostics = res.attrs["graph_window_diagnostics"]
    assert (diagnostics["skip_reason"] == "low_branch_purity").any()
    if not res.empty:
        assert res["branch_purity"].min() >= 0.95


def test_gene_set_and_window_indices_are_reusable():
    import pyfgsea
    import pyfgsea.trajectory as traj

    traj.HAS_SCANPY = True
    adata, gene_sets, _truth = pyfgsea.make_synthetic_trajectory_truth(
        "monotonic_up",
        n_cells=80,
        n_genes=40,
        pathway_size=8,
        seed=49,
    )
    first = pyfgsea.run_trajectory_gsea(
        adata,
        gene_sets,
        pseudotime_key="dpt_pseudotime",
        window_size=20,
        step=20,
        min_size=5,
        max_size=100,
        nperm_nes=8,
        sample_size=8,
    )
    gene_set_index = first.attrs["gene_set_index"]
    window_index = first.attrs["window_index"]

    second = pyfgsea.run_trajectory_gsea(
        adata,
        gene_sets,
        pseudotime_key="dpt_pseudotime",
        ranker="local_slope",
        window_size=20,
        step=20,
        min_size=5,
        max_size=100,
        nperm_nes=8,
        sample_size=8,
        gene_set_index=gene_set_index,
        window_index=window_index,
    )

    assert isinstance(gene_set_index, pyfgsea.GeneSetIndex)
    assert isinstance(window_index, pyfgsea.WindowIndex)
    assert second.attrs["gene_set_index"].hash == gene_set_index.hash
    assert second.attrs["window_index"].hash == window_index.hash
    assert len(window_index.out_cell_indices(0)) == adata.n_obs - 20
