import numpy as np
import pandas as pd

import pyfgsea
import pyfgsea.trajectory as traj


def test_calibrate_events_uses_per_event_window_max_then_bh_by_default():
    observed = pd.DataFrame(
        {
            "Pathway": ["Strong", "Weak"],
            "peak_NES": [5.0, 1.0],
            "trough_NES": [0.0, 0.0],
            "AUC": [2.0, 0.1],
            "AUC_abs": [2.0, 0.1],
            "duration": [0.2, 0.05],
        }
    )
    null = pd.DataFrame(
        {
            "Pathway": ["Strong", "Weak", "Strong", "Weak"],
            "perm_id": [0, 0, 1, 1],
            "peak_NES": [2.0, 1.5, 2.5, 1.0],
            "trough_NES": [0.0, 0.0, 0.0, 0.0],
            "AUC": [0.5, 0.4, 0.6, 0.2],
            "AUC_abs": [0.5, 0.4, 0.6, 0.2],
            "duration": [0.1, 0.1, 0.1, 0.1],
        }
    )

    calibrated = pyfgsea.calibrate_events(observed, null)

    assert {"event_p", "event_fdr", "event_p_peak_NES_abs"}.issubset(
        calibrated.columns
    )
    assert calibrated.loc[0, "event_p"] < calibrated.loc[1, "event_p"]
    assert calibrated.attrs["calibration"]["global_null"] is False
    assert set(calibrated["multiplicity_method"]) == {"per_event_max_window_bh"}
    assert set(calibrated["across_event_adjustment"]) == {"BH"}
    assert calibrated["event_q"].notna().all()


def test_calibrate_events_family_max_is_fwer_without_bh():
    observed = pd.DataFrame(
        {
            "Pathway": ["A", "B"],
            "peak_NES": [3.0, 1.0],
            "trough_NES": [0.0, 0.0],
            "AUC": [1.0, 0.2],
            "AUC_abs": [1.0, 0.2],
            "duration": [0.2, 0.1],
        }
    )
    null = pd.DataFrame(
        {
            "Pathway": ["A", "B", "A", "B"],
            "perm_id": [0, 0, 1, 1],
            "peak_NES": [2.0, 1.5, 2.5, 1.0],
            "trough_NES": [0.0, 0.0, 0.0, 0.0],
            "AUC": [0.5, 0.4, 0.6, 0.2],
            "AUC_abs": [0.5, 0.4, 0.6, 0.2],
            "duration": [0.1, 0.1, 0.1, 0.1],
        }
    )
    calibrated = pyfgsea.calibrate_events(observed, null, global_null=True)
    assert calibrated["event_q"].isna().all()
    assert calibrated["event_fdr"].isna().all()
    assert calibrated["family_fwer_p"].notna().all()
    assert set(calibrated["across_event_adjustment"]) == {"none"}


def test_selected_window_calibration_separates_three_targets():
    observed = np.array([[4.0, 0.0], [0.0, 2.0]])
    permuted = np.array(
        [
            [[1.0, 3.0], [1.0, 1.0]],
            [[2.0, 1.0], [0.5, 2.5]],
            [[0.5, 0.5], [1.5, 0.5]],
        ]
    )
    table = pyfgsea.calibrate_selected_window_statistics(
        observed, permuted, event_ids=["a", "b"]
    )
    assert list(table["selected_window_index"]) == [0, 1]
    assert (table["naive_selected_window_p"] <= table["event_p"]).all()
    assert (table["event_p"] <= table["family_fwer_p"]).all()
    assert set(table["across_event_adjustment"]) == {"BH"}


def test_selected_window_calibration_counts_floating_point_ties():
    observed = np.array([[100.0, 2.0]])
    # The first null draw is the theoretical identity permutation but differs
    # by ordinary matrix-reduction roundoff.
    permuted = np.array([[[100.0 - 5e-9, 1.0]], [[3.0, 2.0]]])
    table = pyfgsea.calibrate_selected_window_statistics(observed, permuted)
    assert table.loc[0, "event_p"] >= 2.0 / 3.0


def test_calibrate_comparison_uses_group_label_null():
    observed = pd.DataFrame(
        {
            "Pathway": ["A", "B"],
            "delta_AUC": [3.0, 0.2],
            "delta_peak_time": [0.15, 0.01],
        }
    )
    null = pd.DataFrame(
        {
            "Pathway": ["A", "B", "A", "B"],
            "perm_id": [0, 0, 1, 1],
            "delta_AUC": [0.8, 0.5, 1.0, 0.4],
            "delta_peak_time": [0.03, 0.02, 0.04, 0.01],
        }
    )

    calibrated = pyfgsea.calibrate_comparison(observed, null)

    assert {"comparison_p", "comparison_fdr", "comparison_p_delta_AUC_abs"}.issubset(
        calibrated.columns
    )
    assert calibrated.loc[0, "comparison_p"] < calibrated.loc[1, "comparison_p"]
    assert calibrated.attrs["calibration"]["type"] == "group_label_permutation"


def test_run_event_permutation_fdr_smoke():
    traj.HAS_SCANPY = True
    adata, gene_sets, _truth = pyfgsea.make_synthetic_trajectory_truth(
        "monotonic_up",
        n_cells=60,
        n_genes=40,
        pathway_size=8,
        seed=12,
    )

    calibrated = pyfgsea.run_event_permutation_fdr(
        adata,
        gene_sets,
        pseudotime_key="dpt_pseudotime",
        n_permutations=2,
        seed=12,
        window_size=20,
        step=20,
        min_size=5,
        max_size=100,
        nperm_nes=8,
        sample_size=8,
        event_kwargs={"min_consecutive": 1},
    )

    assert set(calibrated) == {
        "results",
        "events",
        "null_events",
        "calibrated_events",
    }
    assert "event_fdr" in calibrated["calibrated_events"].columns
    if not calibrated["null_events"].empty:
        assert calibrated["null_events"]["perm_id"].nunique() <= 2


def test_estimate_event_fdr_supports_api_aliases_and_gene_label_null():
    traj.HAS_SCANPY = True
    adata, gene_sets, _truth = pyfgsea.make_synthetic_trajectory_truth(
        "monotonic_up",
        n_cells=60,
        n_genes=40,
        pathway_size=8,
        seed=21,
    )

    events = pyfgsea.estimate_event_fdr(
        adata=adata,
        gmt_path=gene_sets,
        pseudotime_key="dpt_pseudotime",
        event_stats=["max_abs_nes", "auc_abs", "longest_run", "peak_sharpness"],
        null="gene_label_permutation",
        n_perm=2,
        n_jobs=2,
        seed=21,
        window_size=20,
        step=20,
        min_size=5,
        max_size=100,
        nperm_nes=8,
        sample_size=8,
        event_kwargs={"min_consecutive": 1},
    )

    assert "event_fdr" in events.columns
    assert {
        "pathway",
        "event_stat",
        "observed",
        "null_mean",
        "null_sd",
        "event_p",
        "event_q",
        "minimum_attainable_p",
        "n_perm",
        "null_model",
        "calibration_warning",
        "calibration_status",
    }.issubset(events.columns)
    assert "max_abs_NES" in set(events["event_stat"])
    assert events.attrs["calibration_kind"] == "gene_label_permutation"


def test_estimate_event_fdr_pseudotime_within_replicate_long_table():
    traj.HAS_SCANPY = True
    adata, gene_sets, _truth = pyfgsea.make_synthetic_trajectory_truth(
        "monotonic_up",
        n_cells=72,
        n_genes=40,
        pathway_size=8,
        seed=23,
    )
    adata.obs["donor"] = np.where(np.arange(adata.n_obs) % 3 == 0, "D1", "D2")
    res = pyfgsea.run_trajectory_gsea(
        adata,
        gene_sets,
        pseudotime_key="dpt_pseudotime",
        window_size=24,
        step=24,
        min_size=5,
        max_size=100,
        nperm_nes=8,
        sample_size=8,
    )

    table = pyfgsea.estimate_event_fdr(
        adata=adata,
        gmt_path=gene_sets,
        pseudotime_key="dpt_pseudotime",
        result=res,
        event_stats=["max_abs_nes", "auc_abs", "longest_run", "peak_sharpness"],
        null="pseudotime_within_replicate_permutation",
        replicate_key="donor",
        n_perm=2,
        seed=23,
        window_size=24,
        step=24,
        min_size=5,
        max_size=100,
        nperm_nes=8,
        sample_size=8,
        event_kwargs={"min_consecutive": 1},
    )

    assert {"event_p", "event_q", "event_fdr"}.issubset(table.columns)
    assert set(table["null_model"]) == {"pseudotime_within_replicate_permutation"}
    assert set(table["n_perm"]) == {2}
    assert {"max_abs_NES", "AUC_abs", "longest_significant_run", "peak_sharpness"}.issubset(
        set(table["event_stat"])
    )


def test_estimate_event_fdr_early_stop_records_effective_permutations():
    traj.HAS_SCANPY = True
    adata, gene_sets, _truth = pyfgsea.make_synthetic_trajectory_truth(
        "monotonic_up",
        n_cells=72,
        n_genes=40,
        pathway_size=8,
        seed=26,
    )
    adata.obs["donor"] = np.where(np.arange(adata.n_obs) % 3 == 0, "D1", "D2")

    table = pyfgsea.estimate_event_fdr(
        adata=adata,
        gmt_path=gene_sets,
        pseudotime_key="dpt_pseudotime",
        event_stats=["max_abs_nes"],
        null="pseudotime_within_replicate_permutation",
        replicate_key="donor",
        n_perm=5,
        early_stop=True,
        early_stop_interval=1,
        early_stop_threshold=0.05,
        seed=26,
        window_size=24,
        step=24,
        min_size=5,
        max_size=100,
        nperm_nes=8,
        sample_size=8,
        event_kwargs={"min_consecutive": 1},
    )

    assert {"n_perm_effective", "early_stopped", "minimum_attainable_p"}.issubset(
        table.columns
    )
    assert table["early_stopped"].any()
    assert int(table["n_perm_effective"].max()) < 5
    expected_floor = 1.0 / (1.0 + table["n_perm_effective"].astype(float))
    np.testing.assert_allclose(table["minimum_attainable_p"], expected_floor)


def test_event_fdr_power_report_and_targeted_family():
    traj.HAS_SCANPY = True
    adata, gene_sets, _truth = pyfgsea.make_synthetic_trajectory_truth(
        "monotonic_up",
        n_cells=72,
        n_genes=40,
        pathway_size=8,
        seed=27,
    )

    table = pyfgsea.estimate_event_fdr(
        adata=adata,
        gmt_path=gene_sets,
        pseudotime_key="dpt_pseudotime",
        event_stats=["max_abs_nes"],
        null="pseudotime_permutation",
        hypothesis_family="target_family",
        pathways=["TRUE_SIGNAL"],
        n_perm=2,
        seed=27,
        window_size=24,
        step=24,
        min_size=5,
        max_size=100,
        nperm_nes=8,
        sample_size=8,
        event_kwargs={"min_consecutive": 1},
    )

    assert set(table["pathway"]) == {"TRUE_SIGNAL"}
    assert {"minimum_attainable_q", "q_threshold_reachable", "recommended_min_n_perm"}.issubset(
        table.columns
    )
    report = table.attrs["power_report"]
    assert report.loc[0, "hypothesis_family"] == "target_family"
    assert int(report.loc[0, "n_tests"]) == 1
    assert np.isclose(report.loc[0, "minimum_attainable_p"], 1 / 3)
    assert np.isclose(report.loc[0, "minimum_attainable_q"], 1 / 3)
    assert int(report.loc[0, "recommended_min_n_perm"]) == 19


def test_targeted_directional_calibration_uses_family_size():
    observed = pd.DataFrame(
        {
            "Pathway": ["Ery", "Mye", "Other"],
            "branch_a_peak_NES": [4.0, 1.0, 0.5],
            "branch_b_peak_NES": [1.0, 3.0, 0.4],
        }
    )
    null = pd.DataFrame(
        {
            "Pathway": ["Ery", "Mye", "Ery", "Mye"],
            "perm_id": [0, 0, 1, 1],
            "branch_a_peak_NES": [2.0, 1.5, 2.5, 1.0],
            "branch_b_peak_NES": [1.5, 2.0, 2.0, 1.6],
        }
    )

    out = pyfgsea.targeted_directional_calibration(
        observed,
        null,
        expected_direction={"Ery": "branch_a", "Mye": "branch_b"},
        n_perm=2,
    )

    assert set(out["Pathway"]) == {"Ery", "Mye"}
    assert {"directional_p", "directional_q", "minimum_attainable_q"}.issubset(
        out.columns
    )
    assert out.attrs["power_report"].loc[0, "n_tests"] == 2

    csv_like = pd.DataFrame({"Pathway": ["A", "B"], "comparison_fdr": [1.0, 1.0]})
    power = pyfgsea.event_fdr_power_report(
        csv_like,
        hypothesis_family="loaded_csv_family",
        n_perm=20,
    )
    assert np.isclose(power["minimum_attainable_p"].iloc[0], 1 / 21)
    assert np.isclose(power["minimum_attainable_q"].iloc[0], 2 / 21)
    assert int(power["recommended_min_n_perm"].iloc[0]) == 39


def test_estimate_event_fdr_condition_label_by_replicate():
    traj.HAS_SCANPY = True
    adata, gene_sets, _truth = pyfgsea.make_synthetic_trajectory_truth(
        "condition_delayed_activation",
        n_cells=120,
        n_genes=40,
        pathway_size=8,
        seed=24,
    )
    condition = adata.obs["condition"].astype(str).to_numpy()
    seen_control = np.cumsum(condition == "control") - 1
    seen_case = np.cumsum(condition == "case") - 1
    adata.obs["donor"] = np.where(
        condition == "control",
        np.char.add("ctrl_", ((seen_control % 3) + 1).astype(str)),
        np.char.add("case_", ((seen_case % 3) + 1).astype(str)),
    )

    table = pyfgsea.estimate_event_fdr(
        adata=adata,
        gmt_path=gene_sets,
        pseudotime_key="dpt_pseudotime",
        event_stats=["delta_peak_time", "delta_AUC"],
        null="condition_label_permutation_by_replicate",
        condition_key="condition",
        replicate_key="donor",
        control="control",
        case="case",
        n_perm=2,
        seed=24,
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

    assert {"delta_AUC", "delta_peak_time"}.issubset(set(table["event_stat"]))
    assert set(table["null_model"]) == {"condition_label_permutation_by_replicate"}
    assert np.isfinite(table["event_p"]).any()


def test_estimate_event_fdr_condition_label_within_pseudotime_bins():
    traj.HAS_SCANPY = True
    adata, gene_sets, _truth = pyfgsea.make_synthetic_trajectory_truth(
        "condition_delayed_activation",
        n_cells=100,
        n_genes=40,
        pathway_size=8,
        seed=28,
    )

    table = pyfgsea.estimate_event_fdr(
        adata=adata,
        gmt_path=gene_sets,
        pseudotime_key="dpt_pseudotime",
        event_stats=["delta_AUC"],
        null="condition_label_permutation_within_pseudotime_bins",
        condition_key="condition",
        control="control",
        case="case",
        n_pseudotime_bins=4,
        n_perm=2,
        mode="matched_window",
        balance="weights",
        seed=28,
        window_size=24,
        step=24,
        min_size=5,
        max_size=100,
        nperm_nes=8,
        sample_size=8,
        event_kwargs={"min_consecutive": 1},
    )

    assert {"event_p", "event_q", "event_fdr"}.issubset(table.columns)
    assert set(table["null_model"]) == {
        "condition_label_permutation_within_pseudotime_bins"
    }
    assert table.attrs["calibration_kind"] == "condition_label_permutation_within_pseudotime_bins"


def test_estimate_event_fdr_branch_labels_within_pseudotime_bins():
    traj.HAS_SCANPY = True
    adata, gene_sets, _truth = pyfgsea.make_synthetic_trajectory_truth(
        "branch_specific_activation",
        n_cells=120,
        n_genes=40,
        pathway_size=8,
        seed=25,
    )

    table = pyfgsea.estimate_event_fdr(
        adata=adata,
        gmt_path=gene_sets,
        pseudotime_key="dpt_pseudotime",
        event_stats=["delta_AUC", "delta_peak_time"],
        null="branch_label_permutation_within_pseudotime_bins",
        branch_key="branch",
        branch_a="branch_a",
        branch_b="branch_b",
        n_pseudotime_bins=4,
        n_perm=2,
        seed=25,
        window_size=24,
        step=24,
        min_size=5,
        max_size=100,
        nperm_nes=8,
        sample_size=8,
        event_kwargs={"min_consecutive": 1},
    )

    assert {"pathway", "event_stat", "observed", "event_p", "event_q"}.issubset(
        table.columns
    )
    assert set(table["null_model"]) == {"branch_label_permutation_within_pseudotime_bins"}
    assert {"delta_AUC", "delta_peak_time"}.issubset(set(table["event_stat"]))


def test_estimate_event_fdr_dispatches_pseudobulk_sample_null():
    traj.HAS_SCANPY = True
    adata, gene_sets, _truth = pyfgsea.make_synthetic_trajectory_truth(
        "condition_delayed_activation",
        n_cells=100,
        n_genes=40,
        pathway_size=8,
        seed=22,
    )
    idx = np.arange(adata.n_obs)
    condition = adata.obs["condition"].astype(str).to_numpy()
    adata.obs["sample"] = np.where(
        condition == "control",
        np.where((idx // 2) % 2 == 0, "ctrl_1", "ctrl_2"),
        np.where((idx // 2) % 2 == 0, "case_1", "case_2"),
    )

    events = pyfgsea.estimate_event_fdr(
        adata=adata,
        gmt_path=gene_sets,
        pseudotime_key="dpt_pseudotime",
        null="pseudobulk_permutation",
        condition_key="condition",
        sample_key="sample",
        control="control",
        case="case",
        n_perm=1,
        seed=22,
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

    assert "event_fdr" in events.columns
    assert events.attrs["calibration_kind"] == "pseudobulk_permutation"


def test_run_comparison_permutation_calibration_smoke():
    traj.HAS_SCANPY = True
    adata, gene_sets, _truth = pyfgsea.make_synthetic_trajectory_truth(
        "condition_delayed_activation",
        n_cells=80,
        n_genes=40,
        pathway_size=8,
        seed=13,
    )

    calibrated = pyfgsea.run_comparison_permutation_calibration(
        adata,
        gene_sets,
        condition_key="condition",
        control="control",
        case="case",
        n_permutations=2,
        seed=13,
        pseudotime_key="dpt_pseudotime",
        window_size=20,
        step=20,
        min_size=5,
        max_size=100,
        nperm_nes=8,
        sample_size=8,
        event_kwargs={"min_consecutive": 1},
    )

    assert set(calibrated) == {
        "comparison",
        "null_comparisons",
        "calibrated_comparison",
        "results",
        "events",
    }
    assert "comparison_fdr" in calibrated["calibrated_comparison"].columns
    if not calibrated["null_comparisons"].empty:
        assert calibrated["null_comparisons"]["perm_id"].nunique() <= 2
