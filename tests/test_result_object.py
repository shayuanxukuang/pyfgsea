import numpy as np

import pyfgsea
import pyfgsea.trajectory as traj


def test_trajectory_event_result_collects_tables_metadata_and_diagnostics():
    traj.HAS_SCANPY = True
    adata, gene_sets, _truth = pyfgsea.make_synthetic_trajectory_truth(
        "monotonic_up",
        n_cells=72,
        n_genes=40,
        pathway_size=8,
        seed=51,
    )
    adata.obs["donor"] = np.where(np.arange(adata.n_obs) % 2 == 0, "D1", "D2")
    results = pyfgsea.run_trajectory_gsea(
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
    events = pyfgsea.summarize_events(results, min_consecutive=1)
    event_fdr = pyfgsea.estimate_event_fdr(
        adata=adata,
        gmt_path=gene_sets,
        result=results,
        pseudotime_key="dpt_pseudotime",
        null="pseudotime_within_replicate_permutation",
        replicate_key="donor",
        n_perm=2,
        seed=51,
        window_size=24,
        step=24,
        min_size=5,
        max_size=100,
        nperm_nes=8,
        sample_size=8,
        event_kwargs={"min_consecutive": 1},
    )

    obj = pyfgsea.make_trajectory_event_result(
        adata=adata,
        gmt_path=gene_sets,
        results=results,
        events=events,
        event_fdr=event_fdr,
        seed=51,
        replicate_key="donor",
    )

    assert isinstance(obj, pyfgsea.TrajectoryEventResult)
    assert obj.calibration_status == "discovery_ready"
    assert obj.timing_status == "conditional_only"
    assert not obj.windows.empty
    assert not obj.diagnostics.empty
    assert obj.metadata["pyfgsea_version"] is not None
    assert obj.metadata["gmt_hash"] is not None
    assert obj.metadata["gene_universe_hash"] is not None
    assert obj.metadata["event_fdr_null_model"] == [
        "pseudotime_within_replicate_permutation"
    ]
    assert obj.metadata["n_perm"] == 2
    assert {
        "window_level",
        "event_level",
        "robustness_level",
    }.issubset(obj.evidence_layers)
    assert set(obj.summary()["table"]).issuperset({"results", "event_fdr", "diagnostics"})
    assert "activation_onset" in obj.events.columns
    assert "activation_onset" not in obj.to_tables()["events"].columns
    assert "activation_onset" in obj.to_tables(
        include_conditional_timing=True
    )["events"].columns


def test_result_object_marks_null_calibration_failed():
    import pandas as pd

    results = pd.DataFrame(
        {
            "Pathway": ["P"],
            "window_id": [0],
            "pt_start": [0.0],
            "pt_end": [0.2],
            "pt_mid": [0.1],
            "NES": [1.0],
        }
    )
    bad_fdr = pd.DataFrame(
        {
            "pathway": ["P"],
            "event_stat": ["max_abs_NES"],
            "observed": [1.0],
            "event_p": [float("nan")],
            "event_q": [float("nan")],
            "calibration_status": ["null_calibration_failed"],
            "calibration_warning": ["no_null_events"],
        }
    )
    obj = pyfgsea.make_trajectory_event_result(results=results, event_fdr=bad_fdr)
    assert obj.calibration_status == "null_calibration_failed"
