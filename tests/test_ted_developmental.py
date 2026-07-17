import numpy as np
import pandas as pd
import pytest

import pyfgsea

pytestmark = pytest.mark.slow


def test_ted_time_reports_real_and_pseudotime_onsets():
    curves = pd.DataFrame(
        {
            "pathway": ["cell_cycle_exit"] * 4,
            "lineage": ["erythroid"] * 4,
            "real_time": [0, 1, 2, 3],
            "pseudotime": [0.0, 0.2, 0.6, 1.0],
            "score": [0.0, 1.1, 2.0, 1.8],
            "q": [1.0, 0.01, 0.001, 0.01],
        }
    )

    table = pyfgsea.run_ted_time(curves, score_col="score")["ted_time_event_table"]

    row = table.iloc[0]
    assert row["real_time_onset"] == 1
    assert row["pseudotime_onset"] == 0.2
    assert row["event_time_confidence"] == "high_real_pseudo_concordant"
    assert "real_vs_pseudo_disagreement" in table.columns


def test_delay_classifier_separates_loss_and_delay():
    reference = pd.DataFrame(
        {
            "pathway": ["terminal_output", "late_peak"],
            "AUC": [10.0, 10.0],
            "peak_time": [0.5, 0.4],
            "activation_onset": [0.2, 0.1],
            "terminal_output": [3.0, 3.0],
        }
    )
    comparison = pd.DataFrame(
        {
            "pathway": ["terminal_output", "late_peak"],
            "AUC": [2.0, 9.0],
            "peak_time": [0.5, 0.8],
            "activation_onset": [0.2, 0.45],
            "terminal_output": [0.1, 2.9],
        }
    )

    modes = pyfgsea.classify_ted_delay_modes(
        reference,
        comparison,
        id_cols=["pathway"],
        auc_threshold=0.5,
        time_shift_threshold=0.1,
    )["developmental_event_mode"]

    by_pathway = modes.set_index("pathway")["developmental_event_mode"].to_dict()
    assert by_pathway["terminal_output"] == "true_loss"
    assert by_pathway["late_peak"] == "developmental_delay"


def test_ot_dynamic_returns_couplings_event_flow_and_fate_links():
    features = pd.DataFrame(
        {
            "x": [0.0, 0.2, 1.0, 1.2],
            "y": [0.0, 0.1, 1.0, 1.1],
        },
        index=["c0", "c1", "c2", "c3"],
    )
    metadata = pd.DataFrame(
        {
            "real_time": [0, 0, 1, 1],
            "condition": ["WT", "mut", "WT", "mut"],
        },
        index=features.index,
    )
    event_scores = pd.DataFrame({"maturation": [0.1, 0.0, 1.2, 0.4]}, index=features.index)
    fate = pd.DataFrame({"terminal_fate": [0.0, 0.1, 0.9, 0.5]}, index=features.index)

    tables = pyfgsea.run_ted_ot_dynamic(
        features,
        metadata,
        event_scores=event_scores,
        condition_col="condition",
        reference_label="WT",
        case_label="mut",
        fate_probability=fate,
        epsilon=0.2,
    )

    assert not tables["ot_cell_couplings"].empty
    assert tables["ot_event_flow"].iloc[0]["event"] == "maturation"
    assert not tables["counterfactual_event_loss"].empty
    assert tables["fate_probability_linked_event"].iloc[0]["future_fate"] == "terminal_fate"


def test_multiome_lag_promotes_chromatin_first_fate_candidate():
    atac = pd.DataFrame({"event_id": ["osmotic_response"], "onset": [1.0], "direction": ["activation"]})
    rna = pd.DataFrame({"event_id": ["osmotic_response"], "onset": [4.0], "direction": ["activation"]})
    fate = pd.DataFrame({"event_id": ["osmotic_response"], "onset": [6.0], "direction": ["activation"]})

    tables = pyfgsea.run_ted_multiome_lag(
        atac_events=atac,
        rna_events=rna,
        cell_state_events=fate,
        min_driver_lag_score=0.2,
    )

    lag = tables["multiome_event_lag_table"].iloc[0]
    assert lag["Lag_ATAC_to_RNA"] == 3.0
    assert tables["chromatin_first_mechanism_candidates"].iloc[0]["claim_grade"] == "ATAC->RNA->fate strong candidate"


def test_lineage_tree_scores_sister_asymmetry_and_prebranch_candidates():
    scores = pd.DataFrame(
        {"fate_tf": [1.0, 2.0, -1.0], "cell_cycle": [0.0, 0.2, 0.1]},
        index=["AB", "ABa", "ABp"],
    )
    edges = pd.DataFrame({"parent": ["AB", "AB"], "child": ["ABa", "ABp"]})

    tables = pyfgsea.run_ted_lineage_tree(scores, edges, divergence_quantile=0.5)

    sister = tables["sister_branch_divergence_score"]
    fate_row = sister[sister["event"] == "fate_tf"].iloc[0]
    assert fate_row["sister_cell_asymmetry_event"] == 3.0
    assert not tables["prebranch_event_candidates"].empty


def test_spatial_neighborhood_boundary_and_axis_tables():
    scores = pd.DataFrame(
        {"QC_program": [2.0, 1.8, 0.1, 0.2]},
        index=["s0", "s1", "s2", "s3"],
    )
    spatial = pd.DataFrame(
        {
            "x": [0.0, 0.0, 1.0, 1.0],
            "y": [0.0, 0.2, 0.0, 0.2],
            "cell_type": ["QC", "QC", "cortex", "cortex"],
        },
        index=scores.index,
    )

    tables = pyfgsea.run_ted_spatial_neighborhood(scores, spatial, k_neighbors=2)

    assert not tables["spatial_neighborhood_event_table"].empty
    assert not tables["boundary_specific_event"].empty
    assert "axis_q" in tables["spatial_event_propagation"].columns


def test_cross_kingdom_ontology_and_claim_ceiling():
    ontology = pyfgsea.build_cross_kingdom_event_ontology(
        event_tables={
            "arabidopsis": pd.DataFrame({"pathway": ["auxin hormone response"]}),
            "mouse": pd.DataFrame({"pathway": ["BMP morphogen response"]}),
        },
        species_gene_sets={"arabidopsis": {"cell wall remodeling": ["EXP1", "CESA1"]}},
    )

    assert "hormone_morphogen_response" in set(ontology["event_grammar_similarity"]["event_grammar"])
    claim = pyfgsea.assign_developmental_claim_ceiling(
        pd.DataFrame(
            {
                "event_id": ["zscape_event"],
                "event_q": [0.01],
                "block_q": [0.02],
                "multiome_support": [True],
            }
        )
    )["developmental_claim_ceiling"]
    assert claim.iloc[0]["claim_level_numeric"] == 3.5
