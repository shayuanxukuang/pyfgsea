from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCP = ROOT / "data" / "processed" / "ted_known_source" / "SCP1064"


def test_scp1064_claim_boundary_records_heavy_shuffle_downgrade():
    claim = pd.read_csv(SCP / "results" / "scp1064_claim_boundary.tsv", sep="\t").iloc[0]
    assert claim["status"] == "partial"
    assert claim["claim_boundary"] == "E1-V1_partial"
    assert str(claim["robust_event"]).lower() == "true"
    assert str(claim["known_source_metadata"]).lower() == "true"
    assert str(claim["outcome_alignment_pass"]).lower() == "true"
    assert str(claim["negative_control_pass"]).lower() == "false"
    assert str(claim["heavy_shuffle_pass"]).lower() == "false"


def test_scp1064_is_never_promoted_to_level4_causal_rescue():
    claim = pd.read_csv(SCP / "results" / "scp1064_claim_boundary.tsv", sep="\t").iloc[0]
    assert str(claim["matched_rescue_design"]).lower() == "false"
    assert str(claim["same_system"]).lower() == "false"
    assert str(claim["level4_causal_rescue"]).lower() == "false"


def test_global_claim_table_contains_scp1064_without_level4():
    claims = pd.read_csv(
        ROOT / "results" / "ted_known_source_validation" / "tables" / "ted_dataset_level_claim_boundary.tsv",
        sep="\t",
    )
    scp = claims[claims["dataset"].eq("SCP1064")].iloc[0]
    assert scp["claim_boundary"] == "E1-V1_partial"
    assert str(scp["negative_control_pass"]).lower() == "false"
    assert str(scp["level4_causal_rescue"]).lower() == "false"
