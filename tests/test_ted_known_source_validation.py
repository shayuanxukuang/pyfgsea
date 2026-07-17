from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "ted_known_source_validation" / "tables"


def test_known_source_outputs_exist_for_required_datasets():
    for name in [
        "ted_event_recovery_summary.tsv",
        "ted_outcome_alignment_summary.tsv",
        "ted_dataset_level_claim_boundary.tsv",
        "ted_negative_control_summary.tsv",
        "ted_specificity_summary.tsv",
        "gse153056_ted_event_scores.tsv",
        "gse153056_pdl1_outcome_alignment.tsv",
        "gse93735_reversal_index.tsv",
    ]:
        assert (TABLES / name).exists(), name


def test_gse153056_outcome_alignment_passes():
    claim = pd.read_csv(TABLES / "gse153056_claim_boundary.tsv", sep="\t")
    assert claim.loc[0, "status"] == "pass"
    assert claim.loc[0, "claim_boundary"] == "outcome_supported_event"
    align = pd.read_csv(TABLES / "gse153056_pdl1_outcome_alignment.tsv", sep="\t")
    assert align["outcome_alignment_pass"].astype(str).str.lower().eq("true").all()
    assert float(align["outcome_correlation"].iloc[0]) > 0.25


def test_gse93735_reversal_passes_and_controls_do_not_match_primary():
    claim = pd.read_csv(TABLES / "gse93735_claim_boundary.tsv", sep="\t")
    assert claim.loc[0, "status"] == "pass"
    assert claim.loc[0, "claim_boundary"] == "reversal_supported_event"
    reversal = pd.read_csv(TABLES / "gse93735_reversal_index.tsv", sep="\t")
    primary = reversal[reversal["role"].eq("primary_positive_reversal_axis")].iloc[0]
    controls = reversal[reversal["role"].eq("negative_control_axis")]
    assert float(primary["reversal_fraction"]) > 0.2
    assert controls["reversal_fraction"].astype(float).max() < float(primary["reversal_fraction"])


def test_submission_table_excludes_unfinished_candidates():
    claims = pd.read_csv(TABLES / "ted_dataset_level_claim_boundary.tsv", sep="\t")
    unfinished = claims[claims["dataset"].isin(["GSE90546", "GSE90063", "GSE133344"])]
    assert unfinished.empty
