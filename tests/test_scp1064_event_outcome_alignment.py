from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCP = ROOT / "data" / "processed" / "ted_known_source" / "SCP1064"


def test_scp1064_event_recovery_and_alignment_outputs_are_nonempty():
    recovery = pd.read_csv(SCP / "results" / "scp1064_event_recovery_summary.tsv", sep="\t")
    alignment = pd.read_csv(SCP / "results" / "scp1064_outcome_alignment_summary.tsv", sep="\t")
    specificity = pd.read_csv(SCP / "results" / "scp1064_specificity_summary.tsv", sep="\t")
    assert not recovery.empty
    assert not alignment.empty
    assert not specificity.empty
    assert recovery["robust_event"].astype(str).str.lower().eq("true").any()
    assert alignment["outcome_alignment_pass"].astype(str).str.lower().eq("true").any()
    assert str(specificity.loc[0, "negative_control_pass"]).lower() == "false"
    assert specificity.loc[0, "heavy_shuffle_status"] == "completed_full_cell_with_library_batch_not_estimable"
    assert str(specificity.loc[0, "leave_one_unit_refit_pass"]).lower() == "true"


def test_scp1064_alignment_contains_cell_guide_and_target_levels():
    alignment = pd.read_csv(SCP / "results" / "scp1064_outcome_alignment_summary.tsv", sep="\t")
    assert {"cell", "guide", "target"}.issubset(set(alignment["level"]))
    for level in ["cell", "guide", "target"]:
        subset = alignment[alignment["level"].eq(level)]
        assert not subset.empty


def test_scp1064_author_effect_reference_is_recorded():
    alignment = pd.read_csv(SCP / "results" / "scp1064_author_effect_alignment.tsv", sep="\t")
    summary = pd.read_csv(SCP / "results" / "scp1064_author_effect_summary.tsv", sep="\t")
    assert not alignment.empty
    assert not summary.empty
    assert {"matrix", "axis", "spearman", "direction_match"}.issubset(alignment.columns)
    assert {"axis", "max_abs_spearman", "mean_direction_match", "author_effect_support"}.issubset(summary.columns)
