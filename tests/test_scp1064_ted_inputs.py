from pathlib import Path

import pandas as pd
import pytest
import yaml


pytestmark = pytest.mark.external_data


ROOT = Path(__file__).resolve().parents[1]
SCP = ROOT / "data" / "processed" / "ted_known_source" / "SCP1064"


def test_scp1064_ted_input_files_exist():
    for path in [
        SCP / "ted_inputs" / "ted_rna_event_input.h5ad",
        SCP / "ted_inputs" / "ted_cell_metadata.tsv",
        SCP / "ted_inputs" / "ted_protein_outcome_table.tsv.gz",
        SCP / "metadata" / "perturbation_metadata.tsv",
        SCP / "metadata" / "guide_assignment.tsv",
    ]:
        assert path.exists(), path
        assert path.stat().st_size > 0, path


def test_scp1064_event_axes_include_primary_and_negative_controls():
    axes = yaml.safe_load((ROOT / "config" / "scp1064_event_axes.yml").read_text())
    for axis in [
        "immune_evasion_antigen_presentation",
        "ifn_jak_stat_response",
        "t_cell_interaction_or_cytokine_response",
    ]:
        assert axis in axes
        assert axes[axis]["positive_markers"]
    assert "negative_controls" in axes
    for control in ["ribosome", "mitochondrial", "housekeeping"]:
        assert control in axes["negative_controls"]


def test_scp1064_protein_outcome_table_is_nonempty():
    outcome = pd.read_csv(SCP / "ted_inputs" / "ted_protein_outcome_table.tsv.gz", sep="\t", nrows=100)
    required = {
        "cell_id",
        "guide_id",
        "target_gene",
        "protein_name",
        "protein_value",
        "protein_value_normalized",
    }
    assert required.issubset(outcome.columns)
    assert not outcome.empty
