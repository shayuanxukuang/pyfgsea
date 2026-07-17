from pathlib import Path

import pandas as pd
import pytest


pytestmark = pytest.mark.external_data


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "ted_known_source"


def _qc(dataset: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED / dataset / "qc_summary.tsv", sep="\t")


def test_gse153056_preprocessing_outputs_are_complete():
    root = PROCESSED / "GSE153056"
    for name in [
        "expression_matrix.tsv.gz",
        "protein_matrix.tsv.gz",
        "guide_matrix.tsv.gz",
        "cell_metadata.tsv.gz",
        "arrayed_cell_metadata.tsv.gz",
        "gene_mapping.tsv",
        "protein_mapping.tsv",
        "qc_summary.tsv",
    ]:
        assert (root / name).exists(), name
    meta = pd.read_csv(root / "cell_metadata.tsv.gz", sep="\t", nrows=100)
    assert {"cell_id", "condition", "perturbed_gene", "guide_id"}.issubset(meta.columns)
    assert meta["condition"].notna().all()
    qc = _qc("GSE153056").set_index("metric")["value"]
    assert qc["processing_status"] == "pass"


def test_gse93735_preprocessing_outputs_are_complete():
    root = PROCESSED / "GSE93735"
    for name in ["expression_matrix.tsv.gz", "sample_metadata.tsv", "gene_mapping.tsv", "qc_summary.tsv"]:
        assert (root / name).exists(), name
    meta = pd.read_csv(root / "sample_metadata.tsv", sep="\t")
    assert {"sample", "condition", "timepoint", "intervention"}.issubset(meta.columns)
    assert {"Control", "LPS", "Dex_LPS"}.issubset(set(meta["condition"]))
    qc = _qc("GSE93735").set_index("metric")["value"]
    assert qc["processing_status"] == "pass"


def test_gse90546_records_raw_required_boundary():
    root = PROCESSED / "GSE90546"
    assert (root / "upr_branch_annotation.tsv").exists()
    qc = _qc("GSE90546").set_index("metric")["value"]
    assert qc["processing_status"] == "metadata_only_raw_required_not_present"
    assert not (root / "expression_matrix.tsv.gz").exists()
