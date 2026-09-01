"""Fail-closed status and input-validation behavior."""

import numpy as np
import pandas as pd
import pytest

from pyfgsea import run_gsea


def _frame(scores):
    genes = [f"G{index:03d}" for index in range(len(scores))]
    return pd.DataFrame({"gene": genes, "score": scores}), genes


def test_max_level_limit_is_not_reported_as_a_small_pvalue():
    data, genes = _frame(np.linspace(10.0, -2.0, 500))
    result = run_gsea(
        data,
        {"top": genes[:10]},
        gene_col="gene",
        score_col="score",
        min_size=1,
        max_size=30,
        sample_size=21,
        nperm_simple=101,
        nperm_nes=10,
        calculate_nes=False,
        score_type="pos",
        eps=0.0,
        max_levels=0,
        seed=19,
    )
    row = result.iloc[0]
    assert row["status"] == "max_level_exceeded"
    assert "max_levels=0" in row["termination_reason"]
    assert np.isnan(row["P-value"])
    assert np.isnan(row["log2err"])

@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_ranked_statistics_are_rejected(bad):
    scores = np.linspace(1.0, -1.0, 100)
    scores[7] = bad
    data, genes = _frame(scores)
    with pytest.raises(ValueError, match="NaN|infinite|finite"):
        run_gsea(
            data,
            {"path": genes[:15]},
            gene_col="gene",
            score_col="score",
            min_size=5,
            max_size=30,
        )


def test_all_zero_statistics_have_a_trivial_resolved_result():
    data, genes = _frame(np.zeros(100))
    result = run_gsea(
        data,
        # Evenly spaced hits avoid making arbitrary gene-id tie order look like
        # a biologically meaningful top-enriched pathway.
        {"path": genes[4::5]},
        gene_col="gene",
        score_col="score",
        min_size=5,
        max_size=30,
        sample_size=21,
        nperm_simple=51,
        nperm_nes=10,
        calculate_nes=False,
        eps=0.0,
        seed=20,
    )
    row = result.iloc[0]
    assert row["status"] == "resolved"
    assert np.isfinite(row["ES"])
    assert 0.0 < row["P-value"] <= 1.0
    assert np.isfinite(row["log_pval"])


def test_all_initial_samples_at_target_do_not_report_resolved_nan():
    data, genes = _frame(np.linspace(3.0, -3.0, 12))
    result = run_gsea(
        data,
        {"top_singleton": [genes[0]]},
        gene_col="gene",
        score_col="score",
        min_size=1,
        max_size=5,
        sample_size=3,
        nperm_simple=11,
        nperm_nes=5,
        calculate_nes=False,
        score_type="pos",
        mode="aligned",
        eps=0.0,
        max_levels=100,
        seed=2484,
    )
    row = result.iloc[0]
    assert row["status"] == "resolved"
    assert np.isfinite(row["P-value"])
    assert 0.0 < row["P-value"] <= 1.0
    assert np.isfinite(row["log_pval"])


def test_unbalanced_same_sign_nulls_return_explicit_no_level_progress():
    scores = np.linspace(8.0, 0.1, 240)
    data, genes = _frame(scores)
    result = run_gsea(
        data,
        {"bottom": genes[-20:]},
        gene_col="gene",
        score_col="score",
        min_size=5,
        max_size=50,
        sample_size=21,
        nperm_simple=101,
        nperm_nes=5,
        calculate_nes=False,
        score_type="std",
        eps=0.0,
        seed=404,
    )
    row = result.iloc[0]
    assert row["status"] == "no_level_progress"
    assert "same-sign null samples" in row["termination_reason"]
    assert np.isnan(row["P-value"])
    assert np.isnan(row["log_pval"])
    assert np.isnan(row["log2err"])
