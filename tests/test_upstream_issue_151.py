"""Regression for alserglab/fgsea#151 and the PR #178 tie fix."""

import numpy as np
import pandas as pd
import pytest

from pyfgsea import run_gsea


def test_issue_151_top_tied_statistics_resolve_deep_tail():
    # Zero-padding makes the deterministic gene-id tie policy preserve the
    # intended top-ten pathway while every ranked statistic remains tied.
    genes = [f"G{index:04d}" for index in range(5000)]
    data = pd.DataFrame({"gene": genes, "score": np.ones(len(genes))})

    with pytest.warns(UserWarning, match="non-zero ties"):
        result = run_gsea(
            data,
            {"top_ten": genes[:10]},
            gene_col="gene",
            score_col="score",
            min_size=1,
            max_size=100,
            sample_size=21,
            nperm_simple=101,
            nperm_nes=10,
            calculate_nes=False,
            score_type="pos",
            tie_policy="gene_id",
            mode="aligned",
            eps=0.0,
            seed=151,
        )

    row = result.iloc[0]
    assert row["status"] == "resolved"
    assert row["termination_reason"] == (
        "aligned expected log-error selected the multilevel compound ruler; "
        "multilevel compound boundary resolved"
    )
    assert np.isfinite(row["P-value"])
    assert 0.0 < row["P-value"] < 1e-28
    assert np.isfinite(row["log_pval"])
    assert np.isfinite(row["log2err"])
    assert row["n_levels"] > 0
    assert row["n_levels"] < 10_000
