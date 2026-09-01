"""P-value floors, non-zero representation, and fgsea log-error semantics."""

import numpy as np
import pandas as pd
import pytest

from pyfgsea import run_gsea
from pyfgsea.wrapper import multilevel_error


def _extreme_fixture(n_genes=600, pathway_size=8):
    genes = [f"G{index:04d}" for index in range(n_genes)]
    data = pd.DataFrame(
        {"gene": genes, "score": np.linspace(20.0, -5.0, n_genes)}
    )
    return data, {"top": genes[:pathway_size]}


def test_multilevel_error_grows_with_tail_depth():
    shallow = multilevel_error(1e-3, 101)
    deep = multilevel_error(1e-30, 101)
    assert np.isfinite(shallow)
    assert np.isfinite(deep)
    assert 0.0 < shallow < deep


@pytest.mark.parametrize("pvalue", [0.0, -0.1, 1.1, np.nan, np.inf])
def test_multilevel_error_rejects_invalid_probability(pvalue):
    with pytest.raises(ValueError):
        multilevel_error(pvalue, 101)


def test_eps_floor_has_nan_error_and_explicit_status():
    data, pathways = _extreme_fixture()
    result = run_gsea(
        data,
        pathways,
        gene_col="gene",
        score_col="score",
        min_size=1,
        max_size=30,
        sample_size=21,
        nperm_simple=101,
        nperm_nes=10,
        calculate_nes=False,
        score_type="pos",
        eps=1e-3,
        seed=11,
    )
    row = result.iloc[0]
    assert row["status"] == "eps_floor"
    assert "below eps" in row["termination_reason"]
    assert row["P-value"] == pytest.approx(1e-3)
    assert 0.0 < row["P-value"] <= 1.0
    assert row["log_pval"] < np.log(1e-3)
    assert np.isnan(row["log2err"])
    assert row["pval_capped"]
    assert row["padj"] == pytest.approx(1e-3)
    # This fixture is far beyond eps, so it must enter the ruler and can stop
    # once the retained multilevel mass proves that the eps floor was crossed.
    assert row["n_levels"] > 0
    assert np.isfinite(row["acceptance_rate_mean"])


def test_no_resolved_result_silently_returns_zero_probability():
    data, pathways = _extreme_fixture(n_genes=300, pathway_size=10)
    result = run_gsea(
        data,
        pathways,
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
        seed=12,
    )
    resolved = result.loc[result["status"].isin({"resolved", "numerical_underflow"})]
    assert not resolved.empty
    assert (resolved["P-value"] > 0.0).all()
    assert np.isfinite(resolved["log_pval"]).all()
