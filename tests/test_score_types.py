"""Public score-type semantics and compatibility aliases."""

import numpy as np
import pandas as pd
import pytest

from pyfgsea import run_gsea
from pyfgsea.wrapper import (
    build_tail_curve,
    calculate_es,
    fgsea_multilevel,
    get_random_es_means,
    query_tail_curve,
)


def _direction_fixture():
    genes = [f"G{index:03d}" for index in range(240)]
    scores = np.linspace(8.0, -8.0, len(genes))
    data = pd.DataFrame({"gene": genes, "score": scores})
    pathways = {"top": genes[:20], "bottom": genes[-20:]}
    return data, pathways


@pytest.mark.parametrize("score_type", ["std", "pos", "neg"])
def test_current_score_types_are_strict_and_bounded(score_type):
    data, pathways = _direction_fixture()
    result = run_gsea(
        data,
        pathways,
        gene_col="gene",
        score_col="score",
        min_size=5,
        max_size=50,
        sample_size=21,
        nperm_simple=101,
        nperm_nes=20,
        calculate_nes=False,
        score_type=score_type,
        eps=1e-12,
        seed=4,
    ).set_index("Pathway")

    assert result["P-value"].dropna().between(0.0, 1.0).all()
    if score_type in {"std", "pos"}:
        assert result.loc["top", "ES"] > 0.99
    if score_type in {"std", "neg"}:
        assert result.loc["bottom", "ES"] < -0.99
    if score_type == "pos":
        assert result.loc["bottom", "ES"] == pytest.approx(0.0)
    if score_type == "neg":
        assert result.loc["top", "ES"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "scores",
    [
        np.linspace(8.0, 0.1, 240),
        -np.linspace(0.1, 8.0, 240),
        np.concatenate([np.linspace(8.0, 0.1, 230), np.linspace(-0.1, -8.0, 10)]),
    ],
    ids=["all-positive", "all-negative", "strongly-unbalanced"],
)
@pytest.mark.parametrize("score_type", ["std", "pos", "neg"])
def test_score_types_on_one_sign_and_unbalanced_rankings(scores, score_type):
    genes = [f"G{index:03d}" for index in range(len(scores))]
    result = run_gsea(
        pd.DataFrame({"gene": genes, "score": scores}),
        {"top": genes[:20], "bottom": genes[-20:]},
        gene_col="gene",
        score_col="score",
        min_size=5,
        max_size=50,
        sample_size=21,
        nperm_simple=101,
        nperm_nes=10,
        calculate_nes=False,
        score_type=score_type,
        eps=0.0,
        seed=404,
    ).set_index("Pathway")

    if score_type == "std":
        assert set(result["status"]).issubset({"resolved", "no_level_progress"})
        resolved = result["status"] == "resolved"
        assert resolved.any()
        assert np.isfinite(result.loc[resolved, "P-value"]).all()
        assert (result.loc[resolved, "P-value"] > 0.0).all()
        unresolved = ~resolved
        assert result.loc[unresolved, "P-value"].isna().all()
        assert result.loc[unresolved, "log2err"].isna().all()
    else:
        assert (result["status"] == "resolved").all()
        assert np.isfinite(result["P-value"]).all()
        assert (result["P-value"] > 0.0).all()
    if score_type == "pos":
        assert (result["ES"] >= 0.0).all()
        assert result.loc["top", "P-value"] <= result.loc["bottom", "P-value"]
    elif score_type == "neg":
        assert (result["ES"] <= 0.0).all()
        assert result.loc["bottom", "P-value"] <= result.loc["top", "P-value"]


def test_python_default_explicit_std_and_none_are_equivalent():
    data, pathways = _direction_fixture()
    common = dict(
        gene_col="gene",
        score_col="score",
        min_size=5,
        max_size=50,
        sample_size=21,
        nperm_simple=101,
        nperm_nes=10,
        calculate_nes=False,
        seed=8,
    )
    default = run_gsea(data, pathways, **common)
    explicit = run_gsea(data, pathways, score_type="std", **common)
    compatibility_none = run_gsea(data, pathways, score_type=None, **common)
    pd.testing.assert_frame_equal(default, explicit, check_dtype=False)
    pd.testing.assert_frame_equal(default, compatibility_none, check_dtype=False)


def _rust_result_snapshot(results):
    numeric = np.asarray(
        [
            [
                result.es,
                result.pval,
                result.log_pval,
                result.log2err,
                result.n_levels,
                result.acceptance_rate_min,
                result.acceptance_rate_mean,
            ]
            for result in results
        ],
        dtype=float,
    )
    labels = [
        (
            result.status,
            result.termination_reason,
            result.ranking_hash,
            result.algorithm_revision,
        )
        for result in results
    ]
    return numeric, labels


def test_rust_default_explicit_std_and_none_are_equivalent():
    data, _ = _direction_fixture()
    scores = data["score"].to_numpy(dtype=float)
    pathways = [list(range(20)), list(range(220, 240))]
    common = dict(nperm_simple=101, max_levels=1000)

    default = fgsea_multilevel(scores, pathways, 21, 8, 1.0, 0.0, **common)
    explicit = fgsea_multilevel(
        scores, pathways, 21, 8, 1.0, 0.0, score_type="std", **common
    )
    compatibility_none = fgsea_multilevel(
        scores, pathways, 21, 8, 1.0, 0.0, score_type=None, **common
    )

    default_numeric, default_labels = _rust_result_snapshot(default)
    for candidate in (explicit, compatibility_none):
        numeric, labels = _rust_result_snapshot(candidate)
        np.testing.assert_equal(numeric, default_numeric)
        assert labels == default_labels


def test_std_equal_positive_and_negative_excursions_return_zero_es():
    # With uniform hit weights, a centered block has an exact -0.5 excursion
    # before the block and +0.5 at its end. Current fgsea std semantics return
    # zero when those magnitudes are exactly equal.
    genes = [f"G{index:03d}" for index in range(100)]
    data = pd.DataFrame(
        {"gene": genes, "score": np.linspace(10.0, -10.0, len(genes))}
    )
    result = run_gsea(
        data,
        {"centered": genes[40:60]},
        gene_col="gene",
        score_col="score",
        min_size=5,
        max_size=30,
        sample_size=21,
        nperm_simple=51,
        nperm_nes=10,
        calculate_nes=False,
        gsea_param=0.0,
        score_type="std",
        eps=0.0,
        seed=10,
    )

    row = result.iloc[0]
    assert row["ES"] == pytest.approx(0.0, abs=0.0)
    assert row["P-value"] == pytest.approx(1.0)
    assert row["log_pval"] == pytest.approx(0.0)
    assert row["status"] == "resolved"
    assert row["termination_reason"] == "zero enrichment score"


def test_public_float_es_helper_uses_the_same_exact_tie_semantics():
    assert calculate_es(np.ones(4), [1, 2], 0.0) == pytest.approx(0.0, abs=0.0)


@pytest.mark.parametrize("score_type, observed", [("pos", 0.5), ("neg", -0.5)])
def test_public_tail_curve_directional_modes_ignore_legacy_sign_switch(
    score_type, observed
):
    scores = np.linspace(4.0, -4.0, 16)
    positive_sign = build_tail_curve(
        scores, 3, 21, 17, 1.0, 0.0, score_type=score_type, sign=1
    )
    negative_sign = build_tail_curve(
        scores, 3, 21, 17, 1.0, 0.0, score_type=score_type, sign=-1
    )
    np.testing.assert_equal(
        positive_sign.populations[-1], negative_sign.populations[-1]
    )
    assert np.asarray(positive_sign.populations[-1]).min() >= 0.0
    assert query_tail_curve(
        positive_sign, observed, score_type=score_type, sign=1
    ) == query_tail_curve(
        positive_sign, observed, score_type=score_type, sign=-1
    )


def test_public_tail_curve_applies_eps_floor_without_precision_claim():
    curve = build_tail_curve(
        np.linspace(4.0, -4.0, 16),
        3,
        21,
        17,
        1.0,
        0.9,
        score_type="pos",
        sign=1,
    )
    pvalue, log2err = query_tail_curve(curve, 1.0, score_type="pos", sign=1)
    assert curve.eps == pytest.approx(0.9)
    assert pvalue == pytest.approx(0.9)
    assert np.isnan(log2err)


def test_zero_pos_null_scores_contribute_to_both_nes_mode_denominators():
    positive_mean, negative_mean = get_random_es_means(
        np.linspace(4.0, -4.0, 4), [1], 100, 7, 1.0, "pos"
    )[0]
    assert positive_mean > 0.0
    assert negative_mean == pytest.approx(0.0, abs=0.0)


def test_legacy_absolute_tail_retains_nonzero_equal_magnitude_extreme():
    result = fgsea_multilevel(
        np.linspace(10.0, -10.0, 100),
        [list(range(40, 60))],
        21,
        10,
        0.0,
        0.0,
        score_type="two_sided_abs",
        nperm_simple=51,
    )[0]
    assert abs(result.es) == pytest.approx(0.5)
    assert result.approximate


@pytest.mark.parametrize("alias", ["two_sided_abs", "one_sided_signed"])
def test_legacy_score_types_warn_and_are_marked_approximate(alias):
    data, pathways = _direction_fixture()
    with pytest.warns(FutureWarning, match="deprecated"):
        result = run_gsea(
            data,
            pathways,
            gene_col="gene",
            score_col="score",
            min_size=5,
            max_size=50,
            sample_size=21,
            nperm_simple=51,
            nperm_nes=10,
            calculate_nes=False,
            score_type=alias,
            seed=9,
        )
    assert result["approximate"].all()


def test_unknown_score_type_is_rejected():
    data, pathways = _direction_fixture()
    with pytest.raises(ValueError, match="score_type"):
        run_gsea(data, pathways, score_type="two-sided", min_size=5, max_size=50)
