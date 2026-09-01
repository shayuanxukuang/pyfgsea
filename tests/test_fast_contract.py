"""Compatibility coverage for fast precheck routing and selection provenance."""

import numpy as np
import pandas as pd

from pyfgsea import run_gsea
from pyfgsea.wrapper import fgsea_multilevel


def _routing_fixture():
    genes = [f"G{index:03d}" for index in range(240)]
    scores = np.linspace(8.0, -8.0, len(genes))
    data = pd.DataFrame({"gene": genes, "score": scores})
    # A dispersed pathway has a deliberately shallow positive-tail target, so
    # the same deterministic precheck can exercise either route by changing
    # only the routing threshold.
    pathway = genes[::12]
    return data, genes, scores, pathway


def test_precheck_eps_selects_shallow_or_deep_fast_route():
    data, _, _, pathway = _routing_fixture()
    common = dict(
        gene_col="gene",
        score_col="score",
        min_size=5,
        max_size=50,
        sample_size=21,
        nperm_nes=5,
        calculate_nes=False,
        score_type="pos",
        mode="fast",
        precheck_n=101,
        eps=0.0,
        max_levels=100,
        seed=29,
    )

    shallow = run_gsea(data, {"dispersed": pathway}, precheck_eps=0.0, **common).iloc[0]
    deep = run_gsea(data, {"dispersed": pathway}, precheck_eps=1.0, **common).iloc[0]

    assert shallow["status"] == "resolved"
    assert shallow["n_levels"] == 0
    assert (
        shallow["termination_reason"]
        == "fast precheck selected the shallow simple estimator"
    )
    assert shallow["approximate"]

    assert deep["status"] == "resolved"
    assert deep["n_levels"] > 0
    assert deep["termination_reason"].startswith(
        "fast precheck selected the deep multilevel compound ruler; "
    )
    assert deep["approximate"]


def test_fast_mode_defaults_to_64_sample_precheck():
    data, _, _, pathway = _routing_fixture()
    result = run_gsea(
        data,
        {"dispersed": pathway},
        gene_col="gene",
        score_col="score",
        min_size=5,
        max_size=50,
        sample_size=21,
        nperm_nes=5,
        calculate_nes=False,
        score_type="pos",
        mode="fast",
        eps=0.0,
        max_levels=100,
        seed=29,
    )

    assert result.attrs["params"]["precheck_n"] == 64
    assert result.attrs["params"]["precheck_eps"] == 0.005


def test_termination_reason_distinguishes_aligned_and_legacy_selection():
    _, _, scores, _ = _routing_fixture()
    aligned = fgsea_multilevel(
        scores,
        [[index for index in range(0, len(scores), 12)]],
        21,
        29,
        1.0,
        0.0,
        score_type="pos",
        mode="aligned",
        nperm_simple=101,
        max_levels=100,
    )[0]
    legacy = fgsea_multilevel(
        scores,
        [list(range(20))],
        21,
        29,
        1.0,
        0.0,
        score_type="two_sided_abs",
        mode="aligned",
        nperm_simple=101,
        max_levels=100,
    )[0]

    assert aligned.n_levels == 0
    assert (
        aligned.termination_reason
        == "aligned expected log-error selected the simple estimator"
    )
    assert legacy.n_levels == 0
    assert (
        legacy.termination_reason
        == "legacy absolute-tail mode forced the simple estimator"
    )
