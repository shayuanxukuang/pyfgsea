"""Regression coverage for rank ties and gene-set ES boundary ties."""

import numpy as np
import pandas as pd
import pytest

from pyfgsea import run_gsea


def _tied_fixture(order):
    genes = np.asarray([f"G{index:03d}" for index in range(300)])
    scores = np.repeat(np.linspace(12.0, -12.0, 30), 10)
    frame = pd.DataFrame({"gene": genes[order], "score": scores[order]})
    pathways = {
        "top": genes[:12].tolist(),
        "middle": genes[120:136].tolist(),
        "bottom": genes[-18:].tolist(),
    }
    return frame, pathways


def _run(frame, pathways):
    with pytest.warns(UserWarning, match="non-zero ties"):
        return run_gsea(
            frame,
            pathways,
            gene_col="gene",
            score_col="score",
            min_size=5,
            max_size=50,
            sample_size=21,
            nperm_simple=101,
            nperm_nes=20,
            calculate_nes=False,
            tie_policy="gene_id",
            mode="aligned",
            eps=0.0,
            seed=178,
        ).sort_values("Pathway").reset_index(drop=True)


def test_gene_id_tie_policy_is_storage_order_invariant():
    natural = np.arange(300)
    shuffled = np.random.default_rng(178).permutation(300)
    frame_a, pathways = _tied_fixture(natural)
    frame_b, _ = _tied_fixture(shuffled)

    left = _run(frame_a, pathways)
    right = _run(frame_b, pathways)
    pd.testing.assert_frame_equal(left, right, check_dtype=False)


def test_compound_boundaries_make_repeated_es_deterministic_progress():
    frame, pathways = _tied_fixture(np.arange(300))
    first = _run(frame, pathways)
    second = _run(frame, pathways)

    pd.testing.assert_frame_equal(first, second, check_dtype=False)
    assert not first["status"].isin({"no_level_progress", "mixing_failure"}).any()
    deep = first.loc[first["n_levels"] > 0]
    assert not deep.empty
    assert deep["acceptance_rate_min"].between(0.0, 1.0).all()
    assert deep["acceptance_rate_mean"].between(0.0, 1.0).all()
