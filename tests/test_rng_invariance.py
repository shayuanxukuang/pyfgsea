"""Stable random streams across ordering and Rayon thread counts."""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd

from pyfgsea import run_gsea


def _fixture():
    genes = [f"G{index:03d}" for index in range(220)]
    scores = np.sin(np.arange(220) * 0.17) + np.linspace(2.0, -2.0, 220)
    data = pd.DataFrame({"gene": genes, "score": scores})
    pathways = {
        "alpha": genes[3:21],
        "beta": genes[80:101],
        "gamma": genes[170:190],
    }
    return data, pathways


def _run(data, pathways, *, use_batched=True):
    return (
        run_gsea(
            data,
            pathways,
            gene_col="gene",
            score_col="score",
            min_size=5,
            max_size=50,
            sample_size=21,
            nperm_simple=101,
            nperm_nes=30,
            calculate_nes=False,
            score_type="std",
            eps=1e-12,
            seed=2025,
            use_batched=use_batched,
        )
        .sort_values("Pathway")
        .reset_index(drop=True)
    )


def test_pathway_dictionary_order_does_not_change_results():
    data, pathways = _fixture()
    forward = _run(data, pathways)
    reverse = _run(data, dict(reversed(list(pathways.items()))))
    pd.testing.assert_frame_equal(forward, reverse, check_dtype=False)


def test_public_ranking_hash_matches_result_provenance():
    data, pathways = _fixture()
    result = _run(data, pathways)
    params = result.attrs["params"]
    assert set(result["ranking_hash"]) == {params["ranking_hash"]}
    assert len(params["cache_ranking_hash"]) == 64


def test_batched_nonbatched_and_group_order_do_not_change_results():
    data, pathways = _fixture()
    batched = _run(data, pathways, use_batched=True)
    nonbatched = _run(data, dict(reversed(list(pathways.items()))), use_batched=False)
    pd.testing.assert_frame_equal(batched, nonbatched, check_dtype=False)


_SUBPROCESS_PROGRAM = textwrap.dedent(
    """
    import json
    import numpy as np
    import pandas as pd
    from pyfgsea import run_gsea

    genes = [f"G{index:03d}" for index in range(220)]
    scores = np.sin(np.arange(220) * 0.17) + np.linspace(2.0, -2.0, 220)
    frame = pd.DataFrame({"gene": genes, "score": scores})
    pathways = {
        "alpha": genes[3:21],
        "beta": genes[80:101],
        "gamma": genes[170:190],
    }
    result = run_gsea(
        frame, pathways, gene_col="gene", score_col="score",
        min_size=5, max_size=50, sample_size=21, nperm_simple=101,
        nperm_nes=10, calculate_nes=False, score_type="std",
        eps=1e-12, seed=2025,
    ).sort_values("Pathway")
    columns = [
        "Pathway", "ES", "P-value", "log_pval", "log2err", "status",
        "n_levels", "acceptance_rate_min", "acceptance_rate_mean",
        "ranking_hash", "algorithm_revision",
    ]
    print(json.dumps(result[columns].to_dict("records"), sort_keys=True))
    """
)


def _subprocess_result(thread_count):
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = str(thread_count)
    completed = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_PROGRAM],
        cwd=os.getcwd(),
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_rayon_thread_count_does_not_change_results():
    snapshots = [_subprocess_result(count) for count in (1, 2, 4, 8)]
    assert all(snapshot == snapshots[0] for snapshot in snapshots[1:])
