"""Regression tests for the trajectory runner's ranked NES background."""

import inspect

import anndata as ad
import numpy as np
import pandas as pd

from pyfgsea import GseaRunner, run_gsea
from pyfgsea.trajectory import run_trajectory_gsea


def _fixture(permutation=None):
    genes = np.asarray([f"G{index:03d}" for index in range(180)])
    scores = np.sin(np.arange(180) * 0.13) + np.linspace(2.5, -2.5, 180)
    pathways = {
        "early": genes[5:25].tolist(),
        "middle": genes[75:100].tolist(),
        "late": genes[145:170].tolist(),
    }
    if permutation is None:
        permutation = np.arange(len(genes))
    return genes[permutation], scores[permutation], pathways


def _runner_result(genes, scores, pathways, *, use_nes_cache=False):
    index = {gene: position for position, gene in enumerate(genes)}
    runner = GseaRunner(
        list(pathways),
        [[index[gene] for gene in pathways[name]] for name in pathways],
        min_size=5,
        max_size=50,
        gene_ids=genes,
        tie_policy="gene_id",
    )
    result = runner.run(
        scores,
        sample_size=21,
        nperm_simple=101,
        nperm_nes=40,
        seed=73,
        bin_width=0,
        use_nes_cache=use_nes_cache,
    ).sort_values("Pathway").reset_index(drop=True)
    return runner, result


def test_trajectory_defaults_disable_cross_window_cache_and_size_binning():
    trajectory_signature = inspect.signature(run_trajectory_gsea)
    runner_signature = inspect.signature(GseaRunner.run)
    assert trajectory_signature.parameters["use_nes_cache"].default is False
    assert trajectory_signature.parameters["bin_width"].default == 0
    assert runner_signature.parameters["use_nes_cache"].default is False
    assert runner_signature.parameters["bin_width"].default == 0


def test_unsorted_runner_nes_matches_static_ranked_analysis():
    permutation = np.random.default_rng(73).permutation(180)
    genes, scores, pathways = _fixture(permutation)
    _, trajectory = _runner_result(genes, scores, pathways)
    static = run_gsea(
        pd.DataFrame({"gene": genes, "score": scores}),
        pathways,
        gene_col="gene",
        score_col="score",
        min_size=5,
        max_size=50,
        sample_size=21,
        nperm_simple=101,
        nperm_nes=40,
        seed=73,
        bin_width=0,
    ).sort_values("Pathway").reset_index(drop=True)

    pd.testing.assert_series_equal(trajectory["ES"], static["ES"], check_names=False)
    pd.testing.assert_series_equal(trajectory["NES"], static["NES"], check_names=False)
    assert (trajectory["null_curve_size"] == trajectory["Size"]).all()


def test_gene_storage_order_does_not_change_trajectory_result():
    genes_a, scores_a, pathways = _fixture()
    genes_b, scores_b, _ = _fixture(np.random.default_rng(91).permutation(180))
    _, first = _runner_result(genes_a, scores_a, pathways)
    _, second = _runner_result(genes_b, scores_b, pathways)
    pd.testing.assert_frame_equal(first, second, check_dtype=False)


def test_nes_cache_key_includes_current_ranking():
    genes, scores, pathways = _fixture()
    runner, first = _runner_result(genes, scores, pathways, use_nes_cache=True)
    first_key = runner._nes_cache_key
    changed_scores = np.sign(scores) * np.square(np.abs(scores) + 0.25)
    second = runner.run(
        changed_scores,
        sample_size=21,
        nperm_simple=101,
        nperm_nes=40,
        seed=73,
        bin_width=0,
        use_nes_cache=True,
    ).sort_values("Pathway").reset_index(drop=True)

    assert first_key is not None
    assert runner._nes_cache_key != first_key
    assert not np.allclose(first["NES"], second["NES"], equal_nan=True)
    assert (second["null_curve_size"] == second["Size"]).all()


def test_each_trajectory_window_matches_manual_static_run(tmp_path):
    rng = np.random.default_rng(2026)
    n_cells = 30
    genes = np.asarray([f"G{index:03d}" for index in range(60)])
    expression = rng.lognormal(mean=0.2, sigma=0.6, size=(n_cells, len(genes)))
    expression[:, :10] += np.linspace(0.0, 2.0, n_cells)[:, None]
    expression[:, 45:55] += np.linspace(2.0, 0.0, n_cells)[:, None]
    adata = ad.AnnData(expression)
    adata.var_names = genes
    adata.obs["dpt_pseudotime"] = np.linspace(0.0, 1.0, n_cells)
    pathways = {
        "early": genes[:10].tolist(),
        "middle": genes[20:32].tolist(),
        "late": genes[45:55].tolist(),
    }
    gmt_path = tmp_path / "pathways.gmt"
    gmt_path.write_text(
        "".join(
            f"{name}\ttest\t" + "\t".join(members) + "\n"
            for name, members in pathways.items()
        ),
        encoding="utf-8",
    )
    common = dict(
        min_size=5,
        max_size=20,
        sample_size=11,
        nperm_simple=31,
        nperm_nes=20,
        seed=73,
        eps=0.0,
        bin_width=0,
        score_type="std",
        mode="aligned",
        tie_policy="gene_id",
        gsea_param=1.0,
        max_levels=1000,
    )
    trajectory = run_trajectory_gsea(
        adata,
        str(gmt_path),
        window_size=10,
        step=10,
        pseudotime_key="dpt_pseudotime",
        use_nes_cache=False,
        **common,
    )

    manual_rows = []
    total = expression.sum(axis=0)
    for window_id, start in enumerate((0, 10, 20)):
        window = expression[start : start + 10]
        scores = window.mean(axis=0) - (total - window.sum(axis=0)) / 20.0
        static = run_gsea(
            pd.DataFrame({"gene": genes, "score": scores}),
            pathways,
            gene_col="gene",
            score_col="score",
            **common,
        )
        static["window_id"] = window_id
        manual_rows.append(static)
    manual = pd.concat(manual_rows, ignore_index=True)

    columns = [
        "Pathway",
        "window_id",
        "ES",
        "NES",
        "P-value",
        "log_pval",
        "log2err",
        "padj",
        "status",
        "ranking_hash",
        "null_curve_size",
    ]
    left = trajectory[columns].sort_values(["window_id", "Pathway"]).reset_index(drop=True)
    right = manual[columns].sort_values(["window_id", "Pathway"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False, rtol=0.0, atol=0.0)
    assert trajectory.groupby("window_id")["ranking_hash"].first().nunique() == 3
    assert (trajectory["null_curve_size"] == trajectory["Size"]).all()
