"""Regression tests for the trajectory runner's ranked NES background."""

import inspect
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from pyfgsea import GseaRunner, run_gsea
from pyfgsea import trajectory as trajectory_module
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
    result = (
        runner.run(
            scores,
            sample_size=21,
            nperm_simple=101,
            nperm_nes=40,
            seed=73,
            bin_width=0,
            use_nes_cache=use_nes_cache,
        )
        .sort_values("Pathway")
        .reset_index(drop=True)
    )
    return runner, result


def test_trajectory_defaults_disable_cross_window_cache_and_size_binning():
    trajectory_signature = inspect.signature(run_trajectory_gsea)
    runner_signature = inspect.signature(GseaRunner.run)
    assert trajectory_signature.parameters["use_nes_cache"].default is False
    assert trajectory_signature.parameters["bin_width"].default == 0
    assert trajectory_signature.parameters["window_size"].default == 500
    assert trajectory_signature.parameters["step"].default == 50
    assert trajectory_signature.parameters["nperm_nes"].default == 2000
    assert runner_signature.parameters["use_nes_cache"].default is False
    assert runner_signature.parameters["bin_width"].default == 0


def test_missing_pseudotime_requires_an_explicit_root(monkeypatch):
    adata = ad.AnnData(np.ones((3, 2)))
    monkeypatch.setattr(trajectory_module, "HAS_SCANPY", True)

    with pytest.raises(ValueError, match="explicit root"):
        run_trajectory_gsea(adata, "unused.gmt")


def test_missing_custom_pseudotime_must_be_precomputed(monkeypatch):
    adata = ad.AnnData(np.ones((3, 2)))
    adata.uns["iroot"] = 0
    monkeypatch.setattr(trajectory_module, "HAS_SCANPY", True)

    with pytest.raises(ValueError, match="Custom pseudotime columns must be computed"):
        run_trajectory_gsea(adata, "unused.gmt", pseudotime_key="trajectory_time")


def test_unknown_root_gene_fails_before_dpt_processing(monkeypatch):
    adata = ad.AnnData(np.ones((3, 2)))
    adata.var_names = ["G1", "G2"]
    adata.obs["dpt_pseudotime"] = [0.0, 0.5, 1.0]
    monkeypatch.setattr(trajectory_module, "HAS_SCANPY", True)

    with pytest.raises(ValueError, match="root_gene is not present"):
        run_trajectory_gsea(adata, "unused.gmt", root_gene="MISSING")
    assert "dpt_pseudotime" in adata.obs


@pytest.mark.parametrize("location", ["uns", "var"])
def test_scanpy_expression_root_is_accepted(location):
    adata = ad.AnnData(np.array([[0.0, 0.0], [2.0, 2.0], [5.0, 5.0]]))
    root = np.array([2.1, 1.9])
    if location == "uns":
        adata.uns["xroot"] = root
        expected = "xroot_uns"
    else:
        adata.var["xroot"] = root
        expected = "xroot_var"

    assert trajectory_module._explicit_root_source(adata) == expected
    assert trajectory_module._expression_root_index(adata, expected) == 1


@pytest.mark.parametrize("location", ["uns", "var"])
def test_compute_dpt_uses_scanpy_expression_root(location, monkeypatch):
    adata = ad.AnnData(np.array([[0.0, 0.0], [2.0, 2.0], [5.0, 5.0]]))
    root = np.array([2.1, 1.9])
    if location == "uns":
        adata.uns["xroot"] = root
    else:
        adata.var["xroot"] = root
    observed = {}

    def dpt(copy):
        observed["iroot"] = copy.uns["iroot"]
        copy.obs["dpt_pseudotime"] = [0.5, 0.0, 1.0]

    fake_scanpy = SimpleNamespace(
        pp=SimpleNamespace(
            highly_variable_genes=lambda *args, **kwargs: None,
            neighbors=lambda *args, **kwargs: None,
        ),
        tl=SimpleNamespace(
            pca=lambda *args, **kwargs: None,
            diffmap=lambda *args, **kwargs: None,
            dpt=dpt,
        ),
    )
    monkeypatch.setattr(trajectory_module, "HAS_SCANPY", True)
    monkeypatch.setattr(trajectory_module, "sc", fake_scanpy)

    result = trajectory_module._compute_dpt(
        adata,
        n_top_genes=2,
        n_pcs=1,
        n_neighbors=1,
    )

    assert observed["iroot"] == 1
    assert result.uns["iroot"] == 1
    np.testing.assert_array_equal(result.obs["dpt_pseudotime"], [0.5, 0.0, 1.0])


def test_lineage_subset_remaps_positional_root():
    adata = ad.AnnData(np.ones((4, 2)))
    adata.obs_names = ["c0", "c1", "c2", "c3"]
    adata.obs["lineage"] = ["other", "keep", "keep", "keep"]
    adata.uns["iroot"] = 2

    subset = trajectory_module._subset_lineage(
        adata,
        "lineage",
        "keep",
        root_cell="c2",
    )

    assert subset.obs_names.tolist() == ["c1", "c2", "c3"]
    assert subset.uns["iroot"] == 1


def test_lineage_subset_remaps_non_string_root_name():
    adata = ad.AnnData(np.ones((3, 2)))
    adata.obs.index = pd.RangeIndex(3)
    adata.obs["lineage"] = ["drop", "keep", "keep"]

    subset = trajectory_module._subset_lineage(
        adata,
        "lineage",
        "keep",
        root_cell=trajectory_module._root_cell_from_index(adata, 2),
    )

    assert subset.obs_names.tolist() == ["1", "2"]
    assert subset.uns["iroot"] == 1


def test_runner_remaps_explicit_root_by_cell_name(monkeypatch):
    adata = ad.AnnData(np.ones((4, 2)))
    adata.obs_names = ["c0", "c1", "root", "c3"]
    adata.obs["lineage"] = ["other", "keep", "keep", "keep"]
    adata.uns["iroot"] = 2
    monkeypatch.setattr(trajectory_module, "HAS_SCANPY", True)

    def stop_after_root_remap(subset, root_gene=None):
        assert subset.obs_names.tolist() == ["c1", "root", "c3"]
        assert subset.uns["iroot"] == 1
        assert subset.obs_names[subset.uns["iroot"]] == "root"
        raise RuntimeError("root remapped")

    monkeypatch.setattr(trajectory_module, "_compute_dpt", stop_after_root_remap)
    with pytest.raises(RuntimeError, match="root remapped"):
        run_trajectory_gsea(
            adata,
            "unused.gmt",
            lineage_col="lineage",
            lineage_keyword="keep",
        )


def test_lineage_root_requires_unique_cell_names(monkeypatch):
    adata = ad.AnnData(np.ones((3, 2)))
    adata.obs_names = ["duplicate", "duplicate", "other"]
    adata.obs["lineage"] = ["drop", "keep", "keep"]
    adata.uns["iroot"] = 0
    monkeypatch.setattr(trajectory_module, "HAS_SCANPY", True)

    with pytest.raises(ValueError, match="root cell name is not unique"):
        run_trajectory_gsea(
            adata,
            "unused.gmt",
            lineage_col="lineage",
            lineage_keyword="keep",
        )


def test_lineage_subset_rejects_excluded_root():
    adata = ad.AnnData(np.ones((3, 2)))
    adata.obs["lineage"] = ["drop", "keep", "keep"]
    adata.uns["iroot"] = 0

    with pytest.raises(ValueError, match="root cell is excluded"):
        trajectory_module._subset_lineage(
            adata,
            "lineage",
            "keep",
            root_cell="0",
        )


def test_lineage_subset_rejects_excluded_expression_root(monkeypatch):
    adata = ad.AnnData(np.array([[0.0, 0.0], [2.0, 2.0], [5.0, 5.0]]))
    adata.obs["lineage"] = ["drop", "keep", "keep"]
    adata.uns["xroot"] = np.array([0.0, 0.0])
    monkeypatch.setattr(trajectory_module, "HAS_SCANPY", True)

    with pytest.raises(ValueError, match="root cell is excluded"):
        run_trajectory_gsea(
            adata,
            "unused.gmt",
            lineage_col="lineage",
            lineage_keyword="keep",
        )


def test_lineage_subset_remaps_expression_root(monkeypatch):
    adata = ad.AnnData(np.array([[0.0, 0.0], [2.0, 2.0], [5.0, 5.0]]))
    adata.obs["lineage"] = ["drop", "keep", "keep"]
    adata.uns["xroot"] = np.array([5.0, 5.0])
    monkeypatch.setattr(trajectory_module, "HAS_SCANPY", True)

    def stop_after_root_remap(subset, root_gene=None):
        assert subset.obs_names.tolist() == ["1", "2"]
        assert subset.uns["iroot"] == 1
        raise RuntimeError("root remapped")

    monkeypatch.setattr(trajectory_module, "_compute_dpt", stop_after_root_remap)
    with pytest.raises(RuntimeError, match="root remapped"):
        run_trajectory_gsea(
            adata,
            "unused.gmt",
            lineage_col="lineage",
            lineage_keyword="keep",
        )


def test_root_gene_recompute_failure_preserves_input_pseudotime(monkeypatch):
    adata = ad.AnnData(np.array([[2.0, 0.0], [1.0, 1.0], [0.0, 2.0]]))
    adata.var_names = ["G1", "G2"]
    original = np.array([0.0, 0.5, 1.0])
    adata.obs["dpt_pseudotime"] = original
    monkeypatch.setattr(trajectory_module, "HAS_SCANPY", True)

    def fail_dpt(copy, root_gene=None):
        assert copy is not adata
        assert "dpt_pseudotime" not in copy.obs
        raise RuntimeError("DPT failed")

    monkeypatch.setattr(trajectory_module, "_compute_dpt", fail_dpt)
    with pytest.raises(RuntimeError, match="DPT failed"):
        run_trajectory_gsea(adata, "unused.gmt", root_gene="G1")

    np.testing.assert_array_equal(adata.obs["dpt_pseudotime"], original)


def test_nonfinite_dpt_filter_does_not_mutate_input_root(monkeypatch):
    adata = ad.AnnData(np.ones((4, 2)))
    adata.obs_names = ["disconnected", "before", "root", "after"]
    adata.uns["iroot"] = 2
    monkeypatch.setattr(trajectory_module, "HAS_SCANPY", True)

    def disconnected_dpt(copy, root_gene=None):
        copy.obs["dpt_pseudotime"] = [np.inf, 0.0, 0.5, 1.0]
        copy.uns["iroot"] = 2
        return copy

    monkeypatch.setattr(trajectory_module, "_compute_dpt", disconnected_dpt)
    monkeypatch.setattr(trajectory_module, "load_gmt", lambda path: {})
    result = run_trajectory_gsea(
        adata,
        "unused.gmt",
        window_size=2,
        step=1,
        min_size=1,
        max_size=1,
    )

    assert result.empty
    assert result.attrs["params"]["root_index"] == 1
    assert result.attrs["params"]["root_cell"] == "root"
    assert adata.uns["iroot"] == 2
    assert "dpt_pseudotime" not in adata.obs


def test_root_gene_is_selected_after_log_normalization(monkeypatch):
    adata = ad.AnnData(np.array([[10.0, 0.0], [5.0, 0.0]]))
    adata.var_names = ["G1", "G2"]
    original = adata.X.copy()
    observed = {}

    def normalize(copy):
        copy.X = np.array([[1.0, 0.0], [2.0, 0.0]])
        return copy

    def dpt(copy):
        observed["iroot"] = copy.uns["iroot"]
        copy.obs["dpt_pseudotime"] = [1.0, 0.0]

    fake_scanpy = SimpleNamespace(
        pp=SimpleNamespace(
            highly_variable_genes=lambda *args, **kwargs: None,
            neighbors=lambda *args, **kwargs: None,
        ),
        tl=SimpleNamespace(
            pca=lambda *args, **kwargs: None,
            diffmap=lambda *args, **kwargs: None,
            dpt=dpt,
        ),
    )
    monkeypatch.setattr(trajectory_module, "HAS_SCANPY", True)
    monkeypatch.setattr(trajectory_module, "sc", fake_scanpy)
    monkeypatch.setattr(trajectory_module, "_ensure_log1p", normalize)

    trajectory_module._compute_dpt(
        adata,
        root_gene="G1",
        n_top_genes=2,
        n_pcs=1,
        n_neighbors=1,
    )

    assert observed["iroot"] == 1
    np.testing.assert_array_equal(adata.X, original)


def test_unsorted_runner_nes_matches_static_ranked_analysis():
    permutation = np.random.default_rng(73).permutation(180)
    genes, scores, pathways = _fixture(permutation)
    _, trajectory = _runner_result(genes, scores, pathways)
    static = (
        run_gsea(
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
        )
        .sort_values("Pathway")
        .reset_index(drop=True)
    )

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
    second = (
        runner.run(
            changed_scores,
            sample_size=21,
            nperm_simple=101,
            nperm_nes=40,
            seed=73,
            bin_width=0,
            use_nes_cache=True,
        )
        .sort_values("Pathway")
        .reset_index(drop=True)
    )

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
    left = (
        trajectory[columns].sort_values(["window_id", "Pathway"]).reset_index(drop=True)
    )
    right = manual[columns].sort_values(["window_id", "Pathway"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False, rtol=0.0, atol=0.0)
    assert trajectory.groupby("window_id")["ranking_hash"].first().nunique() == 3
    assert (trajectory["null_curve_size"] == trajectory["Size"]).all()
