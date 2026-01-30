import numpy as np
import pandas as pd
import pytest
from pyfgsea import run_gsea

def make_toy(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(n)]
    scores = rng.standard_normal(n)
    df = pd.DataFrame({"gene": genes, "score": scores}).sort_values("score", ascending=False)
    pathways = {"A": genes[:80], "B": genes[200:320], "C": genes[1200:1300]}
    return df, pathways

def test_smoke_runs():
    df, pathways = make_toy()
    res = run_gsea(df, pathways, gene_col="gene", score_col="score", nperm_nes=100, seed=42, eps=1e-10)
    assert set(["Pathway","ES","P-value","padj","Size"]).issubset(res.columns)
    assert res["P-value"].between(0,1).all()
    assert res["padj"].between(0,1).all()

def test_deterministic_seed():
    df, pathways = make_toy(1)
    r1 = run_gsea(df, pathways, gene_col="gene", score_col="score", nperm_nes=50, seed=7, eps=1e-10)
    r2 = run_gsea(df, pathways, gene_col="gene", score_col="score", nperm_nes=50, seed=7, eps=1e-10)
    pd.testing.assert_frame_equal(
        r1.sort_values("Pathway").reset_index(drop=True),
        r2.sort_values("Pathway").reset_index(drop=True),
        check_dtype=False,
    )

def test_batched_matches_nonbatched():
    df, pathways = make_toy(2)
    # Using small nperm to be fast; verify consistency
    rb = run_gsea(df, pathways, gene_col="gene", score_col="score", nperm_nes=50, seed=1, use_batched=True, eps=1e-10)
    rn = run_gsea(df, pathways, gene_col="gene", score_col="score", nperm_nes=50, seed=1, use_batched=False, eps=1e-10)
    
    rb = rb.sort_values("Pathway").reset_index(drop=True)
    rn = rn.sort_values("Pathway").reset_index(drop=True)
    
    # ES should be identical
    pd.testing.assert_series_equal(rb["ES"], rn["ES"], check_names=False)
    
    # NES might have slight float diffs if implementation varies, but usually should be close
    if "NES" in rb.columns and "NES" in rn.columns:
        pd.testing.assert_series_equal(rb["NES"], rn["NES"], check_names=False, rtol=1e-5)
