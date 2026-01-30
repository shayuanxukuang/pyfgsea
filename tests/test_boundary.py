import pandas as pd
import numpy as np
import pytest
from pyfgsea import run_gsea

def test_series_input_compatibility():
    """Test that passing a pd.Series (index=genes, values=scores) works."""
    genes = [f"G{i}" for i in range(100)]
    scores = np.random.randn(100)
    series = pd.Series(scores, index=genes, name="my_scores")
    
    # Create a dummy pathway
    gmt = {"Path1": genes[:20], "Path2": genes[50:70]}
    
    res = run_gsea(series, gmt, min_size=15, max_size=500, seed=42)
    assert not res.empty
    assert "Pathway" in res.columns
    assert "ES" in res.columns

def test_boundary_values():
    """Test edge cases for scores (ties, zeros) and set sizes."""
    # 1. All zeros
    genes = [f"G{i}" for i in range(100)]
    scores = np.zeros(100)
    df = pd.DataFrame({"gene": genes, "score": scores})
    gmt = {"Path1": genes[:20]}
    
    # Should run without crashing, though results might be trivial
    res = run_gsea(df, gmt, gene_col="gene", score_col="score", seed=42)
    assert not res.empty
    # With all ties, ES might be 0 or close to it, p-value should be calculable (likely 1.0 or similar)
    assert res["P-value"].notna().all()

    # 2. Size filtering boundary
    # min_size=15, max_size=20
    gmt_boundary = {
        "TooSmall": genes[:14],   # 14 genes
        "JustRightMin": genes[:15], # 15 genes
        "JustRightMax": genes[:20], # 20 genes
        "TooLarge": genes[:21]    # 21 genes
    }
    
    res_bound = run_gsea(
        df, gmt_boundary, gene_col="gene", score_col="score", 
        min_size=15, max_size=20, seed=42
    )
    
    paths = set(res_bound["Pathway"])
    assert "TooSmall" not in paths
    assert "TooLarge" not in paths
    assert "JustRightMin" in paths
    assert "JustRightMax" in paths

