
import pandas as pd
import numpy as np
import os
import logging
from typing import Union, Any, List, Optional
from .wrapper import load_gmt, prepare_pathways, GseaRunner

logger = logging.getLogger(__name__)

HAS_SCANPY = False
try:
    import scanpy as sc
    HAS_SCANPY = True
except ImportError:
    pass

def _ensure_log1p(adata):
    """Ensure data is log1p transformed (internal helper)."""
    if not HAS_SCANPY:
        return adata
    
    if adata.X.max() > 20: 
        logger.info("Data appears raw (max > 20). Normalizing and log1p transforming.")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    return adata

def _subset_lineage(adata, lineage_col=None, lineage_keyword=None):
    if lineage_col and lineage_keyword:
        m = adata.obs[lineage_col].astype(str).str.contains(lineage_keyword, case=False, na=False)
        adata = adata[m].copy()
        logger.info(f"Subset lineage '{lineage_keyword}': {adata.n_obs} cells")
    return adata

def _compute_dpt(adata, root_gene=None, n_top_genes=2000, n_pcs=30, n_neighbors=15):
    if not HAS_SCANPY:
        raise ImportError("scanpy is required for pseudotime computation")
        
    if "dpt_pseudotime" in adata.obs:
        logger.info("Using existing 'dpt_pseudotime' in adata.obs.")
        return adata
        
    adata = _ensure_log1p(adata)
    logger.info("Re-processing manifold (PCA -> Neighbors -> Diffmap)...")
    
    adata_graph = adata.copy()
    sc.pp.highly_variable_genes(adata_graph, n_top_genes=n_top_genes, subset=True)
    
    try:
        sc.tl.pca(adata_graph, n_comps=n_pcs, svd_solver="arpack")
    except Exception:
        sc.tl.pca(adata_graph, n_comps=n_pcs, svd_solver="arpack")
        
    sc.pp.neighbors(adata_graph, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.diffmap(adata_graph)
    
    if root_gene is not None and root_gene in adata.var_names:
        x = adata[:, root_gene].X
        if hasattr(x, "todense"): 
            x_dense = x.todense()
        elif hasattr(x, "toarray"): 
            x_dense = x.toarray()
        else: 
            x_dense = x
            
        root_idx = int(np.asarray(x_dense).ravel().argmax())
        adata_graph.uns["iroot"] = root_idx
        logger.info(f"Using root gene {root_gene}, iroot={root_idx}")
        
    sc.tl.dpt(adata_graph)
    adata.obs["dpt_pseudotime"] = adata_graph.obs["dpt_pseudotime"]
    return adata

def _make_windows(sorted_idx, window_size, step):
    windows = []
    n = len(sorted_idx)
    for start in range(0, n - window_size + 1, step):
        win = sorted_idx[start:start + window_size]
        windows.append((start, start + window_size, win))
    return windows

def _rank_logfc_fast(X, window_indices, sum_total, n_all):
    n_in = len(window_indices)
    n_out = n_all - n_in
    sum_in = np.asarray(X[window_indices].sum(axis=0)).ravel()
    mu_in = sum_in / max(n_in, 1)
    mu_out = (sum_total - sum_in) / max(n_out, 1)
    return mu_in - mu_out

def run_trajectory_gsea(
    adata: Any,
    gmt_path: str,
    lineage_col: Optional[str] = None,
    lineage_keyword: Optional[str] = None,
    root_gene: Optional[str] = None,
    window_size: int = 500,
    step: int = 100,
    out_csv: Optional[str] = None,
    min_size: int = 15,
    max_size: int = 500,
    sample_size: int = 101,
    seed: int = 42,
    eps: float = 1e-50,
    nperm_nes: int = 100,
    pseudotime_key: str = "dpt_pseudotime",
    bin_width: int = 10, 
    calculate_nes: bool = True,
    use_nes_cache: bool = True
) -> pd.DataFrame:
    """
    Rolling-window GSEA along pseudotime (Trajectory Analysis).
    """
    if not HAS_SCANPY:
        raise ImportError("scanpy is required for trajectory analysis")
        
    if isinstance(adata, str):
        adata = sc.read_h5ad(adata)

    adata = _subset_lineage(adata, lineage_col, lineage_keyword)
    
    if pseudotime_key not in adata.obs:
         adata = _compute_dpt(adata, root_gene=root_gene)
    elif root_gene is not None:
         # If root_gene is provided, recompute DPT might be intended, 
         # but usually if key exists we use it. 
         # Following original logic: recompute if root_gene is explicit?
         # Or maybe just use existing. Let's stick to "use existing if present unless forced"
         # But here the original code recomputed if root_gene was present.
         # Let's keep original logic for safety.
         adata = _compute_dpt(adata, root_gene=root_gene)
    
    pt = adata.obs[pseudotime_key].to_numpy()
    ok = np.isfinite(pt)
    if not ok.all():
        adata = adata[ok].copy()
        pt = pt[ok]

    order = np.argsort(pt)
    windows = _make_windows(order, window_size=window_size, step=step)
    logger.info(f"Windows: {len(windows)} (size={window_size}, step={step})")

    gmt = load_gmt(gmt_path)
    genes = np.array(adata.var_names)
    pathway_names, pathway_indices = prepare_pathways(genes, gmt, min_size, max_size)
    
    if not pathway_indices:
        return pd.DataFrame()
    
    # Initialize Runner
    runner = GseaRunner(pathway_names, pathway_indices, min_size, max_size)
    
    X = adata.X
    n_all = X.shape[0]
    sum_total = np.asarray(X.sum(axis=0)).ravel()
    
    all_rows = []
    
    import time
    t_start = time.time()
    
    logger.info(f"Starting GSEA loop (Caching: {use_nes_cache}, nperm_nes: {nperm_nes})...")

    for wi, (s, e, window_indices) in enumerate(windows):
        logfc_vector = _rank_logfc_fast(X, window_indices, sum_total, n_all)
        scores = np.asarray(logfc_vector, dtype=np.float64)
        scores[~np.isfinite(scores)] = 0.0
        
        # Stateful Run
        res = runner.run(
            scores,
            sample_size=sample_size,
            seed=seed + wi,
            eps=eps,
            nperm_nes=nperm_nes,
            bin_width=bin_width,
            calculate_nes=calculate_nes,
            use_nes_cache=use_nes_cache
        )

        if not res.empty:
            pt_vals = pt[window_indices]
            res['window_id'] = wi
            res['pt_start'] = pt_vals.min()
            res['pt_end'] = pt_vals.max()
            res['pt_mid'] = (res['pt_start'] + res['pt_end']) / 2.0
            all_rows.append(res)
            
        if wi % 10 == 0 and wi > 0:
            elapsed = time.time() - t_start
            fps = (wi + 1) / elapsed
            print(f"Processed {wi+1}/{len(windows)} windows ({fps:.1f} win/s)...", end='\r')

    print(f"\nDone.")
    if not all_rows:
        return pd.DataFrame()

    df = pd.concat(all_rows, ignore_index=True)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
        
    return df
