def ensure_pseudotime(adata, key='dpt_pseudotime'):
    if key not in adata.obs:
        import scanpy as sc
        sc.tl.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.diffmap(adata)
        # Hack: set iroot if missing
        if 'iroot' not in adata.uns:
            adata.uns['iroot'] = 0
        sc.tl.dpt(adata)
    return adata