import scanpy as sc
import os

def load_adata(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return sc.read_h5ad(path)