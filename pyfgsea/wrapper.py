
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Sequence, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from . import _core as _ext
except ImportError:
    # Fallback for development/editable mode where _core might be in the same directory
    import _core as _ext

# Expose core functions
fgsea_multilevel = _ext.fgsea_multilevel
fgsea_multilevel_batched = _ext.fgsea_multilevel_batched
fgsea_multilevel_batched_scores = _ext.fgsea_multilevel_batched_scores
get_random_es_means = _ext.get_random_es_means
build_tail_curve = _ext.build_tail_curve
query_tail_curve = _ext.query_tail_curve
calculate_es = _ext.calculate_es
GseaPrerankedRunner = _ext.GseaPrerankedRunner

__all__ = ['run_gsea', 'load_gmt', 'GseaRunner']

GeneSetDict = Dict[str, Sequence[str]]

def load_gmt(gmt_path: str) -> GeneSetDict:
    """Parses a GMT file into a dictionary."""
    pathways = {}
    with open(gmt_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3: continue
            name = parts[0]
            # description = parts[1] # Ignored
            genes = parts[2:]
            pathways[name] = genes
    return pathways

def prepare_pathways(
    genes: Sequence[str], 
    gmt: Union[str, GeneSetDict], 
    min_size: int = 15, 
    max_size: int = 500
) -> Tuple[List[str], List[List[int]]]:
    """
    Filters pathways by size and maps gene symbols to indices.
    """
    if isinstance(gmt, str):
        raw_pathways = load_gmt(gmt)
    else:
        raw_pathways = gmt

    gene_to_idx = {g: i for i, g in enumerate(genes)}

    # Pre-filter and map genes to indices using generator logic
    valid_pathways_gen = (
        (name, sorted([gene_to_idx[g] for g in gene_set if g in gene_to_idx]))
        for name, gene_set in raw_pathways.items()
    )

    # Filter by size and unzip
    filtered_pathways = [
        (name, indices) 
        for name, indices in valid_pathways_gen 
        if min_size <= len(indices) <= max_size
    ]

    if not filtered_pathways:
        logger.warning(f"No valid pathways found after filtering (min_size={min_size}, max_size={max_size}). "
                       f"Raw pathways: {len(raw_pathways)}. "
                       f"Ensure gene symbols match your input data.")
        return [], []

    pathway_names, pathway_indices = zip(*filtered_pathways)
    return list(pathway_names), list(pathway_indices)

class GseaRunner:
    """
    Stateful GSEA runner that caches pathway definitions in Rust to optimize repeated runs.
    """
    def __init__(self, pathway_names: List[str], pathway_indices: List[List[int]], min_size: int = 15, max_size: int = 500):
        self.pathway_names = pathway_names
        self.min_size = min_size
        self.max_size = max_size
        self.rust_runner = GseaPrerankedRunner(pathway_indices, min_size, max_size)
        
        # Pre-calculate sizes for NES lookup
        self.sizes = [len(p) for p in pathway_indices]
        self.unique_sizes = sorted(list(set(self.sizes)))
        
        # Cache for NES means
        self._nes_cache = None

    def run(
        self,
        scores: np.ndarray,
        sample_size: int = 101,
        seed: int = 42,
        nperm_nes: int = 500,
        gsea_param: float = 1.0,
        eps: float = 1e-50,
        score_type: str = 'two_sided_abs',
        calculate_nes: bool = True,
        bin_width: Optional[int] = None,
        precheck_n: Optional[int] = None,
        precheck_eps: Optional[float] = None,
        use_nes_cache: bool = False
    ) -> pd.DataFrame:
        scores = np.ascontiguousarray(scores, dtype=np.float64)
        
        # NES Background Calculation
        mean_lookup = {}
        if calculate_nes:
            if use_nes_cache and self._nes_cache is not None:
                mean_lookup = self._nes_cache
            else:
                means_vec = get_random_es_means(scores, self.unique_sizes, nperm_nes, seed, gsea_param)
                # Check structure of means_vec
                if not (isinstance(means_vec, list) and all(len(x) == 2 for x in means_vec)):
                    raise ValueError("get_random_es_means returned invalid format. Expected list of (pos_mean, neg_mean).")
                mean_lookup = {size: means for size, means in zip(self.unique_sizes, means_vec)}
                if use_nes_cache:
                    self._nes_cache = mean_lookup

        # Run GSEA via Rust core
        multi_results = self.rust_runner.run(
            scores, 
            sample_size, 
            seed, 
            gsea_param, 
            eps, 
            score_type, 
            bin_width, 
            precheck_n, 
            precheck_eps
        )
        
        # Format results
        results = []
        for i, name in enumerate(self.pathway_names):
            res_obj = multi_results[i]
            es = res_obj.es
            pval = res_obj.pval
            
            size = self.sizes[i]
            nes = 0.0
            
            if calculate_nes and size in mean_lookup:
                pos_mean, neg_mean = mean_lookup[size]
                if es > 0:
                    nes = es / pos_mean
                elif es < 0:
                    nes = es / abs(neg_mean)

            results.append({
                "Pathway": name,
                "ES": es,
                "NES": nes,
                "P-value": pval,
                "log2err": res_obj.log2err,
                "Size": size,
                "n_levels": res_obj.debug_info.current_level if res_obj.debug_info else None
            })
            
        res_df = pd.DataFrame(results)
        if res_df.empty:
            return res_df
            
        res_df = res_df.sort_values("P-value")
        
        # BH adjustment
        m = len(res_df)
        padj = res_df['P-value'].values * m / np.arange(1, m + 1)
        padj = np.minimum.accumulate(padj[::-1])[::-1]
        res_df['padj'] = np.minimum(padj, 1.0)
            
        res_df['pval_capped'] = res_df['P-value'] <= eps
        
        return res_df

def run_gsea(
    data: pd.DataFrame,
    gmt: Union[str, GeneSetDict],
    gene_col: Union[str, int] = 0,
    score_col: Union[str, int] = 1,
    min_size: int = 15,
    max_size: int = 500,
    sample_size: int = 101,
    seed: int = 42,
    nperm_nes: int = 500,
    gsea_param: float = 1.0,
    eps: float = 1e-50,
    dedup_genes: str = 'max_abs',
    score_type: str = 'two_sided_abs',
    use_batched: bool = True,
    bin_width: Optional[int] = None,
    calculate_nes: bool = True
) -> pd.DataFrame:
    """
    Main entry point for running GSEA on a pandas DataFrame.
    """
    # 1. Data Preprocessing
    df = data.copy()
    
    # Normalize column access
    score_c = df.columns[score_col] if isinstance(score_col, int) else score_col
    gene_c = df.columns[gene_col] if isinstance(gene_col, int) else gene_col
    
    # Type conversion
    df[score_c] = df[score_c].astype(float)
    df[gene_c] = df[gene_c].astype(str)

    # Gene Deduplication
    if dedup_genes == 'max_abs':
        df['__abs_score'] = df[score_c].abs()
        df = df.sort_values('__abs_score', ascending=False).drop_duplicates(subset=gene_c, keep='first')
        df = df.drop(columns=['__abs_score'])
    elif dedup_genes == 'first':
        df = df.drop_duplicates(subset=gene_c, keep='first')
    
    # Sort by score descending
    df = df.sort_values(by=score_c, ascending=False).reset_index(drop=True)
    
    genes = df[gene_c].values
    scores = np.ascontiguousarray(df[score_c].values, dtype=np.float64)
    
    # 2. Prepare Pathways
    pathway_names, pathway_indices = prepare_pathways(genes, gmt, min_size, max_size)
            
    if not pathway_indices:
        logger.warning(f"No valid pathways found after filtering (min_size={min_size}, max_size={max_size}). "
                       f"Ensure gene symbols match your input data.")
        return pd.DataFrame()

    # 3. NES Background (Optional)
    mean_lookup = {}
    if calculate_nes:
        sizes = [len(p) for p in pathway_indices]
        unique_sizes = sorted(list(set(sizes)))
        means_vec = get_random_es_means(scores, unique_sizes, nperm_nes, seed, gsea_param)
        if not (isinstance(means_vec, list) and all(len(x) == 2 for x in means_vec)):
            raise ValueError("get_random_es_means returned invalid format. Expected list of (pos_mean, neg_mean).")
        mean_lookup = {size: means for size, means in zip(unique_sizes, means_vec)}
    
    # 4. Run Core GSEA
    if use_batched:
        multi_results = fgsea_multilevel_batched(scores, pathway_indices, sample_size, seed, gsea_param, eps, score_type, bin_width)
    else:
        multi_results = fgsea_multilevel(scores, pathway_indices, sample_size, seed, gsea_param, eps, score_type)
    
    # 5. Format Results
    results = []
    for i, name in enumerate(pathway_names):
        res_obj = multi_results[i]
        es = res_obj.es
        size = len(pathway_indices[i])
        
        nes = np.nan
        if calculate_nes and size in mean_lookup:
            pos_mean, neg_mean = mean_lookup[size]
            if es > 0:
                nes = es / pos_mean if pos_mean > 1e-9 else np.nan
            elif es < 0:
                nes = es / abs(neg_mean) if abs(neg_mean) > 1e-9 else np.nan
            else:
                nes = 0.0
            
        results.append({
            "Pathway": name,
            "ES": es,
            "NES": nes,
            "P-value": res_obj.pval,
            "log2err": res_obj.log2err,
            "Size": size,
            "n_levels": res_obj.debug_info.current_level if res_obj.debug_info else None
        })
        
    res_df = pd.DataFrame(results)
    if res_df.empty:
        return res_df

    res_df = res_df.sort_values("P-value")
    
    # BH Adjustment
    m = len(res_df)
    padj = res_df['P-value'].values * m / np.arange(1, m + 1)
    padj = np.minimum.accumulate(padj[::-1])[::-1]
    res_df['padj'] = np.minimum(padj, 1.0)
    
    res_df['pval_capped'] = res_df['P-value'] <= eps
    
    # Attach reproducibility metadata
    res_df.attrs['params'] = {
        'mode': 'multilevel',
        'sample_size': sample_size,
        'seed': seed,
        'nperm_nes': nperm_nes,
        'gsea_param': gsea_param,
        'eps': eps,
        'dedup_genes': dedup_genes,
        'min_size': min_size,
        'max_size': max_size
    }
    
    return res_df
