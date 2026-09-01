import pandas as pd
import numpy as np
import os
import logging
from typing import Any, Optional, Sequence
from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
from .wrapper import (
    GseaRunner,
    _normalize_run_options,
    load_gmt,
    prepare_pathways,
)
from .leading_edge import compute_leading_edges
from .validation import _expression_matrix

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
        m = (
            adata.obs[lineage_col]
            .astype(str)
            .str.contains(lineage_keyword, case=False, na=False)
        )
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
        sc.tl.pca(adata_graph, n_comps=n_pcs, svd_solver="auto")

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


_SUPPORTED_RANKERS = {
    "mean_diff",
    "wilcoxon",
    "t_stat",
    "z_score",
    "cohens_d",
    "detection_weighted",
    "local_slope",
    "smooth_slope",
    "neighbor_contrast",
}


def _stable_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class GeneSetIndex:
    pathway_names: list[str]
    pathway_indices: list[list[int]]
    sizes: list[int]
    gene_universe_hash: str
    gmt_hash: str
    min_size: int
    max_size: int
    params: dict = field(default_factory=dict)
    _hash: Optional[str] = field(default=None, init=False, repr=False, compare=False)

    @property
    def hash(self) -> str:
        if self._hash is None:
            self._hash = _stable_hash(
                {
                    "pathway_names": self.pathway_names,
                    "pathway_indices": [
                        list(map(int, idx)) for idx in self.pathway_indices
                    ],
                    "gene_universe_hash": self.gene_universe_hash,
                    "gmt_hash": self.gmt_hash,
                    "min_size": self.min_size,
                    "max_size": self.max_size,
                    "params": self.params,
                }
            )
        return self._hash

    def __eq__(self, other) -> bool:
        return isinstance(other, GeneSetIndex) and self.hash == other.hash


@dataclass
class WindowIndex:
    windows: list[tuple[int, int, np.ndarray]]
    weight_map: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    window_mode: str = "cell_count"
    n_obs: int = 0
    params: dict = field(default_factory=dict)
    _hash: Optional[str] = field(default=None, init=False, repr=False, compare=False)

    @property
    def hash(self) -> str:
        if self._hash is None:
            self._hash = _stable_hash(
                {
                    "window_mode": self.window_mode,
                    "windows": [
                        (int(s), int(e), np.asarray(idx, dtype=int).tolist())
                        for s, e, idx in self.windows
                    ],
                    "weight_keys": sorted(map(int, self.weight_map.keys())),
                    "params": self.params,
                    "n_obs": self.n_obs,
                }
            )
        return self._hash

    def __eq__(self, other) -> bool:
        return isinstance(other, WindowIndex) and self.hash == other.hash

    def out_cell_indices(self, window_id: int) -> np.ndarray:
        for s, _e, idx in self.windows:
            if int(s) == int(window_id):
                mask = np.ones(self.n_obs, dtype=bool)
                mask[np.asarray(idx, dtype=int)] = False
                return np.where(mask)[0]
        raise KeyError(f"window_id {window_id!r} not found")


def _load_gene_sets(gmt):
    return load_gmt(gmt) if isinstance(gmt, str) else gmt


def _make_unique_gene_names(genes: Sequence[str]) -> np.ndarray:
    seen = {}
    unique = []
    for gene in map(str, genes):
        count = seen.get(gene, 0)
        if count == 0:
            unique.append(gene)
        else:
            unique.append(f"{gene}-{count}")
        seen[gene] = count + 1
    return np.asarray(unique)


def _normalize_ranker(ranker: str) -> str:
    normalized = ranker.lower().replace("-", "_")
    aliases = {
        "mean_difference": "mean_diff",
        "logfc": "mean_diff",
        "welch_t": "t_stat",
        "t": "t_stat",
        "z": "z_score",
        "cohen_d": "cohens_d",
        "detection": "detection_weighted",
        "dropout_weighted": "detection_weighted",
        "slope": "local_slope",
        "gam_slope": "smooth_slope",
        "smoothed_slope": "smooth_slope",
        "neighbor": "neighbor_contrast",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _SUPPORTED_RANKERS:
        supported = ", ".join(sorted(_SUPPORTED_RANKERS))
        raise ValueError(f"Unsupported ranker '{ranker}'. Supported rankers: {supported}")
    return normalized


def _parse_gene_set_weights(gene_set) -> dict[str, float]:
    if isinstance(gene_set, dict):
        return {str(gene): float(weight) for gene, weight in gene_set.items()}

    parsed = {}
    for item in gene_set:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            parsed[str(item[0])] = float(item[1])
            continue
        text = str(item)
        parsed[text] = 1.0
    return parsed


def _prepare_gene_sets_for_mode(
    gmt,
    gene_set_mode: str = "standard",
    min_abs_gene_weight: float = 0.0,
):
    raw = _load_gene_sets(gmt)
    mode = gene_set_mode.lower().replace("-", "_")
    if mode in {"standard", "unsigned"}:
        out = {}
        for name, gene_set in raw.items():
            weights = _parse_gene_set_weights(gene_set)
            out[name] = [
                gene
                for gene, weight in weights.items()
                if abs(float(weight)) >= min_abs_gene_weight
            ]
        return out

    if mode not in {"split_signed", "signed_split"}:
        raise ValueError("gene_set_mode must be one of 'standard' or 'split_signed'")

    out = {}
    for name, gene_set in raw.items():
        weights = _parse_gene_set_weights(gene_set)
        pos = [
            gene
            for gene, weight in weights.items()
            if weight > 0 and abs(weight) >= min_abs_gene_weight
        ]
        neg = [
            gene
            for gene, weight in weights.items()
            if weight < 0 and abs(weight) >= min_abs_gene_weight
        ]
        unsigned = [
            gene
            for gene, weight in weights.items()
            if weight == 0 and abs(weight) >= min_abs_gene_weight
        ]
        if pos:
            out[f"{name}__positive"] = pos
        if neg:
            out[f"{name}__negative"] = neg
        if not pos and not neg and unsigned:
            out[name] = unsigned
    return out


def build_gene_set_index(
    genes: Sequence[str],
    gmt,
    min_size: int = 15,
    max_size: int = 500,
    gene_set_mode: str = "standard",
    min_abs_gene_weight: float = 0.0,
) -> GeneSetIndex:
    genes = np.asarray(genes, dtype=str)
    prepared = _prepare_gene_sets_for_mode(
        gmt,
        gene_set_mode=gene_set_mode,
        min_abs_gene_weight=min_abs_gene_weight,
    )
    pathway_names, pathway_indices = prepare_pathways(genes, prepared, min_size, max_size)
    return GeneSetIndex(
        pathway_names=list(pathway_names),
        pathway_indices=[np.asarray(idx, dtype=int).tolist() for idx in pathway_indices],
        sizes=[int(len(idx)) for idx in pathway_indices],
        gene_universe_hash=_stable_hash(list(map(str, genes))),
        gmt_hash=_stable_hash(prepared),
        min_size=int(min_size),
        max_size=int(max_size),
        params={
            "gene_set_mode": gene_set_mode,
            "min_abs_gene_weight": min_abs_gene_weight,
        },
    )


def _make_cell_count_windows(sorted_idx, window_size, step):
    windows = []
    n = len(sorted_idx)
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if step <= 0:
        raise ValueError("step must be positive")
    for start in range(0, n - window_size + 1, step):
        win = sorted_idx[start : start + window_size]
        windows.append((start, start + window_size, win))
    return windows


def _default_target_span(pt_sorted: np.ndarray) -> float:
    pt_min = float(np.nanmin(pt_sorted))
    pt_max = float(np.nanmax(pt_sorted))
    span = pt_max - pt_min
    if span <= 0:
        return 1.0
    return span * 0.05


def _trim_to_max_cells(lo: int, hi: int, center_pos: int, max_cells: int, n: int):
    count = hi - lo
    if max_cells is None or count <= max_cells:
        return lo, hi

    half = max_cells // 2
    lo = max(0, center_pos - half)
    hi = min(n, lo + max_cells)
    lo = max(0, hi - max_cells)
    return lo, hi


def _expand_to_min_cells(lo: int, hi: int, center_pos: int, min_cells: int, n: int):
    count = hi - lo
    if min_cells is None or count >= min_cells:
        return lo, hi

    half = min_cells // 2
    lo = max(0, center_pos - half)
    hi = min(n, lo + min_cells)
    lo = max(0, hi - min_cells)
    return lo, hi


def _make_pseudotime_span_windows(
    sorted_idx: np.ndarray,
    pt: np.ndarray,
    target_span: Optional[float],
    span_step: Optional[float],
    min_cells: int,
    max_cells: Optional[int],
):
    pt_sorted = pt[sorted_idx]
    target_span = _default_target_span(pt_sorted) if target_span is None else target_span
    if target_span <= 0:
        raise ValueError("target_span must be positive")

    span_step = target_span if span_step is None else span_step
    if span_step <= 0:
        raise ValueError("span_step must be positive")

    windows = []
    seen = set()
    n = len(sorted_idx)
    start = float(pt_sorted.min())
    stop = float(pt_sorted.max())

    while start <= stop:
        end = start + target_span
        lo = int(np.searchsorted(pt_sorted, start, side="left"))
        hi = int(np.searchsorted(pt_sorted, end, side="right"))
        center_pos = int(np.searchsorted(pt_sorted, (start + end) / 2.0, side="left"))
        lo, hi = _trim_to_max_cells(lo, hi, center_pos, max_cells, n)

        if hi > lo and (hi - lo) >= min_cells and (lo, hi) not in seen:
            windows.append((lo, hi, sorted_idx[lo:hi]))
            seen.add((lo, hi))
        start += span_step

    return windows


def _make_adaptive_windows(
    sorted_idx: np.ndarray,
    pt: np.ndarray,
    target_span: Optional[float],
    span_step: Optional[float],
    min_cells: int,
    max_cells: Optional[int],
):
    pt_sorted = pt[sorted_idx]
    target_span = _default_target_span(pt_sorted) if target_span is None else target_span
    if target_span <= 0:
        raise ValueError("target_span must be positive")

    span_step = target_span if span_step is None else span_step
    if span_step <= 0:
        raise ValueError("span_step must be positive")

    windows = []
    seen = set()
    n = len(sorted_idx)
    center = float(pt_sorted.min())
    stop = float(pt_sorted.max())
    half_span = target_span / 2.0

    while center <= stop:
        lo = int(np.searchsorted(pt_sorted, center - half_span, side="left"))
        hi = int(np.searchsorted(pt_sorted, center + half_span, side="right"))
        center_pos = int(np.searchsorted(pt_sorted, center, side="left"))
        lo, hi = _expand_to_min_cells(lo, hi, center_pos, min_cells, n)
        lo, hi = _trim_to_max_cells(lo, hi, center_pos, max_cells, n)

        if hi > lo and (hi - lo) >= min_cells and (lo, hi) not in seen:
            windows.append((lo, hi, sorted_idx[lo:hi]))
            seen.add((lo, hi))
        center += span_step

    return windows


def _make_windows(
    sorted_idx,
    window_size,
    step,
    pt: Optional[np.ndarray] = None,
    window_mode: str = "cell_count",
    min_cells: Optional[int] = None,
    max_cells: Optional[int] = None,
    target_span: Optional[float] = None,
    span_step: Optional[float] = None,
):
    window_mode = window_mode.lower().replace("-", "_")
    if window_mode == "cell_count":
        return _make_cell_count_windows(sorted_idx, window_size, step)

    if pt is None:
        raise ValueError("pt is required for pseudotime_span and adaptive windows")

    min_cells = 1 if min_cells is None else int(min_cells)
    if min_cells <= 0:
        raise ValueError("min_cells must be positive")
    if max_cells is not None:
        max_cells = int(max_cells)
        if max_cells < min_cells:
            raise ValueError("max_cells must be greater than or equal to min_cells")

    if window_mode == "pseudotime_span":
        return _make_pseudotime_span_windows(
            sorted_idx, pt, target_span, span_step, min_cells, max_cells
        )
    if window_mode == "adaptive":
        if max_cells is None:
            max_cells = window_size
        return _make_adaptive_windows(
            sorted_idx, pt, target_span, span_step, min_cells, max_cells
        )

    raise ValueError(
        "window_mode must be one of 'cell_count', 'pseudotime_span', 'adaptive', "
        "or 'graph_adaptive'"
    )


def _graph_neighbors(graph, node: int) -> np.ndarray:
    row = graph.getrow(node) if hasattr(graph, "getrow") else np.asarray(graph[node])
    if hasattr(row, "indices"):
        return row.indices
    return np.where(np.asarray(row).ravel() > 0)[0]


def _graph_distances_from_anchor(
    graph,
    anchor: int,
    radius: int,
    n_obs: int,
    allowed_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    distances = np.full(n_obs, np.inf, dtype=float)
    if allowed_mask is not None and not bool(allowed_mask[anchor]):
        return distances
    distances[anchor] = 0.0
    queue = deque([anchor])
    while queue:
        node = queue.popleft()
        next_distance = distances[node] + 1.0
        if next_distance > radius:
            continue
        for neighbor in _graph_neighbors(graph, int(node)):
            if allowed_mask is not None and not bool(allowed_mask[neighbor]):
                continue
            if not np.isfinite(distances[neighbor]):
                distances[neighbor] = next_distance
                queue.append(int(neighbor))
    return distances


def _weight_entropy(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    weights = weights[np.isfinite(weights) & (weights > 0)]
    if len(weights) <= 1:
        return 0.0
    p = weights / float(weights.sum())
    return float(-np.sum(p * np.log(p)) / np.log(len(p)))


def _make_graph_adaptive_windows(
    adata,
    sorted_idx: np.ndarray,
    pt: np.ndarray,
    graph_key: str,
    graph_radius: int,
    target_span: Optional[float],
    span_step: Optional[float],
    min_cells: int,
    max_cells: Optional[int],
    bandwidth_pt: Optional[float],
    bandwidth_graph: Optional[float],
    branch_key: Optional[str],
    min_branch_purity: float,
    fate_weights: Optional[np.ndarray],
):
    if graph_key not in adata.obsp:
        raise ValueError(f"graph_key '{graph_key}' not found in adata.obsp")
    if graph_radius < 1:
        raise ValueError("graph_radius must be at least 1")
    if not 0 <= min_branch_purity <= 1:
        raise ValueError("min_branch_purity must be in [0, 1]")

    graph = adata.obsp[graph_key]
    if hasattr(graph, "maximum"):
        graph = graph.maximum(graph.T)
    pt_sorted = pt[sorted_idx]
    span = float(np.nanmax(pt_sorted) - np.nanmin(pt_sorted))
    if span <= 0:
        span = 1.0
    target_span = max(span * 0.05, 1e-9) if target_span is None else float(target_span)
    if target_span <= 0:
        raise ValueError("target_span must be positive for graph_adaptive windows")
    span_step = target_span if span_step is None else float(span_step)
    if span_step <= 0:
        raise ValueError("span_step must be positive for graph_adaptive windows")
    bandwidth_pt = target_span if bandwidth_pt is None else float(bandwidth_pt)
    bandwidth_graph = max(float(graph_radius), 1e-9) if bandwidth_graph is None else float(bandwidth_graph)
    if bandwidth_pt <= 0 or bandwidth_graph <= 0:
        raise ValueError("bandwidth_pt and bandwidth_graph must be positive")

    branch_values = None
    if branch_key is not None:
        if branch_key not in adata.obs:
            raise ValueError(f"branch_key '{branch_key}' not found in adata.obs")
        branch_values = adata.obs[branch_key].astype(str).to_numpy()

    min_cells = max(int(min_cells), 1)
    max_cells = None if max_cells is None else int(max_cells)
    windows = []
    weight_map = {}
    diagnostics = []
    seen = set()
    n_obs = len(pt)
    center = float(np.nanmin(pt_sorted))
    stop = float(np.nanmax(pt_sorted))
    window_id = 0

    while center <= stop + 1e-12:
        anchor = int(sorted_idx[np.argmin(np.abs(pt_sorted - center))])
        anchor_pt = float(pt[anchor])
        in_span = np.abs(pt - anchor_pt) <= target_span
        graph_dist = _graph_distances_from_anchor(
            graph,
            anchor,
            graph_radius,
            n_obs,
            allowed_mask=in_span,
        )
        in_graph = np.isfinite(graph_dist) & (graph_dist <= graph_radius)
        candidate = np.where(in_graph & in_span)[0]

        branch_purity = np.nan
        branch_label = ""
        if len(candidate) and branch_values is not None:
            labels, counts = np.unique(branch_values[candidate], return_counts=True)
            winner = int(np.argmax(counts))
            branch_label = str(labels[winner])
            branch_purity = float(counts[winner] / max(len(candidate), 1))

        skipped_reason = ""
        if len(candidate) < min_cells:
            skipped_reason = "too_few_cells"
        elif np.isfinite(branch_purity) and branch_purity < min_branch_purity:
            skipped_reason = "low_branch_purity"

        if skipped_reason:
            diagnostics.append(
                {
                    "window_id": window_id,
                    "anchor_cell": anchor,
                    "anchor_pseudotime": anchor_pt,
                    "n_cells": int(len(candidate)),
                    "effective_n_cells": np.nan,
                    "pseudotime_span": float(np.nanmax(pt[candidate]) - np.nanmin(pt[candidate])) if len(candidate) else 0.0,
                    "graph_radius": int(graph_radius),
                    "mean_graph_distance": float(np.nanmean(graph_dist[candidate])) if len(candidate) else np.nan,
                    "branch_purity": branch_purity,
                    "branch_label": branch_label,
                    "weight_sum": 0.0,
                    "weight_entropy": np.nan,
                    "fate_weight_mean": np.nan,
                    "skipped": True,
                    "skip_reason": skipped_reason,
                }
            )
            center += span_step
            window_id += 1
            continue

        pt_weight = np.exp(-np.abs(pt[candidate] - anchor_pt) / bandwidth_pt)
        graph_weight = np.exp(-graph_dist[candidate] / bandwidth_graph)
        weights = pt_weight * graph_weight
        if fate_weights is not None:
            weights = weights * fate_weights[candidate]

        if max_cells is not None and len(candidate) > max_cells:
            keep = np.argsort(-weights)[:max_cells]
            candidate = candidate[keep]
            weights = weights[keep]

        if (not np.isfinite(weights).all()) or float(weights.sum()) <= 0:
            diagnostics.append(
                {
                    "window_id": window_id,
                    "anchor_cell": anchor,
                    "anchor_pseudotime": anchor_pt,
                    "n_cells": int(len(candidate)),
                    "effective_n_cells": np.nan,
                    "pseudotime_span": float(np.nanmax(pt[candidate]) - np.nanmin(pt[candidate])) if len(candidate) else 0.0,
                    "graph_radius": int(graph_radius),
                    "mean_graph_distance": float(np.nanmean(graph_dist[candidate])) if len(candidate) else np.nan,
                    "branch_purity": branch_purity,
                    "branch_label": branch_label,
                    "weight_sum": 0.0,
                    "weight_entropy": np.nan,
                    "fate_weight_mean": np.nan,
                    "skipped": True,
                    "skip_reason": "zero_weight_sum",
                }
            )
            center += span_step
            window_id += 1
            continue

        key = tuple(sorted(map(int, candidate)))
        if key in seen or len(candidate) < min_cells:
            center += span_step
            window_id += 1
            continue
        seen.add(key)

        weight_sum = float(weights.sum())
        effective_n = float((weight_sum * weight_sum) / max(float(np.square(weights).sum()), 1e-12))
        fate_mean = float(np.nanmean(fate_weights[candidate])) if fate_weights is not None else np.nan
        diagnostics.append(
            {
                "window_id": window_id,
                "anchor_cell": anchor,
                "anchor_pseudotime": anchor_pt,
                "n_cells": int(len(candidate)),
                "effective_n_cells": effective_n,
                "pseudotime_span": float(np.nanmax(pt[candidate]) - np.nanmin(pt[candidate])),
                "graph_radius": int(graph_radius),
                "mean_graph_distance": float(np.average(graph_dist[candidate], weights=np.maximum(weights, 1e-12))),
                "branch_purity": branch_purity,
                "branch_label": branch_label,
                "weight_sum": weight_sum,
                "weight_entropy": _weight_entropy(weights),
                "fate_weight_mean": fate_mean,
                "skipped": False,
                "skip_reason": "",
            }
        )
        windows.append((window_id, window_id + 1, np.asarray(candidate, dtype=int)))
        weight_map[window_id] = (np.asarray(candidate, dtype=int), np.asarray(weights, dtype=float))
        center += span_step
        window_id += 1

    return windows, weight_map, pd.DataFrame(diagnostics)


def build_window_index(
    adata,
    order: np.ndarray,
    pt: np.ndarray,
    window_size: int = 500,
    step: int = 100,
    window_mode: str = "cell_count",
    min_cells: Optional[int] = None,
    max_cells: Optional[int] = None,
    target_span: Optional[float] = None,
    span_step: Optional[float] = None,
    graph_key: str = "connectivities",
    graph_radius: int = 2,
    bandwidth_pt: Optional[float] = None,
    bandwidth_graph: Optional[float] = None,
    branch_key: Optional[str] = None,
    min_branch_purity: float = 0.75,
    fate_weights: Optional[np.ndarray] = None,
) -> WindowIndex:
    window_mode_normalized = window_mode.lower().replace("-", "_")
    graph_weight_map = {}
    graph_diagnostics = pd.DataFrame()
    if window_mode_normalized == "graph_adaptive":
        windows, graph_weight_map, graph_diagnostics = _make_graph_adaptive_windows(
            adata=adata,
            sorted_idx=order,
            pt=pt,
            graph_key=graph_key,
            graph_radius=graph_radius,
            target_span=target_span,
            span_step=span_step,
            min_cells=1 if min_cells is None else int(min_cells),
            max_cells=max_cells,
            bandwidth_pt=bandwidth_pt,
            bandwidth_graph=bandwidth_graph,
            branch_key=branch_key,
            min_branch_purity=min_branch_purity,
            fate_weights=fate_weights,
        )
    else:
        windows = _make_windows(
            order,
            window_size=window_size,
            step=step,
            pt=pt,
            window_mode=window_mode_normalized,
            min_cells=min_cells,
            max_cells=max_cells,
            target_span=target_span,
            span_step=span_step,
        )
    return WindowIndex(
        windows=windows,
        weight_map=graph_weight_map,
        diagnostics=graph_diagnostics,
        window_mode=window_mode_normalized,
        n_obs=len(pt),
        params={
            "window_size": window_size,
            "step": step,
            "min_cells": min_cells,
            "max_cells": max_cells,
            "target_span": target_span,
            "span_step": span_step,
            "graph_key": graph_key,
            "graph_radius": graph_radius,
            "bandwidth_pt": bandwidth_pt,
            "bandwidth_graph": bandwidth_graph,
            "branch_key": branch_key,
            "min_branch_purity": min_branch_purity,
        },
    )


def _axis_sum(X, indices: Optional[Sequence[int]] = None):
    block = X if indices is None else X[indices]
    return np.asarray(block.sum(axis=0)).ravel()


def _axis_sum_squares(X, indices: Optional[Sequence[int]] = None):
    block = X if indices is None else X[indices]
    if hasattr(block, "power"):
        return np.asarray(block.power(2).sum(axis=0)).ravel()
    return np.asarray(np.square(block).sum(axis=0)).ravel()


def _detection_count(X, indices: Optional[Sequence[int]] = None):
    block = X if indices is None else X[indices]
    return np.asarray((block > 0).sum(axis=0)).ravel()


def _axis_weighted_sum(X, weights: np.ndarray, indices: Optional[Sequence[int]] = None):
    block = X if indices is None else X[indices]
    w = weights if indices is None else weights[indices]
    return np.asarray(block.T @ w).ravel()


def _axis_weighted_sum_squares(
    X,
    weights: np.ndarray,
    indices: Optional[Sequence[int]] = None,
):
    block = X if indices is None else X[indices]
    w = weights if indices is None else weights[indices]
    if hasattr(block, "power"):
        return np.asarray(block.power(2).T @ w).ravel()
    return np.asarray(np.square(block).T @ w).ravel()


def _weighted_detection_sum(X, weights: np.ndarray, indices: Optional[Sequence[int]] = None):
    block = X if indices is None else X[indices]
    w = weights if indices is None else weights[indices]
    return np.asarray((block > 0).T @ w).ravel()


def _window_moments(
    X,
    window_indices,
    n_all,
    weights=None,
    weight_total=None,
    weight_sq_total=None,
    need_sum_sq: bool = False,
    need_detection: bool = False,
) -> dict:
    moments = {
        "n_in": int(len(window_indices)),
        "n_out": int(n_all - len(window_indices)),
    }
    if weights is not None:
        window_weights = weights[window_indices]
        w_in = float(window_weights.sum())
        w_sq_in = float(np.square(window_weights).sum())
        w_out = float(weight_total - w_in)
        w_sq_out = float(weight_sq_total - w_sq_in) if weight_sq_total is not None else 0.0
        moments.update(
            {
                "w_in": w_in,
                "w_out": w_out,
                "w_sq_in": w_sq_in,
                "w_sq_out": w_sq_out,
            }
        )
        if w_in > 0:
            moments["sum_in"] = _axis_weighted_sum(X, weights, window_indices)
            if need_sum_sq:
                moments["sum_sq_in"] = _axis_weighted_sum_squares(
                    X, weights, window_indices
                )
            if need_detection:
                moments["det_in"] = _weighted_detection_sum(X, weights, window_indices)
        return moments

    moments["sum_in"] = _axis_sum(X, window_indices)
    if need_sum_sq:
        moments["sum_sq_in"] = _axis_sum_squares(X, window_indices)
    if need_detection:
        moments["det_in"] = _detection_count(X, window_indices)
    return moments


def _effective_weight_n(weight_sum: float, weight_sq_sum: float) -> float:
    if weight_sum <= 0 or weight_sq_sum <= 0:
        return 0.0
    return float((weight_sum * weight_sum) / weight_sq_sum)


def _mean_and_var(sum_values, sum_squares, n):
    if n <= 0:
        mean = np.zeros_like(sum_values, dtype=np.float64)
        var = np.zeros_like(sum_values, dtype=np.float64)
        return mean, var

    mean = sum_values / n
    if n == 1:
        var = np.zeros_like(mean, dtype=np.float64)
    else:
        var = (sum_squares - (sum_values * sum_values / n)) / (n - 1)
        var = np.maximum(var, 0.0)
    return mean, var


def _weighted_mean_and_var(sum_values, sum_squares, weight_sum, weight_sq_sum):
    if weight_sum <= 0:
        mean = np.zeros_like(sum_values, dtype=np.float64)
        var = np.zeros_like(sum_values, dtype=np.float64)
        return mean, var, 0.0

    mean = sum_values / weight_sum
    pop_var = np.maximum(sum_squares / weight_sum - mean * mean, 0.0)
    n_eff = _effective_weight_n(weight_sum, weight_sq_sum)
    if n_eff > 1:
        var = pop_var * n_eff / (n_eff - 1.0)
    else:
        var = np.zeros_like(mean, dtype=np.float64)
    return mean, var, n_eff


def _rank_logfc_fast(
    X,
    window_indices,
    sum_total,
    n_all,
    weights=None,
    weight_total=None,
    moments=None,
):
    moments = moments or _window_moments(
        X, window_indices, n_all, weights=weights, weight_total=weight_total
    )
    if weights is not None:
        w_in = float(moments["w_in"])
        w_out = float(moments["w_out"])
        if w_in <= 0 or w_out <= 0:
            return np.zeros(X.shape[1], dtype=np.float64)
        sum_in = moments["sum_in"]
        mu_in = sum_in / w_in
        mu_out = (sum_total - sum_in) / w_out
        return mu_in - mu_out

    n_in = moments["n_in"]
    n_out = moments["n_out"]
    sum_in = moments["sum_in"]
    mu_in = sum_in / max(n_in, 1)
    mu_out = (sum_total - sum_in) / max(n_out, 1)
    return mu_in - mu_out


def _rank_t_stat(
    X,
    window_indices,
    sum_total,
    sum_sq_total,
    n_all,
    weights=None,
    weight_total=None,
    weight_sq_total=None,
    moments=None,
):
    moments = moments or _window_moments(
        X,
        window_indices,
        n_all,
        weights=weights,
        weight_total=weight_total,
        weight_sq_total=weight_sq_total,
        need_sum_sq=True,
    )
    if weights is not None:
        w_in = float(moments["w_in"])
        w_sq_in = float(moments["w_sq_in"])
        w_out = float(moments["w_out"])
        w_sq_out = float(moments["w_sq_out"])
        if w_in <= 0 or w_out <= 0:
            return np.zeros(X.shape[1], dtype=np.float64)
        sum_in = moments["sum_in"]
        sum_sq_in = moments["sum_sq_in"]
        mu_in, var_in, n_eff_in = _weighted_mean_and_var(sum_in, sum_sq_in, w_in, w_sq_in)
        mu_out, var_out, n_eff_out = _weighted_mean_and_var(
            sum_total - sum_in,
            sum_sq_total - sum_sq_in,
            w_out,
            w_sq_out,
        )
        denom = np.sqrt(
            var_in / max(n_eff_in, 1.0) + var_out / max(n_eff_out, 1.0)
        )
        return (mu_in - mu_out) / np.maximum(denom, 1e-12)

    n_in = moments["n_in"]
    n_out = moments["n_out"]
    sum_in = moments["sum_in"]
    sum_sq_in = moments["sum_sq_in"]
    mu_in, var_in = _mean_and_var(sum_in, sum_sq_in, n_in)
    mu_out, var_out = _mean_and_var(sum_total - sum_in, sum_sq_total - sum_sq_in, n_out)
    denom = np.sqrt(var_in / max(n_in, 1) + var_out / max(n_out, 1))
    return (mu_in - mu_out) / np.maximum(denom, 1e-12)


def _rank_cohens_d(
    X,
    window_indices,
    sum_total,
    sum_sq_total,
    n_all,
    weights=None,
    weight_total=None,
    weight_sq_total=None,
    moments=None,
):
    moments = moments or _window_moments(
        X,
        window_indices,
        n_all,
        weights=weights,
        weight_total=weight_total,
        weight_sq_total=weight_sq_total,
        need_sum_sq=True,
    )
    if weights is not None:
        w_in = float(moments["w_in"])
        w_sq_in = float(moments["w_sq_in"])
        w_out = float(moments["w_out"])
        w_sq_out = float(moments["w_sq_out"])
        if w_in <= 0 or w_out <= 0:
            return np.zeros(X.shape[1], dtype=np.float64)
        sum_in = moments["sum_in"]
        sum_sq_in = moments["sum_sq_in"]
        mu_in, var_in, n_eff_in = _weighted_mean_and_var(sum_in, sum_sq_in, w_in, w_sq_in)
        mu_out, var_out, n_eff_out = _weighted_mean_and_var(
            sum_total - sum_in,
            sum_sq_total - sum_sq_in,
            w_out,
            w_sq_out,
        )
        pooled_num = max(n_eff_in - 1.0, 0.0) * var_in + max(n_eff_out - 1.0, 0.0) * var_out
        pooled_den = max(n_eff_in + n_eff_out - 2.0, 1.0)
        pooled_sd = np.sqrt(pooled_num / pooled_den)
        return (mu_in - mu_out) / np.maximum(pooled_sd, 1e-12)

    n_in = moments["n_in"]
    n_out = moments["n_out"]
    sum_in = moments["sum_in"]
    sum_sq_in = moments["sum_sq_in"]
    mu_in, var_in = _mean_and_var(sum_in, sum_sq_in, n_in)
    mu_out, var_out = _mean_and_var(sum_total - sum_in, sum_sq_total - sum_sq_in, n_out)
    pooled_num = (max(n_in - 1, 0) * var_in) + (max(n_out - 1, 0) * var_out)
    pooled_den = max(n_in + n_out - 2, 1)
    pooled_sd = np.sqrt(pooled_num / pooled_den)
    return (mu_in - mu_out) / np.maximum(pooled_sd, 1e-12)


def _rank_detection_weighted(
    X,
    window_indices,
    sum_total,
    det_total,
    n_all,
    weights=None,
    weight_total=None,
    moments=None,
):
    moments = moments or _window_moments(
        X,
        window_indices,
        n_all,
        weights=weights,
        weight_total=weight_total,
        need_detection=True,
    )
    if weights is not None:
        w_in = float(moments["w_in"])
        w_out = float(moments["w_out"])
        if w_in <= 0 or w_out <= 0:
            return np.zeros(X.shape[1], dtype=np.float64)
        sum_in = moments["sum_in"]
        mean_diff = (sum_in / w_in) - ((sum_total - sum_in) / w_out)

        det_in_sum = moments["det_in"]
        det_in = det_in_sum / w_in
        det_out = (det_total - det_in_sum) / w_out
        observed = np.maximum(det_in, det_out)
        detection_contrast = np.abs(det_in - det_out)

        weight = np.sqrt(np.clip(observed, 0.0, 1.0)) * (
            0.25 + 0.75 * np.clip(detection_contrast, 0.0, 1.0)
        )
        return mean_diff * weight

    n_in = moments["n_in"]
    n_out = moments["n_out"]
    sum_in = moments["sum_in"]
    mean_diff = (sum_in / max(n_in, 1)) - ((sum_total - sum_in) / max(n_out, 1))

    det_in_count = moments["det_in"]
    det_in = det_in_count / max(n_in, 1)
    det_out = (det_total - det_in_count) / max(n_out, 1)
    observed = np.maximum(det_in, det_out)
    detection_contrast = np.abs(det_in - det_out)

    weight = np.sqrt(np.clip(observed, 0.0, 1.0)) * (
        0.25 + 0.75 * np.clip(detection_contrast, 0.0, 1.0)
    )
    return mean_diff * weight


def _rank_local_slope(X, window_indices, pt, weights=None):
    pt_window = np.asarray(pt[window_indices], dtype=np.float64)
    if weights is not None:
        w = np.asarray(weights[window_indices], dtype=np.float64)
        w_sum = float(w.sum())
        if w_sum <= 0:
            return np.zeros(X.shape[1], dtype=np.float64)
        centered_pt = pt_window - float(np.dot(w, pt_window) / w_sum)
        weighted_centered = w * centered_pt
        denom = float(np.dot(weighted_centered, centered_pt))
    else:
        centered_pt = pt_window - pt_window.mean()
        weighted_centered = centered_pt
        denom = float(np.dot(centered_pt, centered_pt))
    if denom <= 1e-12:
        return np.zeros(X.shape[1], dtype=np.float64)

    block = X[window_indices]
    numerator = np.asarray(block.T @ weighted_centered).ravel()
    return numerator / denom


def _rank_smooth_slope(X, pt, center, bandwidth, weights=None):
    pt = np.asarray(pt, dtype=np.float64)
    if bandwidth is None:
        span = float(np.nanmax(pt) - np.nanmin(pt))
        bandwidth = max(span * 0.15, 1e-6)
    if bandwidth <= 0:
        raise ValueError("smooth_slope_bandwidth must be positive")

    x = pt - float(center)
    kernel = np.exp(-0.5 * np.square(x / bandwidth))
    if weights is not None:
        kernel = kernel * np.asarray(weights, dtype=np.float64)

    weight_sum = float(kernel.sum())
    if weight_sum <= 0:
        return np.zeros(X.shape[1], dtype=np.float64)

    x_centered = x - float(np.dot(kernel, x) / weight_sum)
    weighted_x = kernel * x_centered
    denom = float(np.dot(weighted_x, x_centered))
    if denom <= 1e-12:
        return np.zeros(X.shape[1], dtype=np.float64)

    numerator = np.asarray(X.T @ weighted_x).ravel()
    return numerator / denom


def _rank_neighbor_contrast(X, window_indices, neighbor_indices, weights=None):
    if neighbor_indices is None or len(neighbor_indices) == 0:
        return np.zeros(X.shape[1], dtype=np.float64)

    if weights is not None:
        w_in = float(weights[window_indices].sum())
        w_neighbor = float(weights[neighbor_indices].sum())
        if w_in <= 0 or w_neighbor <= 0:
            return np.zeros(X.shape[1], dtype=np.float64)
        mean_window = _axis_weighted_sum(X, weights, window_indices) / w_in
        mean_neighbor = _axis_weighted_sum(X, weights, neighbor_indices) / w_neighbor
        return mean_window - mean_neighbor

    mean_window = _axis_sum(X, window_indices) / max(len(window_indices), 1)
    mean_neighbor = _axis_sum(X, neighbor_indices) / max(len(neighbor_indices), 1)
    return mean_window - mean_neighbor


def _rank_wilcoxon(X, window_indices, n_all, weights=None):
    try:
        from scipy.stats import rankdata
    except ImportError as exc:
        raise ImportError("ranker='wilcoxon' requires scipy") from exc

    values = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    window_mask = np.zeros(n_all, dtype=bool)
    window_mask[window_indices] = True
    n_in = int(window_mask.sum())
    n_out = n_all - n_in
    if n_in == 0 or n_out == 0:
        return np.zeros(values.shape[1], dtype=np.float64)

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        w_in = float(w[window_mask].sum())
        w_out = float(w[~window_mask].sum())
        if w_in <= 0 or w_out <= 0:
            return np.zeros(values.shape[1], dtype=np.float64)
        n_eff_in = _effective_weight_n(w_in, float(np.square(w[window_mask]).sum()))
        n_eff_out = _effective_weight_n(w_out, float(np.square(w[~window_mask]).sum()))
        scores = np.zeros(values.shape[1], dtype=np.float64)
        for gene_idx in range(values.shape[1]):
            ranks = rankdata(values[:, gene_idx], method="average")
            mean_in = float(np.dot(w[window_mask], ranks[window_mask]) / w_in)
            mean_out = float(np.dot(w[~window_mask], ranks[~window_mask]) / w_out)
            mean_all = float(np.dot(w, ranks) / w.sum())
            var_all = float(np.dot(w, np.square(ranks - mean_all)) / w.sum())
            denom = np.sqrt(var_all * (1.0 / max(n_eff_in, 1.0) + 1.0 / max(n_eff_out, 1.0)))
            scores[gene_idx] = (mean_in - mean_out) / max(denom, 1e-12)
        return scores

    expected_u = n_in * n_out / 2.0
    sd_u = np.sqrt(n_in * n_out * (n_in + n_out + 1) / 12.0)
    scores = np.zeros(values.shape[1], dtype=np.float64)
    for gene_idx in range(values.shape[1]):
        ranks = rankdata(values[:, gene_idx], method="average")
        u_stat = ranks[window_mask].sum() - (n_in * (n_in + 1) / 2.0)
        scores[gene_idx] = (u_stat - expected_u) / max(sd_u, 1e-12)
    return scores


def _neighbor_indices(order, start, end):
    width = end - start
    left = order[max(0, start - width) : start]
    right = order[end : min(len(order), end + width)]
    if len(left) == 0:
        return right
    if len(right) == 0:
        return left
    return np.concatenate([left, right])


def _rank_gene_scores(
    X,
    window_indices,
    ranker,
    sum_total,
    n_all,
    sum_sq_total=None,
    det_total=None,
    pt=None,
    neighbor_indices=None,
    weights=None,
    weight_total=None,
    weight_sq_total=None,
    smooth_center=None,
    smooth_bandwidth=None,
):
    moment_rankers = {
        "mean_diff",
        "t_stat",
        "z_score",
        "cohens_d",
        "detection_weighted",
    }
    moments = None
    if ranker in moment_rankers:
        moments = _window_moments(
            X,
            window_indices,
            n_all,
            weights=weights,
            weight_total=weight_total,
            weight_sq_total=weight_sq_total,
            need_sum_sq=ranker in {"t_stat", "z_score", "cohens_d"},
            need_detection=ranker == "detection_weighted",
        )
    if ranker == "mean_diff":
        return _rank_logfc_fast(
            X,
            window_indices,
            sum_total,
            n_all,
            weights=weights,
            weight_total=weight_total,
            moments=moments,
        )
    if ranker in {"t_stat", "z_score"}:
        return _rank_t_stat(
            X,
            window_indices,
            sum_total,
            sum_sq_total,
            n_all,
            weights=weights,
            weight_total=weight_total,
            weight_sq_total=weight_sq_total,
            moments=moments,
        )
    if ranker == "cohens_d":
        return _rank_cohens_d(
            X,
            window_indices,
            sum_total,
            sum_sq_total,
            n_all,
            weights=weights,
            weight_total=weight_total,
            weight_sq_total=weight_sq_total,
            moments=moments,
        )
    if ranker == "detection_weighted":
        return _rank_detection_weighted(
            X,
            window_indices,
            sum_total,
            det_total,
            n_all,
            weights=weights,
            weight_total=weight_total,
            moments=moments,
        )
    if ranker == "local_slope":
        return _rank_local_slope(X, window_indices, pt, weights=weights)
    if ranker == "smooth_slope":
        return _rank_smooth_slope(
            X,
            pt=pt,
            center=smooth_center,
            bandwidth=smooth_bandwidth,
            weights=weights,
        )
    if ranker == "neighbor_contrast":
        return _rank_neighbor_contrast(
            X, window_indices, neighbor_indices, weights=weights
        )
    if ranker == "wilcoxon":
        return _rank_wilcoxon(X, window_indices, n_all, weights=weights)

    raise ValueError(f"Unsupported ranker '{ranker}'")


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
    bin_width: Optional[int] = 0,
    calculate_nes: bool = True,
    use_nes_cache: bool = False,
    mode: str = "aligned",
    score_type: str = "std",
    tie_policy: str = "gene_id",
    nperm_simple: Optional[int] = 1000,
    max_levels: Optional[int] = None,
    ranker: str = "mean_diff",
    window_mode: str = "cell_count",
    min_cells: Optional[int] = None,
    max_cells: Optional[int] = None,
    target_span: Optional[float] = None,
    span_step: Optional[float] = None,
    return_leading_edge: bool = False,
    gsea_param: float = 1.0,
    layer: Optional[str] = None,
    use_raw: bool = False,
    dropna: bool = True,
    make_var_names_unique: bool = False,
    cell_weight_key: Optional[str] = None,
    smooth_slope_bandwidth: Optional[float] = None,
    gene_set_mode: str = "standard",
    min_abs_gene_weight: float = 0.0,
    graph_key: str = "connectivities",
    graph_radius: int = 2,
    bandwidth_pt: Optional[float] = None,
    bandwidth_graph: Optional[float] = None,
    branch_key: Optional[str] = None,
    min_branch_purity: float = 0.75,
    experimental: bool = False,
    gene_set_index: Optional[GeneSetIndex] = None,
    window_index: Optional[WindowIndex] = None,
) -> pd.DataFrame:
    """
    Rolling-window GSEA along pseudotime (Trajectory Analysis).
    """
    gsea_options = _normalize_run_options(
        sample_size=sample_size,
        seed=seed,
        nperm_nes=nperm_nes,
        gsea_param=gsea_param,
        eps=eps,
        score_type=score_type,
        mode=mode,
        tie_policy=tie_policy,
        bin_width=bin_width,
        calculate_nes=calculate_nes,
        nperm_simple=nperm_simple,
        max_levels=max_levels,
    )
    sample_size = gsea_options["sample_size"]
    seed = gsea_options["seed"]
    nperm_nes = gsea_options["nperm_nes"]
    gsea_param = gsea_options["gsea_param"]
    eps = gsea_options["eps"]
    score_type = gsea_options["score_type"]
    mode = gsea_options["mode"]
    tie_policy = gsea_options["tie_policy"]
    bin_width = gsea_options["bin_width"]
    nperm_simple = gsea_options["nperm_simple"]
    max_levels = gsea_options["max_levels"]
    if not isinstance(use_nes_cache, (bool, np.bool_)):
        raise TypeError("use_nes_cache must be boolean")

    if not HAS_SCANPY:
        raise ImportError("scanpy is required for trajectory analysis")

    if isinstance(adata, str):
        adata = sc.read_h5ad(adata)

    adata = _subset_lineage(adata, lineage_col, lineage_keyword)

    if pseudotime_key not in adata.obs:
        adata = _compute_dpt(adata, root_gene=root_gene)
    elif root_gene is not None:
        # If root_gene is explicitly provided, we assume the user wants to recompute DPT
        # based on this new root, even if pseudotime_key exists.
        logger.info(
            f"Pseudotime key '{pseudotime_key}' exists, but root_gene provided. Recomputing DPT..."
        )
        # Remove existing key to force recompute in _compute_dpt or handle inside
        if "dpt_pseudotime" in adata.obs:
            del adata.obs["dpt_pseudotime"]
        adata = _compute_dpt(adata, root_gene=root_gene)

    pt = adata.obs[pseudotime_key].to_numpy()
    ok = np.isfinite(pt)
    if not ok.all():
        n_bad = int((~ok).sum())
        if not dropna:
            raise ValueError(
                f"pseudotime_key '{pseudotime_key}' contains {n_bad} non-finite values; "
                "pass dropna=True to drop those cells."
            )
        logger.warning(f"Dropping {n_bad} cells with non-finite pseudotime.")
        adata = adata[ok].copy()
        pt = pt[ok]

    cell_weights = None
    if cell_weight_key is not None:
        if cell_weight_key not in adata.obs:
            raise ValueError(f"cell_weight_key '{cell_weight_key}' not found in adata.obs")
        cell_weights = pd.to_numeric(
            adata.obs[cell_weight_key], errors="coerce"
        ).to_numpy(dtype=np.float64)
        if not np.isfinite(cell_weights).all():
            raise ValueError(f"cell_weight_key '{cell_weight_key}' contains non-finite values")
        if (cell_weights < 0).any():
            raise ValueError(f"cell_weight_key '{cell_weight_key}' contains negative weights")
        if float(cell_weights.sum()) <= 0:
            raise ValueError(f"cell_weight_key '{cell_weight_key}' sums to zero")

    X, genes, expression_source = _expression_matrix(adata, layer=layer, use_raw=use_raw)
    genes = np.asarray(genes)
    duplicated = pd.Index(genes).duplicated()
    if duplicated.any():
        if not make_var_names_unique:
            examples = ", ".join(map(str, pd.Index(genes)[duplicated][:5]))
            raise ValueError(
                "Expression gene names are duplicated. "
                "Pass make_var_names_unique=True or fix adata.var_names before running. "
                f"Examples: {examples}"
            )
        logger.warning("Expression gene names are duplicated; making names unique for this run.")
        genes = _make_unique_gene_names(genes)

    ranker = _normalize_ranker(ranker)
    window_mode_normalized = window_mode.lower().replace("-", "_")
    order = np.argsort(pt)
    if window_index is not None:
        if window_index.n_obs not in {0, len(pt)}:
            raise ValueError(
                "window_index was built for a different number of cells; rebuild it "
                "after filtering/subsetting adata."
            )
        windows = window_index.windows
        graph_weight_map = window_index.weight_map
        graph_diagnostics = window_index.diagnostics
        window_mode_normalized = window_index.window_mode
    else:
        window_index = build_window_index(
            adata=adata,
            order=order,
            pt=pt,
            window_size=window_size,
            step=step,
            window_mode=window_mode_normalized,
            min_cells=min_cells,
            max_cells=max_cells,
            target_span=target_span,
            span_step=span_step,
            graph_key=graph_key,
            graph_radius=graph_radius,
            bandwidth_pt=bandwidth_pt,
            bandwidth_graph=bandwidth_graph,
            branch_key=branch_key,
            min_branch_purity=min_branch_purity,
            fate_weights=cell_weights,
        )
        windows = window_index.windows
        graph_weight_map = window_index.weight_map
        graph_diagnostics = window_index.diagnostics
    if window_mode_normalized == "graph_adaptive" and not experimental:
        logger.warning(
            "window_mode='graph_adaptive' is experimental; pass experimental=True "
            "to mark analyses that intentionally use topology-aware windows."
        )
    logger.info(
        f"Windows: {len(windows)} (mode={window_mode_normalized}, size={window_size}, "
        f"step={step}, ranker={ranker})"
    )

    if gene_set_index is not None:
        if gene_set_index.gene_universe_hash != _stable_hash(list(map(str, genes))):
            raise ValueError(
                "gene_set_index was built for a different gene universe; rebuild it "
                "for this AnnData/layer/raw expression source."
            )
    else:
        gene_set_index = build_gene_set_index(
            genes,
            gmt_path,
            min_size=min_size,
            max_size=max_size,
            gene_set_mode=gene_set_mode,
            min_abs_gene_weight=min_abs_gene_weight,
        )
    pathway_names = gene_set_index.pathway_names
    pathway_indices = [np.asarray(idx, dtype=int) for idx in gene_set_index.pathway_indices]
    graph_mode = window_mode_normalized == "graph_adaptive"

    if not pathway_indices:
        logger.warning("No valid pathways after gene overlap and size filtering.")
        empty = pd.DataFrame()
        if graph_mode:
            empty.attrs["graph_window_diagnostics"] = graph_diagnostics
        empty.attrs["gene_set_index"] = gene_set_index
        empty.attrs["window_index"] = window_index
        return empty

    # Initialize Runner
    runner = GseaRunner(
        pathway_names,
        pathway_indices,
        min_size,
        max_size,
        gene_ids=genes,
        tie_policy=tie_policy,
    )

    n_all = X.shape[0]
    weight_total = float(cell_weights.sum()) if cell_weights is not None and not graph_mode else None
    weight_sq_total = (
        float(np.square(cell_weights).sum())
        if cell_weights is not None and not graph_mode
        else None
    )
    sum_total = (
        _axis_weighted_sum(X, cell_weights)
        if cell_weights is not None and not graph_mode
        else _axis_sum(X)
        if not graph_mode
        else None
    )
    sum_sq_total = (
        _axis_weighted_sum_squares(X, cell_weights)
        if cell_weights is not None and not graph_mode and ranker in {"t_stat", "z_score", "cohens_d"}
        else _axis_sum_squares(X)
        if not graph_mode and ranker in {"t_stat", "z_score", "cohens_d"}
        else None
    )
    det_total = (
        _weighted_detection_sum(X, cell_weights)
        if cell_weights is not None and not graph_mode and ranker == "detection_weighted"
        else _detection_count(X)
        if not graph_mode and ranker == "detection_weighted"
        else None
    )
    graph_diag_by_id = (
        graph_diagnostics.set_index("window_id").to_dict("index")
        if not graph_diagnostics.empty and "window_id" in graph_diagnostics
        else {}
    )

    all_rows = []

    import time

    t_start = time.time()

    logger.info(
        f"Starting GSEA loop (Caching: {use_nes_cache}, nperm_nes: {nperm_nes})..."
    )

    for loop_idx, (s, e, window_indices) in enumerate(windows):
        wi = int(s) if graph_mode else loop_idx
        graph_info = graph_diag_by_id.get(wi, {})
        rank_weights = cell_weights
        rank_weight_total = weight_total
        rank_weight_sq_total = weight_sq_total
        rank_sum_total = sum_total
        rank_sum_sq_total = sum_sq_total
        rank_det_total = det_total
        if graph_mode:
            local_window, local_weights = graph_weight_map[wi]
            if ranker == "smooth_slope":
                rank_weights = np.zeros(n_all, dtype=np.float64)
            elif cell_weights is None:
                rank_weights = np.ones(n_all, dtype=np.float64)
            else:
                rank_weights = np.asarray(cell_weights, dtype=np.float64).copy()
            rank_weights[local_window] = local_weights
            rank_weight_total = float(rank_weights.sum())
            rank_weight_sq_total = float(np.square(rank_weights).sum())
            rank_sum_total = _axis_weighted_sum(X, rank_weights)
            rank_sum_sq_total = (
                _axis_weighted_sum_squares(X, rank_weights)
                if ranker in {"t_stat", "z_score", "cohens_d"}
                else None
            )
            rank_det_total = (
                _weighted_detection_sum(X, rank_weights)
                if ranker == "detection_weighted"
                else None
            )

        if ranker == "neighbor_contrast" and graph_mode:
            anchor_pt = float(graph_info.get("anchor_pseudotime", np.nanmedian(pt[window_indices])))
            flank_span = float(target_span) if target_span is not None else _default_target_span(pt[order])
            window_mask = np.zeros(n_all, dtype=bool)
            window_mask[window_indices] = True
            neighbors = np.where((np.abs(pt - anchor_pt) <= flank_span * 2.0) & ~window_mask)[0]
        else:
            neighbors = _neighbor_indices(order, s, e) if ranker == "neighbor_contrast" else None
        pt_vals = pt[window_indices]
        rank_vector = _rank_gene_scores(
            X,
            window_indices,
            ranker=ranker,
            sum_total=rank_sum_total,
            n_all=n_all,
            sum_sq_total=rank_sum_sq_total,
            det_total=rank_det_total,
            pt=pt,
            neighbor_indices=neighbors,
            weights=rank_weights,
            weight_total=rank_weight_total,
            weight_sq_total=rank_weight_sq_total,
            smooth_center=float(np.nanmedian(pt_vals)),
            smooth_bandwidth=smooth_slope_bandwidth,
        )
        scores = np.asarray(rank_vector, dtype=np.float64)
        scores[~np.isfinite(scores)] = 0.0

        # The master seed is stable across windows. The Rust core derives task
        # streams from the ranking hash and exact pathway size.
        res = runner.run(
            scores,
            sample_size=sample_size,
            seed=seed,
            eps=eps,
            nperm_nes=nperm_nes,
            gsea_param=gsea_param,
            score_type=score_type,
            bin_width=bin_width,
            calculate_nes=calculate_nes,
            use_nes_cache=use_nes_cache,
            mode=mode,
            nperm_simple=nperm_simple,
            max_levels=max_levels,
        )

        if not res.empty:
            if return_leading_edge:
                leading_scores = runner.prepare_scores(scores)
                leading_genes = runner.prepared_gene_ids
                if leading_genes is None:
                    leading_genes = genes
                leading_edges = compute_leading_edges(
                    leading_genes,
                    leading_scores,
                    pathway_names,
                    runner.pathway_indices,
                    gsea_param=gsea_param,
                )
                res["leading_edge"] = res["Pathway"].map(leading_edges).fillna("")
                res["leading_edge_size"] = res["leading_edge"].map(
                    lambda value: 0 if value == "" else len(value.split(";"))
                )

            res["window_id"] = wi
            res["pt_start"] = pt_vals.min()
            res["pt_end"] = pt_vals.max()
            res["pt_mid"] = (res["pt_start"] + res["pt_end"]) / 2.0
            res["n_cells"] = len(window_indices)
            if graph_mode:
                for key in (
                    "anchor_pseudotime",
                    "effective_n_cells",
                    "pseudotime_span",
                    "graph_radius",
                    "mean_graph_distance",
                    "branch_purity",
                    "branch_label",
                    "weight_sum",
                    "weight_entropy",
                    "fate_weight_mean",
                ):
                    res[key] = graph_info.get(key, np.nan)
                res["experimental"] = bool(experimental)
            elif cell_weights is not None:
                window_weights = cell_weights[window_indices]
                res["weight_sum"] = float(window_weights.sum())
                res["weight_mean"] = float(window_weights.mean())
            res["ranker"] = ranker
            res["window_mode"] = window_mode_normalized
            all_rows.append(res)

        if loop_idx % 10 == 0 and loop_idx > 0:
            elapsed = time.time() - t_start
            fps = (loop_idx + 1) / elapsed
            print(
                f"Processed {loop_idx + 1}/{len(windows)} windows ({fps:.1f} win/s)...",
                end="\r",
            )

    print("\nDone.")
    if not all_rows:
        empty = pd.DataFrame()
        if graph_mode:
            empty.attrs["graph_window_diagnostics"] = graph_diagnostics
        empty.attrs["gene_set_index"] = gene_set_index
        empty.attrs["window_index"] = window_index
        empty.attrs["trajectory_params"] = {
            "ranker": ranker,
            "window_mode": window_mode_normalized,
            "window_size": window_size,
            "step": step,
            "min_cells": min_cells,
            "max_cells": max_cells,
            "target_span": target_span,
            "span_step": span_step,
            "pseudotime_key": pseudotime_key,
            "min_size": min_size,
            "max_size": max_size,
            "sample_size": sample_size,
            "seed": seed,
            "eps": eps,
            "nperm_nes": nperm_nes,
            "bin_width": bin_width,
            "calculate_nes": calculate_nes,
            "use_nes_cache": use_nes_cache,
            "mode": mode,
            "score_type": score_type,
            "tie_policy": tie_policy,
            "nperm_simple": nperm_simple,
            "max_levels": max_levels,
            "approximate_mode": bool(
                mode == "fast"
                or bin_width > 0
                or score_type in {"two_sided_abs", "one_sided_signed"}
            ),
            "gsea_param": gsea_param,
            "graph_key": graph_key,
            "graph_radius": graph_radius,
            "bandwidth_pt": bandwidth_pt,
            "bandwidth_graph": bandwidth_graph,
            "branch_key": branch_key,
            "min_branch_purity": min_branch_purity,
            "experimental": experimental,
        }
        return empty

    df = pd.concat(all_rows, ignore_index=True)
    df.attrs["trajectory_params"] = {
        "ranker": ranker,
        "window_mode": window_mode_normalized,
        "window_size": window_size,
        "step": step,
        "min_cells": min_cells,
        "max_cells": max_cells,
        "target_span": target_span,
        "span_step": span_step,
        "pseudotime_key": pseudotime_key,
        "min_size": min_size,
        "max_size": max_size,
        "sample_size": sample_size,
        "seed": seed,
        "eps": eps,
        "nperm_nes": nperm_nes,
        "bin_width": bin_width,
        "calculate_nes": calculate_nes,
        "use_nes_cache": use_nes_cache,
        "mode": mode,
        "score_type": score_type,
        "tie_policy": tie_policy,
        "nperm_simple": nperm_simple,
        "max_levels": max_levels,
        "approximate_mode": bool(
            mode == "fast"
            or bin_width > 0
            or score_type in {"two_sided_abs", "one_sided_signed"}
        ),
        "return_leading_edge": return_leading_edge,
        "gsea_param": gsea_param,
        "expression_source": expression_source,
        "layer": layer,
        "use_raw": use_raw,
        "dropna": dropna,
        "make_var_names_unique": make_var_names_unique,
        "cell_weight_key": cell_weight_key,
        "smooth_slope_bandwidth": smooth_slope_bandwidth,
        "gene_set_mode": gene_set_mode,
        "min_abs_gene_weight": min_abs_gene_weight,
        "graph_key": graph_key,
        "graph_radius": graph_radius,
        "bandwidth_pt": bandwidth_pt,
        "bandwidth_graph": bandwidth_graph,
        "branch_key": branch_key,
        "min_branch_purity": min_branch_purity,
        "experimental": experimental,
    }
    if graph_mode:
        df.attrs["graph_window_diagnostics"] = graph_diagnostics
    df.attrs["gene_set_index"] = gene_set_index
    df.attrs["window_index"] = window_index
    if out_csv:
        dirpath = os.path.dirname(out_csv)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        df.to_csv(out_csv, index=False)

    return df
