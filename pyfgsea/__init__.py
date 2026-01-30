from .wrapper import (
    run_gsea,
    load_gmt,
    GseaRunner,
    prepare_pathways,
    get_random_es_means,
)

__version__ = "0.1.3"

try:
    from .wrapper import run_scanpy  # type: ignore
except Exception:
    pass

from .trajectory import run_trajectory_gsea
from .plotting import plot_trajectory_heatmap, plot_pathway_dynamics

# Explicitly expose API to top level
__all__ = [
    "run_gsea",
    "load_gmt",
    "run_scanpy",
    "GseaRunner",
    "prepare_pathways",
    "get_random_es_means",
    "run_trajectory_gsea",
    "plot_trajectory_heatmap",
    "plot_pathway_dynamics",
]
