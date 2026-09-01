from __future__ import annotations

from typing import Any, Sequence

from .overview import plot_overview_heatmap as plot_overview_heatmap
from .fastproof import plot_fastproof as plot_fastproof


def plot_trajectory_heatmap(
    df: Any,
    save_path: str | None = None,
    *,
    pathways: Sequence[str] | None = None,
    n_top_pathways: int = 30,
    sort_by_peak: bool = True,
    cmap: str = "RdBu_r",
    figsize: tuple[float, float] = (10, 8),
) -> None:
    """Plot trajectory NES values while preserving the historical call shape.

    The second positional argument remains ``save_path``. Optional plotting
    controls are keyword-only so the maintained implementation cannot silently
    reinterpret an older call as a pathway list.
    """
    from ..plotting_utils import plot_trajectory_heatmap as _plot

    _plot(
        df,
        pathways=None if pathways is None else list(pathways),
        n_top_pathways=n_top_pathways,
        sort_by_peak=sort_by_peak,
        cmap=cmap,
        figsize=figsize,
        save_path=save_path,
    )


def plot_pathway_dynamics(
    df: Any,
    pathways: Sequence[str] | None = None,
    *,
    figsize: tuple[float, float] = (10, 4),
    save_path: str | None = None,
) -> None:
    """Plot selected pathway trajectories using the maintained implementation."""
    if pathways is None:
        raise ValueError("pathways must contain at least one pathway name")

    from ..plotting_utils import plot_pathway_dynamics as _plot

    _plot(df, pathways=list(pathways), figsize=figsize, save_path=save_path)


__all__ = [
    "plot_fastproof",
    "plot_overview_heatmap",
    "plot_pathway_dynamics",
    "plot_trajectory_heatmap",
]
