from __future__ import annotations

from typing import Any

import pytest

import pyfgsea
from pyfgsea import plotting
from pyfgsea import plotting_utils


def test_star_import_only_advertises_existing_public_names() -> None:
    namespace: dict[str, Any] = {}
    exec("from pyfgsea import *", namespace)

    assert "run_scanpy" not in pyfgsea.__all__
    assert "run_scanpy" not in namespace
    assert all(hasattr(pyfgsea, name) for name in pyfgsea.__all__)


def test_plotting_exports_only_implemented_functions() -> None:
    assert plotting.__all__ == [
        "plot_pathway_dynamics",
        "plot_trajectory_heatmap",
    ]


def test_heatmap_wrapper_preserves_second_positional_save_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def fake_plot(df: Any, **kwargs: Any) -> None:
        observed["df"] = df
        observed.update(kwargs)

    monkeypatch.setattr(plotting_utils, "plot_trajectory_heatmap", fake_plot)
    plotting.plot_trajectory_heatmap(
        "frame",
        "legacy-output.png",
        pathways=("A", "B"),
        n_top_pathways=7,
    )

    assert observed["df"] == "frame"
    assert observed["save_path"] == "legacy-output.png"
    assert observed["pathways"] == ["A", "B"]
    assert observed["n_top_pathways"] == 7


def test_pathway_dynamics_wrapper_rejects_missing_pathways_and_forwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="at least one pathway"):
        plotting.plot_pathway_dynamics("frame")

    observed: dict[str, Any] = {}

    def fake_plot(df: Any, **kwargs: Any) -> None:
        observed["df"] = df
        observed.update(kwargs)

    monkeypatch.setattr(plotting_utils, "plot_pathway_dynamics", fake_plot)
    plotting.plot_pathway_dynamics(
        "frame", ("A",), figsize=(8, 3), save_path="dynamics.png"
    )

    assert observed == {
        "df": "frame",
        "pathways": ["A"],
        "figsize": (8, 3),
        "save_path": "dynamics.png",
    }
