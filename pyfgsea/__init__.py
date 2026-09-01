from importlib.metadata import PackageNotFoundError, version as _distribution_version
from pathlib import Path as _Path
import re as _re

from .wrapper import (
    run_gsea,
    load_gmt,
    GseaRunner,
    prepare_pathways,
    get_random_es_means,
    multilevel_error,
)
from .trajectory import run_trajectory_gsea
from .plotting import plot_trajectory_heatmap, plot_pathway_dynamics

_cargo_manifest = _Path(__file__).resolve().parent.parent / "Cargo.toml"
if _cargo_manifest.is_file():
    _package_section = _cargo_manifest.read_text(encoding="utf-8").split(
        "[package]", 1
    )[1]
    _package_section = _package_section.split("[", 1)[0]
    _version_match = _re.search(
        r'^version\s*=\s*"([^"]+)"', _package_section, _re.MULTILINE
    )
    if _version_match is None:
        raise RuntimeError("Cargo.toml [package] is missing version")
    _cargo_version = _version_match.group(1)
    _rc_match = _re.fullmatch(r"(\d+\.\d+\.\d+)-rc(\d+)", _cargo_version)
    __version__ = (
        f"{_rc_match.group(1)}rc{_rc_match.group(2)}"
        if _rc_match is not None
        else _cargo_version
    )
else:
    try:
        __version__ = _distribution_version("pyfgsea")
    except PackageNotFoundError:
        __version__ = "0+unknown"

# Explicitly expose API to top level
__all__ = [
    "__version__",
    "run_gsea",
    "load_gmt",
    "GseaRunner",
    "prepare_pathways",
    "get_random_es_means",
    "multilevel_error",
    "run_trajectory_gsea",
    "plot_trajectory_heatmap",
    "plot_pathway_dynamics",
]
