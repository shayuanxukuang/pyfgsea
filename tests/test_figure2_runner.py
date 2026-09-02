from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_gse155254_figure2.py"
SPEC = importlib.util.spec_from_file_location("figure2_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _small_adata() -> ad.AnnData:
    return ad.AnnData(
        X=np.array(
            [
                [-1.0, 0.0, 1.0],
                [-0.5, 0.5, 1.5],
                [0.5, 1.0, -1.0],
                [1.0, 1.5, -0.5],
            ]
        ),
        obs=pd.DataFrame(
            {
                "sample_id": ["sample1", "sample1", "sample2", "sample2"],
                "condition": ["Disease"] * 4,
                "dpt_pseudotime": [0.0, 0.25, 0.75, 1.0],
            },
            index=["sample1:a", "sample1:b", "sample2:c", "sample2:d"],
        ),
        var=pd.DataFrame(index=["A", "B", "C"]),
    )


def _small_results() -> pd.DataFrame:
    rows = []
    for window_id in range(2):
        for pathway_index, pathway in enumerate(("P1", "P2", "P3"), start=1):
            rows.append(
                {
                    "Pathway": pathway,
                    "ES": pathway_index / 10.0,
                    "NES": pathway_index / 5.0,
                    "P-value": pathway_index / 10.0,
                    "padj": pathway_index / 10.0,
                    "status": "resolved",
                    "window_id": window_id,
                    "observed_pathway_size": pathway_index + 10,
                    "null_curve_size": pathway_index + 10,
                    "size_binned": False,
                    "algorithm_revision": runner.EXPECTED_ALGORITHM_REVISION,
                }
            )
    return pd.DataFrame(rows)


def _small_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "EXPECTED_DATASET_SHAPE", (4, 3))
    monkeypatch.setattr(runner, "EXPECTED_N_WINDOWS", 2)
    monkeypatch.setattr(runner, "EXPECTED_N_PATHWAYS", 3)
    monkeypatch.setattr(runner, "EXPECTED_N_ROWS", 6)


def test_fixed_parameters_match_the_recorded_contract() -> None:
    record = json.loads(runner.PARAMETER_PATH.read_text(encoding="utf-8"))
    assert record["parameters"] == runner.PARAMETER_CONTRACT
    runner._validate_parameter_record()


def test_installed_runtime_records_version_and_native_core(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_file = tmp_path / "site-packages" / "pyfgsea" / "__init__.py"
    package_file.parent.mkdir(parents=True)
    package_file.write_text("", encoding="utf-8")
    suffix = importlib.machinery.EXTENSION_SUFFIXES[-1]
    core_file = package_file.parent / f"_core{suffix}"
    core_file.write_bytes(b"native fixture")
    package = SimpleNamespace(
        __version__=runner.DEFAULT_EXPECTED_VERSION,
        __file__=str(package_file),
    )
    wrapper = SimpleNamespace(
        _ext=SimpleNamespace(__file__=str(core_file)),
        _algorithm_revision=lambda: runner.EXPECTED_ALGORITHM_REVISION,
    )
    monkeypatch.delitem(sys.modules, "pyfgsea", raising=False)
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: package if name == "pyfgsea" else wrapper,
    )
    monkeypatch.setattr(
        runner.importlib.metadata,
        "version",
        lambda _name: runner.DEFAULT_EXPECTED_VERSION,
    )

    loaded, info = runner._load_installed_runtime(runner.DEFAULT_EXPECTED_VERSION)

    assert loaded is package
    assert info["distribution_version"] == runner.DEFAULT_EXPECTED_VERSION
    assert info["native_core_file"] == str(core_file.resolve())


@pytest.mark.parametrize("failure", ["version", "revision", "core"])
def test_installed_runtime_rejects_wrong_functional_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    package_file = tmp_path / "site" / "pyfgsea" / "__init__.py"
    package_file.parent.mkdir(parents=True)
    package_file.write_text("", encoding="utf-8")
    core_file = package_file.parent / (
        "_core.txt"
        if failure == "core"
        else f"_core{importlib.machinery.EXTENSION_SUFFIXES[-1]}"
    )
    core_file.write_bytes(b"fixture")
    version = "0.1.4" if failure == "version" else runner.DEFAULT_EXPECTED_VERSION
    revision = (
        "legacy-revision"
        if failure == "revision"
        else runner.EXPECTED_ALGORITHM_REVISION
    )
    package = SimpleNamespace(__version__=version, __file__=str(package_file))
    wrapper = SimpleNamespace(
        _ext=SimpleNamespace(__file__=str(core_file)),
        _algorithm_revision=lambda: revision,
    )
    monkeypatch.delitem(sys.modules, "pyfgsea", raising=False)
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: package if name == "pyfgsea" else wrapper,
    )
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _name: version)

    with pytest.raises(runner.Figure2RunError):
        runner._load_installed_runtime(runner.DEFAULT_EXPECTED_VERSION)


def test_result_validation_requires_complete_exact_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _small_contract(monkeypatch)
    summary = runner._validate_results(_small_results())
    assert summary == {
        "complete_grid": True,
        "n_rows": 6,
        "n_windows": 2,
        "n_pathways": 3,
        "resolved_rows": 6,
        "pathway_size_policy": "exact",
    }

    mismatched = _small_results()
    mismatched.loc[0, "null_curve_size"] += 1
    with pytest.raises(runner.Figure2RunError, match="exact pathway sizes"):
        runner._validate_results(mismatched)


def test_runner_writes_only_table_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _small_contract(monkeypatch)
    dataset = tmp_path / "input.h5ad"
    _small_adata().write_h5ad(dataset)
    gene_sets = tmp_path / "pathways.gmt"
    gene_sets.write_text("P1\tfixture\tA\nP2\tfixture\tB\nP3\tfixture\tC\n")
    observed: dict[str, object] = {}

    def run_trajectory(
        adata: ad.AnnData, gmt_path: str, **parameters: object
    ) -> pd.DataFrame:
        observed.update(
            {
                "shape": adata.shape,
                "gmt_path": gmt_path,
                "parameters": parameters,
            }
        )
        return _small_results()

    package = SimpleNamespace(run_trajectory_gsea=run_trajectory)
    runtime = {
        "distribution_version": runner.DEFAULT_EXPECTED_VERSION,
        "package_version": runner.DEFAULT_EXPECTED_VERSION,
        "algorithm_revision": runner.EXPECTED_ALGORITHM_REVISION,
        "package_file": "installed/pyfgsea/__init__.py",
        "native_core_file": "installed/pyfgsea/_core.pyd",
    }
    monkeypatch.setattr(
        runner,
        "_load_installed_runtime",
        lambda _expected: (package, runtime),
    )
    output = tmp_path / "output"

    summary = runner.run_figure2(dataset, gene_sets, output)

    assert {path.name for path in output.iterdir()} == {
        "trajectory_results.csv",
        "run_summary.json",
    }
    assert summary["status"] == "complete"
    assert summary["dataset"]["expression_matrix"] == "scaled adata.X"
    assert observed == {
        "shape": (4, 3),
        "gmt_path": str(gene_sets.resolve()),
        "parameters": runner.RUN_PARAMETERS,
    }
