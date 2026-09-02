from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from repro.supplement_rc8 import run_null_calibration as calibration


def _small_inputs() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    genes = [f"g{index}" for index in range(8)]
    ranks = pd.DataFrame({"Gene": genes, "Score": np.arange(8, dtype=float)})
    pathways = {"a": genes[:4], "b": genes[4:]}
    return ranks, pathways


def _valid_engine(calls: list[tuple[pd.DataFrame, dict[str, object]]]):
    def run_gsea(
        data: pd.DataFrame, pathways: dict[str, list[str]], **kwargs: object
    ) -> pd.DataFrame:
        calls.append((data.copy(), dict(kwargs)))
        names = sorted(pathways)
        pvalues = np.linspace(0.2, 0.8, len(names))
        return pd.DataFrame(
            {
                "Pathway": names,
                "ES": np.linspace(-0.5, 0.5, len(names)),
                "NES": np.linspace(-1.0, 1.0, len(names)),
                "P-value": pvalues,
                "padj": np.minimum(1.0, pvalues * 1.1),
                "status": "resolved",
            }
        )

    return run_gsea


def test_loads_the_fixed_figure1_inputs() -> None:
    ranks, pathways, paths = calibration._load_inputs(calibration.DEFAULT_INPUT_DIR)

    assert len(ranks) == calibration.EXPECTED_GENE_COUNT
    assert len(pathways) == calibration.EXPECTED_PATHWAY_COUNT
    assert set(paths) == {"ranks", "pathways"}


def test_uniform_distance_uses_both_ecdf_sides() -> None:
    values = np.array([0.125, 0.375, 0.625, 0.875])

    assert calibration._ks_uniform(values) == pytest.approx(0.125)


def test_execute_calibration_runs_distinct_real_permutations() -> None:
    ranks, pathways = _small_inputs()
    calls: list[tuple[pd.DataFrame, dict[str, object]]] = []

    raw, summary = calibration._execute_calibration(
        ranks,
        pathways,
        replicates=3,
        base_seed=100,
        run_gsea=_valid_engine(calls),
    )

    assert len(calls) == 3
    assert len(raw) == 6
    assert len(summary) == 4
    assert list(raw.groupby("replicate")["permutation_seed"].first()) == [100, 101, 102]
    original_scores = sorted(ranks["Score"])
    assert all(sorted(call[0]["Score"]) == original_scores for call in calls)
    assert all(list(call[0]["Gene"]) == list(ranks["Gene"]) for call in calls)
    assert len({tuple(call[0]["Score"]) for call in calls}) == 3
    assert all(call[1]["mode"] == "aligned" for call in calls)
    assert summary.iloc[-1]["replicate"] == "pooled_descriptive"


def test_execute_calibration_rejects_nonfinite_pvalues() -> None:
    ranks, pathways = _small_inputs()

    def invalid_engine(*args: object, **kwargs: object) -> pd.DataFrame:
        del args, kwargs
        return pd.DataFrame(
            {
                "Pathway": ["a", "b"],
                "ES": [0.1, 0.2],
                "NES": [0.2, 0.3],
                "P-value": [0.5, np.nan],
                "padj": [0.6, 0.7],
                "status": ["resolved", "unresolved"],
            }
        )

    with pytest.raises(calibration.NullCalibrationError, match="non-finite"):
        calibration._execute_calibration(
            ranks,
            pathways,
            replicates=2,
            base_seed=100,
            run_gsea=invalid_engine,
        )


def test_run_calibration_writes_only_completed_real_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    ranks_path = input_dir / "ranks.csv"
    pathways_path = input_dir / "pathways.gmt"
    ranks_path.write_text("fixed ranks\n", encoding="utf-8")
    pathways_path.write_text("fixed pathways\n", encoding="utf-8")
    ranks, pathways = _small_inputs()
    calls: list[tuple[pd.DataFrame, dict[str, object]]] = []
    monkeypatch.setattr(
        calibration,
        "_installed_package",
        lambda: (
            _valid_engine(calls),
            {
                "distribution_version": "0.2.0rc8",
                "module_version": "0.2.0rc8",
                "algorithm_revision": "fgsea-1.38-pr178-v1",
                "module_path": "installed/pyfgsea/__init__.py",
                "native_core_path": "installed/pyfgsea/_core.pyd",
            },
        ),
    )
    monkeypatch.setattr(
        calibration,
        "_load_inputs",
        lambda _: (
            ranks,
            pathways,
            {"ranks": ranks_path, "pathways": pathways_path},
        ),
    )
    output = tmp_path / "null-result"

    manifest_path = calibration.run_calibration(
        input_dir, output, replicates=2, base_seed=200
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(calls) == 2
    assert manifest["status"] == "complete"
    assert manifest["validation"]["installed_wheel_executed"] is True
    assert manifest["validation"]["hashes_are_pass_fail_checks"] is False
    assert manifest["design"]["equivalence_margin"] is None
    assert manifest["design"]["acceptance_threshold"] is None
    assert all(
        record["sha256_role"] == "provenance_only"
        for record in manifest["artifacts"].values()
    )
    assert (
        len(pd.read_csv(output / "null_calibration_pathway_results.tsv", sep="\t")) == 4
    )
    assert (output / "null_calibration_qq_ecdf.png").is_file()


def test_run_calibration_leaves_no_output_after_engine_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ranks, pathways = _small_inputs()

    def failed_engine(*args: object, **kwargs: object) -> pd.DataFrame:
        del args, kwargs
        raise RuntimeError("engine failed")

    monkeypatch.setattr(
        calibration,
        "_installed_package",
        lambda: (failed_engine, {"distribution_version": "0.2.0rc8"}),
    )
    monkeypatch.setattr(
        calibration,
        "_load_inputs",
        lambda _: (ranks, pathways, {}),
    )
    output = tmp_path / "null-result"

    with pytest.raises(RuntimeError, match="engine failed"):
        calibration.run_calibration(
            tmp_path / "input", output, replicates=2, base_seed=300
        )

    assert not output.exists()
