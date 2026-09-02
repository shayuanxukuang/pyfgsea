from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from repro.supplement_rc8.assemble_figure1 import SupplementError, assemble

COUNTS = {"publication_main": 100, "ties_predeclared": 60}


def _raw() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for lane in ("legacy", "current"):
        lane_shift = 0.001 if lane == "legacy" else -0.0005
        for scenario, count in COUNTS.items():
            for index in range(count):
                r_nes = -2.0 + 4.0 * index / (count - 1)
                py_nes = r_nes + lane_shift + 0.002 * np.sin(index)
                r_es = r_nes / 2.0
                py_es = py_nes / 2.0
                r_pval = (index + 1) / (count + 1)
                py_pval = min(1.0, r_pval * (1.0 + 0.01 * np.cos(index)))
                py_log = -np.log10(py_pval)
                r_log = -np.log10(r_pval)
                rows.append(
                    {
                        "lane": lane,
                        "scenario": scenario,
                        "pathway": f"{scenario}-{index:03d}",
                        "py_es": py_es,
                        "r_es": r_es,
                        "es_difference": py_es - r_es,
                        "py_nes": py_nes,
                        "r_nes": r_nes,
                        "nes_difference": py_nes - r_nes,
                        "py_pval": py_pval,
                        "r_pval": r_pval,
                        "py_padj": min(1.0, py_pval * 1.1),
                        "r_padj": min(1.0, r_pval * 1.1),
                        "py_neg_log10_pval": py_log,
                        "r_neg_log10_pval": r_log,
                        "neg_log10_pval_difference": py_log - r_log,
                    }
                )
    return pd.DataFrame(rows)


def _timing() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for lane in ("legacy", "current"):
        for scenario in COUNTS:
            for engine, scope, elapsed in (
                ("pyfgsea", "run_gsea_call_only", 1.0),
                ("r_fgsea", "Rscript_process_and_internal_fgsea", 2.5),
            ):
                rows.append(
                    {
                        "lane": lane,
                        "scenario": scenario,
                        "engine": engine,
                        "measurement_scope": scope,
                        "elapsed_seconds": elapsed,
                        "peak_rss_bytes": 4096,
                        "peak_increment_bytes": 1024,
                    }
                )
    return pd.DataFrame(rows)


def _write_result(root: Path) -> tuple[Path, pd.DataFrame]:
    root.mkdir()
    raw = _raw()
    raw.to_csv(root / "figure1_pathway_level_raw.tsv", sep="\t", index=False)
    _timing().to_csv(root / "figure1_runtime_memory.tsv", sep="\t", index=False)
    comparison = {
        "schema_version": 1,
        "kind": "figure1_functional_comparison",
        "status": "complete",
        "lanes": {
            "legacy": {"pyfgsea": "0.1.4", "fgsea": "1.32.2"},
            "current": {"pyfgsea": "0.2.0rc8", "fgsea": "1.38.0"},
        },
        "validation": {
            "installed_wheels_executed": True,
            "reference_versions_executed": True,
            "raw_rows_complete": True,
            "metrics_recomputed_from_raw_rows": True,
            "manual_metric_overrides": False,
            "hashes_are_pass_fail_checks": False,
        },
        "artifacts": {
            "figure1_pathway_level_raw.tsv": {"sha256": "deliberately-not-a-valid-hash"}
        },
    }
    (root / "comparison_result.json").write_text(
        json.dumps(comparison), encoding="utf-8"
    )
    return root, raw


def test_assemble_recomputes_descriptive_outputs_without_hash_gate(
    tmp_path: Path,
) -> None:
    result, raw = _write_result(tmp_path / "result")
    output = tmp_path / "supplement"

    manifest_path = assemble(result, output)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["validation"]["hashes_are_pass_fail_checks"] is False
    assert manifest["methods"]["equivalence_margin"] is None
    assert all(
        record["sha256_role"] == "provenance_only"
        for record in manifest["artifacts"].values()
    )

    bland_altman = pd.read_csv(output / "figure1_bland_altman.tsv", sep="\t")
    assert len(bland_altman) == 8
    observed = bland_altman.loc[
        (bland_altman["lane"] == "current")
        & (bland_altman["scenario"] == "publication_main")
        & (bland_altman["metric"] == "NES")
    ].iloc[0]
    group = raw.loc[
        (raw["lane"] == "current") & (raw["scenario"] == "publication_main")
    ]
    difference = group["py_nes"] - group["r_nes"]
    assert observed["mean_bias_py_minus_r"] == pytest.approx(difference.mean())
    assert observed["lower_95_limit"] == pytest.approx(
        difference.mean() - 1.96 * difference.std(ddof=1)
    )

    overlap = pd.read_csv(output / "figure1_pathway_overlap.tsv", sep="\t")
    tail = pd.read_csv(output / "figure1_extreme_tail_cases.tsv", sep="\t")
    tail_summary = pd.read_csv(output / "figure1_extreme_tail_summary.tsv", sep="\t")
    runtime = pd.read_csv(output / "figure1_runtime_descriptive.tsv", sep="\t")
    assert len(overlap) == 8
    assert len(tail) == 40
    assert len(tail_summary) == 4
    assert len(runtime) == 4
    assert (runtime["descriptive_r_over_py_elapsed_ratio"] == 2.5).all()
    assert runtime["interpretation"].str.contains("not an equal-scope").all()
    assert (output / "figure1_bland_altman.png").is_file()
    assert (output / "figure1_bland_altman.pdf").is_file()


def test_assemble_fails_closed_on_inconsistent_raw_values(tmp_path: Path) -> None:
    result, _ = _write_result(tmp_path / "result")
    raw_path = result / "figure1_pathway_level_raw.tsv"
    raw = pd.read_csv(raw_path, sep="\t")
    raw.loc[0, "nes_difference"] = 100.0
    raw.to_csv(raw_path, sep="\t", index=False)
    output = tmp_path / "supplement"

    with pytest.raises(SupplementError, match="nes_difference is inconsistent"):
        assemble(result, output)

    assert not output.exists()


def test_assemble_requires_functional_execution_checks(tmp_path: Path) -> None:
    result, _ = _write_result(tmp_path / "result")
    comparison_path = result / "comparison_result.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison["validation"]["installed_wheels_executed"] = False
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")

    with pytest.raises(SupplementError, match="execution checks"):
        assemble(result, tmp_path / "supplement")


def test_assemble_rejects_a_changed_runtime_scope(tmp_path: Path) -> None:
    result, _ = _write_result(tmp_path / "result")
    timing_path = result / "figure1_runtime_memory.tsv"
    timing = pd.read_csv(timing_path, sep="\t")
    timing.loc[timing["engine"] == "r_fgsea", "measurement_scope"] = "call_only"
    timing.to_csv(timing_path, sep="\t", index=False)

    with pytest.raises(SupplementError, match="timing scope is wrong"):
        assemble(result, tmp_path / "supplement")


def test_assemble_does_not_overwrite_an_existing_directory(tmp_path: Path) -> None:
    result, _ = _write_result(tmp_path / "result")
    output = tmp_path / "supplement"
    output.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        assemble(result, output)
