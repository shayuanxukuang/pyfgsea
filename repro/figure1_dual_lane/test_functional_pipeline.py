from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from repro.figure1_dual_lane.common import GSEA_PARAMETERS, LANE_CONTRACTS
from repro.figure1_dual_lane.compare_functional import _read_lane, compare


def _raw(lane: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario, count in (("publication_main", 100), ("ties_predeclared", 60)):
        for index in range(count):
            r_nes = -2.0 + 4.0 * index / max(count - 1, 1)
            py_nes = r_nes + 0.001 * np.sin(index)
            r_es = r_nes / 2.0
            py_es = py_nes / 2.0
            r_p = 0.01 + 0.98 * (index + 1) / (count + 1)
            py_p = min(1.0, r_p * (1.0 + 0.001 * np.cos(index)))
            py_log = -np.log10(py_p)
            r_log = -np.log10(r_p)
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
                    "py_pval": py_p,
                    "r_pval": r_p,
                    "py_padj": min(1.0, py_p * 1.1),
                    "r_padj": min(1.0, r_p * 1.1),
                    "py_neg_log10_pval": py_log,
                    "r_neg_log10_pval": r_log,
                    "neg_log10_pval_difference": py_log - r_log,
                    "input_tied_score_group_count": 0,
                    "input_tied_gene_count": 0,
                    "input_maximum_tie_multiplicity": 1,
                }
            )
    return pd.DataFrame(rows)


def _timing(lane: str) -> pd.DataFrame:
    rows = []
    for scenario in ("publication_main", "ties_predeclared"):
        for engine, scope in (
            ("pyfgsea", "run_gsea_call_only"),
            ("r_fgsea", "Rscript_process_and_internal_fgsea"),
        ):
            rows.append(
                {
                    "lane": lane,
                    "scenario": scenario,
                    "engine": engine,
                    "measurement_scope": scope,
                    "elapsed_seconds": 1.0,
                    "peak_rss_bytes": 1024,
                    "peak_increment_bytes": 512,
                }
            )
    return pd.DataFrame(rows)


def _write_lane(root: Path, lane: str) -> Path:
    root.mkdir()
    _raw(lane).to_csv(root / "pathway_level_raw.tsv", sep="\t", index=False)
    _timing(lane).to_csv(root / "runtime_memory.tsv", sep="\t", index=False)
    contract = LANE_CONTRACTS[lane]
    result = {
        "schema_version": 1,
        "kind": "figure1_functional_lane",
        "lane": lane,
        "status": "complete",
        "package": {
            "distribution_version": contract["pyfgsea_distribution_version"],
            "module_version": contract["pyfgsea_module_version"],
            "algorithm_revision": contract["algorithm_revision"],
        },
        "reference": {
            "r_version": contract["r_version"],
            "bioconductor_version": contract["bioconductor_version"],
            "fgsea_version": contract["fgsea_version"],
        },
        "gsea_parameters": GSEA_PARAMETERS,
        "input_invariants": {"same": "inputs"},
        "environment": {
            "python": "3.11.9",
            "system": "Linux",
            "machine": "x86_64",
            "implementation": "CPython",
            "numpy": "1.26.4",
            "pandas": "2.2.3",
            "threads": {"RAYON_NUM_THREADS": "1"},
        },
        "provenance": {
            "hashes_are_pass_fail_checks": False,
            "deliberately_invalid_hash": "not-a-test-input",
        },
    }
    path = root / "lane_result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return path


def test_functional_comparison_uses_results_not_provenance_hashes(
    tmp_path: Path,
) -> None:
    legacy = _write_lane(tmp_path / "legacy", "legacy")
    current = _write_lane(tmp_path / "current", "current")
    output = tmp_path / "compared"

    result_path = compare(legacy, current, output)

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "complete"
    assert result["validation"]["installed_wheels_executed"] is True
    assert result["validation"]["hashes_are_pass_fail_checks"] is False
    assert len(pd.read_csv(output / "figure1_pathway_level_raw.tsv", sep="\t")) == 320
    assert all(
        record["sha256_role"] == "provenance_only"
        for record in result["artifacts"].values()
    )


def test_functional_lane_rejects_numerically_inconsistent_rows(tmp_path: Path) -> None:
    result = _write_lane(tmp_path / "legacy", "legacy")
    raw_path = result.parent / "pathway_level_raw.tsv"
    raw = pd.read_csv(raw_path, sep="\t")
    raw.loc[0, "nes_difference"] = 4.0
    raw.to_csv(raw_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="nes_difference is inconsistent"):
        _read_lane(result, "legacy")
