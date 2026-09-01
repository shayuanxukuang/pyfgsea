from __future__ import annotations

import json
import base64
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from repro.data_utils import generate_test_data as historical_generate_test_data
from repro.figure1_dual_lane.adjudicate import concordance_metrics
from repro.figure1_dual_lane.common import read_json, verify_file_record
from repro.figure1_dual_lane.prepare_inputs import (
    PUBLICATION_PARAMETERS,
    generate_test_data,
    prepare_inputs,
)
from repro.figure1_dual_lane.verify_legacy_artifact import _verify_record


def test_publication_generator_matches_historical_utility() -> None:
    expected_ranks, expected_pathways = historical_generate_test_data(
        n_genes=12000, n_sets=100, seed=42
    )
    actual_ranks, actual_pathways = generate_test_data(**PUBLICATION_PARAMETERS)
    pd.testing.assert_frame_equal(actual_ranks, expected_ranks)
    assert actual_pathways == expected_pathways


def test_input_bundle_contains_distinct_predeclared_ties(tmp_path: Path) -> None:
    manifest_path = prepare_inputs(tmp_path / "inputs")
    manifest = read_json(manifest_path)
    assert list(manifest["scenarios"]) == ["publication_main", "ties_predeclared"]
    publication = manifest["scenarios"]["publication_main"]
    ties = manifest["scenarios"]["ties_predeclared"]
    assert publication["score_transform"] == "none"
    assert ties["score_transform"] == "round_binary64_to_1_decimal"
    assert publication["invariants"]["pathway_count"] == 100
    assert publication["invariants"]["tied_score_group_count"] == 0
    assert ties["invariants"]["pathway_count"] == 60
    assert ties["invariants"]["tied_score_group_count"] > 0
    for scenario in (publication, ties):
        for label in ("ranks", "pathways"):
            verify_file_record(manifest_path.parent, scenario[label], label=label)


def test_concordance_metrics_are_derived_from_values() -> None:
    left = pd.Series([1.0, 2.0, 4.0, 8.0])
    same = concordance_metrics(left, left.copy())
    assert same["pearson"] == pytest.approx(1.0)
    assert same["spearman"] == pytest.approx(1.0)
    assert same["rmse"] == pytest.approx(0.0)
    assert same["maximum_absolute_difference"] == pytest.approx(0.0)

    right = pd.Series([1.0, 2.5, 3.0, 7.0])
    changed = concordance_metrics(left, right)
    expected_difference = left.to_numpy() - right.to_numpy()
    assert changed["rmse"] == pytest.approx(
        float(np.sqrt(np.mean(np.square(expected_difference))))
    )
    assert changed["median_absolute_difference"] == pytest.approx(
        float(np.median(np.abs(expected_difference)))
    )
    assert changed["maximum_absolute_difference"] == pytest.approx(1.0)


def test_file_record_tampering_fails(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("before\n", encoding="utf-8")
    record = {
        "path": evidence.name,
        "sha256": __import__("hashlib").sha256(evidence.read_bytes()).hexdigest(),
    }
    evidence.write_text("after\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_file_record(tmp_path, record, label="tampered")


def test_manifest_reader_rejects_non_object(tmp_path: Path) -> None:
    source = tmp_path / "array.json"
    source.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="root must be an object"):
        read_json(source)


def test_legacy_record_windows_separators_and_tampering() -> None:
    member = b"wheel source\n"
    encoded = base64.urlsafe_b64encode(hashlib.sha256(member).digest()).rstrip(b"=")
    record_name = "pyfgsea-0.1.4.dist-info/RECORD"
    record = (
        f"pyfgsea\\__init__.py,sha256={encoded.decode('ascii')},{len(member)}\n"
        f"{record_name.replace('/', chr(92))},,\n"
    ).encode()
    contents = {"pyfgsea/__init__.py": member, record_name: record}
    _verify_record(contents, record_name)
    contents["pyfgsea/__init__.py"] = b"tampered\n"
    with pytest.raises(RuntimeError, match="RECORD hash mismatch"):
        _verify_record(contents, record_name)


def test_legacy_wheel_record_verifier_rejects_unsafe_or_colliding_paths() -> None:
    record_name = "pkg.dist-info/RECORD"
    contents = {
        "pkg/value.py": b"value\n",
        record_name: b"../pkg/value.py,,\npkg.dist-info/RECORD,,\n",
    }
    with pytest.raises(RuntimeError, match="unsafe path"):
        _verify_record(contents, record_name)

    contents[record_name] = b"pkg/value.py,,\npkg\\value.py,,\npkg.dist-info/RECORD,,\n"
    with pytest.raises(RuntimeError, match="normalized-path collision"):
        _verify_record(contents, record_name)
