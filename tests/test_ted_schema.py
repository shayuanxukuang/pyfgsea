from __future__ import annotations

import pandas as pd
from click.testing import CliRunner

from pyfgsea.cli.main import cli
from pyfgsea.ted_schema import ted_table_is_valid, validate_ted_table


def _event_v2(**updates: object) -> pd.DataFrame:
    row: dict[str, object] = {
        "dataset_id": "d",
        "event_id": "e",
        "pathway": "p",
        "direction": "up",
        "event_mode": "activation",
        "event_test_status": "run_supported",
        "event_q": 0.01,
        "event_q_missing_reason": None,
        "e0_reason_code": None,
        "event_support_code": "E2",
        "validation_provenance_code": "V1",
        "evidence_boundary": "E2\u2013V1",
        "supported_interpretation": "Block-robust event with an orthogonal outcome.",
        "unsupported_interpretation_current_evidence": "No intervention reversal or rescue.",
        "identifiability_status": "identifiable",
        "seed": 1,
        # Legacy v1 fields are accepted but do not define the v2 boundary.
        "evidence_tier": 3.0,
        "claim_ceiling": "Level 3 block-robust event",
        "matched_functional_rescue": False,
    }
    row.update(updates)
    return pd.DataFrame([row])


def test_activity_table_schema_passes() -> None:
    table = pd.DataFrame(
        {
            "dataset_id": ["d", "d"],
            "block_id": ["b", "b"],
            "time": [0.0, 1.0],
            "pathway": ["p", "p"],
            "activity": [0.0, 1.0],
        }
    )
    assert ted_table_is_valid(validate_ted_table(table, "activity"))


def test_event_schema_fails_closed_on_unmatched_level4() -> None:
    event = pd.DataFrame(
        [
            {
                "dataset_id": "d",
                "event_id": "e",
                "pathway": "p",
                "direction": "up",
                "event_mode": "activation",
                "event_q": 0.01,
                "evidence_tier": 4.0,
                "claim_ceiling": "Level 4 functional support",
                "identifiability_status": "identifiable",
                "matched_functional_rescue": False,
                "seed": 1,
            }
        ]
    )
    report = validate_ted_table(event, "event")
    assert not ted_table_is_valid(report)
    assert "claim_ceiling_gate" in set(report["check"])


def test_event_schema_v1_remains_the_legacy_fallback() -> None:
    event = pd.DataFrame(
        [
            {
                "dataset_id": "d",
                "event_id": "e",
                "pathway": "p",
                "direction": "up",
                "event_mode": "activation",
                "event_q": 0.01,
                "evidence_tier": 3.0,
                "claim_ceiling": "Level 3 block-robust event",
                "identifiability_status": "identifiable",
                "matched_functional_rescue": False,
                "seed": 1,
            }
        ]
    )
    report = validate_ted_table(event, "event")
    assert ted_table_is_valid(report)
    assert "passed v1 schema" in report.iloc[-1]["message"]


def test_event_schema_v2_is_auto_detected_and_accepts_legacy_fields() -> None:
    report = validate_ted_table(_event_v2(), "event")
    assert ted_table_is_valid(report)
    assert "passed v2 schema" in report.iloc[-1]["message"]


def test_event_schema_v2_accepts_ascii_boundary() -> None:
    report = validate_ted_table(_event_v2(evidence_boundary="E2-V1"), "event")
    assert ted_table_is_valid(report)


def test_event_schema_v2_accepts_true_unicode_en_dash() -> None:
    boundary = "E2\u2013V1"
    assert ord(boundary[2]) == 0x2013
    report = validate_ted_table(_event_v2(evidence_boundary=boundary), "event")
    assert ted_table_is_valid(report)


def test_event_schema_v2_rejects_inconsistent_boundary() -> None:
    report = validate_ted_table(_event_v2(evidence_boundary="E1-V1"), "event")
    assert not ted_table_is_valid(report)
    assert "evidence_boundary_consistency" in set(report["check"])


def test_event_schema_v2_requires_matched_rescue_for_v3() -> None:
    report = validate_ted_table(
        _event_v2(validation_provenance_code="V3", evidence_boundary="E2-V3"),
        "event",
    )
    assert not ted_table_is_valid(report)
    assert "validation_provenance_gate" in set(report["check"])


def test_event_schema_v2_e0_requires_stable_reason_code() -> None:
    event = _event_v2(
        event_q=0.20,
        event_test_status="run_not_supported",
        e0_reason_code=None,
        event_support_code="E0",
        evidence_boundary="E0-V1",
        identifiability_status="limited",
    )
    report = validate_ted_table(event, "event")
    assert not ted_table_is_valid(report)
    assert "e0_reason_code" in set(report["check"])


def test_event_schema_v2_allows_null_q_when_test_not_run() -> None:
    event = _event_v2(
        event_test_status="not_run",
        event_q=None,
        event_q_missing_reason="insufficient_blocks",
        e0_reason_code="E0_not_estimable",
        event_support_code="E0",
        evidence_boundary="E0-V2",
        validation_provenance_code="V2",
        identifiability_status="limited",
    )
    report = validate_ted_table(event, "event")
    assert ted_table_is_valid(report), report.to_dict("records")


def test_event_schema_v2_rejects_null_q_after_test_run() -> None:
    event = _event_v2(
        event_test_status="run_not_supported",
        event_q=None,
        e0_reason_code="E0_not_supported",
        event_support_code="E0",
        validation_provenance_code="V0",
        evidence_boundary="E0-V0",
        identifiability_status="limited",
    )
    report = validate_ted_table(event, "event")
    assert not ted_table_is_valid(report)
    assert "event_q_range" in set(report["check"])


def test_event_schema_v2_allows_numeric_q_for_other_e0_reason() -> None:
    event = _event_v2(
        event_test_status="run_not_supported",
        event_q=0.25,
        e0_reason_code="E0_not_supported",
        event_support_code="E0",
        validation_provenance_code="V0",
        evidence_boundary="E0-V0",
        identifiability_status="limited",
    )
    report = validate_ted_table(event, "event")
    assert ted_table_is_valid(report), report.to_dict("records")


def test_event_schema_v2_rejects_null_q_for_supported_event() -> None:
    report = validate_ted_table(_event_v2(event_q=None), "event")
    assert not ted_table_is_valid(report)
    assert "event_q_range" in set(report["check"])


def test_event_schema_v2_requires_missing_reason_when_test_not_run() -> None:
    event = _event_v2(
        event_test_status="not_run",
        event_q=None,
        event_q_missing_reason=None,
        e0_reason_code="E0_missing_required_design",
        event_support_code="E0",
        validation_provenance_code="V0",
        evidence_boundary="E0-V0",
        identifiability_status="not_identifiable",
    )
    report = validate_ted_table(event, "event")
    assert not ted_table_is_valid(report)
    assert "event_q_missing_reason" in set(report["check"])


def test_event_schema_v2_rejects_missing_reason_after_test_run() -> None:
    report = validate_ted_table(
        _event_v2(event_q_missing_reason="other"), "event"
    )
    assert not ted_table_is_valid(report)
    assert "event_q_missing_reason" in set(report["check"])


def test_event_schema_v2_rejects_e0_reason_on_supported_event() -> None:
    report = validate_ted_table(_event_v2(e0_reason_code="E0_not_supported"), "event")
    assert not ted_table_is_valid(report)
    assert "e0_reason_code" in set(report["check"])


def test_event_schema_v2_accepts_consistent_resampling_stability() -> None:
    event = _event_v2(
        resampling_selection_frequency=0.83,
        discovery_stability_status="stable_core",
        upstream_disagreement_flag=False,
        upstream_method_agreement=0.91,
    )
    report = validate_ted_table(event, "event")
    assert ted_table_is_valid(report), report.to_dict("records")


def test_event_schema_v2_rejects_inconsistent_resampling_stability() -> None:
    report = validate_ted_table(
        _event_v2(
            resampling_selection_frequency=0.49,
            discovery_stability_status="stable_core",
        ),
        "event",
    )
    assert not ted_table_is_valid(report)
    assert "discovery_stability_consistency" in set(report["check"])


def test_event_schema_v2_upstream_disagreement_forbids_e2() -> None:
    report = validate_ted_table(
        _event_v2(upstream_disagreement_flag=True, upstream_method_agreement=0.50),
        "event",
    )
    assert not ted_table_is_valid(report)
    assert "upstream_disagreement_gate" in set(report["check"])


def test_cli_validate_writes_report(tmp_path) -> None:
    path = tmp_path / "activity.tsv"
    report = tmp_path / "report.tsv"
    pd.DataFrame(
        {
            "dataset_id": ["d", "d"],
            "block_id": ["b", "b"],
            "time": [0.0, 1.0],
            "pathway": ["p", "p"],
            "activity": [0.0, 1.0],
        }
    ).to_csv(path, sep="\t", index=False)
    result = CliRunner().invoke(cli, ["validate", str(path), "--kind", "activity", "--report", str(report)])
    assert result.exit_code == 0, result.output
    assert report.exists()


def test_cli_validate_recognizes_event_v2(tmp_path) -> None:
    path = tmp_path / "events_v2.tsv"
    report = tmp_path / "report.tsv"
    _event_v2().to_csv(path, sep="\t", index=False)
    result = CliRunner().invoke(
        cli,
        [
            "validate",
            str(path),
            "--kind",
            "event",
            "--schema-version",
            "v2",
            "--report",
            str(report),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "passed v2 schema" in result.output
    assert report.exists()
