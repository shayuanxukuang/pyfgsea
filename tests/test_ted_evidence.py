from __future__ import annotations

import json

import pytest

from pyfgsea.ted_evidence import (
    EventSupportInputs,
    EventSupportThresholds,
    ValidationProvenanceInputs,
    assign_event_support,
    assign_evidence_boundary,
    assign_validation_provenance,
)


def e1_inputs(**updates: object) -> EventSupportInputs:
    values = {
        "event_family_declared": True,
        "defensible_null_specified": True,
        "biological_units_present": True,
        "condition_batch_confounded": False,
        "identifiability_status": "identifiable",
        "artifact_dominated": False,
        "event_q": 0.05,
        "retained_module": True,
        "basic_controls_pass": True,
    }
    values.update(updates)
    return EventSupportInputs(**values)


def e2_inputs(**updates: object) -> EventSupportInputs:
    values = {
        "effective_blocks": 3,
        "block_q": 0.08,
        "direction_stability": 0.80,
        "mode_identifiable": True,
        "negative_control_pass": True,
        "negative_control_margin": 0.01,
    }
    values.update(updates)
    return e1_inputs(**values)


def test_missing_mandatory_event_input_fails_closed_to_e0() -> None:
    result = assign_event_support(e1_inputs(event_q=None))

    assert result.code == "E0"
    assert "EVENT_Q_MISSING" in result.reason_codes
    assert any(
        gate.gate == "event_q" and gate.status == "missing"
        for gate in result.audit_trail
    )


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"identifiability_status": "not_identifiable"}, "EVENT_NOT_IDENTIFIABLE"),
        ({"artifact_dominated": True}, "EVENT_ARTIFACT_DOMINATED"),
        ({"condition_batch_confounded": True}, "CONDITION_BATCH_COMPLETELY_CONFOUNDED"),
        ({"event_q": 0.10001}, "EVENT_Q_ABOVE_THRESHOLD"),
        ({"event_q": float("nan")}, "EVENT_Q_INVALID"),
    ],
)
def test_hard_event_gates_return_e0(updates: dict[str, object], reason: str) -> None:
    result = assign_event_support(e2_inputs(**updates))

    assert result.code == "E0"
    assert reason in result.reason_codes


def test_supported_event_without_robustness_inputs_is_e1() -> None:
    result = assign_event_support(e1_inputs())

    assert result.code == "E1"
    assert "EFFECTIVE_BLOCKS_MISSING" in result.reason_codes
    assert "BLOCK_SUPPORT_MISSING" in result.reason_codes
    assert "NEGATIVE_CONTROL_STATUS_MISSING" in result.reason_codes


def test_complete_default_robustness_gates_assign_e2() -> None:
    result = assign_event_support(e2_inputs())

    assert result.code == "E2"
    assert result.reason_codes == ("E2_ALL_GATES_PASS",)


def test_block_confidence_interval_is_an_allowed_alternative_to_block_q() -> None:
    result = assign_event_support(e2_inputs(block_q=None, block_ci_excludes_zero=True))

    assert result.code == "E2"
    assert any(
        gate.gate == "block_support" and gate.status == "pass"
        for gate in result.audit_trail
    )


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"effective_blocks": 2}, "INSUFFICIENT_EFFECTIVE_BLOCKS"),
        ({"block_q": 0.11}, "BLOCK_SUPPORT_FAIL"),
        ({"direction_stability": 0.79}, "DIRECTION_STABILITY_BELOW_THRESHOLD"),
        ({"mode_identifiable": False}, "MODE_NOT_IDENTIFIABLE_FOR_E2"),
        ({"negative_control_pass": False}, "NEGATIVE_CONTROL_GATES_FAIL"),
        ({"negative_control_margin": 0.0}, "NEGATIVE_CONTROL_MARGIN_NONPOSITIVE"),
    ],
)
def test_failed_robustness_gate_caps_support_at_e1(
    updates: dict[str, object], reason: str
) -> None:
    result = assign_event_support(e2_inputs(**updates))

    assert result.code == "E1"
    assert reason in result.reason_codes


def test_required_matched_state_overlap_fails_closed_and_attenuation_caps_e2() -> None:
    no_overlap = assign_event_support(
        e2_inputs(matched_state_required=True, matched_state_overlap_pass=False)
    )
    attenuated = assign_event_support(
        e2_inputs(
            matched_state_required=True,
            matched_state_overlap_pass=True,
            matched_state_attenuation=0.50,
        )
    )

    assert no_overlap.code == "E0"
    assert "MATCHED_STATE_OVERLAP_FAIL" in no_overlap.reason_codes
    assert attenuated.code == "E1"
    assert "MATCHED_STATE_ATTENUATION_TOO_LARGE" in attenuated.reason_codes


def test_thresholds_are_explicit_and_configurable() -> None:
    strict = EventSupportThresholds(event_q_max=0.05)

    assert assign_event_support(e2_inputs(event_q=0.05), thresholds=strict).code == "E2"
    failed = assign_event_support(e2_inputs(event_q=0.05001), thresholds=strict)
    assert failed.code == "E0"
    assert "EVENT_Q_ABOVE_THRESHOLD" in failed.reason_codes


def test_no_observed_validation_provenance_is_v0() -> None:
    result = assign_validation_provenance(ValidationProvenanceInputs())

    assert result.code == "V0"
    assert result.reason_codes == ("V0_COMPUTATIONAL_ONLY",)


def test_invalid_observation_flag_fails_closed_with_reason() -> None:
    result = assign_validation_provenance(
        ValidationProvenanceInputs(orthogonal_outcome_observed=None)  # type: ignore[arg-type]
    )

    assert result.code == "V0"
    assert "V1_OBSERVATION_FLAG_INVALID" in result.reason_codes


def test_incomplete_observed_outcome_fails_closed_to_v0() -> None:
    result = assign_validation_provenance(
        ValidationProvenanceInputs(
            orthogonal_outcome_observed=True,
            outcome_assessment_prespecified=True,
            outcome_aligned=True,
            outcome_controls_pass=None,
        )
    )

    assert result.code == "V0"
    assert "V1_OUTCOME_CONTROLS_STATUS_MISSING" in result.reason_codes


def test_complete_outcome_and_intervention_gates_assign_v1_and_v2() -> None:
    v1 = assign_validation_provenance(
        ValidationProvenanceInputs(
            orthogonal_outcome_observed=True,
            outcome_assessment_prespecified=True,
            outcome_aligned=True,
            outcome_controls_pass=True,
        )
    )
    v2 = assign_validation_provenance(
        ValidationProvenanceInputs(
            intervention_reversal_observed=True,
            intervention_contrast_prespecified=True,
            intervention_reversal_pass=True,
            matched_intervention_controls_pass=True,
        )
    )

    assert v1.code == "V1"
    assert v2.code == "V2"


def test_v3_requires_same_system_predicted_recovery_and_controls() -> None:
    incomplete = assign_validation_provenance(
        ValidationProvenanceInputs(
            matched_rescue_observed=True,
            rescue_same_system=False,
            predicted_readout_recovered=True,
            matched_rescue_controls_pass=True,
        )
    )
    complete = assign_validation_provenance(
        ValidationProvenanceInputs(
            matched_rescue_observed=True,
            rescue_same_system=True,
            predicted_readout_recovered=True,
            matched_rescue_controls_pass=True,
        )
    )

    assert incomplete.code == "V0"
    assert "V3_RESCUE_NOT_SAME_SYSTEM" in incomplete.reason_codes
    assert complete.code == "V3"


def test_v4_requires_independence_success_and_recorded_basis() -> None:
    missing_basis = assign_validation_provenance(
        ValidationProvenanceInputs(
            independent_replication_observed=True,
            replication_independent=True,
            independent_replication_pass=True,
        )
    )
    complete = assign_validation_provenance(
        ValidationProvenanceInputs(
            independent_replication_observed=True,
            replication_independent=True,
            independent_replication_pass=True,
            replicated_validation_basis="V2",
        )
    )

    assert missing_basis.code == "V0"
    assert "V4_REPLICATED_BASIS_MISSING" in missing_basis.reason_codes
    assert complete.code == "V4"
    assert complete.replicated_validation_basis == "V2"


def test_failed_higher_candidate_does_not_erase_complete_lower_provenance() -> None:
    result = assign_validation_provenance(
        ValidationProvenanceInputs(
            orthogonal_outcome_observed=True,
            outcome_assessment_prespecified=True,
            outcome_aligned=True,
            outcome_controls_pass=True,
            matched_rescue_observed=True,
            rescue_same_system=None,
            predicted_readout_recovered=True,
            matched_rescue_controls_pass=True,
        )
    )

    assert result.code == "V1"
    assert "V3_RESCUE_SYSTEM_STATUS_MISSING" in result.reason_codes


def test_e_and_v_axes_remain_independent_in_combined_boundary() -> None:
    result = assign_evidence_boundary(
        e1_inputs(event_q=None),
        ValidationProvenanceInputs(
            intervention_reversal_observed=True,
            intervention_contrast_prespecified=True,
            intervention_reversal_pass=True,
            matched_intervention_controls_pass=True,
        ),
    )

    assert result.event_support.code == "E0"
    assert result.validation_provenance.code == "V2"
    assert result.boundary == "E0-V2"


def test_assignments_are_deterministic_and_json_serializable() -> None:
    first = assign_evidence_boundary(e2_inputs(), ValidationProvenanceInputs())
    second = assign_evidence_boundary(e2_inputs(), ValidationProvenanceInputs())

    assert first == second
    assert json.loads(json.dumps(first.as_dict()))["evidence_boundary"] == "E2-V0"

    invalid = assign_event_support(e1_inputs(event_q=float("nan")))
    json.dumps(invalid.as_dict(), allow_nan=False)
