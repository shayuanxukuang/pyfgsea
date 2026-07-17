"""Deterministic TED event-support and validation-provenance assignment.

The module implements the orthogonal evidence descriptor used by TED:

* ``E0``--``E2`` describes support for the event in the study that produced it.
* ``V0``--``V4`` records the strongest qualifying validation provenance.

The axes are intentionally assigned independently.  In particular, a high
validation-provenance code cannot repair a failed event-support gate.  Missing
or malformed evidence inputs never promote an assignment: mandatory E inputs
fail closed to E0, and incomplete V observations remain at the strongest lower
provenance whose complete gates pass.

The functions return both stable reason codes and a structured audit trail so
that a caller can persist the exact basis for an assignment.  They do not infer
that any dataset is externally validated; callers must supply the observed
design facts explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Dict, Literal, Optional, Sequence, Tuple


EventSupportCode = Literal["E0", "E1", "E2"]
ValidationProvenanceCode = Literal["V0", "V1", "V2", "V3", "V4"]
IdentifiabilityStatus = Literal["identifiable", "limited", "not_identifiable"]
AuditStatus = Literal["pass", "fail", "missing", "not_applicable"]


def _json_safe(value: Any) -> Any:
    """Normalize audit values so strict JSON serialization always succeeds."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if isfinite(value) else str(value)
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_safe(item_method())
        except (TypeError, ValueError):
            pass
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return repr(value)
    return numeric if isfinite(numeric) else str(numeric)


@dataclass(frozen=True)
class GateAudit:
    """One immutable gate decision in an evidence-assignment audit trail."""

    gate: str
    status: AuditStatus
    reason_code: str
    criterion: str
    observed: Any = None

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly representation with stable field names."""

        return {
            "gate": self.gate,
            "status": self.status,
            "reason_code": self.reason_code,
            "criterion": self.criterion,
            "observed": _json_safe(self.observed),
        }


@dataclass(frozen=True)
class EventSupportThresholds:
    """Versioned default thresholds for the E0/E1/E2 assignment gates."""

    event_q_max: float = 0.10
    minimum_effective_blocks: int = 3
    block_q_max: float = 0.10
    direction_stability_min: float = 0.80
    negative_control_margin_min: float = 0.0
    matched_state_attenuation_max: float = 0.50

    def __post_init__(self) -> None:
        if not 0.0 <= self.event_q_max <= 1.0:
            raise ValueError("event_q_max must be within [0, 1]")
        if (
            isinstance(self.minimum_effective_blocks, bool)
            or not isinstance(self.minimum_effective_blocks, int)
            or self.minimum_effective_blocks < 1
        ):
            raise ValueError("minimum_effective_blocks must be a positive integer")
        if not 0.0 <= self.block_q_max <= 1.0:
            raise ValueError("block_q_max must be within [0, 1]")
        if not 0.0 <= self.direction_stability_min <= 1.0:
            raise ValueError("direction_stability_min must be within [0, 1]")
        if not isfinite(float(self.negative_control_margin_min)):
            raise ValueError("negative_control_margin_min must be finite")
        if not 0.0 <= self.matched_state_attenuation_max <= 1.0:
            raise ValueError("matched_state_attenuation_max must be within [0, 1]")


@dataclass(frozen=True)
class EventSupportInputs:
    """Observed inputs used to assign TED within-study event support.

    The first nine fields are mandatory for E1.  ``matched_state_*`` fields
    become mandatory when ``matched_state_required`` is true.  E2 additionally
    requires effective block support, stable direction, identifiable mode and
    the relevant negative-control gates.

    ``block_q`` and ``block_ci_excludes_zero`` form an OR gate: either a block
    permutation q value at or below the configured threshold, or a confidence
    interval excluding zero, is sufficient for that component of E2.
    """

    event_family_declared: Optional[bool] = None
    defensible_null_specified: Optional[bool] = None
    biological_units_present: Optional[bool] = None
    condition_batch_confounded: Optional[bool] = None
    identifiability_status: Optional[IdentifiabilityStatus] = None
    artifact_dominated: Optional[bool] = None
    event_q: Optional[float] = None
    retained_module: Optional[bool] = None
    basic_controls_pass: Optional[bool] = None

    matched_state_required: bool = False
    matched_state_overlap_pass: Optional[bool] = None
    matched_state_attenuation: Optional[float] = None

    effective_blocks: Optional[int] = None
    block_q: Optional[float] = None
    block_ci_excludes_zero: Optional[bool] = None
    direction_stability: Optional[float] = None
    mode_identifiable: Optional[bool] = None

    negative_controls_required: bool = True
    negative_control_pass: Optional[bool] = None
    negative_control_margin: Optional[float] = None


@dataclass(frozen=True)
class ValidationProvenanceInputs:
    """Observed design facts used to assign V0--V4 provenance.

    An ``*_observed`` switch marks a candidate provenance for evaluation.  If
    it is false, that provenance is not claimed.  If it is true, every listed
    component for that provenance is mandatory and missing values fail closed.

    V1 requires an aligned orthogonal outcome assessed above prespecified
    controls.  V2 requires a prespecified intervention reversal above matched
    controls.  V3 requires a same-system matched rescue with the predicted
    molecular or functional readout and adequate controls.  V4 requires a
    successful independent replication and a recorded V1/V2/V3 basis.
    """

    orthogonal_outcome_observed: bool = False
    outcome_assessment_prespecified: Optional[bool] = None
    outcome_aligned: Optional[bool] = None
    outcome_controls_pass: Optional[bool] = None

    intervention_reversal_observed: bool = False
    intervention_contrast_prespecified: Optional[bool] = None
    intervention_reversal_pass: Optional[bool] = None
    matched_intervention_controls_pass: Optional[bool] = None

    matched_rescue_observed: bool = False
    rescue_same_system: Optional[bool] = None
    predicted_readout_recovered: Optional[bool] = None
    matched_rescue_controls_pass: Optional[bool] = None

    independent_replication_observed: bool = False
    replication_independent: Optional[bool] = None
    independent_replication_pass: Optional[bool] = None
    replicated_validation_basis: Optional[str] = None


@dataclass(frozen=True)
class EventSupportAssignment:
    """Assigned E code, causal reason codes and complete gate audit."""

    code: EventSupportCode
    reason_codes: Tuple[str, ...]
    audit_trail: Tuple[GateAudit, ...]

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly event-support assignment."""

        return {
            "code": self.code,
            "reason_codes": list(self.reason_codes),
            "audit_trail": [item.as_dict() for item in self.audit_trail],
        }


@dataclass(frozen=True)
class ValidationProvenanceAssignment:
    """Assigned V code, reason codes and complete provenance-gate audit."""

    code: ValidationProvenanceCode
    reason_codes: Tuple[str, ...]
    audit_trail: Tuple[GateAudit, ...]
    replicated_validation_basis: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly validation-provenance assignment."""

        return {
            "code": self.code,
            "reason_codes": list(self.reason_codes),
            "replicated_validation_basis": self.replicated_validation_basis,
            "audit_trail": [item.as_dict() for item in self.audit_trail],
        }


@dataclass(frozen=True)
class EvidenceBoundaryAssignment:
    """Combined orthogonal E/V descriptor and the two independent audits."""

    event_support: EventSupportAssignment
    validation_provenance: ValidationProvenanceAssignment

    @property
    def boundary(self) -> str:
        """Return the schema-compatible combined label, for example E2-V1."""

        return f"{self.event_support.code}-{self.validation_provenance.code}"

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly combined descriptor."""

        return {
            "evidence_boundary": self.boundary,
            "event_support": self.event_support.as_dict(),
            "validation_provenance": self.validation_provenance.as_dict(),
        }


def _append_bool_gate(
    audit: list[GateAudit],
    *,
    gate: str,
    value: Optional[bool],
    true_is_pass: bool,
    pass_code: str,
    fail_code: str,
    missing_code: str,
    criterion: str,
) -> Optional[str]:
    """Audit a strict boolean and return a causal failure code, if any."""

    if value is None:
        audit.append(GateAudit(gate, "missing", missing_code, criterion, None))
        return missing_code
    if not isinstance(value, bool):
        reason = f"{gate.upper().replace('.', '_')}_INVALID"
        audit.append(GateAudit(gate, "fail", reason, criterion, value))
        return reason
    passed = value is true_is_pass
    audit.append(
        GateAudit(
            gate,
            "pass" if passed else "fail",
            pass_code if passed else fail_code,
            criterion,
            value,
        )
    )
    return None if passed else fail_code


def _finite_number(value: Any) -> bool:
    """Return true only for a finite, non-boolean numeric value."""

    if value is None or isinstance(value, bool):
        return False
    try:
        return isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _deduplicate(items: Sequence[str]) -> Tuple[str, ...]:
    """Deduplicate reason codes while preserving deterministic gate order."""

    return tuple(dict.fromkeys(items))


def assign_event_support(
    inputs: EventSupportInputs,
    *,
    thresholds: EventSupportThresholds = EventSupportThresholds(),
) -> EventSupportAssignment:
    """Assign E0, E1 or E2 using explicit fail-closed TED gates.

    Evidence values that are absent, malformed or outside their valid range do
    not raise a data-validation exception; they produce a conservative code and
    an audit reason.  Invalid threshold configuration raises ``ValueError`` at
    construction time because it is a programming/configuration error.
    """

    if not isinstance(inputs, EventSupportInputs):
        raise TypeError("inputs must be an EventSupportInputs instance")
    if not isinstance(thresholds, EventSupportThresholds):
        raise TypeError("thresholds must be an EventSupportThresholds instance")

    audit: list[GateAudit] = []
    e0_reasons: list[str] = []

    base_bool_gates = (
        (
            "event_family_declared",
            inputs.event_family_declared,
            True,
            "EVENT_FAMILY_DECLARED",
            "EVENT_FAMILY_NOT_DECLARED",
            "EVENT_FAMILY_DECLARATION_MISSING",
            "the event family is declared before testing",
        ),
        (
            "defensible_null_specified",
            inputs.defensible_null_specified,
            True,
            "DEFENSIBLE_NULL_SPECIFIED",
            "DEFENSIBLE_NULL_NOT_SPECIFIED",
            "DEFENSIBLE_NULL_STATUS_MISSING",
            "a defensible null is specified",
        ),
        (
            "biological_units_present",
            inputs.biological_units_present,
            True,
            "BIOLOGICAL_UNITS_PRESENT",
            "BIOLOGICAL_UNITS_ABSENT",
            "BIOLOGICAL_UNITS_STATUS_MISSING",
            "independent biological analysis units are present",
        ),
        (
            "condition_batch_confounded",
            inputs.condition_batch_confounded,
            False,
            "CONDITION_BATCH_NOT_COMPLETELY_CONFOUNDED",
            "CONDITION_BATCH_COMPLETELY_CONFOUNDED",
            "CONDITION_BATCH_CONFOUNDING_STATUS_MISSING",
            "condition and batch are not completely confounded",
        ),
        (
            "artifact_dominated",
            inputs.artifact_dominated,
            False,
            "EVENT_NOT_ARTIFACT_DOMINATED",
            "EVENT_ARTIFACT_DOMINATED",
            "ARTIFACT_STATUS_MISSING",
            "the event is not artifact dominated",
        ),
        (
            "retained_module",
            inputs.retained_module,
            True,
            "MODULE_RETAINED",
            "MODULE_NOT_RETAINED",
            "RETAINED_MODULE_STATUS_MISSING",
            "the tested module is retained after input quality control",
        ),
        (
            "basic_controls_pass",
            inputs.basic_controls_pass,
            True,
            "BASIC_CONTROLS_PASS",
            "BASIC_CONTROLS_FAIL",
            "BASIC_CONTROLS_STATUS_MISSING",
            "adequate basic controls pass",
        ),
    )
    for gate_args in base_bool_gates:
        reason = _append_bool_gate(
            audit,
            gate=gate_args[0],
            value=gate_args[1],
            true_is_pass=gate_args[2],
            pass_code=gate_args[3],
            fail_code=gate_args[4],
            missing_code=gate_args[5],
            criterion=gate_args[6],
        )
        if reason is not None:
            e0_reasons.append(reason)

    valid_identifiability = {"identifiable", "limited", "not_identifiable"}
    identifiability = inputs.identifiability_status
    if identifiability is None:
        reason = "IDENTIFIABILITY_STATUS_MISSING"
        audit.append(
            GateAudit(
                "identifiability_status",
                "missing",
                reason,
                "status is identifiable or limited for E1",
                None,
            )
        )
        e0_reasons.append(reason)
    elif identifiability not in valid_identifiability:
        reason = "IDENTIFIABILITY_STATUS_INVALID"
        audit.append(
            GateAudit(
                "identifiability_status",
                "fail",
                reason,
                "status is one of identifiable, limited or not_identifiable",
                identifiability,
            )
        )
        e0_reasons.append(reason)
    elif identifiability == "not_identifiable":
        reason = "EVENT_NOT_IDENTIFIABLE"
        audit.append(
            GateAudit(
                "identifiability_status",
                "fail",
                reason,
                "status is identifiable or limited for E1",
                identifiability,
            )
        )
        e0_reasons.append(reason)
    else:
        audit.append(
            GateAudit(
                "identifiability_status",
                "pass",
                "EVENT_AT_LEAST_LIMITED_IDENTIFIABILITY",
                "status is identifiable or limited for E1",
                identifiability,
            )
        )

    event_q = inputs.event_q
    if event_q is None:
        reason = "EVENT_Q_MISSING"
        audit.append(
            GateAudit(
                "event_q",
                "missing",
                reason,
                f"event_q <= {thresholds.event_q_max}",
                None,
            )
        )
        e0_reasons.append(reason)
    elif not _finite_number(event_q) or not 0.0 <= float(event_q) <= 1.0:
        reason = "EVENT_Q_INVALID"
        audit.append(
            GateAudit(
                "event_q",
                "fail",
                reason,
                f"event_q is finite, within [0, 1], and <= {thresholds.event_q_max}",
                event_q,
            )
        )
        e0_reasons.append(reason)
    elif float(event_q) > thresholds.event_q_max:
        reason = "EVENT_Q_ABOVE_THRESHOLD"
        audit.append(
            GateAudit(
                "event_q",
                "fail",
                reason,
                f"event_q <= {thresholds.event_q_max}",
                float(event_q),
            )
        )
        e0_reasons.append(reason)
    else:
        audit.append(
            GateAudit(
                "event_q",
                "pass",
                "EVENT_Q_GATE_PASS",
                f"event_q <= {thresholds.event_q_max}",
                float(event_q),
            )
        )

    if not isinstance(inputs.matched_state_required, bool):
        reason = "MATCHED_STATE_REQUIREMENT_INVALID"
        audit.append(
            GateAudit(
                "matched_state_required",
                "fail",
                reason,
                "matched_state_required is boolean",
                inputs.matched_state_required,
            )
        )
        e0_reasons.append(reason)
    elif inputs.matched_state_required:
        reason = _append_bool_gate(
            audit,
            gate="matched_state_overlap_pass",
            value=inputs.matched_state_overlap_pass,
            true_is_pass=True,
            pass_code="MATCHED_STATE_OVERLAP_PASS",
            fail_code="MATCHED_STATE_OVERLAP_FAIL",
            missing_code="MATCHED_STATE_OVERLAP_STATUS_MISSING",
            criterion="adequate matched-state overlap is demonstrated",
        )
        if reason is not None:
            e0_reasons.append(reason)
    else:
        audit.append(
            GateAudit(
                "matched_state_overlap_pass",
                "not_applicable",
                "MATCHED_STATE_NOT_REQUIRED",
                "matched-state overlap is required only when composition is plausible",
                inputs.matched_state_overlap_pass,
            )
        )

    if e0_reasons:
        return EventSupportAssignment("E0", _deduplicate(e0_reasons), tuple(audit))

    e2_reasons: list[str] = []

    if inputs.identifiability_status == "identifiable":
        audit.append(
            GateAudit(
                "e2_identifiability",
                "pass",
                "E2_IDENTIFIABILITY_PASS",
                "identifiability_status is identifiable",
                inputs.identifiability_status,
            )
        )
    else:
        reason = "E2_IDENTIFIABILITY_LIMITED"
        audit.append(
            GateAudit(
                "e2_identifiability",
                "fail",
                reason,
                "identifiability_status is identifiable",
                inputs.identifiability_status,
            )
        )
        e2_reasons.append(reason)

    blocks = inputs.effective_blocks
    if blocks is None:
        reason = "EFFECTIVE_BLOCKS_MISSING"
        audit.append(
            GateAudit(
                "effective_blocks",
                "missing",
                reason,
                f"effective_blocks >= {thresholds.minimum_effective_blocks}",
                None,
            )
        )
        e2_reasons.append(reason)
    elif isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 0:
        reason = "EFFECTIVE_BLOCKS_INVALID"
        audit.append(
            GateAudit(
                "effective_blocks",
                "fail",
                reason,
                f"effective_blocks is an integer >= {thresholds.minimum_effective_blocks}",
                blocks,
            )
        )
        e2_reasons.append(reason)
    elif blocks < thresholds.minimum_effective_blocks:
        reason = "INSUFFICIENT_EFFECTIVE_BLOCKS"
        audit.append(
            GateAudit(
                "effective_blocks",
                "fail",
                reason,
                f"effective_blocks >= {thresholds.minimum_effective_blocks}",
                blocks,
            )
        )
        e2_reasons.append(reason)
    else:
        audit.append(
            GateAudit(
                "effective_blocks",
                "pass",
                "EFFECTIVE_BLOCKS_PASS",
                f"effective_blocks >= {thresholds.minimum_effective_blocks}",
                blocks,
            )
        )

    block_q = inputs.block_q
    ci_value = inputs.block_ci_excludes_zero
    block_q_valid = _finite_number(block_q) and 0.0 <= float(block_q) <= 1.0
    block_q_pass = block_q_valid and float(block_q) <= thresholds.block_q_max
    ci_valid = isinstance(ci_value, bool)
    ci_pass = ci_value is True
    block_observed = {"block_q": block_q, "block_ci_excludes_zero": ci_value}
    if block_q_pass or ci_pass:
        audit.append(
            GateAudit(
                "block_support",
                "pass",
                "BLOCK_SUPPORT_PASS",
                f"block_q <= {thresholds.block_q_max} OR block CI excludes zero",
                block_observed,
            )
        )
    elif block_q is None and ci_value is None:
        reason = "BLOCK_SUPPORT_MISSING"
        audit.append(
            GateAudit(
                "block_support",
                "missing",
                reason,
                f"block_q <= {thresholds.block_q_max} OR block CI excludes zero",
                block_observed,
            )
        )
        e2_reasons.append(reason)
    elif (block_q is not None and not block_q_valid) or (
        ci_value is not None and not ci_valid
    ):
        reason = "BLOCK_SUPPORT_INVALID"
        audit.append(
            GateAudit(
                "block_support",
                "fail",
                reason,
                f"valid block_q <= {thresholds.block_q_max} OR boolean CI support",
                block_observed,
            )
        )
        e2_reasons.append(reason)
    else:
        reason = "BLOCK_SUPPORT_FAIL"
        audit.append(
            GateAudit(
                "block_support",
                "fail",
                reason,
                f"block_q <= {thresholds.block_q_max} OR block CI excludes zero",
                block_observed,
            )
        )
        e2_reasons.append(reason)

    stability = inputs.direction_stability
    if stability is None:
        reason = "DIRECTION_STABILITY_MISSING"
        audit.append(
            GateAudit(
                "direction_stability",
                "missing",
                reason,
                f"direction_stability >= {thresholds.direction_stability_min}",
                None,
            )
        )
        e2_reasons.append(reason)
    elif not _finite_number(stability) or not 0.0 <= float(stability) <= 1.0:
        reason = "DIRECTION_STABILITY_INVALID"
        audit.append(
            GateAudit(
                "direction_stability",
                "fail",
                reason,
                "direction_stability is finite and within [0, 1]",
                stability,
            )
        )
        e2_reasons.append(reason)
    elif float(stability) < thresholds.direction_stability_min:
        reason = "DIRECTION_STABILITY_BELOW_THRESHOLD"
        audit.append(
            GateAudit(
                "direction_stability",
                "fail",
                reason,
                f"direction_stability >= {thresholds.direction_stability_min}",
                float(stability),
            )
        )
        e2_reasons.append(reason)
    else:
        audit.append(
            GateAudit(
                "direction_stability",
                "pass",
                "DIRECTION_STABILITY_PASS",
                f"direction_stability >= {thresholds.direction_stability_min}",
                float(stability),
            )
        )

    reason = _append_bool_gate(
        audit,
        gate="mode_identifiable",
        value=inputs.mode_identifiable,
        true_is_pass=True,
        pass_code="MODE_IDENTIFIABILITY_PASS",
        fail_code="MODE_NOT_IDENTIFIABLE_FOR_E2",
        missing_code="MODE_IDENTIFIABILITY_STATUS_MISSING",
        criterion="the event mode is identifiable",
    )
    if reason is not None:
        e2_reasons.append(reason)

    if inputs.matched_state_required:
        attenuation = inputs.matched_state_attenuation
        if attenuation is None:
            reason = "MATCHED_STATE_ATTENUATION_MISSING"
            audit.append(
                GateAudit(
                    "matched_state_attenuation",
                    "missing",
                    reason,
                    f"attenuation < {thresholds.matched_state_attenuation_max}",
                    None,
                )
            )
            e2_reasons.append(reason)
        elif not _finite_number(attenuation) or not 0.0 <= float(attenuation) <= 1.0:
            reason = "MATCHED_STATE_ATTENUATION_INVALID"
            audit.append(
                GateAudit(
                    "matched_state_attenuation",
                    "fail",
                    reason,
                    "attenuation is finite and within [0, 1]",
                    attenuation,
                )
            )
            e2_reasons.append(reason)
        elif float(attenuation) >= thresholds.matched_state_attenuation_max:
            reason = "MATCHED_STATE_ATTENUATION_TOO_LARGE"
            audit.append(
                GateAudit(
                    "matched_state_attenuation",
                    "fail",
                    reason,
                    f"attenuation < {thresholds.matched_state_attenuation_max}",
                    float(attenuation),
                )
            )
            e2_reasons.append(reason)
        else:
            audit.append(
                GateAudit(
                    "matched_state_attenuation",
                    "pass",
                    "MATCHED_STATE_ATTENUATION_PASS",
                    f"attenuation < {thresholds.matched_state_attenuation_max}",
                    float(attenuation),
                )
            )
    else:
        audit.append(
            GateAudit(
                "matched_state_attenuation",
                "not_applicable",
                "MATCHED_STATE_NOT_REQUIRED",
                "matched-state attenuation is required only when matching is required",
                inputs.matched_state_attenuation,
            )
        )

    if not isinstance(inputs.negative_controls_required, bool):
        reason = "NEGATIVE_CONTROL_REQUIREMENT_INVALID"
        audit.append(
            GateAudit(
                "negative_controls_required",
                "fail",
                reason,
                "negative_controls_required is boolean",
                inputs.negative_controls_required,
            )
        )
        e2_reasons.append(reason)
    elif inputs.negative_controls_required:
        reason = _append_bool_gate(
            audit,
            gate="negative_control_pass",
            value=inputs.negative_control_pass,
            true_is_pass=True,
            pass_code="NEGATIVE_CONTROL_GATES_PASS",
            fail_code="NEGATIVE_CONTROL_GATES_FAIL",
            missing_code="NEGATIVE_CONTROL_STATUS_MISSING",
            criterion="all relevant negative-control gates pass",
        )
        if reason is not None:
            e2_reasons.append(reason)

        margin = inputs.negative_control_margin
        if margin is None:
            reason = "NEGATIVE_CONTROL_MARGIN_MISSING"
            audit.append(
                GateAudit(
                    "negative_control_margin",
                    "missing",
                    reason,
                    f"margin > {thresholds.negative_control_margin_min}",
                    None,
                )
            )
            e2_reasons.append(reason)
        elif not _finite_number(margin):
            reason = "NEGATIVE_CONTROL_MARGIN_INVALID"
            audit.append(
                GateAudit(
                    "negative_control_margin",
                    "fail",
                    reason,
                    "negative-control margin is finite",
                    margin,
                )
            )
            e2_reasons.append(reason)
        elif float(margin) <= thresholds.negative_control_margin_min:
            reason = "NEGATIVE_CONTROL_MARGIN_NONPOSITIVE"
            audit.append(
                GateAudit(
                    "negative_control_margin",
                    "fail",
                    reason,
                    f"margin > {thresholds.negative_control_margin_min}",
                    float(margin),
                )
            )
            e2_reasons.append(reason)
        else:
            audit.append(
                GateAudit(
                    "negative_control_margin",
                    "pass",
                    "NEGATIVE_CONTROL_MARGIN_PASS",
                    f"margin > {thresholds.negative_control_margin_min}",
                    float(margin),
                )
            )
    else:
        audit.append(
            GateAudit(
                "negative_control_pass",
                "not_applicable",
                "NEGATIVE_CONTROLS_NOT_REQUIRED",
                "relevant negative controls are explicitly declared not applicable",
                inputs.negative_control_pass,
            )
        )
        audit.append(
            GateAudit(
                "negative_control_margin",
                "not_applicable",
                "NEGATIVE_CONTROLS_NOT_REQUIRED",
                "a control margin is not required when controls are not applicable",
                inputs.negative_control_margin,
            )
        )

    if e2_reasons:
        return EventSupportAssignment("E1", _deduplicate(e2_reasons), tuple(audit))
    return EventSupportAssignment("E2", ("E2_ALL_GATES_PASS",), tuple(audit))


def _audit_provenance_component(
    audit: list[GateAudit],
    failures: list[str],
    *,
    gate: str,
    value: Optional[bool],
    pass_code: str,
    fail_code: str,
    missing_code: str,
    criterion: str,
) -> None:
    reason = _append_bool_gate(
        audit,
        gate=gate,
        value=value,
        true_is_pass=True,
        pass_code=pass_code,
        fail_code=fail_code,
        missing_code=missing_code,
        criterion=criterion,
    )
    if reason is not None:
        failures.append(reason)


def assign_validation_provenance(
    inputs: ValidationProvenanceInputs,
) -> ValidationProvenanceAssignment:
    """Assign the strongest complete V0--V4 provenance without inference.

    A candidate provenance is evaluated only when its ``*_observed`` field is
    true.  Incomplete candidates do not promote the result.  Lower complete
    candidates remain valid when a higher candidate fails, and the higher-gate
    failure remains visible in the returned audit trail and reason codes.
    """

    if not isinstance(inputs, ValidationProvenanceInputs):
        raise TypeError("inputs must be a ValidationProvenanceInputs instance")

    audit: list[GateAudit] = []
    candidate_pass: Dict[str, bool] = {
        "V1": False,
        "V2": False,
        "V3": False,
        "V4": False,
    }
    candidate_failures: Dict[str, Tuple[str, ...]] = {}

    observed_specs = (
        ("V1", "orthogonal_outcome_observed", inputs.orthogonal_outcome_observed),
        ("V2", "intervention_reversal_observed", inputs.intervention_reversal_observed),
        ("V3", "matched_rescue_observed", inputs.matched_rescue_observed),
        (
            "V4",
            "independent_replication_observed",
            inputs.independent_replication_observed,
        ),
    )
    observed_valid: Dict[str, bool] = {}
    for code, gate, value in observed_specs:
        if not isinstance(value, bool):
            reason = f"{code}_OBSERVATION_FLAG_INVALID"
            audit.append(
                GateAudit(gate, "fail", reason, "observation flag is boolean", value)
            )
            candidate_failures[code] = (reason,)
            observed_valid[code] = False
        elif value:
            audit.append(
                GateAudit(
                    gate,
                    "pass",
                    f"{code}_OBSERVED",
                    "candidate evidence is observed",
                    value,
                )
            )
            observed_valid[code] = True
        else:
            audit.append(
                GateAudit(
                    gate,
                    "not_applicable",
                    f"{code}_NOT_OBSERVED",
                    "candidate evidence is explicitly observed before evaluation",
                    value,
                )
            )
            observed_valid[code] = False

    if observed_valid["V1"]:
        failures: list[str] = []
        _audit_provenance_component(
            audit,
            failures,
            gate="outcome_assessment_prespecified",
            value=inputs.outcome_assessment_prespecified,
            pass_code="V1_OUTCOME_ASSESSMENT_PRESPECIFIED",
            fail_code="V1_OUTCOME_ASSESSMENT_NOT_PRESPECIFIED",
            missing_code="V1_OUTCOME_ASSESSMENT_STATUS_MISSING",
            criterion="the outcome assessment and controls were prespecified",
        )
        _audit_provenance_component(
            audit,
            failures,
            gate="outcome_aligned",
            value=inputs.outcome_aligned,
            pass_code="V1_OUTCOME_ALIGNED",
            fail_code="V1_OUTCOME_NOT_ALIGNED",
            missing_code="V1_OUTCOME_ALIGNMENT_MISSING",
            criterion="the orthogonal outcome aligns with the event",
        )
        _audit_provenance_component(
            audit,
            failures,
            gate="outcome_controls_pass",
            value=inputs.outcome_controls_pass,
            pass_code="V1_OUTCOME_CONTROLS_PASS",
            fail_code="V1_OUTCOME_CONTROLS_FAIL",
            missing_code="V1_OUTCOME_CONTROLS_STATUS_MISSING",
            criterion="the outcome exceeds prespecified controls",
        )
        candidate_pass["V1"] = not failures
        candidate_failures["V1"] = _deduplicate(failures)
        audit.append(
            GateAudit(
                "v1_assignment",
                "pass" if not failures else "fail",
                "V1_GATES_PASS" if not failures else "V1_GATES_INCOMPLETE",
                "all V1 components pass",
                not failures,
            )
        )

    if observed_valid["V2"]:
        failures = []
        _audit_provenance_component(
            audit,
            failures,
            gate="intervention_contrast_prespecified",
            value=inputs.intervention_contrast_prespecified,
            pass_code="V2_INTERVENTION_CONTRAST_PRESPECIFIED",
            fail_code="V2_INTERVENTION_CONTRAST_NOT_PRESPECIFIED",
            missing_code="V2_INTERVENTION_CONTRAST_STATUS_MISSING",
            criterion="the intervention contrast was prespecified",
        )
        _audit_provenance_component(
            audit,
            failures,
            gate="intervention_reversal_pass",
            value=inputs.intervention_reversal_pass,
            pass_code="V2_INTERVENTION_REVERSAL_PASS",
            fail_code="V2_INTERVENTION_REVERSAL_FAIL",
            missing_code="V2_INTERVENTION_REVERSAL_STATUS_MISSING",
            criterion="the intervention reverses or attenuates the event",
        )
        _audit_provenance_component(
            audit,
            failures,
            gate="matched_intervention_controls_pass",
            value=inputs.matched_intervention_controls_pass,
            pass_code="V2_MATCHED_CONTROLS_PASS",
            fail_code="V2_MATCHED_CONTROLS_FAIL",
            missing_code="V2_MATCHED_CONTROLS_STATUS_MISSING",
            criterion="the reversal exceeds matched controls",
        )
        candidate_pass["V2"] = not failures
        candidate_failures["V2"] = _deduplicate(failures)
        audit.append(
            GateAudit(
                "v2_assignment",
                "pass" if not failures else "fail",
                "V2_GATES_PASS" if not failures else "V2_GATES_INCOMPLETE",
                "all V2 components pass",
                not failures,
            )
        )

    if observed_valid["V3"]:
        failures = []
        _audit_provenance_component(
            audit,
            failures,
            gate="rescue_same_system",
            value=inputs.rescue_same_system,
            pass_code="V3_RESCUE_SAME_SYSTEM",
            fail_code="V3_RESCUE_NOT_SAME_SYSTEM",
            missing_code="V3_RESCUE_SYSTEM_STATUS_MISSING",
            criterion="the rescue is matched in the relevant system",
        )
        _audit_provenance_component(
            audit,
            failures,
            gate="predicted_readout_recovered",
            value=inputs.predicted_readout_recovered,
            pass_code="V3_PREDICTED_READOUT_RECOVERED",
            fail_code="V3_PREDICTED_READOUT_NOT_RECOVERED",
            missing_code="V3_PREDICTED_READOUT_STATUS_MISSING",
            criterion="the predicted molecular or functional readout recovers",
        )
        _audit_provenance_component(
            audit,
            failures,
            gate="matched_rescue_controls_pass",
            value=inputs.matched_rescue_controls_pass,
            pass_code="V3_MATCHED_RESCUE_CONTROLS_PASS",
            fail_code="V3_MATCHED_RESCUE_CONTROLS_FAIL",
            missing_code="V3_MATCHED_RESCUE_CONTROLS_STATUS_MISSING",
            criterion="matched rescue controls pass",
        )
        candidate_pass["V3"] = not failures
        candidate_failures["V3"] = _deduplicate(failures)
        audit.append(
            GateAudit(
                "v3_assignment",
                "pass" if not failures else "fail",
                "V3_GATES_PASS" if not failures else "V3_GATES_INCOMPLETE",
                "all V3 components pass",
                not failures,
            )
        )

    replicated_basis: Optional[str] = None
    if observed_valid["V4"]:
        failures = []
        _audit_provenance_component(
            audit,
            failures,
            gate="replication_independent",
            value=inputs.replication_independent,
            pass_code="V4_REPLICATION_INDEPENDENT",
            fail_code="V4_REPLICATION_NOT_INDEPENDENT",
            missing_code="V4_REPLICATION_INDEPENDENCE_STATUS_MISSING",
            criterion="the replication is independent of development evidence",
        )
        _audit_provenance_component(
            audit,
            failures,
            gate="independent_replication_pass",
            value=inputs.independent_replication_pass,
            pass_code="V4_REPLICATION_PASS",
            fail_code="V4_REPLICATION_FAIL",
            missing_code="V4_REPLICATION_STATUS_MISSING",
            criterion="the independent replication passes its declared gates",
        )
        basis = inputs.replicated_validation_basis
        if basis is None:
            reason = "V4_REPLICATED_BASIS_MISSING"
            audit.append(
                GateAudit(
                    "replicated_validation_basis",
                    "missing",
                    reason,
                    "replicated basis is one of V1, V2 or V3",
                    None,
                )
            )
            failures.append(reason)
        elif basis not in {"V1", "V2", "V3"}:
            reason = "V4_REPLICATED_BASIS_INVALID"
            audit.append(
                GateAudit(
                    "replicated_validation_basis",
                    "fail",
                    reason,
                    "replicated basis is one of V1, V2 or V3",
                    basis,
                )
            )
            failures.append(reason)
        else:
            replicated_basis = basis
            audit.append(
                GateAudit(
                    "replicated_validation_basis",
                    "pass",
                    "V4_REPLICATED_BASIS_RECORDED",
                    "replicated basis is one of V1, V2 or V3",
                    basis,
                )
            )
        candidate_pass["V4"] = not failures
        candidate_failures["V4"] = _deduplicate(failures)
        audit.append(
            GateAudit(
                "v4_assignment",
                "pass" if not failures else "fail",
                "V4_GATES_PASS" if not failures else "V4_GATES_INCOMPLETE",
                "all V4 components pass and the replicated basis is retained",
                not failures,
            )
        )

    selected: ValidationProvenanceCode = "V0"
    for candidate in ("V1", "V2", "V3", "V4"):
        if candidate_pass[candidate]:
            selected = candidate  # type: ignore[assignment]

    reasons: list[str] = []
    if selected == "V0":
        attempted = [
            code
            for code, _, value in observed_specs
            if value is True or code in candidate_failures
        ]
        if not attempted:
            reasons.append("V0_COMPUTATIONAL_ONLY")
        else:
            reasons.append("NO_QUALIFYING_VALIDATION_PROVENANCE")
            for code in attempted:
                reasons.extend(candidate_failures.get(code, ()))
    else:
        reasons.append(f"{selected}_GATES_PASS")
        selected_number = int(selected[1])
        for code in ("V1", "V2", "V3", "V4"):
            if (
                int(code[1]) > selected_number
                and inputs.__getattribute__(
                    {
                        "V1": "orthogonal_outcome_observed",
                        "V2": "intervention_reversal_observed",
                        "V3": "matched_rescue_observed",
                        "V4": "independent_replication_observed",
                    }[code]
                )
                is True
            ):
                reasons.extend(candidate_failures.get(code, ()))

    return ValidationProvenanceAssignment(
        selected,
        _deduplicate(reasons),
        tuple(audit),
        replicated_basis if selected == "V4" else None,
    )


def assign_evidence_boundary(
    event_inputs: EventSupportInputs,
    provenance_inputs: ValidationProvenanceInputs,
    *,
    thresholds: EventSupportThresholds = EventSupportThresholds(),
) -> EvidenceBoundaryAssignment:
    """Assign both axes independently and return their combined descriptor."""

    return EvidenceBoundaryAssignment(
        event_support=assign_event_support(event_inputs, thresholds=thresholds),
        validation_provenance=assign_validation_provenance(provenance_inputs),
    )


__all__ = [
    "AuditStatus",
    "EvidenceBoundaryAssignment",
    "EventSupportAssignment",
    "EventSupportCode",
    "EventSupportInputs",
    "EventSupportThresholds",
    "GateAudit",
    "IdentifiabilityStatus",
    "ValidationProvenanceAssignment",
    "ValidationProvenanceCode",
    "ValidationProvenanceInputs",
    "assign_event_support",
    "assign_evidence_boundary",
    "assign_validation_provenance",
]
