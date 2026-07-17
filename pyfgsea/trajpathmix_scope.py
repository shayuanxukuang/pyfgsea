from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml


SCHEMA_NAME = "trajpathmix_two_layer_scope_contract"
SCHEMA_VERSION = "1.0.0"
DECISION_ID = "trajpathmix_timing_scope_revision_2026-07-14"
FROZEN_SCOPE_PAYLOAD_SHA256 = (
    "58567ebee0d5da697ba5259f5c3303095a987e15a0e9eb14e9658ebf351786dd"
)


CORE_CAPABILITIES = (
    "donor_level_pathway_functional_inference",
    "fixed_common_grid",
    "missing_donor_bin_remains_missing",
    "whole_donor_condition_or_residual_mapping",
    "covariate_and_design_estimability_audit",
    "simultaneous_pathway_effect_curve",
    "supported_region_integrated_effect",
    "regulation_occupancy_fate_decomposition",
    "pathway_family_maxT_and_BY",
    "donor_influence_LODO_dynamic_leading_edge",
    "fail_closed_on_insufficient_information",
)

CONDITIONAL_TIMING_CAPABILITIES = (
    "onset",
    "duration",
    "activation_delay",
    "phase_shift",
    "transient_versus_sustained",
    "heterochrony",
    "peak_location",
)

REQUIRED_TIMING_GATES = (
    "continuous_common_support",
    "trajectory_method_concordance",
    "independent_direction_anchor",
    "acceptable_location_error",
    "acceptable_mde",
    "calibrated_event_false_positive_rate",
)

TIMING_ONLY_COLUMNS = (
    "activation_onset",
    "suppression_onset",
    "peak_time",
    "trough_time",
    "duration",
    "sharpness",
    "direction_switch",
    "direction_switch_count",
    "recurrence",
    "event_label",
    "event_confidence_class",
    "event_confidence_reason",
)


@dataclass(frozen=True)
class TimingActivationDecision:
    status: str
    context: str
    timing_claim_allowed: bool
    gate_evidence: dict[str, bool]
    failed_gates: tuple[str, ...]
    fallback: str = "supported_region_functional_effect_inference"

    @property
    def activated(self) -> bool:
        return self.status == "activated" and self.timing_claim_allowed

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["failed_gates"] = list(self.failed_gates)
        return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _scope_payload_sha256(config: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in config.items()
        if key != "frozen_payload_sha256" and not str(key).startswith("_")
    }
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def validate_scope_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "decision_id": DECISION_ID,
        "revision_mode": "create_once_append_only",
        "frozen_payload_sha256": FROZEN_SCOPE_PAYLOAD_SHA256,
    }
    for key, expected in checks.items():
        if config.get(key) != expected:
            raise ValueError(
                f"Frozen TrajPathMix scope mismatch for {key}: "
                f"expected {expected!r}, observed {config.get(key)!r}"
            )
    observed_hash = _scope_payload_sha256(config)
    if observed_hash != FROZEN_SCOPE_PAYLOAD_SHA256:
        raise ValueError(
            "Frozen TrajPathMix scope mismatch for recomputed payload SHA-256"
        )
    if config.get("universal_timing_claim", {}).get("status") != "closed":
        raise ValueError("Universal timing claim must remain closed")
    layers = config.get("layers", {})
    if tuple(layers.get("functional_core", {}).get("capabilities", [])) != CORE_CAPABILITIES:
        raise ValueError("Functional-core capability set differs from the freeze")
    conditional = layers.get("conditional_timing", {})
    if conditional.get("default_output") is not False:
        raise ValueError("Conditional timing may not become a default output")
    if tuple(conditional.get("capabilities", [])) != CONDITIONAL_TIMING_CAPABILITIES:
        raise ValueError("Conditional timing capability set differs from the freeze")
    if tuple(config.get("conditional_activation_requires_all", [])) != REQUIRED_TIMING_GATES:
        raise ValueError("Timing activation gates differ from the freeze")
    value = dict(config)
    value["_config_payload_sha256"] = FROZEN_SCOPE_PAYLOAD_SHA256
    return value


def load_scope_contract(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("TrajPathMix scope contract must be a YAML mapping")
    return validate_scope_contract(value)


def evaluate_timing_activation(
    evidence: Mapping[str, bool] | None = None,
    *,
    context: str = "dataset_specific",
) -> TimingActivationDecision:
    """Apply the project-level all-gates timing activation rule.

    Missing evidence fails closed. This decision controls claim permission and
    default output projection; it does not delete the low-level timing code
    used by preregistered calibration and recovery benchmarks.
    """

    supplied = dict(evidence or {})
    unknown = sorted(set(supplied) - set(REQUIRED_TIMING_GATES))
    if unknown:
        raise ValueError(f"Unknown timing-gate evidence: {unknown}")
    invalid = sorted(key for key, value in supplied.items() if not isinstance(value, bool))
    if invalid:
        raise TypeError(f"Timing-gate evidence must be boolean: {invalid}")
    normalized = {
        gate: bool(supplied.get(gate, False)) for gate in REQUIRED_TIMING_GATES
    }
    failed = tuple(gate for gate, passed in normalized.items() if not passed)
    activated = not failed
    return TimingActivationDecision(
        status="activated" if activated else "conditional_only",
        context=str(context),
        timing_claim_allowed=activated,
        gate_evidence=normalized,
        failed_gates=failed,
    )


def annotate_timing_scope(
    events: pd.DataFrame,
    decision: TimingActivationDecision | None = None,
) -> pd.DataFrame:
    """Attach the two-layer scope decision without discarding calculations."""

    resolved = decision or evaluate_timing_activation()
    annotated = events.copy()
    annotated["trajpathmix_primary_layer"] = "functional_core"
    annotated["timing_module_status"] = resolved.status
    annotated["timing_claim_allowed"] = resolved.timing_claim_allowed
    annotated["timing_failed_gates"] = ";".join(resolved.failed_gates)
    return annotated


def project_primary_effect_output(
    events: pd.DataFrame,
    decision: TimingActivationDecision | None = None,
) -> pd.DataFrame:
    """Return the default v1 event table under the two-layer scope.

    Timing columns appear in the default projection only after every frozen
    activation gate passes. Core magnitude and supported-region summaries are
    retained when timing is conditional-only.
    """

    resolved = decision or evaluate_timing_activation()
    projected = annotate_timing_scope(events, resolved)
    if resolved.activated:
        return projected
    return projected.drop(
        columns=[column for column in TIMING_ONLY_COLUMNS if column in projected],
        errors="ignore",
    )


__all__ = [
    "CONDITIONAL_TIMING_CAPABILITIES",
    "CORE_CAPABILITIES",
    "DECISION_ID",
    "FROZEN_SCOPE_PAYLOAD_SHA256",
    "REQUIRED_TIMING_GATES",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "TIMING_ONLY_COLUMNS",
    "TimingActivationDecision",
    "annotate_timing_scope",
    "evaluate_timing_activation",
    "load_scope_contract",
    "project_primary_effect_output",
    "validate_scope_contract",
]
