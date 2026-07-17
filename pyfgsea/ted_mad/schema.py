"""Strict input validation for TED-MAD/ARD YAML protocols."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"

ALLOWED_EVIDENCE_FAMILIES = {
    "family_block_robustness",
    "proliferation_adjusted_mediation",
    "counterfactual_ot",
    "day_stratified_timing",
    "negative_mediator_controls",
    "dynamic_precedence",
    "rescue_prediction",
    "external_GATA1_KD",
    "external_T21_multiome",
    "state_matched",
    "direct_rescue",
    "orthogonal_perturbation",
    "independent_replication",
}


class TedMadValidationError(ValueError):
    """Raised when a TED-MAD protocol input fails schema validation."""


def load_schema(name: str) -> dict[str, Any]:
    """Load a JSON schema bundled with the installed TED-MAD package."""

    path = SCHEMA_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


def _path(error: ValidationError) -> str:
    parts = []
    for part in error.absolute_path:
        if isinstance(part, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{part}]"
            else:
                parts.append(f"[{part}]")
        else:
            parts.append(str(part))
    return ".".join(parts)


def _format_jsonschema_error(error: ValidationError) -> str:
    path = _path(error)
    location = f" at {path}" if path else ""
    if error.validator == "required":
        missing = str(error.message).split("'")[1]
        return f"Missing required field: {missing}{location}"
    if error.validator == "enum":
        value = error.instance
        if error.absolute_path and error.absolute_path[-1] == "evidence_family":
            return f"Invalid evidence_family: {value}{location}"
        return f"Invalid value for {path or 'input'}: {value}"
    if error.validator == "additionalProperties":
        return f"Unexpected field{location}: {error.message}"
    return f"Invalid TED-MAD input{location}: {error.message}"


def _schema_validate(payload: Mapping[str, Any], schema_name: str) -> None:
    schema = load_schema(schema_name)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.absolute_path))
    if errors:
        raise TedMadValidationError(_format_jsonschema_error(errors[0]))


def _hypothesis_ids(hypotheses_input: Mapping[str, Any]) -> set[str]:
    hypotheses = hypotheses_input.get("hypotheses", [])
    return {str(row["hypothesis_id"]) for row in hypotheses}


def _check_hypothesis_ids(ids: Sequence[str], known_ids: set[str]) -> None:
    for hyp_id in ids:
        if hyp_id not in known_ids:
            raise TedMadValidationError(f"Invalid hypothesis id: {hyp_id}")


def validate_hypotheses(hypotheses_input: Mapping[str, Any]) -> set[str]:
    """Validate hypothesis YAML and return the declared hypothesis IDs."""

    _schema_validate(hypotheses_input, "hypotheses.schema.json")
    ids = _hypothesis_ids(hypotheses_input)
    if len(ids) != len(hypotheses_input.get("hypotheses", [])):
        raise TedMadValidationError("Duplicate hypothesis id in hypotheses")
    return ids


def validate_evidence(evidence_input: Mapping[str, Any], hypothesis_ids: set[str]) -> None:
    """Validate evidence YAML and cross-check referenced hypotheses."""

    _schema_validate(evidence_input, "evidence.schema.json")
    for row in evidence_input.get("evidence", []):
        _check_hypothesis_ids(list(row.get("supports", {})), hypothesis_ids)
        _check_hypothesis_ids(list(row.get("weakens", {})), hypothesis_ids)


def validate_experiments(experiments_input: Mapping[str, Any], hypothesis_ids: set[str]) -> None:
    """Validate experiment YAML and cross-check referenced hypotheses."""

    _schema_validate(experiments_input, "experiments.schema.json")
    for row in experiments_input.get("experiments", []):
        _check_hypothesis_ids(row.get("distinguishes", []), hypothesis_ids)
        _check_hypothesis_ids(list(row.get("expected_patterns", {})), hypothesis_ids)
        if "supports_hypotheses" in row:
            _check_hypothesis_ids(row.get("supports_hypotheses", []), hypothesis_ids)
        falsifiers = row.get("falsifiers", [])
        for falsifier in falsifiers:
            if isinstance(falsifier, Mapping) and "hypothesis" in falsifier:
                _check_hypothesis_ids([str(falsifier["hypothesis"])], hypothesis_ids)


def validate_claim_levels(claim_levels_input: Mapping[str, Any]) -> None:
    """Validate a custom claim-level definition file."""

    _schema_validate(claim_levels_input, "claim_levels.schema.json")


def validate_observed_rescue(observed_input: Mapping[str, Any]) -> None:
    """Validate observed rescue-result YAML."""

    _schema_validate(observed_input, "observed_rescue.schema.json")
