from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator


TableKind = Literal["activity", "event"]
# ``auto`` is accepted by the public validator/CLI; resolved versions are v1/v2.
SchemaVersion = Literal["auto", "v1", "v2"]

# Retain the original mapping for callers that imported it when only v1 existed.
SCHEMA_FILES: dict[TableKind, str] = {
    "activity": "ted_activity_table_v1.schema.json",
    "event": "ted_event_report_v1.schema.json",
}
SCHEMA_FILES_BY_VERSION: dict[TableKind, dict[str, str]] = {
    "activity": {"v1": "ted_activity_table_v1.schema.json"},
    "event": {
        "v1": "ted_event_report_v1.schema.json",
        "v2": "ted_event_report_v2.schema.json",
    },
}
EVENT_V2_FIELDS = frozenset(
    {
        "event_test_status",
        "event_q_missing_reason",
        "e0_reason_code",
        "event_support_code",
        "validation_provenance_code",
        "evidence_boundary",
        "supported_interpretation",
        "unsupported_interpretation_current_evidence",
    }
)
EVENT_Q_MISSING_REASONS = frozenset(
    {
        "undeclared_family",
        "no_defensible_null",
        "insufficient_blocks",
        "complete_confounding",
        "insufficient_permutation_resolution",
        "other",
    }
)
E0_REASON_CODES = frozenset(
    {
        "E0_not_supported",
        "E0_not_estimable",
        "E0_not_identifiable",
        "E0_artifact_dominated",
        "E0_missing_required_design",
    }
)


def read_ted_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path)
    return pd.read_csv(path, sep="\t")


def detect_ted_schema_version(frame: pd.DataFrame, kind: TableKind) -> Literal["v1", "v2"]:
    """Detect the built-in schema from v2 marker columns without changing v1 behavior.

    The presence of any v2-only field selects v2. A partially migrated table
    therefore fails against v2 instead of silently falling back to v1.
    """
    if kind not in SCHEMA_FILES_BY_VERSION:
        raise ValueError(f"Unsupported TED table kind: {kind}")
    if kind == "event" and EVENT_V2_FIELDS.intersection(frame.columns):
        return "v2"
    return "v1"


def _resolve_schema_version(
    frame: pd.DataFrame,
    kind: TableKind,
    schema_version: SchemaVersion,
) -> Literal["v1", "v2"]:
    if schema_version == "auto":
        return detect_ted_schema_version(frame, kind)
    if schema_version not in {"v1", "v2"}:
        raise ValueError(f"Unsupported TED schema version: {schema_version}")
    if schema_version not in SCHEMA_FILES_BY_VERSION[kind]:
        raise ValueError(f"TED {kind} tables do not have a built-in {schema_version} schema")
    return schema_version


def _schema_path(kind: TableKind, schema_version: str = "v1") -> Path:
    try:
        schema_file = SCHEMA_FILES_BY_VERSION[kind][schema_version]
    except KeyError as exc:
        raise ValueError(
            f"No built-in TED schema for kind={kind!r}, version={schema_version!r}"
        ) from exc
    packaged = Path(__file__).resolve().parent / "schemas" / schema_file
    if packaged.exists():
        return packaged
    # Source-tree fallback for older editable installs.
    return Path(__file__).resolve().parents[1] / "schemas" / schema_file


def _record_for_json(row: pd.Series) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in row.items():
        if pd.isna(value):
            out[str(key)] = None
        elif isinstance(value, np.generic):
            out[str(key)] = value.item()
        else:
            out[str(key)] = value
    return out


def _issue(
    rows: list[dict[str, object]],
    level: str,
    check: str,
    message: str,
    *,
    row_index: int | None = None,
) -> None:
    rows.append(
        {
            "level": level,
            "check": check,
            "row_index": row_index,
            "message": message,
        }
    )


def validate_ted_table(
    table: pd.DataFrame | str | Path,
    kind: TableKind,
    *,
    schema_path: str | Path | None = None,
    schema_version: SchemaVersion = "auto",
) -> pd.DataFrame:
    """Validate a TED activity or event table against schema and semantic gates.

    Event v1 remains the fallback for legacy tables. Event v2 is selected when
    any E/V marker column is present, or can be requested explicitly with
    ``schema_version="v2"``. A custom ``schema_path`` changes the JSON Schema
    document but retains semantic checks for the resolved built-in version.
    """
    if kind not in SCHEMA_FILES_BY_VERSION:
        raise ValueError(f"Unsupported TED table kind: {kind}")
    frame = read_ted_table(table) if isinstance(table, (str, Path)) else table.copy()
    resolved_version = _resolve_schema_version(frame, kind, schema_version)
    schema_file = Path(schema_path) if schema_path else _schema_path(kind, resolved_version)
    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    issues: list[dict[str, object]] = []

    if frame.empty:
        _issue(issues, "error", "nonempty", "Table has no rows")
        return pd.DataFrame(issues)

    for idx, row in frame.iterrows():
        record = _record_for_json(row)
        for error in sorted(validator.iter_errors(record), key=lambda item: list(item.path)):
            location = ".".join(map(str, error.path)) or "record"
            _issue(
                issues,
                "error",
                "json_schema",
                f"{location}: {error.message}",
                row_index=int(idx) if isinstance(idx, (int, np.integer)) else None,
            )

    if kind == "activity":
        numeric = frame[[col for col in ("time", "activity") if col in frame]].apply(
            pd.to_numeric, errors="coerce"
        )
        if not numeric.empty and not np.isfinite(numeric.to_numpy(dtype=float)).all():
            _issue(issues, "error", "finite_numeric", "time and activity must be finite")
        if {"dataset_id", "pathway", "time"}.issubset(frame.columns):
            support = frame.groupby(["dataset_id", "pathway"], dropna=False)["time"].nunique()
            if (support < 2).any():
                _issue(
                    issues,
                    "error",
                    "time_support",
                    f"{int((support < 2).sum())} dataset/pathway groups have fewer than two time points",
                )
        if "block_id" in frame and frame["block_id"].astype("string").str.strip().isin(["", "<NA>"]).any():
            _issue(issues, "error", "block_id", "block_id contains missing or blank values")
    else:
        if "event_q" in frame:
            q = pd.to_numeric(frame["event_q"], errors="coerce")
            out_of_range = q.notna() & ((q < 0) | (q > 1))
            if resolved_version == "v2" and "event_test_status" in frame:
                status = frame["event_test_status"].astype("string")
                invalid = out_of_range | (status.eq("not_run") & q.notna()) | (
                    status.isin(["run_not_supported", "run_supported"]) & q.isna()
                )
                message = (
                    "event_q must be null exactly when event_test_status=not_run "
                    "and finite within [0, 1] after a valid test run"
                )
            else:
                invalid = out_of_range | q.isna()
                message = "event_q must be finite and within [0, 1]"
            if invalid.any():
                _issue(issues, "error", "event_q_range", message)
        if resolved_version == "v1" and {
            "claim_ceiling",
            "matched_functional_rescue",
        }.issubset(frame.columns):
            high = frame["claim_ceiling"].astype(str).str.match(r"Level\s+[45]")
            rescue = frame["matched_functional_rescue"].astype(str).str.lower().isin(
                {"true", "1", "yes"}
            )
            if (high & ~rescue).any():
                _issue(
                    issues,
                    "error",
                    "claim_ceiling_gate",
                    "Level 4/5 rows require matched_functional_rescue=true",
                )
        if resolved_version == "v2":
            if {"event_test_status", "event_q_missing_reason"}.issubset(frame.columns):
                status = frame["event_test_status"].astype("string")
                raw_missing_reason = frame["event_q_missing_reason"]
                missing_reason = raw_missing_reason.astype("string")
                reason_absent = raw_missing_reason.isna() | missing_reason.str.strip().fillna("").eq("")
                reason_valid = missing_reason.fillna("").isin(EVENT_Q_MISSING_REASONS)
                invalid_not_run = status.eq("not_run") & (reason_absent | ~reason_valid)
                invalid_run = status.isin(["run_not_supported", "run_supported"]) & ~reason_absent
                for idx in frame.index[invalid_not_run]:
                    _issue(
                        issues,
                        "error",
                        "event_q_missing_reason",
                        "not_run rows require a stable event_q_missing_reason",
                        row_index=int(idx) if isinstance(idx, (int, np.integer)) else None,
                    )
                for idx in frame.index[invalid_run]:
                    _issue(
                        issues,
                        "error",
                        "event_q_missing_reason",
                        "rows with a valid test run must leave event_q_missing_reason null",
                        row_index=int(idx) if isinstance(idx, (int, np.integer)) else None,
                    )

            if {"event_support_code", "e0_reason_code"}.issubset(frame.columns):
                raw_reason = frame["e0_reason_code"]
                reason = raw_reason.astype("string")
                e0 = frame["event_support_code"].eq("E0")
                missing_reason = raw_reason.isna() | reason.str.strip().fillna("").eq("")
                invalid_reason = ~reason.fillna("").isin(E0_REASON_CODES)
                for idx in frame.index[e0 & (missing_reason | invalid_reason)]:
                    _issue(
                        issues,
                        "error",
                        "e0_reason_code",
                        "E0 rows require one of the five stable e0_reason_code values",
                        row_index=int(idx) if isinstance(idx, (int, np.integer)) else None,
                    )
                for idx in frame.index[~e0 & ~missing_reason]:
                    _issue(
                        issues,
                        "error",
                        "e0_reason_code",
                        "E1/E2 rows must leave e0_reason_code null",
                        row_index=int(idx) if isinstance(idx, (int, np.integer)) else None,
                    )

            if {"resampling_selection_frequency", "discovery_stability_status"}.issubset(frame.columns):
                frequency = pd.to_numeric(frame["resampling_selection_frequency"], errors="coerce")
                stability = frame["discovery_stability_status"].astype("string")
                assessed = frequency.notna()
                expected = pd.Series("unstable", index=frame.index, dtype="string")
                expected.loc[frequency >= 0.50] = "intermediate"
                expected.loc[frequency >= 0.80] = "stable_core"
                invalid = assessed & stability.ne(expected)
                invalid |= ~assessed & ~stability.isin(["<NA>", "not_assessed"])
                for idx in frame.index[invalid]:
                    _issue(
                        issues,
                        "error",
                        "discovery_stability_consistency",
                        "discovery_stability_status must match resampling_selection_frequency thresholds",
                        row_index=int(idx) if isinstance(idx, (int, np.integer)) else None,
                    )

            if {"upstream_disagreement_flag", "event_support_code"}.issubset(frame.columns):
                disagreement = frame["upstream_disagreement_flag"].astype("boolean").fillna(False)
                invalid = disagreement & frame["event_support_code"].eq("E2")
                for idx in frame.index[invalid]:
                    _issue(
                        issues,
                        "error",
                        "upstream_disagreement_gate",
                        "upstream disagreement forbids E2; return E1 or an ambiguity set",
                        row_index=int(idx) if isinstance(idx, (int, np.integer)) else None,
                    )

            text_fields = [
                column
                for column in (
                    "evidence_boundary",
                    "supported_interpretation",
                    "unsupported_interpretation_current_evidence",
                )
                if column in frame
            ]
            for column in text_fields:
                blank = frame[column].astype("string").str.strip().isin(["", "<NA>"])
                if blank.any():
                    _issue(
                        issues,
                        "error",
                        "nonblank_interpretation",
                        f"{column} contains missing or blank values",
                    )

            required_for_boundary = {
                "event_support_code",
                "validation_provenance_code",
                "evidence_boundary",
            }
            if required_for_boundary.issubset(frame.columns):
                expected = (
                    frame["event_support_code"].astype(str)
                    + "-"
                    + frame["validation_provenance_code"].astype(str)
                )
                observed = frame["evidence_boundary"].astype(str).str.replace(
                    "\u2013", "-", regex=False
                )
                mismatch = observed.ne(expected)
                for idx in frame.index[mismatch]:
                    _issue(
                        issues,
                        "error",
                        "evidence_boundary_consistency",
                        f"evidence_boundary must equal {expected.loc[idx]}",
                        row_index=int(idx) if isinstance(idx, (int, np.integer)) else None,
                    )

            if {"event_support_code", "identifiability_status"}.issubset(frame.columns):
                invalid = frame["identifiability_status"].eq("not_identifiable") & ~frame[
                    "event_support_code"
                ].eq("E0")
                for idx in frame.index[invalid]:
                    _issue(
                        issues,
                        "error",
                        "event_support_identifiability",
                        "not_identifiable rows must use event_support_code=E0",
                        row_index=int(idx) if isinstance(idx, (int, np.integer)) else None,
                    )

            if "validation_provenance_code" in frame:
                v3 = frame["validation_provenance_code"].eq("V3")
                if "matched_functional_rescue" not in frame:
                    rescue = pd.Series(False, index=frame.index)
                else:
                    rescue = frame["matched_functional_rescue"].astype(str).str.lower().isin(
                        {"true", "1", "yes"}
                    )
                for idx in frame.index[v3 & ~rescue]:
                    _issue(
                        issues,
                        "error",
                        "validation_provenance_gate",
                        "V3 requires matched_functional_rescue=true",
                        row_index=int(idx) if isinstance(idx, (int, np.integer)) else None,
                    )

    if not any(row["level"] == "error" for row in issues):
        _issue(
            issues,
            "ok",
            "table",
            f"{kind} table passed {resolved_version} schema and semantic validation",
        )
    return pd.DataFrame(issues, columns=["level", "check", "row_index", "message"])


def ted_table_is_valid(report: pd.DataFrame) -> bool:
    return report.empty or not report["level"].eq("error").any()
