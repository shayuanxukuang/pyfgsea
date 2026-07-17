"""Outcome-agnostic donor-subset robustness for the locked T21 design.

This module deliberately reports robustness diagnostics rather than inference.
It enumerates every three-T21 subset from a fixed 15-donor case cohort and
compares each subset with the same three disomy controls through a user-supplied
effect estimator.  Failed fits remain in the fixed 455-subset denominator.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
import json
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


N_T21_DONORS = 15
N_DISOMY_CONTROLS = 3
T21_SUBSET_SIZE = 3
N_PLANNED_SUBSETS = math.comb(N_T21_DONORS, T21_SUBSET_SIZE)

_CALLBACK_REQUIRED_COLUMNS = frozenset({"candidate", "effect"})
_CALLBACK_ALLOWED_COLUMNS = frozenset(
    {"candidate", "effect", "status", "detail"}
)
_SUCCESS_STATUSES = frozenset({"success", "pass", "passed", "ok"})
_SUMMARY_QUANTILES = (0.05, 0.25, 0.75, 0.95)


@dataclass(frozen=True)
class T21DonorSubsetRobustnessResult:
    """Auditable subset effects and fixed-denominator candidate summaries."""

    subset_effects: pd.DataFrame
    candidate_summary: pd.DataFrame
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "subset_effects": self.subset_effects.copy(),
            "candidate_summary": self.candidate_summary.copy(),
            "metadata": dict(self.metadata),
        }


def _canonical_ids(
    values: Sequence[str],
    *,
    label: str,
    expected_count: Optional[int] = None,
) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a sequence of identifiers, not a string")
    identifiers = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f"Every {label} identifier must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{label} identifiers must be non-empty")
        identifiers.append(stripped)
    if expected_count is not None and len(identifiers) != expected_count:
        raise ValueError(
            f"{label} must contain exactly {expected_count} identifiers; "
            f"found {len(identifiers)}"
        )
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{label} identifiers must be unique")
    return tuple(sorted(identifiers))


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _donor_plan(
    t21_donors: Tuple[str, ...],
    disomy_controls: Tuple[str, ...],
) -> tuple[list[tuple[str, Tuple[str, ...]]], str]:
    subsets = list(combinations(t21_donors, T21_SUBSET_SIZE))
    if len(subsets) != N_PLANNED_SUBSETS or len(set(subsets)) != N_PLANNED_SUBSETS:
        raise RuntimeError("Internal T21 donor-subset plan is not the complete 455 orbit")
    plan = [
        (f"subset_{index:03d}", subset)
        for index, subset in enumerate(subsets, start=1)
    ]
    payload = {
        "schema_name": "t21_three_donor_subset_plan",
        "schema_version": "1.0.0",
        "t21_donors": list(t21_donors),
        "disomy_controls": list(disomy_controls),
        "t21_subset_size": T21_SUBSET_SIZE,
        "n_planned_subsets": N_PLANNED_SUBSETS,
        "subsets": [list(subset) for _, subset in plan],
    }
    digest = sha256(_stable_json(payload).encode("utf-8")).hexdigest()
    return plan, digest


def _normalize_callback_output(
    value: Any,
    *,
    candidate_ids: Tuple[str, ...],
    context: str,
) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        frame = pd.DataFrame(list(value))
    else:
        raise TypeError(
            f"{context} must return a DataFrame or a sequence of row mappings"
        )
    if frame.columns.duplicated().any():
        raise ValueError(f"{context} returned duplicate column names")
    columns = set(map(str, frame.columns))
    missing = sorted(_CALLBACK_REQUIRED_COLUMNS.difference(columns))
    unexpected = sorted(columns.difference(_CALLBACK_ALLOWED_COLUMNS))
    if missing or unexpected:
        raise ValueError(
            f"{context} callback columns differ: missing={missing}, "
            f"unsupported={unexpected}; no inferential-value columns are accepted"
        )
    frame = frame.loc[:, [
        column
        for column in ("candidate", "effect", "status", "detail")
        if column in frame.columns
    ]].copy()
    if len(frame) != len(candidate_ids):
        raise ValueError(
            f"{context} must return exactly one row for each candidate; "
            f"expected {len(candidate_ids)}, found {len(frame)}"
        )
    if frame["candidate"].isna().any():
        raise ValueError(f"{context} candidate identifiers must be non-missing")
    raw_candidates = frame["candidate"].tolist()
    if any(not isinstance(value, str) for value in raw_candidates):
        raise TypeError(f"{context} candidate identifiers must be strings")
    if any(value != value.strip() or not value for value in raw_candidates):
        raise ValueError(
            f"{context} candidate identifiers must be non-empty without outer whitespace"
        )
    if len(raw_candidates) != len(set(raw_candidates)):
        raise ValueError(f"{context} returned duplicate candidate rows")
    observed = set(raw_candidates)
    expected = set(candidate_ids)
    if observed != expected:
        raise ValueError(
            f"{context} candidate set changed: "
            f"missing={sorted(expected-observed)}, unexpected={sorted(observed-expected)}"
        )
    if frame["effect"].map(lambda item: isinstance(item, (bool, np.bool_))).any():
        raise TypeError(f"{context} effects must be numeric, not boolean")
    effects = pd.to_numeric(frame["effect"], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(effects)):
        raise ValueError(f"{context} effects must all be finite")
    frame["effect"] = effects

    if "status" not in frame:
        frame["callback_status"] = "success"
    else:
        if frame["status"].isna().any():
            raise ValueError(f"{context} callback status must be non-missing")
        raw_status = frame["status"].astype(str).str.strip().str.lower()
        if raw_status.eq("").any():
            raise ValueError(f"{context} callback status must be non-empty")
        frame["callback_status"] = raw_status
    frame["status"] = np.where(
        frame["callback_status"].isin(_SUCCESS_STATUSES),
        "success",
        "failed",
    )
    if "detail" not in frame:
        frame["detail"] = ""
    else:
        frame["detail"] = frame["detail"].fillna("").astype(str)

    order = {candidate: index for index, candidate in enumerate(candidate_ids)}
    frame["_candidate_order"] = frame["candidate"].map(order)
    return (
        frame.sort_values("_candidate_order")
        .drop(columns=["_candidate_order"])
        .reset_index(drop=True)
    )


def _reference_effect_map(
    *,
    candidate_ids: Tuple[str, ...],
    t21_donors: Tuple[str, ...],
    disomy_controls: Tuple[str, ...],
    reference_effects: Optional[Mapping[str, float]],
    reference_estimator: Optional[Callable[[Tuple[str, ...], Tuple[str, ...]], Any]],
) -> tuple[dict[str, float], str]:
    if (reference_effects is None) == (reference_estimator is None):
        raise ValueError(
            "Supply exactly one of reference_effects or reference_estimator"
        )
    if reference_effects is not None:
        if not isinstance(reference_effects, Mapping):
            raise TypeError("reference_effects must be a candidate-to-effect mapping")
        normalized_keys = _canonical_ids(
            list(reference_effects.keys()), label="reference_effects"
        )
        if set(normalized_keys) != set(candidate_ids):
            raise ValueError("reference_effects must exactly match candidate_ids")
        result = {}
        for candidate in candidate_ids:
            value = reference_effects[candidate]
            if isinstance(value, (bool, np.bool_)):
                raise TypeError("Reference effects must be numeric, not boolean")
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError("Reference effects must all be finite")
            result[candidate] = numeric
        return result, "explicit_full_cohort_effects"

    assert reference_estimator is not None
    try:
        raw = reference_estimator(t21_donors, disomy_controls)
    except Exception as exc:
        raise RuntimeError(
            "The independent full-cohort reference estimator failed"
        ) from exc
    frame = _normalize_callback_output(
        raw,
        candidate_ids=candidate_ids,
        context="full-cohort reference",
    )
    if not frame["status"].eq("success").all():
        raise ValueError("Every full-cohort reference candidate must succeed")
    return (
        dict(zip(frame["candidate"].astype(str), frame["effect"].astype(float))),
        "independent_full_cohort_callback",
    )


def _effect_direction(effect: float, tolerance: float) -> str:
    if effect > tolerance:
        return "positive"
    if effect < -tolerance:
        return "negative"
    return "zero"


def _failed_rows(
    *,
    candidate_ids: Tuple[str, ...],
    detail: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate": candidate_ids,
            "effect": np.nan,
            "callback_status": "exception",
            "status": "failed",
            "detail": detail,
        }
    )


def _subset_rows(
    *,
    subset_id: str,
    subset: Tuple[str, ...],
    controls: Tuple[str, ...],
    callback_rows: pd.DataFrame,
    reference_effects: Mapping[str, float],
    zero_tolerance: float,
) -> list[dict[str, Any]]:
    rows = []
    for item in callback_rows.itertuples(index=False):
        reference_effect = float(reference_effects[item.candidate])
        reference_direction = _effect_direction(reference_effect, zero_tolerance)
        successful = item.status == "success"
        direction = (
            _effect_direction(float(item.effect), zero_tolerance)
            if successful
            else "not_available"
        )
        rows.append(
            {
                "subset_id": subset_id,
                "candidate": str(item.candidate),
                "t21_donors": _stable_json(list(subset)),
                "t21_donor_1": subset[0],
                "t21_donor_2": subset[1],
                "t21_donor_3": subset[2],
                "disomy_controls": _stable_json(list(controls)),
                "disomy_control_1": controls[0],
                "disomy_control_2": controls[1],
                "disomy_control_3": controls[2],
                "effect": float(item.effect),
                "callback_status": str(item.callback_status),
                "status": str(item.status),
                "detail": str(item.detail),
                "effect_direction": direction,
                "reference_effect": reference_effect,
                "reference_direction": reference_direction,
                "direction_supported": bool(
                    successful and direction == reference_direction
                ),
                "planned_denominator": N_PLANNED_SUBSETS,
                "analysis_role": "robustness_not_inference",
                "no_new_p_value": True,
            }
        )
    return rows


def _extreme_donor_audit(
    successful: pd.DataFrame,
    *,
    t21_donors: Tuple[str, ...],
    extreme_fraction: float,
) -> dict[str, Any]:
    if successful.empty:
        return {
            "n_extreme_subsets": 0,
            "extreme_donor_concentration_hhi": np.nan,
            "extreme_donor_max_appearance_count": 0,
            "extreme_donor_max_appearance_fraction": np.nan,
            "extreme_donor_ids": "[]",
            "extreme_donor_appearance_fraction": "{}",
            "max_abs_effect": np.nan,
            "max_abs_effect_subset_id": "",
            "max_abs_effect_subset_t21_donors": "[]",
            "max_abs_effect_donor_mean_extreme_appearance_fraction": np.nan,
            "max_abs_effect_donor_max_extreme_appearance_fraction": np.nan,
        }
    ordered = successful.assign(_abs_effect=successful["effect"].abs()).sort_values(
        ["_abs_effect", "subset_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    n_extreme = max(1, int(math.ceil(len(ordered) * extreme_fraction)))
    extreme = ordered.head(n_extreme)
    appearances = Counter()
    for item in extreme.itertuples(index=False):
        appearances.update((item.t21_donor_1, item.t21_donor_2, item.t21_donor_3))
    counts = {donor: int(appearances.get(donor, 0)) for donor in t21_donors}
    fractions = {donor: count / n_extreme for donor, count in counts.items()}
    total_appearances = T21_SUBSET_SIZE * n_extreme
    hhi = float(sum((count / total_appearances) ** 2 for count in counts.values()))
    max_count = max(counts.values())
    max_donors = [donor for donor, count in counts.items() if count == max_count]
    max_row = ordered.iloc[0]
    max_subset_donors = [
        str(max_row["t21_donor_1"]),
        str(max_row["t21_donor_2"]),
        str(max_row["t21_donor_3"]),
    ]
    max_subset_fractions = [fractions[donor] for donor in max_subset_donors]
    return {
        "n_extreme_subsets": n_extreme,
        "extreme_donor_concentration_hhi": hhi,
        "extreme_donor_max_appearance_count": max_count,
        "extreme_donor_max_appearance_fraction": max_count / n_extreme,
        "extreme_donor_ids": _stable_json(max_donors),
        "extreme_donor_appearance_fraction": _stable_json(fractions),
        "max_abs_effect": float(max_row["_abs_effect"]),
        "max_abs_effect_subset_id": str(max_row["subset_id"]),
        "max_abs_effect_subset_t21_donors": _stable_json(max_subset_donors),
        "max_abs_effect_donor_mean_extreme_appearance_fraction": float(
            np.mean(max_subset_fractions)
        ),
        "max_abs_effect_donor_max_extreme_appearance_fraction": float(
            np.max(max_subset_fractions)
        ),
    }


def _candidate_summary(
    subset_effects: pd.DataFrame,
    *,
    candidate_ids: Tuple[str, ...],
    t21_donors: Tuple[str, ...],
    extreme_fraction: float,
) -> pd.DataFrame:
    rows = []
    for candidate in candidate_ids:
        planned = subset_effects.loc[subset_effects["candidate"].eq(candidate)].copy()
        if len(planned) != N_PLANNED_SUBSETS:
            raise RuntimeError("Internal candidate subset plan lost rows")
        successful = planned.loc[planned["status"].eq("success")].copy()
        failed = planned.loc[planned["status"].eq("failed")]
        values = successful["effect"].to_numpy(dtype=float)
        quantiles = (
            np.quantile(values, _SUMMARY_QUANTILES)
            if values.size
            else np.full(len(_SUMMARY_QUANTILES), np.nan)
        )
        reference_effect = float(planned["reference_effect"].iloc[0])
        reference_direction = str(planned["reference_direction"].iloc[0])
        row = {
            "candidate": candidate,
            "reference_effect": reference_effect,
            "reference_direction": reference_direction,
            "reference_direction_nonzero": reference_direction != "zero",
            "n_planned": N_PLANNED_SUBSETS,
            "n_success": int(len(successful)),
            "n_failed": int(len(failed)),
            "n_positive_direction": int(
                successful["effect_direction"].eq("positive").sum()
            ),
            "n_negative_direction": int(
                successful["effect_direction"].eq("negative").sum()
            ),
            "n_zero_direction": int(
                successful["effect_direction"].eq("zero").sum()
            ),
            "n_direction_supported": int(planned["direction_supported"].sum()),
            "direction_support_fraction": float(
                planned["direction_supported"].sum() / N_PLANNED_SUBSETS
            ),
            "failed_fraction": float(len(failed) / N_PLANNED_SUBSETS),
            "effect_median": float(np.median(values)) if values.size else np.nan,
            "effect_q05": float(quantiles[0]),
            "effect_q25": float(quantiles[1]),
            "effect_q75": float(quantiles[2]),
            "effect_q95": float(quantiles[3]),
            "effect_min": float(np.min(values)) if values.size else np.nan,
            "effect_max": float(np.max(values)) if values.size else np.nan,
            "robustness_status": (
                "complete_fixed_plan"
                if failed.empty
                else "incomplete_fixed_plan_failures_retained"
            ),
            "analysis_role": "robustness_not_inference",
            "no_new_p_value": True,
            **_extreme_donor_audit(
                successful,
                t21_donors=t21_donors,
                extreme_fraction=extreme_fraction,
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def run_t21_donor_subset_robustness(
    t21_donors: Sequence[str],
    disomy_controls: Sequence[str],
    candidate_ids: Sequence[str],
    estimator: Callable[[Tuple[str, ...], Tuple[str, ...]], Any],
    *,
    reference_effects: Optional[Mapping[str, float]] = None,
    reference_estimator: Optional[
        Callable[[Tuple[str, ...], Tuple[str, ...]], Any]
    ] = None,
    zero_tolerance: float = 1e-12,
    extreme_fraction: float = 0.05,
) -> T21DonorSubsetRobustnessResult:
    """Run all 455 three-T21 versus three-disomy robustness fits.

    ``estimator`` is called positionally with ``(t21_subset, disomy_controls)``
    and must return exactly one row per candidate with columns ``candidate`` and
    ``effect``. Optional ``status`` and ``detail`` columns are retained. No
    inferential-value columns are accepted. Exceptions become fixed-plan failed
    rows; structurally invalid callback output raises immediately.

    Direction support always uses the fixed denominator of 455, so failed fits
    count as unsupported. A zero full-cohort reference treats only successful
    zero-direction subset effects as direction-supported. This is a robustness
    diagnostic and does not create or aggregate a new p value.
    """
    canonical_t21 = _canonical_ids(
        t21_donors,
        label="t21_donors",
        expected_count=N_T21_DONORS,
    )
    canonical_controls = _canonical_ids(
        disomy_controls,
        label="disomy_controls",
        expected_count=N_DISOMY_CONTROLS,
    )
    overlap = sorted(set(canonical_t21).intersection(canonical_controls))
    if overlap:
        raise ValueError(f"T21 and disomy donor IDs must be disjoint: {overlap}")
    canonical_candidates = _canonical_ids(candidate_ids, label="candidate_ids")
    if not canonical_candidates:
        raise ValueError("At least one candidate is required")
    if not callable(estimator):
        raise TypeError("estimator must be callable")
    zero_tolerance = float(zero_tolerance)
    if not np.isfinite(zero_tolerance) or zero_tolerance < 0:
        raise ValueError("zero_tolerance must be finite and non-negative")
    extreme_fraction = float(extreme_fraction)
    if not np.isfinite(extreme_fraction) or not 0 < extreme_fraction <= 1:
        raise ValueError("extreme_fraction must be finite and within (0, 1]")

    plan, donor_plan_sha256 = _donor_plan(canonical_t21, canonical_controls)
    reference_map, reference_source = _reference_effect_map(
        candidate_ids=canonical_candidates,
        t21_donors=canonical_t21,
        disomy_controls=canonical_controls,
        reference_effects=reference_effects,
        reference_estimator=reference_estimator,
    )

    output_rows: list[dict[str, Any]] = []
    for subset_id, subset in plan:
        try:
            raw = estimator(subset, canonical_controls)
        except Exception as exc:
            callback_rows = _failed_rows(
                candidate_ids=canonical_candidates,
                detail=f"{type(exc).__name__}:{exc}",
            )
        else:
            callback_rows = _normalize_callback_output(
                raw,
                candidate_ids=canonical_candidates,
                context=subset_id,
            )
        output_rows.extend(
            _subset_rows(
                subset_id=subset_id,
                subset=subset,
                controls=canonical_controls,
                callback_rows=callback_rows,
                reference_effects=reference_map,
                zero_tolerance=zero_tolerance,
            )
        )

    subset_effects = pd.DataFrame(output_rows)
    expected_rows = N_PLANNED_SUBSETS * len(canonical_candidates)
    if len(subset_effects) != expected_rows:
        raise RuntimeError("Internal subset execution did not preserve the fixed plan")
    candidate_summary = _candidate_summary(
        subset_effects,
        candidate_ids=canonical_candidates,
        t21_donors=canonical_t21,
        extreme_fraction=extreme_fraction,
    )
    metadata = {
        "schema_name": "t21_donor_subset_robustness",
        "schema_version": "1.0.0",
        "method": "all_455_three_T21_subsets_vs_three_disomy_controls",
        "analysis_role": "robustness_not_inference",
        "no_new_p_value": True,
        "failed_subsets_remain_in_planned_denominator": True,
        "direction_support_denominator": "all_455_planned_subsets",
        "zero_reference_policy": "successful_zero_effect_matches_zero_reference",
        "n_t21_donors": N_T21_DONORS,
        "n_disomy_controls": N_DISOMY_CONTROLS,
        "t21_subset_size": T21_SUBSET_SIZE,
        "n_planned_subsets": N_PLANNED_SUBSETS,
        "planned_denominator": N_PLANNED_SUBSETS,
        "n_candidates": len(canonical_candidates),
        "t21_donors": list(canonical_t21),
        "disomy_controls": list(canonical_controls),
        "candidate_ids": list(canonical_candidates),
        "donor_plan_sha256": donor_plan_sha256,
        "reference_source": reference_source,
        "zero_tolerance": zero_tolerance,
        "extreme_fraction": extreme_fraction,
        "summary_quantiles": list(_SUMMARY_QUANTILES),
        "callback_output_columns": sorted(_CALLBACK_ALLOWED_COLUMNS),
    }
    for table in (subset_effects, candidate_summary):
        table.attrs["t21_donor_subset_robustness"] = metadata.copy()
    return T21DonorSubsetRobustnessResult(
        subset_effects=subset_effects,
        candidate_summary=candidate_summary,
        metadata=MappingProxyType(metadata),
    )


__all__ = [
    "N_DISOMY_CONTROLS",
    "N_PLANNED_SUBSETS",
    "N_T21_DONORS",
    "T21DonorSubsetRobustnessResult",
    "T21_SUBSET_SIZE",
    "run_t21_donor_subset_robustness",
]
