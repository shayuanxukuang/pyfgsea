from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
import hashlib
import json
import math
import warnings
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import brentq
from scipy.stats import nct, t

from .trajectory_pseudobulk import (
    _assign_fixed_bins,
    _bh_adjust,
    _by_adjust,
    _curve_statistics,
    _fixed_edges,
    _h5ad_safe_value,
    _normalize_statistic,
    _normalize_tail,
    _prepare_pathways,
    _require_integer,
    _score_pathways,
    _test_scale,
    _validate_stringification_is_injective,
)
from .validation import _expression_matrix


_RESERVED_DONOR_OUTPUT_COLUMNS = {
    "donor",
    "donor_index_all",
    "observed_condition",
    "observed_case",
    "restriction_stratum",
    "n_cells_selected",
    "donor_index",
    "availability_signature",
    "permutation_block",
    "permutation_block_size",
    "mobile_residual_curve",
    "n_selected_bins_available",
    "included_in_inference",
    "exclusion_reason",
    "bin_id",
    "bin_left",
    "bin_right",
    "bin_mid",
    "bin_width",
    "n_cells",
    "available",
    "_index",
    "__stratum_key",
    "__donor_index_all",
    "__original_index",
    "__donor_index",
}


@dataclass
class CovariateAdjustedDonorPseudobulkResult:
    """Auditable outputs from donor-curve covariate-adjusted inference."""

    pathway_tests: pd.DataFrame
    effect_curves: pd.DataFrame
    pseudobulk_adata: Any
    donor_bin_activity: pd.DataFrame
    grid_diagnostics: pd.DataFrame
    segment_diagnostics: pd.DataFrame
    donor_design: pd.DataFrame
    covariate_diagnostics: pd.DataFrame
    design_diagnostics: pd.DataFrame
    power_diagnostics: pd.DataFrame
    pathway_membership: pd.DataFrame
    permutation_summary: pd.DataFrame
    permutation_assignments: pd.DataFrame
    null_statistics: pd.DataFrame
    metadata: Dict[str, Any]

    def to_tables(self) -> Dict[str, pd.DataFrame]:
        """Return independent copies of all tabular outputs."""
        return {
            "pathway_tests": self.pathway_tests.copy(),
            "effect_curves": self.effect_curves.copy(),
            "donor_bin_activity": self.donor_bin_activity.copy(),
            "grid_diagnostics": self.grid_diagnostics.copy(),
            "segment_diagnostics": self.segment_diagnostics.copy(),
            "donor_design": self.donor_design.copy(),
            "covariate_diagnostics": self.covariate_diagnostics.copy(),
            "design_diagnostics": self.design_diagnostics.copy(),
            "power_diagnostics": self.power_diagnostics.copy(),
            "pathway_membership": self.pathway_membership.copy(),
            "permutation_summary": self.permutation_summary.copy(),
            "permutation_assignments": self.permutation_assignments.copy(),
            "null_statistics": self.null_statistics.copy(),
        }


def _formal_inference_donor_bin_view(
    frame: pd.DataFrame, *, label: str
) -> pd.DataFrame:
    """Return included-donor x selected-segment rows from a full-grid artifact.

    New results retain the complete source grid.  Formal consumers must keep
    unavailable rows inside the selected segment (as NaN) while excluding
    source-only bins and donors.  Legacy selected-only tables without the new
    flags remain readable; partially flagged tables fail closed.
    """
    new_flags = {
        "included_in_inference",
        "bin_in_selected_segment",
        "analysis_selected",
    }
    present = new_flags.intersection(frame.columns)
    if not present:
        return frame.copy()
    required = {"available", *new_flags}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"{label} has an incomplete formal-view flag contract: "
            f"missing={sorted(required-set(frame.columns))}"
        )
    flags: dict[str, np.ndarray] = {}
    for column in required:
        values = frame[column]
        if values.isna().any() or not pd.api.types.is_bool_dtype(values.dtype):
            raise ValueError(f"{label}.{column} must be non-missing boolean")
        flags[column] = values.to_numpy(dtype=bool)
    expected_analysis = (
        flags["included_in_inference"]
        & flags["bin_in_selected_segment"]
        & flags["available"]
    )
    if not np.array_equal(flags["analysis_selected"], expected_analysis):
        raise ValueError(f"{label}.analysis_selected violates the formal contract")
    formal_rows = (
        flags["included_in_inference"] & flags["bin_in_selected_segment"]
    )
    return frame.loc[formal_rows].copy()


def _formal_inference_donor_design_view(
    donor_design: pd.DataFrame,
) -> pd.DataFrame:
    """Return the exact ordered donor axis used by formal inference."""
    frame = donor_design.copy()
    if "included_in_inference" in frame:
        values = frame["included_in_inference"]
        if values.isna().any() or not pd.api.types.is_bool_dtype(values.dtype):
            raise ValueError(
                "donor_design.included_in_inference must be non-missing boolean"
            )
        frame = frame.loc[values.to_numpy(dtype=bool)].copy()
    if frame.empty:
        raise ValueError("No donors are included in formal inference")
    if "donor_index" in frame:
        index = pd.to_numeric(frame["donor_index"], errors="coerce").to_numpy(
            dtype=float
        )
        if (
            not np.isfinite(index).all()
            or not np.array_equal(index, np.floor(index))
            or set(index.astype(int)) != set(range(len(frame)))
        ):
            raise ValueError("Included donor_index values must be contiguous")
        frame = frame.assign(donor_index=index.astype(int)).sort_values(
            "donor_index", kind="mergesort"
        )
    else:
        frame = frame.sort_values("donor", kind="mergesort")
    return frame.reset_index(drop=True)


class CovariateDesignError(ValueError):
    """Raised when condition and nuisance effects cannot be separated safely."""

    def __init__(self, message: str, diagnostics: Optional[pd.DataFrame] = None):
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass
class _ResidualPermutationPlan:
    null_mappings: list[np.ndarray]
    residual_space_size: int
    requested_mode: str
    actual_mode: str
    is_exhaustive: bool
    draw_attempts: int
    duplicate_draws: int


@dataclass
class _EncodedDesign:
    reduced: np.ndarray
    terms: list[str]
    encoding: list[dict[str, Any]]


def _as_key_tuple(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("design key collections must be sequences, not strings")
    normalized = tuple(str(value) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError("design key collections must not contain duplicates")
    return normalized


def _validate_public_design_keys(
    condition_key: str,
    donor_key: str,
    pseudotime_key: str,
    continuous_covariate_keys: tuple[str, ...],
    categorical_covariate_keys: tuple[str, ...],
    strata_keys: tuple[str, ...],
) -> None:
    keys = [
        condition_key,
        donor_key,
        pseudotime_key,
        *continuous_covariate_keys,
        *categorical_covariate_keys,
        *strata_keys,
    ]
    if len(set(keys)) != len(keys):
        raise ValueError(
            "condition, donor, pseudotime, continuous covariate, categorical "
            "covariate, and strata keys must be distinct. Restriction strata are "
            "already included as fixed nuisance effects."
        )
    nuisance_keys = {
        *continuous_covariate_keys,
        *categorical_covariate_keys,
        *strata_keys,
    }
    reserved = sorted(nuisance_keys & _RESERVED_DONOR_OUTPUT_COLUMNS)
    if reserved:
        raise ValueError(
            "Covariate and strata keys collide with reserved result columns: "
            f"{reserved}. Rename these input columns before analysis."
        )
    primary_conflicts = []
    if condition_key in _RESERVED_DONOR_OUTPUT_COLUMNS:
        primary_conflicts.append(f"condition_key={condition_key!r}")
    if pseudotime_key in _RESERVED_DONOR_OUTPUT_COLUMNS:
        primary_conflicts.append(f"pseudotime_key={pseudotime_key!r}")
    if (
        donor_key in _RESERVED_DONOR_OUTPUT_COLUMNS
        and donor_key != "donor"
    ):
        primary_conflicts.append(f"donor_key={donor_key!r}")
    if primary_conflicts:
        raise ValueError(
            "Primary design keys collide with reserved internal/result columns: "
            + ", ".join(primary_conflicts)
            + ". Rename these input columns before analysis."
        )


def _build_donor_frame(
    adata,
    *,
    condition_key: str,
    donor_key: str,
    control,
    case,
    pseudotime_key: str,
    continuous_covariate_keys: tuple[str, ...],
    categorical_covariate_keys: tuple[str, ...],
    strata_keys: tuple[str, ...],
    donor_order: str = "lexicographic",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if str(control) == str(case):
        raise ValueError("control and case must be different")
    required = [
        condition_key,
        donor_key,
        pseudotime_key,
        *continuous_covariate_keys,
        *categorical_covariate_keys,
        *strata_keys,
    ]
    missing = [key for key in required if key not in adata.obs]
    if missing:
        raise KeyError(f"Missing adata.obs columns: {missing}")
    obs = adata.obs.loc[:, required].copy()
    if obs[condition_key].isna().any():
        raise ValueError(f"condition_key '{condition_key}' contains missing values")
    for key in (condition_key, donor_key, *categorical_covariate_keys, *strata_keys):
        _validate_stringification_is_injective(obs[key], key)
    selected_mask = obs[condition_key].astype(str).isin([str(control), str(case)])
    if not selected_mask.any():
        raise ValueError("No cells match the requested control/case conditions")
    selected = obs.loc[selected_mask].copy()
    if selected.isna().any().any():
        bad = selected.columns[selected.isna().any()].tolist()
        raise ValueError(f"Selected cells contain missing design values in {bad}")

    pt = pd.to_numeric(selected[pseudotime_key], errors="coerce")
    if not np.isfinite(pt.to_numpy(dtype=float)).all():
        raise ValueError(f"pseudotime_key '{pseudotime_key}' must be finite numeric")
    selected[pseudotime_key] = pt.astype(float)
    for key in continuous_covariate_keys:
        numeric = pd.to_numeric(selected[key], errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"Continuous covariate '{key}' must be finite numeric")
        selected[key] = numeric.astype(float)
    selected[condition_key] = selected[condition_key].astype(str)
    selected[donor_key] = selected[donor_key].astype(str)
    for key in (*categorical_covariate_keys, *strata_keys):
        selected[key] = selected[key].astype(str)

    donor_rows: list[dict[str, Any]] = []
    donor_constant_keys = [
        condition_key,
        *continuous_covariate_keys,
        *categorical_covariate_keys,
        *strata_keys,
    ]
    for donor, group in selected.groupby(donor_key, sort=True):
        donor_row: dict[str, Any] = {"donor": str(donor)}
        for key in donor_constant_keys:
            values = pd.unique(group[key])
            if len(values) != 1:
                raise ValueError(
                    f"Donor-level design key '{key}' is not constant for donor "
                    f"'{donor}'"
                )
            donor_row[key] = values[0]
        stratum_values = tuple(str(donor_row[key]) for key in strata_keys)
        donor_row["__stratum_key"] = stratum_values if strata_keys else ("__all__",)
        donor_row["restriction_stratum"] = (
            json.dumps(stratum_values, ensure_ascii=False, separators=(",", ":"))
            if strata_keys
            else "__all__"
        )
        donor_row["n_cells_selected"] = int(len(group))
        donor_rows.append(donor_row)

    donors = pd.DataFrame(donor_rows)
    if donor_order == "lexicographic":
        donors = donors.sort_values("donor").reset_index(drop=True)
    elif donor_order == "sha256":
        donors = (
            donors.assign(
                __donor_order_digest=donors["donor"].map(
                    lambda value: hashlib.sha256(str(value).encode("utf-8")).hexdigest()
                )
            )
            .sort_values("__donor_order_digest")
            .drop(columns="__donor_order_digest")
            .reset_index(drop=True)
        )
    else:
        raise ValueError("donor_order must be 'lexicographic' or 'sha256'")
    donors["donor_index_all"] = np.arange(len(donors), dtype=int)
    donors["observed_condition"] = donors[condition_key].astype(str)
    donors["observed_case"] = donors["observed_condition"].eq(str(case))
    donor_to_index = donors.set_index("donor")["donor_index_all"].to_dict()
    selected["__donor_index_all"] = selected[donor_key].map(donor_to_index).astype(int)
    selected["__original_index"] = np.flatnonzero(selected_mask.to_numpy())
    return donors, selected


def _count_donor_bins(
    cell_frame: pd.DataFrame,
    pseudotime_key: str,
    edges: np.ndarray,
    n_donors: int,
) -> np.ndarray:
    bin_ids = _assign_fixed_bins(
        cell_frame[pseudotime_key].to_numpy(dtype=float), edges
    )
    counts = np.zeros((n_donors, len(edges) - 1), dtype=int)
    valid = bin_ids >= 0
    np.add.at(
        counts,
        (
            cell_frame.loc[valid, "__donor_index_all"].to_numpy(dtype=int),
            bin_ids[valid],
        ),
        1,
    )
    return counts


def _encode_reduced_design(
    donor_frame: pd.DataFrame,
    *,
    continuous_covariate_keys: tuple[str, ...],
    categorical_covariate_keys: tuple[str, ...],
    strata_keys: tuple[str, ...],
) -> _EncodedDesign:
    n = len(donor_frame)
    if n == 0:
        raise ValueError("Reduced design requires at least one donor")
    columns = [np.ones(n, dtype=float)]
    terms = ["Intercept"]
    encoding: list[dict[str, Any]] = [
        {"term": "Intercept", "source": "intercept", "status": "included"}
    ]

    for key in continuous_covariate_keys:
        values = donor_frame[key].to_numpy(dtype=float)
        mean = float(np.mean(values))
        scale = float(np.std(values, ddof=0))
        floor = 1e-12 * max(1.0, abs(mean))
        if not np.isfinite(scale) or scale <= floor:
            encoding.append(
                {
                    "term": key,
                    "source": key,
                    "kind": "continuous",
                    "status": "omitted_zero_variance",
                    "center": mean,
                    "scale": scale,
                }
            )
            continue
        columns.append((values - mean) / scale)
        terms.append(key)
        encoding.append(
            {
                "term": key,
                "source": key,
                "kind": "continuous",
                "status": "included_centered_scaled",
                "center": mean,
                "scale": scale,
            }
        )

    for key in categorical_covariate_keys:
        values = donor_frame[key].astype(str).to_numpy()
        levels = sorted(pd.unique(values).tolist())
        reference = levels[0]
        if len(levels) == 1:
            encoding.append(
                {
                    "term": key,
                    "source": key,
                    "kind": "categorical",
                    "status": "omitted_single_level",
                    "reference": reference,
                    "levels": levels,
                }
            )
        for level in levels[1:]:
            term = f"{key}[{level}]"
            columns.append((values == level).astype(float))
            terms.append(term)
            encoding.append(
                {
                    "term": term,
                    "source": key,
                    "kind": "categorical",
                    "status": "included_dummy",
                    "reference": reference,
                    "level": level,
                    "levels": levels,
                }
            )

    if strata_keys:
        values = donor_frame["__stratum_key"].tolist()
        levels = sorted(set(values))
        reference = levels[0]
        for level in levels[1:]:
            label = json.dumps(level, ensure_ascii=False, separators=(",", ":"))
            term = f"stratum[{label}]"
            columns.append(np.asarray([value == level for value in values], dtype=float))
            terms.append(term)
            encoding.append(
                {
                    "term": term,
                    "source": "|".join(strata_keys),
                    "kind": "restriction_fixed_effect",
                    "status": "included_dummy",
                    "reference": json.dumps(
                        reference, ensure_ascii=False, separators=(",", ":")
                    ),
                    "level": label,
                }
            )
    return _EncodedDesign(
        reduced=np.column_stack(columns).astype(float),
        terms=terms,
        encoding=encoding,
    )


def _make_pattern_blocks(
    donor_frame: pd.DataFrame,
    available: np.ndarray,
    selected_bins: np.ndarray,
) -> tuple[list[np.ndarray], list[str], list[str]]:
    signatures = [
        "".join("1" if value else "0" for value in row)
        for row in available[:, selected_bins]
    ]
    block_lookup: dict[tuple[Any, str], list[int]] = {}
    for index, (stratum, signature) in enumerate(
        zip(donor_frame["__stratum_key"], signatures)
    ):
        if "1" not in signature:
            continue
        block_lookup.setdefault((stratum, signature), []).append(index)
    ordered = sorted(block_lookup.items(), key=lambda item: repr(item[0]))
    groups = [np.asarray(indices, dtype=int) for _, indices in ordered]
    block_labels = [
        json.dumps(
            [*list(key[0]), key[1]], ensure_ascii=False, separators=(",", ":")
        )
        for key, _ in ordered
    ]
    donor_block = ["" for _ in range(len(donor_frame))]
    for label, indices in zip(block_labels, groups):
        for index in indices:
            donor_block[int(index)] = label
    return groups, signatures, donor_block


def _space_sizes(
    observed_case: np.ndarray,
    groups: list[np.ndarray],
) -> tuple[int, int]:
    residual_space = 1
    label_space = 1
    for indices in groups:
        residual_space *= math.factorial(len(indices))
        label_space *= math.comb(
            len(indices), int(observed_case[indices].sum())
        )
    return int(residual_space), int(label_space)


def _safe_inverse_space(size: int) -> float:
    """Return 1/size without overflowing while converting a huge Python int."""
    if size < 1:
        raise ValueError("permutation space size must be positive")
    log_value = math.log(size)
    if log_value > -math.log(np.nextafter(0.0, 1.0)):
        return 0.0
    return float(math.exp(-log_value))


def _matrix_rank(values: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(values))


def _bin_design_row(
    reduced: np.ndarray,
    condition: np.ndarray,
    available: np.ndarray,
    groups: list[np.ndarray],
    *,
    max_condition_vif: float,
    min_residual_df: int,
    min_donors_per_condition: int,
) -> dict[str, Any]:
    indices = np.flatnonzero(available)
    z = reduced[indices]
    c = condition[indices].astype(float)
    full = np.column_stack([z, c])
    rank_reduced = _matrix_rank(z)
    rank_full = _matrix_rank(full)
    residual_df = int(len(indices) - rank_full)
    pinv_z = np.linalg.pinv(z)
    u = c - z @ (pinv_z @ c)
    condition_information = float(u @ u)
    centered = c - float(np.mean(c)) if len(c) else c
    unadjusted_information = float(centered @ centered)
    information_fraction = (
        condition_information / unadjusted_information
        if unadjusted_information > 0
        else 0.0
    )
    condition_vif = (
        unadjusted_information / condition_information
        if condition_information > 1e-14 * max(1.0, unadjusted_information)
        else math.inf
    )
    leverage = np.diag(full @ np.linalg.pinv(full)) if len(indices) else np.array([])
    singular = np.linalg.svd(full, compute_uv=False) if full.size else np.array([])
    nonzero = singular[singular > np.finfo(float).eps * max(full.shape) * singular[0]] if len(singular) and singular[0] > 0 else np.array([])
    condition_number = (
        float(nonzero[0] / nonzero[-1]) if len(nonzero) else math.inf
    )

    global_to_local = {int(value): local for local, value in enumerate(indices)}
    permutation_information = 0.0
    condition_label_permutation_information = 0.0
    for group in groups:
        local = [global_to_local[int(value)] for value in group if int(value) in global_to_local]
        if len(local) <= 1:
            continue
        values = u[np.asarray(local, dtype=int)]
        permutation_information += float(np.sum((values - np.mean(values)) ** 2))
        labels = c[np.asarray(local, dtype=int)]
        condition_label_permutation_information += float(
            np.sum((labels - np.mean(labels)) ** 2)
        )

    n_case = int(c.sum())
    n_control = int(len(c) - n_case)
    reasons = []
    if n_control < min_donors_per_condition or n_case < min_donors_per_condition:
        reasons.append("insufficient_donors_per_condition")
    condition_estimable = bool(
        rank_full == rank_reduced + 1
        and condition_information > 1e-14 * max(1.0, unadjusted_information)
    )
    if not condition_estimable:
        reasons.append("condition_not_identifiable_from_nuisance")
    if residual_df < min_residual_df:
        reasons.append("insufficient_residual_df")
    if not np.isfinite(condition_vif) or condition_vif > max_condition_vif:
        reasons.append("condition_vif_exceeds_threshold")
    if permutation_information <= 1e-14 * max(1.0, condition_information):
        reasons.append("degenerate_residual_permutation_information")
    if condition_label_permutation_information <= 1e-14:
        reasons.append("no_within_block_condition_label_information")
    return {
        "n_donors_available": int(len(indices)),
        "n_control_available": n_control,
        "n_case_available": n_case,
        "reduced_model_rank": rank_reduced,
        "full_model_rank": rank_full,
        "reduced_model_columns": int(z.shape[1]),
        "full_model_columns": int(full.shape[1]),
        "residual_df_reduced": int(len(indices) - rank_reduced),
        "residual_df_full": residual_df,
        "condition_information": condition_information,
        "unadjusted_condition_information": unadjusted_information,
        "condition_information_fraction": information_fraction,
        "condition_r2_from_nuisance": float(1.0 - information_fraction),
        "condition_vif": float(condition_vif),
        "permutation_information": float(permutation_information),
        "condition_label_permutation_information": float(
            condition_label_permutation_information
        ),
        "condition_estimable": condition_estimable,
        "full_design_condition_number": condition_number,
        "max_leverage": float(np.max(leverage)) if len(leverage) else math.nan,
        "design_gate_pass": not reasons,
        "rejection_reason": "|".join(reasons),
    }


def _select_segment(
    donor_frame: pd.DataFrame,
    counts: np.ndarray,
    edges: np.ndarray,
    *,
    min_cells_per_donor_bin: int,
    min_donors_per_condition: int,
    min_common_bins: int,
    min_residual_df: int,
    max_condition_vif: float,
    continuous_covariate_keys: tuple[str, ...],
    categorical_covariate_keys: tuple[str, ...],
    strata_keys: tuple[str, ...],
) -> tuple[
    np.ndarray,
    np.ndarray,
    _EncodedDesign,
    list[np.ndarray],
    list[str],
    list[str],
    pd.DataFrame,
    pd.DataFrame,
]:
    available_all = counts >= min_cells_per_donor_bin
    n_bins = available_all.shape[1]
    candidates: list[dict[str, Any]] = []
    payloads: dict[tuple[int, int], tuple[Any, ...]] = {}
    observed_case_all = donor_frame["observed_case"].to_numpy(dtype=bool)

    for start in range(n_bins):
        for stop in range(start + min_common_bins, n_bins + 1):
            selected_bins = np.arange(start, stop, dtype=int)
            has_any = available_all[:, selected_bins].any(axis=1)
            candidate_donors = donor_frame.loc[has_any].copy().reset_index(drop=True)
            candidate_available = available_all[has_any]
            candidate_case = observed_case_all[has_any]
            row: dict[str, Any] = {
                "segment_start_bin": int(start),
                "segment_stop_bin_exclusive": int(stop),
                "n_bins": int(stop - start),
                "n_donors_with_any_coverage": int(has_any.sum()),
                "n_donor_bin_observations": int(
                    candidate_available[:, selected_bins].sum()
                ),
            }
            try:
                encoded = _encode_reduced_design(
                    candidate_donors,
                    continuous_covariate_keys=continuous_covariate_keys,
                    categorical_covariate_keys=categorical_covariate_keys,
                    strata_keys=strata_keys,
                )
                groups, signatures, donor_blocks = _make_pattern_blocks(
                    candidate_donors, candidate_available, selected_bins
                )
                residual_space, label_space = _space_sizes(candidate_case, groups)
                bin_rows = [
                    _bin_design_row(
                        encoded.reduced,
                        candidate_case,
                        candidate_available[:, bin_id],
                        groups,
                        max_condition_vif=max_condition_vif,
                        min_residual_df=min_residual_df,
                        min_donors_per_condition=min_donors_per_condition,
                    )
                    for bin_id in selected_bins
                ]
                valid = bool(
                    residual_space > 1
                    and label_space > 1
                    and bin_rows
                    and all(item["design_gate_pass"] for item in bin_rows)
                )
                reasons = sorted(
                    {
                        reason
                        for item in bin_rows
                        for reason in str(item["rejection_reason"]).split("|")
                        if reason
                    }
                )
                if residual_space <= 1:
                    reasons.append("residual_permutation_space_size_one")
                if label_space <= 1:
                    reasons.append("condition_label_space_size_one")
                row.update(
                    {
                        "n_permutation_blocks": int(len(groups)),
                        "n_mobile_blocks": int(sum(len(group) > 1 for group in groups)),
                        "n_immobile_donors": int(
                            sum(len(group) for group in groups if len(group) == 1)
                        ),
                        "residual_permutation_space_size": int(residual_space),
                        "condition_label_space_size": int(label_space),
                        "minimum_residual_df": int(
                            min(item["residual_df_full"] for item in bin_rows)
                        ),
                        "maximum_condition_vif": float(
                            max(item["condition_vif"] for item in bin_rows)
                        ),
                        "minimum_condition_information_fraction": float(
                            min(
                                item["condition_information_fraction"]
                                for item in bin_rows
                            )
                        ),
                        "segment_gate_pass": valid,
                        "rejection_reason": "|".join(sorted(set(reasons))),
                    }
                )
                payloads[(start, stop)] = (
                    has_any,
                    candidate_available,
                    encoded,
                    groups,
                    signatures,
                    donor_blocks,
                    bin_rows,
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                row.update(
                    {
                        "segment_gate_pass": False,
                        "rejection_reason": f"design_encoding_failure:{exc}",
                    }
                )
            candidates.append(row)

    segment_diagnostics = pd.DataFrame(candidates)
    valid = segment_diagnostics[segment_diagnostics["segment_gate_pass"].fillna(False)]
    if valid.empty:
        reasons = (
            segment_diagnostics["rejection_reason"]
            .fillna("")
            .value_counts()
            .head(5)
            .index.tolist()
        )
        raise CovariateDesignError(
            "Unable to reliably distinguish condition from covariates: no "
            "contiguous fixed-grid segment passes the requested donor, "
            "residual-df, VIF, coverage, "
            "and whole-curve permutation gates. Leading reasons: "
            + "; ".join(reasons),
            diagnostics=segment_diagnostics,
        )
    selected_row = valid.sort_values(
        [
            "n_bins",
            "n_donor_bin_observations",
            "n_donors_with_any_coverage",
            "segment_start_bin",
        ],
        ascending=[False, False, False, True],
    ).iloc[0]
    start = int(selected_row["segment_start_bin"])
    stop = int(selected_row["segment_stop_bin_exclusive"])
    selected_bins = np.arange(start, stop, dtype=int)
    segment_diagnostics["selected_segment"] = (
        segment_diagnostics["segment_start_bin"].eq(start)
        & segment_diagnostics["segment_stop_bin_exclusive"].eq(stop)
    )
    (
        has_any,
        candidate_available,
        encoded,
        groups,
        signatures,
        donor_blocks,
        bin_rows,
    ) = payloads[(start, stop)]
    design_diagnostics = pd.DataFrame(bin_rows)
    design_diagnostics.insert(0, "bin_id", selected_bins)
    design_diagnostics.insert(1, "bin_left", edges[selected_bins])
    design_diagnostics.insert(2, "bin_right", edges[selected_bins + 1])
    design_diagnostics.insert(
        3, "bin_mid", (edges[selected_bins] + edges[selected_bins + 1]) / 2.0
    )
    return (
        selected_bins,
        has_any,
        encoded,
        groups,
        signatures,
        donor_blocks,
        segment_diagnostics,
        design_diagnostics,
    )


def _enumerate_residual_mappings(
    n_donors: int, groups: list[np.ndarray]
) -> list[np.ndarray]:
    choices = [list(permutations(group.tolist())) for group in groups]
    mappings = []
    for selected in product(*choices):
        mapping = np.arange(n_donors, dtype=int)
        for group, source_order in zip(groups, selected):
            mapping[group] = np.asarray(source_order, dtype=int)
        mappings.append(mapping)
    return mappings


def _sample_residual_mappings(
    n_donors: int,
    groups: list[np.ndarray],
    target: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], int, int]:
    identity = np.arange(n_donors, dtype=int)
    seen = {identity.tobytes()}
    mappings: list[np.ndarray] = []
    attempts = 0
    duplicates = 0
    max_attempts = max(1000, target * 100)
    while len(mappings) < target and attempts < max_attempts:
        attempts += 1
        mapping = identity.copy()
        for group in groups:
            mapping[group] = rng.permutation(group)
        key = mapping.tobytes()
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        mappings.append(mapping)
    if len(mappings) < target:
        raise RuntimeError(
            "Could not sample the requested number of unique donor-curve "
            "residual permutations; reduce n_permutations or use exhaustive mode"
        )
    return mappings, attempts, duplicates


def _make_residual_plan(
    n_donors: int,
    groups: list[np.ndarray],
    *,
    permutation_mode: str,
    n_permutations: int,
    max_exact_permutations: int,
    seed: int,
) -> _ResidualPermutationPlan:
    requested = str(permutation_mode).lower().replace("-", "_")
    if requested not in {"auto", "exact", "exhaustive", "monte_carlo"}:
        raise ValueError(
            "permutation_mode must be 'auto', 'exact', 'exhaustive', or "
            "'monte_carlo'"
        )
    residual_space = 1
    for group in groups:
        residual_space *= math.factorial(len(group))
    residual_space = int(residual_space)
    if residual_space <= 1:
        raise CovariateDesignError(
            "Residual donor-curve permutation space has size 1 after strata and "
            "availability-pattern restrictions"
        )
    use_exhaustive = requested in {"exact", "exhaustive"} or (
        requested == "auto" and residual_space <= max_exact_permutations
    )
    exhaustive_mc = (
        requested == "monte_carlo"
        and residual_space <= max_exact_permutations
        and n_permutations >= residual_space - 1
    )
    use_exhaustive = use_exhaustive or exhaustive_mc
    identity = np.arange(n_donors, dtype=int)
    if use_exhaustive:
        if residual_space > max_exact_permutations:
            raise ValueError(
                f"Exhaustive residual space has {residual_space} mappings, "
                f"exceeding max_exact_permutations={max_exact_permutations}"
            )
        null = [
            mapping
            for mapping in _enumerate_residual_mappings(n_donors, groups)
            if not np.array_equal(mapping, identity)
        ]
        return _ResidualPermutationPlan(
            null_mappings=null,
            residual_space_size=residual_space,
            requested_mode=requested,
            actual_mode=("exhaustive_via_requested_mc" if exhaustive_mc else "exhaustive"),
            is_exhaustive=True,
            draw_attempts=0,
            duplicate_draws=0,
        )
    target = min(n_permutations, residual_space - 1)
    mappings, attempts, duplicates = _sample_residual_mappings(
        n_donors, groups, target, np.random.default_rng(seed)
    )
    return _ResidualPermutationPlan(
        null_mappings=mappings,
        residual_space_size=residual_space,
        requested_mode=requested,
        actual_mode=("exhaustive_via_unique_sampling" if target == residual_space - 1 else "monte_carlo"),
        is_exhaustive=target == residual_space - 1,
        draw_attempts=attempts,
        duplicate_draws=duplicates,
    )


def _mean_selected_columns(X, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Read only selected genes when the matrix backend permits it."""
    rows = np.asarray(rows, dtype=int)
    columns = np.asarray(columns, dtype=int)
    if isinstance(X, np.ndarray):
        block = X[np.ix_(rows, columns)]
        return np.asarray(block, dtype=float).mean(axis=0)
    if sparse.issparse(X):
        return np.asarray(X[rows][:, columns].mean(axis=0)).ravel()
    module = type(X).__module__.lower()
    if module.startswith("h5py"):
        total = np.zeros(len(columns), dtype=float)
        for row in rows:
            total += np.asarray(X[int(row), columns], dtype=float).ravel()
        return total / len(rows)
    if module.startswith("anndata._core.sparse_dataset"):
        sorted_rows = np.sort(rows)
        total = np.zeros(len(columns), dtype=float)
        split_points = np.flatnonzero(np.diff(sorted_rows) != 1) + 1
        row_runs = np.split(sorted_rows, split_points)
        matrix_format = str(getattr(X, "format", "")).lower()
        chunk_size = 256
        for run in row_runs:
            for offset in range(0, len(run), chunk_size):
                chunk = run[offset : offset + chunk_size]
                start = int(chunk[0])
                stop = int(chunk[-1]) + 1
                if matrix_format == "csc":
                    block = X[start:stop, columns]
                elif matrix_format == "csr":
                    block = X[start:stop]
                    block = block[:, columns]
                else:
                    raise TypeError(
                        "Unsupported backed sparse format. Use backed CSR/CSC, "
                        "an in-memory scipy sparse matrix, or a dense layer."
                    )
                total += np.asarray(block.sum(axis=0)).ravel()
        return total / len(rows)
    block = X[rows]
    block = block[:, columns]
    if sparse.issparse(block):
        return np.asarray(block.mean(axis=0)).ravel()
    return np.asarray(block, dtype=float).mean(axis=0)


def _aggregate_donor_bins_selected(
    X,
    cell_frame: pd.DataFrame,
    *,
    pseudotime_key: str,
    gene_indices: np.ndarray,
    selected_bins: np.ndarray,
    edges: np.ndarray,
    n_donors: int,
    min_cells_per_donor_bin: int,
) -> tuple[np.ndarray, np.ndarray]:
    cell_bins = _assign_fixed_bins(
        cell_frame[pseudotime_key].to_numpy(dtype=float), edges
    )
    pseudobulk = np.full(
        (n_donors, len(selected_bins), len(gene_indices)), np.nan, dtype=float
    )
    counts = np.zeros((n_donors, len(selected_bins)), dtype=int)
    donor_index = cell_frame["__donor_index"].to_numpy(dtype=int)
    original_index = cell_frame["__original_index"].to_numpy(dtype=int)
    for donor in range(n_donors):
        donor_mask = donor_index == donor
        for local_bin, bin_id in enumerate(selected_bins):
            mask = donor_mask & (cell_bins == int(bin_id))
            rows = original_index[mask]
            counts[donor, local_bin] = len(rows)
            if len(rows) < min_cells_per_donor_bin:
                continue
            mean = _mean_selected_columns(X, rows, gene_indices)
            if not np.isfinite(mean).all():
                raise ValueError("Non-finite donor-bin pseudobulk expression encountered")
            pseudobulk[donor, local_bin] = mean
    return pseudobulk, counts


@dataclass
class _BinFit:
    available_indices: np.ndarray
    reduced: np.ndarray
    full: np.ndarray
    pinv_full: np.ndarray
    fitted_reduced: np.ndarray
    residual_reduced: np.ndarray
    beta: np.ndarray
    t_value: np.ndarray
    standard_error: np.ndarray
    residual_sd: np.ndarray
    adjusted_control: np.ndarray
    adjusted_case: np.ndarray
    residual_df: int


def _studentized(beta: np.ndarray, standard_error: np.ndarray) -> np.ndarray:
    beta = np.asarray(beta, dtype=float)
    standard_error = np.asarray(standard_error, dtype=float)
    tolerance = np.finfo(float).eps * np.maximum(1.0, np.abs(beta))
    out = np.zeros_like(beta)
    regular = standard_error > tolerance
    out[regular] = beta[regular] / standard_error[regular]
    deterministic = (~regular) & (np.abs(beta) > tolerance)
    out[deterministic] = np.sign(beta[deterministic]) * np.inf
    return out


def _fit_observed_models(
    scores: np.ndarray,
    reduced: np.ndarray,
    condition: np.ndarray,
) -> tuple[list[_BinFit], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_bins = scores.shape[1]
    n_pathways = scores.shape[2]
    beta_curve = np.full((n_bins, n_pathways), np.nan, dtype=float)
    t_curve = np.full_like(beta_curve, np.nan)
    se_curve = np.full_like(beta_curve, np.nan)
    residual_sd_curve = np.full_like(beta_curve, np.nan)
    control_curve = np.full_like(beta_curve, np.nan)
    case_curve = np.full_like(beta_curve, np.nan)
    fits: list[_BinFit] = []
    for bin_index in range(n_bins):
        available = np.isfinite(scores[:, bin_index, 0])
        indices = np.flatnonzero(available)
        y = scores[indices, bin_index]
        z = reduced[indices]
        c = condition[indices].astype(float)
        full = np.column_stack([z, c])
        pinv_z = np.linalg.pinv(z)
        pinv_full = np.linalg.pinv(full)
        fitted_reduced = z @ (pinv_z @ y)
        residual_reduced = y - fitted_reduced
        coefficients = pinv_full @ y
        beta = coefficients[-1]
        fitted_full = full @ coefficients
        residual_full = y - fitted_full
        residual_df = int(len(indices) - _matrix_rank(full))
        sigma2 = np.sum(residual_full**2, axis=0) / residual_df
        c_residual = c - z @ (pinv_z @ c)
        information = float(c_residual @ c_residual)
        standard_error = np.sqrt(np.maximum(sigma2, 0.0) / information)
        t_value = _studentized(beta, standard_error)
        baseline = np.mean(z @ coefficients[:-1], axis=0)
        beta_curve[bin_index] = beta
        t_curve[bin_index] = t_value
        se_curve[bin_index] = standard_error
        residual_sd_curve[bin_index] = np.sqrt(np.maximum(sigma2, 0.0))
        control_curve[bin_index] = baseline
        case_curve[bin_index] = baseline + beta
        fits.append(
            _BinFit(
                available_indices=indices,
                reduced=z,
                full=full,
                pinv_full=pinv_full,
                fitted_reduced=fitted_reduced,
                residual_reduced=residual_reduced,
                beta=beta,
                t_value=t_value,
                standard_error=standard_error,
                residual_sd=np.sqrt(np.maximum(sigma2, 0.0)),
                adjusted_control=baseline,
                adjusted_case=baseline + beta,
                residual_df=residual_df,
            )
        )
    return (
        fits,
        beta_curve,
        t_curve,
        se_curve,
        residual_sd_curve,
        control_curve,
        case_curve,
    )


def _fit_permuted_curves(
    fits: list[_BinFit], mapping: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    beta_rows = []
    t_rows = []
    for fit in fits:
        sources = mapping[fit.available_indices]
        source_lookup = {
            int(global_index): local
            for local, global_index in enumerate(fit.available_indices)
        }
        try:
            source_local = np.asarray(
                [source_lookup[int(source)] for source in sources], dtype=int
            )
        except KeyError as exc:
            raise RuntimeError(
                "Residual permutation moved a donor curve across availability "
                "patterns; this violates the fixed missingness contract"
            ) from exc
        y_star = fit.fitted_reduced + fit.residual_reduced[source_local]
        coefficients = fit.pinv_full @ y_star
        beta = coefficients[-1]
        residual = y_star - fit.full @ coefficients
        sigma2 = np.sum(residual**2, axis=0) / fit.residual_df
        z = fit.reduced
        c = fit.full[:, -1]
        u = c - z @ (np.linalg.pinv(z) @ c)
        information = float(u @ u)
        standard_error = np.sqrt(np.maximum(sigma2, 0.0) / information)
        beta_rows.append(beta)
        t_rows.append(_studentized(beta, standard_error))
    return np.vstack(beta_rows), np.vstack(t_rows)


def _mapping_hash(mapping: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(mapping, dtype=np.int64).tobytes()).hexdigest()[:16]


def _block_constant_reduced_design(
    reduced: np.ndarray,
    available: np.ndarray,
    groups: list[np.ndarray],
    selected_bins: np.ndarray,
) -> bool:
    for bin_id in selected_bins:
        for group in groups:
            present = group[available[group, bin_id]]
            if len(present) <= 1:
                continue
            rows = reduced[present]
            if not np.allclose(rows, rows[0], rtol=1e-10, atol=1e-12):
                return False
    return True


def _cramers_v(table: np.ndarray) -> float:
    table = np.asarray(table, dtype=float)
    total = float(table.sum())
    if total <= 0 or min(table.shape) <= 1:
        return 0.0
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / total
    valid = expected > 0
    chi2 = float(np.sum(((table - expected) ** 2)[valid] / expected[valid]))
    return float(np.sqrt((chi2 / total) / min(table.shape[0] - 1, table.shape[1] - 1)))


def _covariate_diagnostics(
    donor_frame: pd.DataFrame,
    *,
    control,
    case,
    continuous_covariate_keys: tuple[str, ...],
    categorical_covariate_keys: tuple[str, ...],
    strata_keys: tuple[str, ...],
    encoded: _EncodedDesign,
) -> pd.DataFrame:
    condition = donor_frame["observed_case"].to_numpy(dtype=bool)
    rows: list[dict[str, Any]] = []
    for key in continuous_covariate_keys:
        values = donor_frame[key].to_numpy(dtype=float)
        control_values = values[~condition]
        case_values = values[condition]
        pooled_numerator = (
            max(len(control_values) - 1, 0) * np.var(control_values, ddof=1)
            + max(len(case_values) - 1, 0) * np.var(case_values, ddof=1)
        )
        pooled_denominator = max(len(values) - 2, 1)
        pooled_sd = float(np.sqrt(max(pooled_numerator / pooled_denominator, 0.0)))
        mean_difference = float(np.mean(case_values) - np.mean(control_values))
        correlation = (
            float(np.corrcoef(values, condition.astype(float))[0, 1])
            if np.std(values) > 0
            else 0.0
        )
        rows.append(
            {
                "diagnostic_type": "condition_association",
                "covariate": key,
                "covariate_type": "continuous",
                "n_donors": int(len(values)),
                "control": str(control),
                "case": str(case),
                "control_mean": float(np.mean(control_values)),
                "case_mean": float(np.mean(case_values)),
                "mean_difference_case_minus_control": mean_difference,
                "pooled_sd": pooled_sd,
                "standardized_mean_difference": (
                    mean_difference / pooled_sd if pooled_sd > 0 else math.nan
                ),
                "condition_correlation": correlation,
            }
        )
    categorical_diagnostic_keys = [
        (key, "categorical") for key in categorical_covariate_keys
    ] + [(key, "restriction_stratum") for key in strata_keys]
    if len(strata_keys) > 1:
        categorical_diagnostic_keys.append(
            ("restriction_stratum", "joint_restriction_stratum")
        )
    for key, key_type in categorical_diagnostic_keys:
        levels = sorted(donor_frame[key].astype(str).unique().tolist())
        table = pd.crosstab(
            donor_frame[key].astype(str), donor_frame["observed_condition"]
        ).reindex(index=levels, columns=[str(control), str(case)], fill_value=0)
        shared = int(((table > 0).sum(axis=1) == 2).sum())
        rows.append(
            {
                "diagnostic_type": "condition_association",
                "covariate": key,
                "covariate_type": key_type,
                "n_donors": int(len(donor_frame)),
                "control": str(control),
                "case": str(case),
                "n_levels": int(len(levels)),
                "n_levels_shared_between_conditions": shared,
                "n_condition_exclusive_levels": int(len(levels) - shared),
                "cramers_v": _cramers_v(table.to_numpy()),
                "contingency_json": json.dumps(
                    {
                        level: {
                            str(control): int(table.loc[level, str(control)]),
                            str(case): int(table.loc[level, str(case)]),
                        }
                        for level in levels
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )
    for item in encoded.encoding:
        rows.append(
            {
                "diagnostic_type": "design_encoding",
                "covariate": item.get("source"),
                "covariate_type": item.get("kind", item.get("source")),
                "encoded_term": item.get("term"),
                "encoding_status": item.get("status"),
                "encoding_reference": item.get("reference"),
                "encoding_level": item.get("level"),
                "encoding_center": item.get("center"),
                "encoding_scale": item.get("scale"),
            }
        )
    return pd.DataFrame(rows)


def _required_noncentrality(df: int, alpha: float, power: float) -> float:
    critical = float(t.ppf(1.0 - alpha / 2.0, df))

    def achieved(ncp_value: float) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return float(
                nct.sf(critical, df, ncp_value)
                + nct.cdf(-critical, df, ncp_value)
            )

    if achieved(0.0) >= power:
        return 0.0
    upper = 1.0
    while achieved(upper) < power and upper < 1e4:
        upper *= 2.0
    return float(brentq(lambda value: achieved(value) - power, 0.0, upper))


def run_covariate_adjusted_donor_pseudobulk(
    adata,
    gene_sets,
    *,
    condition_key: str,
    donor_key: str,
    control,
    case,
    pseudotime_key: str = "dpt_pseudotime",
    continuous_covariate_keys: Sequence[str] = (),
    categorical_covariate_keys: Sequence[str] = (),
    strata_keys: Sequence[str] = (),
    donor_order: str = "lexicographic",
    grid_edges: Optional[Sequence[float]] = None,
    n_bins: int = 8,
    pseudotime_range: Tuple[float, float] = (0.0, 1.0),
    min_cells_per_donor_bin: int = 5,
    min_donors_per_condition: int = 3,
    min_common_bins: int = 2,
    min_residual_df: int = 3,
    max_condition_vif: float = 10.0,
    statistic: str = "max_absolute_effect",
    tail: str = "greater",
    calibration_scale: str = "studentized",
    permutation_mode: str = "auto",
    n_permutations: int = 999,
    max_exact_permutations: int = 20000,
    min_size: int = 5,
    max_size: int = 500,
    layer: Optional[str] = None,
    use_raw: bool = False,
    alpha: float = 0.05,
    power_target: float = 0.8,
    seed: int = 42,
    return_null_statistics: bool = False,
    return_permutation_assignments: bool = False,
    return_donor_bin_activity: bool = False,
    retain_all_genes: bool = False,
) -> CovariateAdjustedDonorPseudobulkResult:
    """Fit adjusted pathway curves and calibrate them by donor-curve residuals.

    The fixed donor-by-bin pathway activity is modeled separately at every bin
    with donor-level nuisance covariates and one condition coefficient. The
    reduced-model residual vector belonging to a donor is permuted as one curve
    across every selected bin and pathway. Permutations are restricted by both
    declared strata and the complete selected-bin availability signature, so
    neither bins nor missingness masks move independently.

    Exhaustive enumeration is an exact enumeration of the specified
    Freedman--Lane reference. It is a finite-sample exact residual-group test
    only when the reduced design is invariant inside every permutation block;
    otherwise the returned calibration is explicitly labeled approximate.
    Inference is conditional on the supplied pseudotime and fixed grid.
    """
    statistic = _normalize_statistic(statistic)
    tail = _normalize_tail(tail)
    _test_scale(np.asarray([0.0]), statistic, tail)
    calibration_scale = str(calibration_scale).lower().replace("-", "_")
    if calibration_scale not in {"studentized", "effect"}:
        raise ValueError("calibration_scale must be 'studentized' or 'effect'")
    n_bins = _require_integer(n_bins, "n_bins", 2)
    min_cells_per_donor_bin = _require_integer(
        min_cells_per_donor_bin, "min_cells_per_donor_bin"
    )
    min_donors_per_condition = _require_integer(
        min_donors_per_condition, "min_donors_per_condition"
    )
    min_common_bins = _require_integer(min_common_bins, "min_common_bins")
    min_residual_df = _require_integer(min_residual_df, "min_residual_df")
    n_permutations = _require_integer(n_permutations, "n_permutations")
    max_exact_permutations = _require_integer(
        max_exact_permutations, "max_exact_permutations", 2
    )
    min_size = _require_integer(min_size, "min_size")
    max_size = _require_integer(max_size, "max_size")
    seed = _require_integer(seed, "seed", 0)
    if not np.isfinite(max_condition_vif) or max_condition_vif < 1.0:
        raise ValueError("max_condition_vif must be finite and at least 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if not 0 < power_target < 1:
        raise ValueError("power_target must be between 0 and 1")
    if layer is not None and use_raw:
        raise ValueError("layer and use_raw=True are mutually exclusive")

    continuous_covariate_keys = _as_key_tuple(continuous_covariate_keys)
    categorical_covariate_keys = _as_key_tuple(categorical_covariate_keys)
    strata_keys = _as_key_tuple(strata_keys)
    donor_order = str(donor_order).strip().lower()
    if donor_order not in {"lexicographic", "sha256"}:
        raise ValueError("donor_order must be 'lexicographic' or 'sha256'")
    _validate_public_design_keys(
        condition_key,
        donor_key,
        pseudotime_key,
        continuous_covariate_keys,
        categorical_covariate_keys,
        strata_keys,
    )
    edges = _fixed_edges(grid_edges, n_bins, pseudotime_range)
    if min_common_bins > len(edges) - 1:
        raise ValueError(
            f"min_common_bins={min_common_bins} exceeds the fixed grid's "
            f"{len(edges) - 1} bins"
        )
    donor_design_all, cell_frame_all = _build_donor_frame(
        adata,
        condition_key=condition_key,
        donor_key=donor_key,
        control=control,
        case=case,
        pseudotime_key=pseudotime_key,
        continuous_covariate_keys=continuous_covariate_keys,
        categorical_covariate_keys=categorical_covariate_keys,
        strata_keys=strata_keys,
        donor_order=donor_order,
    )
    counts_all = _count_donor_bins(
        cell_frame_all,
        pseudotime_key,
        edges,
        len(donor_design_all),
    )
    (
        selected_bins,
        included_mask,
        encoded,
        groups,
        signatures,
        donor_blocks,
        segment_diagnostics,
        design_diagnostics,
    ) = _select_segment(
        donor_design_all,
        counts_all,
        edges,
        min_cells_per_donor_bin=min_cells_per_donor_bin,
        min_donors_per_condition=min_donors_per_condition,
        min_common_bins=min_common_bins,
        min_residual_df=min_residual_df,
        max_condition_vif=float(max_condition_vif),
        continuous_covariate_keys=continuous_covariate_keys,
        categorical_covariate_keys=categorical_covariate_keys,
        strata_keys=strata_keys,
    )
    donor_design = donor_design_all.loc[included_mask].copy().reset_index(drop=True)
    condition = donor_design["observed_case"].to_numpy(dtype=bool)
    available_all = counts_all[included_mask] >= min_cells_per_donor_bin
    residual_space, label_space = _space_sizes(condition, groups)
    block_constant = _block_constant_reduced_design(
        encoded.reduced, available_all, groups, selected_bins
    )
    requested_permutation_mode = str(permutation_mode).lower().replace("-", "_")
    if requested_permutation_mode == "exact" and not block_constant:
        raise CovariateDesignError(
            "permutation_mode='exact' requires the reduced nuisance design to "
            "be constant within every strata × availability permutation block. "
            "Use permutation_mode='exhaustive' only if you explicitly accept an "
            "exhaustive Freedman-Lane reference that is not finite-sample exact."
        )
    plan = _make_residual_plan(
        len(donor_design),
        groups,
        permutation_mode=permutation_mode,
        n_permutations=n_permutations,
        max_exact_permutations=max_exact_permutations,
        seed=seed,
    )

    donor_design["donor_index"] = np.arange(len(donor_design), dtype=int)
    donor_design["availability_signature"] = signatures
    donor_design["permutation_block"] = donor_blocks
    block_size = {
        donor_blocks[int(index)]: int(len(group))
        for group in groups
        for index in group
    }
    donor_design["permutation_block_size"] = donor_design[
        "permutation_block"
    ].map(block_size).astype(int)
    donor_design["mobile_residual_curve"] = donor_design[
        "permutation_block_size"
    ].gt(1)
    donor_design["n_selected_bins_available"] = available_all[
        :, selected_bins
    ].sum(axis=1)
    donor_design["included_in_inference"] = True
    donor_design["exclusion_reason"] = ""
    excluded = donor_design_all.loc[~included_mask].copy()
    if not excluded.empty:
        excluded["donor_index"] = np.nan
        excluded["availability_signature"] = "0" * len(selected_bins)
        excluded["permutation_block"] = ""
        excluded["permutation_block_size"] = 0
        excluded["mobile_residual_curve"] = False
        excluded["n_selected_bins_available"] = 0
        excluded["included_in_inference"] = False
        excluded["exclusion_reason"] = "no_coverage_in_selected_segment"
        donor_design_output = pd.concat([donor_design, excluded], ignore_index=True)
    else:
        donor_design_output = donor_design.copy()

    global_indices = donor_design["donor_index_all"].to_numpy(dtype=int)
    global_to_local = {
        int(global_index): local for local, global_index in enumerate(global_indices)
    }
    cell_keep = cell_frame_all["__donor_index_all"].isin(global_indices)
    cell_frame = cell_frame_all.loc[cell_keep].copy()
    cell_frame["__donor_index"] = cell_frame["__donor_index_all"].map(
        global_to_local
    ).astype(int)

    X, genes, expression_source = _expression_matrix(
        adata, layer=layer, use_raw=use_raw
    )
    pathway_names, pathway_indices, pathway_weights, pathway_membership = (
        _prepare_pathways(genes, gene_sets, min_size=min_size, max_size=max_size)
    )
    if retain_all_genes:
        pseudobulk_gene_indices = np.arange(len(genes), dtype=int)
    else:
        pseudobulk_gene_indices = np.unique(np.concatenate(pathway_indices))
    source_to_pseudobulk = {
        int(source): local for local, source in enumerate(pseudobulk_gene_indices)
    }
    local_pathway_indices = [
        np.asarray([source_to_pseudobulk[int(index)] for index in indices], dtype=int)
        for indices in pathway_indices
    ]
    pathway_membership["pseudobulk_gene_index"] = pathway_membership[
        "gene_index"
    ].map(source_to_pseudobulk).astype(int)
    # Materialize the complete frozen source grid for every donor.  Formal
    # inference still uses only the outcome-blind selected segment below, but
    # excluded bins and donors must remain auditable rather than disappearing
    # from the donor-by-bin pseudobulk artifact.
    full_bin_ids = np.arange(len(edges) - 1, dtype=int)
    full_cell_frame = cell_frame_all.copy()
    full_cell_frame["__donor_index"] = full_cell_frame[
        "__donor_index_all"
    ].astype(int)
    pseudobulk_full, full_counts = _aggregate_donor_bins_selected(
        X,
        full_cell_frame,
        pseudotime_key=pseudotime_key,
        gene_indices=pseudobulk_gene_indices,
        selected_bins=full_bin_ids,
        edges=edges,
        n_donors=len(donor_design_all),
        min_cells_per_donor_bin=min_cells_per_donor_bin,
    )
    if not np.array_equal(full_counts, counts_all):
        raise RuntimeError("Internal donor-bin aggregation count mismatch")
    pseudobulk = pseudobulk_full[included_mask][:, selected_bins]
    scores = _score_pathways(
        pseudobulk, local_pathway_indices, pathway_weights
    )
    (
        fits,
        beta_curve,
        t_curve,
        se_curve,
        residual_sd_curve,
        control_curve,
        case_curve,
    ) = _fit_observed_models(scores, encoded.reduced, condition)
    widths = np.diff(edges)[selected_bins]
    effect_stats = _curve_statistics(beta_curve, widths)
    calibration_curve = t_curve if calibration_scale == "studentized" else beta_curve
    if calibration_scale == "studentized" and not np.isfinite(calibration_curve).all():
        raise CovariateDesignError(
            "Studentized calibration is undefined because at least one selected "
            "pathway/bin has zero residual variance with a nonzero condition "
            "coefficient. Inspect residual_sd or use calibration_scale='effect' "
            "only as an explicitly labeled sensitivity analysis."
        )
    calibration_stats = _curve_statistics(calibration_curve, widths)
    observed_effect_stat = effect_stats[statistic]
    observed_calibration_raw = calibration_stats[statistic]
    observed_calibration_stat = _test_scale(
        observed_calibration_raw, statistic, tail
    )
    observed_point_scale = (
        np.abs(calibration_curve)
        if statistic != "signed_integral" or tail == "two_sided"
        else calibration_curve
        if tail == "greater"
        else -calibration_curve
    )

    pathway_exceedances = np.zeros(len(pathway_names), dtype=int)
    pathway_max_t_exceedances = np.zeros(len(pathway_names), dtype=int)
    pointwise_exceedances = np.zeros_like(calibration_curve, dtype=int)
    within_pathway_exceedances = np.zeros_like(calibration_curve, dtype=int)
    global_curve_exceedances = np.zeros_like(calibration_curve, dtype=int)
    null_rows: list[dict[str, Any]] = []
    tolerance = 1e-12
    for perm_id, mapping in enumerate(plan.null_mappings):
        null_beta, null_t = _fit_permuted_curves(fits, mapping)
        null_effect_stats = _curve_statistics(null_beta, widths)
        null_curve = null_t if calibration_scale == "studentized" else null_beta
        if calibration_scale == "studentized" and not np.isfinite(null_curve).all():
            raise CovariateDesignError(
                "Studentized Freedman-Lane calibration became undefined under "
                f"residual mapping {perm_id} because a pathway/bin has zero "
                "permuted residual variance with a nonzero coefficient. Use a "
                "larger donor design or calibration_scale='effect' as a labeled "
                "sensitivity analysis."
            )
        null_stats = _curve_statistics(null_curve, widths)
        null_raw = null_stats[statistic]
        null_test = _test_scale(null_raw, statistic, tail)
        pathway_exceedances += null_test >= (
            observed_calibration_stat - tolerance
        )
        max_pathway_stat = float(np.max(null_test))
        pathway_max_t_exceedances += max_pathway_stat >= (
            observed_calibration_stat - tolerance
        )
        null_point_scale = (
            np.abs(null_curve)
            if statistic != "signed_integral" or tail == "two_sided"
            else null_curve
            if tail == "greater"
            else -null_curve
        )
        pointwise_exceedances += null_point_scale >= (
            observed_point_scale - tolerance
        )
        within_max = np.max(null_point_scale, axis=0)
        within_pathway_exceedances += within_max[None, :] >= (
            observed_point_scale - tolerance
        )
        global_max = float(np.max(null_point_scale))
        global_curve_exceedances += global_max >= (
            observed_point_scale - tolerance
        )
        if return_null_statistics:
            mapping_hash = _mapping_hash(mapping)
            for pathway_index, pathway in enumerate(pathway_names):
                null_rows.append(
                    {
                        "perm_id": int(perm_id),
                        "mapping_hash": mapping_hash,
                        "Pathway": pathway,
                        "effect_statistic": float(
                            null_effect_stats[statistic][pathway_index]
                        ),
                        "raw_calibration_statistic": float(
                            null_raw[pathway_index]
                        ),
                        "calibration_statistic": float(
                            null_test[pathway_index]
                        ),
                        "max_pathway_calibration_statistic": max_pathway_stat,
                    }
                )

    n_null = len(plan.null_mappings)
    denominator = n_null + 1
    p_raw = (pathway_exceedances + 1.0) / denominator
    p_max_t = (pathway_max_t_exceedances + 1.0) / denominator
    q_bh = _bh_adjust(p_raw)
    q_by = _by_adjust(p_raw)
    pointwise_p = (pointwise_exceedances + 1.0) / denominator
    within_pathway_p = (within_pathway_exceedances + 1.0) / denominator
    global_curve_p = (global_curve_exceedances + 1.0) / denominator
    permutation_resolution = 1.0 / denominator

    if block_constant and plan.is_exhaustive:
        exactness_status = (
            "finite_sample_exact_residual_group_conditional_on_invariance"
        )
    elif block_constant:
        exactness_status = (
            "monte_carlo_finite_sample_residual_group_conditional_on_invariance"
        )
    elif plan.is_exhaustive:
        exactness_status = (
            "exhaustive_freedman_lane_reference_not_finite_sample_exact"
        )
    else:
        exactness_status = "monte_carlo_freedman_lane_approximation"
    reference_enumeration = "exhaustive" if plan.is_exhaustive else "monte_carlo"
    label_quotient_applies = bool(block_constant and plan.is_exhaustive)
    condition_label_complement_symmetry_applies = bool(
        (statistic != "signed_integral" or tail == "two_sided")
        and all(
            2 * int(condition[group].sum()) == len(group) for group in groups
        )
    )
    condition_label_assignment_resolution = _safe_inverse_space(label_space)
    condition_assignment_minimum_p = min(
        1.0,
        (2.0 if condition_label_complement_symmetry_applies else 1.0)
        * condition_label_assignment_resolution,
    )
    log10_condition_label_space_size = float(math.log10(label_space))
    log10_condition_assignment_minimum_p = float(
        math.log10(2.0 if condition_label_complement_symmetry_applies else 1.0)
        - log10_condition_label_space_size
    )
    null_model = (
        "whole_donor_reduced_model_residual_curve_permutation_within_"
        "strata_and_availability_pattern"
    )

    nominal_ncp = {
        int(df): _required_noncentrality(int(df), alpha, power_target)
        for df in sorted(set(design_diagnostics["residual_df_full"].astype(int)))
    }
    family_alpha = alpha / len(pathway_names)
    family_ncp = {
        int(df): _required_noncentrality(int(df), family_alpha, power_target)
        for df in sorted(set(design_diagnostics["residual_df_full"].astype(int)))
    }
    power_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    for local_bin, bin_id in enumerate(selected_bins):
        df = int(design_diagnostics.iloc[local_bin]["residual_df_full"])
        for pathway_index, pathway in enumerate(pathway_names):
            se = float(se_curve[local_bin, pathway_index])
            nominal_mde = float(nominal_ncp[df] * se)
            family_mde = float(family_ncp[df] * se)
            common = {
                "Pathway": pathway,
                "bin_id": int(bin_id),
                "bin_left": float(edges[bin_id]),
                "bin_right": float(edges[bin_id + 1]),
                "bin_mid": float((edges[bin_id] + edges[bin_id + 1]) / 2.0),
                "bin_width": float(edges[bin_id + 1] - edges[bin_id]),
            }
            effect_rows.append(
                {
                    **common,
                    "adjusted_control_activity": float(
                        control_curve[local_bin, pathway_index]
                    ),
                    "adjusted_case_activity": float(
                        case_curve[local_bin, pathway_index]
                    ),
                    "beta_condition": float(beta_curve[local_bin, pathway_index]),
                    "condition_t": float(t_curve[local_bin, pathway_index]),
                    "condition_standard_error": se,
                    "residual_sd": float(
                        residual_sd_curve[local_bin, pathway_index]
                    ),
                    "residual_df": df,
                    "pointwise_p": float(
                        pointwise_p[local_bin, pathway_index]
                    ),
                    "within_pathway_maxT_p": float(
                        within_pathway_p[local_bin, pathway_index]
                    ),
                    "global_curve_maxT_p": float(
                        global_curve_p[local_bin, pathway_index]
                    ),
                }
            )
            power_rows.append(
                {
                    **common,
                    "residual_df": df,
                    "condition_standard_error": se,
                    "power_target": float(power_target),
                    "alpha_pointwise": float(alpha),
                    "alpha_bonferroni_pathway_family": float(family_alpha),
                    "pointwise_mde_two_sided": nominal_mde,
                    "bonferroni_mde_two_sided": family_mde,
                    "method": "noncentral_t_pointwise_analytic_approximation",
                    "not_curve_permutation_power": True,
                }
            )
    effect_curves = pd.DataFrame(effect_rows)
    power_diagnostics = pd.DataFrame(power_rows)

    pathway_sizes = (
        pathway_membership.groupby("Pathway").size().reindex(pathway_names).to_numpy()
    )
    peak_local = effect_stats["peak_bin_local"].astype(int)
    pathway_rows = []
    for pathway_index, pathway in enumerate(pathway_names):
        peak_bin = int(selected_bins[peak_local[pathway_index]])
        pathway_power = power_diagnostics[
            power_diagnostics["Pathway"].eq(pathway)
        ]
        pathway_rows.append(
            {
                "Pathway": pathway,
                "pathway_size": int(pathway_sizes[pathway_index]),
                "primary_statistic": statistic,
                "tail": tail,
                "calibration_scale": calibration_scale,
                "observed_statistic": float(
                    observed_effect_stat[pathway_index]
                ),
                "observed_effect_statistic": float(
                    observed_effect_stat[pathway_index]
                ),
                "observed_calibration_statistic": float(
                    observed_calibration_raw[pathway_index]
                ),
                "max_absolute_effect": float(
                    effect_stats["max_absolute_effect"][pathway_index]
                ),
                "integrated_absolute_effect": float(
                    effect_stats["integrated_absolute_effect"][pathway_index]
                ),
                "l2_effect": float(effect_stats["l2_effect"][pathway_index]),
                "signed_integral": float(
                    effect_stats["signed_integral"][pathway_index]
                ),
                "peak_effect": float(effect_stats["peak_effect"][pathway_index]),
                "peak_bin": peak_bin,
                "peak_time": float((edges[peak_bin] + edges[peak_bin + 1]) / 2.0),
                "p_raw": float(p_raw[pathway_index]),
                "q_bh": float(q_bh[pathway_index]),
                "q_by": float(q_by[pathway_index]),
                "p_maxT": float(p_max_t[pathway_index]),
                "event_p": float(p_raw[pathway_index]),
                "event_q": float(q_by[pathway_index]),
                "event_fdr": float(q_by[pathway_index]),
                "fdr_method": "Benjamini-Yekutieli",
                "null_model": null_model,
                "n_perm": int(n_null),
                "n_perm_effective": int(n_null),
                "permutation_p_resolution": float(permutation_resolution),
                "freedman_lane_reference_p_resolution": float(
                    permutation_resolution
                ),
                "residual_permutation_space_size": int(residual_space),
                "condition_label_space_size": int(label_space),
                "condition_label_assignment_resolution": float(
                    condition_label_assignment_resolution
                ),
                "condition_assignment_minimum_p": float(
                    condition_assignment_minimum_p
                ),
                "log10_condition_label_space_size": log10_condition_label_space_size,
                "log10_condition_assignment_minimum_p": (
                    log10_condition_assignment_minimum_p
                ),
                "reference_enumeration": reference_enumeration,
                "exactness_status": exactness_status,
                "label_quotient_applies": label_quotient_applies,
                "condition_label_complement_symmetry_applies": (
                    condition_label_complement_symmetry_applies
                ),
                "reduced_design_block_constant": bool(block_constant),
                "minimum_residual_df": int(
                    design_diagnostics["residual_df_full"].min()
                ),
                "maximum_condition_vif": float(
                    design_diagnostics["condition_vif"].max()
                ),
                "median_pointwise_mde": float(
                    pathway_power["pointwise_mde_two_sided"].median()
                ),
                "maximum_pointwise_mde": float(
                    pathway_power["pointwise_mde_two_sided"].max()
                ),
                "calibration_warning": (
                    "Inference is conditional on fixed pseudotime, fixed grid, "
                    "availability-pattern-preserving whole-donor residual-curve "
                    "exchangeability, and the declared reduced model. Exhaustive "
                    "Freedman-Lane enumeration is not generally finite-sample exact "
                    "when nuisance rows vary within permutation blocks."
                ),
            }
        )
    pathway_tests = pd.DataFrame(pathway_rows).sort_values(
        ["event_fdr", "p_raw", "Pathway"]
    ).reset_index(drop=True)

    covariate_diagnostics = _covariate_diagnostics(
        donor_design,
        control=control,
        case=case,
        continuous_covariate_keys=continuous_covariate_keys,
        categorical_covariate_keys=categorical_covariate_keys,
        strata_keys=strata_keys,
        encoded=encoded,
    )
    grid_rows = []
    observed_case_all = donor_design_all["observed_case"].to_numpy(dtype=bool)
    for bin_id in range(len(edges) - 1):
        available_bin_all = counts_all[:, bin_id] >= min_cells_per_donor_bin
        available_bin_included = available_all[:, bin_id]
        grid_rows.append(
            {
                "bin_id": int(bin_id),
                "bin_left": float(edges[bin_id]),
                "bin_right": float(edges[bin_id + 1]),
                "bin_mid": float((edges[bin_id] + edges[bin_id + 1]) / 2.0),
                "bin_width": float(edges[bin_id + 1] - edges[bin_id]),
                "n_cells": int(counts_all[:, bin_id].sum()),
                "n_donors_available_all": int(available_bin_all.sum()),
                "n_control_available_all": int(
                    (available_bin_all & ~observed_case_all).sum()
                ),
                "n_case_available_all": int(
                    (available_bin_all & observed_case_all).sum()
                ),
                "n_inference_donors_available": int(
                    available_bin_included.sum()
                ),
                "selected_common_support": bool(bin_id in set(selected_bins)),
                "drop_reason": (
                    "" if bin_id in set(selected_bins) else "outside_selected_design_valid_segment"
                ),
            }
        )
    grid_diagnostics = pd.DataFrame(grid_rows)

    activity_rows: list[dict[str, Any]] = []
    if return_donor_bin_activity:
        selected_bin_to_local = {
            int(bin_id): local for local, bin_id in enumerate(selected_bins)
        }
        activity_design = donor_design_output.sort_values(
            "donor_index_all", kind="mergesort"
        )
        for _, donor_row in activity_design.iterrows():
            global_donor_index = int(donor_row["donor_index_all"])
            included = bool(donor_row["included_in_inference"])
            local_donor_index = (
                global_to_local[global_donor_index] if included else None
            )
            for bin_id in full_bin_ids:
                bin_in_selected_segment = int(bin_id) in selected_bin_to_local
                available = bool(
                    full_counts[global_donor_index, bin_id]
                    >= min_cells_per_donor_bin
                )
                analysis_selected = (
                    included and bin_in_selected_segment and available
                )
                for pathway_index, pathway in enumerate(pathway_names):
                    value = (
                        scores[
                            int(local_donor_index),
                            selected_bin_to_local[int(bin_id)],
                            pathway_index,
                        ]
                        if analysis_selected
                        else np.nan
                    )
                    activity_rows.append(
                        {
                            "donor": donor_row["donor"],
                            "observed_condition": donor_row["observed_condition"],
                            "restriction_stratum": donor_row[
                                "restriction_stratum"
                            ],
                            "availability_signature": donor_row[
                                "availability_signature"
                            ],
                            "permutation_block": donor_row["permutation_block"],
                            "bin_id": int(bin_id),
                            "bin_left": float(edges[bin_id]),
                            "bin_right": float(edges[bin_id + 1]),
                            "bin_mid": float(
                                (edges[bin_id] + edges[bin_id + 1]) / 2.0
                            ),
                            "n_cells": int(full_counts[global_donor_index, bin_id]),
                            "available": available,
                            "included_in_inference": included,
                            "bin_in_selected_segment": bin_in_selected_segment,
                            "analysis_selected": analysis_selected,
                            "Pathway": pathway,
                            "activity": float(value) if np.isfinite(value) else np.nan,
                        }
                    )
    donor_bin_activity = pd.DataFrame(
        activity_rows,
        columns=[
            "donor",
            "observed_condition",
            "restriction_stratum",
            "availability_signature",
            "permutation_block",
            "bin_id",
            "bin_left",
            "bin_right",
            "bin_mid",
            "n_cells",
            "available",
            "included_in_inference",
            "bin_in_selected_segment",
            "analysis_selected",
            "Pathway",
            "activity",
        ],
    )

    assignment_rows: list[dict[str, Any]] = []
    if return_permutation_assignments:
        identity = np.arange(len(donor_design), dtype=int)
        mappings = [(-1, identity, True)] + [
            (perm_id, mapping, False)
            for perm_id, mapping in enumerate(plan.null_mappings)
        ]
        for perm_id, mapping, is_identity in mappings:
            mapping_hash = _mapping_hash(mapping)
            for target_index, source_index in enumerate(mapping):
                target = donor_design.iloc[target_index]
                source = donor_design.iloc[int(source_index)]
                assignment_rows.append(
                    {
                        "perm_id": int(perm_id),
                        "mapping_hash": mapping_hash,
                        "is_identity_mapping": bool(is_identity),
                        "target_donor": target["donor"],
                        "residual_source_donor": source["donor"],
                        "permutation_block": target["permutation_block"],
                        "target_availability_signature": target[
                            "availability_signature"
                        ],
                        "source_availability_signature": source[
                            "availability_signature"
                        ],
                    }
                )
    permutation_assignments = pd.DataFrame(
        assignment_rows,
        columns=[
            "perm_id",
            "mapping_hash",
            "is_identity_mapping",
            "target_donor",
            "residual_source_donor",
            "permutation_block",
            "target_availability_signature",
            "source_availability_signature",
        ],
    )
    null_statistics = pd.DataFrame(
        null_rows,
        columns=[
            "perm_id",
            "mapping_hash",
            "Pathway",
            "effect_statistic",
            "raw_calibration_statistic",
            "calibration_statistic",
            "max_pathway_calibration_statistic",
        ],
    )

    permutation_summary = pd.DataFrame(
        [
            {
                "calibration_method": "freedman_lane_whole_donor_residual_curve",
                "requested_mode": plan.requested_mode,
                "actual_mode": plan.actual_mode,
                "reference_enumeration": reference_enumeration,
                "exactness_status": exactness_status,
                "reduced_design_block_constant": bool(block_constant),
                "restriction": "strata_and_availability_signature",
                "strata_keys": "|".join(strata_keys),
                "availability_pattern_restriction": True,
                "residual_permutation_space_size": int(residual_space),
                "condition_label_space_size": int(label_space),
                "n_null_mappings_possible": int(residual_space - 1),
                "n_null_mappings_evaluated": int(n_null),
                "n_reference_mappings": int(denominator),
                "n_permutations_requested": int(n_permutations),
                "identity_mapping_in_null": False,
                "permutation_p_resolution": float(permutation_resolution),
                "freedman_lane_reference_p_resolution": float(
                    permutation_resolution
                ),
                "condition_label_assignment_resolution": float(
                    condition_label_assignment_resolution
                ),
                "condition_assignment_minimum_p": float(
                    condition_assignment_minimum_p
                ),
                "log10_condition_label_space_size": log10_condition_label_space_size,
                "log10_condition_assignment_minimum_p": (
                    log10_condition_assignment_minimum_p
                ),
                "label_quotient_applies": label_quotient_applies,
                "condition_label_complement_symmetry_applies": (
                    condition_label_complement_symmetry_applies
                ),
                "monte_carlo_does_not_increase_condition_label_space": True,
                "n_donors_included": int(len(donor_design)),
                "n_donors_excluded": int((~included_mask).sum()),
                "n_permutation_blocks": int(len(groups)),
                "n_mobile_blocks": int(sum(len(group) > 1 for group in groups)),
                "n_immobile_donors": int(
                    sum(len(group) for group in groups if len(group) == 1)
                ),
                "n_pathways": int(len(pathway_names)),
                "draw_attempts": int(plan.draw_attempts),
                "duplicate_draws": int(plan.duplicate_draws),
                "seed": int(seed),
            }
        ]
    )

    try:
        from . import __version__ as pyfgsea_version
    except Exception:
        pyfgsea_version = None
    gene_universe_hash = hashlib.sha256(
        "\n".join(map(str, genes)).encode("utf-8")
    ).hexdigest()
    pathway_family_payload = (
        pathway_membership[["Pathway", "gene", "weight", "pathway_size"]]
        .sort_values(["Pathway", "gene", "weight"])
        .to_csv(index=False, lineterminator="\n")
        .encode("utf-8")
    )
    pathway_family_hash = hashlib.sha256(pathway_family_payload).hexdigest()
    source_availability = counts_all >= min_cells_per_donor_bin
    selected_counts_for_hash = counts_all[included_mask][:, selected_bins]
    selected_availability_for_hash = source_availability[included_mask][
        :, selected_bins
    ]

    def _small_array_hash(values: np.ndarray, dtype: str) -> str:
        canonical = np.ascontiguousarray(np.asarray(values).astype(dtype))
        return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()

    def _json_contract_hash(value: Any) -> str:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    ordered_donor_ids = donor_design_all["donor"].astype(str).tolist()
    included_donor_ids = donor_design["donor"].astype(str).tolist()
    source_counts_hash = _small_array_hash(counts_all, "<i8")
    source_availability_hash = _small_array_hash(source_availability, "u1")
    included_mask_hash = _small_array_hash(included_mask, "u1")
    selected_bin_mask = np.isin(
        np.arange(len(edges) - 1, dtype=int), selected_bins
    )
    selected_bin_mask_hash = _small_array_hash(selected_bin_mask, "u1")
    selected_counts_hash = _small_array_hash(selected_counts_for_hash, "<i8")
    selected_availability_hash = _small_array_hash(
        selected_availability_for_hash, "u1"
    )
    residual_mappings_hash = _small_array_hash(
        np.stack(plan.null_mappings, axis=0), "<i8"
    )
    grid_edges_hash = _small_array_hash(edges, "<f8")
    pseudobulk_gene_ids = [
        str(genes[index]) for index in pseudobulk_gene_indices
    ]
    pseudobulk_gene_order_hash = _json_contract_hash(pseudobulk_gene_ids)
    support_block_rows = donor_design[
        ["donor", "availability_signature", "permutation_block"]
    ].astype(str).to_dict(orient="records")
    source_grid_support_contract_hash = _json_contract_hash(
        {
            "schema": "donor_major_bin_minor_fixed_grid_support_v1",
            "shape": [int(len(donor_design_all)), int(len(edges) - 1)],
            "ordered_donor_ids": ordered_donor_ids,
            "grid_edges": edges.astype(float).tolist(),
            "grid_edges_sha256_float64_le": grid_edges_hash,
            "min_cells_per_donor_bin": int(min_cells_per_donor_bin),
            "counts_sha256_int64_le": source_counts_hash,
            "availability_sha256_uint8": source_availability_hash,
            "row_order": "donor_major_bin_minor",
        }
    )
    source_grid_matrix_hash = _small_array_hash(pseudobulk_full, "<f8")
    source_grid_matrix_contract_hash = _json_contract_hash(
        {
            "schema": "source_grid_pseudobulk_matrix_v1",
            "support_contract_sha256": source_grid_support_contract_hash,
            "shape": list(map(int, pseudobulk_full.shape)),
            "dtype": "float64_le",
            "matrix_sha256_float64_le": source_grid_matrix_hash,
            "pseudobulk_gene_order_sha256": pseudobulk_gene_order_hash,
            "expression_source": expression_source,
            "layer": layer,
            "use_raw": bool(use_raw),
        }
    )
    import inspect

    selector_functions = (
        _encode_reduced_design,
        _make_pattern_blocks,
        _space_sizes,
        _bin_design_row,
        _select_segment,
    )
    selector_source_bundle = "\n\n".join(
        f"{function.__module__}.{function.__qualname__}\n"
        f"{inspect.getsource(function)}"
        for function in selector_functions
    )
    selector_code_hash = hashlib.sha256(
        selector_source_bundle.encode("utf-8")
    ).hexdigest()
    selected_support_contract_hash = _json_contract_hash(
        {
            "schema": "selected_contiguous_support_v1",
            "source_grid_support_contract_sha256": (
                source_grid_support_contract_hash
            ),
            "selector_id": "trajectory_covariate_pseudobulk._select_segment_v1",
            "selector_implementation_bundle_sha256": selector_code_hash,
            "selector_tie_break": (
                "longest_then_most_donor_bin_observations_then_most_donors_"
                "then_earliest"
            ),
            "support_policy": {
                "min_cells_per_donor_bin": int(min_cells_per_donor_bin),
                "min_donors_per_condition": int(min_donors_per_condition),
                "min_common_bins": int(min_common_bins),
                "min_residual_df": int(min_residual_df),
                "max_condition_vif": float(max_condition_vif),
            },
            "selected_bin_ids": selected_bins.astype(int).tolist(),
            "selected_bin_mask_sha256_uint8": selected_bin_mask_hash,
            "included_donor_ids": included_donor_ids,
            "included_donor_mask_sha256_uint8": included_mask_hash,
            "selected_counts_sha256_int64_le": selected_counts_hash,
            "selected_availability_sha256_uint8": selected_availability_hash,
            "reduced_design_sha256_float64_le": _small_array_hash(
                encoded.reduced, "<f8"
            ),
            "reduced_model_terms": list(encoded.terms),
            "design_encoding": encoded.encoding,
            "support_blocks": support_block_rows,
            "residual_permutation_space_size": int(residual_space),
            "condition_label_space_size": int(label_space),
            "actual_permutation_mode": str(plan.actual_mode),
            "n_null_mappings_evaluated": int(n_null),
            "residual_mappings_sha256_int64_le": residual_mappings_hash,
            "n_permutations_requested": int(n_permutations),
            "max_exact_permutations": int(max_exact_permutations),
            "seed": int(seed),
        }
    )

    metadata = {
        "method": "covariate_adjusted_fixed_grid_donor_pseudobulk_freedman_lane",
        "pyfgsea_version": pyfgsea_version,
        "gene_universe_hash": gene_universe_hash,
        "pathway_family_hash": pathway_family_hash,
        "pathway_score": "weighted_mean_gene_z_across_donor_bin_pseudobulk",
        "estimand": "conditional_regulation_adjusted_condition_coefficient",
        "condition_key": condition_key,
        "donor_key": donor_key,
        "control": str(control),
        "case": str(case),
        "pseudotime_key": pseudotime_key,
        "continuous_covariate_keys": list(continuous_covariate_keys),
        "categorical_covariate_keys": list(categorical_covariate_keys),
        "strata_keys": list(strata_keys),
        "donor_order_rule": donor_order,
        "strata_enter_reduced_model_as_fixed_effects": True,
        "reduced_model_terms": encoded.terms,
        "design_encoding_json": json.dumps(
            encoded.encoding, ensure_ascii=False, separators=(",", ":")
        ),
        "grid_edges": edges.tolist(),
        "grid_edges_sha256_float64_le": grid_edges_hash,
        "source_grid_n_bins": int(len(edges) - 1),
        "source_grid_donor_ids": ordered_donor_ids,
        "source_grid_counts_sha256_int64_le": source_counts_hash,
        "source_grid_availability_sha256_uint8": source_availability_hash,
        "source_grid_support_contract_sha256": (
            source_grid_support_contract_hash
        ),
        "source_grid_pseudobulk_sha256_float64_le": source_grid_matrix_hash,
        "source_grid_matrix_contract_sha256": source_grid_matrix_contract_hash,
        "selected_bin_ids": selected_bins.astype(int).tolist(),
        "selected_bin_mask_sha256_uint8": selected_bin_mask_hash,
        "included_donor_ids": included_donor_ids,
        "included_donor_mask_sha256_uint8": included_mask_hash,
        "selected_counts_sha256_int64_le": selected_counts_hash,
        "selected_availability_sha256_uint8": selected_availability_hash,
        "selected_support_contract_sha256": selected_support_contract_hash,
        "support_selector_implementation_sha256": selector_code_hash,
        "reduced_design_sha256_float64_le": _small_array_hash(
            encoded.reduced, "<f8"
        ),
        "support_blocks_sha256": _json_contract_hash(support_block_rows),
        "pseudobulk_gene_order_sha256": pseudobulk_gene_order_hash,
        "segment_selection": (
            "longest_design_valid_contiguous_segment_then_most_donor_bin_"
            "observations_then_most_donors_then_earliest"
        ),
        "missing_curve_policy": (
            "whole_curve_permutation_within_identical_selected_bin_"
            "availability_signature"
        ),
        "min_cells_per_donor_bin": int(min_cells_per_donor_bin),
        "min_donors_per_condition": int(min_donors_per_condition),
        "min_common_bins": int(min_common_bins),
        "min_residual_df": int(min_residual_df),
        "max_condition_vif": float(max_condition_vif),
        "statistic": statistic,
        "tail": tail,
        "calibration_scale": calibration_scale,
        "calibration_method": "freedman_lane_whole_donor_residual_curve",
        "reference_enumeration": reference_enumeration,
        "exactness_status": exactness_status,
        "reduced_design_block_constant": bool(block_constant),
        "label_quotient_applies": label_quotient_applies,
        "condition_label_complement_symmetry_applies": (
            condition_label_complement_symmetry_applies
        ),
        "residual_permutation_space_size": int(residual_space),
        "condition_label_space_size": int(label_space),
        "n_null_mappings_evaluated": int(n_null),
        "residual_mappings_sha256_int64_le": residual_mappings_hash,
        "n_permutations_requested": int(n_permutations),
        "p_value_rule": "(1 + null mappings >= observed) / (B + 1)",
        "maxT_scope": "single_step_across_pathways",
        "maxT_strong_fwer_condition": "requires_subset_pivotality",
        "bh_dependency_assumption": "independent_or_positive_regression_dependence",
        "by_available_for_arbitrary_dependence": True,
        "expression_source": expression_source,
        "layer": layer,
        "use_raw": bool(use_raw),
        "raw_counts_offset_model": "not_estimated",
        "retain_all_genes": bool(retain_all_genes),
        "n_expression_genes": int(len(genes)),
        "n_pseudobulk_genes": int(len(pseudobulk_gene_indices)),
        "return_donor_bin_activity": bool(return_donor_bin_activity),
        "min_size": int(min_size),
        "max_size": int(max_size),
        "alpha": float(alpha),
        "power_target": float(power_target),
        "power_method": "noncentral_t_pointwise_analytic_approximation_not_curve_power",
        "seed": int(seed),
        "inference_scope": "conditional_on_fixed_input_pseudotime_and_grid",
        "exchangeability_assumption": (
            "whole donor residual curves exchangeable within declared strata "
            "and identical availability signatures"
        ),
        "trajectory_reestimation_required_for_unconditional_test": True,
        "identity_mapping_in_null": False,
        "permutation_p_resolution": float(permutation_resolution),
        "freedman_lane_reference_p_resolution": float(permutation_resolution),
        "condition_label_assignment_resolution": float(
            condition_label_assignment_resolution
        ),
        "condition_assignment_minimum_p": float(
            condition_assignment_minimum_p
        ),
        "log10_condition_label_space_size": log10_condition_label_space_size,
        "log10_condition_assignment_minimum_p": (
            log10_condition_assignment_minimum_p
        ),
    }
    for table in (
        pathway_tests,
        effect_curves,
        grid_diagnostics,
        design_diagnostics,
        covariate_diagnostics,
        power_diagnostics,
    ):
        table.attrs["covariate_adjusted_donor_pseudobulk"] = metadata.copy()

    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError(
            "run_covariate_adjusted_donor_pseudobulk requires anndata"
        ) from exc
    pseudobulk_obs_rows: list[dict[str, Any]] = []
    retained_design_columns = [
        "donor",
        "observed_condition",
        *continuous_covariate_keys,
        *categorical_covariate_keys,
        *strata_keys,
        "restriction_stratum",
        "availability_signature",
        "permutation_block",
    ]
    pseudobulk_design = donor_design_output.sort_values(
        "donor_index_all", kind="mergesort"
    ).reset_index(drop=True)
    if not np.array_equal(
        pd.to_numeric(
            pseudobulk_design["donor_index_all"], errors="coerce"
        ).to_numpy(dtype=float),
        np.arange(len(donor_design_all), dtype=float),
    ):
        raise RuntimeError("Donor order changed while materializing the source grid")
    selected_bin_set = set(map(int, selected_bins))
    for donor_index, donor_row in pseudobulk_design.iterrows():
        included = bool(donor_row["included_in_inference"])
        for bin_id in full_bin_ids:
            available = bool(
                full_counts[donor_index, bin_id] >= min_cells_per_donor_bin
            )
            bin_in_selected_segment = bool(int(bin_id) in selected_bin_set)
            row = {
                key: donor_row[key]
                for key in retained_design_columns
                if key in donor_row.index
            }
            row.update(
                {
                    "bin_id": int(bin_id),
                    "bin_left": float(edges[bin_id]),
                    "bin_right": float(edges[bin_id + 1]),
                    "bin_mid": float((edges[bin_id] + edges[bin_id + 1]) / 2.0),
                    "n_cells": int(full_counts[donor_index, bin_id]),
                    "available": available,
                    "included_in_inference": included,
                    "bin_in_selected_segment": bin_in_selected_segment,
                    "analysis_selected": bool(
                        included and bin_in_selected_segment and available
                    ),
                }
            )
            pseudobulk_obs_rows.append(row)
    pseudobulk_obs = pd.DataFrame(
        pseudobulk_obs_rows,
        index=[
            f"{row['donor']}__bin{row['bin_id']}" for row in pseudobulk_obs_rows
        ],
    )
    source_grid_matrix = pseudobulk_full.reshape(-1, pseudobulk_full.shape[-1])
    formal_matrix = source_grid_matrix.copy()
    formal_matrix[
        ~pseudobulk_obs["analysis_selected"].to_numpy(dtype=bool)
    ] = np.nan
    pseudobulk_adata = ad.AnnData(
        X=formal_matrix,
        obs=pseudobulk_obs,
        var=pd.DataFrame(
            index=[str(genes[index]) for index in pseudobulk_gene_indices]
        ),
    )
    pseudobulk_adata.layers["source_grid_pseudobulk"] = source_grid_matrix
    pseudobulk_adata.uns["covariate_adjusted_donor_pseudobulk"] = {
        key: _h5ad_safe_value(value)
        for key, value in metadata.items()
        if value is not None
    }
    pseudobulk_adata.uns["pseudobulk_matrix_contract"] = {
        "X": "formal_selected_support_only; all other donor-bins are NaN",
        "source_grid_pseudobulk_layer": (
            "complete fixed source grid; values are present only when the "
            "donor-bin meets min_cells_per_donor_bin"
        ),
        "source_grid_rows": int(len(donor_design_all) * len(full_bin_ids)),
        "source_grid_bins": int(len(full_bin_ids)),
    }
    donor_design_output = donor_design_output.drop(
        columns=["__stratum_key"], errors="ignore"
    ).sort_values("donor_index_all", kind="mergesort").reset_index(drop=True)
    return CovariateAdjustedDonorPseudobulkResult(
        pathway_tests=pathway_tests,
        effect_curves=effect_curves,
        pseudobulk_adata=pseudobulk_adata,
        donor_bin_activity=donor_bin_activity,
        grid_diagnostics=grid_diagnostics,
        segment_diagnostics=segment_diagnostics,
        donor_design=donor_design_output,
        covariate_diagnostics=covariate_diagnostics,
        design_diagnostics=design_diagnostics,
        power_diagnostics=power_diagnostics,
        pathway_membership=pathway_membership,
        permutation_summary=permutation_summary,
        permutation_assignments=permutation_assignments,
        null_statistics=null_statistics,
        metadata=metadata,
    )
