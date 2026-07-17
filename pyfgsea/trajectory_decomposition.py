from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .trajectory_covariate_pseudobulk import (
    CovariateAdjustedDonorPseudobulkResult,
    CovariateDesignError,
    _block_constant_reduced_design,
    _encode_reduced_design,
    _fit_observed_models,
    _fit_permuted_curves,
    _formal_inference_donor_bin_view,
    _formal_inference_donor_design_view,
    _mapping_hash,
    _make_residual_plan,
    run_covariate_adjusted_donor_pseudobulk,
)
from .trajectory_covariate_simulation import (
    ArrayFreedmanLanePlan,
    run_array_freedman_lane_calibration_batch,
)
from .trajectory_pseudobulk import (
    _assign_fixed_bins,
    _bh_adjust,
    _by_adjust,
    _curve_statistics,
    _fixed_edges,
    _normalize_statistic,
    _normalize_tail,
    _require_integer,
    _test_scale,
    _validate_stringification_is_injective,
)


@dataclass
class RegulationOccupancyFateResult:
    """Parallel donor-level regulation, occupancy, and fate results.

    The three components are associations on a shared donor design and shared
    whole-donor residual mapping stream.  They are not a causal mediation
    decomposition and are not assumed to add to a common outcome.
    """

    regulation_tests: pd.DataFrame
    regulation_curves: pd.DataFrame
    occupancy_tests: pd.DataFrame
    occupancy_curves: pd.DataFrame
    fate_tests: pd.DataFrame
    fate_effects: pd.DataFrame
    donor_state_counts: pd.DataFrame
    donor_fate_counts: pd.DataFrame
    component_summary: pd.DataFrame
    component_diagnostics: pd.DataFrame
    donor_design: pd.DataFrame
    design_diagnostics: pd.DataFrame
    permutation_summary: pd.DataFrame
    permutation_assignments: pd.DataFrame
    null_statistics: pd.DataFrame
    regulation_result: CovariateAdjustedDonorPseudobulkResult
    metadata: Dict[str, Any]

    def to_tables(self) -> Dict[str, pd.DataFrame]:
        """Return independent copies of all flat tabular outputs."""
        return {
            "regulation_tests": self.regulation_tests.copy(),
            "regulation_curves": self.regulation_curves.copy(),
            "occupancy_tests": self.occupancy_tests.copy(),
            "occupancy_curves": self.occupancy_curves.copy(),
            "fate_tests": self.fate_tests.copy(),
            "fate_effects": self.fate_effects.copy(),
            "donor_state_counts": self.donor_state_counts.copy(),
            "donor_fate_counts": self.donor_fate_counts.copy(),
            "component_summary": self.component_summary.copy(),
            "component_diagnostics": self.component_diagnostics.copy(),
            "donor_design": self.donor_design.copy(),
            "design_diagnostics": self.design_diagnostics.copy(),
            "permutation_summary": self.permutation_summary.copy(),
            "permutation_assignments": self.permutation_assignments.copy(),
            "null_statistics": self.null_statistics.copy(),
        }


@dataclass
class _ComponentCalibration:
    tests: pd.DataFrame
    curves: pd.DataFrame
    null_statistics: pd.DataFrame
    observed_scaled: np.ndarray
    null_scaled: np.ndarray


def _as_key_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence, not a string")
    result = tuple(str(value) for value in values)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicate keys")
    return result


def _validate_complete_cohort_donor_design(
    adata,
    *,
    condition_key: str,
    donor_key: str,
    control,
    case,
    continuous_covariate_keys: tuple[str, ...],
    categorical_covariate_keys: tuple[str, ...],
    strata_keys: tuple[str, ...],
) -> set[str]:
    """Validate donor-level design fields before branch-restricted fitting."""
    donor_constant_keys = (
        condition_key,
        *continuous_covariate_keys,
        *categorical_covariate_keys,
        *strata_keys,
    )
    required = [donor_key, *donor_constant_keys]
    if len(set(required)) != len(required):
        raise ValueError(
            "condition, donor, continuous covariate, categorical covariate, and "
            "strata keys must be distinct"
        )
    missing = [key for key in required if key not in adata.obs]
    if missing:
        raise KeyError(f"Missing adata.obs columns: {missing}")

    obs = adata.obs.loc[:, required].copy()
    _validate_stringification_is_injective(obs[donor_key], donor_key)
    _validate_stringification_is_injective(obs[condition_key], condition_key)
    requested = (
        obs[condition_key].notna()
        & obs[condition_key].astype(str).isin([str(control), str(case)])
    )
    if not requested.any():
        raise ValueError("No cells match the requested control/case conditions")
    if obs.loc[requested, donor_key].isna().any():
        raise ValueError(f"donor_key '{donor_key}' contains missing values")
    requested_donors = set(obs.loc[requested, donor_key].astype(str))

    donor_values = obs[donor_key].astype("string")
    cohort = obs.loc[
        donor_values.notna() & donor_values.isin(requested_donors)
    ].copy()
    donor_constant_columns = list(donor_constant_keys)
    if cohort.loc[:, donor_constant_columns].isna().any().any():
        bad = cohort.loc[:, donor_constant_columns].columns[
            cohort.loc[:, donor_constant_columns].isna().any()
        ].tolist()
        raise ValueError(
            "Complete requested donor cohort contains missing donor-level design "
            f"values in {bad}"
        )

    for key in (condition_key, *categorical_covariate_keys, *strata_keys):
        _validate_stringification_is_injective(cohort[key], key)
        cohort[key] = cohort[key].astype(str)
    cohort[donor_key] = cohort[donor_key].astype(str)
    for key in continuous_covariate_keys:
        numeric = pd.to_numeric(cohort[key], errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"Continuous covariate '{key}' must be finite numeric")
        cohort[key] = numeric.astype(float)

    for donor, group in cohort.groupby(donor_key, sort=True):
        for key in donor_constant_keys:
            if len(pd.unique(group[key])) != 1:
                raise ValueError(
                    f"Donor-level design key '{key}' is not constant for donor "
                    f"'{donor}' across the complete cohort before branch filtering"
                )
    return requested_donors


def _groups_from_design(donor_design: pd.DataFrame) -> list[np.ndarray]:
    lookup: dict[str, list[int]] = {}
    for index, block in enumerate(donor_design["permutation_block"].astype(str)):
        lookup.setdefault(block, []).append(index)
    groups = [
        np.asarray(indices, dtype=int)
        for _, indices in sorted(lookup.items(), key=lambda item: item[0])
    ]
    if not groups or sum(len(group) for group in groups) != len(donor_design):
        raise RuntimeError("Invalid donor permutation blocks in regulation result")
    return groups


def _reduced_design(
    donor_design: pd.DataFrame,
    *,
    continuous_covariate_keys: tuple[str, ...],
    categorical_covariate_keys: tuple[str, ...],
    strata_keys: tuple[str, ...],
):
    frame = donor_design.copy()
    frame["__stratum_key"] = [
        tuple(str(row[key]) for key in strata_keys)
        if strata_keys
        else ("__all__",)
        for _, row in frame.iterrows()
    ]
    return _encode_reduced_design(
        frame,
        continuous_covariate_keys=continuous_covariate_keys,
        categorical_covariate_keys=categorical_covariate_keys,
        strata_keys=strata_keys,
    )


def _exactness_status(
    block_constant: bool,
    denominator_block_constant: bool,
    is_exhaustive: bool,
) -> str:
    if not denominator_block_constant and is_exhaustive:
        return (
            "exhaustive_joint_freedman_lane_reference_denominator_"
            "heterogeneity_not_exact"
        )
    if not denominator_block_constant:
        return (
            "monte_carlo_joint_freedman_lane_denominator_heterogeneity_"
            "approximation"
        )
    if block_constant and is_exhaustive:
        return "finite_sample_exact_joint_residual_group_conditional_on_invariance"
    if block_constant:
        return "monte_carlo_joint_residual_group_conditional_on_invariance"
    if is_exhaustive:
        return "exhaustive_joint_freedman_lane_reference_not_finite_sample_exact"
    return "monte_carlo_joint_freedman_lane_approximation"


def _softmax_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def _clr(smoothed_proportions: np.ndarray) -> np.ndarray:
    values = np.asarray(smoothed_proportions, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("Compositional responses require at least two parts")
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Smoothed compositional proportions must be finite and positive")
    logs = np.log(values)
    return logs - logs.mean(axis=1, keepdims=True)


def occupancy_response_from_counts(
    counts: np.ndarray,
    *,
    pseudocount: float = 0.5,
    min_cells_per_donor: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the production occupancy CLR response from donor-by-bin counts.

    This array entry point is deliberately shared by the fitted decomposition
    and design calibration.  Zeros remain observed compositional zeros and are
    handled only by the frozen additive pseudocount.
    """
    values = np.asarray(counts)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("Occupancy counts must be a donor-by-bin matrix")
    if not np.issubdtype(values.dtype, np.integer):
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError("Occupancy counts must be finite non-negative integers")
        values = values.astype(np.int64)
    else:
        values = values.astype(np.int64, copy=False)
    if np.any(values < 0):
        raise ValueError("Occupancy counts must be non-negative")
    if not math.isfinite(float(pseudocount)) or float(pseudocount) <= 0:
        raise ValueError("pseudocount must be finite and positive")
    if isinstance(min_cells_per_donor, bool) or int(min_cells_per_donor) < 1:
        raise ValueError("min_cells_per_donor must be a positive integer")
    totals = values.sum(axis=1)
    if np.any(totals < int(min_cells_per_donor)):
        raise CovariateDesignError(
            "State-occupancy denominators are below min_cells_per_donor"
        )
    raw = values / totals[:, None]
    smoothed = (values + float(pseudocount)) / (
        totals[:, None] + float(pseudocount) * values.shape[1]
    )
    return _clr(smoothed)[:, :, None], raw, smoothed


def fate_response_from_masses(
    masses: np.ndarray,
    denominators: Sequence[float] | None = None,
    *,
    pseudocount: float = 0.5,
    min_fate_cells_per_donor: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the production fate CLR response from donor-by-fate masses."""
    values = np.asarray(masses, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("Fate masses must be a donor-by-fate matrix")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Fate masses must be finite and non-negative")
    if denominators is None:
        denominator_array = values.sum(axis=1)
    else:
        denominator_array = np.asarray(denominators, dtype=float)
    if denominator_array.shape != (values.shape[0],):
        raise ValueError("Fate denominators must have one value per donor")
    if (
        not np.isfinite(denominator_array).all()
        or np.any(denominator_array < int(min_fate_cells_per_donor))
        or np.any(values.sum(axis=1) > denominator_array + 1e-8)
    ):
        raise CovariateDesignError(
            "Fate denominators are invalid or below min_fate_cells_per_donor"
        )
    if not math.isfinite(float(pseudocount)) or float(pseudocount) <= 0:
        raise ValueError("pseudocount must be finite and positive")
    raw = values / denominator_array[:, None]
    smoothed = (values + float(pseudocount)) / (
        denominator_array[:, None] + float(pseudocount) * values.shape[1]
    )
    return _clr(smoothed)[:, None, :], raw, smoothed


def _activity_tensor(
    result: CovariateAdjustedDonorPseudobulkResult,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    formal_design = _formal_inference_donor_design_view(result.donor_design)
    donors = formal_design["donor"].astype(str).tolist()
    bins = np.asarray(result.metadata["selected_bin_ids"], dtype=int)
    pathways = result.pathway_tests["Pathway"].astype(str).tolist()
    donor_index = {value: index for index, value in enumerate(donors)}
    bin_index = {int(value): index for index, value in enumerate(bins)}
    pathway_index = {value: index for index, value in enumerate(pathways)}
    scores = np.full((len(donors), len(bins), len(pathways)), np.nan, dtype=float)
    activity = _formal_inference_donor_bin_view(
        result.donor_bin_activity, label="donor_bin_activity"
    )
    if activity.empty:
        raise RuntimeError(
            "Regulation donor-bin activity was not retained for joint calibration"
        )
    for row in activity.itertuples(index=False):
        d = donor_index[str(row.donor)]
        b = bin_index[int(row.bin_id)]
        p = pathway_index[str(row.Pathway)]
        scores[d, b, p] = float(row.activity)
    first_mask = np.isfinite(scores[:, :, 0])
    if not all(np.array_equal(np.isfinite(scores[:, :, index]), first_mask) for index in range(scores.shape[2])):
        raise RuntimeError("Pathway activity availability differs across pathways")
    return scores, pathways, bins


def _calibrate_component(
    scores: np.ndarray,
    feature_names: Sequence[str],
    *,
    component: str,
    axis_ids: np.ndarray,
    axis_left: np.ndarray,
    axis_right: np.ndarray,
    reduced: np.ndarray,
    condition: np.ndarray,
    mappings: Sequence[np.ndarray],
    statistic: str,
    tail: str,
    calibration_scale: str,
    return_null_statistics: bool,
) -> _ComponentCalibration:
    feature_names = [str(value) for value in feature_names]
    if scores.shape != (len(condition), len(axis_ids), len(feature_names)):
        raise ValueError(f"Invalid {component} response tensor shape")
    widths = np.asarray(axis_right, dtype=float) - np.asarray(axis_left, dtype=float)
    if not np.isfinite(widths).all() or np.any(widths <= 0):
        raise ValueError(f"{component} curve widths must be finite and positive")
    (
        fits,
        beta,
        t_value,
        standard_error,
        residual_sd,
        adjusted_control,
        adjusted_case,
    ) = _fit_observed_models(scores, reduced, condition)
    calibration_curve = t_value if calibration_scale == "studentized" else beta
    if not np.isfinite(calibration_curve).all():
        raise CovariateDesignError(
            f"{component} has an undefined {calibration_scale} statistic; "
            "the donor-level response has insufficient residual variation"
        )
    effect_stats = _curve_statistics(beta, widths)
    calibration_stats = _curve_statistics(calibration_curve, widths)
    raw_calibration = calibration_stats[statistic]
    observed_scaled = _test_scale(raw_calibration, statistic, tail)

    null_scaled_rows: list[np.ndarray] = []
    null_rows: list[dict[str, Any]] = []
    exceed = np.zeros(len(feature_names), dtype=int)
    component_max_exceed = np.zeros(len(feature_names), dtype=int)
    for permutation_id, mapping in enumerate(mappings):
        null_beta, null_t = _fit_permuted_curves(fits, mapping)
        null_curve = null_t if calibration_scale == "studentized" else null_beta
        if not np.isfinite(null_curve).all():
            raise CovariateDesignError(
                f"A permuted {component} {calibration_scale} statistic is undefined"
            )
        null_effect = _curve_statistics(null_beta, widths)[statistic]
        null_raw = _curve_statistics(null_curve, widths)[statistic]
        null_scaled = _test_scale(null_raw, statistic, tail)
        null_scaled_rows.append(null_scaled)
        exceed += null_scaled >= observed_scaled - 1e-12
        maximum = float(np.max(null_scaled))
        component_max_exceed += maximum >= observed_scaled - 1e-12
        if return_null_statistics:
            for index, feature in enumerate(feature_names):
                null_rows.append(
                    {
                        "perm_id": int(permutation_id),
                        "component": component,
                        "feature": feature,
                        "effect_statistic": float(null_effect[index]),
                        "raw_calibration_statistic": float(null_raw[index]),
                        "calibration_statistic": float(null_scaled[index]),
                        "component_max_calibration_statistic": maximum,
                    }
                )
    null_scaled_matrix = np.vstack(null_scaled_rows)
    denominator = len(mappings) + 1
    p_raw = (exceed + 1.0) / denominator
    p_component_max_t = (component_max_exceed + 1.0) / denominator
    peak_local = effect_stats["peak_bin_local"].astype(int)
    test_rows = []
    for index, feature in enumerate(feature_names):
        local = int(peak_local[index])
        test_rows.append(
            {
                "component": component,
                "feature": feature,
                "statistic": statistic,
                "tail": tail,
                "calibration_scale": calibration_scale,
                "effect_statistic": float(effect_stats[statistic][index]),
                "raw_calibration_statistic": float(raw_calibration[index]),
                "calibration_statistic": float(observed_scaled[index]),
                "peak_axis_id": int(axis_ids[local]),
                "peak_effect": float(effect_stats["peak_effect"][index]),
                "p_raw": float(p_raw[index]),
                "p_component_maxT": float(p_component_max_t[index]),
            }
        )
    tests = pd.DataFrame(test_rows)
    tests["q_component_bh"] = _bh_adjust(tests["p_raw"].to_numpy(dtype=float))
    tests["q_component_by"] = _by_adjust(tests["p_raw"].to_numpy(dtype=float))

    curve_rows = []
    for local, axis_id in enumerate(axis_ids):
        for feature_index, feature in enumerate(feature_names):
            curve_rows.append(
                {
                    "component": component,
                    "feature": feature,
                    "axis_id": int(axis_id),
                    "axis_left": float(axis_left[local]),
                    "axis_right": float(axis_right[local]),
                    "axis_mid": float((axis_left[local] + axis_right[local]) / 2.0),
                    "beta_clr": float(beta[local, feature_index]),
                    "standard_error_clr": float(standard_error[local, feature_index]),
                    "t_value": float(t_value[local, feature_index]),
                    "residual_sd_clr": float(residual_sd[local, feature_index]),
                    "adjusted_control_clr": float(adjusted_control[local, feature_index]),
                    "adjusted_case_clr": float(adjusted_case[local, feature_index]),
                }
            )
    return _ComponentCalibration(
        tests=tests,
        curves=pd.DataFrame(curve_rows),
        null_statistics=pd.DataFrame(
            null_rows,
            columns=[
                "perm_id",
                "component",
                "feature",
                "effect_statistic",
                "raw_calibration_statistic",
                "calibration_statistic",
                "component_max_calibration_statistic",
            ],
        ),
        observed_scaled=observed_scaled,
        null_scaled=null_scaled_matrix,
    )


def calibrate_compositional_component_arrays(
    scores: np.ndarray,
    feature_names: Sequence[str],
    *,
    component: str,
    axis_ids: Sequence[int],
    axis_left: Sequence[float],
    axis_right: Sequence[float],
    reduced_design: np.ndarray,
    condition: Sequence[bool],
    null_mappings: Sequence[np.ndarray],
    statistic: str = "max_absolute_effect",
    tail: str = "greater",
    calibration_scale: str = "studentized",
    return_null_statistics: bool = False,
) -> dict[str, Any]:
    """Run the production decomposition component kernel on aligned arrays.

    The calibration runner uses this entry point instead of a scalar surrogate,
    so detector calibration and the fitted regulation/occupancy/fate workflow
    execute the same CLR, curve-statistic, and whole-donor residual machinery.
    """
    statistic = _normalize_statistic(statistic)
    tail = _normalize_tail(tail)
    calibration_scale = str(calibration_scale).lower().replace("-", "_")
    if calibration_scale not in {"studentized", "effect"}:
        raise ValueError("calibration_scale must be studentized or effect")
    calibration = _calibrate_component(
        np.asarray(scores, dtype=float),
        feature_names,
        component=str(component),
        axis_ids=np.asarray(axis_ids, dtype=int),
        axis_left=np.asarray(axis_left, dtype=float),
        axis_right=np.asarray(axis_right, dtype=float),
        reduced=np.asarray(reduced_design, dtype=float),
        condition=np.asarray(condition, dtype=bool),
        mappings=tuple(np.asarray(mapping, dtype=int) for mapping in null_mappings),
        statistic=statistic,
        tail=tail,
        calibration_scale=calibration_scale,
        return_null_statistics=bool(return_null_statistics),
    )
    return {
        "tests": calibration.tests,
        "curves": calibration.curves,
        "null_statistics": calibration.null_statistics,
        "observed_scaled": calibration.observed_scaled,
        "null_scaled": calibration.null_scaled,
    }


def calibrate_compositional_component_arrays_batch(
    scores: np.ndarray,
    feature_names: Sequence[str],
    *,
    component: str,
    axis_ids: Sequence[int],
    axis_left: Sequence[float],
    axis_right: Sequence[float],
    reduced_design: np.ndarray,
    condition: Sequence[bool],
    null_mappings: Sequence[np.ndarray],
    statistic: str = "max_absolute_effect",
    tail: str = "greater",
    calibration_scale: str = "studentized",
    alpha: float = 0.05,
    mapping_batch_size: int = 16,
) -> dict[str, np.ndarray]:
    """Batch the production component maxT kernel over simulation replicates."""
    values = np.asarray(scores, dtype=float)
    names = [str(value) for value in feature_names]
    axis = np.asarray(axis_ids, dtype=int)
    left = np.asarray(axis_left, dtype=float)
    right = np.asarray(axis_right, dtype=float)
    reduced = np.asarray(reduced_design, dtype=float)
    condition_array = np.asarray(condition, dtype=bool)
    mappings = tuple(np.asarray(value, dtype=int) for value in null_mappings)
    if values.ndim != 4 or values.shape[1:] != (
        len(condition_array),
        len(axis),
        len(names),
    ):
        raise ValueError(f"Invalid batched {component} response tensor shape")
    widths = right - left
    if (
        left.shape != axis.shape
        or right.shape != axis.shape
        or not np.isfinite(widths).all()
        or np.any(widths <= 0)
    ):
        raise ValueError(f"{component} curve widths must be finite and positive")
    plan = ArrayFreedmanLanePlan(
        reduced_design=reduced,
        condition=condition_array,
        widths=widths,
        null_mappings=mappings,
        residual_space_size=len(mappings) + 1,
        restricted_label_space_size=0,
        requested_mode="bound_component_stream",
        actual_mode="bound_component_stream",
        n_null_mappings=len(mappings),
        monte_carlo_p_resolution=1.0 / (len(mappings) + 1),
        seed=0,
        reference_enumeration="bound_component_stream",
        exactness_status="inherited_from_bound_component_stream",
        availability_mask_sha256="",
    )
    calibrated = run_array_freedman_lane_calibration_batch(
        values,
        plan,
        statistic=statistic,
        tail=tail,
        calibration_scale=calibration_scale,
        alpha=float(alpha),
        family_index=None,
        mapping_batch_size=int(mapping_batch_size),
    )
    return {
        "p_raw": calibrated["p_raw"],
        "p_component_maxT": calibrated["p_maxT"],
        "q_component_by": calibrated["q_by"],
        "component_maxT_reject": calibrated["p_maxT"] <= float(alpha),
        "beta_curve": calibrated["beta_curve"],
        "t_curve": calibrated["calibration_curve"],
        "standard_error_curve": calibrated["standard_error_curve"],
    }


def _cell_selection(
    adata,
    donor_design: pd.DataFrame,
    *,
    condition_key: str,
    donor_key: str,
    control,
    case,
    pseudotime_key: str,
    edges: np.ndarray,
    selected_bins: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    required = [condition_key, donor_key, pseudotime_key]
    missing = [key for key in required if key not in adata.obs]
    if missing:
        raise KeyError(f"Missing adata.obs columns: {missing}")
    frame = adata.obs.loc[:, required].copy()
    frame[condition_key] = frame[condition_key].astype(str)
    frame[donor_key] = frame[donor_key].astype(str)
    frame[pseudotime_key] = pd.to_numeric(frame[pseudotime_key], errors="coerce")
    allowed_donors = set(donor_design["donor"].astype(str))
    selected = (
        frame[condition_key].isin([str(control), str(case)])
        & frame[donor_key].isin(allowed_donors)
        & np.isfinite(frame[pseudotime_key].to_numpy(dtype=float))
    )
    bins = _assign_fixed_bins(frame[pseudotime_key].to_numpy(dtype=float), edges)
    selected &= np.isin(bins, selected_bins)
    donor_lookup = {
        value: index
        for index, value in enumerate(
            donor_design.sort_values("donor_index")["donor"].astype(str)
        )
    }
    donor_indices = np.full(len(frame), -1, dtype=int)
    selected_positions = np.flatnonzero(selected.to_numpy())
    donor_indices[selected_positions] = [
        donor_lookup[value]
        for value in frame.iloc[selected_positions][donor_key].astype(str)
    ]
    return frame, bins, donor_indices


def _occupancy_response(
    frame: pd.DataFrame,
    bins: np.ndarray,
    donor_indices: np.ndarray,
    donor_design: pd.DataFrame,
    *,
    selected_bins: np.ndarray,
    pseudocount: float,
    min_cells_per_donor: int,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    n_donors = len(donor_design)
    counts = np.zeros((n_donors, len(selected_bins)), dtype=int)
    bin_lookup = {int(value): index for index, value in enumerate(selected_bins)}
    valid = donor_indices >= 0
    for donor, bin_id in zip(donor_indices[valid], bins[valid]):
        counts[int(donor), bin_lookup[int(bin_id)]] += 1
    totals = counts.sum(axis=1)
    insufficient = np.flatnonzero(totals < min_cells_per_donor)
    if len(insufficient):
        names = donor_design.iloc[insufficient]["donor"].astype(str).tolist()
        raise CovariateDesignError(
            "State-occupancy denominators are below min_cells_per_donor for "
            f"donors {names}; donors are not silently dropped from joint inference"
        )
    response, raw, smoothed = occupancy_response_from_counts(
        counts,
        pseudocount=pseudocount,
        min_cells_per_donor=min_cells_per_donor,
    )
    rows = []
    for donor_index, donor_row in donor_design.sort_values("donor_index").iterrows():
        for local, bin_id in enumerate(selected_bins):
            rows.append(
                {
                    "donor": str(donor_row["donor"]),
                    "observed_condition": str(donor_row["observed_condition"]),
                    "permutation_block": str(donor_row["permutation_block"]),
                    "bin_id": int(bin_id),
                    "n_cells": int(counts[donor_index, local]),
                    "donor_total_cells_selected_grid": int(totals[donor_index]),
                    "observed_proportion": float(raw[donor_index, local]),
                    "smoothed_proportion": float(smoothed[donor_index, local]),
                    "clr_occupancy": float(response[donor_index, local, 0]),
                    "observed_zero_retained": bool(counts[donor_index, local] == 0),
                }
            )
    return response, pd.DataFrame(rows), raw, counts


def _validate_eligibility(values: pd.Series, key: str) -> np.ndarray:
    if values.isna().any():
        raise ValueError(f"fate_eligibility_key '{key}' contains missing values")
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy(dtype=bool)
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not np.isin(numeric, [0.0, 1.0]).all():
        raise ValueError(f"fate_eligibility_key '{key}' must be boolean or 0/1")
    return numeric.astype(bool)


def _fate_response(
    adata,
    frame: pd.DataFrame,
    donor_indices: np.ndarray,
    donor_design: pd.DataFrame,
    *,
    fate_key: Optional[str],
    fate_probability_keys: Optional[Mapping[str, str]],
    fate_eligibility_key: Optional[str],
    pseudocount: float,
    min_fate_cells_per_donor: int,
) -> tuple[Optional[np.ndarray], pd.DataFrame, list[str], Optional[np.ndarray]]:
    if fate_key is not None and fate_probability_keys:
        raise ValueError("fate_key and fate_probability_keys are mutually exclusive")
    if fate_key is None and not fate_probability_keys:
        return None, pd.DataFrame(), [], None
    selected = donor_indices >= 0
    if fate_eligibility_key is not None:
        if fate_eligibility_key not in adata.obs:
            raise KeyError(
                f"fate_eligibility_key '{fate_eligibility_key}' not found in adata.obs"
            )
        analysis_positions = np.flatnonzero(selected)
        eligibility = np.zeros(len(selected), dtype=bool)
        eligibility[analysis_positions] = _validate_eligibility(
            adata.obs.iloc[analysis_positions][fate_eligibility_key],
            fate_eligibility_key,
        )
        selected &= eligibility
    elif fate_key is not None:
        raise ValueError(
            "Hard terminal fate selection requires an externally fixed "
            "fate_eligibility_key; pooled-data adaptive terminal selection is forbidden"
        )
    positions = np.flatnonzero(selected)
    n_donors = len(donor_design)
    denominators = np.bincount(
        donor_indices[positions], minlength=n_donors
    ).astype(int)
    insufficient = np.flatnonzero(denominators < min_fate_cells_per_donor)
    if len(insufficient):
        names = donor_design.iloc[insufficient]["donor"].astype(str).tolist()
        raise CovariateDesignError(
            "Fate denominators are below min_fate_cells_per_donor for donors "
            f"{names}; donors are not silently dropped from joint inference"
        )

    if fate_key is not None:
        if fate_key not in adata.obs:
            raise KeyError(f"fate_key '{fate_key}' not found in adata.obs")
        values = adata.obs.iloc[positions][fate_key]
        if values.isna().any():
            raise ValueError("Eligible hard-fate cells contain missing fate labels")
        _validate_stringification_is_injective(values, fate_key)
        labels = values.astype(str).to_numpy()
        names = sorted(pd.unique(labels).tolist())
        if len(names) < 2:
            raise CovariateDesignError("Hard fate selection requires at least two fates")
        name_to_index = {name: index for index, name in enumerate(names)}
        counts = np.zeros((n_donors, len(names)), dtype=float)
        for donor, label in zip(donor_indices[positions], labels):
            counts[int(donor), name_to_index[str(label)]] += 1.0
        source = "hard_terminal_labels"
    else:
        mapping = {str(name): str(key) for name, key in fate_probability_keys.items()}
        names = list(mapping)
        if len(names) < 2 or len(names) != len(set(names)):
            raise ValueError("fate_probability_keys must define at least two unique fates")
        if len(set(mapping.values())) != len(mapping):
            raise ValueError(
                "Each fate in fate_probability_keys must use a distinct source column"
            )
        missing = [key for key in mapping.values() if key not in adata.obs]
        if missing:
            raise KeyError(f"Missing fate probability columns: {missing}")
        probabilities = adata.obs.iloc[positions][list(mapping.values())].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(probabilities).all():
            raise ValueError("Fate probabilities must be finite")
        if np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ValueError("Fate probabilities must lie in [0, 1]")
        row_sums = probabilities.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6, rtol=1e-6):
            raise ValueError("Fate probabilities must sum to one for every eligible cell")
        counts = np.zeros((n_donors, len(names)), dtype=float)
        for donor_index in range(n_donors):
            mask = donor_indices[positions] == donor_index
            counts[donor_index] = probabilities[mask].sum(axis=0)
        source = "soft_lineage_probabilities"

    response, raw, smoothed = fate_response_from_masses(
        counts,
        denominators,
        pseudocount=pseudocount,
        min_fate_cells_per_donor=min_fate_cells_per_donor,
    )
    rows = []
    for donor_index, donor_row in donor_design.sort_values("donor_index").iterrows():
        for fate_index, fate in enumerate(names):
            rows.append(
                {
                    "donor": str(donor_row["donor"]),
                    "observed_condition": str(donor_row["observed_condition"]),
                    "permutation_block": str(donor_row["permutation_block"]),
                    "fate": fate,
                    "fate_source": source,
                    "fate_mass": float(counts[donor_index, fate_index]),
                    "eligible_cell_denominator": int(denominators[donor_index]),
                    "observed_proportion": float(raw[donor_index, fate_index]),
                    "smoothed_proportion": float(smoothed[donor_index, fate_index]),
                    "clr_fate": float(response[donor_index, 0, fate_index]),
                    "observed_zero_retained": bool(counts[donor_index, fate_index] == 0),
                }
            )
    return response, pd.DataFrame(rows), names, raw


def _attach_probability_curves(
    curves: pd.DataFrame,
    *,
    raw_proportions: np.ndarray,
    condition: np.ndarray,
) -> pd.DataFrame:
    result = curves.copy()
    features = result["feature"].drop_duplicates().tolist()
    axes = result["axis_id"].drop_duplicates().tolist()
    control_clr = (
        result.pivot(index="axis_id", columns="feature", values="adjusted_control_clr")
        .reindex(index=axes, columns=features)
        .to_numpy(dtype=float)
    )
    case_clr = (
        result.pivot(index="axis_id", columns="feature", values="adjusted_case_clr")
        .reindex(index=axes, columns=features)
        .to_numpy(dtype=float)
    )
    if len(features) == 1:
        # One feature spread across pseudotime bins is the composition.  This
        # inverse-CLR is an Aitchison compositional center, not E[P | C].
        control_probability = _softmax_rows(control_clr[:, 0][None, :]).ravel()[:, None]
        case_probability = _softmax_rows(case_clr[:, 0][None, :]).ravel()[:, None]
        raw_control = raw_proportions[~condition].mean(axis=0)[:, None]
        raw_case = raw_proportions[condition].mean(axis=0)[:, None]
    else:
        control_probability = _softmax_rows(control_clr)
        case_probability = _softmax_rows(case_clr)
        raw_control = raw_proportions[~condition].mean(axis=0)[None, :]
        raw_case = raw_proportions[condition].mean(axis=0)[None, :]
    lookup = {}
    for axis_index, axis in enumerate(axes):
        for feature_index, feature in enumerate(features):
            lookup[(axis, feature)] = (
                control_probability[axis_index, feature_index],
                case_probability[axis_index, feature_index],
                raw_control[axis_index, feature_index],
                raw_case[axis_index, feature_index],
            )
    adjusted_control = []
    adjusted_case = []
    unadjusted_control = []
    unadjusted_case = []
    for row in result.itertuples(index=False):
        values = lookup[(row.axis_id, row.feature)]
        adjusted_control.append(values[0])
        adjusted_case.append(values[1])
        unadjusted_control.append(values[2])
        unadjusted_case.append(values[3])
    result["adjusted_control_compositional_center"] = adjusted_control
    result["adjusted_case_compositional_center"] = adjusted_case
    result["delta_compositional_center"] = (
        result["adjusted_case_compositional_center"]
        - result["adjusted_control_compositional_center"]
    )
    result["unadjusted_control_mean_probability"] = unadjusted_control
    result["unadjusted_case_mean_probability"] = unadjusted_case
    result["unadjusted_delta_mean_probability"] = (
        result["unadjusted_case_mean_probability"]
        - result["unadjusted_control_mean_probability"]
    )
    return result


def run_regulation_occupancy_fate_decomposition(
    adata,
    gene_sets,
    *,
    condition_key: str,
    donor_key: str,
    control,
    case,
    pseudotime_key: str = "dpt_pseudotime",
    regulation_branch_key: Optional[str] = None,
    regulation_branch=None,
    fate_key: Optional[str] = None,
    fate_probability_keys: Optional[Mapping[str, str]] = None,
    fate_eligibility_key: Optional[str] = None,
    continuous_covariate_keys: Sequence[str] = (),
    categorical_covariate_keys: Sequence[str] = (),
    strata_keys: Sequence[str] = (),
    grid_edges: Optional[Sequence[float]] = None,
    n_bins: int = 8,
    pseudotime_range: Tuple[float, float] = (0.0, 1.0),
    min_cells_per_donor_bin: int = 5,
    min_cells_per_donor: int = 1,
    min_fate_cells_per_donor: int = 1,
    min_donors_per_condition: int = 3,
    min_common_bins: Optional[int] = None,
    min_residual_df: int = 3,
    max_condition_vif: float = 10.0,
    occupancy_pseudocount: float = 0.5,
    fate_pseudocount: float = 0.5,
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
) -> RegulationOccupancyFateResult:
    """Separate conditional regulation, state occupancy, and fate selection.

    Conditional regulation is the adjusted pathway activity difference at a
    fixed pseudotime bin.  Occupancy is the donor's compositional cell
    distribution over the same outcome-blind contiguous segment of the
    complete predeclared source grid.  Fate selection is a
    donor-level terminal-label composition or mean soft lineage probability.
    All components use equal donor weight and the same whole-donor residual
    mappings.  Observed zeros in occupancy/fate counts remain zero rather than
    becoming missing donor-bins.

    The output is a parallel association decomposition, not causal mediation.
    It is conditional on the supplied pseudotime and supplied branch/fate
    alignment.  Hard fates require an externally fixed eligibility mask.
    """
    statistic = _normalize_statistic(statistic)
    tail = _normalize_tail(tail)
    _test_scale(np.asarray([0.0]), statistic, tail)
    calibration_scale = str(calibration_scale).lower().replace("-", "_")
    if calibration_scale not in {"studentized", "effect"}:
        raise ValueError("calibration_scale must be 'studentized' or 'effect'")
    if calibration_scale != "studentized":
        raise ValueError(
            "Joint maxT across pathway-activity and compositional CLR responses "
            "requires calibration_scale='studentized'; effect-scale component "
            "statistics have different units"
        )
    min_cells_per_donor = _require_integer(
        min_cells_per_donor, "min_cells_per_donor"
    )
    min_fate_cells_per_donor = _require_integer(
        min_fate_cells_per_donor, "min_fate_cells_per_donor"
    )
    for name, value in {
        "occupancy_pseudocount": occupancy_pseudocount,
        "fate_pseudocount": fate_pseudocount,
    }.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    continuous_covariate_keys = _as_key_tuple(
        continuous_covariate_keys, "continuous_covariate_keys"
    )
    categorical_covariate_keys = _as_key_tuple(
        categorical_covariate_keys, "categorical_covariate_keys"
    )
    strata_keys = _as_key_tuple(strata_keys, "strata_keys")
    if (regulation_branch_key is None) != (regulation_branch is None):
        raise ValueError(
            "regulation_branch_key and regulation_branch must be supplied together"
        )
    if fate_eligibility_key is not None and fate_key is None and not fate_probability_keys:
        raise ValueError(
            "fate_eligibility_key was supplied but no fate response was requested"
        )
    declared_edges = _fixed_edges(grid_edges, n_bins, pseudotime_range)
    declared_n_bins = len(declared_edges) - 1
    if declared_n_bins < 2:
        raise ValueError("Occupancy CLR inference requires at least two fixed bins")
    effective_min_common_bins = 2 if min_common_bins is None else int(min_common_bins)
    if not 2 <= effective_min_common_bins <= declared_n_bins:
        raise ValueError(
            "min_common_bins must be between 2 and the number of fixed source bins"
        )

    complete_cohort_donors: Optional[set[str]] = None
    regulation_adata = adata
    if regulation_branch_key is not None:
        if regulation_branch_key not in adata.obs:
            raise KeyError(
                f"regulation_branch_key '{regulation_branch_key}' not found in adata.obs"
            )
        complete_cohort_donors = _validate_complete_cohort_donor_design(
            adata,
            condition_key=condition_key,
            donor_key=donor_key,
            control=control,
            case=case,
            continuous_covariate_keys=continuous_covariate_keys,
            categorical_covariate_keys=categorical_covariate_keys,
            strata_keys=strata_keys,
        )
        condition_selected = adata.obs[condition_key].astype(str).isin(
            [str(control), str(case)]
        )
        branch_values = adata.obs.loc[condition_selected, regulation_branch_key]
        if branch_values.isna().any():
            raise ValueError(
                f"regulation_branch_key '{regulation_branch_key}' contains missing values"
            )
        _validate_stringification_is_injective(
            branch_values, regulation_branch_key
        )
        branch_mask = adata.obs[regulation_branch_key].astype(str).eq(
            str(regulation_branch)
        )
        if not (condition_selected & branch_mask).any():
            raise ValueError("No analysis cells match regulation_branch")
        regulation_adata = adata[branch_mask.to_numpy()]

    regulation = run_covariate_adjusted_donor_pseudobulk(
        regulation_adata,
        gene_sets,
        condition_key=condition_key,
        donor_key=donor_key,
        control=control,
        case=case,
        pseudotime_key=pseudotime_key,
        continuous_covariate_keys=continuous_covariate_keys,
        categorical_covariate_keys=categorical_covariate_keys,
        strata_keys=strata_keys,
        grid_edges=declared_edges,
        n_bins=n_bins,
        pseudotime_range=pseudotime_range,
        min_cells_per_donor_bin=min_cells_per_donor_bin,
        min_donors_per_condition=min_donors_per_condition,
        min_common_bins=effective_min_common_bins,
        min_residual_df=min_residual_df,
        max_condition_vif=max_condition_vif,
        statistic=statistic,
        tail=tail,
        calibration_scale=calibration_scale,
        permutation_mode=permutation_mode,
        n_permutations=n_permutations,
        max_exact_permutations=max_exact_permutations,
        min_size=min_size,
        max_size=max_size,
        layer=layer,
        use_raw=use_raw,
        alpha=alpha,
        power_target=power_target,
        seed=seed,
        return_null_statistics=False,
        return_permutation_assignments=True,
        return_donor_bin_activity=True,
    )
    donor_design = _formal_inference_donor_design_view(regulation.donor_design)
    if complete_cohort_donors is None:
        requested_cells = adata.obs[condition_key].astype(str).isin(
            [str(control), str(case)]
        )
        requested_donors = set(
            adata.obs.loc[requested_cells, donor_key].astype(str).unique()
        )
    else:
        requested_donors = complete_cohort_donors
    included_donors = set(donor_design["donor"].astype(str))
    if included_donors != requested_donors:
        missing_donors = sorted(requested_donors - included_donors)
        raise CovariateDesignError(
            "The joint decomposition requires a predeclared shared donor cohort; "
            "regulation support excluded donors " + str(missing_donors)
        )
    condition = donor_design["observed_case"].to_numpy(dtype=bool)
    encoded = _reduced_design(
        donor_design,
        continuous_covariate_keys=continuous_covariate_keys,
        categorical_covariate_keys=categorical_covariate_keys,
        strata_keys=strata_keys,
    )
    groups = _groups_from_design(donor_design)
    plan = _make_residual_plan(
        len(donor_design),
        groups,
        permutation_mode=permutation_mode,
        n_permutations=n_permutations,
        max_exact_permutations=max_exact_permutations,
        seed=seed,
    )
    expected_mapping_hashes = [
        _mapping_hash(mapping) for mapping in plan.null_mappings
    ]
    recorded_mapping_hashes = (
        regulation.permutation_assignments.loc[
            ~regulation.permutation_assignments["is_identity_mapping"],
            ["perm_id", "mapping_hash"],
        ]
        .drop_duplicates()
        .sort_values("perm_id")["mapping_hash"]
        .astype(str)
        .tolist()
    )
    if recorded_mapping_hashes != expected_mapping_hashes:
        raise RuntimeError(
            "Joint calibration mappings do not match the audited regulation stream"
        )
    edges = np.asarray(regulation.metadata["grid_edges"], dtype=float)
    selected_bins = np.asarray(regulation.metadata["selected_bin_ids"], dtype=int)
    if (
        len(selected_bins) < effective_min_common_bins
        or np.any(np.diff(selected_bins) != 1)
        or np.any(selected_bins < 0)
        or np.any(selected_bins >= declared_n_bins)
    ):
        raise CovariateDesignError(
            "The regulation model did not retain a valid outcome-blind contiguous "
            "segment of the fixed source grid"
        )
    selected_left = edges[selected_bins]
    selected_right = edges[selected_bins + 1]

    frame, cell_bins, donor_indices = _cell_selection(
        adata,
        donor_design,
        condition_key=condition_key,
        donor_key=donor_key,
        control=control,
        case=case,
        pseudotime_key=pseudotime_key,
        edges=edges,
        selected_bins=selected_bins,
    )
    occupancy_scores, donor_state_counts, raw_occupancy, occupancy_counts = (
        _occupancy_response(
            frame,
            cell_bins,
            donor_indices,
            donor_design,
            selected_bins=selected_bins,
            pseudocount=float(occupancy_pseudocount),
            min_cells_per_donor=min_cells_per_donor,
        )
    )
    fate_scores, donor_fate_counts, fate_names, raw_fate = _fate_response(
        adata,
        frame,
        donor_indices,
        donor_design,
        fate_key=fate_key,
        fate_probability_keys=fate_probability_keys,
        fate_eligibility_key=fate_eligibility_key,
        pseudocount=float(fate_pseudocount),
        min_fate_cells_per_donor=min_fate_cells_per_donor,
    )
    occupancy_denominators = (
        donor_state_counts[["donor", "donor_total_cells_selected_grid"]]
        .drop_duplicates()
        .set_index("donor")
        .reindex(donor_design["donor"].astype(str))[
            "donor_total_cells_selected_grid"
        ]
        .to_numpy(dtype=int)
    )
    denominator_arrays = [occupancy_denominators]
    if not donor_fate_counts.empty:
        fate_denominators = (
            donor_fate_counts[["donor", "eligible_cell_denominator"]]
            .drop_duplicates()
            .set_index("donor")
            .reindex(donor_design["donor"].astype(str))["eligible_cell_denominator"]
            .to_numpy(dtype=int)
        )
        denominator_arrays.append(fate_denominators)
    denominator_block_constant = all(
        all(len(np.unique(values[group])) <= 1 for group in groups)
        for values in denominator_arrays
    )
    if (
        str(permutation_mode).lower().replace("-", "_") == "exact"
        and not denominator_block_constant
    ):
        raise CovariateDesignError(
            "permutation_mode='exact' is unavailable because occupancy/fate "
            "donor denominators vary within a residual-mapping block. Use "
            "'exhaustive' only as an explicitly approximate sensitivity analysis."
        )
    regulation_scores, pathway_names, regulation_bins = _activity_tensor(regulation)

    regulation_calibration = _calibrate_component(
        regulation_scores,
        pathway_names,
        component="conditional_regulation",
        axis_ids=regulation_bins,
        axis_left=selected_left,
        axis_right=selected_right,
        reduced=encoded.reduced,
        condition=condition,
        mappings=plan.null_mappings,
        statistic=statistic,
        tail=tail,
        calibration_scale=calibration_scale,
        return_null_statistics=return_null_statistics,
    )
    occupancy_calibration = _calibrate_component(
        occupancy_scores,
        ["state_occupancy"],
        component="state_occupancy",
        axis_ids=selected_bins,
        axis_left=selected_left,
        axis_right=selected_right,
        reduced=encoded.reduced,
        condition=condition,
        mappings=plan.null_mappings,
        statistic=statistic,
        tail=tail,
        calibration_scale=calibration_scale,
        return_null_statistics=return_null_statistics,
    )
    occupancy_curves = _attach_probability_curves(
        occupancy_calibration.curves,
        raw_proportions=raw_occupancy,
        condition=condition,
    )
    calibrations = [regulation_calibration, occupancy_calibration]
    if fate_scores is not None:
        fate_calibration = _calibrate_component(
            fate_scores,
            fate_names,
            component="fate_selection",
            axis_ids=np.asarray([0], dtype=int),
            axis_left=np.asarray([0.0]),
            axis_right=np.asarray([1.0]),
            reduced=encoded.reduced,
            condition=condition,
            mappings=plan.null_mappings,
            statistic=statistic,
            tail=tail,
            calibration_scale=calibration_scale,
            return_null_statistics=return_null_statistics,
        )
        fate_effects = _attach_probability_curves(
            fate_calibration.curves,
            raw_proportions=raw_fate,
            condition=condition,
        )
        calibrations.append(fate_calibration)
    else:
        fate_calibration = None
        fate_effects = pd.DataFrame(
            columns=[
                "component",
                "feature",
                "axis_id",
                "axis_left",
                "axis_right",
                "axis_mid",
                "beta_clr",
                "standard_error_clr",
                "t_value",
                "residual_sd_clr",
                "adjusted_control_clr",
                "adjusted_case_clr",
                "adjusted_control_compositional_center",
                "adjusted_case_compositional_center",
                "delta_compositional_center",
                "unadjusted_control_mean_probability",
                "unadjusted_case_mean_probability",
                "unadjusted_delta_mean_probability",
            ]
        )

    all_observed = np.concatenate([item.observed_scaled for item in calibrations])
    all_null = np.column_stack([item.null_scaled for item in calibrations])
    joint_null_max = np.max(all_null, axis=1)
    denominator = len(plan.null_mappings) + 1
    p_joint = (
        (joint_null_max[:, None] >= all_observed[None, :] - 1e-12).sum(axis=0)
        + 1.0
    ) / denominator
    all_tests = pd.concat([item.tests for item in calibrations], ignore_index=True)
    all_tests["p_joint_maxT"] = p_joint
    all_tests["q_joint_bh"] = _bh_adjust(all_tests["p_raw"].to_numpy(dtype=float))
    all_tests["q_joint_by"] = _by_adjust(all_tests["p_raw"].to_numpy(dtype=float))

    regulation_tests = all_tests[
        all_tests["component"].eq("conditional_regulation")
    ].reset_index(drop=True)
    occupancy_tests = all_tests[
        all_tests["component"].eq("state_occupancy")
    ].reset_index(drop=True)
    fate_tests = all_tests[
        all_tests["component"].eq("fate_selection")
    ].reset_index(drop=True)
    regulation_curves = regulation_calibration.curves.rename(
        columns={
            "beta_clr": "beta_activity",
            "standard_error_clr": "standard_error_activity",
            "residual_sd_clr": "residual_sd_activity",
            "adjusted_control_clr": "adjusted_control_activity",
            "adjusted_case_clr": "adjusted_case_activity",
        }
    )

    component_summary_rows = []
    for component, tests in (
        ("conditional_regulation", regulation_tests),
        ("state_occupancy", occupancy_tests),
        ("fate_selection", fate_tests),
    ):
        component_summary_rows.append(
            {
                "component": component,
                "status": "not_requested" if tests.empty else "fitted",
                "n_tests": int(len(tests)),
                "n_raw_rejections": int((tests["p_raw"] <= alpha).sum()) if len(tests) else 0,
                "n_component_maxT_rejections": int(
                    (tests["p_component_maxT"] <= alpha).sum()
                ) if len(tests) else 0,
                "n_joint_maxT_rejections": int(
                    (tests["p_joint_maxT"] <= alpha).sum()
                ) if len(tests) else 0,
                "interpretation_scope": (
                    "parallel_donor_level_association_not_causal_mediation"
                ),
            }
        )
    component_summary = pd.DataFrame(component_summary_rows)

    totals = occupancy_counts.sum(axis=1)
    diagnostics_rows = [
        {
            "component": "state_occupancy",
            "diagnostic": "observed_zero_donor_bins",
            "value": float((occupancy_counts == 0).sum()),
            "status": "retained_as_zero",
            "detail": "observed zero donor-bin counts are outcomes, not missingness",
        },
        {
            "component": "state_occupancy",
            "diagnostic": "case_to_control_mean_cell_yield_ratio",
            "value": float(totals[condition].mean() / totals[~condition].mean()),
            "status": "descriptive",
            "detail": "donors remain equally weighted in inference",
        },
        {
            "component": "joint",
            "diagnostic": "joint_test_count",
            "value": float(len(all_tests)),
            "status": "joint_maxT_reference",
            "detail": "regulation pathways plus occupancy plus requested fates",
        },
    ]
    if not donor_fate_counts.empty:
        diagnostics_rows.append(
            {
                "component": "fate_selection",
                "diagnostic": "observed_zero_donor_fates",
                "value": float(donor_fate_counts["observed_zero_retained"].sum()),
                "status": "retained_as_zero",
                "detail": "zero fate mass is not missingness",
            }
        )
    component_diagnostics = pd.DataFrame(diagnostics_rows)

    null_statistics = (
        pd.concat(
            [item.null_statistics for item in calibrations], ignore_index=True
        )
        if return_null_statistics
        else pd.DataFrame(
            columns=[
                "perm_id",
                "component",
                "feature",
                "effect_statistic",
                "raw_calibration_statistic",
                "calibration_statistic",
                "component_max_calibration_statistic",
            ]
        )
    )
    if return_null_statistics:
        joint = pd.DataFrame(
            {
                "perm_id": np.arange(len(plan.null_mappings), dtype=int),
                "joint_max_calibration_statistic": joint_null_max,
            }
        )
        null_statistics = null_statistics.merge(joint, on="perm_id", how="left")

    available = np.ones((len(donor_design), len(selected_bins)), dtype=bool)
    block_constant = _block_constant_reduced_design(
        encoded.reduced,
        available,
        groups,
        np.arange(len(selected_bins), dtype=int),
    )
    exactness = _exactness_status(
        block_constant,
        denominator_block_constant,
        plan.is_exhaustive,
    )
    permutation_summary = regulation.permutation_summary.copy()
    permutation_summary["calibration_method"] = (
        "joint_freedman_lane_whole_donor_residual_tensor"
    )
    permutation_summary["joint_components"] = (
        "conditional_regulation|state_occupancy"
        + ("|fate_selection" if fate_scores is not None else "")
    )
    permutation_summary["n_joint_tests"] = int(len(all_tests))
    permutation_summary["joint_maxT"] = True
    permutation_summary["joint_exactness_status"] = exactness
    permutation_summary["joint_residual_exchangeability_required"] = True
    permutation_summary["denominator_block_constant"] = bool(
        denominator_block_constant
    )
    permutation_summary["strong_fwer_requires_subset_pivotality"] = True
    assignments = regulation.permutation_assignments.copy()
    assignments["component_scope"] = "shared_across_all_components"

    metadata = {
        "method": "donor_level_regulation_occupancy_fate_decomposition",
        "estimands": {
            "conditional_regulation": "adjusted pathway activity at fixed pseudotime",
            "state_occupancy": (
                "adjusted CLR contrast over the complete predeclared grid; "
                "unadjusted arithmetic mean probabilities also reported"
            ),
            "fate_selection": "donor terminal/lineage composition",
        },
        "interpretation_scope": "parallel_associations_not_causal_mediation",
        "condition_key": condition_key,
        "donor_key": donor_key,
        "control": str(control),
        "case": str(case),
        "pseudotime_key": pseudotime_key,
        "regulation_branch_key": regulation_branch_key,
        "regulation_branch": (
            None if regulation_branch is None else str(regulation_branch)
        ),
        "selected_bin_ids": selected_bins.tolist(),
        "grid_edges": edges.tolist(),
        "occupancy_transform": "centered_log_ratio_with_additive_pseudocount",
        "occupancy_pseudocount": float(occupancy_pseudocount),
        "occupancy_zero_policy": "observed_zero_retained_not_missing",
        "probability_backtransform_scope": (
            "adjusted inverse-CLR values are compositional centers, not E[P|C]"
        ),
        "fate_transform": (
            "not_requested"
            if fate_scores is None
            else "centered_log_ratio_with_additive_pseudocount"
        ),
        "fate_pseudocount": float(fate_pseudocount),
        "fate_key": fate_key,
        "fate_probability_keys": dict(fate_probability_keys or {}),
        "fate_eligibility_key": fate_eligibility_key,
        "hard_fate_requires_external_fixed_eligibility": True,
        "continuous_covariate_keys": list(continuous_covariate_keys),
        "categorical_covariate_keys": list(categorical_covariate_keys),
        "strata_keys": list(strata_keys),
        "statistic": statistic,
        "tail": tail,
        "calibration_scale": calibration_scale,
        "joint_maxT": True,
        "joint_exactness_status": exactness,
        "denominator_block_constant": bool(denominator_block_constant),
        "joint_exactness_condition": (
            "reduced design block-invariant and the joint regulation/occupancy/"
            "fate donor residual tensor exchangeable within declared blocks"
        ),
        "branch_alignment_scope": "conditional_on_supplied_branch_and_pseudotime_alignment",
        "strong_fwer_condition": (
            "joint maxT strong FWER additionally relies on subset pivotality"
        ),
        "n_joint_tests": int(len(all_tests)),
        "residual_permutation_space_size": int(plan.residual_space_size),
        "n_null_mappings_evaluated": int(len(plan.null_mappings)),
        "seed": int(seed),
    }
    for table in (
        regulation_tests,
        regulation_curves,
        occupancy_tests,
        occupancy_curves,
        fate_tests,
        fate_effects,
        donor_state_counts,
        donor_fate_counts,
        component_summary,
        component_diagnostics,
        permutation_summary,
        assignments,
        null_statistics,
    ):
        table.attrs["regulation_occupancy_fate_decomposition"] = metadata.copy()
    return RegulationOccupancyFateResult(
        regulation_tests=regulation_tests,
        regulation_curves=regulation_curves,
        occupancy_tests=occupancy_tests,
        occupancy_curves=occupancy_curves,
        fate_tests=fate_tests,
        fate_effects=fate_effects,
        donor_state_counts=donor_state_counts,
        donor_fate_counts=donor_fate_counts,
        component_summary=component_summary,
        component_diagnostics=component_diagnostics,
        donor_design=donor_design.copy(),
        design_diagnostics=regulation.design_diagnostics.copy(),
        permutation_summary=permutation_summary,
        permutation_assignments=assignments,
        null_statistics=null_statistics,
        regulation_result=regulation,
        metadata=metadata,
    )
