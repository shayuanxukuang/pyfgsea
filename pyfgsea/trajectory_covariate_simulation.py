from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .trajectory_covariate_pseudobulk import (
    CovariateAdjustedDonorPseudobulkResult,
    CovariateDesignError,
    _block_constant_reduced_design,
    _by_adjust,
    _curve_statistics,
    _encode_reduced_design,
    _fit_observed_models,
    _fit_permuted_curves,
    _make_residual_plan,
    _normalize_statistic,
    _normalize_tail,
    _require_integer,
    _space_sizes,
    _test_scale,
)


@dataclass
class CovariateAdjustedDesignSimulationResult:
    """Design-matched donor-curve calibration and power summaries."""

    replicate_metrics: pd.DataFrame
    simulation_metrics: pd.DataFrame
    summary: pd.DataFrame
    design_diagnostics: pd.DataFrame
    metadata: Dict[str, Any]

    def to_tables(self) -> Dict[str, pd.DataFrame]:
        return {
            "replicate_metrics": self.replicate_metrics.copy(),
            "simulation_metrics": self.simulation_metrics.copy(),
            "summary": self.summary.copy(),
            "design_diagnostics": self.design_diagnostics.copy(),
        }


@dataclass(frozen=True)
class ArrayFreedmanLanePlan:
    """Frozen residual-curve reference for array-level design calibration."""

    reduced_design: np.ndarray
    condition: np.ndarray
    widths: np.ndarray
    null_mappings: tuple[np.ndarray, ...]
    residual_space_size: int
    restricted_label_space_size: int
    requested_mode: str
    actual_mode: str
    n_null_mappings: int
    monte_carlo_p_resolution: float
    seed: int
    reference_enumeration: str
    exactness_status: str
    availability_mask_sha256: str


def _shared_curve_calibration_kernel(
    scores: np.ndarray,
    *,
    reduced_design: np.ndarray,
    condition: np.ndarray,
    widths: np.ndarray,
    null_mappings: Sequence[np.ndarray],
    statistic: str,
    tail: str,
    calibration_scale: str,
    alpha: float,
    family_index: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Calibrate one donor-curve array through the formal shared FL kernel."""
    (
        fits,
        beta_curve,
        t_curve,
        se_curve,
        _residual_sd_curve,
        control_curve,
        case_curve,
    ) = _fit_observed_models(scores, reduced_design, condition)
    calibration_curve = t_curve if calibration_scale == "studentized" else beta_curve
    if not np.isfinite(calibration_curve).all():
        raise CovariateDesignError(
            "A studentized array statistic is undefined under the observed design"
        )
    effect_statistics = _curve_statistics(beta_curve, widths)[statistic]
    raw_calibration = _curve_statistics(calibration_curve, widths)[statistic]
    observed = _test_scale(raw_calibration, statistic, tail)
    exceed = np.zeros(scores.shape[2], dtype=int)
    max_exceed = np.zeros(scores.shape[2], dtype=int)
    families = None
    within_family_exceed = np.zeros(scores.shape[2], dtype=int)
    family_gate_exceed = np.zeros(scores.shape[2], dtype=int)
    if family_index is not None:
        families = np.asarray(family_index, dtype=int)
        if families.shape != (scores.shape[2],) or np.any(families < -1):
            raise ValueError("family_index must use -1 or one non-negative code per pathway")
        assigned_families = np.unique(families[families >= 0])
        if not len(assigned_families):
            raise ValueError("family_index must assign at least one pathway")
        observed_family_max = {
            int(family): float(np.max(observed[families == family]))
            for family in assigned_families
        }
    for mapping in null_mappings:
        null_beta, null_t = _fit_permuted_curves(fits, mapping)
        null_curve = null_t if calibration_scale == "studentized" else null_beta
        if not np.isfinite(null_curve).all():
            raise CovariateDesignError(
                "A studentized array statistic is undefined under a residual mapping"
            )
        null_raw = _curve_statistics(null_curve, widths)[statistic]
        null_stat = _test_scale(null_raw, statistic, tail)
        exceed += null_stat >= observed - 1e-12
        max_exceed += float(np.max(null_stat)) >= observed - 1e-12
        if families is not None:
            null_family_max = {
                int(family): float(np.max(null_stat[families == family]))
                for family in assigned_families
            }
            global_family_max = max(null_family_max.values())
            for family in assigned_families:
                members = families == family
                within_family_exceed[members] += (
                    null_family_max[int(family)] >= observed[members] - 1e-12
                )
                family_gate_exceed[members] += (
                    global_family_max
                    >= observed_family_max[int(family)] - 1e-12
                )
    denominator = len(null_mappings) + 1
    p_raw = (exceed + 1.0) / denominator
    p_max_t = (max_exceed + 1.0) / denominator
    q_by = _by_adjust(p_raw)
    if families is None:
        p_within_family = np.full(scores.shape[2], np.nan, dtype=float)
        p_family_gate = np.full(scores.shape[2], np.nan, dtype=float)
    else:
        p_within_family = (within_family_exceed + 1.0) / denominator
        p_family_gate = (family_gate_exceed + 1.0) / denominator
        p_within_family[families < 0] = 1.0
        p_family_gate[families < 0] = 1.0
    hierarchical_reject = np.isfinite(p_within_family) & (p_within_family <= alpha) & (
        p_family_gate <= alpha
    )
    production_hierarchical_reject = np.isfinite(p_family_gate) & (
        p_max_t <= alpha
    ) & (p_family_gate <= alpha)
    return {
        "beta_curve": beta_curve,
        "standard_error_curve": se_curve,
        "adjusted_control_curve": control_curve,
        "adjusted_case_curve": case_curve,
        "residual_df_by_bin": np.asarray(
            [fit.residual_df for fit in fits], dtype=int
        ),
        "calibration_curve": calibration_curve,
        "effect_statistic": effect_statistics,
        "calibration_statistic": raw_calibration,
        "p_raw": p_raw,
        "p_maxT": p_max_t,
        "q_by": q_by,
        "raw_reject": p_raw <= alpha,
        "maxT_reject": p_max_t <= alpha,
        "by_reject": q_by <= alpha,
        "p_pathway_within_family_maxT": p_within_family,
        "p_family_gate": p_family_gate,
        "within_family_maxT_reject": p_within_family <= alpha,
        "family_gate_reject": p_family_gate <= alpha,
        "family_hierarchical_reject": hierarchical_reject,
        "production_family_hierarchical_reject": production_hierarchical_reject,
    }


def make_array_freedman_lane_plan(
    *,
    reduced_design: np.ndarray,
    condition: np.ndarray,
    available: np.ndarray,
    widths: np.ndarray,
    max_exact_permutations: int = 20_000,
    permutation_mode: str = "auto",
    n_permutations: int = 999,
    seed: int = 42,
) -> ArrayFreedmanLanePlan:
    """Build an exhaustive availability-restricted residual-curve plan."""
    import hashlib

    reduced = np.asarray(reduced_design, dtype=float)
    condition_array = np.asarray(condition, dtype=bool)
    available_array = np.asarray(available, dtype=bool)
    widths_array = np.asarray(widths, dtype=float)
    if (
        reduced.ndim != 2
        or available_array.ndim != 2
        or reduced.shape[0] != len(condition_array)
        or available_array.shape[0] != len(condition_array)
        or widths_array.shape != (available_array.shape[1],)
    ):
        raise ValueError("Array Freedman-Lane design arrays are not aligned")
    if np.any(available_array.sum(axis=0) <= reduced.shape[1] + 1):
        raise CovariateDesignError(
            "At least one fixed-grid bin lacks residual degrees of freedom"
        )
    lookup: dict[bytes, list[int]] = {}
    for index, row in enumerate(available_array):
        lookup.setdefault(row.tobytes(), []).append(index)
    groups = [
        np.asarray(indices, dtype=int)
        for _, indices in sorted(lookup.items(), key=lambda item: item[0])
    ]
    residual_space, restricted_label_space = _space_sizes(condition_array, groups)
    plan = _make_residual_plan(
        len(condition_array),
        groups,
        permutation_mode=permutation_mode,
        n_permutations=int(n_permutations),
        max_exact_permutations=max_exact_permutations,
        seed=seed,
    )
    block_constant = _block_constant_reduced_design(
        reduced,
        available_array,
        groups,
        np.arange(available_array.shape[1], dtype=int),
    )
    exactness_status = (
        "finite_sample_exact_residual_group_conditional_on_invariance"
        if block_constant and plan.is_exhaustive
        else (
            "exhaustive_freedman_lane_reference_not_finite_sample_exact"
            if plan.is_exhaustive
            else "monte_carlo_freedman_lane_reference_not_finite_sample_exact"
        )
    )
    return ArrayFreedmanLanePlan(
        reduced_design=reduced,
        condition=condition_array,
        widths=widths_array,
        null_mappings=tuple(np.asarray(mapping, dtype=int) for mapping in plan.null_mappings),
        residual_space_size=int(residual_space),
        restricted_label_space_size=int(restricted_label_space),
        requested_mode=str(plan.requested_mode),
        actual_mode=str(plan.actual_mode),
        n_null_mappings=int(len(plan.null_mappings)),
        monte_carlo_p_resolution=1.0 / (len(plan.null_mappings) + 1),
        seed=int(seed),
        reference_enumeration=("exhaustive" if plan.is_exhaustive else "monte_carlo"),
        exactness_status=exactness_status,
        availability_mask_sha256=hashlib.sha256(available_array.tobytes()).hexdigest(),
    )


def run_array_freedman_lane_calibration(
    scores: np.ndarray,
    plan: ArrayFreedmanLanePlan,
    *,
    statistic: str = "max_absolute_effect",
    tail: str = "greater",
    calibration_scale: str = "studentized",
    alpha: float = 0.05,
    family_index: Optional[Sequence[int]] = None,
) -> dict[str, np.ndarray]:
    """Run the same array kernel used by the fitted-design simulator."""
    statistic = _normalize_statistic(statistic)
    tail = _normalize_tail(tail)
    calibration_scale = str(calibration_scale).lower().replace("-", "_")
    if calibration_scale not in {"studentized", "effect"}:
        raise ValueError("calibration_scale must be studentized or effect")
    return _shared_curve_calibration_kernel(
        np.asarray(scores, dtype=float),
        reduced_design=plan.reduced_design,
        condition=plan.condition,
        widths=plan.widths,
        null_mappings=plan.null_mappings,
        statistic=statistic,
        tail=tail,
        calibration_scale=calibration_scale,
        alpha=float(alpha),
        family_index=(
            None if family_index is None else np.asarray(family_index, dtype=int)
        ),
    )


def _batch_studentized(
    beta: np.ndarray, standard_error: np.ndarray
) -> np.ndarray:
    beta_array = np.asarray(beta, dtype=float)
    error_array = np.asarray(standard_error, dtype=float)
    tolerance = np.finfo(float).eps * np.maximum(1.0, np.abs(beta_array))
    out = np.zeros_like(beta_array)
    regular = error_array > tolerance
    out[regular] = beta_array[regular] / error_array[regular]
    deterministic = (~regular) & (np.abs(beta_array) > tolerance)
    out[deterministic] = np.sign(beta_array[deterministic]) * np.inf
    return out


def run_array_freedman_lane_calibration_batch(
    scores: np.ndarray,
    plan: ArrayFreedmanLanePlan,
    *,
    statistic: str = "max_absolute_effect",
    tail: str = "greater",
    calibration_scale: str = "studentized",
    alpha: float = 0.05,
    family_index: Optional[Sequence[int]] = None,
    mapping_batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """Vectorize the scalar shared kernel over replicate and mapping batches.

    This path intentionally supports the frozen T21 formal statistic only. It
    uses the same reduced/full OLS projections, residual mappings, studentized
    max-absolute statistic, maxT rule, BY adjustment, and family gate as the
    scalar production kernel.
    """
    values = np.asarray(scores, dtype=float)
    if values.ndim != 4:
        raise ValueError("Batch scores require replicate/donor/bin/pathway axes")
    if values.shape[1] != len(plan.condition):
        raise ValueError("Batch scores and Freedman-Lane plan are donor-misaligned")
    statistic = _normalize_statistic(statistic)
    tail = _normalize_tail(tail)
    calibration_scale = str(calibration_scale).lower().replace("-", "_")
    if (
        statistic != "max_absolute_effect"
        or tail != "greater"
        or calibration_scale != "studentized"
    ):
        raise ValueError(
            "The batch kernel is frozen to studentized max_absolute_effect/greater"
        )
    mapping_batch_size = _require_integer(
        mapping_batch_size, "mapping_batch_size"
    )
    n_replicates, _, n_bins, n_pathways = values.shape
    reduced = np.asarray(plan.reduced_design, dtype=float)
    condition = np.asarray(plan.condition, dtype=bool)
    mappings = np.asarray(plan.null_mappings, dtype=int)
    if mappings.ndim != 2 or mappings.shape[1] != len(condition):
        raise ValueError("Freedman-Lane mappings are not rectangular")

    beta_curve = np.empty((n_replicates, n_bins, n_pathways), dtype=float)
    t_curve = np.empty_like(beta_curve)
    standard_error_curve = np.empty_like(beta_curve)
    adjusted_control_curve = np.empty_like(beta_curve)
    adjusted_case_curve = np.empty_like(beta_curve)
    residual_df_by_bin = np.empty(n_bins, dtype=int)
    bin_payloads: list[dict[str, np.ndarray | float | int]] = []
    for bin_index in range(n_bins):
        available = np.isfinite(values[0, :, bin_index, 0])
        if not np.array_equal(
            np.isfinite(values[:, :, bin_index, 0]),
            np.broadcast_to(available, (n_replicates, len(available))),
        ):
            raise ValueError("Every batch replicate must share the frozen missingness")
        indices = np.flatnonzero(available)
        y = values[:, indices, bin_index, :]
        if not np.isfinite(y).all():
            raise ValueError("Available batch scores must be finite")
        z = reduced[indices]
        c = condition[indices].astype(float)
        full = np.column_stack([z, c])
        pinv_z = np.linalg.pinv(z)
        pinv_full = np.linalg.pinv(full)
        hat_reduced = z @ pinv_z
        residual_reduced_projection = np.eye(len(indices)) - hat_reduced
        residual_full_projection = np.eye(len(indices)) - full @ pinv_full
        fitted_reduced = np.einsum(
            "ij,rjp->rip", hat_reduced, y, optimize=True
        )
        residual_reduced = np.einsum(
            "ij,rjp->rip", residual_reduced_projection, y, optimize=True
        )
        beta = np.einsum("i,rip->rp", pinv_full[-1], y, optimize=True)
        residual_full = np.einsum(
            "ij,rjp->rip", residual_full_projection, y, optimize=True
        )
        residual_df = int(len(indices) - np.linalg.matrix_rank(full))
        sigma2 = np.sum(residual_full**2, axis=1) / residual_df
        c_residual = c - z @ (pinv_z @ c)
        information = float(c_residual @ c_residual)
        standard_error = np.sqrt(np.maximum(sigma2, 0.0) / information)
        t_value = _batch_studentized(beta, standard_error)
        baseline_weight = np.mean(z, axis=0) @ pinv_full[:-1]
        baseline = np.einsum("i,rip->rp", baseline_weight, y, optimize=True)
        beta_curve[:, bin_index, :] = beta
        t_curve[:, bin_index, :] = t_value
        standard_error_curve[:, bin_index, :] = standard_error
        adjusted_control_curve[:, bin_index, :] = baseline
        adjusted_case_curve[:, bin_index, :] = baseline + beta
        residual_df_by_bin[bin_index] = residual_df
        global_to_local = np.full(len(condition), -1, dtype=int)
        global_to_local[indices] = np.arange(len(indices), dtype=int)
        source_local = global_to_local[mappings[:, indices]]
        if np.any(source_local < 0):
            raise RuntimeError(
                "Residual mapping crossed the frozen availability signature"
            )
        bin_payloads.append(
            {
                "fitted_reduced": fitted_reduced,
                "residual_reduced": residual_reduced,
                "source_local": source_local,
                "beta_weight": pinv_full[-1],
                "residual_full_projection": residual_full_projection,
                "residual_df": residual_df,
                "information": information,
            }
        )

    observed = np.max(np.abs(t_curve), axis=1)
    effect_statistics = np.max(np.abs(beta_curve), axis=1)
    exceed = np.zeros((n_replicates, n_pathways), dtype=np.int64)
    max_exceed = np.zeros_like(exceed)
    families = None if family_index is None else np.asarray(family_index, dtype=int)
    within_family_exceed = np.zeros_like(exceed)
    family_gate_exceed = np.zeros_like(exceed)
    if families is not None:
        if families.shape != (n_pathways,) or np.any(families < -1):
            raise ValueError("family_index must use -1 or a non-negative family code")
        assigned_families = np.unique(families[families >= 0])
        if not len(assigned_families):
            raise ValueError("family_index must assign at least one pathway")
        observed_family_max = np.stack(
            [
                np.max(observed[:, families == family], axis=1)
                for family in assigned_families
            ],
            axis=1,
        )
    else:
        assigned_families = np.empty(0, dtype=int)
        observed_family_max = np.empty((n_replicates, 0), dtype=float)

    for start in range(0, len(mappings), mapping_batch_size):
        stop = min(start + mapping_batch_size, len(mappings))
        batch_count = stop - start
        null_stat = np.zeros(
            (batch_count, n_replicates, n_pathways), dtype=float
        )
        for payload in bin_payloads:
            fitted_reduced = np.asarray(payload["fitted_reduced"], dtype=float)
            residual_reduced = np.asarray(payload["residual_reduced"], dtype=float)
            source_local = np.asarray(payload["source_local"], dtype=int)[start:stop]
            permuted = np.take(residual_reduced, source_local, axis=1)
            # np.take returns replicate,mapping,donor,pathway.
            y_star = (
                np.transpose(permuted, (1, 0, 2, 3))
                + fitted_reduced[None, :, :, :]
            )
            null_beta = np.einsum(
                "i,mrip->mrp",
                np.asarray(payload["beta_weight"], dtype=float),
                y_star,
                optimize=True,
            )
            null_residual = np.einsum(
                "ij,mrjp->mrip",
                np.asarray(payload["residual_full_projection"], dtype=float),
                y_star,
                optimize=True,
            )
            null_sigma2 = np.sum(null_residual**2, axis=2) / int(
                payload["residual_df"]
            )
            null_error = np.sqrt(
                np.maximum(null_sigma2, 0.0) / float(payload["information"])
            )
            null_t = _batch_studentized(null_beta, null_error)
            null_stat = np.maximum(null_stat, np.abs(null_t))
        exceed += np.sum(
            null_stat >= observed[None, :, :] - 1e-12, axis=0
        )
        global_max = np.max(null_stat, axis=2)
        max_exceed += np.sum(
            global_max[:, :, None] >= observed[None, :, :] - 1e-12,
            axis=0,
        )
        if families is not None:
            null_family_max = np.stack(
                [
                    np.max(null_stat[:, :, families == family], axis=2)
                    for family in assigned_families
                ],
                axis=2,
            )
            global_family_max = np.max(null_family_max, axis=2)
            for family_position, family in enumerate(assigned_families):
                members = families == family
                within_family_exceed[:, members] += np.sum(
                    null_family_max[:, :, family_position, None]
                    >= observed[:, members][None, :, :] - 1e-12,
                    axis=0,
                )
                family_gate_exceed[:, members] += np.sum(
                    global_family_max[:, :, None]
                    >= observed_family_max[:, family_position][None, :, None]
                    - 1e-12,
                    axis=0,
                )

    denominator = len(mappings) + 1
    p_raw = (exceed + 1.0) / denominator
    p_max_t = (max_exceed + 1.0) / denominator
    q_by = np.vstack([_by_adjust(row) for row in p_raw])
    if families is None:
        p_within_family = np.full_like(p_raw, np.nan)
        p_family_gate = np.full_like(p_raw, np.nan)
    else:
        p_within_family = (within_family_exceed + 1.0) / denominator
        p_family_gate = (family_gate_exceed + 1.0) / denominator
        p_within_family[:, families < 0] = 1.0
        p_family_gate[:, families < 0] = 1.0
    hierarchical_reject = (
        np.isfinite(p_within_family)
        & (p_within_family <= alpha)
        & (p_family_gate <= alpha)
    )
    production_hierarchical_reject = (
        np.isfinite(p_family_gate)
        & (p_max_t <= alpha)
        & (p_family_gate <= alpha)
    )
    return {
        "beta_curve": beta_curve,
        "standard_error_curve": standard_error_curve,
        "adjusted_control_curve": adjusted_control_curve,
        "adjusted_case_curve": adjusted_case_curve,
        "residual_df_by_bin": residual_df_by_bin,
        "calibration_curve": t_curve,
        "effect_statistic": effect_statistics,
        "calibration_statistic": observed,
        "p_raw": p_raw,
        "p_maxT": p_max_t,
        "q_by": q_by,
        "raw_reject": p_raw <= alpha,
        "maxT_reject": p_max_t <= alpha,
        "by_reject": q_by <= alpha,
        "p_pathway_within_family_maxT": p_within_family,
        "p_family_gate": p_family_gate,
        "within_family_maxT_reject": p_within_family <= alpha,
        "family_gate_reject": p_family_gate <= alpha,
        "family_hierarchical_reject": hierarchical_reject,
        "production_family_hierarchical_reject": production_hierarchical_reject,
    }


def _as_finite_effect_sizes(effect_sizes: Sequence[float]) -> tuple[float, ...]:
    if isinstance(effect_sizes, (str, bytes)):
        raise ValueError("effect_sizes must be a finite numeric sequence")
    values = tuple(float(value) for value in effect_sizes)
    if not values or not np.isfinite(values).all():
        raise ValueError("effect_sizes must contain finite numeric values")
    return values


def _groups_from_design(donor_design: pd.DataFrame) -> list[np.ndarray]:
    lookup: dict[str, list[int]] = {}
    for index, block in enumerate(donor_design["permutation_block"].astype(str)):
        lookup.setdefault(block, []).append(index)
    return [
        np.asarray(indices, dtype=int)
        for _, indices in sorted(lookup.items(), key=lambda item: item[0])
    ]


def _availability_from_design(donor_design: pd.DataFrame, n_bins: int) -> np.ndarray:
    signatures = donor_design["availability_signature"].astype(str).tolist()
    if any(len(signature) != n_bins for signature in signatures):
        raise ValueError(
            "donor_design availability signatures do not match the selected grid"
        )
    if any(set(signature) - {"0", "1"} for signature in signatures):
        raise ValueError("availability signatures must contain only 0/1")
    return np.asarray(
        [[character == "1" for character in signature] for signature in signatures],
        dtype=bool,
    )


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return math.nan, math.nan
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def _effect_profile(values: Optional[Sequence[float]], n_bins: int) -> np.ndarray:
    if values is None:
        return np.ones(n_bins, dtype=float)
    profile = np.asarray(list(values), dtype=float)
    if profile.shape != (n_bins,) or not np.isfinite(profile).all():
        raise ValueError(f"effect_profile must contain {n_bins} finite values")
    maximum = float(np.max(np.abs(profile)))
    if maximum <= 0:
        raise ValueError("effect_profile must contain at least one nonzero value")
    return profile / maximum


def _ar1_covariance(n_bins: int, correlation: float) -> np.ndarray:
    indices = np.arange(n_bins)
    return correlation ** np.abs(indices[:, None] - indices[None, :])


def run_covariate_adjusted_design_simulation(
    fitted_result: CovariateAdjustedDonorPseudobulkResult,
    *,
    effect_sizes: Sequence[float] = (0.0, 0.5, 1.0),
    effect_profile: Optional[Sequence[float]] = None,
    n_simulations: int = 200,
    n_pathways: int = 20,
    n_signal_pathways: int = 1,
    residual_sd: float = 1.0,
    ar1_correlation: float = 0.6,
    pathway_correlation: float = 0.2,
    nuisance_scale: float = 0.75,
    case_variance_ratio: float = 1.0,
    extreme_donor_scale: float = 1.0,
    extreme_donor: Optional[str] = None,
    statistic: Optional[str] = None,
    tail: Optional[str] = None,
    calibration_scale: Optional[str] = None,
    permutation_mode: str = "auto",
    n_permutations: int = 999,
    max_exact_permutations: int = 20000,
    alpha: float = 0.05,
    seed: int = 42,
) -> CovariateAdjustedDesignSimulationResult:
    """Simulate donor pathway curves on an observed adjusted-analysis design.

    This harness reuses the fitted result's donor rows, condition imbalance,
    encoded nuisance design, selected bins, missingness signatures, and
    residual-permutation blocks. It measures conditional-regulation type-I
    error, maxT/BY behavior, and amplitude power under correlated Gaussian
    donor curves. It does not simulate cell occupancy, trajectory speed,
    pathway-gene overlap, or onset estimation; those require separate scenario
    generators and must not be inferred from this report.
    """
    if not isinstance(fitted_result, CovariateAdjustedDonorPseudobulkResult):
        raise TypeError(
            "fitted_result must be a CovariateAdjustedDonorPseudobulkResult"
        )
    n_simulations = _require_integer(n_simulations, "n_simulations")
    n_pathways = _require_integer(n_pathways, "n_pathways")
    n_signal_pathways = _require_integer(
        n_signal_pathways, "n_signal_pathways", 0
    )
    if n_signal_pathways > n_pathways:
        raise ValueError("n_signal_pathways cannot exceed n_pathways")
    n_permutations = _require_integer(n_permutations, "n_permutations")
    max_exact_permutations = _require_integer(
        max_exact_permutations, "max_exact_permutations", 2
    )
    seed = _require_integer(seed, "seed", 0)
    effect_sizes = _as_finite_effect_sizes(effect_sizes)
    for name, value in {
        "residual_sd": residual_sd,
        "nuisance_scale": nuisance_scale,
        "case_variance_ratio": case_variance_ratio,
        "extreme_donor_scale": extreme_donor_scale,
    }.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name, value in {
        "ar1_correlation": ar1_correlation,
        "pathway_correlation": pathway_correlation,
    }.items():
        if not 0 <= value < 1:
            raise ValueError(f"{name} must be in [0, 1)")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    metadata_source = fitted_result.metadata
    statistic = _normalize_statistic(
        statistic if statistic is not None else metadata_source["statistic"]
    )
    tail = _normalize_tail(tail if tail is not None else metadata_source["tail"])
    _test_scale(np.asarray([0.0]), statistic, tail)
    calibration_scale = str(
        calibration_scale
        if calibration_scale is not None
        else metadata_source["calibration_scale"]
    ).lower().replace("-", "_")
    if calibration_scale not in {"studentized", "effect"}:
        raise ValueError("calibration_scale must be 'studentized' or 'effect'")

    donor_design = fitted_result.donor_design[
        fitted_result.donor_design["included_in_inference"].astype(bool)
    ].copy()
    donor_design = donor_design.sort_values("donor_index").reset_index(drop=True)
    continuous_keys = tuple(metadata_source["continuous_covariate_keys"])
    categorical_keys = tuple(metadata_source["categorical_covariate_keys"])
    strata_keys = tuple(metadata_source["strata_keys"])
    donor_design["__stratum_key"] = [
        tuple(str(row[key]) for key in strata_keys) if strata_keys else ("__all__",)
        for _, row in donor_design.iterrows()
    ]
    encoded = _encode_reduced_design(
        donor_design,
        continuous_covariate_keys=continuous_keys,
        categorical_covariate_keys=categorical_keys,
        strata_keys=strata_keys,
    )
    condition = donor_design["observed_case"].astype(bool).to_numpy()
    selected_bins = np.asarray(metadata_source["selected_bin_ids"], dtype=int)
    n_bins = len(selected_bins)
    profile = _effect_profile(effect_profile, n_bins)
    available = _availability_from_design(donor_design, n_bins)
    groups = _groups_from_design(donor_design)
    residual_space, label_space = _space_sizes(condition, groups)
    block_constant = _block_constant_reduced_design(
        encoded.reduced,
        available,
        groups,
        np.arange(n_bins, dtype=int),
    )
    requested_mode = str(permutation_mode).lower().replace("-", "_")
    if requested_mode == "exact" and not block_constant:
        raise CovariateDesignError(
            "permutation_mode='exact' is unavailable because the observed "
            "reduced design varies within simulation permutation blocks"
        )
    plan = _make_residual_plan(
        len(donor_design),
        groups,
        permutation_mode=permutation_mode,
        n_permutations=n_permutations,
        max_exact_permutations=max_exact_permutations,
        seed=seed + 1,
    )
    edges = np.asarray(metadata_source["grid_edges"], dtype=float)
    widths = np.diff(edges)[selected_bins]
    covariance = _ar1_covariance(n_bins, ar1_correlation)
    rng = np.random.default_rng(seed)
    pathway_names = [f"SIM_PATHWAY_{index:04d}" for index in range(n_pathways)]
    extreme_index = None
    if extreme_donor is not None:
        matches = np.flatnonzero(donor_design["donor"].astype(str).eq(str(extreme_donor)))
        if len(matches) != 1:
            raise ValueError("extreme_donor must identify one included donor")
        extreme_index = int(matches[0])
    elif extreme_donor_scale != 1.0:
        extreme_index = len(donor_design) - 1

    replicate_rows: list[dict[str, Any]] = []
    simulation_rows: list[dict[str, Any]] = []
    for effect_size in effect_sizes:
        signal_count = n_signal_pathways if effect_size != 0 else 0
        signal_mask = np.arange(n_pathways) < signal_count
        pathway_effects = signal_mask.astype(float) * float(effect_size)
        for simulation_id in range(n_simulations):
            nuisance_coefficients = rng.normal(
                0.0,
                nuisance_scale,
                size=(encoded.reduced.shape[1], n_bins, n_pathways),
            )
            mean = np.einsum(
                "dq,qbp->dbp", encoded.reduced, nuisance_coefficients
            )
            mean += (
                condition[:, None, None]
                * profile[None, :, None]
                * pathway_effects[None, None, :]
            )
            common = rng.multivariate_normal(
                np.zeros(n_bins), covariance, size=len(donor_design)
            )[:, :, None]
            independent = rng.multivariate_normal(
                np.zeros(n_bins),
                covariance,
                size=(len(donor_design), n_pathways),
            ).transpose(0, 2, 1)
            error = residual_sd * (
                math.sqrt(pathway_correlation) * common
                + math.sqrt(1.0 - pathway_correlation) * independent
            )
            error[condition] *= math.sqrt(case_variance_ratio)
            if extreme_index is not None:
                error[extreme_index] *= extreme_donor_scale
            scores = mean + error
            scores[~available] = np.nan

            calibrated = _shared_curve_calibration_kernel(
                scores,
                reduced_design=encoded.reduced,
                condition=condition,
                widths=widths,
                null_mappings=plan.null_mappings,
                statistic=statistic,
                tail=tail,
                calibration_scale=calibration_scale,
                alpha=alpha,
            )
            effect_statistics = calibrated["effect_statistic"]
            raw_calibration = calibrated["calibration_statistic"]
            p_raw = calibrated["p_raw"]
            p_max_t = calibrated["p_maxT"]
            q_by = calibrated["q_by"]
            raw_reject = calibrated["raw_reject"]
            max_reject = calibrated["maxT_reject"]
            by_reject = calibrated["by_reject"]
            null_mask = ~signal_mask
            false_by = int((by_reject & null_mask).sum())
            total_by = int(by_reject.sum())
            simulation_rows.append(
                {
                    "effect_size": float(effect_size),
                    "simulation_id": int(simulation_id),
                    "n_signal_pathways": int(signal_count),
                    "n_null_pathways": int(null_mask.sum()),
                    "raw_null_rejection_fraction": float(
                        raw_reject[null_mask].mean()
                    ),
                    "maxT_any_false_rejection": bool(
                        (max_reject & null_mask).any()
                    ),
                    "by_any_false_rejection": bool((by_reject & null_mask).any()),
                    "by_false_discovery_proportion": float(
                        false_by / total_by if total_by else 0.0
                    ),
                    "raw_signal_power": float(
                        raw_reject[signal_mask].mean()
                        if signal_count
                        else math.nan
                    ),
                    "maxT_signal_power": float(
                        max_reject[signal_mask].mean()
                        if signal_count
                        else math.nan
                    ),
                    "by_signal_power": float(
                        by_reject[signal_mask].mean()
                        if signal_count
                        else math.nan
                    ),
                }
            )
            for pathway_index, pathway in enumerate(pathway_names):
                replicate_rows.append(
                    {
                        "effect_size": float(effect_size),
                        "simulation_id": int(simulation_id),
                        "Pathway": pathway,
                        "is_signal": bool(signal_mask[pathway_index]),
                        "true_curve_scale": float(pathway_effects[pathway_index]),
                        "observed_effect_statistic": float(
                            effect_statistics[pathway_index]
                        ),
                        "observed_calibration_statistic": float(
                            raw_calibration[pathway_index]
                        ),
                        "p_raw": float(p_raw[pathway_index]),
                        "p_maxT": float(p_max_t[pathway_index]),
                        "q_by": float(q_by[pathway_index]),
                        "raw_reject": bool(raw_reject[pathway_index]),
                        "maxT_reject": bool(max_reject[pathway_index]),
                        "by_reject": bool(by_reject[pathway_index]),
                    }
                )

    replicate_metrics = pd.DataFrame(replicate_rows)
    simulation_metrics = pd.DataFrame(simulation_rows)
    summary_rows = []
    for effect_size, group in simulation_metrics.groupby("effect_size", sort=False):
        fwer_successes = int(group["maxT_any_false_rejection"].sum())
        by_false_successes = int(group["by_any_false_rejection"].sum())
        fwer_low, fwer_high = _wilson_interval(fwer_successes, len(group))
        by_low, by_high = _wilson_interval(by_false_successes, len(group))
        raw_values = group["raw_null_rejection_fraction"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "effect_size": float(effect_size),
                "n_simulations": int(len(group)),
                "n_pathways": int(n_pathways),
                "n_signal_pathways": int(group["n_signal_pathways"].iloc[0]),
                "raw_type_i_error": float(np.mean(raw_values)),
                "raw_type_i_monte_carlo_se": float(
                    np.std(raw_values, ddof=1) / math.sqrt(len(raw_values))
                    if len(raw_values) > 1
                    else math.nan
                ),
                "maxT_fwer": float(fwer_successes / len(group)),
                "maxT_fwer_wilson_low": fwer_low,
                "maxT_fwer_wilson_high": fwer_high,
                "by_any_false_rejection_rate": float(
                    by_false_successes / len(group)
                ),
                "by_any_false_wilson_low": by_low,
                "by_any_false_wilson_high": by_high,
                "mean_by_false_discovery_proportion": float(
                    group["by_false_discovery_proportion"].mean()
                ),
                "raw_power": float(group["raw_signal_power"].mean()),
                "maxT_power": float(group["maxT_signal_power"].mean()),
                "by_power": float(group["by_signal_power"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    exactness_status = (
        "finite_sample_exact_residual_group_conditional_on_invariance"
        if block_constant and plan.is_exhaustive
        else "monte_carlo_finite_sample_residual_group_conditional_on_invariance"
        if block_constant
        else "exhaustive_freedman_lane_reference_not_finite_sample_exact"
        if plan.is_exhaustive
        else "monte_carlo_freedman_lane_approximation"
    )
    metadata = {
        "method": "design_matched_covariate_adjusted_donor_curve_simulation",
        "source_analysis_method": metadata_source["method"],
        "effect_sizes": list(effect_sizes),
        "effect_profile": profile.tolist(),
        "n_simulations": int(n_simulations),
        "n_pathways": int(n_pathways),
        "n_signal_pathways_nonzero_scenario": int(n_signal_pathways),
        "residual_sd": float(residual_sd),
        "ar1_correlation": float(ar1_correlation),
        "pathway_correlation": float(pathway_correlation),
        "nuisance_scale": float(nuisance_scale),
        "case_variance_ratio": float(case_variance_ratio),
        "extreme_donor_scale": float(extreme_donor_scale),
        "extreme_donor": extreme_donor,
        "statistic": statistic,
        "tail": tail,
        "calibration_scale": calibration_scale,
        "alpha": float(alpha),
        "seed": int(seed),
        "residual_permutation_space_size": int(residual_space),
        "condition_label_space_size": int(label_space),
        "n_null_mappings_evaluated": int(len(plan.null_mappings)),
        "reference_enumeration": (
            "exhaustive" if plan.is_exhaustive else "monte_carlo"
        ),
        "exactness_status": exactness_status,
        "design_matched_features": [
            "donor_rows",
            "condition_imbalance",
            "nuisance_design",
            "selected_grid",
            "missingness_signatures",
            "permutation_blocks",
        ],
        "not_simulated": [
            "cell_occupancy",
            "trajectory_speed",
            "pseudotime_reestimation",
            "pathway_gene_overlap",
            "onset_or_peak_estimation",
            "leave_one_donor_out_stability",
        ],
    }
    for table in (replicate_metrics, simulation_metrics, summary):
        table.attrs["covariate_adjusted_design_simulation"] = metadata.copy()
    return CovariateAdjustedDesignSimulationResult(
        replicate_metrics=replicate_metrics,
        simulation_metrics=simulation_metrics,
        summary=summary,
        design_diagnostics=fitted_result.design_diagnostics.copy(),
        metadata=metadata,
    )
