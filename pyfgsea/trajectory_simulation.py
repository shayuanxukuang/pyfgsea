from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd


PRIMARY_PATHWAY = "DELAYED_BRANCH_PATHWAY"
OVERLAPPING_PATHWAY = "OVERLAPPING_PATHWAY"
NULL_PATHWAY = "NULL_PATHWAY"


@dataclass
class DonorTrajectorySimulation:
    """Container for a donor-aware trajectory simulation and its truth tables."""

    adata: Any
    gene_sets: Dict[str, list[str]]
    pathway_truth: pd.DataFrame
    donor_truth: pd.DataFrame
    config: Dict[str, Any]

    def to_tables(self) -> Dict[str, pd.DataFrame]:
        """Return independent copies of the simulation truth tables."""
        cell_columns = [
            "condition",
            "donor",
            "batch",
            "branch",
            "pseudotime_true",
            "dpt_pseudotime",
            "pseudotime_sd",
            "lineage_prob_branch_a",
            "lineage_prob_branch_b",
            "active_branch_probability",
        ]
        return {
            "pathway_truth": self.pathway_truth.copy(),
            "donor_truth": self.donor_truth.copy(),
            "cell_truth": self.adata.obs[cell_columns].copy(),
        }


def _resolve_pair(
    value: Union[int, Tuple[int, int]],
    name: str,
) -> Tuple[int, int]:
    if isinstance(value, (int, np.integer)):
        pair = (int(value), int(value))
    else:
        try:
            pair = tuple(int(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive integer or a two-value tuple") from exc
        if len(pair) != 2:
            raise ValueError(f"{name} must be a positive integer or a two-value tuple")
    if any(item <= 0 for item in pair):
        raise ValueError(f"{name} values must be positive")
    return pair


def _draw_cell_count(
    mean_cells: int,
    cell_count_cv: float,
    rng: np.random.Generator,
) -> int:
    if cell_count_cv == 0:
        return int(mean_cells)
    sigma = np.sqrt(np.log1p(cell_count_cv**2))
    mean_log = np.log(float(mean_cells)) - 0.5 * sigma**2
    return max(4, int(np.rint(rng.lognormal(mean=mean_log, sigma=sigma))))


def _sigmoid_profile(
    pseudotime: np.ndarray,
    onset: float,
    width: float,
) -> np.ndarray:
    scaled = np.clip((np.asarray(pseudotime, dtype=float) - onset) / width, -60, 60)
    return 1.0 / (1.0 + np.exp(-scaled))


def _validate_parameters(
    n_genes: int,
    pathway_size: int,
    overlap_fraction: float,
    cell_count_cv: float,
    control_onset: float,
    case_onset: float,
    branch_point: float,
    activation_width: float,
    active_branch: str,
    branch_a_probability_control: float,
    branch_a_probability_case: float,
    donor_intercept_sd: float,
    donor_slope_sd: float,
    donor_time_shift_sd: float,
    pseudotime_noise_sd: float,
    lineage_probability_noise_sd: float,
    within_pathway_correlation: float,
    noise_sd: float,
    batch_effect_size: float,
    dropout_rate: float,
    n_pseudotime_draws: int,
) -> int:
    if pathway_size < 2:
        raise ValueError("pathway_size must be at least 2")
    if not 0 <= overlap_fraction < 1:
        raise ValueError("overlap_fraction must be in [0, 1)")
    n_overlap = min(pathway_size - 1, int(np.rint(pathway_size * overlap_fraction)))
    required_genes = 3 * pathway_size - n_overlap
    if n_genes < required_genes:
        raise ValueError(
            "n_genes must be at least "
            f"{required_genes} for the requested pathway size and overlap"
        )
    if cell_count_cv < 0:
        raise ValueError("cell_count_cv must be non-negative")
    for name, value in {
        "control_onset": control_onset,
        "case_onset": case_onset,
        "branch_point": branch_point,
    }.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if activation_width <= 0:
        raise ValueError("activation_width must be positive")
    if active_branch not in {"branch_a", "branch_b"}:
        raise ValueError("active_branch must be 'branch_a' or 'branch_b'")
    for name, value in {
        "branch_a_probability_control": branch_a_probability_control,
        "branch_a_probability_case": branch_a_probability_case,
    }.items():
        if not 0 < value < 1:
            raise ValueError(f"{name} must be between 0 and 1")
    for name, value in {
        "donor_intercept_sd": donor_intercept_sd,
        "donor_slope_sd": donor_slope_sd,
        "donor_time_shift_sd": donor_time_shift_sd,
        "pseudotime_noise_sd": pseudotime_noise_sd,
        "lineage_probability_noise_sd": lineage_probability_noise_sd,
        "noise_sd": noise_sd,
        "batch_effect_size": batch_effect_size,
    }.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    if not 0 <= within_pathway_correlation < 1:
        raise ValueError("within_pathway_correlation must be in [0, 1)")
    if not 0 <= dropout_rate < 1:
        raise ValueError("dropout_rate must be in [0, 1)")
    if n_pseudotime_draws < 1:
        raise ValueError("n_pseudotime_draws must be at least 1")
    return n_overlap


def simulate_donor_branch_trajectory(
    n_donors_per_condition: Union[int, Tuple[int, int]] = 4,
    cells_per_donor: Union[int, Tuple[int, int]] = 120,
    cell_count_cv: float = 0.35,
    n_genes: int = 180,
    pathway_size: int = 16,
    overlap_fraction: float = 0.25,
    control_onset: float = 0.35,
    case_onset: float = 0.55,
    branch_point: float = 0.25,
    active_branch: str = "branch_b",
    effect_size: float = 1.5,
    activation_width: float = 0.06,
    branch_a_probability_control: float = 0.50,
    branch_a_probability_case: float = 0.50,
    donor_intercept_sd: float = 0.35,
    donor_slope_sd: float = 0.15,
    donor_time_shift_sd: float = 0.025,
    pseudotime_noise_sd: float = 0.04,
    n_pseudotime_draws: int = 10,
    lineage_probability_noise_sd: float = 0.04,
    within_pathway_correlation: float = 0.35,
    noise_sd: float = 0.35,
    batch_effect_size: float = 0.15,
    dropout_rate: float = 0.02,
    seed: int = 0,
) -> DonorTrajectorySimulation:
    """Simulate a multi-donor, two-condition, bifurcating trajectory.

    The primary pathway activates only in ``active_branch``. Its logistic
    50%-activation midpoint is ``control_onset`` in control cells and
    ``case_onset`` in case cells, so the midpoint delay is known exactly.
    Cells from the same donor share pathway intercept and slope effects. An
    independent donor-correlated null pathway has no condition effect and can
    be used to measure pseudoreplication. The returned AnnData also keeps
    true and noisy pseudotime, multiple pseudotime draws, and soft lineage
    probabilities so uncertainty-aware methods can be benchmarked without
    treating inferred trajectory coordinates as truth.

    ``n_donors_per_condition`` and ``cells_per_donor`` accept either one value
    for both conditions or ``(control, case)`` pairs, allowing donor and cell
    count imbalance to be controlled independently.
    """
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError(
            "simulate_donor_branch_trajectory requires anndata"
        ) from exc

    donor_counts = _resolve_pair(n_donors_per_condition, "n_donors_per_condition")
    cell_means = _resolve_pair(cells_per_donor, "cells_per_donor")
    n_overlap = _validate_parameters(
        n_genes=n_genes,
        pathway_size=pathway_size,
        overlap_fraction=overlap_fraction,
        cell_count_cv=cell_count_cv,
        control_onset=control_onset,
        case_onset=case_onset,
        branch_point=branch_point,
        activation_width=activation_width,
        active_branch=active_branch,
        branch_a_probability_control=branch_a_probability_control,
        branch_a_probability_case=branch_a_probability_case,
        donor_intercept_sd=donor_intercept_sd,
        donor_slope_sd=donor_slope_sd,
        donor_time_shift_sd=donor_time_shift_sd,
        pseudotime_noise_sd=pseudotime_noise_sd,
        lineage_probability_noise_sd=lineage_probability_noise_sd,
        within_pathway_correlation=within_pathway_correlation,
        noise_sd=noise_sd,
        batch_effect_size=batch_effect_size,
        dropout_rate=dropout_rate,
        n_pseudotime_draws=n_pseudotime_draws,
    )

    rng = np.random.default_rng(seed)
    conditions = ("control", "case")
    branch_probabilities = (
        branch_a_probability_control,
        branch_a_probability_case,
    )
    onset_by_condition = {"control": control_onset, "case": case_onset}

    obs_frames = []
    donor_rows = []
    donor_intercepts = {}
    donor_slopes = {}
    null_donor_intercepts = {}
    null_donor_slopes = {}
    donor_time_shifts = {}
    for condition_idx, condition in enumerate(conditions):
        for donor_idx in range(donor_counts[condition_idx]):
            donor = f"{condition}_{donor_idx + 1}"
            batch = f"batch_{donor_idx % 2 + 1}"
            n_cells = _draw_cell_count(
                cell_means[condition_idx], cell_count_cv, rng
            )
            pseudotime_true = np.sort(rng.beta(1.2, 1.2, size=n_cells))
            branch_a = rng.random(n_cells) < branch_probabilities[condition_idx]
            branch = np.where(branch_a, "branch_a", "branch_b")
            time_shift = float(rng.normal(0.0, donor_time_shift_sd))
            pseudotime_observed = np.clip(
                pseudotime_true
                + time_shift
                + rng.normal(0.0, pseudotime_noise_sd, size=n_cells),
                0.0,
                1.0,
            )

            denominator = max(1.0 - branch_point, np.finfo(float).eps)
            lineage_certainty = np.sqrt(
                np.clip((pseudotime_true - branch_point) / denominator, 0.0, 1.0)
            )
            branch_sign = np.where(branch_a, 1.0, -1.0)
            probability_a = np.clip(
                0.5
                + 0.49 * lineage_certainty * branch_sign
                + rng.normal(0.0, lineage_probability_noise_sd, size=n_cells),
                0.01,
                0.99,
            )
            probability_b = 1.0 - probability_a

            donor_intercept = float(rng.normal(0.0, donor_intercept_sd))
            donor_slope = float(rng.normal(0.0, donor_slope_sd))
            null_donor_intercept = float(rng.normal(0.0, donor_intercept_sd))
            null_donor_slope = float(rng.normal(0.0, donor_slope_sd))
            donor_intercepts[donor] = donor_intercept
            donor_slopes[donor] = donor_slope
            null_donor_intercepts[donor] = null_donor_intercept
            null_donor_slopes[donor] = null_donor_slope
            donor_time_shifts[donor] = time_shift

            cell_ids = [f"{donor}_cell_{idx + 1}" for idx in range(n_cells)]
            obs_frames.append(
                pd.DataFrame(
                    {
                        "condition": condition,
                        "donor": donor,
                        "batch": batch,
                        "branch": branch,
                        "pseudotime_true": pseudotime_true,
                        "dpt_pseudotime": pseudotime_observed,
                        "pseudotime_sd": pseudotime_noise_sd,
                        "lineage_prob_branch_a": probability_a,
                        "lineage_prob_branch_b": probability_b,
                        "active_branch_probability": (
                            probability_a
                            if active_branch == "branch_a"
                            else probability_b
                        ),
                    },
                    index=cell_ids,
                )
            )
            donor_rows.append(
                {
                    "donor": donor,
                    "condition": condition,
                    "batch": batch,
                    "n_cells": n_cells,
                    "branch_a_fraction": float(branch_a.mean()),
                    "branch_b_fraction": float((~branch_a).mean()),
                    "pathway_random_intercept": donor_intercept,
                    "pathway_random_slope": donor_slope,
                    "null_pathway_random_intercept": null_donor_intercept,
                    "null_pathway_random_slope": null_donor_slope,
                    "pseudotime_shift": time_shift,
                }
            )

    obs = pd.concat(obs_frames, axis=0)
    for column in ("condition", "donor", "batch", "branch"):
        obs[column] = pd.Categorical(obs[column])

    genes = [f"Gene_{idx}" for idx in range(n_genes)]
    target_indices = np.arange(pathway_size)
    overlap_unique_start = pathway_size
    overlap_unique_stop = overlap_unique_start + pathway_size - n_overlap
    overlap_indices = np.concatenate(
        [
            target_indices[:n_overlap],
            np.arange(overlap_unique_start, overlap_unique_stop),
        ]
    )
    null_indices = np.arange(overlap_unique_stop, overlap_unique_stop + pathway_size)
    gene_sets = {
        PRIMARY_PATHWAY: [genes[idx] for idx in target_indices],
        OVERLAPPING_PATHWAY: [genes[idx] for idx in overlap_indices],
        NULL_PATHWAY: [genes[idx] for idx in null_indices],
    }

    n_cells_total = len(obs)
    baseline = rng.normal(1.5, 0.15, size=n_genes)
    expected = np.broadcast_to(baseline, (n_cells_total, n_genes)).copy()
    true_time = obs["pseudotime_true"].to_numpy(dtype=float)
    condition_values = obs["condition"].astype(str).to_numpy()
    branch_values = obs["branch"].astype(str).to_numpy()
    donor_values = obs["donor"].astype(str).to_numpy()
    onset = np.asarray([onset_by_condition[value] for value in condition_values])
    pathway_profile = _sigmoid_profile(true_time, onset, activation_width)
    pathway_profile *= (branch_values == active_branch).astype(float)
    pathway_effect = effect_size * pathway_profile
    pathway_effect += np.asarray(
        [donor_intercepts[value] for value in donor_values], dtype=float
    )
    pathway_effect += np.asarray(
        [donor_slopes[value] for value in donor_values], dtype=float
    ) * (true_time - 0.5)
    gene_loadings = rng.lognormal(mean=0.0, sigma=0.10, size=pathway_size)
    gene_loadings /= gene_loadings.mean()
    expected[:, target_indices] += pathway_effect[:, None] * gene_loadings[None, :]

    null_pathway_effect = np.asarray(
        [null_donor_intercepts[value] for value in donor_values], dtype=float
    )
    null_pathway_effect += np.asarray(
        [null_donor_slopes[value] for value in donor_values], dtype=float
    ) * (true_time - 0.5)
    null_gene_loadings = rng.lognormal(mean=0.0, sigma=0.10, size=pathway_size)
    null_gene_loadings /= null_gene_loadings.mean()
    expected[:, null_indices] += (
        null_pathway_effect[:, None] * null_gene_loadings[None, :]
    )

    batch_candidates = np.setdiff1d(
        np.arange(n_genes),
        np.union1d(target_indices, null_indices),
        assume_unique=True,
    )
    n_batch_genes = min(max(1, n_genes // 10), len(batch_candidates))
    batch_indices = batch_candidates[-n_batch_genes:]
    batch_two = (obs["batch"].astype(str).to_numpy() == "batch_2").astype(float)
    expected[:, batch_indices] += batch_effect_size * batch_two[:, None]
    expected = np.clip(expected, 0.05, None)

    noise = rng.normal(0.0, noise_sd, size=expected.shape)
    if noise_sd > 0 and pathway_size > 0:
        shared_noise = rng.normal(0.0, noise_sd, size=(n_cells_total, 1))
        independent_noise = rng.normal(
            0.0, noise_sd, size=(n_cells_total, pathway_size)
        )
        noise[:, target_indices] = (
            np.sqrt(within_pathway_correlation) * shared_noise
            + np.sqrt(1.0 - within_pathway_correlation) * independent_noise
        )
        null_shared_noise = rng.normal(0.0, noise_sd, size=(n_cells_total, 1))
        null_independent_noise = rng.normal(
            0.0, noise_sd, size=(n_cells_total, pathway_size)
        )
        noise[:, null_indices] = (
            np.sqrt(within_pathway_correlation) * null_shared_noise
            + np.sqrt(1.0 - within_pathway_correlation) * null_independent_noise
        )
    expression = np.clip(expected + noise, 0.0, None)
    if dropout_rate > 0:
        expression[rng.random(expression.shape) < dropout_rate] = 0.0

    var = pd.DataFrame(index=genes)
    var["in_primary_pathway"] = False
    var["in_overlapping_pathway"] = False
    var["in_null_pathway"] = False
    var["batch_affected"] = False
    var["primary_pathway_loading"] = 0.0
    var["null_pathway_loading"] = 0.0
    var.iloc[target_indices, var.columns.get_loc("in_primary_pathway")] = True
    var.iloc[overlap_indices, var.columns.get_loc("in_overlapping_pathway")] = True
    var.iloc[null_indices, var.columns.get_loc("in_null_pathway")] = True
    var.iloc[batch_indices, var.columns.get_loc("batch_affected")] = True
    var.iloc[
        target_indices, var.columns.get_loc("primary_pathway_loading")
    ] = gene_loadings
    var.iloc[
        null_indices, var.columns.get_loc("null_pathway_loading")
    ] = null_gene_loadings

    adata = ad.AnnData(
        X=expression.astype(np.float32),
        obs=obs,
        var=var,
    )
    adata.layers["expected_expression"] = expected.astype(np.float32)
    cell_time_shifts = np.asarray(
        [donor_time_shifts[value] for value in donor_values], dtype=float
    )
    pseudotime_draws = np.clip(
        true_time[:, None]
        + cell_time_shifts[:, None]
        + rng.normal(
            0.0,
            pseudotime_noise_sd,
            size=(n_cells_total, n_pseudotime_draws),
        ),
        0.0,
        1.0,
    )
    adata.obsm["pseudotime_draws"] = pd.DataFrame(
        pseudotime_draws,
        index=adata.obs_names,
        columns=[f"draw_{idx}" for idx in range(n_pseudotime_draws)],
    )
    adata.obsm["lineage_probabilities"] = pd.DataFrame(
        obs[["lineage_prob_branch_a", "lineage_prob_branch_b"]].to_numpy(),
        index=adata.obs_names,
        columns=["branch_a", "branch_b"],
    )

    realized_overlap = n_overlap / pathway_size
    has_fixed_signal = not np.isclose(effect_size, 0.0)
    has_condition_contrast = has_fixed_signal and not np.isclose(
        case_onset, control_onset
    )
    has_branch_composition_contrast = not np.isclose(
        branch_a_probability_case, branch_a_probability_control
    )
    has_composition_induced_pathway_contrast = (
        has_fixed_signal and has_branch_composition_contrast
    )
    direction = "activation" if effect_size > 0 else "suppression"
    if not has_fixed_signal and has_branch_composition_contrast:
        primary_truth_class = "null_expression_with_fate_selection_contrast"
    elif not has_fixed_signal:
        primary_truth_class = "null_with_donor_heterogeneity"
    elif case_onset > control_onset:
        primary_truth_class = f"condition_delayed_branch_{direction}"
    elif case_onset < control_onset:
        primary_truth_class = f"condition_advanced_branch_{direction}"
    else:
        primary_truth_class = f"condition_matched_branch_{direction}"
    truth_control_onset = control_onset if has_fixed_signal else np.nan
    truth_case_onset = case_onset if has_fixed_signal else np.nan
    truth_delay = case_onset - control_onset if has_fixed_signal else np.nan
    truth_active_branch = active_branch if has_fixed_signal else "none"
    pathway_truth = pd.DataFrame(
        [
            {
                "pathway": PRIMARY_PATHWAY,
                "truth_scope": "primary_estimand",
                "truth_class": primary_truth_class,
                "is_primary_signal": has_fixed_signal,
                "is_strict_negative_control": not has_fixed_signal,
                "is_null_condition": not has_condition_contrast,
                "is_null_condition_branch_conditional": not has_condition_contrast,
                "is_null_condition_marginal": not (
                    has_condition_contrast
                    or has_composition_induced_pathway_contrast
                ),
                "is_null_fate_selection": not has_branch_composition_contrast,
                "branch_composition_contrast": has_branch_composition_contrast,
                "is_null_branch": not has_fixed_signal,
                "is_null_trajectory": not has_fixed_signal,
                "effect_direction": direction if has_fixed_signal else "none",
                "control_onset": truth_control_onset,
                "case_onset": truth_case_onset,
                "activation_delay": truth_delay,
                "onset_definition": "logistic_50pct_midpoint",
                "delay_definition": "case_midpoint_minus_control_midpoint",
                "delay_sign": "case_minus_control",
                "time_scale": "pseudotime_true",
                "estimand_scope": "branch_conditional_population_fixed_effect",
                "reference_condition": "control",
                "query_condition": "case",
                "active_branch": truth_active_branch,
                "branch_point": branch_point,
                "effect_size": effect_size,
                "expected_effect_fraction": 1.0 if has_fixed_signal else 0.0,
                "overlap_with_primary": 1.0,
            },
            {
                "pathway": OVERLAPPING_PATHWAY,
                "truth_scope": "overlap_sensitivity_control",
                "truth_class": "overlap_contaminated",
                "is_primary_signal": False,
                "is_strict_negative_control": False,
                "is_null_condition": not has_condition_contrast,
                "is_null_condition_branch_conditional": not has_condition_contrast,
                "is_null_condition_marginal": not (
                    has_condition_contrast
                    or has_composition_induced_pathway_contrast
                ),
                "is_null_fate_selection": not has_branch_composition_contrast,
                "branch_composition_contrast": has_branch_composition_contrast,
                "is_null_branch": not has_fixed_signal,
                "is_null_trajectory": not has_fixed_signal,
                "effect_direction": direction if has_fixed_signal else "none",
                "control_onset": truth_control_onset,
                "case_onset": truth_case_onset,
                "activation_delay": truth_delay,
                "onset_definition": "logistic_50pct_midpoint",
                "delay_definition": "case_midpoint_minus_control_midpoint",
                "delay_sign": "case_minus_control",
                "time_scale": "pseudotime_true",
                "estimand_scope": "overlap_sensitivity",
                "reference_condition": "control",
                "query_condition": "case",
                "active_branch": truth_active_branch,
                "branch_point": branch_point,
                "effect_size": effect_size,
                "expected_effect_fraction": (
                    realized_overlap if has_fixed_signal else 0.0
                ),
                "overlap_with_primary": realized_overlap,
            },
            {
                "pathway": NULL_PATHWAY,
                "truth_scope": "donor_correlated_negative_control",
                "truth_class": "null_with_donor_heterogeneity",
                "is_primary_signal": False,
                "is_strict_negative_control": True,
                "is_null_condition": True,
                "is_null_condition_branch_conditional": True,
                "is_null_condition_marginal": True,
                "is_null_fate_selection": not has_branch_composition_contrast,
                "branch_composition_contrast": has_branch_composition_contrast,
                "is_null_branch": True,
                "is_null_trajectory": True,
                "effect_direction": "none",
                "control_onset": np.nan,
                "case_onset": np.nan,
                "activation_delay": np.nan,
                "onset_definition": "not_applicable",
                "delay_definition": "not_applicable",
                "delay_sign": "not_applicable",
                "time_scale": "pseudotime_true",
                "estimand_scope": "population_fixed_effect",
                "reference_condition": "control",
                "query_condition": "case",
                "active_branch": "none",
                "branch_point": branch_point,
                "effect_size": 0.0,
                "expected_effect_fraction": 0.0,
                "overlap_with_primary": 0.0,
            },
        ]
    )
    donor_truth = pd.DataFrame(donor_rows)
    config = {
        "seed": int(seed),
        "conditions": list(conditions),
        "n_donors_per_condition": list(donor_counts),
        "cells_per_donor": list(cell_means),
        "cell_count_cv": float(cell_count_cv),
        "n_genes": int(n_genes),
        "pathway_size": int(pathway_size),
        "overlap_fraction_requested": float(overlap_fraction),
        "overlap_fraction_realized": float(realized_overlap),
        "control_onset": float(control_onset),
        "case_onset": float(case_onset),
        "activation_delay": float(case_onset - control_onset),
        "branch_point": float(branch_point),
        "active_branch": active_branch,
        "effect_size": float(effect_size),
        "activation_width": float(activation_width),
        "branch_a_probabilities": [
            float(branch_a_probability_control),
            float(branch_a_probability_case),
        ],
        "branch_composition_contrast": bool(has_branch_composition_contrast),
        "donor_intercept_sd": float(donor_intercept_sd),
        "donor_slope_sd": float(donor_slope_sd),
        "donor_time_shift_sd": float(donor_time_shift_sd),
        "pseudotime_noise_sd": float(pseudotime_noise_sd),
        "n_pseudotime_draws": int(n_pseudotime_draws),
        "pseudotime_draw_definition": "independent_true_time_plus_donor_shift_draws",
        "lineage_probability_noise_sd": float(lineage_probability_noise_sd),
        "within_pathway_correlation": float(within_pathway_correlation),
        "noise_sd": float(noise_sd),
        "batch_effect_size": float(batch_effect_size),
        "dropout_rate": float(dropout_rate),
        "onset_definition": "logistic_50pct_midpoint",
        "delay_definition": "case_midpoint_minus_control_midpoint",
        "delay_sign": "case_minus_control",
        "time_scale": "pseudotime_true",
    }
    adata.uns["trajectory_simulation"] = config.copy()
    adata.uns["trajectory_simulation"]["pathway_truth_json"] = pathway_truth.to_json(
        orient="records"
    )

    return DonorTrajectorySimulation(
        adata=adata,
        gene_sets=gene_sets,
        pathway_truth=pathway_truth,
        donor_truth=donor_truth,
        config=config,
    )
