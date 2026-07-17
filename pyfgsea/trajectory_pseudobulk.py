from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
import hashlib
import json
import math
import os
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .validation import _expression_matrix
from .wrapper import load_gmt


@dataclass
class FixedGridDonorPseudobulkResult:
    """Auditable tables from a fixed-grid donor-level pathway test."""

    pathway_tests: pd.DataFrame
    effect_curves: pd.DataFrame
    pseudobulk_adata: Any
    donor_bin_activity: pd.DataFrame
    grid_diagnostics: pd.DataFrame
    donor_design: pd.DataFrame
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
            "donor_design": self.donor_design.copy(),
            "pathway_membership": self.pathway_membership.copy(),
            "permutation_summary": self.permutation_summary.copy(),
            "permutation_assignments": self.permutation_assignments.copy(),
            "null_statistics": self.null_statistics.copy(),
        }


@dataclass
class _PermutationPlan:
    null_assignments: list[np.ndarray]
    permutation_space_size: int
    requested_mode: str
    actual_mode: str
    is_exact: bool
    draw_attempts: int
    duplicate_draws: int


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    values = p[finite]
    order = np.argsort(values)
    ranked = values[order]
    n_tests = len(ranked)
    adjusted = ranked * n_tests / np.arange(1, n_tests + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[finite] = restored
    return out


def _by_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    finite_count = int(np.isfinite(p).sum())
    if finite_count == 0:
        return np.full(len(p), np.nan, dtype=float)
    harmonic = float(np.sum(1.0 / np.arange(1, finite_count + 1)))
    return np.minimum(_bh_adjust(p) * harmonic, 1.0)


def _normalize_statistic(statistic: str) -> str:
    normalized = str(statistic).lower().replace("-", "_")
    aliases = {
        "max_abs": "max_absolute_effect",
        "max_abs_effect": "max_absolute_effect",
        "integrated_abs": "integrated_absolute_effect",
        "auc_abs": "integrated_absolute_effect",
        "l2": "l2_effect",
        "signed_auc": "signed_integral",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {
        "max_absolute_effect",
        "integrated_absolute_effect",
        "l2_effect",
        "signed_integral",
    }
    if normalized not in allowed:
        raise ValueError(f"statistic must be one of {sorted(allowed)}")
    return normalized


def _normalize_tail(tail: str) -> str:
    normalized = str(tail).lower().replace("-", "_")
    aliases = {"two_sided": "two_sided", "two_tailed": "two_sided"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"greater", "less", "two_sided"}:
        raise ValueError("tail must be 'greater', 'less', or 'two_sided'")
    return normalized


def _require_integer(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _test_scale(values: np.ndarray, statistic: str, tail: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if statistic != "signed_integral":
        if tail != "greater":
            raise ValueError(
                "Magnitude statistics already encode a two-sided effect and require "
                "tail='greater'"
            )
        return values
    if tail == "greater":
        return values
    if tail == "less":
        return -values
    return np.abs(values)


def _fixed_edges(
    grid_edges: Optional[Sequence[float]],
    n_bins: int,
    pseudotime_range: Tuple[float, float],
) -> np.ndarray:
    if grid_edges is None:
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        if len(pseudotime_range) != 2:
            raise ValueError("pseudotime_range must contain two values")
        lower, upper = map(float, pseudotime_range)
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError("pseudotime_range must be finite and strictly increasing")
        edges = np.linspace(lower, upper, int(n_bins) + 1)
    else:
        edges = np.asarray(list(grid_edges), dtype=float)
        if edges.ndim != 1 or len(edges) < 3:
            raise ValueError("grid_edges must contain at least three values")
        if not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
            raise ValueError("grid_edges must be finite and strictly increasing")
    return edges


def _assign_fixed_bins(pseudotime: np.ndarray, edges: np.ndarray) -> np.ndarray:
    values = np.asarray(pseudotime, dtype=float)
    bins = np.searchsorted(edges, values, side="right") - 1
    bins[values == edges[-1]] = len(edges) - 2
    outside = (values < edges[0]) | (values > edges[-1])
    bins[outside] = -1
    return bins.astype(int)


def _load_pathway_mapping(gene_sets) -> Mapping:
    if isinstance(gene_sets, (str, os.PathLike)):
        return load_gmt(str(gene_sets))
    if not isinstance(gene_sets, Mapping):
        raise TypeError("gene_sets must be a GMT path or a pathway mapping")
    return gene_sets


def _prepare_pathways(
    genes: np.ndarray,
    gene_sets,
    min_size: int,
    max_size: int,
) -> tuple[list[str], list[np.ndarray], list[np.ndarray], pd.DataFrame]:
    if min_size < 1:
        raise ValueError("min_size must be positive")
    if max_size < min_size:
        raise ValueError("max_size must be at least min_size")
    gene_names = [str(gene) for gene in genes]
    if len(set(gene_names)) != len(gene_names):
        raise ValueError("Expression gene names must be unique")
    gene_to_index = {gene: idx for idx, gene in enumerate(gene_names)}
    raw = _load_pathway_mapping(gene_sets)

    names = []
    indices = []
    weights = []
    membership_rows = []
    normalized_pathway_names = set()
    for pathway, members in raw.items():
        pathway_name = str(pathway)
        if pathway_name in normalized_pathway_names:
            raise ValueError(
                "Pathway names must remain unique after string normalization; "
                f"duplicate '{pathway_name}'"
            )
        normalized_pathway_names.add(pathway_name)
        if isinstance(members, Mapping):
            raw_members = [(str(gene), float(weight)) for gene, weight in members.items()]
        else:
            if isinstance(members, (str, bytes)):
                raise TypeError(
                    f"Pathway '{pathway}' members must be a gene sequence, not a string"
                )
            raw_members = []
            for item in members:
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    raw_members.append((str(item[0]), float(item[1])))
                else:
                    raw_members.append((str(item), 1.0))
        seen = set()
        matched = []
        for gene, weight in raw_members:
            if gene in seen or gene not in gene_to_index:
                continue
            seen.add(gene)
            if not np.isfinite(weight) or weight == 0:
                continue
            matched.append((gene, gene_to_index[gene], weight))
        if not min_size <= len(matched) <= max_size:
            continue
        names.append(pathway_name)
        indices.append(np.asarray([item[1] for item in matched], dtype=int))
        weights.append(np.asarray([item[2] for item in matched], dtype=float))
        for gene, gene_index, weight in matched:
            membership_rows.append(
                {
                    "Pathway": pathway_name,
                    "gene": gene,
                    "gene_index": int(gene_index),
                    "weight": float(weight),
                    "pathway_size": int(len(matched)),
                }
            )
    if not names:
        raise ValueError(
            "No pathways remain after gene matching and min_size/max_size filtering"
        )
    return names, indices, weights, pd.DataFrame(membership_rows)


def _validate_stringification_is_injective(series: pd.Series, key: str) -> None:
    seen = {}
    for value in pd.unique(series.dropna()):
        normalized = str(value)
        signature = (
            type(value).__module__,
            type(value).__qualname__,
            repr(value),
        )
        if normalized in seen and seen[normalized] != signature:
            raise ValueError(
                f"Column '{key}' contains distinct values that both normalize to "
                f"'{normalized}'; use an unambiguous donor/design identifier"
            )
        seen[normalized] = signature


def _validate_design(
    adata,
    condition_key: str,
    donor_key: str,
    control,
    case,
    pseudotime_key: str,
    strata_keys: Sequence[str],
    min_donors_per_condition: int,
    drop_noninformative_strata: bool,
) -> tuple[pd.DataFrame, np.ndarray, list[str], np.ndarray, list[np.ndarray]]:
    if str(control) == str(case):
        raise ValueError("control and case must be different")
    if min_donors_per_condition < 1:
        raise ValueError("min_donors_per_condition must be positive")
    required = [condition_key, donor_key, pseudotime_key, *strata_keys]
    missing = [key for key in required if key not in adata.obs]
    if missing:
        raise KeyError(f"Missing adata.obs columns: {missing}")
    if adata.obs[condition_key].isna().any():
        raise ValueError(f"condition_key '{condition_key}' contains missing values")

    condition_all = adata.obs[condition_key].astype(str)
    selected_mask = condition_all.isin([str(control), str(case)]).to_numpy()
    if not selected_mask.any():
        raise ValueError("No cells match the requested control/case conditions")
    selected = adata.obs.loc[selected_mask, required].copy()
    if selected[[donor_key, pseudotime_key, *strata_keys]].isna().any().any():
        raise ValueError("Selected cells contain missing donor, pseudotime, or strata values")
    numeric_pt = pd.to_numeric(selected[pseudotime_key], errors="coerce")
    if not np.isfinite(numeric_pt.to_numpy(dtype=float)).all():
        raise ValueError(f"pseudotime_key '{pseudotime_key}' must be finite numeric")
    selected[pseudotime_key] = numeric_pt
    for key in (condition_key, donor_key, *strata_keys):
        _validate_stringification_is_injective(selected[key], key)
    selected[condition_key] = selected[condition_key].astype(str)
    selected[donor_key] = selected[donor_key].astype(str)
    for key in strata_keys:
        selected[key] = selected[key].astype(str)

    donor_rows = []
    for donor, group in selected.groupby(donor_key, sort=True):
        conditions = sorted(group[condition_key].unique())
        if len(conditions) != 1:
            raise ValueError(
                "Each donor must map to exactly one condition. "
                f"Donor '{donor}' maps to {conditions}"
            )
        stratum_values = []
        for key in strata_keys:
            values = sorted(group[key].unique())
            if len(values) != 1:
                raise ValueError(
                    f"Restriction key '{key}' is not donor-constant for donor '{donor}'"
                )
            stratum_values.append(values[0])
        stratum_key = tuple(stratum_values) if strata_keys else ("__all__",)
        stratum = (
            json.dumps(stratum_values, ensure_ascii=False, separators=(",", ":"))
            if strata_keys
            else "__all__"
        )
        donor_row = {
            "donor": str(donor),
            "observed_condition": conditions[0],
            "stratum": stratum,
            "__stratum_key": stratum_key,
            "n_cells_selected": int(len(group)),
        }
        donor_row.update(
            {key: value for key, value in zip(strata_keys, stratum_values)}
        )
        donor_rows.append(donor_row)
    donor_design = pd.DataFrame(donor_rows)
    informative = (
        donor_design.groupby("__stratum_key")["observed_condition"]
        .nunique()
        .loc[lambda values: values == 2]
        .index
    )
    donor_design["included_in_inference"] = donor_design["__stratum_key"].isin(
        informative
    )
    donor_design["exclusion_reason"] = np.where(
        donor_design["included_in_inference"],
        "",
        "noninformative_single_condition_stratum",
    )
    noninformative = donor_design[~donor_design["included_in_inference"]]
    if not noninformative.empty and not drop_noninformative_strata:
        examples = ", ".join(noninformative["stratum"].drop_duplicates().head(5))
        raise ValueError(
            "Restricted design contains single-condition strata with no "
            "within-stratum exchangeability. Set drop_noninformative_strata=True "
            f"to target the informative-strata subset explicitly. Examples: {examples}"
        )
    included = donor_design[donor_design["included_in_inference"]].copy()
    if included.empty:
        raise ValueError(
            "No exchangeable donor strata contain both control and case; "
            "the restricted permutation space is nonidentifiable"
        )
    condition_counts = included["observed_condition"].value_counts()
    for condition in (str(control), str(case)):
        if int(condition_counts.get(condition, 0)) < min_donors_per_condition:
            raise ValueError(
                f"Condition '{condition}' has fewer than "
                f"min_donors_per_condition={min_donors_per_condition} included donors"
            )

    donor_names = sorted(included["donor"].tolist())
    donor_to_index = {donor: idx for idx, donor in enumerate(donor_names)}
    donor_design["donor_index"] = donor_design["donor"].map(donor_to_index)
    included = donor_design[donor_design["included_in_inference"]].copy()
    included = included.sort_values("donor_index")
    observed_case = (
        included["observed_condition"].to_numpy(dtype=str) == str(case)
    )
    stratum_groups = []
    stratum_weights = {}
    for stratum, group in included.groupby("__stratum_key", sort=True):
        group_indices = group["donor_index"].to_numpy(dtype=int)
        stratum_groups.append(group_indices)
        stratum_weights[stratum] = len(group_indices) / len(included)
    donor_design["stratum_weight"] = donor_design["__stratum_key"].map(
        stratum_weights
    )

    selected_original_indices = np.where(selected_mask)[0]
    selected_donors = selected[donor_key].to_numpy(dtype=str)
    included_cell_mask = np.isin(selected_donors, donor_names)
    cell_indices = selected_original_indices[included_cell_mask]
    cell_frame = selected.iloc[np.where(included_cell_mask)[0]].copy()
    cell_frame["__donor_index"] = cell_frame[donor_key].map(donor_to_index).astype(int)
    cell_frame["__original_index"] = cell_indices

    included_lookup = included.set_index("donor")
    donor_design.loc[donor_design["included_in_inference"], "observed_case"] = (
        donor_design.loc[donor_design["included_in_inference"], "donor"]
        .map(included_lookup["observed_condition"])
        .eq(str(case))
    )
    return donor_design, observed_case, donor_names, cell_frame, stratum_groups


def _permutation_space_size(
    observed_case: np.ndarray,
    stratum_groups: list[np.ndarray],
) -> int:
    total = 1
    for indices in stratum_groups:
        k_case = int(observed_case[indices].sum())
        total *= math.comb(len(indices), k_case)
    return int(total)


def _enumerate_assignments(
    n_donors: int,
    observed_case: np.ndarray,
    stratum_groups: list[np.ndarray],
) -> list[np.ndarray]:
    choices = []
    for indices in stratum_groups:
        k_case = int(observed_case[indices].sum())
        choices.append(list(combinations(indices.tolist(), k_case)))
    assignments = []
    for selected_groups in product(*choices):
        assignment = np.zeros(n_donors, dtype=bool)
        for selected in selected_groups:
            assignment[list(selected)] = True
        assignments.append(assignment)
    return assignments


def _sample_unique_assignments(
    n_donors: int,
    observed_case: np.ndarray,
    stratum_groups: list[np.ndarray],
    target: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], int, int]:
    observed_key = observed_case.tobytes()
    seen = {observed_key}
    assignments = []
    attempts = 0
    duplicates = 0
    max_attempts = max(1000, target * 100)
    while len(assignments) < target and attempts < max_attempts:
        attempts += 1
        assignment = np.zeros(n_donors, dtype=bool)
        for indices in stratum_groups:
            k_case = int(observed_case[indices].sum())
            selected = rng.choice(indices, size=k_case, replace=False)
            assignment[selected] = True
        key = assignment.tobytes()
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        assignments.append(assignment)
    if len(assignments) < target:
        raise RuntimeError(
            "Could not obtain the requested number of unique Monte Carlo donor "
            "assignments; reduce n_permutations or use exact mode"
        )
    return assignments, attempts, duplicates


def _make_permutation_plan(
    observed_case: np.ndarray,
    stratum_groups: list[np.ndarray],
    permutation_mode: str,
    n_permutations: int,
    max_exact_permutations: int,
    seed: int,
) -> _PermutationPlan:
    requested = str(permutation_mode).lower().replace("-", "_")
    if requested not in {"auto", "exact", "monte_carlo"}:
        raise ValueError("permutation_mode must be 'auto', 'exact', or 'monte_carlo'")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    if max_exact_permutations < 2:
        raise ValueError("max_exact_permutations must be at least 2")
    total = _permutation_space_size(observed_case, stratum_groups)
    if total <= 1:
        raise ValueError(
            "The restricted donor-label permutation space has size 1; condition "
            "is not exchangeable under the requested strata"
        )

    use_exact = requested == "exact" or (
        requested == "auto" and total <= max_exact_permutations
    )
    exhaustive_mc = (
        requested == "monte_carlo"
        and total <= max_exact_permutations
        and n_permutations >= total - 1
    )
    use_exact = use_exact or exhaustive_mc
    if use_exact:
        if total > max_exact_permutations:
            raise ValueError(
                f"Exact permutation space has {total} assignments, exceeding "
                f"max_exact_permutations={max_exact_permutations}"
            )
        all_assignments = _enumerate_assignments(
            len(observed_case), observed_case, stratum_groups
        )
        null_assignments = [
            assignment
            for assignment in all_assignments
            if not np.array_equal(assignment, observed_case)
        ]
        actual = "exact_exhaustive" if exhaustive_mc else "exact"
        return _PermutationPlan(
            null_assignments=null_assignments,
            permutation_space_size=total,
            requested_mode=requested,
            actual_mode=actual,
            is_exact=True,
            draw_attempts=0,
            duplicate_draws=0,
        )

    target = min(int(n_permutations), total - 1)
    rng = np.random.default_rng(seed)
    assignments, attempts, duplicates = _sample_unique_assignments(
        len(observed_case), observed_case, stratum_groups, target, rng
    )
    exhaustive = target == total - 1
    return _PermutationPlan(
        null_assignments=assignments,
        permutation_space_size=total,
        requested_mode=requested,
        actual_mode=(
            "exact_exhaustive_via_unique_sampling"
            if exhaustive
            else "monte_carlo"
        ),
        is_exact=exhaustive,
        draw_attempts=attempts,
        duplicate_draws=duplicates,
    )


def _support_segments(supported: np.ndarray) -> tuple[np.ndarray, Optional[int]]:
    supported = np.asarray(supported, dtype=bool)
    segment_ids = np.full(len(supported), -1, dtype=int)
    segments = []
    start = None
    segment_id = -1
    for idx, value in enumerate(supported):
        if value and start is None:
            start = idx
            segment_id += 1
        if value:
            segment_ids[idx] = segment_id
        if start is not None and (not value or idx == len(supported) - 1):
            stop = idx if not value else idx + 1
            segments.append((segment_id, start, stop))
            start = None
    if not segments:
        return segment_ids, None
    selected = sorted(segments, key=lambda item: (-(item[2] - item[1]), item[1]))[0][0]
    return segment_ids, int(selected)


def _build_grid_support(
    cell_frame: pd.DataFrame,
    donor_names: list[str],
    donor_key: str,
    condition_key: str,
    control,
    case,
    pseudotime_key: str,
    edges: np.ndarray,
    observed_case: np.ndarray,
    stratum_groups: list[np.ndarray],
    min_cells_per_donor_bin: int,
    min_donors_per_condition: int,
    min_common_bins: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    if min_cells_per_donor_bin < 1:
        raise ValueError("min_cells_per_donor_bin must be positive")
    if min_common_bins < 1:
        raise ValueError("min_common_bins must be positive")
    bin_ids = _assign_fixed_bins(
        cell_frame[pseudotime_key].to_numpy(dtype=float), edges
    )
    n_bins = len(edges) - 1
    counts = np.zeros((len(donor_names), n_bins), dtype=int)
    valid = bin_ids >= 0
    np.add.at(
        counts,
        (
            cell_frame.loc[valid, "__donor_index"].to_numpy(dtype=int),
            bin_ids[valid],
        ),
        1,
    )
    available = counts >= int(min_cells_per_donor_bin)
    observed_control = ~observed_case
    rows = []
    supported = np.zeros(n_bins, dtype=bool)
    for bin_id in range(n_bins):
        min_case_total = 0
        min_control_total = 0
        every_stratum_identifiable = True
        for indices in stratum_groups:
            n_stratum = len(indices)
            k_case = int(observed_case[indices].sum())
            available_count = int(available[indices, bin_id].sum())
            missing_count = n_stratum - available_count
            min_case = max(0, k_case - missing_count)
            min_control = max(0, (n_stratum - k_case) - missing_count)
            min_case_total += min_case
            min_control_total += min_control
            every_stratum_identifiable &= min_case >= 1 and min_control >= 1
        gate = (
            min_case_total >= min_donors_per_condition
            and min_control_total >= min_donors_per_condition
            and every_stratum_identifiable
        )
        supported[bin_id] = gate
        rows.append(
            {
                "bin_id": int(bin_id),
                "bin_left": float(edges[bin_id]),
                "bin_right": float(edges[bin_id + 1]),
                "bin_mid": float((edges[bin_id] + edges[bin_id + 1]) / 2.0),
                "bin_width": float(edges[bin_id + 1] - edges[bin_id]),
                "n_cells": int(counts[:, bin_id].sum()),
                "min_cells_any_donor": int(counts[:, bin_id].min()),
                "median_cells_per_donor": float(np.median(counts[:, bin_id])),
                "n_donors_available": int(available[:, bin_id].sum()),
                "n_control_available_observed": int(
                    available[observed_control, bin_id].sum()
                ),
                "n_case_available_observed": int(
                    available[observed_case, bin_id].sum()
                ),
                "worst_case_control_donors": int(min_control_total),
                "worst_case_case_donors": int(min_case_total),
                "all_strata_identifiable": bool(every_stratum_identifiable),
                "permutation_invariant_support_gate": bool(gate),
            }
        )
    diagnostics = pd.DataFrame(rows)
    segment_ids, selected_segment = _support_segments(supported)
    diagnostics["support_segment_id"] = np.where(segment_ids >= 0, segment_ids, np.nan)
    selected = (
        supported & (segment_ids == selected_segment)
        if selected_segment is not None
        else np.zeros(n_bins, dtype=bool)
    )
    diagnostics["selected_common_support"] = selected
    diagnostics["drop_reason"] = np.select(
        [selected, supported],
        ["", "outside_longest_supported_segment"],
        default="permutation_space_support_failure",
    )
    selected_bins = np.where(selected)[0]
    if len(selected_bins) < min_common_bins:
        supported_ids = np.where(supported)[0].tolist()
        raise ValueError(
            "Insufficient contiguous common support: longest valid segment has "
            f"{len(selected_bins)} bins, requires {min_common_bins}; "
            f"permutation-invariant supported bins={supported_ids}"
        )
    return diagnostics, counts, available, selected_bins


def _aggregate_donor_bins(
    X,
    cell_frame: pd.DataFrame,
    pseudotime_key: str,
    gene_indices: np.ndarray,
    selected_bins: np.ndarray,
    edges: np.ndarray,
    n_donors: int,
    min_cells_per_donor_bin: int,
) -> tuple[np.ndarray, np.ndarray]:
    cell_bins = _assign_fixed_bins(
        cell_frame[pseudotime_key].to_numpy(dtype=float),
        edges,
    )
    n_genes = len(gene_indices)
    pseudobulk = np.full(
        (n_donors, len(selected_bins), n_genes), np.nan, dtype=float
    )
    counts = np.zeros((n_donors, len(selected_bins)), dtype=int)
    for donor_index in range(n_donors):
        donor_mask = cell_frame["__donor_index"].to_numpy(dtype=int) == donor_index
        for local_bin, bin_id in enumerate(selected_bins):
            mask = donor_mask & (cell_bins == int(bin_id))
            original_indices = cell_frame.loc[mask, "__original_index"].to_numpy(dtype=int)
            counts[donor_index, local_bin] = len(original_indices)
            if len(original_indices) < min_cells_per_donor_bin:
                continue
            block = X[original_indices]
            block = block[:, gene_indices]
            mean = np.asarray(block.mean(axis=0)).ravel()
            if not np.isfinite(mean).all():
                raise ValueError("Non-finite donor-bin pseudobulk expression encountered")
            pseudobulk[donor_index, local_bin] = mean
    return pseudobulk, counts


def _score_pathways(
    pseudobulk: np.ndarray,
    pathway_indices: list[np.ndarray],
    pathway_weights: list[np.ndarray],
) -> np.ndarray:
    flat = pseudobulk.reshape(-1, pseudobulk.shape[-1])
    center = np.nanmean(flat, axis=0)
    scale = np.nanstd(flat, axis=0, ddof=1)
    scale_floor = 1e-6 * np.maximum(np.abs(center), 1.0)
    near_constant = ~np.isfinite(scale) | (scale <= scale_floor)
    scale[near_constant] = 1.0
    standardized = (pseudobulk - center[None, None, :]) / scale[None, None, :]
    standardized[:, :, near_constant] = 0.0
    scores = np.full(
        (pseudobulk.shape[0], pseudobulk.shape[1], len(pathway_indices)),
        np.nan,
        dtype=float,
    )
    available = np.isfinite(pseudobulk).all(axis=2)
    for pathway_index, (indices, weights) in enumerate(
        zip(pathway_indices, pathway_weights)
    ):
        denominator = float(np.abs(weights).sum())
        values = np.tensordot(
            standardized[:, :, indices], weights, axes=([2], [0])
        ) / denominator
        values[~available] = np.nan
        scores[:, :, pathway_index] = values
    return scores


def _stratified_condition_curves(
    scores: np.ndarray,
    case_assignment: np.ndarray,
    stratum_groups: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_donors = scores.shape[0]
    control_curve = np.zeros(scores.shape[1:], dtype=float)
    case_curve = np.zeros(scores.shape[1:], dtype=float)
    for indices in stratum_groups:
        weight = len(indices) / n_donors
        for bin_index in range(scores.shape[1]):
            available = np.isfinite(scores[indices, bin_index, 0])
            case_mask = case_assignment[indices] & available
            control_mask = (~case_assignment[indices]) & available
            if not case_mask.any() or not control_mask.any():
                raise RuntimeError(
                    "Calibration failure: a frozen common-support bin became "
                    "unidentified under a legal donor assignment"
                )
            case_values = scores[indices[case_mask], bin_index]
            control_values = scores[indices[control_mask], bin_index]
            case_curve[bin_index] += weight * np.mean(case_values, axis=0)
            control_curve[bin_index] += weight * np.mean(control_values, axis=0)
    delta = case_curve - control_curve
    if not np.isfinite(delta).all():
        raise RuntimeError("Calibration failure: non-finite pathway contrast")
    return control_curve, case_curve, delta


def _curve_statistics(delta: np.ndarray, widths: np.ndarray) -> Dict[str, np.ndarray]:
    absolute = np.abs(delta)
    peak_indices = np.argmax(absolute, axis=0)
    pathway_indices = np.arange(delta.shape[1])
    return {
        "max_absolute_effect": np.max(absolute, axis=0),
        "integrated_absolute_effect": np.sum(absolute * widths[:, None], axis=0),
        "l2_effect": np.sqrt(np.sum(np.square(delta) * widths[:, None], axis=0)),
        "signed_integral": np.sum(delta * widths[:, None], axis=0),
        "peak_bin_local": peak_indices.astype(int),
        "peak_effect": delta[peak_indices, pathway_indices],
    }


def _assignment_hash(assignment: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(assignment, dtype=np.uint8).tobytes()).hexdigest()[:16]


def _h5ad_safe_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, int) and not -(2**63) <= value < 2**63:
        return str(value)
    if isinstance(value, tuple):
        return [_h5ad_safe_value(item) for item in value]
    if isinstance(value, list):
        return [_h5ad_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _h5ad_safe_value(item) for key, item in value.items()}
    return value


def run_fixed_grid_donor_pseudobulk(
    adata,
    gene_sets,
    *,
    condition_key: str,
    donor_key: str,
    control,
    case,
    pseudotime_key: str = "dpt_pseudotime",
    grid_edges: Optional[Sequence[float]] = None,
    n_bins: int = 8,
    pseudotime_range: Tuple[float, float] = (0.0, 1.0),
    min_cells_per_donor_bin: int = 5,
    min_donors_per_condition: int = 3,
    min_common_bins: int = 2,
    strata_keys: Sequence[str] = (),
    drop_noninformative_strata: bool = False,
    statistic: str = "max_absolute_effect",
    tail: str = "greater",
    permutation_mode: str = "auto",
    n_permutations: int = 999,
    max_exact_permutations: int = 20000,
    min_size: int = 5,
    max_size: int = 500,
    layer: Optional[str] = None,
    use_raw: bool = False,
    alpha: float = 0.05,
    seed: int = 42,
    return_null_statistics: bool = False,
    return_permutation_assignments: bool = False,
    return_donor_bin_activity: bool = False,
    retain_all_genes: bool = False,
) -> FixedGridDonorPseudobulkResult:
    """Test pathway curve differences on a fixed donor-by-pseudotime grid.

    Cells are averaged once within each donor and fixed pseudotime bin. Pathway
    activity is the weighted mean of gene z-scores across donor-bin
    pseudobulks. The complete donor trajectory receives one condition label in
    every assignment. Small assignment spaces are enumerated exactly; larger
    spaces use unique Monte Carlo assignments. When ``strata_keys`` are given,
    condition counts are preserved within their joint donor-level strata.

    The returned p-values are conditional on the supplied pseudotime and on
    donor-label exchangeability within the declared strata.
    """
    statistic = _normalize_statistic(statistic)
    tail = _normalize_tail(tail)
    _test_scale(np.asarray([0.0]), statistic, tail)
    n_bins = _require_integer(n_bins, "n_bins", 2)
    min_cells_per_donor_bin = _require_integer(
        min_cells_per_donor_bin, "min_cells_per_donor_bin"
    )
    min_donors_per_condition = _require_integer(
        min_donors_per_condition, "min_donors_per_condition"
    )
    min_common_bins = _require_integer(min_common_bins, "min_common_bins")
    n_permutations = _require_integer(n_permutations, "n_permutations")
    max_exact_permutations = _require_integer(
        max_exact_permutations, "max_exact_permutations", 2
    )
    min_size = _require_integer(min_size, "min_size")
    max_size = _require_integer(max_size, "max_size")
    seed = _require_integer(seed, "seed", 0)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if isinstance(strata_keys, (str, bytes)):
        raise ValueError("strata_keys must be a sequence of column names")
    strata_keys = tuple(str(key) for key in strata_keys)
    if len(set(strata_keys)) != len(strata_keys):
        raise ValueError("strata_keys must be unique")
    design_keys = [condition_key, donor_key, pseudotime_key, *strata_keys]
    if len(set(design_keys)) != len(design_keys):
        raise ValueError(
            "condition_key, donor_key, pseudotime_key, and strata_keys must "
            "refer to distinct columns"
        )
    if layer is not None and use_raw:
        raise ValueError("layer and use_raw=True are mutually exclusive")

    edges = _fixed_edges(grid_edges, n_bins, pseudotime_range)
    (
        donor_design,
        observed_case,
        donor_names,
        cell_frame,
        stratum_groups,
    ) = _validate_design(
        adata,
        condition_key=condition_key,
        donor_key=donor_key,
        control=control,
        case=case,
        pseudotime_key=pseudotime_key,
        strata_keys=strata_keys,
        min_donors_per_condition=min_donors_per_condition,
        drop_noninformative_strata=drop_noninformative_strata,
    )
    (
        grid_diagnostics,
        all_bin_counts,
        all_bin_available,
        selected_bins,
    ) = _build_grid_support(
        cell_frame,
        donor_names=donor_names,
        donor_key=donor_key,
        condition_key=condition_key,
        control=control,
        case=case,
        pseudotime_key=pseudotime_key,
        edges=edges,
        observed_case=observed_case,
        stratum_groups=stratum_groups,
        min_cells_per_donor_bin=min_cells_per_donor_bin,
        min_donors_per_condition=min_donors_per_condition,
        min_common_bins=min_common_bins,
    )
    plan = _make_permutation_plan(
        observed_case,
        stratum_groups,
        permutation_mode=permutation_mode,
        n_permutations=n_permutations,
        max_exact_permutations=max_exact_permutations,
        seed=seed,
    )

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
        int(source_index): local_index
        for local_index, source_index in enumerate(pseudobulk_gene_indices)
    }
    local_pathway_indices = [
        np.asarray(
            [source_to_pseudobulk[int(source_index)] for source_index in indices],
            dtype=int,
        )
        for indices in pathway_indices
    ]
    pathway_membership["pseudobulk_gene_index"] = pathway_membership[
        "gene_index"
    ].map(source_to_pseudobulk).astype(int)

    pseudobulk, tested_counts = _aggregate_donor_bins(
        X,
        cell_frame,
        pseudotime_key=pseudotime_key,
        gene_indices=pseudobulk_gene_indices,
        selected_bins=selected_bins,
        edges=edges,
        n_donors=len(donor_names),
        min_cells_per_donor_bin=min_cells_per_donor_bin,
    )
    if not np.array_equal(tested_counts, all_bin_counts[:, selected_bins]):
        raise RuntimeError("Internal donor-bin aggregation count mismatch")
    scores = _score_pathways(
        pseudobulk, local_pathway_indices, pathway_weights
    )
    control_curve, case_curve, observed_delta = _stratified_condition_curves(
        scores, observed_case, stratum_groups
    )
    widths = np.diff(edges)[selected_bins]
    observed_stats = _curve_statistics(observed_delta, widths)
    observed_raw_stat = observed_stats[statistic]
    observed_test_stat = _test_scale(observed_raw_stat, statistic, tail)
    observed_point_scale = (
        np.abs(observed_delta)
        if statistic != "signed_integral" or tail == "two_sided"
        else observed_delta if tail == "greater" else -observed_delta
    )

    pathway_exceedances = np.zeros(len(pathway_names), dtype=int)
    pathway_max_t_exceedances = np.zeros(len(pathway_names), dtype=int)
    pointwise_exceedances = np.zeros(observed_delta.shape, dtype=int)
    within_pathway_exceedances = np.zeros(observed_delta.shape, dtype=int)
    global_curve_exceedances = np.zeros(observed_delta.shape, dtype=int)
    null_rows = []
    tolerance = 1e-12
    for perm_id, assignment in enumerate(plan.null_assignments):
        _, _, null_delta = _stratified_condition_curves(
            scores, assignment, stratum_groups
        )
        null_stats = _curve_statistics(null_delta, widths)
        null_raw_stat = null_stats[statistic]
        null_test_stat = _test_scale(null_raw_stat, statistic, tail)
        pathway_exceedances += null_test_stat >= (observed_test_stat - tolerance)
        max_pathway_stat = float(np.max(null_test_stat))
        pathway_max_t_exceedances += max_pathway_stat >= (
            observed_test_stat - tolerance
        )

        null_point_scale = (
            np.abs(null_delta)
            if statistic != "signed_integral" or tail == "two_sided"
            else null_delta if tail == "greater" else -null_delta
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
            assignment_hash = _assignment_hash(assignment)
            for pathway_index, pathway in enumerate(pathway_names):
                null_rows.append(
                    {
                        "perm_id": int(perm_id),
                        "assignment_hash": assignment_hash,
                        "Pathway": pathway,
                        "raw_statistic": float(null_raw_stat[pathway_index]),
                        "calibration_statistic": float(
                            null_test_stat[pathway_index]
                        ),
                        "max_pathway_calibration_statistic": max_pathway_stat,
                    }
                )

    n_null = len(plan.null_assignments)
    denominator = n_null + 1
    p_raw = (pathway_exceedances + 1.0) / denominator
    p_max_t = (pathway_max_t_exceedances + 1.0) / denominator
    q_bh = _bh_adjust(p_raw)
    q_by = _by_adjust(p_raw)
    pointwise_p = (pointwise_exceedances + 1.0) / denominator
    within_pathway_p = (within_pathway_exceedances + 1.0) / denominator
    global_curve_p = (global_curve_exceedances + 1.0) / denominator
    permutation_p_resolution = 1.0 / denominator
    complement_symmetry_applies = bool(
        plan.is_exact
        and (statistic != "signed_integral" or tail == "two_sided")
        and all(
            2 * int(observed_case[indices].sum()) == len(indices)
            for indices in stratum_groups
        )
    )
    minimum_p = min(
        1.0,
        (2.0 if complement_symmetry_applies else 1.0) / denominator,
    )
    bonferroni_floor = min(1.0, len(pathway_names) * minimum_p)
    single_discovery_resolution_limited = bonferroni_floor > alpha
    restricted = bool(strata_keys)
    calibration_status = (
        "calibrated_restricted_exact"
        if restricted and plan.is_exact
        else "calibrated_exact"
        if plan.is_exact
        else "calibrated_restricted_monte_carlo"
        if restricted
        else "calibrated_monte_carlo"
    )

    pathway_sizes = (
        pathway_membership.groupby("Pathway").size().reindex(pathway_names).to_numpy()
    )
    peak_local = observed_stats["peak_bin_local"].astype(int)
    pathway_rows = []
    null_model = (
        "whole_donor_label_permutation_within_strata"
        if restricted
        else "whole_donor_label_permutation"
    )
    for pathway_index, pathway in enumerate(pathway_names):
        peak_bin = int(selected_bins[peak_local[pathway_index]])
        pathway_rows.append(
            {
                "Pathway": pathway,
                "pathway_size": int(pathway_sizes[pathway_index]),
                "primary_statistic": statistic,
                "tail": tail,
                "observed_statistic": float(observed_raw_stat[pathway_index]),
                "calibration_statistic": float(
                    observed_test_stat[pathway_index]
                ),
                "max_absolute_effect": float(
                    observed_stats["max_absolute_effect"][pathway_index]
                ),
                "integrated_absolute_effect": float(
                    observed_stats["integrated_absolute_effect"][pathway_index]
                ),
                "l2_effect": float(observed_stats["l2_effect"][pathway_index]),
                "signed_integral": float(
                    observed_stats["signed_integral"][pathway_index]
                ),
                "peak_effect": float(observed_stats["peak_effect"][pathway_index]),
                "peak_bin": peak_bin,
                "peak_time": float(
                    (edges[peak_bin] + edges[peak_bin + 1]) / 2.0
                ),
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
                "minimum_attainable_p": float(minimum_p),
                "permutation_p_resolution": float(permutation_p_resolution),
                "complement_symmetry_applies": complement_symmetry_applies,
                "bonferroni_resolution_floor": float(bonferroni_floor),
                "single_discovery_bonferroni_resolution_reachable": bool(
                    not single_discovery_resolution_limited
                ),
                "resolution_note": (
                    "single_minimum_p_cannot_pass_bonferroni_alpha"
                    if single_discovery_resolution_limited
                    else "single_minimum_p_can_pass_bonferroni_alpha"
                ),
                "calibration_warning": (
                    "Inference is conditional on fixed pseudotime and whole-donor "
                    "exchangeability within declared strata"
                    + (
                        "; estimand is restricted to informative strata after "
                        "explicitly dropping single-condition strata"
                        if (~donor_design["included_in_inference"]).any()
                        else ""
                    )
                ),
                "permutation_space_size": int(plan.permutation_space_size),
                "n_null_assignments_possible": int(
                    plan.permutation_space_size - 1
                ),
                "n_null_assignments_evaluated": int(n_null),
                "n_reference_assignments": int(denominator),
                "permutation_mode": plan.actual_mode,
                "restriction": "within_strata" if restricted else "unrestricted",
                "calibration_status": calibration_status,
                "calibrated_under_exchangeability_assumption": True,
                "reference": str(control),
                "query": str(case),
            }
        )
    pathway_tests = pd.DataFrame(pathway_rows).sort_values(
        ["event_fdr", "p_raw", "Pathway"]
    ).reset_index(drop=True)

    effect_rows = []
    for local_bin, bin_id in enumerate(selected_bins):
        for pathway_index, pathway in enumerate(pathway_names):
            effect_rows.append(
                {
                    "Pathway": pathway,
                    "bin_id": int(bin_id),
                    "bin_left": float(edges[bin_id]),
                    "bin_right": float(edges[bin_id + 1]),
                    "bin_mid": float((edges[bin_id] + edges[bin_id + 1]) / 2.0),
                    "bin_width": float(edges[bin_id + 1] - edges[bin_id]),
                    "control_activity": float(
                        control_curve[local_bin, pathway_index]
                    ),
                    "case_activity": float(case_curve[local_bin, pathway_index]),
                    "delta_activity": float(
                        observed_delta[local_bin, pathway_index]
                    ),
                    "pointwise_p": float(pointwise_p[local_bin, pathway_index]),
                    "within_pathway_maxT_p": float(
                        within_pathway_p[local_bin, pathway_index]
                    ),
                    "global_curve_maxT_p": float(
                        global_curve_p[local_bin, pathway_index]
                    ),
                }
            )
    effect_curves = pd.DataFrame(effect_rows)

    included_design = (
        donor_design[donor_design["included_in_inference"]]
        .sort_values("donor_index")
        .set_index("donor_index")
    )
    activity_rows = []
    selected_bin_to_local = {
        int(bin_id): local for local, bin_id in enumerate(selected_bins)
    }
    if return_donor_bin_activity:
        for donor_index, donor in enumerate(donor_names):
            donor_info = included_design.loc[donor_index]
            for bin_id in range(len(edges) - 1):
                is_tested = bin_id in selected_bin_to_local
                local_bin = selected_bin_to_local.get(bin_id)
                for pathway_index, pathway in enumerate(pathway_names):
                    activity = (
                        scores[donor_index, local_bin, pathway_index]
                        if is_tested
                        else np.nan
                    )
                    activity_rows.append(
                        {
                            "donor": donor,
                            "observed_condition": donor_info["observed_condition"],
                            "stratum": donor_info["stratum"],
                            "bin_id": int(bin_id),
                            "bin_left": float(edges[bin_id]),
                            "bin_right": float(edges[bin_id + 1]),
                            "bin_mid": float(
                                (edges[bin_id] + edges[bin_id + 1]) / 2.0
                            ),
                            "n_cells": int(all_bin_counts[donor_index, bin_id]),
                            "available": bool(
                                all_bin_available[donor_index, bin_id]
                            ),
                            "tested_bin": bool(is_tested),
                            "Pathway": pathway,
                            "activity": (
                                float(activity) if np.isfinite(activity) else np.nan
                            ),
                        }
                    )
    donor_bin_activity = pd.DataFrame(
        activity_rows,
        columns=[
            "donor",
            "observed_condition",
            "stratum",
            "bin_id",
            "bin_left",
            "bin_right",
            "bin_mid",
            "n_cells",
            "available",
            "tested_bin",
            "Pathway",
            "activity",
        ],
    )
    donor_design["n_cells_in_grid"] = donor_design["donor_index"].map(
        {
            index: int(all_bin_counts[index].sum())
            for index in range(len(donor_names))
        }
    )
    donor_design["n_tested_bins_available"] = donor_design["donor_index"].map(
        {
            index: int(all_bin_available[index, selected_bins].sum())
            for index in range(len(donor_names))
        }
    )

    assignment_rows = []
    if return_permutation_assignments:
        assignments_to_write = [(-1, observed_case, True)] + [
            (perm_id, assignment, False)
            for perm_id, assignment in enumerate(plan.null_assignments)
        ]
        for perm_id, assignment, is_observed in assignments_to_write:
            assignment_hash = _assignment_hash(assignment)
            for donor_index, donor in enumerate(donor_names):
                assignment_rows.append(
                    {
                        "perm_id": int(perm_id),
                        "assignment_hash": assignment_hash,
                        "is_observed_assignment": bool(is_observed),
                        "donor": donor,
                        "stratum": included_design.loc[donor_index, "stratum"],
                        "permuted_condition": (
                            str(case) if assignment[donor_index] else str(control)
                        ),
                    }
                )
    permutation_assignments = pd.DataFrame(
        assignment_rows,
        columns=[
            "perm_id",
            "assignment_hash",
            "is_observed_assignment",
            "donor",
            "stratum",
            "permuted_condition",
        ],
    )
    null_statistics = pd.DataFrame(
        null_rows,
        columns=[
            "perm_id",
            "assignment_hash",
            "Pathway",
            "raw_statistic",
            "calibration_statistic",
            "max_pathway_calibration_statistic",
        ],
    )
    permutation_summary = pd.DataFrame(
        [
            {
                "requested_mode": plan.requested_mode,
                "actual_mode": plan.actual_mode,
                "enumeration": "exact" if plan.is_exact else "monte_carlo",
                "restriction": "within_strata" if restricted else "unrestricted",
                "strata_keys": "|".join(strata_keys),
                "permutation_space_size": int(plan.permutation_space_size),
                "n_null_assignments_possible": int(
                    plan.permutation_space_size - 1
                ),
                "n_null_assignments_evaluated": int(n_null),
                "n_reference_assignments": int(denominator),
                "n_permutations_requested": int(n_permutations),
                "observed_assignment_in_null": False,
                "minimum_attainable_p": float(minimum_p),
                "permutation_p_resolution": float(permutation_p_resolution),
                "complement_symmetry_applies": complement_symmetry_applies,
                "n_donors_included": int(len(donor_names)),
                "n_donors_excluded": int(
                    (~donor_design["included_in_inference"]).sum()
                ),
                "estimand_population": (
                    "informative_strata_only"
                    if (~donor_design["included_in_inference"]).any()
                    else "all_included_donors"
                ),
                "n_informative_strata": int(len(stratum_groups)),
                "n_pathways": int(len(pathway_names)),
                "recommended_reference_assignments_for_bonferroni_alpha": int(
                    math.ceil(
                        (2 if complement_symmetry_applies else 1)
                        * len(pathway_names)
                        / alpha
                    )
                ),
                "draw_attempts": int(plan.draw_attempts),
                "duplicate_draws": int(plan.duplicate_draws),
                "seed": int(seed),
                "calibration_status": calibration_status,
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
        pathway_membership[
            ["Pathway", "gene", "weight", "pathway_size"]
        ]
        .sort_values(["Pathway", "gene", "weight"])
        .to_csv(index=False, lineterminator="\n")
        .encode("utf-8")
    )
    pathway_family_hash = hashlib.sha256(pathway_family_payload).hexdigest()
    metadata = {
        "method": "fixed_grid_donor_pseudobulk_pathway_permutation",
        "pyfgsea_version": pyfgsea_version,
        "gene_universe_hash": gene_universe_hash,
        "pathway_family_hash": pathway_family_hash,
        "pathway_score": "weighted_mean_gene_z_across_donor_bin_pseudobulk",
        "condition_key": condition_key,
        "donor_key": donor_key,
        "control": str(control),
        "case": str(case),
        "pseudotime_key": pseudotime_key,
        "grid_edges": edges.tolist(),
        "selected_bin_ids": selected_bins.astype(int).tolist(),
        "common_support": "longest_contiguous_permutation_invariant_segment",
        "min_cells_per_donor_bin": int(min_cells_per_donor_bin),
        "min_donors_per_condition": int(min_donors_per_condition),
        "min_common_bins": int(min_common_bins),
        "strata_keys": list(strata_keys),
        "drop_noninformative_strata": bool(drop_noninformative_strata),
        "estimand_population": (
            "informative_strata_only"
            if (~donor_design["included_in_inference"]).any()
            else "all_included_donors"
        ),
        "statistic": statistic,
        "tail": tail,
        "permutation_mode": plan.actual_mode,
        "permutation_space_size": int(plan.permutation_space_size),
        "n_null_assignments_evaluated": int(n_null),
        "n_permutations_requested": int(n_permutations),
        "p_value_rule": "(1 + null assignments >= observed) / (B + 1)",
        "bh_dependency_assumption": "independent_or_positive_regression_dependence",
        "by_available_for_arbitrary_dependence": True,
        "maxT_scope": "single_step_across_pathways",
        "maxT_strong_fwer_condition": "requires_subset_pivotality",
        "expression_source": expression_source,
        "layer": layer,
        "use_raw": bool(use_raw),
        "retain_all_genes": bool(retain_all_genes),
        "n_expression_genes": int(len(genes)),
        "n_pseudobulk_genes": int(len(pseudobulk_gene_indices)),
        "return_donor_bin_activity": bool(return_donor_bin_activity),
        "min_size": int(min_size),
        "max_size": int(max_size),
        "alpha": float(alpha),
        "seed": int(seed),
        "inference_scope": "conditional_on_fixed_input_pseudotime",
        "exchangeability_assumption": (
            "whole donors exchangeable within declared strata"
            if restricted
            else "whole donors exchangeable between conditions"
        ),
        "observed_assignment_in_null": False,
        "permutation_p_resolution": float(permutation_p_resolution),
        "minimum_attainable_p": float(minimum_p),
        "complement_symmetry_applies": complement_symmetry_applies,
    }
    for table in (pathway_tests, effect_curves, grid_diagnostics):
        table.attrs["fixed_grid_donor_pseudobulk"] = metadata.copy()

    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError(
            "run_fixed_grid_donor_pseudobulk requires anndata"
        ) from exc
    pseudobulk_obs_rows = []
    for donor_index, donor in enumerate(donor_names):
        donor_info = included_design.loc[donor_index]
        for local_bin, bin_id in enumerate(selected_bins):
            pseudobulk_obs_rows.append(
                {
                    "donor": donor,
                    "observed_condition": donor_info["observed_condition"],
                    "stratum": donor_info["stratum"],
                    "bin_id": int(bin_id),
                    "bin_left": float(edges[bin_id]),
                    "bin_right": float(edges[bin_id + 1]),
                    "bin_mid": float((edges[bin_id] + edges[bin_id + 1]) / 2.0),
                    "n_cells": int(tested_counts[donor_index, local_bin]),
                    "available": bool(
                        tested_counts[donor_index, local_bin]
                        >= min_cells_per_donor_bin
                    ),
                }
            )
    pseudobulk_obs = pd.DataFrame(
        pseudobulk_obs_rows,
        index=[
            f"{row['donor']}__bin{row['bin_id']}" for row in pseudobulk_obs_rows
        ],
    )
    pseudobulk_adata = ad.AnnData(
        X=pseudobulk.reshape(-1, pseudobulk.shape[-1]),
        obs=pseudobulk_obs,
        var=pd.DataFrame(
            index=[str(genes[index]) for index in pseudobulk_gene_indices]
        ),
    )
    pseudobulk_adata.uns["fixed_grid_donor_pseudobulk"] = {
        key: _h5ad_safe_value(value)
        for key, value in metadata.items()
        if value is not None
    }
    donor_design_output = donor_design.drop(
        columns=["__stratum_key"], errors="ignore"
    )
    return FixedGridDonorPseudobulkResult(
        pathway_tests=pathway_tests,
        effect_curves=effect_curves,
        pseudobulk_adata=pseudobulk_adata,
        donor_bin_activity=donor_bin_activity,
        grid_diagnostics=grid_diagnostics,
        donor_design=donor_design_output,
        pathway_membership=pathway_membership,
        permutation_summary=permutation_summary,
        permutation_assignments=permutation_assignments,
        null_statistics=null_statistics,
        metadata=metadata,
    )
