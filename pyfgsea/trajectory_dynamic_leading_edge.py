from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import sparse

from .trajectory_covariate_pseudobulk import (
    CovariateAdjustedDonorPseudobulkResult,
    _encode_reduced_design,
    _formal_inference_donor_bin_view,
    _formal_inference_donor_design_view,
)


def _fitted_source_fingerprint(fitted) -> str:
    """Hash the fitted design and observed pathway effects used by this layer."""

    metadata_keys = (
        "method",
        "gene_universe_hash",
        "pathway_family_hash",
        "condition_key",
        "donor_key",
        "control",
        "case",
        "pseudotime_key",
        "continuous_covariate_keys",
        "categorical_covariate_keys",
        "strata_keys",
        "grid_edges",
        "selected_bin_ids",
        "statistic",
        "tail",
        "calibration_scale",
        "exactness_status",
        "n_null_mappings_evaluated",
    )
    metadata = getattr(fitted, "metadata", {})
    payload = {
        key: metadata.get(key)
        for key in metadata_keys
        if key in metadata
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
    frame_specs = (
        (
            "pathway_tests",
            ("Pathway", "primary_statistic", "tail", "calibration_scale", "observed_calibration_statistic", "p_maxT"),
            ("Pathway",),
        ),
        (
            "effect_curves",
            ("Pathway", "bin_id", "beta_condition"),
            ("Pathway", "bin_id"),
        ),
        (
            "donor_design",
            (
                "donor",
                "observed_condition",
                "restriction_stratum",
                "availability_signature",
                "permutation_block",
            ),
            ("donor",),
        ),
    )
    for attribute, preferred_columns, sort_columns in frame_specs:
        frame = getattr(fitted, attribute, None)
        if not isinstance(frame, pd.DataFrame):
            digest.update(f"{attribute}:<absent>".encode("utf-8"))
            continue
        columns = [column for column in preferred_columns if column in frame]
        stable = frame.loc[:, columns].copy()
        available_sort = [column for column in sort_columns if column in stable]
        if available_sort:
            stable = stable.sort_values(available_sort, kind="mergesort")
        digest.update(attribute.encode("utf-8"))
        digest.update(
            stable.to_csv(
                index=False,
                lineterminator="\n",
                float_format="%.17g",
                na_rep="<NA>",
            ).encode("utf-8")
        )
    return digest.hexdigest()


_CONTRIBUTION_COLUMNS = [
    "Pathway",
    "gene",
    "weight",
    "pathway_abs_weight_sum",
    "bin_id",
    "bin_left",
    "bin_right",
    "bin_mid",
    "bin_width",
    "gene_pseudobulk_center",
    "gene_pseudobulk_observed_sample_sd",
    "gene_pseudobulk_effective_scale",
    "gene_pseudobulk_scale",
    "gene_near_constant",
    "gene_condition_beta_raw",
    "gene_condition_beta_z",
    "contribution",
    "pathway_beta_condition",
    "contribution_closure_error",
]

_EVENT_GENE_COLUMNS = [
    "Pathway",
    "event_id",
    "gene",
    "weight",
    "pathway_abs_weight_sum",
    "event_region_source",
    "onset_bin",
    "peak_bin",
    "peak_definition",
    "end_bin",
    "n_event_bins",
    "event_duration",
    "integrated_contribution",
    "integrated_absolute_contribution",
    "onset_contribution",
    "peak_contribution",
    "mean_absolute_contribution",
    "maximum_absolute_contribution",
    "bin_direction_agreement_fraction",
    "pathway_event_effect",
    "pathway_event_absolute_effect",
    "absolute_contribution_fraction",
    "absolute_contribution_rank",
    "cumulative_absolute_contribution_fraction",
    "in_dynamic_leading_edge",
    "n_donors_with_influence",
    "condition_aligned_influence_support",
    "n_lodo_attempted",
    "n_lodo_valid",
    "n_lodo_failed",
    "lodo_direction_support",
    "lodo_inclusion_support",
    "lodo_pathway_direction_support",
    "lodo_median_absolute_change",
    "lodo_maximum_absolute_change",
    "lodo_maximum_absolute_relative_change",
]

_DONOR_INFLUENCE_COLUMNS = [
    "Pathway",
    "event_id",
    "gene",
    "weight",
    "donor",
    "observed_condition",
    "n_event_bins",
    "n_bins_available",
    "integrated_condition_aligned_influence",
    "original_integrated_contribution",
    "direction_agrees_with_integrated_contribution",
    "influence_fraction_of_integrated_contribution",
]

_LODO_COLUMNS = [
    "Pathway",
    "event_id",
    "gene",
    "weight",
    "dropped_donor",
    "dropped_condition",
    "n_donors_remaining",
    "status",
    "failure_reason",
    "original_integrated_contribution",
    "lodo_integrated_contribution",
    "absolute_change",
    "absolute_relative_change",
    "direction_preserved",
    "original_in_dynamic_leading_edge",
    "lodo_in_dynamic_leading_edge",
    "original_pathway_event_effect",
    "lodo_pathway_event_effect",
    "pathway_direction_preserved",
]


@dataclass
class DynamicLeadingEdgeResult:
    """Gene-level decomposition of an adjusted donor-pseudobulk effect.

    The tables are mechanistic decompositions and sensitivity diagnostics of
    an already fitted pathway model.  They intentionally contain no gene-level
    p-values or q-values.
    """

    gene_contribution_curves: pd.DataFrame
    event_gene_summary: pd.DataFrame
    donor_influence: pd.DataFrame
    lodo_stability: pd.DataFrame
    metadata: dict[str, Any]

    def to_tables(self) -> dict[str, pd.DataFrame]:
        return {
            "gene_contribution_curves": self.gene_contribution_curves.copy(),
            "event_gene_summary": self.event_gene_summary.copy(),
            "donor_influence": self.donor_influence.copy(),
            "lodo_stability": self.lodo_stability.copy(),
        }


def _as_dense(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        values = matrix.toarray()
    else:
        try:
            values = np.asarray(matrix, dtype=float)
        except (TypeError, ValueError):
            values = np.asarray(matrix[...], dtype=float)
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("fitted.pseudobulk_adata.X must be two-dimensional")
    return values


def _normalize_pathways(fitted, pathways: Optional[Sequence[str]]) -> list[str]:
    effect = fitted.effect_curves
    membership = fitted.pathway_membership
    if "Pathway" not in effect or "Pathway" not in membership:
        raise ValueError("fitted result is missing Pathway columns")
    available = list(dict.fromkeys(effect["Pathway"].astype(str).tolist()))
    membership_names = set(membership["Pathway"].astype(str))
    available = [name for name in available if name in membership_names]
    if pathways is None:
        selected = available
    else:
        if isinstance(pathways, (str, bytes)):
            raise ValueError("pathways must be a sequence, not a string")
        selected = [str(value) for value in pathways]
        if len(set(selected)) != len(selected):
            raise ValueError("pathways must not contain duplicates")
    if not selected:
        raise ValueError("At least one pathway must be selected")
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"Selected pathways are absent from fitted result: {missing}")
    return selected


def _integer_bin(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be an integer bin id")
    numeric = float(value)
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{label} must be an integer bin id")
    return int(numeric)


def _default_peak_bins(fitted, pathways: list[str]) -> dict[str, int]:
    tests = fitted.pathway_tests.copy()
    effect = fitted.effect_curves.copy()
    peaks: dict[str, int] = {}
    if {"Pathway", "peak_bin"}.issubset(tests.columns):
        for pathway in pathways:
            rows = tests[tests["Pathway"].astype(str).eq(pathway)]
            if len(rows) == 1 and pd.notna(rows.iloc[0]["peak_bin"]):
                peaks[pathway] = _integer_bin(
                    rows.iloc[0]["peak_bin"], f"peak_bin for {pathway}"
                )
    for pathway in pathways:
        if pathway in peaks:
            continue
        rows = effect[effect["Pathway"].astype(str).eq(pathway)].copy()
        if rows.empty or "beta_condition" not in rows:
            raise ValueError(f"No fixed pathway effect curve is available for {pathway}")
        beta = pd.to_numeric(rows["beta_condition"], errors="coerce")
        if not np.isfinite(beta.to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite pathway effect curve for {pathway}")
        peaks[pathway] = int(rows.iloc[int(np.argmax(np.abs(beta)))]["bin_id"])
    return peaks


def _normalize_event_regions(
    fitted,
    pathways: list[str],
    event_regions,
    valid_bins: set[int],
) -> tuple[pd.DataFrame, str]:
    rows: list[dict[str, Any]] = []
    requested_events: set[tuple[str, str]] = set()
    if event_regions is None:
        peaks = _default_peak_bins(fitted, pathways)
        for pathway in pathways:
            event_id = f"{pathway}|fixed_peak"
            requested_events.add((pathway, event_id))
            rows.append(
                {
                    "Pathway": pathway,
                    "event_id": event_id,
                    "bin_id": peaks[pathway],
                }
            )
        source = "fixed_pathway_peak"
    elif isinstance(event_regions, Mapping):
        unexpected = sorted(set(map(str, event_regions)) - set(pathways))
        if unexpected:
            raise ValueError(
                f"event_regions contains unselected pathways: {unexpected}"
            )
        for pathway in pathways:
            if pathway not in event_regions:
                raise ValueError(f"event_regions is missing pathway '{pathway}'")
            spec = event_regions[pathway]
            event_id = f"{pathway}|event_001"
            if isinstance(spec, Mapping):
                event_id = str(spec.get("event_id", event_id))
                spec = spec.get("bins", spec.get("bin_ids", spec.get("bin_id")))
                if spec is None:
                    raise ValueError(
                        f"event_regions['{pathway}'] must define bins or bin_id"
                    )
            if np.isscalar(spec):
                bins = [_integer_bin(spec, f"event bin for {pathway}")]
            else:
                bins = [
                    _integer_bin(value, f"event bin for {pathway}")
                    for value in spec
                ]
            if not bins:
                raise ValueError(f"event region for '{pathway}' is empty")
            requested_events.add((pathway, event_id))
            for bin_id in bins:
                rows.append(
                    {"Pathway": pathway, "event_id": event_id, "bin_id": bin_id}
                )
        source = "user_fixed_bins"
    elif isinstance(event_regions, pd.DataFrame):
        frame = event_regions.copy()
        if "Pathway" not in frame:
            raise ValueError("event_regions DataFrame requires a Pathway column")
        frame["Pathway"] = frame["Pathway"].astype(str)
        unexpected = sorted(set(frame["Pathway"]) - set(pathways))
        if unexpected:
            raise ValueError(
                f"event_regions contains unselected pathways: {unexpected}"
            )
        if "event_id" not in frame:
            frame["event_id"] = frame["Pathway"].map(
                lambda value: f"{value}|event_001"
            )
        if "bin_id" in frame:
            for item in frame.itertuples(index=False):
                event_key = (str(item.Pathway), str(item.event_id))
                requested_events.add(event_key)
                rows.append(
                    {
                        "Pathway": event_key[0],
                        "event_id": event_key[1],
                        "bin_id": _integer_bin(
                            item.bin_id, f"event bin for {item.Pathway}"
                        ),
                    }
                )
        elif {"start_bin", "end_bin"}.issubset(frame.columns):
            for item in frame.itertuples(index=False):
                start = _integer_bin(item.start_bin, "start_bin")
                end = _integer_bin(item.end_bin, "end_bin")
                if end < start:
                    raise ValueError("event region end_bin must be at least start_bin")
                event_key = (str(item.Pathway), str(item.event_id))
                requested_events.add(event_key)
                for bin_id in sorted(value for value in valid_bins if start <= value <= end):
                    rows.append(
                        {
                            "Pathway": event_key[0],
                            "event_id": event_key[1],
                            "bin_id": bin_id,
                        }
                    )
        else:
            raise ValueError(
                "event_regions DataFrame requires bin_id or start_bin/end_bin"
            )
        source = "user_fixed_table"
    else:
        raise TypeError("event_regions must be None, a mapping, or a DataFrame")

    normalized = pd.DataFrame(rows, columns=["Pathway", "event_id", "bin_id"])
    normalized = normalized.drop_duplicates().reset_index(drop=True)
    observed_events = set(
        zip(normalized["Pathway"].astype(str), normalized["event_id"].astype(str))
    )
    missing_events = sorted(requested_events - observed_events)
    if missing_events:
        labels = [f"{pathway}/{event_id}" for pathway, event_id in missing_events]
        raise ValueError(
            "event_regions contains events with no bins in the fitted grid: "
            f"{labels}"
        )
    missing_pathways = sorted(set(pathways) - set(normalized["Pathway"]))
    if missing_pathways:
        raise ValueError(f"event_regions is missing pathways: {missing_pathways}")
    invalid = sorted(set(normalized["bin_id"]) - valid_bins)
    if invalid:
        raise ValueError(f"event_regions references bins outside fitted grid: {invalid}")
    empty_ids = normalized["event_id"].astype(str).str.len().eq(0)
    if empty_ids.any():
        raise ValueError("event_id must not be empty")
    return normalized, source


def _build_tensor(fitted, selected_pathways: list[str]):
    required = {
        "pseudobulk_adata",
        "donor_design",
        "effect_curves",
        "pathway_membership",
        "pathway_tests",
        "metadata",
    }
    missing = [name for name in required if not hasattr(fitted, name)]
    if missing:
        raise TypeError(f"fitted is missing required result attributes: {missing}")
    pb = fitted.pseudobulk_adata
    if not hasattr(pb, "obs") or not hasattr(pb, "var_names") or not hasattr(pb, "X"):
        raise TypeError("fitted.pseudobulk_adata must be an AnnData-like object")
    obs = pb.obs.copy()
    for column in ("donor", "bin_id", "available"):
        if column not in obs:
            raise ValueError(f"pseudobulk_adata.obs is missing '{column}'")
    values = _as_dense(pb.X)
    if values.shape != (len(obs), len(pb.var_names)):
        raise ValueError("pseudobulk matrix shape does not match obs/var dimensions")
    formal_obs = _formal_inference_donor_bin_view(
        obs, label="pseudobulk_adata.obs"
    )
    formal_row_indices = obs.index.get_indexer(formal_obs.index)
    if np.any(formal_row_indices < 0):
        raise RuntimeError("Formal pseudobulk rows are not aligned to the matrix")
    obs = formal_obs
    values = values[formal_row_indices]
    genes = np.asarray(list(map(str, pb.var_names)), dtype=object)
    if len(set(genes)) != len(genes):
        raise ValueError("pseudobulk gene names must be unique")

    donor_design = _formal_inference_donor_design_view(fitted.donor_design)
    donor_design["donor"] = donor_design["donor"].astype(str)
    if donor_design["donor"].duplicated().any():
        raise ValueError("donor_design contains duplicate included donors")

    effect = fitted.effect_curves.copy()
    effect["Pathway"] = effect["Pathway"].astype(str)
    effect = effect[effect["Pathway"].isin(selected_pathways)].copy()
    required_effect = {
        "Pathway",
        "bin_id",
        "bin_left",
        "bin_right",
        "bin_mid",
        "bin_width",
        "beta_condition",
    }
    missing_effect = required_effect - set(effect.columns)
    if missing_effect:
        raise ValueError(f"effect_curves is missing columns: {sorted(missing_effect)}")
    effect["bin_id"] = effect["bin_id"].map(
        lambda value: _integer_bin(value, "effect bin_id")
    )
    bins = sorted(effect["bin_id"].unique().tolist())
    expected_pairs = pd.MultiIndex.from_product(
        [selected_pathways, bins], names=["Pathway", "bin_id"]
    )
    observed_pairs = pd.MultiIndex.from_frame(effect[["Pathway", "bin_id"]])
    if observed_pairs.has_duplicates or set(expected_pairs) != set(observed_pairs):
        raise ValueError("effect_curves must have exactly one row per pathway and bin")

    donor_to_index = {
        donor: index for index, donor in enumerate(donor_design["donor"].tolist())
    }
    bin_to_index = {bin_id: index for index, bin_id in enumerate(bins)}
    tensor = np.full((len(donor_design), len(bins), len(genes)), np.nan, dtype=float)
    seen: set[tuple[int, int]] = set()
    for row_index, row in enumerate(obs.itertuples()):
        donor = str(row.donor)
        bin_id = _integer_bin(row.bin_id, "pseudobulk bin_id")
        if donor not in donor_to_index or bin_id not in bin_to_index:
            raise ValueError(
                "pseudobulk_adata contains a donor/bin outside the fitted inference"
            )
        key = (donor_to_index[donor], bin_to_index[bin_id])
        if key in seen:
            raise ValueError("pseudobulk_adata contains duplicate donor/bin rows")
        seen.add(key)
        row_values = values[row_index]
        finite = np.isfinite(row_values)
        if finite.any() and not finite.all():
            raise ValueError("pseudobulk donor/bin rows must be wholly finite or missing")
        available = bool(row.available)
        if available != bool(finite.all()):
            raise ValueError("pseudobulk availability flag does not match expression")
        tensor[key] = row_values
    expected_keys = {
        (donor_index, bin_index)
        for donor_index in range(len(donor_design))
        for bin_index in range(len(bins))
    }
    if seen != expected_keys:
        raise ValueError("pseudobulk_adata does not contain the full donor-by-bin grid")

    membership = fitted.pathway_membership.copy()
    required_membership = {"Pathway", "gene", "weight"}
    missing_membership = required_membership - set(membership.columns)
    if missing_membership:
        raise ValueError(
            f"pathway_membership is missing columns: {sorted(missing_membership)}"
        )
    membership["Pathway"] = membership["Pathway"].astype(str)
    membership["gene"] = membership["gene"].astype(str)
    membership = membership[membership["Pathway"].isin(selected_pathways)].copy()
    membership["weight"] = pd.to_numeric(membership["weight"], errors="coerce")
    if (
        membership.empty
        or not np.isfinite(membership["weight"].to_numpy(dtype=float)).all()
        or membership["weight"].eq(0).any()
    ):
        raise ValueError("Selected pathway membership has invalid weights")
    if membership.duplicated(["Pathway", "gene"]).any():
        raise ValueError("pathway_membership contains duplicate pathway/gene rows")
    missing_genes = sorted(set(membership["gene"]) - set(genes))
    if missing_genes:
        raise ValueError(
            f"Selected pathway genes are absent from pseudobulk_adata: {missing_genes}"
        )
    return tensor, genes, donor_design, bins, effect, membership


def _prepare_design(fitted, donor_design: pd.DataFrame):
    metadata = fitted.metadata
    continuous = tuple(map(str, metadata.get("continuous_covariate_keys", [])))
    categorical = tuple(map(str, metadata.get("categorical_covariate_keys", [])))
    strata = tuple(map(str, metadata.get("strata_keys", [])))
    required = [*continuous, *categorical, *strata, "observed_case"]
    missing = [column for column in required if column not in donor_design]
    if missing:
        raise ValueError(f"donor_design is missing design columns: {missing}")
    frame = donor_design.copy()
    frame["__stratum_key"] = [
        tuple(str(row[key]) for key in strata) if strata else ("__all__",)
        for _, row in frame.iterrows()
    ]
    encoded = _encode_reduced_design(
        frame,
        continuous_covariate_keys=continuous,
        categorical_covariate_keys=categorical,
        strata_keys=strata,
    )
    expected_terms = list(map(str, metadata.get("reduced_model_terms", encoded.terms)))
    if encoded.terms != expected_terms:
        raise ValueError(
            "Reconstructed reduced-model terms do not match the fitted result"
        )
    condition = frame["observed_case"].astype(bool).to_numpy(dtype=float)
    return frame, encoded.reduced, condition, continuous, categorical, strata


def _standardize_tensor(values: np.ndarray):
    flat = values.reshape(-1, values.shape[-1])
    with np.errstate(invalid="ignore", divide="ignore"):
        center = np.nanmean(flat, axis=0)
        observed_sample_sd = np.nanstd(flat, axis=0, ddof=1)
    if not np.isfinite(center).all():
        raise ValueError("At least one pseudobulk gene has no finite observations")
    scale_floor = 1e-6 * np.maximum(np.abs(center), 1.0)
    near_constant = ~np.isfinite(observed_sample_sd) | (
        observed_sample_sd <= scale_floor
    )
    effective_scale = observed_sample_sd.copy()
    effective_scale[near_constant] = 1.0
    standardized = (
        values - center[None, None, :]
    ) / effective_scale[None, None, :]
    standardized[:, :, near_constant] = np.where(
        np.isfinite(values[:, :, near_constant]), 0.0, np.nan
    )
    return (
        standardized,
        center,
        observed_sample_sd,
        effective_scale,
        near_constant,
    )


def _fit_gene_curves(
    raw: np.ndarray,
    standardized: np.ndarray,
    reduced: np.ndarray,
    condition: np.ndarray,
):
    n_donors, n_bins, n_genes = raw.shape
    beta_raw = np.full((n_bins, n_genes), np.nan, dtype=float)
    beta_z = np.full((n_bins, n_genes), np.nan, dtype=float)
    influence = np.full((n_donors, n_bins, n_genes), np.nan, dtype=float)
    for bin_index in range(n_bins):
        available = np.isfinite(raw[:, bin_index]).all(axis=1)
        indices = np.flatnonzero(available)
        z = reduced[indices]
        c = condition[indices]
        full = np.column_stack([z, c])
        rank_z = int(np.linalg.matrix_rank(z))
        rank_full = int(np.linalg.matrix_rank(full))
        if rank_full != rank_z + 1:
            raise ValueError(
                f"Condition is not estimable when reconstructing fitted bin {bin_index}"
            )
        pinv_z = np.linalg.pinv(z)
        pinv_full = np.linalg.pinv(full)
        raw_y = raw[indices, bin_index]
        z_y = standardized[indices, bin_index]
        beta_raw[bin_index] = (pinv_full @ raw_y)[-1]
        beta_z[bin_index] = (pinv_full @ z_y)[-1]
        residualized_condition = c - z @ (pinv_z @ c)
        information = float(residualized_condition @ residualized_condition)
        if information <= 1e-14:
            raise ValueError(
                f"Condition has zero residual information in fitted bin {bin_index}"
            )
        reduced_residual = z_y - z @ (pinv_z @ z_y)
        local_influence = (
            residualized_condition[:, None] * reduced_residual / information
        )
        influence[indices, bin_index] = local_influence
        if not np.allclose(
            np.sum(local_influence, axis=0),
            beta_z[bin_index],
            rtol=1e-9,
            atol=1e-11,
        ):
            raise RuntimeError("Internal donor influence decomposition did not close")
    return beta_raw, beta_z, influence


def _leading_membership(
    genes: Sequence[str], absolute_values: np.ndarray, fraction: float
) -> tuple[dict[str, int], dict[str, float], dict[str, bool]]:
    table = pd.DataFrame(
        {"gene": list(map(str, genes)), "absolute": np.asarray(absolute_values, float)}
    ).sort_values(["absolute", "gene"], ascending=[False, True], kind="mergesort")
    total = float(table["absolute"].sum())
    ranks: dict[str, int] = {}
    cumulative: dict[str, float] = {}
    selected: dict[str, bool] = {}
    if not np.isfinite(total) or total <= 1e-15:
        for rank, gene in enumerate(table["gene"], start=1):
            ranks[gene] = rank
            cumulative[gene] = 0.0
            selected[gene] = False
        return ranks, cumulative, selected
    table["cumulative"] = table["absolute"].cumsum() / total
    cutoff_position = int(np.flatnonzero(table["cumulative"].to_numpy() >= fraction)[0])
    cutoff = float(table.iloc[cutoff_position]["absolute"])
    tolerance = 1e-12 * max(1.0, cutoff)
    for rank, row in enumerate(table.itertuples(index=False), start=1):
        ranks[row.gene] = rank
        cumulative[row.gene] = float(row.cumulative)
        selected[row.gene] = bool(row.absolute > 0 and row.absolute >= cutoff - tolerance)
    return ranks, cumulative, selected


def _same_direction(left: float, right: float) -> Any:
    tolerance = 1e-12 * max(1.0, abs(left), abs(right))
    if abs(left) <= tolerance or abs(right) <= tolerance:
        return np.nan
    return bool(np.sign(left) == np.sign(right))


def _event_definitions(
    fitted,
    event_bins: pd.DataFrame,
    source: str,
    effect: pd.DataFrame,
    bins: list[int],
) -> pd.DataFrame:
    default_peaks = _default_peak_bins(
        fitted, list(dict.fromkeys(event_bins["Pathway"].tolist()))
    )
    bin_to_local = {value: index for index, value in enumerate(bins)}
    rows = []
    for (pathway, event_id), group in event_bins.groupby(
        ["Pathway", "event_id"], sort=False
    ):
        group_bins = sorted(group["bin_id"].astype(int).unique().tolist())
        curve = effect[effect["Pathway"].eq(pathway)].set_index("bin_id")
        group_bins = sorted(group_bins, key=lambda value: float(curve.loc[value, "bin_mid"]))
        fixed_peak = default_peaks[pathway]
        if fixed_peak in group_bins:
            peak_bin = fixed_peak
            peak_definition = "fitted_pathway_peak"
        else:
            beta = curve.loc[group_bins, "beta_condition"].to_numpy(dtype=float)
            peak_bin = group_bins[int(np.argmax(np.abs(beta)))]
            peak_definition = "pathway_max_abs_within_fixed_event_region"
        widths = curve.loc[group_bins, "bin_width"].to_numpy(dtype=float)
        beta = curve.loc[group_bins, "beta_condition"].to_numpy(dtype=float)
        rows.append(
            {
                "Pathway": pathway,
                "event_id": event_id,
                "event_region_source": source,
                "bin_ids": group_bins,
                "local_bin_indices": [bin_to_local[value] for value in group_bins],
                "onset_bin": int(group_bins[0]),
                "peak_bin": int(peak_bin),
                "peak_definition": peak_definition,
                "end_bin": int(group_bins[-1]),
                "n_event_bins": int(len(group_bins)),
                "event_duration": float(np.sum(widths)),
                "pathway_event_effect": float(np.sum(beta * widths)),
                "pathway_event_absolute_effect": float(np.sum(np.abs(beta) * widths)),
            }
        )
    return pd.DataFrame(rows)


def _contribution_tables(
    genes: np.ndarray,
    beta_raw: np.ndarray,
    beta_z: np.ndarray,
    center: np.ndarray,
    observed_sample_sd: np.ndarray,
    effective_scale: np.ndarray,
    near_constant: np.ndarray,
    bins: list[int],
    effect: pd.DataFrame,
    membership: pd.DataFrame,
):
    gene_to_index = {str(gene): index for index, gene in enumerate(genes)}
    rows = []
    max_error = 0.0
    for pathway, members in membership.groupby("Pathway", sort=False):
        members = members.sort_values("gene", kind="mergesort")
        denominator = float(np.abs(members["weight"].to_numpy(dtype=float)).sum())
        if denominator <= 0:
            raise ValueError(f"Pathway '{pathway}' has zero absolute weight sum")
        curve = effect[effect["Pathway"].eq(pathway)].set_index("bin_id")
        for local_bin, bin_id in enumerate(bins):
            path_beta = float(curve.loc[bin_id, "beta_condition"])
            contributions = []
            pending = []
            for member in members.itertuples(index=False):
                gene_index = gene_to_index[str(member.gene)]
                contribution = float(member.weight) * beta_z[local_bin, gene_index] / denominator
                contributions.append(contribution)
                pending.append((member, gene_index, contribution))
            closure_error = float(np.sum(contributions) - path_beta)
            max_error = max(max_error, abs(closure_error))
            tolerance = 1e-10 + 1e-8 * abs(path_beta)
            if abs(closure_error) > tolerance:
                raise RuntimeError(
                    "Gene contribution reconstruction does not close to fitted "
                    f"pathway beta for pathway={pathway!r}, bin={bin_id}: "
                    f"error={closure_error:.6g}"
                )
            grid = curve.loc[bin_id]
            for member, gene_index, contribution in pending:
                rows.append(
                    {
                        "Pathway": pathway,
                        "gene": str(member.gene),
                        "weight": float(member.weight),
                        "pathway_abs_weight_sum": denominator,
                        "bin_id": int(bin_id),
                        "bin_left": float(grid["bin_left"]),
                        "bin_right": float(grid["bin_right"]),
                        "bin_mid": float(grid["bin_mid"]),
                        "bin_width": float(grid["bin_width"]),
                        "gene_pseudobulk_center": float(center[gene_index]),
                        "gene_pseudobulk_observed_sample_sd": float(
                            observed_sample_sd[gene_index]
                        ),
                        "gene_pseudobulk_effective_scale": float(
                            effective_scale[gene_index]
                        ),
                        # Backward-compatible alias for the scale actually used.
                        "gene_pseudobulk_scale": float(effective_scale[gene_index]),
                        "gene_near_constant": bool(near_constant[gene_index]),
                        "gene_condition_beta_raw": float(beta_raw[local_bin, gene_index]),
                        "gene_condition_beta_z": float(beta_z[local_bin, gene_index]),
                        "contribution": contribution,
                        "pathway_beta_condition": path_beta,
                        "contribution_closure_error": closure_error,
                    }
                )
    return pd.DataFrame(rows, columns=_CONTRIBUTION_COLUMNS), max_error


def _donor_influence_table(
    event_definitions: pd.DataFrame,
    membership: pd.DataFrame,
    genes: np.ndarray,
    raw: np.ndarray,
    influence: np.ndarray,
    donor_design: pd.DataFrame,
    effect: pd.DataFrame,
    contribution_curves: pd.DataFrame,
) -> pd.DataFrame:
    gene_to_index = {str(gene): index for index, gene in enumerate(genes)}
    rows = []
    for event in event_definitions.itertuples(index=False):
        members = membership[membership["Pathway"].eq(event.Pathway)].copy()
        denominator = float(np.abs(members["weight"].to_numpy(dtype=float)).sum())
        curve = effect[effect["Pathway"].eq(event.Pathway)].set_index("bin_id")
        widths = curve.loc[event.bin_ids, "bin_width"].to_numpy(dtype=float)
        original = contribution_curves[
            contribution_curves["Pathway"].eq(event.Pathway)
            & contribution_curves["bin_id"].isin(event.bin_ids)
        ]
        original_lookup = (
            original.assign(
                weighted=lambda frame: frame["contribution"] * frame["bin_width"]
            )
            .groupby("gene")["weighted"]
            .sum()
            .to_dict()
        )
        for member in members.itertuples(index=False):
            gene = str(member.gene)
            gene_index = gene_to_index[gene]
            factor = float(member.weight) / denominator
            donor_values = []
            for donor_index, donor in donor_design.iterrows():
                local_values = influence[
                    donor_index, event.local_bin_indices, gene_index
                ] * factor
                available = np.isfinite(
                    raw[donor_index, event.local_bin_indices]
                ).all(axis=1)
                n_available = int(available.sum())
                integrated = (
                    float(np.sum(local_values[available] * widths[available]))
                    if n_available
                    else np.nan
                )
                donor_values.append(integrated)
                original_value = float(original_lookup[gene])
                rows.append(
                    {
                        "Pathway": event.Pathway,
                        "event_id": event.event_id,
                        "gene": gene,
                        "weight": float(member.weight),
                        "donor": str(donor["donor"]),
                        "observed_condition": str(donor["observed_condition"]),
                        "n_event_bins": int(event.n_event_bins),
                        "n_bins_available": n_available,
                        "integrated_condition_aligned_influence": integrated,
                        "original_integrated_contribution": original_value,
                        "direction_agrees_with_integrated_contribution": (
                            _same_direction(integrated, original_value)
                            if np.isfinite(integrated)
                            else np.nan
                        ),
                        "influence_fraction_of_integrated_contribution": (
                            integrated / original_value
                            if np.isfinite(integrated) and abs(original_value) > 1e-12
                            else np.nan
                        ),
                    }
                )
            finite = np.asarray(donor_values, dtype=float)
            finite = finite[np.isfinite(finite)]
            if not np.allclose(
                finite.sum(), original_lookup[gene], rtol=1e-9, atol=1e-11
            ):
                raise RuntimeError("Event-level donor influence did not close")
    return pd.DataFrame(rows, columns=_DONOR_INFLUENCE_COLUMNS)


def _event_gene_summary(
    event_definitions: pd.DataFrame,
    membership: pd.DataFrame,
    contribution_curves: pd.DataFrame,
    donor_influence: pd.DataFrame,
    cumulative_abs_fraction: float,
) -> pd.DataFrame:
    rows = []
    for event in event_definitions.itertuples(index=False):
        members = membership[membership["Pathway"].eq(event.Pathway)]
        event_curves = contribution_curves[
            contribution_curves["Pathway"].eq(event.Pathway)
            & contribution_curves["bin_id"].isin(event.bin_ids)
        ].copy()
        summaries: list[dict[str, Any]] = []
        for member in members.itertuples(index=False):
            gene = str(member.gene)
            curve = event_curves[event_curves["gene"].eq(gene)].sort_values("bin_mid")
            values = curve["contribution"].to_numpy(dtype=float)
            widths = curve["bin_width"].to_numpy(dtype=float)
            path_values = curve["pathway_beta_condition"].to_numpy(dtype=float)
            agreements = [
                _same_direction(value, path_value)
                for value, path_value in zip(values, path_values)
            ]
            finite_agreement = [value for value in agreements if pd.notna(value)]
            peak_row = curve[curve["bin_id"].eq(event.peak_bin)]
            integrated = float(np.sum(values * widths))
            influence_rows = donor_influence[
                donor_influence["Pathway"].eq(event.Pathway)
                & donor_influence["event_id"].eq(event.event_id)
                & donor_influence["gene"].eq(gene)
            ]
            support = pd.to_numeric(
                influence_rows["direction_agrees_with_integrated_contribution"],
                errors="coerce",
            ).dropna()
            summaries.append(
                {
                    "Pathway": event.Pathway,
                    "event_id": event.event_id,
                    "gene": gene,
                    "weight": float(member.weight),
                    "pathway_abs_weight_sum": float(
                        curve["pathway_abs_weight_sum"].iloc[0]
                    ),
                    "event_region_source": event.event_region_source,
                    "onset_bin": int(event.onset_bin),
                    "peak_bin": int(event.peak_bin),
                    "peak_definition": event.peak_definition,
                    "end_bin": int(event.end_bin),
                    "n_event_bins": int(event.n_event_bins),
                    "event_duration": float(event.event_duration),
                    "integrated_contribution": integrated,
                    "integrated_absolute_contribution": float(
                        np.sum(np.abs(values) * widths)
                    ),
                    "onset_contribution": float(values[0]),
                    "peak_contribution": float(peak_row["contribution"].iloc[0]),
                    "mean_absolute_contribution": float(np.mean(np.abs(values))),
                    "maximum_absolute_contribution": float(np.max(np.abs(values))),
                    "bin_direction_agreement_fraction": (
                        float(np.mean(finite_agreement))
                        if finite_agreement
                        else np.nan
                    ),
                    "pathway_event_effect": float(event.pathway_event_effect),
                    "pathway_event_absolute_effect": float(
                        event.pathway_event_absolute_effect
                    ),
                    "n_donors_with_influence": int(
                        influence_rows["integrated_condition_aligned_influence"]
                        .notna()
                        .sum()
                    ),
                    "condition_aligned_influence_support": (
                        float(support.mean()) if len(support) else np.nan
                    ),
                }
            )
        abs_values = np.asarray(
            [item["integrated_absolute_contribution"] for item in summaries],
            dtype=float,
        )
        genes = [item["gene"] for item in summaries]
        ranks, cumulative, selected = _leading_membership(
            genes, abs_values, cumulative_abs_fraction
        )
        total_abs = float(abs_values.sum())
        for item in summaries:
            gene = item["gene"]
            value = float(item["integrated_absolute_contribution"])
            item.update(
                {
                    "absolute_contribution_fraction": (
                        value / total_abs if total_abs > 0 else 0.0
                    ),
                    "absolute_contribution_rank": int(ranks[gene]),
                    "cumulative_absolute_contribution_fraction": float(
                        cumulative[gene]
                    ),
                    "in_dynamic_leading_edge": bool(selected[gene]),
                    "n_lodo_attempted": 0,
                    "n_lodo_valid": 0,
                    "n_lodo_failed": 0,
                    "lodo_direction_support": np.nan,
                    "lodo_inclusion_support": np.nan,
                    "lodo_pathway_direction_support": np.nan,
                    "lodo_median_absolute_change": np.nan,
                    "lodo_maximum_absolute_change": np.nan,
                    "lodo_maximum_absolute_relative_change": np.nan,
                }
            )
            rows.append(item)
    return pd.DataFrame(rows, columns=_EVENT_GENE_COLUMNS)


def _lodo_gene_curves(
    raw: np.ndarray,
    donor_subset: pd.DataFrame,
    condition: np.ndarray,
    continuous: tuple[str, ...],
    categorical: tuple[str, ...],
    strata: tuple[str, ...],
    min_residual_df: int,
    min_donors_per_condition: int,
    max_condition_vif: float,
):
    frame = donor_subset.copy().reset_index(drop=True)
    frame["__stratum_key"] = [
        tuple(str(row[key]) for key in strata) if strata else ("__all__",)
        for _, row in frame.iterrows()
    ]
    encoded = _encode_reduced_design(
        frame,
        continuous_covariate_keys=continuous,
        categorical_covariate_keys=categorical,
        strata_keys=strata,
    )
    standardized, _, _, _, _ = _standardize_tensor(raw)
    beta = np.full((raw.shape[1], raw.shape[2]), np.nan, dtype=float)
    failures: dict[int, str] = {}
    minimum_df = np.inf
    for bin_index in range(raw.shape[1]):
        available = np.isfinite(raw[:, bin_index]).all(axis=1)
        indices = np.flatnonzero(available)
        z = encoded.reduced[indices]
        c = condition[indices]
        full = np.column_stack([z, c])
        rank_z = int(np.linalg.matrix_rank(z)) if len(indices) else 0
        rank_full = int(np.linalg.matrix_rank(full)) if len(indices) else 0
        residual_df = int(len(indices) - rank_full)
        minimum_df = min(minimum_df, residual_df)
        n_case = int(c.sum())
        n_control = int(len(c) - n_case)
        centered_condition = c - float(np.mean(c)) if len(c) else c
        unadjusted_information = float(centered_condition @ centered_condition)
        u = c - z @ (np.linalg.pinv(z) @ c) if len(indices) else c
        information = float(u @ u)
        condition_vif = (
            unadjusted_information / information
            if information > 1e-14 * max(1.0, unadjusted_information)
            else np.inf
        )
        reasons = []
        if (
            n_control < min_donors_per_condition
            or n_case < min_donors_per_condition
        ):
            reasons.append(
                "insufficient_donors_per_condition_"
                f"control_{n_control}_case_{n_case}_below_{min_donors_per_condition}"
            )
        if rank_full != rank_z + 1:
            reasons.append("condition_not_estimable")
        if residual_df < min_residual_df:
            reasons.append(
                f"residual_df_{residual_df}_below_{min_residual_df}"
            )
        if information <= 1e-14 * max(1.0, unadjusted_information):
            reasons.append("zero_condition_information")
        if not np.isfinite(condition_vif) or condition_vif > max_condition_vif:
            reasons.append(
                "condition_vif_"
                f"{condition_vif:.6g}_exceeds_{max_condition_vif:.6g}"
            )
        if reasons:
            failures[bin_index] = "|".join(reasons)
            continue
        beta[bin_index] = (
            np.linalg.pinv(full) @ standardized[indices, bin_index]
        )[-1]
    return beta, failures, int(minimum_df) if np.isfinite(minimum_df) else -1


def _lodo_table(
    event_definitions: pd.DataFrame,
    event_summary: pd.DataFrame,
    raw: np.ndarray,
    genes: np.ndarray,
    donor_design: pd.DataFrame,
    membership: pd.DataFrame,
    effect: pd.DataFrame,
    continuous: tuple[str, ...],
    categorical: tuple[str, ...],
    strata: tuple[str, ...],
    cumulative_abs_fraction: float,
    min_residual_df: int,
    min_donors_per_condition: int,
    max_condition_vif: float,
) -> pd.DataFrame:
    gene_to_index = {str(gene): index for index, gene in enumerate(genes)}
    rows = []
    for dropped_index, dropped in donor_design.iterrows():
        keep = np.arange(len(donor_design)) != int(dropped_index)
        subset = donor_design.loc[keep].copy().reset_index(drop=True)
        condition = subset["observed_case"].astype(bool).to_numpy(dtype=float)
        try:
            beta, failures, _minimum_df = _lodo_gene_curves(
                raw[keep],
                subset,
                condition,
                continuous,
                categorical,
                strata,
                min_residual_df,
                min_donors_per_condition,
                max_condition_vif,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            beta = np.full((raw.shape[1], raw.shape[2]), np.nan)
            failures = {index: f"design_refit_failure:{exc}" for index in range(raw.shape[1])}
        for event in event_definitions.itertuples(index=False):
            members = membership[membership["Pathway"].eq(event.Pathway)].copy()
            denominator = float(np.abs(members["weight"].to_numpy(dtype=float)).sum())
            curve = effect[effect["Pathway"].eq(event.Pathway)].set_index("bin_id")
            widths = curve.loc[event.bin_ids, "bin_width"].to_numpy(dtype=float)
            failed_bins = [
                (bin_id, failures[local])
                for bin_id, local in zip(event.bin_ids, event.local_bin_indices)
                if local in failures
            ]
            original_group = event_summary[
                event_summary["Pathway"].eq(event.Pathway)
                & event_summary["event_id"].eq(event.event_id)
            ].set_index("gene")
            if failed_bins:
                reason = ";".join(
                    f"bin_{bin_id}:{failure}" for bin_id, failure in failed_bins
                )
                for member in members.itertuples(index=False):
                    original = original_group.loc[str(member.gene)]
                    rows.append(
                        {
                            "Pathway": event.Pathway,
                            "event_id": event.event_id,
                            "gene": str(member.gene),
                            "weight": float(member.weight),
                            "dropped_donor": str(dropped["donor"]),
                            "dropped_condition": str(dropped["observed_condition"]),
                            "n_donors_remaining": int(keep.sum()),
                            "status": "failed",
                            "failure_reason": reason,
                            "original_integrated_contribution": float(
                                original["integrated_contribution"]
                            ),
                            "lodo_integrated_contribution": np.nan,
                            "absolute_change": np.nan,
                            "absolute_relative_change": np.nan,
                            "direction_preserved": np.nan,
                            "original_in_dynamic_leading_edge": bool(
                                original["in_dynamic_leading_edge"]
                            ),
                            "lodo_in_dynamic_leading_edge": np.nan,
                            "original_pathway_event_effect": float(
                                event.pathway_event_effect
                            ),
                            "lodo_pathway_event_effect": np.nan,
                            "pathway_direction_preserved": np.nan,
                        }
                    )
                continue

            lodo_values: dict[str, float] = {}
            lodo_abs: dict[str, float] = {}
            for member in members.itertuples(index=False):
                gene = str(member.gene)
                gene_index = gene_to_index[gene]
                values = (
                    float(member.weight)
                    * beta[event.local_bin_indices, gene_index]
                    / denominator
                )
                lodo_values[gene] = float(np.sum(values * widths))
                lodo_abs[gene] = float(np.sum(np.abs(values) * widths))
            _, _, selected = _leading_membership(
                list(lodo_abs),
                np.asarray(list(lodo_abs.values()), dtype=float),
                cumulative_abs_fraction,
            )
            lodo_pathway = float(sum(lodo_values.values()))
            path_preserved = _same_direction(
                lodo_pathway, float(event.pathway_event_effect)
            )
            for member in members.itertuples(index=False):
                gene = str(member.gene)
                original = original_group.loc[gene]
                original_value = float(original["integrated_contribution"])
                lodo_value = float(lodo_values[gene])
                change = abs(lodo_value - original_value)
                rows.append(
                    {
                        "Pathway": event.Pathway,
                        "event_id": event.event_id,
                        "gene": gene,
                        "weight": float(member.weight),
                        "dropped_donor": str(dropped["donor"]),
                        "dropped_condition": str(dropped["observed_condition"]),
                        "n_donors_remaining": int(keep.sum()),
                        "status": "ok",
                        "failure_reason": "",
                        "original_integrated_contribution": original_value,
                        "lodo_integrated_contribution": lodo_value,
                        "absolute_change": change,
                        "absolute_relative_change": (
                            change / abs(original_value)
                            if abs(original_value) > 1e-12
                            else np.nan
                        ),
                        "direction_preserved": _same_direction(
                            lodo_value, original_value
                        ),
                        "original_in_dynamic_leading_edge": bool(
                            original["in_dynamic_leading_edge"]
                        ),
                        "lodo_in_dynamic_leading_edge": bool(selected[gene]),
                        "original_pathway_event_effect": float(
                            event.pathway_event_effect
                        ),
                        "lodo_pathway_event_effect": lodo_pathway,
                        "pathway_direction_preserved": path_preserved,
                    }
                )
    return pd.DataFrame(rows, columns=_LODO_COLUMNS)


def _attach_lodo_summary(
    event_summary: pd.DataFrame, lodo: pd.DataFrame
) -> pd.DataFrame:
    out = event_summary.copy()
    if lodo.empty:
        return out
    for index, row in out.iterrows():
        group = lodo[
            lodo["Pathway"].eq(row["Pathway"])
            & lodo["event_id"].eq(row["event_id"])
            & lodo["gene"].eq(row["gene"])
        ]
        valid = group[group["status"].eq("ok")]
        failed = group[group["status"].ne("ok")]
        direction = pd.to_numeric(valid["direction_preserved"], errors="coerce").dropna()
        inclusion = pd.to_numeric(
            valid["lodo_in_dynamic_leading_edge"], errors="coerce"
        ).dropna()
        path_direction = pd.to_numeric(
            valid["pathway_direction_preserved"], errors="coerce"
        ).dropna()
        changes = pd.to_numeric(valid["absolute_change"], errors="coerce").dropna()
        relative = pd.to_numeric(
            valid["absolute_relative_change"], errors="coerce"
        ).dropna()
        out.at[index, "n_lodo_attempted"] = int(group["dropped_donor"].nunique())
        out.at[index, "n_lodo_valid"] = int(valid["dropped_donor"].nunique())
        out.at[index, "n_lodo_failed"] = int(failed["dropped_donor"].nunique())
        out.at[index, "lodo_direction_support"] = (
            float(direction.mean()) if len(direction) else np.nan
        )
        out.at[index, "lodo_inclusion_support"] = (
            float(inclusion.mean()) if len(inclusion) else np.nan
        )
        out.at[index, "lodo_pathway_direction_support"] = (
            float(path_direction.mean()) if len(path_direction) else np.nan
        )
        out.at[index, "lodo_median_absolute_change"] = (
            float(changes.median()) if len(changes) else np.nan
        )
        out.at[index, "lodo_maximum_absolute_change"] = (
            float(changes.max()) if len(changes) else np.nan
        )
        out.at[index, "lodo_maximum_absolute_relative_change"] = (
            float(relative.max()) if len(relative) else np.nan
        )
    for column in ("n_lodo_attempted", "n_lodo_valid", "n_lodo_failed"):
        out[column] = out[column].astype(int)
    return out.reindex(columns=_EVENT_GENE_COLUMNS)


def decompose_covariate_adjusted_leading_edge(
    fitted: CovariateAdjustedDonorPseudobulkResult,
    *,
    pathways: Optional[Sequence[str]] = None,
    event_regions=None,
    cumulative_abs_fraction: float = 0.8,
    lodo: bool = True,
    min_lodo_residual_df: int = 1,
) -> DynamicLeadingEdgeResult:
    """Decompose adjusted pathway effects into dynamic gene contributions.

    For pathway ``P`` and bin ``b``, the returned contribution is

    ``weight_g * beta_z(g, b) / sum(abs(pathway_weights))``.

    It therefore closes exactly to the fitted pathway ``beta_condition`` under
    the standardization used by ``run_covariate_adjusted_donor_pseudobulk``.
    Event regions are fixed by the caller; when omitted, only the already
    fitted pathway peak bin is used.  Gene ranks never define event timing.

    Leave-one-donor-out rows recompute gene standardization and refit the
    nuisance/condition model on the remaining donors while holding the fitted
    grid, pathway membership, and event bins fixed.  The source fit's minimum
    donors per condition and maximum condition-VIF gates are replayed for every
    LODO bin.  LODO is a sensitivity diagnostic, not a gene-level hypothesis
    test.
    """
    try:
        cumulative_abs_fraction = float(cumulative_abs_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError("cumulative_abs_fraction must be numeric") from exc
    if not 0 < cumulative_abs_fraction <= 1:
        raise ValueError("cumulative_abs_fraction must be in (0, 1]")
    if isinstance(min_lodo_residual_df, (bool, np.bool_)):
        raise ValueError("min_lodo_residual_df must be a positive integer")
    try:
        minimum_df = int(min_lodo_residual_df)
    except (TypeError, ValueError) as exc:
        raise ValueError("min_lodo_residual_df must be a positive integer") from exc
    if minimum_df != min_lodo_residual_df or minimum_df < 1:
        raise ValueError("min_lodo_residual_df must be a positive integer")

    selected_pathways = _normalize_pathways(fitted, pathways)
    raw, genes, donor_design, bins, effect, membership = _build_tensor(
        fitted, selected_pathways
    )
    (
        donor_design,
        reduced,
        condition,
        continuous,
        categorical,
        strata,
    ) = _prepare_design(fitted, donor_design)
    source_min_donors_value = fitted.metadata.get("min_donors_per_condition")
    if isinstance(source_min_donors_value, (bool, np.bool_)):
        raise ValueError(
            "fitted metadata min_donors_per_condition must be a positive integer"
        )
    try:
        source_min_donors = int(source_min_donors_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "fitted metadata min_donors_per_condition must be a positive integer"
        ) from exc
    if source_min_donors < 1 or source_min_donors != source_min_donors_value:
        raise ValueError(
            "fitted metadata min_donors_per_condition must be a positive integer"
        )
    try:
        source_max_condition_vif = float(
            fitted.metadata.get("max_condition_vif")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "fitted metadata max_condition_vif must be finite and at least 1"
        ) from exc
    if (
        not np.isfinite(source_max_condition_vif)
        or source_max_condition_vif < 1.0
    ):
        raise ValueError(
            "fitted metadata max_condition_vif must be finite and at least 1"
        )
    (
        standardized,
        center,
        observed_sample_sd,
        effective_scale,
        near_constant,
    ) = _standardize_tensor(raw)
    beta_raw, beta_z, influence = _fit_gene_curves(
        raw, standardized, reduced, condition
    )
    contribution_curves, max_closure_error = _contribution_tables(
        genes,
        beta_raw,
        beta_z,
        center,
        observed_sample_sd,
        effective_scale,
        near_constant,
        bins,
        effect,
        membership,
    )
    normalized_regions, region_source = _normalize_event_regions(
        fitted, selected_pathways, event_regions, set(bins)
    )
    event_definitions = _event_definitions(
        fitted, normalized_regions, region_source, effect, bins
    )
    donor_influence = _donor_influence_table(
        event_definitions,
        membership,
        genes,
        raw,
        influence,
        donor_design,
        effect,
        contribution_curves,
    )
    event_summary = _event_gene_summary(
        event_definitions,
        membership,
        contribution_curves,
        donor_influence,
        cumulative_abs_fraction,
    )
    if lodo:
        lodo_stability = _lodo_table(
            event_definitions,
            event_summary,
            raw,
            genes,
            donor_design,
            membership,
            effect,
            continuous,
            categorical,
            strata,
            cumulative_abs_fraction,
            minimum_df,
            source_min_donors,
            source_max_condition_vif,
        )
        event_summary = _attach_lodo_summary(event_summary, lodo_stability)
    else:
        lodo_stability = pd.DataFrame(columns=_LODO_COLUMNS)

    metadata = {
        "method": "covariate_adjusted_dynamic_leading_edge_decomposition",
        "source_method": fitted.metadata.get("method"),
        "source_pathway_family_hash": fitted.metadata.get("pathway_family_hash"),
        "source_gene_universe_hash": fitted.metadata.get("gene_universe_hash"),
        "source_fit_fingerprint": _fitted_source_fingerprint(fitted),
        "pathways": selected_pathways,
        "event_region_source": region_source,
        "n_events": int(len(event_definitions)),
        "n_genes": int(membership["gene"].nunique()),
        "n_donors": int(len(donor_design)),
        "contribution_formula": (
            "weight_g * standardized_gene_condition_beta / "
            "sum_abs_pathway_weights"
        ),
        "standardization": (
            "gene mean and sample SD across the fitted donor-by-bin pseudobulk "
            "grid; near-constant genes use effective scale 1 and contribute zero"
        ),
        "standardization_scale_columns": {
            "observed_sample_sd": "gene_pseudobulk_observed_sample_sd",
            "effective_scale": "gene_pseudobulk_effective_scale",
            "legacy_effective_scale_alias": "gene_pseudobulk_scale",
        },
        "maximum_absolute_closure_error": float(max_closure_error),
        "cumulative_abs_fraction": float(cumulative_abs_fraction),
        "lodo": bool(lodo),
        "lodo_refit": (
            "recompute_standardization_and_refit_on_remaining_donors_with_"
            "fixed_grid_pathways_and_event_bins"
            if lodo
            else "not_run"
        ),
        "min_lodo_residual_df": int(minimum_df),
        "lodo_min_donors_per_condition": int(source_min_donors),
        "lodo_max_condition_vif": float(source_max_condition_vif),
        "gene_level_p_values": False,
        "gene_level_q_values": False,
        "inference_boundary": (
            "mechanistic decomposition and donor sensitivity of fitted pathway "
            "effects; not gene-level inference"
        ),
        "donor_support_definition": (
            "Frisch-Waugh-Lovell condition-aligned reduced-residual influence"
        ),
    }
    for table in (
        contribution_curves,
        event_summary,
        donor_influence,
        lodo_stability,
    ):
        table.attrs["dynamic_leading_edge"] = metadata.copy()
    return DynamicLeadingEdgeResult(
        gene_contribution_curves=contribution_curves,
        event_gene_summary=event_summary,
        donor_influence=donor_influence,
        lodo_stability=lodo_stability,
        metadata=metadata,
    )


__all__ = [
    "DynamicLeadingEdgeResult",
    "decompose_covariate_adjusted_leading_edge",
]
