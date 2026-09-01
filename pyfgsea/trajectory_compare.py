from itertools import combinations
from typing import Optional, Sequence
import warnings

import numpy as np
import pandas as pd

from .trajectory import (
    _axis_sum_squares,
    _axis_sum,
    _detection_count,
    _axis_weighted_sum,
    _make_windows,
    _neighbor_indices,
    _normalize_ranker,
    _prepare_gene_sets_for_mode,
    _rank_gene_scores,
    run_trajectory_gsea,
)
from .trajectory_events import summarize_events
from .trajectory_alignment import run_aligned_trajectory_contrast
from .validation import _expression_matrix
from .wrapper import GseaRunner, prepare_pathways


def _unique_obs_values(adata, key: str) -> list:
    values = pd.Series(adata.obs[key]).dropna().unique().tolist()
    return sorted(values, key=lambda value: str(value))


def _subset_adata(adata, key: str, value):
    mask = pd.Series(adata.obs[key]).astype(str).to_numpy() == str(value)
    return adata[mask].copy()


def _bh_adjust_values(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(list(values), dtype=float)
    out = np.full(len(p), np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    pf = p[finite]
    order = np.argsort(pf)
    ranked = pf[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[finite] = restored
    return out


def _permute_labels_within_time_bins(
    labels: np.ndarray,
    pt: np.ndarray,
    *,
    n_bins: int,
    rng,
) -> np.ndarray:
    labels = np.asarray(labels).copy()
    pt = np.asarray(pt, dtype=float)
    finite = np.isfinite(pt)
    if finite.sum() == 0:
        rng.shuffle(labels)
        return labels
    ranks = pd.Series(pt[finite]).rank(method="first")
    bins = pd.qcut(
        ranks,
        q=min(int(n_bins), int(finite.sum())),
        labels=False,
        duplicates="drop",
    )
    out = labels.copy()
    finite_idx = np.where(finite)[0]
    for bin_id in pd.Series(bins).dropna().unique():
        idx = finite_idx[np.asarray(bins) == bin_id]
        out[idx] = rng.permutation(out[idx])
    return out


def _calibrate_aligned_contrast_with_null(
    observed: pd.DataFrame,
    null_events: pd.DataFrame,
    *,
    n_permutations: int,
    null_model: str,
    calibration_status: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if observed is None or observed.empty:
        return pd.DataFrame(), pd.DataFrame()
    calibrated = observed.copy()
    stats = pd.to_numeric(
        calibrated.get("observed_event_statistic", calibrated.get("contrast_C")),
        errors="coerce",
    ).to_numpy(dtype=float)
    if null_events is None or null_events.empty:
        calibrated["event_p"] = np.nan
        calibrated["event_q"] = np.nan
        calibrated["event_fdr"] = np.nan
        calibrated["n_perm"] = int(n_permutations)
        calibrated["minimum_attainable_p"] = np.nan
        calibrated["minimum_attainable_q"] = np.nan
        calibrated["null_model"] = null_model
        calibrated["calibration_status"] = "null_calibration_failed"
    else:
        null_values = pd.to_numeric(
            null_events.get("observed_event_statistic", null_events.get("contrast_C")),
            errors="coerce",
        ).dropna().to_numpy(dtype=float)
        p_values = []
        for observed_stat in stats:
            if not np.isfinite(observed_stat) or len(null_values) == 0:
                p_values.append(np.nan)
            else:
                p_values.append(
                    (1.0 + float(np.sum(null_values >= observed_stat)))
                    / (1.0 + float(len(null_values)))
                )
        q_values = _bh_adjust_values(p_values)
        min_p = 1.0 / (1.0 + float(len(null_values)))
        calibrated["event_p"] = p_values
        calibrated["event_q"] = q_values
        calibrated["event_fdr"] = q_values
        calibrated["n_perm"] = int(n_permutations)
        calibrated["minimum_attainable_p"] = min_p
        calibrated["minimum_attainable_q"] = np.minimum(1.0, len(calibrated) * min_p)
        calibrated["null_model"] = null_model
        calibrated["calibration_status"] = calibration_status
    fdr_cols = [
        col
        for col in (
            "condition_A",
            "condition_B",
            "pathway",
            "event_id",
            "event_statistic_used",
            "observed_event_statistic",
            "event_p",
            "event_q",
            "event_fdr",
            "n_perm",
            "minimum_attainable_p",
            "minimum_attainable_q",
            "null_model",
            "calibration_status",
            "alignment_anchor_set",
            "alignment_quality",
        )
        if col in calibrated.columns
    ]
    return calibrated, calibrated[fdr_cols].copy()


def _is_significant(row: pd.Series, label: str, fdr_threshold: float) -> bool:
    col = f"{label}_window_fdr_min"
    return col in row.index and np.isfinite(row[col]) and row[col] <= fdr_threshold


def _dominant_direction(row: pd.Series, label: str) -> str:
    peak = row.get(f"{label}_peak_NES", np.nan)
    trough = row.get(f"{label}_trough_NES", np.nan)
    if np.isfinite(peak) and np.isfinite(trough) and abs(trough) > abs(peak):
        return "suppression"
    if np.isfinite(peak) and peak < 0:
        return "suppression"
    return "activation"


def _comparison_interpretation(
    row: pd.Series,
    reference_label: str,
    query_label: str,
    time_tolerance: float,
    auc_tolerance: float,
    fdr_threshold: float,
) -> tuple[str, str]:
    ref_sig = _is_significant(row, reference_label, fdr_threshold)
    query_sig = _is_significant(row, query_label, fdr_threshold)
    query_direction = _dominant_direction(row, query_label)
    ref_direction = _dominant_direction(row, reference_label)

    if query_sig and not ref_sig:
        return f"gained {query_direction}", "specific_program"
    if ref_sig and not query_sig:
        return f"lost {ref_direction}", "specific_program"

    delta_peak = row.get("delta_peak_time", np.nan)
    delta_auc = row.get("delta_AUC", np.nan)
    if np.isfinite(delta_peak) and abs(delta_peak) > time_tolerance:
        if delta_peak < 0:
            return "earlier activation", "divergence_program"
        return "delayed activation", "divergence_program"

    if np.isfinite(delta_auc) and abs(delta_auc) > auc_tolerance:
        if delta_auc > 0:
            return "increased integrated activity", "divergence_program"
        return "decreased integrated activity", "divergence_program"

    if ref_sig and query_sig:
        return "shared dynamic program", "shared_program"
    return "no clear difference", "unclear"


def compare_baseline_event_tables(
    left_events: pd.DataFrame,
    right_events: pd.DataFrame,
    left_name: str = "PyFgsea-TED",
    right_name: str = "baseline",
    pathway_col: str = "Pathway",
    top_n: int = 25,
) -> pd.DataFrame:
    """Compare TED event summaries against score-then-smooth baseline events."""
    if left_events is None or left_events.empty or right_events is None or right_events.empty:
        return pd.DataFrame()
    if pathway_col not in left_events.columns or pathway_col not in right_events.columns:
        raise ValueError(f"Both event tables must contain '{pathway_col}'")

    merged = pd.merge(
        left_events,
        right_events,
        on=pathway_col,
        how="inner",
        suffixes=("_left", "_right"),
    )
    if merged.empty:
        return pd.DataFrame()

    auc_left = pd.to_numeric(merged.get("AUC_left"), errors="coerce")
    auc_right = pd.to_numeric(merged.get("AUC_right"), errors="coerce")
    finite_auc = np.isfinite(auc_left) & np.isfinite(auc_right)
    auc_correlation = (
        float(np.corrcoef(auc_left[finite_auc], auc_right[finite_auc])[0, 1])
        if int(finite_auc.sum()) >= 2
        else np.nan
    )

    left_rank = left_events.copy()
    right_rank = right_events.copy()
    left_rank["__strength"] = pd.to_numeric(
        left_rank.get("AUC_abs", left_rank.get("AUC", np.nan)), errors="coerce"
    ).abs()
    right_rank["__strength"] = pd.to_numeric(
        right_rank.get("AUC_abs", right_rank.get("AUC", np.nan)), errors="coerce"
    ).abs()
    left_top = set(left_rank.nlargest(top_n, "__strength")[pathway_col].astype(str))
    right_top = set(right_rank.nlargest(top_n, "__strength")[pathway_col].astype(str))
    top_event_overlap = (
        len(left_top & right_top) / max(len(left_top | right_top), 1)
        if left_top or right_top
        else np.nan
    )

    random_mask = merged[pathway_col].astype(str).str.contains(
        "random|shuffle|null|background", case=False, regex=True
    )
    false_positive_under_random_sets = float(random_mask.mean()) if len(merged) else np.nan
    runtime = np.nan
    for table in (left_events, right_events):
        baseline = table.attrs.get("baseline", {}) if hasattr(table, "attrs") else {}
        if "runtime_seconds" in baseline:
            runtime = baseline["runtime_seconds"] if not np.isfinite(runtime) else runtime + baseline["runtime_seconds"]

    left_labels = (
        merged["event_label_left"].astype(str)
        if "event_label_left" in merged
        else pd.Series([""] * len(merged), index=merged.index)
    )
    right_labels = (
        merged["event_label_right"].astype(str)
        if "event_label_right" in merged
        else pd.Series([""] * len(merged), index=merged.index)
    )
    out = pd.DataFrame(
        {
            pathway_col: merged[pathway_col],
            "left_name": left_name,
            "right_name": right_name,
            "left_event_label": left_labels,
            "right_event_label": right_labels,
            "event_label_agreement": left_labels == right_labels,
            "left_peak_time": pd.to_numeric(merged.get("peak_time_left"), errors="coerce"),
            "right_peak_time": pd.to_numeric(merged.get("peak_time_right"), errors="coerce"),
            "peak_time_delta": pd.to_numeric(merged.get("peak_time_right"), errors="coerce")
            - pd.to_numeric(merged.get("peak_time_left"), errors="coerce"),
            "left_AUC": auc_left,
            "right_AUC": auc_right,
            "AUC_delta": auc_right - auc_left,
            "AUC_correlation": auc_correlation,
            "top_event_overlap": top_event_overlap,
            "false_positive_under_random_sets": false_positive_under_random_sets,
            "runtime": runtime,
        }
    )
    out.attrs["comparison"] = {
        "left_name": left_name,
        "right_name": right_name,
        "top_n": int(top_n),
        "type": "event_table_baseline_comparison",
    }
    return out.sort_values("peak_time_delta", key=lambda s: s.abs(), na_position="last").reset_index(drop=True)


def compare_event_tables(
    events: pd.DataFrame,
    group_col=None,
    reference=None,
    query=None,
    pathway_col: str = "Pathway",
    reference_label: str = "control",
    query_label: str = "case",
    time_tolerance: float = 0.02,
    auc_tolerance: float = 0.25,
    fdr_threshold: float = 0.05,
    left_name: Optional[str] = None,
    right_name: Optional[str] = None,
    top_n: int = 25,
) -> pd.DataFrame:
    """Compare pathway event summaries between two conditions or branches."""
    if isinstance(group_col, pd.DataFrame):
        return compare_baseline_event_tables(
            events,
            group_col,
            left_name=left_name or reference or "PyFgsea-TED",
            right_name=right_name or query or "baseline",
            pathway_col=pathway_col,
            top_n=top_n,
        )
    if events is None or events.empty:
        return pd.DataFrame()
    if group_col is None:
        raise ValueError("group_col is required for condition/branch event comparison")
    if group_col not in events.columns:
        raise ValueError(f"Missing group column '{group_col}'")

    left = events[events[group_col].astype(str) == str(reference)].copy()
    right = events[events[group_col].astype(str) == str(query)].copy()
    if left.empty or right.empty:
        return pd.DataFrame()

    merged = pd.merge(
        left,
        right,
        on=pathway_col,
        how="outer",
        suffixes=(f"_{reference_label}", f"_{query_label}"),
    )

    rows = []
    for _, row in merged.iterrows():
        out = {
            pathway_col: row[pathway_col],
            "reference": reference,
            "query": query,
        }
        for metric in (
            "peak_time",
            "peak_NES",
            "trough_time",
            "trough_NES",
            "duration",
            "AUC",
            "window_fdr_min",
            "event_label",
        ):
            out[f"{reference_label}_{metric}"] = row.get(f"{metric}_{reference_label}", np.nan)
            out[f"{query_label}_{metric}"] = row.get(f"{metric}_{query_label}", np.nan)

        out["delta_peak_time"] = (
            out[f"{query_label}_peak_time"] - out[f"{reference_label}_peak_time"]
        )
        out["delta_AUC"] = out[f"{query_label}_AUC"] - out[f"{reference_label}_AUC"]
        out["delta_duration"] = (
            out[f"{query_label}_duration"] - out[f"{reference_label}_duration"]
        )
        interpretation, program_type = _comparison_interpretation(
            pd.Series(out),
            reference_label,
            query_label,
            time_tolerance,
            auc_tolerance,
            fdr_threshold,
        )
        out["interpretation"] = interpretation
        out["program_type"] = program_type
        rows.append(out)

    return pd.DataFrame(rows).sort_values(
        ["program_type", "interpretation", pathway_col]
    ).reset_index(drop=True)


def _sample_condition_map(adata, condition_key: str, sample_key: str) -> dict[str, str]:
    if sample_key not in adata.obs:
        raise ValueError(f"sample_key '{sample_key}' not found in adata.obs")
    if condition_key not in adata.obs:
        raise ValueError(f"condition_key '{condition_key}' not found in adata.obs")

    frame = adata.obs[[sample_key, condition_key]].dropna().astype(str)
    mapping = {}
    mixed = []
    for sample, group in frame.groupby(sample_key, sort=False):
        values = group[condition_key].unique().tolist()
        if len(values) != 1:
            mixed.append(str(sample))
        else:
            mapping[str(sample)] = str(values[0])
    if mixed:
        examples = ", ".join(mixed[:5])
        raise ValueError(
            "Each biological sample must map to exactly one condition for "
            f"replicate-aware comparison. Mixed samples: {examples}"
        )
    return mapping


def _median_or_nan(values) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    return float(np.nanmedian(arr))


def _mode_string(values) -> str:
    series = pd.Series(values).dropna().astype(str)
    if series.empty:
        return ""
    return str(series.value_counts().index[0])


def _sample_consistency(control_auc: np.ndarray, case_auc: np.ndarray) -> float:
    control_auc = np.asarray(control_auc, dtype=float)
    case_auc = np.asarray(case_auc, dtype=float)
    control_auc = control_auc[np.isfinite(control_auc)]
    case_auc = case_auc[np.isfinite(case_auc)]
    if len(control_auc) == 0 or len(case_auc) == 0:
        return np.nan

    control_median = float(np.nanmedian(control_auc))
    case_median = float(np.nanmedian(case_auc))
    delta = case_median - control_median
    if delta == 0:
        return np.nan
    if delta > 0:
        flags = np.concatenate([case_auc >= control_median, control_auc <= case_median])
    else:
        flags = np.concatenate([case_auc <= control_median, control_auc >= case_median])
    return float(np.mean(flags)) if len(flags) else np.nan


def _pseudobulk_rank_scores(
    sample_means: np.ndarray,
    sample_conditions: np.ndarray,
    control,
    case,
    ranker: str,
) -> np.ndarray:
    ranker = ranker.lower().replace("-", "_")
    aliases = {
        "mean_difference": "mean_diff",
        "welch_t": "t_stat",
        "t": "t_stat",
        "z": "z_score",
        "cohen_d": "cohens_d",
    }
    ranker = aliases.get(ranker, ranker)

    control_mat = sample_means[sample_conditions.astype(str) == str(control)]
    case_mat = sample_means[sample_conditions.astype(str) == str(case)]
    if len(control_mat) == 0 or len(case_mat) == 0:
        return np.zeros(sample_means.shape[1], dtype=float)

    control_mean = control_mat.mean(axis=0)
    case_mean = case_mat.mean(axis=0)
    diff = case_mean - control_mean
    if ranker == "mean_diff":
        return diff

    control_var = control_mat.var(axis=0, ddof=1) if len(control_mat) > 1 else np.zeros(sample_means.shape[1])
    case_var = case_mat.var(axis=0, ddof=1) if len(case_mat) > 1 else np.zeros(sample_means.shape[1])
    if ranker in {"t_stat", "z_score"}:
        denom = np.sqrt(case_var / max(len(case_mat), 1) + control_var / max(len(control_mat), 1))
        return diff / np.maximum(denom, 1e-12)
    if ranker == "cohens_d":
        denom_n = max(len(case_mat) + len(control_mat) - 2, 1)
        pooled = ((len(case_mat) - 1) * case_var + (len(control_mat) - 1) * control_var) / denom_n
        return diff / np.maximum(np.sqrt(pooled), 1e-12)

    raise ValueError("pseudobulk_ranker must be one of mean_diff, t_stat, z_score, or cohens_d")


def _event_direction(row: pd.Series) -> str:
    peak = row.get("peak_NES", np.nan)
    trough = row.get("trough_NES", np.nan)
    if np.isfinite(trough) and abs(trough) > abs(peak):
        return "case_depleted"
    if np.isfinite(peak) and peak < 0:
        return "case_depleted"
    return "case_enriched"


def _events_to_pseudobulk_comparison(
    events: pd.DataFrame,
    control,
    case,
    pathway_col: str = "Pathway",
) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame()
    rows = []
    for _, row in events.iterrows():
        direction = _event_direction(row)
        out = row.to_dict()
        out["reference"] = control
        out["query"] = case
        out["delta_AUC"] = row.get("AUC", np.nan)
        out["delta_peak_time"] = row.get("peak_time", np.nan)
        out["sample_consistency"] = np.nan
        out["interpretation"] = (
            "case-enriched trajectory event"
            if direction == "case_enriched"
            else "case-depleted trajectory event"
        )
        out["program_type"] = "pseudobulk_condition_event"
        out["event_type"] = direction
        rows.append(out)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["window_fdr_min", pathway_col], na_position="last"
    ).reset_index(drop=True)


def _pseudobulk_condition_results(
    adata,
    gmt_path: str,
    condition_key: str,
    sample_key: str,
    control,
    case,
    pseudotime_key: str,
    sample_to_condition: dict[str, str],
    window_size: int,
    step: int,
    window_mode: str,
    min_cells: Optional[int],
    max_cells: Optional[int],
    target_span: Optional[float],
    span_step: Optional[float],
    min_cells_per_sample: int,
    min_samples_per_condition: int,
    pseudobulk_ranker: str,
    min_size: int,
    max_size: int,
    sample_size: int,
    seed: int,
    eps: float,
    nperm_nes: int,
    bin_width: int,
    calculate_nes: bool,
    use_nes_cache: bool,
    layer: Optional[str],
    use_raw: bool,
    gene_set_mode: str,
    min_abs_gene_weight: float,
    gsea_param: float,
    gsea_mode: str,
    score_type: str,
    tie_policy: str,
    nperm_simple: Optional[int],
    max_levels: Optional[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pseudotime_key not in adata.obs:
        raise ValueError(f"pseudotime_key '{pseudotime_key}' not found in adata.obs")

    pt = pd.to_numeric(adata.obs[pseudotime_key], errors="coerce").to_numpy(dtype=float)
    condition_values = adata.obs[condition_key].astype(str).to_numpy()
    sample_values = adata.obs[sample_key].astype(str).to_numpy()
    keep = (
        np.isfinite(pt)
        & np.isin(condition_values, [str(control), str(case)])
        & np.isin(sample_values, list(sample_to_condition))
    )
    if not keep.any():
        return pd.DataFrame(), pd.DataFrame()

    work = adata[keep].copy()
    pt = pt[keep]
    sample_values = sample_values[keep]
    X, genes, expression_source = _expression_matrix(work, layer=layer, use_raw=use_raw)
    genes = np.asarray(genes)
    duplicated = pd.Index(genes).duplicated()
    if duplicated.any():
        examples = ", ".join(map(str, pd.Index(genes)[duplicated][:5]))
        raise ValueError(f"Expression gene names are duplicated. Examples: {examples}")

    gene_sets = _prepare_gene_sets_for_mode(
        gmt_path,
        gene_set_mode=gene_set_mode,
        min_abs_gene_weight=min_abs_gene_weight,
    )
    pathway_names, pathway_indices = prepare_pathways(genes, gene_sets, min_size, max_size)
    if not pathway_indices:
        return pd.DataFrame(), pd.DataFrame()
    runner = GseaRunner(
        pathway_names,
        pathway_indices,
        min_size,
        max_size,
        gene_ids=genes,
        tie_policy=tie_policy,
    )

    order = np.argsort(pt)
    windows = _make_windows(
        order,
        window_size=window_size,
        step=step,
        pt=pt,
        window_mode=window_mode,
        min_cells=min_cells,
        max_cells=max_cells,
        target_span=target_span,
        span_step=span_step,
    )

    result_frames = []
    diagnostic_rows = []
    sample_order = sorted(sample_to_condition, key=str)
    for wi, (_s, _e, window_indices) in enumerate(windows):
        sample_means = []
        sample_conditions = []
        sample_counts = []
        for sample in sample_order:
            idx = window_indices[sample_values[window_indices] == str(sample)]
            if len(idx) < min_cells_per_sample:
                continue
            sample_means.append(_axis_sum(X, idx) / max(len(idx), 1))
            sample_conditions.append(sample_to_condition[str(sample)])
            sample_counts.append(len(idx))

        if not sample_means:
            continue
        sample_means = np.vstack(sample_means).astype(float)
        sample_conditions = np.asarray(sample_conditions, dtype=str)
        n_control = int(np.sum(sample_conditions == str(control)))
        n_case = int(np.sum(sample_conditions == str(case)))
        diagnostic_rows.append(
            {
                "window_id": wi,
                "pt_start": float(np.nanmin(pt[window_indices])),
                "pt_end": float(np.nanmax(pt[window_indices])),
                "pt_mid": float((np.nanmin(pt[window_indices]) + np.nanmax(pt[window_indices])) / 2.0),
                "n_cells": int(len(window_indices)),
                "n_pseudobulk_samples": int(len(sample_conditions)),
                "n_control_samples": n_control,
                "n_case_samples": n_case,
                "median_cells_per_sample": float(np.median(sample_counts)),
            }
        )
        if n_control < min_samples_per_condition or n_case < min_samples_per_condition:
            continue

        scores = _pseudobulk_rank_scores(
            sample_means,
            sample_conditions,
            control=control,
            case=case,
            ranker=pseudobulk_ranker,
        )
        scores = np.asarray(scores, dtype=np.float64)
        scores[~np.isfinite(scores)] = 0.0

        res = runner.run(
            scores,
            sample_size=sample_size,
            seed=seed,
            eps=eps,
            nperm_nes=nperm_nes,
            gsea_param=gsea_param,
            score_type=score_type,
            bin_width=bin_width,
            calculate_nes=calculate_nes,
            use_nes_cache=use_nes_cache,
            mode=gsea_mode,
            nperm_simple=nperm_simple,
            max_levels=max_levels,
        )
        if res.empty:
            continue
        pt_vals = pt[window_indices]
        pt_start = float(np.nanmin(pt_vals))
        pt_end = float(np.nanmax(pt_vals))
        res = res.copy()
        res["window_id"] = wi
        res["pt_start"] = pt_start
        res["pt_end"] = pt_end
        res["pt_mid"] = float((pt_start + pt_end) / 2.0)
        res["n_cells"] = int(len(window_indices))
        res["n_pseudobulk_samples"] = int(len(sample_conditions))
        res["n_control_samples"] = n_control
        res["n_case_samples"] = n_case
        res["median_cells_per_sample"] = float(np.median(sample_counts))
        res["ranker"] = f"pseudobulk_{pseudobulk_ranker}"
        res["window_mode"] = window_mode
        res["expression_source"] = expression_source
        res["reference"] = control
        res["query"] = case
        result_frames.append(res)

    results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    diagnostics = pd.DataFrame(diagnostic_rows)
    return results, diagnostics


def _summarize_events_by_group(
    results: pd.DataFrame,
    group_col: str,
    event_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    if results is None or results.empty:
        return pd.DataFrame()
    if group_col not in results.columns:
        raise ValueError(f"Missing group column '{group_col}'")

    event_kwargs = {} if event_kwargs is None else dict(event_kwargs)
    frames = []
    for value, group in results.groupby(group_col, sort=False):
        events = summarize_events(group, **event_kwargs)
        if not events.empty:
            events[group_col] = value
            frames.append(events)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _sample_rank_stat(
    X,
    pt: np.ndarray,
    sample_all: np.ndarray,
    window_indices: np.ndarray,
    ranker: str,
    smooth_slope_bandwidth: Optional[float] = None,
) -> np.ndarray:
    sample_all = np.asarray(sample_all, dtype=int)
    sample_set = set(map(int, sample_all))
    sample_window = [idx for idx in window_indices if int(idx) in sample_set]
    if not sample_window:
        return np.array([], dtype=float)

    X_sample = X[sample_all]
    pt_sample = pt[sample_all]
    local_lookup = {int(global_idx): local_idx for local_idx, global_idx in enumerate(sample_all)}
    local_window = np.asarray([local_lookup[int(idx)] for idx in sample_window], dtype=int)

    ranker = _normalize_ranker(ranker)
    sum_total = _axis_sum(X_sample)
    sum_sq_total = (
        _axis_sum_squares(X_sample)
        if ranker in {"t_stat", "z_score", "cohens_d"}
        else None
    )
    det_total = _detection_count(X_sample) if ranker == "detection_weighted" else None
    neighbor_indices = None
    if ranker == "neighbor_contrast":
        local_order = np.argsort(pt_sample)
        pos = np.where(np.isin(local_order, local_window))[0]
        if len(pos):
            neighbor_indices = _neighbor_indices(local_order, int(pos.min()), int(pos.max()) + 1)

    scores = _rank_gene_scores(
        X_sample,
        local_window,
        ranker=ranker,
        sum_total=sum_total,
        n_all=X_sample.shape[0],
        sum_sq_total=sum_sq_total,
        det_total=det_total,
        pt=pt_sample,
        neighbor_indices=neighbor_indices,
        smooth_center=float(np.nanmedian(pt_sample[local_window])),
        smooth_bandwidth=smooth_slope_bandwidth,
    )
    scores = np.asarray(scores, dtype=float)
    scores[~np.isfinite(scores)] = 0.0
    return scores


def _row_sum(X) -> np.ndarray:
    return np.asarray(X.sum(axis=1)).ravel().astype(float)


def _row_detection(X) -> np.ndarray:
    return np.asarray((X > 0).sum(axis=1)).ravel().astype(float)


def _smd(left: np.ndarray, right: np.ndarray) -> float:
    left = pd.to_numeric(pd.Series(left), errors="coerce").dropna().to_numpy(dtype=float)
    right = pd.to_numeric(pd.Series(right), errors="coerce").dropna().to_numpy(dtype=float)
    if len(left) == 0 or len(right) == 0:
        return np.nan
    var = (float(np.var(left, ddof=1)) if len(left) > 1 else 0.0) + (
        float(np.var(right, ddof=1)) if len(right) > 1 else 0.0
    )
    pooled = np.sqrt(var / 2.0)
    diff = float(np.mean(right) - np.mean(left))
    if pooled <= 0:
        return 0.0 if abs(diff) <= 0 else np.inf
    return diff / pooled


def _nearest_to_center(
    indices: np.ndarray,
    pt: np.ndarray,
    center: float,
    n: int,
    *,
    covariate: Optional[np.ndarray] = None,
    covariate_center: Optional[float] = None,
    covariate_weight: float = 0.0,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if len(indices) <= n:
        return indices
    score = np.abs(pt[indices] - center)
    if covariate is not None and covariate_weight > 0:
        cov = np.asarray(covariate, dtype=float)
        local = cov[indices]
        center_cov = (
            float(covariate_center)
            if covariate_center is not None and np.isfinite(covariate_center)
            else float(np.nanmedian(local))
        )
        scale = float(np.nanstd(local))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        score = score + float(covariate_weight) * np.abs(local - center_cov) / scale
    order = np.argsort(score, kind="mergesort")
    return indices[order[:n]]


def _matched_window_diagnostics(
    *,
    window_id: int,
    pt: np.ndarray,
    qc: dict[str, np.ndarray],
    control_all: np.ndarray,
    case_all: np.ndarray,
    control_selected: np.ndarray,
    case_selected: np.ndarray,
    pt_start: float,
    pt_end: float,
    pt_mid: float,
    balance_smd_threshold: float,
    min_effective_cells: int,
    window_merge_level: int = 0,
    skip_reason: str = "",
) -> dict:
    row = {
        "window_id": int(window_id),
        "window_merge_level": int(window_merge_level),
        "pt_start": float(pt_start),
        "pt_end": float(pt_end),
        "pt_mid": float(pt_mid),
        "n_control": int(len(control_all)),
        "n_case": int(len(case_all)),
        "effective_n_control": int(len(control_selected)),
        "effective_n_case": int(len(case_selected)),
        "skip_reason": skip_reason,
    }
    checks = []
    smd_pass_by_name = {}
    for name, values in {"pt": pt, **qc}.items():
        before = _smd(values[control_all], values[case_all])
        after = _smd(values[control_selected], values[case_selected])
        row[f"{name}_smd_before"] = before
        row[f"{name}_smd_after"] = after
        row[f"{name}_smd_after_pass"] = bool(
            np.isfinite(after) and abs(after) <= balance_smd_threshold
        )
        smd_pass_by_name[name] = bool(row[f"{name}_smd_after_pass"])
        if np.isfinite(after):
            checks.append(abs(after) <= balance_smd_threshold)
    imbalance_values = np.asarray(
        [
            row.get("pt_smd_after", np.nan),
            row.get("detection_rate_smd_after", np.nan),
            row.get("n_counts_smd_after", np.nan),
            row.get("n_genes_smd_after", np.nan),
        ],
        dtype=float,
    )
    finite_imbalance = np.abs(imbalance_values[np.isfinite(imbalance_values)])
    row["imbalance_score"] = (
        float(finite_imbalance.max()) if finite_imbalance.size else np.inf
    )
    row["effective_n_control_pass"] = bool(row["effective_n_control"] >= min_effective_cells)
    row["effective_n_case_pass"] = bool(row["effective_n_case"] >= min_effective_cells)
    effective_ratio = min(row["effective_n_control"], row["effective_n_case"]) / max(
        int(min_effective_cells), 1
    )
    effective_ratio = float(min(max(effective_ratio, 0.0), 1.0))
    smd_score = 1.0 - min(float(row["imbalance_score"]) / max(balance_smd_threshold, 1e-12), 1.0)
    row["balance_score"] = float(max(0.0, smd_score) * effective_ratio)
    row["balance_pass"] = bool(
        checks
        and all(checks)
        and row["effective_n_control_pass"]
        and row["effective_n_case_pass"]
    )
    core_values = np.asarray(
        [
            row.get("pt_smd_after", np.nan),
            row.get("detection_rate_smd_after", np.nan),
            row.get("n_genes_smd_after", np.nan),
        ],
        dtype=float,
    )
    finite_core = np.abs(core_values[np.isfinite(core_values)])
    row["core_imbalance_score"] = float(finite_core.max()) if finite_core.size else np.inf
    core_smd_score = 1.0 - min(
        float(row["core_imbalance_score"]) / max(balance_smd_threshold, 1e-12), 1.0
    )
    row["core_balance_score"] = float(max(0.0, core_smd_score) * effective_ratio)
    row["core_balance_pass"] = bool(
        all(smd_pass_by_name.get(name, False) for name in ("pt", "detection_rate", "n_genes"))
        and row["effective_n_control_pass"]
        and row["effective_n_case_pass"]
    )
    return row


def _matched_condition_results(
    adata,
    gmt_path: str,
    condition_key: str,
    control,
    case,
    pseudotime_key: str,
    window_size: int,
    step: int,
    ranker: str,
    window_mode: str,
    min_cells: Optional[int],
    max_cells: Optional[int],
    target_span: Optional[float],
    span_step: Optional[float],
    balance: str,
    n_balance_resamples: int,
    balance_smd_threshold: float,
    n_counts_balance_weight: float,
    max_window_merge: int,
    min_size: int,
    max_size: int,
    sample_size: int,
    seed: int,
    eps: float,
    nperm_nes: int,
    bin_width: int,
    calculate_nes: bool,
    use_nes_cache: bool,
    layer: Optional[str],
    use_raw: bool,
    gene_set_mode: str,
    min_abs_gene_weight: float,
    gsea_param: float,
    gsea_mode: str,
    score_type: str,
    tie_policy: str,
    nperm_simple: Optional[int],
    max_levels: Optional[int],
    smooth_slope_bandwidth: Optional[float] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pseudotime_key not in adata.obs:
        raise ValueError(f"pseudotime_key '{pseudotime_key}' not found in adata.obs")

    balance = str(balance).lower().replace("-", "_")
    if balance in {"density", "density_weighted", "inverse_density"}:
        balance = "weights"
    if balance not in {"weights", "resample", "none"}:
        raise ValueError("balance must be one of 'weights', 'resample', or 'none'")

    pt = pd.to_numeric(adata.obs[pseudotime_key], errors="coerce").to_numpy(dtype=float)
    condition_values = adata.obs[condition_key].astype(str).to_numpy()
    keep = np.isfinite(pt) & np.isin(condition_values, [str(control), str(case)])
    if not keep.any():
        return pd.DataFrame(), pd.DataFrame()

    work = adata[keep].copy()
    pt = pt[keep]
    condition_values = condition_values[keep]
    X, genes, expression_source = _expression_matrix(work, layer=layer, use_raw=use_raw)
    genes = np.asarray(genes)
    duplicated = pd.Index(genes).duplicated()
    if duplicated.any():
        examples = ", ".join(map(str, pd.Index(genes)[duplicated][:5]))
        raise ValueError(f"Expression gene names are duplicated. Examples: {examples}")

    gene_sets = _prepare_gene_sets_for_mode(
        gmt_path,
        gene_set_mode=gene_set_mode,
        min_abs_gene_weight=min_abs_gene_weight,
    )
    pathway_names, pathway_indices = prepare_pathways(genes, gene_sets, min_size, max_size)
    if not pathway_indices:
        return pd.DataFrame(), pd.DataFrame()
    runner = GseaRunner(
        pathway_names,
        pathway_indices,
        min_size,
        max_size,
        gene_ids=genes,
        tie_policy=tie_policy,
    )
    ranker = _normalize_ranker(ranker)

    ordered = np.argsort(pt)
    common_windows = _make_windows(
        ordered,
        window_size=window_size,
        step=step,
        pt=pt,
        window_mode=window_mode,
        min_cells=min_cells,
        max_cells=max_cells,
        target_span=target_span,
        span_step=span_step,
    )
    min_window_cells = int(
        min_cells if min_cells is not None else max(3, min(max(window_size // 4, 1), 20))
    )
    max_window_merge = max(0, int(max_window_merge))
    rng = np.random.default_rng(seed)
    qc = {
        "detection_rate": _row_detection(X) / max(int(X.shape[1]), 1),
        "n_counts": _row_sum(X),
        "n_genes": _row_detection(X),
    }
    control_all = np.where(condition_values == str(control))[0]
    case_all = np.where(condition_values == str(case))[0]
    condition_all = {str(control): control_all, str(case): case_all}
    result_frames = []
    diagnostic_rows = []

    for wi, (_s, _e, _window_indices) in enumerate(common_windows):
        best = None
        for merge_level in range(max_window_merge + 1):
            left = max(0, wi - merge_level)
            right = min(len(common_windows), wi + merge_level + 1)
            merged = np.unique(
                np.concatenate(
                    [np.asarray(common_windows[idx][2], dtype=int) for idx in range(left, right)]
                )
            )
            if len(merged) == 0:
                continue
            pt_vals = pt[merged]
            pt_start = float(np.nanmin(pt_vals))
            pt_end = float(np.nanmax(pt_vals))
            pt_mid = float((pt_start + pt_end) / 2.0)
            window_by_condition = {
                str(control): merged[condition_values[merged] == str(control)],
                str(case): merged[condition_values[merged] == str(case)],
            }
            n_match = min(
                len(window_by_condition[str(control)]),
                len(window_by_condition[str(case)]),
            )
            if max_cells is not None:
                n_match = min(n_match, int(max_cells))
            if balance == "none":
                n_match = max(
                    len(window_by_condition[str(control)]),
                    len(window_by_condition[str(case)]),
                )
            if n_match < min_window_cells:
                diag = _matched_window_diagnostics(
                    window_id=wi,
                    pt=pt,
                    qc=qc,
                    control_all=window_by_condition[str(control)],
                    case_all=window_by_condition[str(case)],
                    control_selected=np.array([], dtype=int),
                    case_selected=np.array([], dtype=int),
                    pt_start=pt_start,
                    pt_end=pt_end,
                    pt_mid=pt_mid,
                    balance_smd_threshold=balance_smd_threshold,
                    min_effective_cells=min_window_cells,
                    window_merge_level=merge_level,
                    skip_reason="insufficient_matched_cells",
                )
                candidate = (diag, window_by_condition, {}, n_match)
            else:
                selected_for_diag = {}
                n_counts_center = float(np.nanmedian(qc["n_counts"][merged]))
                for condition in (str(control), str(case)):
                    candidates = window_by_condition[condition]
                    selected_for_diag[condition] = (
                        candidates
                        if balance == "none"
                        else _nearest_to_center(
                            candidates,
                            pt,
                            pt_mid,
                            n_match,
                            covariate=qc["n_counts"],
                            covariate_center=n_counts_center,
                            covariate_weight=n_counts_balance_weight,
                        )
                    )
                diag = _matched_window_diagnostics(
                    window_id=wi,
                    pt=pt,
                    qc=qc,
                    control_all=window_by_condition[str(control)],
                    case_all=window_by_condition[str(case)],
                    control_selected=selected_for_diag[str(control)],
                    case_selected=selected_for_diag[str(case)],
                    pt_start=pt_start,
                    pt_end=pt_end,
                    pt_mid=pt_mid,
                    balance_smd_threshold=balance_smd_threshold,
                    min_effective_cells=min_window_cells,
                    window_merge_level=merge_level,
                )
                candidate = (diag, window_by_condition, selected_for_diag, n_match)
            if best is None or candidate[0].get("balance_score", 0.0) > best[0].get("balance_score", 0.0):
                best = candidate
            if bool(candidate[0].get("balance_pass", False)):
                best = candidate
                break
        if best is None:
            continue
        diag, window_by_condition, selected_for_diag, n_match = best
        pt_start = float(diag.get("pt_start", np.nan))
        pt_end = float(diag.get("pt_end", np.nan))
        pt_mid = float(diag.get("pt_mid", np.nan))
        diagnostic_rows.append(diag)
        if not selected_for_diag:
            continue

        for condition in (str(control), str(case)):
            sample_all = condition_all[condition]
            candidates = window_by_condition[condition]
            score_reps = []
            reps = max(1, int(n_balance_resamples)) if balance == "resample" else 1
            for _ in range(reps):
                if balance == "resample" and len(candidates) > n_match:
                    selected = rng.choice(candidates, size=n_match, replace=False)
                elif balance == "none":
                    selected = candidates
                else:
                    selected = selected_for_diag[condition]
                scores = _sample_rank_stat(
                    X,
                    pt,
                    sample_all,
                    selected,
                    ranker=ranker,
                    smooth_slope_bandwidth=smooth_slope_bandwidth,
                )
                if scores.size:
                    score_reps.append(scores)
            if not score_reps:
                continue
            scores = np.mean(np.vstack(score_reps), axis=0)
            scores = np.asarray(scores, dtype=np.float64)
            scores[~np.isfinite(scores)] = 0.0
            res = runner.run(
                scores,
                sample_size=sample_size,
                seed=seed,
                eps=eps,
                nperm_nes=nperm_nes,
                gsea_param=gsea_param,
                score_type=score_type,
                bin_width=bin_width,
                calculate_nes=calculate_nes,
                use_nes_cache=use_nes_cache,
                mode=gsea_mode,
                nperm_simple=nperm_simple,
                max_levels=max_levels,
            )
            if res.empty:
                continue
            selected = selected_for_diag[condition] if balance != "none" else candidates
            res = res.copy()
            res["window_id"] = int(wi)
            res["pt_start"] = pt_start
            res["pt_end"] = pt_end
            res["pt_mid"] = pt_mid
            res["n_cells"] = int(len(selected))
            res["n_control"] = int(len(window_by_condition[str(control)]))
            res["n_case"] = int(len(window_by_condition[str(case)]))
            res["effective_n_control"] = int(len(selected_for_diag[str(control)]))
            res["effective_n_case"] = int(len(selected_for_diag[str(case)]))
            res["window_merge_level"] = int(diag.get("window_merge_level", 0))
            res["balance_pass"] = bool(diag.get("balance_pass", False))
            res["balance_score"] = float(diag.get("balance_score", np.nan))
            res["imbalance_score"] = float(diag.get("imbalance_score", np.nan))
            res["ranker"] = f"matched_{balance}_{ranker}"
            res["base_ranker"] = ranker
            res["window_mode"] = "matched_window"
            res["balance"] = balance
            res["expression_source"] = expression_source
            res[condition_key] = condition
            result_frames.append(res)

    results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    diagnostics = pd.DataFrame(diagnostic_rows)
    return results, diagnostics


def _sample_balanced_condition_results(
    adata,
    gmt_path: str,
    condition_key: str,
    sample_key: str,
    control,
    case,
    pseudotime_key: str,
    sample_to_condition: dict[str, str],
    window_size: int,
    step: int,
    ranker: str,
    window_mode: str,
    min_cells: Optional[int],
    max_cells: Optional[int],
    target_span: Optional[float],
    span_step: Optional[float],
    min_cells_per_replicate: int,
    min_replicates_per_condition: int,
    min_size: int,
    max_size: int,
    sample_size: int,
    seed: int,
    eps: float,
    nperm_nes: int,
    bin_width: int,
    calculate_nes: bool,
    use_nes_cache: bool,
    layer: Optional[str],
    use_raw: bool,
    gene_set_mode: str,
    min_abs_gene_weight: float,
    gsea_param: float,
    gsea_mode: str,
    score_type: str,
    tie_policy: str,
    nperm_simple: Optional[int],
    max_levels: Optional[int],
    smooth_slope_bandwidth: Optional[float] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pseudotime_key not in adata.obs:
        raise ValueError(f"pseudotime_key '{pseudotime_key}' not found in adata.obs")

    pt = pd.to_numeric(adata.obs[pseudotime_key], errors="coerce").to_numpy(dtype=float)
    condition_values = adata.obs[condition_key].astype(str).to_numpy()
    sample_values = adata.obs[sample_key].astype(str).to_numpy()
    keep = (
        np.isfinite(pt)
        & np.isin(condition_values, [str(control), str(case)])
        & np.isin(sample_values, list(sample_to_condition))
    )
    if not keep.any():
        return pd.DataFrame(), pd.DataFrame()

    work = adata[keep].copy()
    pt = pt[keep]
    condition_values = condition_values[keep]
    sample_values = sample_values[keep]
    X, genes, expression_source = _expression_matrix(work, layer=layer, use_raw=use_raw)
    genes = np.asarray(genes)
    duplicated = pd.Index(genes).duplicated()
    if duplicated.any():
        examples = ", ".join(map(str, pd.Index(genes)[duplicated][:5]))
        raise ValueError(f"Expression gene names are duplicated. Examples: {examples}")

    gene_sets = _prepare_gene_sets_for_mode(
        gmt_path,
        gene_set_mode=gene_set_mode,
        min_abs_gene_weight=min_abs_gene_weight,
    )
    pathway_names, pathway_indices = prepare_pathways(genes, gene_sets, min_size, max_size)
    if not pathway_indices:
        return pd.DataFrame(), pd.DataFrame()
    runner = GseaRunner(
        pathway_names,
        pathway_indices,
        min_size,
        max_size,
        gene_ids=genes,
        tie_policy=tie_policy,
    )
    ranker = _normalize_ranker(ranker)

    result_frames = []
    diagnostic_rows = []
    for condition in (control, case):
        condition = str(condition)
        condition_indices = np.where(condition_values == condition)[0]
        if len(condition_indices) == 0:
            continue
        condition_samples = [
            sample
            for sample, sample_condition in sample_to_condition.items()
            if str(sample_condition) == condition
        ]
        ordered = condition_indices[np.argsort(pt[condition_indices])]
        windows = _make_windows(
            ordered,
            window_size=window_size,
            step=step,
            pt=pt,
            window_mode=window_mode,
            min_cells=min_cells,
            max_cells=max_cells,
            target_span=target_span,
            span_step=span_step,
        )

        for wi, (_s, _e, window_indices) in enumerate(windows):
            sample_scores = []
            sample_counts = []
            included_samples = []
            for sample in condition_samples:
                sample_all = np.where(
                    (condition_values == condition) & (sample_values == str(sample))
                )[0]
                sample_window_count = int(
                    np.isin(window_indices, sample_all, assume_unique=False).sum()
                )
                if sample_window_count < min_cells_per_replicate:
                    continue
                if len(sample_all) <= sample_window_count and ranker not in {
                    "local_slope",
                    "smooth_slope",
                }:
                    continue
                scores = _sample_rank_stat(
                    X,
                    pt,
                    sample_all,
                    window_indices,
                    ranker=ranker,
                    smooth_slope_bandwidth=smooth_slope_bandwidth,
                )
                if scores.size == 0:
                    continue
                sample_scores.append(scores)
                sample_counts.append(sample_window_count)
                included_samples.append(sample)

            pt_vals = pt[window_indices]
            n_reps = len(included_samples)
            replicate_support = (
                float(n_reps / max(len(condition_samples), 1))
                if condition_samples
                else np.nan
            )
            diagnostic_rows.append(
                {
                    "condition": condition,
                    "window_id": wi,
                    "pt_start": float(np.nanmin(pt_vals)),
                    "pt_end": float(np.nanmax(pt_vals)),
                    "pt_mid": float((np.nanmin(pt_vals) + np.nanmax(pt_vals)) / 2.0),
                    "n_cells": int(len(window_indices)),
                    "n_replicates": int(n_reps),
                    "n_replicates_total": int(len(condition_samples)),
                    "replicate_support": replicate_support,
                    "min_cells_per_replicate": int(min(sample_counts)) if sample_counts else 0,
                    "included_replicates": ";".join(map(str, included_samples)),
                }
            )
            if n_reps < min_replicates_per_condition:
                continue

            scores = np.mean(np.vstack(sample_scores), axis=0)
            scores = np.asarray(scores, dtype=np.float64)
            scores[~np.isfinite(scores)] = 0.0
            res = runner.run(
                scores,
                sample_size=sample_size,
                seed=seed,
                eps=eps,
                nperm_nes=nperm_nes,
                gsea_param=gsea_param,
                score_type=score_type,
                bin_width=bin_width,
                calculate_nes=calculate_nes,
                use_nes_cache=use_nes_cache,
                mode=gsea_mode,
                nperm_simple=nperm_simple,
                max_levels=max_levels,
            )
            if res.empty:
                continue
            pt_start = float(np.nanmin(pt_vals))
            pt_end = float(np.nanmax(pt_vals))
            res = res.copy()
            res["window_id"] = wi
            res["pt_start"] = pt_start
            res["pt_end"] = pt_end
            res["pt_mid"] = float((pt_start + pt_end) / 2.0)
            res["n_cells"] = int(len(window_indices))
            res["n_replicates"] = int(n_reps)
            res["n_replicates_total"] = int(len(condition_samples))
            res["replicate_support"] = replicate_support
            res["min_cells_per_replicate"] = int(min(sample_counts))
            res["ranker"] = f"sample_balanced_{ranker}"
            res["base_ranker"] = ranker
            res["window_mode"] = window_mode
            res["expression_source"] = expression_source
            res[condition_key] = condition
            result_frames.append(res)

    results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    diagnostics = pd.DataFrame(diagnostic_rows)
    return results, diagnostics


def _compare_sample_event_tables(
    events: pd.DataFrame,
    condition_key: str,
    sample_key: str,
    control,
    case,
    pathway_col: str = "Pathway",
    time_tolerance: float = 0.02,
    auc_tolerance: float = 0.25,
    fdr_threshold: float = 0.05,
) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame()
    for col in (condition_key, sample_key, pathway_col):
        if col not in events.columns:
            raise ValueError(f"Missing required event column '{col}'")

    rows = []
    for pathway, group in events.groupby(pathway_col, sort=False):
        control_group = group[group[condition_key].astype(str) == str(control)]
        case_group = group[group[condition_key].astype(str) == str(case)]
        if control_group.empty or case_group.empty:
            continue

        out = {
            pathway_col: pathway,
            "reference": control,
            "query": case,
            "control_n_samples": int(control_group[sample_key].nunique()),
            "case_n_samples": int(case_group[sample_key].nunique()),
            "control_peak_time": _median_or_nan(control_group["peak_time"]),
            "case_peak_time": _median_or_nan(case_group["peak_time"]),
            "control_AUC": _median_or_nan(control_group["AUC"]),
            "case_AUC": _median_or_nan(case_group["AUC"]),
            "control_peak_NES": _median_or_nan(control_group["peak_NES"]),
            "case_peak_NES": _median_or_nan(case_group["peak_NES"]),
            "control_window_fdr_min": _median_or_nan(control_group["window_fdr_min"]),
            "case_window_fdr_min": _median_or_nan(case_group["window_fdr_min"]),
            "control_event_label": _mode_string(control_group.get("event_label", [])),
            "case_event_label": _mode_string(case_group.get("event_label", [])),
        }
        out["delta_peak_time"] = out["case_peak_time"] - out["control_peak_time"]
        out["delta_AUC"] = out["case_AUC"] - out["control_AUC"]
        out["sample_consistency"] = _sample_consistency(
            control_group["AUC"].to_numpy(dtype=float),
            case_group["AUC"].to_numpy(dtype=float),
        )
        interpretation, program_type = _comparison_interpretation(
            pd.Series(out),
            "control",
            "case",
            time_tolerance,
            auc_tolerance,
            fdr_threshold,
        )
        out["interpretation"] = interpretation
        out["program_type"] = program_type
        out["event_type"] = program_type
        rows.append(out)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["program_type", "interpretation", pathway_col]
    ).reset_index(drop=True)


def _run_per_sample_trajectory_events(
    adata,
    gmt_path: str,
    condition_key: str,
    sample_key: str,
    sample_to_condition: dict[str, str],
    pseudotime_key: str,
    min_sample_cells: int,
    event_kwargs: Optional[dict],
    seed: int,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_kwargs = {} if event_kwargs is None else dict(event_kwargs)
    result_frames = []
    event_frames = []
    for sample_idx, sample in enumerate(sorted(sample_to_condition, key=str)):
        mask = adata.obs[sample_key].astype(str).to_numpy() == str(sample)
        if int(mask.sum()) < min_sample_cells:
            continue
        sample_adata = adata[mask].copy()
        res = run_trajectory_gsea(
            sample_adata,
            gmt_path=gmt_path,
            pseudotime_key=pseudotime_key,
            seed=seed + sample_idx,
            **kwargs,
        )
        if res is None or res.empty:
            continue
        condition = sample_to_condition[str(sample)]
        res = res.copy()
        res[condition_key] = condition
        res[sample_key] = sample
        result_frames.append(res)

        events = summarize_events(res, **event_kwargs)
        if not events.empty:
            events = events.copy()
            events[condition_key] = condition
            events[sample_key] = sample
            event_frames.append(events)

    results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    return results, events


def _bootstrap_replicate_event_ci(
    sample_events: pd.DataFrame,
    condition_key: str,
    sample_key: str,
    control,
    case,
    n_bootstrap: int,
    seed: int,
    pathway_col: str = "Pathway",
) -> pd.DataFrame:
    if sample_events is None or sample_events.empty or n_bootstrap <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    metrics = ["AUC", "peak_time", "duration"]
    for pathway, group in sample_events.groupby(pathway_col, sort=False):
        control_group = group[group[condition_key].astype(str) == str(control)]
        case_group = group[group[condition_key].astype(str) == str(case)]
        if control_group.empty or case_group.empty:
            continue
        row = {pathway_col: pathway}
        for metric in metrics:
            control_values = pd.to_numeric(control_group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            case_values = pd.to_numeric(case_group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if len(control_values) == 0 or len(case_values) == 0:
                row[f"delta_{metric}_ci_low"] = np.nan
                row[f"delta_{metric}_ci_high"] = np.nan
                continue
            deltas = []
            for _ in range(n_bootstrap):
                ctrl = rng.choice(control_values, size=len(control_values), replace=True)
                qry = rng.choice(case_values, size=len(case_values), replace=True)
                deltas.append(float(np.nanmedian(qry) - np.nanmedian(ctrl)))
            row[f"delta_{metric}_ci_low"] = float(np.quantile(deltas, 0.025))
            row[f"delta_{metric}_ci_high"] = float(np.quantile(deltas, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def _add_replicate_support_columns(
    comparison: pd.DataFrame,
    results: pd.DataFrame,
    sample_comparison: pd.DataFrame,
    diagnostics: pd.DataFrame,
    control,
    case,
    condition_key: str,
    pathway_col: str = "Pathway",
) -> pd.DataFrame:
    if comparison is None or comparison.empty:
        return pd.DataFrame()

    out = comparison.copy()
    if not sample_comparison.empty and pathway_col in sample_comparison.columns:
        keep_cols = [
            pathway_col,
            "sample_consistency",
            "control_n_samples",
            "case_n_samples",
        ]
        support = sample_comparison[[col for col in keep_cols if col in sample_comparison.columns]]
        out = out.drop(columns=["sample_consistency"], errors="ignore")
        out = pd.merge(out, support, on=pathway_col, how="left")

    n_control = 0
    n_case = 0
    if not results.empty and condition_key in results.columns:
        condition_counts = (
            results.groupby(condition_key)["n_replicates_total"].max().to_dict()
            if "n_replicates_total" in results.columns
            else {}
        )
        n_control = int(condition_counts.get(str(control), 0))
        n_case = int(condition_counts.get(str(case), 0))

    out["n_replicates_control"] = n_control
    out["n_replicates_case"] = n_case
    if "control_n_samples" in out:
        out["n_replicates_control"] = out["control_n_samples"].fillna(n_control).astype(int)
    if "case_n_samples" in out:
        out["n_replicates_case"] = out["case_n_samples"].fillna(n_case).astype(int)

    if not results.empty and "replicate_support" in results.columns:
        support_by_condition = results.groupby(condition_key)["replicate_support"].median()
        out["replicate_support"] = float(
            min(
                support_by_condition.get(str(control), np.nan),
                support_by_condition.get(str(case), np.nan),
            )
        )
    else:
        out["replicate_support"] = np.nan

    if diagnostics is not None and not diagnostics.empty and "min_cells_per_replicate" in diagnostics:
        positive = pd.to_numeric(diagnostics["min_cells_per_replicate"], errors="coerce")
        positive = positive[positive > 0]
        out["min_cells_per_replicate"] = int(positive.min()) if len(positive) else 0
    else:
        out["min_cells_per_replicate"] = 0

    out["sample_consistency"] = out.get("sample_consistency", np.nan)
    if "event_type" not in out.columns:
        out["event_type"] = out.get("program_type", "")
    return out


def _add_matched_condition_support(
    comparison: pd.DataFrame,
    results: pd.DataFrame,
    diagnostics: pd.DataFrame,
    control,
    case,
    condition_key: str,
    pathway_col: str = "Pathway",
    min_balance_pass_rate: float = 0.8,
    fdr_threshold: float = 0.05,
) -> pd.DataFrame:
    if comparison is None or comparison.empty:
        return pd.DataFrame()
    out = comparison.copy()
    if diagnostics is not None and not diagnostics.empty:
        balance = diagnostics.get("balance_pass", pd.Series(dtype=bool)).fillna(False).astype(bool)
        out["balance_pass_rate"] = float(balance.mean()) if len(balance) else np.nan
        if "balance_score" in diagnostics:
            score = pd.to_numeric(diagnostics["balance_score"], errors="coerce")
            out["median_balance_score"] = float(score.median()) if score.notna().any() else np.nan
        else:
            out["median_balance_score"] = np.nan
        if "core_balance_pass" in diagnostics:
            core_balance = diagnostics["core_balance_pass"].fillna(False).astype(bool)
            out["core_balance_pass_rate"] = (
                float(core_balance.mean()) if len(core_balance) else np.nan
            )
        else:
            core_balance = pd.Series(dtype=bool)
            out["core_balance_pass_rate"] = np.nan
        if "core_balance_score" in diagnostics:
            core_score = pd.to_numeric(diagnostics["core_balance_score"], errors="coerce")
            out["median_core_balance_score"] = (
                float(core_score.median()) if core_score.notna().any() else np.nan
            )
        else:
            out["median_core_balance_score"] = np.nan
        out["comparable_window_count"] = int(len(diagnostics))
        out["balanced_anchor_fraction"] = out["balance_pass_rate"]
        out["unbalanced_window_count"] = int((~balance).sum())
        out["comparable_pseudotime_span"] = float(
            pd.to_numeric(diagnostics["pt_end"], errors="coerce").max()
            - pd.to_numeric(diagnostics["pt_start"], errors="coerce").min()
        ) if {"pt_start", "pt_end"}.issubset(diagnostics.columns) else np.nan
        for col in (
            "pt_smd_after",
            "detection_rate_smd_after",
            "n_counts_smd_after",
            "n_genes_smd_after",
            "imbalance_score",
        ):
            if col in diagnostics:
                values = pd.to_numeric(diagnostics[col], errors="coerce").abs()
                out[f"max_abs_{col}"] = float(values.max()) if values.notna().any() else np.nan
        out["balance_pass"] = bool(
            np.isfinite(out["balance_pass_rate"].iloc[0])
            and out["balance_pass_rate"].iloc[0] >= min_balance_pass_rate
        )
        out["core_balance_pass"] = bool(
            np.isfinite(out["core_balance_pass_rate"].iloc[0])
            and out["core_balance_pass_rate"].iloc[0] >= min_balance_pass_rate
        )
    else:
        out["balance_pass_rate"] = np.nan
        out["median_balance_score"] = np.nan
        out["core_balance_pass_rate"] = np.nan
        out["median_core_balance_score"] = np.nan
        out["comparable_window_count"] = 0
        out["balanced_anchor_fraction"] = np.nan
        out["unbalanced_window_count"] = 0
        out["comparable_pseudotime_span"] = np.nan
        out["balance_pass"] = False
        out["core_balance_pass"] = False

    support_rows = []
    if results is not None and not results.empty and condition_key in results:
        diag = diagnostics.set_index("window_id") if diagnostics is not None and not diagnostics.empty and "window_id" in diagnostics else pd.DataFrame()
        for pathway, group in results.groupby(pathway_col, sort=False):
            pivot = group.pivot_table(
                index="window_id",
                columns=condition_key,
                values="NES",
                aggfunc="mean",
            )
            if str(control) not in pivot or str(case) not in pivot:
                continue
            delta = (
                pd.to_numeric(pivot[str(case)], errors="coerce")
                - pd.to_numeric(pivot[str(control)], errors="coerce")
            ).dropna()
            event_windows = delta.index.to_numpy()
            if "padj" in group.columns:
                padj_pivot = group.pivot_table(
                    index="window_id",
                    columns=condition_key,
                    values="padj",
                    aggfunc="min",
                )
                support_mask = pd.Series(False, index=delta.index)
                for condition in (str(control), str(case)):
                    if condition in padj_pivot:
                        condition_padj = pd.to_numeric(
                            padj_pivot.loc[delta.index, condition], errors="coerce"
                        )
                        support_mask = support_mask | (condition_padj <= fdr_threshold)
                if support_mask.any():
                    event_windows = support_mask[support_mask].index.to_numpy()
            if delta.empty:
                sign_consistency = np.nan
            else:
                direction = np.sign(float(delta.median()))
                if direction == 0:
                    sign_consistency = np.nan
                else:
                    sign_consistency = float((np.sign(delta) == direction).mean())
            if not diag.empty and len(event_windows):
                event_diag = diag.reindex(event_windows)
                event_balance = event_diag.get("balance_pass", pd.Series(dtype=bool)).fillna(False).astype(bool)
                event_core_balance = (
                    event_diag.get("core_balance_pass", pd.Series(dtype=bool))
                    .fillna(False)
                    .astype(bool)
                )
                event_scores = pd.to_numeric(
                    event_diag.get("balance_score", pd.Series(dtype=float)), errors="coerce"
                )
                event_core_scores = pd.to_numeric(
                    event_diag.get("core_balance_score", pd.Series(dtype=float)),
                    errors="coerce",
                )
                event_balance_coverage = float(event_balance.mean()) if len(event_balance) else np.nan
                event_core_balance_coverage = (
                    float(event_core_balance.mean()) if len(event_core_balance) else np.nan
                )
                event_median_balance_score = (
                    float(event_scores.median()) if event_scores.notna().any() else np.nan
                )
                event_median_core_balance_score = (
                    float(event_core_scores.median())
                    if event_core_scores.notna().any()
                    else np.nan
                )
                balanced_event_windows = int(event_balance.sum()) if len(event_balance) else 0
                core_balanced_event_windows = (
                    int(event_core_balance.sum()) if len(event_core_balance) else 0
                )
            else:
                event_balance_coverage = np.nan
                event_core_balance_coverage = np.nan
                event_median_balance_score = np.nan
                event_median_core_balance_score = np.nan
                balanced_event_windows = 0
                core_balanced_event_windows = 0
            support_rows.append(
                {
                    pathway_col: pathway,
                    "matched_event_windows": int(len(delta)),
                    "balanced_event_windows": balanced_event_windows,
                    "core_balanced_event_windows": core_balanced_event_windows,
                    "event_balance_coverage": event_balance_coverage,
                    "event_core_balance_coverage": event_core_balance_coverage,
                    "event_median_balance_score": event_median_balance_score,
                    "event_median_core_balance_score": event_median_core_balance_score,
                    "sign_consistency": sign_consistency,
                }
            )
    if support_rows:
        out = pd.merge(out, pd.DataFrame(support_rows), on=pathway_col, how="left")
    else:
        out["matched_event_windows"] = 0
        out["balanced_event_windows"] = 0
        out["core_balanced_event_windows"] = 0
        out["event_balance_coverage"] = np.nan
        out["event_core_balance_coverage"] = np.nan
        out["event_median_balance_score"] = np.nan
        out["event_median_core_balance_score"] = np.nan
        out["sign_consistency"] = np.nan
    event_balance_coverage = pd.to_numeric(
        out.get("event_balance_coverage"), errors="coerce"
    )
    event_core_balance_coverage = pd.to_numeric(
        out.get("event_core_balance_coverage"), errors="coerce"
    )
    event_balance_score = pd.to_numeric(
        out.get("event_median_balance_score"), errors="coerce"
    )
    event_core_balance_score = pd.to_numeric(
        out.get("event_median_core_balance_score"), errors="coerce"
    )
    strict_event_balanced = (
        (event_balance_coverage >= 0.8)
        | (event_balance_score >= 0.8)
        | out["balance_pass"].astype(bool)
    )
    core_event_balanced = (
        (event_core_balance_coverage >= 0.8)
        | (event_core_balance_score >= 0.8)
        | out["core_balance_pass"].astype(bool)
    )
    out["n_counts_sensitivity_flag"] = np.select(
        [strict_event_balanced, core_event_balanced],
        ["balanced", "core_balanced_ncounts_shift"],
        default="not_comparable",
    )
    out["eligible_condition_event"] = (
        (
            out["balance_pass"].astype(bool)
            | (pd.to_numeric(out.get("event_balance_coverage"), errors="coerce") >= 0.7)
            | (pd.to_numeric(out.get("event_median_balance_score"), errors="coerce") >= 0.7)
            | (pd.to_numeric(out.get("event_core_balance_coverage"), errors="coerce") >= 0.7)
            | (pd.to_numeric(out.get("event_median_core_balance_score"), errors="coerce") >= 0.7)
        )
        & (pd.to_numeric(out.get("matched_event_windows"), errors="coerce") >= 2)
        & (pd.to_numeric(out.get("sign_consistency"), errors="coerce") >= 0.7)
    )
    out["calibration_status"] = np.select(
        [out["balance_pass"].astype(bool), out["core_balance_pass"].astype(bool)],
        ["matched_window_balanced", "matched_window_core_balanced_counts_shift"],
        default="descriptive_only_imbalanced_windows",
    )
    return out


def summarize_matched_balance_diagnostics(
    diagnostics: pd.DataFrame,
    smd_threshold: float = 0.25,
    min_effective_cells: Optional[int] = None,
) -> pd.DataFrame:
    """Summarize why matched condition windows pass or fail local balance."""
    if diagnostics is None or diagnostics.empty:
        return pd.DataFrame(
            columns=["metric", "pass_rate", "n_windows", "threshold", "median_abs_value"]
        )
    rows = []

    def add_bool(metric: str, values: pd.Series, threshold) -> None:
        vals = values.dropna()
        rows.append(
            {
                "metric": metric,
                "pass_rate": float(vals.astype(bool).mean()) if len(vals) else np.nan,
                "n_windows": int(len(vals)),
                "threshold": threshold,
                "median_abs_value": np.nan,
            }
        )

    for prefix in ("pt", "detection_rate", "n_counts", "n_genes"):
        col = f"{prefix}_smd_after"
        pass_col = f"{prefix}_smd_after_pass"
        if pass_col in diagnostics:
            add_bool(pass_col, diagnostics[pass_col], smd_threshold)
        elif col in diagnostics:
            vals = pd.to_numeric(diagnostics[col], errors="coerce").abs()
            rows.append(
                {
                    "metric": pass_col,
                    "pass_rate": float((vals <= smd_threshold).mean())
                    if vals.notna().any()
                    else np.nan,
                    "n_windows": int(vals.notna().sum()),
                    "threshold": smd_threshold,
                    "median_abs_value": float(vals.median()) if vals.notna().any() else np.nan,
                }
            )

    for col in ("effective_n_control", "effective_n_case"):
        if col not in diagnostics:
            continue
        vals = pd.to_numeric(diagnostics[col], errors="coerce")
        threshold = (
            min_effective_cells
            if min_effective_cells is not None
            else float(vals[vals > 0].min()) if (vals > 0).any() else np.nan
        )
        rows.append(
            {
                "metric": f"{col}_pass",
                "pass_rate": float((vals >= threshold).mean())
                if np.isfinite(threshold) and vals.notna().any()
                else np.nan,
                "n_windows": int(vals.notna().sum()),
                "threshold": threshold,
                "median_abs_value": float(vals.median()) if vals.notna().any() else np.nan,
            }
        )

    if "balance_pass" in diagnostics:
        add_bool("overall_balance_pass", diagnostics["balance_pass"], "all_components")
    if "core_balance_pass" in diagnostics:
        add_bool(
            "core_balance_pass",
            diagnostics["core_balance_pass"],
            "pt+detection_rate+n_genes+effective_n",
        )
    if "balance_score" in diagnostics:
        vals = pd.to_numeric(diagnostics["balance_score"], errors="coerce")
        rows.append(
            {
                "metric": "median_balance_score",
                "pass_rate": np.nan,
                "n_windows": int(vals.notna().sum()),
                "threshold": np.nan,
                "median_abs_value": float(vals.median()) if vals.notna().any() else np.nan,
            }
        )
    if "core_balance_score" in diagnostics:
        vals = pd.to_numeric(diagnostics["core_balance_score"], errors="coerce")
        rows.append(
            {
                "metric": "median_core_balance_score",
                "pass_rate": np.nan,
                "n_windows": int(vals.notna().sum()),
                "threshold": np.nan,
                "median_abs_value": float(vals.median()) if vals.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_pseudobulk_condition_gsea(
    adata,
    gmt_path: str,
    condition_key: str,
    sample_key: str,
    control: Optional[str] = None,
    case: Optional[str] = None,
    pseudotime_key: str = "dpt_pseudotime",
    window_size: int = 500,
    step: int = 100,
    window_mode: str = "cell_count",
    min_cells: Optional[int] = None,
    max_cells: Optional[int] = None,
    target_span: Optional[float] = None,
    span_step: Optional[float] = None,
    min_cells_per_sample: int = 3,
    min_samples_per_condition: int = 2,
    pseudobulk_ranker: str = "t_stat",
    min_size: int = 15,
    max_size: int = 500,
    sample_size: int = 101,
    seed: int = 42,
    eps: float = 1e-50,
    nperm_nes: int = 100,
    bin_width: Optional[int] = 0,
    calculate_nes: bool = True,
    use_nes_cache: bool = False,
    event_kwargs: Optional[dict] = None,
    n_permutations: int = 100,
    calibration_stats: Sequence[str] = ("max_abs_NES", "AUC_abs", "duration"),
    primary_stat: str = "max_abs_NES",
    global_null: bool = True,
    layer: Optional[str] = None,
    use_raw: bool = False,
    gene_set_mode: str = "standard",
    min_abs_gene_weight: float = 0.0,
    gsea_param: float = 1.0,
    mode: str = "aligned",
    score_type: str = "std",
    tie_policy: str = "gene_id",
    nperm_simple: Optional[int] = 1000,
    max_levels: Optional[int] = None,
) -> pd.DataFrame:
    """
    Differential trajectory GSEA on sample-window pseudobulk profiles.

    Cells are aggregated within each pseudotime window for every biological
    sample. Genes are ranked by case-vs-control differences across those
    sample pseudobulks, then pathway events are calibrated by sample-label
    permutation.
    """
    if condition_key not in adata.obs:
        raise ValueError(f"condition_key '{condition_key}' not found in adata.obs")
    if sample_key not in adata.obs:
        raise ValueError(f"sample_key '{sample_key}' not found in adata.obs")
    if min_cells_per_sample <= 0:
        raise ValueError("min_cells_per_sample must be positive")
    if min_samples_per_condition <= 0:
        raise ValueError("min_samples_per_condition must be positive")
    if n_permutations < 0:
        raise ValueError("n_permutations must be non-negative")

    values = _unique_obs_values(adata, condition_key)
    if len(values) < 2:
        raise ValueError("At least two condition values are required")
    control = values[0] if control is None else control
    case = values[1] if case is None else case

    labels = adata.obs[condition_key].astype(str).to_numpy()
    keep = np.isin(labels, [str(control), str(case)])
    work = adata[keep].copy()
    sample_to_condition = _sample_condition_map(work, condition_key, sample_key)
    event_kwargs = {} if event_kwargs is None else dict(event_kwargs)

    base_kwargs = dict(
        gmt_path=gmt_path,
        condition_key=condition_key,
        sample_key=sample_key,
        control=control,
        case=case,
        pseudotime_key=pseudotime_key,
        window_size=window_size,
        step=step,
        window_mode=window_mode,
        min_cells=min_cells,
        max_cells=max_cells,
        target_span=target_span,
        span_step=span_step,
        min_cells_per_sample=min_cells_per_sample,
        min_samples_per_condition=min_samples_per_condition,
        pseudobulk_ranker=pseudobulk_ranker,
        min_size=min_size,
        max_size=max_size,
        sample_size=sample_size,
        seed=seed,
        eps=eps,
        nperm_nes=nperm_nes,
        bin_width=bin_width,
        calculate_nes=calculate_nes,
        use_nes_cache=use_nes_cache,
        layer=layer,
        use_raw=use_raw,
        gene_set_mode=gene_set_mode,
        min_abs_gene_weight=min_abs_gene_weight,
        gsea_param=gsea_param,
        gsea_mode=mode,
        score_type=score_type,
        tie_policy=tie_policy,
        nperm_simple=nperm_simple,
        max_levels=max_levels,
    )
    results, diagnostics = _pseudobulk_condition_results(
        work,
        sample_to_condition=sample_to_condition,
        **base_kwargs,
    )
    events = summarize_events(results, **event_kwargs)

    rng = np.random.default_rng(seed)
    null_frames = []
    samples = sorted(sample_to_condition, key=str)
    conditions = np.asarray([sample_to_condition[sample] for sample in samples])
    for perm_id in range(n_permutations):
        perm_map = dict(zip(samples, rng.permutation(conditions)))
        perm_results, _diagnostics = _pseudobulk_condition_results(
            work,
            sample_to_condition=perm_map,
            **{**base_kwargs, "seed": seed + perm_id + 1},
        )
        perm_events = summarize_events(perm_results, **event_kwargs)
        if not perm_events.empty:
            perm_events = perm_events.copy()
            perm_events.attrs.clear()
            perm_events["perm_id"] = perm_id
            null_frames.append(perm_events)

    null_events = pd.concat(null_frames, ignore_index=True) if null_frames else pd.DataFrame()
    from .calibration import calibrate_events

    calibrated_events = calibrate_events(
        events,
        null_events,
        stats=calibration_stats,
        primary_stat=primary_stat,
        global_null=global_null,
    )
    comparison = _events_to_pseudobulk_comparison(calibrated_events, control, case)
    if not comparison.empty:
        comparison["event_p"] = comparison.get("event_p", np.nan)
        comparison["event_fdr"] = comparison.get("event_fdr", np.nan)
    comparison.attrs["results"] = results
    comparison.attrs["events"] = events
    comparison.attrs["calibrated_events"] = calibrated_events
    comparison.attrs["null_events"] = null_events
    comparison.attrs["diagnostics"] = diagnostics
    comparison.attrs["pseudobulk"] = {
        "condition_key": condition_key,
        "sample_key": sample_key,
        "control": control,
        "case": case,
        "pseudobulk_ranker": pseudobulk_ranker,
        "n_permutations": int(n_permutations),
        "min_cells_per_sample": int(min_cells_per_sample),
        "min_samples_per_condition": int(min_samples_per_condition),
    }
    return comparison


def compare_trajectory_gsea_replicate_aware(
    adata,
    gmt_path: str,
    condition_key: str,
    sample_key: str,
    control: Optional[str] = None,
    case: Optional[str] = None,
    pseudotime_key: str = "dpt_pseudotime",
    event_kwargs: Optional[dict] = None,
    n_permutations: int = 100,
    n_bootstrap: int = 100,
    seed: int = 42,
    min_sample_cells: Optional[int] = None,
    min_cells_per_replicate: Optional[int] = None,
    min_replicates_per_condition: int = 3,
    time_tolerance: float = 0.02,
    auc_tolerance: float = 0.25,
    fdr_threshold: float = 0.05,
    **kwargs,
) -> pd.DataFrame:
    """
    Compare condition events with sample-balanced trajectory ranking.
    """
    if n_permutations < 0:
        raise ValueError("n_permutations must be non-negative")
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be non-negative")
    if condition_key not in adata.obs:
        raise ValueError(f"condition_key '{condition_key}' not found in adata.obs")
    if sample_key not in adata.obs:
        raise ValueError(f"sample_key '{sample_key}' not found in adata.obs")

    values = _unique_obs_values(adata, condition_key)
    if len(values) < 2:
        raise ValueError("At least two condition values are required")
    control = values[0] if control is None else control
    case = values[1] if case is None else case

    labels = adata.obs[condition_key].astype(str).to_numpy()
    keep = np.isin(labels, [str(control), str(case)])
    work = adata[keep].copy()
    sample_to_condition = _sample_condition_map(work, condition_key, sample_key)

    event_kwargs = {} if event_kwargs is None else dict(event_kwargs)
    run_kwargs = dict(kwargs)
    window_size = int(run_kwargs.pop("window_size", 500))
    step = int(run_kwargs.pop("step", 100))
    ranker = run_kwargs.pop("ranker", "mean_diff")
    window_mode = run_kwargs.pop("window_mode", "cell_count")
    min_cells = run_kwargs.pop("min_cells", None)
    max_cells = run_kwargs.pop("max_cells", None)
    target_span = run_kwargs.pop("target_span", None)
    span_step = run_kwargs.pop("span_step", None)
    min_size = int(run_kwargs.pop("min_size", 15))
    max_size = int(run_kwargs.pop("max_size", 500))
    sample_size = int(run_kwargs.pop("sample_size", 101))
    eps = float(run_kwargs.pop("eps", 1e-50))
    nperm_nes = int(run_kwargs.pop("nperm_nes", 100))
    bin_width = run_kwargs.pop("bin_width", 0)
    calculate_nes = bool(run_kwargs.pop("calculate_nes", True))
    use_nes_cache = bool(run_kwargs.pop("use_nes_cache", False))
    layer = run_kwargs.pop("layer", None)
    use_raw = bool(run_kwargs.pop("use_raw", False))
    gene_set_mode = run_kwargs.pop("gene_set_mode", "standard")
    min_abs_gene_weight = float(run_kwargs.pop("min_abs_gene_weight", 0.0))
    gsea_param = float(run_kwargs.pop("gsea_param", 1.0))
    gsea_mode = run_kwargs.pop("gsea_mode", "aligned")
    score_type = run_kwargs.pop("score_type", "std")
    tie_policy = run_kwargs.pop("tie_policy", "gene_id")
    nperm_simple = run_kwargs.pop("nperm_simple", 1000)
    max_levels = run_kwargs.pop("max_levels", None)
    smooth_slope_bandwidth = run_kwargs.pop("smooth_slope_bandwidth", None)
    min_sample_cells = (
        int(kwargs.get("window_size", 1)) if min_sample_cells is None else int(min_sample_cells)
    )
    min_cells_per_replicate = (
        3 if min_cells_per_replicate is None else int(min_cells_per_replicate)
    )
    if min_cells_per_replicate <= 0:
        raise ValueError("min_cells_per_replicate must be positive")
    if min_replicates_per_condition <= 0:
        raise ValueError("min_replicates_per_condition must be positive")
    n_control_reps = sum(str(value) == str(control) for value in sample_to_condition.values())
    n_case_reps = sum(str(value) == str(case) for value in sample_to_condition.values())
    low_replicate_count = min(n_control_reps, n_case_reps) < 3
    effective_min_replicates = (
        1 if low_replicate_count else min_replicates_per_condition
    )

    results, diagnostics = _sample_balanced_condition_results(
        work,
        gmt_path=gmt_path,
        condition_key=condition_key,
        sample_key=sample_key,
        control=control,
        case=case,
        pseudotime_key=pseudotime_key,
        sample_to_condition=sample_to_condition,
        window_size=window_size,
        step=step,
        ranker=ranker,
        window_mode=window_mode,
        min_cells=min_cells,
        max_cells=max_cells,
        target_span=target_span,
        span_step=span_step,
        min_cells_per_replicate=min_cells_per_replicate,
        min_replicates_per_condition=effective_min_replicates,
        min_size=min_size,
        max_size=max_size,
        sample_size=sample_size,
        seed=seed,
        eps=eps,
        nperm_nes=nperm_nes,
        bin_width=bin_width,
        calculate_nes=calculate_nes,
        use_nes_cache=use_nes_cache,
        layer=layer,
        use_raw=use_raw,
        gene_set_mode=gene_set_mode,
        min_abs_gene_weight=min_abs_gene_weight,
        gsea_param=gsea_param,
        gsea_mode=gsea_mode,
        score_type=score_type,
        tie_policy=tie_policy,
        nperm_simple=nperm_simple,
        max_levels=max_levels,
        smooth_slope_bandwidth=smooth_slope_bandwidth,
    )
    events = _summarize_events_by_group(results, condition_key, event_kwargs=event_kwargs)
    comparison = compare_event_tables(
        events,
        group_col=condition_key,
        reference=control,
        query=case,
        reference_label="control",
        query_label="case",
        time_tolerance=time_tolerance,
        auc_tolerance=auc_tolerance,
        fdr_threshold=fdr_threshold,
    )

    sample_results, sample_events = _run_per_sample_trajectory_events(
        work,
        gmt_path=gmt_path,
        condition_key=condition_key,
        sample_key=sample_key,
        sample_to_condition=sample_to_condition,
        pseudotime_key=pseudotime_key,
        min_sample_cells=min_sample_cells,
        event_kwargs=event_kwargs,
        seed=seed,
        window_size=window_size,
        step=step,
        ranker=ranker,
        window_mode=window_mode,
        min_cells=min_cells,
        max_cells=max_cells,
        target_span=target_span,
        span_step=span_step,
        min_size=min_size,
        max_size=max_size,
        sample_size=sample_size,
        eps=eps,
        nperm_nes=nperm_nes,
        bin_width=bin_width,
        calculate_nes=calculate_nes,
        use_nes_cache=use_nes_cache,
        layer=layer,
        use_raw=use_raw,
        gene_set_mode=gene_set_mode,
        min_abs_gene_weight=min_abs_gene_weight,
        gsea_param=gsea_param,
        mode=gsea_mode,
        score_type=score_type,
        tie_policy=tie_policy,
        nperm_simple=nperm_simple,
        max_levels=max_levels,
        smooth_slope_bandwidth=smooth_slope_bandwidth,
    )
    sample_comparison = _compare_sample_event_tables(
        sample_events,
        condition_key=condition_key,
        sample_key=sample_key,
        control=control,
        case=case,
        time_tolerance=time_tolerance,
        auc_tolerance=auc_tolerance,
        fdr_threshold=fdr_threshold,
    )
    comparison = _add_replicate_support_columns(
        comparison,
        results,
        sample_comparison,
        diagnostics,
        control=control,
        case=case,
        condition_key=condition_key,
    )
    ci = _bootstrap_replicate_event_ci(
        sample_events,
        condition_key=condition_key,
        sample_key=sample_key,
        control=control,
        case=case,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    if not comparison.empty and not ci.empty:
        comparison = pd.merge(comparison, ci, on="Pathway", how="left")

    rng = np.random.default_rng(seed)
    null_frames = []
    if not low_replicate_count and not events.empty and n_permutations > 0:
        samples = sorted(sample_to_condition, key=str)
        sample_conditions = np.asarray([sample_to_condition[sample] for sample in samples])
        for perm_id in range(n_permutations):
            perm_map = dict(zip(samples, rng.permutation(sample_conditions)))
            perm_results, _perm_diagnostics = _sample_balanced_condition_results(
                work,
                gmt_path=gmt_path,
                condition_key=condition_key,
                sample_key=sample_key,
                control=control,
                case=case,
                pseudotime_key=pseudotime_key,
                sample_to_condition=perm_map,
                window_size=window_size,
                step=step,
                ranker=ranker,
                window_mode=window_mode,
                min_cells=min_cells,
                max_cells=max_cells,
                target_span=target_span,
                span_step=span_step,
                min_cells_per_replicate=min_cells_per_replicate,
                min_replicates_per_condition=effective_min_replicates,
                min_size=min_size,
                max_size=max_size,
                sample_size=sample_size,
                seed=seed + perm_id + 1,
                eps=eps,
                nperm_nes=nperm_nes,
                bin_width=bin_width,
                calculate_nes=calculate_nes,
                use_nes_cache=use_nes_cache,
                layer=layer,
                use_raw=use_raw,
                gene_set_mode=gene_set_mode,
                min_abs_gene_weight=min_abs_gene_weight,
                gsea_param=gsea_param,
                gsea_mode=gsea_mode,
                score_type=score_type,
                tie_policy=tie_policy,
                nperm_simple=nperm_simple,
                max_levels=max_levels,
                smooth_slope_bandwidth=smooth_slope_bandwidth,
            )
            perm_events = _summarize_events_by_group(
                perm_results, condition_key, event_kwargs=event_kwargs
            )
            cmp_df = compare_event_tables(
                perm_events,
                group_col=condition_key,
                reference=control,
                query=case,
                reference_label="control",
                query_label="case",
                time_tolerance=time_tolerance,
                auc_tolerance=auc_tolerance,
                fdr_threshold=fdr_threshold,
            )
            if not cmp_df.empty:
                cmp_df["perm_id"] = perm_id
                null_frames.append(cmp_df)

    null_comparisons = (
        pd.concat(null_frames, ignore_index=True) if null_frames else pd.DataFrame()
    )
    from .calibration import calibrate_comparison

    if low_replicate_count:
        calibrated = comparison.copy()
        calibrated["comparison_p"] = np.nan
        calibrated["comparison_fdr"] = np.nan
        calibrated["event_p"] = np.nan
        calibrated["event_fdr"] = np.nan
        calibrated["replicate_aware_p"] = np.nan
        calibrated["replicate_aware_q"] = np.nan
        calibrated["calibration_method"] = "none_low_replicate_count"
        calibrated["calibration_status"] = "descriptive_only_low_replicate_count"
    elif n_permutations == 0:
        calibrated = comparison.copy()
        calibrated["comparison_p"] = np.nan
        calibrated["comparison_fdr"] = np.nan
        calibrated["event_p"] = np.nan
        calibrated["event_fdr"] = np.nan
        calibrated["replicate_aware_p"] = np.nan
        calibrated["replicate_aware_q"] = np.nan
        calibrated["calibration_method"] = "none_no_permutations"
        calibrated["calibration_status"] = "descriptive_only_no_permutations"
    elif comparison.empty:
        calibrated = comparison.copy()
        calibrated["comparison_p"] = np.nan
        calibrated["comparison_fdr"] = np.nan
        calibrated["event_p"] = np.nan
        calibrated["event_fdr"] = np.nan
        calibrated["replicate_aware_p"] = np.nan
        calibrated["replicate_aware_q"] = np.nan
        calibrated["calibration_method"] = "sample_label_permutation"
        calibrated["calibration_status"] = "null_calibration_failed"
    else:
        calibrated = calibrate_comparison(
            comparison,
            null_comparisons,
            stats=("delta_AUC_abs", "delta_peak_time_abs", "delta_duration"),
            primary_stat="delta_AUC_abs",
            global_null=True,
        )
        calibrated["event_p"] = calibrated["comparison_p"]
        calibrated["event_fdr"] = calibrated["comparison_fdr"]
        calibrated["replicate_aware_p"] = calibrated["comparison_p"]
        calibrated["replicate_aware_q"] = calibrated["comparison_fdr"]
        calibrated["calibration_method"] = "sample_label_permutation"
        calibrated["calibration_status"] = "calibrated"
    calibrated.attrs["results"] = results
    calibrated.attrs["events"] = events
    calibrated.attrs["sample_results"] = sample_results
    calibrated.attrs["sample_events"] = sample_events
    calibrated.attrs["sample_comparison"] = sample_comparison
    calibrated.attrs["null_comparisons"] = null_comparisons
    calibrated.attrs["diagnostics"] = diagnostics
    calibrated.attrs["replicate_aware"] = {
        "condition_key": condition_key,
        "sample_key": sample_key,
        "replicate_key": sample_key,
        "control": control,
        "case": case,
        "n_permutations": int(n_permutations),
        "n_bootstrap": int(n_bootstrap),
        "min_sample_cells": int(min_sample_cells),
        "min_cells_per_replicate": int(min_cells_per_replicate),
        "min_replicates_per_condition": int(min_replicates_per_condition),
        "ranker": ranker,
        "method": "sample_balanced_ranking",
    }
    return calibrated


def _fit_pathway_mixed_effect(
    group: pd.DataFrame,
    condition_key: str,
    sample_key: str,
    control,
    case,
) -> dict:
    data = group[[condition_key, sample_key, "pt_mid", "NES"]].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data[data[condition_key].astype(str).isin([str(control), str(case)])]
    if data.empty:
        return {
            "mixed_event_p": np.nan,
            "mixed_effect_method": "insufficient_data",
        }

    data["case_indicator"] = (data[condition_key].astype(str) == str(case)).astype(float)
    data["pt_mid_centered"] = pd.to_numeric(data["pt_mid"], errors="coerce") - float(
        pd.to_numeric(data["pt_mid"], errors="coerce").median()
    )
    data["condition_time"] = data["case_indicator"] * data["pt_mid_centered"]
    data["NES"] = pd.to_numeric(data["NES"], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["NES", "pt_mid_centered", "condition_time"]
    )

    n_samples = int(data[sample_key].nunique())
    n_control = int(data.loc[data[condition_key].astype(str) == str(control), sample_key].nunique())
    n_case = int(data.loc[data[condition_key].astype(str) == str(case), sample_key].nunique())
    if n_control < 3 or n_case < 3 or len(data) < 4:
        return {
            "mixed_event_p": np.nan,
            "mixed_effect_method": "descriptive_only_low_replicate_count",
            "mixed_n_samples": n_samples,
            "mixed_n_control_samples": n_control,
            "mixed_n_case_samples": n_case,
            "mixed_n_windows": int(len(data)),
        }

    formula = "NES ~ case_indicator + pt_mid_centered + condition_time"
    method = "mixedlm"
    fit = None
    try:
        import statsmodels.formula.api as smf

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.mixedlm(formula, data=data, groups=data[sample_key]).fit(
                reml=False,
                method="lbfgs",
                disp=False,
            )
    except Exception:
        fit = None

    if fit is None:
        try:
            import statsmodels.formula.api as smf

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if n_samples >= 3:
                    fit = smf.ols(formula, data=data).fit(
                        cov_type="cluster",
                        cov_kwds={"groups": data[sample_key]},
                    )
                    method = "ols_cluster"
                else:
                    fit = smf.ols(formula, data=data).fit()
                    method = "ols"
        except Exception:
            fit = None

    if fit is None:
        return {
            "mixed_event_p": np.nan,
            "mixed_effect_method": "model_failed",
            "mixed_n_samples": n_samples,
            "mixed_n_windows": int(len(data)),
        }

    params = getattr(fit, "params", pd.Series(dtype=float))
    pvalues = getattr(fit, "pvalues", pd.Series(dtype=float))
    p_condition = float(pvalues.get("case_indicator", np.nan))
    p_time = float(pvalues.get("condition_time", np.nan))
    p_candidates = np.asarray([p_condition, p_time], dtype=float)
    p_candidates = p_candidates[np.isfinite(p_candidates)]
    mixed_p = float(min(np.min(p_candidates) * len(p_candidates), 1.0)) if len(p_candidates) else np.nan

    return {
        "mixed_coef_condition": float(params.get("case_indicator", np.nan)),
        "mixed_p_condition": p_condition,
        "mixed_coef_condition_time": float(params.get("condition_time", np.nan)),
        "mixed_p_condition_time": p_time,
        "mixed_event_p": mixed_p,
        "mixed_effect_method": method,
        "mixed_n_samples": n_samples,
        "mixed_n_control_samples": n_control,
        "mixed_n_case_samples": n_case,
        "mixed_n_windows": int(len(data)),
    }


def calibrate_mixed_effect_events(
    comparison: pd.DataFrame,
    sample_results: pd.DataFrame,
    condition_key: str,
    sample_key: str,
    control,
    case,
    pathway_col: str = "Pathway",
) -> pd.DataFrame:
    """
    Add mixed-effect curve-level p-values/FDRs to a comparison table.

    The model is fitted per pathway on per-sample window NES values:
    ``NES ~ condition + pseudotime + condition:pseudotime + (1 | sample)``.
    If the mixed model cannot be fitted, the implementation falls back to OLS
    with cluster-robust sample standard errors when possible.
    """
    if comparison is None or comparison.empty:
        return pd.DataFrame()
    if sample_results is None or sample_results.empty:
        calibrated = comparison.copy()
        calibrated["mixed_event_p"] = np.nan
        calibrated["mixed_event_fdr"] = np.nan
        return calibrated
    for col in (pathway_col, condition_key, sample_key, "pt_mid", "NES"):
        if col not in sample_results.columns:
            raise ValueError(f"sample_results is missing required column '{col}'")

    rows = []
    for pathway, group in sample_results.groupby(pathway_col, sort=False):
        row = {pathway_col: pathway}
        row.update(
            _fit_pathway_mixed_effect(
                group,
                condition_key=condition_key,
                sample_key=sample_key,
                control=control,
                case=case,
            )
        )
        rows.append(row)
    mixed = pd.DataFrame(rows)
    from .calibration import _bh_adjust

    if not mixed.empty:
        mixed["mixed_event_fdr"] = _bh_adjust(
            pd.to_numeric(mixed["mixed_event_p"], errors="coerce").to_numpy(dtype=float)
        )

    calibrated = comparison.copy()
    if "event_p" in calibrated.columns:
        calibrated["permutation_event_p"] = calibrated["event_p"]
    if "event_fdr" in calibrated.columns:
        calibrated["permutation_event_fdr"] = calibrated["event_fdr"]
    calibrated = pd.merge(calibrated, mixed, on=pathway_col, how="left")
    calibrated["event_p"] = calibrated["mixed_event_p"]
    calibrated["event_fdr"] = calibrated["mixed_event_fdr"]
    calibrated.attrs.update(comparison.attrs)
    calibrated.attrs["mixed_effect"] = {
        "condition_key": condition_key,
        "sample_key": sample_key,
        "control": control,
        "case": case,
        "model": "NES ~ condition + pseudotime + condition:pseudotime + (1|sample)",
    }
    return calibrated


def compare_trajectory_gsea_mixed_effect(
    adata,
    gmt_path: str,
    condition_key: str,
    sample_key: str,
    control: Optional[str] = None,
    case: Optional[str] = None,
    pseudotime_key: str = "dpt_pseudotime",
    event_kwargs: Optional[dict] = None,
    n_permutations: int = 0,
    seed: int = 42,
    **kwargs,
) -> pd.DataFrame:
    """
    Replicate-aware condition comparison with mixed-effect event calibration.
    """
    comparison = compare_trajectory_gsea_replicate_aware(
        adata,
        gmt_path=gmt_path,
        condition_key=condition_key,
        sample_key=sample_key,
        control=control,
        case=case,
        pseudotime_key=pseudotime_key,
        event_kwargs=event_kwargs,
        n_permutations=n_permutations,
        seed=seed,
        **kwargs,
    )
    if comparison.empty:
        return comparison

    resolved_control = comparison["reference"].iloc[0]
    resolved_case = comparison["query"].iloc[0]
    calibrated = calibrate_mixed_effect_events(
        comparison,
        comparison.attrs.get("sample_results", pd.DataFrame()),
        condition_key=condition_key,
        sample_key=sample_key,
        control=resolved_control,
        case=resolved_case,
    )
    return calibrated


def compare_trajectory_gsea(
    adata,
    gmt_path: str,
    condition_key: str,
    control: Optional[str] = None,
    case: Optional[str] = None,
    compare: str = "case_vs_control",
    sample_key: Optional[str] = None,
    replicate_key: Optional[str] = None,
    pseudotime_key: str = "dpt_pseudotime",
    mode: Optional[str] = None,
    event_kwargs: Optional[dict] = None,
    n_permutations: int = 100,
    seed: int = 42,
    time_tolerance: float = 0.02,
    auc_tolerance: float = 0.25,
    fdr_threshold: float = 0.05,
    **kwargs,
) -> pd.DataFrame:
    """
    Run trajectory GSEA separately by condition and compare pathway events.

    The returned comparison table has the full per-condition results attached in
    ``comparison.attrs["results"]`` and event summaries in
    ``comparison.attrs["events"]``.
    """
    if condition_key not in adata.obs:
        raise ValueError(f"condition_key '{condition_key}' not found in adata.obs")
    if replicate_key is not None:
        if sample_key is not None and str(sample_key) != str(replicate_key):
            raise ValueError("Pass only one of sample_key or replicate_key, or use the same value")
        sample_key = replicate_key
    if mode == "pseudobulk" or compare == "pseudobulk":
        if sample_key is None:
            raise ValueError("sample_key is required for pseudobulk comparison")
        pseudobulk_kwargs = dict(kwargs)
        pseudobulk_ranker = pseudobulk_kwargs.pop("pseudobulk_ranker", None)
        fallback_ranker = pseudobulk_kwargs.pop("ranker", "t_stat")
        if pseudobulk_ranker is None:
            pseudobulk_ranker = fallback_ranker
        return run_pseudobulk_condition_gsea(
            adata,
            gmt_path=gmt_path,
            condition_key=condition_key,
            sample_key=sample_key,
            control=control,
            case=case,
            pseudotime_key=pseudotime_key,
            event_kwargs=event_kwargs,
            n_permutations=n_permutations,
            seed=seed,
            pseudobulk_ranker=pseudobulk_ranker,
            **pseudobulk_kwargs,
        )
    if mode == "mixed_effect" or compare == "mixed_effect":
        if sample_key is None:
            raise ValueError("sample_key is required for mixed-effect comparison")
        return compare_trajectory_gsea_mixed_effect(
            adata,
            gmt_path=gmt_path,
            condition_key=condition_key,
            sample_key=sample_key,
            control=control,
            case=case,
            pseudotime_key=pseudotime_key,
            event_kwargs=event_kwargs,
            n_permutations=n_permutations,
            seed=seed,
            time_tolerance=time_tolerance,
            auc_tolerance=auc_tolerance,
            fdr_threshold=fdr_threshold,
            **kwargs,
        )
    if mode == "replicate_aware" or compare == "replicate_aware":
        if sample_key is None:
            raise ValueError("sample_key is required for replicate-aware comparison")
        return compare_trajectory_gsea_replicate_aware(
            adata,
            gmt_path=gmt_path,
            condition_key=condition_key,
            sample_key=sample_key,
            control=control,
            case=case,
            pseudotime_key=pseudotime_key,
            event_kwargs=event_kwargs,
            n_permutations=n_permutations,
            seed=seed,
            time_tolerance=time_tolerance,
            auc_tolerance=auc_tolerance,
            fdr_threshold=fdr_threshold,
            **kwargs,
        )
    if mode in {"aligned_contrast", "alignment_contrast"} or compare in {
        "aligned_contrast",
        "alignment_contrast",
    }:
        values = _unique_obs_values(adata, condition_key)
        if len(values) < 2:
            raise ValueError("At least two condition values are required")
        control = values[0] if control is None else control
        case = values[1] if case is None else case
        event_kwargs = {} if event_kwargs is None else dict(event_kwargs)
        run_kwargs = dict(kwargs)
        anchor_pathways = run_kwargs.pop(
            "alignment_anchor_pathways",
            run_kwargs.pop("anchor_pathways", None),
        )
        anchor_set_name = run_kwargs.pop("alignment_anchor_set", "alignment_anchors")
        alignment_lambda = float(run_kwargs.pop("alignment_lambda", 1.0))
        alignment_permutation = run_kwargs.pop("alignment_permutation", "pathway_label")
        permutation_bins = int(run_kwargs.pop("permutation_bins", 5))
        contrast_threshold = float(
            run_kwargs.pop(
                "contrast_threshold",
                event_kwargs.get("nes_threshold", 0.5),
            )
        )
        alignment_min_consecutive = int(
            run_kwargs.pop(
                "alignment_min_consecutive",
                event_kwargs.get("min_consecutive", 2),
            )
        )
        alignment_sensitivity = bool(run_kwargs.pop("alignment_sensitivity", True))
        run_kwargs["mode"] = run_kwargs.pop("gsea_mode", "aligned")
        precomputed_pathway_null = str(alignment_permutation).lower() in {
            "pathway_label",
            "pathway_label_permutation",
        }
        primary_n_permutations = int(n_permutations) if precomputed_pathway_null else 0
        result_frames = []
        event_frames = []
        for condition in (control, case):
            res = run_trajectory_gsea(
                _subset_adata(adata, condition_key, condition),
                gmt_path=gmt_path,
                pseudotime_key=pseudotime_key,
                seed=seed,
                **run_kwargs,
            )
            if res is None or res.empty:
                continue
            res = res.copy()
            res[condition_key] = condition
            result_frames.append(res)

            events = summarize_events(res, **event_kwargs)
            if not events.empty:
                events[condition_key] = condition
                event_frames.append(events)

        results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
        events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
        tables = run_aligned_trajectory_contrast(
            results,
            condition_col=condition_key,
            condition_a=str(control),
            condition_b=str(case),
            anchor_pathways=anchor_pathways,
            contrast_threshold=contrast_threshold,
            min_consecutive=alignment_min_consecutive,
            alignment_lambda=alignment_lambda,
            anchor_set_name=anchor_set_name,
            time_tolerance=time_tolerance,
            auc_tolerance=auc_tolerance,
            n_permutations=primary_n_permutations,
            seed=seed,
            sensitivity=alignment_sensitivity,
        )
        comparison = tables["differential_event_table"]

        null_events = pd.DataFrame()
        if (
            str(alignment_permutation).lower()
            in {"condition_within_pseudotime_bins", "cell_label_within_pseudotime_bins"}
            and int(n_permutations) > 0
        ):
            labels = adata.obs[condition_key].astype(str).to_numpy()
            keep = np.isin(labels, [str(control), str(case)])
            work = adata[keep].copy()
            original_labels = work.obs[condition_key].astype(str).to_numpy()
            pt = pd.to_numeric(work.obs[pseudotime_key], errors="coerce").to_numpy(dtype=float)
            rng = np.random.default_rng(seed)
            null_frames = []
            for perm_id in range(int(n_permutations)):
                permuted = _permute_labels_within_time_bins(
                    original_labels,
                    pt,
                    n_bins=permutation_bins,
                    rng=rng,
                )
                perm_work = work.copy()
                perm_work.obs[condition_key] = permuted
                perm_results = []
                for condition in (control, case):
                    res = run_trajectory_gsea(
                        _subset_adata(perm_work, condition_key, condition),
                        gmt_path=gmt_path,
                        pseudotime_key=pseudotime_key,
                        seed=seed + perm_id + 1,
                        **run_kwargs,
                    )
                    if res is None or res.empty:
                        continue
                    res = res.copy()
                    res[condition_key] = condition
                    perm_results.append(res)
                if len(perm_results) < 2:
                    continue
                perm_results_df = pd.concat(perm_results, ignore_index=True)
                perm_tables = run_aligned_trajectory_contrast(
                    perm_results_df,
                    condition_col=condition_key,
                    condition_a=str(control),
                    condition_b=str(case),
                    anchor_pathways=anchor_pathways,
                    contrast_threshold=contrast_threshold,
                    min_consecutive=alignment_min_consecutive,
                    alignment_lambda=alignment_lambda,
                    anchor_set_name=anchor_set_name,
                    time_tolerance=time_tolerance,
                    auc_tolerance=auc_tolerance,
                    n_permutations=0,
                    seed=seed + perm_id + 1,
                    sensitivity=False,
                )
                perm_events = perm_tables["differential_event_table"]
                if not perm_events.empty:
                    perm_events = perm_events.copy()
                    perm_events["perm_id"] = int(perm_id)
                    null_frames.append(perm_events)
            null_events = (
                pd.concat(null_frames, ignore_index=True)
                if null_frames
                else pd.DataFrame()
            )
            comparison, event_fdr = _calibrate_aligned_contrast_with_null(
                comparison,
                null_events,
                n_permutations=int(n_permutations),
                null_model="condition_label_permutation_within_pseudotime_bins",
                calibration_status="screening_cell_label_permutation_no_replicates",
            )
            tables["differential_event_table"] = comparison
            tables["differential_event_fdr"] = event_fdr

        comparison.attrs["results"] = results
        comparison.attrs["events"] = events
        comparison.attrs["null_events"] = null_events
        for key, table in tables.items():
            if key != "differential_event_table":
                comparison.attrs[key] = table
        comparison.attrs["aligned_contrast"] = {
            "condition_key": condition_key,
            "control": control,
            "case": case,
            "anchor_set_name": anchor_set_name,
            "anchor_pathways": anchor_pathways,
            "contrast_threshold": contrast_threshold,
            "alignment_lambda": alignment_lambda,
            "n_permutations": int(n_permutations),
            "alignment_permutation": alignment_permutation,
            "permutation_bins": permutation_bins,
        }
        return comparison
    if mode in {"matched_window", "density_balanced"} or compare in {
        "matched_window",
        "density_balanced",
    }:
        values = _unique_obs_values(adata, condition_key)
        if len(values) < 2:
            raise ValueError("At least two condition values are required")
        control = values[0] if control is None else control
        case = values[1] if case is None else case
        event_kwargs = {} if event_kwargs is None else dict(event_kwargs)
        run_kwargs = dict(kwargs)
        window_size = int(run_kwargs.pop("window_size", 500))
        step = int(run_kwargs.pop("step", 100))
        ranker = run_kwargs.pop("ranker", "mean_diff")
        window_mode = run_kwargs.pop("window_mode", "cell_count")
        min_cells = run_kwargs.pop("min_cells", None)
        max_cells = run_kwargs.pop("max_cells", None)
        target_span = run_kwargs.pop("target_span", None)
        span_step = run_kwargs.pop("span_step", None)
        balance = run_kwargs.pop("balance", "weights")
        if mode == "density_balanced" or compare == "density_balanced":
            balance = "weights" if balance is None else balance
        n_balance_resamples = int(run_kwargs.pop("n_balance_resamples", 30))
        balance_smd_threshold = float(run_kwargs.pop("balance_smd_threshold", 0.25))
        n_counts_balance_weight = float(run_kwargs.pop("n_counts_balance_weight", 0.0))
        max_window_merge = int(run_kwargs.pop("max_window_merge", 0))
        min_size = int(run_kwargs.pop("min_size", 15))
        max_size = int(run_kwargs.pop("max_size", 500))
        sample_size = int(run_kwargs.pop("sample_size", 101))
        eps = float(run_kwargs.pop("eps", 1e-50))
        nperm_nes = int(run_kwargs.pop("nperm_nes", 100))
        bin_width = run_kwargs.pop("bin_width", 0)
        calculate_nes = bool(run_kwargs.pop("calculate_nes", True))
        use_nes_cache = bool(run_kwargs.pop("use_nes_cache", False))
        layer = run_kwargs.pop("layer", None)
        use_raw = bool(run_kwargs.pop("use_raw", False))
        gene_set_mode = run_kwargs.pop("gene_set_mode", "standard")
        min_abs_gene_weight = float(run_kwargs.pop("min_abs_gene_weight", 0.0))
        gsea_param = float(run_kwargs.pop("gsea_param", 1.0))
        gsea_mode = run_kwargs.pop("gsea_mode", "aligned")
        score_type = run_kwargs.pop("score_type", "std")
        tie_policy = run_kwargs.pop("tie_policy", "gene_id")
        nperm_simple = run_kwargs.pop("nperm_simple", 1000)
        max_levels = run_kwargs.pop("max_levels", None)
        smooth_slope_bandwidth = run_kwargs.pop("smooth_slope_bandwidth", None)
        results, diagnostics = _matched_condition_results(
            adata,
            gmt_path=gmt_path,
            condition_key=condition_key,
            control=control,
            case=case,
            pseudotime_key=pseudotime_key,
            window_size=window_size,
            step=step,
            ranker=ranker,
            window_mode=window_mode,
            min_cells=min_cells,
            max_cells=max_cells,
            target_span=target_span,
            span_step=span_step,
            balance=balance,
            n_balance_resamples=n_balance_resamples,
            balance_smd_threshold=balance_smd_threshold,
            n_counts_balance_weight=n_counts_balance_weight,
            max_window_merge=max_window_merge,
            min_size=min_size,
            max_size=max_size,
            sample_size=sample_size,
            seed=seed,
            eps=eps,
            nperm_nes=nperm_nes,
            bin_width=bin_width,
            calculate_nes=calculate_nes,
            use_nes_cache=use_nes_cache,
            layer=layer,
            use_raw=use_raw,
            gene_set_mode=gene_set_mode,
            min_abs_gene_weight=min_abs_gene_weight,
            gsea_param=gsea_param,
            gsea_mode=gsea_mode,
            score_type=score_type,
            tie_policy=tie_policy,
            nperm_simple=nperm_simple,
            max_levels=max_levels,
            smooth_slope_bandwidth=smooth_slope_bandwidth,
        )
        events = _summarize_events_by_group(results, condition_key, event_kwargs=event_kwargs)
        comparison = compare_event_tables(
            events,
            group_col=condition_key,
            reference=control,
            query=case,
            reference_label="control",
            query_label="case",
            time_tolerance=time_tolerance,
            auc_tolerance=auc_tolerance,
            fdr_threshold=fdr_threshold,
        )
        comparison = _add_matched_condition_support(
            comparison,
            results,
            diagnostics,
            control=control,
            case=case,
            condition_key=condition_key,
            fdr_threshold=fdr_threshold,
        )
        comparison.attrs["results"] = results
        comparison.attrs["events"] = events
        comparison.attrs["diagnostics"] = diagnostics
        comparison.attrs["balance_summary"] = summarize_matched_balance_diagnostics(
            diagnostics,
            smd_threshold=balance_smd_threshold,
            min_effective_cells=int(
                min_cells
                if min_cells is not None
                else max(3, min(max(window_size // 4, 1), 20))
            ),
        )
        comparison.attrs["matched_window"] = {
            "condition_key": condition_key,
            "control": control,
            "case": case,
            "balance": balance,
            "n_balance_resamples": int(n_balance_resamples),
            "balance_smd_threshold": float(balance_smd_threshold),
            "n_counts_balance_weight": float(n_counts_balance_weight),
            "max_window_merge": int(max_window_merge),
        }
        return comparison

    values = _unique_obs_values(adata, condition_key)
    if len(values) < 2:
        raise ValueError("At least two condition values are required")
    control = values[0] if control is None else control
    case = values[1] if case is None else case
    if compare != "case_vs_control":
        raise ValueError("Only compare='case_vs_control' is currently supported")

    event_kwargs = {} if event_kwargs is None else dict(event_kwargs)
    run_kwargs = dict(kwargs)
    run_kwargs["mode"] = run_kwargs.pop("gsea_mode", "aligned")
    result_frames = []
    event_frames = []
    for condition in (control, case):
        res = run_trajectory_gsea(
            _subset_adata(adata, condition_key, condition),
            gmt_path=gmt_path,
            pseudotime_key=pseudotime_key,
            seed=seed,
            **run_kwargs,
        )
        if res is None or res.empty:
            continue
        res = res.copy()
        res[condition_key] = condition
        result_frames.append(res)

        events = summarize_events(res, **event_kwargs)
        if not events.empty:
            events[condition_key] = condition
            event_frames.append(events)

    results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    comparison = compare_event_tables(
        events,
        group_col=condition_key,
        reference=control,
        query=case,
        reference_label="control",
        query_label="case",
        time_tolerance=time_tolerance,
        auc_tolerance=auc_tolerance,
        fdr_threshold=fdr_threshold,
    )
    comparison.attrs["results"] = results
    comparison.attrs["events"] = events
    return comparison


def run_branch_gsea(
    adata,
    gmt_path: str,
    branch_key: str,
    branches: Optional[Sequence[str]] = None,
    mode: str = "branch_vs_branch",
    event_kwargs: Optional[dict] = None,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """
    Run branch-aware trajectory GSEA and classify shared/divergent programs.
    """
    if branch_key not in adata.obs:
        raise ValueError(f"branch_key '{branch_key}' not found in adata.obs")
    requested_ranker = kwargs.get("ranker")
    if mode == "branch_contrast" or requested_ranker == "branch_contrast":
        kwargs = dict(kwargs)
        kwargs.pop("ranker", None)
        return run_branch_contrast_gsea(
            adata,
            gmt_path=gmt_path,
            branch_key=branch_key,
            branches=branches,
            event_kwargs=event_kwargs,
            **kwargs,
        )
    if mode != "branch_vs_branch":
        raise ValueError("Only mode='branch_vs_branch' is currently supported")

    selected = list(branches) if branches is not None else _unique_obs_values(adata, branch_key)
    if len(selected) < 2:
        raise ValueError("At least two branches are required")

    event_kwargs = {} if event_kwargs is None else dict(event_kwargs)
    run_kwargs = dict(kwargs)
    run_kwargs["mode"] = run_kwargs.pop("gsea_mode", "aligned")
    result_frames = []
    event_frames = []
    for branch in selected:
        res = run_trajectory_gsea(
            _subset_adata(adata, branch_key, branch),
            gmt_path=gmt_path,
            **run_kwargs,
        )
        if res is None or res.empty:
            continue
        res = res.copy()
        res[branch_key] = branch
        result_frames.append(res)

        events = summarize_events(res, **event_kwargs)
        if not events.empty:
            events[branch_key] = branch
            event_frames.append(events)

    results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()

    comparisons = []
    for left, right in combinations(selected, 2):
        cmp_df = compare_event_tables(
            events,
            group_col=branch_key,
            reference=left,
            query=right,
            reference_label="branch_a",
            query_label="branch_b",
        )
        if not cmp_df.empty:
            cmp_df["branch_a"] = left
            cmp_df["branch_b"] = right
            comparisons.append(cmp_df)

    comparison_df = (
        pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame()
    )
    return {"results": results, "events": events, "comparisons": comparison_df}


def _matched_reference_indices(
    pt: np.ndarray,
    branch_values: np.ndarray,
    reference,
    pt_start: float,
    pt_end: float,
    center: float,
    min_reference_cells: int,
    max_reference_cells: Optional[int],
) -> np.ndarray:
    ref_all = np.where(branch_values.astype(str) == str(reference))[0]
    in_span = ref_all[(pt[ref_all] >= pt_start) & (pt[ref_all] <= pt_end)]
    if len(in_span) >= min_reference_cells:
        if max_reference_cells is not None and len(in_span) > max_reference_cells:
            order = np.argsort(np.abs(pt[in_span] - center))
            return in_span[order[:max_reference_cells]]
        return in_span

    order = np.argsort(np.abs(pt[ref_all] - center))
    n_take = min(len(ref_all), max(min_reference_cells, len(in_span)))
    if max_reference_cells is not None:
        n_take = min(n_take, max_reference_cells)
    return ref_all[order[:n_take]]


def _contrast_mean_diff(X, target_indices, reference_indices, weights=None):
    if len(target_indices) == 0 or len(reference_indices) == 0:
        return np.zeros(X.shape[1], dtype=np.float64)

    if weights is not None:
        target_weight = float(weights[target_indices].sum())
        reference_weight = float(weights[reference_indices].sum())
        if target_weight <= 0 or reference_weight <= 0:
            return np.zeros(X.shape[1], dtype=np.float64)
        target_mean = _axis_weighted_sum(X, weights, target_indices) / target_weight
        reference_mean = _axis_weighted_sum(X, weights, reference_indices) / reference_weight
        return target_mean - reference_mean

    target_mean = _axis_sum(X, target_indices) / max(len(target_indices), 1)
    reference_mean = _axis_sum(X, reference_indices) / max(len(reference_indices), 1)
    return target_mean - reference_mean


def run_branch_contrast_gsea(
    adata,
    gmt_path: str,
    branch_key: str,
    branches: Optional[Sequence[str]] = None,
    pseudotime_key: str = "dpt_pseudotime",
    window_size: int = 500,
    step: int = 100,
    min_reference_cells: int = 20,
    max_reference_cells: Optional[int] = None,
    min_size: int = 15,
    max_size: int = 500,
    sample_size: int = 101,
    seed: int = 42,
    eps: float = 1e-50,
    nperm_nes: int = 100,
    bin_width: Optional[int] = 0,
    calculate_nes: bool = True,
    use_nes_cache: bool = False,
    event_kwargs: Optional[dict] = None,
    layer: Optional[str] = None,
    use_raw: bool = False,
    cell_weight_key: Optional[str] = None,
    gene_set_mode: str = "standard",
    min_abs_gene_weight: float = 0.0,
    gsea_param: float = 1.0,
    gsea_mode: str = "aligned",
    score_type: str = "std",
    tie_policy: str = "gene_id",
    nperm_simple: Optional[int] = 1000,
    max_levels: Optional[int] = None,
) -> dict[str, pd.DataFrame]:
    """
    Compare branches at matched pseudotime windows using branch contrast ranks.

    For each target branch window, genes are ranked by
    ``mean(target branch window) - mean(reference branch cells at matched
    pseudotime)``. This is useful for same-pseudotime lineage divergence.
    """
    if branch_key not in adata.obs:
        raise ValueError(f"branch_key '{branch_key}' not found in adata.obs")
    if pseudotime_key not in adata.obs:
        raise ValueError(f"pseudotime_key '{pseudotime_key}' not found in adata.obs")

    selected = list(branches) if branches is not None else _unique_obs_values(adata, branch_key)
    if len(selected) != 2:
        raise ValueError("run_branch_contrast_gsea currently requires exactly two branches")
    if min_reference_cells <= 0:
        raise ValueError("min_reference_cells must be positive")

    event_kwargs = {} if event_kwargs is None else dict(event_kwargs)
    pt = pd.to_numeric(adata.obs[pseudotime_key], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(pt).all():
        raise ValueError("pseudotime contains non-finite values; clean or drop cells first")
    branch_values = adata.obs[branch_key].astype(str).to_numpy()

    weights = None
    if cell_weight_key is not None:
        if cell_weight_key not in adata.obs:
            raise ValueError(f"cell_weight_key '{cell_weight_key}' not found in adata.obs")
        weights = pd.to_numeric(adata.obs[cell_weight_key], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("cell weights must be finite, non-negative, and sum to a positive value")

    X, genes, expression_source = _expression_matrix(adata, layer=layer, use_raw=use_raw)
    gene_sets = _prepare_gene_sets_for_mode(
        gmt_path,
        gene_set_mode=gene_set_mode,
        min_abs_gene_weight=min_abs_gene_weight,
    )
    pathway_names, pathway_indices = prepare_pathways(genes, gene_sets, min_size, max_size)
    if not pathway_indices:
        return {"results": pd.DataFrame(), "events": pd.DataFrame(), "comparisons": pd.DataFrame()}

    runner = GseaRunner(
        pathway_names,
        pathway_indices,
        min_size,
        max_size,
        gene_ids=genes,
        tie_policy=tie_policy,
    )
    result_frames = []
    event_frames = []

    for target, reference in ((selected[0], selected[1]), (selected[1], selected[0])):
        target_indices_all = np.where(branch_values == str(target))[0]
        ordered_target = target_indices_all[np.argsort(pt[target_indices_all])]
        windows = _make_windows(ordered_target, window_size=window_size, step=step, pt=pt)

        for wi, (_s, _e, window_indices) in enumerate(windows):
            pt_vals = pt[window_indices]
            pt_start = float(np.nanmin(pt_vals))
            pt_end = float(np.nanmax(pt_vals))
            center = float((pt_start + pt_end) / 2.0)
            reference_indices = _matched_reference_indices(
                pt,
                branch_values,
                reference,
                pt_start,
                pt_end,
                center,
                min_reference_cells=min_reference_cells,
                max_reference_cells=max_reference_cells,
            )
            if len(reference_indices) == 0:
                continue

            scores = _contrast_mean_diff(X, window_indices, reference_indices, weights=weights)
            scores = np.asarray(scores, dtype=np.float64)
            scores[~np.isfinite(scores)] = 0.0

            res = runner.run(
                scores,
                sample_size=sample_size,
                seed=seed,
                eps=eps,
                nperm_nes=nperm_nes,
                gsea_param=gsea_param,
                score_type=score_type,
                bin_width=bin_width,
                calculate_nes=calculate_nes,
                use_nes_cache=use_nes_cache,
                mode=gsea_mode,
                nperm_simple=nperm_simple,
                max_levels=max_levels,
            )
            if res.empty:
                continue

            res = res.copy()
            res["window_id"] = wi
            res["pt_start"] = pt_start
            res["pt_end"] = pt_end
            res["pt_mid"] = center
            res["n_cells"] = len(window_indices)
            res["n_reference_cells"] = len(reference_indices)
            res["ranker"] = "branch_contrast"
            res["window_mode"] = "cell_count"
            res[branch_key] = target
            res["contrast_reference"] = reference
            res["expression_source"] = expression_source
            if weights is not None:
                res["weight_sum"] = float(weights[window_indices].sum())
                res["reference_weight_sum"] = float(weights[reference_indices].sum())
            result_frames.append(res)

        target_results = [frame for frame in result_frames if not frame.empty and frame[branch_key].iloc[0] == target]
        if target_results:
            target_result = pd.concat(target_results, ignore_index=True)
            events = summarize_events(target_result, **event_kwargs)
            if not events.empty:
                events[branch_key] = target
                events["contrast_reference"] = reference
                event_frames.append(events)

    results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    comparison = compare_event_tables(
        events,
        group_col=branch_key,
        reference=selected[0],
        query=selected[1],
        reference_label="branch_a",
        query_label="branch_b",
    )
    if not comparison.empty:
        comparison["branch_a"] = selected[0]
        comparison["branch_b"] = selected[1]
        comparison["ranker"] = "branch_contrast"

    return {"results": results, "events": events, "comparisons": comparison}
