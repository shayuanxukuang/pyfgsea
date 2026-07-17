from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .trajectory import _load_gene_sets, run_trajectory_gsea
from .trajectory_compare import compare_trajectory_gsea
from .trajectory_events import summarize_events
from .validation import _expression_matrix


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return out

    p_finite = p[finite]
    order = np.argsort(p_finite)
    ranked = p_finite[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[finite] = restored
    return out


def calibrate_selected_window_statistics(
    observed_window_statistics: np.ndarray,
    permuted_window_statistics: np.ndarray,
    *,
    event_ids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Compare selected-window, event-FDR, and family-FWER calibrations.

    Parameters
    ----------
    observed_window_statistics
        Absolute or signed statistics with shape ``(events, windows)``.
    permuted_window_statistics
        Statistics recomputed after the complete selection pipeline in each
        permutation, with shape ``(permutations, events, windows)``.
    event_ids
        Optional stable event labels.

    Notes
    -----
    The primary TED event-FDR route maximizes within each event in every
    permutation, forms one selection-adjusted p value per event, and applies
    BH across the declared event family.  The family-wide maxT route is a
    separate FWER sensitivity analysis and is not BH-adjusted.  The naive
    p value holds the observed selected window fixed in the null draws and is
    included only as an anti-selection comparator.
    """
    observed = np.asarray(observed_window_statistics, dtype=float)
    permuted = np.asarray(permuted_window_statistics, dtype=float)
    if observed.ndim != 2:
        raise ValueError("observed_window_statistics must have shape (events, windows)")
    if permuted.ndim != 3 or permuted.shape[1:] != observed.shape:
        raise ValueError(
            "permuted_window_statistics must have shape "
            "(permutations, events, windows) matching the observed array"
        )
    if permuted.shape[0] < 1:
        raise ValueError("At least one permutation is required")

    n_events, _n_windows = observed.shape
    ids = list(event_ids) if event_ids is not None else [f"event_{i}" for i in range(n_events)]
    if len(ids) != n_events or len(set(ids)) != n_events:
        raise ValueError("event_ids must be unique and match the number of events")

    observed_abs = np.abs(observed)
    permuted_abs = np.abs(permuted)
    selected_window_index = np.nanargmax(observed_abs, axis=1)
    observed_selected = observed_abs[np.arange(n_events), selected_window_index]
    perm_index = np.arange(permuted.shape[0])[:, None]
    event_index = np.arange(n_events)[None, :]
    fixed_window_null = permuted_abs[
        perm_index,
        event_index,
        selected_window_index[None, :],
    ]
    per_event_max_null = np.nanmax(permuted_abs, axis=2)
    family_max_null = np.nanmax(per_event_max_null, axis=1)
    denominator = float(permuted.shape[0] + 1)

    tie_tolerance = np.maximum(1.0, observed_selected) * 1e-10
    naive_p = (
        1.0
        + np.sum(
            fixed_window_null >= observed_selected[None, :] - tie_tolerance[None, :],
            axis=0,
        )
    ) / denominator
    event_p = (
        1.0
        + np.sum(
            per_event_max_null >= observed_selected[None, :] - tie_tolerance[None, :],
            axis=0,
        )
    ) / denominator
    event_q = _bh_adjust(event_p)
    family_fwer_p = (
        1.0
        + np.sum(
            family_max_null[:, None]
            >= observed_selected[None, :] - tie_tolerance[None, :],
            axis=0,
        )
    ) / denominator

    return pd.DataFrame(
        {
            "event_id": ids,
            "selected_window_index": selected_window_index.astype(int),
            "observed_selected_abs_stat": observed_selected,
            "naive_selected_window_p": naive_p,
            "event_p": event_p,
            "event_q": event_q,
            "family_fwer_p": family_fwer_p,
            "window_selection_correction": "per_event_permutation_max",
            "across_event_adjustment": "BH",
            "primary_error_rate": "event-level FDR",
            "family_wide_sensitivity": "maxT FWER; no BH",
            "n_perm": int(permuted.shape[0]),
        }
    )


def _normalize_event_stat(stat: str) -> str:
    normalized = str(stat).replace("-", "_")
    aliases = {
        "max_abs_nes": "max_abs_NES",
        "max_abs_NES": "max_abs_NES",
        "auc_abs": "AUC_abs",
        "auc_abs_nes": "AUC_abs",
        "AUC_abs_NES": "AUC_abs",
        "longest_run": "duration",
        "longest_significant_run": "duration",
        "peak_sharpness": "sharpness",
        "delta_auc": "delta_AUC_abs",
        "delta_AUC": "delta_AUC_abs",
        "delta_auc_abs": "delta_AUC_abs",
        "delta_peak_time": "delta_peak_time_abs",
        "delta_peak_time_abs": "delta_peak_time_abs",
        "direction_switches": "direction_switch_count",
        "recurrence_score": "recurrence",
    }
    return aliases.get(normalized, normalized)


def _normalize_comparison_stat(stat: str) -> str:
    normalized = str(stat).replace("-", "_")
    aliases = {
        "delta_auc": "delta_AUC_abs",
        "delta_AUC": "delta_AUC_abs",
        "delta_auc_abs": "delta_AUC_abs",
        "delta_peak_time": "delta_peak_time_abs",
        "delta_duration_abs": "delta_duration",
        "branch_divergence_score": "delta_AUC_abs",
    }
    return aliases.get(normalized, normalized)


def _event_stat_values(events: pd.DataFrame, stat: str) -> pd.Series:
    stat = _normalize_event_stat(stat)
    if stat == "peak_NES_abs":
        return events["peak_NES"].abs()
    if stat == "trough_NES_abs":
        return events["trough_NES"].abs()
    if stat == "max_abs_NES":
        return pd.concat([events["peak_NES"].abs(), events["trough_NES"].abs()], axis=1).max(axis=1)
    if stat == "AUC_abs":
        return events["AUC_abs"].abs() if "AUC_abs" in events else events["AUC"].abs()
    if stat == "duration":
        return events["duration"].abs()
    if stat == "sharpness":
        return events["sharpness"].abs()
    if stat == "recurrence":
        return events["recurrence"].abs()
    if stat in events.columns:
        return events[stat].abs()
    raise ValueError(f"Unsupported event calibration stat '{stat}'")


def _comparison_stat_values(comparison: pd.DataFrame, stat: str) -> pd.Series:
    stat = _normalize_comparison_stat(stat)
    if stat == "delta_AUC_abs":
        return comparison["delta_AUC"].abs()
    if stat == "delta_peak_time_abs":
        return comparison["delta_peak_time"].abs()
    if stat in comparison.columns:
        return comparison[stat].abs()
    raise ValueError(f"Unsupported comparison calibration stat '{stat}'")


def _display_event_stat(stat: str) -> str:
    stat = _normalize_event_stat(stat)
    names = {
        "max_abs_NES": "max_abs_NES",
        "AUC_abs": "AUC_abs",
        "duration": "longest_significant_run",
        "sharpness": "peak_sharpness",
        "delta_AUC_abs": "delta_AUC",
        "delta_peak_time_abs": "delta_peak_time",
    }
    return names.get(stat, stat)


def _pathway_column(df: pd.DataFrame) -> str:
    if "Pathway" in df.columns:
        return "Pathway"
    if "pathway" in df.columns:
        return "pathway"
    raise ValueError("Could not find a pathway column")


def _normalize_pathway_name(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("hallmark_", "")
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def _filter_pathway_family(
    df: pd.DataFrame,
    pathways: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if df is None or df.empty or pathways is None:
        return df
    path_col = _pathway_column(df)
    wanted = {_normalize_pathway_name(pathway) for pathway in pathways}
    out = df[df[path_col].map(_normalize_pathway_name).isin(wanted)].copy()
    out.attrs.clear()
    return out


def _filter_gene_sets_for_family(
    gene_sets,
    pathways: Optional[Sequence[str]] = None,
):
    if pathways is None:
        return gene_sets
    raw = _load_gene_sets(gene_sets)
    wanted = {_normalize_pathway_name(pathway) for pathway in pathways}
    return {
        name: genes
        for name, genes in raw.items()
        if _normalize_pathway_name(name) in wanted
    }


def event_fdr_power_report(
    calibration_table: pd.DataFrame,
    q_threshold: float = 0.05,
    hypothesis_family: str = "hallmark_all",
    n_perm: Optional[int] = None,
) -> pd.DataFrame:
    """
    Report whether a permutation-calibrated event FDR table has enough power.

    With a small number of permutations and many tests, the smallest attainable
    empirical p-value can still be too large to produce a BH q-value below the
    requested threshold. This report makes that limitation explicit.
    """
    columns = [
        "hypothesis_family",
        "event_stat",
        "n_perm",
        "n_perm_effective",
        "n_tests",
        "minimum_attainable_p",
        "minimum_attainable_q",
        "q_threshold",
        "q_threshold_reachable",
        "discovery_possible_at_q_threshold",
        "recommended_min_n_perm",
    ]
    if calibration_table is None or calibration_table.empty:
        return pd.DataFrame(columns=columns)

    df = calibration_table.copy()
    n_perm_override = n_perm
    calibration_meta = df.attrs.get("calibration", {}) if hasattr(df, "attrs") else {}

    def numeric_column(group: pd.DataFrame, column: str) -> pd.Series:
        if column not in group.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(group[column], errors="coerce")

    stat_col = "event_stat" if "event_stat" in df.columns else None
    groups = df.groupby(stat_col, sort=False) if stat_col else [(None, df)]
    rows = []
    for stat, group in groups:
        p_floor = numeric_column(group, "minimum_attainable_p").dropna().min()
        if not np.isfinite(p_floor):
            if n_perm_override is not None:
                n_eff_values = pd.Series([n_perm_override])
            else:
                n_eff_values = numeric_column(group, "n_perm_effective").dropna()
            if n_eff_values.empty:
                n_eff_values = numeric_column(group, "n_perm").dropna()
            if n_eff_values.empty and calibration_meta.get("n_permutations") is not None:
                n_eff_values = pd.Series([calibration_meta.get("n_permutations")])
            n_eff = n_eff_values.min() if not n_eff_values.empty else np.nan
            p_floor = 1.0 / (1.0 + float(n_eff)) if np.isfinite(n_eff) else np.nan
        n_tests = int(len(group))
        min_q = float(min(1.0, p_floor * n_tests)) if np.isfinite(p_floor) else np.nan
        recommended_min_n_perm = (
            int(np.ceil(n_tests / float(q_threshold) - 1.0))
            if q_threshold > 0 and n_tests > 0
            else np.nan
        )
        n_perm_values = (
            pd.Series([n_perm_override])
            if n_perm_override is not None
            else numeric_column(group, "n_perm").dropna()
        )
        if n_perm_values.empty and calibration_meta.get("n_permutations") is not None:
            n_perm_values = pd.Series([calibration_meta.get("n_permutations")])
        n_perm_display = n_perm_values.max() if not n_perm_values.empty else np.nan
        n_eff_values = numeric_column(group, "n_perm_effective").dropna()
        if n_eff_values.empty:
            n_eff_values = n_perm_values
        n_eff = n_eff_values.max() if not n_eff_values.empty else np.nan
        rows.append(
            {
                "hypothesis_family": hypothesis_family,
                "event_stat": stat if stat is not None else "primary",
                "n_perm": int(n_perm_display) if np.isfinite(n_perm_display) else np.nan,
                "n_perm_effective": int(n_eff) if np.isfinite(n_eff) else np.nan,
                "n_tests": n_tests,
                "minimum_attainable_p": float(p_floor) if np.isfinite(p_floor) else np.nan,
                "minimum_attainable_q": min_q,
                "q_threshold": float(q_threshold),
                "q_threshold_reachable": bool(
                    np.isfinite(min_q) and min_q <= q_threshold
                ),
                "discovery_possible_at_q_threshold": bool(
                    np.isfinite(min_q) and min_q <= q_threshold
                ),
                "recommended_min_n_perm": recommended_min_n_perm,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _attach_power_report(
    table: pd.DataFrame,
    q_threshold: float,
    hypothesis_family: str,
    n_perm: Optional[int] = None,
) -> pd.DataFrame:
    if table is None:
        return table
    report = event_fdr_power_report(
        table,
        q_threshold=q_threshold,
        hypothesis_family=hypothesis_family,
        n_perm=n_perm,
    )
    table.attrs["power_report"] = report
    table.attrs["hypothesis_family"] = hypothesis_family
    if table.empty:
        return table
    if "event_stat" in table.columns and not report.empty:
        lookup = report.set_index("event_stat")["minimum_attainable_q"].to_dict()
        possible = report.set_index("event_stat")[
            "discovery_possible_at_q_threshold"
        ].to_dict()
        table["minimum_attainable_q"] = table["event_stat"].map(lookup)
        table[f"discovery_possible_at_q_{str(q_threshold).replace('.', '_')}"] = (
            table["event_stat"].map(possible).fillna(False).astype(bool)
        )
        table["q_threshold_reachable"] = table[
            f"discovery_possible_at_q_{str(q_threshold).replace('.', '_')}"
        ]
        rec = report.set_index("event_stat")["recommended_min_n_perm"].to_dict()
        table["recommended_min_n_perm"] = table["event_stat"].map(rec)
        limited = ~table["q_threshold_reachable"].astype(bool)
        if limited.any():
            warning = "empirical_resolution_limited"
            current = table.get("calibration_warning", pd.Series([""] * len(table), index=table.index))
            current = current.fillna("").astype(str)
            table["calibration_warning"] = np.where(
                limited & current.ne(""),
                current + ";" + warning,
                np.where(limited, warning, current),
            )
    return table


def _stat_values_for_kind(df: pd.DataFrame, stat: str, kind: str) -> pd.Series:
    if kind == "comparison":
        return _comparison_stat_values(df, stat)
    return _event_stat_values(df, stat)


def _event_calibration_long_table(
    observed_table: pd.DataFrame,
    null_table: pd.DataFrame,
    stats: Iterable[str],
    kind: str,
    n_perm: int,
    null_model: str,
    global_null: bool = False,
    base_warning: str = "",
    n_perm_effective: Optional[int] = None,
    early_stopped: bool = False,
) -> pd.DataFrame:
    n_perm_effective = int(n_perm if n_perm_effective is None else n_perm_effective)
    if observed_table is None or observed_table.empty:
        return pd.DataFrame(
            columns=[
                "pathway",
                "event_stat",
                "observed",
                "null_mean",
                "null_sd",
                "event_p",
                "event_q",
                "event_fdr",
                "family_fwer_p",
                "multiplicity_method",
                "window_selection_correction",
                "across_event_adjustment",
                "primary_error_rate",
                "minimum_attainable_p",
                "minimum_attainable_q",
                "n_perm",
                "n_perm_effective",
                "early_stopped",
                "null_model",
                "calibration_warning",
                "calibration_status",
            ]
        )

    stats = [_normalize_event_stat(stat) for stat in stats]
    observed_path_col = _pathway_column(observed_table)
    null_path_col = _pathway_column(null_table) if null_table is not None and not null_table.empty else None
    rows = []
    null_empty = null_table is None or null_table.empty or "perm_id" not in null_table.columns

    for stat in stats:
        obs_values = _stat_values_for_kind(observed_table, stat, kind)
        obs_tmp = pd.DataFrame(
            {
                "pathway": observed_table[observed_path_col].astype(str).to_numpy(),
                "observed": obs_values.to_numpy(dtype=float),
            }
        )

        null_tmp = pd.DataFrame()
        global_values = None
        if not null_empty:
            null_values = _stat_values_for_kind(null_table, stat, kind)
            null_tmp = pd.DataFrame(
                {
                    "pathway": null_table[null_path_col].astype(str).to_numpy(),
                    "perm_id": null_table["perm_id"].to_numpy(),
                    "stat": null_values.to_numpy(dtype=float),
                }
            )
            null_tmp = null_tmp[np.isfinite(null_tmp["stat"])]
            if global_null and not null_tmp.empty:
                global_values = np.zeros(n_perm_effective, dtype=float)
                grouped = null_tmp.groupby("perm_id")["stat"].max()
                for perm_id, value in grouped.items():
                    if 0 <= int(perm_id) < n_perm_effective:
                        global_values[int(perm_id)] = float(value)

        stat_rows = []
        for _, obs_row in obs_tmp.iterrows():
            pathway = obs_row["pathway"]
            observed = float(obs_row["observed"]) if np.isfinite(obs_row["observed"]) else np.nan
            warnings = [base_warning] if base_warning else []
            if null_empty:
                null_values = np.array([], dtype=float)
                warnings.append("no_null_events")
            elif global_values is not None:
                null_values = global_values
                warnings.append("global_max_stat_null")
            else:
                pathway_null = null_tmp[null_tmp["pathway"] == pathway]
                null_values = np.zeros(n_perm_effective, dtype=float)
                if pathway_null.empty:
                    warnings.append("pathway_missing_in_null")
                else:
                    grouped = pathway_null.groupby("perm_id")["stat"].max()
                    for perm_id, value in grouped.items():
                        if 0 <= int(perm_id) < n_perm_effective:
                            null_values[int(perm_id)] = float(value)

            if len(null_values) == 0 or not np.isfinite(observed):
                event_p = np.nan
                null_mean = np.nan
                null_sd = np.nan
            else:
                null_values = null_values[np.isfinite(null_values)]
                null_mean = float(np.mean(null_values)) if len(null_values) else np.nan
                null_sd = float(np.std(null_values, ddof=1)) if len(null_values) > 1 else 0.0
                event_p = (1.0 + float(np.sum(null_values >= observed))) / (
                    1.0 + float(n_perm_effective)
                )

            stat_rows.append(
                {
                    "pathway": pathway,
                    "event_stat": _display_event_stat(stat),
                    "observed": observed,
                    "null_mean": null_mean,
                    "null_sd": null_sd,
                    "event_p": event_p,
                    "minimum_attainable_p": 1.0 / (1.0 + float(n_perm_effective)),
                    "minimum_attainable_q": np.nan,
                    "n_perm": int(n_perm),
                    "n_perm_effective": int(n_perm_effective),
                    "early_stopped": bool(early_stopped),
                    "null_model": null_model,
                    "calibration_warning": ";".join(w for w in warnings if w),
                }
            )

        if stat_rows:
            p_values = np.asarray([row["event_p"] for row in stat_rows], dtype=float)
            q_values = np.full_like(p_values, np.nan) if global_null else _bh_adjust(p_values)
            for row, q_value in zip(stat_rows, q_values, strict=True):
                row["event_q"] = q_value
                row["event_fdr"] = q_value
                row["family_fwer_p"] = row["event_p"] if global_null else np.nan
                row["multiplicity_method"] = (
                    "family_wide_maxT_fwer" if global_null else "per_event_max_window_bh"
                )
                row["window_selection_correction"] = "per_event_permutation_max"
                row["across_event_adjustment"] = "none" if global_null else "BH"
                row["primary_error_rate"] = (
                    "family-wise error rate" if global_null else "event-level FDR"
                )
                warning = row.get("calibration_warning", "")
                if "descriptive_only_low_replicate_count" in warning:
                    row["calibration_status"] = "descriptive_only_low_replicates"
                elif "no_null_events" in warning or "pathway_missing_in_null" in warning:
                    row["calibration_status"] = "null_calibration_failed"
                else:
                    row["calibration_status"] = "discovery_ready"
            rows.extend(stat_rows)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out[
        [
            "pathway",
            "event_stat",
            "observed",
            "null_mean",
            "null_sd",
            "event_p",
            "event_q",
            "event_fdr",
            "family_fwer_p",
            "multiplicity_method",
            "window_selection_correction",
            "across_event_adjustment",
            "primary_error_rate",
            "minimum_attainable_p",
            "minimum_attainable_q",
            "n_perm",
            "n_perm_effective",
            "early_stopped",
            "null_model",
            "calibration_warning",
            "calibration_status",
        ]
    ]


def _event_early_stop_ready(
    observed_table: pd.DataFrame,
    null_table: pd.DataFrame,
    stats: Iterable[str],
    n_perm_planned: int,
    threshold: float,
) -> bool:
    if observed_table is None or observed_table.empty:
        return True
    if null_table is None or null_table.empty or "perm_id" not in null_table.columns:
        return False
    observed_path_col = _pathway_column(observed_table)
    null_path_col = _pathway_column(null_table)
    for stat in [_normalize_event_stat(stat) for stat in stats]:
        observed = pd.DataFrame(
            {
                "pathway": observed_table[observed_path_col].astype(str).to_numpy(),
                "observed": _event_stat_values(observed_table, stat).to_numpy(dtype=float),
            }
        )
        null_values = _event_stat_values(null_table, stat)
        null_tmp = pd.DataFrame(
            {
                "pathway": null_table[null_path_col].astype(str).to_numpy(),
                "stat": null_values.to_numpy(dtype=float),
            }
        )
        null_tmp = null_tmp[np.isfinite(null_tmp["stat"])]
        for _, row in observed.iterrows():
            obs = float(row["observed"]) if np.isfinite(row["observed"]) else np.nan
            if not np.isfinite(obs):
                continue
            pathway_null = null_tmp.loc[null_tmp["pathway"] == row["pathway"], "stat"]
            exceed = int((pathway_null >= obs).sum()) if len(pathway_null) else 0
            best_possible_p = (1.0 + exceed) / (1.0 + float(n_perm_planned))
            if best_possible_p <= threshold:
                return False
    return True


def _null_distribution(null_table: pd.DataFrame, values: pd.Series, global_null: bool) -> np.ndarray:
    finite_values = values.to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if len(finite_values) == 0:
        return np.array([], dtype=float)

    if global_null and "perm_id" in null_table.columns:
        tmp = null_table[["perm_id"]].copy()
        tmp["__stat"] = values.to_numpy(dtype=float)
        return (
            tmp.dropna(subset=["__stat"])
            .groupby("perm_id")["__stat"]
            .max()
            .to_numpy(dtype=float)
        )
    return finite_values


def _empirical_p_values(observed: np.ndarray, null_values: np.ndarray) -> np.ndarray:
    p_values = np.full(len(observed), np.nan, dtype=float)
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if len(null_values) == 0:
        return p_values

    for idx, value in enumerate(observed):
        if np.isfinite(value):
            p_values[idx] = (1.0 + np.sum(null_values >= value)) / (len(null_values) + 1.0)
    return p_values


def calibrate_events(
    observed_events: pd.DataFrame,
    null_events: pd.DataFrame,
    stats: Iterable[str] = ("peak_NES_abs", "AUC_abs", "duration"),
    primary_stat: str = "peak_NES_abs",
    global_null: bool = False,
) -> pd.DataFrame:
    """
    Add empirical event-level p-values and FDRs using permutation null events.

    The default is the TED event-FDR target: each observed event is compared
    with the same event statistic recomputed in every permutation (the event
    statistic already contains its within-event window maximum), followed by
    BH across events. ``global_null=True`` is retained as a family-wide maxT
    FWER sensitivity analysis; it is not followed by BH.
    """
    stats = [_normalize_event_stat(stat) for stat in stats]
    primary_stat = _normalize_event_stat(primary_stat)
    if primary_stat not in stats:
        stats.append(primary_stat)
    if observed_events is None or observed_events.empty:
        return pd.DataFrame()
    calibrated = observed_events.copy()
    if null_events is None or null_events.empty:
        for stat in stats:
            calibrated[f"event_p_{stat}"] = np.nan
            calibrated[f"event_fdr_{stat}"] = np.nan
        calibrated["event_p"] = np.nan
        calibrated["event_fdr"] = np.nan
        return calibrated

    observed_path_col = _pathway_column(calibrated)
    null_path_col = _pathway_column(null_events)
    for stat in stats:
        observed = _event_stat_values(calibrated, stat).to_numpy(dtype=float)
        null_stat = _event_stat_values(null_events, stat)
        if global_null:
            null_values = _null_distribution(null_events, null_stat, global_null=True)
            p_values = _empirical_p_values(observed, null_values)
        else:
            p_values = np.full(len(calibrated), np.nan, dtype=float)
            null_tmp = pd.DataFrame(
                {
                    "pathway": null_events[null_path_col].map(_normalize_pathway_name),
                    "perm_id": null_events.get("perm_id", pd.Series(np.arange(len(null_events)))),
                    "stat": null_stat.to_numpy(dtype=float),
                }
            )
            for idx, (pathway, value) in enumerate(
                zip(calibrated[observed_path_col], observed, strict=True)
            ):
                if not np.isfinite(value):
                    continue
                event_null = null_tmp[
                    null_tmp["pathway"].eq(_normalize_pathway_name(pathway))
                ]
                if event_null.empty:
                    continue
                if "perm_id" in event_null:
                    values = event_null.groupby("perm_id")["stat"].max().to_numpy(dtype=float)
                else:
                    values = event_null["stat"].to_numpy(dtype=float)
                p_values[idx] = _empirical_p_values(np.asarray([value]), values)[0]
        calibrated[f"event_p_{stat}"] = p_values
        if global_null:
            calibrated[f"event_fwer_p_{stat}"] = p_values
            calibrated[f"event_fdr_{stat}"] = np.nan
        else:
            calibrated[f"event_fdr_{stat}"] = _bh_adjust(p_values)

    calibrated["event_p"] = calibrated[f"event_p_{primary_stat}"]
    if global_null:
        calibrated["family_fwer_p"] = calibrated[f"event_fwer_p_{primary_stat}"]
        calibrated["event_q"] = np.nan
        calibrated["event_fdr"] = np.nan
        multiplicity_method = "family_wide_maxT_fwer"
        across_event_adjustment = "none"
        primary_error_rate = "family-wise error rate"
    else:
        calibrated["event_q"] = calibrated[f"event_fdr_{primary_stat}"]
        calibrated["event_fdr"] = calibrated["event_q"]
        calibrated["family_fwer_p"] = np.nan
        multiplicity_method = "per_event_max_window_bh"
        across_event_adjustment = "BH"
        primary_error_rate = "event-level FDR"
    calibrated["multiplicity_method"] = multiplicity_method
    calibrated["window_selection_correction"] = "per_event_permutation_max"
    calibrated["across_event_adjustment"] = across_event_adjustment
    calibrated["primary_error_rate"] = primary_error_rate
    calibrated.attrs["calibration"] = {
        "type": "event_permutation",
        "stats": list(stats),
        "primary_stat": primary_stat,
        "global_null": global_null,
        "multiplicity_method": multiplicity_method,
        "across_event_adjustment": across_event_adjustment,
        "primary_error_rate": primary_error_rate,
        "n_null_events": int(len(null_events)),
        "n_permutations": int(null_events["perm_id"].nunique())
        if "perm_id" in null_events.columns
        else None,
    }
    return calibrated


def calibrate_comparison(
    observed_comparison: pd.DataFrame,
    null_comparisons: pd.DataFrame,
    stats: Iterable[str] = ("delta_AUC_abs", "delta_peak_time_abs"),
    primary_stat: str = "delta_AUC_abs",
    global_null: bool = True,
) -> pd.DataFrame:
    """Add empirical p-values/FDRs to condition or branch comparison rows."""
    stats = [_normalize_comparison_stat(stat) for stat in stats]
    primary_stat = _normalize_comparison_stat(primary_stat)
    if primary_stat not in stats:
        stats.append(primary_stat)
    if observed_comparison is None or observed_comparison.empty:
        return pd.DataFrame()
    calibrated = observed_comparison.copy()
    if null_comparisons is None or null_comparisons.empty:
        for stat in stats:
            calibrated[f"comparison_p_{stat}"] = np.nan
            calibrated[f"comparison_fdr_{stat}"] = np.nan
        calibrated["comparison_p"] = np.nan
        calibrated["comparison_fdr"] = np.nan
        return calibrated

    for stat in stats:
        observed = _comparison_stat_values(calibrated, stat).to_numpy(dtype=float)
        null_values = _null_distribution(
            null_comparisons,
            _comparison_stat_values(null_comparisons, stat),
            global_null=global_null,
        )
        p_values = _empirical_p_values(observed, null_values)
        calibrated[f"comparison_p_{stat}"] = p_values
        calibrated[f"comparison_fdr_{stat}"] = _bh_adjust(p_values)

    calibrated["comparison_p"] = calibrated[f"comparison_p_{primary_stat}"]
    calibrated["comparison_fdr"] = calibrated[f"comparison_fdr_{primary_stat}"]
    calibrated.attrs["calibration"] = {
        "type": "group_label_permutation",
        "stats": list(stats),
        "primary_stat": primary_stat,
        "global_null": global_null,
        "n_null_comparisons": int(len(null_comparisons)),
        "n_permutations": int(null_comparisons["perm_id"].nunique())
        if "perm_id" in null_comparisons.columns
        else None,
    }
    return calibrated


def targeted_directional_calibration(
    observed_comparison: pd.DataFrame,
    null_comparisons: pd.DataFrame,
    expected_direction: Mapping[str, str],
    reference_label: str = "branch_a",
    query_label: str = "branch_b",
    value: str = "peak_NES",
    hypothesis_family: str = "targeted_directional_hypotheses",
    n_perm: Optional[int] = None,
    q_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Calibrate pre-specified directional branch/condition hypotheses.

    ``expected_direction`` maps pathway names to the group expected to have the
    larger value, e.g. ``{"HEME": "branch_a"}``. Only those pathways are tested
    and BH correction is performed within that targeted family.
    """
    if observed_comparison is None or observed_comparison.empty:
        return pd.DataFrame()
    if not expected_direction:
        raise ValueError("expected_direction must contain at least one pathway")

    ref_col = f"{reference_label}_{value}"
    query_col = f"{query_label}_{value}"
    if ref_col not in observed_comparison or query_col not in observed_comparison:
        raise ValueError(
            f"Could not find directional value columns '{ref_col}' and '{query_col}'"
        )
    if null_comparisons is not None and not null_comparisons.empty:
        if ref_col not in null_comparisons or query_col not in null_comparisons:
            raise ValueError(
                f"Could not find directional value columns '{ref_col}' and '{query_col}' in null_comparisons"
            )

    observed_path_col = _pathway_column(observed_comparison)
    null_path_col = (
        _pathway_column(null_comparisons)
        if null_comparisons is not None and not null_comparisons.empty
        else None
    )
    direction_lookup = {
        _normalize_pathway_name(pathway): str(direction)
        for pathway, direction in expected_direction.items()
    }
    rows = []
    for _, row in observed_comparison.iterrows():
        pathway = str(row[observed_path_col])
        key = _normalize_pathway_name(pathway)
        if key not in direction_lookup:
            continue
        expected = direction_lookup[key]
        ref_value = float(row[ref_col])
        query_value = float(row[query_col])
        if expected in {reference_label, "reference", "control", "branch_a"}:
            observed_stat = ref_value - query_value
        elif expected in {query_label, "query", "case", "branch_b"}:
            observed_stat = query_value - ref_value
        else:
            raise ValueError(
                "expected_direction values must match reference/query labels or "
                "one of reference, query, control, case, branch_a, branch_b"
            )

        null_values = np.array([], dtype=float)
        if (
            null_comparisons is not None
            and not null_comparisons.empty
            and null_path_col is not None
        ):
            null_group = null_comparisons[
                null_comparisons[null_path_col].map(_normalize_pathway_name) == key
            ]
            if not null_group.empty:
                ref_null = pd.to_numeric(null_group[ref_col], errors="coerce")
                query_null = pd.to_numeric(null_group[query_col], errors="coerce")
                if expected in {reference_label, "reference", "control", "branch_a"}:
                    null_values = (ref_null - query_null).to_numpy(dtype=float)
                else:
                    null_values = (query_null - ref_null).to_numpy(dtype=float)
                null_values = null_values[np.isfinite(null_values)]

        effective_perm = int(n_perm) if n_perm is not None else int(len(null_values))
        if effective_perm <= 0 or not np.isfinite(observed_stat):
            p_value = np.nan
            null_mean = np.nan
            null_sd = np.nan
            p_floor = np.nan
        else:
            p_value = (1.0 + float(np.sum(null_values >= observed_stat))) / (
                1.0 + float(effective_perm)
            )
            null_mean = float(np.mean(null_values)) if len(null_values) else 0.0
            null_sd = float(np.std(null_values, ddof=1)) if len(null_values) > 1 else 0.0
            p_floor = 1.0 / (1.0 + float(effective_perm))

        rows.append(
            {
                "Pathway": pathway,
                "hypothesis_family": hypothesis_family,
                "expected_direction": expected,
                "directional_value": value,
                "reference_value": ref_value,
                "query_value": query_value,
                "directional_stat": observed_stat,
                "null_mean": null_mean,
                "null_sd": null_sd,
                "directional_p": p_value,
                "minimum_attainable_p": p_floor,
                "n_perm": effective_perm,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["directional_q"] = _bh_adjust(out["directional_p"].to_numpy(dtype=float))
    n_tests = len(out)
    out["minimum_attainable_q"] = np.minimum(
        1.0,
        pd.to_numeric(out["minimum_attainable_p"], errors="coerce") * n_tests,
    )
    out[f"discovery_possible_at_q_{str(q_threshold).replace('.', '_')}"] = (
        out["minimum_attainable_q"] <= q_threshold
    )
    out.attrs["power_report"] = event_fdr_power_report(
        out.rename(
            columns={
                "directional_p": "event_p",
                "directional_q": "event_q",
            }
        ).assign(event_stat=f"directional_{value}"),
        q_threshold=q_threshold,
        hypothesis_family=hypothesis_family,
    )
    return out.sort_values(["directional_q", "directional_p", "Pathway"]).reset_index(
        drop=True
    )


def _copy_with_permuted_obs(adata, key: str, values: np.ndarray):
    copied = adata.copy()
    copied.obs[key] = values
    return copied


def _gene_label_permuted_sets(
    gene_sets,
    gene_universe: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, list[str]]:
    universe = np.asarray(gene_universe, dtype=str)
    if len(universe) == 0:
        raise ValueError("gene universe is empty")

    universe_set = set(map(str, universe))
    out = {}
    for name, genes in _load_gene_sets(gene_sets).items():
        overlap = [str(gene) for gene in genes if str(gene) in universe_set]
        if not overlap:
            out[name] = []
            continue
        size = min(len(overlap), len(universe))
        out[name] = rng.choice(universe, size=size, replace=False).astype(str).tolist()
    return out


def _permuted_within_groups(values: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(values).copy()
    groups = np.asarray(groups)
    finite = np.isfinite(out.astype(float)) if np.issubdtype(out.dtype, np.number) else pd.notna(out)
    for group in pd.Series(groups[finite]).dropna().unique():
        idx = np.where(finite & (groups == group))[0]
        if len(idx) > 1:
            out[idx] = rng.permutation(out[idx])
    return out


def _map_permutations(n_jobs: int, func, items):
    if int(n_jobs) <= 1:
        return [func(item) for item in items]
    with ThreadPoolExecutor(max_workers=int(n_jobs)) as pool:
        return list(pool.map(func, items))


def _condition_labels_permuted_by_replicate(
    adata,
    condition_key: str,
    sample_key: str,
    rng: np.random.Generator,
) -> np.ndarray:
    labels = adata.obs[condition_key].astype(str).to_numpy()
    samples = adata.obs[sample_key].astype(str).to_numpy()
    out = labels.copy()

    per_sample = pd.DataFrame({"sample": samples, "condition": labels})
    n_conditions_per_sample = per_sample.groupby("sample")["condition"].nunique()
    pure_samples = bool((n_conditions_per_sample == 1).all())
    if pure_samples:
        sample_conditions = (
            per_sample.drop_duplicates("sample")
            .set_index("sample")["condition"]
            .astype(str)
        )
        shuffled = rng.permutation(sample_conditions.to_numpy())
        mapping = dict(zip(sample_conditions.index.to_numpy(), shuffled))
        return np.asarray([mapping[sample] for sample in samples], dtype=object)

    for sample in pd.Series(samples).dropna().unique():
        idx = np.where(samples == sample)[0]
        if len(idx) > 1:
            out[idx] = rng.permutation(out[idx])
    return out


def _condition_labels_permuted_within_pseudotime_bins(
    adata,
    condition_key: str,
    pseudotime_key: str,
    control,
    case,
    rng: np.random.Generator,
    n_bins: int = 10,
    sample_key: Optional[str] = None,
) -> np.ndarray:
    labels = adata.obs[condition_key].astype(str).to_numpy()
    pt = pd.to_numeric(adata.obs[pseudotime_key], errors="coerce").to_numpy(dtype=float)
    bins = _pseudotime_bins(pt, n_bins)
    out = labels.copy()
    keep = np.isin(labels, [str(control), str(case)])
    if sample_key is not None:
        samples = adata.obs[sample_key].astype(str).to_numpy()
        for sample in pd.Series(samples[keep]).dropna().unique():
            sample_mask = keep & (samples == sample)
            for bin_id in np.unique(bins[sample_mask]):
                if bin_id < 0:
                    continue
                idx = np.where(sample_mask & (bins == bin_id))[0]
                if len(idx) > 1:
                    out[idx] = rng.permutation(out[idx])
        return out

    for bin_id in np.unique(bins[keep]):
        if bin_id < 0:
            continue
        idx = np.where(keep & (bins == bin_id))[0]
        if len(idx) > 1:
            out[idx] = rng.permutation(out[idx])
    return out


def _condition_replicate_warning(
    adata,
    condition_key: str,
    sample_key: str,
    control,
    case,
) -> str:
    labels = adata.obs[condition_key].astype(str)
    samples = adata.obs[sample_key].astype(str)
    counts = {}
    for condition in (str(control), str(case)):
        counts[condition] = int(samples[labels == condition].nunique())
    if min(counts.values()) < 3:
        return "descriptive_only_low_replicate_count"
    return ""


def _pseudotime_bins(pt: np.ndarray, n_bins: int) -> np.ndarray:
    if n_bins < 1:
        raise ValueError("n_pseudotime_bins must be at least 1")
    pt = np.asarray(pt, dtype=float)
    bins = np.full(len(pt), -1, dtype=int)
    finite = np.isfinite(pt)
    if not finite.any():
        return bins
    try:
        labels = pd.qcut(pt[finite], q=min(n_bins, int(finite.sum())), labels=False, duplicates="drop")
    except ValueError:
        labels = pd.cut(pt[finite], bins=min(n_bins, int(finite.sum())), labels=False, include_lowest=True)
    labels = pd.Series(labels).fillna(-1).astype(int).to_numpy()
    bins[finite] = labels
    return bins


def _branch_labels_permuted_within_pseudotime_bins(
    adata,
    branch_key: str,
    pseudotime_key: str,
    branch_a,
    branch_b,
    n_bins: int,
    rng: np.random.Generator,
) -> np.ndarray:
    labels = adata.obs[branch_key].astype(str).to_numpy()
    pt = pd.to_numeric(adata.obs[pseudotime_key], errors="coerce").to_numpy(dtype=float)
    bins = _pseudotime_bins(pt, n_bins)
    out = labels.copy()
    keep = np.isin(labels, [str(branch_a), str(branch_b)])
    for bin_id in np.unique(bins[keep]):
        if bin_id < 0:
            continue
        idx = np.where(keep & (bins == bin_id))[0]
        if len(idx) > 1:
            out[idx] = rng.permutation(out[idx])
    return out


def run_event_permutation_fdr(
    adata,
    gmt_path,
    pseudotime_key: str = "dpt_pseudotime",
    n_permutations: int = 100,
    seed: int = 42,
    event_kwargs: Optional[dict] = None,
    calibration_stats: Iterable[str] = ("peak_NES_abs", "AUC_abs", "duration"),
    primary_stat: str = "peak_NES_abs",
    global_null: bool = False,
    **trajectory_kwargs,
) -> dict[str, pd.DataFrame]:
    """
    Calibrate pathway events by permuting pseudotime labels.

    Returns ``results``, ``events``, ``null_events``, and ``calibrated_events``.
    """
    if pseudotime_key not in adata.obs:
        raise ValueError(f"pseudotime_key '{pseudotime_key}' not found in adata.obs")
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")

    event_kwargs = {} if event_kwargs is None else dict(event_kwargs)
    rng = np.random.default_rng(seed)
    observed_results = run_trajectory_gsea(
        adata,
        gmt_path=gmt_path,
        pseudotime_key=pseudotime_key,
        seed=seed,
        **trajectory_kwargs,
    )
    observed_events = summarize_events(observed_results, **event_kwargs)

    pt = pd.to_numeric(adata.obs[pseudotime_key], errors="coerce").to_numpy()
    null_frames = []
    for perm_id in range(n_permutations):
        permuted = pt.copy()
        finite = np.isfinite(permuted)
        permuted[finite] = rng.permutation(permuted[finite])
        perm_adata = _copy_with_permuted_obs(adata, pseudotime_key, permuted)
        perm_results = run_trajectory_gsea(
            perm_adata,
            gmt_path=gmt_path,
            pseudotime_key=pseudotime_key,
            seed=seed + perm_id + 1,
            **trajectory_kwargs,
        )
        perm_events = summarize_events(perm_results, **event_kwargs)
        if not perm_events.empty:
            perm_events = perm_events.copy()
            perm_events.attrs.clear()
            perm_events["perm_id"] = perm_id
            null_frames.append(perm_events)

    null_events = pd.concat(null_frames, ignore_index=True) if null_frames else pd.DataFrame()
    calibrated = calibrate_events(
        observed_events,
        null_events,
        stats=calibration_stats,
        primary_stat=primary_stat,
        global_null=global_null,
    )
    return {
        "results": observed_results,
        "events": observed_events,
        "null_events": null_events,
        "calibrated_events": calibrated,
    }


def estimate_event_fdr(
    result: Optional[pd.DataFrame] = None,
    adata=None,
    gmt_path=None,
    pseudotime_key: str = "dpt_pseudotime",
    event_stats: Iterable[str] = (
        "max_abs_NES",
        "AUC_abs",
        "longest_significant_run",
        "peak_sharpness",
    ),
    null: str = "pseudotime_permutation",
    n_perm: int = 100,
    seed: int = 42,
    event_kwargs: Optional[dict] = None,
    primary_stat: Optional[str] = None,
    global_null: bool = False,
    early_stop: bool = False,
    early_stop_interval: int = 50,
    early_stop_threshold: float = 0.05,
    hypothesis_family: str = "hallmark_all",
    pathways: Optional[Sequence[str]] = None,
    q_threshold: float = 0.05,
    condition_key: Optional[str] = None,
    sample_key: Optional[str] = None,
    replicate_key: Optional[str] = None,
    control=None,
    case=None,
    branch_key: Optional[str] = None,
    branch_a=None,
    branch_b=None,
    n_pseudotime_bins: int = 10,
    layer: Optional[str] = None,
    use_raw: bool = False,
    n_jobs: int = 1,
    **trajectory_kwargs,
) -> pd.DataFrame:
    """
    Estimate trajectory/event-level FDR for pathway events.

    Window-level FDR remains attached to individual rolling windows. This
    function calibrates pathway-level event statistics against permutation nulls
    and returns an event or comparison table with ``event_fdr``.
    """
    if n_perm < 1:
        raise ValueError("n_perm must be at least 1")
    if adata is None:
        raise ValueError("adata is required for permutation-based event FDR")
    if gmt_path is None:
        raise ValueError("gmt_path is required for permutation-based event FDR")
    if int(n_jobs) < 1:
        raise ValueError("n_jobs must be at least 1")

    null = null.lower().replace("-", "_")
    if pathways is not None and hypothesis_family == "hallmark_all":
        hypothesis_family = "custom_targeted_hypotheses"
    calibration_gmt = _filter_gene_sets_for_family(gmt_path, pathways)
    event_kwargs = {} if event_kwargs is None else dict(event_kwargs)
    stats = [_normalize_event_stat(stat) for stat in event_stats]
    primary_stat = _normalize_event_stat(primary_stat or stats[0])
    comparison_stats = [
        stat for stat in stats if stat in {"delta_AUC_abs", "delta_peak_time_abs"}
    ]
    trajectory_stats = [
        stat for stat in stats if stat not in {"delta_AUC_abs", "delta_peak_time_abs"}
    ]
    if replicate_key is not None:
        if sample_key is not None and str(sample_key) != str(replicate_key):
            raise ValueError("Pass only one of sample_key or replicate_key, or use the same value")
        sample_key = replicate_key

    if null == "condition_label_permutation_within_pseudotime_bins":
        if condition_key is None:
            raise ValueError("condition_key is required for condition_label_permutation_within_pseudotime_bins")
        if pseudotime_key not in adata.obs:
            raise ValueError(f"pseudotime_key '{pseudotime_key}' not found in adata.obs")
        if control is None or case is None:
            values = pd.Series(adata.obs[condition_key]).dropna().astype(str).unique().tolist()
            if len(values) < 2:
                raise ValueError("At least two condition values are required")
            control = values[0] if control is None else control
            case = values[1] if case is None else case
        cmp_stats = comparison_stats or ["delta_AUC_abs", "delta_peak_time_abs"]
        mode = trajectory_kwargs.pop("mode", "matched_window")
        observed = compare_trajectory_gsea(
            adata,
            gmt_path=calibration_gmt,
            condition_key=condition_key,
            sample_key=sample_key,
            control=control,
            case=case,
            mode=mode,
            pseudotime_key=pseudotime_key,
            n_permutations=0,
            seed=seed,
            event_kwargs=event_kwargs,
            layer=layer,
            use_raw=use_raw,
            **trajectory_kwargs,
        )
        observed = _filter_pathway_family(observed, pathways)
        rng = np.random.default_rng(seed)
        null_frames = []
        for perm_id in range(n_perm):
            permuted = _condition_labels_permuted_within_pseudotime_bins(
                adata,
                condition_key=condition_key,
                pseudotime_key=pseudotime_key,
                control=control,
                case=case,
                rng=rng,
                n_bins=n_pseudotime_bins,
                sample_key=sample_key,
            )
            perm_adata = _copy_with_permuted_obs(adata, condition_key, permuted)
            cmp_df = compare_trajectory_gsea(
                perm_adata,
                gmt_path=calibration_gmt,
                condition_key=condition_key,
                sample_key=sample_key,
                control=control,
                case=case,
                mode=mode,
                pseudotime_key=pseudotime_key,
                n_permutations=0,
                seed=seed + perm_id + 1,
                event_kwargs=event_kwargs,
                layer=layer,
                use_raw=use_raw,
                **trajectory_kwargs,
            )
            if not cmp_df.empty:
                cmp_df = cmp_df.copy()
                cmp_df.attrs.clear()
                cmp_df["perm_id"] = perm_id
                cmp_df = _filter_pathway_family(cmp_df, pathways)
                null_frames.append(cmp_df)
        null_comparisons = (
            pd.concat(null_frames, ignore_index=True) if null_frames else pd.DataFrame()
        )
        calibrated = _event_calibration_long_table(
            observed,
            null_comparisons,
            stats=cmp_stats,
            kind="comparison",
            n_perm=n_perm,
            null_model=null,
            global_null=global_null,
        )
        calibrated = _attach_power_report(
            calibrated,
            q_threshold=q_threshold,
            hypothesis_family=hypothesis_family,
        )
        calibrated.attrs["comparison"] = observed
        calibrated.attrs["calibration_kind"] = null
        calibrated.attrs["n_null_comparisons"] = int(len(null_comparisons))
        return calibrated

    if null == "condition_label_permutation_by_replicate":
        if condition_key is None:
            raise ValueError("condition_key is required for condition_label_permutation_by_replicate")
        if sample_key is None:
            raise ValueError("replicate_key or sample_key is required for condition_label_permutation_by_replicate")
        if control is None or case is None:
            values = pd.Series(adata.obs[condition_key]).dropna().astype(str).unique().tolist()
            if len(values) < 2:
                raise ValueError("At least two condition values are required")
            control = values[0] if control is None else control
            case = values[1] if case is None else case
        cmp_stats = comparison_stats or ["delta_AUC_abs", "delta_peak_time_abs"]
        mode = trajectory_kwargs.pop("mode", None)
        if mode is None and sample_key is not None:
            mode = "replicate_aware"
        observed = compare_trajectory_gsea(
            adata,
            gmt_path=calibration_gmt,
            condition_key=condition_key,
            sample_key=sample_key,
            control=control,
            case=case,
            mode=mode,
            pseudotime_key=pseudotime_key,
            n_permutations=0,
            seed=seed,
            event_kwargs=event_kwargs,
            layer=layer,
            use_raw=use_raw,
            **trajectory_kwargs,
        )
        observed = _filter_pathway_family(observed, pathways)
        rng = np.random.default_rng(seed)
        null_frames = []
        for perm_id in range(n_perm):
            permuted = _condition_labels_permuted_by_replicate(
                adata, condition_key, sample_key, rng
            )
            perm_adata = _copy_with_permuted_obs(adata, condition_key, permuted)
            cmp_df = compare_trajectory_gsea(
                perm_adata,
                gmt_path=calibration_gmt,
                condition_key=condition_key,
                sample_key=sample_key,
                control=control,
                case=case,
                mode=mode,
                pseudotime_key=pseudotime_key,
                n_permutations=0,
                seed=seed + perm_id + 1,
                event_kwargs=event_kwargs,
                layer=layer,
                use_raw=use_raw,
                **trajectory_kwargs,
            )
            if not cmp_df.empty:
                cmp_df = cmp_df.copy()
                cmp_df.attrs.clear()
                cmp_df["perm_id"] = perm_id
                cmp_df = _filter_pathway_family(cmp_df, pathways)
                null_frames.append(cmp_df)
        null_comparisons = (
            pd.concat(null_frames, ignore_index=True) if null_frames else pd.DataFrame()
        )
        warning = _condition_replicate_warning(
            adata, condition_key, sample_key, control, case
        )
        calibrated = _event_calibration_long_table(
            observed,
            null_comparisons,
            stats=cmp_stats,
            kind="comparison",
            n_perm=n_perm,
            null_model=null,
            global_null=global_null,
            base_warning=warning,
        )
        calibrated = _attach_power_report(
            calibrated,
            q_threshold=q_threshold,
            hypothesis_family=hypothesis_family,
        )
        calibrated.attrs["comparison"] = observed
        calibrated.attrs["calibration_kind"] = null
        calibrated.attrs["n_null_comparisons"] = int(len(null_comparisons))
        return calibrated

    if null == "branch_label_permutation_within_pseudotime_bins":
        if branch_key is None:
            raise ValueError("branch_key is required for branch_label_permutation_within_pseudotime_bins")
        if pseudotime_key not in adata.obs:
            raise ValueError(f"pseudotime_key '{pseudotime_key}' not found in adata.obs")
        if branch_a is None or branch_b is None:
            values = pd.Series(adata.obs[branch_key]).dropna().astype(str).unique().tolist()
            if len(values) < 2:
                raise ValueError("At least two branch values are required")
            branch_a = values[0] if branch_a is None else branch_a
            branch_b = values[1] if branch_b is None else branch_b
        cmp_stats = comparison_stats or ["delta_AUC_abs", "delta_peak_time_abs"]
        observed = compare_trajectory_gsea(
            adata,
            gmt_path=calibration_gmt,
            condition_key=branch_key,
            control=branch_a,
            case=branch_b,
            pseudotime_key=pseudotime_key,
            n_permutations=0,
            seed=seed,
            event_kwargs=event_kwargs,
            layer=layer,
            use_raw=use_raw,
            **trajectory_kwargs,
        )
        observed = _filter_pathway_family(observed, pathways)
        rng = np.random.default_rng(seed)
        null_frames = []
        for perm_id in range(n_perm):
            permuted = _branch_labels_permuted_within_pseudotime_bins(
                adata,
                branch_key=branch_key,
                pseudotime_key=pseudotime_key,
                branch_a=branch_a,
                branch_b=branch_b,
                n_bins=n_pseudotime_bins,
                rng=rng,
            )
            perm_adata = _copy_with_permuted_obs(adata, branch_key, permuted)
            cmp_df = compare_trajectory_gsea(
                perm_adata,
                gmt_path=calibration_gmt,
                condition_key=branch_key,
                control=branch_a,
                case=branch_b,
                pseudotime_key=pseudotime_key,
                n_permutations=0,
                seed=seed + perm_id + 1,
                event_kwargs=event_kwargs,
                layer=layer,
                use_raw=use_raw,
                **trajectory_kwargs,
            )
            if not cmp_df.empty:
                cmp_df = cmp_df.copy()
                cmp_df.attrs.clear()
                cmp_df["perm_id"] = perm_id
                cmp_df = _filter_pathway_family(cmp_df, pathways)
                null_frames.append(cmp_df)
        null_comparisons = (
            pd.concat(null_frames, ignore_index=True) if null_frames else pd.DataFrame()
        )
        calibrated = _event_calibration_long_table(
            observed,
            null_comparisons,
            stats=cmp_stats,
            kind="comparison",
            n_perm=n_perm,
            null_model=null,
            global_null=global_null,
        )
        calibrated = _attach_power_report(
            calibrated,
            q_threshold=q_threshold,
            hypothesis_family=hypothesis_family,
        )
        calibrated.attrs["calibration_kind"] = null
        calibrated.attrs["n_null_comparisons"] = int(len(null_comparisons))
        return calibrated

    if null in {"pseudobulk_permutation", "sample_label_permutation", "mixed_effect"}:
        if condition_key is None:
            raise ValueError(f"condition_key is required for {null}")
        if sample_key is None:
            raise ValueError(f"sample_key is required for {null}")
        mode = {
            "pseudobulk_permutation": "pseudobulk",
            "sample_label_permutation": "replicate_aware",
            "mixed_effect": "mixed_effect",
        }[null]
        comparison = compare_trajectory_gsea(
            adata,
            gmt_path=calibration_gmt,
            condition_key=condition_key,
            sample_key=sample_key,
            control=control,
            case=case,
            mode=mode,
            pseudotime_key=pseudotime_key,
            n_permutations=0 if null == "mixed_effect" else n_perm,
            seed=seed,
            event_kwargs=event_kwargs,
            layer=layer,
            use_raw=use_raw,
            **trajectory_kwargs,
        )
        if pathways is not None:
            comparison = _filter_pathway_family(comparison, pathways)
        comparison.attrs["calibration_kind"] = null
        return comparison

    if null == "condition_label_permutation":
        if condition_key is None:
            raise ValueError("condition_key is required for condition_label_permutation")
        if control is None or case is None:
            values = pd.Series(adata.obs[condition_key]).dropna().astype(str).unique().tolist()
            if len(values) < 2:
                raise ValueError("At least two condition values are required")
            control = values[0] if control is None else control
            case = values[1] if case is None else case
        out = run_comparison_permutation_calibration(
            adata,
            gmt_path=calibration_gmt,
            condition_key=condition_key,
            control=control,
            case=case,
            n_permutations=n_perm,
            seed=seed,
            event_kwargs=event_kwargs,
            calibration_stats=("delta_AUC_abs", "delta_peak_time_abs"),
            primary_stat="delta_AUC_abs",
            global_null=global_null,
            pseudotime_key=pseudotime_key,
            layer=layer,
            use_raw=use_raw,
            **trajectory_kwargs,
        )
        if pathways is None:
            calibrated = out["calibrated_comparison"].copy()
        else:
            calibrated = calibrate_comparison(
                _filter_pathway_family(out["comparison"], pathways),
                _filter_pathway_family(out["null_comparisons"], pathways),
                stats=("delta_AUC_abs", "delta_peak_time_abs"),
                primary_stat="delta_AUC_abs",
                global_null=global_null,
            )
        calibrated["event_p"] = calibrated["comparison_p"]
        calibrated["event_fdr"] = calibrated["comparison_fdr"]
        calibrated = _attach_power_report(
            calibrated,
            q_threshold=q_threshold,
            hypothesis_family=hypothesis_family,
        )
        calibrated.attrs["calibration_kind"] = null
        calibrated.attrs["n_null_comparisons"] = int(len(out["null_comparisons"]))
        return calibrated

    if null == "branch_label_permutation":
        if branch_key is None:
            raise ValueError("branch_key is required for branch_label_permutation")
        if branch_a is None or branch_b is None:
            values = pd.Series(adata.obs[branch_key]).dropna().astype(str).unique().tolist()
            if len(values) < 2:
                raise ValueError("At least two branch values are required")
            branch_a = values[0] if branch_a is None else branch_a
            branch_b = values[1] if branch_b is None else branch_b
        out = run_branch_permutation_calibration(
            adata,
            gmt_path=calibration_gmt,
            branch_key=branch_key,
            branch_a=branch_a,
            branch_b=branch_b,
            n_permutations=n_perm,
            seed=seed,
            event_kwargs=event_kwargs,
            calibration_stats=("delta_AUC_abs", "delta_peak_time_abs"),
            primary_stat="delta_AUC_abs",
            global_null=global_null,
            pseudotime_key=pseudotime_key,
            layer=layer,
            use_raw=use_raw,
            **trajectory_kwargs,
        )
        if pathways is None:
            calibrated = out["calibrated_comparison"].copy()
        else:
            calibrated = calibrate_comparison(
                _filter_pathway_family(out["comparison"], pathways),
                _filter_pathway_family(out["null_comparisons"], pathways),
                stats=("delta_AUC_abs", "delta_peak_time_abs"),
                primary_stat="delta_AUC_abs",
                global_null=global_null,
            )
        calibrated["event_p"] = calibrated["comparison_p"]
        calibrated["event_fdr"] = calibrated["comparison_fdr"]
        calibrated = _attach_power_report(
            calibrated,
            q_threshold=q_threshold,
            hypothesis_family=hypothesis_family,
        )
        calibrated.attrs["calibration_kind"] = null
        calibrated.attrs["n_null_comparisons"] = int(len(out["null_comparisons"]))
        return calibrated

    if result is None:
        result = run_trajectory_gsea(
            adata,
            gmt_path=calibration_gmt,
            pseudotime_key=pseudotime_key,
            seed=seed,
            layer=layer,
            use_raw=use_raw,
            **trajectory_kwargs,
        )
    gene_set_index = result.attrs.get("gene_set_index") if hasattr(result, "attrs") else None
    if gene_set_index is not None and pathways is not None:
        wanted = {_normalize_pathway_name(pathway) for pathway in pathways}
        indexed = {_normalize_pathway_name(pathway) for pathway in gene_set_index.pathway_names}
        if not indexed.issubset(wanted):
            gene_set_index = None
    window_index = result.attrs.get("window_index") if hasattr(result, "attrs") else None
    observed_events = summarize_events(result, **event_kwargs)
    observed_events = _filter_pathway_family(observed_events, pathways)

    rng = np.random.default_rng(seed)
    null_frames = []
    n_perm_effective = n_perm
    early_stopped = False
    trajectory_stats = trajectory_stats or ["max_abs_NES", "AUC_abs", "duration", "sharpness"]
    if null in {"pseudotime_permutation", "pseudotime_within_replicate_permutation"}:
        if pseudotime_key not in adata.obs:
            raise ValueError(f"pseudotime_key '{pseudotime_key}' not found in adata.obs")
        if null == "pseudotime_within_replicate_permutation" and sample_key is None:
            raise ValueError("replicate_key or sample_key is required for pseudotime_within_replicate_permutation")
        pt = pd.to_numeric(adata.obs[pseudotime_key], errors="coerce").to_numpy()
        groups = (
            adata.obs[sample_key].astype(str).to_numpy()
            if null == "pseudotime_within_replicate_permutation"
            else None
        )
        def run_pseudotime_perm(perm_id: int):
            local_rng = np.random.default_rng(seed + 100_003 * (perm_id + 1))
            permuted = pt.copy()
            if groups is None:
                finite = np.isfinite(permuted)
                permuted[finite] = local_rng.permutation(permuted[finite])
            else:
                permuted = _permuted_within_groups(permuted, groups, local_rng)
            perm_adata = _copy_with_permuted_obs(adata, pseudotime_key, permuted)
            perm_results = run_trajectory_gsea(
                perm_adata,
                gmt_path=calibration_gmt,
                pseudotime_key=pseudotime_key,
                seed=seed + perm_id + 1,
                layer=layer,
                use_raw=use_raw,
                gene_set_index=gene_set_index,
                **trajectory_kwargs,
            )
            perm_events = summarize_events(perm_results, **event_kwargs)
            if not perm_events.empty:
                perm_events = perm_events.copy()
                perm_events.attrs.clear()
                perm_events["perm_id"] = perm_id
                perm_events = _filter_pathway_family(perm_events, pathways)
            return perm_events, perm_results.attrs.get("gene_set_index")

        if int(n_jobs) > 1 and not early_stop:
            for perm_events, _idx in _map_permutations(
                int(n_jobs), run_pseudotime_perm, range(n_perm)
            ):
                if not perm_events.empty:
                    null_frames.append(perm_events)
        else:
            for perm_id in range(n_perm):
                perm_events, perm_gene_index = run_pseudotime_perm(perm_id)
                gene_set_index = gene_set_index or perm_gene_index
                if not perm_events.empty:
                    null_frames.append(perm_events)
                if (
                    early_stop
                    and early_stop_interval > 0
                    and (perm_id + 1) % early_stop_interval == 0
                ):
                    current_null = (
                        pd.concat(null_frames, ignore_index=True)
                        if null_frames
                        else pd.DataFrame()
                    )
                    if _event_early_stop_ready(
                        observed_events,
                        current_null,
                        trajectory_stats,
                        n_perm_planned=n_perm,
                        threshold=early_stop_threshold,
                    ):
                        n_perm_effective = perm_id + 1
                        early_stopped = True
                        break
    elif null == "gene_label_permutation":
        _X, genes, _source = _expression_matrix(adata, layer=layer, use_raw=use_raw)
        def run_gene_label_perm(perm_id: int):
            local_rng = np.random.default_rng(seed + 200_003 * (perm_id + 1))
            perm_sets = _gene_label_permuted_sets(calibration_gmt, genes, local_rng)
            perm_results = run_trajectory_gsea(
                adata,
                gmt_path=perm_sets,
                pseudotime_key=pseudotime_key,
                seed=seed + perm_id + 1,
                layer=layer,
                use_raw=use_raw,
                window_index=window_index,
                **trajectory_kwargs,
            )
            perm_events = summarize_events(perm_results, **event_kwargs)
            if not perm_events.empty:
                perm_events = perm_events.copy()
                perm_events.attrs.clear()
                perm_events["perm_id"] = perm_id
                perm_events = _filter_pathway_family(perm_events, pathways)
            return perm_events, perm_results.attrs.get("window_index")

        if int(n_jobs) > 1 and not early_stop:
            for perm_events, _idx in _map_permutations(
                int(n_jobs), run_gene_label_perm, range(n_perm)
            ):
                if not perm_events.empty:
                    null_frames.append(perm_events)
        else:
            for perm_id in range(n_perm):
                perm_events, perm_window_index = run_gene_label_perm(perm_id)
                window_index = window_index or perm_window_index
                if not perm_events.empty:
                    null_frames.append(perm_events)
                if (
                    early_stop
                    and early_stop_interval > 0
                    and (perm_id + 1) % early_stop_interval == 0
                ):
                    current_null = (
                        pd.concat(null_frames, ignore_index=True)
                        if null_frames
                        else pd.DataFrame()
                    )
                    if _event_early_stop_ready(
                        observed_events,
                        current_null,
                        trajectory_stats,
                        n_perm_planned=n_perm,
                        threshold=early_stop_threshold,
                    ):
                        n_perm_effective = perm_id + 1
                        early_stopped = True
                        break
    else:
        raise ValueError(
            "null must be one of 'pseudotime_permutation', "
            "'pseudotime_within_replicate_permutation', 'gene_label_permutation', "
            "'condition_label_permutation', 'condition_label_permutation_by_replicate', "
            "'condition_label_permutation_within_pseudotime_bins', "
            "'branch_label_permutation', 'branch_label_permutation_within_pseudotime_bins', "
            "'pseudobulk_permutation', 'sample_label_permutation', or 'mixed_effect'"
        )

    null_events = pd.concat(null_frames, ignore_index=True) if null_frames else pd.DataFrame()
    calibrated = _event_calibration_long_table(
        observed_events,
        null_events,
        stats=trajectory_stats,
        kind="event",
        n_perm=n_perm,
        null_model=null,
        global_null=global_null,
        n_perm_effective=n_perm_effective,
        early_stopped=early_stopped,
    )
    calibrated = _attach_power_report(
        calibrated,
        q_threshold=q_threshold,
        hypothesis_family=hypothesis_family,
    )
    calibrated.attrs["calibration_kind"] = null
    calibrated.attrs["n_null_events"] = int(len(null_events))
    return calibrated


def run_group_comparison_permutation_calibration(
    adata,
    gmt_path,
    group_key: str,
    reference,
    query,
    n_permutations: int = 100,
    seed: int = 42,
    event_kwargs: Optional[dict] = None,
    calibration_stats: Iterable[str] = ("delta_AUC_abs", "delta_peak_time_abs"),
    primary_stat: str = "delta_AUC_abs",
    global_null: bool = True,
    **trajectory_kwargs,
) -> dict[str, pd.DataFrame]:
    """
    Calibrate condition/branch comparison by permuting group labels.

    Only cells whose group is ``reference`` or ``query`` are used. Group sizes
    are preserved by shuffling labels across those cells.
    """
    if group_key not in adata.obs:
        raise ValueError(f"group_key '{group_key}' not found in adata.obs")
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")

    labels = adata.obs[group_key].astype(str).to_numpy()
    keep = np.isin(labels, [str(reference), str(query)])
    if keep.sum() == 0:
        raise ValueError("No cells match the requested reference/query groups")
    work = adata[keep].copy()
    labels = work.obs[group_key].astype(str).to_numpy()
    if len(set(labels)) < 2:
        raise ValueError("Both reference and query groups must contain cells")
    work.obs[group_key] = labels
    reference = str(reference)
    query = str(query)

    event_kwargs = {} if event_kwargs is None else dict(event_kwargs)
    observed = compare_trajectory_gsea(
        work,
        gmt_path=gmt_path,
        condition_key=group_key,
        control=reference,
        case=query,
        event_kwargs=event_kwargs,
        seed=seed,
        **trajectory_kwargs,
    )

    rng = np.random.default_rng(seed)
    null_frames = []
    for perm_id in range(n_permutations):
        permuted = rng.permutation(labels)
        perm_adata = _copy_with_permuted_obs(work, group_key, permuted)
        cmp_df = compare_trajectory_gsea(
            perm_adata,
            gmt_path=gmt_path,
            condition_key=group_key,
            control=reference,
            case=query,
            event_kwargs=event_kwargs,
            seed=seed + perm_id + 1,
            **trajectory_kwargs,
        )
        if not cmp_df.empty:
            cmp_df = cmp_df.copy()
            cmp_df.attrs.clear()
            cmp_df["perm_id"] = perm_id
            null_frames.append(cmp_df)

    null_comparisons = (
        pd.concat(null_frames, ignore_index=True) if null_frames else pd.DataFrame()
    )
    calibrated = calibrate_comparison(
        observed,
        null_comparisons,
        stats=calibration_stats,
        primary_stat=primary_stat,
        global_null=global_null,
    )
    calibrated.attrs["results"] = observed.attrs.get("results", pd.DataFrame())
    calibrated.attrs["events"] = observed.attrs.get("events", pd.DataFrame())
    return {
        "comparison": observed,
        "null_comparisons": null_comparisons,
        "calibrated_comparison": calibrated,
        "results": observed.attrs.get("results", pd.DataFrame()),
        "events": observed.attrs.get("events", pd.DataFrame()),
    }


def run_comparison_permutation_calibration(
    adata,
    gmt_path,
    condition_key: str,
    control,
    case,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Condition-comparison alias for group label permutation calibration."""
    return run_group_comparison_permutation_calibration(
        adata,
        gmt_path=gmt_path,
        group_key=condition_key,
        reference=control,
        query=case,
        **kwargs,
    )


def run_branch_permutation_calibration(
    adata,
    gmt_path,
    branch_key: str,
    branch_a,
    branch_b,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Branch-comparison alias for group label permutation calibration."""
    return run_group_comparison_permutation_calibration(
        adata,
        gmt_path=gmt_path,
        group_key=branch_key,
        reference=branch_a,
        query=branch_b,
        **kwargs,
    )
