import math
from typing import Optional

import numpy as np
import pandas as pd


def _pick_column(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    lower_to_original = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        lower = candidate.lower()
        if lower in lower_to_original:
            return lower_to_original[lower]
    return None


def _trapz(values: np.ndarray, x: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, x=x))
    return float(np.trapz(values, x=x))


def _stable_onset(
    time: np.ndarray,
    values: np.ndarray,
    significant: np.ndarray,
    direction: int,
    min_consecutive: int,
) -> float:
    signed = values * direction
    active = significant & (signed > 0)
    run = 0
    start = None
    for idx, flag in enumerate(active):
        if flag:
            if run == 0:
                start = idx
            run += 1
            if run >= min_consecutive:
                return float(time[start])
        else:
            run = 0
            start = None
    return math.nan


def stable_event_timing(
    time: np.ndarray,
    values: np.ndarray,
    significant: np.ndarray,
    *,
    min_consecutive: int = 2,
) -> dict[str, float]:
    """Return the production consecutive-window onset and duration estimands."""
    time_array = np.asarray(time, dtype=float)
    value_array = np.asarray(values, dtype=float)
    significant_array = np.asarray(significant, dtype=bool)
    if (
        time_array.ndim != 1
        or value_array.shape != time_array.shape
        or significant_array.shape != time_array.shape
        or len(time_array) < 1
        or not np.isfinite(time_array).all()
        or not np.isfinite(value_array).all()
        or int(min_consecutive) < 1
    ):
        raise ValueError("Stable event-timing arrays are invalid")
    return {
        "activation_onset": _stable_onset(
            time_array,
            value_array,
            significant_array,
            direction=1,
            min_consecutive=int(min_consecutive),
        ),
        "suppression_onset": _stable_onset(
            time_array,
            value_array,
            significant_array,
            direction=-1,
            min_consecutive=int(min_consecutive),
        ),
        "duration": _center_duration(time_array, significant_array),
    }


def estimate_half_rise_onset(
    time: np.ndarray, values: np.ndarray
) -> float:
    """Estimate adjusted-group onset by the monotone half-rise crossing.

    This is the frozen quantitative shift estimand for T21 adjusted control and
    case curves. Event-call onset remains the consecutive-window estimand above.
    """
    grid = np.asarray(time, dtype=float)
    curve = np.asarray(values, dtype=float)
    if (
        grid.ndim != 1
        or curve.shape != grid.shape
        or len(grid) < 2
        or not np.isfinite(grid).all()
        or not np.isfinite(curve).all()
        or not np.all(np.diff(grid) > 0)
    ):
        return math.nan
    monotone = np.maximum.accumulate(curve)
    edge = max(1, len(curve) // 5)
    low = float(np.mean(monotone[:edge]))
    high = float(np.mean(monotone[-edge:]))
    if high - low <= np.finfo(float).eps:
        return math.nan
    target = low + 0.5 * (high - low)
    reached = np.flatnonzero(monotone >= target)
    if not len(reached):
        return math.nan
    index = int(reached[0])
    if index == 0:
        return float(grid[0])
    x0, x1 = float(grid[index - 1]), float(grid[index])
    y0, y1 = float(monotone[index - 1]), float(monotone[index])
    if y1 <= y0:
        return x1
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))


def _first_direction_switch(time: np.ndarray, values: np.ndarray) -> tuple[float, int]:
    switches = []
    for idx in range(1, len(values)):
        prev = values[idx - 1]
        cur = values[idx]
        if not np.isfinite(prev) or not np.isfinite(cur):
            continue
        if prev == 0 or cur == 0:
            continue
        if np.sign(prev) == np.sign(cur):
            continue

        t0 = time[idx - 1]
        t1 = time[idx]
        frac = abs(prev) / (abs(prev) + abs(cur))
        switches.append(float(t0 + frac * (t1 - t0)))

    if not switches:
        return math.nan, 0
    return switches[0], len(switches)


def _interval_union_length(starts: np.ndarray, ends: np.ndarray) -> float:
    intervals = sorted(
        (float(start), float(end))
        for start, end in zip(starts, ends)
        if np.isfinite(start) and np.isfinite(end) and end >= start
    )
    if not intervals:
        return 0.0

    total = 0.0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    total += cur_end - cur_start
    return float(total)


def _center_duration(time: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    if len(time) < 2:
        return 0.0
    spacing = float(np.nanmedian(np.diff(np.sort(time))))
    return float(mask.sum() * spacing)


def _sharpness(time: np.ndarray, values: np.ndarray, peak_idx: int) -> float:
    if len(time) < 2 or peak_idx < 0:
        return math.nan
    peak = float(values[peak_idx])
    amplitude = abs(peak)
    if amplitude <= 0:
        return math.nan

    total_span = float(np.nanmax(time) - np.nanmin(time))
    if total_span <= 0:
        return math.nan

    same_direction = np.sign(values) == np.sign(peak)
    half_height = same_direction & (np.abs(values) >= amplitude / 2.0)
    width = _center_duration(time, half_height)
    return float(1.0 - min(width / total_span, 1.0))


def _count_prominent_peaks(values: np.ndarray, threshold: float) -> int:
    if len(values) < 3:
        return int(np.nanmax(np.abs(values)) >= threshold)

    count = 0
    for idx in range(1, len(values) - 1):
        cur = values[idx]
        if abs(cur) < threshold:
            continue
        if cur > values[idx - 1] and cur >= values[idx + 1]:
            count += 1
        elif cur < values[idx - 1] and cur <= values[idx + 1]:
            count += 1
    return count


def _time_bin(value: float, t_min: float, t_max: float) -> str:
    if not np.isfinite(value) or t_max <= t_min:
        return "unclear"
    rel = (value - t_min) / (t_max - t_min)
    if rel < 1.0 / 3.0:
        return "early"
    if rel > 2.0 / 3.0:
        return "late"
    return "mid"


def _event_label(
    time: np.ndarray,
    values: np.ndarray,
    significant: np.ndarray,
    activation_onset: float,
    suppression_onset: float,
    peak_time: float,
    trough_time: float,
    duration: float,
    switch_count: int,
    recurrence: int,
) -> str:
    t_min = float(np.nanmin(time))
    t_max = float(np.nanmax(time))
    span = max(t_max - t_min, 1e-12)
    duration_fraction = duration / span
    has_activation = np.isfinite(activation_onset)
    has_suppression = np.isfinite(suppression_onset)

    if recurrence > 1:
        return "recurrent pathway program"

    if switch_count > 0 and has_activation and has_suppression:
        if suppression_onset < activation_onset:
            return "early suppression / late recovery"
        return "biphasic pathway program"

    if has_activation:
        phase = _time_bin(peak_time, t_min, t_max)
        if duration_fraction >= 0.4:
            return f"{phase} sustained activation"
        return f"{phase}-trajectory transient activation"

    if has_suppression:
        phase = _time_bin(trough_time, t_min, t_max)
        if duration_fraction >= 0.4:
            return f"{phase} sustained suppression"
        return f"{phase}-trajectory transient suppression"

    if significant.any():
        dominant = "activation" if abs(values.max()) >= abs(values.min()) else "suppression"
        phase_time = peak_time if dominant == "activation" else trough_time
        phase = _time_bin(phase_time, t_min, t_max)
        return f"{phase} weak {dominant}"

    return "no clear event"


def _event_confidence_class(
    duration: float,
    significant_window_count: int,
    window_fdr_min: float,
    fdr_threshold: float,
    switch_count: int,
    recurrence: int,
) -> tuple[str, str]:
    if not np.isfinite(window_fdr_min) or window_fdr_min > fdr_threshold:
        return "descriptive_only", "no_window_level_support"
    if significant_window_count <= 1 or duration <= 0:
        return "single_window_pulse", "zero_duration_or_single_supported_window"
    if recurrence > 1:
        return "recurrent_multi_window", "multiple_prominent_peaks"
    if switch_count > 0:
        return "switching_multi_window", "direction_switch_detected"
    return "multi_window_event", "supported_by_nonzero_duration"


def summarize_events(
    result: pd.DataFrame,
    pathway_col: Optional[str] = None,
    time_col: str = "pt_mid",
    nes_col: str = "NES",
    fdr_col: Optional[str] = "padj",
    fdr_threshold: float = 0.05,
    nes_threshold: float = 0.0,
    min_consecutive: int = 2,
) -> pd.DataFrame:
    """
    Summarize rolling-window trajectory GSEA curves into pathway-level events.

    Parameters
    ----------
    result
        DataFrame returned by ``run_trajectory_gsea``.
    pathway_col
        Pathway column. If omitted, ``Pathway``/``pathway`` is detected.
    time_col
        Pseudotime coordinate used for event timing.
    nes_col
        NES-like curve column.
    fdr_col
        Window-level FDR column. If unavailable or ``None``, all windows are
        treated as eligible and ``nes_threshold`` controls event support.
    fdr_threshold
        Window-level significance threshold.
    nes_threshold
        Minimum absolute NES considered event-supporting.
    min_consecutive
        Number of consecutive supporting windows required for activation or
        suppression onset.
    """
    if result is None or result.empty:
        return pd.DataFrame()

    pathway_col = pathway_col or _pick_column(result, "Pathway", "pathway")
    if pathway_col is None:
        raise ValueError("Could not find a pathway column")
    if time_col not in result.columns:
        raise ValueError(f"Missing time column '{time_col}'")
    if nes_col not in result.columns:
        raise ValueError(f"Missing NES column '{nes_col}'")
    if min_consecutive <= 0:
        raise ValueError("min_consecutive must be positive")

    start_col = _pick_column(result, "pt_start")
    end_col = _pick_column(result, "pt_end")
    fdr_col = _pick_column(result, fdr_col) if fdr_col else None

    rows = []
    for pathway, group in result.groupby(pathway_col, sort=False):
        group = group.sort_values(time_col)
        time = group[time_col].to_numpy(dtype=np.float64)
        values = group[nes_col].to_numpy(dtype=np.float64)
        finite = np.isfinite(time) & np.isfinite(values)
        if not finite.any():
            continue
        group = group.loc[finite]
        time = time[finite]
        values = values[finite]

        if fdr_col and fdr_col in group.columns:
            fdr = group[fdr_col].to_numpy(dtype=np.float64)
            significant = np.isfinite(fdr) & (fdr <= fdr_threshold)
            window_fdr_min = float(np.nanmin(fdr)) if len(fdr) else math.nan
        else:
            significant = np.ones(len(group), dtype=bool)
            window_fdr_min = math.nan

        if nes_threshold > 0:
            significant = significant & (np.abs(values) >= nes_threshold)
        significant_window_count = int(np.sum(significant))

        peak_idx = int(np.nanargmax(values))
        trough_idx = int(np.nanargmin(values))
        peak_time = float(time[peak_idx])
        trough_time = float(time[trough_idx])
        peak_nes = float(values[peak_idx])
        trough_nes = float(values[trough_idx])

        if start_col and end_col:
            sig_group = group.loc[significant]
            duration = _interval_union_length(
                sig_group[start_col].to_numpy(dtype=np.float64),
                sig_group[end_col].to_numpy(dtype=np.float64),
            )
        else:
            duration = _center_duration(time, significant)

        timing = stable_event_timing(
            time,
            values,
            significant,
            min_consecutive=min_consecutive,
        )
        activation_onset = timing["activation_onset"]
        suppression_onset = timing["suppression_onset"]
        switch_time, switch_count = _first_direction_switch(time, values)

        auc = _trapz(values, time)
        auc_abs = _trapz(np.abs(values), time)
        prominent_threshold = max(float(nes_threshold), abs(peak_nes) * 0.5, abs(trough_nes) * 0.5)
        recurrence = max(_count_prominent_peaks(values, prominent_threshold) - 1, 0)

        label = _event_label(
            time,
            values,
            significant,
            activation_onset,
            suppression_onset,
            peak_time,
            trough_time,
            duration,
            switch_count,
            recurrence,
        )
        confidence_class, confidence_reason = _event_confidence_class(
            duration,
            significant_window_count,
            window_fdr_min,
            fdr_threshold,
            switch_count,
            recurrence,
        )

        rows.append(
            {
                pathway_col: pathway,
                "activation_onset": activation_onset,
                "suppression_onset": suppression_onset,
                "peak_time": peak_time,
                "peak_NES": peak_nes,
                "trough_time": trough_time,
                "trough_NES": trough_nes,
                "duration": duration,
                "AUC": auc,
                "integrated_NES": auc,
                "AUC_abs": auc_abs,
                "sharpness": _sharpness(time, values, peak_idx),
                "direction_switch": switch_time,
                "direction_switch_count": switch_count,
                "recurrence": recurrence,
                "window_fdr_min": window_fdr_min,
                "significant_window_count": significant_window_count,
                "event_window_count": significant_window_count,
                "event_label": label,
                "event_confidence_class": confidence_class,
                "event_confidence_reason": confidence_reason,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(
        ["window_fdr_min", "peak_NES"], ascending=[True, False], na_position="last"
    ).reset_index(drop=True)


summarize_pathway_events = summarize_events
