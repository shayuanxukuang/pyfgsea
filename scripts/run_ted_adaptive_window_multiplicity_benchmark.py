from __future__ import annotations

"""End-to-end adaptive-window multiplicity benchmark for TED.

The benchmark starts from block-level dynamic profiles, scans candidate windows,
repeats the complete scan under paired-block sign permutations, and compares:

1. a naive p value that holds the observed selected window fixed;
2. the primary per-event max-window p value followed by BH across events; and
3. a family-wide maxT FWER p value with no subsequent BH.

It is a controlled simulation, not evidence of external biological validity.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pyfgsea.calibration import calibrate_selected_window_statistics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "ted_adaptive_window_multiplicity"
SEED = 20260716
TARGET_Q = 0.10
MODES = ("activation", "suppression", "delay", "loss", "redirection")
ARTIFACTS = ("none", "composition", "stress", "batch_time", "missing_intermediates")
EFFECTS = {"null": 0.0, "weak": 0.45, "moderate": 0.85, "strong": 1.30}


def bh(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def scenario_registry() -> pd.DataFrame:
    """Return a balanced 36-scenario matrix covering every requested level."""
    blocks = (3, 5, 10)
    pathways = (20, 50, 100)
    windows = (10, 25, 50)
    correlations = (0.0, 0.3, 0.7)
    effects = tuple(EFFECTS)
    rows: list[dict[str, object]] = []
    for i in range(36):
        rows.append(
            {
                "scenario_id": f"AW{i + 1:02d}",
                "n_blocks": blocks[i % 3],
                "n_pathways": pathways[(i // 3) % 3],
                "n_windows": windows[(i // 9) % 3],
                "pathway_correlation": correlations[(i * 2 + i // 3) % 3],
                "effect_size": effects[i % 4],
                "artifact": (
                    "none" if i in {0, 4, 20} else ARTIFACTS[(i // 2) % len(ARTIFACTS)]
                ),
                "n_perm": 5000 if i in {5, 11, 17, 23, 29, 35} else 1000,
                "target_q": TARGET_Q,
                "signal_modes": ";".join(MODES),
            }
        )
    frame = pd.DataFrame(rows)
    for column, required in {
        "n_blocks": blocks,
        "n_pathways": pathways,
        "n_windows": windows,
        "pathway_correlation": correlations,
        "effect_size": effects,
        "artifact": ARTIFACTS,
    }.items():
        if set(required) - set(frame[column]):
            raise AssertionError(f"Scenario registry misses levels for {column}")
    return frame


def window_matrix(n_windows: int, n_time: int = 60) -> tuple[np.ndarray, np.ndarray]:
    widths = np.resize(np.array([5, 7, 9, 11], dtype=int), n_windows)
    centers = np.linspace(0.08, 0.92, n_windows)
    matrix = np.zeros((n_windows, n_time), dtype=float)
    center_time = np.empty(n_windows, dtype=float)
    for index, (center, width) in enumerate(zip(centers, widths, strict=True)):
        middle = int(round(center * (n_time - 1)))
        start = max(0, min(n_time - int(width), middle - int(width) // 2))
        stop = start + int(width)
        matrix[index, start:stop] = 1.0 / float(width)
        center_time[index] = (start + stop - 1) / (2.0 * (n_time - 1))
    return matrix, center_time


def mode_template(mode: str, time: np.ndarray) -> np.ndarray:
    sigmoid = lambda x, center, scale=22.0: 1.0 / (1.0 + np.exp(-scale * (x - center)))
    if mode == "activation":
        values = sigmoid(time, 0.35)
    elif mode == "suppression":
        values = -sigmoid(time, 0.35)
    elif mode == "delay":
        values = sigmoid(time, 0.55) - sigmoid(time, 0.35)
    elif mode == "loss":
        values = -sigmoid(time, 0.52)
    elif mode == "redirection":
        values = sigmoid(time, 0.35) - 1.65 * sigmoid(time, 0.68)
    else:
        raise ValueError(mode)
    maximum = np.max(np.abs(values))
    return values / maximum if maximum else values


def timing_truth(template: np.ndarray, time: np.ndarray) -> tuple[float, float]:
    magnitude = np.abs(template)
    peak = float(time[int(np.argmax(magnitude))])
    threshold = 0.2 * float(np.max(magnitude))
    onset = float(time[int(np.flatnonzero(magnitude >= threshold)[0])])
    return onset, peak


def simulate_profiles(
    row: pd.Series, rng: np.random.Generator
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    n_blocks = int(row.n_blocks)
    n_pathways = int(row.n_pathways)
    rho = float(row.pathway_correlation)
    effect = float(EFFECTS[str(row.effect_size)])
    n_time = 60
    time = np.linspace(0.0, 1.0, n_time)
    common = rng.normal(size=(n_blocks, 1, n_time))
    individual = rng.normal(size=(n_blocks, n_pathways, n_time))
    noise = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * individual
    noise = 0.45 * noise
    profiles = noise + rng.normal(0.0, 0.12, size=(n_blocks, n_pathways, 1))

    truth_signal = np.zeros(n_pathways, dtype=bool)
    truth_mode = np.full(n_pathways, "null", dtype=object)
    truth_onset = np.full(n_pathways, np.nan)
    truth_peak = np.full(n_pathways, np.nan)
    if effect > 0:
        n_signal = max(5, int(round(0.25 * n_pathways)))
        truth_signal[:n_signal] = True
        for pathway in range(n_signal):
            mode = MODES[pathway % len(MODES)]
            template = mode_template(mode, time)
            block_scale = rng.normal(1.0, 0.12, size=(n_blocks, 1))
            profiles[:, pathway, :] += effect * block_scale * template[None, :]
            truth_mode[pathway] = mode
            truth_onset[pathway], truth_peak[pathway] = timing_truth(template, time)

    null_indices = np.flatnonzero(~truth_signal)
    artifact = str(row.artifact)
    n_artifact = (
        max(1, int(round(0.15 * len(null_indices))))
        if len(null_indices) and artifact != "none"
        else 0
    )
    artifact_indices = null_indices[:n_artifact]
    artifact_score = np.clip(rng.normal(0.10, 0.08, n_pathways), 0.0, 1.0)
    missing_fraction = np.zeros(n_pathways, dtype=float)
    if n_artifact:
        if artifact == "composition":
            template = mode_template("activation", time)
            profiles[:, artifact_indices, :] += 0.75 * template
            artifact_score[artifact_indices] = rng.normal(0.88, 0.05, n_artifact)
        elif artifact == "stress":
            template = np.exp(-0.5 * ((time - 0.50) / 0.10) ** 2)
            profiles[:, artifact_indices, :] += 0.85 * template
            artifact_score[artifact_indices] = rng.normal(0.84, 0.06, n_artifact)
        elif artifact == "batch_time":
            template = np.linspace(-1.0, 1.0, n_time)
            signs = rng.choice([-1.0, 1.0], size=(n_blocks, n_artifact, 1))
            profiles[:, artifact_indices, :] += 0.95 * signs * template
            artifact_score[artifact_indices] = rng.normal(0.72, 0.10, n_artifact)
        elif artifact == "missing_intermediates":
            profiles[:, artifact_indices, 22:38] = np.nan
            missing_fraction[artifact_indices] = 16.0 / n_time
            artifact_score[artifact_indices] = rng.normal(0.78, 0.08, n_artifact)

    truth = pd.DataFrame(
        {
            "event_id": [f"pathway_{i:03d}" for i in range(n_pathways)],
            "truth_signal": truth_signal,
            "truth_mode": truth_mode,
            "truth_artifact": np.isin(np.arange(n_pathways), artifact_indices),
            "truth_onset": truth_onset,
            "truth_peak": truth_peak,
        }
    )
    return profiles, truth, time, np.clip(artifact_score, 0.0, 1.0), missing_fraction


def block_window_statistics(profiles: np.ndarray, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(profiles).astype(float)
    filled = np.nan_to_num(profiles, nan=0.0)
    sums = np.einsum("bpt,wt->bpw", filled, windows > 0, optimize=True)
    counts = np.einsum("bpt,wt->bpw", valid, windows > 0, optimize=True)
    block_means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    finite = np.isfinite(block_means)
    n_finite = np.sum(finite, axis=0)
    mean = np.divide(
        np.nansum(block_means, axis=0),
        n_finite,
        out=np.zeros(block_means.shape[1:], dtype=float),
        where=n_finite > 0,
    )
    centered = np.where(finite, block_means - mean[None, :, :], 0.0)
    variance = np.divide(
        np.sum(centered**2, axis=0),
        np.maximum(n_finite - 1, 1),
        out=np.zeros_like(mean),
        where=n_finite > 1,
    )
    se = np.sqrt(variance / np.maximum(n_finite, 1))
    statistic = np.divide(mean, se, out=np.zeros_like(mean), where=se > 1e-12)
    return block_means, statistic


def permuted_statistics(
    block_means: np.ndarray, n_perm: int, rng: np.random.Generator
) -> np.ndarray:
    n_blocks = block_means.shape[0]
    filled = np.nan_to_num(block_means, nan=0.0)
    valid = np.isfinite(block_means).astype(float)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n_blocks))
    signed_sum = np.einsum("kb,bpw->kpw", signs, filled, optimize=True)
    counts = np.sum(valid, axis=0)
    mean = np.divide(signed_sum, counts[None, :, :], out=np.zeros_like(signed_sum), where=counts[None, :, :] > 0)
    sumsq = np.einsum("bpw->pw", filled**2, optimize=True)
    variance_numerator = sumsq[None, :, :] - counts[None, :, :] * mean**2
    variance = np.divide(
        variance_numerator,
        np.maximum(counts - 1.0, 1.0)[None, :, :],
        out=np.zeros_like(mean),
        where=counts[None, :, :] > 1,
    )
    se = np.sqrt(np.maximum(variance, 0.0) / np.maximum(counts, 1.0)[None, :, :])
    return np.divide(mean, se, out=np.zeros_like(mean), where=se > 1e-12)


def evidence_truth(truth: pd.DataFrame, effect_size: str, n_blocks: int) -> np.ndarray:
    result = np.full(len(truth), "E0", dtype=object)
    eligible = truth["truth_signal"].to_numpy() & ~truth["truth_artifact"].to_numpy()
    result[eligible] = "E1"
    if effect_size in {"moderate", "strong"} and n_blocks >= 5:
        result[eligible] = "E2"
    return result


def evidence_calls(
    calls: np.ndarray,
    truth: pd.DataFrame,
    selected_effect: np.ndarray,
    block_means: np.ndarray,
    selected_window: np.ndarray,
    artifact_score: np.ndarray,
    missing_fraction: np.ndarray,
    n_blocks: int,
    exact_resolution: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e = np.full(len(calls), "E0", dtype=object)
    test_status = np.full(len(calls), "run_not_supported", dtype=object)
    missing_reason = np.full(len(calls), None, dtype=object)
    if exact_resolution > TARGET_Q:
        test_status[:] = "not_run"
        missing_reason[:] = "insufficient_permutation_resolution"
        return e, test_status, missing_reason
    gated = calls & (artifact_score < 0.70) & (missing_fraction <= 0.20)
    e[gated] = "E1"
    stability = np.zeros(len(calls), dtype=float)
    for event in range(len(calls)):
        values = block_means[:, event, int(selected_window[event])]
        values = values[np.isfinite(values)]
        if len(values) and selected_effect[event] != 0:
            stability[event] = np.mean(np.sign(values) == np.sign(selected_effect[event]))
    e[gated & (n_blocks >= 5) & (stability >= 0.80)] = "E2"
    test_status[e != "E0"] = "run_supported"
    return e, test_status, missing_reason


def ordinal_error(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    rank = {"E0": 0, "E1": 1, "E2": 2}
    delta = np.array([rank[p] - rank[t] for t, p in zip(truth, prediction, strict=True)])
    return float(np.mean(delta > 0)), float(np.mean(delta < 0))


def method_metrics(
    truth_signal: np.ndarray,
    calls: np.ndarray,
    selected_window: np.ndarray,
    window_centers: np.ndarray,
    truth: pd.DataFrame,
    truth_e: np.ndarray,
    pred_e: np.ndarray,
) -> dict[str, float]:
    false = calls & ~truth_signal
    true = calls & truth_signal
    n_called = int(np.sum(calls))
    n_signal = int(np.sum(truth_signal))
    fdp = float(np.sum(false) / n_called) if n_called else 0.0
    power = float(np.sum(true) / n_signal) if n_signal else np.nan
    timing = true & truth["truth_onset"].notna().to_numpy()
    if np.any(timing):
        selected_time = window_centers[selected_window[timing]]
        onset_mae = float(np.mean(np.abs(selected_time - truth.loc[timing, "truth_onset"].to_numpy(float))))
        peak_mae = float(np.mean(np.abs(selected_time - truth.loc[timing, "truth_peak"].to_numpy(float))))
    else:
        onset_mae = np.nan
        peak_mae = np.nan
    promotion, demotion = ordinal_error(truth_e, pred_e)
    return {
        "fdp": fdp,
        "family_wise_false_positive": float(np.any(false)),
        "power": power,
        "onset_location_mae": onset_mae,
        "peak_location_mae": peak_mae,
        "false_e_promotion": promotion,
        "false_e_demotion": demotion,
        "non_e0_call_fraction": float(np.mean(pred_e != "E0")),
        "n_calls": n_called,
        "n_false_calls": int(np.sum(false)),
        "n_true_calls": int(np.sum(true)),
    }


def run_replicate(row: pd.Series, replicate: int, seed: int) -> tuple[list[dict[str, object]], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    profiles, truth, _time, artifact_score, missing_fraction = simulate_profiles(row, rng)
    windows, centers = window_matrix(int(row.n_windows), profiles.shape[2])
    block_means, observed = block_window_statistics(profiles, windows)
    permuted = permuted_statistics(block_means, int(row.n_perm), rng)
    calibration = calibrate_selected_window_statistics(
        observed,
        permuted,
        event_ids=truth["event_id"].tolist(),
    )
    selected = calibration["selected_window_index"].to_numpy(int)
    selected_effect = observed[np.arange(len(truth)), selected]
    exact_resolution = 1.0 / (2 ** int(row.n_blocks))
    truth_e = evidence_truth(truth, str(row.effect_size), int(row.n_blocks))
    methods = {
        "naive_selected_window_bh": bh(calibration["naive_selected_window_p"].to_numpy()) <= TARGET_Q,
        "per_event_max_window_bh": calibration["event_q"].to_numpy() <= TARGET_Q,
        "family_wide_maxT_fwer": calibration["family_fwer_p"].to_numpy() <= TARGET_Q,
    }
    metric_rows: list[dict[str, object]] = []
    event_frames: list[pd.DataFrame] = []
    for method, calls in methods.items():
        pred_e, status, missing_reason = evidence_calls(
            calls,
            truth,
            selected_effect,
            block_means,
            selected,
            artifact_score,
            missing_fraction,
            int(row.n_blocks),
            exact_resolution,
        )
        metric_rows.append(
            {
                **row.to_dict(),
                "replicate": replicate,
                "seed": seed,
                "method": method,
                "exact_sign_permutation_resolution": exact_resolution,
                **method_metrics(
                    truth["truth_signal"].to_numpy(),
                    calls,
                    selected,
                    centers,
                    truth,
                    truth_e,
                    pred_e,
                ),
            }
        )
        local = truth.copy()
        local["scenario_id"] = row.scenario_id
        local["replicate"] = replicate
        local["method"] = method
        local["called"] = calls
        local["truth_E"] = truth_e
        local["predicted_E"] = pred_e
        local["event_test_status"] = status
        local["event_q_missing_reason"] = missing_reason
        local["event_q"] = (
            calibration["event_q"].to_numpy()
            if method == "per_event_max_window_bh"
            else np.where(method == "naive_selected_window_bh", bh(calibration["naive_selected_window_p"].to_numpy()), np.nan)
        )
        local["family_fwer_p"] = calibration["family_fwer_p"].to_numpy()
        local["selected_window_index"] = selected
        local["selected_window_center"] = centers[selected]
        event_frames.append(local)
    return metric_rows, pd.concat(event_frames, ignore_index=True)


def ci(values: pd.Series) -> tuple[float, float, float]:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if not len(finite):
        return np.nan, np.nan, np.nan
    mean = float(np.mean(finite))
    if len(finite) == 1:
        return mean, mean, mean
    se = float(np.std(finite, ddof=1) / np.sqrt(len(finite)))
    return mean, max(0.0, mean - 1.96 * se), min(1.0, mean + 1.96 * se)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, group in metrics.groupby("method", sort=False):
        row: dict[str, object] = {"method": method, "n_scenario_replicates": len(group)}
        for metric in (
            "fdp",
            "family_wise_false_positive",
            "power",
            "onset_location_mae",
            "peak_location_mae",
            "false_e_promotion",
            "false_e_demotion",
            "non_e0_call_fraction",
        ):
            mean, low, high = ci(group[metric])
            row[f"mean_{metric}"] = mean
            row[f"ci95_low_{metric}"] = low
            row[f"ci95_high_{metric}"] = high
        row["worst_scenario_mean_fdp"] = float(group.groupby("scenario_id")["fdp"].mean().max())
        row["worst_scenario_fwer"] = float(group.groupby("scenario_id")["family_wise_false_positive"].mean().max())
        rows.append(row)
    return pd.DataFrame(rows)


def stratified_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    strata = {
        "all_scenarios": np.ones(len(metrics), dtype=bool),
        "clean_all": metrics["artifact"].eq("none").to_numpy(),
        "clean_null": (
            metrics["artifact"].eq("none") & metrics["effect_size"].eq("null")
        ).to_numpy(),
        "clean_signal": (
            metrics["artifact"].eq("none") & ~metrics["effect_size"].eq("null")
        ).to_numpy(),
        "artifact_stress": ~metrics["artifact"].eq("none").to_numpy(),
    }
    frames: list[pd.DataFrame] = []
    for name, mask in strata.items():
        local = metrics.loc[mask]
        if local.empty:
            continue
        table = summarize(local)
        table.insert(0, "analysis_stratum", name)
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repetitions", type=int, default=12)
    parser.add_argument("--clean-null-repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--scenario-limit", type=int, default=None)
    args = parser.parse_args()
    if args.repetitions < 1 or args.clean_null_repetitions < args.repetitions:
        raise ValueError("repetitions must be positive and clean-null repetitions cannot be smaller")
    registry = scenario_registry()
    if args.scenario_limit is not None:
        registry = registry.iloc[: args.scenario_limit].copy()
    args.out.mkdir(parents=True, exist_ok=True)
    registry.to_csv(args.out / "scenario_registry.tsv", sep="\t", index=False)

    metric_rows: list[dict[str, object]] = []
    event_frames: list[pd.DataFrame] = []
    for scenario_index, (_, row) in enumerate(registry.iterrows()):
        local_repetitions = (
            args.clean_null_repetitions
            if str(row.effect_size) == "null" and str(row.artifact) == "none"
            else args.repetitions
        )
        for replicate in range(local_repetitions):
            seed = int(args.seed + scenario_index * 100_003 + replicate * 1_009)
            local_metrics, local_events = run_replicate(row, replicate, seed)
            metric_rows.extend(local_metrics)
            event_frames.append(local_events)
        print(f"completed {row.scenario_id}: {local_repetitions} replicates x {int(row.n_perm)} permutations", flush=True)

    metrics = pd.DataFrame(metric_rows)
    events = pd.concat(event_frames, ignore_index=True)
    summary = summarize(metrics)
    stratified = stratified_summary(metrics)
    metrics.to_csv(args.out / "replicate_metrics.tsv", sep="\t", index=False)
    events.to_csv(args.out / "event_call_audit.tsv.gz", sep="\t", index=False, compression="gzip")
    summary.to_csv(args.out / "method_summary.tsv", sep="\t", index=False)
    stratified.to_csv(args.out / "method_summary_by_stratum.tsv", sep="\t", index=False)
    factor_rows: list[pd.DataFrame] = []
    for factor in (
        "n_blocks",
        "n_pathways",
        "n_windows",
        "pathway_correlation",
        "effect_size",
        "artifact",
    ):
        grouped = (
            metrics.groupby(["method", factor], dropna=False)
            .agg(
                mean_fdp=("fdp", "mean"),
                empirical_fwer=("family_wise_false_positive", "mean"),
                mean_power=("power", "mean"),
                mean_false_e_promotion=("false_e_promotion", "mean"),
                mean_false_e_demotion=("false_e_demotion", "mean"),
                mean_non_e0_call_fraction=("non_e0_call_fraction", "mean"),
                n=("replicate", "size"),
            )
            .reset_index()
            .rename(columns={factor: "factor_level"})
        )
        grouped.insert(1, "factor", factor)
        factor_rows.append(grouped)
    pd.concat(factor_rows, ignore_index=True).to_csv(
        args.out / "factor_level_summary.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        [
            ["fdp", "false calls / all calls; zero when no calls", "each scenario replicate"],
            ["family_wise_false_positive", "indicator of one or more false calls", "each scenario replicate"],
            ["power", "true calls / simulated true events", "true events; null scenarios excluded from mean"],
            ["false_e_promotion", "predicted E rank > truth E rank, including E0->E2 and E1->E2", "all events"],
            ["false_e_demotion", "predicted E rank < truth E rank, including E2->E0/E1 and E1->E0", "all events"],
            ["non_e0_call_fraction", "fraction assigned E1 or E2; not selective-prediction coverage", "all events"],
        ],
        columns=["metric", "definition", "denominator"],
    ).to_csv(args.out / "metric_definitions.tsv", sep="\t", index=False)
    (args.out / "run_config.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "repetitions": args.repetitions,
                "clean_null_repetitions": args.clean_null_repetitions,
                "scenario_count": len(registry),
                "target_q": TARGET_Q,
                "primary_method": "per_event_max_window_bh",
                "critical_scenario_permutations": 5000,
                "other_scenario_permutations": 1000,
                "note": "Balanced coverage matrix, not the 6,480-cell full factorial. Every requested factor level is represented.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
