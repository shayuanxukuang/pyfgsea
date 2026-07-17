from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_zscape_leave_one_embryo_full_refit as zscape


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = (
    ROOT
    / "results"
    / "ted_submission_supplement"
    / "zscape_repeated_holdout_stability"
)
SEED = 20260716
Q_THRESHOLDS = (0.01, 0.025, 0.05, 0.10)
SUBSAMPLING_FRACTIONS = (0.20, 0.40, 0.60, 0.80, 0.90)


def aggregate_subset(
    abundance: pd.DataFrame,
    template: pd.DataFrame,
    kept_embryos: set[str],
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    subset = abundance[abundance["embryo"].astype(str).isin(kept_embryos)].copy()
    subset["cell_type_fraction_sq"] = np.square(
        subset["cell_type_fraction"].to_numpy(dtype=float)
    )
    controls = subset[subset["is_control"].astype(bool)]
    mutants = subset[~subset["is_control"].astype(bool)]

    mutant = (
        mutants.groupby(
            ["gene_target", "timepoint", "cell_type_broad"], observed=True
        )
        .agg(
            mut_n=("cell_type_fraction", "size"),
            mut_sum=("cell_type_fraction", "sum"),
            mut_sumsq=("cell_type_fraction_sq", "sum"),
        )
        .reset_index()
    )
    control = (
        controls.groupby(["timepoint", "cell_type_broad"], observed=True)
        .agg(
            ctrl_n=("cell_type_fraction", "size"),
            ctrl_sum=("cell_type_fraction", "sum"),
            ctrl_sumsq=("cell_type_fraction_sq", "sum"),
        )
        .reset_index()
    )
    keys = template[["gene_target", "timepoint", "cell_type_broad"]]
    sufficient = keys.merge(
        mutant,
        on=["gene_target", "timepoint", "cell_type_broad"],
        how="left",
    ).merge(control, on=["timepoint", "cell_type_broad"], how="left")
    for column in [
        "mut_n",
        "mut_sum",
        "mut_sumsq",
        "ctrl_n",
        "ctrl_sum",
        "ctrl_sumsq",
    ]:
        sufficient[column] = pd.to_numeric(
            sufficient[column], errors="coerce"
        ).fillna(0.0)

    mean_mut, mean_ctrl, effect, p = zscape.welch_from_sufficient(
        sufficient["mut_n"].to_numpy(dtype=float),
        sufficient["mut_sum"].to_numpy(dtype=float),
        sufficient["mut_sumsq"].to_numpy(dtype=float),
        sufficient["ctrl_n"].to_numpy(dtype=float),
        sufficient["ctrl_sum"].to_numpy(dtype=float),
        sufficient["ctrl_sumsq"].to_numpy(dtype=float),
    )
    fit = {
        "mean_mut": mean_mut,
        "mean_ctrl": mean_ctrl,
        "effect": effect,
        "p": p,
        "q": zscape.bh(p),
    }
    return fit, sufficient


def stratified_holdout(
    units: pd.DataFrame,
    rng: np.random.Generator,
    fraction: float,
) -> set[str]:
    strata = ["is_control", "gene_target", "timepoint"]
    work = units.copy().reset_index(drop=True)
    work["stratum"] = work[strata].astype(str).agg("|".join, axis=1)
    sizes = work.groupby("stratum", observed=True).size().to_dict()
    capacities = {key: max(0, size - 2) for key, size in sizes.items()}
    target = int(round(fraction * len(work)))
    held: set[str] = set()
    held_per_stratum = {key: 0 for key in sizes}
    # Sample globally to hit the requested fraction exactly, while enforcing
    # the per-stratum constraint that at least two units remain estimable.
    for index in rng.permutation(len(work)):
        row = work.iloc[int(index)]
        key = str(row["stratum"])
        if held_per_stratum[key] >= capacities[key]:
            continue
        held.add(str(row["embryo"]))
        held_per_stratum[key] += 1
        if len(held) == target:
            break
    if len(held) != target:
        raise RuntimeError(
            f"Could not draw requested stratified holdout: {len(held)} != {target}"
        )
    return held


def stratified_halves(
    units: pd.DataFrame, rng: np.random.Generator
) -> tuple[set[str], set[str]]:
    first: set[str] = set()
    second: set[str] = set()
    strata = ["is_control", "gene_target", "timepoint"]
    groups = list(units.groupby(strata, observed=True, sort=False))
    odd_group_indices = [index for index, (_, group) in enumerate(groups) if len(group) % 2]
    # Assign the unmatched unit in odd-sized strata so that the two global
    # halves are equal (or differ by one when the total is odd).
    n_extra_first = len(odd_group_indices) // 2
    extra_first = set(
        rng.choice(odd_group_indices, size=n_extra_first, replace=False).tolist()
    ) if n_extra_first else set()
    for group_index, (_, group) in enumerate(groups):
        embryos = rng.permutation(group["embryo"].astype(str).to_numpy())
        cut = len(embryos) // 2 + int(group_index in extra_first)
        first.update(embryos[:cut].tolist())
        second.update(embryos[cut:].tolist())
    return first, second


def stratified_subsample(
    units: pd.DataFrame, rng: np.random.Generator, fraction: float
) -> set[str]:
    """Keep a stratified fraction, retaining at least two units when available."""
    kept: set[str] = set()
    strata = ["is_control", "gene_target", "timepoint"]
    for _, group in units.groupby(strata, observed=True, sort=False):
        embryos = rng.permutation(group["embryo"].astype(str).to_numpy())
        n_keep = min(len(embryos), max(min(2, len(embryos)), int(round(fraction * len(embryos)))))
        kept.update(embryos[:n_keep].tolist())
    return kept


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = int((left | right).sum())
    return float((left & right).sum() / union) if union else 1.0


def fit_comparison(
    events: pd.DataFrame,
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    groups: dict[tuple[str, str], np.ndarray],
    q_threshold: float,
) -> dict[str, float | int]:
    reference_called = np.isfinite(reference["q"]) & (
        reference["q"] <= q_threshold
    )
    candidate_called = np.isfinite(candidate["q"]) & (
        candidate["q"] <= q_threshold
    )
    selected_groups = set(
        zip(
            events.loc[reference_called, "gene_target"].astype(str),
            events.loc[reference_called, "cell_type_broad"].astype(str),
            strict=False,
        )
    )
    reference_modes = zscape.classify_groups(
        events, reference, groups, selected_groups=selected_groups
    )
    modes = zscape.classify_groups(
        events, candidate, groups, selected_groups=selected_groups
    )
    common = sorted(set(reference_modes) & set(modes))
    finite = reference_called & np.isfinite(candidate["effect"])
    return {
        "n_reference_calls": int(reference_called.sum()),
        "n_candidate_calls": int(candidate_called.sum()),
        "event_jaccard": jaccard(reference_called, candidate_called),
        "direction_agreement": (
            float(
                np.mean(
                    np.sign(reference["effect"][finite])
                    == np.sign(candidate["effect"][finite])
                )
            )
            if finite.any()
            else np.nan
        ),
        "event_mode_agreement": (
            float(np.mean([reference_modes[key] == modes[key] for key in common]))
            if common
            else np.nan
        ),
    }


def paired_comparison(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray], q_threshold: float
) -> dict[str, float | int]:
    left_called = np.isfinite(left["q"]) & (left["q"] <= q_threshold)
    right_called = np.isfinite(right["q"]) & (right["q"] <= q_threshold)
    common = left_called & right_called
    finite = np.isfinite(left["effect"]) & np.isfinite(right["effect"])
    return {
        "n_left_calls": int(left_called.sum()),
        "n_right_calls": int(right_called.sum()),
        "event_jaccard": jaccard(left_called, right_called),
        "direction_agreement_on_common_calls": (
            float(
                np.mean(
                    np.sign(left["effect"][common])
                    == np.sign(right["effect"][common])
                )
            )
            if common.any()
            else np.nan
        ),
        "effect_spearman_all_estimable": (
            float(pd.Series(left["effect"][finite]).corr(pd.Series(right["effect"][finite]), method="spearman"))
            if finite.sum() > 2
            else np.nan
        ),
    }


def quantile_summary(frame: pd.DataFrame, column: str) -> dict[str, float]:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return {
        f"{column}_median": float(values.median()),
        f"{column}_iqr_low": float(values.quantile(0.25)),
        f"{column}_iqr_high": float(values.quantile(0.75)),
        f"{column}_minimum": float(values.min()),
    }


def resample_event_rows(
    events: pd.DataFrame,
    fit: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    *,
    scheme: str,
    repeat: int,
    side: str,
) -> pd.DataFrame:
    finite = np.isfinite(fit["effect"])
    ranks = pd.Series(np.abs(fit["effect"])).rank(pct=True).to_numpy(float)
    keys = events[["gene_target", "timepoint", "cell_type_broad"]].astype(str).agg("|".join, axis=1)
    return pd.DataFrame(
        {
            "event_key": keys,
            "gene_target": events["gene_target"].astype(str),
            "timepoint": events["timepoint"],
            "cell_type_broad": events["cell_type_broad"].astype(str),
            "scheme": scheme,
            "repeat": repeat,
            "side": side,
            "selected": np.isfinite(fit["q"]) & (fit["q"] <= 0.05),
            "selected_in_full": np.isfinite(reference["q"])
            & (reference["q"] <= 0.05),
            "effect": fit["effect"],
            "effect_rank_percentile": np.where(finite, ranks, np.nan),
            "direction_matches_full": np.where(
                finite & np.isfinite(reference["effect"]),
                np.sign(fit["effect"]) == np.sign(reference["effect"]),
                np.nan,
            ),
        }
    )


def summarize_event_stability(long: pd.DataFrame) -> pd.DataFrame:
    identity = ["event_key", "gene_target", "timepoint", "cell_type_broad"]
    rows: list[dict[str, object]] = []
    for keys, group in long.groupby(identity, sort=False, dropna=False):
        holdout = group[group.scheme.eq("holdout")]
        split = group[group.scheme.eq("split_half")]
        holdout_frequency = float(holdout.selected.mean()) if len(holdout) else np.nan
        split_frequency = float(split.selected.mean()) if len(split) else np.nan
        combined_frequency = float(np.nanmin([holdout_frequency, split_frequency]))
        ranks = pd.to_numeric(group.effect_rank_percentile, errors="coerce").dropna()
        directions = pd.to_numeric(
            group.direction_matches_full, errors="coerce"
        ).dropna()
        rank_stability = float(1.0 - (ranks.quantile(0.75) - ranks.quantile(0.25))) if len(ranks) else np.nan
        if combined_frequency >= 0.80:
            status = "stable_core"
        elif combined_frequency >= 0.50:
            status = "intermediate"
        else:
            status = "unstable"
        rows.append(
            {
                **dict(zip(identity, keys, strict=True)),
                "selected_in_full_fit": bool(group.selected_in_full.iloc[0]),
                "selection_frequency_20pct_holdout": holdout_frequency,
                "selection_frequency_split_half": split_frequency,
                "resampling_selection_frequency": combined_frequency,
                "effect_direction_frequency": (
                    float(directions.mean()) if len(directions) else np.nan
                ),
                "median_effect": float(pd.to_numeric(group.effect, errors="coerce").median()),
                "effect_rank_stability": rank_stability,
                "discovery_stability_status": status,
            }
        )
    return pd.DataFrame(rows)


def run(
    holdout_repeats: int,
    split_repeats: int,
    holdout_fraction: float,
    subsampling_repeats: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    abundance = pd.read_csv(zscape.ABUNDANCE, sep="\t")
    abundance["embryo"] = abundance["embryo"].astype(str)
    abundance["cell_type_fraction"] = pd.to_numeric(
        abundance["cell_type_fraction"], errors="coerce"
    )
    events, context = zscape.build_sufficient(abundance)
    full = zscape.fit_arrays(events)
    groups = zscape.group_indices(events)
    units = abundance[
        ["embryo", "is_control", "gene_target", "timepoint"]
    ].drop_duplicates()
    all_embryos = set(units["embryo"].astype(str))
    rng = np.random.default_rng(seed)

    primary_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    event_resamples: list[pd.DataFrame] = []
    for repeat in range(holdout_repeats):
        held = stratified_holdout(units, rng, holdout_fraction)
        candidate, sufficient = aggregate_subset(abundance, events, all_embryos - held)
        for q_threshold in Q_THRESHOLDS:
            metrics = fit_comparison(
                events,
                full,
                candidate,
                groups,
                q_threshold,
            )
            row = {
                "repeat": repeat + 1,
                "q_threshold": q_threshold,
                "n_embryos_held_out": len(held),
                "held_out_fraction": len(held) / len(all_embryos),
                "minimum_mutant_n": float(sufficient["mut_n"].min()),
                "minimum_control_n": float(sufficient["ctrl_n"].min()),
                **metrics,
            }
            threshold_rows.append(row)
            if q_threshold == 0.05:
                primary_rows.append(row)
                event_resamples.append(
                    resample_event_rows(
                        events,
                        candidate,
                        full,
                        scheme="holdout",
                        repeat=repeat + 1,
                        side="kept_80pct",
                    )
                )

    split_rows: list[dict[str, object]] = []
    for repeat in range(split_repeats):
        left_units, right_units = stratified_halves(units, rng)
        left, _ = aggregate_subset(abundance, events, left_units)
        right, _ = aggregate_subset(abundance, events, right_units)
        split_rows.append(
            {
                "repeat": repeat + 1,
                "q_threshold": 0.05,
                "n_left_embryos": len(left_units),
                "n_right_embryos": len(right_units),
                **paired_comparison(left, right, 0.05),
            }
        )
        event_resamples.append(
            resample_event_rows(events, left, full, scheme="split_half", repeat=repeat + 1, side="left")
        )
        event_resamples.append(
            resample_event_rows(events, right, full, scheme="split_half", repeat=repeat + 1, side="right")
        )

    subsampling_rows: list[dict[str, object]] = []
    for fraction in SUBSAMPLING_FRACTIONS:
        for repeat in range(subsampling_repeats):
            kept = stratified_subsample(units, rng, fraction)
            candidate, sufficient = aggregate_subset(abundance, events, kept)
            finite = np.isfinite(candidate["effect"]) & np.isfinite(full["effect"])
            eligible = (
                np.isfinite(candidate["q"])
                & (candidate["q"] <= 0.05)
                & (sufficient["mut_n"].to_numpy(float) >= 3)
                & (sufficient["ctrl_n"].to_numpy(float) >= 3)
            )
            # The subsampling curve needs event-set overlap and effect ranks, not
            # the comparatively expensive group-level mode reclassification.
            # Holdout mode stability is computed above with the full routine.
            reference_called = np.isfinite(full["q"]) & (full["q"] <= 0.05)
            candidate_called = np.isfinite(candidate["q"]) & (candidate["q"] <= 0.05)
            subsampling_rows.append(
                {
                    "target_retained_fraction": fraction,
                    "repeat": repeat + 1,
                    "actual_retained_fraction": len(kept) / len(all_embryos),
                    "event_jaccard": jaccard(reference_called, candidate_called),
                    "effect_spearman": float(
                        pd.Series(full["effect"][finite]).corr(
                            pd.Series(candidate["effect"][finite]), method="spearman"
                        )
                    ) if finite.sum() > 2 else np.nan,
                    "n_e2_eligible_calls": int(eligible.sum()),
                }
            )

    primary = pd.DataFrame(primary_rows)
    threshold_long = pd.DataFrame(threshold_rows)
    threshold = (
        threshold_long.groupby("q_threshold", as_index=False)
        .agg(
            n_repeats=("repeat", "size"),
            median_event_jaccard=("event_jaccard", "median"),
            iqr_low_event_jaccard=("event_jaccard", lambda x: x.quantile(0.25)),
            iqr_high_event_jaccard=("event_jaccard", lambda x: x.quantile(0.75)),
            minimum_event_jaccard=("event_jaccard", "min"),
            median_direction_agreement=("direction_agreement", "median"),
            median_event_mode_agreement=("event_mode_agreement", "median"),
        )
    )
    split = pd.DataFrame(split_rows)
    event_long = pd.concat(event_resamples, ignore_index=True)
    event_stability = summarize_event_stability(event_long)
    subsampling_long = pd.DataFrame(subsampling_rows)
    subsampling_summary = (
        subsampling_long.groupby("target_retained_fraction", as_index=False)
        .agg(
            n_repeats=("repeat", "size"),
            median_actual_retained_fraction=("actual_retained_fraction", "median"),
            median_event_jaccard=("event_jaccard", "median"),
            iqr_low_event_jaccard=("event_jaccard", lambda x: x.quantile(0.25)),
            iqr_high_event_jaccard=("event_jaccard", lambda x: x.quantile(0.75)),
            median_effect_spearman=("effect_spearman", "median"),
            median_e2_eligible_calls=("n_e2_eligible_calls", "median"),
        )
    )
    summary_row: dict[str, object] = {
        "n_embryos": len(all_embryos),
        "holdout_repeats": holdout_repeats,
        "target_holdout_fraction": holdout_fraction,
        "median_actual_holdout_fraction": float(primary["held_out_fraction"].median()),
        "split_half_repeats": split_repeats,
        "subsampling_repeats_per_fraction": subsampling_repeats,
        "analysis_unit": "embryo",
        "inference_recomputed": "Welch event tests;BH event-FDR;event modes",
    }
    for column in ["event_jaccard", "direction_agreement", "event_mode_agreement"]:
        summary_row.update(quantile_summary(primary, column))
    for column in [
        "event_jaccard",
        "direction_agreement_on_common_calls",
        "effect_spearman_all_estimable",
    ]:
        for key, value in quantile_summary(split, column).items():
            summary_row[f"split_half_{key}"] = value
    return {
        "repeated_20pct_holdout_metrics": primary,
        "threshold_sensitivity": threshold,
        "threshold_sensitivity_long": threshold_long,
        "split_half_metrics": split,
        "event_resampling_long": event_long,
        "event_selection_frequency": event_stability,
        "subsampling_curve_long": subsampling_long,
        "subsampling_curve": subsampling_summary,
        "summary": pd.DataFrame([summary_row]),
    }


def write_manifest(outdir: Path) -> None:
    manifest = []
    for path in sorted(outdir.glob("*")):
        if path.is_file() and path.name != "manifest.tsv":
            manifest.append(
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    pd.DataFrame(manifest).to_csv(outdir / "manifest.tsv", sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repeated embryo-level holdout and split-half stability for ZSCAPE"
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--holdout-repeats", type=int, default=100)
    parser.add_argument("--split-repeats", type=int, default=50)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--subsampling-repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Recompute the output manifest without rerunning resampling.",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.manifest_only:
        write_manifest(args.outdir)
        print(f"Updated manifest: {(args.outdir / 'manifest.tsv').resolve()}")
        return
    started = time.perf_counter()
    outputs = run(
        args.holdout_repeats,
        args.split_repeats,
        args.holdout_fraction,
        args.subsampling_repeats,
        args.seed,
    )
    for name, table in outputs.items():
        if name == "event_resampling_long":
            # Keep every event-resample row but partition the long table so the
            # submission archive stays below its per-file evidence limit.
            for scheme, scheme_table in table.groupby("scheme", sort=True):
                maximum = int(scheme_table["repeat"].max())
                for start in range(1, maximum + 1, 10):
                    end = min(start + 9, maximum)
                    part = scheme_table[scheme_table["repeat"].between(start, end)]
                    part.to_csv(
                        args.outdir
                        / f"event_resampling_{scheme}_repeats_{start:03d}_{end:03d}.tsv.gz",
                        sep="\t",
                        index=False,
                        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
                    )
        else:
            table.to_csv(args.outdir / f"{name}.tsv", sep="\t", index=False)
    elapsed = time.perf_counter() - started
    summary = outputs["summary"].iloc[0]
    report = [
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: run + validate",
        "- Origin Date: 2026-07-16",
        "- Verification Status: ANALYZED",
        "- Version Label: zscape_repeated_holdout_stability_v1",
        "",
        "# ZSCAPE repeated holdout stability",
        "",
        f"Repeated stratified 20% holdouts: {int(summary['holdout_repeats'])}; "
        f"median actual held-out fraction: {summary['median_actual_holdout_fraction']:.3f}.",
        f"Median event Jaccard: {summary['event_jaccard_median']:.3f} "
        f"(IQR {summary['event_jaccard_iqr_low']:.3f}-{summary['event_jaccard_iqr_high']:.3f}; "
        f"minimum {summary['event_jaccard_minimum']:.3f}).",
        f"Median direction agreement: {summary['direction_agreement_median']:.3f}; "
        f"minimum {summary['direction_agreement_minimum']:.3f}; median event-mode agreement: "
        f"{summary['event_mode_agreement_median']:.3f} (IQR "
        f"{summary['event_mode_agreement_iqr_low']:.3f}-{summary['event_mode_agreement_iqr_high']:.3f}; "
        f"minimum {summary['event_mode_agreement_minimum']:.3f}).",
        f"Repeated stratified split halves: {int(summary['split_half_repeats'])}; "
        f"median event Jaccard: {summary['split_half_event_jaccard_median']:.3f} "
        f"(IQR {summary['split_half_event_jaccard_iqr_low']:.3f}-"
        f"{summary['split_half_event_jaccard_iqr_high']:.3f}; minimum "
        f"{summary['split_half_event_jaccard_minimum']:.3f}).",
        f"Split-half common-call direction agreement median: "
        f"{summary['split_half_direction_agreement_on_common_calls_median']:.3f}; "
        f"minimum {summary['split_half_direction_agreement_on_common_calls_minimum']:.3f}. "
        f"All-estimable effect Spearman median: "
        f"{summary['split_half_effect_spearman_all_estimable_median']:.3f} "
        f"(IQR {summary['split_half_effect_spearman_all_estimable_iqr_low']:.3f}-"
        f"{summary['split_half_effect_spearman_all_estimable_iqr_high']:.3f}; minimum "
        f"{summary['split_half_effect_spearman_all_estimable_minimum']:.3f}).",
        "All inference starts from embryo-level abundance; cells are not treated as independent replicates.",
        "This is a retrospective no-retuning stability audit, not a prospective external validation.",
        f"Per-event selection frequencies and a five-point subsampling curve used {args.subsampling_repeats} repeats per retained fraction.",
    ]
    (args.outdir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    config = {
        "seed": args.seed,
        "holdout_repeats": args.holdout_repeats,
        "split_repeats": args.split_repeats,
        "target_holdout_fraction": args.holdout_fraction,
        "subsampling_repeats_per_fraction": args.subsampling_repeats,
        "subsampling_retained_fractions": list(SUBSAMPLING_FRACTIONS),
        "q_thresholds": list(Q_THRESHOLDS),
        "source": str(zscape.ABUNDANCE),
        "runtime_seconds": elapsed,
        "prospective_holdout_claimed": False,
    }
    (args.outdir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    write_manifest(args.outdir)
    print(outputs["summary"].to_string(index=False))
    print(f"Completed in {elapsed:.1f}s: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
