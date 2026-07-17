from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "ted_submission_calibration"
DEFAULT_OUT = ROOT / "results" / "bib_manuscript_revision" / "packet_bootstrap"
SEED = 20260716


def metric_row(frame: pd.DataFrame) -> dict[str, float]:
    labels = sorted(frame["truth_packet_class"].unique())
    truth_tier = (frame["truth_evidence_tier"] * 2).round().astype(int)
    pred_tier = (frame["ted_evidence_tier"] * 2).round().astype(int)
    return {
        "packet_class_macro_f1": float(
            f1_score(frame["truth_packet_class"], frame["ted_packet_class"], labels=labels, average="macro", zero_division=0)
        ),
        "packet_class_balanced_accuracy": float(balanced_accuracy_score(frame["truth_packet_class"], frame["ted_packet_class"])),
        "legacy_tier_weighted_kappa": float(cohen_kappa_score(truth_tier, pred_tier, weights="quadratic")),
        "legacy_tier_mean_absolute_error": float(np.mean(np.abs(frame["truth_evidence_tier"] - frame["ted_evidence_tier"]))),
        "legacy_tier_false_escalation": float(np.mean(frame["ted_evidence_tier"] > frame["truth_evidence_tier"])),
        "legacy_tier_false_downgrade": float(np.mean(frame["ted_evidence_tier"] < frame["truth_evidence_tier"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified packet bootstrap intervals for the TED controlled-truth benchmark")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    truth = pd.read_csv(args.input / "controlled_truth_key.tsv", sep="\t")
    pred = pd.read_csv(args.input / "ted_packet_predictions.tsv", sep="\t")
    joined = truth.merge(pred, on="packet_id", validate="one_to_one")
    full = metric_row(joined)
    rng = np.random.default_rng(args.seed)
    strata = [group.index.to_numpy() for _, group in joined.groupby("truth_packet_class", sort=True)]
    rows: list[dict[str, float | int]] = []
    for replicate in range(args.replicates):
        sampled = np.concatenate([rng.choice(index, size=len(index), replace=True) for index in strata])
        values = metric_row(joined.loc[sampled])
        values["bootstrap_replicate"] = replicate
        rows.append(values)
    boot = pd.DataFrame(rows)
    summary_rows = []
    for metric, estimate in full.items():
        summary_rows.append(
            {
                "metric": metric,
                "estimate": estimate,
                "ci95_low": float(boot[metric].quantile(0.025)),
                "ci95_high": float(boot[metric].quantile(0.975)),
                "bootstrap_replicates": args.replicates,
                "bootstrap_unit": "packet stratified by controlled truth packet class",
                "qualification": "legacy_tier metrics use the version-1 single-axis truth_evidence_tier and do not validate the revised E/V axes"
                if metric.startswith("legacy_tier")
                else "controlled synthetic packet benchmark",
            }
        )
    boot.to_csv(args.out / "packet_bootstrap_replicates.tsv.gz", sep="\t", index=False, compression="gzip")
    pd.DataFrame(summary_rows).to_csv(args.out / "packet_bootstrap_summary.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "seed": args.seed,
                "bootstrap_replicates": args.replicates,
                "n_packets": len(joined),
                "n_truth_packet_classes": joined["truth_packet_class"].nunique(),
                "sampling": "nonparametric within-truth-packet-class resampling with replacement",
            }
        ]
    ).to_csv(args.out / "packet_bootstrap_run_config.tsv", sep="\t", index=False)
    print(f"Wrote {args.replicates} packet bootstrap replicates to {args.out}")


if __name__ == "__main__":
    main()
