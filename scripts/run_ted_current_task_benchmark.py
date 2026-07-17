from __future__ import annotations

"""Leakage-audited current-task comparison for the TED controlled packets.

This script deliberately calls the final partition a *retrospective shifted audit*.
The packet generator and TED rules pre-date this split, so the partition is useful
for detecting gross overfitting and comparing executable baselines, but it is not
an untouched post-freeze test set.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260716
PACKET_CLASS_ORDER = [
    "activation",
    "suppression",
    "developmental_delay",
    "true_loss",
    "fate_redirection",
    "composition_artifact",
    "stress_dominated",
    "not_identifiable",
    "outcome_supported",
    "reversal_supported",
]
E_ORDER = ["E0", "E1", "E2"]
FEATURES = [
    "n_blocks",
    "curve_effect",
    "peak_shift",
    "terminal_recovery",
    "alternative_fate_gain",
    "composition_score",
    "stress_score",
    "outcome_alignment",
    "reversal_score",
    "block_direction_stability",
    "matched_state_balance",
    "negative_control_margin",
    "missing_evidence_fraction",
]


def _replicate(packet_id: str) -> int:
    return int(packet_id.rsplit("_", 1)[1])


def _partition(replicate: int) -> str:
    if replicate <= 8:
        return "development"
    if replicate <= 12:
        return "tuning"
    return "retrospective_shifted_audit"


def _truth_e(row: pd.Series) -> str:
    if (not bool(row["truth_event_present"])) or bool(row["truth_artifact_sensitive"]):
        return "E0"
    # The controlled truth uses only latent event/artifact class and the generated
    # number of independent blocks. It intentionally does not reuse the noisy
    # stability, balance, control-margin or missingness fields consumed by TED.
    return "E2" if int(row["n_blocks"]) >= 3 else "E1"


def _ted_e(row: pd.Series) -> str:
    if (
        (not bool(row["ted_event_present"]))
        or bool(row["ted_artifact_sensitive"])
        or float(row["ted_identifiability"]) < 0.58
    ):
        return "E0"
    robust = (
        int(row["n_blocks"]) >= 3
        and float(row["block_direction_stability"]) >= 0.80
        and float(row["matched_state_balance"]) >= 0.50
        and float(row["negative_control_margin"]) >= 0.50
        and float(row["missing_evidence_fraction"]) <= 0.50
    )
    return "E2" if robust else "E1"


def _promotion_demotion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    rank = {label: i for i, label in enumerate(E_ORDER)}
    delta = np.asarray([rank[p] - rank[t] for t, p in zip(y_true, y_pred, strict=True)])
    return float(np.mean(delta > 0)), float(np.mean(delta < 0))


def _metrics(task: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if task == "artifact":
        macro = f1_score(y_true, y_pred, average="binary", pos_label=True, zero_division=0)
        balanced = balanced_accuracy_score(y_true, y_pred)
        return {
            "primary_metric": float(macro),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced),
            "false_e_promotion": np.nan,
            "false_e_demotion": np.nan,
            "non_e0_call_fraction": np.nan,
        }
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    balanced = balanced_accuracy_score(y_true, y_pred)
    promotion = demotion = non_e0_call_fraction = np.nan
    if task == "evidence":
        promotion, demotion = _promotion_demotion(y_true, y_pred)
        non_e0_call_fraction = float(np.mean(np.asarray(y_pred) != "E0"))
    return {
        "primary_metric": float(macro),
        "macro_f1": float(macro),
        "balanced_accuracy": float(balanced),
        "false_e_promotion": promotion,
        "false_e_demotion": demotion,
        "non_e0_call_fraction": non_e0_call_fraction,
    }


def _candidates(task: str) -> dict[str, list[tuple[str, object]]]:
    # All candidate grids are fixed here and are evaluated only on the tuning split.
    logistic = [
        (
            f"C={c}",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=c,
                            max_iter=5000,
                            random_state=SEED,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
        )
        for c in (0.1, 1.0, 10.0)
    ]
    forest = [
        (
            f"depth={depth};leaf={leaf}",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=depth,
                min_samples_leaf=leaf,
                random_state=SEED,
                class_weight="balanced_subsample",
                n_jobs=1,
            ),
        )
        for depth in (4, None)
        for leaf in (1, 3)
    ]
    hist = [
        (
            f"lr={rate};leaf={leaves}",
            HistGradientBoostingClassifier(
                learning_rate=rate,
                max_leaf_nodes=leaves,
                max_iter=200,
                random_state=SEED,
            ),
        )
        for rate in (0.05, 0.10)
        for leaves in (15, 31)
    ]
    return {"logistic": logistic, "random_forest": forest, "hist_gradient_boosting": hist}


def _fit_selected(
    task: str,
    x_dev: pd.DataFrame,
    y_dev: np.ndarray,
    x_tune: pd.DataFrame,
    y_tune: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    selected: dict[str, object] = {}
    tuning_rows: list[dict[str, object]] = []
    for family, candidates in _candidates(task).items():
        scored: list[tuple[float, str, object]] = []
        for configuration, model in candidates:
            model.fit(x_dev, y_dev)
            pred = model.predict(x_tune)
            score = _metrics(task, y_tune, pred)["primary_metric"]
            scored.append((score, configuration, model))
            tuning_rows.append(
                {
                    "task": task,
                    "model_family": family,
                    "configuration": configuration,
                    "tuning_primary_metric": score,
                }
            )
        # Lexicographic tie breaking makes selection deterministic.
        score, configuration, _ = sorted(scored, key=lambda item: (-item[0], item[1]))[0]
        chosen = next(model for cfg, model in candidates if cfg == configuration)
        chosen.fit(pd.concat([x_dev, x_tune]), np.concatenate([y_dev, y_tune]))
        selected[family] = chosen
        for row in tuning_rows:
            if row["task"] == task and row["model_family"] == family:
                row["selected"] = row["configuration"] == configuration
                row["selected_tuning_metric"] = score
    return selected, tuning_rows


def _bootstrap_delta(
    task: str,
    y_true: np.ndarray,
    ted_pred: np.ndarray,
    baseline_pred: np.ndarray,
    strata: np.ndarray,
    replicates: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED + {"packet_class": 1, "artifact": 2, "evidence": 3}[task])
    groups = [np.flatnonzero(strata == label) for label in sorted(set(strata.tolist()))]
    labels = sorted(set(y_true.tolist()) | set(ted_pred.tolist()) | set(baseline_pred.tolist()), key=str)

    def primary_fast(truth: np.ndarray, prediction: np.ndarray) -> float:
        if task == "artifact":
            tp = int(np.sum((truth == True) & (prediction == True)))  # noqa: E712
            fp = int(np.sum((truth == False) & (prediction == True)))  # noqa: E712
            fn = int(np.sum((truth == True) & (prediction == False)))  # noqa: E712
            denominator = 2 * tp + fp + fn
            return float(2 * tp / denominator) if denominator else 0.0
        f1_values = []
        for label in labels:
            tp = int(np.sum((truth == label) & (prediction == label)))
            fp = int(np.sum((truth != label) & (prediction == label)))
            fn = int(np.sum((truth == label) & (prediction != label)))
            denominator = 2 * tp + fp + fn
            f1_values.append(2 * tp / denominator if denominator else 0.0)
        return float(np.mean(f1_values))

    deltas: list[float] = []
    for _ in range(replicates):
        sampled = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        ted_value = primary_fast(y_true[sampled], ted_pred[sampled])
        base_value = primary_fast(y_true[sampled], baseline_pred[sampled])
        deltas.append(ted_value - base_value)
    point = primary_fast(y_true, ted_pred) - primary_fast(y_true, baseline_pred)
    low, high = np.quantile(np.asarray(deltas), [0.025, 0.975])
    return float(point), float(low), float(high)


def run(input_dir: Path, output_dir: Path, bootstrap_replicates: int) -> None:
    packets = pd.read_csv(input_dir / "controlled_packet_features.tsv", sep="\t")
    truth = pd.read_csv(input_dir / "controlled_truth_key.tsv", sep="\t")
    ted = pd.read_csv(input_dir / "ted_packet_predictions.tsv", sep="\t")
    data = packets.merge(truth, on="packet_id", validate="one_to_one").merge(ted, on="packet_id", validate="one_to_one")
    data["replicate"] = data["packet_id"].map(_replicate)
    data["partition"] = data["replicate"].map(_partition)
    data["truth_E"] = data.apply(_truth_e, axis=1)
    data["ted_E"] = data.apply(_ted_e, axis=1)

    development = data["partition"] == "development"
    tuning = data["partition"] == "tuning"
    audit = data["partition"] == "retrospective_shifted_audit"
    x_dev, x_tune, x_audit = data.loc[development, FEATURES], data.loc[tuning, FEATURES], data.loc[audit, FEATURES]

    targets = {
        "packet_class": (data["truth_packet_class"].to_numpy(), data["ted_packet_class"].to_numpy()),
        "artifact": (data["truth_artifact_sensitive"].astype(bool).to_numpy(), data["ted_artifact_sensitive"].astype(bool).to_numpy()),
        "evidence": (data["truth_E"].to_numpy(), data["ted_E"].to_numpy()),
    }
    result_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    confusion_frames: list[pd.DataFrame] = []

    for task, (truth_all, ted_all) in targets.items():
        selected, local_tuning = _fit_selected(task, x_dev, truth_all[development], x_tune, truth_all[tuning])
        tuning_rows.extend(local_tuning)
        y_true = truth_all[audit]
        predictions: dict[str, np.ndarray] = {"TED_fixed_rules": ted_all[audit]}
        dummy = DummyClassifier(strategy="most_frequent", random_state=SEED)
        dummy.fit(pd.concat([x_dev, x_tune]), np.concatenate([truth_all[development], truth_all[tuning]]))
        predictions["dummy_most_frequent"] = dummy.predict(x_audit)
        for family, model in selected.items():
            predictions[family] = model.predict(x_audit)

        tune_best = max(
            selected,
            key=lambda family: max(
                row["selected_tuning_metric"]
                for row in local_tuning
                if row["model_family"] == family and row.get("selected")
            ),
        )
        for method, prediction in predictions.items():
            values = _metrics(task, y_true, prediction)
            result_rows.append(
                {
                    "task": task,
                    "method": method,
                    "audit_n": len(y_true),
                    "auditable_reason_codes": method == "TED_fixed_rules",
                    **values,
                }
            )
            for packet_id, truth_value, pred_value in zip(data.loc[audit, "packet_id"], y_true, prediction, strict=True):
                prediction_rows.append(
                    {"packet_id": packet_id, "task": task, "method": method, "truth": truth_value, "prediction": pred_value}
                )
            labels = PACKET_CLASS_ORDER if task == "packet_class" else ([False, True] if task == "artifact" else E_ORDER)
            matrix = confusion_matrix(y_true, prediction, labels=labels)
            confusion_frames.append(
                pd.DataFrame(matrix, index=labels, columns=labels)
                .rename_axis("truth")
                .reset_index()
                .melt(id_vars="truth", var_name="prediction", value_name="count")
                .assign(task=task, method=method)
            )
        point, low, high = _bootstrap_delta(
            task,
            y_true,
            predictions["TED_fixed_rules"],
            predictions[tune_best],
            data.loc[audit, "truth_packet_class"].to_numpy(),
            bootstrap_replicates,
        )
        delta_rows.append(
            {
                "task": task,
                "comparison": f"TED_fixed_rules-minus-{tune_best}",
                "paired_primary_metric_delta": point,
                "bootstrap_95ci_low": low,
                "bootstrap_95ci_high": high,
                "bootstrap_replicates": bootstrap_replicates,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "current_task_packet_partitions.tsv", sep="\t", index=False)
    pd.DataFrame(tuning_rows).to_csv(output_dir / "baseline_tuning_audit.tsv", sep="\t", index=False)
    pd.DataFrame(result_rows).to_csv(output_dir / "current_task_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(delta_rows).to_csv(output_dir / "paired_deltas.tsv", sep="\t", index=False)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "audit_predictions.tsv", sep="\t", index=False)
    pd.concat(confusion_frames, ignore_index=True).to_csv(output_dir / "current_task_confusions.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "metric": "false_e_promotion",
                "definition": "predicted E rank exceeds truth E rank; includes E0->E1, E0->E2, and E1->E2",
                "denominator": "all shifted-audit packets",
            },
            {
                "metric": "false_e_demotion",
                "definition": "predicted E rank is below truth E rank; includes E2->E1, E2->E0, and E1->E0",
                "denominator": "all shifted-audit packets",
            },
            {
                "metric": "non_e0_call_fraction",
                "definition": "fraction assigned E1 or E2; this is not selective-prediction coverage",
                "denominator": "all shifted-audit packets",
            },
        ]
    ).to_csv(output_dir / "e_metric_definitions.tsv", sep="\t", index=False)
    split_rows = []
    for name, mask in (("development", development), ("tuning", tuning), ("retrospective_shifted_audit", audit)):
        split_rows.append(
            {
                "partition": name,
                "n_packets": int(mask.sum()),
                "replicate_range": f"{int(data.loc[mask, 'replicate'].min())}-{int(data.loc[mask, 'replicate'].max())}",
                "truth_used_for_threshold_or_model_selection": name != "retrospective_shifted_audit",
                "outcome_masked_during_prediction": name == "retrospective_shifted_audit",
                "post_freeze": False,
                "interpretation": (
                    "model development" if name == "development" else "hyperparameter selection" if name == "tuning" else "retrospective OOD diagnostic; not an untouched final test"
                ),
            }
        )
    pd.DataFrame(split_rows).to_csv(output_dir / "split_and_leakage_audit.tsv", sep="\t", index=False)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "bootstrap_replicates": bootstrap_replicates,
                "features": FEATURES,
                "warning": "Retrospective shifted audit only; generator and TED rules pre-date this split, so this is not an untouched post-freeze test.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("results/ted_submission_calibration"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/ted_current_task_benchmark"))
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args()
    run(args.input_dir, args.output_dir, args.bootstrap_replicates)


if __name__ == "__main__":
    main()
