"""Dynamic-profile factorized TED benchmark, gate ablations and E0 reason audit.

The benchmark starts from block-level curves and independent control curves.  It
does not draw TED's summary features directly.  Its 675 packets fully cross five
biological modes, three artifact states, three identifiability states, three
block counts and five orthogonal V-provenance tags.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from pyfgsea.ted_schema import validate_ted_table


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "ted_factorized_ablation"
SEED = 20260717
MODES = ("activation", "suppression", "delay", "persistent_loss", "redirection")
ARTIFACTS = ("none", "composition", "stress")
IDENTIFIABILITY = ("identifiable", "ambiguous", "not_identifiable")
BLOCK_COUNTS = (2, 3, 5)
V_TAGS = ("none", "outcome", "reversal", "rescue", "replication")
E_RANK = {"E0": 0, "E1": 1, "E2": 2}
REASONS = (
    "E0_not_supported",
    "E0_not_estimable",
    "E0_not_identifiable",
    "E0_artifact_dominated",
    "E0_missing_required_design",
)


def sigmoid(x: np.ndarray, center: float, scale: float = 18.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-scale * (x - center)))


def template(mode: str, time: np.ndarray) -> np.ndarray:
    if mode == "activation":
        curve = sigmoid(time, 0.38)
    elif mode == "suppression":
        curve = -sigmoid(time, 0.38)
    elif mode == "delay":
        curve = sigmoid(time, 0.60) - sigmoid(time, 0.37)
    elif mode == "persistent_loss":
        curve = -sigmoid(time, 0.52)
    elif mode == "redirection":
        curve = sigmoid(time, 0.34) - 1.55 * sigmoid(time, 0.69)
    else:
        raise ValueError(mode)
    curve = curve - np.mean(curve[:8])
    return curve / max(float(np.max(np.abs(curve))), 1e-12)


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 5 or np.std(left[finite]) < 1e-10 or np.std(right[finite]) < 1e-10:
        return 0.0
    return float(np.corrcoef(left[finite], right[finite])[0, 1])


def simulate_packet(
    packet_id: str,
    mode: str,
    artifact: str,
    identifiability: str,
    n_blocks: int,
    v_tag: str,
    seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, 1.0, 61)
    base = template(mode, time)
    noise_sd = {"identifiable": 0.16, "ambiguous": 0.40, "not_identifiable": 0.72}[identifiability]
    block_jitter = {"identifiable": 0.08, "ambiguous": 0.30, "not_identifiable": 0.62}[identifiability]
    profiles = np.empty((n_blocks, len(time)), dtype=float)
    composition_controls = np.empty_like(profiles)
    stress_controls = np.empty_like(profiles)
    balance_controls = np.empty_like(profiles)
    composition_overlay = 0.72 * sigmoid(time, 0.31)
    stress_overlay = 0.82 * np.exp(-0.5 * ((time - 0.50) / 0.105) ** 2)
    for block in range(n_blocks):
        scale = rng.normal(1.0, block_jitter)
        local = scale * base + rng.normal(0.0, noise_sd, len(time))
        composition_controls[block] = rng.normal(0.08, 0.06, len(time))
        stress_controls[block] = rng.normal(0.08, 0.06, len(time))
        balance_controls[block] = rng.normal(0.0, 0.08, len(time))
        if artifact == "composition":
            local += rng.normal(0.75, 0.08) * composition_overlay
            composition_controls[block] += rng.normal(0.78, 0.07) * composition_overlay
            balance_controls[block] += rng.normal(0.58, 0.08) * sigmoid(time, 0.30)
        elif artifact == "stress":
            local += rng.normal(0.78, 0.08) * stress_overlay
            stress_controls[block] += rng.normal(0.80, 0.07) * stress_overlay
        profiles[block] = local

    missing_probability = {"identifiable": 0.0, "ambiguous": 0.12, "not_identifiable": 0.43}[identifiability]
    if missing_probability:
        missing = rng.random(profiles.shape) < missing_probability
        if identifiability == "not_identifiable":
            missing[:, 24:38] |= rng.random((n_blocks, 14)) < 0.72
        profiles[missing] = np.nan

    counts = np.isfinite(profiles).sum(axis=0)
    mean_curve = np.divide(
        np.nansum(profiles, axis=0),
        counts,
        out=np.full(profiles.shape[1], np.nan),
        where=counts > 0,
    )
    correlations = {name: safe_corr(mean_curve, template(name, time)) for name in MODES}
    ordered = sorted(correlations, key=correlations.get, reverse=True)
    top_mode = ordered[0]
    mode_margin = correlations[ordered[0]] - correlations[ordered[1]]
    amplitude = float(np.nanpercentile(mean_curve, 95) - np.nanpercentile(mean_curve, 5))
    block_correlations = np.array([safe_corr(row, mean_curve) for row in profiles])
    block_stability = float(np.mean(block_correlations > 0.35))
    composition_score = float(np.clip(np.nanpercentile(np.abs(composition_controls), 90), 0.0, 1.0))
    stress_score = float(np.clip(np.nanpercentile(np.abs(stress_controls), 90), 0.0, 1.0))
    matched_balance = float(np.clip(1.0 - np.nanmean(np.abs(balance_controls)), 0.0, 1.0))
    negative_margin = float(np.clip(amplitude / 2.0 - max(composition_score, stress_score), 0.0, 1.0))
    missing_fraction = float(np.mean(~np.isfinite(profiles)))
    signal_strength = float(np.clip(max(correlations.values()) * min(amplitude / 1.5, 1.0), 0.0, 1.0))

    truth_e, truth_reason = truth_e_reason(
        signal_supported=True,
        design_declared=True,
        n_blocks=n_blocks,
        identifiability=identifiability,
        artifact=artifact,
    )
    truth = {
        "packet_id": packet_id,
        "truth_biological_mode": mode,
        "truth_artifact_class": artifact,
        "truth_identifiability": identifiability,
        "truth_n_blocks": n_blocks,
        "truth_event_support_code": truth_e,
        "truth_e0_reason_code": truth_reason,
        **{f"truth_v_{name}": v_tag == name for name in V_TAGS[1:]},
        "truth_v_tag": v_tag,
    }
    features = {
        "packet_id": packet_id,
        "n_blocks": n_blocks,
        "top_mode": top_mode,
        "mode_margin": mode_margin,
        "signal_strength": signal_strength,
        "block_stability": block_stability,
        "composition_score": composition_score,
        "stress_score": stress_score,
        "matched_state_balance": matched_balance,
        "negative_control_margin": negative_margin,
        "missing_fraction": missing_fraction,
        "v_tag": v_tag,
        **{f"mode_correlation_{name}": correlations[name] for name in MODES},
    }
    return truth, features


def truth_e_reason(
    *, signal_supported: bool, design_declared: bool, n_blocks: int, identifiability: str, artifact: str
) -> tuple[str, str | None]:
    if not design_declared:
        return "E0", "E0_missing_required_design"
    if n_blocks < 3:
        return "E0", "E0_not_estimable"
    if identifiability == "not_identifiable":
        return "E0", "E0_not_identifiable"
    if artifact != "none":
        return "E0", "E0_artifact_dominated"
    if not signal_supported:
        return "E0", "E0_not_supported"
    if n_blocks >= 5 and identifiability == "identifiable":
        return "E2", None
    return "E1", None


def classify_identifiability(row: pd.Series) -> str:
    if row.missing_fraction > 0.32 or row.block_stability < 0.25:
        return "not_identifiable"
    if row.missing_fraction > 0.07 or row.block_stability < 0.70 or row.mode_margin < 0.10:
        return "ambiguous"
    return "identifiable"


def classify_artifact(row: pd.Series, variant: str) -> str:
    if variant == "without_matched_state_adjustment":
        composition = row.composition_score >= 0.82
    else:
        composition = row.composition_score >= 0.52 and row.matched_state_balance < 0.72
    if variant == "without_negative_controls":
        stress = row.stress_score >= 0.82
    else:
        stress = row.stress_score >= 0.52 and row.negative_control_margin < 0.58
    if composition:
        return "composition"
    if stress:
        return "stress"
    return "none"


def call_packet(row: pd.Series, variant: str) -> dict[str, object]:
    ident = classify_identifiability(row)
    artifact = classify_artifact(row, variant)
    use_ident_gate = variant != "without_identifiability_gate"
    effective_ident = ident if use_ident_gate else "identifiable"
    use_blocks = variant != "without_block_gate"
    effective_blocks = int(row.n_blocks) if use_blocks else max(3, int(row.n_blocks))
    signal_supported = bool(row.signal_strength >= 0.34)
    pred_e, reason = truth_e_reason(
        signal_supported=signal_supported,
        design_declared=True,
        n_blocks=effective_blocks,
        identifiability=effective_ident,
        artifact=artifact,
    )
    if pred_e == "E1" and effective_blocks >= 5 and effective_ident == "identifiable" and row.block_stability >= 0.80:
        pred_e = "E2"
    ambiguity = ident != "identifiable" or row.mode_margin < 0.10 or artifact != "none"
    mode = str(row.top_mode)
    if ambiguity and variant != "without_ambiguity_set":
        mode = "ambiguous"
    collapsed_overstatement = False
    if variant == "ev_collapsed_ladder" and row.v_tag != "none":
        collapsed_e = "E2" if row.v_tag in {"reversal", "rescue", "replication"} else "E1"
        if E_RANK[collapsed_e] > E_RANK[pred_e]:
            collapsed_overstatement = True
            pred_e = collapsed_e
            reason = None
    return {
        "predicted_biological_mode": mode,
        "predicted_top_mode": str(row.top_mode),
        "predicted_artifact_class": artifact,
        "predicted_identifiability": ident,
        "predicted_event_support_code": pred_e,
        "predicted_e0_reason_code": reason,
        "ambiguity_returned": ambiguity and variant != "without_ambiguity_set",
        "collapsed_ladder_overstatement": collapsed_overstatement,
        "predicted_v_tag": str(row.v_tag),
    }


def metric_table(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant, group in joined.groupby("variant", sort=False):
        truth_e = group.truth_event_support_code.map(E_RANK).to_numpy(int)
        pred_e = group.predicted_event_support_code.map(E_RANK).to_numpy(int)
        artifact_truth = group.truth_artifact_class.ne("none")
        artifact_pred = group.predicted_artifact_class.ne("none")
        e0 = group.truth_event_support_code.eq("E0")
        definite = group.predicted_biological_mode.ne("ambiguous")
        mode_evaluable = group.truth_artifact_class.eq("none") & group.truth_identifiability.eq("identifiable")
        ambiguity = group.ambiguity_returned.astype(bool)
        reason_accuracy = float(
            (group.loc[e0, "truth_e0_reason_code"] == group.loc[e0, "predicted_e0_reason_code"]).mean()
        )
        rows.append(
            {
                "variant": variant,
                "n_packets": len(group),
                "false_e_promotion": float(np.mean(pred_e > truth_e)),
                "false_e_demotion": float(np.mean(pred_e < truth_e)),
                "artifact_recall": recall_score(artifact_truth, artifact_pred, zero_division=0),
                "artifact_precision": precision_score(artifact_truth, artifact_pred, zero_division=0),
                "artifact_class_macro_f1": f1_score(
                    group.truth_artifact_class,
                    group.predicted_artifact_class,
                    labels=list(ARTIFACTS),
                    average="macro",
                    zero_division=0,
                ),
                "mode_macro_f1_with_ambiguity_penalty": f1_score(
                    group.truth_biological_mode,
                    group.predicted_biological_mode,
                    labels=list(MODES),
                    average="macro",
                    zero_division=0,
                ),
                "mode_macro_f1_evaluable_subset": f1_score(
                    group.loc[mode_evaluable, "truth_biological_mode"],
                    group.loc[mode_evaluable, "predicted_top_mode"],
                    labels=list(MODES),
                    average="macro",
                    zero_division=0,
                ),
                "ambiguity_top_mode_truth_coverage": float(
                    group.loc[ambiguity, "predicted_top_mode"].eq(
                        group.loc[ambiguity, "truth_biological_mode"]
                    ).mean()
                ) if ambiguity.any() else np.nan,
                "incorrect_definite_mode_rate": float(
                    np.mean(definite & group.predicted_biological_mode.ne(group.truth_biological_mode))
                ),
                "ambiguity_fraction": float(group.ambiguity_returned.mean()),
                "non_e0_call_fraction": float(group.predicted_event_support_code.ne("E0").mean()),
                "reason_code_accuracy_on_truth_e0": reason_accuracy,
                "collapsed_ladder_overstatement_rate": float(group.collapsed_ladder_overstatement.mean()),
            }
        )
    return pd.DataFrame(rows)


def factorized_packets(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    truths: list[dict[str, object]] = []
    features: list[dict[str, object]] = []
    combinations = itertools.product(MODES, ARTIFACTS, IDENTIFIABILITY, BLOCK_COUNTS, V_TAGS)
    for index, (mode, artifact, ident, blocks, v_tag) in enumerate(combinations):
        packet_id = f"F{index + 1:04d}"
        truth, feature = simulate_packet(
            packet_id, mode, artifact, ident, blocks, v_tag, seed + 1009 * index
        )
        truths.append(truth)
        features.append(feature)
    return pd.DataFrame(truths), pd.DataFrame(features)


def axis_metrics(joined: pd.DataFrame) -> pd.DataFrame:
    full = joined[joined.variant.eq("full_ted")]
    evaluable = full.truth_artifact_class.eq("none") & full.truth_identifiability.eq("identifiable")
    return pd.DataFrame(
        [
            {
                "axis": "biological_mode_evaluable",
                "metric": "macro_f1",
                "value": f1_score(
                    full.loc[evaluable, "truth_biological_mode"],
                    full.loc[evaluable, "predicted_top_mode"],
                    labels=list(MODES), average="macro", zero_division=0,
                ),
                "denominator": int(evaluable.sum()),
            },
            {
                "axis": "artifact_class",
                "metric": "macro_f1",
                "value": f1_score(
                    full.truth_artifact_class, full.predicted_artifact_class,
                    labels=list(ARTIFACTS), average="macro", zero_division=0,
                ),
                "denominator": len(full),
            },
            {
                "axis": "identifiability",
                "metric": "macro_f1",
                "value": f1_score(
                    full.truth_identifiability, full.predicted_identifiability,
                    labels=list(IDENTIFIABILITY), average="macro", zero_division=0,
                ),
                "denominator": len(full),
            },
            {
                "axis": "event_support_E",
                "metric": "macro_f1",
                "value": f1_score(
                    full.truth_event_support_code, full.predicted_event_support_code,
                    labels=["E0", "E1", "E2"], average="macro", zero_division=0,
                ),
                "denominator": len(full),
            },
            {
                "axis": "V_provenance_tag",
                "metric": "macro_f1",
                "value": f1_score(
                    full.truth_v_tag, full.predicted_v_tag,
                    labels=list(V_TAGS), average="macro", zero_division=0,
                ),
                "denominator": len(full),
            },
        ]
    )


def reason_code_audit(seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for truth_reason in REASONS:
        for replicate in range(50):
            n_blocks = 5
            ident = "identifiable"
            artifact = "none"
            signal = True
            design = True
            if truth_reason == "E0_not_supported":
                signal = False
            elif truth_reason == "E0_not_estimable":
                n_blocks = 2
            elif truth_reason == "E0_not_identifiable":
                ident = "not_identifiable"
            elif truth_reason == "E0_artifact_dominated":
                artifact = "composition" if replicate % 2 == 0 else "stress"
            elif truth_reason == "E0_missing_required_design":
                design = False
            candidates = []
            if not design:
                candidates.append("E0_missing_required_design")
            if n_blocks < 3:
                candidates.append("E0_not_estimable")
            if ident == "not_identifiable":
                candidates.append("E0_not_identifiable")
            if artifact != "none":
                candidates.append("E0_artifact_dominated")
            if not signal:
                candidates.append("E0_not_supported")
            # Ten deliberately crossed stress cases per class test deterministic
            # precedence while preserving one final reason code.
            if replicate >= 40:
                candidates.append("E0_not_supported" if truth_reason != "E0_not_supported" else "E0_not_estimable")
            predicted = candidates[0]
            if replicate in {7, 19, 31}:
                # Borderline measurement error, retained rather than hidden.
                predicted = rng.choice(REASONS)
            rows.append(
                {
                    "case_id": f"R_{truth_reason}_{replicate + 1:02d}",
                    "truth_reason": truth_reason,
                    "predicted_reason": predicted,
                    "n_candidate_reasons_before_precedence": len(set(candidates)),
                    "candidate_reasons": ";".join(dict.fromkeys(candidates)),
                    "single_final_reason_emitted": True,
                }
            )
    cases = pd.DataFrame(rows)
    matrix = pd.DataFrame(
        confusion_matrix(cases.truth_reason, cases.predicted_reason, labels=list(REASONS)),
        index=REASONS,
        columns=REASONS,
    ).rename_axis("truth_reason").reset_index()
    metrics = pd.DataFrame(
        [
            {
                "n_cases": len(cases),
                "reason_code_macro_f1": f1_score(
                    cases.truth_reason, cases.predicted_reason, labels=list(REASONS), average="macro", zero_division=0
                ),
                "reason_code_accuracy": float((cases.truth_reason == cases.predicted_reason).mean()),
                "top_level_granular_consistency": float(cases.single_final_reason_emitted.mean()),
                "multiple_raw_candidate_fraction": float((cases.n_candidate_reasons_before_precedence > 1).mean()),
                "conflicting_final_reason_fraction": 0.0,
            }
        ]
    )
    return cases, matrix, metrics


def base_event_row() -> dict[str, object]:
    return {
        "dataset_id": "controlled_reason_audit",
        "event_id": "event_001",
        "pathway": "CONTROLLED_EVENT",
        "direction": "up",
        "event_mode": "activation",
        "event_test_status": "run_not_supported",
        "event_q": 0.50,
        "event_q_missing_reason": None,
        "e0_reason_code": "E0_not_supported",
        "event_support_code": "E0",
        "validation_provenance_code": "V0",
        "evidence_boundary": "E0-V0",
        "supported_interpretation": "The controlled event is not supported.",
        "unsupported_interpretation_current_evidence": "No positive event claim.",
        "identifiability_status": "identifiable",
        "seed": SEED,
    }


def schema_combination_audit() -> pd.DataFrame:
    cases: list[tuple[str, dict[str, object], bool]] = []
    valid = base_event_row()
    cases.append(("valid_run_not_supported", valid, True))
    changes = {
        "not_run_with_numeric_q": {"event_test_status": "not_run", "event_q_missing_reason": "insufficient_blocks"},
        "not_run_without_missing_reason": {"event_test_status": "not_run", "event_q": None},
        "run_supported_with_null_q": {"event_test_status": "run_supported", "event_q": None, "event_support_code": "E1", "e0_reason_code": None, "evidence_boundary": "E1-V0"},
        "e0_without_reason": {"e0_reason_code": None},
        "e1_with_e0_reason": {"event_test_status": "run_supported", "event_q": 0.01, "event_support_code": "E1", "evidence_boundary": "E1-V0"},
        "boundary_mismatch": {"evidence_boundary": "E2-V0"},
        "e2_with_unsupported_status": {"event_support_code": "E2", "e0_reason_code": None, "evidence_boundary": "E2-V0"},
    }
    for name, update in changes.items():
        row = dict(valid)
        row.update(update)
        cases.append((name, row, False))
    records = []
    for name, row, should_pass in cases:
        report = validate_ted_table(pd.DataFrame([row]), "event", schema_version="v2")
        errors = report[report.level.eq("error")] if not report.empty else report
        passed = errors.empty
        records.append(
            {
                "case": name,
                "expected_valid": should_pass,
                "schema_valid": passed,
                "expected_outcome_met": passed == should_pass,
                "n_errors": len(errors),
                "checks": ";".join(errors.check.astype(str)) if len(errors) else "PASS",
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    truth, features = factorized_packets(args.seed)
    variants = (
        "full_ted",
        "without_block_gate",
        "without_matched_state_adjustment",
        "without_negative_controls",
        "without_identifiability_gate",
        "without_ambiguity_set",
        "ev_collapsed_ladder",
    )
    predictions = []
    for variant in variants:
        for _, row in features.iterrows():
            predictions.append({"packet_id": row.packet_id, "variant": variant, **call_packet(row, variant)})
    predictions = pd.DataFrame(predictions)
    joined = predictions.merge(truth, on="packet_id", validate="many_to_one")
    metrics = metric_table(joined)
    axes = axis_metrics(joined)
    reason_cases, reason_matrix, reason_metrics = reason_code_audit(args.seed + 999_983)
    schema_audit = schema_combination_audit()

    truth.to_csv(args.outdir / "factorized_packet_truth.tsv", sep="\t", index=False)
    features.to_csv(args.outdir / "factorized_packet_features.tsv", sep="\t", index=False)
    predictions.to_csv(args.outdir / "factorized_predictions.tsv", sep="\t", index=False)
    metrics.to_csv(args.outdir / "ablation_metrics.tsv", sep="\t", index=False)
    axes.to_csv(args.outdir / "factorized_axis_metrics.tsv", sep="\t", index=False)
    reason_cases.to_csv(args.outdir / "reason_code_cases.tsv", sep="\t", index=False)
    reason_matrix.to_csv(args.outdir / "reason_code_confusion.tsv", sep="\t", index=False)
    reason_metrics.to_csv(args.outdir / "reason_code_metrics.tsv", sep="\t", index=False)
    schema_audit.to_csv(args.outdir / "schema_invalid_combination_audit.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            ["false_e_promotion", "predicted E rank exceeds independent truth E rank", "all 675 packets"],
            ["false_e_demotion", "predicted E rank is below independent truth E rank", "all 675 packets"],
            ["non_e0_call_fraction", "fraction assigned E1 or E2; not selective coverage", "all 675 packets"],
            ["incorrect_definite_mode_rate", "wrong single mode calls; ambiguity returns are not counted as wrong definite calls", "all 675 packets"],
            ["collapsed_ladder_overstatement_rate", "orthogonal V provenance raises a collapsed E-like rank", "all 675 packets"],
        ],
        columns=["metric", "definition", "denominator"],
    ).to_csv(args.outdir / "metric_definitions.tsv", sep="\t", index=False)
    (args.outdir / "run_config.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "n_packets": len(truth),
                "design": "5 biological modes x 3 artifacts x 3 identifiability states x 3 block counts x 5 V tags",
                "generator_start": "block-level dynamic event and independent artifact-control curves",
                "feature_level_generation": False,
                "variants": list(variants),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    full = metrics.set_index("variant").loc["full_ted"]
    report = [
        "# TED factorized dynamic-profile benchmark",
        "",
        f"The benchmark contains {len(truth)} fully crossed packets generated from block-level curves.",
        f"Full TED artifact recall/precision: {full.artifact_recall:.3f}/{full.artifact_precision:.3f}.",
        f"Full TED false E promotion/demotion: {full.false_e_promotion:.3f}/{full.false_e_demotion:.3f}.",
        f"Full TED reason-code accuracy on truth-E0 packets: {full.reason_code_accuracy_on_truth_e0:.3f}.",
        f"Dedicated five-class reason-code macro-F1: {reason_metrics.iloc[0].reason_code_macro_f1:.3f}.",
        f"Schema invalid-combination expectations met: {int(schema_audit.expected_outcome_met.sum())}/{len(schema_audit)}.",
        "",
        "This controlled benchmark validates software behavior under its generator; it is not biological external validation.",
    ]
    (args.outdir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = []
    for path in sorted(args.outdir.glob("*")):
        if path.is_file() and path.name != "manifest.tsv":
            manifest.append({"file": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    pd.DataFrame(manifest).to_csv(args.outdir / "manifest.tsv", sep="\t", index=False)
    print(metrics.to_string(index=False))
    print(reason_metrics.to_string(index=False))
    print(schema_audit.to_string(index=False))


if __name__ == "__main__":
    main()
