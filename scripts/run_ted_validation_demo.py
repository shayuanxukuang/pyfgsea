from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from pyfgsea.ted_schema import ted_table_is_valid, validate_ted_table


SEED = 20260715


def bh(values: pd.Series) -> np.ndarray:
    p = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.clip(ranked, 0, 1)
    return out


def make_activity() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    modes = {
        "ERYTHROID_ACTIVATION": "activation",
        "INFLAMMATION_REVERSAL": "reversal_pattern",
        "STRESS_CONTROL": "stress_dominated",
        "NULL_PATHWAY": "not_identifiable",
    }
    for block in range(1, 7):
        block_shift = rng.normal(0, 0.08)
        for time in np.linspace(0, 1, 12):
            for pathway, mode in modes.items():
                if mode == "activation":
                    signal = 1.4 / (1 + np.exp(-10 * (time - 0.55)))
                elif mode == "reversal_pattern":
                    signal = 1.2 * np.exp(-((time - 0.45) / 0.18) ** 2) - 0.45 * time
                elif mode == "stress_dominated":
                    signal = 0.45 * np.sin(4 * np.pi * time) + 0.4 * (block % 2)
                else:
                    signal = 0.0
                rows.append(
                    {
                        "dataset_id": "TED_VALIDATION_DEMO_V2",
                        "block_id": f"B{block}",
                        "condition": "demo",
                        "trajectory": "known_time",
                        "time": float(time),
                        "pathway": pathway,
                        "activity": float(signal + block_shift + rng.normal(0, 0.12)),
                        "weight": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def call_events(activity: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pathway, sub in activity.groupby("pathway", sort=True):
        block_effects = []
        for _, block in sub.groupby("block_id"):
            early = block.loc[block["time"] <= 0.30, "activity"].mean()
            late = block.loc[block["time"] >= 0.70, "activity"].mean()
            block_effects.append(float(late - early))
        effect = float(np.mean(block_effects))
        test = stats.ttest_1samp(block_effects, 0.0)
        stability = float(np.mean(np.sign(block_effects) == np.sign(effect))) if effect else 0.0
        if pathway == "INFLAMMATION_REVERSAL":
            event_mode = "reversal_pattern"
        elif pathway == "STRESS_CONTROL":
            event_mode = "stress_dominated"
        elif abs(effect) < 0.15:
            event_mode = "not_identifiable"
        else:
            event_mode = "activation" if effect > 0 else "suppression"
        rows.append(
            {
                "dataset_id": "TED_VALIDATION_DEMO_V2",
                "event_id": f"demo::{pathway}",
                "pathway": pathway,
                "direction": "up" if effect > 0 else "down",
                "event_mode": event_mode,
                "effect_size": effect,
                "event_p": float(test.pvalue),
                "event_q": 1.0,
                "ambiguity_set": event_mode if stability >= 0.8 else f"{event_mode};not_identifiable",
                "matched_functional_rescue": False,
                "block_direction_stability": stability,
                "seed": SEED,
            }
        )
    events = pd.DataFrame(rows)
    events["event_q"] = bh(events["event_p"])
    support_codes: list[str] = []
    e0_reasons: list[str | None] = []
    for row in events.itertuples(index=False):
        artifact_dominated = row.event_mode in {"stress_dominated", "not_identifiable"}
        if artifact_dominated or row.event_q > 0.05:
            support_codes.append("E0")
            e0_reasons.append(
                "E0_not_identifiable"
                if row.event_mode == "not_identifiable"
                else "E0_artifact_dominated"
                if row.event_mode == "stress_dominated"
                else "E0_not_supported"
            )
        elif row.block_direction_stability >= 0.8 and abs(row.effect_size) >= 0.15:
            support_codes.append("E2")
            e0_reasons.append(None)
        else:
            support_codes.append("E1")
            e0_reasons.append(None)
    events["event_support_code"] = support_codes
    events["e0_reason_code"] = e0_reasons
    events["event_test_status"] = np.where(
        events["event_support_code"].eq("E0"), "run_not_supported", "run_supported"
    )
    events["event_q_missing_reason"] = None
    events["validation_provenance_code"] = "V0"
    events["evidence_boundary"] = events["event_support_code"] + "–V0"
    events["identifiability_status"] = events["event_support_code"].map({"E0": "limited", "E1": "limited", "E2": "identifiable"})
    events.loc[events["event_mode"].eq("not_identifiable"), "identifiability_status"] = "not_identifiable"
    events.loc[events["event_mode"].eq("not_identifiable"), "direction"] = "not_identifiable"
    events["supported_interpretation"] = events["event_support_code"].map(
        {
            "E0": "No event interpretation is supported beyond the recorded E0 reason under the available design.",
            "E1": "A statistically supported synthetic event without robust mode identification.",
            "E2": "A block-robust, mode-identifiable synthetic event.",
        }
    )
    events["unsupported_interpretation_current_evidence"] = events[
        "event_support_code"
    ].map(
        {
            "E0": "A supported dynamic event or any external validation.",
            "E1": "Block-robust mode identity or any external validation.",
            "E2": "Orthogonal outcome, intervention reversal, matched rescue, or independent replication.",
        }
    )

    # Deprecated v1 fields remain populated so downstream transition code can
    # read the v2 table while migrating to the orthogonal E/V contract.
    events["evidence_tier"] = events["event_support_code"].map(
        {"E0": 1.0, "E1": 2.0, "E2": 3.0}
    )
    events["claim_ceiling"] = events["event_support_code"].map(
        {
            "E0": "Level 1 descriptive trend",
            "E1": "Level 2 statistically supported event",
            "E2": "Level 3 block-robust event",
        }
    )
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic TED validation demo")
    parser.add_argument("--outdir", type=Path, default=Path("results/ted_validation_demo"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    activity = make_activity()
    events = call_events(activity)
    activity_report = validate_ted_table(activity, "activity")
    event_report = validate_ted_table(events, "event", schema_version="v2")
    if not ted_table_is_valid(activity_report) or not ted_table_is_valid(event_report):
        raise RuntimeError("Demo output failed TED schema validation")
    activity.to_csv(args.outdir / "demo_activity.tsv", sep="\t", index=False)
    events.to_csv(args.outdir / "demo_events_v2.tsv", sep="\t", index=False)
    # Keep the original filename as a byte-equivalent compatibility alias.
    events.to_csv(args.outdir / "demo_events.tsv", sep="\t", index=False)
    pd.concat(
        [activity_report.assign(table="activity"), event_report.assign(table="event")],
        ignore_index=True,
    ).to_csv(args.outdir / "demo_validation.tsv", sep="\t", index=False)
    html = "<h1>TED validation demo (event schema v2)</h1>" + events.to_html(
        index=False, float_format=lambda x: f"{x:.4g}"
    )
    (args.outdir / "demo_report.html").write_text(html, encoding="utf-8")
    (args.outdir / "demo_manifest.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "event_schema": "ted_event_report_v2",
                "input": "demo_activity.tsv",
                "event_report": "demo_events_v2.tsv",
                "compatibility_alias": "demo_events.tsv",
                "html_report": "demo_report.html",
                "evidence_boundary": "Synthetic demo evidence is computational only (V0); no external validation is implied.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"TED validation demo complete: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
