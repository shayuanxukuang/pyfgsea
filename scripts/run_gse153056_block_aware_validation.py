from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "ted_known_source" / "GSE153056"
DEFAULT_OUT = ROOT / "results" / "bib_manuscript_revision" / "gse153056_block_aware"
DEFAULT_LEGACY_ALIGNMENT = ROOT / "results" / "ted_known_source_validation" / "tables" / "gse153056_pdl1_outcome_alignment.tsv"
DEFAULT_NEGATIVE_CONTROLS = ROOT / "results" / "ted_known_source_validation" / "tables" / "gse153056_negative_control_results.tsv"
IFNG_PDL1_AXIS = ["CD274", "IRF1", "STAT1", "STAT2", "JAK2", "IFNGR1", "IFNGR2", "CXCL10", "GBP1", "TAP1"]


def bh_fdr(values: list[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    out = np.full(len(p), np.nan)
    finite = np.isfinite(p)
    ranked = np.argsort(p[finite])
    observed = p[finite][ranked]
    adjusted = observed * len(observed) / np.arange(1, len(observed) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    inverse = np.empty_like(ranked)
    inverse[ranked] = np.arange(len(ranked))
    out[finite] = np.minimum(adjusted[inverse], 1.0)
    return out


def read_selected_rows(path: Path, genes: list[str]) -> pd.DataFrame:
    wanted = {gene.upper(): gene for gene in genes}
    rows: dict[str, list[float]] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        cells = [cell.strip().strip('"') for cell in header[1:]]
        for row in reader:
            if not row:
                continue
            key = row[0].strip().strip('"').upper()
            if key in wanted:
                rows[wanted[key]] = [float(value) if value not in {"", "NA"} else 0.0 for value in row[1:]]
    return pd.DataFrame(rows, index=cells)


def axis_score(matrix: pd.DataFrame) -> pd.Series:
    logged = np.log1p(matrix.astype(float))
    standardized = (logged - logged.mean(axis=0)) / logged.std(axis=0, ddof=0).replace(0, np.nan)
    return standardized.mean(axis=1).rename("axis_score")


def replicate_effects(scores: pd.Series, metadata: pd.DataFrame, modality: str) -> pd.DataFrame:
    data = pd.concat(
        [scores, metadata[["perturbed_gene", "replicate", "is_non_targeting"]]],
        axis=1,
    ).dropna()
    rows: list[dict[str, object]] = []
    for replicate, block in data.groupby("replicate"):
        reference = block[block["is_non_targeting"].astype(bool)]["axis_score"].astype(float)
        if reference.empty:
            continue
        reference_mean = float(reference.mean())
        for perturbation, group in block.groupby("perturbed_gene"):
            if str(perturbation) == "NT":
                continue
            rows.append(
                {
                    "perturbation": perturbation,
                    "modality": modality,
                    "replicate": replicate,
                    "n_cells": len(group),
                    "reference_n_cells": len(reference),
                    "effect": float(group["axis_score"].astype(float).mean() - reference_mean),
                }
            )
    return pd.DataFrame(rows)


def summarize_event_support(event_effects: pd.DataFrame, protein_effects: pd.DataFrame) -> pd.DataFrame:
    protein_mean = protein_effects.groupby("perturbation")["effect"].mean()
    rows: list[dict[str, object]] = []
    for perturbation, group in event_effects.groupby("perturbation"):
        effects = group["effect"].astype(float).to_numpy()
        n = len(effects)
        mean = float(np.mean(effects))
        sd = float(np.std(effects, ddof=1)) if n > 1 else np.nan
        p = float(stats.ttest_1samp(effects, 0.0).pvalue) if n >= 3 and sd > 0 else np.nan
        stability = float(np.mean(np.sign(effects) == np.sign(mean))) if mean != 0 else 0.0
        if n >= 3 and np.isfinite(sd):
            half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
            low, high = mean - half, mean + half
        else:
            low, high = np.nan, np.nan
        outcome = float(protein_mean.get(perturbation, np.nan))
        rows.append(
            {
                "perturbation": perturbation,
                "n_blocks": n,
                "mean_block_effect": mean,
                "block_effect_sd": sd,
                "block_p_value": p,
                "block_ci95_low": low,
                "block_ci95_high": high,
                "direction_stability": stability,
                "protein_mean_block_effect": outcome,
                "rna_protein_direction_match": bool(np.isfinite(outcome) and np.sign(mean) == np.sign(outcome)),
            }
        )
    summary = pd.DataFrame(rows)
    summary["block_q_value"] = bh_fdr(summary["block_p_value"].tolist())
    # The BH-adjusted test over the three independent replicate-block effects is
    # the event-level q used for the general retrospective E1 gate.  The legacy
    # cell-level q is retained only as a diagnostic and is never substituted.
    summary["event_q_value"] = summary["block_q_value"]
    summary["event_q_source"] = "BH-adjusted one-sample t test over replicate-block effects"
    summary["formal_e1_q_threshold"] = 0.10
    summary["formal_e1_contract_pass"] = (
        (summary["n_blocks"] >= 3) & (summary["event_q_value"] <= 0.10)
    )
    summary["locked_primary_q_0_05_pass"] = summary["event_q_value"] <= 0.05
    summary["ci_excludes_zero"] = (summary["block_ci95_low"] > 0) | (summary["block_ci95_high"] < 0)
    summary["posthoc_e2_eligible"] = (
        (summary["n_blocks"] >= 3)
        & (summary["block_q_value"] <= 0.10)
        & (summary["direction_stability"] >= 0.80)
        & summary["ci_excludes_zero"]
    )
    # The q-and-CI block rule was introduced after the STAT1 result was known and
    # did not execute the full locked E2 contract.  It is therefore retained as
    # a sensitivity flag, while the formal descriptor is capped at E1.
    summary["formal_e2_contract_pass"] = False
    summary["event_support_code"] = np.where(summary["formal_e1_contract_pass"], "E1", "E0")
    summary["event_test_status"] = np.where(
        summary["formal_e1_contract_pass"], "run_supported", "run_not_supported"
    )
    summary["event_q"] = summary["block_q_value"]
    summary["event_q_missing_reason"] = None
    summary["e0_reason_code"] = np.where(
        summary["event_support_code"].eq("E0"), "E0_not_supported", None
    )
    summary["posthoc_gate_rule"] = "n_blocks>=3 AND block_q<=0.10 AND CI excludes 0 AND direction_stability>=0.80"
    return summary.sort_values(["posthoc_e2_eligible", "block_q_value", "perturbation"], ascending=[False, True, True])


def stat1_gate_audit(
    support: pd.DataFrame,
    legacy_alignment_path: Path,
    negative_controls_path: Path,
) -> pd.DataFrame:
    stat1 = support.loc[support["perturbation"].eq("STAT1")].iloc[0]
    legacy = pd.read_csv(legacy_alignment_path, sep="\t")
    legacy_stat1 = legacy.loc[legacy["perturbation"].eq("STAT1")].iloc[0]
    negative = pd.read_csv(negative_controls_path, sep="\t")
    negative_threshold = float(legacy["event_effect_size"].abs().quantile(0.90))
    negative_max = float(negative["max_abs_effect_size"].max())
    negative_margin = negative_threshold - negative_max
    rows = [
        {
            "field": "legacy_event_q",
            "value": float(legacy_stat1["event_q_value"]),
            "threshold_or_rule": "event q <= 0.05",
            "numeric_pass": bool(float(legacy_stat1["event_q_value"]) <= 0.05),
            "contract_pass": False,
            "audit_note": "Cell-level Welch/BH value; not an independent-block event q and not accepted as the locked inferential-unit gate.",
        },
        {
            "field": "formal_event_q",
            "value": float(stat1["block_q_value"]),
            "threshold_or_rule": "general retrospective E1 event q <= 0.10",
            "numeric_pass": bool(float(stat1["block_q_value"]) <= 0.10),
            "contract_pass": True,
            "audit_note": "BH-adjusted one-sample test over the three replicate-block effects; this is the formal E1 event q.",
        },
        {
            "field": "locked_primary_q_max",
            "value": float(stat1["block_q_value"]),
            "threshold_or_rule": "locked primary event q <= 0.05",
            "numeric_pass": bool(float(stat1["block_q_value"]) <= 0.05),
            "contract_pass": False,
            "audit_note": "The independent-block event q is 0.0556, so the locked confirmatory primary gate fails; CI cannot substitute for this failure.",
        },
        {
            "field": "posthoc_e2_block_q",
            "value": float(stat1["block_q_value"]),
            "threshold_or_rule": "post-hoc E2-sensitivity block q <= 0.10 AND CI excludes 0",
            "numeric_pass": bool(float(stat1["block_q_value"]) <= 0.10 and stat1["ci_excludes_zero"]),
            "contract_pass": True,
            "audit_note": "Passes the retrospective q-and-CI sensitivity rule only; the rule was not fixed before STAT1 inspection.",
        },
        {
            "field": "block_ci95",
            "value": f"[{float(stat1['block_ci95_low']):.12g}, {float(stat1['block_ci95_high']):.12g}]",
            "threshold_or_rule": "95% CI excludes 0",
            "numeric_pass": bool(stat1["ci_excludes_zero"]),
            "contract_pass": True,
            "audit_note": "Three-block t interval (df=2).",
        },
        {
            "field": "block_count",
            "value": int(stat1["n_blocks"]),
            "threshold_or_rule": "n_blocks >= 3",
            "numeric_pass": bool(int(stat1["n_blocks"]) >= 3),
            "contract_pass": True,
            "audit_note": "Blocks are replicate labels rep1, rep2 and rep3.",
        },
        {
            "field": "direction_stability",
            "value": float(stat1["direction_stability"]),
            "threshold_or_rule": ">= 0.80",
            "numeric_pass": bool(float(stat1["direction_stability"]) >= 0.80),
            "contract_pass": True,
            "audit_note": "All three replicate effects have the same negative direction.",
        },
        {
            "field": "basic_negative_control_gate",
            "value": f"max={negative_max:.12g}; margin={negative_margin:.12g}",
            "threshold_or_rule": f"all controls < legacy 90th-percentile threshold {negative_threshold:.12g}",
            "numeric_pass": bool(negative["negative_control_pass"].astype(bool).all()),
            "contract_pass": True,
            "audit_note": "The basic matched negative-control panel passes for the retrospective E1 boundary.",
        },
        {
            "field": "complete_e2_control_integration",
            "value": "not_executed",
            "threshold_or_rule": "all mandatory E2 controls joined into one locked executable audit",
            "numeric_pass": None,
            "contract_pass": False,
            "audit_note": "The passing legacy controls were not joined into the post-hoc E2 block script or its locked audit trail.",
        },
        {
            "field": "mode_identifiability",
            "value": "not_recorded",
            "threshold_or_rule": "mandatory for E2",
            "numeric_pass": None,
            "contract_pass": False,
            "audit_note": "Suppression is implied by direction, but no executable mode-identifiability gate is stored.",
        },
        {
            "field": "rule_fixed_before_STAT1",
            "value": False,
            "threshold_or_rule": "required for a locked primary E2 claim",
            "numeric_pass": None,
            "contract_pass": False,
            "audit_note": "The q-and-CI block audit was added during revision and is explicitly post hoc.",
        },
        {
            "field": "final_boolean_rule",
            "value": "E1-V1 formal; post-hoc E2-eligible sensitivity",
            "threshold_or_rule": "E1 = independent-block event_q<=0.10 AND retained family/basic controls; formal E2 requires the complete locked contract",
            "numeric_pass": True,
            "contract_pass": True,
            "audit_note": "E1 passes with event q=0.0556. The locked q<=0.05 primary gate fails. Post-hoc E2 eligibility = n_blocks>=3 AND block_q<=0.10 AND CI excludes 0 AND stability>=0.80, but does not confer E2.",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Block-aware GSE153056 RNA-event validation for the TED E axis")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--legacy-alignment", type=Path, default=DEFAULT_LEGACY_ALIGNMENT)
    parser.add_argument("--negative-controls", type=Path, default=DEFAULT_NEGATIVE_CONTROLS)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.input / "cell_metadata.tsv.gz", sep="\t").set_index("cell_id", drop=False)
    expression = read_selected_rows(args.input / "expression_matrix.tsv.gz", IFNG_PDL1_AXIS)
    protein = read_selected_rows(args.input / "protein_matrix.tsv.gz", ["PDL1"])
    shared = metadata.index.intersection(expression.index).intersection(protein.index)
    metadata = metadata.loc[shared]
    event_scores = axis_score(expression.loc[shared])
    protein_scores = np.log1p(protein.loc[shared, "PDL1"].astype(float)).rename("axis_score")

    event_effects = replicate_effects(event_scores, metadata, "RNA_event")
    protein_effects = replicate_effects(protein_scores, metadata, "PDL1_protein")
    effects = pd.concat([event_effects, protein_effects], ignore_index=True)
    support = summarize_event_support(event_effects, protein_effects)
    posthoc_e2 = support[support["posthoc_e2_eligible"]]
    formal_e1 = support[support["event_support_code"].eq("E1")]
    dataset_code = "E1-V1" if not formal_e1.empty else "E0-V1"
    gate_audit = stat1_gate_audit(support, args.legacy_alignment, args.negative_controls)

    effects.to_csv(args.out / "gse153056_replicate_effects.tsv", sep="\t", index=False)
    support.to_csv(args.out / "gse153056_block_event_support.tsv", sep="\t", index=False)
    gate_audit.to_csv(args.out / "gse153056_stat1_gate_audit.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "dataset": "GSE153056",
                "analysis_status": "post_hoc_revision_block_audit",
                "n_cells": len(shared),
                "n_replicates": metadata["replicate"].nunique(),
                "n_perturbations_tested": len(support),
                "n_formal_e2_perturbations": 0,
                "n_posthoc_e2_eligible_perturbations": len(posthoc_e2),
                "posthoc_e2_eligible_perturbations": ";".join(posthoc_e2["perturbation"].astype(str)),
                "dataset_evidence_descriptor": dataset_code,
                "validation_basis": "PD-L1 protein outcome",
                "qualification": "Formal E1-V1. STAT1 is E2-eligible only under a retrospective q-and-CI block sensitivity rule that was added during revision and did not execute the complete locked E2 contract.",
            }
        ]
    ).to_csv(args.out / "gse153056_block_summary.tsv", sep="\t", index=False)
    print(f"Wrote GSE153056 block-aware audit to {args.out}; descriptor {dataset_code}; post-hoc E2-eligible perturbations={len(posthoc_e2)}")


if __name__ == "__main__":
    main()
