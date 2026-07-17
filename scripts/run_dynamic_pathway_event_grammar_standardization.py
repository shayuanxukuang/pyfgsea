from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch


plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data_external" / "StepXX_dynamic_pathway_event_grammar_standardization"
DEFAULT_SUBMISSION = ROOT / "GenomeBiology_submission_files_only"
DEFAULT_PAUL15 = ROOT / "results" / "paul15_branch_validation"
DEFAULT_GSE271399 = ROOT / "data_external" / "deliverables_all_ted_rounds" / "GSE271399_T21_GATA1s"


EVENT_COLUMNS = [
    "event_id",
    "dataset",
    "organism",
    "input_type",
    "contrast_or_context",
    "pathway",
    "event_type",
    "event_subtype",
    "biological_meaning",
    "onset_time",
    "peak_time",
    "shutdown_time",
    "duration",
    "branch_specificity",
    "spatial_specificity",
    "rescue_status",
    "event_effect",
    "event_FDR",
    "event_FDR_available",
    "event_FDR_reason_if_missing",
    "block_stability",
    "negative_control_margin",
    "robustness_score",
    "prediction_robustness_score",
    "validation_robustness_score",
    "validation_status",
    "claim_boundary",
    "supported_interpretation",
    "unsupported_interpretation",
    "source_file",
]


def read_table(path: Path, sep: str | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if sep is None:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=sep)


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def robustness_score(
    event_fdr: float | None = None,
    block_stability: float | None = None,
    negative_control_margin: float | None = None,
    effect: float | None = None,
) -> float:
    """Compact 0-1 score for display, not a replacement for component gates."""
    parts: list[float] = []
    if event_fdr is not None and not math.isnan(event_fdr):
        parts.append(max(0.0, min(1.0, 1.0 - min(event_fdr, 0.25) / 0.25)))
    if block_stability is not None and not math.isnan(block_stability):
        parts.append(max(0.0, min(1.0, block_stability)))
    if negative_control_margin is not None and not math.isnan(negative_control_margin):
        parts.append(max(0.0, min(1.0, 0.5 + negative_control_margin / 2.0)))
    if effect is not None and not math.isnan(effect):
        parts.append(max(0.0, min(1.0, abs(effect) / 3.0)))
    if not parts:
        return math.nan
    return round(float(sum(parts) / len(parts)), 3)


def add_row(rows: list[dict[str, Any]], **kwargs: Any) -> None:
    row = {col: "" for col in EVENT_COLUMNS}
    row.update(kwargs)
    raw_fdr = row.get("event_FDR", "")
    fdr = as_float(raw_fdr)
    if not math.isnan(fdr):
        row["event_FDR"] = fdr
        row["event_FDR_available"] = True
        row["event_FDR_reason_if_missing"] = ""
    else:
        reason = fmt(raw_fdr) or "not_provided"
        row["event_FDR"] = ""
        row["event_FDR_available"] = False
        row["event_FDR_reason_if_missing"] = reason
    if not row.get("validation_status"):
        row["validation_status"] = "not_functional_validation_test"
    rows.append(row)


def standardize_paul15(rows: list[dict[str, Any]], paul15_dir: Path) -> None:
    events = read_table(paul15_dir / "paul15_branch_events.csv")
    comps = read_table(paul15_dir / "paul15_branch_comparisons.csv")
    checks = read_table(paul15_dir / "paul15_expected_program_checks.csv")

    if not events.empty:
        # Use the strongest interpretable branch events to populate onset, peak and shutdown grammar.
        events = events.sort_values(["window_fdr_min", "AUC_abs"], ascending=[True, False])
        for _, r in events.head(8).iterrows():
            pathway = fmt(r.get("Pathway"))
            branch = fmt(r.get("ted_branch"))
            fdr = as_float(r.get("window_fdr_min"))
            peak = as_float(r.get("peak_time"))
            duration = as_float(r.get("duration"))
            peak_effect = as_float(r.get("peak_NES"))
            onset = as_float(r.get("activation_onset"))
            shutdown = as_float(r.get("suppression_onset"))
            branch_spec = "branch=" + branch if branch else "branch_specificity_not_available"
            common = dict(
                dataset="Paul15_branch_validation",
                organism="mouse",
                input_type="single-cell trajectory branch benchmark",
                contrast_or_context=branch or "branch trajectory",
                pathway=pathway,
                event_FDR=fdr,
                branch_specificity=branch_spec,
                spatial_specificity="not_applicable",
                rescue_status="not_tested",
                block_stability="branch_window_FDR_supported" if fdr <= 0.10 else "not_block_level",
                negative_control_margin="supported_by_branch_permutation" if fdr <= 0.10 else "weak",
                claim_boundary="tier_event_FDR_supported_or_branch_event_candidate",
                source_file="results/paul15_branch_validation/paul15_branch_events.csv",
            )
            if not math.isnan(onset):
                add_row(
                    rows,
                    **common,
                    event_id=f"paul15_onset_{pathway}_{branch}".replace(" ", "_"),
                    event_type="onset",
                    event_subtype="activation_onset",
                    biological_meaning="pathway activation begins along a branch trajectory",
                    onset_time=round(onset, 4),
                    peak_time=round(peak, 4) if not math.isnan(peak) else "",
                    duration=round(duration, 4) if not math.isnan(duration) else "",
                    event_effect=round(peak_effect, 4) if not math.isnan(peak_effect) else "",
                    robustness_score=robustness_score(fdr, None, 0.25 if fdr <= 0.10 else -0.25, peak_effect),
                    supported_interpretation="branch-associated pathway onset event",
                    unsupported_interpretation="validated causal branch fate mechanism",
                )
            if not math.isnan(peak):
                add_row(
                    rows,
                    **common,
                    event_id=f"paul15_peak_{pathway}_{branch}".replace(" ", "_"),
                    event_type="peak",
                    event_subtype="trajectory_peak",
                    biological_meaning="pathway reaches maximum activity within the branch trajectory",
                    onset_time=round(onset, 4) if not math.isnan(onset) else "",
                    peak_time=round(peak, 4),
                    duration=round(duration, 4) if not math.isnan(duration) else "",
                    event_effect=round(peak_effect, 4) if not math.isnan(peak_effect) else "",
                    robustness_score=robustness_score(fdr, None, 0.25 if fdr <= 0.10 else -0.25, peak_effect),
                    supported_interpretation="branch-local pathway peak event",
                    unsupported_interpretation="functional proof of lineage fate control",
                )
            if not math.isnan(shutdown):
                add_row(
                    rows,
                    **common,
                    event_id=f"paul15_shutdown_{pathway}_{branch}".replace(" ", "_"),
                    event_type="shutdown",
                    event_subtype="suppression_onset",
                    biological_meaning="pathway activity switches toward suppression or trough phase",
                    onset_time=round(onset, 4) if not math.isnan(onset) else "",
                    peak_time=round(peak, 4) if not math.isnan(peak) else "",
                    shutdown_time=round(shutdown, 4),
                    duration=round(duration, 4) if not math.isnan(duration) else "",
                    event_effect=round(as_float(r.get("trough_NES")), 4),
                    robustness_score=robustness_score(fdr, None, 0.25 if fdr <= 0.10 else -0.25, as_float(r.get("trough_NES"))),
                    supported_interpretation="trajectory shutdown/suppression event",
                    unsupported_interpretation="validated irreversible pathway termination",
                )

    if not comps.empty:
        comps = comps.sort_values("delta_AUC", key=lambda s: s.abs(), ascending=False)
        for _, r in comps.head(6).iterrows():
            pathway = fmt(r.get("Pathway"))
            fdr_a = as_float(r.get("branch_a_window_fdr_min"))
            fdr_b = as_float(r.get("branch_b_window_fdr_min"))
            fdr = min(fdr_a, fdr_b)
            effect = as_float(r.get("delta_AUC"))
            branch_specificity = f"{fmt(r.get('branch_a'))}_vs_{fmt(r.get('branch_b'))}; delta_AUC={round(effect, 4)}"
            add_row(
                rows,
                event_id=f"paul15_branch_specific_{pathway}".replace(" ", "_"),
                dataset="Paul15_branch_validation",
                organism="mouse",
                input_type="single-cell trajectory branch benchmark",
                contrast_or_context=f"{fmt(r.get('branch_a'))}_vs_{fmt(r.get('branch_b'))}",
                pathway=pathway,
                event_type="branch-specific",
                event_subtype=fmt(r.get("interpretation")) or "branch_divergence",
                biological_meaning="pathway event differs across sibling or alternative trajectory branches",
                onset_time="",
                peak_time=f"{round(as_float(r.get('branch_a_peak_time')), 4)}|{round(as_float(r.get('branch_b_peak_time')), 4)}",
                duration=f"{round(as_float(r.get('branch_a_duration')), 4)}|{round(as_float(r.get('branch_b_duration')), 4)}",
                branch_specificity=branch_specificity,
                spatial_specificity="not_applicable",
                rescue_status="not_tested",
                event_effect=round(effect, 4),
                event_FDR=fdr,
                block_stability="branch_window_FDR_supported" if fdr <= 0.10 else "weak",
                negative_control_margin="branch_permutation_supported" if fdr <= 0.10 else "weak",
                robustness_score=robustness_score(fdr, None, 0.30 if fdr <= 0.10 else -0.20, effect),
                claim_boundary="tier_branch_event_candidate",
                supported_interpretation="branch-specific pathway event",
                unsupported_interpretation="validated fate-determining pathway mechanism",
                source_file="results/paul15_branch_validation/paul15_branch_comparisons.csv",
            )

    if not checks.empty:
        for _, r in checks[checks["observed_support"].astype(str).str.lower().eq("supported")].head(4).iterrows():
            pathway = fmt(r.get("Pathway"))
            effect = as_float(r.get("branch_a_peak_NES")) - as_float(r.get("branch_b_peak_NES"))
            fdr = min(as_float(r.get("branch_a_window_fdr_min")), as_float(r.get("branch_b_window_fdr_min")))
            add_row(
                rows,
                event_id=f"paul15_expected_branch_support_{pathway}".replace(" ", "_"),
                dataset="Paul15_branch_validation",
                organism="mouse",
                input_type="single-cell branch/fate benchmark",
                contrast_or_context=f"expected_{fmt(r.get('expected_branch'))}_program",
                pathway=pathway,
                event_type="branch-specific",
                event_subtype="expected_fate_program_support",
                biological_meaning="known lineage-associated program is recovered as a branch-skewed pathway event",
                peak_time="branch_peak_comparison",
                branch_specificity=f"expected_branch={fmt(r.get('expected_branch'))}",
                spatial_specificity="not_applicable",
                rescue_status="not_tested",
                event_effect=round(effect, 4),
                event_FDR=fdr,
                block_stability="expected_program_supported",
                negative_control_margin="expected_program_check_pass",
                robustness_score=robustness_score(fdr, None, 0.35, effect),
                claim_boundary="tier_branch_event_candidate",
                supported_interpretation="expected branch/fate pathway event recovered",
                unsupported_interpretation="new validated lineage mechanism",
                source_file="results/paul15_branch_validation/paul15_expected_program_checks.csv",
            )


def standardize_bombyx(rows: list[dict[str, Any]], submission_dir: Path) -> None:
    summary = read_table(submission_dir / "bombyx_spatial_pathway_localization_summary.tsv")
    method = read_table(submission_dir / "bombyx_method_comparison_summary.tsv")
    method_lookup = {}
    if not method.empty:
        method_lookup = {str(r["gene_set"]): r for _, r in method.iterrows()}

    if summary.empty:
        return
    for _, r in summary.sort_values("best_dynamic_signal", ascending=False).head(6).iterrows():
        gene_set = fmt(r.get("gene_set"))
        label = fmt(r.get("gene_set_label"))
        effect = as_float(r.get("best_dynamic_signal"))
        fdr = as_float(r.get("matched_expression_empirical_p"))
        spatial_z = as_float(r.get("spatial_specificity_z"))
        margin = as_float(r.get("shuffle_null_separation"))
        mrow = method_lookup.get(gene_set)
        static_signal = as_float(mrow.get("global_static_signal")) if mrow is not None else as_float(r.get("global_static_signal"))
        dynamic_gain = as_float(r.get("dynamic_gain_over_static"))
        event_type = "spatial-localized"
        subtype = "spatial_trajectory_localization"
        if "20e" in gene_set.lower():
            subtype = "hormone_associated_window_support"
        add_row(
            rows,
            event_id=f"bombyx_spatial_{gene_set}".replace(" ", "_"),
            dataset="STT0000176_Bombyx_metamorphosis",
            organism="Bombyx mori",
            input_type="Stereo-seq spatial/trajectory bin rankings",
            contrast_or_context="wing_disc_metamorphosis_spatial_windows",
            pathway=label or gene_set,
            event_type=event_type,
            event_subtype=subtype,
            biological_meaning="pathway activity localizes to spatial or trajectory windows during metamorphosis",
            onset_time="stage/region window",
            peak_time="best_dynamic_window",
            duration="spatial_window_support",
            branch_specificity="not_applicable",
            spatial_specificity=f"z={round(spatial_z, 3)}; sections={fmt(r.get('section_consistency'))}",
            rescue_status="not_tested",
            event_effect=round(effect, 4),
            event_FDR=fdr,
            block_stability=fmt(r.get("section_consistency")),
            negative_control_margin=round(margin, 4),
            robustness_score=robustness_score(fdr, None, margin, effect),
            claim_boundary=fmt(r.get("interpretation_boundary_short")) or "spatial_localization_only",
            supported_interpretation="spatial/trajectory pathway localization event",
            unsupported_interpretation="direct hormonal causality or new metamorphosis mechanism from enrichment alone",
            source_file="GenomeBiology_submission_files_only/bombyx_spatial_pathway_localization_summary.tsv",
        )

        if not math.isnan(dynamic_gain) and dynamic_gain > 0:
            add_row(
                rows,
                event_id=f"bombyx_peak_{gene_set}".replace(" ", "_"),
                dataset="STT0000176_Bombyx_metamorphosis",
                organism="Bombyx mori",
                input_type="Stereo-seq spatial/trajectory bin rankings",
                contrast_or_context="dynamic_vs_static_pathway_localization",
                pathway=label or gene_set,
                event_type="peak",
                event_subtype="dynamic_window_peak_over_static_baseline",
                biological_meaning="dynamic window has stronger pathway signal than global/static baseline",
                onset_time="stage/region window",
                peak_time="best_dynamic_window",
                duration="localized_window",
                branch_specificity="not_applicable",
                spatial_specificity=f"dynamic_gain={round(dynamic_gain, 4)}; static={round(static_signal, 4)}",
                rescue_status="not_tested",
                event_effect=round(dynamic_gain, 4),
                event_FDR=fdr,
                block_stability=fmt(r.get("section_consistency")),
                negative_control_margin=round(margin, 4),
                robustness_score=robustness_score(fdr, None, margin, dynamic_gain),
                claim_boundary="dynamic localization event; not causal mechanism",
                supported_interpretation="dynamic pathway peak/localization event",
                unsupported_interpretation="pathway mechanism inferred from static enrichment alone",
                source_file="GenomeBiology_submission_files_only/bombyx_method_comparison_summary.tsv",
            )


def standardize_gse271399(rows: list[dict[str, Any]], gse_dir: Path) -> None:
    scorecard = read_table(gse_dir / "gse271399_family_evidence_scorecard.tsv")
    rescue = read_table(gse_dir / "full_length_gata1_rescue_prediction_table.tsv")
    if not scorecard.empty:
        sub = scorecard[
            (scorecard["family_id"].astype(str).eq("ERYTHROID_EVENT_LOSS_FAMILY"))
            & (scorecard["contrast_or_effect"].astype(str).eq("T21_GATA1s_vs_T21_wtGATA1"))
        ].copy()
        sub = sub.sort_values("family_delta_auc")
        for _, r in sub.head(3).iterrows():
            effect = as_float(r.get("family_delta_auc"))
            direction_stability = as_float(r.get("direction_stability"))
            margin = as_float(r.get("specificity_percentile"))
            add_row(
                rows,
                event_id=f"gse271399_shutdown_{fmt(r.get('trajectory'))}_{fmt(r.get('family_id'))}".replace(" ", "_"),
                dataset="GSE271399_T21_GATA1s",
                organism="human",
                input_type="single-cell hematopoietic differentiation trajectory",
                contrast_or_context=fmt(r.get("contrast_or_effect")),
                pathway=fmt(r.get("family_id")),
                event_type="shutdown",
                event_subtype="event_loss_family",
                biological_meaning="erythroid pathway family is suppressed or fails to activate in the perturbation contrast",
                onset_time="erythroid trajectory window",
                peak_time="loss_family_window",
                shutdown_time="late erythroid output window",
                duration="trajectory_family_AUC",
                branch_specificity=fmt(r.get("trajectory")),
                spatial_specificity="not_applicable",
                rescue_status="predicted_rescue_required_not_observed",
                event_effect=round(effect, 4),
                event_FDR="not_estimable_from_independent_biological_blocks",
                block_stability=(
                    f"design_stratum_direction_stability={direction_stability:.3f};"
                    " not_biological_replication"
                ),
                negative_control_margin=round(margin, 4),
                robustness_score=robustness_score(None, None, margin - 0.5, effect),
                claim_boundary="E1-V0 direction-consistent candidate; design-stratum sensitivity only",
                supported_interpretation="direction-consistent erythroid event-loss candidate without biological-block robustness",
                unsupported_interpretation="GATA1s/T21 causal erythroid failure mechanism proven",
                source_file="data_external/deliverables_all_ted_rounds/GSE271399_T21_GATA1s/gse271399_family_evidence_scorecard.tsv",
            )
    if not rescue.empty:
        for _, r in rescue.head(4).iterrows():
            score = as_float(r.get("predicted_full_length_GATA1_rescue_score"))
            add_row(
                rows,
                event_id=f"gse271399_failed_rescue_pending_{fmt(r.get('model_readout'))}".replace(" ", "_"),
                dataset="GSE271399_T21_GATA1s",
                organism="human",
                input_type="single-cell hematopoietic differentiation trajectory plus rescue prediction",
                contrast_or_context="full_length_GATA1_rescue_prediction",
                pathway=fmt(r.get("readout")),
                event_type="failed-rescue-extension",
                event_subtype="predicted_rescue_axis_without_observed_rescue",
                biological_meaning="TED predicts rescue direction but matched rescue data are not present in the primary system",
                onset_time="not_applicable",
                peak_time="not_applicable",
                shutdown_time="not_applicable",
                duration="not_applicable",
                branch_specificity="erythroid trajectory",
                spatial_specificity="not_applicable",
                rescue_status="predicted_only_missing_matched_rescue",
                event_effect=round(score, 4),
                event_FDR="not_applicable_prediction_table",
                block_stability="inherits_primary_event_family",
                negative_control_margin="requires rescue negative controls",
                robustness_score=round(min(0.45, max(0.0, score * 0.45)), 3) if not math.isnan(score) else "",
                prediction_robustness_score=round(min(1.0, max(0.0, score)), 3) if not math.isnan(score) else "",
                validation_robustness_score=0.0,
                validation_status="prediction_only_no_matched_rescue",
                claim_boundary="rescue prediction only; not functionally validated",
                supported_interpretation="experimentally testable rescue-axis prediction",
                unsupported_interpretation="completed functional rescue validation",
                source_file="data_external/deliverables_all_ted_rounds/GSE271399_T21_GATA1s/full_length_gata1_rescue_prediction_table.tsv",
            )


def standardize_gse123013(rows: list[dict[str, Any]], submission_dir: Path) -> None:
    table = read_table(submission_dir / "gse123013_root_fate_event_table.tsv")
    if table.empty:
        return
    for _, r in table.head(4).iterrows():
        effect = as_float(r.get("event_strength"))
        add_row(
            rows,
            event_id=f"gse123013_branch_fate_{fmt(r.get('event_family'))}".replace(" ", "_"),
            dataset="GSE123013_Arabidopsis_root",
            organism="Arabidopsis thaliana",
            input_type="root epidermis mutant scRNA-seq TED-lite",
            contrast_or_context="rhd6_vs_gl2_vs_WT",
            pathway=fmt(r.get("event_family")),
            event_type="branch-specific",
            event_subtype=fmt(r.get("event_mode")),
            biological_meaning="opposite or fate-like root epidermal axis differs by genotype but is stress-sensitive",
            onset_time="not_resolved",
            peak_time="not_resolved",
            duration="not_resolved",
            branch_specificity=fmt(r.get("expected_pattern")),
            spatial_specificity="not_applicable",
            rescue_status="not_tested",
            event_effect=round(effect, 4),
            event_FDR="not_available_TED_lite",
            block_stability="stress_adjusted_downgraded",
            negative_control_margin="stress_wound_controls_dominate_or_attenuate",
            robustness_score=robustness_score(None, None, -0.25, effect),
            claim_boundary=fmt(r.get("claim_boundary")),
            supported_interpretation="stress-sensitive plant fate-switch candidate",
            unsupported_interpretation="strong root-hair causal mechanism",
            source_file="GenomeBiology_submission_files_only/gse123013_root_fate_event_table.tsv",
        )


def event_type_definitions() -> str:
    return """# Dynamic pathway event type definitions

This file defines the standardized event grammar used by `dynamic_pathway_event_table.tsv`.
The goal is to convert pathway or module activity curves into auditable pathway-event rows.

## Core event types

| event_type | Definition | Required evidence | Unsupported escalation |
|---|---|---|---|
| onset | A pathway begins activation or suppression at an ordered time, pseudotime, branch or spatial window. | ordered coordinate, effect direction, event_FDR or window-level support | causal initiation mechanism |
| peak | A pathway reaches maximal activity in a trajectory, time, branch or spatial window. | peak coordinate, effect size, comparator or null support | mechanism inferred from a curve maximum alone |
| shutdown | A pathway enters suppression, trough or persistent loss after a prior active or expected state. | suppression/trough coordinate or event-loss family, robustness or contrast support | irreversible biological termination without validation |
| branch-specific | A pathway event differs between branches, fates, genotypes or lineages. | branch labels or fate scaffold, branch-specific effect, null or expected-program support | fate-determining causal mechanism |
| spatial-localized | A pathway event is localized to a spatial region, tissue section or spatial trajectory window. | spatial/bin/region context, dynamic signal, spatial/null support | direct spatial causal mechanism from enrichment alone |

## Extension event types

| event_type | Definition | Boundary |
|---|---|---|
| rescued-extension | A perturbed event reverses in a matched rescue or intervention design. | Requires matched rescue/intervention contrast and negative controls. |
| failed-rescue-extension | TED predicts rescue axes or observes incomplete reversal, but matched rescue validation is absent or fails. | Report as prediction or failed public gate, not completed functional validation. |

## Required row fields

Each standardized event row includes `event_FDR` when a numeric event-level FDR is available,
`event_FDR_available`, `event_FDR_reason_if_missing`, `robustness_score`,
`claim_boundary`, `supported_interpretation` and `unsupported_interpretation`.
Prediction-only rescue extensions additionally report `prediction_robustness_score`,
`validation_robustness_score` and `validation_status`, so rescue predictions are not
mistaken for matched functional validation.
"""


def event_calling_rules_yaml() -> str:
    return """version: 1
name: dynamic_pathway_event_grammar_standardization
purpose: >
  Convert pathway/module score curves and TED event outputs into a common
  pathway-event grammar table.

required_core_fields:
  - dataset
  - pathway
  - event_type
  - event_FDR
  - event_FDR_available
  - event_FDR_reason_if_missing
  - robustness_score
  - validation_status
  - claim_boundary
  - supported_interpretation
  - unsupported_interpretation

event_types:
  onset:
    required_evidence:
      - ordered coordinate or branch/spatial window
      - activation_onset or suppression_onset estimate
      - event_FDR or window-level support
    unsupported_escalation: causal initiation mechanism
  peak:
    required_evidence:
      - peak_time or best_dynamic_window
      - peak effect or dynamic gain
      - null or baseline comparison when available
    unsupported_escalation: mechanism from curve maximum alone
  shutdown:
    required_evidence:
      - suppression_onset, trough, persistent loss, or event-loss family
      - direction and robustness support
    unsupported_escalation: irreversible termination without validation
  branch-specific:
    required_evidence:
      - branch/fate/lineage or genotype scaffold
      - branch-specific effect or expected-program support
      - null or held-out support when available
    unsupported_escalation: fate-determining causal mechanism
  spatial-localized:
    required_evidence:
      - spatial region, bin, section, or spatial trajectory window
      - dynamic signal stronger than static or matched control
      - spatial/null support
    unsupported_escalation: direct spatial or hormonal causality
  rescued-extension:
    required_evidence:
      - matched intervention or rescue contrast
      - reversal toward reference
      - negative controls below primary axis
    unsupported_escalation: primary-system causality if rescue is public or unmatched
  failed-rescue-extension:
    required_evidence:
      - predicted rescue axis or failed public gate
      - explicit missing rescue/intervention evidence
    unsupported_escalation: completed rescue validation

robustness_score:
  scale: 0_to_1
  components:
    - event_FDR_support
    - block_or_section_or_branch_stability
    - negative_control_margin
    - absolute_event_effect
  note: >
    The score is a compact display summary. Component gates remain the auditable
    evidence fields and should not be replaced by the aggregate score.
"""


def claim_boundary_mapping(events: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "event_type": "onset",
            "minimum_boundary": "tier_event_FDR_supported",
            "upgrade_evidence": "block/branch support plus negative-control margin",
            "supported_interpretation_template": "pathway onset event",
            "unsupported_interpretation_template": "causal initiation mechanism",
        },
        {
            "event_type": "peak",
            "minimum_boundary": "tier_event_FDR_supported",
            "upgrade_evidence": "dynamic gain over static baseline plus null support",
            "supported_interpretation_template": "pathway peak/localization event",
            "unsupported_interpretation_template": "mechanism from curve maximum alone",
        },
        {
            "event_type": "shutdown",
            "minimum_boundary": "tier_robust_event_or_computational_candidate",
            "upgrade_evidence": "block robust persistent loss plus matched-state and negative-control gates",
            "supported_interpretation_template": "pathway shutdown or event-loss family",
            "unsupported_interpretation_template": "validated irreversible pathway termination",
        },
        {
            "event_type": "branch-specific",
            "minimum_boundary": "tier_branch_event_candidate",
            "upgrade_evidence": "expected-program support, branch permutation, independent validation",
            "supported_interpretation_template": "branch-specific pathway event",
            "unsupported_interpretation_template": "validated fate-determining mechanism",
        },
        {
            "event_type": "spatial-localized",
            "minimum_boundary": "tier_spatial_localization_candidate",
            "upgrade_evidence": "section consistency, matched-expression null, spatial block shuffle",
            "supported_interpretation_template": "spatial/trajectory pathway localization event",
            "unsupported_interpretation_template": "direct spatial or hormonal causality",
        },
        {
            "event_type": "rescued-extension",
            "minimum_boundary": "tier_public_or_matched_intervention_candidate",
            "upgrade_evidence": "matched rescue/intervention reversal with negative controls",
            "supported_interpretation_template": "matched rescue/restoration event",
            "unsupported_interpretation_template": "primary-system functional causality without matched validation",
        },
        {
            "event_type": "failed-rescue-extension",
            "minimum_boundary": "tier_rescue_prediction_or_failed_gate",
            "upgrade_evidence": "matched rescue experiment that passes reversal and specificity gates",
            "supported_interpretation_template": "rescue-axis prediction or failed rescue gate",
            "unsupported_interpretation_template": "completed functional rescue validation",
        },
    ]
    counts = events.groupby("event_type").size().rename("n_rows").reset_index()
    df = pd.DataFrame(rows).merge(counts, on="event_type", how="left")
    df["n_rows"] = df["n_rows"].fillna(0).astype(int)
    return df


def robustness_summary(events: pd.DataFrame) -> pd.DataFrame:
    tmp = events.copy()
    tmp["robustness_score_numeric"] = pd.to_numeric(tmp["robustness_score"], errors="coerce")
    tmp["event_FDR_numeric"] = pd.to_numeric(tmp["event_FDR"], errors="coerce")
    tmp["prediction_robustness_score_numeric"] = pd.to_numeric(tmp["prediction_robustness_score"], errors="coerce")
    tmp["validation_robustness_score_numeric"] = pd.to_numeric(tmp["validation_robustness_score"], errors="coerce")
    tmp["event_FDR_available_bool"] = tmp["event_FDR_available"].astype(str).str.lower().isin(["true", "1", "yes"])
    return (
        tmp.groupby(["dataset", "event_type"], dropna=False)
        .agg(
            n_events=("event_id", "count"),
            mean_robustness_score=("robustness_score_numeric", "mean"),
            median_robustness_score=("robustness_score_numeric", "median"),
            mean_prediction_robustness_score=("prediction_robustness_score_numeric", "mean"),
            mean_validation_robustness_score=("validation_robustness_score_numeric", "mean"),
            min_event_FDR=("event_FDR_numeric", "min"),
            n_with_numeric_event_FDR=("event_FDR_numeric", lambda s: int(s.notna().sum())),
            n_event_FDR_available=("event_FDR_available_bool", lambda s: int(s.sum())),
            n_prediction_only=("validation_status", lambda s: int(s.astype(str).str.contains("prediction_only", case=False, na=False).sum())),
            n_with_negative_control_margin=("negative_control_margin", lambda s: int((s.astype(str) != "").sum())),
            n_with_claim_boundary=("claim_boundary", lambda s: int((s.astype(str) != "").sum())),
        )
        .reset_index()
    )


def baseline_comparison(submission_dir: Path, events: pd.DataFrame) -> pd.DataFrame:
    baseline = read_table(submission_dir / "baseline_two_layer_comparison.tsv")
    rows: list[dict[str, Any]] = []
    if not baseline.empty:
        for _, r in baseline.iterrows():
            rows.append(
                {
                    "comparison_id": fmt(r.get("method_or_adapter")),
                    "baseline_curve_or_score_output": fmt(r.get("native_task_output")),
                    "ted_event_output": fmt(r.get("same_data_ted_object_task")),
                    "baseline_event_type_accuracy_mean": r.get("baseline_event_type_accuracy_mean", ""),
                    "TED_event_type_accuracy_mean": r.get("TED_event_type_accuracy_mean", ""),
                    "baseline_overclaim_rate_mean": r.get("baseline_overclaim_rate_mean", ""),
                    "TED_overclaim_rate_mean": r.get("TED_overclaim_rate_mean", ""),
                    "why_event_table_is_different": fmt(r.get("conclusion")),
                }
            )
    rows.append(
        {
            "comparison_id": "dynamic_pathway_event_table",
            "baseline_curve_or_score_output": "ssGSEA/AUCell/GSVA/rolling fgsea curves or scores",
            "ted_event_output": "pathway x event rows with onset/peak/shutdown/branch/spatial labels, event_FDR, robustness and claim boundary",
            "baseline_event_type_accuracy_mean": "not_a_native_field",
            "TED_event_type_accuracy_mean": "see benchmark_compact_table and dynamic_pathway_event_table",
            "baseline_overclaim_rate_mean": "not_a_native_field",
            "TED_overclaim_rate_mean": "explicitly audited",
            "why_event_table_is_different": "The event table stores biological event grammar and missing evidence; score curves store activity magnitude only.",
        }
    )
    explicit_curve_baselines = [
        ("ssGSEA_score_curve", "sample/window-level single-sample enrichment scores"),
        ("AUCell_score_curve", "cell-level rank activity calls"),
        ("GSVA_score_curve", "sample/window-level pathway activity scores"),
        ("rolling_fgsea_profile", "rolling-window enrichment NES and p-values"),
    ]
    for comparison_id, curve_output in explicit_curve_baselines:
        rows.append(
            {
                "comparison_id": comparison_id,
                "baseline_curve_or_score_output": curve_output,
                "ted_event_output": "standardized event rows with event type, event_FDR, robustness_score, claim_boundary, supported_interpretation and unsupported_interpretation",
                "baseline_event_type_accuracy_mean": "not_a_native_field",
                "TED_event_type_accuracy_mean": "event_type_available_for_all_rows",
                "baseline_overclaim_rate_mean": "not_a_native_field",
                "TED_overclaim_rate_mean": "claim_boundary_available_for_all_rows",
                "why_event_table_is_different": "The baseline is an upstream activity profile. TED standardizes the profile into onset/peak/shutdown/branch/spatial event grammar with audit fields.",
            }
        )
    rows.append(
        {
            "comparison_id": "standardized_event_grammar_coverage",
            "baseline_curve_or_score_output": "curve has local maxima/minima but no standardized event row",
            "ted_event_output": f"{events['event_type'].nunique()} event types across {events['dataset'].nunique()} datasets",
            "baseline_event_type_accuracy_mean": "",
            "TED_event_type_accuracy_mean": "",
            "baseline_overclaim_rate_mean": "",
            "TED_overclaim_rate_mean": "",
            "why_event_table_is_different": "The standardized table turns curve features into audit-ready event objects.",
        }
    )
    return pd.DataFrame(rows)


def make_event_grammar_figure(events: pd.DataFrame, out_pdf: Path) -> None:
    out_png = out_pdf.with_suffix(".png")
    type_order = ["onset", "peak", "shutdown", "branch-specific", "spatial-localized", "failed-rescue-extension"]
    counts = events["event_type"].value_counts().reindex(type_order).fillna(0)
    datasets = sorted(events["dataset"].unique())

    fig = plt.figure(figsize=(15.5, 8.8), facecolor="white")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.25, 1.05], height_ratios=[1, 1], wspace=0.5, hspace=0.48)
    ax_flow = fig.add_subplot(gs[:, 0])
    ax_counts = fig.add_subplot(gs[0, 1])
    ax_heat = fig.add_subplot(gs[1, 1])
    ax_table = fig.add_subplot(gs[:, 2])

    for ax in [ax_flow, ax_table]:
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle("Dynamic pathway event grammar standardization", x=0.02, y=0.985, ha="left", fontsize=18, fontweight="bold")

    ax_flow.set_title("A. From pathway curves to event grammar", loc="left", fontsize=11.5, fontweight="bold", pad=10)
    flow_items = [
        ("Pathway/module activity curve", "#e8f1fb"),
        ("Detect ordered features\nonset, peak, shutdown", "#edf7ed"),
        ("Attach context\nbranch, spatial, rescue", "#fff6df"),
        ("Write event table row\nevent-FDR, robustness, boundary", "#f6e9f4"),
    ]
    y_positions = [0.83, 0.60, 0.37, 0.14]
    for i, ((label, color), y) in enumerate(zip(flow_items, y_positions)):
        patch = FancyBboxPatch((0.06, y - 0.07), 0.84, 0.13, boxstyle="round,pad=0.02", facecolor=color, edgecolor="#5a6670")
        ax_flow.add_patch(patch)
        ax_flow.text(0.48, y - 0.005, label, ha="center", va="center", fontsize=9.5)
        if i < len(flow_items) - 1:
            ax_flow.annotate("", xy=(0.48, y_positions[i + 1] + 0.08), xytext=(0.48, y - 0.09), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax_flow.text(0.06, 0.02, "Output: dynamic_pathway_event_table.tsv", fontsize=9.2, fontweight="bold", color="#263238")

    ax_counts.set_title("B. Event-type coverage in standardized table", loc="left", fontsize=11.5, fontweight="bold", pad=10)
    colors = ["#4c78a8", "#59a14f", "#e15759", "#f28e2b", "#b07aa1", "#9c755f"]
    ax_counts.barh(range(len(type_order)), counts.values, color=colors)
    count_labels = ["onset", "peak", "shutdown", "branch-specific", "spatial-localized", "failed-rescue\nprediction-only"]
    ax_counts.set_yticks(range(len(type_order)), labels=count_labels)
    ax_counts.invert_yaxis()
    ax_counts.set_xlabel("event rows")
    ax_counts.text(0.98, 0.06, f"N = {len(events)} standardized event rows", transform=ax_counts.transAxes, ha="right", va="bottom", fontsize=9)
    ax_counts.spines[["top", "right"]].set_visible(False)

    ax_heat.set_title("C. Dataset x event-type coverage", loc="left", fontsize=11.5, fontweight="bold", pad=10)
    matrix = pd.crosstab(events["dataset"], events["event_type"]).reindex(index=datasets, columns=type_order, fill_value=0)
    ax_heat.imshow(matrix.values, aspect="auto", cmap="Blues")
    heat_x_labels = ["onset", "peak", "shutdown", "branch-\nspecific", "spatial-\nlocalized", "failed-\nrescue\npred."]
    dataset_label_map = {
        "GSE123013_Arabidopsis_root": "GSE123013\nArabidopsis root",
        "GSE271399_T21_GATA1s": "GSE271399\nerythroid",
        "Paul15_branch_validation": "Paul15\nbranch",
        "STT0000176_Bombyx_metamorphosis": "Bombyx\nStereo-seq",
    }
    ax_heat.set_xticks(range(len(type_order)), labels=heat_x_labels, rotation=35, ha="right", fontsize=8.5)
    ax_heat.set_yticks(range(len(datasets)), labels=[dataset_label_map.get(d, d.replace("_", "\n")) for d in datasets], fontsize=8.5)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = int(matrix.iloc[i, j])
            ax_heat.text(j, i, str(val), ha="center", va="center", fontsize=9, color="black" if val < 4 else "white")

    ax_table.set_title("D. Required event-row fields", loc="left", fontsize=11.5, fontweight="bold", pad=10)
    fields = [
        "pathway",
        "event_type",
        "event_FDR",
        "event_FDR_available",
        "robustness_score",
        "claim_boundary",
        "supported_interpretation",
        "unsupported_interpretation",
    ]
    y = 0.84
    for field in fields:
        patch = FancyBboxPatch((0.04, y - 0.036), 0.88, 0.062, boxstyle="round,pad=0.012", facecolor="#f5f5f5", edgecolor="#b0b7bd")
        ax_table.add_patch(patch)
        ax_table.text(0.08, y - 0.006, field, ha="left", va="center", fontsize=8.8, fontweight="bold")
        y -= 0.082
    ax_table.text(
        0.04,
        0.18,
        "Scores alone: activity changes.\nTED event rows: event type,\nrobustness, and supported\ninterpretation.",
        fontsize=9.3,
        va="top",
    )

    fig.subplots_adjust(left=0.045, right=0.985, top=0.90, bottom=0.12)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def write_outputs(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    standardize_paul15(rows, Path(args.paul15_dir))
    standardize_bombyx(rows, Path(args.submission_dir))
    standardize_gse271399(rows, Path(args.gse271399_dir))
    standardize_gse123013(rows, Path(args.submission_dir))

    events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    events.to_csv(out / "dynamic_pathway_event_table.tsv", sep="\t", index=False)
    (out / "event_type_definition.md").write_text(event_type_definitions(), encoding="utf-8")
    (out / "event_calling_rules.yaml").write_text(event_calling_rules_yaml(), encoding="utf-8")
    robustness_summary(events).to_csv(out / "event_robustness_summary.tsv", sep="\t", index=False)
    baseline_comparison(Path(args.submission_dir), events).to_csv(out / "baseline_score_vs_event_comparison.tsv", sep="\t", index=False)
    claim_boundary_mapping(events).to_csv(out / "claim_boundary_mapping.tsv", sep="\t", index=False)
    make_event_grammar_figure(events, out / "figure1_event_grammar_overview.pdf")

    manifest = pd.DataFrame(
        [
            {"file": p.name, "bytes": p.stat().st_size}
            for p in sorted(out.iterdir())
            if p.is_file()
        ]
    )
    manifest.to_csv(out / "step_output_manifest.tsv", sep="\t", index=False)
    print(f"Wrote {len(events)} standardized event rows to {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standardize TED/PyFgsea outputs as dynamic pathway event grammar.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory")
    parser.add_argument("--submission-dir", default=str(DEFAULT_SUBMISSION), help="Genome Biology submission output directory")
    parser.add_argument("--paul15-dir", default=str(DEFAULT_PAUL15), help="Paul15 branch validation result directory")
    parser.add_argument("--gse271399-dir", default=str(DEFAULT_GSE271399), help="GSE271399 deliverable directory")
    return parser.parse_args()


if __name__ == "__main__":
    write_outputs(parse_args())
