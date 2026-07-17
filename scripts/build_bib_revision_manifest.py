from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "bib_manuscript_revision"

PRIMARY_EVIDENCE = [
    "results/ted_submission_calibration/controlled_packet_features.tsv",
    "results/ted_submission_calibration/controlled_truth_metrics.tsv",
    "results/ted_submission_calibration/controlled_truth_key.tsv",
    "results/ted_submission_calibration/ted_packet_predictions.tsv",
    "results/ted_submission_calibration/ambiguity_calibration.tsv",
    "results/ted_submission_calibration/evidence_tier_selective_coverage.tsv",
    "results/ted_submission_calibration/event_fdr_calibration.tsv",
    "results/ted_submission_calibration/confounded_null_calibration.tsv",
    "results/ted_submission_calibration/confounded_signal_calibration.tsv",
    "results/ted_submission_calibration/controlled_packet_class_factorization.tsv",
    "results/ted_submission_calibration/packet_class_confusion_matrix.tsv",
    "results/ted_submission_calibration/failure_modes_and_applicability.tsv",
    "results/ted_submission_calibration/manifest.tsv",
    "results/ted_submission_calibration/run_config.json",
    "results/ted_current_task_benchmark/audit_predictions.tsv",
    "results/ted_current_task_benchmark/baseline_tuning_audit.tsv",
    "results/ted_current_task_benchmark/current_task_confusions.tsv",
    "results/ted_current_task_benchmark/current_task_metrics.tsv",
    "results/ted_current_task_benchmark/e_metric_definitions.tsv",
    "results/ted_current_task_benchmark/current_task_packet_partitions.tsv",
    "results/ted_current_task_benchmark/paired_deltas.tsv",
    "results/ted_current_task_benchmark/run_config.json",
    "results/ted_current_task_benchmark/split_and_leakage_audit.tsv",
    "results/ted_factorized_ablation/factorized_packet_truth.tsv",
    "results/ted_factorized_ablation/factorized_packet_features.tsv",
    "results/ted_factorized_ablation/factorized_predictions.tsv",
    "results/ted_factorized_ablation/factorized_axis_metrics.tsv",
    "results/ted_factorized_ablation/ablation_metrics.tsv",
    "results/ted_factorized_ablation/reason_code_cases.tsv",
    "results/ted_factorized_ablation/reason_code_confusion.tsv",
    "results/ted_factorized_ablation/reason_code_metrics.tsv",
    "results/ted_factorized_ablation/schema_invalid_combination_audit.tsv",
    "results/ted_factorized_ablation/metric_definitions.tsv",
    "results/ted_factorized_ablation/run_config.json",
    "results/ted_factorized_ablation/manifest.tsv",
    "results/ted_adaptive_window_multiplicity/scenario_registry.tsv",
    "results/ted_adaptive_window_multiplicity/replicate_metrics.tsv",
    "results/ted_adaptive_window_multiplicity/event_call_audit.tsv.gz",
    "results/ted_adaptive_window_multiplicity/method_summary.tsv",
    "results/ted_adaptive_window_multiplicity/method_summary_by_stratum.tsv",
    "results/ted_adaptive_window_multiplicity/factor_level_summary.tsv",
    "results/ted_adaptive_window_multiplicity/metric_definitions.tsv",
    "results/ted_adaptive_window_multiplicity/run_config.json",
    "results/ted_submission_supplement/zscape_repeated_holdout_stability/summary.tsv",
    "results/ted_submission_supplement/zscape_repeated_holdout_stability/repeated_20pct_holdout_metrics.tsv",
    "results/ted_submission_supplement/zscape_repeated_holdout_stability/split_half_metrics.tsv",
    "results/ted_submission_supplement/zscape_repeated_holdout_stability/threshold_sensitivity.tsv",
    "results/ted_submission_supplement/zscape_repeated_holdout_stability/run_config.json",
    "results/ted_submission_supplement/zscape_repeated_holdout_stability/event_selection_frequency.tsv",
    "results/ted_submission_supplement/zscape_repeated_holdout_stability/subsampling_curve_long.tsv",
    "results/ted_submission_supplement/zscape_repeated_holdout_stability/subsampling_curve.tsv",
    "results/ted_submission_supplement/zscape_leave_one_embryo_full_refit/summary.tsv",
    "results/ted_submission_supplement/zscape_leave_one_embryo_full_refit/leave_one_embryo_refits.tsv",
    "results/ted_submission_supplement/zscape_leave_one_embryo_full_refit/full_event_table.tsv",
    "results/ted_submission_supplement/zscape_leave_one_embryo_full_refit/gse271399_estimability_audit.tsv",
    "results/ted_submission_supplement/cross_dataset_holdout/cross_dataset_summary.tsv",
    "results/ted_submission_supplement/cross_dataset_holdout/leave_one_dataset_out.tsv",
    "results/ted_submission_supplement/cross_dataset_holdout/cross_dataset_primary_endpoints.tsv",
    "data/processed/ted_known_source/SCP1064/results/scp1064_heavy_shuffle_summary.tsv",
    "data/processed/ted_known_source/SCP1064/results/scp1064_heavy_shuffle_manifest.tsv",
    "data/processed/ted_known_source/SCP1064/results/scp1064_leave_one_unit_refits.tsv",
    "data/processed/ted_known_source/SCP1064/results/scp1064_leave_one_unit_summary.tsv",
    "data/processed/ted_known_source/SCP1064/results/scp1064_random_gene_set_null.tsv",
    "data/processed/ted_known_source/SCP1064/results/scp1064_specificity_summary.tsv",
    "data/processed/ted_known_source/SCP1064/results/scp1064_claim_boundary.tsv",
    "data/processed/ted_known_source/SCP1064/results/scp1064_heavy_shuffle_estimability.tsv",
    "results/ted_submission_supplement/upstream_sensitivity/upstream_sensitivity_summary.tsv",
    "results/ted_submission_supplement/upstream_sensitivity/upstream_method_registry.tsv",
    "results/ted_real_data_upstream_sensitivity/real_data_event_calls.tsv",
    "results/ted_real_data_upstream_sensitivity/real_data_upstream_metrics.tsv",
    "results/ted_real_data_upstream_sensitivity/upstream_event_agreement.tsv",
    "results/ted_real_data_upstream_sensitivity/upstream_method_registry.tsv",
    "results/ted_real_data_upstream_sensitivity/run_config.json",
    "results/ted_real_data_upstream_sensitivity/manifest.tsv",
    "results/ted_post_freeze_protocol_candidate/README.md",
    "results/ted_post_freeze_protocol_candidate/protocol.json",
    "results/ted_post_freeze_protocol_candidate/reporting_and_exclusion_rules.tsv",
    "results/ted_post_freeze_protocol_candidate/activation_checklist.tsv",
    "results/ted_submission_supplement/direct_external_baselines_docker/direct_external_baseline_metric_table.tsv",
    "results/ted_submission_supplement/direct_external_baselines_docker/direct_external_baseline_execution_manifest.tsv",
    "results/ted_submission_supplement/event_layer_scaling/ted_event_layer_scaling.tsv",
    "results/ted_submission_supplement/event_layer_scaling/ted_event_layer_scaling_summary.tsv",
    "results/ted_submission_supplement/event_layer_scaling/environment.json",
    "results/ted_submission_supplement/event_layer_scaling/manifest.tsv",
    "results/ted_submission_supplement/event_layer_scaling/scaling_report.md",
    "results/ted_submission_supplement/event_layer_scaling_quick/ted_event_layer_scaling.tsv",
    "results/ted_submission_supplement/parallel_threads_1/ted_performance_summary.csv",
    "results/ted_submission_supplement/parallel_threads_4/ted_performance_summary.csv",
    "results/ted_known_source_validation/tables/gse153056_pdl1_outcome_alignment.tsv",
    "results/ted_known_source_validation/tables/gse153056_negative_control_results.tsv",
    "results/ted_known_source_validation/tables/gse93735_reversal_index.tsv",
    "results/ted_known_source_validation/tables/gse93735_negative_control_results.tsv",
    "results/bib_manuscript_revision/gse93735_ev_boundary.tsv",
    "results/bib_manuscript_revision/gse271399_design_stratum_audit.tsv",
    "data_external/deliverables_all_ted_rounds/GSE271399_T21_GATA1s/gse271399_family_block_permutation_fdr.tsv",
    "data_external/deliverables_all_ted_rounds/GSE271399_T21_GATA1s/gse271399_block_bootstrap_family_effects.tsv",
    "results/ted_submission_supplement/verification_summary.tsv",
    "results/ted_submission_supplement/ev_v2_verification_2026-07-16.tsv",
    "results/ted_submission_supplement/final_verification_2026-07-16.md",
    "results/ted_submission_supplement/empty_env_validation_demo_20260716/environment.json",
    "results/ted_submission_supplement/empty_env_validation_demo_20260716/cli_activity_validation.tsv",
    "results/ted_submission_supplement/empty_env_validation_demo_20260716/cli_event_v2_validation.tsv",
    "results/ted_submission_supplement/empty_env_validation_demo_20260716/demo_events_v2.tsv",
    "results/ted_submission_supplement/empty_env_validation_demo_20260716/demo_validation.tsv",
    "results/ted_submission_supplement/wheel_dist_ev_v2/pyfgsea-0.1.4-cp38-abi3-win_amd64.whl",
    "results/ted_submission_supplement/evidence_manifest.tsv",
    "results/ted_validation_demo/activity_cli_validation.tsv",
    "results/ted_validation_demo/event_cli_validation.tsv",
    "results/ted_validation_demo/demo_events_v2.tsv",
]

SUPPLEMENTARY_SENSITIVITY = [
    "results/ted_submission_calibration/rule_perturbation_sensitivity_profiles.tsv",
    "results/ted_submission_calibration/rule_perturbation_sensitivity_calls.tsv",
    "results/ted_submission_calibration/rule_perturbation_sensitivity_summary.tsv",
    "data_external/StepXX_dynamic_pathway_event_grammar_standardization/dynamic_pathway_event_table.tsv",
    "data_external/StepXX_dynamic_pathway_event_grammar_standardization/event_type_definition.md",
    "data_external/StepXX_dynamic_pathway_event_grammar_standardization/event_calling_rules.yaml",
    "data_external/StepXX_dynamic_pathway_event_grammar_standardization/event_robustness_summary.tsv",
    "data_external/StepXX_dynamic_pathway_event_grammar_standardization/baseline_score_vs_event_comparison.tsv",
]

REVISION_FILES = [
    "GenomeBiology_known_source_submission_package/06_latex_source/TED_GenomeBiology_Main_Manuscript_Only/main.tex",
    "GenomeBiology_known_source_submission_package/06_latex_source/TED_GenomeBiology_Main_Manuscript_Only/main.pdf",
    "GenomeBiology_known_source_submission_package/06_latex_source/TED_GenomeBiology_Main_Manuscript_Only/references.bib",
    "GenomeBiology_known_source_submission_package/06_latex_source/TED_GenomeBiology_LaTeX_submission/supplementary_files/supplementary_information.tex",
    "GenomeBiology_known_source_submission_package/06_latex_source/TED_GenomeBiology_LaTeX_submission/supplementary_files/supplementary_information.pdf",
    "GenomeBiology_known_source_submission_package/06_latex_source/TED_GenomeBiology_LaTeX_submission/supplementary_files/references.bib",
    "GenomeBiology_known_source_submission_package/06_latex_source/TED_GenomeBiology_LaTeX_submission/supplementary_files/figures/supplementary_figure_s4_dynamic_pathway_event_grammar.pdf",
    "GenomeBiology_known_source_submission_package/06_latex_source/TED_GenomeBiology_LaTeX_submission/supplementary_files/figures/supplementary_figure_s5_scp1064.pdf",
    "BIB_submission_candidate_2026-07-16/README.md",
    "BIB_submission_candidate_2026-07-16/TED_BIB_main_manuscript.pdf",
    "BIB_submission_candidate_2026-07-16/TED_BIB_main_source.zip",
    "BIB_submission_candidate_2026-07-16/TED_BIB_supplementary_information.pdf",
    "scripts/build_bib_main_figures.py",
    "scripts/build_bib_revision_manifest.py",
    "scripts/build_bib_submission_candidate_manifest.py",
    "scripts/build_bib_submission_bundle.py",
    "scripts/build_bib_conservative_ev_audits.py",
    "scripts/run_dynamic_pathway_event_grammar_standardization.py",
    "scripts/222_build_scp1064_validation_figure.py",
    "scripts/run_gse153056_block_aware_validation.py",
    "scripts/run_ted_packet_bootstrap_ci.py",
    "scripts/run_ted_current_task_benchmark.py",
    "scripts/run_ted_factorized_ablation_benchmark.py",
    "scripts/run_ted_adaptive_window_multiplicity_benchmark.py",
    "scripts/run_zscape_repeated_holdout_stability.py",
    "scripts/run_ted_real_data_upstream_sensitivity.py",
    "scripts/run_ted_event_layer_scaling.py",
    "scripts/run_ted_validation_demo.py",
    "pyfgsea/ted_schema.py",
    "pyfgsea/calibration.py",
    "pyfgsea/ted_evidence.py",
    "pyfgsea/cli/main.py",
    "pyfgsea/schemas/ted_event_report_v2.schema.json",
    "schemas/ted_event_report_v2.schema.json",
    "tests/test_ted_schema.py",
    "tests/test_calibration.py",
    "tests/test_ted_evidence.py",
    "docs/output_schema.md",
    "docs/ted_validation_demo.md",
    "docs/bib_figure_chart_map.md",
    "docs/ted_bib_revision_log_2026-07-16.md",
    "docs/ted_pre_submission_revision_tracking_2026-07-16.md",
    "docs/bib_submission_readiness_2026-07-16.md",
    "docs/ted_public_release_audit_2026-07-16.md",
    "docs/docker_ci_readiness_2026-07-16.md",
    "README.md",
    "config/ted_external_validation_protocol_v1.yaml",
    "results/ted_submission_supplement/requested_experiment_audit/requested_experiment_support_audit.tsv",
    "results/bib_manuscript_revision/evidence_axis_legacy_crosswalk.tsv",
    "results/bib_manuscript_revision/manuscript_metric_source_map.tsv",
    "results/bib_manuscript_revision/graphical_abstract_alt_text.txt",
    "results/bib_manuscript_revision/figure_manifest.tsv",
]

REVISION_DATA_DIRS = [
    ("manuscript_experiment", "results/bib_manuscript_revision/gse153056_block_aware"),
    ("manuscript_experiment", "results/bib_manuscript_revision/packet_bootstrap"),
    ("test_data", "results/ted_validation_demo"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_record = {
        "release_preparation_date": "2026-07-16",
        "figure_command": "python scripts/build_bib_main_figures.py",
        "manifest_command": "python scripts/build_bib_revision_manifest.py",
        "python_version": platform.python_version(),
        "figure_source_directory": "results/bib_manuscript_revision/figure_source_data",
        "canonical_main_source": "GenomeBiology_known_source_submission_package/06_latex_source/TED_GenomeBiology_Main_Manuscript_Only/main.tex",
    }
    record_path = OUT / "build_record.json"
    legacy_record_path = OUT / "revision_build_record.json"
    record_text = json.dumps(build_record, indent=2) + "\n"
    record_path.write_text(record_text, encoding="utf-8")
    legacy_record_path.write_text(record_text, encoding="utf-8")

    grouped: list[tuple[str, Path]] = []
    grouped.extend(("primary_evidence", ROOT / rel) for rel in PRIMARY_EVIDENCE)
    grouped.extend(("supplementary_sensitivity", ROOT / rel) for rel in SUPPLEMENTARY_SENSITIVITY)
    grouped.extend(("manuscript_source", ROOT / rel) for rel in REVISION_FILES)
    for group, rel in REVISION_DATA_DIRS:
        grouped.extend((group, path) for path in sorted((ROOT / rel).rglob("*")) if path.is_file())
    grouped.append(("manuscript_source", record_path))
    grouped.append(("manuscript_source", legacy_record_path))
    grouped.extend(("figure_source_data", path) for path in sorted((OUT / "figure_source_data").glob("*.tsv")))
    grouped.extend(("figure_output", path) for path in sorted((OUT / "figures").glob("*.*")))

    missing = [path for _, path in grouped if not path.is_file()]
    if missing:
        formatted = "\n".join(str(path.relative_to(ROOT)) for path in missing)
        raise FileNotFoundError(f"Required manuscript evidence files are missing:\n{formatted}")

    rows = [
        {
            "evidence_group": group,
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for group, path in grouped
    ]
    frame = pd.DataFrame(rows).drop_duplicates(subset=["path"]).sort_values(["evidence_group", "path"])
    neutral_manifest = OUT / "evidence_manifest.tsv"
    legacy_manifest = OUT / "revision_evidence_manifest.tsv"
    frame.to_csv(neutral_manifest, sep="\t", index=False)
    frame.to_csv(legacy_manifest, sep="\t", index=False)
    print(f"Wrote {len(frame)} manuscript evidence entries to {neutral_manifest}")


if __name__ == "__main__":
    main()
