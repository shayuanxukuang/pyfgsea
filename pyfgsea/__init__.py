from .wrapper import (
    run_gsea,
    load_gmt,
    GseaRunner,
    prepare_pathways,
    get_random_es_means,
)

__version__ = "0.1.4"

try:
    from .wrapper import run_scanpy  # type: ignore
except Exception:
    pass

from .trajectory import (
    GeneSetIndex,
    WindowIndex,
    build_gene_set_index,
    build_window_index,
    run_trajectory_gsea,
)
from .trajectory_events import summarize_events, summarize_pathway_events
from .trajectory_grid import (
    run_ranker_consensus,
    run_trajectory_gsea_grid,
    summarize_event_consensus,
)
from .leading_edge import leading_edge_dynamics, get_dynamic_leading_edge
from .trajectory_compare import (
    calibrate_mixed_effect_events,
    compare_baseline_event_tables,
    compare_event_tables,
    compare_trajectory_gsea_mixed_effect,
    compare_trajectory_gsea_replicate_aware,
    run_branch_contrast_gsea,
    compare_trajectory_gsea,
    run_branch_gsea,
    run_pseudobulk_condition_gsea,
    summarize_matched_balance_diagnostics,
)
from .trajectory_alignment import (
    run_aligned_trajectory_contrast,
    write_aligned_trajectory_contrast,
)
from .trajectory_fate import (
    run_fate_predictive_events,
    write_fate_predictive_events,
)
from .trajectory_modules import (
    discover_dynamic_gene_modules,
    write_dynamic_gene_modules,
)
from .event_graph import build_event_graph, write_event_graph
from .event_drivers import score_event_drivers, write_event_driver_scores
from .event_replication import match_cross_dataset_events, write_cross_dataset_replication
from .phenotype_linked import associate_phenotype_events, write_phenotype_event_association
from .discovery_score import score_biological_discovery, write_biological_discovery_score
from .baselines import run_score_then_smooth_baseline
from .bootstrap import bootstrap_trajectory_gsea
from .validation import validate_inputs, validate_trajectory_result
from .trajectory_benchmark import (
    TRUTH_TYPES,
    make_synthetic_trajectory_truth,
    run_synthetic_truth_benchmark,
    score_synthetic_events,
)
from .trajectory_simulation import (
    DonorTrajectorySimulation,
    simulate_donor_branch_trajectory,
)
from .trajectory_pseudobulk import (
    FixedGridDonorPseudobulkResult,
    run_fixed_grid_donor_pseudobulk,
)
from .trajectory_covariate_pseudobulk import (
    CovariateAdjustedDonorPseudobulkResult,
    CovariateDesignError,
    run_covariate_adjusted_donor_pseudobulk,
)
from .trajectory_covariate_simulation import (
    CovariateAdjustedDesignSimulationResult,
    run_covariate_adjusted_design_simulation,
)
from .t21_covariate_design import run_t21_formal_covariate_pseudobulk
from .trajectory_decomposition import (
    RegulationOccupancyFateResult,
    run_regulation_occupancy_fate_decomposition,
)
from .trajectory_robustness import (
    TrajectoryRobustnessResult,
    summarize_trajectory_robustness,
)
from .trajectory_dynamic_leading_edge import (
    DynamicLeadingEdgeResult,
    decompose_covariate_adjusted_leading_edge,
)
from .trajectory_pathway_families import (
    PathwayFamilyInferenceResult,
    PathwayRedundancyResult,
    cluster_exploratory_pathway_redundancy,
    run_predeclared_pathway_family_inference,
)
from .t21_donor_subset_robustness import (
    T21DonorSubsetRobustnessResult,
    run_t21_donor_subset_robustness,
)
from .t21_pathway_universe import (
    T21PathwayUniverseResult,
    T21PathwayUniverseValidationError,
    build_t21_pathway_universe,
    publish_t21_pathway_universe,
    validate_t21_pathway_universe_outputs,
)
from .t21_pathway_decomposition import (
    T21Chr21EffectComponents,
    T21Chr21PathwayDecomposition,
    build_t21_chr21_pathway_decomposition,
    reconstruct_t21_chr21_effect_components,
)
from .calibration import (
    calibrate_selected_window_statistics,
    calibrate_comparison,
    calibrate_events,
    estimate_event_fdr,
    event_fdr_power_report,
    run_branch_permutation_calibration,
    run_comparison_permutation_calibration,
    run_event_permutation_fdr,
    run_group_comparison_permutation_calibration,
    targeted_directional_calibration,
)
from .design import detect_experimental_design
from .diagnostics import add_window_detection_metrics, technical_confound_diagnostics
from .result import (
    TrajectoryEventResult,
    build_metadata,
    make_trajectory_event_result,
)
from .trajpathmix_scope import (
    CONDITIONAL_TIMING_CAPABILITIES,
    CORE_CAPABILITIES,
    REQUIRED_TIMING_GATES,
    TimingActivationDecision,
    evaluate_timing_activation,
    project_primary_effect_output,
)
from .ted_perturbation import PerturbationEventResult, run_ted_perturbation
from .ted_mad import (
    adjudicate_mechanism,
    design_rescue_experiments,
    generate_decision_report,
    interpret_rescue_results,
    run_negative_control_benchmark,
    run_retrospective_evidence_hiding_benchmark,
    run_synthetic_mechanism_benchmark,
)
from .ted_developmental import (
    TED_DEVELOPMENTAL_OUTPUTS,
    assign_developmental_claim_ceiling,
    build_cross_kingdom_event_ontology,
    classify_ted_delay_modes,
    run_ted_lineage_tree,
    run_ted_multiome_lag,
    run_ted_ot_dynamic,
    run_ted_spatial_neighborhood,
    run_ted_time,
    write_ted_developmental_tables,
)
from .reliability import (
    RELIABILITY_TRUTH_TYPES,
    apply_synthetic_discovery_gate,
    apply_synthetic_gate_sweep,
    run_null_calibration_benchmark,
    run_reliability_ablation_study,
    run_reliability_synthetic_truth_benchmark,
    summarize_synthetic_fpr_breakdown,
)
from .plotting import plot_trajectory_heatmap, plot_pathway_dynamics

# Explicitly expose API to top level
__all__ = [
    "run_gsea",
    "load_gmt",
    "run_scanpy",
    "GseaRunner",
    "prepare_pathways",
    "get_random_es_means",
    "run_trajectory_gsea",
    "GeneSetIndex",
    "WindowIndex",
    "build_gene_set_index",
    "build_window_index",
    "summarize_events",
    "summarize_pathway_events",
    "run_trajectory_gsea_grid",
    "run_ranker_consensus",
    "summarize_event_consensus",
    "leading_edge_dynamics",
    "get_dynamic_leading_edge",
    "calibrate_mixed_effect_events",
    "compare_baseline_event_tables",
    "compare_event_tables",
    "compare_trajectory_gsea_mixed_effect",
    "compare_trajectory_gsea_replicate_aware",
    "run_branch_contrast_gsea",
    "compare_trajectory_gsea",
    "run_branch_gsea",
    "run_pseudobulk_condition_gsea",
    "summarize_matched_balance_diagnostics",
    "run_aligned_trajectory_contrast",
    "write_aligned_trajectory_contrast",
    "run_fate_predictive_events",
    "write_fate_predictive_events",
    "discover_dynamic_gene_modules",
    "write_dynamic_gene_modules",
    "build_event_graph",
    "write_event_graph",
    "score_event_drivers",
    "write_event_driver_scores",
    "match_cross_dataset_events",
    "write_cross_dataset_replication",
    "associate_phenotype_events",
    "write_phenotype_event_association",
    "score_biological_discovery",
    "write_biological_discovery_score",
    "validate_inputs",
    "validate_trajectory_result",
    "TRUTH_TYPES",
    "make_synthetic_trajectory_truth",
    "run_synthetic_truth_benchmark",
    "score_synthetic_events",
    "DonorTrajectorySimulation",
    "simulate_donor_branch_trajectory",
    "FixedGridDonorPseudobulkResult",
    "run_fixed_grid_donor_pseudobulk",
    "CovariateAdjustedDonorPseudobulkResult",
    "CovariateDesignError",
    "run_covariate_adjusted_donor_pseudobulk",
    "run_t21_formal_covariate_pseudobulk",
    "CovariateAdjustedDesignSimulationResult",
    "run_covariate_adjusted_design_simulation",
    "RegulationOccupancyFateResult",
    "run_regulation_occupancy_fate_decomposition",
    "TrajectoryRobustnessResult",
    "summarize_trajectory_robustness",
    "DynamicLeadingEdgeResult",
    "decompose_covariate_adjusted_leading_edge",
    "PathwayFamilyInferenceResult",
    "PathwayRedundancyResult",
    "run_predeclared_pathway_family_inference",
    "cluster_exploratory_pathway_redundancy",
    "T21DonorSubsetRobustnessResult",
    "run_t21_donor_subset_robustness",
    "T21PathwayUniverseResult",
    "T21PathwayUniverseValidationError",
    "build_t21_pathway_universe",
    "publish_t21_pathway_universe",
    "validate_t21_pathway_universe_outputs",
    "T21Chr21EffectComponents",
    "T21Chr21PathwayDecomposition",
    "build_t21_chr21_pathway_decomposition",
    "reconstruct_t21_chr21_effect_components",
    "run_score_then_smooth_baseline",
    "bootstrap_trajectory_gsea",
    "calibrate_comparison",
    "calibrate_selected_window_statistics",
    "calibrate_events",
    "estimate_event_fdr",
    "event_fdr_power_report",
    "run_branch_permutation_calibration",
    "run_comparison_permutation_calibration",
    "run_event_permutation_fdr",
    "run_group_comparison_permutation_calibration",
    "targeted_directional_calibration",
    "detect_experimental_design",
    "add_window_detection_metrics",
    "technical_confound_diagnostics",
    "TrajectoryEventResult",
    "build_metadata",
    "make_trajectory_event_result",
    "CONDITIONAL_TIMING_CAPABILITIES",
    "CORE_CAPABILITIES",
    "REQUIRED_TIMING_GATES",
    "TimingActivationDecision",
    "evaluate_timing_activation",
    "project_primary_effect_output",
    "PerturbationEventResult",
    "run_ted_perturbation",
    "adjudicate_mechanism",
    "design_rescue_experiments",
    "generate_decision_report",
    "interpret_rescue_results",
    "run_negative_control_benchmark",
    "run_retrospective_evidence_hiding_benchmark",
    "run_synthetic_mechanism_benchmark",
    "TED_DEVELOPMENTAL_OUTPUTS",
    "assign_developmental_claim_ceiling",
    "build_cross_kingdom_event_ontology",
    "classify_ted_delay_modes",
    "run_ted_lineage_tree",
    "run_ted_multiome_lag",
    "run_ted_ot_dynamic",
    "run_ted_spatial_neighborhood",
    "run_ted_time",
    "write_ted_developmental_tables",
    "RELIABILITY_TRUTH_TYPES",
    "apply_synthetic_discovery_gate",
    "apply_synthetic_gate_sweep",
    "run_null_calibration_benchmark",
    "run_reliability_ablation_study",
    "run_reliability_synthetic_truth_benchmark",
    "summarize_synthetic_fpr_breakdown",
    "plot_trajectory_heatmap",
    "plot_pathway_dynamics",
]
