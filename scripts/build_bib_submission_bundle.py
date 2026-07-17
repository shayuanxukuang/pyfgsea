from __future__ import annotations

"""Build the current BIB submission bundle without invoking legacy rc7 builders.

The workflow is deliberately split in two:

1. ``--stage`` synchronizes the authoritative manuscript and supplement sources,
   current figures, and small source-data mirrors.  It also writes the clean,
   deterministic main-source ZIP.
2. The caller regenerates ``results/bib_manuscript_revision/evidence_manifest.tsv``.
3. ``--finalize`` verifies every hash in that manifest before writing anything,
   then creates Additional file 3 and the deterministic LaTeX package archives.

The script uses explicit evidence and figure allowlists.  It never imports,
executes, or modifies the historical Genome Biology/rc7 package generators.
"""

import argparse
import csv
import hashlib
import io
import os
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]

KNOWN_SOURCE_ROOT = ROOT / "GenomeBiology_known_source_submission_package"
UPLOAD_MAIN = KNOWN_SOURCE_ROOT / "01_main_manuscript"
KNOWN_LATEX_ROOT = KNOWN_SOURCE_ROOT / "06_latex_source"
AUTHORITATIVE_MAIN = KNOWN_LATEX_ROOT / "TED_GenomeBiology_Main_Manuscript_Only"
CANONICAL_FULL = KNOWN_LATEX_ROOT / "TED_GenomeBiology_LaTeX_submission"
AUTHORITATIVE_SUPPLEMENT = CANONICAL_FULL / "supplementary_files"

LATEX_MIRROR_ROOT = ROOT / "latex_submission_package"
FULL_MIRROR = LATEX_MIRROR_ROOT / "TED_GenomeBiology_LaTeX_submission"

CANDIDATE = ROOT / "BIB_submission_candidate_2026-07-16"
CANDIDATE_MAIN_SOURCE = CANDIDATE / "main_source"
CANDIDATE_SUPPLEMENT = CANDIDATE / "supplement"
CANDIDATE_FIGURES = CANDIDATE / "figures"

REVISION_RESULTS = ROOT / "results" / "bib_manuscript_revision"
FIGURE_RESULTS = REVISION_RESULTS / "figures"
FIGURE_SOURCE = REVISION_RESULTS / "figure_source_data"
EVIDENCE_MANIFEST = REVISION_RESULTS / "evidence_manifest.tsv"

HEAVY_ROOT = ROOT / "data" / "processed" / "ted_known_source" / "SCP1064" / "results"
HEAVY_MANIFEST = HEAVY_ROOT / "scp1064_heavy_shuffle_manifest.tsv"

ADDITIONAL3_NAME = "Additional_file_3_Machine_Readable_Files.zip"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
MAX_EVIDENCE_FILE_BYTES = 5 * 1024 * 1024
MAX_ADDITIONAL3_PAYLOAD_BYTES = 25 * 1024 * 1024

EXCLUDED_BUILD_ENDINGS = (
    ".aux",
    ".log",
    ".fls",
    ".fdb_latexmk",
    ".blg",
    ".out",
    ".synctex.gz",
    ".toc",
)
IMAGE_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".svg"}

MAIN_FIGURE_STEMS = (
    "figure1_problem_definition_ted_algorithm",
    "figure2_frozen_benchmark_design",
    "figure3_primary_heldout_performance",
    "figure4_robustness_portability_scalability",
    "figure5_independent_real_data_validation",
)
PUBLIC_FIGURE_FILES = tuple(
    f"{stem}.{suffix}" for stem in MAIN_FIGURE_STEMS for suffix in ("pdf", "png")
) + (
    "graphical_abstract.pdf",
    "graphical_abstract.png",
    "graphical_abstract_alt_text.txt",
)
MAIN_SOURCE_FIGURE_FILES = tuple(f"{stem}.pdf" for stem in MAIN_FIGURE_STEMS)

SUPPLEMENT_FIGURE_FILES = (
    "parameter_sensitivity_multiverse_figure.png",
    "supplementary_figure_s2_bombyx_workflow_localization.png",
    "supplementary_figure_s3_bombyx_controls_windows.png",
    "supplementary_figure_s4_dynamic_pathway_event_grammar.pdf",
    "supplementary_figure_s5_scp1064.pdf",
)
ROOT_SOURCE_REFRESH = {
    "rule_perturbation_sensitivity_profiles.tsv": ROOT
    / "results"
    / "ted_submission_calibration"
    / "rule_perturbation_sensitivity_profiles.tsv",
    "rule_perturbation_sensitivity_calls.tsv": ROOT
    / "results"
    / "ted_submission_calibration"
    / "rule_perturbation_sensitivity_calls.tsv",
    "rule_perturbation_sensitivity_summary.tsv": ROOT
    / "results"
    / "ted_submission_calibration"
    / "rule_perturbation_sensitivity_summary.tsv",
    "gse153056_stat1_gate_audit.tsv": ROOT
    / "results"
    / "bib_manuscript_revision"
    / "gse153056_block_aware"
    / "gse153056_stat1_gate_audit.tsv",
    "gse93735_ev_boundary.tsv": ROOT
    / "results"
    / "bib_manuscript_revision"
    / "gse93735_ev_boundary.tsv",
    "gse271399_design_stratum_audit.tsv": ROOT
    / "results"
    / "bib_manuscript_revision"
    / "gse271399_design_stratum_audit.tsv",
    "dynamic_pathway_event_table.tsv": ROOT
    / "data_external"
    / "StepXX_dynamic_pathway_event_grammar_standardization"
    / "dynamic_pathway_event_table.tsv",
    "event_type_definition.md": ROOT
    / "data_external"
    / "StepXX_dynamic_pathway_event_grammar_standardization"
    / "event_type_definition.md",
    "event_calling_rules.yaml": ROOT
    / "data_external"
    / "StepXX_dynamic_pathway_event_grammar_standardization"
    / "event_calling_rules.yaml",
    "event_robustness_summary.tsv": ROOT
    / "data_external"
    / "StepXX_dynamic_pathway_event_grammar_standardization"
    / "event_robustness_summary.tsv",
    "baseline_score_vs_event_comparison.tsv": ROOT
    / "data_external"
    / "StepXX_dynamic_pathway_event_grammar_standardization"
    / "baseline_score_vs_event_comparison.tsv",
}
CURRENT_IMAGE_NAMES = {
    name.casefold()
    for name in PUBLIC_FIGURE_FILES + SUPPLEMENT_FIGURE_FILES
    if Path(name).suffix.casefold() in IMAGE_SUFFIXES
}

HEAVY_FILE_NAMES = (
    "scp1064_heavy_shuffle_estimability.tsv",
    "scp1064_heavy_shuffle_summary.tsv",
    "scp1064_leave_one_unit_refits.tsv",
    "scp1064_leave_one_unit_summary.tsv",
    "scp1064_random_gene_set_null.tsv",
    "scp1064_specificity_summary.tsv",
)

# Explicitly small, submission-relevant July 2026 evidence.  No wheel, PDF,
# cell-level score matrix, raw archive, or recursively discovered result is
# eligible for Additional file 3.
SUBMISSION_EVIDENCE_RELATIVE = (
    # Controlled truth, packet factorization, E calibration and rule sensitivity.
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
    "results/ted_submission_calibration/rule_perturbation_sensitivity_profiles.tsv",
    "results/ted_submission_calibration/rule_perturbation_sensitivity_calls.tsv",
    "results/ted_submission_calibration/rule_perturbation_sensitivity_summary.tsv",
    # Dynamic-event grammar source tables cited in Supplementary Figure S4.
    "data_external/StepXX_dynamic_pathway_event_grammar_standardization/dynamic_pathway_event_table.tsv",
    "data_external/StepXX_dynamic_pathway_event_grammar_standardization/event_type_definition.md",
    "data_external/StepXX_dynamic_pathway_event_grammar_standardization/event_calling_rules.yaml",
    "data_external/StepXX_dynamic_pathway_event_grammar_standardization/event_robustness_summary.tsv",
    "data_external/StepXX_dynamic_pathway_event_grammar_standardization/baseline_score_vs_event_comparison.tsv",
    # Leakage-audited current-task baselines and E-risk comparison.
    "results/ted_current_task_benchmark/audit_predictions.tsv",
    "results/ted_current_task_benchmark/baseline_tuning_audit.tsv",
    "results/ted_current_task_benchmark/current_task_confusions.tsv",
    "results/ted_current_task_benchmark/current_task_metrics.tsv",
    "results/ted_current_task_benchmark/e_metric_definitions.tsv",
    "results/ted_current_task_benchmark/current_task_packet_partitions.tsv",
    "results/ted_current_task_benchmark/paired_deltas.tsv",
    "results/ted_current_task_benchmark/run_config.json",
    "results/ted_current_task_benchmark/split_and_leakage_audit.tsv",
    # Primary factorized dynamic benchmark, one-gate ablations and reason-code audit.
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
    # Full block-profile adaptive-window multiplicity benchmark.
    "results/ted_adaptive_window_multiplicity/scenario_registry.tsv",
    "results/ted_adaptive_window_multiplicity/replicate_metrics.tsv",
    "results/ted_adaptive_window_multiplicity/event_call_audit.tsv.gz",
    "results/ted_adaptive_window_multiplicity/method_summary.tsv",
    "results/ted_adaptive_window_multiplicity/method_summary_by_stratum.tsv",
    "results/ted_adaptive_window_multiplicity/factor_level_summary.tsv",
    "results/ted_adaptive_window_multiplicity/metric_definitions.tsv",
    "results/ted_adaptive_window_multiplicity/run_config.json",
    # Packet bootstrap and dataset-level holdout evidence.
    "results/bib_manuscript_revision/packet_bootstrap/packet_bootstrap_replicates.tsv.gz",
    "results/bib_manuscript_revision/packet_bootstrap/packet_bootstrap_run_config.tsv",
    "results/bib_manuscript_revision/packet_bootstrap/packet_bootstrap_summary.tsv",
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
    # Exact STAT1 gate reconstruction and public outcome/reversal evidence.
    "results/bib_manuscript_revision/gse153056_block_aware/gse153056_block_event_support.tsv",
    "results/bib_manuscript_revision/gse153056_block_aware/gse153056_block_summary.tsv",
    "results/bib_manuscript_revision/gse153056_block_aware/gse153056_replicate_effects.tsv",
    "results/bib_manuscript_revision/gse153056_block_aware/gse153056_stat1_gate_audit.tsv",
    "results/bib_manuscript_revision/gse271399_design_stratum_audit.tsv",
    "results/bib_manuscript_revision/gse93735_ev_boundary.tsv",
    "results/ted_known_source_validation/tables/gse153056_pdl1_outcome_alignment.tsv",
    "results/ted_known_source_validation/tables/gse153056_negative_control_results.tsv",
    "results/ted_known_source_validation/tables/gse93735_reversal_index.tsv",
    "results/ted_known_source_validation/tables/gse93735_negative_control_results.tsv",
    "data_external/deliverables_all_ted_rounds/GSE271399_T21_GATA1s/gse271399_family_block_permutation_fdr.tsv",
    "data_external/deliverables_all_ted_rounds/GSE271399_T21_GATA1s/gse271399_block_bootstrap_family_effects.tsv",
    # Upstream sensitivity, executable baselines and scaling.
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
    # Executable schema/demo verification and source-to-claim maps.
    "results/ted_submission_supplement/verification_summary.tsv",
    "results/ted_submission_supplement/ev_v2_verification_2026-07-16.tsv",
    "results/ted_submission_supplement/final_verification_2026-07-16.md",
    "results/ted_submission_supplement/empty_env_validation_demo_20260716/environment.json",
    "results/ted_submission_supplement/empty_env_validation_demo_20260716/cli_activity_validation.tsv",
    "results/ted_submission_supplement/empty_env_validation_demo_20260716/cli_event_v2_validation.tsv",
    "results/ted_submission_supplement/empty_env_validation_demo_20260716/demo_events_v2.tsv",
    "results/ted_submission_supplement/empty_env_validation_demo_20260716/demo_validation.tsv",
    "results/ted_validation_demo/activity_cli_validation.tsv",
    "results/ted_validation_demo/event_cli_validation.tsv",
    "results/ted_validation_demo/demo_events_v2.tsv",
    "schemas/ted_event_report_v2.schema.json",
    "results/bib_manuscript_revision/evidence_axis_legacy_crosswalk.tsv",
    "results/bib_manuscript_revision/manuscript_metric_source_map.tsv",
    "results/bib_manuscript_revision/figure_manifest.tsv",
    "results/ted_submission_supplement/requested_experiment_audit/requested_experiment_support_audit.tsv",
    "config/ted_external_validation_protocol_v1.yaml",
    "docs/ted_pre_submission_revision_tracking_2026-07-16.md",
)

ALLOWED_EVIDENCE_ENDINGS = (
    ".tsv",
    ".tsv.gz",
    ".csv",
    ".json",
    ".md",
    ".yaml",
    ".yml",
)


class BundleError(RuntimeError):
    """Raised for a reproducibility or package-safety failure."""


@dataclass(frozen=True)
class ManifestRecord:
    relative_path: str
    path: Path
    size: int
    sha256: str


@dataclass(frozen=True)
class ZipPayload:
    archive_path: str
    source_label: str
    source: Optional[Path] = None
    data: Optional[bytes] = None

    def read_bytes(self) -> bytes:
        if (self.source is None) == (self.data is None):
            raise BundleError(
                f"ZIP payload {self.archive_path!r} must have exactly one source"
            )
        if self.source is not None:
            return self.source.read_bytes()
        assert self.data is not None
        return self.data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_relative(value: str, *, label: str) -> str:
    if not value or "\\" in value:
        raise BundleError(f"Unsafe {label} path {value!r}; POSIX relative paths are required")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise BundleError(f"Unsafe {label} path {value!r}")
    return pure.as_posix()


def repo_file(relative: str) -> Path:
    normalized = normalize_relative(relative, label="repository")
    path = (ROOT / Path(*PurePosixPath(normalized).parts)).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise BundleError(f"Repository path escapes the workspace: {relative!r}")
    return path


def require_files(paths: Iterable[Path], *, context: str) -> None:
    issues: List[str] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not path.exists():
            issues.append(f"missing: {path.relative_to(ROOT)}")
        elif not path.is_file():
            issues.append(f"not a regular file: {path.relative_to(ROOT)}")
        elif path.stat().st_size == 0:
            issues.append(f"empty: {path.relative_to(ROOT)}")
    if issues:
        details = "\n  - ".join(issues)
        raise BundleError(f"{context} failed:\n  - {details}")


def ensure_output_path(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    root = ROOT.resolve()
    if absolute == root or not absolute.is_relative_to(root):
        raise BundleError(f"Refusing to write outside the workspace: {path}")
    for component in (absolute, *absolute.parents):
        if component == root:
            break
        if component.exists() and component.is_symlink():
            raise BundleError(f"Refusing to write through a symlink: {component}")
    resolved = absolute.resolve()
    if not resolved.is_relative_to(root):
        raise BundleError(f"Resolved output path escapes the workspace: {path}")
    return absolute


def atomic_copy(source: Path, destination: Path) -> None:
    require_files([source], context="Copy preflight")
    destination = ensure_output_path(destination)
    if source.resolve() == destination:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise BundleError(f"Refusing to replace symlink: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_bytes(destination: Path, data: bytes) -> None:
    destination = ensure_output_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise BundleError(f"Refusing to replace symlink: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def managed_directories() -> set:
    return {
        path.resolve()
        for path in (
            AUTHORITATIVE_MAIN / "figures",
            UPLOAD_MAIN,
            AUTHORITATIVE_SUPPLEMENT / "figures",
            CANONICAL_FULL / "figures",
            FULL_MIRROR / "figures",
            FULL_MIRROR / "supplementary_files" / "figures",
            CANDIDATE_MAIN_SOURCE,
            CANDIDATE_SUPPLEMENT,
            CANDIDATE_FIGURES,
            CANDIDATE / "schema",
            CANDIDATE / "test_data",
            CANDIDATE / "evidence",
            KNOWN_SOURCE_ROOT / "03_figures",
            KNOWN_SOURCE_ROOT / "03_figures_final_upload",
            KNOWN_SOURCE_ROOT / "05_source_data_and_audits",
            KNOWN_SOURCE_ROOT / "05_source_data_and_audits" / "figure_source_data",
            KNOWN_SOURCE_ROOT / "05_source_data_and_audits" / "july_2026_evidence",
            CANONICAL_FULL / "tables" / "figure_source_data",
            CANONICAL_FULL / "tables" / "july_2026_evidence",
            CANONICAL_FULL / "tables",
            FULL_MIRROR / "tables" / "figure_source_data",
            FULL_MIRROR / "tables" / "july_2026_evidence",
            FULL_MIRROR / "tables",
        )
    }


def sync_exact_directory(destination: Path, sources: Mapping[str, Path]) -> None:
    destination = ensure_output_path(destination)
    if destination not in managed_directories():
        raise BundleError(f"Directory is not in the exact-sync allowlist: {destination}")
    normalized_sources: Dict[str, Path] = {}
    for relative, source in sources.items():
        normalized = normalize_relative(relative, label="managed-directory")
        if normalized.casefold() in {item.casefold() for item in normalized_sources}:
            raise BundleError(f"Duplicate managed path for {destination}: {normalized}")
        normalized_sources[normalized] = source
    if not normalized_sources:
        raise BundleError(f"Refusing to synchronize an empty directory: {destination}")
    require_files(normalized_sources.values(), context=f"Source preflight for {destination}")

    if destination.exists() and not destination.is_dir():
        raise BundleError(f"Managed destination is not a directory: {destination}")
    if destination.is_symlink():
        raise BundleError(f"Managed destination must not be a symlink: {destination}")

    existing: List[Path] = []
    if destination.exists():
        existing = list(destination.rglob("*"))
        symlinks = [path for path in existing if path.is_symlink()]
        if symlinks:
            rendered = "\n  - ".join(str(path) for path in symlinks)
            raise BundleError(
                f"Refusing exact synchronization across symlinks:\n  - {rendered}"
            )

    destination.mkdir(parents=True, exist_ok=True)
    for relative, source in sorted(normalized_sources.items()):
        target = destination / Path(*PurePosixPath(relative).parts)
        atomic_copy(source, target)

    desired = {item.casefold() for item in normalized_sources}
    for path in sorted(
        (item for item in destination.rglob("*") if item.is_file()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        relative = path.relative_to(destination).as_posix()
        if relative.casefold() not in desired:
            path.unlink()
    for path in sorted(
        (item for item in destination.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        try:
            path.rmdir()
        except OSError:
            pass


def is_excluded_build_file(name: str) -> bool:
    lowered = name.casefold()
    return (
        lowered.startswith("~$")
        or lowered.startswith(".~lock.")
        or lowered.endswith(EXCLUDED_BUILD_ENDINGS)
    )


def is_obsolete_figure(path: PurePosixPath) -> bool:
    name = path.name.casefold()
    if Path(name).suffix.casefold() not in IMAGE_SUFFIXES:
        return False
    figure_like = name.startswith(
        ("figure", "supplementary_figure", "graphical_abstract", "parameter_sensitivity")
    )
    return figure_like and name not in CURRENT_IMAGE_NAMES


def should_include_in_latex_archive(relative: str, *, include_manifest: bool = True) -> bool:
    normalized = normalize_relative(relative, label="archive")
    pure = PurePosixPath(normalized)
    lowered_parts = [part.casefold() for part in pure.parts]
    if any(part.startswith("_preview") for part in lowered_parts):
        return False
    if any(part in {"__pycache__", ".git"} for part in lowered_parts):
        return False
    if is_excluded_build_file(pure.name):
        return False
    if pure.name.casefold().endswith(".tmp"):
        return False
    if not include_manifest and pure.name.casefold() == "latex_package_manifest.tsv":
        return False
    if is_obsolete_figure(pure):
        return False
    return True


def collect_tree_payloads(
    base: Path, *, include_manifest: bool = True
) -> List[ZipPayload]:
    if not base.is_dir():
        raise BundleError(f"Archive source directory is missing: {base}")
    payloads: List[ZipPayload] = []
    for path in sorted(base.rglob("*")):
        if path.is_symlink():
            raise BundleError(f"Archive source must not contain symlinks: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(base).as_posix()
        if should_include_in_latex_archive(relative, include_manifest=include_manifest):
            payloads.append(
                ZipPayload(
                    archive_path=relative,
                    source_label=path.relative_to(ROOT).as_posix(),
                    source=path,
                )
            )
    if not payloads:
        raise BundleError(f"No eligible files found for archive source: {base}")
    return payloads


def write_deterministic_zip(destination: Path, payloads: Sequence[ZipPayload]) -> None:
    destination = ensure_output_path(destination)
    if not payloads:
        raise BundleError(f"Refusing to write an empty ZIP: {destination}")

    normalized: Dict[str, ZipPayload] = {}
    folded_names = set()
    for payload in payloads:
        archive_path = normalize_relative(payload.archive_path, label="ZIP member")
        folded = archive_path.casefold()
        if folded in folded_names:
            raise BundleError(f"Duplicate ZIP member: {archive_path}")
        folded_names.add(folded)
        normalized[archive_path] = payload

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            allowZip64=True,
        ) as archive:
            for archive_path in sorted(normalized):
                payload = normalized[archive_path]
                info = zipfile.ZipInfo(archive_path, date_time=FIXED_ZIP_TIMESTAMP)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                info.flag_bits |= 0x800
                archive.writestr(info, payload.read_bytes(), compresslevel=9)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def figure_payload_map() -> Dict[str, Path]:
    mapping = {
        name: (REVISION_RESULTS / "graphical_abstract_alt_text.txt")
        if name.endswith(".txt")
        else FIGURE_RESULTS / name
        for name in PUBLIC_FIGURE_FILES
    }
    require_files(mapping.values(), context="Current main-figure preflight")
    return mapping


def supplement_figure_map() -> Dict[str, Path]:
    mapping = {
        name: AUTHORITATIVE_SUPPLEMENT / "figures" / name
        for name in SUPPLEMENT_FIGURE_FILES
    }
    mapping["supplementary_figure_s4_dynamic_pathway_event_grammar.pdf"] = (
        ROOT
        / "data_external"
        / "StepXX_dynamic_pathway_event_grammar_standardization"
        / "figure1_event_grammar_overview.pdf"
    )
    mapping["supplementary_figure_s5_scp1064.pdf"] = (
        ROOT
        / "results"
        / "ted_known_source_validation"
        / "figures"
        / "supplementary_figure_scp1064.pdf"
    )
    require_files(mapping.values(), context="Supplementary-figure preflight")
    return mapping


def figure_source_map() -> Dict[str, Path]:
    if not FIGURE_SOURCE.is_dir():
        raise BundleError(f"Figure source-data directory is missing: {FIGURE_SOURCE}")
    mapping = {path.name: path for path in sorted(FIGURE_SOURCE.glob("*.tsv"))}
    if not mapping:
        raise BundleError(f"No figure source-data TSV files found under {FIGURE_SOURCE}")
    require_files(mapping.values(), context="Figure source-data preflight")
    return mapping


def evidence_source_map() -> Dict[str, Path]:
    mapping = {relative: repo_file(relative) for relative in SUBMISSION_EVIDENCE_RELATIVE}
    for name in HEAVY_FILE_NAMES:
        relative = (HEAVY_ROOT / name).relative_to(ROOT).as_posix()
        mapping[relative] = HEAVY_ROOT / name
    require_files(mapping.values(), context="July 2026 evidence preflight")
    return mapping


def main_source_map() -> Dict[str, Path]:
    mapping = {
        "main.tex": AUTHORITATIVE_MAIN / "main.tex",
        "main.bbl": AUTHORITATIVE_MAIN / "main.bbl",
        "references.bib": AUTHORITATIVE_MAIN / "references.bib",
        "README_compile.txt": AUTHORITATIVE_MAIN / "README_compile.txt",
    }
    mapping.update(
        {f"figures/{name}": FIGURE_RESULTS / name for name in MAIN_SOURCE_FIGURE_FILES}
    )
    return mapping


def upload_main_map() -> Dict[str, Path]:
    mapping = {
        "main_manuscript_with_figures.tex": AUTHORITATIVE_MAIN / "main.tex",
        "main_manuscript_with_figures.pdf": AUTHORITATIVE_MAIN / "main.pdf",
        "main_manuscript_with_figures.bbl": AUTHORITATIVE_MAIN / "main.bbl",
        "references.bib": AUTHORITATIVE_MAIN / "references.bib",
        "README_compile.txt": AUTHORITATIVE_MAIN / "README_compile.txt",
    }
    mapping.update(
        {
            f"figures/{name}": AUTHORITATIVE_MAIN / "figures" / name
            for name in MAIN_SOURCE_FIGURE_FILES
        }
    )
    require_files(mapping.values(), context="Loose upload-main preflight")
    return mapping


def supplement_source_map() -> Dict[str, Path]:
    mapping = {
        "supplementary_information.tex": AUTHORITATIVE_SUPPLEMENT
        / "supplementary_information.tex",
        "supplementary_information.bbl": AUTHORITATIVE_SUPPLEMENT
        / "supplementary_information.bbl",
        "supplementary_information.pdf": AUTHORITATIVE_SUPPLEMENT
        / "supplementary_information.pdf",
        "references.bib": AUTHORITATIVE_SUPPLEMENT / "references.bib",
        "README_compile.txt": CANONICAL_FULL / "README_compile.txt",
    }
    mapping.update(
        {
            f"figures/{name}": AUTHORITATIVE_SUPPLEMENT / "figures" / name
            for name in SUPPLEMENT_FIGURE_FILES
        }
    )
    return mapping


def figure_package_map(figures: Mapping[str, Path]) -> Dict[str, Path]:
    mapping = dict(figures)
    upload_manifest = (
        KNOWN_SOURCE_ROOT / "03_figures_final_upload" / "final_main_figure_upload_manifest.tsv"
    )
    require_files([upload_manifest], context="Current figure-upload manifest preflight")
    mapping[upload_manifest.name] = upload_manifest
    return mapping


def candidate_schema_map() -> Dict[str, Path]:
    names = (
        "ted_activity_table_v1.schema.json",
        "ted_event_report_v1.schema.json",
        "ted_event_report_v2.schema.json",
    )
    return {name: ROOT / "schemas" / name for name in names}


def candidate_test_data_map() -> Dict[str, Path]:
    base = ROOT / "results" / "ted_validation_demo"
    mapping = {
        f"ted_validation_demo/{path.name}": path
        for path in sorted(base.glob("*"))
        if path.is_file()
    }
    if not mapping:
        raise BundleError(f"Validation-demo output is missing: {base}")
    return mapping


def stage() -> None:
    main_files = {
        name: AUTHORITATIVE_MAIN / name
        for name in ("main.tex", "main.pdf", "main.bbl", "references.bib", "README_compile.txt")
    }
    supplement_files = {
        name: AUTHORITATIVE_SUPPLEMENT / name
        for name in (
            "supplementary_information.tex",
            "supplementary_information.pdf",
            "supplementary_information.bbl",
            "references.bib",
            "Additional_file_2_Supplementary_Tables.xlsx",
        )
    }
    require_files(main_files.values(), context="Authoritative main-manuscript preflight")
    require_files(
        supplement_files.values(), context="Authoritative Supplementary Information preflight"
    )

    figures = figure_payload_map()
    packaged_figures = figure_package_map(figures)
    supplement_figures = supplement_figure_map()
    source_data = figure_source_map()
    evidence = evidence_source_map()
    require_files(main_source_map().values(), context="Main source-package preflight")
    require_files(supplement_source_map().values(), context="Supplement source preflight")

    # Exact-sync only explicitly managed generated directories.  This removes
    # old main figures and LaTeX build debris from the candidate source trees.
    sync_exact_directory(AUTHORITATIVE_MAIN / "figures", figures)
    sync_exact_directory(UPLOAD_MAIN, upload_main_map())
    sync_exact_directory(CANONICAL_FULL / "figures", figures)
    sync_exact_directory(FULL_MIRROR / "figures", figures)
    sync_exact_directory(AUTHORITATIVE_SUPPLEMENT / "figures", supplement_figures)
    sync_exact_directory(FULL_MIRROR / "supplementary_files" / "figures", supplement_figures)
    sync_exact_directory(CANDIDATE_MAIN_SOURCE, main_source_map())
    sync_exact_directory(CANDIDATE_SUPPLEMENT, supplement_source_map())
    sync_exact_directory(CANDIDATE_FIGURES, figures)
    sync_exact_directory(KNOWN_SOURCE_ROOT / "03_figures", packaged_figures)
    sync_exact_directory(KNOWN_SOURCE_ROOT / "03_figures_final_upload", packaged_figures)
    sync_exact_directory(CANDIDATE / "schema", candidate_schema_map())
    sync_exact_directory(CANDIDATE / "test_data", candidate_test_data_map())

    root_source_map = {
        **ROOT_SOURCE_REFRESH,
        **{f"figure_source_data/{name}": path for name, path in source_data.items()},
        **{f"july_2026_evidence/{name}": path for name, path in evidence.items()},
    }
    sync_exact_directory(
        KNOWN_SOURCE_ROOT / "05_source_data_and_audits", root_source_map
    )
    clean_table_map = {
        **{f"figure_source_data/{name}": path for name, path in source_data.items()},
        **{f"july_2026_evidence/{name}": path for name, path in evidence.items()},
    }
    sync_exact_directory(CANONICAL_FULL / "tables", clean_table_map)
    sync_exact_directory(FULL_MIRROR / "tables", clean_table_map)

    copy_pairs: List[Tuple[Path, Path]] = []
    for name, source in main_files.items():
        copy_pairs.extend(((source, CANONICAL_FULL / name), (source, FULL_MIRROR / name)))
    copy_pairs.extend(
        (
            (main_files["main.pdf"], CANDIDATE / "TED_BIB_main_manuscript.pdf"),
            (
                supplement_files["supplementary_information.pdf"],
                CANDIDATE / "TED_BIB_supplementary_information.pdf",
            ),
            (
                supplement_files["supplementary_information.pdf"],
                AUTHORITATIVE_SUPPLEMENT / "Additional_file_1_Supplementary_Information.pdf",
            ),
            (
                supplement_files["supplementary_information.pdf"],
                FULL_MIRROR / "supplementary_information.pdf",
            ),
            (
                supplement_files["supplementary_information.tex"],
                FULL_MIRROR / "supplementary_information.tex",
            ),
            (
                supplement_files["supplementary_information.bbl"],
                FULL_MIRROR / "supplementary_information.bbl",
            ),
        )
    )
    for name in (
        "supplementary_information.tex",
        "supplementary_information.pdf",
        "supplementary_information.bbl",
        "references.bib",
    ):
        copy_pairs.append((supplement_files[name], FULL_MIRROR / "supplementary_files" / name))
    copy_pairs.extend(
        (
            (
                supplement_files["supplementary_information.pdf"],
                FULL_MIRROR
                / "supplementary_files"
                / "Additional_file_1_Supplementary_Information.pdf",
            ),
            (
                supplement_files["Additional_file_2_Supplementary_Tables.xlsx"],
                FULL_MIRROR
                / "supplementary_files"
                / "Additional_file_2_Supplementary_Tables.xlsx",
            ),
        )
    )
    for source, destination in copy_pairs:
        atomic_copy(source, destination)

    main_source_zip = CANDIDATE / "TED_BIB_main_source.zip"
    write_deterministic_zip(main_source_zip, collect_tree_payloads(CANDIDATE_MAIN_SOURCE))
    print("Staged authoritative manuscript/supplement sources and current figures.")
    print(f"Wrote deterministic source ZIP: {main_source_zip.relative_to(ROOT)}")
    print("Next: regenerate results/bib_manuscript_revision/evidence_manifest.tsv, then use --finalize.")


def validate_evidence_manifest(path: Path) -> Dict[str, ManifestRecord]:
    require_files([path], context="Evidence-manifest preflight")
    records: Dict[str, ManifestRecord] = {}
    folded_paths = set()
    issues: List[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required_columns = {"path", "bytes", "sha256"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise BundleError(
                f"Evidence manifest must contain {sorted(required_columns)}; found {reader.fieldnames}"
            )
        for line_number, row in enumerate(reader, start=2):
            raw_relative = (row.get("path") or "").strip()
            try:
                relative = normalize_relative(raw_relative, label="evidence-manifest")
                source = repo_file(relative)
                expected_size = int((row.get("bytes") or "").strip())
                expected_sha = (row.get("sha256") or "").strip().casefold()
            except (BundleError, ValueError) as exc:
                issues.append(f"line {line_number}: {exc}")
                continue
            if len(expected_sha) != 64 or any(char not in "0123456789abcdef" for char in expected_sha):
                issues.append(f"line {line_number}: invalid SHA256 for {relative}")
                continue
            folded = relative.casefold()
            if folded in folded_paths:
                issues.append(f"line {line_number}: duplicate path {relative}")
                continue
            folded_paths.add(folded)
            if not source.is_file():
                issues.append(f"missing manifest member: {relative}")
                continue
            actual_size = source.stat().st_size
            if actual_size != expected_size:
                issues.append(
                    f"size mismatch for {relative}: manifest={expected_size}, actual={actual_size}"
                )
                continue
            actual_sha = sha256_file(source)
            if actual_sha != expected_sha:
                issues.append(
                    f"SHA256 mismatch for {relative}: manifest={expected_sha}, actual={actual_sha}"
                )
                continue
            records[relative] = ManifestRecord(relative, source, actual_size, actual_sha)
    if issues:
        shown = issues[:30]
        suffix = "" if len(issues) <= len(shown) else f"\n  ... {len(issues) - len(shown)} more"
        raise BundleError(
            "Evidence manifest validation failed before finalization:\n  - "
            + "\n  - ".join(shown)
            + suffix
        )
    if not records:
        raise BundleError(f"Evidence manifest contains no validated records: {path}")
    return records


def compare_files(authority: Path, mirror: Path, *, label: str) -> None:
    require_files((authority, mirror), context=f"{label} preflight")
    authority_hash = sha256_file(authority)
    mirror_hash = sha256_file(mirror)
    if authority_hash != mirror_hash:
        raise BundleError(
            f"{label} is not synchronized: {authority.relative_to(ROOT)} != {mirror.relative_to(ROOT)}"
        )


def validate_source_zip(path: Path, source_dir: Path) -> None:
    require_files([path], context="Staged main-source ZIP preflight")
    expected = {
        payload.archive_path
        for payload in collect_tree_payloads(source_dir)
    }
    with zipfile.ZipFile(path, "r") as archive:
        infos = archive.infolist()
        actual = {info.filename for info in infos}
        if len(actual) != len(infos):
            raise BundleError(f"Duplicate entries found in staged source ZIP: {path}")
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise BundleError(
                f"Staged source ZIP membership mismatch; missing={missing}, unexpected={unexpected}"
            )
        bad_timestamps = [info.filename for info in infos if info.date_time != FIXED_ZIP_TIMESTAMP]
        if bad_timestamps:
            raise BundleError(
                f"Staged source ZIP has non-deterministic timestamps: {bad_timestamps}"
            )
        excluded = [
            info.filename
            for info in infos
            if not should_include_in_latex_archive(info.filename)
        ]
        if excluded:
            raise BundleError(f"Staged source ZIP contains excluded files: {excluded}")


def validate_stage_outputs() -> None:
    comparisons = (
        (AUTHORITATIVE_MAIN / "main.tex", CANONICAL_FULL / "main.tex", "canonical full main.tex"),
        (AUTHORITATIVE_MAIN / "main.tex", FULL_MIRROR / "main.tex", "full mirror main.tex"),
        (
            AUTHORITATIVE_MAIN / "main.tex",
            CANDIDATE_MAIN_SOURCE / "main.tex",
            "candidate main.tex",
        ),
        (
            AUTHORITATIVE_SUPPLEMENT / "supplementary_information.tex",
            FULL_MIRROR / "supplementary_files" / "supplementary_information.tex",
            "full mirror supplement source",
        ),
        (
            AUTHORITATIVE_SUPPLEMENT / "supplementary_information.tex",
            CANDIDATE_SUPPLEMENT / "supplementary_information.tex",
            "candidate supplement source",
        ),
        (
            AUTHORITATIVE_MAIN / "main.pdf",
            CANDIDATE / "TED_BIB_main_manuscript.pdf",
            "candidate main PDF",
        ),
        (
            AUTHORITATIVE_SUPPLEMENT / "supplementary_information.pdf",
            CANDIDATE / "TED_BIB_supplementary_information.pdf",
            "candidate supplement PDF",
        ),
    )
    for authority, mirror, label in comparisons:
        compare_files(authority, mirror, label=label)
    validate_source_zip(CANDIDATE / "TED_BIB_main_source.zip", CANDIDATE_MAIN_SOURCE)


def validate_heavy_manifest(
    evidence_records: Mapping[str, ManifestRecord]
) -> Dict[str, Path]:
    require_files([HEAVY_MANIFEST], context="Heavy-shuffle manifest preflight")
    rows: Dict[str, Tuple[int, str]] = {}
    with HEAVY_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"file", "bytes", "sha256"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise BundleError(
                f"Heavy-shuffle manifest must contain {sorted(required)}; found {reader.fieldnames}"
            )
        for line_number, row in enumerate(reader, start=2):
            name = (row.get("file") or "").strip()
            if Path(name).name != name or not name:
                raise BundleError(f"Unsafe heavy-shuffle member at line {line_number}: {name!r}")
            if name in rows:
                raise BundleError(f"Duplicate heavy-shuffle member: {name}")
            try:
                size = int((row.get("bytes") or "").strip())
            except ValueError as exc:
                raise BundleError(f"Invalid heavy-shuffle byte count for {name}") from exc
            rows[name] = (size, (row.get("sha256") or "").strip().casefold())

    expected = set(HEAVY_FILE_NAMES)
    actual = set(rows)
    if actual != expected:
        raise BundleError(
            "Heavy-shuffle manifest must list exactly the six frozen members; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )

    validated: Dict[str, Path] = {}
    for name in HEAVY_FILE_NAMES:
        source = HEAVY_ROOT / name
        require_files([source], context=f"Heavy-shuffle member preflight ({name})")
        expected_size, expected_sha = rows[name]
        actual_size = source.stat().st_size
        actual_sha = sha256_file(source)
        if actual_size != expected_size or actual_sha != expected_sha:
            raise BundleError(
                f"Heavy-shuffle manifest mismatch for {name}: "
                f"manifest=({expected_size}, {expected_sha}), actual=({actual_size}, {actual_sha})"
            )
        relative = source.relative_to(ROOT).as_posix()
        record = evidence_records.get(relative)
        if record is None:
            raise BundleError(
                f"Validated evidence manifest does not include heavy-shuffle member: {relative}"
            )
        if record.size != actual_size or record.sha256 != actual_sha:
            raise BundleError(f"Evidence and heavy-shuffle manifests disagree for {relative}")
        validated[name] = source
    return validated


def build_additional3_payloads(
    evidence_records: Mapping[str, ManifestRecord], heavy_files: Mapping[str, Path]
) -> List[ZipPayload]:
    selected = list(SUBMISSION_EVIDENCE_RELATIVE)
    figure_sources = figure_source_map()
    selected.extend(
        path.relative_to(ROOT).as_posix() for path in figure_sources.values()
    )
    if len({item.casefold() for item in selected}) != len(selected):
        raise BundleError("Additional file 3 evidence allowlist contains duplicate paths")

    payloads: List[ZipPayload] = []
    total_size = 0
    for relative in sorted(selected):
        if not relative.casefold().endswith(ALLOWED_EVIDENCE_ENDINGS):
            raise BundleError(f"Disallowed Additional file 3 evidence type: {relative}")
        record = evidence_records.get(relative)
        if record is None:
            raise BundleError(
                "Submission evidence is absent from the validated revision manifest: "
                f"{relative}. Re-run scripts/build_bib_revision_manifest.py after --stage."
            )
        if record.size > MAX_EVIDENCE_FILE_BYTES:
            raise BundleError(
                f"Submission evidence exceeds the {MAX_EVIDENCE_FILE_BYTES}-byte per-file limit: "
                f"{relative} ({record.size} bytes)"
            )
        total_size += record.size
        payloads.append(
            ZipPayload(
                archive_path=f"evidence/{relative}",
                source_label=relative,
                source=record.path,
            )
        )

    for name in HEAVY_FILE_NAMES:
        source = heavy_files[name]
        size = source.stat().st_size
        if size > MAX_EVIDENCE_FILE_BYTES:
            raise BundleError(f"Heavy-shuffle member exceeds the per-file limit: {name}")
        total_size += size
        payloads.append(
            ZipPayload(
                archive_path=f"heavy_shuffle/{name}",
                source_label=source.relative_to(ROOT).as_posix(),
                source=source,
            )
        )

    heavy_manifest_relative = HEAVY_MANIFEST.relative_to(ROOT).as_posix()
    heavy_manifest_record = evidence_records.get(heavy_manifest_relative)
    if heavy_manifest_record is None:
        raise BundleError(
            "Validated evidence manifest does not include the heavy-shuffle manifest: "
            f"{heavy_manifest_relative}"
        )
    heavy_manifest_size = HEAVY_MANIFEST.stat().st_size
    heavy_manifest_sha = sha256_file(HEAVY_MANIFEST)
    if (
        heavy_manifest_record.size != heavy_manifest_size
        or heavy_manifest_record.sha256 != heavy_manifest_sha
    ):
        raise BundleError(
            f"Evidence manifest disagrees with heavy-shuffle manifest file: {heavy_manifest_relative}"
        )
    total_size += heavy_manifest_size
    payloads.append(
        ZipPayload(
            archive_path="heavy_shuffle/scp1064_heavy_shuffle_manifest.tsv",
            source_label=heavy_manifest_relative,
            source=HEAVY_MANIFEST,
        )
    )

    manifest_size = EVIDENCE_MANIFEST.stat().st_size
    total_size += manifest_size
    payloads.append(
        ZipPayload(
            archive_path="metadata/evidence_manifest.tsv",
            source_label=EVIDENCE_MANIFEST.relative_to(ROOT).as_posix(),
            source=EVIDENCE_MANIFEST,
        )
    )
    if total_size > MAX_ADDITIONAL3_PAYLOAD_BYTES:
        raise BundleError(
            f"Additional file 3 payload is {total_size} bytes, exceeding the "
            f"{MAX_ADDITIONAL3_PAYLOAD_BYTES}-byte safety limit"
        )

    rows: List[Dict[str, object]] = []
    for payload in sorted(payloads, key=lambda item: item.archive_path):
        assert payload.source is not None
        rows.append(
            {
                "archive_path": payload.archive_path,
                "source_path": payload.source_label,
                "bytes": payload.source.stat().st_size,
                "sha256": sha256_file(payload.source),
            }
        )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=("archive_path", "source_path", "bytes", "sha256"),
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    payloads.append(
        ZipPayload(
            archive_path="MANIFEST.tsv",
            source_label="generated payload checksum manifest",
            data=buffer.getvalue().encode("utf-8"),
        )
    )
    return payloads


def package_manifest_bytes(base: Path) -> bytes:
    payloads = collect_tree_payloads(base, include_manifest=False)
    rows = []
    for payload in sorted(payloads, key=lambda item: item.archive_path):
        assert payload.source is not None
        rows.append(
            {
                "path": payload.archive_path,
                "bytes": payload.source.stat().st_size,
                "sha256": sha256_file(payload.source),
            }
        )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=("path", "bytes", "sha256"),
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def write_upload_package_manifest(paths: Sequence[Path]) -> None:
    require_files(paths, context="Top-level upload-package manifest")
    rows: List[Dict[str, object]] = []
    for path in paths:
        timestamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        rows.append(
            {
                "file": path.relative_to(KNOWN_SOURCE_ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "last_write_time_utc": timestamp.isoformat().replace("+00:00", "Z"),
            }
        )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=("file", "size_bytes", "sha256", "last_write_time_utc"),
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_bytes(
        KNOWN_SOURCE_ROOT / "upload_package_manifest.tsv",
        buffer.getvalue().encode("utf-8"),
    )


def finalize() -> None:
    # This must remain the first operation: no final package is mutated until
    # every path, size and SHA256 in the revision evidence manifest validates.
    evidence_records = validate_evidence_manifest(EVIDENCE_MANIFEST)

    validate_stage_outputs()
    heavy_files = validate_heavy_manifest(evidence_records)
    additional3_payloads = build_additional3_payloads(evidence_records, heavy_files)
    require_files(
        [AUTHORITATIVE_SUPPLEMENT / "Additional_file_2_Supplementary_Tables.xlsx"],
        context="Final package preflight",
    )

    canonical_additional3 = AUTHORITATIVE_SUPPLEMENT / ADDITIONAL3_NAME
    write_deterministic_zip(canonical_additional3, additional3_payloads)
    for destination in (
        FULL_MIRROR / "supplementary_files" / ADDITIONAL3_NAME,
        FULL_MIRROR / "tables" / ADDITIONAL3_NAME,
        CANDIDATE / ADDITIONAL3_NAME,
    ):
        atomic_copy(canonical_additional3, destination)

    candidate_evidence = CANDIDATE / "evidence"
    candidate_evidence_map = {
        "evidence_manifest.tsv": EVIDENCE_MANIFEST,
        "revision_evidence_manifest.tsv": EVIDENCE_MANIFEST,
        "manuscript_metric_source_map.tsv": REVISION_RESULTS / "manuscript_metric_source_map.tsv",
        "evidence_axis_legacy_crosswalk.tsv": REVISION_RESULTS / "evidence_axis_legacy_crosswalk.tsv",
        "ev_v2_verification_2026-07-16.tsv": ROOT
        / "results"
        / "ted_submission_supplement"
        / "ev_v2_verification_2026-07-16.tsv",
        "final_verification_2026-07-16.md": ROOT
        / "results"
        / "ted_submission_supplement"
        / "final_verification_2026-07-16.md",
    }
    sync_exact_directory(candidate_evidence, candidate_evidence_map)

    canonical_package_manifest = CANONICAL_FULL / "latex_package_manifest.tsv"
    atomic_write_bytes(canonical_package_manifest, package_manifest_bytes(CANONICAL_FULL))
    atomic_copy(canonical_package_manifest, FULL_MIRROR / "latex_package_manifest.tsv")

    canonical_full_zip = KNOWN_LATEX_ROOT / "TED_GenomeBiology_LaTeX_submission.zip"
    write_deterministic_zip(canonical_full_zip, collect_tree_payloads(CANONICAL_FULL))
    atomic_copy(
        canonical_full_zip,
        LATEX_MIRROR_ROOT / "TED_GenomeBiology_LaTeX_submission.zip",
    )

    main_only_zip = KNOWN_LATEX_ROOT / "TED_GenomeBiology_Main_Manuscript_Only.zip"
    write_deterministic_zip(main_only_zip, collect_tree_payloads(AUTHORITATIVE_MAIN))
    write_upload_package_manifest(
        (
            UPLOAD_MAIN / "main_manuscript_with_figures.pdf",
            UPLOAD_MAIN / "main_manuscript_with_figures.tex",
            canonical_full_zip,
            main_only_zip,
        )
    )

    print(f"Validated {len(evidence_records)} evidence-manifest hashes before writing outputs.")
    print(f"Wrote deterministic Additional file 3: {canonical_additional3.relative_to(ROOT)}")
    print(f"Wrote deterministic full LaTeX archive: {canonical_full_zip.relative_to(ROOT)}")
    print(f"Wrote deterministic main-only archive: {main_only_zip.relative_to(ROOT)}")
    print("Run scripts/build_bib_submission_candidate_manifest.py last.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Safely stage or finalize the July 2026 TED BIB submission bundle. "
            "Run --stage, rebuild evidence_manifest.tsv, then run --finalize."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--stage",
        action="store_true",
        help=(
            "Synchronize authoritative sources/figures and source-data mirrors, "
            "then build the deterministic candidate main-source ZIP."
        ),
    )
    mode.add_argument(
        "--finalize",
        action="store_true",
        help=(
            "Validate every revision evidence-manifest hash, then build Additional "
            "file 3, package manifests and deterministic LaTeX ZIPs."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.stage:
            stage()
        else:
            finalize()
    except (BundleError, OSError, csv.Error, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
