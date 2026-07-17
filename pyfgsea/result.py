from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from .trajpathmix_scope import (
    annotate_timing_scope,
    evaluate_timing_activation,
    project_primary_effect_output,
)


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame()


def _json_default(value: Any):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Index, pd.Series)):
        return value.astype(str).tolist()
    return str(value)


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_gene_sets(gmt_path_or_sets: Any) -> Optional[str]:
    if gmt_path_or_sets is None:
        return None
    if isinstance(gmt_path_or_sets, (str, os.PathLike)):
        path = Path(gmt_path_or_sets)
        if path.exists():
            return _hash_bytes(path.read_bytes())
    try:
        payload = json.dumps(
            gmt_path_or_sets,
            sort_keys=True,
            default=_json_default,
            separators=(",", ":"),
        ).encode("utf-8")
        return _hash_bytes(payload)
    except TypeError:
        return None


def _hash_gene_universe(adata=None, results: Optional[pd.DataFrame] = None) -> Optional[str]:
    genes = None
    if adata is not None and hasattr(adata, "var_names"):
        genes = [str(gene) for gene in adata.var_names]
    elif results is not None and "gene_universe" in results.attrs:
        genes = [str(gene) for gene in results.attrs["gene_universe"]]
    if genes is None:
        return None
    return _hash_bytes("\n".join(genes).encode("utf-8"))


def _git_commit() -> Optional[str]:
    repo = Path(__file__).resolve().parents[1]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    commit = completed.stdout.strip()
    return commit or None


def _pyfgsea_version() -> Optional[str]:
    try:
        from . import __version__

        return __version__
    except Exception:
        return None


def _window_table(results: Optional[pd.DataFrame]) -> pd.DataFrame:
    if results is None or results.empty or "window_id" not in results.columns:
        return pd.DataFrame()
    keep = [
        col
        for col in (
            "window_id",
            "pt_start",
            "pt_end",
            "pt_mid",
            "n_cells",
            "window_mode",
            "ranker",
            "anchor_pseudotime",
            "effective_n_cells",
            "pseudotime_span",
            "graph_radius",
            "mean_graph_distance",
            "branch_purity",
            "weight_sum",
            "weight_entropy",
            "fate_weight_mean",
        )
        if col in results.columns
    ]
    return results[keep].drop_duplicates("window_id").reset_index(drop=True)


def build_metadata(
    *,
    adata=None,
    gmt_path=None,
    results: Optional[pd.DataFrame] = None,
    event_fdr: Optional[pd.DataFrame] = None,
    bootstrap: Optional[pd.DataFrame] = None,
    seed: Optional[int] = None,
    ranker: Optional[str] = None,
    window_mode: Optional[str] = None,
    cell_weight_key: Optional[str] = None,
    replicate_key: Optional[str] = None,
    condition_key: Optional[str] = None,
    branch_key: Optional[str] = None,
    graph_key: Optional[str] = None,
    layer: Optional[str] = None,
    use_raw: Optional[bool] = None,
    **extra,
) -> dict[str, Any]:
    params = {}
    if results is not None:
        params.update(results.attrs.get("trajectory_params", {}))

    metadata = {
        "pyfgsea_version": _pyfgsea_version(),
        "git_commit": _git_commit(),
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed if seed is not None else params.get("seed"),
        "threads": (
            os.environ.get("RAYON_NUM_THREADS")
            or os.environ.get("PYFGSEA_NUM_THREADS")
            or os.environ.get("OMP_NUM_THREADS")
        ),
        "ranker": ranker if ranker is not None else params.get("ranker"),
        "window_mode": window_mode if window_mode is not None else params.get("window_mode"),
        "cell_weight_key": cell_weight_key if cell_weight_key is not None else params.get("cell_weight_key"),
        "replicate_key": replicate_key,
        "condition_key": condition_key,
        "branch_key": branch_key if branch_key is not None else params.get("branch_key"),
        "graph_key": graph_key if graph_key is not None else params.get("graph_key"),
        "gmt_hash": _hash_gene_sets(gmt_path),
        "gene_universe_hash": _hash_gene_universe(adata=adata, results=results),
        "adata_shape": tuple(adata.shape) if adata is not None and hasattr(adata, "shape") else None,
        "expression_source": params.get("expression_source"),
        "layer": layer if layer is not None else params.get("layer"),
        "use_raw": use_raw if use_raw is not None else params.get("use_raw"),
        "event_fdr_null_model": None,
        "n_perm": None,
        "n_boot": None,
    }
    if event_fdr is not None and not event_fdr.empty:
        if "null_model" in event_fdr.columns:
            metadata["event_fdr_null_model"] = sorted(
                map(str, pd.Series(event_fdr["null_model"]).dropna().unique())
            )
        if "n_perm" in event_fdr.columns:
            metadata["n_perm"] = int(pd.to_numeric(event_fdr["n_perm"], errors="coerce").max())
    if bootstrap is not None:
        boot_meta = bootstrap.attrs.get("bootstrap", {}) if hasattr(bootstrap, "attrs") else {}
        metadata["n_boot"] = boot_meta.get("n_boot")
    metadata.update({key: value for key, value in extra.items() if value is not None})
    return metadata


def infer_calibration_status(
    *,
    results: Optional[pd.DataFrame] = None,
    event_fdr: Optional[pd.DataFrame] = None,
    consensus: Optional[pd.DataFrame] = None,
    diagnostics: Optional[pd.DataFrame] = None,
) -> str:
    if results is not None and results.empty:
        return "insufficient_cells"
    if diagnostics is not None and not diagnostics.empty:
        if "skip_reason" in diagnostics.columns and (diagnostics["skip_reason"] == "low_branch_purity").any():
            return "low_branch_purity"
        if "skip_reason" in diagnostics.columns and (diagnostics["skip_reason"] == "too_few_cells").any():
            return "insufficient_cells"
    if event_fdr is not None and not event_fdr.empty:
        statuses = set(event_fdr.get("calibration_status", pd.Series(dtype=str)).dropna().astype(str))
        warnings = ";".join(event_fdr.get("calibration_warning", pd.Series(dtype=str)).dropna().astype(str))
        if "descriptive_only_low_replicates" in statuses or "descriptive_only_low_replicate_count" in warnings:
            return "descriptive_only_low_replicates"
        if "null_calibration_failed" in statuses:
            return "null_calibration_failed"
        return "discovery_ready"
    if consensus is not None and not consensus.empty:
        if "recommendation" in consensus.columns:
            rec = consensus["recommendation"].astype(str).str.lower()
            if rec.str.contains("unstable", na=False).any():
                return "unstable_consensus"
        if "ranker_agreement" in consensus.columns:
            agreement = pd.to_numeric(consensus["ranker_agreement"], errors="coerce")
            if np.isfinite(agreement).any() and float(np.nanmedian(agreement)) < 0.5:
                return "unstable_consensus"
    return "discovery_ready" if event_fdr is not None else "exploratory_only"


def build_diagnostics(
    *,
    windows: Optional[pd.DataFrame] = None,
    results: Optional[pd.DataFrame] = None,
    event_fdr: Optional[pd.DataFrame] = None,
    bootstrap: Optional[pd.DataFrame] = None,
    consensus: Optional[pd.DataFrame] = None,
    leading_edges: Optional[pd.DataFrame] = None,
    baselines: Optional[pd.DataFrame] = None,
    extra_diagnostics: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    rows = []

    def add(module: str, diagnostic: str, value: Any, status: str = "ok", message: str = ""):
        rows.append(
            {
                "module": module,
                "diagnostic": diagnostic,
                "value": value,
                "status": status,
                "message": message,
            }
        )

    if windows is not None and not windows.empty:
        if "n_cells" in windows.columns:
            add("window", "min_n_cells", int(pd.to_numeric(windows["n_cells"], errors="coerce").min()))
        if "pseudotime_span" in windows.columns:
            add("window", "median_pseudotime_span", float(pd.to_numeric(windows["pseudotime_span"], errors="coerce").median()))
        if "effective_n_cells" in windows.columns:
            add("window", "median_effective_n_cells", float(pd.to_numeric(windows["effective_n_cells"], errors="coerce").median()))
        if "branch_purity" in windows.columns:
            purity = pd.to_numeric(windows["branch_purity"], errors="coerce")
            if np.isfinite(purity).any():
                add("graph_adaptive", "min_branch_purity", float(np.nanmin(purity)))
        if "weight_entropy" in windows.columns:
            entropy = pd.to_numeric(windows["weight_entropy"], errors="coerce")
            if np.isfinite(entropy).any():
                add("graph_adaptive", "median_weight_entropy", float(np.nanmedian(entropy)))

    if event_fdr is not None and not event_fdr.empty:
        if "n_perm" in event_fdr.columns:
            n_perm = int(pd.to_numeric(event_fdr["n_perm"], errors="coerce").max())
            add("event_fdr", "n_perm", n_perm)
            add("event_fdr", "minimum_attainable_p", 1.0 / (1.0 + n_perm))
        if "null_model" in event_fdr.columns:
            add("event_fdr", "null_model", ",".join(sorted(map(str, event_fdr["null_model"].dropna().unique()))))
        if "calibration_warning" in event_fdr.columns:
            warnings = sorted(set(";".join(event_fdr["calibration_warning"].dropna().astype(str)).split(";")) - {""})
            add("event_fdr", "calibration_warning", ",".join(warnings), status="warning" if warnings else "ok")

    if bootstrap is not None and not bootstrap.empty:
        boot_meta = bootstrap.attrs.get("bootstrap", {})
        add("bootstrap", "resample_unit", boot_meta.get("resample"))
        add("bootstrap", "n_boot", boot_meta.get("n_boot"))
        ci_cols = [col for col in bootstrap.columns if col.endswith("_upper") or col.endswith("_lower")]
        add("bootstrap", "ci_columns", len(ci_cols))

    if consensus is not None and not consensus.empty:
        if "ranker_agreement" in consensus.columns:
            add("consensus", "median_ranker_agreement", float(pd.to_numeric(consensus["ranker_agreement"], errors="coerce").median()))
        if "recommendation" in consensus.columns:
            unstable = int(consensus["recommendation"].astype(str).str.lower().str.contains("unstable", na=False).sum())
            add("consensus", "unstable_events", unstable, status="warning" if unstable else "ok")

    if leading_edges is not None and not leading_edges.empty:
        if "core_gene_count" in leading_edges.columns:
            add("leading_edge", "median_core_gene_count", float(pd.to_numeric(leading_edges["core_gene_count"], errors="coerce").median()))
        if "turnover_score" in leading_edges.columns:
            add("leading_edge", "median_turnover_score", float(pd.to_numeric(leading_edges["turnover_score"], errors="coerce").median()))

    if baselines is not None and not baselines.empty:
        if "event_label_agreement" in baselines.columns:
            add("baseline", "event_label_agreement", float(pd.to_numeric(baselines["event_label_agreement"], errors="coerce").mean()))
        if "AUC_correlation" in baselines.columns:
            add("baseline", "AUC_correlation", float(pd.to_numeric(baselines["AUC_correlation"], errors="coerce").mean()))

    if extra_diagnostics is not None and not extra_diagnostics.empty:
        extra = extra_diagnostics.copy()
        for col in ("module", "diagnostic", "value", "status", "message"):
            if col not in extra.columns:
                extra[col] = ""
        rows.extend(extra[["module", "diagnostic", "value", "status", "message"]].to_dict("records"))

    return pd.DataFrame(rows)


@dataclass
class TrajectoryEventResult:
    windows: pd.DataFrame = field(default_factory=_empty_df)
    results: pd.DataFrame = field(default_factory=_empty_df)
    events: pd.DataFrame = field(default_factory=_empty_df)
    event_fdr: pd.DataFrame = field(default_factory=_empty_df)
    bootstrap: pd.DataFrame = field(default_factory=_empty_df)
    consensus: pd.DataFrame = field(default_factory=_empty_df)
    leading_edges: pd.DataFrame = field(default_factory=_empty_df)
    comparisons: pd.DataFrame = field(default_factory=_empty_df)
    baselines: pd.DataFrame = field(default_factory=_empty_df)
    diagnostics: pd.DataFrame = field(default_factory=_empty_df)
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence_layers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_tables(
        cls,
        *,
        results: Optional[pd.DataFrame] = None,
        events: Optional[pd.DataFrame] = None,
        event_fdr: Optional[pd.DataFrame] = None,
        bootstrap: Optional[pd.DataFrame] = None,
        consensus: Optional[pd.DataFrame] = None,
        leading_edges: Optional[pd.DataFrame] = None,
        comparisons: Optional[pd.DataFrame] = None,
        baselines: Optional[pd.DataFrame] = None,
        diagnostics: Optional[pd.DataFrame] = None,
        metadata: Optional[dict[str, Any]] = None,
        timing_gate_evidence: Optional[Mapping[str, bool]] = None,
        timing_context: str = "dataset_specific",
        adata=None,
        gmt_path=None,
        **metadata_kwargs,
    ) -> "TrajectoryEventResult":
        results = results.copy() if results is not None else pd.DataFrame()
        windows = _window_table(results)
        graph_diag = results.attrs.get("graph_window_diagnostics") if hasattr(results, "attrs") else None
        extra_diag = diagnostics
        if graph_diag is not None and not graph_diag.empty:
            extra_diag = pd.concat(
                [
                    extra_diag if extra_diag is not None else pd.DataFrame(),
                    graph_diag.assign(
                        module="graph_adaptive",
                        diagnostic="window_construction",
                        value="",
                        status=np.where(graph_diag.get("skipped", False), "warning", "ok"),
                        message=graph_diag.get("skip_reason", ""),
                    )[["module", "diagnostic", "value", "status", "message"]],
                ],
                ignore_index=True,
            )
        merged_metadata = build_metadata(
            adata=adata,
            gmt_path=gmt_path,
            results=results,
            event_fdr=event_fdr,
            bootstrap=bootstrap,
            **metadata_kwargs,
        )
        if metadata:
            merged_metadata.update(metadata)
        built_diagnostics = build_diagnostics(
            windows=windows,
            results=results,
            event_fdr=event_fdr,
            bootstrap=bootstrap,
            consensus=consensus,
            leading_edges=leading_edges,
            baselines=baselines,
            extra_diagnostics=extra_diag,
        )
        status = infer_calibration_status(
            results=results,
            event_fdr=event_fdr,
            consensus=consensus,
            diagnostics=built_diagnostics,
        )
        merged_metadata["calibration_status"] = status
        timing_decision = evaluate_timing_activation(
            timing_gate_evidence,
            context=timing_context,
        )
        merged_metadata["timing_scope"] = timing_decision.to_dict()
        evidence_layers = {
            "window_level": "NES/window_q support local visualization only",
            "event_level": (
                "curve magnitude and event_q support functional-effect inference; "
                "timing attributes remain conditional on the timing_scope gates"
            ),
            "robustness_level": "bootstrap, consensus, replicate, leading-edge, and baseline diagnostics support interpretation",
        }
        return cls(
            windows=windows,
            results=results,
            events=(
                annotate_timing_scope(events, timing_decision)
                if events is not None
                else pd.DataFrame()
            ),
            event_fdr=event_fdr.copy() if event_fdr is not None else pd.DataFrame(),
            bootstrap=bootstrap.copy() if bootstrap is not None else pd.DataFrame(),
            consensus=consensus.copy() if consensus is not None else pd.DataFrame(),
            leading_edges=leading_edges.copy() if leading_edges is not None else pd.DataFrame(),
            comparisons=comparisons.copy() if comparisons is not None else pd.DataFrame(),
            baselines=baselines.copy() if baselines is not None else pd.DataFrame(),
            diagnostics=built_diagnostics,
            metadata=merged_metadata,
            evidence_layers=evidence_layers,
        )

    @property
    def calibration_status(self) -> str:
        return str(self.metadata.get("calibration_status", "exploratory_only"))

    @property
    def timing_status(self) -> str:
        return str(
            self.metadata.get("timing_scope", {}).get(
                "status", "conditional_only"
            )
        )

    def to_tables(
        self, *, include_conditional_timing: bool = False
    ) -> dict[str, pd.DataFrame]:
        timing_scope = self.metadata.get("timing_scope", {})
        decision = evaluate_timing_activation(
            timing_scope.get("gate_evidence"),
            context=str(timing_scope.get("context", "dataset_specific")),
        )
        events = (
            self.events.copy()
            if include_conditional_timing
            else project_primary_effect_output(self.events, decision)
        )
        return {
            "windows": self.windows,
            "results": self.results,
            "events": events,
            "event_fdr": self.event_fdr,
            "bootstrap": self.bootstrap,
            "consensus": self.consensus,
            "leading_edges": self.leading_edges,
            "comparisons": self.comparisons,
            "baselines": self.baselines,
            "diagnostics": self.diagnostics,
        }

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "table": name,
                    "rows": int(len(table)),
                    "columns": int(len(table.columns)),
                }
                for name, table in self.to_tables().items()
            ]
        )


def make_trajectory_event_result(**kwargs) -> TrajectoryEventResult:
    return TrajectoryEventResult.from_tables(**kwargs)
