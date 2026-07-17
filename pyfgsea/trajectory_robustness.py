"""Formal sensitivity summaries for externally generated trajectory events.

This module is deliberately analyzer agnostic.  It does not infer pseudotime,
rerun an event caller, choose a significance threshold, or combine p-values.
Instead, it joins event calls to a *planned* variant manifest and reports how
well pre-specified candidate pathways survive trajectory draws, trajectory
methods, and candidate-specific leave-pathway-out (LPO) analyses.

The manifest is the denominator.  A planned run that is absent from ``events``
therefore contributes zero support; it is never silently removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd


_VARIANT_ALIASES = ("variant_id", "analysis_id", "run_id")
_PATHWAY_ALIASES = ("pathway", "Pathway", "gene_set", "term")
_SUPPORT_ALIASES = (
    "supported",
    "event_supported",
    "detected",
    "event_detected",
    "is_significant",
    "significant",
)
_DIRECTION_ALIASES = (
    "direction",
    "effect_direction",
    "event_direction",
    "peak_effect",
    "event_score",
    "delta_AUC",
    "peak_NES",
)
_ONSET_ALIASES = ("onset", "onset_time", "event_onset")
_PEAK_ALIASES = ("peak_time", "event_peak", "event_peak_time")


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame()


@dataclass
class TrajectoryRobustnessResult:
    """Auditable tables from :func:`summarize_trajectory_robustness`.

    ``pathway_summary`` contains equal-weight draw and method summaries.
    ``variant_support`` is the complete candidate-by-planned-variant table,
    including missing analyses.  No table contains a combined p-value.
    """

    pathway_summary: pd.DataFrame = field(default_factory=_empty_frame)
    variant_support: pd.DataFrame = field(default_factory=_empty_frame)
    draw_support: pd.DataFrame = field(default_factory=_empty_frame)
    method_support: pd.DataFrame = field(default_factory=_empty_frame)
    lpo_diagnostics: pd.DataFrame = field(default_factory=_empty_frame)
    manifest_diagnostics: pd.DataFrame = field(default_factory=_empty_frame)
    manifest: pd.DataFrame = field(default_factory=_empty_frame)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> pd.DataFrame:
        """Alias for the candidate-level pathway summary."""

        return self.pathway_summary

    @property
    def candidate_summary(self) -> pd.DataFrame:
        """Alias for the candidate-level pathway summary."""

        return self.pathway_summary

    def to_tables(self) -> dict[str, pd.DataFrame]:
        return {
            "pathway_summary": self.pathway_summary,
            "variant_support": self.variant_support,
            "draw_support": self.draw_support,
            "method_support": self.method_support,
            "lpo_diagnostics": self.lpo_diagnostics,
            "manifest_diagnostics": self.manifest_diagnostics,
            "manifest": self.manifest,
        }


def _column(
    frame: pd.DataFrame,
    aliases: Iterable[str],
    *,
    required: bool = False,
    label: str = "column",
) -> Optional[str]:
    for name in aliases:
        if name in frame.columns:
            return name
    if required:
        raise ValueError(f"Missing {label}; expected one of {list(aliases)!r}")
    return None


def _not_blank(value: Any) -> bool:
    return value is not None and not pd.isna(value) and bool(str(value).strip())


def _supplied(value: Any) -> bool:
    """Whether a manifest cell was supplied, preserving explicit empty sets."""

    if value is None:
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return True
    if isinstance(missing, (bool, np.bool_)):
        return not bool(missing)
    return True


def _strict_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1", "planned", "supported"}:
            return True
        if normalized in {"false", "no", "n", "0", "unplanned", "unsupported"}:
            return False
    raise ValueError(f"{label} must be an explicit boolean, got {value!r}")


def _orientation(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or value is None or pd.isna(value):
        raise ValueError("orientation must be explicit for every planned variant")
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if numeric == 1.0:
            return 1
        if numeric == -1.0:
            return -1
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {
        "1",
        "+1",
        "forward",
        "aligned",
        "reference_aligned",
        "increasing",
    }:
        return 1
    if normalized in {
        "_1",  # ``-1`` after replacing the minus sign
        "reverse",
        "reversed",
        "reference_reversed",
        "decreasing",
    }:
        return -1
    raise ValueError(
        "orientation must be one of +1/-1 or forward/reverse; "
        f"got {value!r}"
    )


def _direction(value: Any) -> float:
    if value is None or pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        return float(np.sign(numeric)) if np.isfinite(numeric) and numeric != 0 else np.nan
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {
        "positive",
        "up",
        "upregulated",
        "activation",
        "activated",
        "increasing",
        "enriched",
        "+1",
        "1",
    }:
        return 1.0
    if normalized in {
        "negative",
        "down",
        "downregulated",
        "suppression",
        "suppressed",
        "decreasing",
        "depleted",
        "_1",
    }:
        return -1.0
    return np.nan


def _numeric(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def _gene_set(value: Any) -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return set()
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                decoded = json.loads(stripped)
            except (json.JSONDecodeError, TypeError):
                decoded = None
            if decoded is not None:
                return _gene_set(decoded)
        return {token.strip() for token in re.split(r"[;,|]", stripped) if token.strip()}
    if isinstance(value, Mapping):
        values = value.keys()
    elif isinstance(value, (set, frozenset, list, tuple, np.ndarray, pd.Series, pd.Index)):
        values = value
    else:
        values = [value]
    return {str(item).strip() for item in values if _not_blank(item)}


def _membership_map(
    pathway_membership: Mapping[str, Iterable[str]] | pd.DataFrame | None,
) -> dict[str, set[str]]:
    if pathway_membership is None:
        return {}
    if isinstance(pathway_membership, Mapping):
        return {
            str(pathway): _gene_set(genes)
            for pathway, genes in pathway_membership.items()
        }
    if not isinstance(pathway_membership, pd.DataFrame):
        raise TypeError("pathway_membership must be a mapping, DataFrame, or None")
    pathway_col = _column(
        pathway_membership,
        _PATHWAY_ALIASES,
        required=True,
        label="pathway_membership pathway column",
    )
    gene_col = _column(
        pathway_membership,
        ("gene", "symbol", "member", "genes", "members", "gene_symbols"),
        required=True,
        label="pathway_membership gene/member column",
    )
    out: dict[str, set[str]] = {}
    for pathway, group in pathway_membership.groupby(pathway_col, sort=False, dropna=False):
        if not _not_blank(pathway):
            raise ValueError("pathway_membership contains a blank pathway")
        genes: set[str] = set()
        for value in group[gene_col]:
            genes.update(_gene_set(value))
        out[str(pathway)] = genes
    return out


def _canonical_manifest(manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not isinstance(manifest, pd.DataFrame) or manifest.empty:
        raise ValueError("manifest must be a non-empty DataFrame")
    source = manifest.copy()
    variant_col = _column(source, _VARIANT_ALIASES, required=True, label="variant id")
    orientation_col = _column(
        source, ("orientation", "trajectory_orientation"), required=True, label="orientation"
    )
    method_col = _column(
        source, ("method_id", "trajectory_method", "method"), required=True, label="method"
    )
    source_col = _column(
        source,
        ("source", "trajectory_source", "pseudotime_source", "source_id"),
        required=True,
        label="trajectory/pseudotime source provenance",
    )
    draw_col = _column(
        source,
        ("draw_id", "pseudotime_draw_id", "trajectory_draw", "draw"),
    )
    planned_col = _column(source, ("planned", "is_planned", "preplanned"))
    status_col = _column(source, ("status", "run_status", "analysis_status"))
    if planned_col is None and status_col is None:
        raise ValueError(
            "manifest requires either an explicit planned boolean or a status column"
        )
    role_col = _column(
        source,
        ("variant_kind", "variant_role", "variant_type", "analysis_type", "role"),
    )
    primary_col = _column(source, ("is_primary", "primary"))
    lpo_col = _column(
        source,
        ("lpo_pathway", "leave_pathway_out", "excluded_pathway", "target_pathway"),
    )
    lpo_flag_col = _column(source, ("is_lpo", "leave_pathway_out_variant"))
    reference_id_col = _column(
        source,
        ("reference_id", "reference_source", "reference_dataset", "reference_atlas_id"),
    )
    reference_version_col = _column(
        source,
        ("reference_version", "reference_hash", "reference_fingerprint", "reference_provenance"),
    )
    mapping_method_col = _column(
        source,
        ("mapping_method", "reference_mapping_method"),
    )
    reference_fit_scope_col = _column(
        source,
        ("reference_fit_scope", "fit_scope", "reference_training_scope"),
    )
    cell_mask_col = _column(
        source,
        ("cell_mask_hash", "cell_mask_id", "cell_set_hash", "common_cell_mask_id"),
        required=True,
        label="cell mask hash/id",
    )
    feature_col = _column(
        source,
        ("trajectory_features", "trajectory_feature_genes", "feature_genes", "features"),
    )
    exclusion_col = _column(
        source,
        ("excluded_features", "excluded_genes", "removed_features", "lpo_excluded_genes"),
    )
    time_min_col = _column(source, ("pseudotime_min", "time_min", "trajectory_time_min"))
    time_max_col = _column(source, ("pseudotime_max", "time_max", "trajectory_time_max"))

    out = pd.DataFrame(index=source.index)
    out["variant_id"] = source[variant_col].map(lambda value: str(value).strip() if _not_blank(value) else "")
    if (out["variant_id"] == "").any():
        raise ValueError("manifest variant_id values must be non-blank")
    duplicated = out.loc[out["variant_id"].duplicated(keep=False), "variant_id"].unique()
    if len(duplicated):
        raise ValueError(f"manifest variant_id values must be unique: {duplicated.tolist()!r}")

    if status_col is not None:
        out["status"] = source[status_col].map(
            lambda value: str(value).strip().lower() if _not_blank(value) else ""
        )
        if out["status"].eq("").any():
            raise ValueError("manifest status values must be non-blank")
    else:
        out["status"] = ""
    if planned_col is not None:
        out["planned"] = [
            _strict_bool(value, label=f"manifest[{planned_col!r}]") for value in source[planned_col]
        ]
        if status_col is None:
            out["status"] = np.where(out["planned"], "planned", "unplanned")
        unplanned_status = out["status"].str.replace("-", "_", regex=False).isin(
            {"unplanned", "not_planned", "cancelled_before_plan"}
        )
        if (out["planned"] & unplanned_status).any():
            raise ValueError("manifest planned=True contradicts an explicitly unplanned status")
        if ((~out["planned"]) & (~unplanned_status)).any():
            raise ValueError(
                "Failed, incomplete, rejected, and not-run variants cannot be marked "
                "planned=False; only an explicitly unplanned status may leave the denominator"
            )
    else:
        # Failed, rejected, incomplete, and not-run analyses remain in the
        # pre-specified denominator. Only an explicit *unplanned* status opts
        # a row out.
        out["planned"] = ~out["status"].str.replace("-", "_", regex=False).isin(
            {"unplanned", "not_planned", "cancelled_before_plan"}
        )
    if not out["planned"].any():
        raise ValueError("manifest contains no planned variants")

    out["orientation"] = [
        _orientation(value) if planned else np.nan
        for value, planned in zip(source[orientation_col], out["planned"])
    ]
    out["method_id"] = source[method_col].map(
        lambda value: str(value).strip() if _not_blank(value) else ""
    )
    out["source"] = source[source_col].map(
        lambda value: str(value).strip() if _not_blank(value) else ""
    )
    out["draw_id"] = (
        source[draw_col].map(
            lambda value: str(value).strip() if _not_blank(value) else ""
        )
        if draw_col is not None
        else ""
    )
    missing_method_or_source = out["planned"] & (
        out["method_id"].eq("") | out["source"].eq("")
    )
    if missing_method_or_source.any():
        variants = out.loc[missing_method_or_source, "variant_id"].tolist()
        raise ValueError(
            "Every planned variant must have explicit method and source provenance; "
            f"missing for {variants!r}"
        )

    roles = (
        source[role_col].map(
            lambda value: str(value).strip().lower().replace("-", "_")
            if _not_blank(value)
            else ""
        )
        if role_col is not None
        else pd.Series("", index=source.index, dtype=object)
    )
    lpo_targets = (
        source[lpo_col].map(lambda value: str(value).strip() if _not_blank(value) else "")
        if lpo_col is not None
        else pd.Series("", index=source.index, dtype=object)
    )
    lpo_flags = pd.Series(False, index=source.index)
    if lpo_flag_col is not None:
        lpo_flags = pd.Series(
            [_strict_bool(value, label=f"manifest[{lpo_flag_col!r}]") for value in source[lpo_flag_col]],
            index=source.index,
        )
    role_is_lpo = roles.isin(
        {"lpo", "leave_pathway_out", "leave_one_pathway_out"}
    )
    out["is_lpo"] = lpo_flags | role_is_lpo | lpo_targets.ne("")
    out["lpo_pathway"] = lpo_targets
    out["is_primary"] = roles.eq("primary")
    if primary_col is not None:
        out["is_primary"] |= pd.Series(
            [_strict_bool(value, label=f"manifest[{primary_col!r}]") for value in source[primary_col]],
            index=source.index,
        )
    out["variant_kind"] = roles
    out.loc[out["variant_kind"].eq("") & out["is_primary"], "variant_kind"] = "primary"
    out.loc[out["variant_kind"].eq("") & out["is_lpo"], "variant_kind"] = (
        "leave_pathway_out"
    )
    out.loc[out["variant_kind"].eq(""), "variant_kind"] = "unspecified"
    missing_lpo_target = out["planned"] & out["is_lpo"] & out["lpo_pathway"].eq("")
    if missing_lpo_target.any():
        variants = out.loc[missing_lpo_target, "variant_id"].tolist()
        raise ValueError(
            "Every planned LPO variant must name its candidate-specific lpo_pathway; "
            f"missing for {variants!r}"
        )
    explicit_draw = out["draw_id"].ne("")
    inferred_draw = out["variant_kind"].isin({"primary", "draw"})
    out["draw_eligible"] = (~out["is_lpo"]) & (explicit_draw | inferred_draw)
    out.loc[out["draw_eligible"] & out["draw_id"].eq(""), "draw_id"] = out["variant_id"]
    out["reference_id"] = (
        source[reference_id_col].map(
            lambda value: str(value).strip() if _not_blank(value) else ""
        )
        if reference_id_col is not None
        else ""
    )
    out["reference_version"] = (
        source[reference_version_col].map(
            lambda value: str(value).strip() if _not_blank(value) else ""
        )
        if reference_version_col is not None
        else ""
    )
    out["mapping_method"] = (
        source[mapping_method_col].map(
            lambda value: str(value).strip() if _not_blank(value) else ""
        )
        if mapping_method_col is not None
        else ""
    )
    out["reference_fit_scope"] = (
        source[reference_fit_scope_col].map(
            lambda value: str(value).strip() if _not_blank(value) else ""
        )
        if reference_fit_scope_col is not None
        else ""
    )
    out["cell_mask_id"] = source[cell_mask_col].map(
        lambda value: str(value).strip() if _not_blank(value) else ""
    )
    missing_mask = out["planned"] & out["cell_mask_id"].eq("")
    if missing_mask.any():
        variants = out.loc[missing_mask, "variant_id"].tolist()
        raise ValueError(
            "Every planned variant must provide a non-blank cell_mask_hash/id; "
            f"missing for {variants!r}"
        )
    out["feature_genes"] = (
        source[feature_col].map(_gene_set) if feature_col is not None else [set() for _ in source.index]
    )
    out["feature_genes_explicit"] = (
        source[feature_col].map(_supplied) if feature_col is not None else False
    )
    out["excluded_genes"] = (
        source[exclusion_col].map(_gene_set)
        if exclusion_col is not None
        else [set() for _ in source.index]
    )
    out["excluded_genes_explicit"] = (
        source[exclusion_col].map(_supplied) if exclusion_col is not None else False
    )

    if (time_min_col is None) != (time_max_col is None):
        raise ValueError("pseudotime_min and pseudotime_max must be supplied together")
    if time_min_col is None:
        out["pseudotime_min"] = 0.0
        out["pseudotime_max"] = 1.0
        default_time_range = True
    else:
        out["pseudotime_min"] = source[time_min_col].map(_numeric)
        out["pseudotime_max"] = source[time_max_col].map(_numeric)
        default_time_range = False
    invalid_range = out["planned"] & (
        (~np.isfinite(out["pseudotime_min"]))
        | (~np.isfinite(out["pseudotime_max"]))
        | (out["pseudotime_max"] <= out["pseudotime_min"])
    )
    if invalid_range.any():
        variants = out.loc[invalid_range, "variant_id"].tolist()
        raise ValueError(f"Invalid pseudotime bounds for variants {variants!r}")

    metadata = {
        "default_unit_pseudotime_range_used": default_time_range,
        "feature_manifest_column_present": feature_col is not None,
        "exclusion_manifest_column_present": exclusion_col is not None,
        "reference_id_column_present": reference_id_col is not None,
        "reference_version_column_present": reference_version_col is not None,
        "mapping_method_column_present": mapping_method_col is not None,
        "reference_fit_scope_column_present": reference_fit_scope_col is not None,
        "cell_mask_column_present": True,
        "source_column_present": True,
        "draw_id_column_present": draw_col is not None,
        "status_column_present": status_col is not None,
        "planned_column_present": planned_col is not None,
    }
    return out.reset_index(drop=True), metadata


def _primary_variant(manifest: pd.DataFrame, primary_variant_id: Optional[str]) -> str:
    if primary_variant_id is not None:
        primary = str(primary_variant_id).strip()
        if primary not in set(manifest.loc[manifest["planned"], "variant_id"]):
            raise ValueError(f"primary_variant_id {primary!r} is not a planned manifest variant")
    else:
        candidates = manifest.loc[manifest["planned"] & manifest["is_primary"], "variant_id"]
        if len(candidates) != 1:
            raise ValueError(
                "primary_variant_id must be supplied unless exactly one planned manifest "
                "row is explicitly marked primary"
            )
        primary = str(candidates.iloc[0])
    row = manifest.loc[manifest["variant_id"].eq(primary)].iloc[0]
    if bool(row["is_lpo"]):
        raise ValueError("The primary variant cannot be an LPO variant")
    return primary


def _canonical_events(
    events: pd.DataFrame,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not isinstance(events, pd.DataFrame):
        raise TypeError("events must be a DataFrame")
    ignored = [
        str(col)
        for col in events.columns
        if re.search(r"(^|_)(p|q|fdr|pvalue|p_value|qvalue|q_value)($|_)", str(col).lower())
    ]
    if events.empty:
        return pd.DataFrame(
            columns=[
                "variant_id",
                "pathway",
                "event_observed",
                "supported",
                "direction",
                "onset",
                "peak_time",
                "aligned_direction",
                "aligned_onset",
                "aligned_peak_time",
            ]
        ), {"support_definition": "row_presence", "ignored_p_value_columns": ignored}

    variant_col = _column(events, _VARIANT_ALIASES, required=True, label="event variant id")
    pathway_col = _column(events, _PATHWAY_ALIASES, required=True, label="event pathway")
    support_col = _column(events, _SUPPORT_ALIASES)
    direction_col = _column(events, _DIRECTION_ALIASES)
    onset_col = _column(events, _ONSET_ALIASES)
    peak_col = _column(events, _PEAK_ALIASES)
    activation_col = _column(events, ("activation_onset",))
    suppression_col = _column(events, ("suppression_onset",))
    event_reference_id_col = _column(
        events,
        ("reference_id", "reference_source", "reference_dataset", "reference_atlas_id"),
    )
    event_reference_version_col = _column(
        events,
        ("reference_version", "reference_hash", "reference_fingerprint", "reference_provenance"),
    )
    event_source_col = _column(
        events,
        ("source", "trajectory_source", "pseudotime_source", "source_id"),
    )

    out = pd.DataFrame(index=events.index)
    out["variant_id"] = events[variant_col].map(
        lambda value: str(value).strip() if _not_blank(value) else ""
    )
    out["pathway"] = events[pathway_col].map(
        lambda value: str(value).strip() if _not_blank(value) else ""
    )
    if out["variant_id"].eq("").any() or out["pathway"].eq("").any():
        raise ValueError("events variant_id and pathway values must be non-blank")
    unknown = sorted(set(out["variant_id"]) - set(manifest["variant_id"]))
    if unknown:
        raise ValueError(f"events contain variants absent from the planned manifest: {unknown!r}")
    duplicate = out.duplicated(["variant_id", "pathway"], keep=False)
    if duplicate.any():
        pairs = out.loc[duplicate, ["variant_id", "pathway"]].drop_duplicates().to_dict("records")
        raise ValueError(
            "events must contain at most one row per variant and pathway; "
            f"duplicate pairs: {pairs!r}"
        )

    out["event_observed"] = True
    if support_col is None:
        out["supported"] = True
        support_definition = "row_presence"
    else:
        out["supported"] = [
            _strict_bool(value, label=f"events[{support_col!r}]") for value in events[support_col]
        ]
        support_definition = f"explicit:{support_col}"
    out["direction"] = (
        events[direction_col].map(_direction) if direction_col is not None else np.nan
    )

    if onset_col is not None:
        out["onset"] = events[onset_col].map(_numeric)
    else:
        out["onset"] = np.nan
        if activation_col is not None or suppression_col is not None:
            activation = (
                events[activation_col].map(_numeric)
                if activation_col is not None
                else pd.Series(np.nan, index=events.index)
            )
            suppression = (
                events[suppression_col].map(_numeric)
                if suppression_col is not None
                else pd.Series(np.nan, index=events.index)
            )
            out["onset"] = np.where(out["direction"].lt(0), suppression, activation)
    out["peak_time"] = events[peak_col].map(_numeric) if peak_col is not None else np.nan

    missing_direction = out["supported"] & ~np.isfinite(out["direction"])
    if missing_direction.any():
        pairs = out.loc[missing_direction, ["variant_id", "pathway"]].to_dict("records")
        raise ValueError(
            "Every supported event must have a non-zero, interpretable direction; "
            f"missing for {pairs!r}"
        )

    joined = out.merge(
        manifest[
            [
                "variant_id",
                "orientation",
                "pseudotime_min",
                "pseudotime_max",
                "reference_id",
                "reference_version",
                "source",
            ]
        ],
        on="variant_id",
        how="left",
        validate="many_to_one",
    )
    for timing_col in ("onset", "peak_time"):
        finite = np.isfinite(joined[timing_col])
        outside = finite & (
            (joined[timing_col] < joined["pseudotime_min"] - 1e-12)
            | (joined[timing_col] > joined["pseudotime_max"] + 1e-12)
        )
        if outside.any():
            pairs = joined.loc[outside, ["variant_id", "pathway", timing_col]].to_dict("records")
            raise ValueError(
                f"{timing_col} lies outside the declared pseudotime range: {pairs!r}"
            )
    # Event direction is an effect/enrichment direction, not a temporal slope.
    # Reversing the pseudotime coordinate therefore mirrors event timing but
    # must not invert delta_AUC, peak_NES, or another effect sign.
    joined["aligned_direction"] = joined["direction"]
    span = joined["pseudotime_max"] - joined["pseudotime_min"]
    valid_span = np.isfinite(span) & span.gt(0)
    for timing_col, aligned_col in (
        ("onset", "aligned_onset"),
        ("peak_time", "aligned_peak_time"),
    ):
        relative = (joined[timing_col] - joined["pseudotime_min"]) / span
        finite = np.isfinite(joined[timing_col]) & valid_span
        joined[aligned_col] = np.where(
            finite & joined["orientation"].eq(1),
            relative,
            np.where(
                finite & joined["orientation"].eq(-1),
                1.0 - relative,
                np.nan,
            ),
        )

    # If the event producer repeats provenance, it must agree with the plan.
    if event_reference_id_col is not None:
        event_values = events[event_reference_id_col].map(
            lambda value: str(value).strip() if _not_blank(value) else ""
        ).to_numpy()
        mismatch = (event_values != "") & (event_values != joined["reference_id"].to_numpy())
        if mismatch.any():
            raise ValueError("event reference_id disagrees with its planned manifest provenance")
    if event_reference_version_col is not None:
        event_values = events[event_reference_version_col].map(
            lambda value: str(value).strip() if _not_blank(value) else ""
        ).to_numpy()
        mismatch = (event_values != "") & (
            event_values != joined["reference_version"].to_numpy()
        )
        if mismatch.any():
            raise ValueError("event reference version/hash disagrees with its planned manifest provenance")
    if event_source_col is not None:
        event_values = events[event_source_col].map(
            lambda value: str(value).strip() if _not_blank(value) else ""
        ).to_numpy()
        mismatch = (event_values != "") & (event_values != joined["source"].to_numpy())
        if mismatch.any():
            raise ValueError("event trajectory/pseudotime source disagrees with its manifest")

    return joined, {
        "support_definition": support_definition,
        "ignored_p_value_columns": ignored,
    }


def _dispersion(values: Iterable[Any]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "median": np.nan, "mad": np.nan, "iqr": np.nan}
    median = float(np.median(array))
    return {
        "n": int(len(array)),
        "median": median,
        "mad": float(np.median(np.abs(array - median))),
        "iqr": float(np.percentile(array, 75) - np.percentile(array, 25)),
    }


def _unit_direction(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite[np.isfinite(finite)]
    if finite.empty:
        return np.nan
    total = float(np.sign(finite).sum())
    return float(np.sign(total)) if total != 0 else 0.0


def _axis_table(frame: pd.DataFrame, axis: str) -> pd.DataFrame:
    columns = [
        "pathway",
        axis,
        "planned_variant_count",
        "observed_variant_count",
        "supporting_variant_count",
        "support_fraction_within_unit",
        "unit_direction",
        "unit_onset",
        "unit_peak_time",
    ]
    rows: list[dict[str, Any]] = []
    for (pathway, unit), group in frame.groupby(["pathway", axis], sort=False, dropna=False):
        supported = group["supported"].astype(bool)
        supported_group = group.loc[supported]
        onset_values = pd.to_numeric(supported_group["aligned_onset"], errors="coerce")
        peak_values = pd.to_numeric(
            supported_group["aligned_peak_time"], errors="coerce"
        )
        rows.append(
            {
                "pathway": pathway,
                axis: unit,
                "planned_variant_count": int(len(group)),
                "observed_variant_count": int(group["event_observed"].sum()),
                "supporting_variant_count": int(supported.sum()),
                "support_fraction_within_unit": float(supported.mean()),
                "unit_direction": _unit_direction(supported_group["aligned_direction"]),
                "unit_onset": (
                    float(np.nanmedian(onset_values))
                    if np.isfinite(onset_values.to_numpy(dtype=float)).any()
                    else np.nan
                ),
                "unit_peak_time": (
                    float(np.nanmedian(peak_values))
                    if np.isfinite(peak_values.to_numpy(dtype=float)).any()
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _reference_direction(
    candidate: pd.DataFrame,
    primary_variant_id: str,
) -> tuple[float, str]:
    aligned = pd.to_numeric(candidate["aligned_direction"], errors="coerce")
    primary = candidate.loc[
        candidate["variant_id"].eq(primary_variant_id)
        & candidate["supported"]
        & np.isfinite(aligned.to_numpy(dtype=float))
    ]
    if not primary.empty:
        return float(primary.iloc[0]["aligned_direction"]), "primary_variant"
    # The same sensitivity variants being assessed must not select their own
    # reference direction.  Without a supported primary direction, direction
    # consistency is intentionally unavailable rather than circularly defined.
    return np.nan, "unavailable"


def _axis_summary(
    axis_table: pd.DataFrame,
    *,
    axis: str,
    reference_direction: float,
) -> dict[str, Any]:
    planned = int(len(axis_table))
    observed = int((axis_table["observed_variant_count"] > 0).sum()) if planned else 0
    supporting = int((axis_table["supporting_variant_count"] > 0).sum()) if planned else 0
    complete = int(
        (axis_table["observed_variant_count"] == axis_table["planned_variant_count"]).sum()
    ) if planned else 0
    unit_directions = pd.to_numeric(axis_table.get("unit_direction"), errors="coerce")
    direction_observed = np.isfinite(unit_directions)
    if np.isfinite(reference_direction) and direction_observed.any():
        aligned = unit_directions[direction_observed].eq(reference_direction)
        consistency = float(aligned.mean())
        planned_aligned_rate = float(aligned.sum() / planned) if planned else np.nan
        discordant = int((~aligned).sum())
    else:
        consistency = np.nan
        planned_aligned_rate = np.nan
        discordant = 0
    onset = _dispersion(axis_table.get("unit_onset", pd.Series(dtype=float)))
    peak = _dispersion(axis_table.get("unit_peak_time", pd.Series(dtype=float)))
    prefix = "draw" if axis == "draw_id" else "method"
    return {
        f"n_planned_{prefix}s": planned,
        f"n_observed_{prefix}s": observed,
        f"n_complete_{prefix}s": complete,
        f"n_supporting_{prefix}s": supporting,
        f"{prefix}_equal_weight_support_rate": (
            float(axis_table["support_fraction_within_unit"].mean()) if planned else np.nan
        ),
        f"{prefix}_direction_n": int(direction_observed.sum()),
        f"{prefix}_direction_consistency": consistency,
        f"{prefix}_planned_aligned_direction_rate": planned_aligned_rate,
        f"{prefix}_direction_discordant_count": discordant,
        f"{prefix}_onset_n": onset["n"],
        f"{prefix}_onset_median": onset["median"],
        f"{prefix}_onset_mad": onset["mad"],
        f"{prefix}_onset_iqr": onset["iqr"],
        f"{prefix}_peak_n": peak["n"],
        f"{prefix}_peak_median": peak["median"],
        f"{prefix}_peak_mad": peak["mad"],
        f"{prefix}_peak_iqr": peak["iqr"],
    }


def _manifest_gates(
    manifest: pd.DataFrame,
    primary_variant_id: str,
) -> tuple[pd.DataFrame, dict[str, bool]]:
    planned = manifest.loc[manifest["planned"]].copy()
    primary = planned.loc[planned["variant_id"].eq(primary_variant_id)].iloc[0]
    source_complete = bool(planned["source"].ne("").all())
    reference_anchored = planned["variant_kind"].isin(
        {"reference", "reference_anchored", "reference_mapping"}
    )
    anchored_rows = planned.loc[reference_anchored]
    anchored_complete = bool(
        anchored_rows["reference_id"].ne("").all()
        and anchored_rows["reference_version"].ne("").all()
        and anchored_rows["mapping_method"].ne("").all()
        and anchored_rows["reference_fit_scope"].ne("").all()
    )
    nonblank = planned.loc[
        planned["reference_id"].ne("") & planned["reference_version"].ne("")
    ]
    provenance_consistent = bool(
        nonblank.groupby("reference_id")["reference_version"].nunique().le(1).all()
    )
    if not provenance_consistent:
        conflicts = (
            nonblank.groupby("reference_id")["reference_version"]
            .agg(lambda values: sorted(set(values)))
            .loc[lambda values: values.map(len).gt(1)]
            .to_dict()
        )
        raise ValueError(
            "A reference_id maps to conflicting versions/hashes in the manifest: "
            f"{conflicts!r}"
        )
    if not anchored_rows.empty:
        source_reference_consistent = bool(
            anchored_rows.groupby("source")["reference_id"].nunique().le(1).all()
        )
        if not source_reference_consistent:
            raise ValueError(
                "A reference-anchored source maps to conflicting reference_id values"
            )
    else:
        source_reference_consistent = True
    provenance_complete = source_complete and anchored_complete
    mask_complete = bool(planned["cell_mask_id"].ne("").all())
    common_mask = bool(
        mask_complete
        and _not_blank(primary["cell_mask_id"])
        and planned["cell_mask_id"].eq(primary["cell_mask_id"]).all()
    )
    gates = {
        "reference_provenance_complete": provenance_complete,
        "reference_provenance_consistent": (
            provenance_consistent and source_reference_consistent
        ),
        "reference_provenance_gate": bool(
            provenance_complete and provenance_consistent and source_reference_consistent
        ),
        "source_provenance_complete_gate": source_complete,
        "reference_anchored_fields_complete_gate": anchored_complete,
        "cell_mask_provenance_complete": mask_complete,
        "common_cell_mask_gate": common_mask,
        "orientation_explicit_gate": True,
        "draw_manifest_complete_gate": bool(
            planned.loc[planned["draw_eligible"], "draw_id"].ne("").all()
        ),
        "method_manifest_complete_gate": bool(planned["method_id"].ne("").all()),
        "status_manifest_complete_gate": bool(planned["status"].ne("").all()),
    }
    messages = {
        "reference_provenance_complete": "all sources are named and reference-anchored variants declare reference id, version/hash, mapping, and fit scope",
        "reference_provenance_consistent": "reference ids, versions, and anchored sources are internally consistent",
        "reference_provenance_gate": "trajectory/reference provenance is complete and internally consistent",
        "source_provenance_complete_gate": "every planned variant names its pseudotime/trajectory source",
        "reference_anchored_fields_complete_gate": "reference-anchored variants declare reference_id, reference_version/hash, mapping_method, and reference_fit_scope",
        "cell_mask_provenance_complete": "every planned variant names its analyzed cell mask",
        "common_cell_mask_gate": "every planned variant uses the primary variant's cell mask",
        "orientation_explicit_gate": "every planned trajectory declares forward or reverse orientation",
        "draw_manifest_complete_gate": "every draw-eligible variant has an explicit or deterministic draw id",
        "method_manifest_complete_gate": "every planned variant has an explicit method id",
        "status_manifest_complete_gate": "every planned variant has explicit plan/run status",
    }
    diagnostics = pd.DataFrame(
        [
            {
                "gate": key,
                "passed": bool(value),
                "status": "pass" if value else "fail",
                "message": messages[key],
            }
            for key, value in gates.items()
        ]
    )
    return diagnostics, gates


def _lpo_diagnostics(
    *,
    candidates: list[str],
    detail: pd.DataFrame,
    manifest: pd.DataFrame,
    membership: dict[str, set[str]],
    primary_variant_id: str,
) -> pd.DataFrame:
    primary = manifest.loc[manifest["variant_id"].eq(primary_variant_id)].iloc[0]
    primary_features = set(primary["feature_genes"])
    primary_features_explicit = bool(primary["feature_genes_explicit"])
    provenance_fields = (
        "method_id",
        "source",
        "reference_id",
        "reference_version",
        "mapping_method",
        "reference_fit_scope",
    )
    rows: list[dict[str, Any]] = []
    for pathway in candidates:
        candidate_genes = membership.get(pathway)
        lpo_manifest = manifest.loc[
            manifest["planned"]
            & manifest["is_lpo"]
            & manifest["lpo_pathway"].eq(pathway)
        ]
        candidate_detail = detail.loc[detail["pathway"].eq(pathway)].set_index("variant_id")
        for _, variant in lpo_manifest.iterrows():
            event = candidate_detail.loc[variant["variant_id"]]
            lpo_features = set(variant["feature_genes"])
            excluded = set(variant["excluded_genes"])
            membership_available = candidate_genes is not None and len(candidate_genes) > 0
            feature_provenance_available = bool(
                primary_features_explicit
                and variant["feature_genes_explicit"]
                and variant["excluded_genes_explicit"]
            )
            if membership_available and primary_features_explicit:
                expected = set(candidate_genes) & primary_features
            else:
                expected = set()
            missing_expected = expected - excluded
            remaining_candidate = (
                set(candidate_genes) & lpo_features if membership_available else set()
            )
            actual_removed = primary_features - lpo_features
            undeclared_removed = actual_removed - excluded
            declared_but_not_removed = excluded - actual_removed
            novel_features = lpo_features - primary_features
            feature_exclusion_overlap = lpo_features & excluded
            feature_set_exact = bool(
                feature_provenance_available
                and not undeclared_removed
                and not declared_but_not_removed
                and not novel_features
            )
            if membership_available and feature_provenance_available:
                feature_gate = bool(
                    not missing_expected
                    and not remaining_candidate
                    and not feature_exclusion_overlap
                    and feature_set_exact
                )
                feature_reason = (
                    "pass" if feature_gate else "feature_set_not_exact_primary_minus_excluded"
                )
            elif not membership_available:
                feature_gate = False
                feature_reason = "pathway_membership_unavailable"
            else:
                feature_gate = False
                feature_reason = "feature_or_exclusion_provenance_unavailable"
            mask_match = bool(
                _not_blank(primary["cell_mask_id"])
                and variant["cell_mask_id"] == primary["cell_mask_id"]
            )
            provenance_mismatch_fields = [
                field for field in provenance_fields if variant[field] != primary[field]
            ]
            reference_match = bool(
                _not_blank(primary["source"])
                and _not_blank(variant["source"])
                and not provenance_mismatch_fields
            )
            if feature_reason != "pass":
                reason = feature_reason
            elif not mask_match:
                reason = "cell_mask_mismatch"
            elif not reference_match:
                reason = "provenance_mismatch_primary"
            else:
                reason = "pass"
            rows.append(
                {
                    "pathway": pathway,
                    "variant_id": variant["variant_id"],
                    "lpo_pathway": variant["lpo_pathway"],
                    "candidate_specific_lpo": True,
                    "event_observed": bool(event["event_observed"]),
                    "supported": bool(event["supported"]),
                    "aligned_direction": event["aligned_direction"],
                    "aligned_onset": event["aligned_onset"],
                    "aligned_peak_time": event["aligned_peak_time"],
                    "candidate_gene_count": len(candidate_genes) if membership_available else 0,
                    "primary_candidate_feature_overlap_count": len(expected),
                    "expected_exclusion_count": len(expected),
                    "expected_exclusion_missing_count": len(missing_expected),
                    "expected_exclusion_fraction": (
                        float(len(expected & excluded) / len(expected)) if expected else 1.0
                    ),
                    "remaining_candidate_feature_overlap_count": len(remaining_candidate),
                    "feature_exclusion_overlap_count": len(feature_exclusion_overlap),
                    "actual_removed_feature_count": len(actual_removed),
                    "undeclared_removed_feature_count": len(undeclared_removed),
                    "declared_but_not_removed_feature_count": len(declared_but_not_removed),
                    "novel_feature_count": len(novel_features),
                    "pathway_membership_available": membership_available,
                    "feature_provenance_available": feature_provenance_available,
                    "feature_set_exact_primary_minus_excluded": feature_set_exact,
                    "feature_exclusion_gate": feature_gate,
                    "common_cell_mask_gate": mask_match,
                    "provenance_match_primary": reference_match,
                    "provenance_mismatch_fields": ";".join(provenance_mismatch_fields),
                    "reference_provenance_gate": reference_match,
                    "lpo_design_gate": bool(feature_gate and mask_match and reference_match),
                    "failure_reason": reason,
                }
            )
    columns = [
        "pathway",
        "variant_id",
        "lpo_pathway",
        "candidate_specific_lpo",
        "event_observed",
        "supported",
        "aligned_direction",
        "aligned_onset",
        "aligned_peak_time",
        "candidate_gene_count",
        "primary_candidate_feature_overlap_count",
        "expected_exclusion_count",
        "expected_exclusion_missing_count",
        "expected_exclusion_fraction",
        "remaining_candidate_feature_overlap_count",
        "feature_exclusion_overlap_count",
        "actual_removed_feature_count",
        "undeclared_removed_feature_count",
        "declared_but_not_removed_feature_count",
        "novel_feature_count",
        "pathway_membership_available",
        "feature_provenance_available",
        "feature_set_exact_primary_minus_excluded",
        "feature_exclusion_gate",
        "common_cell_mask_gate",
        "provenance_match_primary",
        "provenance_mismatch_fields",
        "reference_provenance_gate",
        "lpo_design_gate",
        "failure_reason",
    ]
    return pd.DataFrame(rows, columns=columns)


def summarize_trajectory_robustness(
    events: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    candidate_pathways: Iterable[str],
    pathway_membership: Mapping[str, Iterable[str]] | pd.DataFrame | None = None,
    primary_variant_id: Optional[str] = None,
) -> TrajectoryRobustnessResult:
    """Summarize pre-planned trajectory sensitivity analyses without p pooling.

    Parameters
    ----------
    events
        At most one row per variant and pathway.  A supplied ``supported`` (or
        recognized alias) column is used verbatim.  If it is absent, row
        presence is the event call.  P/q/FDR columns are deliberately ignored.
    manifest
        One row per planned analysis variant. Required fields (recognized
        aliases are accepted) are ``variant_id``, ``method``, trajectory
        ``source``, explicit ``orientation``, ``cell_mask_hash``, and either
        ``status`` or a ``planned`` boolean. ``draw_id`` is optional; a planned
        ``variant_kind='draw'`` uses its variant id when no draw id is supplied.
        Failed and incomplete planned rows stay in the denominator. Reference-
        anchored rows are additionally audited for ``reference_id``,
        ``reference_version``/hash, ``mapping_method``, and
        ``reference_fit_scope``. LPO rows must explicitly name their
        ``lpo_pathway``.
    candidate_pathways
        The pre-specified candidate universe.  Every candidate is crossed with
        every planned non-LPO variant, so missing runs remain in denominators.
    pathway_membership
        Mapping or long DataFrame used to verify candidate-specific LPO feature
        removal.  Without it, LPO support is reported but the feature gate
        cannot pass.
    primary_variant_id
        Primary analysis used for direction and provenance anchoring.  If
        omitted, exactly one planned manifest row must be marked primary.

    Notes
    -----
    Event timing is first affinely normalized from each declared pseudotime
    range to relative ``[0, 1]`` coordinates. Reversed trajectories then use
    ``1 - relative_time``. Effect/enrichment direction is not inverted by a
    coordinate reversal. Draw and method summaries are computed separately:
    variants are averaged within an axis unit, then units receive equal weight.
    This function does not calculate a robustness p-value.
    """

    if candidate_pathways is None or isinstance(candidate_pathways, (str, bytes)):
        raise ValueError(
            "candidate_pathways must be a non-string iterable of non-blank pathways"
        )
    raw_candidates = list(candidate_pathways)
    if not raw_candidates or any(not _not_blank(value) for value in raw_candidates):
        raise ValueError("candidate_pathways must contain at least one non-blank pathway")
    candidates = [str(value).strip() for value in raw_candidates]
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate_pathways must be unique; duplicates change no denominator")

    normalized_manifest, manifest_meta = _canonical_manifest(manifest)
    primary = _primary_variant(normalized_manifest, primary_variant_id)
    normalized_events, event_meta = _canonical_events(events, normalized_manifest)
    membership = _membership_map(pathway_membership)
    manifest_diagnostics, gates = _manifest_gates(normalized_manifest, primary)

    planned = normalized_manifest.loc[normalized_manifest["planned"]].copy()
    candidate_frame = pd.DataFrame({"pathway": candidates})
    candidate_frame["__key"] = 1
    planned_frame = planned.copy()
    planned_frame["__key"] = 1
    detail = candidate_frame.merge(planned_frame, on="__key", how="inner").drop(columns="__key")
    event_keep = [
        "variant_id",
        "pathway",
        "event_observed",
        "supported",
        "direction",
        "onset",
        "peak_time",
        "aligned_direction",
        "aligned_onset",
        "aligned_peak_time",
    ]
    detail = detail.merge(
        normalized_events[event_keep],
        on=["variant_id", "pathway"],
        how="left",
        validate="one_to_one",
    )
    # ``eq(True)`` maps unmatched (NaN) rows to False without pandas' legacy
    # object-downcasting behavior.
    detail["event_observed"] = detail["event_observed"].eq(True)
    detail["supported"] = detail["supported"].eq(True)
    for column in (
        "direction",
        "aligned_direction",
        "onset",
        "aligned_onset",
        "peak_time",
        "aligned_peak_time",
    ):
        detail[column] = pd.to_numeric(detail[column], errors="coerce")

    general = detail.loc[~detail["is_lpo"]].copy()
    draw_support = _axis_table(general.loc[general["draw_eligible"]], "draw_id")
    method_support = _axis_table(general, "method_id")
    lpo = _lpo_diagnostics(
        candidates=candidates,
        detail=detail,
        manifest=normalized_manifest,
        membership=membership,
        primary_variant_id=primary,
    )

    primary_manifest = normalized_manifest.loc[
        normalized_manifest["variant_id"].eq(primary)
    ].iloc[0]
    summary_rows: list[dict[str, Any]] = []
    for pathway in candidates:
        candidate = general.loc[general["pathway"].eq(pathway)]
        candidate_draws = draw_support.loc[draw_support["pathway"].eq(pathway)]
        candidate_methods = method_support.loc[method_support["pathway"].eq(pathway)]
        reference_direction, direction_source = _reference_direction(candidate, primary)
        primary_event = candidate.loc[candidate["variant_id"].eq(primary)].iloc[0]
        pathway_lpo = lpo.loc[lpo["pathway"].eq(pathway)]
        supported_lpo = pathway_lpo.loc[pathway_lpo["supported"]]
        lpo_onset = _dispersion(
            supported_lpo.get("aligned_onset", pd.Series(dtype=float))
        )
        lpo_peak = _dispersion(
            supported_lpo.get("aligned_peak_time", pd.Series(dtype=float))
        )
        if not pathway_lpo.empty and np.isfinite(reference_direction):
            lpo_direction = pd.to_numeric(pathway_lpo["aligned_direction"], errors="coerce")
            observed_lpo_direction = np.isfinite(lpo_direction) & pathway_lpo["supported"]
            lpo_direction_consistency = (
                float(lpo_direction[observed_lpo_direction].eq(reference_direction).mean())
                if observed_lpo_direction.any()
                else np.nan
            )
        else:
            lpo_direction_consistency = np.nan
        primary_onset = _numeric(primary_event["aligned_onset"])
        primary_peak = _numeric(primary_event["aligned_peak_time"])
        lpo_onset_shift = (
            float(lpo_onset["median"] - primary_onset)
            if np.isfinite(lpo_onset["median"]) and np.isfinite(primary_onset)
            else np.nan
        )
        lpo_peak_shift = (
            float(lpo_peak["median"] - primary_peak)
            if np.isfinite(lpo_peak["median"]) and np.isfinite(primary_peak)
            else np.nan
        )
        has_lpo = not pathway_lpo.empty
        lpo_feature_gate = bool(has_lpo and pathway_lpo["feature_exclusion_gate"].all())
        lpo_mask_gate = bool(has_lpo and pathway_lpo["common_cell_mask_gate"].all())
        lpo_reference_gate = bool(has_lpo and pathway_lpo["reference_provenance_gate"].all())
        candidate_specific_lpo_gate = bool(
            has_lpo and pathway_lpo["candidate_specific_lpo"].all()
        )
        summary_rows.append(
            {
                "pathway": pathway,
                "primary_variant_id": primary,
                "primary_supported": bool(primary_event["supported"]),
                "primary_aligned_direction": primary_event["aligned_direction"],
                "reference_direction": reference_direction,
                "reference_direction_source": direction_source,
                "n_planned_variants": int(len(candidate)),
                "n_observed_variants": int(candidate["event_observed"].sum()),
                "n_supporting_variants": int(candidate["supported"].sum()),
                "planned_variant_support_rate": float(candidate["supported"].mean()),
                **_axis_summary(
                    candidate_draws, axis="draw_id", reference_direction=reference_direction
                ),
                **_axis_summary(
                    candidate_methods, axis="method_id", reference_direction=reference_direction
                ),
                "n_planned_lpo_variants": int(len(pathway_lpo)),
                "n_observed_lpo_variants": int(pathway_lpo["event_observed"].sum()),
                "n_supporting_lpo_variants": int(pathway_lpo["supported"].sum()),
                "lpo_support_rate": (
                    float(pathway_lpo["supported"].mean()) if has_lpo else np.nan
                ),
                "lpo_direction_consistency": lpo_direction_consistency,
                "lpo_onset_n": lpo_onset["n"],
                "lpo_onset_median": lpo_onset["median"],
                "lpo_onset_mad": lpo_onset["mad"],
                "lpo_onset_iqr": lpo_onset["iqr"],
                "lpo_onset_shift_from_primary": lpo_onset_shift,
                "lpo_peak_n": lpo_peak["n"],
                "lpo_peak_median": lpo_peak["median"],
                "lpo_peak_mad": lpo_peak["mad"],
                "lpo_peak_iqr": lpo_peak["iqr"],
                "lpo_peak_shift_from_primary": lpo_peak_shift,
                "candidate_specific_lpo_gate": candidate_specific_lpo_gate,
                "lpo_feature_exclusion_gate": lpo_feature_gate,
                "lpo_common_cell_mask_gate": lpo_mask_gate,
                "lpo_reference_provenance_gate": lpo_reference_gate,
                "reference_provenance_gate": gates["reference_provenance_gate"],
                "common_cell_mask_gate": gates["common_cell_mask_gate"],
                "robustness_design_gates_pass": bool(
                    gates["reference_provenance_gate"]
                    and gates["common_cell_mask_gate"]
                    and candidate_specific_lpo_gate
                    and lpo_feature_gate
                    and lpo_mask_gate
                    and lpo_reference_gate
                ),
                "p_values_combined": False,
            }
        )

    pathway_summary = pd.DataFrame(summary_rows)
    safe_manifest = normalized_manifest.copy()
    safe_manifest["feature_genes"] = safe_manifest["feature_genes"].map(
        lambda values: ";".join(sorted(values))
    )
    safe_manifest["excluded_genes"] = safe_manifest["excluded_genes"].map(
        lambda values: ";".join(sorted(values))
    )
    variant_columns = [
        "pathway",
        "variant_id",
        "planned",
        "status",
        "variant_kind",
        "is_primary",
        "is_lpo",
        "lpo_pathway",
        "draw_id",
        "draw_eligible",
        "method_id",
        "source",
        "orientation",
        "reference_id",
        "reference_version",
        "mapping_method",
        "reference_fit_scope",
        "cell_mask_id",
        "event_observed",
        "supported",
        "direction",
        "aligned_direction",
        "onset",
        "aligned_onset",
        "peak_time",
        "aligned_peak_time",
    ]
    metadata = {
        **manifest_meta,
        **event_meta,
        "primary_variant_id": primary,
        "primary_reference_id": primary_manifest["reference_id"],
        "primary_reference_version": primary_manifest["reference_version"],
        "primary_source": primary_manifest["source"],
        "primary_cell_mask_id": primary_manifest["cell_mask_id"],
        "candidate_pathways": candidates,
        "planned_variant_count": int(planned.loc[~planned["is_lpo"]].shape[0]),
        "planned_lpo_variant_count": int(planned.loc[planned["is_lpo"]].shape[0]),
        "p_values_combined": False,
        "p_value_policy": "ignored; support decisions are supplied by the event producer",
        "weighting": "variants within draw/method unit, then equal weight across units",
        "missing_planned_event_policy": "zero support; retained in denominator",
        "timing_alignment": (
            "declared pseudotime range normalized to relative [0,1]; "
            "reverse orientation uses 1-relative_time"
        ),
        "direction_alignment": (
            "effect/enrichment direction unchanged by pseudotime orientation"
        ),
    }
    return TrajectoryRobustnessResult(
        pathway_summary=pathway_summary,
        variant_support=detail[variant_columns].reset_index(drop=True),
        draw_support=draw_support.reset_index(drop=True),
        method_support=method_support.reset_index(drop=True),
        lpo_diagnostics=lpo.reset_index(drop=True),
        manifest_diagnostics=manifest_diagnostics.reset_index(drop=True),
        manifest=safe_manifest.reset_index(drop=True),
        metadata=metadata,
    )


__all__ = ["TrajectoryRobustnessResult", "summarize_trajectory_robustness"]
