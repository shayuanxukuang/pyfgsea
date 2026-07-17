"""Predeclared pathway-family inference and exploratory redundancy summaries.

The inferential entry point reuses the complete whole-donor permutation null
from a covariate-adjusted fit.  Family definitions must be supplied explicitly;
effect-curve clustering is kept in a separate exploratory API so that a family
cannot be learned and formally tested on the same outcomes by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .trajectory_covariate_pseudobulk import (
    CovariateAdjustedDonorPseudobulkResult,
)
from .trajectory_dynamic_leading_edge import _fitted_source_fingerprint
from .trajectory_pseudobulk import (
    _bh_adjust,
    _by_adjust,
    _normalize_statistic,
    _normalize_tail,
    _test_scale,
)


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame()


@dataclass
class PathwayFamilyInferenceResult:
    """Auditable family gates built from a fitted donor-permutation reference."""

    family_tests: pd.DataFrame = field(default_factory=_empty_frame)
    member_tests: pd.DataFrame = field(default_factory=_empty_frame)
    family_membership: pd.DataFrame = field(default_factory=_empty_frame)
    family_effect_curves: pd.DataFrame = field(default_factory=_empty_frame)
    pairwise_redundancy: pd.DataFrame = field(default_factory=_empty_frame)
    leading_edge_genes: pd.DataFrame = field(default_factory=_empty_frame)
    null_statistics: pd.DataFrame = field(default_factory=_empty_frame)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_tables(self) -> dict[str, pd.DataFrame]:
        return {
            "family_tests": self.family_tests.copy(),
            "member_tests": self.member_tests.copy(),
            "family_membership": self.family_membership.copy(),
            "family_effect_curves": self.family_effect_curves.copy(),
            "pairwise_redundancy": self.pairwise_redundancy.copy(),
            "leading_edge_genes": self.leading_edge_genes.copy(),
            "null_statistics": self.null_statistics.copy(),
        }


@dataclass
class PathwayRedundancyResult:
    """Exploratory overlap/curve clusters with no inferential interpretation."""

    clusters: pd.DataFrame = field(default_factory=_empty_frame)
    pairwise_redundancy: pd.DataFrame = field(default_factory=_empty_frame)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_tables(self) -> dict[str, pd.DataFrame]:
        return {
            "clusters": self.clusters.copy(),
            "pairwise_redundancy": self.pairwise_redundancy.copy(),
        }


def _nonblank(value: Any, label: str) -> str:
    if value is None or pd.isna(value) or not str(value).strip():
        raise ValueError(f"{label} must be non-blank")
    return str(value).strip()


def _normalize_families(
    definitions: Mapping[str, Iterable[str]] | pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    if isinstance(definitions, pd.DataFrame):
        redundancy_metadata = definitions.attrs.get("pathway_redundancy", {})
        if isinstance(redundancy_metadata, Mapping) and (
            redundancy_metadata.get("outcome_derived") is True
            or redundancy_metadata.get("formal_inference") is False
        ):
            raise ValueError(
                "Outcome-derived pathway redundancy clusters cannot be reused "
                "as predeclared family definitions on the same data"
            )
        if "formal_family_test" in definitions.columns:
            formal_values = definitions["formal_family_test"]
            try:
                formal_flags = _strict_bool_series(
                    formal_values, "formal_family_test"
                )
            except ValueError as exc:
                raise ValueError(
                    "family_definitions has an invalid or non-formal provenance flag"
                ) from exc
            if not formal_flags.all():
                raise ValueError(
                    "family_definitions is explicitly marked as non-formal/outcome-derived"
                )
        family_col = "family" if "family" in definitions else "Family"
        pathway_col = "Pathway" if "Pathway" in definitions else "pathway"
        if family_col not in definitions or pathway_col not in definitions:
            raise ValueError(
                "family_definitions DataFrame requires family and Pathway columns"
            )
        for item in definitions[[family_col, pathway_col]].itertuples(index=False):
            rows.append(
                {
                    "family": _nonblank(item[0], "family"),
                    "Pathway": _nonblank(item[1], "Pathway"),
                }
            )
    elif isinstance(definitions, Mapping):
        for family, members in definitions.items():
            family_name = _nonblank(family, "family")
            if isinstance(members, (str, bytes)):
                raise ValueError(
                    f"Members of family '{family_name}' must be an iterable, not a string"
                )
            member_list = list(members)
            if not member_list:
                raise ValueError(f"Family '{family_name}' has no pathways")
            for pathway in member_list:
                rows.append(
                    {
                        "family": family_name,
                        "Pathway": _nonblank(pathway, "Pathway"),
                    }
                )
    else:
        raise TypeError("family_definitions must be a mapping or DataFrame")
    frame = pd.DataFrame(rows, columns=["family", "Pathway"])
    if frame.empty:
        raise ValueError("At least one pathway family is required")
    duplicates = frame.duplicated(["family", "Pathway"], keep=False)
    if duplicates.any():
        raise ValueError("family_definitions contains duplicate family/pathway rows")
    repeated = frame[frame.duplicated("Pathway", keep=False)]["Pathway"].unique()
    if len(repeated):
        raise ValueError(
            "Formal family inference requires disjoint families; repeated pathways: "
            f"{sorted(map(str, repeated))}"
        )
    return frame.sort_values(["family", "Pathway"]).reset_index(drop=True)


def _family_hash(membership: pd.DataFrame, definition_id: str) -> str:
    payload = {
        "family_definition_id": definition_id,
        "membership": membership[["family", "Pathway"]].to_dict("records"),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _available_pathways(fitted) -> set[str]:
    required = {"Pathway", "observed_calibration_statistic"}
    missing = required - set(fitted.pathway_tests.columns)
    if missing:
        raise ValueError(f"fitted.pathway_tests is missing columns: {sorted(missing)}")
    pathways = fitted.pathway_tests["Pathway"].astype(str)
    if pathways.duplicated().any():
        raise ValueError("fitted.pathway_tests must contain one row per pathway")
    return set(pathways)


def _null_matrix(
    fitted, pathways: Sequence[str]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    null = fitted.null_statistics.copy()
    required = {"perm_id", "mapping_hash", "Pathway", "calibration_statistic"}
    missing = required - set(null.columns)
    if null.empty or missing:
        raise ValueError(
            "Formal family inference requires fitted null_statistics from "
            "return_null_statistics=True"
        )
    null["Pathway"] = null["Pathway"].astype(str)
    available = set(fitted.pathway_tests["Pathway"].astype(str))
    null_pathways = set(null["Pathway"])
    if null_pathways != available:
        raise ValueError(
            "null_statistics must contain the complete fitted pathway universe"
        )
    if null["mapping_hash"].isna().any() or null["mapping_hash"].astype(str).str.strip().eq("").any():
        raise ValueError("Every null mapping requires a non-blank mapping_hash")
    if null.duplicated(["perm_id", "Pathway"]).any():
        raise ValueError("null_statistics has duplicate perm_id/pathway rows")
    hash_counts = null.groupby("perm_id")["mapping_hash"].nunique(dropna=False)
    if not hash_counts.eq(1).all():
        raise ValueError("Each perm_id must have exactly one whole-donor mapping hash")
    mapping_to_perm = null[["perm_id", "mapping_hash"]].drop_duplicates()
    if mapping_to_perm["mapping_hash"].duplicated().any():
        raise ValueError("Each mapping_hash must identify exactly one perm_id")
    counts = null.groupby("perm_id")["Pathway"].nunique()
    if not counts.eq(len(available)).all():
        raise ValueError(
            "null_statistics must be rectangular over every fitted pathway and mapping"
        )
    expected_mappings = fitted.metadata.get("n_null_mappings_evaluated")
    if expected_mappings is None:
        raise ValueError(
            "fitted metadata must record n_null_mappings_evaluated for family inference"
        )
    if int(expected_mappings) != int(null["perm_id"].nunique()):
        raise ValueError(
            "null_statistics does not contain every evaluated whole-donor mapping"
        )
    if fitted.metadata.get("identity_mapping_in_null") is not False:
        raise ValueError(
            "Source metadata must confirm identity_mapping_in_null=False for the +1 rule"
        )
    pivot = null.pivot(
        index=["perm_id", "mapping_hash"],
        columns="Pathway",
        values="calibration_statistic",
    ).sort_index()
    full_values = pivot.to_numpy(dtype=float)
    if not np.isfinite(full_values).all():
        raise ValueError("null_statistics contains non-finite calibration statistics")
    selected_values = pivot.reindex(columns=list(pathways)).to_numpy(dtype=float)
    global_values = np.max(full_values, axis=1)
    return (
        pivot.reset_index()[["perm_id", "mapping_hash"]],
        selected_values,
        global_values,
    )


def _gene_sets(fitted, pathways: Sequence[str]) -> dict[str, set[str]]:
    membership = fitted.pathway_membership.copy()
    required = {"Pathway", "gene"}
    missing = required - set(membership.columns)
    if missing:
        raise ValueError(f"fitted.pathway_membership is missing columns: {sorted(missing)}")
    membership["Pathway"] = membership["Pathway"].astype(str)
    membership["gene"] = membership["gene"].astype(str)
    sets = {
        pathway: set(
            membership.loc[membership["Pathway"].eq(pathway), "gene"].tolist()
        )
        for pathway in pathways
    }
    missing_sets = [pathway for pathway, genes in sets.items() if not genes]
    if missing_sets:
        raise ValueError(f"Pathways have no recorded member genes: {missing_sets}")
    return sets


def _curve_vectors(fitted, pathways: Sequence[str]) -> tuple[list[int], dict[str, np.ndarray]]:
    curves = fitted.effect_curves.copy()
    required = {"Pathway", "bin_id", "beta_condition"}
    missing = required - set(curves.columns)
    if missing:
        raise ValueError(f"fitted.effect_curves is missing columns: {sorted(missing)}")
    curves["Pathway"] = curves["Pathway"].astype(str)
    subset = curves.loc[curves["Pathway"].isin(pathways)].copy()
    bins = sorted(pd.to_numeric(subset["bin_id"], errors="raise").astype(int).unique())
    vectors: dict[str, np.ndarray] = {}
    for pathway in pathways:
        rows = subset.loc[subset["Pathway"].eq(pathway)].copy()
        if rows["bin_id"].duplicated().any():
            raise ValueError(f"Duplicate effect-curve bins for pathway '{pathway}'")
        vector = (
            rows.assign(bin_id=pd.to_numeric(rows["bin_id"]).astype(int))
            .set_index("bin_id")["beta_condition"]
            .reindex(bins)
            .to_numpy(dtype=float)
        )
        if not np.isfinite(vector).all():
            raise ValueError(f"Incomplete/non-finite effect curve for pathway '{pathway}'")
        vectors[pathway] = vector
    return bins, vectors


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or np.std(first) <= 1e-14 or np.std(second) <= 1e-14:
        return np.nan
    return float(np.corrcoef(first, second)[0, 1])


def _pairwise_table(
    fitted,
    pathways: Sequence[str],
    family_lookup: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    genes = _gene_sets(fitted, pathways)
    _, curves = _curve_vectors(fitted, pathways)
    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(pathways):
        for right in pathways[left_index + 1 :]:
            intersection = genes[left] & genes[right]
            union = genes[left] | genes[right]
            minimum = min(len(genes[left]), len(genes[right]))
            correlation = _correlation(curves[left], curves[right])
            rows.append(
                {
                    "pathway_a": left,
                    "pathway_b": right,
                    "same_predeclared_family": (
                        bool(family_lookup)
                        and family_lookup.get(left) == family_lookup.get(right)
                    ),
                    "n_genes_a": len(genes[left]),
                    "n_genes_b": len(genes[right]),
                    "n_shared_genes": len(intersection),
                    "gene_jaccard": len(intersection) / len(union),
                    "gene_overlap_coefficient": len(intersection) / minimum,
                    "effect_curve_correlation": correlation,
                    "absolute_effect_curve_correlation": (
                        abs(correlation) if np.isfinite(correlation) else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _family_curves(fitted, membership: pd.DataFrame) -> pd.DataFrame:
    curves = fitted.effect_curves.merge(membership, on="Pathway", how="inner")
    rows = []
    passthrough = [
        column
        for column in ("bin_id", "bin_left", "bin_right", "bin_mid", "bin_width")
        if column in curves
    ]
    for keys, group in curves.groupby(["family", *passthrough], sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(["family", *passthrough], keys))
        values = group["beta_condition"].to_numpy(dtype=float)
        row.update(
            {
                "n_member_pathways": int(group["Pathway"].nunique()),
                "mean_member_beta_condition": float(np.mean(values)),
                "median_member_beta_condition": float(np.median(values)),
                "minimum_member_beta_condition": float(np.min(values)),
                "maximum_member_beta_condition": float(np.max(values)),
                "descriptive_only": True,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _strict_bool_series(values: pd.Series, label: str) -> pd.Series:
    parsed: list[bool] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            parsed.append(bool(value))
        elif isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            parsed.append(value.strip().lower() == "true")
        else:
            raise ValueError(
                f"{label} must contain only booleans or canonical true/false strings"
            )
    return pd.Series(parsed, index=values.index, dtype=bool)


def _leading_edge_table(dynamic, membership: pd.DataFrame, fitted) -> pd.DataFrame:
    columns = [
        "family",
        "gene",
        "n_family_pathways",
        "n_pathways_with_gene",
        "pathway_fraction",
        "pathways_with_gene",
        "event_count",
        "signed_integrated_contribution_sum",
        "absolute_integrated_contribution_sum",
        "family_gene_role",
        "shared_leading_edge_membership",
        "n_positive_pathways",
        "n_negative_pathways",
        "n_zero_pathways",
        "direction_consistent_across_pathways",
        "gene_level_inference",
    ]
    if dynamic is None:
        return pd.DataFrame(columns=columns)
    dynamic_metadata = getattr(dynamic, "metadata", None)
    if not isinstance(dynamic_metadata, Mapping):
        raise ValueError("dynamic leading-edge input requires source provenance metadata")
    provenance_pairs = {
        "source_method": fitted.metadata.get("method"),
        "source_pathway_family_hash": fitted.metadata.get("pathway_family_hash"),
        "source_gene_universe_hash": fitted.metadata.get("gene_universe_hash"),
        "source_fit_fingerprint": _fitted_source_fingerprint(fitted),
    }
    for dynamic_key, expected in provenance_pairs.items():
        observed = dynamic_metadata.get(dynamic_key)
        if expected is None or observed is None or str(observed) != str(expected):
            raise ValueError(
                f"dynamic leading-edge {dynamic_key} does not match the fitted source"
            )
    events = dynamic.event_gene_summary.copy()
    required = {
        "Pathway",
        "gene",
        "event_id",
        "in_dynamic_leading_edge",
        "integrated_contribution",
        "integrated_absolute_contribution",
    }
    missing = required - set(events.columns)
    if missing:
        raise ValueError(
            f"dynamic leading-edge summary is missing columns: {sorted(missing)}"
        )
    if events.duplicated(["Pathway", "event_id", "gene"]).any():
        raise ValueError(
            "dynamic leading-edge summary has duplicate Pathway/event_id/gene rows"
        )
    for column in (
        "integrated_contribution",
        "integrated_absolute_contribution",
    ):
        numeric = pd.to_numeric(events[column], errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"dynamic leading-edge {column} must be finite numeric")
        events[column] = numeric.astype(float)
    events["in_dynamic_leading_edge"] = _strict_bool_series(
        events["in_dynamic_leading_edge"], "in_dynamic_leading_edge"
    )
    selected = set(membership["Pathway"].astype(str))
    covered = set(events["Pathway"].astype(str))
    absent = sorted(selected - covered)
    if absent:
        raise ValueError(
            "dynamic leading-edge input must cover every family member pathway; "
            f"missing {absent}"
        )
    events["Pathway"] = events["Pathway"].astype(str)
    events = events.loc[events["Pathway"].isin(selected)].copy()
    member_genes = {
        pathway: set(
            fitted.pathway_membership.loc[
                fitted.pathway_membership["Pathway"].astype(str).eq(pathway), "gene"
            ].astype(str)
        )
        for pathway in membership["Pathway"].astype(str)
    }
    invalid_genes = events.loc[
        events.apply(
            lambda row: str(row["gene"])
            not in member_genes.get(str(row["Pathway"]), set()),
            axis=1,
        ),
        ["Pathway", "event_id", "gene"],
    ]
    if not invalid_genes.empty:
        raise ValueError(
            "dynamic leading-edge genes must belong to their fitted pathway membership"
        )
    events = events.loc[
        events["in_dynamic_leading_edge"]
    ].merge(membership, on="Pathway", how="inner")
    family_sizes = membership.groupby("family")["Pathway"].nunique().to_dict()
    rows = []
    for (family, gene), group in events.groupby(["family", "gene"], sort=True):
        pathways = sorted(group["Pathway"].unique())
        n_family = int(family_sizes[family])
        pathway_contribution = group.groupby("Pathway")[
            "integrated_contribution"
        ].sum()
        signs = np.sign(pathway_contribution.to_numpy(dtype=float))
        n_positive = int(np.sum(signs > 0))
        n_negative = int(np.sum(signs < 0))
        n_zero = int(np.sum(signs == 0))
        direction_consistent = bool(
            n_zero == 0 and not (n_positive > 0 and n_negative > 0)
        )
        shared = len(pathways) > 1
        rows.append(
            {
                "family": family,
                "gene": str(gene),
                "n_family_pathways": n_family,
                "n_pathways_with_gene": len(pathways),
                "pathway_fraction": len(pathways) / n_family,
                "pathways_with_gene": ";".join(pathways),
                "event_count": int(group[["Pathway", "event_id"]].drop_duplicates().shape[0]),
                "signed_integrated_contribution_sum": float(
                    group["integrated_contribution"].sum()
                ),
                "absolute_integrated_contribution_sum": float(
                    group["integrated_absolute_contribution"].sum()
                ),
                "family_gene_role": (
                    "shared_direction_consistent"
                    if shared and direction_consistent
                    else "shared_direction_mixed"
                    if shared
                    else "pathway_specific"
                ),
                "shared_leading_edge_membership": shared,
                "n_positive_pathways": n_positive,
                "n_negative_pathways": n_negative,
                "n_zero_pathways": n_zero,
                "direction_consistent_across_pathways": direction_consistent,
                "gene_level_inference": False,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def run_predeclared_pathway_family_inference(
    fitted: CovariateAdjustedDonorPseudobulkResult,
    family_definitions: Mapping[str, Iterable[str]] | pd.DataFrame,
    *,
    family_definition_id: str,
    dynamic_leading_edge=None,
    alpha: float = 0.05,
    return_null_statistics: bool = False,
) -> PathwayFamilyInferenceResult:
    """Test predeclared pathway families on the fitted whole-donor null.

    Each family statistic is the maximum calibrated statistic among its member
    pathways. A second maximum across families gives single-step family maxT.
    Member interpretation is declared supported only when both the family maxT
    gate and the source fit's global pathway maxT gate pass. The latter retains
    the original pathway-wide maxT boundary, whose strong-FWER interpretation
    still requires the source model's subset-pivotality condition.
    """

    definition_id = _nonblank(family_definition_id, "family_definition_id")
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha must be numeric") from exc
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    membership = _normalize_families(family_definitions)
    available = _available_pathways(fitted)
    selected = membership["Pathway"].tolist()
    missing = sorted(set(selected) - available)
    if missing:
        raise ValueError(f"Family pathways are absent from fitted result: {missing}")

    tests = fitted.pathway_tests.copy()
    tests["Pathway"] = tests["Pathway"].astype(str)
    source = tests.set_index("Pathway").loc[selected].reset_index()
    required_source = {
        "p_maxT",
        "observed_calibration_statistic",
        "primary_statistic",
        "tail",
        "calibration_scale",
    }
    missing_source = required_source - set(source.columns)
    if missing_source:
        raise ValueError(f"fitted.pathway_tests is missing columns: {sorted(missing_source)}")
    for column in ("primary_statistic", "tail", "calibration_scale"):
        values = source[column].astype(str).unique()
        if len(values) != 1:
            raise ValueError(
                f"All family members must share one source {column}; found {values.tolist()}"
            )
    statistic = _normalize_statistic(source["primary_statistic"].iloc[0])
    tail = _normalize_tail(source["tail"].iloc[0])
    calibration_scale = str(source["calibration_scale"].iloc[0])
    observed_raw_lookup = source.set_index("Pathway")[
        "observed_calibration_statistic"
    ].astype(float)
    if not np.isfinite(observed_raw_lookup.to_numpy()).all():
        raise ValueError("Observed pathway calibration statistics must be finite")
    source_p_max = pd.to_numeric(source["p_maxT"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(source_p_max).all() or np.any(
        (source_p_max < 0) | (source_p_max > 1)
    ):
        raise ValueError("Source pathway p_maxT values must be finite and in [0, 1]")
    observed_lookup = pd.Series(
        _test_scale(observed_raw_lookup.to_numpy(), statistic, tail),
        index=observed_raw_lookup.index,
        name="observed_test_scale_statistic",
    )

    mapping_rows, null_matrix, global_pathway_null = _null_matrix(fitted, selected)
    denominator = len(mapping_rows) + 1
    tolerance = 1e-12
    recomputed_global_p_max = np.asarray(
        [
            (1 + np.sum(global_pathway_null >= value - tolerance)) / denominator
            for value in observed_lookup.to_numpy(dtype=float)
        ],
        dtype=float,
    )
    if not np.allclose(source_p_max, recomputed_global_p_max, atol=1e-12, rtol=0):
        raise ValueError(
            "Source pathway p_maxT does not match the complete audited null mapping stream"
        )
    recomputed_p_max_lookup = pd.Series(
        recomputed_global_p_max, index=source["Pathway"].astype(str)
    )
    path_index = {pathway: index for index, pathway in enumerate(selected)}
    families = sorted(membership["family"].unique())
    observed_family = np.empty(len(families), dtype=float)
    null_family = np.empty((len(mapping_rows), len(families)), dtype=float)
    leading_members: list[str] = []
    for family_index, family in enumerate(families):
        members = membership.loc[membership["family"].eq(family), "Pathway"].tolist()
        member_indices = [path_index[pathway] for pathway in members]
        member_observed = observed_lookup.reindex(members)
        observed_family[family_index] = float(member_observed.max())
        leading_members.append(
            sorted(member_observed[member_observed.eq(member_observed.max())].index)[0]
        )
        null_family[:, family_index] = np.max(null_matrix[:, member_indices], axis=1)
    global_family_null = np.max(null_family, axis=1)
    family_p_raw = np.asarray(
        [
            (1 + np.sum(null_family[:, index] >= value - tolerance)) / denominator
            for index, value in enumerate(observed_family)
        ],
        dtype=float,
    )
    family_p_max = np.asarray(
        [
            (1 + np.sum(global_family_null >= value - tolerance)) / denominator
            for value in observed_family
        ],
        dtype=float,
    )
    family_tests = pd.DataFrame(
        {
            "family": families,
            "n_member_pathways": [
                int(membership["family"].eq(family).sum()) for family in families
            ],
            "primary_statistic": statistic,
            "tail": tail,
            "calibration_scale": calibration_scale,
            "observed_family_max_calibration_statistic": observed_family,
            "observed_family_max_test_scale_statistic": observed_family,
            "leading_member_pathway": leading_members,
            "p_family_raw": family_p_raw,
            "q_family_bh": _bh_adjust(family_p_raw),
            "q_family_by": _by_adjust(family_p_raw),
            "p_family_maxT": family_p_max,
            "family_maxT_gate": family_p_max <= alpha,
            "alpha": alpha,
            "family_definition_id": definition_id,
        }
    )

    family_index = {family: index for index, family in enumerate(families)}
    member_rows = []
    for item in membership.itertuples(index=False):
        source_row = source.loc[source["Pathway"].eq(item.Pathway)].iloc[0]
        observed_raw = float(source_row["observed_calibration_statistic"])
        observed = float(observed_lookup.loc[item.Pathway])
        within_null = null_family[:, family_index[item.family]]
        within_p = (1 + np.sum(within_null >= observed - tolerance)) / denominator
        family_row = family_tests.loc[family_tests["family"].eq(item.family)].iloc[0]
        member_rows.append(
            {
                "family": item.family,
                "Pathway": item.Pathway,
                "observed_calibration_statistic": observed_raw,
                "observed_test_scale_statistic": observed,
                "p_pathway_raw": float(source_row.get("p_raw", np.nan)),
                "q_pathway_bh": float(source_row.get("q_bh", np.nan)),
                "q_pathway_by": float(source_row.get("q_by", np.nan)),
                "p_pathway_within_family_maxT": float(within_p),
                "p_pathway_global_maxT": float(
                    recomputed_p_max_lookup.loc[item.Pathway]
                ),
                "p_family_maxT": float(family_row["p_family_maxT"]),
                "family_maxT_gate": bool(family_row["family_maxT_gate"]),
                "global_pathway_maxT_gate": bool(
                    recomputed_p_max_lookup.loc[item.Pathway] <= alpha
                ),
                "hierarchical_supported": bool(
                    family_row["family_maxT_gate"]
                    and recomputed_p_max_lookup.loc[item.Pathway] <= alpha
                ),
                "alpha": alpha,
            }
        )
    member_tests = pd.DataFrame(member_rows)

    family_lookup = dict(zip(membership["Pathway"], membership["family"]))
    pairwise = _pairwise_table(fitted, selected, MappingProxyType(family_lookup))
    effects = _family_curves(fitted, membership)
    leading = _leading_edge_table(dynamic_leading_edge, membership, fitted)
    family_id_hash = _family_hash(membership, definition_id)
    null_rows = []
    if return_null_statistics:
        for row_index, mapping in mapping_rows.iterrows():
            for index, family in enumerate(families):
                null_rows.append(
                    {
                        "perm_id": mapping["perm_id"],
                        "mapping_hash": mapping["mapping_hash"],
                        "family": family,
                        "family_max_calibration_statistic": float(
                            null_family[row_index, index]
                        ),
                        "global_family_max_calibration_statistic": float(
                            global_family_null[row_index]
                        ),
                    }
                )
    null_output = pd.DataFrame(null_rows)
    metadata = {
        "method": "predeclared_pathway_family_maxT",
        "family_definition_id": definition_id,
        "family_definition_sha256": family_id_hash,
        "n_families": len(families),
        "n_member_pathways": len(selected),
        "n_null_mappings": len(mapping_rows),
        "permutation_denominator_including_identity": denominator,
        "source_exactness_status": fitted.metadata.get("exactness_status"),
        "source_pathway_family_hash": fitted.metadata.get("pathway_family_hash"),
        "family_statistic": "maximum member tail-transformed calibration statistic",
        "source_primary_statistic": statistic,
        "source_tail": tail,
        "source_calibration_scale": calibration_scale,
        "observed_statistic_transform": (
            "same _test_scale transform as source null calibration_statistic"
        ),
        "family_multiple_testing": "single_step_maxT_across_predeclared_families",
        "family_maxT_strong_fwer_condition": "requires_subset_pivotality",
        "member_support_rule": (
            "family_maxT_gate AND source_global_pathway_maxT_gate"
        ),
        "source_global_pathway_maxT_recomputed_and_verified": True,
        "within_family_member_p_role": "descriptive sensitivity; not the final global gate",
        "effect_curve_summary_role": "descriptive; not a fitted family effect",
        "family_definitions_must_be_outcome_independent": True,
        "data_driven_cluster_inference_allowed": False,
        "leading_edge_gene_inference": False,
        "alpha": alpha,
    }
    for table in (
        family_tests,
        member_tests,
        membership,
        effects,
        pairwise,
        leading,
        null_output,
    ):
        table.attrs["pathway_family_inference"] = metadata.copy()
    return PathwayFamilyInferenceResult(
        family_tests=family_tests,
        member_tests=member_tests,
        family_membership=membership,
        family_effect_curves=effects,
        pairwise_redundancy=pairwise,
        leading_edge_genes=leading,
        null_statistics=null_output,
        metadata=metadata,
    )


def cluster_exploratory_pathway_redundancy(
    fitted: CovariateAdjustedDonorPseudobulkResult,
    *,
    pathways: Optional[Sequence[str]] = None,
    min_gene_jaccard: float = 0.25,
    min_effect_curve_correlation: float = 0.8,
) -> PathwayRedundancyResult:
    """Cluster redundant pathways descriptively using overlap and curve shape.

    Connected components join pathway pairs passing both thresholds. Clusters
    are outcome-derived and therefore must not be used as formal family tests
    on the same fitted data.
    """

    available = sorted(_available_pathways(fitted))
    if pathways is None:
        selected = available
    else:
        if isinstance(pathways, (str, bytes)):
            raise ValueError("pathways must be a sequence, not a string")
        selected = [_nonblank(value, "pathway") for value in pathways]
        if len(set(selected)) != len(selected):
            raise ValueError("pathways must be unique")
        missing = sorted(set(selected) - set(available))
        if missing:
            raise ValueError(f"Selected pathways are absent from fitted result: {missing}")
    if not selected:
        raise ValueError("At least one pathway is required")
    for name, value in {
        "min_gene_jaccard": min_gene_jaccard,
        "min_effect_curve_correlation": min_effect_curve_correlation,
    }.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if not 0 <= numeric <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
        if name == "min_gene_jaccard":
            min_gene_jaccard = numeric
        else:
            min_effect_curve_correlation = numeric

    pairwise = _pairwise_table(fitted, selected)
    parent = {pathway: pathway for pathway in selected}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            low, high = sorted((left_root, right_root))
            parent[high] = low

    for row in pairwise.itertuples(index=False):
        if (
            row.gene_jaccard >= min_gene_jaccard
            and np.isfinite(row.effect_curve_correlation)
            and row.effect_curve_correlation >= min_effect_curve_correlation
        ):
            union(row.pathway_a, row.pathway_b)
    components: dict[str, list[str]] = {}
    for pathway in selected:
        components.setdefault(find(pathway), []).append(pathway)
    ordered = sorted((sorted(members) for members in components.values()), key=lambda x: x[0])
    tests = fitted.pathway_tests.copy().set_index("Pathway")
    cluster_rows = []
    for cluster_number, members in enumerate(ordered, start=1):
        cluster_id = f"cluster_{cluster_number:03d}"
        representative = sorted(
            members,
            key=lambda pathway: (
                -float(tests.loc[pathway].get("integrated_absolute_effect", 0.0)),
                pathway,
            ),
        )[0]
        for pathway in members:
            cluster_rows.append(
                {
                    "cluster_id": cluster_id,
                    "Pathway": pathway,
                    "cluster_size": len(members),
                    "descriptive_representative": representative,
                    "is_descriptive_representative": pathway == representative,
                    "source_p_raw": float(tests.loc[pathway].get("p_raw", np.nan)),
                    "source_p_maxT": float(tests.loc[pathway].get("p_maxT", np.nan)),
                    "formal_family_test": False,
                }
            )
    clusters = pd.DataFrame(cluster_rows)
    metadata = {
        "method": "exploratory_overlap_and_effect_curve_connected_components",
        "min_gene_jaccard": float(min_gene_jaccard),
        "min_effect_curve_correlation": float(min_effect_curve_correlation),
        "n_pathways": len(selected),
        "n_clusters": len(ordered),
        "outcome_derived": True,
        "formal_inference": False,
        "reuse_as_predeclared_family_on_same_data": False,
    }
    clusters.attrs["pathway_redundancy"] = metadata.copy()
    pairwise.attrs["pathway_redundancy"] = metadata.copy()
    return PathwayRedundancyResult(
        clusters=clusters,
        pairwise_redundancy=pairwise,
        metadata=metadata,
    )


__all__ = [
    "PathwayFamilyInferenceResult",
    "PathwayRedundancyResult",
    "run_predeclared_pathway_family_inference",
    "cluster_exploratory_pathway_redundancy",
]
