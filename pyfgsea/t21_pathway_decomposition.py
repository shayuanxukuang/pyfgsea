"""Chromosome-21 pathway sensitivity decomposition for the T21 workflow.

This module constructs scorer-compatible total, trans-chromosomal, and
chromosome-21-contribution gene sets and reconstructs their fitted effects.
The decomposition is an *operational sensitivity analysis*: it describes how
the pathway score changes when chromosome-21 members are removed.  It is not a
causal mediation analysis, proof of trans regulation, or gene-level causal
attribution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd


_SCHEMA_VERSION = "1.0.0"
_TOTAL_SUFFIX = "::total"
_TRANS_SUFFIX = "::trans"
_CONTRIBUTION_SUFFIX = "::chr21_contribution"
_COMPONENTS = ("total", "trans", "chr21_contribution")
_REQUIRED_SCORER = "weighted_mean_gene_z_across_donor_bin_pseudobulk"

_MEMBERSHIP_COLUMNS = [
    "Pathway",
    "base_pathway",
    "component",
    "gene_id",
    "weight",
    "original_weight",
    "is_chr21",
    "pathway_size",
    "score_abs_weight_sum",
    "A",
    "B",
    "C",
    "effect_multiplier",
]

_SUMMARY_COLUMNS = [
    "Pathway",
    "has_chr21",
    "n_total_genes",
    "n_chr21_genes",
    "n_trans_genes",
    "A",
    "B",
    "C",
    "total_expanded_pathway",
    "trans_expanded_pathway",
    "chr21_contribution_expanded_pathway",
    "trans_status",
    "chr21_contribution_status",
    "total_formal_inference",
    "trans_formal_inference",
    "chr21_contribution_formal_inference",
    "closure_contract",
]

_GRID_COLUMNS = [
    "bin_id",
    "bin_left",
    "bin_right",
    "bin_mid",
    "bin_width",
    "residual_df",
]
_LINEAR_EFFECT_COLUMNS = [
    "adjusted_control_activity",
    "adjusted_case_activity",
    "beta_condition",
]
_SCALE_EFFECT_CURVE_COLUMNS = [
    *_LINEAR_EFFECT_COLUMNS,
    "condition_standard_error",
    "residual_sd",
]
_CURVE_INFERENCE_COLUMNS = [
    "condition_t",
    "pointwise_p",
    "within_pathway_maxT_p",
    "global_curve_maxT_p",
]
_TEST_EFFECT_COLUMNS = [
    "observed_statistic",
    "observed_effect_statistic",
    "max_absolute_effect",
    "integrated_absolute_effect",
    "l2_effect",
    "signed_integral",
    "peak_effect",
    "median_pointwise_mde",
    "maximum_pointwise_mde",
]
_TEST_P_COLUMNS = [
    "p_raw",
    "q_bh",
    "q_by",
    "p_maxT",
    "event_p",
    "event_q",
    "event_fdr",
]
_FIT_SIGNATURE_KEYS = [
    "method",
    "pathway_score",
    "condition_key",
    "donor_key",
    "control",
    "case",
    "pseudotime_key",
    "continuous_covariate_keys",
    "categorical_covariate_keys",
    "strata_keys",
    "design_encoding_json",
    "grid_edges",
    "grid_edges_sha256_float64_le",
    "source_grid_support_contract_sha256",
    "selected_bin_ids",
    "selected_support_contract_sha256",
    "reduced_design_sha256_float64_le",
    "support_blocks_sha256",
    "statistic",
    "tail",
    "calibration_scale",
    "calibration_method",
    "n_null_mappings_evaluated",
]


@dataclass
class T21Chr21PathwayDecomposition:
    """Immutable-by-contract plan for operational chr21 score sensitivity.

    The contained mappings can be passed directly to the existing weighted
    donor-pseudobulk scorer.  Callers should treat all fields as immutable after
    construction; :func:`reconstruct_t21_chr21_effect_components` verifies the
    stored stable hash before using the plan.
    """

    expanded_gene_sets: dict[str, dict[str, float]]
    membership: pd.DataFrame
    pathway_summary: pd.DataFrame
    metadata: dict[str, Any]

    @property
    def stable_hash(self) -> str:
        """Return the plan's canonical SHA-256 digest."""
        return str(self.metadata["stable_hash"])

    def to_tables(self) -> dict[str, pd.DataFrame]:
        """Return independent copies of the plan tables."""
        return {
            "membership": self.membership.copy(),
            "pathway_summary": self.pathway_summary.copy(),
        }


@dataclass
class T21Chr21EffectComponents:
    """Reconstructed operational score components from one shared fit.

    These components quantify score sensitivity to chromosome-21 membership.
    They are not a causal decomposition or evidence that the trans component is
    mediated by chromosome-21 dosage.
    """

    effect_curves: pd.DataFrame
    component_tests: pd.DataFrame
    null_statistics: pd.DataFrame
    pathway_summary: pd.DataFrame
    metadata: dict[str, Any]

    def to_tables(self) -> dict[str, pd.DataFrame]:
        """Return independent copies of all reconstructed tables."""
        return {
            "effect_curves": self.effect_curves.copy(),
            "component_tests": self.component_tests.copy(),
            "null_statistics": self.null_statistics.copy(),
            "pathway_summary": self.pathway_summary.copy(),
        }


def _strict_identifier(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{context} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{context} must be non-empty with no edge whitespace")
    if "\x00" in value:
        raise ValueError(f"{context} must not contain NUL characters")
    return value


def _strict_weight(value: Any, pathway: str, gene: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(
            f"Weight for gene '{gene}' in pathway '{pathway}' must be real numeric"
        )
    weight = float(value)
    if not np.isfinite(weight) or weight == 0.0:
        raise ValueError(
            f"Weight for gene '{gene}' in pathway '{pathway}' must be finite and nonzero"
        )
    return weight


def _canonical_gene_sets(gene_sets: Any) -> dict[str, dict[str, float]]:
    if not isinstance(gene_sets, Mapping):
        raise TypeError("gene_sets must be a mapping of pathway names to members")
    if not gene_sets:
        raise ValueError("gene_sets must not be empty")

    canonical: dict[str, dict[str, float]] = {}
    for raw_name, raw_members in gene_sets.items():
        name = _strict_identifier(raw_name, "Pathway name")
        if name.endswith((_TOTAL_SUFFIX, _TRANS_SUFFIX, _CONTRIBUTION_SUFFIX)):
            raise ValueError(
                f"Pathway name '{name}' uses a reserved decomposition suffix"
            )
        if name in canonical:
            raise ValueError(f"Duplicate pathway name '{name}'")

        parsed: dict[str, float] = {}
        if isinstance(raw_members, Mapping):
            items = raw_members.items()
            for raw_gene, raw_weight in items:
                gene = _strict_identifier(
                    raw_gene, f"Member of pathway '{name}'"
                )
                if gene in parsed:
                    raise ValueError(
                        f"Pathway '{name}' contains duplicate member '{gene}'"
                    )
                parsed[gene] = _strict_weight(raw_weight, name, gene)
        else:
            if isinstance(raw_members, np.ndarray):
                if raw_members.ndim != 1:
                    raise TypeError(
                        f"Pathway '{name}' members must be a one-dimensional sequence"
                    )
                raw_members = raw_members.tolist()
            if isinstance(raw_members, (str, bytes)) or not isinstance(
                raw_members, Sequence
            ):
                raise TypeError(
                    f"Pathway '{name}' members must be a gene_id sequence or weight mapping"
                )
            for raw_gene in raw_members:
                gene = _strict_identifier(
                    raw_gene, f"Member of pathway '{name}'"
                )
                if gene in parsed:
                    raise ValueError(
                        f"Pathway '{name}' contains duplicate member '{gene}'"
                    )
                parsed[gene] = 1.0

        if not parsed:
            raise ValueError(f"Pathway '{name}' must contain at least one member")
        canonical[name] = {gene: parsed[gene] for gene in sorted(parsed)}
    return {name: canonical[name] for name in sorted(canonical)}


def _annotation_lookup(
    gene_annotation: Any,
    *,
    gene_id_column: str,
    is_chr21_column: str,
) -> dict[str, bool]:
    gene_id_column = _strict_identifier(gene_id_column, "gene_id_column")
    is_chr21_column = _strict_identifier(is_chr21_column, "is_chr21_column")
    if gene_id_column == is_chr21_column:
        raise ValueError("gene_id_column and is_chr21_column must be distinct")

    if isinstance(gene_annotation, pd.DataFrame):
        missing = [
            column
            for column in (gene_id_column, is_chr21_column)
            if column not in gene_annotation.columns
        ]
        if missing:
            raise KeyError(f"gene_annotation is missing columns {missing}")
        genes = gene_annotation[gene_id_column].tolist()
        flags = gene_annotation[is_chr21_column].tolist()
    elif isinstance(gene_annotation, Mapping):
        genes = list(gene_annotation.keys())
        flags = list(gene_annotation.values())
    else:
        raise TypeError(
            "gene_annotation must be a DataFrame or gene_id-to-bool mapping"
        )

    lookup: dict[str, bool] = {}
    for raw_gene, raw_flag in zip(genes, flags):
        gene = _strict_identifier(raw_gene, "gene_annotation gene_id")
        if gene in lookup:
            raise ValueError(f"gene_annotation contains duplicate gene_id '{gene}'")
        if not isinstance(raw_flag, (bool, np.bool_)):
            raise TypeError(
                f"gene_annotation is_chr21 for gene '{gene}' must be an actual bool"
            )
        lookup[gene] = bool(raw_flag)
    return lookup


def _abs_weight_sum(weights: Mapping[str, float]) -> float:
    values = np.asarray(list(weights.values()), dtype=float)
    total = float(np.abs(values).sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("Internal non-positive absolute-weight denominator")
    return total


def _append_membership(
    rows: list[dict[str, Any]],
    *,
    base_pathway: str,
    expanded_pathway: str,
    component: str,
    score_weights: Mapping[str, float],
    original_weights: Mapping[str, float],
    annotation: Mapping[str, bool],
    A: float,
    B: float,
    C: float,
) -> None:
    denominator = _abs_weight_sum(score_weights)
    multiplier = C if component == "chr21_contribution" else 1.0
    for gene, weight in score_weights.items():
        rows.append(
            {
                "Pathway": expanded_pathway,
                "base_pathway": base_pathway,
                "component": component,
                "gene_id": gene,
                "weight": float(weight),
                "original_weight": float(original_weights[gene]),
                "is_chr21": bool(annotation[gene]),
                "pathway_size": int(len(score_weights)),
                "score_abs_weight_sum": denominator,
                "A": float(A),
                "B": float(B),
                "C": float(C),
                "effect_multiplier": float(multiplier),
            }
        )


def _json_scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            return None
        return number
    return str(value)


def _canonical_table_records(
    table: pd.DataFrame, *, sort_by: Sequence[str]
) -> list[dict[str, Any]]:
    columns = sorted(map(str, table.columns))
    frame = table.loc[:, columns].copy()
    frame = frame.sort_values(list(sort_by), kind="mergesort").reset_index(drop=True)
    return [
        {column: _json_scalar(row[column]) for column in columns}
        for _, row in frame.iterrows()
    ]


def _plan_hash(
    expanded_gene_sets: Mapping[str, Mapping[str, float]],
    membership: pd.DataFrame,
    pathway_summary: pd.DataFrame,
    *,
    min_trans_genes: int,
    gene_id_column: str,
    is_chr21_column: str,
) -> str:
    expanded = {
        name: {gene: float(weights[gene]) for gene in sorted(weights)}
        for name, weights in sorted(expanded_gene_sets.items())
    }
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "min_trans_genes": int(min_trans_genes),
        "gene_id_column": gene_id_column,
        "is_chr21_column": is_chr21_column,
        "required_scorer": _REQUIRED_SCORER,
        "expanded_gene_sets": expanded,
        "membership": _canonical_table_records(
            membership, sort_by=["Pathway", "gene_id"]
        ),
        "pathway_summary": _canonical_table_records(
            pathway_summary, sort_by=["Pathway"]
        ),
    }
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def build_t21_chr21_pathway_decomposition(
    gene_sets: Mapping[str, Sequence[str] | Mapping[str, float]],
    gene_annotation: pd.DataFrame | Mapping[str, bool],
    *,
    gene_id_column: str = "gene_id",
    is_chr21_column: str = "is_chr21",
    min_trans_genes: int = 1,
) -> T21Chr21PathwayDecomposition:
    """Build weighted-scorer inputs for an operational chr21 sensitivity check.

    For a pathway with weights ``w``, let ``A=sum(abs(w_all))`` and
    ``B=sum(abs(w_trans))``.  The contribution set receives raw coefficients
    ``w/A`` for chromosome-21 genes and ``w/A - w/B`` for trans genes.  If
    ``C`` is the absolute sum of those coefficients, multiplying the existing
    scorer's contribution output by ``C`` recovers ``total - trans``.

    Pathways without chromosome-21 genes produce only a ``::total`` scoring
    set.  Their trans component is later treated as an alias of total and their
    contribution as deterministic zero; no artificial zero-weight set is made.
    Pathways with chromosome-21 members fail closed if fewer than
    ``min_trans_genes`` trans genes remain.

    This construction is an operational pathway-score sensitivity analysis,
    not a causal decomposition or evidence of chromosome-21 mediation.
    """

    if isinstance(min_trans_genes, (bool, np.bool_)) or not isinstance(
        min_trans_genes, (int, np.integer)
    ):
        raise TypeError("min_trans_genes must be an integer")
    if int(min_trans_genes) < 1:
        raise ValueError("min_trans_genes must be at least 1")
    min_trans_genes = int(min_trans_genes)

    canonical = _canonical_gene_sets(gene_sets)
    annotation = _annotation_lookup(
        gene_annotation,
        gene_id_column=gene_id_column,
        is_chr21_column=is_chr21_column,
    )
    requested_genes = sorted(
        {gene for weights in canonical.values() for gene in weights}
    )
    missing = [gene for gene in requested_genes if gene not in annotation]
    if missing:
        preview = ", ".join(missing[:10])
        raise KeyError(
            "gene_annotation is missing pathway members: "
            f"{preview}{' ...' if len(missing) > 10 else ''}"
        )

    expanded: dict[str, dict[str, float]] = {}
    membership_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for pathway, original_weights in canonical.items():
        total_weights = dict(original_weights)
        chr21_weights = {
            gene: weight
            for gene, weight in total_weights.items()
            if annotation[gene]
        }
        trans_weights = {
            gene: weight
            for gene, weight in total_weights.items()
            if not annotation[gene]
        }
        A = _abs_weight_sum(total_weights)
        B = _abs_weight_sum(trans_weights) if trans_weights else 0.0
        total_name = f"{pathway}{_TOTAL_SUFFIX}"
        if total_name in expanded:
            raise ValueError(f"Expanded pathway name collision at '{total_name}'")
        expanded[total_name] = total_weights

        if not chr21_weights:
            C = 0.0
            _append_membership(
                membership_rows,
                base_pathway=pathway,
                expanded_pathway=total_name,
                component="total",
                score_weights=total_weights,
                original_weights=original_weights,
                annotation=annotation,
                A=A,
                B=A,
                C=C,
            )
            summary_rows.append(
                {
                    "Pathway": pathway,
                    "has_chr21": False,
                    "n_total_genes": len(total_weights),
                    "n_chr21_genes": 0,
                    "n_trans_genes": len(trans_weights),
                    "A": A,
                    "B": A,
                    "C": C,
                    "total_expanded_pathway": total_name,
                    "trans_expanded_pathway": total_name,
                    "chr21_contribution_expanded_pathway": "",
                    "trans_status": "alias_total",
                    "chr21_contribution_status": "deterministic_zero",
                    "total_formal_inference": True,
                    "trans_formal_inference": False,
                    "chr21_contribution_formal_inference": False,
                    "closure_contract": "total_minus_trans_equals_zero",
                }
            )
            continue

        if len(trans_weights) < min_trans_genes:
            raise ValueError(
                f"Pathway '{pathway}' has {len(trans_weights)} trans genes after "
                f"removing chromosome-21 members; min_trans_genes={min_trans_genes}"
            )

        trans_name = f"{pathway}{_TRANS_SUFFIX}"
        contribution_name = f"{pathway}{_CONTRIBUTION_SUFFIX}"
        if trans_name in expanded or contribution_name in expanded:
            raise ValueError(f"Expanded pathway name collision for '{pathway}'")

        contribution_weights: dict[str, float] = {}
        for gene, weight in total_weights.items():
            if annotation[gene]:
                coefficient = weight / A
            else:
                coefficient = weight / A - weight / B
            if not np.isfinite(coefficient) or coefficient == 0.0:
                raise RuntimeError(
                    f"Internal invalid contribution coefficient for '{pathway}/{gene}'"
                )
            contribution_weights[gene] = float(coefficient)
        C = _abs_weight_sum(contribution_weights)
        expanded[trans_name] = trans_weights
        expanded[contribution_name] = contribution_weights

        for component, expanded_name, score_weights in (
            ("total", total_name, total_weights),
            ("trans", trans_name, trans_weights),
            (
                "chr21_contribution",
                contribution_name,
                contribution_weights,
            ),
        ):
            _append_membership(
                membership_rows,
                base_pathway=pathway,
                expanded_pathway=expanded_name,
                component=component,
                score_weights=score_weights,
                original_weights=original_weights,
                annotation=annotation,
                A=A,
                B=B,
                C=C,
            )
        summary_rows.append(
            {
                "Pathway": pathway,
                "has_chr21": True,
                "n_total_genes": len(total_weights),
                "n_chr21_genes": len(chr21_weights),
                "n_trans_genes": len(trans_weights),
                "A": A,
                "B": B,
                "C": C,
                "total_expanded_pathway": total_name,
                "trans_expanded_pathway": trans_name,
                "chr21_contribution_expanded_pathway": contribution_name,
                "trans_status": "fitted",
                "chr21_contribution_status": "fitted_then_multiplied_by_C",
                "total_formal_inference": True,
                "trans_formal_inference": True,
                "chr21_contribution_formal_inference": True,
                "closure_contract": (
                    "C_times_normalized_contribution_equals_total_minus_trans"
                ),
            }
        )

    expanded = {name: expanded[name] for name in sorted(expanded)}
    membership = pd.DataFrame(membership_rows, columns=_MEMBERSHIP_COLUMNS).sort_values(
        ["base_pathway", "component", "Pathway", "gene_id"], kind="mergesort"
    ).reset_index(drop=True)
    pathway_summary = pd.DataFrame(
        summary_rows, columns=_SUMMARY_COLUMNS
    ).sort_values("Pathway", kind="mergesort").reset_index(drop=True)

    stable_hash = _plan_hash(
        expanded,
        membership,
        pathway_summary,
        min_trans_genes=min_trans_genes,
        gene_id_column=gene_id_column,
        is_chr21_column=is_chr21_column,
    )
    metadata = {
        "method": "t21_chr21_pathway_score_sensitivity_decomposition",
        "schema_version": _SCHEMA_VERSION,
        "stable_hash": stable_hash,
        "stable_hash_algorithm": "sha256_canonical_json",
        "gene_id_column": gene_id_column,
        "is_chr21_column": is_chr21_column,
        "min_trans_genes": min_trans_genes,
        "n_base_pathways": int(len(pathway_summary)),
        "n_expanded_gene_sets": int(len(expanded)),
        "required_scorer": _REQUIRED_SCORER,
        "score_denominator": "sum_absolute_weights",
        "contribution_rescaling": "multiply_fitted_contribution_by_C",
        "interpretation_scope": (
            "operational_pathway_score_sensitivity_not_causal_mediation"
        ),
        "no_chr21_policy": (
            "fit_total_only_trans_alias_total_contribution_deterministic_zero"
        ),
    }
    membership.attrs["t21_chr21_pathway_decomposition"] = metadata.copy()
    pathway_summary.attrs["t21_chr21_pathway_decomposition"] = metadata.copy()
    return T21Chr21PathwayDecomposition(
        expanded_gene_sets=expanded,
        membership=membership,
        pathway_summary=pathway_summary,
        metadata=metadata,
    )


def _expected_scorer_membership(
    expanded_gene_sets: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pathway, weights in sorted(expanded_gene_sets.items()):
        for gene, weight in sorted(weights.items()):
            rows.append(
                {
                    "Pathway": pathway,
                    "gene": gene,
                    "weight": float(weight),
                    "pathway_size": int(len(weights)),
                }
            )
    return pd.DataFrame(
        rows, columns=["Pathway", "gene", "weight", "pathway_size"]
    )


def _assert_exact_membership(
    actual: pd.DataFrame,
    plan: T21Chr21PathwayDecomposition,
) -> None:
    expected = _expected_scorer_membership(plan.expanded_gene_sets)
    required = list(expected.columns)
    missing = [column for column in required if column not in actual.columns]
    if missing:
        raise ValueError(f"fitted.pathway_membership is missing columns {missing}")
    if actual.empty:
        raise ValueError("fitted.pathway_membership must not be empty")
    if actual[["Pathway", "gene"]].isna().any().any():
        raise ValueError("fitted.pathway_membership contains missing identifiers")
    if actual.duplicated(["Pathway", "gene"]).any():
        raise ValueError("fitted.pathway_membership contains duplicate members")
    if not all(isinstance(value, str) for value in actual["Pathway"]):
        raise TypeError("fitted pathway names must be strings")
    if not all(isinstance(value, str) for value in actual["gene"]):
        raise TypeError("fitted pathway gene identifiers must be strings")

    observed = actual.loc[:, required].sort_values(
        ["Pathway", "gene"], kind="mergesort"
    ).reset_index(drop=True)
    expected = expected.sort_values(
        ["Pathway", "gene"], kind="mergesort"
    ).reset_index(drop=True)
    if not observed[["Pathway", "gene"]].equals(
        expected[["Pathway", "gene"]]
    ):
        raise ValueError(
            "fitted.pathway_membership identifiers do not exactly match the plan"
        )
    observed_sizes = pd.to_numeric(observed["pathway_size"], errors="coerce")
    if observed_sizes.isna().any() or not np.array_equal(
        observed_sizes.to_numpy(dtype=int),
        expected["pathway_size"].to_numpy(dtype=int),
    ):
        raise ValueError(
            "fitted.pathway_membership pathway_size does not exactly match the plan"
        )
    observed_weights = pd.to_numeric(observed["weight"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(observed_weights).all() or not np.array_equal(
        observed_weights, expected["weight"].to_numpy(dtype=float)
    ):
        raise ValueError(
            "fitted.pathway_membership weights do not exactly match the plan"
        )

    plan_membership = plan.membership.rename(columns={"gene_id": "gene"})
    planned = plan_membership.loc[:, required].sort_values(
        ["Pathway", "gene"], kind="mergesort"
    ).reset_index(drop=True)
    if not planned[["Pathway", "gene"]].equals(
        expected[["Pathway", "gene"]]
    ) or not np.array_equal(
        planned["weight"].to_numpy(dtype=float),
        expected["weight"].to_numpy(dtype=float),
    ):
        raise ValueError("plan membership is inconsistent with expanded_gene_sets")


def _metadata_value_equal(left: Any, right: Any) -> bool:
    return json.dumps(
        left, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ) == json.dumps(
        right, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )


def _fit_signature(fitted: Any) -> str:
    metadata = getattr(fitted, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise TypeError("fitted.metadata must be a mapping")
    missing = [key for key in _FIT_SIGNATURE_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"fitted.metadata is missing fit invariants {missing}")
    if metadata["pathway_score"] != _REQUIRED_SCORER:
        raise ValueError(
            "fitted result does not use the required absolute-weight-normalized scorer"
        )
    payload = {key: metadata[key] for key in _FIT_SIGNATURE_KEYS}
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def _validate_table_fit_attrs(fitted: Any) -> None:
    metadata = fitted.metadata
    for name in ("effect_curves", "pathway_tests"):
        table = getattr(fitted, name)
        recorded = table.attrs.get("covariate_adjusted_donor_pseudobulk")
        if recorded is None:
            continue
        for key in _FIT_SIGNATURE_KEYS:
            if key not in recorded or not _metadata_value_equal(
                recorded[key], metadata[key]
            ):
                raise ValueError(
                    f"{name} does not originate from the same covariate fit metadata"
                )


def _validate_effect_grid(
    effect_curves: pd.DataFrame,
    expected_pathways: set[str],
    metadata: Mapping[str, Any],
) -> pd.DataFrame:
    required = [
        "Pathway",
        *_GRID_COLUMNS,
        *_SCALE_EFFECT_CURVE_COLUMNS,
        *_CURVE_INFERENCE_COLUMNS,
    ]
    missing = [column for column in required if column not in effect_curves.columns]
    if missing:
        raise ValueError(f"fitted.effect_curves is missing columns {missing}")
    if effect_curves.empty:
        raise ValueError("fitted.effect_curves must not be empty")
    if set(effect_curves["Pathway"]) != expected_pathways:
        raise ValueError("fitted.effect_curves pathway set does not match the plan")
    if effect_curves.duplicated(["Pathway", "bin_id"]).any():
        raise ValueError("fitted.effect_curves contains duplicate pathway-bin rows")

    selected_bins = np.asarray(metadata["selected_bin_ids"])
    if selected_bins.ndim != 1 or len(selected_bins) == 0:
        raise ValueError("fitted.metadata selected_bin_ids must be non-empty")
    numeric_selected = pd.to_numeric(
        pd.Series(selected_bins), errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(numeric_selected).all() or not np.equal(
        numeric_selected, np.floor(numeric_selected)
    ).all():
        raise ValueError("fitted.metadata selected_bin_ids must be integer-valued")
    selected_bins = numeric_selected.astype(int)

    reference: pd.DataFrame | None = None
    for pathway in sorted(expected_pathways):
        block = effect_curves[effect_curves["Pathway"].eq(pathway)].sort_values(
            "bin_id", kind="mergesort"
        )
        bins = pd.to_numeric(block["bin_id"], errors="coerce").to_numpy(dtype=float)
        if not np.array_equal(bins, selected_bins.astype(float)):
            raise ValueError(
                f"fitted.effect_curves has an incomplete effect grid for '{pathway}'"
            )
        for column in [*_GRID_COLUMNS[1:], *_SCALE_EFFECT_CURVE_COLUMNS]:
            values = pd.to_numeric(block[column], errors="coerce").to_numpy(
                dtype=float
            )
            if not np.isfinite(values).all():
                raise ValueError(
                    f"fitted.effect_curves column '{column}' is non-finite for '{pathway}'"
                )
        if reference is None:
            reference = block.loc[:, _GRID_COLUMNS].reset_index(drop=True)
        else:
            current = block.loc[:, _GRID_COLUMNS].reset_index(drop=True)
            for column in _GRID_COLUMNS:
                if not np.array_equal(
                    pd.to_numeric(current[column], errors="coerce").to_numpy(
                        dtype=float
                    ),
                    pd.to_numeric(reference[column], errors="coerce").to_numpy(
                        dtype=float
                    ),
                ):
                    raise ValueError(
                        "Expanded pathways do not share one exact covariate-fit grid"
                    )
    assert reference is not None
    return reference


def _validate_pathway_tests(
    pathway_tests: pd.DataFrame,
    expected_pathways: set[str],
    metadata: Mapping[str, Any],
) -> pd.DataFrame:
    required = [
        "Pathway",
        "primary_statistic",
        "tail",
        "calibration_scale",
        "observed_effect_statistic",
        *_TEST_P_COLUMNS,
    ]
    missing = [column for column in required if column not in pathway_tests.columns]
    if missing:
        raise ValueError(f"fitted.pathway_tests is missing columns {missing}")
    if len(pathway_tests) != len(expected_pathways):
        raise ValueError("fitted.pathway_tests must have exactly one row per expanded set")
    if pathway_tests["Pathway"].duplicated().any() or set(
        pathway_tests["Pathway"]
    ) != expected_pathways:
        raise ValueError("fitted.pathway_tests pathway rows do not match the plan")
    for column, expected in (
        ("primary_statistic", metadata["statistic"]),
        ("tail", metadata["tail"]),
        ("calibration_scale", metadata["calibration_scale"]),
    ):
        if not pathway_tests[column].map(str).eq(str(expected)).all():
            raise ValueError(
                f"fitted.pathway_tests '{column}' does not come from one shared fit"
            )
    for column in _TEST_P_COLUMNS:
        values = pd.to_numeric(pathway_tests[column], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"fitted.pathway_tests '{column}' must contain valid p/q values")
    return pathway_tests.set_index("Pathway", drop=False)


def _validate_null_statistics(
    fitted: Any,
    expected_pathways: set[str],
) -> pd.DataFrame:
    null = fitted.null_statistics
    required = [
        "perm_id",
        "mapping_hash",
        "Pathway",
        "effect_statistic",
        "raw_calibration_statistic",
        "calibration_statistic",
        "max_pathway_calibration_statistic",
    ]
    missing = [column for column in required if column not in null.columns]
    if missing:
        raise ValueError(f"fitted.null_statistics is missing columns {missing}")
    if null.empty:
        raise ValueError(
            "fitted.null_statistics is required; fit with return_null_statistics=True"
        )
    if set(null["Pathway"]) != expected_pathways:
        raise ValueError("fitted.null_statistics pathway set does not match the plan")
    if null.duplicated(["perm_id", "Pathway"]).any():
        raise ValueError("fitted.null_statistics contains duplicate permutation rows")

    numeric_perm = pd.to_numeric(null["perm_id"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(numeric_perm).all() or not np.equal(
        numeric_perm, np.floor(numeric_perm)
    ).all():
        raise ValueError("fitted.null_statistics perm_id must be integer-valued")
    work = null.copy()
    work["perm_id"] = numeric_perm.astype(int)
    perm_ids = np.sort(work["perm_id"].unique())
    if not np.array_equal(perm_ids, np.arange(len(perm_ids), dtype=int)):
        raise ValueError("fitted.null_statistics perm_id values must be contiguous from zero")

    expected_count = len(expected_pathways)
    if not work.groupby("perm_id").size().eq(expected_count).all():
        raise ValueError("fitted.null_statistics is not rectangular over pathways")
    for pathway, block in work.groupby("Pathway", sort=True):
        if not np.array_equal(
            np.sort(block["perm_id"].to_numpy(dtype=int)), perm_ids
        ):
            raise ValueError(
                f"fitted.null_statistics mapping stream is incomplete for '{pathway}'"
            )
    hashes = work.groupby("perm_id")["mapping_hash"].agg(
        lambda values: tuple(pd.unique(values))
    )
    if not hashes.map(len).eq(1).all() or any(
        not isinstance(values[0], str) or not values[0] for values in hashes
    ):
        raise ValueError(
            "Expanded pathways do not share the same permutation mapping stream"
        )
    for column in required[3:]:
        values = pd.to_numeric(work[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"fitted.null_statistics '{column}' must be finite")

    recorded_n = int(fitted.metadata["n_null_mappings_evaluated"])
    if recorded_n != len(perm_ids):
        raise ValueError(
            "fitted.metadata n_null_mappings_evaluated disagrees with null_statistics"
        )
    summary = getattr(fitted, "permutation_summary", None)
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        if "n_null_mappings_evaluated" not in summary.columns:
            raise ValueError(
                "fitted.permutation_summary lacks n_null_mappings_evaluated"
            )
        summary_n = pd.to_numeric(
            summary["n_null_mappings_evaluated"], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(summary_n).all() or not np.equal(summary_n, recorded_n).all():
            raise ValueError(
                "fitted.permutation_summary disagrees with the shared mapping stream"
            )
    return work.sort_values(["perm_id", "Pathway"], kind="mergesort").reset_index(
        drop=True
    )


def _scaled_effect_block(
    source: pd.DataFrame,
    *,
    base_pathway: str,
    component: str,
    expanded_pathway: str,
    C: float,
    scale: float,
    formal_inference: bool,
    deterministic: bool,
    inference_status: str,
) -> pd.DataFrame:
    block = source.copy()
    block["expanded_pathway"] = expanded_pathway
    block["Pathway"] = base_pathway
    block["component"] = component
    block["C"] = float(C)
    block["effect_multiplier"] = float(scale)
    block["formal_inference"] = bool(formal_inference)
    block["deterministic"] = bool(deterministic)
    block["inference_status"] = inference_status
    if scale != 1.0:
        for column in _SCALE_EFFECT_CURVE_COLUMNS:
            block[column] = pd.to_numeric(block[column], errors="raise") * scale
    return block


def _deterministic_zero_effect(
    total: pd.DataFrame,
    *,
    base_pathway: str,
) -> pd.DataFrame:
    block = _scaled_effect_block(
        total,
        base_pathway=base_pathway,
        component="chr21_contribution",
        expanded_pathway="",
        C=0.0,
        scale=0.0,
        formal_inference=False,
        deterministic=True,
        inference_status="deterministic_zero_no_formal_inference",
    )
    for column in _LINEAR_EFFECT_COLUMNS:
        block[column] = 0.0
    for column in [
        "condition_standard_error",
        "residual_sd",
        *_CURVE_INFERENCE_COLUMNS,
    ]:
        block[column] = np.nan
    return block


def _validate_curve_closure(
    effect_curves: pd.DataFrame,
    *,
    rtol: float,
    atol: float,
) -> None:
    for pathway, group in effect_curves.groupby("Pathway", sort=True):
        blocks = {
            component: block.sort_values("bin_id", kind="mergesort")
            for component, block in group.groupby("component", sort=False)
        }
        if set(blocks) != set(_COMPONENTS):
            raise RuntimeError(
                f"Reconstructed pathway '{pathway}' lacks a required component"
            )
        for column in _LINEAR_EFFECT_COLUMNS:
            total = blocks["total"][column].to_numpy(dtype=float)
            trans = blocks["trans"][column].to_numpy(dtype=float)
            contribution = blocks["chr21_contribution"][column].to_numpy(
                dtype=float
            )
            if not np.allclose(
                total - trans,
                contribution,
                rtol=rtol,
                atol=atol,
                equal_nan=False,
            ):
                maximum = float(np.max(np.abs(total - trans - contribution)))
                raise ValueError(
                    f"Effect closure failed for pathway '{pathway}', column "
                    f"'{column}' (maximum error {maximum:.3g})"
                )


def _scaled_test_row(
    source: pd.Series,
    *,
    base_pathway: str,
    component: str,
    expanded_pathway: str,
    C: float,
    scale: float,
    calibration_scale: str,
    formal_inference: bool,
    deterministic: bool,
    inference_status: str,
    p_value_source: str,
) -> dict[str, Any]:
    row = source.to_dict()
    row.update(
        {
            "Pathway": base_pathway,
            "component": component,
            "expanded_pathway": expanded_pathway,
            "C": float(C),
            "effect_multiplier": float(scale),
            "formal_inference": bool(formal_inference),
            "deterministic": bool(deterministic),
            "inference_status": inference_status,
            "p_value_source": p_value_source,
        }
    )
    if scale != 1.0:
        for column in _TEST_EFFECT_COLUMNS:
            if column in row:
                row[column] = float(row[column]) * scale
        if calibration_scale == "effect":
            for column in (
                "observed_calibration_statistic",
                "raw_calibration_statistic",
                "calibration_statistic",
            ):
                if column in row:
                    row[column] = float(row[column]) * scale
    return row


def _deterministic_zero_test(
    total: pd.Series,
    *,
    base_pathway: str,
    calibration_scale: str,
) -> dict[str, Any]:
    row = _scaled_test_row(
        total,
        base_pathway=base_pathway,
        component="chr21_contribution",
        expanded_pathway="",
        C=0.0,
        scale=0.0,
        calibration_scale=calibration_scale,
        formal_inference=False,
        deterministic=True,
        inference_status="deterministic_zero_no_formal_inference",
        p_value_source="not_applicable",
    )
    for column in _TEST_EFFECT_COLUMNS:
        if column in row:
            row[column] = 0.0
    for column in [
        "observed_calibration_statistic",
        "peak_bin",
        "peak_time",
        *_TEST_P_COLUMNS,
    ]:
        if column in row:
            row[column] = np.nan
    row["pathway_size"] = 0
    return row


def _add_wide_component_p_values(component_tests: pd.DataFrame) -> pd.DataFrame:
    result = component_tests.copy()
    for p_column in _TEST_P_COLUMNS:
        if p_column not in result.columns:
            continue
        lookup = result.pivot(
            index="Pathway", columns="component", values=p_column
        )
        for component in _COMPONENTS:
            output = f"{component}_{p_column}"
            values = lookup[component] if component in lookup.columns else pd.Series(
                np.nan, index=lookup.index
            )
            result[output] = result["Pathway"].map(values)
    if "total_p_raw" in result:
        result["p_total"] = result["total_p_raw"]
        result["p_trans"] = result["trans_p_raw"]
        result["p_chr21_contribution"] = result[
            "chr21_contribution_p_raw"
        ]
    return result


def _scaled_null_block(
    source: pd.DataFrame,
    *,
    base_pathway: str,
    component: str,
    expanded_pathway: str,
    C: float,
    scale: float,
    calibration_scale: str,
    formal_inference: bool,
    deterministic: bool,
    inference_status: str,
) -> pd.DataFrame:
    block = source.copy()
    block["expanded_pathway"] = expanded_pathway
    block["Pathway"] = base_pathway
    block["component"] = component
    block["C"] = float(C)
    block["effect_multiplier"] = float(scale)
    block["formal_inference"] = bool(formal_inference)
    block["deterministic"] = bool(deterministic)
    block["inference_status"] = inference_status
    if scale != 1.0:
        block["effect_statistic"] = pd.to_numeric(
            block["effect_statistic"], errors="raise"
        ) * scale
        if calibration_scale == "effect":
            for column in (
                "raw_calibration_statistic",
                "calibration_statistic",
            ):
                block[column] = pd.to_numeric(block[column], errors="raise") * scale
    return block


def _deterministic_zero_null(
    total: pd.DataFrame,
    *,
    base_pathway: str,
    calibration_scale: str,
) -> pd.DataFrame:
    block = _scaled_null_block(
        total,
        base_pathway=base_pathway,
        component="chr21_contribution",
        expanded_pathway="",
        C=0.0,
        scale=0.0,
        calibration_scale=calibration_scale,
        formal_inference=False,
        deterministic=True,
        inference_status="deterministic_zero_no_formal_inference",
    )
    block["effect_statistic"] = 0.0
    for column in (
        "raw_calibration_statistic",
        "calibration_statistic",
        "max_pathway_calibration_statistic",
    ):
        block[column] = np.nan
    return block


def _result_hash(
    effect_curves: pd.DataFrame,
    component_tests: pd.DataFrame,
    null_statistics: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> str:
    digest = sha256()
    for name, table, sort_by in (
        ("effect_curves", effect_curves, ["Pathway", "component", "bin_id"]),
        ("component_tests", component_tests, ["Pathway", "component"]),
        (
            "null_statistics",
            null_statistics,
            ["perm_id", "Pathway", "component"],
        ),
    ):
        canonical = table.sort_values(sort_by, kind="mergesort")
        canonical = canonical.reindex(sorted(canonical.columns), axis=1)
        digest.update(name.encode("utf-8"))
        digest.update(
            canonical.to_csv(
                index=False,
                float_format="%.17g",
                na_rep="<NA>",
                lineterminator="\n",
            ).encode("utf-8")
        )
    digest.update(
        json.dumps(
            dict(metadata),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def reconstruct_t21_chr21_effect_components(
    fitted: Any,
    plan: T21Chr21PathwayDecomposition,
    *,
    closure_rtol: float = 1e-10,
    closure_atol: float = 1e-12,
) -> T21Chr21EffectComponents:
    """Reconstruct total, trans, and chr21-contribution effects from one fit.

    The function fails closed unless the fitted pathway membership exactly
    matches ``plan``, every expanded set shares a complete effect grid, and the
    null-statistic table is rectangular over one common permutation mapping
    stream.  Contribution curves and homogeneous effect statistics are
    multiplied by the plan's ``C`` before closure is checked.

    For pathways with no chromosome-21 members, trans is reported as a total
    alias and contribution as deterministic zero.  The deterministic component
    receives no p-value and no claim of formal inference.

    This is an operational score-sensitivity reconstruction, not a causal
    mediation decomposition.
    """

    if not isinstance(plan, T21Chr21PathwayDecomposition):
        raise TypeError("plan must be a T21Chr21PathwayDecomposition")
    for name in (
        "pathway_tests",
        "effect_curves",
        "pathway_membership",
        "null_statistics",
        "metadata",
    ):
        if not hasattr(fitted, name):
            raise TypeError(f"fitted is missing required attribute '{name}'")
    for name, value in (
        ("closure_rtol", closure_rtol),
        ("closure_atol", closure_atol),
    ):
        if not isinstance(value, Real) or not np.isfinite(float(value)) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    closure_rtol = float(closure_rtol)
    closure_atol = float(closure_atol)

    expected_plan_hash = _plan_hash(
        plan.expanded_gene_sets,
        plan.membership,
        plan.pathway_summary,
        min_trans_genes=int(plan.metadata["min_trans_genes"]),
        gene_id_column=str(plan.metadata["gene_id_column"]),
        is_chr21_column=str(plan.metadata["is_chr21_column"]),
    )
    if plan.metadata.get("stable_hash") != expected_plan_hash:
        raise ValueError("plan stable hash does not match its current contents")

    fit_signature = _fit_signature(fitted)
    _validate_table_fit_attrs(fitted)
    _assert_exact_membership(fitted.pathway_membership, plan)
    expected_pathways = set(plan.expanded_gene_sets)
    _validate_effect_grid(fitted.effect_curves, expected_pathways, fitted.metadata)
    tests_by_pathway = _validate_pathway_tests(
        fitted.pathway_tests, expected_pathways, fitted.metadata
    )
    source_null = _validate_null_statistics(fitted, expected_pathways)
    calibration_scale = str(fitted.metadata["calibration_scale"])

    effect_blocks: list[pd.DataFrame] = []
    test_rows: list[dict[str, Any]] = []
    null_blocks: list[pd.DataFrame] = []
    source_effect = {
        pathway: block.sort_values("bin_id", kind="mergesort").reset_index(drop=True)
        for pathway, block in fitted.effect_curves.groupby("Pathway", sort=False)
    }
    source_null_blocks = {
        pathway: block.sort_values("perm_id", kind="mergesort").reset_index(drop=True)
        for pathway, block in source_null.groupby("Pathway", sort=False)
    }

    for summary in plan.pathway_summary.itertuples(index=False):
        base = str(summary.Pathway)
        C = float(summary.C)
        total_name = str(summary.total_expanded_pathway)
        total_effect = source_effect[total_name]
        total_null = source_null_blocks[total_name]
        effect_blocks.append(
            _scaled_effect_block(
                total_effect,
                base_pathway=base,
                component="total",
                expanded_pathway=total_name,
                C=C,
                scale=1.0,
                formal_inference=True,
                deterministic=False,
                inference_status="formal_joint_fit",
            )
        )
        test_rows.append(
            _scaled_test_row(
                tests_by_pathway.loc[total_name],
                base_pathway=base,
                component="total",
                expanded_pathway=total_name,
                C=C,
                scale=1.0,
                calibration_scale=calibration_scale,
                formal_inference=True,
                deterministic=False,
                inference_status="formal_joint_fit",
                p_value_source="fitted_total_set",
            )
        )
        null_blocks.append(
            _scaled_null_block(
                total_null,
                base_pathway=base,
                component="total",
                expanded_pathway=total_name,
                C=C,
                scale=1.0,
                calibration_scale=calibration_scale,
                formal_inference=True,
                deterministic=False,
                inference_status="formal_joint_fit",
            )
        )

        if not bool(summary.has_chr21):
            effect_blocks.append(
                _scaled_effect_block(
                    total_effect,
                    base_pathway=base,
                    component="trans",
                    expanded_pathway=total_name,
                    C=0.0,
                    scale=1.0,
                    formal_inference=False,
                    deterministic=True,
                    inference_status="alias_total_no_additional_inference",
                )
            )
            effect_blocks.append(
                _deterministic_zero_effect(total_effect, base_pathway=base)
            )
            test_rows.append(
                _scaled_test_row(
                    tests_by_pathway.loc[total_name],
                    base_pathway=base,
                    component="trans",
                    expanded_pathway=total_name,
                    C=0.0,
                    scale=1.0,
                    calibration_scale=calibration_scale,
                    formal_inference=False,
                    deterministic=True,
                    inference_status="alias_total_no_additional_inference",
                    p_value_source="total_alias",
                )
            )
            test_rows.append(
                _deterministic_zero_test(
                    tests_by_pathway.loc[total_name],
                    base_pathway=base,
                    calibration_scale=calibration_scale,
                )
            )
            null_blocks.append(
                _scaled_null_block(
                    total_null,
                    base_pathway=base,
                    component="trans",
                    expanded_pathway=total_name,
                    C=0.0,
                    scale=1.0,
                    calibration_scale=calibration_scale,
                    formal_inference=False,
                    deterministic=True,
                    inference_status="alias_total_no_additional_inference",
                )
            )
            null_blocks.append(
                _deterministic_zero_null(
                    total_null,
                    base_pathway=base,
                    calibration_scale=calibration_scale,
                )
            )
            continue

        trans_name = str(summary.trans_expanded_pathway)
        contribution_name = str(summary.chr21_contribution_expanded_pathway)
        trans_effect = source_effect[trans_name]
        contribution_effect = source_effect[contribution_name]
        trans_null = source_null_blocks[trans_name]
        contribution_null = source_null_blocks[contribution_name]
        effect_blocks.extend(
            [
                _scaled_effect_block(
                    trans_effect,
                    base_pathway=base,
                    component="trans",
                    expanded_pathway=trans_name,
                    C=C,
                    scale=1.0,
                    formal_inference=True,
                    deterministic=False,
                    inference_status="formal_joint_fit",
                ),
                _scaled_effect_block(
                    contribution_effect,
                    base_pathway=base,
                    component="chr21_contribution",
                    expanded_pathway=contribution_name,
                    C=C,
                    scale=C,
                    formal_inference=True,
                    deterministic=False,
                    inference_status="formal_joint_fit_scaled_by_C",
                ),
            ]
        )
        test_rows.extend(
            [
                _scaled_test_row(
                    tests_by_pathway.loc[trans_name],
                    base_pathway=base,
                    component="trans",
                    expanded_pathway=trans_name,
                    C=C,
                    scale=1.0,
                    calibration_scale=calibration_scale,
                    formal_inference=True,
                    deterministic=False,
                    inference_status="formal_joint_fit",
                    p_value_source="fitted_trans_set",
                ),
                _scaled_test_row(
                    tests_by_pathway.loc[contribution_name],
                    base_pathway=base,
                    component="chr21_contribution",
                    expanded_pathway=contribution_name,
                    C=C,
                    scale=C,
                    calibration_scale=calibration_scale,
                    formal_inference=True,
                    deterministic=False,
                    inference_status="formal_joint_fit_scaled_by_C",
                    p_value_source="fitted_contribution_set",
                ),
            ]
        )
        null_blocks.extend(
            [
                _scaled_null_block(
                    trans_null,
                    base_pathway=base,
                    component="trans",
                    expanded_pathway=trans_name,
                    C=C,
                    scale=1.0,
                    calibration_scale=calibration_scale,
                    formal_inference=True,
                    deterministic=False,
                    inference_status="formal_joint_fit",
                ),
                _scaled_null_block(
                    contribution_null,
                    base_pathway=base,
                    component="chr21_contribution",
                    expanded_pathway=contribution_name,
                    C=C,
                    scale=C,
                    calibration_scale=calibration_scale,
                    formal_inference=True,
                    deterministic=False,
                    inference_status="formal_joint_fit_scaled_by_C",
                ),
            ]
        )

    effect_curves = pd.concat(effect_blocks, ignore_index=True).sort_values(
        ["Pathway", "component", "bin_id"], kind="mergesort"
    ).reset_index(drop=True)
    _validate_curve_closure(
        effect_curves, rtol=closure_rtol, atol=closure_atol
    )
    component_tests = _add_wide_component_p_values(
        pd.DataFrame(test_rows).sort_values(
            ["Pathway", "component"], kind="mergesort"
        ).reset_index(drop=True)
    )
    null_statistics = pd.concat(null_blocks, ignore_index=True).sort_values(
        ["perm_id", "Pathway", "component"], kind="mergesort"
    ).reset_index(drop=True)
    if not null_statistics.groupby("perm_id")["mapping_hash"].nunique().eq(1).all():
        raise RuntimeError("Reconstructed components lost the shared mapping stream")

    metadata = {
        "method": "reconstructed_t21_chr21_pathway_effect_components",
        "schema_version": _SCHEMA_VERSION,
        "source_plan_hash": plan.stable_hash,
        "source_fit_signature": fit_signature,
        "n_base_pathways": int(len(plan.pathway_summary)),
        "n_effect_rows": int(len(effect_curves)),
        "n_null_mappings": int(source_null["perm_id"].nunique()),
        "null_mapping_stream_validated_rectangular": True,
        "pathway_membership_validated_exact": True,
        "effect_grid_validated_complete_and_shared": True,
        "closure_validated": True,
        "closure_rtol": closure_rtol,
        "closure_atol": closure_atol,
        "contribution_rescaling": "multiply_fitted_contribution_by_C",
        "no_chr21_policy": (
            "trans_alias_total_and_contribution_deterministic_zero_without_p_value"
        ),
        "interpretation_scope": (
            "operational_pathway_score_sensitivity_not_causal_mediation"
        ),
    }
    metadata["stable_hash"] = _result_hash(
        effect_curves, component_tests, null_statistics, metadata
    )
    metadata["stable_hash_algorithm"] = "sha256_canonical_tables"
    for table in (effect_curves, component_tests, null_statistics):
        table.attrs["t21_chr21_effect_components"] = metadata.copy()
    summary = plan.pathway_summary.copy()
    summary.attrs["t21_chr21_effect_components"] = metadata.copy()
    return T21Chr21EffectComponents(
        effect_curves=effect_curves,
        component_tests=component_tests,
        null_statistics=null_statistics,
        pathway_summary=summary,
        metadata=metadata,
    )


__all__ = [
    "T21Chr21EffectComponents",
    "T21Chr21PathwayDecomposition",
    "build_t21_chr21_pathway_decomposition",
    "reconstruct_t21_chr21_effect_components",
]
