from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .trajectory_covariate_pseudobulk import (
    _encode_reduced_design,
    run_covariate_adjusted_donor_pseudobulk,
)


CANONICAL_T21_DONOR_DESIGN_VERSION = "1.0.0"
_UNRESOLVED_BATCH_VALUES = frozenset(
    {
        "",
        "na",
        "n/a",
        "nan",
        "none",
        "not_available",
        "not_reported",
        "not_resolved",
        "omitted_not_identifiable",
        "unknown",
    }
)


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def canonical_t21_donor_design_spec() -> dict[str, Any]:
    """Return the frozen donor-design contract shared by profile and analysis."""
    return {
        "schema_name": "t21_canonical_donor_design",
        "schema_version": CANONICAL_T21_DONOR_DESIGN_VERSION,
        "required_donor_fields": [
            "donor_id",
            "condition",
            "pcw",
            "technical_batch",
        ],
        "missing_value_policy": "fail_closed_for_every_required_donor_field",
        "continuous_covariate_keys": ["pcw"],
        "categorical_covariate_rule": (
            "include_technical_batch_only_when_all_values_are_resolved_and_"
            "at_least_two_levels_are_present"
        ),
        "primary_technical_batch_status": "omitted_not_identifiable",
        "strata_keys": [],
        "sex_role": "sensitivity_only_not_in_primary_nuisance_design",
        "encoding": "trajectory_covariate_pseudobulk._encode_reduced_design",
        "continuous_encoding": "centered_and_population_sd_scaled",
        "categorical_encoding": "lexicographic_reference_treatment_dummies",
        "donor_order": "sha256_of_raw_donor_id_then_frozen_anonymous_slots",
    }


def canonical_t21_donor_design_spec_sha256() -> str:
    return _sha256_payload(canonical_t21_donor_design_spec())


def validate_canonical_t21_donor_design_spec(
    spec: Mapping[str, Any],
) -> str:
    expected = canonical_t21_donor_design_spec()
    if not isinstance(spec, Mapping) or dict(spec) != expected:
        raise ValueError("Analysis plan canonical donor-design contract changed")
    return canonical_t21_donor_design_spec_sha256()


def canonical_signature_scores(
    signature: np.ndarray, *, max_components: int
) -> np.ndarray:
    """Build deterministic centered contrasts equivariant to donor-row order."""
    values = np.asarray(signature, dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] < 2
        or max_components < 1
        or not np.isfinite(values).all()
    ):
        raise ValueError("Sensitivity signatures must be finite donor-by-feature arrays")
    centered = values - values.mean(axis=0, keepdims=True)
    if values.shape[1] == 0 or np.allclose(centered, 0.0):
        return np.zeros((len(values), 0), dtype=float)
    # Ordered modified Gram-Schmidt on the frozen anonymous signature columns
    # avoids arbitrary rotations when singular values are tied. A row
    # permutation simply applies the same permutation to every returned score.
    basis: list[np.ndarray] = []
    tolerance = np.finfo(float).eps * max(centered.shape) * max(
        1.0, float(np.linalg.norm(centered))
    )
    for feature in range(centered.shape[1]):
        candidate = centered[:, feature].copy()
        for vector in basis:
            candidate -= float(candidate @ vector) * vector
        norm = float(np.linalg.norm(candidate))
        if norm <= tolerance:
            continue
        basis.append(candidate / norm)
        if len(basis) == int(max_components):
            break
    if not basis:
        return np.zeros((len(values), 0), dtype=float)
    scores = np.column_stack(basis)
    scores /= np.std(scores, axis=0, ddof=0, keepdims=True)
    return scores


def build_t21_sensitivity_signature_matrix(
    *, sex_signature: np.ndarray, batch_signature: np.ndarray
) -> tuple[np.ndarray, dict[str, int]]:
    """Build non-primary sex/batch stress covariates without changing X0."""
    sex = canonical_signature_scores(sex_signature, max_components=1)
    batch = canonical_signature_scores(batch_signature, max_components=2)
    if sex.shape[0] != batch.shape[0]:
        raise ValueError("Sex and batch signatures are not donor aligned")
    matrix = np.column_stack([sex, batch])
    return matrix, {
        "sex_components": int(sex.shape[1]),
        "batch_components": int(batch.shape[1]),
    }


@dataclass(frozen=True)
class CanonicalT21DonorDesign:
    donor_frame: pd.DataFrame
    continuous_covariate_keys: tuple[str, ...]
    categorical_covariate_keys: tuple[str, ...]
    strata_keys: tuple[str, ...]
    reduced_design: np.ndarray
    terms: tuple[str, ...]
    encoding: tuple[Mapping[str, Any], ...]
    technical_batch_status: str
    spec_sha256: str
    reduced_design_sha256: str
    terms_sha256: str
    encoding_sha256: str

    @property
    def nuisance_matrix(self) -> np.ndarray:
        """Reduced design without the intercept, for outcome simulation only."""
        return np.asarray(self.reduced_design[:, 1:], dtype=float)

    def audit_manifest(self) -> dict[str, Any]:
        """Return a serializable manifest; callers must anonymize batch labels."""
        return {
            "schema_version": CANONICAL_T21_DONOR_DESIGN_VERSION,
            "spec_sha256": self.spec_sha256,
            "continuous_covariate_keys": list(self.continuous_covariate_keys),
            "categorical_covariate_keys": list(self.categorical_covariate_keys),
            "strata_keys": list(self.strata_keys),
            "technical_batch_status": self.technical_batch_status,
            "reduced_design": np.asarray(self.reduced_design, dtype=float).tolist(),
            "reduced_design_sha256": self.reduced_design_sha256,
            "terms": list(self.terms),
            "terms_sha256": self.terms_sha256,
            "encoding": [dict(value) for value in self.encoding],
            "encoding_sha256": self.encoding_sha256,
        }


def _validated_donor_values(
    donor_ids: Sequence[Any],
    conditions: Sequence[Any],
    pcw: Sequence[float],
    technical_batch: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    donors = np.asarray(donor_ids, dtype=object)
    group = np.asarray(conditions, dtype=object)
    ages = np.asarray(pcw, dtype=float)
    batches = np.asarray(technical_batch, dtype=object)
    if any(value.ndim != 1 for value in (donors, group, ages, batches)):
        raise ValueError("Canonical T21 donor-design inputs must be one-dimensional")
    if not len(donors) or not (
        len(donors) == len(group) == len(ages) == len(batches)
    ):
        raise ValueError("Canonical T21 donor-design inputs are not donor aligned")
    donor_text = np.asarray([str(value).strip() for value in donors], dtype=object)
    condition_text = np.asarray([str(value).strip() for value in group], dtype=object)
    batch_text = np.asarray([str(value).strip() for value in batches], dtype=object)
    if (
        any(not value for value in donor_text)
        or any(not value for value in condition_text)
        or len(set(donor_text.tolist())) != len(donor_text)
        or not np.isfinite(ages).all()
    ):
        raise ValueError(
            "Canonical T21 donor fields must be complete, finite, and donor-unique"
        )
    return donor_text, condition_text, ages, batch_text


def _technical_batch_status(values: np.ndarray) -> tuple[str, tuple[str, ...]]:
    normalized = np.asarray(
        [str(value).strip().lower() for value in values], dtype=object
    )
    unresolved = np.asarray(
        [value in _UNRESOLVED_BATCH_VALUES for value in normalized], dtype=bool
    )
    if unresolved.all():
        return "omitted_not_identifiable", tuple()
    if unresolved.any():
        raise ValueError(
            "Technical batch is only partially resolved; formal inference fails closed"
        )
    levels = tuple(sorted(set(str(value) for value in values)))
    if len(levels) < 2:
        return "omitted_single_level", tuple()
    return "included_identifiable", ("technical_batch",)


def build_t21_canonical_donor_design(
    *,
    donor_ids: Sequence[Any],
    conditions: Sequence[Any],
    pcw: Sequence[float],
    technical_batch: Sequence[Any],
    control: Any = "disomy",
    case: Any = "T21",
    expected_primary_batch_status: str | None = "omitted_not_identifiable",
    donor_order_mode: str = "sha256",
) -> CanonicalT21DonorDesign:
    """Build the exact nuisance design used by selection and production fitting.

    PCW is deliberately passed to the production encoder as a continuous donor
    field. Sex is not a primary nuisance term. Technical batch is included as a
    categorical term only when it is completely resolved and has multiple
    levels; partial resolution is an error.
    """
    donors, group, ages, batches = _validated_donor_values(
        donor_ids, conditions, pcw, technical_batch
    )
    donor_order_mode = str(donor_order_mode).strip().lower()
    if donor_order_mode == "sha256":
        canonical_order = np.asarray(
            sorted(
                range(len(donors)),
                key=lambda index: hashlib.sha256(
                    str(donors[index]).encode("utf-8")
                ).hexdigest(),
            ),
            dtype=int,
        )
    elif donor_order_mode == "provided_frozen_slots":
        expected = [f"D{index + 1:03d}" for index in range(len(donors))]
        if donors.tolist() != expected:
            raise ValueError(
                "provided_frozen_slots requires ordered sequential anonymous slots"
            )
        canonical_order = np.arange(len(donors), dtype=int)
    elif donor_order_mode == "provided_frozen_subset":
        if any(
            not isinstance(value, str)
            or len(value) != 4
            or not value.startswith("D")
            or not value[1:].isdigit()
            for value in donors
        ) or donors.tolist() != sorted(donors.tolist()):
            raise ValueError(
                "provided_frozen_subset requires increasing anonymous donor slots"
            )
        canonical_order = np.arange(len(donors), dtype=int)
    else:
        raise ValueError(
            "donor_order_mode must be 'sha256', 'provided_frozen_slots', or "
            "'provided_frozen_subset'"
        )
    donors = donors[canonical_order]
    group = group[canonical_order]
    ages = ages[canonical_order]
    batches = batches[canonical_order]
    control_text, case_text = str(control), str(case)
    if control_text == case_text:
        raise ValueError("control and case must differ")
    observed = set(str(value) for value in group)
    if not observed.issubset({control_text, case_text}) or observed != {
        control_text,
        case_text,
    }:
        raise ValueError("Canonical donor conditions must contain both frozen groups")
    batch_status, categorical_keys = _technical_batch_status(batches)
    if (
        expected_primary_batch_status is not None
        and batch_status != str(expected_primary_batch_status)
    ):
        raise ValueError(
            "Technical-batch identifiability differs from the frozen primary design: "
            f"expected {expected_primary_batch_status!r}, observed {batch_status!r}"
        )
    donor_frame = pd.DataFrame(
        {
            "donor": donors,
            "observed_condition": group,
            "observed_case": np.asarray(
                [str(value) == case_text for value in group], dtype=bool
            ),
            "pcw": ages,
            "technical_batch": batches,
            "__stratum_key": [("__all__",) for _ in donors],
        }
    )
    encoded = _encode_reduced_design(
        donor_frame,
        continuous_covariate_keys=("pcw",),
        categorical_covariate_keys=categorical_keys,
        strata_keys=tuple(),
    )
    reduced = np.asarray(encoded.reduced, dtype=float)
    if (
        reduced.shape[0] != len(donors)
        or reduced.shape[1] < 2
        or not np.isfinite(reduced).all()
    ):
        raise ValueError("Canonical T21 reduced design is incomplete")
    terms = tuple(str(value) for value in encoded.terms)
    encoding = tuple(dict(value) for value in encoded.encoding)
    return CanonicalT21DonorDesign(
        donor_frame=donor_frame,
        continuous_covariate_keys=("pcw",),
        categorical_covariate_keys=categorical_keys,
        strata_keys=tuple(),
        reduced_design=reduced,
        terms=terms,
        encoding=encoding,
        technical_batch_status=batch_status,
        spec_sha256=canonical_t21_donor_design_spec_sha256(),
        reduced_design_sha256=covariate_matrix_sha256(reduced),
        terms_sha256=_sha256_payload(list(terms)),
        encoding_sha256=_sha256_payload(list(encoding)),
    )


def donor_level_design_from_obs(
    obs: pd.DataFrame,
    *,
    donor_key: str,
    condition_key: str,
    pcw_key: str = "pcw",
    technical_batch_key: str = "technical_batch",
    control: Any = "disomy",
    case: Any = "T21",
    expected_primary_batch_status: str | None = "omitted_not_identifiable",
) -> CanonicalT21DonorDesign:
    """Reduce cell metadata to donor constants and build the canonical design."""
    required = [donor_key, condition_key, pcw_key, technical_batch_key]
    missing = [key for key in required if key not in obs]
    if missing:
        raise KeyError(f"Missing canonical T21 design columns: {missing}")
    selected = obs.loc[
        obs[condition_key].astype(str).isin([str(control), str(case)]), required
    ].copy()
    if selected.empty or selected.isna().any().any():
        raise ValueError("Canonical T21 donor-design cells are missing")
    rows: list[dict[str, Any]] = []
    for donor, group in selected.groupby(donor_key, sort=True, observed=True):
        row: dict[str, Any] = {"donor_id": str(donor)}
        for source, target in (
            (condition_key, "condition"),
            (pcw_key, "pcw"),
            (technical_batch_key, "technical_batch"),
        ):
            values = pd.unique(group[source])
            if len(values) != 1:
                raise ValueError(
                    f"Canonical T21 field {source!r} is not donor-constant for {donor!r}"
                )
            row[target] = values[0]
        rows.append(row)
    frame = pd.DataFrame(rows)
    return build_t21_canonical_donor_design(
        donor_ids=frame["donor_id"],
        conditions=frame["condition"],
        pcw=pd.to_numeric(frame["pcw"], errors="raise"),
        technical_batch=frame["technical_batch"],
        control=control,
        case=case,
        expected_primary_batch_status=expected_primary_batch_status,
    )


def _run_t21_bound_covariate_pseudobulk(
    scrna_path: str | Path,
    trajectory_path: str | Path,
    fates_path: str | Path,
    pathway_universe_path: str | Path,
    design_profile_path: str | Path,
):
    """Internal zero-override numeric kernel for already gated bound artifacts."""
    import anndata as ad
    import zarr

    condition_key = "condition"
    donor_key = "donor_id"
    control = "disomy"
    case = "T21"
    pcw_key = "pcw"
    technical_batch_key = "technical_batch"
    expected_primary_batch_status = str(
        canonical_t21_donor_design_spec()["primary_technical_batch_status"]
    )
    root = Path(__file__).resolve().parents[1]
    paths = {
        "scrna": Path(scrna_path).resolve(),
        "trajectory": Path(trajectory_path).resolve(),
        "fates": Path(fates_path).resolve(),
        "pathway_universe": Path(pathway_universe_path).resolve(),
        "design_profile": Path(design_profile_path).resolve(),
    }
    for role, path in paths.items():
        if role == "trajectory":
            if not path.is_dir():
                raise FileNotFoundError(path)
        elif not path.is_file():
            raise FileNotFoundError(path)

    # Local imports avoid the profile-builder/canonical-design import cycle.
    from .t21_calibration_profile import (
        _pathway_structure,
        load_calibration_design_profile,
    )
    from .t21_data_product import (
        cell_id_set_hash,
        formal_t21_analysis_view,
        sha256_file,
        stable_json,
        tree_digest,
        validate_scrna_contract,
        validate_trajectory_scrna_alignment,
    )
    from .t21_expression_preprocessing import (
        compute_pooled_gene_support_chunked,
        filter_gene_sets_for_supported_expression,
        formal_expression_preprocessing_contract,
        formal_expression_preprocessing_contract_sha256,
        formal_expression_preprocessing_source_sha256,
        validate_t21_formal_expression,
    )
    from .t21_preunblinding_calibration import (
        _make_profile_shared_curve_plan,
        derive_profile_simulation_parameters,
        load_runner_spec,
    )

    bound_design_profile = load_calibration_design_profile(
        paths["design_profile"], repository_root=root
    )
    bound_profile_payload_sha256 = str(
        bound_design_profile["integrity"]["profile_payload_sha256"]
    )
    profile_design = bound_design_profile.get("design")
    if not isinstance(profile_design, Mapping):  # pragma: no cover - schema guarded
        raise ValueError("Bound calibration design profile lacks its design payload")
    bound_canonical_manifest = profile_design.get("canonical_formal_design")
    if not isinstance(bound_canonical_manifest, Mapping):
        raise ValueError("Bound calibration profile lacks a canonical design manifest")
    input_bindings = bound_design_profile.get("input_bindings")
    scrna_binding = (
        input_bindings.get("scrna") if isinstance(input_bindings, Mapping) else None
    )
    if not isinstance(scrna_binding, Mapping):  # pragma: no cover - schema guarded
        raise ValueError("Bound calibration profile lacks the scRNA product binding")
    trajectory_binding = input_bindings.get("trajectory")
    fates_binding = input_bindings.get("fates")
    pathway_binding = input_bindings.get("pathway_universe")
    if not all(
        isinstance(value, Mapping)
        for value in (trajectory_binding, fates_binding, pathway_binding)
    ):
        raise ValueError("Bound profile lacks formal external-artifact bindings")
    exact_file_bindings = {
        "scrna": sha256_file(paths["scrna"]),
        "fates": sha256_file(paths["fates"]),
        "pathway_universe": sha256_file(paths["pathway_universe"]),
    }
    for role, observed_hash in exact_file_bindings.items():
        binding = input_bindings[role]
        if binding.get("file_sha256") != observed_hash:
            raise ValueError(f"Formal {role} artifact differs from the bound profile")
    if trajectory_binding.get("tree_digest_sha256") != tree_digest(
        paths["trajectory"]
    ):
        raise ValueError("Formal trajectory tree differs from the bound profile")

    required_manifest = {
        "spec_sha256",
        "technical_batch_status",
        "reduced_design_sha256",
        "terms_sha256",
        "encoding_sha256",
    }
    if required_manifest - set(bound_canonical_manifest):
        raise ValueError("Bound canonical design manifest is incomplete")
    if (
        bound_canonical_manifest["spec_sha256"]
        != canonical_t21_donor_design_spec_sha256()
        or bound_canonical_manifest["technical_batch_status"]
        != expected_primary_batch_status
    ):
        raise ValueError("Bound canonical design contract differs from the formal plan")
    runner_path = root / "config" / "t21_preunblinding_calibration_runner_v2.yaml"
    runner_spec = load_runner_spec(runner_path)
    derived = derive_profile_simulation_parameters(
        runner_spec, bound_design_profile
    )
    residual_mapping_seed = int(runner_spec["inference"]["residual_mapping_seed"])
    shared_plan = _make_profile_shared_curve_plan(
        runner_spec, derived, seed=residual_mapping_seed
    )

    adata = ad.read_h5ad(paths["scrna"], backed="r")
    try:
        # Validate the full immutable product before any internal selection.
        full_summary = validate_scrna_contract(
            adata,
            strict_analysis_labels=True,
            require_formal_expression=False,
        )
        candidate_logical_bindings = {
            "cell_set_hash": full_summary["cell_id_set_hash"],
            "gene_order_hash": full_summary["gene_order_hash"],
            "donor_set_hash": cell_id_set_hash(
                pd.unique(adata.obs[donor_key].astype(str))
            ),
        }
        for key, observed_hash in candidate_logical_bindings.items():
            if scrna_binding.get(key) != observed_hash:
                raise ValueError(
                    "Candidate scRNA expression product differs from bound profile "
                    f"at {key}"
                )
        expression_metadata = adata.uns.get("t21_data_product", {}).get(
            "expression_contract"
        )
        if not isinstance(expression_metadata, Mapping):
            raise ValueError("Formal H5AD lacks expression-contract metadata")
        if stable_json(expression_metadata.get("contract")) != stable_json(
            formal_expression_preprocessing_contract()
        ):
            raise ValueError("Formal H5AD expression contract changed")
        expression_validation = expression_metadata.get("validation")
        if not isinstance(expression_validation, Mapping):
            raise ValueError("Formal H5AD lacks expression-validation metadata")
        expression_expected = {
            "expression_contract_sha256": (
                formal_expression_preprocessing_contract_sha256()
            ),
            "expression_implementation_sha256": (
                formal_expression_preprocessing_source_sha256()
            ),
            "x_semantic_sha256": str(
                expression_validation.get("expression_csr_semantic_sha256", "")
            ),
        }
        for key, expected_value in expression_expected.items():
            if scrna_binding.get(key) != expected_value:
                raise ValueError(f"Formal expression binding changed at {key}")

        for start in range(0, int(adata.n_obs), 4096):
            stop = min(start + 4096, int(adata.n_obs))
            count_block = adata.layers["counts"][start:stop, :]
            expression_block = adata.X[start:stop, :]
            if hasattr(count_block, "to_memory"):
                count_block = count_block.to_memory()
            if hasattr(expression_block, "to_memory"):
                expression_block = expression_block.to_memory()
            validate_t21_formal_expression(count_block, expression_block)

        trajectory_summary = validate_trajectory_scrna_alignment(
            paths["trajectory"], adata.obs
        )
        for key in ("grid_hash", "cell_id_set_hash", "donor_set_hash"):
            profile_key = "cell_set_hash" if key == "cell_id_set_hash" else key
            if trajectory_binding.get(profile_key) != trajectory_summary.get(key):
                raise ValueError(f"Formal trajectory binding changed at {profile_key}")
        formal_view = formal_t21_analysis_view(
            adata,
            trajectory_path=paths["trajectory"],
            fates_path=paths["fates"],
        )
        view_expected = {
            "formal_analysis_cell_set_hash": formal_view["analysis_cell_set_hash"],
            "formal_analysis_cell_order_hash": formal_view["analysis_cell_order_hash"],
            "formal_analysis_cell_count": formal_view["n_analysis_cells"],
        }
        for binding_role in ("scrna", "trajectory", "fates"):
            for key, expected_value in view_expected.items():
                if input_bindings[binding_role].get(key) != expected_value:
                    raise ValueError(
                        f"Formal {binding_role} analysis-view binding changed at {key}"
                    )

        analysis_mask = np.asarray(formal_view["analysis_mask"], dtype=bool)

        def count_row_reader(start: int, stop: int):
            block = adata.layers["counts"][start:stop, :]
            return block.to_memory() if hasattr(block, "to_memory") else block

        support = compute_pooled_gene_support_chunked(
            count_row_reader,
            n_cells=int(adata.n_obs),
            n_genes=int(adata.n_vars),
            analysis_cell_mask=analysis_mask,
            gene_ids=adata.var_names.astype(str).tolist(),
            chunk_size=4096,
        )
        support_contract = support.to_contract_dict()
        support_expected = {
            "formal_gene_order_bound_support_sha256": (
                support.gene_order_bound_support_sha256
            ),
            "formal_support_contract_sha256": support.support_contract_sha256,
            "formal_support_mask_sha256_uint8": support.support_mask_sha256_uint8,
            "formal_analysis_cell_mask_sha256_uint8": (
                support.analysis_cell_mask_sha256_uint8
            ),
            "formal_supported_gene_count": support.n_supported_genes,
        }
        for key, expected_value in support_expected.items():
            if scrna_binding.get(key) != expected_value:
                raise ValueError(f"Formal pooled-gene support changed at {key}")

        pathway_frame = pd.read_csv(
            paths["pathway_universe"], sep="\t", dtype=str, keep_default_na=False
        )
        required_pathway_columns = {
            "pathway_id",
            "gene_id",
            "gene_order",
            "pathway_universe_logical_sha256",
        }
        if required_pathway_columns.difference(pathway_frame):
            raise ValueError("Formal pathway universe lacks required columns")
        logical_hashes = sorted(
            set(pathway_frame["pathway_universe_logical_sha256"].astype(str))
        )
        if logical_hashes != [str(pathway_binding["logical_sha256"])]:
            raise ValueError("Formal pathway-universe logical hash changed")
        raw_gene_sets = {
            str(pathway_id): tuple(
                group.sort_values(
                    "gene_order", key=lambda values: pd.to_numeric(values, errors="raise")
                )["gene_id"].astype(str)
            )
            for pathway_id, group in pathway_frame.groupby(
                "pathway_id", sort=True, observed=True
            )
        }
        filtered_gene_sets = filter_gene_sets_for_supported_expression(
            raw_gene_sets, support, min_size=5, max_size=500
        )
        gene_sets = filtered_gene_sets.as_mapping()
        pathway_structure, original_logical_hash = _pathway_structure(
            paths["pathway_universe"],
            supported_gene_ids=set(support.supported_gene_ids),
            min_size=5,
            max_size=500,
        )
        if original_logical_hash != pathway_binding.get("logical_sha256") or (
            pathway_structure["supported_pathway_universe_logical_sha256"]
            != pathway_binding.get("supported_logical_sha256")
        ):
            raise ValueError("Supported pathway-universe contract changed")
        if len(gene_sets) != int(pathway_structure["n_pathways"]):
            raise ValueError("Formal pathway filtering differs from the bound profile")

        trajectory_group = zarr.open_group(paths["trajectory"], mode="r")
        bin_left = np.asarray(trajectory_group["axes/bin_left"][:], dtype=float)
        bin_right = np.asarray(trajectory_group["axes/bin_right"][:], dtype=float)
        grid_edges = np.concatenate((bin_left[:1], bin_right))
        if not np.array_equal(
            bin_left, np.asarray(bound_design_profile["fixed_grid"]["bin_left"], dtype=float)
        ) or not np.array_equal(
            bin_right, np.asarray(bound_design_profile["fixed_grid"]["bin_right"], dtype=float)
        ):
            raise ValueError("Formal trajectory grid differs from the bound profile")

        analysis_adata = adata[analysis_mask].to_memory()
        private_pseudotime_key = "__t21_bound_primary_pseudotime"
        analysis_adata.obs[private_pseudotime_key] = np.asarray(
            formal_view["primary_pseudotime"], dtype=float
        )[analysis_mask]
        design = donor_level_design_from_obs(
            analysis_adata.obs,
            donor_key=donor_key,
            condition_key=condition_key,
            pcw_key=pcw_key,
            technical_batch_key=technical_batch_key,
            control=control,
            case=case,
            expected_primary_batch_status=expected_primary_batch_status,
        )
        expected_hashes = {
            "spec_sha256": design.spec_sha256,
            "reduced_design_sha256": design.reduced_design_sha256,
            "terms_sha256": design.terms_sha256,
            "encoding_sha256": design.encoding_sha256,
        }
        for key, observed in expected_hashes.items():
            if bound_canonical_manifest.get(key) != observed:
                raise ValueError(
                    f"Candidate donor design differs from bound profile at {key}"
                )

        support_settings = runner_spec["inference"]["support_selection"]
        result = run_covariate_adjusted_donor_pseudobulk(
            analysis_adata,
            gene_sets,
            condition_key=condition_key,
            donor_key=donor_key,
            control=control,
            case=case,
            pseudotime_key=private_pseudotime_key,
            continuous_covariate_keys=(pcw_key,),
            categorical_covariate_keys=(
                (technical_batch_key,)
                if design.categorical_covariate_keys
                else tuple()
            ),
            strata_keys=tuple(),
            donor_order="sha256",
            grid_edges=grid_edges,
            n_bins=20,
            pseudotime_range=(0.0, 1.0),
            min_cells_per_donor_bin=int(
                support_settings["min_cells_per_donor_bin"]
            ),
            min_donors_per_condition=int(
                support_settings["min_donors_per_condition"]
            ),
            min_common_bins=int(support_settings["min_common_bins"]),
            min_residual_df=int(support_settings["min_residual_df"]),
            max_condition_vif=float(support_settings["max_condition_vif"]),
            statistic="max_absolute_effect",
            tail="greater",
            calibration_scale="studentized",
            permutation_mode=str(runner_spec["inference"]["residual_reference_mode"]),
            n_permutations=int(
                runner_spec["inference"]["monte_carlo_residual_mappings"]
            ),
            max_exact_permutations=int(
                runner_spec["inference"]["max_exhaustive_residual_mappings"]
            ),
            min_size=5,
            max_size=500,
            layer=None,
            use_raw=False,
            alpha=float(runner_spec["design"]["alpha"]),
            power_target=float(runner_spec["power"]["target_power"]),
            seed=residual_mapping_seed,
            return_null_statistics=False,
            return_permutation_assignments=True,
            return_donor_bin_activity=False,
            retain_all_genes=False,
        )
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    observed = result.donor_design.sort_values("donor_index").reset_index(drop=True)
    result_design = build_t21_canonical_donor_design(
        donor_ids=observed["donor"],
        conditions=observed["observed_condition"],
        pcw=observed[pcw_key],
        technical_batch=(
            observed[technical_batch_key]
            if technical_batch_key in observed
            else np.repeat("omitted_not_identifiable", len(observed))
        ),
        control=control,
        case=case,
        expected_primary_batch_status=expected_primary_batch_status,
    )
    if any(
        getattr(result_design, attribute) != getattr(design, attribute)
        for attribute in (
            "spec_sha256",
            "reduced_design_sha256",
            "terms_sha256",
            "encoding_sha256",
        )
    ):
        raise RuntimeError("Production donor design differs from its canonical preflight")

    def array_hash(values: Any, dtype: str) -> str:
        return hashlib.sha256(
            np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype))).tobytes()
        ).hexdigest()

    source_counts = np.asarray(derived["source_primary_draw_cell_count"], dtype=np.int64)
    source_available = np.asarray(
        derived["source_support_available_mask"], dtype=bool
    )
    selected_counts = np.asarray(
        derived["occupancy_baseline_cell_count"], dtype=np.int64
    )
    selected_available = np.asarray(
        derived["primary_draw_available_mask"], dtype=bool
    )
    selected_bin_mask = np.asarray(derived["selected_bin_mask"], dtype=bool)
    included_donor_mask = np.asarray(derived["included_donor_mask"], dtype=bool)
    residual_mappings = np.stack(shared_plan.null_mappings, axis=0)
    expected_metadata = {
        "source_grid_n_bins": 20,
        "grid_edges_sha256_float64_le": array_hash(grid_edges, "<f8"),
        "source_grid_counts_sha256_int64_le": array_hash(source_counts, "<i8"),
        "source_grid_availability_sha256_uint8": array_hash(source_available, "u1"),
        "selected_bin_ids": list(map(int, derived["selected_bin_indices"])),
        "selected_bin_mask_sha256_uint8": array_hash(selected_bin_mask, "u1"),
        "included_donor_mask_sha256_uint8": array_hash(included_donor_mask, "u1"),
        "selected_counts_sha256_int64_le": array_hash(selected_counts, "<i8"),
        "selected_availability_sha256_uint8": array_hash(selected_available, "u1"),
        "reduced_design_sha256_float64_le": array_hash(
            np.asarray(derived["support_reduced_design"], dtype=float), "<f8"
        ),
        "residual_permutation_space_size": int(shared_plan.residual_space_size),
        "condition_label_space_size": int(shared_plan.restricted_label_space_size),
        "n_null_mappings_evaluated": int(shared_plan.n_null_mappings),
        "residual_mappings_sha256_int64_le": array_hash(
            residual_mappings, "<i8"
        ),
        "n_permutations_requested": int(
            runner_spec["inference"]["monte_carlo_residual_mappings"]
        ),
        "max_exact_permutations": int(
            runner_spec["inference"]["max_exhaustive_residual_mappings"]
        ),
        "seed": residual_mapping_seed,
        "reference_enumeration": shared_plan.reference_enumeration,
        "exactness_status": shared_plan.exactness_status,
        "permutation_p_resolution": float(shared_plan.monte_carlo_p_resolution),
        "freedman_lane_reference_p_resolution": float(
            shared_plan.monte_carlo_p_resolution
        ),
        "expression_source": "X",
        "layer": None,
        "use_raw": False,
        "condition_key": condition_key,
        "donor_key": donor_key,
        "control": control,
        "case": case,
        "pseudotime_key": private_pseudotime_key,
        "continuous_covariate_keys": [pcw_key],
        "categorical_covariate_keys": list(design.categorical_covariate_keys),
        "strata_keys": [],
        "donor_order_rule": "sha256",
        "min_cells_per_donor_bin": int(
            support_settings["min_cells_per_donor_bin"]
        ),
        "min_donors_per_condition": int(
            support_settings["min_donors_per_condition"]
        ),
        "min_common_bins": int(support_settings["min_common_bins"]),
        "min_residual_df": int(support_settings["min_residual_df"]),
        "max_condition_vif": float(support_settings["max_condition_vif"]),
        "statistic": "max_absolute_effect",
        "tail": "greater",
        "calibration_scale": "studentized",
        "min_size": 5,
        "max_size": 500,
        "alpha": float(runner_spec["design"]["alpha"]),
        "power_target": float(runner_spec["power"]["target_power"]),
        "return_donor_bin_activity": False,
        "retain_all_genes": False,
    }
    for key, expected_value in expected_metadata.items():
        if result.metadata.get(key) != expected_value:
            raise RuntimeError(
                f"Production formal metadata differs from the bound plan at {key}"
            )
    if not np.array_equal(
        np.asarray(result.metadata.get("grid_edges"), dtype=float), grid_edges
    ):
        raise RuntimeError("Production grid edges differ from the trajectory artifact")
    if result.metadata.get("reduced_model_terms") != list(design.terms):
        raise RuntimeError("Production reduced-model terms changed")
    if json.loads(str(result.metadata.get("design_encoding_json"))) != list(
        design.encoding
    ):
        raise RuntimeError("Production design encoding changed")
    result.metadata.update(
        {
            "t21_formal_canonical_design_used": True,
            "bound_profile_payload_sha256": bound_profile_payload_sha256,
            "bound_profile_file_sha256": sha256_file(paths["design_profile"]),
            "bound_scrna_cell_set_hash": candidate_logical_bindings["cell_set_hash"],
            "bound_scrna_gene_order_hash": candidate_logical_bindings[
                "gene_order_hash"
            ],
            "bound_scrna_donor_set_hash": candidate_logical_bindings["donor_set_hash"],
            "canonical_donor_design_spec_sha256": design.spec_sha256,
            "canonical_reduced_design_sha256": design.reduced_design_sha256,
            "canonical_terms_sha256": design.terms_sha256,
            "canonical_encoding_sha256": design.encoding_sha256,
            "bound_scrna_file_sha256": exact_file_bindings["scrna"],
            "bound_trajectory_tree_digest_sha256": tree_digest(paths["trajectory"]),
            "bound_fates_file_sha256": exact_file_bindings["fates"],
            "bound_pathway_universe_file_sha256": exact_file_bindings[
                "pathway_universe"
            ],
            "bound_pathway_universe_logical_sha256": original_logical_hash,
            "bound_supported_pathway_universe_logical_sha256": pathway_structure[
                "supported_pathway_universe_logical_sha256"
            ],
            "bound_expression_contract_sha256": expression_expected[
                "expression_contract_sha256"
            ],
            "bound_expression_implementation_sha256": expression_expected[
                "expression_implementation_sha256"
            ],
            "bound_x_semantic_sha256": expression_expected["x_semantic_sha256"],
            "formal_analysis_cell_set_hash": formal_view["analysis_cell_set_hash"],
            "formal_analysis_cell_order_hash": formal_view[
                "analysis_cell_order_hash"
            ],
            "formal_analysis_cell_count": formal_view["n_analysis_cells"],
            "formal_gene_support_contract": support_contract,
            "formal_gene_support_filter_before_min_size_and_weights": True,
            "formal_pathway_filter_audit": list(filtered_gene_sets.audit),
            "formal_residual_mapping_seed": residual_mapping_seed,
        }
    )
    return result


def _require_exact_passing_release_gates(
    rows: Sequence[Mapping[str, Any]], required_gates: set[str]
) -> None:
    gate_rows: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if isinstance(row, Mapping):
            gate_rows.setdefault(str(row.get("gate")), []).append(row)
    if any(
        len(gate_rows.get(gate, [])) != 1
        or gate_rows[gate][0].get("status") != "pass"
        for gate in required_gates
    ):
        raise ValueError("Published release lacks a unique passing formal gate")


def _require_release_profile_matches_calibration_report(
    *,
    profile_path: str | Path,
    profile: Mapping[str, Any],
    report: Mapping[str, Any],
    repository_root: str | Path,
) -> None:
    """Bind the formal-analysis profile to the exact profile calibrated in report."""
    from .t21_data_product import sha256_file

    root = Path(repository_root).resolve()
    formal_profile_path = Path(profile_path).resolve()
    bindings = report.get("input_bindings")
    integrity = profile.get("integrity")
    if not isinstance(bindings, Mapping) or not isinstance(integrity, Mapping):
        raise ValueError("Published calibration lacks design-profile bindings")
    formal_file_sha256 = sha256_file(formal_profile_path)
    if (
        bindings.get("design_profile_sha256") != formal_file_sha256
        or bindings.get("design_profile_payload_sha256")
        != integrity.get("profile_payload_sha256")
    ):
        raise ValueError(
            "Formal analysis profile differs from calibration report bindings"
        )
    records = [
        record
        for record in report.get("output_artifacts", [])
        if isinstance(record, Mapping)
        and record.get("role") == "bound_calibration_design_profile"
    ]
    if len(records) != 1:
        raise ValueError(
            "Calibration report requires exactly one bound design-profile artifact"
        )
    record = records[0]
    relative = Path(str(record.get("relative_path", "")))
    if relative.is_absolute() or ".." in relative.parts or not str(relative):
        raise ValueError("Calibration report design-profile path is invalid")
    report_profile_path = (root / relative).resolve()
    try:
        report_profile_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Calibration report design profile escapes repository") from exc
    if report_profile_path != formal_profile_path:
        raise ValueError(
            "Formal analysis profile path differs from calibrated profile artifact"
        )
    if (
        int(record.get("bytes", -1)) != formal_profile_path.stat().st_size
        or record.get("sha256") != formal_file_sha256
    ):
        raise ValueError("Calibration report design-profile artifact changed")


def _validated_published_t21_release_inputs(
    release_dir: str | Path,
) -> tuple[Path, Path, Path, Path, Path]:
    """Validate the published release/calibration gate and resolve bound inputs."""
    import jsonschema
    import yaml

    from .t21_data_product import (
        FINAL_PRODUCT_NAMES,
        sha256_file,
        tree_digest,
        validate_pre_unblinding_calibration,
        validate_unblinding_decision,
    )
    from .t21_preunblinding_calibration import (
        validate_pre_unblinding_calibration_artifacts,
    )

    root = Path(__file__).resolve().parents[1]
    release = Path(release_dir).resolve()
    canonical_release_root = (
        root / "data_external" / "t21_data_product_v1" / "releases"
    ).resolve()
    try:
        relative_release = release.relative_to(canonical_release_root)
    except ValueError as exc:
        raise ValueError(
            "Formal T21 analysis accepts only an immutable published release"
        ) from exc
    if len(relative_release.parts) != 1 or release.name.startswith("."):
        raise ValueError("Nested, staging, and pending release paths are forbidden")
    manifest_path = release / FINAL_PRODUCT_NAMES["provenance"]
    receipt_path = canonical_release_root / f"{release.name}.receipt.json"
    if not manifest_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError("Published release manifest/receipt is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_schema = json.loads(
        (root / "schemas" / "t21_data_provenance_manifest_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.validate(manifest, manifest_schema)
    if (
        manifest.get("release_id") != release.name
        or manifest.get("release_status") not in {"validated_release_candidate", "final"}
    ):
        raise ValueError("Release manifest is not a published validated release")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema_name") != "t21_release_receipt"
        or receipt.get("release_id") != release.name
        or receipt.get("manifest_sha256") != sha256_file(manifest_path)
        or receipt.get("status") != "validated_release_candidate"
    ):
        raise ValueError("Published release receipt does not bind the manifest")

    output_records_by_role: dict[str, list[Mapping[str, Any]]] = {}
    for record in manifest.get("outputs", []):
        if isinstance(record, Mapping):
            output_records_by_role.setdefault(str(record.get("role")), []).append(
                record
            )
    resolved_outputs: dict[str, Path] = {}
    for role in ("scrna", "donor_design", "trajectory", "fates"):
        records = output_records_by_role.get(role, [])
        if len(records) != 1:
            raise ValueError(
                f"Release manifest requires exactly one output role {role}"
            )
        record = records[0]
        expected_path = (release / FINAL_PRODUCT_NAMES[role]).resolve()
        recorded_path = (root / str(record.get("relative_path", ""))).resolve()
        if recorded_path != expected_path:
            raise ValueError(f"Release output path changed for {role}")
        if role == "trajectory":
            if not expected_path.is_dir() or record.get(
                "tree_digest_sha256"
            ) != tree_digest(expected_path):
                raise ValueError("Published trajectory tree changed")
        elif not expected_path.is_file() or record.get("sha256") != sha256_file(
            expected_path
        ):
            raise ValueError(f"Published {role} artifact changed")
        resolved_outputs[role] = expected_path

    required_gates = {
        "four_data_product_contracts",
        "trajectory_donor_bin_recomputation",
        "pre_unblinding_calibration",
        "canonical_pathway_universe_integrity",
        "unblinding_decision",
        "captured_evidence_snapshot_stability",
    }
    _require_exact_passing_release_gates(
        manifest.get("validation_gates", []), required_gates
    )

    sources_by_role: dict[str, list[Mapping[str, Any]]] = {}
    for record in manifest.get("sources", []):
        if isinstance(record, Mapping):
            sources_by_role.setdefault(str(record.get("role")), []).append(record)

    def one_bound_source(role: str) -> tuple[Path, Mapping[str, Any]]:
        records = sources_by_role.get(role, [])
        if len(records) != 1:
            raise ValueError(f"Release manifest requires exactly one source role {role}")
        record = records[0]
        relative = Path(str(record.get("repository_relative_path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Release source role {role} escapes the repository")
        path = (root / relative).resolve()
        path.relative_to(root.resolve())
        if (
            not path.is_file()
            or int(record.get("bytes", -1)) != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
        ):
            raise ValueError(f"Release source role {role} changed")
        return path, record

    report_path, _ = one_bound_source("pre_unblinding_calibration_report")
    policy_path, _ = one_bound_source("calibration_acceptance_policy")
    decision_path, _ = one_bound_source("unblinding_decision")
    profile_path, _ = one_bound_source(
        "calibration_bound_calibration_design_profile"
    )
    from .t21_calibration_profile import load_calibration_design_profile

    bound_profile = load_calibration_design_profile(
        profile_path, repository_root=root
    )
    if bound_profile["input_bindings"]["donor_design"][
        "file_sha256"
    ] != sha256_file(resolved_outputs["donor_design"]):
        raise ValueError("Published donor design differs from the bound profile")
    canonical_policy = (root / "config" / "t21_calibration_acceptance_v2.yaml").resolve()
    if policy_path != canonical_policy:
        raise ValueError("Published release does not bind the canonical v2 policy")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_schema_path = root / "schemas" / "t21_calibration_report_v2.schema.json"
    jsonschema.validate(
        report, json.loads(report_schema_path.read_text(encoding="utf-8"))
    )
    _require_release_profile_matches_calibration_report(
        profile_path=profile_path,
        profile=bound_profile,
        report=report,
        repository_root=root,
    )
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    expected_bindings = report.get("input_bindings")
    if not isinstance(expected_bindings, Mapping):
        raise ValueError("Published calibration report lacks exact bindings")
    calibration_summary = validate_pre_unblinding_calibration(
        report,
        policy,
        expected_bindings=expected_bindings,
        repository_root=root,
    )
    if calibration_summary.get("publication_calibration_eligible") is not True:
        raise ValueError("Published calibration is not publication eligible")
    raw_summary = validate_pre_unblinding_calibration_artifacts(
        report,
        repository_root=root,
        runner_spec_path=root
        / "config"
        / "t21_preunblinding_calibration_runner_v2.yaml",
    )
    if raw_summary.get("design_profile_usage_verified") is not True:
        raise ValueError("Published calibration raw artifacts failed profile binding")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    jsonschema.validate(
        decision,
        json.loads(
            (root / "schemas" / "t21_unblinding_decision_v1.schema.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    validate_unblinding_decision(
        decision,
        report_sha256=sha256_file(report_path),
        policy_sha256=sha256_file(policy_path),
    )

    cross = manifest.get("cross_artifact_contract")
    if not isinstance(cross, Mapping):
        raise ValueError("Release manifest lacks its cross-artifact contract")
    pathway_relative = Path(str(cross.get("pathway_universe_relative_path", "")))
    pathway_path = (root / "data_external" / "t21_data_product_v1" / pathway_relative).resolve()
    pathway_path.relative_to(
        (root / "data_external" / "t21_data_product_v1").resolve()
    )
    if (
        not pathway_path.is_file()
        or cross.get("pathway_universe_sha256") != sha256_file(pathway_path)
    ):
        raise ValueError("Published pathway universe changed")
    return (
        resolved_outputs["scrna"],
        resolved_outputs["trajectory"],
        resolved_outputs["fates"],
        pathway_path,
        profile_path,
    )


def run_t21_formal_covariate_pseudobulk(release_dir: str | Path):
    """Run real T21 inference only from a fully published, unlocked release."""
    inputs = _validated_published_t21_release_inputs(release_dir)
    result = _run_t21_bound_covariate_pseudobulk(*inputs)
    result.metadata.update(
        {
            "formal_public_release_gate_verified": True,
            "formal_release_directory": Path(release_dir).resolve().name,
            "pre_unblinding_calibration_gate_verified": True,
            "human_attested_unblinding_decision_verified": True,
            "cryptographic_signature_claimed": False,
        }
    )
    return result


def covariate_matrix_sha256(matrix: np.ndarray) -> str:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Covariate matrix must be finite and two-dimensional")
    return _sha256_payload(values.tolist())
