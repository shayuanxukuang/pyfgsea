"""Frozen, outcome-blinded pathway universe construction for the T21 product.

The builder in this module is intentionally strict.  It binds a declarative
YAML specification to byte-verified gene-reference and GMT inputs, maps HGNC
symbols to frozen Ensembl identifiers, and emits deterministic logical content.
Missing symbols are exclusions recorded in an audit table; ambiguous symbols
are accepted only when the specification contains an exact resolution.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
from types import MappingProxyType
import tempfile
from typing import Any, Mapping, Optional, Sequence, Tuple, Union
from uuid import uuid4

import pandas as pd
import yaml
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver


UNIVERSE_TSV_NAME = "t21_pathway_universe_v1.tsv"
FAMILIES_TSV_NAME = "t21_pathway_families_v1.tsv"
MAPPING_AUDIT_TSV_NAME = "t21_pathway_mapping_audit_v1.tsv"
BUILD_RECORD_JSON_NAME = "t21_pathway_universe_build_record_v1.json"
OUTPUT_FILE_NAMES = (
    UNIVERSE_TSV_NAME,
    FAMILIES_TSV_NAME,
    MAPPING_AUDIT_TSV_NAME,
    BUILD_RECORD_JSON_NAME,
)

MEMBERSHIP_COLUMNS = (
    "universe_id",
    "pathway_id",
    "pathway_label",
    "source_id",
    "source_collection",
    "gene_order",
    "source_symbol",
    "gene_id",
    "is_chr21",
    "mapping_status",
    "level_1_family_id",
    "level_2_included",
    "level_2_multiple_testing",
    "outcome_blinded",
    "spec_sha256",
    "gene_reference_sha256",
    "source_gmt_sha256",
    "source_inputs_sha256",
    "pathway_logical_sha256",
    "pathway_universe_logical_sha256",
)
FAMILY_COLUMNS = (
    "universe_id",
    "analysis_level",
    "family_id",
    "family_label",
    "pathways_json",
    "n_pathways",
    "role",
    "formal_inference",
    "multiple_testing",
    "interpretation_limit",
    "outcome_blinded",
    "spec_sha256",
    "gene_reference_sha256",
    "source_inputs_sha256",
    "pathway_universe_logical_sha256",
)
MAPPING_AUDIT_COLUMNS = (
    "universe_id",
    "pathway_id",
    "pathway_label",
    "source_id",
    "source_member_order",
    "source_symbol",
    "gene_id",
    "is_chr21",
    "included",
    "mapping_status",
    "detail",
    "outcome_blinded",
    "spec_sha256",
    "gene_reference_sha256",
    "source_gmt_sha256",
    "source_inputs_sha256",
    "pathway_universe_logical_sha256",
)

_ROOT_KEYS = frozenset(
    {
        "schema_name",
        "schema_version",
        "universe_id",
        "frozen_at_utc",
        "outcome_blinded_at_freeze",
        "real_pathway_results_inspected",
        "gene_reference",
        "sources",
        "mapping_policy",
        "pathways",
        "level_1",
        "level_2",
        "level_3",
        "coverage_gaps",
        "claim_rule",
    }
)
_GENE_REFERENCE_KEYS = frozenset(
    {
        "path",
        "sha256",
        "gene_id_column",
        "gene_symbol_column",
        "chromosome_21_column",
        "genome_build",
    }
)
_SOURCE_KEYS = frozenset(
    {
        "collection",
        "provider",
        "url",
        "retrieved_at_utc",
        "path",
        "sha256",
        "expected_pathways",
    }
)
_MAPPING_POLICY_KEYS = frozenset(
    {
        "source_namespace",
        "output_namespace",
        "absent_symbol_action",
        "ambiguous_symbol_action",
        "minimum_mapped_genes",
        "maximum_mapped_genes",
        "ambiguous_symbol_resolutions",
        "ambiguity_resolution_rule",
    }
)
_LEVEL_1_KEYS = frozenset(
    {"role", "formal_inference", "multiple_testing", "families"}
)
_FAMILY_KEYS = frozenset(
    {"label", "pathways", "interpretation_limit"}
)
_LEVEL_2_KEYS = frozenset(
    {"role", "formal_inference", "multiple_testing", "pathways"}
)
_LEVEL_3_KEYS = frozenset(
    {
        "role",
        "formal_inference",
        "multiple_testing",
        "allowed_uses",
        "prohibited_uses",
    }
)
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_EMPTY_LIMITS = frozenset({"", "none", "n/a", "na", "not_applicable"})


class T21PathwayUniverseValidationError(ValueError):
    """Raised when a frozen pathway-universe contract is violated."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


@dataclass(frozen=True)
class T21PathwayUniverseResult:
    """Validated pathway sets and their audit-ready normalized tables."""

    gene_sets: Mapping[str, Tuple[str, ...]]
    membership: pd.DataFrame
    families: pd.DataFrame
    mapping_audit: pd.DataFrame
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "gene_sets": dict(self.gene_sets),
            "membership": self.membership.copy(),
            "families": self.families.copy(),
            "mapping_audit": self.mapping_audit.copy(),
            "metadata": dict(self.metadata),
        }


def _stable_json(value: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _logical_hash(value: Any) -> str:
    return sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise T21PathwayUniverseValidationError(
            f"{context} must be a YAML mapping"
        )
    if any(not isinstance(key, str) for key in value):
        raise T21PathwayUniverseValidationError(
            f"{context} keys must all be strings"
        )
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], context: str
) -> None:
    observed = set(value)
    missing = sorted(expected.difference(observed))
    unexpected = sorted(observed.difference(expected))
    if missing or unexpected:
        raise T21PathwayUniverseValidationError(
            f"{context} keys differ: missing={missing}, unexpected={unexpected}"
        )


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise T21PathwayUniverseValidationError(
            f"{context} must be a non-empty string"
        )
    if value != value.strip():
        raise T21PathwayUniverseValidationError(
            f"{context} must not contain outer whitespace"
        )
    return value


def _identifier(value: Any, context: str) -> str:
    result = _text(value, context)
    if _IDENTIFIER.fullmatch(result) is None:
        raise T21PathwayUniverseValidationError(
            f"{context} is not a canonical identifier: {result!r}"
        )
    return result


def _boolean(value: Any, context: str) -> bool:
    if value is not True and value is not False:
        raise T21PathwayUniverseValidationError(
            f"{context} must be a YAML boolean"
        )
    return bool(value)


def _positive_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise T21PathwayUniverseValidationError(
            f"{context} must be a positive integer"
        )
    return value


def _sha(value: Any, context: str) -> str:
    result = _text(value, context)
    if _SHA256.fullmatch(result) is None:
        raise T21PathwayUniverseValidationError(
            f"{context} must be a 64-character SHA256"
        )
    return result.lower()


def _utc_timestamp(value: Any, context: str) -> str:
    result = _text(value, context)
    if not result.endswith("Z"):
        raise T21PathwayUniverseValidationError(
            f"{context} must be an ISO-8601 UTC timestamp ending in Z"
        )
    try:
        parsed = datetime.fromisoformat(result[:-1] + "+00:00")
    except ValueError as exc:
        raise T21PathwayUniverseValidationError(
            f"{context} must be an ISO-8601 UTC timestamp"
        ) from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise T21PathwayUniverseValidationError(f"{context} must be UTC")
    return result


def _string_list(value: Any, context: str, *, nonempty: bool = True) -> list[str]:
    if not isinstance(value, list):
        raise T21PathwayUniverseValidationError(f"{context} must be a YAML list")
    result = [_text(item, f"{context}[{index}]") for index, item in enumerate(value)]
    if nonempty and not result:
        raise T21PathwayUniverseValidationError(f"{context} must not be empty")
    if len(result) != len(set(result)):
        raise T21PathwayUniverseValidationError(
            f"{context} must not contain duplicate values"
        )
    return result


def _validate_spec(raw: Any) -> dict[str, Any]:
    spec = _mapping(raw, "specification")
    _exact_keys(spec, _ROOT_KEYS, "specification")
    if _text(spec["schema_name"], "schema_name") != "t21_pathway_universe_spec":
        raise T21PathwayUniverseValidationError(
            "schema_name must be t21_pathway_universe_spec"
        )
    if _text(spec["schema_version"], "schema_version") != "1.0.0":
        raise T21PathwayUniverseValidationError("schema_version must be 1.0.0")
    _identifier(spec["universe_id"], "universe_id")
    _utc_timestamp(spec["frozen_at_utc"], "frozen_at_utc")
    if _boolean(spec["outcome_blinded_at_freeze"], "outcome_blinded_at_freeze") is not True:
        raise T21PathwayUniverseValidationError(
            "outcome_blinded_at_freeze must be true"
        )
    if _boolean(
        spec["real_pathway_results_inspected"], "real_pathway_results_inspected"
    ) is not False:
        raise T21PathwayUniverseValidationError(
            "real_pathway_results_inspected must be false"
        )

    gene_reference = _mapping(spec["gene_reference"], "gene_reference")
    _exact_keys(gene_reference, _GENE_REFERENCE_KEYS, "gene_reference")
    for key in (
        "path",
        "gene_id_column",
        "gene_symbol_column",
        "chromosome_21_column",
        "genome_build",
    ):
        _text(gene_reference[key], f"gene_reference.{key}")
    _sha(gene_reference["sha256"], "gene_reference.sha256")

    sources = _mapping(spec["sources"], "sources")
    if not sources:
        raise T21PathwayUniverseValidationError("sources must not be empty")
    for source_id, source_value in sources.items():
        _identifier(source_id, f"sources key {source_id!r}")
        source = _mapping(source_value, f"sources.{source_id}")
        _exact_keys(source, _SOURCE_KEYS, f"sources.{source_id}")
        for key in (
            "collection",
            "provider",
            "url",
            "path",
        ):
            _text(source[key], f"sources.{source_id}.{key}")
        _utc_timestamp(
            source["retrieved_at_utc"],
            f"sources.{source_id}.retrieved_at_utc",
        )
        _sha(source["sha256"], f"sources.{source_id}.sha256")
        _positive_integer(
            source["expected_pathways"],
            f"sources.{source_id}.expected_pathways",
        )

    policy = _mapping(spec["mapping_policy"], "mapping_policy")
    _exact_keys(policy, _MAPPING_POLICY_KEYS, "mapping_policy")
    expected_policy_values = {
        "source_namespace": "HGNC_gene_symbol",
        "output_namespace": "Ensembl_gene_id",
        "absent_symbol_action": "exclude_and_record",
        "ambiguous_symbol_action": "require_explicit_resolution",
    }
    for key, expected in expected_policy_values.items():
        if _text(policy[key], f"mapping_policy.{key}") != expected:
            raise T21PathwayUniverseValidationError(
                f"mapping_policy.{key} must be {expected}"
            )
    minimum = _positive_integer(
        policy["minimum_mapped_genes"], "mapping_policy.minimum_mapped_genes"
    )
    maximum = _positive_integer(
        policy["maximum_mapped_genes"], "mapping_policy.maximum_mapped_genes"
    )
    if minimum > maximum:
        raise T21PathwayUniverseValidationError(
            "mapping_policy minimum_mapped_genes exceeds maximum_mapped_genes"
        )
    resolutions = _mapping(
        policy["ambiguous_symbol_resolutions"],
        "mapping_policy.ambiguous_symbol_resolutions",
    )
    for symbol, gene_id in resolutions.items():
        _text(symbol, "ambiguous resolution symbol")
        _text(gene_id, f"ambiguous resolution for {symbol}")
    _text(
        policy["ambiguity_resolution_rule"],
        "mapping_policy.ambiguity_resolution_rule",
    )

    pathways = _mapping(spec["pathways"], "pathways")
    if not pathways:
        raise T21PathwayUniverseValidationError("pathways must not be empty")
    for pathway_id, label in pathways.items():
        _identifier(pathway_id, f"pathway ID {pathway_id!r}")
        _text(label, f"pathways.{pathway_id}")
    labels = list(pathways.values())
    if len(labels) != len(set(labels)):
        raise T21PathwayUniverseValidationError(
            "pathways source labels must be unique"
        )
    expected_pathways = sum(
        source["expected_pathways"] for source in sources.values()
    )
    if len(pathways) != expected_pathways:
        raise T21PathwayUniverseValidationError(
            "pathways count must equal the sum of sources.expected_pathways; "
            f"expected {expected_pathways}, found {len(pathways)}"
        )

    level_1 = _mapping(spec["level_1"], "level_1")
    _exact_keys(level_1, _LEVEL_1_KEYS, "level_1")
    _text(level_1["role"], "level_1.role")
    if _boolean(level_1["formal_inference"], "level_1.formal_inference") is not True:
        raise T21PathwayUniverseValidationError(
            "Level 1 formal_inference must be true"
        )
    if (
        _text(level_1["multiple_testing"], "level_1.multiple_testing")
        != "whole_donor_permutation_family_maxT"
    ):
        raise T21PathwayUniverseValidationError(
            "Level 1 multiple_testing must be whole_donor_permutation_family_maxT"
        )
    families = _mapping(level_1["families"], "level_1.families")
    if not 12 <= len(families) <= 20:
        raise T21PathwayUniverseValidationError(
            f"Level 1 must declare 12-20 families; found {len(families)}"
        )
    assigned: dict[str, str] = {}
    for family_id, family_value in families.items():
        _identifier(family_id, f"level_1 family ID {family_id!r}")
        family = _mapping(family_value, f"level_1.families.{family_id}")
        _exact_keys(family, _FAMILY_KEYS, f"level_1.families.{family_id}")
        label = _text(family["label"], f"level_1.families.{family_id}.label")
        family_pathways = _string_list(
            family["pathways"], f"level_1.families.{family_id}.pathways"
        )
        unknown = sorted(set(family_pathways).difference(pathways))
        if unknown:
            raise T21PathwayUniverseValidationError(
                f"Level 1 family {family_id} references unknown pathways: {unknown}"
            )
        for pathway_id in family_pathways:
            if pathway_id in assigned:
                raise T21PathwayUniverseValidationError(
                    "Level 1 families must be disjoint; "
                    f"{pathway_id} occurs in {assigned[pathway_id]} and {family_id}"
                )
            assigned[pathway_id] = family_id
        limit = _text(
            family["interpretation_limit"],
            f"level_1.families.{family_id}.interpretation_limit",
        )
        proxy = "proxy" in family_id.lower() or "proxy" in label.lower()
        if proxy and limit.strip().lower() in _EMPTY_LIMITS:
            raise T21PathwayUniverseValidationError(
                f"Proxy family {family_id} requires a substantive interpretation_limit"
            )

    level_2 = _mapping(spec["level_2"], "level_2")
    _exact_keys(level_2, _LEVEL_2_KEYS, "level_2")
    _text(level_2["role"], "level_2.role")
    if _boolean(level_2["formal_inference"], "level_2.formal_inference") is not True:
        raise T21PathwayUniverseValidationError(
            "Level 2 formal_inference must be true"
        )
    if (
        _text(level_2["multiple_testing"], "level_2.multiple_testing")
        != "Benjamini_Yekutieli"
    ):
        raise T21PathwayUniverseValidationError(
            "Level 2 multiple_testing must be Benjamini_Yekutieli"
        )
    if (
        _text(level_2["pathways"], "level_2.pathways")
        != "all_declared_hallmark_pathways"
    ):
        raise T21PathwayUniverseValidationError(
            "Level 2 must include all_declared_hallmark_pathways"
        )

    level_3 = _mapping(spec["level_3"], "level_3")
    _exact_keys(level_3, _LEVEL_3_KEYS, "level_3")
    if (
        _text(level_3["role"], "level_3.role")
        != "interpretation_only_GO_or_Reactome"
    ):
        raise T21PathwayUniverseValidationError(
            "Level 3 role must be interpretation_only_GO_or_Reactome"
        )
    if _boolean(level_3["formal_inference"], "level_3.formal_inference") is not False:
        raise T21PathwayUniverseValidationError(
            "Level 3 formal_inference must be false (fail closed)"
        )
    if (
        _text(level_3["multiple_testing"], "level_3.multiple_testing")
        != "not_applicable"
    ):
        raise T21PathwayUniverseValidationError(
            "Level 3 multiple_testing must be not_applicable"
        )
    _string_list(level_3["allowed_uses"], "level_3.allowed_uses")
    prohibited = _string_list(
        level_3["prohibited_uses"], "level_3.prohibited_uses"
    )
    required_prohibited = {"primary_discovery_claim", "unreplicated_formal_event_claim"}
    if not required_prohibited.issubset(prohibited):
        raise T21PathwayUniverseValidationError(
            "Level 3 prohibited_uses must block primary and unreplicated formal claims"
        )

    _string_list(spec["coverage_gaps"], "coverage_gaps")
    _text(spec["claim_rule"], "claim_rule")
    return spec


def _read_spec(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise T21PathwayUniverseValidationError(
            f"Could not read strict pathway-universe YAML {path}: {exc}"
        ) from exc
    return _validate_spec(raw)


def _resolve_repo_file(repo_root: Path, raw_path: Any, context: str) -> Path:
    path_text = _text(raw_path, context)
    relative = Path(path_text)
    if relative.is_absolute():
        raise T21PathwayUniverseValidationError(
            f"{context} must be repository-relative"
        )
    root = repo_root.resolve()
    try:
        resolved = (root / relative).resolve(strict=True)
    except OSError as exc:
        raise T21PathwayUniverseValidationError(
            f"{context} does not resolve to a readable repository file: {path_text}"
        ) from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise T21PathwayUniverseValidationError(
            f"{context} escapes the repository root: {path_text}"
        ) from exc
    if not resolved.is_file():
        raise T21PathwayUniverseValidationError(
            f"{context} is not a regular file: {path_text}"
        )
    return resolved


def _verify_hash(path: Path, expected: Any, context: str) -> str:
    expected_hash = _sha(expected, f"{context}.sha256")
    observed = _sha256_file(path)
    if observed != expected_hash:
        raise T21PathwayUniverseValidationError(
            f"{context} SHA256 mismatch: expected {expected_hash}, observed {observed}"
        )
    return observed


def _read_gene_reference(
    path: Path, config: Mapping[str, Any]
) -> tuple[dict[str, list[tuple[str, bool]]], int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as exc:
            raise T21PathwayUniverseValidationError(
                "gene_reference is empty"
            ) from exc
    if len(header) != len(set(header)):
        raise T21PathwayUniverseValidationError(
            "gene_reference contains duplicate column names"
        )
    required = {
        str(config["gene_id_column"]),
        str(config["gene_symbol_column"]),
        str(config["chromosome_21_column"]),
    }
    missing = sorted(required.difference(header))
    if missing:
        raise T21PathwayUniverseValidationError(
            f"gene_reference is missing configured columns: {missing}"
        )
    try:
        frame = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            na_filter=False,
        )
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise T21PathwayUniverseValidationError(
            f"Could not parse gene_reference TSV: {exc}"
        ) from exc
    if frame.empty:
        raise T21PathwayUniverseValidationError("gene_reference has no gene rows")
    gene_id_column = str(config["gene_id_column"])
    symbol_column = str(config["gene_symbol_column"])
    chr21_column = str(config["chromosome_21_column"])
    for column in (gene_id_column, symbol_column, chr21_column):
        values = frame[column].astype(str)
        if values.str.strip().eq("").any() or values.ne(values.str.strip()).any():
            raise T21PathwayUniverseValidationError(
                f"gene_reference.{column} values must be non-empty without outer whitespace"
            )
    if frame[gene_id_column].duplicated().any():
        duplicates = sorted(
            frame.loc[frame[gene_id_column].duplicated(False), gene_id_column]
            .astype(str)
            .unique()
            .tolist()
        )
        raise T21PathwayUniverseValidationError(
            f"gene_reference gene IDs must be unique: {duplicates[:5]}"
        )
    chr_values = frame[chr21_column].astype(str).str.lower()
    invalid_chr = sorted(set(chr_values).difference({"true", "false"}))
    if invalid_chr:
        raise T21PathwayUniverseValidationError(
            f"gene_reference.{chr21_column} must contain only true/false: {invalid_chr}"
        )
    symbol_index: dict[str, list[tuple[str, bool]]] = {}
    for gene_id, symbol, is_chr21 in zip(
        frame[gene_id_column].astype(str),
        frame[symbol_column].astype(str),
        chr_values.eq("true"),
    ):
        symbol_index.setdefault(symbol, []).append((gene_id, bool(is_chr21)))
    for records in symbol_index.values():
        records.sort(key=lambda item: item[0])
    return symbol_index, len(frame)


def _read_gmt(
    path: Path, *, source_id: str, expected_pathways: int
) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise T21PathwayUniverseValidationError(
            f"Could not parse GMT source {source_id}: {exc}"
        ) from exc
    if not lines:
        raise T21PathwayUniverseValidationError(
            f"GMT source {source_id} is empty"
        )
    pathways = []
    seen_labels: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise T21PathwayUniverseValidationError(
                f"GMT source {source_id} contains a blank line at {line_number}"
            )
        fields = line.split("\t")
        if len(fields) < 3:
            raise T21PathwayUniverseValidationError(
                f"GMT source {source_id} line {line_number} has no members"
            )
        label = _text(fields[0], f"GMT {source_id} line {line_number} label")
        if label in seen_labels:
            raise T21PathwayUniverseValidationError(
                f"GMT source {source_id} has duplicate pathway label {label!r}"
            )
        seen_labels.add(label)
        members = [
            _text(symbol, f"GMT {source_id} {label} member")
            for symbol in fields[2:]
        ]
        if len(members) != len(set(members)):
            duplicates = sorted(
                symbol for symbol in set(members) if members.count(symbol) > 1
            )
            raise T21PathwayUniverseValidationError(
                f"GMT pathway {label!r} has duplicate members: {duplicates[:5]}"
            )
        pathways.append(
            {
                "source_id": source_id,
                "source_line": line_number,
                "label": label,
                "members": members,
            }
        )
    if len(pathways) != expected_pathways:
        raise T21PathwayUniverseValidationError(
            f"GMT source {source_id} must contain exactly {expected_pathways} "
            f"pathways; found {len(pathways)}"
        )
    return pathways


def _path_text(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _source_inputs_hash(source_records: Mapping[str, Mapping[str, Any]]) -> str:
    return _logical_hash(
        [
            {
                "source_id": source_id,
                "sha256": source_records[source_id]["sha256"],
            }
            for source_id in sorted(source_records)
        ]
    )


def build_t21_pathway_universe(
    spec_path: Union[os.PathLike[str], str],
    *,
    repo_root: Optional[Union[os.PathLike[str], str]] = None,
) -> T21PathwayUniverseResult:
    """Validate a frozen spec and construct its normalized pathway universe."""

    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    if not root.is_dir():
        raise T21PathwayUniverseValidationError(
            f"Repository root is not a directory: {root}"
        )
    specification_path = Path(spec_path).resolve()
    if not specification_path.is_file():
        raise T21PathwayUniverseValidationError(
            f"Specification is not a file: {specification_path}"
        )
    spec = _read_spec(specification_path)
    spec_sha256 = _sha256_file(specification_path)

    gene_config = spec["gene_reference"]
    gene_path = _resolve_repo_file(root, gene_config["path"], "gene_reference.path")
    gene_reference_sha256 = _verify_hash(
        gene_path, gene_config["sha256"], "gene_reference"
    )
    symbol_index, n_reference_genes = _read_gene_reference(gene_path, gene_config)

    source_records: dict[str, dict[str, Any]] = {}
    raw_pathways: list[dict[str, Any]] = []
    observed_labels: set[str] = set()
    for source_id in sorted(spec["sources"]):
        source = spec["sources"][source_id]
        source_path = _resolve_repo_file(
            root, source["path"], f"sources.{source_id}.path"
        )
        source_sha256 = _verify_hash(
            source_path, source["sha256"], f"sources.{source_id}"
        )
        parsed = _read_gmt(
            source_path,
            source_id=source_id,
            expected_pathways=int(source["expected_pathways"]),
        )
        repeated = sorted(
            observed_labels.intersection(item["label"] for item in parsed)
        )
        if repeated:
            raise T21PathwayUniverseValidationError(
                f"GMT pathway labels must be globally unique across sources: {repeated}"
            )
        observed_labels.update(item["label"] for item in parsed)
        raw_pathways.extend(parsed)
        source_records[source_id] = {
            "collection": str(source["collection"]),
            "path": _path_text(source_path, root),
            "sha256": source_sha256,
            "expected_pathways": int(source["expected_pathways"]),
            "observed_pathways": len(parsed),
        }

    label_to_id = {label: pathway_id for pathway_id, label in spec["pathways"].items()}
    expected_labels = set(label_to_id)
    missing_labels = sorted(expected_labels.difference(observed_labels))
    unexpected_labels = sorted(observed_labels.difference(expected_labels))
    if missing_labels or unexpected_labels:
        raise T21PathwayUniverseValidationError(
            "Declared pathways differ from GMT labels: "
            f"missing={missing_labels}, unexpected={unexpected_labels}"
        )

    policy = spec["mapping_policy"]
    resolutions = {
        str(symbol): str(gene_id)
        for symbol, gene_id in policy["ambiguous_symbol_resolutions"].items()
    }
    source_inputs_sha256 = _source_inputs_hash(source_records)
    audit_semantic_rows: list[dict[str, Any]] = []
    mapped_by_pathway: dict[str, list[dict[str, Any]]] = {}
    pathway_source: dict[str, str] = {}
    used_ambiguous_symbols: set[str] = set()
    for raw_pathway in raw_pathways:
        source_id = str(raw_pathway["source_id"])
        pathway_label = str(raw_pathway["label"])
        pathway_id = label_to_id[pathway_label]
        pathway_source[pathway_id] = source_id
        included_rows: list[dict[str, Any]] = []
        for member_order, symbol in enumerate(raw_pathway["members"], start=1):
            candidates = symbol_index.get(symbol, [])
            if not candidates:
                audit_semantic_rows.append(
                    {
                        "pathway_id": pathway_id,
                        "pathway_label": pathway_label,
                        "source_id": source_id,
                        "source_member_order": member_order,
                        "source_symbol": symbol,
                        "gene_id": "",
                        "is_chr21": None,
                        "included": False,
                        "mapping_status": "excluded_absent_symbol",
                        "detail": "symbol_absent_from_gene_reference",
                    }
                )
                continue
            if len(candidates) == 1:
                gene_id, is_chr21 = candidates[0]
                status = "mapped_unique_symbol"
                detail = "unique_symbol_match"
            else:
                used_ambiguous_symbols.add(symbol)
                if symbol not in resolutions:
                    candidate_ids = [item[0] for item in candidates]
                    raise T21PathwayUniverseValidationError(
                        f"Ambiguous symbol {symbol!r} requires an exact configured "
                        f"resolution among {candidate_ids}"
                    )
                resolved_id = resolutions[symbol]
                selected = [item for item in candidates if item[0] == resolved_id]
                if len(selected) != 1:
                    candidate_ids = [item[0] for item in candidates]
                    raise T21PathwayUniverseValidationError(
                        f"Ambiguous resolution {symbol}->{resolved_id} is not exactly "
                        f"one of the gene-reference candidates {candidate_ids}"
                    )
                gene_id, is_chr21 = selected[0]
                status = "mapped_ambiguous_explicit_resolution"
                detail = f"explicit_resolution:{resolved_id}"
            row = {
                "pathway_id": pathway_id,
                "pathway_label": pathway_label,
                "source_id": source_id,
                "source_member_order": member_order,
                "source_symbol": symbol,
                "gene_id": gene_id,
                "is_chr21": bool(is_chr21),
                "included": True,
                "mapping_status": status,
                "detail": detail,
            }
            audit_semantic_rows.append(row.copy())
            included_rows.append(row)
        duplicate_gene_ids = sorted(
            gene_id
            for gene_id in {row["gene_id"] for row in included_rows}
            if sum(row["gene_id"] == gene_id for row in included_rows) > 1
        )
        if duplicate_gene_ids:
            raise T21PathwayUniverseValidationError(
                f"Mapped pathway {pathway_id} contains duplicate gene IDs: "
                f"{duplicate_gene_ids[:5]}"
            )
        mapped_count = len(included_rows)
        minimum = int(policy["minimum_mapped_genes"])
        maximum = int(policy["maximum_mapped_genes"])
        if not minimum <= mapped_count <= maximum:
            raise T21PathwayUniverseValidationError(
                f"Mapped pathway {pathway_id} has {mapped_count} genes outside "
                f"the configured [{minimum}, {maximum}] range"
            )
        mapped_by_pathway[pathway_id] = sorted(
            included_rows, key=lambda row: (row["gene_id"], row["source_symbol"])
        )

    unused_resolutions = sorted(set(resolutions).difference(used_ambiguous_symbols))
    if unused_resolutions:
        raise T21PathwayUniverseValidationError(
            "Configured ambiguous_symbol_resolutions must exactly match ambiguous "
            f"symbols used by the GMT inputs; unused={unused_resolutions}"
        )

    families_config = spec["level_1"]["families"]
    family_lookup: dict[str, str] = {}
    family_semantic_rows = []
    for family_id in sorted(families_config):
        family = families_config[family_id]
        pathway_ids = sorted(str(item) for item in family["pathways"])
        for pathway_id in pathway_ids:
            family_lookup[pathway_id] = family_id
        family_semantic_rows.append(
            {
                "family_id": family_id,
                "family_label": str(family["label"]),
                "pathways": pathway_ids,
                "role": str(spec["level_1"]["role"]),
                "formal_inference": True,
                "multiple_testing": str(spec["level_1"]["multiple_testing"]),
                "interpretation_limit": str(family["interpretation_limit"]),
            }
        )

    gene_sets = {
        pathway_id: tuple(row["gene_id"] for row in mapped_by_pathway[pathway_id])
        for pathway_id in sorted(mapped_by_pathway)
    }
    pathway_hashes = {
        pathway_id: _logical_hash(
            {
                "pathway_id": pathway_id,
                "pathway_label": str(spec["pathways"][pathway_id]),
                "gene_ids": list(gene_sets[pathway_id]),
            }
        )
        for pathway_id in sorted(gene_sets)
    }
    membership_semantic = [
        {
            "pathway_id": pathway_id,
            "gene_id": row["gene_id"],
            "source_symbol": row["source_symbol"],
            "is_chr21": row["is_chr21"],
            "mapping_status": row["mapping_status"],
        }
        for pathway_id in sorted(mapped_by_pathway)
        for row in mapped_by_pathway[pathway_id]
    ]
    audit_for_hash = sorted(
        (
            {
                key: row[key]
                for key in (
                    "pathway_id",
                    "source_symbol",
                    "gene_id",
                    "is_chr21",
                    "included",
                    "mapping_status",
                    "detail",
                )
            }
            for row in audit_semantic_rows
        ),
        key=lambda row: (
            row["pathway_id"],
            row["source_symbol"],
            row["gene_id"],
            row["mapping_status"],
        ),
    )
    logical_payload = {
        "schema_name": "t21_pathway_universe_logical_content",
        "schema_version": "1.0.0",
        "universe_id": spec["universe_id"],
        "gene_sets": [
            {
                "pathway_id": pathway_id,
                "pathway_label": spec["pathways"][pathway_id],
                "gene_ids": list(gene_sets[pathway_id]),
            }
            for pathway_id in sorted(gene_sets)
        ],
        "mapping_audit": audit_for_hash,
        "level_1_families": family_semantic_rows,
        "level_2": {
            "pathways": sorted(gene_sets),
            "role": spec["level_2"]["role"],
            "formal_inference": True,
            "multiple_testing": "Benjamini_Yekutieli",
        },
        "level_3": {
            "role": spec["level_3"]["role"],
            "formal_inference": False,
            "multiple_testing": spec["level_3"]["multiple_testing"],
            "allowed_uses": sorted(spec["level_3"]["allowed_uses"]),
            "prohibited_uses": sorted(spec["level_3"]["prohibited_uses"]),
        },
    }
    logical_hashes = {
        "gene_sets_sha256": _logical_hash(
            {pathway_id: list(gene_sets[pathway_id]) for pathway_id in sorted(gene_sets)}
        ),
        "membership_sha256": _logical_hash(membership_semantic),
        "families_sha256": _logical_hash(family_semantic_rows),
        "mapping_audit_sha256": _logical_hash(audit_for_hash),
        "pathway_universe_sha256": _logical_hash(logical_payload),
    }
    universe_hash = logical_hashes["pathway_universe_sha256"]
    universe_id = str(spec["universe_id"])

    membership_rows = []
    for pathway_id in sorted(mapped_by_pathway):
        source_id = pathway_source[pathway_id]
        source_record = source_records[source_id]
        for gene_order, row in enumerate(mapped_by_pathway[pathway_id], start=1):
            membership_rows.append(
                {
                    "universe_id": universe_id,
                    "pathway_id": pathway_id,
                    "pathway_label": str(spec["pathways"][pathway_id]),
                    "source_id": source_id,
                    "source_collection": source_record["collection"],
                    "gene_order": gene_order,
                    "source_symbol": row["source_symbol"],
                    "gene_id": row["gene_id"],
                    "is_chr21": row["is_chr21"],
                    "mapping_status": row["mapping_status"],
                    "level_1_family_id": family_lookup.get(pathway_id, ""),
                    "level_2_included": True,
                    "level_2_multiple_testing": "Benjamini_Yekutieli",
                    "outcome_blinded": True,
                    "spec_sha256": spec_sha256,
                    "gene_reference_sha256": gene_reference_sha256,
                    "source_gmt_sha256": source_record["sha256"],
                    "source_inputs_sha256": source_inputs_sha256,
                    "pathway_logical_sha256": pathway_hashes[pathway_id],
                    "pathway_universe_logical_sha256": universe_hash,
                }
            )
    membership = pd.DataFrame(membership_rows, columns=MEMBERSHIP_COLUMNS)

    family_rows = [
        {
            "universe_id": universe_id,
            "analysis_level": "level_1",
            "family_id": row["family_id"],
            "family_label": row["family_label"],
            "pathways_json": _stable_json(row["pathways"]),
            "n_pathways": len(row["pathways"]),
            "role": row["role"],
            "formal_inference": True,
            "multiple_testing": row["multiple_testing"],
            "interpretation_limit": row["interpretation_limit"],
            "outcome_blinded": True,
            "spec_sha256": spec_sha256,
            "gene_reference_sha256": gene_reference_sha256,
            "source_inputs_sha256": source_inputs_sha256,
            "pathway_universe_logical_sha256": universe_hash,
        }
        for row in family_semantic_rows
    ]
    families = pd.DataFrame(family_rows, columns=FAMILY_COLUMNS)

    audit_rows = []
    for row in sorted(
        audit_semantic_rows,
        key=lambda item: (item["pathway_id"], item["source_member_order"]),
    ):
        source_record = source_records[row["source_id"]]
        audit_rows.append(
            {
                "universe_id": universe_id,
                "pathway_id": row["pathway_id"],
                "pathway_label": row["pathway_label"],
                "source_id": row["source_id"],
                "source_member_order": row["source_member_order"],
                "source_symbol": row["source_symbol"],
                "gene_id": row["gene_id"],
                "is_chr21": row["is_chr21"],
                "included": row["included"],
                "mapping_status": row["mapping_status"],
                "detail": row["detail"],
                "outcome_blinded": True,
                "spec_sha256": spec_sha256,
                "gene_reference_sha256": gene_reference_sha256,
                "source_gmt_sha256": source_record["sha256"],
                "source_inputs_sha256": source_inputs_sha256,
                "pathway_universe_logical_sha256": universe_hash,
            }
        )
    mapping_audit = pd.DataFrame(audit_rows, columns=MAPPING_AUDIT_COLUMNS)

    n_absent = int(mapping_audit["included"].eq(False).sum())
    n_ambiguous = int(
        mapping_audit["mapping_status"]
        .eq("mapped_ambiguous_explicit_resolution")
        .sum()
    )
    metadata = {
        "schema_name": "t21_pathway_universe_result",
        "schema_version": "1.0.0",
        "universe_id": universe_id,
        "frozen_at_utc": str(spec["frozen_at_utc"]),
        "outcome_blinded": True,
        "real_pathway_results_inspected": False,
        "spec": {
            "path": _path_text(specification_path, root),
            "sha256": spec_sha256,
        },
        "gene_reference": {
            "path": _path_text(gene_path, root),
            "sha256": gene_reference_sha256,
            "genome_build": str(gene_config["genome_build"]),
            "n_genes": n_reference_genes,
        },
        "sources": source_records,
        "source_inputs_sha256": source_inputs_sha256,
        "n_pathways": len(gene_sets),
        "n_memberships": len(membership),
        "n_mapping_audit_rows": len(mapping_audit),
        "n_absent_symbol_exclusions": n_absent,
        "n_ambiguous_symbol_memberships_resolved": n_ambiguous,
        "ambiguous_symbols_resolved": sorted(used_ambiguous_symbols),
        "n_level_1_families": len(families),
        "level_1": {
            "formal_inference": True,
            "multiple_testing": "whole_donor_permutation_family_maxT",
            "family_pathways_disjoint": True,
        },
        "level_2": {
            "formal_inference": True,
            "multiple_testing": "Benjamini_Yekutieli",
            "n_pathways": len(gene_sets),
            "pathway_ids": sorted(gene_sets),
        },
        "level_3": {
            "role": "interpretation_only_GO_or_Reactome",
            "formal_inference": False,
            "multiple_testing": "not_applicable",
            "allowed_uses": list(spec["level_3"]["allowed_uses"]),
            "prohibited_uses": list(spec["level_3"]["prohibited_uses"]),
        },
        "mapping_policy": {
            "source_namespace": "HGNC_gene_symbol",
            "output_namespace": "Ensembl_gene_id",
            "absent_symbol_action": "exclude_and_record",
            "ambiguous_symbol_action": "require_explicit_resolution",
            "minimum_mapped_genes": int(policy["minimum_mapped_genes"]),
            "maximum_mapped_genes": int(policy["maximum_mapped_genes"]),
            "ambiguity_resolution_rule": str(policy["ambiguity_resolution_rule"]),
        },
        "logical_hashes": logical_hashes,
        "output_columns": {
            UNIVERSE_TSV_NAME: list(MEMBERSHIP_COLUMNS),
            FAMILIES_TSV_NAME: list(FAMILY_COLUMNS),
            MAPPING_AUDIT_TSV_NAME: list(MAPPING_AUDIT_COLUMNS),
        },
    }
    for table in (membership, families, mapping_audit):
        table.attrs["t21_pathway_universe"] = metadata.copy()
    return T21PathwayUniverseResult(
        gene_sets=MappingProxyType(gene_sets),
        membership=membership,
        families=families,
        mapping_audit=mapping_audit,
        metadata=MappingProxyType(metadata),
    )


def _format_tsv_value(value: Any) -> str:
    if value is None or value is pd.NA or bool(pd.isna(value)):
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    return str(value)


def _frame_to_tsv_bytes(frame: pd.DataFrame, columns: Sequence[str]) -> bytes:
    if tuple(frame.columns) != tuple(columns):
        raise T21PathwayUniverseValidationError(
            "Internal canonical TSV column order changed"
        )
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(columns)
    for row in frame.itertuples(index=False, name=None):
        writer.writerow([_format_tsv_value(value) for value in row])
    return buffer.getvalue().encode("utf-8")


def render_t21_pathway_universe_outputs(
    result: T21PathwayUniverseResult,
) -> dict[str, bytes]:
    """Render the three canonical TSVs and deterministic build record."""

    payloads = {
        UNIVERSE_TSV_NAME: _frame_to_tsv_bytes(
            result.membership, MEMBERSHIP_COLUMNS
        ),
        FAMILIES_TSV_NAME: _frame_to_tsv_bytes(result.families, FAMILY_COLUMNS),
        MAPPING_AUDIT_TSV_NAME: _frame_to_tsv_bytes(
            result.mapping_audit, MAPPING_AUDIT_COLUMNS
        ),
    }
    metadata = dict(result.metadata)
    output_records = {}
    frames = {
        UNIVERSE_TSV_NAME: result.membership,
        FAMILIES_TSV_NAME: result.families,
        MAPPING_AUDIT_TSV_NAME: result.mapping_audit,
    }
    for filename in (
        UNIVERSE_TSV_NAME,
        FAMILIES_TSV_NAME,
        MAPPING_AUDIT_TSV_NAME,
    ):
        content = payloads[filename]
        output_records[filename] = {
            "filename": filename,
            "bytes": len(content),
            "sha256": _sha256_bytes(content),
            "rows": len(frames[filename]),
            "columns": list(frames[filename].columns),
            "canonical_tsv": True,
            "outcome_blinded": True,
            "pathway_universe_logical_sha256": metadata["logical_hashes"][
                "pathway_universe_sha256"
            ],
        }
    module_path = Path(__file__).resolve()
    record = {
        "schema_name": "t21_pathway_universe_build_record",
        "schema_version": "1.0.0",
        "universe_id": metadata["universe_id"],
        "frozen_at_utc": metadata["frozen_at_utc"],
        "outcome_blinded": True,
        "real_pathway_results_inspected": False,
        "deterministic_rebuild": True,
        "builder": {
            "module": "pyfgsea.t21_pathway_universe",
            "sha256": _sha256_file(module_path),
        },
        "inputs": {
            "spec": metadata["spec"],
            "gene_reference": metadata["gene_reference"],
            "sources": metadata["sources"],
            "source_inputs_sha256": metadata["source_inputs_sha256"],
        },
        "validation": {
            "strict_yaml": True,
            "paths_repository_confined": True,
            "input_byte_hashes_verified": True,
            "gmt_pathways_unique": True,
            "gmt_members_unique_within_pathway": True,
            "ambiguous_symbols_exactly_resolved": True,
            "missing_symbols_excluded_and_recorded": True,
            "level_1_family_count_12_to_20": True,
            "level_1_families_disjoint": True,
            "level_1_formal_maxT": True,
            "level_2_all_declared_pathways_with_BY": True,
            "level_3_interpretation_only": True,
            "proxy_interpretation_limits_present": True,
        },
        "counts": {
            "pathways": metadata["n_pathways"],
            "memberships": metadata["n_memberships"],
            "level_1_families": metadata["n_level_1_families"],
            "mapping_audit_rows": metadata["n_mapping_audit_rows"],
            "absent_symbol_exclusions": metadata["n_absent_symbol_exclusions"],
            "ambiguous_symbol_memberships_resolved": metadata[
                "n_ambiguous_symbol_memberships_resolved"
            ],
        },
        "logical_hashes": metadata["logical_hashes"],
        "outputs": output_records,
        "build_record_filename": BUILD_RECORD_JSON_NAME,
    }
    payloads[BUILD_RECORD_JSON_NAME] = (
        _stable_json(record, pretty=True) + "\n"
    ).encode("utf-8")
    return payloads


def _stage_bytes(path: Path, content: bytes) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _replace_file(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def _publish_atomic_payloads(output_dir: Path, payloads: Mapping[str, bytes]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if set(payloads) != set(OUTPUT_FILE_NAMES):
        raise T21PathwayUniverseValidationError(
            "Atomic publication requires exactly the four fixed output files"
        )
    targets = {name: output_dir / name for name in OUTPUT_FILE_NAMES}
    for target in targets.values():
        if target.exists() and not target.is_file():
            raise T21PathwayUniverseValidationError(
                f"Output target exists but is not a file: {target}"
            )
    staged: dict[str, Path] = {}
    try:
        for filename in OUTPUT_FILE_NAMES:
            staged[filename] = _stage_bytes(targets[filename], payloads[filename])
    except Exception:
        for path in staged.values():
            path.unlink(missing_ok=True)
        raise

    token = uuid4().hex
    backups: dict[str, Path] = {}
    published: list[str] = []
    safe_to_delete_backups = False
    try:
        for filename in OUTPUT_FILE_NAMES:
            target = targets[filename]
            if target.exists():
                backup = target.with_name(f".{target.name}.{token}.bak")
                _replace_file(target, backup)
                backups[filename] = backup
        for filename in OUTPUT_FILE_NAMES:
            _replace_file(staged[filename], targets[filename])
            published.append(filename)
        safe_to_delete_backups = True
    except Exception as publication_error:
        rollback_errors = []
        for filename in reversed(published):
            try:
                targets[filename].unlink(missing_ok=True)
            except OSError as exc:
                rollback_errors.append(str(exc))
        for filename in reversed(OUTPUT_FILE_NAMES):
            backup = backups.get(filename)
            if backup is not None and backup.exists():
                try:
                    _replace_file(backup, targets[filename])
                except OSError as exc:
                    rollback_errors.append(str(exc))
        if rollback_errors:
            raise RuntimeError(
                "Atomic pathway-universe publication and rollback both failed: "
                + "; ".join(rollback_errors)
            ) from publication_error
        safe_to_delete_backups = True
        raise
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)
        if safe_to_delete_backups:
            for path in backups.values():
                path.unlink(missing_ok=True)


def publish_t21_pathway_universe(
    spec_path: Union[os.PathLike[str], str],
    output_dir: Union[os.PathLike[str], str],
    *,
    repo_root: Optional[Union[os.PathLike[str], str]] = None,
) -> tuple[T21PathwayUniverseResult, dict[str, Any]]:
    """Build and atomically publish all four fixed pathway-universe artifacts."""

    result = build_t21_pathway_universe(spec_path, repo_root=repo_root)
    payloads = render_t21_pathway_universe_outputs(result)
    destination = Path(output_dir).resolve()
    _publish_atomic_payloads(destination, payloads)
    for filename, expected in payloads.items():
        observed = (destination / filename).read_bytes()
        if observed != expected:
            raise RuntimeError(
                f"Published pathway-universe output differs from staged bytes: {filename}"
            )
    record = json.loads(payloads[BUILD_RECORD_JSON_NAME].decode("utf-8"))
    return result, record


def validate_t21_pathway_universe_outputs(
    spec_path: Union[os.PathLike[str], str],
    output_dir: Union[os.PathLike[str], str],
    *,
    repo_root: Optional[Union[os.PathLike[str], str]] = None,
) -> T21PathwayUniverseResult:
    """Rebuild in memory and require byte-identical existing outputs."""

    result = build_t21_pathway_universe(spec_path, repo_root=repo_root)
    expected = render_t21_pathway_universe_outputs(result)
    destination = Path(output_dir).resolve()
    for filename in OUTPUT_FILE_NAMES:
        path = destination / filename
        if not path.is_file():
            raise T21PathwayUniverseValidationError(
                f"Rebuild comparison output is missing: {path}"
            )
        observed = path.read_bytes()
        if observed != expected[filename]:
            raise T21PathwayUniverseValidationError(
                f"Rebuild comparison failed for {filename}: expected "
                f"{_sha256_bytes(expected[filename])}, observed {_sha256_bytes(observed)}"
            )
    return result


__all__ = [
    "BUILD_RECORD_JSON_NAME",
    "FAMILIES_TSV_NAME",
    "MAPPING_AUDIT_TSV_NAME",
    "OUTPUT_FILE_NAMES",
    "T21PathwayUniverseResult",
    "T21PathwayUniverseValidationError",
    "UNIVERSE_TSV_NAME",
    "build_t21_pathway_universe",
    "publish_t21_pathway_universe",
    "render_t21_pathway_universe_outputs",
    "validate_t21_pathway_universe_outputs",
]
