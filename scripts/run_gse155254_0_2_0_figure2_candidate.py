"""Generate a fail-closed, artifact-bound Figure 2 recalculation receipt.

The historical record does not uniquely identify the Figure 2 step size and
NES-permutation count.  The project author accepted the explicit 500-cell,
50-cell-step, 2000-permutation contract on 2026-09-01.  This runner binds that
contract to a clean annotated release candidate and to the exact installed
wheel/native core recorded by ``verify_pyfgsea_artifacts.py``.  It does not
mark the manuscript or publication as accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "repro" / "figure2_gse155254" / "author_parameter_contract.json"
REFERENCE_MANIFEST_PATH = REPO_ROOT / "reference_manifest.json"
EXPECTED_VERSION = "0.2.0rc3"
EXPECTED_CARGO_VERSION = "0.2.0-rc3"
EXPECTED_ALGORITHM_REVISION = "fgsea-1.38-pr178-v1"
EXPECTED_FGSEA_REFERENCE = "1.38.0"
EXPECTED_DATASET_SHA256 = "9d9d1db60fe06037c5bfcf1a6ce06adfa74fe6ef715d910ef8b7e004d05cd21e"
EXPECTED_GENE_SETS_SHA256 = "92203149acdfa1e7d583fad5da99487244ce353b4d10d3193cd64329e334da66"
EXPECTED_DATASET_SHAPE = (3576, 3000)
EXPECTED_N_WINDOWS = 62
EXPECTED_N_PATHWAYS = 43
EXPECTED_N_ROWS = EXPECTED_N_WINDOWS * EXPECTED_N_PATHWAYS
BH_ATOL = 1e-14
RELEASE_TAG_PATTERN = re.compile(r"v0\.2\.0-rc3")

PARAMETERS: dict[str, Any] = {
    "pseudotime_key": "dpt_pseudotime",
    "window_size": 500,
    "step": 50,
    "min_size": 15,
    "max_size": 500,
    "sample_size": 101,
    "seed": 42,
    "eps": 1e-50,
    "nperm_nes": 2000,
    "nperm_simple": 1000,
    "gsea_param": 1.0,
    "mode": "aligned",
    "score_type": "std",
    "tie_policy": "gene_id",
    "bin_width": 0,
    "calculate_nes": True,
    "use_nes_cache": False,
    "max_levels": None,
}
ACCEPTED_VALUES = {
    "window_size": 500,
    "step": 50,
    "nperm_nes": 2000,
    "score_type": "std",
    "bin_width": 0,
    "use_nes_cache": False,
}
TARGET_PATHWAYS = ("heme Metabolism", "E2F Targets")
RESEARCH_CONTEXT = {
    "research_mode": "in-silico computational analysis",
    "input_provenance": (
        "processed single-cell data derived from public accession GSE155254; "
        "the exact H5AD and Hallmark GMT bytes are enforced by SHA-256"
    ),
    "computational_operation": "rolling-window trajectory GSEA recalculation",
    "intended_artifact": "parameter-bound Figure 2 tables, plots, and run receipt",
    "claim_boundary": "descriptive numerical/software-alignment evidence only",
    "physical_experiment_requested": False,
}


class BindingError(RuntimeError):
    """Raised when a formal evidence-binding gate fails."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _git(*args: str, binary: bool = False) -> str | bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=not binary,
    )
    return completed.stdout


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BindingError(f"{context} must be a JSON object")
    return value


def _load_json(path: Path, context: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise BindingError(f"{context} does not exist: {path}")
    try:
        return _require_mapping(json.loads(path.read_text(encoding="utf-8")), context)
    except json.JSONDecodeError as error:
        raise BindingError(f"{context} is not valid JSON: {path}") from error


def _load_artifact_verifier() -> Any:
    verifier_path = REPO_ROOT / "scripts" / "verify_pyfgsea_artifacts.py"
    spec = importlib.util.spec_from_file_location(
        "pyfgsea_artifact_verifier_for_figure2", verifier_path
    )
    if spec is None or spec.loader is None:
        raise BindingError(f"could not load artifact verifier: {verifier_path}")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    return verifier


def _require_external_output(output_dir: Path) -> Path:
    output_dir = output_dir.expanduser().resolve()
    if _is_within(output_dir, REPO_ROOT):
        raise BindingError("--output-dir must be outside the verified Git worktree")
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing evidence directory: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return output_dir


def _capture_release_git_state(expected_commit: str, expected_tag: str) -> dict[str, Any]:
    expected_commit = expected_commit.lower()
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise BindingError("--expected-git-commit must be a full lowercase 40-hex SHA")
    if RELEASE_TAG_PATTERN.fullmatch(expected_tag) is None:
        raise BindingError("--expected-git-tag must be v0.2.0-rc3")

    head = str(_git("rev-parse", "HEAD")).strip().lower()
    if head != expected_commit:
        raise BindingError(f"repository HEAD is {head}, expected {expected_commit}")
    status = bytes(
        _git(
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            binary=True,
        )
    )
    if status:
        raise BindingError("repository must be clean for a formal Figure 2 run")

    tag_ref = f"refs/tags/{expected_tag}"
    object_type = str(_git("cat-file", "-t", tag_ref)).strip()
    if object_type != "tag":
        raise BindingError(f"release tag {expected_tag!r} is not annotated")
    tag_object = str(_git("rev-parse", tag_ref)).strip().lower()
    peeled_commit = str(_git("rev-parse", f"{tag_ref}^{{}}")).strip().lower()
    if peeled_commit != expected_commit:
        raise BindingError(
            f"release tag {expected_tag!r} peels to {peeled_commit}, expected {expected_commit}"
        )
    tree = str(_git("rev-parse", "HEAD^{tree}")).strip().lower()
    return {
        "commit": head,
        "tree": tree,
        "clean": True,
        "status_sha256": _sha256_bytes(status),
        "release_tag": {
            "name": expected_tag,
            "annotated": True,
            "tag_object": tag_object,
            "peeled_commit": peeled_commit,
        },
    }


def _require_unchanged_git_state(start: Mapping[str, Any]) -> dict[str, Any]:
    tag = _require_mapping(start.get("release_tag"), "start release tag")
    end = _capture_release_git_state(str(start["commit"]), str(tag["name"]))
    if end != dict(start):
        raise BindingError("release Git identity changed during the Figure 2 run")
    return end


def _load_author_contract() -> dict[str, Any]:
    payload = dict(_load_json(CONTRACT_PATH, "author parameter contract"))
    if payload.get("author_parameter_contract_status") != "accepted":
        raise BindingError("Figure 2 author parameter contract is not accepted")
    if payload.get("accepted_at_date") != "2026-09-01":
        raise BindingError("Figure 2 contract has an unexpected acceptance date")
    if payload.get("publication_accepted") is not False:
        raise BindingError("parameter acceptance must not imply publication acceptance")
    accepted = _require_mapping(payload.get("accepted_values"), "accepted_values")
    if dict(accepted) != ACCEPTED_VALUES:
        raise BindingError(
            f"accepted Figure 2 values differ from the runner: {dict(accepted)!r}"
        )
    semantics = _require_mapping(
        payload.get("pathway_size_semantics"), "pathway_size_semantics"
    )
    if semantics.get("policy") != "exact":
        raise BindingError("Figure 2 pathway size policy must be exact")
    enforced = _require_mapping(semantics.get("enforced_by"), "pathway size enforcement")
    if dict(enforced) != {"bin_width": 0, "mode": "aligned"}:
        raise BindingError("exact pathway-size enforcement differs from the runner")
    for key, value in ACCEPTED_VALUES.items():
        if PARAMETERS[key] != value:
            raise BindingError(f"runner parameter {key} differs from accepted contract")
    return {
        "path": CONTRACT_PATH.relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256_file(CONTRACT_PATH),
        "contract": payload,
    }


def _load_reference_contract() -> dict[str, Any]:
    payload = _load_json(REFERENCE_MANIFEST_PATH, "reference manifest")
    if payload.get("schema_version") != 2:
        raise BindingError("reference manifest schema_version must be 2")
    profiles = _require_mapping(payload.get("profiles"), "reference profiles")
    current = _require_mapping(
        profiles.get("current_conformance"), "current reference profile"
    )
    expected = {
        "pyfgsea_version": EXPECTED_VERSION,
        "fgsea_version": EXPECTED_FGSEA_REFERENCE,
        "pyfgsea_algorithm_revision": EXPECTED_ALGORITHM_REVISION,
    }
    if any(current.get(key) != value for key, value in expected.items()):
        raise BindingError("current fgsea reference contract differs from the RC3 runner")
    return {
        "manifest_path": REFERENCE_MANIFEST_PATH.relative_to(REPO_ROOT).as_posix(),
        "manifest_sha256": sha256_file(REFERENCE_MANIFEST_PATH),
        "profile": "current_conformance",
        "fgsea_alignment_target_version": EXPECTED_FGSEA_REFERENCE,
        "pyfgsea_algorithm_revision": EXPECTED_ALGORITHM_REVISION,
        "interpretation": "alignment target; this Figure 2 run does not execute R fgsea",
        "r_reference_run_performed": False,
    }


def _verify_input(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise BindingError(
            f"{label} hash mismatch: expected {expected_sha256}, found {observed}"
        )
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": observed}


def _reverify_release_artifacts(
    receipt_git: Mapping[str, Any],
    installed_tests: Mapping[str, Any],
    sdist: Mapping[str, Any],
    wheel: Mapping[str, Any],
    sdist_path: Path,
    wheel_path: Path,
    git_state: Mapping[str, Any],
) -> dict[str, Any]:
    verifier = _load_artifact_verifier()
    try:
        expected_sources = verifier._git_source_manifest(
            REPO_ROOT, str(git_state["commit"])
        )
        expected_hash_manifest = verifier._source_hash_manifest(expected_sources)
        recorded_hash_manifest = _require_mapping(
            receipt_git.get("source_manifest"), "artifact receipt source manifest"
        )
        if dict(recorded_hash_manifest) != expected_hash_manifest:
            raise BindingError(
                "artifact receipt source manifest differs from the release commit"
            )
        source_manifest_sha256 = verifier._source_manifest_sha256(expected_sources)
        expected_test_sources = verifier._git_installed_test_manifest(
            REPO_ROOT, str(git_state["commit"])
        )
        expected_test_manifest = verifier._source_hash_manifest(expected_test_sources)
        recorded_test_manifest = _require_mapping(
            installed_tests.get("test_source_manifest"),
            "installed-test source manifest",
        )
        if dict(recorded_test_manifest) != expected_test_manifest:
            raise BindingError(
                "installed-test source manifest differs from the release commit"
            )
        test_manifest_sha256 = verifier._source_manifest_sha256(
            expected_test_sources
        )
        if installed_tests.get("test_source_manifest_sha256") != test_manifest_sha256:
            raise BindingError("installed-test source aggregate hash differs from Git")
        actual_sdist = verifier._verify_sdist(
            sdist_path, expected_sources, expected_version=EXPECTED_VERSION
        )
        actual_wheel = verifier._verify_wheel(
            wheel_path, expected_sources, expected_version=EXPECTED_VERSION
        )
    except BindingError:
        raise
    except verifier.VerificationError as error:
        raise BindingError(f"artifact source-boundary verification failed: {error}") from error

    for context, recorded, actual in (
        ("sdist", sdist, actual_sdist),
        ("wheel", wheel, actual_wheel),
    ):
        for key, value in actual.items():
            if key == "path":
                # Build-host absolute paths are provenance only. The relocated
                # bundle path has already been resolved and hash-checked.
                continue
            if recorded.get(key) != value:
                raise BindingError(
                    f"artifact receipt {context} field {key!r} differs from "
                    "independent verification"
                )
    return {
        "source_manifest_sha256": source_manifest_sha256,
        "source_file_count": len(expected_sources),
        "source_manifest_matches_commit": True,
        "test_source_manifest_sha256": test_manifest_sha256,
        "test_source_file_count": len(expected_test_sources),
        "test_source_manifest_matches_commit": True,
        "sdist_reverified": True,
        "wheel_reverified": True,
    }


def _resolve_bundle_artifact(
    receipt_path: Path, record: Mapping[str, Any], label: str
) -> Path:
    """Resolve one artifact from a relocatable ``dist/`` + ``evidence/`` bundle."""
    if receipt_path.parent.name != "evidence":
        raise BindingError(
            "artifact receipt must use the <bundle>/evidence/receipt.json layout"
        )
    raw = record.get("bundle_path")
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise BindingError(f"artifact {label} has no safe POSIX bundle_path")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise BindingError(f"artifact {label} has an unsafe bundle_path: {raw!r}")
    bundle_root = receipt_path.parent.parent.resolve()
    artifact_path = bundle_root.joinpath(*relative.parts).resolve()
    if not _is_within(artifact_path, bundle_root):
        raise BindingError(f"artifact {label} escapes the downloaded bundle")
    return artifact_path


def _verify_installed_test_evidence(
    receipt_path: Path,
    value: Any,
    *,
    git_commit: str,
    wheel_sha256: str,
) -> dict[str, Any]:
    evidence = _require_mapping(value, "installed-test evidence")
    expected_fields = {
        "status": "passed",
        "git_commit": git_commit,
        "test_paths": ["tests", "repro/figure1_dual_lane/test_pipeline.py"],
        "pytest_version": "8.4.2",
        "trajectory_extra_installed": True,
        "isolated_python": True,
        "import_mode": "importlib",
        "cwd_outside_worktree": True,
        "wheel_sha256": wheel_sha256,
    }
    if any(evidence.get(key) != expected for key, expected in expected_fields.items()):
        raise BindingError("installed-test evidence contract is incomplete or inconsistent")

    test_manifest = _require_mapping(
        evidence.get("test_source_manifest"), "installed-test source manifest"
    )
    if not test_manifest or re.fullmatch(
        r"[0-9a-f]{64}", str(evidence.get("test_source_manifest_sha256", ""))
    ) is None:
        raise BindingError("installed-test source manifest is empty or unhashed")

    counts = _require_mapping(evidence.get("counts"), "installed-test counts")
    expected_count_keys = {"passed", "total", "failed", "errors", "skipped"}
    if set(counts) != expected_count_keys or not all(
        isinstance(counts[key], int) and counts[key] >= 0 for key in expected_count_keys
    ):
        raise BindingError("installed-test counts are malformed")
    if (
        counts["total"] <= 0
        or counts["failed"] != 0
        or counts["errors"] != 0
        or counts["passed"] + counts["skipped"] != counts["total"]
    ):
        raise BindingError("installed-test receipt does not record a passing suite")

    junit = _require_mapping(evidence.get("junit"), "installed-test JUnit")
    junit_path = _resolve_bundle_artifact(receipt_path, junit, "installed-test JUnit")
    if (
        not junit_path.is_file()
        or sha256_file(junit_path) != junit.get("sha256")
        or junit_path.stat().st_size != junit.get("bytes")
    ):
        raise BindingError("installed-test JUnit is missing or has changed")
    return {
        "status": "passed",
        "git_commit": git_commit,
        "wheel_sha256": wheel_sha256,
        "test_paths": ["tests", "repro/figure1_dual_lane/test_pipeline.py"],
        "test_source_manifest": dict(test_manifest),
        "test_source_manifest_sha256": evidence["test_source_manifest_sha256"],
        "pytest_version": evidence["pytest_version"],
        "counts": dict(counts),
        "junit": {
            "path": str(junit_path),
            "bundle_path": junit["bundle_path"],
            "bytes": junit["bytes"],
            "sha256": junit["sha256"],
        },
    }


def _verify_artifact_receipt(
    receipt_path: Path,
    git_state: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve()
    if _is_within(receipt_path, REPO_ROOT):
        raise BindingError("artifact receipt must be outside the verified Git worktree")
    payload = _load_json(receipt_path, "artifact receipt")
    if payload.get("schema_version") != 1:
        raise BindingError("artifact receipt schema_version must be 1")
    if (
        payload.get("status") != "passed"
        or payload.get("all_artifact_chain_gates_passed") is not True
    ):
        raise BindingError("artifact receipt did not pass every artifact-chain gate")

    receipt_git = _require_mapping(payload.get("git"), "artifact receipt git")
    if str(receipt_git.get("commit", "")).lower() != git_state["commit"]:
        raise BindingError("artifact receipt commit does not match the Figure 2 checkout")
    if str(receipt_git.get("tree", "")).lower() != git_state["tree"]:
        raise BindingError("artifact receipt tree does not match the Figure 2 checkout")
    if receipt_git.get("clean_before_and_after") is not True:
        raise BindingError("artifact receipt did not preserve a clean checkout")
    _require_mapping(
        receipt_git.get("source_manifest"), "artifact receipt source manifest"
    )
    expected_tag = _require_mapping(git_state.get("release_tag"), "expected release tag")
    receipt_tag = _require_mapping(receipt_git.get("release_tag"), "artifact release tag")
    for key in ("name", "annotated", "tag_object", "peeled_commit"):
        if receipt_tag.get(key) != expected_tag.get(key):
            raise BindingError(f"artifact receipt release-tag field {key!r} does not match")

    expected = _require_mapping(payload.get("expected"), "artifact receipt expected")
    if expected.get("cargo_version") != EXPECTED_CARGO_VERSION:
        raise BindingError("artifact receipt has an unexpected Cargo version")
    if expected.get("pyfgsea_version") != EXPECTED_VERSION:
        raise BindingError("artifact receipt has an unexpected PyFgsea version")
    if expected.get("algorithm_revision") != EXPECTED_ALGORITHM_REVISION:
        raise BindingError("artifact receipt has an unexpected algorithm revision")

    chain = _require_mapping(payload.get("artifact_chain"), "artifact chain")
    sdist = _require_mapping(chain.get("sdist"), "artifact sdist")
    wheel = _require_mapping(chain.get("wheel"), "artifact wheel")
    installed = _require_mapping(chain.get("installed"), "installed artifact")
    installed_tests = _verify_installed_test_evidence(
        receipt_path,
        payload.get("installed_tests"),
        git_commit=git_state["commit"],
        wheel_sha256=str(wheel.get("sha256", "")),
    )
    bundle = _require_mapping(payload.get("artifact_bundle"), "artifact bundle")
    expected_bundle = {
        "schema_version": 1,
        "layout": "receipt-parent-parent-v1",
        "bundle_root_relative_to_receipt": "..",
        "sdist": sdist.get("bundle_path"),
        "wheel": wheel.get("bundle_path"),
        "installed_tests_junit": installed_tests["junit"]["bundle_path"],
    }
    if dict(bundle) != expected_bundle:
        raise BindingError("artifact bundle contract is missing or inconsistent")
    sdist_path = _resolve_bundle_artifact(receipt_path, sdist, "sdist")
    wheel_path = _resolve_bundle_artifact(receipt_path, wheel, "wheel")
    if _is_within(sdist_path, REPO_ROOT) or _is_within(wheel_path, REPO_ROOT):
        raise BindingError("verified sdist and wheel must be outside the Git worktree")
    if not sdist_path.is_file() or sha256_file(sdist_path) != sdist.get("sha256"):
        raise BindingError("artifact receipt sdist is missing or has changed")
    if not wheel_path.is_file() or sha256_file(wheel_path) != wheel.get("sha256"):
        raise BindingError("artifact receipt wheel is missing or has changed")
    if wheel.get("build_input_sdist_sha256") != sdist.get("sha256"):
        raise BindingError("wheel is not bound to the verified sdist hash")
    if wheel.get("wheel_built_from_verified_sdist") is not True:
        raise BindingError("artifact receipt does not prove sdist-to-wheel construction")
    if wheel.get("wheel_member_boundary_exact") is not True:
        raise BindingError("artifact wheel member boundary was not verified")
    if sdist.get("pyfgsea_source_set_exact") is not True:
        raise BindingError("artifact sdist package source boundary was not exact")
    if sdist.get("native_binary_count") != 0:
        raise BindingError("artifact sdist contains a native binary")
    if wheel.get("pyfgsea_source_set_exact") is not True:
        raise BindingError("artifact wheel package source boundary was not exact")
    if sdist.get("cargo_version") != EXPECTED_CARGO_VERSION:
        raise BindingError(
            f"artifact sdist Cargo version is not {EXPECTED_CARGO_VERSION}"
        )
    if sdist.get("metadata_version") != EXPECTED_VERSION:
        raise BindingError(f"artifact sdist metadata version is not {EXPECTED_VERSION}")
    if wheel.get("metadata_version") != EXPECTED_VERSION:
        raise BindingError(f"artifact wheel metadata version is not {EXPECTED_VERSION}")
    if wheel.get("verified_source_manifest_sha256") != sdist.get(
        "verified_source_manifest_sha256"
    ):
        raise BindingError("sdist and wheel source manifests differ")
    if installed.get("core_sha256") != wheel.get("core_sha256"):
        raise BindingError("installed core hash differs from the wheel core hash")
    if installed.get("direct_url_wheel_sha256") != wheel.get("sha256"):
        raise BindingError("installed direct_url hash differs from the wheel hash")
    if installed.get("pyfgsea_version") != EXPECTED_VERSION:
        raise BindingError(f"installed artifact module version is not {EXPECTED_VERSION}")
    if installed.get("distribution_version") != EXPECTED_VERSION:
        raise BindingError(
            f"installed artifact distribution version is not {EXPECTED_VERSION}"
        )
    if installed.get("algorithm_revision") != EXPECTED_ALGORITHM_REVISION:
        raise BindingError("installed artifact algorithm revision is unexpected")
    if installed.get("package_and_core_inside_venv") is not True:
        raise BindingError("artifact receipt package/core paths are not inside its venv")

    source_reverification = _reverify_release_artifacts(
        receipt_git,
        installed_tests,
        sdist,
        wheel,
        sdist_path,
        wheel_path,
        git_state,
    )

    core_member = str(wheel.get("core_member", ""))
    with zipfile.ZipFile(wheel_path) as archive:
        try:
            core_bytes = archive.read(core_member)
        except KeyError as error:
            raise BindingError(f"wheel does not contain {core_member!r}") from error
    if _sha256_bytes(core_bytes) != wheel.get("core_sha256"):
        raise BindingError("native core bytes no longer match the artifact receipt")

    return {
        "path": str(receipt_path),
        "sha256": sha256_file(receipt_path),
        "status": "passed",
        "source_reverification": source_reverification,
        "sdist": {
            "path": str(sdist_path),
            "sha256": sdist["sha256"],
        },
        "wheel": {
            "path": str(wheel_path),
            "sha256": wheel["sha256"],
            "core_sha256": wheel["core_sha256"],
            "core_member": core_member,
            "verified_source_manifest_sha256": wheel[
                "verified_source_manifest_sha256"
            ],
        },
        "installed": dict(installed),
        "installed_tests": installed_tests,
    }


def _direct_url_sha256(direct_url: Mapping[str, Any]) -> str:
    archive_info = _require_mapping(direct_url.get("archive_info"), "direct_url archive_info")
    hashes = archive_info.get("hashes")
    if isinstance(hashes, Mapping) and isinstance(hashes.get("sha256"), str):
        return str(hashes["sha256"]).lower()
    value = archive_info.get("hash")
    if isinstance(value, str) and value.startswith("sha256="):
        return value.removeprefix("sha256=").lower()
    raise BindingError("installed direct_url.json has no SHA-256")


def _verify_installed_pyfgsea(artifact: Mapping[str, Any]) -> dict[str, Any]:
    import pyfgsea

    core = importlib.import_module("pyfgsea._core")
    distribution = importlib.metadata.distribution("pyfgsea")
    package_path = Path(pyfgsea.__file__).resolve()
    core_path = Path(core.__file__).resolve()
    executable = Path(sys.executable).resolve()
    prefix = Path(sys.prefix).resolve()
    if _is_within(package_path, REPO_ROOT) or _is_within(core_path, REPO_ROOT):
        raise BindingError("Figure 2 must import PyFgsea from the verified wheel, not the checkout")
    if not _is_within(package_path, prefix) or not _is_within(core_path, prefix):
        raise BindingError("loaded PyFgsea package/core are outside the active environment")

    base_prefix = Path(getattr(sys, "base_prefix", sys.prefix)).resolve()
    if base_prefix == prefix:
        raise BindingError("Figure 2 must run in a fresh virtual environment")
    if pyfgsea.__version__ != EXPECTED_VERSION or distribution.version != EXPECTED_VERSION:
        raise BindingError(
            f"loaded PyFgsea module/distribution version is not {EXPECTED_VERSION}"
        )
    revision = core.algorithm_revision()
    if revision != EXPECTED_ALGORITHM_REVISION:
        raise BindingError(f"unexpected statistical core revision: {revision}")
    core_sha256 = sha256_file(core_path)
    wheel = _require_mapping(artifact.get("wheel"), "receipt wheel")
    if core_sha256 != wheel.get("core_sha256"):
        raise BindingError("loaded native core hash differs from the verified wheel")
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise BindingError("installed distribution has no direct_url.json")
    direct_url = _require_mapping(json.loads(direct_url_text), "direct_url.json")
    direct_url_sha256 = _direct_url_sha256(direct_url)
    if direct_url_sha256 != wheel.get("sha256"):
        raise BindingError("loaded distribution direct_url hash differs from the verified wheel")
    wheel_path = Path(str(wheel.get("path", ""))).resolve()
    verified_members: dict[str, str] = {}
    with zipfile.ZipFile(wheel_path) as archive:
        for member in sorted(archive.namelist()):
            if not member.startswith("pyfgsea/") or member.endswith("/"):
                continue
            installed_path = Path(distribution.locate_file(member)).resolve()
            if not installed_path.is_file():
                raise BindingError(f"installed wheel member is missing: {member}")
            wheel_member_sha = _sha256_bytes(archive.read(member))
            if sha256_file(installed_path) != wheel_member_sha:
                raise BindingError(f"installed wheel member has changed: {member}")
            verified_members[member] = wheel_member_sha
    package_root = package_path.parent
    actual_members = {
        "pyfgsea/" + path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix.lower() != ".pyc"
    }
    if actual_members != set(verified_members):
        missing = sorted(set(verified_members) - actual_members)
        extra = sorted(actual_members - set(verified_members))
        raise BindingError(
            "installed pyfgsea member boundary differs from the wheel; "
            f"missing={missing!r}, extra={extra!r}"
        )
    return {
        "python_executable": str(executable),
        "sys_prefix": str(prefix),
        "base_prefix": str(base_prefix),
        "package_file": str(package_path),
        "core_file": str(core_path),
        "module_version": pyfgsea.__version__,
        "distribution_version": distribution.version,
        "algorithm_revision": revision,
        "core_sha256": core_sha256,
        "direct_url_wheel_sha256": direct_url_sha256,
        "verified_package_member_count": len(verified_members),
        "installed_package_member_boundary_exact": True,
        "verified_package_members_sha256": _sha256_bytes(
            json.dumps(
                verified_members, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ),
    }


def formal_preflight(
    dataset: Path,
    gene_sets: Path,
    artifact_receipt: Path,
    expected_commit: str,
    expected_tag: str,
) -> dict[str, Any]:
    git_state = _capture_release_git_state(expected_commit, expected_tag)
    contract = _load_author_contract()
    reference_contract = _load_reference_contract()
    artifact = _verify_artifact_receipt(artifact_receipt, git_state)
    installed = _verify_installed_pyfgsea(artifact)
    inputs = {
        "dataset": _verify_input(dataset, EXPECTED_DATASET_SHA256, "dataset"),
        "gene_sets": _verify_input(gene_sets, EXPECTED_GENE_SETS_SHA256, "gene sets"),
    }
    return {
        "git": git_state,
        "author_parameter_contract": contract,
        "reference_contract": reference_contract,
        "artifact_receipt": artifact,
        "installed_artifact": installed,
        "inputs": inputs,
        "parameters": dict(PARAMETERS),
        "research_context": dict(RESEARCH_CONTEXT),
        "environment_versions": _environment_versions(),
    }


def bh_adjust(values: Iterable[float]) -> np.ndarray:
    pvalues = np.asarray(list(values), dtype=np.float64)
    adjusted = np.full(pvalues.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(pvalues)
    if not valid.any():
        return adjusted
    selected = pvalues[valid]
    order = np.argsort(selected, kind="mergesort")
    ranked = selected[order]
    scaled = ranked * selected.size / np.arange(1, selected.size + 1)
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    restored = np.empty_like(scaled)
    restored[order] = np.clip(scaled, 0.0, 1.0)
    adjusted[valid] = restored
    return adjusted


def _audit_results(results: pd.DataFrame) -> dict[str, Any]:
    required = {
        "Pathway",
        "NES",
        "P-value",
        "padj",
        "status",
        "window_id",
        "pt_mid",
    }
    missing = sorted(required.difference(results.columns))
    if missing:
        raise BindingError(f"trajectory result is missing required columns: {missing}")
    if len(results) != EXPECTED_N_ROWS:
        raise BindingError(f"expected {EXPECTED_N_ROWS} rows, found {len(results)}")
    duplicate_count = int(results.duplicated(["window_id", "Pathway"]).sum())
    if duplicate_count:
        raise BindingError(f"trajectory grid has {duplicate_count} duplicate keys")
    window_ids = sorted(int(value) for value in results["window_id"].unique())
    if window_ids != list(range(EXPECTED_N_WINDOWS)):
        raise BindingError("trajectory window IDs are not the complete 0..61 grid")
    if int(results["Pathway"].nunique()) != EXPECTED_N_PATHWAYS:
        raise BindingError("trajectory result does not contain exactly 43 pathways")
    per_window = results.groupby("window_id")["Pathway"].nunique()
    per_pathway = results.groupby("Pathway")["window_id"].nunique()
    if not (per_window == EXPECTED_N_PATHWAYS).all() or not (
        per_pathway == EXPECTED_N_WINDOWS
    ).all():
        raise BindingError("trajectory result is not a complete 62 x 43 grid")
    status_counts = {
        str(key): int(value)
        for key, value in results["status"].value_counts(dropna=False).sort_index().items()
    }
    if status_counts != {"resolved": EXPECTED_N_ROWS}:
        raise BindingError(f"formal Figure 2 requires all rows resolved: {status_counts}")
    pvalues = results["P-value"].to_numpy(dtype=np.float64)
    padj = results["padj"].to_numpy(dtype=np.float64)
    nes = results["NES"].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(pvalues).all()
        or (pvalues <= 0).any()
        or (pvalues > 1).any()
    ):
        raise BindingError("formal results require p-values in (0, 1]")
    if (
        not np.isfinite(padj).all()
        or (padj < 0).any()
        or (padj > 1).any()
        or not np.isfinite(nes).all()
    ):
        raise BindingError("formal results require padj in [0, 1] and finite NES")

    independent = np.empty(len(results), dtype=np.float64)
    for indices in results.groupby("window_id", sort=False).indices.values():
        independent[indices] = bh_adjust(results.iloc[indices]["P-value"])
    max_abs_diff = float(np.max(np.abs(padj - independent)))
    if max_abs_diff > BH_ATOL:
        raise BindingError(
            f"independent within-window BH differs by {max_abs_diff}, tolerance {BH_ATOL}"
        )
    return {
        "n_rows": int(len(results)),
        "n_windows": int(results["window_id"].nunique()),
        "n_pathways": int(results["Pathway"].nunique()),
        "expected_grid": [EXPECTED_N_WINDOWS, EXPECTED_N_PATHWAYS],
        "complete_grid": True,
        "duplicate_key_count": duplicate_count,
        "status_counts": status_counts,
        "resolved_rows": status_counts["resolved"],
        "bh": {
            "scope": "within-window",
            "tolerance_absolute": BH_ATOL,
            "max_absolute_difference": max_abs_diff,
            "matches_core": True,
        },
    }


def pathway_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pathway, group in results.groupby("Pathway", sort=True):
        ordered = group.sort_values(["pt_mid", "window_id"], kind="mergesort")
        peak = ordered.loc[ordered["NES"].abs().idxmax()]
        peak_nes = float(peak["NES"])
        rows.append(
            {
                "Pathway": pathway,
                "start_pt": float(ordered.iloc[0]["pt_mid"]),
                "start_NES": float(ordered.iloc[0]["NES"]),
                "end_pt": float(ordered.iloc[-1]["pt_mid"]),
                "end_NES": float(ordered.iloc[-1]["NES"]),
                "peak_pt": float(peak["pt_mid"]),
                "peak_NES": peak_nes,
                "max_abs_NES": abs(peak_nes),
                "peak_direction": "positive" if peak_nes > 0 else "negative",
                "min_padj": float(ordered["padj"].min()),
                "significant_windows_fdr_0_05": int((ordered["padj"] < 0.05).sum()),
                "n_windows": int(len(ordered)),
            }
        )
    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        ["max_abs_NES", "min_padj", "Pathway"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    summary.insert(0, "top_pathway_rank", np.arange(1, len(summary) + 1))
    return summary


def plot_target_curves(results: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    import matplotlib.pyplot as plt

    missing = [name for name in TARGET_PATHWAYS if name not in set(results["Pathway"])]
    if missing:
        raise BindingError(f"required Figure 2 pathways are missing: {missing}")
    fig, axes = plt.subplots(len(TARGET_PATHWAYS), 1, figsize=(8.2, 6.6), sharex=True)
    for axis, pathway in zip(np.atleast_1d(axes), TARGET_PATHWAYS):
        curve = results.loc[results["Pathway"] == pathway].sort_values("pt_mid")
        axis.axhline(0.0, color="#777777", linewidth=0.8)
        axis.plot(curve["pt_mid"], curve["NES"], color="#155e75", linewidth=2.0)
        significant = curve["padj"] < 0.05
        axis.scatter(
            curve.loc[significant, "pt_mid"],
            curve.loc[significant, "NES"],
            color="#c2410c",
            s=19,
            label="within-window BH FDR < 0.05",
            zorder=3,
        )
        axis.set_title(pathway)
        axis.set_ylabel("NES")
        axis.legend(loc="best", frameon=False, fontsize=8)
    axes[-1].set_xlabel("Pseudotime midpoint")
    fig.suptitle("PyFgsea 0.2.0rc3 Figure 2 parameter-bound recalculation")
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _artifact_record(
    path: Path, final_dir: Path, relative_path: Path | None = None
) -> dict[str, Any]:
    relative_path = relative_path or Path(path.name)
    return {
        "path": str(final_dir / relative_path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _environment_versions() -> dict[str, str]:
    packages = ("scanpy", "anndata", "numpy", "pandas", "scipy", "matplotlib", "h5py")
    return {name: importlib.metadata.version(name) for name in packages}


def _publish_evidence(
    staging_dir: Path,
    output_dir: Path,
    manifest: Mapping[str, Any],
) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {output_dir}")
    staging_dir.replace(output_dir)
    pending_manifest = output_dir / ".run_manifest.json.pending"
    final_manifest = output_dir / "run_manifest.json"
    try:
        pending_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(pending_manifest, final_manifest)
    except BaseException:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return output_dir


def run(
    output_dir: Path,
    dataset: Path,
    gene_sets: Path,
    artifact_receipt: Path,
    expected_commit: str,
    expected_tag: str,
) -> Path:
    started_at = utc_now()
    timer = time.perf_counter()
    output_dir = _require_external_output(output_dir)
    preflight = formal_preflight(
        dataset, gene_sets, artifact_receipt, expected_commit, expected_tag
    )
    start_git = preflight["git"]

    from pyfgsea.trajectory import run_trajectory_gsea

    import scanpy as sc

    dataset_path = Path(preflight["inputs"]["dataset"]["path"])
    gene_sets_path = Path(preflight["inputs"]["gene_sets"]["path"])
    incomplete_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.incomplete-", dir=output_dir.parent)
    )
    try:
        frozen_dir = incomplete_dir / "frozen_inputs"
        frozen_dir.mkdir()
        frozen_dataset = frozen_dir / dataset_path.name
        frozen_gene_sets = frozen_dir / gene_sets_path.name
        shutil.copyfile(dataset_path, frozen_dataset)
        shutil.copyfile(gene_sets_path, frozen_gene_sets)
        frozen_inputs = {
            "dataset": _verify_input(
                frozen_dataset, EXPECTED_DATASET_SHA256, "frozen dataset copy"
            ),
            "gene_sets": _verify_input(
                frozen_gene_sets,
                EXPECTED_GENE_SETS_SHA256,
                "frozen gene-set copy",
            ),
        }

        adata = sc.read_h5ad(frozen_dataset)
        if adata.shape != EXPECTED_DATASET_SHAPE:
            raise BindingError(f"unexpected GSE155254 shape: {adata.shape}")
        if PARAMETERS["pseudotime_key"] not in adata.obs:
            raise BindingError("the frozen processed input lacks dpt_pseudotime")
        results = run_trajectory_gsea(
            adata,
            gmt_path=str(frozen_gene_sets),
            lineage_col=None,
            lineage_keyword=None,
            root_gene=None,
            out_csv=None,
            **PARAMETERS,
        )
        if results.empty:
            raise BindingError("trajectory recalculation returned no pathways")
        result_audit = _audit_results(results)
        results = results.sort_values(
            ["window_id", "padj", "P-value", "Pathway"], kind="mergesort"
        ).reset_index(drop=True)
        summary = pathway_summary(results)
        targets = summary.loc[summary["Pathway"].isin(TARGET_PATHWAYS)].copy()
        if len(targets) != len(TARGET_PATHWAYS):
            raise BindingError("target pathway summary is incomplete")

        result_path = incomplete_dir / "trajectory_results.csv"
        summary_path = incomplete_dir / "pathway_summary.csv"
        target_path = incomplete_dir / "figure2_target_pathway_summary.csv"
        png_path = incomplete_dir / "figure2_formal.png"
        pdf_path = incomplete_dir / "figure2_formal.pdf"
        results.to_csv(result_path, index=False)
        summary.to_csv(summary_path, index=False)
        targets.to_csv(target_path, index=False)
        plot_target_curves(results, png_path, pdf_path)
        artifacts = {
            path.name: _artifact_record(path, output_dir)
            for path in (result_path, summary_path, target_path, png_path, pdf_path)
        }
        frozen_input_artifacts = {
            "dataset": _artifact_record(
                frozen_dataset,
                output_dir,
                Path("frozen_inputs") / frozen_dataset.name,
            ),
            "gene_sets": _artifact_record(
                frozen_gene_sets,
                output_dir,
                Path("frozen_inputs") / frozen_gene_sets.name,
            ),
        }

        end_git = _require_unchanged_git_state(start_git)
        end_contract = _load_author_contract()
        end_reference_contract = _load_reference_contract()
        end_artifact = _verify_artifact_receipt(artifact_receipt, end_git)
        end_installed = _verify_installed_pyfgsea(end_artifact)
        end_inputs = {
            "dataset": _verify_input(
                dataset_path, EXPECTED_DATASET_SHA256, "dataset"
            ),
            "gene_sets": _verify_input(
                gene_sets_path, EXPECTED_GENE_SETS_SHA256, "gene sets"
            ),
        }
        end_environment = _environment_versions()
        if end_contract != preflight["author_parameter_contract"]:
            raise BindingError("author parameter contract changed during the run")
        if end_reference_contract != preflight["reference_contract"]:
            raise BindingError("fgsea reference contract changed during the run")
        if end_artifact != preflight["artifact_receipt"]:
            raise BindingError("artifact receipt or release artifacts changed during the run")
        if end_installed != preflight["installed_artifact"]:
            raise BindingError("installed wheel environment changed during the run")
        if end_inputs != preflight["inputs"]:
            raise BindingError("source inputs changed during the run")
        if end_environment != preflight["environment_versions"]:
            raise BindingError("dependency versions changed during the run")
        if frozen_inputs["dataset"]["sha256"] != end_inputs["dataset"]["sha256"]:
            raise BindingError("frozen dataset bytes differ from the final source input")
        if frozen_inputs["gene_sets"]["sha256"] != end_inputs["gene_sets"]["sha256"]:
            raise BindingError("frozen gene-set bytes differ from the final source input")

        release_tag_name = str(start_git["release_tag"]["name"])
        manifest = {
            "schema_version": 2,
            "artifact_type": "pyfgsea-figure2-formal-recalculation",
            "binding_status": "bound-to-clean-tag-and-verified-wheel",
            "author_parameter_contract": {
                "status": "accepted",
                "publication_accepted": False,
                "path": preflight["author_parameter_contract"]["path"],
                "sha256": preflight["author_parameter_contract"]["sha256"],
                "contract_id": preflight["author_parameter_contract"]["contract"][
                    "contract_id"
                ],
                "accepted_at_date": preflight["author_parameter_contract"][
                    "contract"
                ]["accepted_at_date"],
            },
            "reference_contract": preflight["reference_contract"],
            "evidence_binding": (
                f"bound-to-clean-{release_tag_name}-and-verified-wheel"
            ),
            "publication_accepted": False,
            "publication_binding": "pending-manuscript-impact-adjudication",
            "binding_blockers": [
                "manuscript and supplement have not yet been impact-adjudicated",
                "legacy/current Figure 1 and supplemental lanes remain incomplete",
            ],
            "research_context": dict(RESEARCH_CONTEXT),
            "started_at_utc": started_at,
            "finished_at_utc": utc_now(),
            "elapsed_seconds": time.perf_counter() - timer,
            "command": [str(value) for value in sys.argv],
            "working_directory": os.getcwd(),
            "git": {"start": start_git, "end": end_git, "unchanged": True},
            "runner": {
                "path": Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "artifact_receipt": preflight["artifact_receipt"],
            "installed_artifact": preflight["installed_artifact"],
            "verified_external_state_unchanged_before_and_after": True,
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "packages": end_environment,
            },
            "parameters": dict(PARAMETERS),
            "inputs": preflight["inputs"],
            "frozen_input_artifacts": frozen_input_artifacts,
            "dataset_shape": [int(adata.n_obs), int(adata.n_vars)],
            "result_audit": result_audit,
            "artifacts": artifacts,
        }
        return _publish_evidence(incomplete_dir, output_dir, manifest)
    except BaseException:
        if incomplete_dir.exists():
            shutil.rmtree(incomplete_dir, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--gene-sets", type=Path, required=True)
    parser.add_argument("--artifact-receipt", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--expected-git-tag", required=True)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify Git, contract, inputs, and the installed artifact without computing GSEA.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.preflight_only:
        receipt = formal_preflight(
            args.dataset,
            args.gene_sets,
            args.artifact_receipt,
            args.expected_git_commit,
            args.expected_git_tag,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    output = run(
        args.output_dir,
        args.dataset,
        args.gene_sets,
        args.artifact_receipt,
        args.expected_git_commit,
        args.expected_git_tag,
    )
    print(f"Formal Figure 2 evidence written without overwrite: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
