"""Reproduce Figure 2 from fixed inputs, parameters, and release artifacts.

The runner verifies the clean annotated release tag, installed wheel and native
core, recorded parameters, reference version, inputs, results, and output files.
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
PARAMETER_PATH = REPO_ROOT / "repro" / "figure2_gse155254" / "figure2_parameters.json"
REFERENCE_MANIFEST_PATH = REPO_ROOT / "reference_manifest.json"
EXPECTED_VERSION = "0.2.0rc8"
EXPECTED_CARGO_VERSION = "0.2.0-rc8"
EXPECTED_ALGORITHM_REVISION = "fgsea-1.38-pr178-v1"
EXPECTED_FGSEA_REFERENCE = "1.38.0"
EXPECTED_DATASET_SHA256 = "9d9d1db60fe06037c5bfcf1a6ce06adfa74fe6ef715d910ef8b7e004d05cd21e"
EXPECTED_GENE_SETS_SHA256 = "92203149acdfa1e7d583fad5da99487244ce353b4d10d3193cd64329e334da66"
EXPECTED_DATASET_SHAPE = (3576, 3000)
EXPECTED_N_WINDOWS = 62
EXPECTED_N_PATHWAYS = 43
EXPECTED_N_ROWS = EXPECTED_N_WINDOWS * EXPECTED_N_PATHWAYS
BH_ATOL = 1e-14
RELEASE_TAG_PATTERN = re.compile(r"v0\.2\.0-rc8")

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
RECORDED_PARAMETERS: dict[str, Any] = {
    **PARAMETERS,
    "pathway_size_policy": "exact",
}
TARGET_PATHWAYS = ("heme Metabolism", "E2F Targets")


class Figure2Error(RuntimeError):
    """Raised when Figure 2 inputs or verification checks are invalid."""


# Compatibility for callers that imported the previous exception name.
BindingError = Figure2Error


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
        raise Figure2Error(
            f"{context} is not a JSON object. Replace it with an object and rerun."
        )
    return value


def _load_json(path: Path, context: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise Figure2Error(
            f"{context} was not found at {path}. Restore the file and rerun."
        )
    try:
        return _require_mapping(json.loads(path.read_text(encoding="utf-8")), context)
    except json.JSONDecodeError as error:
        raise Figure2Error(
            f"{context} at {path} is not valid JSON. Fix the JSON and rerun."
        ) from error


def _load_artifact_verifier() -> Any:
    verifier_path = REPO_ROOT / "scripts" / "verify_pyfgsea_artifacts.py"
    spec = importlib.util.spec_from_file_location(
        "pyfgsea_artifact_verifier_for_figure2", verifier_path
    )
    if spec is None or spec.loader is None:
        raise Figure2Error(
            f"The artifact verifier could not be loaded from {verifier_path}. "
            "Restore scripts/verify_pyfgsea_artifacts.py and rerun."
        )
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    return verifier


def _require_external_output(output_dir: Path) -> Path:
    output_dir = output_dir.expanduser().resolve()
    if _is_within(output_dir, REPO_ROOT):
        raise Figure2Error(
            "The output directory is inside the Git checkout. Choose a directory "
            "outside the checkout and rerun."
        )
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing evidence directory: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return output_dir


def _capture_release_git_state(expected_commit: str, expected_tag: str) -> dict[str, Any]:
    expected_commit = expected_commit.lower()
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise Figure2Error(
            "The expected commit is not a full lowercase 40-character SHA. "
            "Pass the release commit from git rev-parse HEAD and rerun."
        )
    if RELEASE_TAG_PATTERN.fullmatch(expected_tag) is None:
        raise Figure2Error(
            "The expected tag is not v0.2.0-rc8. Pass the RC8 tag and rerun."
        )

    head = str(_git("rev-parse", "HEAD")).strip().lower()
    if head != expected_commit:
        raise Figure2Error(
            f"The checkout is at {head}, not {expected_commit}. Check out the "
            "expected release commit and rerun."
        )
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
        raise Figure2Error(
            "The Git checkout contains tracked or untracked changes. Commit, stash, "
            "or remove them and rerun from the release commit."
        )

    tag_ref = f"refs/tags/{expected_tag}"
    object_type = str(_git("cat-file", "-t", tag_ref)).strip()
    if object_type != "tag":
        raise Figure2Error(
            f"The release tag {expected_tag!r} is lightweight, not annotated. "
            "Use the annotated release-candidate tag and rerun."
        )
    tag_object = str(_git("rev-parse", tag_ref)).strip().lower()
    peeled_commit = str(_git("rev-parse", f"{tag_ref}^{{}}")).strip().lower()
    if peeled_commit != expected_commit:
        raise Figure2Error(
            f"The release tag {expected_tag!r} points to {peeled_commit}, not "
            f"{expected_commit}. Use the matching tag and commit and rerun."
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
        raise Figure2Error(
            "The commit, tree, tag, or worktree state changed while Figure 2 was "
            "running. Restore the starting release checkout and rerun."
        )
    return end


def _load_figure2_parameters() -> dict[str, Any]:
    payload = dict(_load_json(PARAMETER_PATH, "Figure 2 parameter file"))
    expected_keys = {"schema_version", "dataset", "recorded_on", "parameters"}
    if set(payload) != expected_keys:
        raise Figure2Error(
            "Figure 2 parameter file has unexpected fields. Keep only "
            "schema_version, dataset, recorded_on, and parameters, then rerun."
        )
    if payload.get("schema_version") != 1:
        raise Figure2Error(
            "Figure 2 parameter schema is not version 1. Set schema_version to 1 and rerun."
        )
    if payload.get("dataset") != "GSE155254":
        raise Figure2Error(
            "Figure 2 parameter file names a different dataset. Set dataset to "
            "GSE155254 and rerun."
        )
    if payload.get("recorded_on") != "2026-09-01":
        raise Figure2Error(
            "Figure 2 parameter date differs from the recorded configuration. "
            "Restore recorded_on to 2026-09-01 and rerun."
        )
    recorded = _require_mapping(payload.get("parameters"), "Figure 2 parameters")
    if recorded.get("pathway_size_policy") != "exact":
        raise Figure2Error(
            "Figure 2 pathway_size_policy is not exact. Set it to exact and rerun."
        )
    if dict(recorded) != RECORDED_PARAMETERS:
        raise Figure2Error(
            f"Figure 2 parameters differ from the runner: {dict(recorded)!r}. "
            "Make the JSON parameters match RECORDED_PARAMETERS and rerun."
        )
    return {
        "path": PARAMETER_PATH.relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256_file(PARAMETER_PATH),
        "record": payload,
    }


def _load_reference_contract() -> dict[str, Any]:
    payload = _load_json(REFERENCE_MANIFEST_PATH, "reference manifest")
    if payload.get("schema_version") != 2:
        raise Figure2Error(
            "The reference manifest schema is not version 2. Restore the version 2 "
            "reference_manifest.json and rerun."
        )
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
        raise Figure2Error(
            "The current reference profile does not specify PyFgsea 0.2.0rc8, "
            "fgsea 1.38.0, and the expected algorithm revision. Restore those "
            "values in reference_manifest.json and rerun."
        )
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
        raise Figure2Error(
            f"The {label} file was not found at {path}. Restore the file and rerun."
        )
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise Figure2Error(
            f"The {label} SHA-256 is {observed}, not {expected_sha256}. Use the "
            "recorded Figure 2 input file and rerun."
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
            raise Figure2Error(
                "The receipt source-file list differs from the release commit. "
                "Regenerate the artifact bundle from that commit and rerun."
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
            raise Figure2Error(
                "The installed-test file list differs from the release commit. "
                "Regenerate the artifact bundle from that commit and rerun."
            )
        test_manifest_sha256 = verifier._source_manifest_sha256(
            expected_test_sources
        )
        if installed_tests.get("test_source_manifest_sha256") != test_manifest_sha256:
            raise Figure2Error(
                "The installed-test file summary does not match Git. Regenerate the "
                "artifact bundle from the release commit and rerun."
            )
        actual_sdist = verifier._verify_sdist(
            sdist_path, expected_sources, expected_version=EXPECTED_VERSION
        )
        actual_wheel = verifier._verify_wheel(
            wheel_path, expected_sources, expected_version=EXPECTED_VERSION
        )
    except Figure2Error:
        raise
    except verifier.VerificationError as error:
        raise Figure2Error(
            f"The sdist or wheel contents do not match the release source: {error}. "
            "Rebuild the artifact bundle from the release commit and rerun."
        ) from error

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
                raise Figure2Error(
                    f"The recorded {context} field {key!r} differs from the file "
                    "that was independently checked. Regenerate the artifact "
                    "bundle and rerun."
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
        raise Figure2Error(
            "The receipt is not at <bundle>/evidence/receipt.json. Restore the "
            "downloaded bundle layout and rerun."
        )
    raw = record.get("bundle_path")
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise Figure2Error(
            f"The {label} bundle_path is missing or invalid. Regenerate the receipt "
            "with a relative POSIX path and rerun."
        )
    relative = PurePosixPath(raw)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise Figure2Error(
            f"The {label} bundle_path {raw!r} leaves the bundle. Regenerate the "
            "receipt with a path under dist/ or evidence/ and rerun."
        )
    bundle_root = receipt_path.parent.parent.resolve()
    artifact_path = bundle_root.joinpath(*relative.parts).resolve()
    if not _is_within(artifact_path, bundle_root):
        raise Figure2Error(
            f"The {label} path resolves outside the downloaded bundle. Restore the "
            "bundle and rerun."
        )
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
        raise Figure2Error(
            "The installed-test record is missing required values or names a "
            "different commit or wheel. Run the artifact verifier again and rerun "
            "Figure 2 with its receipt."
        )

    test_manifest = _require_mapping(
        evidence.get("test_source_manifest"), "installed-test source manifest"
    )
    if not test_manifest or re.fullmatch(
        r"[0-9a-f]{64}", str(evidence.get("test_source_manifest_sha256", ""))
    ) is None:
        raise Figure2Error(
            "The installed-test file list is empty or has no SHA-256 summary. Run "
            "the artifact verifier again and rerun Figure 2 with its receipt."
        )

    counts = _require_mapping(evidence.get("counts"), "installed-test counts")
    expected_count_keys = {"passed", "total", "failed", "errors", "skipped"}
    if set(counts) != expected_count_keys or not all(
        isinstance(counts[key], int) and counts[key] >= 0 for key in expected_count_keys
    ):
        raise Figure2Error(
            "The installed-test counts are missing or invalid. Run the artifact "
            "verifier again and rerun Figure 2 with its receipt."
        )
    if (
        counts["total"] <= 0
        or counts["failed"] != 0
        or counts["errors"] != 0
        or counts["passed"] + counts["skipped"] != counts["total"]
    ):
        raise Figure2Error(
            "The installed-wheel test suite did not pass. Fix the failed tests, "
            "regenerate the artifact bundle, and rerun Figure 2."
        )

    junit = _require_mapping(evidence.get("junit"), "installed-test JUnit")
    junit_path = _resolve_bundle_artifact(receipt_path, junit, "installed-test JUnit")
    if (
        not junit_path.is_file()
        or sha256_file(junit_path) != junit.get("sha256")
        or junit_path.stat().st_size != junit.get("bytes")
    ):
        raise Figure2Error(
            "The installed-test JUnit file is missing or its hash or size changed. "
            "Restore or regenerate the artifact bundle and rerun."
        )
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
        raise Figure2Error(
            "The artifact receipt is inside the Git checkout. Use the receipt from "
            "an external downloaded artifact bundle and rerun."
        )
    payload = _load_json(receipt_path, "artifact receipt")
    if payload.get("schema_version") != 1:
        raise Figure2Error(
            "The artifact receipt schema is not version 1. Regenerate the bundle "
            "with the current artifact verifier and rerun."
        )
    if (
        payload.get("status") != "passed"
        or payload.get("all_artifact_chain_gates_passed") is not True
    ):
        raise Figure2Error(
            "The artifact receipt reports a failed verification check. Fix that "
            "failure, regenerate a passing artifact bundle, and rerun Figure 2."
        )

    receipt_git = _require_mapping(payload.get("git"), "artifact receipt git")
    if str(receipt_git.get("commit", "")).lower() != git_state["commit"]:
        raise Figure2Error(
            "The artifact receipt names a different commit than the Figure 2 "
            "checkout. Use the matching checkout and receipt and rerun."
        )
    if str(receipt_git.get("tree", "")).lower() != git_state["tree"]:
        raise Figure2Error(
            "The artifact receipt names a different Git tree than the Figure 2 "
            "checkout. Use the matching checkout and receipt and rerun."
        )
    if receipt_git.get("clean_before_and_after") is not True:
        raise Figure2Error(
            "The artifact build did not record a clean checkout before and after. "
            "Rebuild from a clean release checkout and rerun."
        )
    _require_mapping(
        receipt_git.get("source_manifest"), "artifact receipt source manifest"
    )
    expected_tag = _require_mapping(git_state.get("release_tag"), "expected release tag")
    receipt_tag = _require_mapping(receipt_git.get("release_tag"), "artifact release tag")
    for key in ("name", "annotated", "tag_object", "peeled_commit"):
        if receipt_tag.get(key) != expected_tag.get(key):
            raise Figure2Error(
                f"The artifact receipt release-tag field {key!r} does not match "
                "the checkout. Use the receipt from the same annotated tag and rerun."
            )

    expected = _require_mapping(payload.get("expected"), "artifact receipt expected")
    if expected.get("cargo_version") != EXPECTED_CARGO_VERSION:
        raise Figure2Error(
            f"The artifact receipt does not record Cargo version "
            f"{EXPECTED_CARGO_VERSION}. Rebuild RC8 artifacts and rerun."
        )
    if expected.get("pyfgsea_version") != EXPECTED_VERSION:
        raise Figure2Error(
            f"The artifact receipt does not record PyFgsea {EXPECTED_VERSION}. "
            "Use the RC8 artifact bundle and rerun."
        )
    if expected.get("algorithm_revision") != EXPECTED_ALGORITHM_REVISION:
        raise Figure2Error(
            "The artifact receipt names a different algorithm revision. Rebuild "
            "the RC8 artifacts with the expected Rust core and rerun."
        )

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
        raise Figure2Error(
            "The artifact bundle paths do not match the receipt. Restore or "
            "regenerate the complete bundle and rerun."
        )
    sdist_path = _resolve_bundle_artifact(receipt_path, sdist, "sdist")
    wheel_path = _resolve_bundle_artifact(receipt_path, wheel, "wheel")
    if _is_within(sdist_path, REPO_ROOT) or _is_within(wheel_path, REPO_ROOT):
        raise Figure2Error(
            "The sdist or wheel path is inside the Git checkout. Use artifacts from "
            "an external downloaded bundle and rerun."
        )
    if not sdist_path.is_file() or sha256_file(sdist_path) != sdist.get("sha256"):
        raise Figure2Error(
            "The sdist is missing or its SHA-256 changed. Restore or regenerate the "
            "artifact bundle and rerun."
        )
    if not wheel_path.is_file() or sha256_file(wheel_path) != wheel.get("sha256"):
        raise Figure2Error(
            "The wheel is missing or its SHA-256 changed. Restore or regenerate the "
            "artifact bundle and rerun."
        )
    if wheel.get("build_input_sdist_sha256") != sdist.get("sha256"):
        raise Figure2Error(
            "The wheel was not built from the recorded sdist. Rebuild the wheel from "
            "that sdist, regenerate the receipt, and rerun."
        )
    if wheel.get("wheel_built_from_verified_sdist") is not True:
        raise Figure2Error(
            "The receipt does not confirm that the wheel came from the checked "
            "sdist. Regenerate the artifact bundle with the current verifier."
        )
    if wheel.get("wheel_member_boundary_exact") is not True:
        raise Figure2Error(
            "The wheel file list was not checked exactly. Regenerate the artifact "
            "bundle with the current verifier and rerun."
        )
    if sdist.get("pyfgsea_source_set_exact") is not True:
        raise Figure2Error(
            "The sdist PyFgsea source-file list is not exact. Rebuild it from the "
            "release commit, regenerate the receipt, and rerun."
        )
    if sdist.get("native_binary_count") != 0:
        raise Figure2Error(
            "The sdist contains a native binary. Rebuild a source-only sdist and rerun."
        )
    if wheel.get("pyfgsea_source_set_exact") is not True:
        raise Figure2Error(
            "The wheel PyFgsea file list is not exact. Rebuild it from the checked "
            "sdist, regenerate the receipt, and rerun."
        )
    if sdist.get("cargo_version") != EXPECTED_CARGO_VERSION:
        raise Figure2Error(
            f"artifact sdist Cargo version is not {EXPECTED_CARGO_VERSION}"
        )
    if sdist.get("metadata_version") != EXPECTED_VERSION:
        raise Figure2Error(f"artifact sdist metadata version is not {EXPECTED_VERSION}")
    if wheel.get("metadata_version") != EXPECTED_VERSION:
        raise Figure2Error(f"artifact wheel metadata version is not {EXPECTED_VERSION}")
    if wheel.get("verified_source_manifest_sha256") != sdist.get(
        "verified_source_manifest_sha256"
    ):
        raise Figure2Error(
            "The sdist and wheel contain different source-file summaries. Rebuild "
            "the wheel from the checked sdist and rerun."
        )
    if installed.get("core_sha256") != wheel.get("core_sha256"):
        raise Figure2Error(
            "The installed native core differs from the wheel. Reinstall the checked "
            "wheel in a fresh environment and rerun."
        )
    if installed.get("direct_url_wheel_sha256") != wheel.get("sha256"):
        raise Figure2Error(
            "The installed distribution points to a different wheel. Reinstall the "
            "checked wheel in a fresh environment and rerun."
        )
    if installed.get("pyfgsea_version") != EXPECTED_VERSION:
        raise Figure2Error(f"installed artifact module version is not {EXPECTED_VERSION}")
    if installed.get("distribution_version") != EXPECTED_VERSION:
        raise Figure2Error(
            f"installed artifact distribution version is not {EXPECTED_VERSION}"
        )
    if installed.get("algorithm_revision") != EXPECTED_ALGORITHM_REVISION:
        raise Figure2Error(
            "The installed native core reports a different algorithm revision. "
            "Install the RC8 wheel from the checked bundle and rerun."
        )
    if installed.get("package_and_core_inside_venv") is not True:
        raise Figure2Error(
            "The receipt records PyFgsea or its core outside the test environment. "
            "Regenerate the artifact bundle in a fresh virtual environment and rerun."
        )

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
            raise Figure2Error(
                f"The wheel does not contain {core_member!r}. Rebuild the wheel and rerun."
            ) from error
    if _sha256_bytes(core_bytes) != wheel.get("core_sha256"):
        raise Figure2Error(
            "The native core in the wheel differs from the receipt. Restore or "
            "regenerate the artifact bundle and rerun."
        )

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
    raise Figure2Error(
        "The installed direct_url.json has no wheel SHA-256. Reinstall the checked "
        "wheel with pip in a fresh environment and rerun."
    )


def _verify_installed_pyfgsea(artifact: Mapping[str, Any]) -> dict[str, Any]:
    import pyfgsea

    core = importlib.import_module("pyfgsea._core")
    distribution = importlib.metadata.distribution("pyfgsea")
    package_path = Path(pyfgsea.__file__).resolve()
    core_path = Path(core.__file__).resolve()
    executable = Path(sys.executable).resolve()
    prefix = Path(sys.prefix).resolve()
    if _is_within(package_path, REPO_ROOT) or _is_within(core_path, REPO_ROOT):
        raise Figure2Error(
            "Figure 2 imported PyFgsea from the source checkout. Install the checked "
            "wheel in a fresh environment and rerun outside the checkout."
        )
    if not _is_within(package_path, prefix) or not _is_within(core_path, prefix):
        raise Figure2Error(
            "PyFgsea or its native core was loaded from outside the active "
            "environment. Reinstall the checked wheel in that environment and rerun."
        )

    base_prefix = Path(getattr(sys, "base_prefix", sys.prefix)).resolve()
    if base_prefix == prefix:
        raise Figure2Error(
            "Figure 2 is running without an isolated virtual environment. Create a "
            "fresh environment, install the checked wheel, and rerun."
        )
    if pyfgsea.__version__ != EXPECTED_VERSION or distribution.version != EXPECTED_VERSION:
        raise Figure2Error(
            f"loaded PyFgsea module/distribution version is not {EXPECTED_VERSION}"
        )
    revision = core.algorithm_revision()
    if revision != EXPECTED_ALGORITHM_REVISION:
        raise Figure2Error(
            f"The native core reports revision {revision!r}, not "
            f"{EXPECTED_ALGORITHM_REVISION!r}. Install the checked RC8 wheel and rerun."
        )
    core_sha256 = sha256_file(core_path)
    wheel = _require_mapping(artifact.get("wheel"), "receipt wheel")
    if core_sha256 != wheel.get("core_sha256"):
        raise Figure2Error(
            "The loaded native core differs from the checked wheel. Reinstall that "
            "wheel in a fresh environment and rerun."
        )
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise Figure2Error(
            "The installed distribution has no direct_url.json. Install the checked "
            "wheel directly with pip and rerun."
        )
    direct_url = _require_mapping(json.loads(direct_url_text), "direct_url.json")
    direct_url_sha256 = _direct_url_sha256(direct_url)
    if direct_url_sha256 != wheel.get("sha256"):
        raise Figure2Error(
            "The installed distribution came from a different wheel. Install the "
            "checked wheel from the artifact bundle and rerun."
        )
    wheel_path = Path(str(wheel.get("path", ""))).resolve()
    verified_members: dict[str, str] = {}
    with zipfile.ZipFile(wheel_path) as archive:
        for member in sorted(archive.namelist()):
            if not member.startswith("pyfgsea/") or member.endswith("/"):
                continue
            installed_path = Path(distribution.locate_file(member)).resolve()
            if not installed_path.is_file():
                raise Figure2Error(
                    f"The installed wheel file {member} is missing. Reinstall the "
                    "checked wheel in a fresh environment and rerun."
                )
            wheel_member_sha = _sha256_bytes(archive.read(member))
            if sha256_file(installed_path) != wheel_member_sha:
                raise Figure2Error(
                    f"The installed wheel file {member} has changed. Reinstall the "
                    "checked wheel in a fresh environment and rerun."
                )
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
        raise Figure2Error(
            "The installed PyFgsea file list differs from the wheel: "
            f"missing={missing!r}, extra={extra!r}. Reinstall the checked wheel "
            "in a fresh environment and rerun."
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
    parameter_record = _load_figure2_parameters()
    reference_contract = _load_reference_contract()
    artifact = _verify_artifact_receipt(artifact_receipt, git_state)
    installed = _verify_installed_pyfgsea(artifact)
    inputs = {
        "dataset": _verify_input(dataset, EXPECTED_DATASET_SHA256, "dataset"),
        "gene_sets": _verify_input(gene_sets, EXPECTED_GENE_SETS_SHA256, "gene sets"),
    }
    return {
        "git": git_state,
        "figure2_parameters": parameter_record,
        "reference_contract": reference_contract,
        "artifact_receipt": artifact,
        "installed_artifact": installed,
        "inputs": inputs,
        "parameters": dict(RECORDED_PARAMETERS),
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


def _validate_results(results: pd.DataFrame) -> dict[str, Any]:
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
        raise Figure2Error(
            f"The trajectory output is missing columns: {missing}. Return every "
            "required Figure 2 column and rerun."
        )
    if len(results) != EXPECTED_N_ROWS:
        raise Figure2Error(
            f"The trajectory output has {len(results)} rows, not {EXPECTED_N_ROWS}. "
            "Use the recorded window and pathway inputs and rerun."
        )
    duplicate_count = int(results.duplicated(["window_id", "Pathway"]).sum())
    if duplicate_count:
        raise Figure2Error(
            f"The trajectory output has {duplicate_count} duplicate window-pathway "
            "keys. Remove the duplicates at their source and rerun."
        )
    window_ids = sorted(int(value) for value in results["window_id"].unique())
    if window_ids != list(range(EXPECTED_N_WINDOWS)):
        raise Figure2Error(
            "The trajectory output does not contain window IDs 0 through 61. Check "
            "the recorded window parameters and input data, then rerun."
        )
    if int(results["Pathway"].nunique()) != EXPECTED_N_PATHWAYS:
        raise Figure2Error(
            "The trajectory output does not contain exactly 43 pathways. Use the "
            "recorded gene-set file and parameters, then rerun."
        )
    per_window = results.groupby("window_id")["Pathway"].nunique()
    per_pathway = results.groupby("Pathway")["window_id"].nunique()
    if not (per_window == EXPECTED_N_PATHWAYS).all() or not (
        per_pathway == EXPECTED_N_WINDOWS
    ).all():
        raise Figure2Error(
            "The trajectory output is not a complete 62 by 43 window-pathway grid. "
            "Check the recorded inputs and parameters, then rerun."
        )
    status_counts = {
        str(key): int(value)
        for key, value in results["status"].value_counts(dropna=False).sort_index().items()
    }
    if status_counts != {"resolved": EXPECTED_N_ROWS}:
        raise Figure2Error(
            f"Figure 2 contains unresolved rows: {status_counts}. Resolve the "
            "reported numerical failures before rerunning."
        )
    pvalues = results["P-value"].to_numpy(dtype=np.float64)
    padj = results["padj"].to_numpy(dtype=np.float64)
    nes = results["NES"].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(pvalues).all()
        or (pvalues <= 0).any()
        or (pvalues > 1).any()
    ):
        raise Figure2Error(
            "Figure 2 contains non-finite or out-of-range p-values. Fix the "
            "calculation so every p-value is in (0, 1], then rerun."
        )
    if (
        not np.isfinite(padj).all()
        or (padj < 0).any()
        or (padj > 1).any()
        or not np.isfinite(nes).all()
    ):
        raise Figure2Error(
            "Figure 2 contains an invalid adjusted p-value or NES. Fix the "
            "calculation so padj is in [0, 1] and NES is finite, then rerun."
        )

    independent = np.empty(len(results), dtype=np.float64)
    for indices in results.groupby("window_id", sort=False).indices.values():
        independent[indices] = bh_adjust(results.iloc[indices]["P-value"])
    max_abs_diff = float(np.max(np.abs(padj - independent)))
    if max_abs_diff > BH_ATOL:
        raise Figure2Error(
            f"The recorded adjusted p-values differ from an independent within-window "
            f"BH calculation by {max_abs_diff}, above {BH_ATOL}. Fix the adjustment "
            "calculation and rerun."
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


# Compatibility for callers that used the previous helper name.
_audit_results = _validate_results


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
        raise Figure2Error(
            f"Figure 2 is missing required pathways: {missing}. Use the recorded "
            "gene-set file and rerun."
        )
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
    fig.suptitle("PyFgsea 0.2.0rc8 Figure 2 reproduction")
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
            raise Figure2Error(
                f"The GSE155254 matrix shape is {adata.shape}, not "
                f"{EXPECTED_DATASET_SHAPE}. Use the recorded processed dataset and rerun."
            )
        if PARAMETERS["pseudotime_key"] not in adata.obs:
            raise Figure2Error(
                "The processed dataset has no dpt_pseudotime column. Use the recorded "
                "processed dataset or restore that column and rerun."
            )
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
            raise Figure2Error(
                "The trajectory calculation returned no pathways. Check the recorded "
                "gene sets and parameters, then rerun."
            )
        result_validation = _validate_results(results)
        results = results.sort_values(
            ["window_id", "padj", "P-value", "Pathway"], kind="mergesort"
        ).reset_index(drop=True)
        summary = pathway_summary(results)
        targets = summary.loc[summary["Pathway"].isin(TARGET_PATHWAYS)].copy()
        if len(targets) != len(TARGET_PATHWAYS):
            raise Figure2Error(
                "The Figure 2 target summary is incomplete. Restore both recorded "
                "target pathways in the gene-set file and rerun."
            )

        result_path = incomplete_dir / "trajectory_results.csv"
        summary_path = incomplete_dir / "pathway_summary.csv"
        target_path = incomplete_dir / "figure2_target_pathway_summary.csv"
        png_path = incomplete_dir / "figure2.png"
        pdf_path = incomplete_dir / "figure2.pdf"
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
        end_parameters = _load_figure2_parameters()
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
        if end_parameters != preflight["figure2_parameters"]:
            raise Figure2Error(
                "Figure 2 parameters changed during the run. Restore the recorded "
                "parameter file and rerun from a clean checkout."
            )
        if end_reference_contract != preflight["reference_contract"]:
            raise Figure2Error(
                "The fgsea reference profile changed while Figure 2 was running. "
                "Restore reference_manifest.json and rerun from a clean checkout."
            )
        if end_artifact != preflight["artifact_receipt"]:
            raise Figure2Error(
                "The artifact receipt, sdist, or wheel changed while Figure 2 was "
                "running. Restore the downloaded bundle and rerun."
            )
        if end_installed != preflight["installed_artifact"]:
            raise Figure2Error(
                "The installed PyFgsea environment changed while Figure 2 was running. "
                "Create a fresh environment, install the checked wheel, and rerun."
            )
        if end_inputs != preflight["inputs"]:
            raise Figure2Error(
                "The dataset or gene-set file changed while Figure 2 was running. "
                "Restore the recorded inputs and rerun."
            )
        if end_environment != preflight["environment_versions"]:
            raise Figure2Error(
                "A dependency version changed while Figure 2 was running. Recreate "
                "the environment with fixed package versions and rerun."
            )
        if frozen_inputs["dataset"]["sha256"] != end_inputs["dataset"]["sha256"]:
            raise Figure2Error(
                "The copied dataset differs from the source dataset. Restore the "
                "recorded input file and rerun."
            )
        if frozen_inputs["gene_sets"]["sha256"] != end_inputs["gene_sets"]["sha256"]:
            raise Figure2Error(
                "The copied gene-set file differs from the source file. Restore the "
                "recorded input file and rerun."
            )

        release_tag_name = str(start_git["release_tag"]["name"])
        manifest = {
            "schema_version": 2,
            "artifact_type": "pyfgsea-figure2-reproduction",
            "verification_status": "verified",
            "figure2_parameters": {
                "path": preflight["figure2_parameters"]["path"],
                "sha256": preflight["figure2_parameters"]["sha256"],
                "dataset": preflight["figure2_parameters"]["record"]["dataset"],
                "recorded_on": preflight["figure2_parameters"]["record"][
                    "recorded_on"
                ],
            },
            "reference_contract": preflight["reference_contract"],
            "verification_basis": f"clean-{release_tag_name}-and-verified-wheel",
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
            "parameters": dict(RECORDED_PARAMETERS),
            "inputs": preflight["inputs"],
            "frozen_input_artifacts": frozen_input_artifacts,
            "dataset_shape": [int(adata.n_obs), int(adata.n_vars)],
            "result_validation": result_validation,
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
        help=(
            "Verify Git, parameters, inputs, and the installed wheel without "
            "computing GSEA."
        ),
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
    print(f"Figure 2 outputs written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
