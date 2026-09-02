#!/usr/bin/env python3
"""Run one exact PyFgsea/R-fgsea Figure 1 lane from immutable inputs."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping
from urllib.parse import unquote, urlparse

# Configure the single-thread evidence process before importing NumPy or the
# Rust extension, both of which may initialize a global worker pool on import.
for _thread_variable in (
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

try:
    import psutil
except ImportError:  # pragma: no cover - checked before the run
    psutil = None  # type: ignore[assignment]

try:
    from .common import (
        EPS,
        GSEA_PARAMETERS,
        LANE_CONTRACTS,
        LEGACY_PYFGSEA_COMMIT,
        LEGACY_PYFGSEA_TREE,
        LEGACY_PYPI_WHEEL_SHA256,
        LOG10_FLOOR,
        PUBLICATION_SOURCE_COMMIT,
        REFERENCE_ARTIFACT_CONTRACTS,
        SCENARIOS,
        SUITE_VERSION,
        assert_finite_range,
        ensure_empty_output_dir,
        file_record,
        read_json,
        require_sha256,
        sha256_file,
        verify_clean_git_checkout,
        verify_file_record,
        verify_git_unchanged,
        write_json,
    )
except ImportError:  # pragma: no cover - direct script execution
    from common import (  # type: ignore
        EPS,
        GSEA_PARAMETERS,
        LANE_CONTRACTS,
        LEGACY_PYFGSEA_COMMIT,
        LEGACY_PYFGSEA_TREE,
        LEGACY_PYPI_WHEEL_SHA256,
        LOG10_FLOOR,
        PUBLICATION_SOURCE_COMMIT,
        REFERENCE_ARTIFACT_CONTRACTS,
        SCENARIOS,
        SUITE_VERSION,
        assert_finite_range,
        ensure_empty_output_dir,
        file_record,
        read_json,
        require_sha256,
        sha256_file,
        verify_clean_git_checkout,
        verify_file_record,
        verify_git_unchanged,
        write_json,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
R_HELPER = SCRIPT_DIR / "run_reference.R"
EXPECTED_PUBLICATION_PARAMETERS = {"n_genes": 12000, "n_sets": 100, "seed": 42}
EXPECTED_TIES_PARAMETERS = {
    "n_genes": 4000,
    "n_sets": 60,
    "seed": 4242,
    "score_round_decimals": 1,
}
EXPECTED_HISTORICAL_SOURCE_PATH = (
    "bioinfor0208/revision/generate_revision_assets.py"
)
EXPECTED_HISTORICAL_GENERATOR_CALL = (
    "generate_test_data(n_genes=12000,n_sets=100,seed=42)"
)
EXPECTED_SCENARIO_INVARIANTS = {
    "publication_main": {
        "gene_count": 12000,
        "pathway_count": 100,
        "minimum_pathway_size": 15,
        "maximum_pathway_size": 199,
        "tied_score_group_count": 0,
        "tied_gene_count": 0,
        "maximum_tie_multiplicity": 1,
    },
    "ties_predeclared": {
        "gene_count": 4000,
        "pathway_count": 60,
        "minimum_pathway_size": 16,
        "maximum_pathway_size": 192,
        "tied_score_group_count": 86,
        "tied_gene_count": 3994,
        "maximum_tie_multiplicity": 141,
    },
}


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _resolve_current_bundle_file(
    receipt_path: Path, record: Mapping[str, Any], label: str
) -> Path:
    if receipt_path.parent.name != "evidence":
        raise ValueError(
            "current receipt must use the <bundle>/evidence/receipt.json layout"
        )
    raw = record.get("bundle_path")
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise ValueError(f"{label} has no safe POSIX bundle_path")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError(f"{label} has an unsafe bundle_path: {raw!r}")
    root = receipt_path.parent.parent.resolve()
    resolved = root.joinpath(*relative.parts).resolve()
    if not _is_within(resolved, root):
        raise ValueError(f"{label} escapes the downloaded artifact bundle")
    return resolved


def _file_url_to_path(url: str) -> Path:
    parsed = urlparse(url)
    if parsed.scheme != "file":
        raise RuntimeError(f"wheel direct_url is not a local file URL: {url}")
    decoded = unquote(parsed.path)
    if (
        os.name == "nt"
        and len(decoded) >= 3
        and decoded[0] == "/"
        and decoded[2] == ":"
    ):
        decoded = decoded[1:]
    if parsed.netloc and parsed.netloc not in {"", "localhost"}:
        decoded = f"//{parsed.netloc}{decoded}"
    return Path(decoded).resolve()


def _direct_url_sha256(payload: Mapping[str, Any]) -> str:
    archive = payload.get("archive_info")
    if not isinstance(archive, Mapping):
        raise RuntimeError("installed distribution direct_url lacks archive_info")
    hashes = archive.get("hashes")
    if isinstance(hashes, Mapping) and isinstance(hashes.get("sha256"), str):
        return require_sha256(str(hashes["sha256"]), label="direct_url wheel hash")
    raw_hash = archive.get("hash")
    if isinstance(raw_hash, str) and raw_hash.startswith("sha256="):
        return require_sha256(raw_hash.split("=", 1)[1], label="direct_url wheel hash")
    raise RuntimeError("installed distribution direct_url lacks a SHA-256 wheel hash")


def _verify_artifact_receipt(
    lane: str, receipt_path: Path, evidence_git: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve wheel/core hashes only from an upstream artifact-chain receipt."""

    receipt = read_json(receipt_path)
    if receipt.get("schema_version") != 1 or receipt.get("status") != "passed":
        raise ValueError("artifact report schema or status is invalid")
    if lane == "legacy":
        if receipt.get("kind") != "figure1_legacy_artifact_receipt":
            raise ValueError("legacy reference run requires a legacy artifact report")
        if receipt.get("suite_version") != SUITE_VERSION:
            raise ValueError("legacy artifact receipt belongs to a different suite")
        if receipt.get("all_legacy_artifact_gates_passed") is not True:
            raise ValueError("legacy artifact report did not pass all checks")
        git = receipt.get("git")
        if not isinstance(git, Mapping):
            raise ValueError("legacy artifact receipt lacks Git identity")
        if (
            git.get("commit") != LEGACY_PYFGSEA_COMMIT
            or git.get("tree") != LEGACY_PYFGSEA_TREE
            or git.get("tag") != "v0.1.4"
            or git.get("clean") is not True
        ):
            raise ValueError("legacy artifact is not bound to the clean v0.1.4 tag")
        if receipt.get("evidence_git") != dict(evidence_git):
            raise ValueError(
                "legacy artifact verifier used a different evidence checkout"
            )
        expected = receipt.get("expected")
        if expected != {
            "distribution_version": "0.1.4",
            "module_declared_version": "0.1.3",
            "algorithm_revision_contract": "legacy-no-revision-api",
        }:
            raise ValueError("legacy artifact version/revision contract is invalid")
        script = receipt.get("script")
        if not isinstance(script, Mapping) or script.get("sha256") != sha256_file(
            SCRIPT_DIR / "verify_legacy_artifact.py"
        ):
            raise ValueError("legacy artifact receipt used a different verifier script")
        wheel = receipt.get("wheel")
        if not isinstance(wheel, Mapping):
            raise ValueError("legacy artifact receipt lacks wheel evidence")
        if (
            wheel.get("source_set_exact") is not True
            or wheel.get("source_bytes_equal_after_crlf_normalization") is not True
            or wheel.get("record_hashes_and_sizes_valid") is not True
        ):
            raise ValueError("legacy wheel is not source-bound to v0.1.4")
        wheel_hash = require_sha256(str(wheel.get("sha256", "")), label="legacy wheel")
        if (
            LEGACY_PYPI_WHEEL_SHA256.get(str(wheel.get("filename"))) != wheel_hash
            or wheel.get("authoritative_pypi_sha256") != wheel_hash
        ):
            raise ValueError("legacy wheel is not an official PyPI 0.1.4 artifact")
        core_hash = require_sha256(
            str(wheel.get("core_sha256", "")), label="legacy core"
        )
        source_commit = LEGACY_PYFGSEA_COMMIT
        source_tree = LEGACY_PYFGSEA_TREE
        receipt_kind = str(receipt["kind"])
    else:
        if receipt.get("all_artifact_chain_gates_passed") is not True:
            raise ValueError(
                "current artifact report did not pass all package checks"
            )
        expected = receipt.get("expected")
        expected_fields = {
            "cargo_version": "0.2.0-rc7",
            "pyfgsea_version": "0.2.0rc7",
            "algorithm_revision": "fgsea-1.38-pr178-v1",
        }
        if not isinstance(expected, Mapping) or any(
            expected.get(key) != value for key, value in expected_fields.items()
        ):
            raise ValueError("current artifact version/revision contract is invalid")
        git = receipt.get("git")
        if not isinstance(git, Mapping):
            raise ValueError("current artifact receipt lacks Git identity")
        if (
            git.get("commit") != evidence_git["commit"]
            or git.get("tree") != evidence_git["tree"]
            or git.get("head_matches_commit") is not True
            or git.get("clean_before_and_after") is not True
        ):
            raise ValueError(
                "current artifact is not bound to the evidence commit/tree"
            )
        release_tag = git.get("release_tag")
        expected_tag_fields = {
            "name": evidence_git["tag"],
            "annotated": True,
            "tag_object": evidence_git["tag_object"],
            "peeled_commit": evidence_git["commit"],
        }
        if not isinstance(release_tag, Mapping) or any(
            release_tag.get(key) != value
            for key, value in expected_tag_fields.items()
        ):
            raise ValueError(
                "current artifact receipt is not bound to the annotated evidence tag"
            )
        chain = receipt.get("artifact_chain")
        if not isinstance(chain, Mapping):
            raise ValueError("current artifact receipt lacks its artifact chain")
        sdist, wheel, installed = (
            chain.get("sdist"),
            chain.get("wheel"),
            chain.get("installed"),
        )
        if not all(isinstance(item, Mapping) for item in (sdist, wheel, installed)):
            raise ValueError("current artifact chain is incomplete")
        assert isinstance(sdist, Mapping)
        assert isinstance(wheel, Mapping)
        assert isinstance(installed, Mapping)
        if (
            sdist.get("native_binary_count") != 0
            or sdist.get("pyfgsea_source_set_exact") is not True
            or wheel.get("pyfgsea_source_set_exact") is not True
            or wheel.get("wheel_member_boundary_exact") is not True
            or wheel.get("wheel_built_from_verified_sdist") is not True
            or wheel.get("build_input_sdist_sha256") != sdist.get("sha256")
            or wheel.get("verified_source_manifest_sha256")
            != sdist.get("verified_source_manifest_sha256")
        ):
            raise ValueError("current sdist-to-wheel source chain is not exact")
        wheel_hash = require_sha256(str(wheel.get("sha256", "")), label="current wheel")
        core_hash = require_sha256(
            str(wheel.get("core_sha256", "")), label="current core"
        )
        if (
            installed.get("direct_url_wheel_sha256") != wheel_hash
            or installed.get("core_sha256") != core_hash
            or installed.get("package_and_core_inside_venv") is not True
        ):
            raise ValueError("current clean-install evidence differs from its wheel")
        installed_tests = receipt.get("installed_tests")
        if not isinstance(installed_tests, Mapping):
            raise ValueError("current artifact receipt lacks installed-test evidence")
        expected_test_fields = {
            "status": "passed",
            "git_commit": evidence_git["commit"],
            "test_paths": ["tests", "repro/figure1_dual_lane/test_pipeline.py"],
            "pytest_version": "8.4.2",
            "trajectory_extra_installed": True,
            "isolated_python": True,
            "import_mode": "importlib",
            "cwd_outside_worktree": True,
            "wheel_sha256": wheel_hash,
        }
        if any(
            installed_tests.get(key) != value
            for key, value in expected_test_fields.items()
        ):
            raise ValueError("current installed-test contract is incomplete")
        test_manifest = installed_tests.get("test_source_manifest")
        if not isinstance(test_manifest, Mapping) or not test_manifest:
            raise ValueError("current installed-test source manifest is empty")
        require_sha256(
            str(installed_tests.get("test_source_manifest_sha256", "")),
            label="installed-test source manifest",
        )
        counts = installed_tests.get("counts")
        if not isinstance(counts, Mapping) or (
            counts.get("total", 0) <= 0
            or counts.get("failed") != 0
            or counts.get("errors") != 0
            or counts.get("passed", 0) + counts.get("skipped", 0)
            != counts.get("total")
        ):
            raise ValueError("current installed-test receipt is not passing")
        junit = installed_tests.get("junit")
        if not isinstance(junit, Mapping):
            raise ValueError("current installed-test receipt lacks JUnit evidence")
        junit_path = _resolve_current_bundle_file(
            receipt_path, junit, "installed-test JUnit"
        )
        junit_hash = require_sha256(
            str(junit.get("sha256", "")), label="installed-test JUnit"
        )
        if (
            not junit_path.is_file()
            or sha256_file(junit_path) != junit_hash
            or junit_path.stat().st_size != junit.get("bytes")
        ):
            raise ValueError("current installed-test JUnit is missing or changed")
        source_commit = str(git["commit"])
        source_tree = str(git["tree"])
        receipt_kind = "pyfgsea_release_artifact_chain"

    return {
        "receipt_path": str(receipt_path.resolve()),
        "receipt_sha256": sha256_file(receipt_path),
        "receipt_kind": receipt_kind,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "release_tag": evidence_git["tag"],
        "release_tag_object": evidence_git["tag_object"],
        "wheel_sha256": wheel_hash,
        "core_sha256": core_hash,
    }


def _verify_reference_receipt(
    lane: str, receipt_path: Path, evidence_git: Mapping[str, Any]
) -> dict[str, Any]:
    """Check the R runner against one OCI build report and image tarball."""

    contract = REFERENCE_ARTIFACT_CONTRACTS[lane]
    lane_contract = LANE_CONTRACTS[lane]
    receipt = read_json(receipt_path)
    if (
        receipt.get("schema_version") != 1
        or receipt.get("status") != "passed"
        or receipt.get("profile") != contract["profile"]
    ):
        raise ValueError("reference OCI receipt has the wrong status/profile")
    if (
        receipt.get("git_commit") != evidence_git["commit"]
        or receipt.get("git_tree") != evidence_git["tree"]
        or receipt.get("release_tag") != evidence_git["tag"]
        or receipt.get("release_tag_object") != evidence_git["tag_object"]
        or receipt.get("platform") != "linux/amd64"
    ):
        raise ValueError(
            "reference OCI receipt is not bound to this evidence tree/platform"
        )
    dockerfile = REPO_ROOT / contract["dockerfile"]
    if (
        receipt.get("dockerfile") != contract["dockerfile"]
        or receipt.get("dockerfile_sha256") != sha256_file(dockerfile)
        or receipt.get("base_image_digest") != contract["base_image_digest"]
    ):
        raise ValueError("reference OCI Dockerfile/base digest contract differs")
    built_digest = str(receipt.get("built_image_digest", ""))
    if not built_digest.startswith("sha256:"):
        raise ValueError("reference OCI receipt lacks a built image digest")
    require_sha256(built_digest.split(":", 1)[1], label="built OCI image digest")
    archive = receipt.get("oci_archive")
    if not isinstance(archive, Mapping):
        raise ValueError("reference OCI receipt lacks archive evidence")
    require_sha256(str(archive.get("sha256", "")), label="OCI archive")
    verification = receipt.get("reference_verification")
    expected_verification = {
        "REFERENCE_ID": contract["reference_id"],
        "R_VERSION": lane_contract["r_version"],
        "BIOCONDUCTOR_VERSION": lane_contract["bioconductor_version"],
        "FGSEA_VERSION": lane_contract["fgsea_version"],
    }
    if not isinstance(verification, Mapping) or any(
        verification.get(key) != value for key, value in expected_verification.items()
    ):
        raise ValueError("reference OCI in-image verification differs from the lane")
    evidence_files = receipt.get("evidence_files")
    if not isinstance(evidence_files, Mapping):
        raise ValueError("reference OCI receipt lacks extracted evidence hashes")
    for required in (
        "fgsea.tar.gz",
        "renv.resolved.lock",
        "reference-verification.txt",
        "R-sessionInfo.txt",
        "dpkg-packages.tsv",
        "evidence-sha256.txt",
    ):
        record = evidence_files.get(required)
        if not isinstance(record, Mapping):
            raise ValueError(f"reference OCI receipt lacks {required}")
        require_sha256(str(record.get("sha256", "")), label=f"OCI {required}")
    if (
        evidence_files["fgsea.tar.gz"].get("sha256") != contract["fgsea_tarball_sha256"]
        or receipt.get("registry_push_performed") is not False
    ):
        raise ValueError("reference fgsea tarball hash or no-push contract differs")

    runtime_reference_id = os.environ.get("FGSEA_REFERENCE_ID", "").strip()
    runtime_digest = os.environ.get("FGSEA_REFERENCE_IMAGE_DIGEST", "").strip()
    if platform.system() != "Linux" or platform.machine().lower() not in {
        "x86_64",
        "amd64",
    }:
        raise RuntimeError("reference OCI lane must execute on linux/amd64")
    if (
        runtime_reference_id != contract["reference_id"]
        or runtime_digest != built_digest
    ):
        raise RuntimeError(
            "run inside the receipt-bound OCI environment and set its reference ID/digest"
        )
    evidence_root = Path("/opt/reference")
    if not evidence_root.is_dir():
        raise RuntimeError(
            "reference OCI evidence directory is missing: /opt/reference"
        )
    runtime_evidence: dict[str, dict[str, str]] = {}
    for filename in (
        "fgsea.tar.gz",
        "renv.resolved.lock",
        "reference-verification.txt",
        "R-sessionInfo.txt",
        "dpkg-packages.tsv",
        "evidence-sha256.txt",
    ):
        evidence_path = evidence_root / filename
        actual_hash = sha256_file(evidence_path)
        if actual_hash != evidence_files[filename]["sha256"]:
            raise RuntimeError(
                f"runtime /opt/reference/{filename} differs from the OCI receipt"
            )
        runtime_evidence[filename] = {
            "path": str(evidence_path),
            "sha256": actual_hash,
        }
    tarball_path = evidence_root / "fgsea.tar.gz"
    return {
        "receipt_path": str(receipt_path.resolve()),
        "receipt_sha256": sha256_file(receipt_path),
        "profile": contract["profile"],
        "release_tag": evidence_git["tag"],
        "release_tag_object": evidence_git["tag_object"],
        "built_image_digest": built_digest,
        "oci_archive_sha256": str(archive["sha256"]),
        "fgsea_tarball_path": str(tarball_path),
        "fgsea_tarball_sha256": contract["fgsea_tarball_sha256"],
        "runtime_evidence": runtime_evidence,
    }


def verify_installed_lane(
    lane: str, *, expected_wheel_sha256: str, expected_core_sha256: str
) -> tuple[Any, dict[str, Any]]:
    """Check the imported module and native core against the installed wheel."""

    contract = LANE_CONTRACTS[lane]
    expected_wheel = require_sha256(expected_wheel_sha256, label="expected wheel hash")
    expected_core = require_sha256(expected_core_sha256, label="expected core hash")

    try:
        distribution = importlib.metadata.distribution("pyfgsea")
        import pyfgsea
        from pyfgsea import wrapper
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise RuntimeError("an installed PyFgsea wheel is required") from exc

    distribution_version = distribution.version
    if distribution_version != contract["pyfgsea_distribution_version"]:
        raise RuntimeError(
            f"distribution mismatch: expected {contract['pyfgsea_distribution_version']}, "
            f"found {distribution_version}"
        )
    module_version = str(getattr(pyfgsea, "__version__", "0+unknown"))
    if module_version != contract["pyfgsea_module_version"]:
        raise RuntimeError(
            f"module version mismatch: expected {contract['pyfgsea_module_version']}, "
            f"found {module_version}"
        )

    module_path = Path(pyfgsea.__file__).resolve()
    core = getattr(wrapper, "_ext", None)
    core_file = getattr(core, "__file__", None)
    if core_file is None:
        raise RuntimeError("PyFgsea wrapper did not expose a file-backed Rust core")
    core_path = Path(core_file).resolve()
    if _is_within(module_path, REPO_ROOT) or _is_within(core_path, REPO_ROOT):
        raise RuntimeError(
            "PyFgsea resolved from the evidence source checkout; install and import "
            "the wheel in a clean environment instead"
        )
    distribution_files = distribution.files
    if distribution_files is None:
        raise RuntimeError("installed distribution does not expose its file manifest")
    installed_paths = {
        Path(distribution.locate_file(item)).resolve() for item in distribution_files
    }
    if module_path not in installed_paths or core_path not in installed_paths:
        raise RuntimeError(
            "imported PyFgsea module/core do not belong to the selected distribution"
        )
    actual_core = sha256_file(core_path)
    if actual_core != expected_core:
        raise RuntimeError(
            f"native core hash mismatch: expected {expected_core}, found {actual_core}"
        )

    direct_url_text = distribution.read_text("direct_url.json")
    if not direct_url_text:
        raise RuntimeError(
            "installed distribution has no direct_url.json; install the exact local wheel"
        )
    direct_url = json.loads(direct_url_text)
    if not isinstance(direct_url, dict) or not isinstance(direct_url.get("url"), str):
        raise RuntimeError("installed distribution direct_url.json is invalid")
    recorded_wheel = _direct_url_sha256(direct_url)
    if recorded_wheel != expected_wheel:
        raise RuntimeError(
            f"installed wheel hash mismatch: expected {expected_wheel}, found {recorded_wheel}"
        )
    wheel_path = _file_url_to_path(str(direct_url["url"]))
    if wheel_path.suffix != ".whl" or not wheel_path.is_file():
        raise RuntimeError(f"direct_url wheel is missing or not a wheel: {wheel_path}")
    if sha256_file(wheel_path) != expected_wheel:
        raise RuntimeError("local wheel bytes no longer match direct_url.json")

    wrapper_revision_reader = getattr(wrapper, "_algorithm_revision", None)
    core_revision_reader = getattr(core, "algorithm_revision", None)
    if lane == "legacy":
        if callable(wrapper_revision_reader) or callable(core_revision_reader):
            raise RuntimeError(
                "legacy v0.1.4 contract expects the historical core with no revision API"
            )
        revision: str | None = None
    else:
        if not callable(wrapper_revision_reader) or not callable(core_revision_reader):
            raise RuntimeError("current core revision APIs are missing")
        wrapper_revision = str(wrapper_revision_reader())
        core_revision = str(core_revision_reader())
        if wrapper_revision != core_revision:
            raise RuntimeError("wrapper and core algorithm revisions disagree")
        expected_revision = contract["algorithm_revision"]
        if wrapper_revision != expected_revision:
            raise RuntimeError(
                f"algorithm revision mismatch: expected {expected_revision}, "
                f"found {wrapper_revision}"
            )
        revision = wrapper_revision

    identity = {
        "lane": lane,
        "distribution_version": distribution_version,
        "module_declared_version": module_version,
        "historical_version_mismatch_expected": lane == "legacy",
        "module_path": str(module_path),
        "module_sha256": sha256_file(module_path),
        "core_path": str(core_path),
        "core_sha256": actual_core,
        "algorithm_revision": revision,
        "algorithm_revision_contract": (
            "legacy-no-revision-api" if lane == "legacy" else revision
        ),
        "wheel_path": str(wheel_path),
        "wheel_sha256": expected_wheel,
        "direct_url": direct_url,
    }
    return pyfgsea, identity


def _validate_input_manifest(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Path]]]:
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema_version") != 2
        or manifest.get("kind") != "figure1_input_manifest"
    ):
        raise ValueError("input manifest has the wrong schema or kind")
    if manifest.get("suite_version") != SUITE_VERSION:
        raise ValueError("input manifest belongs to a different suite version")
    historical = manifest.get("historical_generator")
    if not isinstance(historical, Mapping):
        raise ValueError("input manifest lacks historical generator metadata")
    expected_historical_fields = {
        "source_commit": PUBLICATION_SOURCE_COMMIT,
        "source_path": EXPECTED_HISTORICAL_SOURCE_PATH,
        "generator_call": EXPECTED_HISTORICAL_GENERATOR_CALL,
    }
    if any(
        historical.get(key) != value
        for key, value in expected_historical_fields.items()
    ):
        raise ValueError("input manifest cites the wrong historical generator")
    generator = manifest.get("generator")
    expected_generator_hash = sha256_file(SCRIPT_DIR / "prepare_inputs.py")
    if (
        not isinstance(generator, Mapping)
        or generator.get("mode") != "copy_commit_bound_frozen_bytes"
        or generator.get("script_sha256") != expected_generator_hash
        or generator.get("frozen_input_root")
        != "repro/figure1_dual_lane/frozen_inputs"
    ):
        raise ValueError(
            "input manifest was not generated by this suite's input script"
        )
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, Mapping) or set(scenarios) != set(SCENARIOS):
        raise ValueError(f"input manifest must contain exactly {SCENARIOS}")

    expected_parameters = {
        "publication_main": EXPECTED_PUBLICATION_PARAMETERS,
        "ties_predeclared": EXPECTED_TIES_PARAMETERS,
    }
    expected_transforms = {
        "publication_main": (
            "frozen_bytes_canonicalized_to_12_significant_decimal_digits"
        ),
        "ties_predeclared": "frozen_bytes_round_binary64_to_1_decimal",
    }
    expected_transform_parameters = {
        "publication_main": {"significant_decimal_digits": 12},
        "ties_predeclared": {"round_decimal_places": 1},
    }
    resolved: dict[str, dict[str, Path]] = {}
    for name in SCENARIOS:
        scenario = scenarios[name]
        if not isinstance(scenario, Mapping):
            raise ValueError(f"scenario record is invalid: {name}")
        if scenario.get("parameters") != expected_parameters[name]:
            raise ValueError(f"scenario parameters differ from the contract: {name}")
        if scenario.get("invariants") != EXPECTED_SCENARIO_INVARIANTS[name]:
            raise ValueError(f"scenario invariants differ from the contract: {name}")
        if (
            scenario.get("materialization") != "copy_commit_bound_frozen_bytes"
            or scenario.get("score_transform") != expected_transforms[name]
            or scenario.get("score_transform_parameters")
            != expected_transform_parameters[name]
        ):
            raise ValueError(f"scenario freeze contract differs: {name}")
        ranks_record = scenario.get("ranks")
        pathways_record = scenario.get("pathways")
        frozen_source = scenario.get("frozen_source")
        if not isinstance(ranks_record, Mapping) or not isinstance(
            pathways_record, Mapping
        ) or not isinstance(frozen_source, Mapping):
            raise ValueError(f"scenario input records are invalid: {name}")
        if ranks_record.get("path") != f"{name}/ranks.csv" or pathways_record.get(
            "path"
        ) != f"{name}/pathways.gmt":
            raise ValueError(f"scenario materialized input paths differ: {name}")
        resolved[name] = {
            "ranks": verify_file_record(
                manifest_path.parent, ranks_record, label=f"{name} ranks"
            ),
            "pathways": verify_file_record(
                manifest_path.parent, pathways_record, label=f"{name} pathways"
            ),
        }
        for label, resolved_path in resolved[name].items():
            filename = "ranks.csv" if label == "ranks" else "pathways.gmt"
            source_record = frozen_source.get(label)
            if not isinstance(source_record, Mapping) or source_record.get(
                "path"
            ) != f"{name}/{filename}":
                raise ValueError(f"{name}/{label} frozen source path differs")
            frozen_path = verify_file_record(
                SCRIPT_DIR / "frozen_inputs",
                source_record,
                label=f"{name} frozen {label}",
            )
            if resolved_path.read_bytes() != frozen_path.read_bytes():
                raise ValueError(
                    f"{name}/{label} differs from the commit-bound frozen bytes"
                )
    return manifest, resolved


def _memory_sampler(
    pid: int,
    stop: threading.Event,
    peak: list[int],
    samples: list[int],
    errors: list[str],
) -> None:
    if psutil is None:  # pragma: no cover - checked before the reference run
        raise RuntimeError("psutil is required for peak-RSS evidence")
    try:
        process = psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
        errors.append(repr(exc))
        return
    while not stop.is_set():
        try:
            total = process.memory_info().rss
            for child in process.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            peak[0] = max(peak[0], total)
            samples[0] += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
            if samples[0] == 0:
                errors.append(repr(exc))
            return
        stop.wait(0.02)


def _measure_call(function: Callable[[], Any]) -> tuple[Any, dict[str, float | int]]:
    if psutil is None:  # pragma: no cover - checked before the reference run
        raise RuntimeError("psutil is required for peak-RSS evidence")
    process = psutil.Process(os.getpid())
    baseline = process.memory_info().rss
    stop = threading.Event()
    peak = [baseline]
    samples = [0]
    errors: list[str] = []
    monitor = threading.Thread(
        target=_memory_sampler,
        args=(os.getpid(), stop, peak, samples, errors),
        daemon=True,
    )
    monitor.start()
    started = time.perf_counter()
    try:
        result = function()
    finally:
        elapsed = time.perf_counter() - started
        stop.set()
        monitor.join()
    if errors or samples[0] == 0:
        raise RuntimeError(f"Python peak-RSS sampler failed: {errors}")
    return result, {
        "elapsed_seconds": elapsed,
        "baseline_rss_bytes": baseline,
        "peak_rss_bytes": peak[0],
        "peak_increment_bytes": max(0, peak[0] - baseline),
    }


def _measure_process(
    command: list[str], *, environment: Mapping[str, str]
) -> tuple[subprocess.CompletedProcess[str], dict[str, float | int]]:
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(environment),
    )
    stop = threading.Event()
    peak = [0]
    samples = [0]
    errors: list[str] = []
    monitor = threading.Thread(
        target=_memory_sampler,
        args=(process.pid, stop, peak, samples, errors),
        daemon=True,
    )
    monitor.start()
    stdout, stderr = process.communicate()
    elapsed = time.perf_counter() - started
    stop.set()
    monitor.join()
    if errors or samples[0] == 0 or peak[0] <= 0:
        raise RuntimeError(f"R peak-RSS sampler failed: {errors}")
    completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    return completed, {
        "elapsed_seconds": elapsed,
        "baseline_rss_bytes": 0,
        "peak_rss_bytes": peak[0],
        "peak_increment_bytes": peak[0],
    }


def _load_rank_and_gmt(
    ranks_path: Path, pathways_path: Path
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    ranks = pd.read_csv(ranks_path)
    if list(ranks.columns) != ["Gene", "Score"]:
        raise ValueError("rank input must have exactly Gene,Score columns")
    if ranks["Gene"].isna().any() or ranks["Gene"].duplicated().any():
        raise ValueError("rank genes must be nonmissing and unique")
    assert_finite_range(ranks["Score"].to_numpy(dtype=float), label="rank score")

    pathways: dict[str, list[str]] = {}
    with pathways_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\n\r").split("\t")
            if len(parts) < 3:
                raise ValueError(f"invalid GMT line {line_number}")
            name, members = parts[0], parts[2:]
            if name in pathways:
                raise ValueError(f"duplicate pathway name in GMT: {name}")
            if len(members) != len(set(members)):
                raise ValueError(f"duplicate members in GMT pathway: {name}")
            pathways[name] = members
    if not pathways:
        raise ValueError("GMT contains no pathways")
    return ranks, pathways


def _verify_scenario_invariants(
    name: str,
    ranks: pd.DataFrame,
    pathways: Mapping[str, list[str]],
    manifest: Mapping[str, Any],
) -> None:
    counts = ranks["Score"].value_counts()
    tied = counts[counts > 1]
    sizes = [len(set(members)) for members in pathways.values()]
    actual = {
        "gene_count": int(len(ranks)),
        "pathway_count": int(len(pathways)),
        "minimum_pathway_size": int(min(sizes)),
        "maximum_pathway_size": int(max(sizes)),
        "tied_score_group_count": int(len(tied)),
        "tied_gene_count": int(tied.sum()) if len(tied) else 0,
        "maximum_tie_multiplicity": int(tied.max()) if len(tied) else 1,
    }
    declared = manifest["scenarios"][name].get("invariants")
    if actual != declared:
        raise ValueError(f"{name} actual input invariants differ from its manifest")
    parameters = manifest["scenarios"][name]["parameters"]
    if actual["gene_count"] != parameters["n_genes"]:
        raise ValueError(f"{name} gene count differs from its generator parameters")


def _run_python_engine(
    pyfgsea: Any, lane: str, ranks: pd.DataFrame, pathways: dict[str, list[str]]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    common_arguments: dict[str, Any] = {
        "gene_col": "Gene",
        "score_col": "Score",
        "min_size": GSEA_PARAMETERS["min_size"],
        "max_size": GSEA_PARAMETERS["max_size"],
        "sample_size": GSEA_PARAMETERS["sample_size"],
        "seed": GSEA_PARAMETERS["python_seed"],
        "nperm_nes": GSEA_PARAMETERS["nperm_nes"],
        "eps": GSEA_PARAMETERS["eps"],
    }
    if lane == "current":
        common_arguments.update(
            {
                "score_type": "std",
                "mode": "aligned",
                "tie_policy": "gene_id",
                "bin_width": 0,
                "use_batched": True,
                "nperm_simple": 1000,
            }
        )
    signature = inspect.signature(pyfgsea.run_gsea)
    unsupported = sorted(set(common_arguments).difference(signature.parameters))
    if unsupported:
        raise RuntimeError(
            f"installed run_gsea lacks required arguments: {unsupported}"
        )
    result = pyfgsea.run_gsea(ranks, pathways, **common_arguments)
    if not isinstance(result, pd.DataFrame) or result.empty:
        raise RuntimeError("PyFgsea returned no pathway results")
    required = ["Pathway", "ES", "NES", "P-value", "padj", "Size"]
    missing = sorted(set(required).difference(result.columns))
    if missing:
        raise RuntimeError(f"PyFgsea result is missing columns: {missing}")
    selected = result[required].rename(
        columns={
            "Pathway": "pathway",
            "ES": "py_es",
            "NES": "py_nes",
            "P-value": "py_pval",
            "padj": "py_padj",
            "Size": "py_size",
        }
    )
    if "log_pval" in result.columns:
        selected["py_log_pval_native"] = pd.to_numeric(
            result["log_pval"], errors="raise"
        )
    else:
        selected["py_log_pval_native"] = np.log(
            np.maximum(pd.to_numeric(selected["py_pval"]), LOG10_FLOOR)
        )
    selected = selected.sort_values("pathway", kind="mergesort").reset_index(drop=True)
    effective_arguments: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if name in {"data", "gmt"}:
            continue
        value = common_arguments.get(name, parameter.default)
        if value is inspect.Parameter.empty:
            raise RuntimeError(f"run_gsea parameter has no recorded value: {name}")
        if value is None or isinstance(value, (str, int, float, bool)):
            effective_arguments[name] = value
        else:
            effective_arguments[name] = repr(value)
    return selected, {
        "passed_arguments": common_arguments,
        "effective_arguments": effective_arguments,
    }


def _validate_engine_table(frame: pd.DataFrame, *, prefix: str) -> None:
    if frame["pathway"].isna().any() or frame["pathway"].duplicated().any():
        raise ValueError(f"{prefix} pathway identifiers must be nonmissing and unique")
    assert_finite_range(frame[f"{prefix}_es"], label=f"{prefix} ES")
    assert_finite_range(frame[f"{prefix}_nes"], label=f"{prefix} NES")
    assert_finite_range(
        frame[f"{prefix}_pval"], label=f"{prefix} p-value", lower=0.0, upper=1.0
    )
    assert_finite_range(
        frame[f"{prefix}_padj"],
        label=f"{prefix} adjusted p-value",
        lower=0.0,
        upper=1.0,
    )
    assert_finite_range(frame[f"{prefix}_size"], label=f"{prefix} size", lower=1.0)
    native_log_column = f"{prefix}_log_pval_native"
    if native_log_column in frame.columns:
        assert_finite_range(frame[native_log_column], label=native_log_column)


def _read_r_environment(
    path: Path, contract: Mapping[str, str | None]
) -> dict[str, str]:
    frame = pd.read_csv(path, sep="\t", dtype=str)
    if list(frame.columns) != ["key", "value"] or frame["key"].duplicated().any():
        raise RuntimeError("R environment sidecar is invalid")
    details = dict(zip(frame["key"], frame["value"]))
    expected = {
        "r_version": contract["r_version"],
        "bioconductor_version": contract["bioconductor_version"],
        "fgsea_version": contract["fgsea_version"],
        "r_seed": str(GSEA_PARAMETERS["r_seed"]),
        "score_type": "std",
    }
    for key, value in expected.items():
        if details.get(key) != value:
            raise RuntimeError(f"R sidecar mismatch for {key}: {details.get(key)!r}")
    return details


def run_lane(args: argparse.Namespace) -> Path:
    if psutil is None:
        raise RuntimeError("psutil is required for peak-RSS evidence")
    lane = args.lane
    contract = LANE_CONTRACTS[lane]
    if (
        os.environ.get("FGSEA_REFERENCE_VERSION", "").strip()
        != contract["fgsea_version"]
    ):
        raise RuntimeError(
            "FGSEA_REFERENCE_VERSION must be set explicitly to "
            f"{contract['fgsea_version']} for the {lane} lane"
        )
    for variable in (
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        os.environ[variable] = "1"

    output_dir = Path(args.output_dir).resolve()
    if _is_within(output_dir, REPO_ROOT):
        raise ValueError("Choose an output directory outside the repository")
    output_dir = ensure_empty_output_dir(output_dir)
    initial_git = verify_clean_git_checkout(
        REPO_ROOT,
        expected_commit=args.expected_git_commit,
        expected_tag=args.expected_git_tag,
    )
    manifest_path = Path(args.input_manifest).resolve()
    manifest, input_paths = _validate_input_manifest(manifest_path)
    artifact_receipt_path = Path(args.artifact_receipt).resolve()
    artifact_binding = _verify_artifact_receipt(
        lane, artifact_receipt_path, initial_git
    )
    reference_receipt_path = Path(args.reference_receipt).resolve()
    reference_binding = _verify_reference_receipt(
        lane, reference_receipt_path, initial_git
    )
    pyfgsea, package_identity = verify_installed_lane(
        lane,
        expected_wheel_sha256=artifact_binding["wheel_sha256"],
        expected_core_sha256=artifact_binding["core_sha256"],
    )
    package_identity["artifact_binding"] = artifact_binding
    package_identity["reference_binding"] = reference_binding
    write_json(output_dir / "installed_identity.json", package_identity)
    artifact_receipt_copy = output_dir / "artifact_receipt.json"
    shutil.copyfile(artifact_receipt_path, artifact_receipt_copy)
    reference_receipt_copy = output_dir / "reference_oci_receipt.json"
    shutil.copyfile(reference_receipt_path, reference_receipt_copy)

    configured_rscript = os.environ.get("PYFGSEA_REFERENCE_RSCRIPT", "").strip()
    if configured_rscript:
        rscript_path = Path(configured_rscript).expanduser().resolve()
        if not rscript_path.is_file():
            raise RuntimeError(
                f"PYFGSEA_REFERENCE_RSCRIPT is not a file: {rscript_path}"
            )
        rscript = str(rscript_path)
    else:
        rscript = shutil.which("Rscript") or ""
    if not rscript:
        raise RuntimeError("Rscript is required for the exact R/fgsea reference lane")
    if not R_HELPER.is_file():
        raise RuntimeError(f"R helper is missing: {R_HELPER}")

    all_raw: list[pd.DataFrame] = []
    timing_rows: list[dict[str, Any]] = []
    r_commands: list[list[str]] = []
    python_arguments: dict[str, Any] | None = None
    output_files: dict[str, Path] = {
        "artifact_receipt": artifact_receipt_copy,
        "installed_identity": output_dir / "installed_identity.json",
        "reference_oci_receipt": reference_receipt_copy,
    }
    started_at = datetime.now(timezone.utc).isoformat()

    for scenario_name in SCENARIOS:
        ranks_path = input_paths[scenario_name]["ranks"]
        pathways_path = input_paths[scenario_name]["pathways"]
        ranks, pathways = _load_rank_and_gmt(ranks_path, pathways_path)
        _verify_scenario_invariants(scenario_name, ranks, pathways, manifest)

        (py_result, current_arguments), py_timing = _measure_call(
            lambda: _run_python_engine(pyfgsea, lane, ranks, pathways)
        )
        if python_arguments is None:
            python_arguments = current_arguments
        elif python_arguments != current_arguments:
            raise RuntimeError("Python parameters changed between scenarios")
        _validate_engine_table(py_result, prefix="py")
        py_path = output_dir / f"pyfgsea_{scenario_name}.tsv"
        py_result.to_csv(
            py_path, sep="\t", index=False, float_format="%.17g", lineterminator="\n"
        )
        output_files[f"pyfgsea_{scenario_name}"] = py_path
        timing_rows.append(
            {
                "lane": lane,
                "scenario": scenario_name,
                "engine": "pyfgsea",
                "measurement_scope": "run_gsea_call_only",
                "engine_elapsed_seconds": py_timing["elapsed_seconds"],
                **py_timing,
            }
        )

        r_result_path = output_dir / f"r_fgsea_{scenario_name}.tsv"
        r_environment_path = output_dir / f"r_environment_{scenario_name}.tsv"
        r_session_path = output_dir / f"r_sessionInfo_{scenario_name}.txt"
        r_stdout_path = output_dir / f"r_stdout_{scenario_name}.txt"
        r_stderr_path = output_dir / f"r_stderr_{scenario_name}.txt"
        r_command = [
            rscript,
            "--vanilla",
            str(R_HELPER),
            str(ranks_path),
            str(pathways_path),
            str(r_result_path),
            str(r_environment_path),
            str(r_session_path),
            str(contract["fgsea_version"]),
            str(contract["r_version"]),
            str(contract["bioconductor_version"]),
            str(GSEA_PARAMETERS["r_seed"]),
            str(GSEA_PARAMETERS["min_size"]),
            str(GSEA_PARAMETERS["max_size"]),
            str(GSEA_PARAMETERS["sample_size"]),
        ]
        r_environment = os.environ.copy()
        r_environment["PYFGSEA_FIGURE1_EPS"] = f"{EPS:.17g}"
        r_environment["OMP_NUM_THREADS"] = "1"
        r_environment["OPENBLAS_NUM_THREADS"] = "1"
        r_environment["MKL_NUM_THREADS"] = "1"
        completed, r_timing = _measure_process(r_command, environment=r_environment)
        with r_stdout_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(completed.stdout)
        with r_stderr_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(
                f"R reference failed for {scenario_name}; see {r_stderr_path}"
            )
        r_commands.append(r_command)
        for label, path in {
            f"r_fgsea_{scenario_name}": r_result_path,
            f"r_environment_{scenario_name}": r_environment_path,
            f"r_session_{scenario_name}": r_session_path,
            f"r_stdout_{scenario_name}": r_stdout_path,
            f"r_stderr_{scenario_name}": r_stderr_path,
        }.items():
            output_files[label] = path
        r_details = _read_r_environment(r_environment_path, contract)
        timing_rows.append(
            {
                "lane": lane,
                "scenario": scenario_name,
                "engine": "r_fgsea",
                "measurement_scope": "Rscript_process_and_internal_fgsea",
                "engine_elapsed_seconds": float(r_details["elapsed_seconds"]),
                **r_timing,
            }
        )

        r_result = pd.read_csv(r_result_path, sep="\t").rename(
            columns={
                "ES": "r_es",
                "NES": "r_nes",
                "pval": "r_pval",
                "padj": "r_padj",
                "size": "r_size",
            }
        )
        _validate_engine_table(r_result, prefix="r")
        if set(py_result["pathway"]) != set(r_result["pathway"]):
            only_python = sorted(set(py_result["pathway"]) - set(r_result["pathway"]))
            only_r = sorted(set(r_result["pathway"]) - set(py_result["pathway"]))
            raise RuntimeError(
                "Python/R pathway universe mismatch; "
                f"Python-only={only_python[:5]}, R-only={only_r[:5]}"
            )
        merged = py_result.merge(
            r_result, on="pathway", how="inner", validate="one_to_one"
        )
        if not (merged["py_size"].astype(int) == merged["r_size"].astype(int)).all():
            raise RuntimeError("Python/R mapped pathway sizes disagree")
        expected_pathways = int(
            manifest["scenarios"][scenario_name]["invariants"]["pathway_count"]
        )
        if len(merged) != expected_pathways:
            raise RuntimeError(
                f"{scenario_name} returned {len(merged)} pathways, expected {expected_pathways}"
            )
        merged.insert(0, "lane", lane)
        merged.insert(1, "scenario", scenario_name)
        merged["es_difference"] = merged["py_es"] - merged["r_es"]
        merged["nes_difference"] = merged["py_nes"] - merged["r_nes"]
        merged["py_neg_log10_pval"] = -np.log10(
            np.maximum(merged["py_pval"], LOG10_FLOOR)
        )
        merged["r_neg_log10_pval"] = -np.log10(
            np.maximum(merged["r_pval"], LOG10_FLOOR)
        )
        merged["neg_log10_pval_difference"] = (
            merged["py_neg_log10_pval"] - merged["r_neg_log10_pval"]
        )
        invariants = manifest["scenarios"][scenario_name]["invariants"]
        merged["input_tied_score_group_count"] = int(
            invariants["tied_score_group_count"]
        )
        merged["input_tied_gene_count"] = int(invariants["tied_gene_count"])
        merged["input_maximum_tie_multiplicity"] = int(
            invariants["maximum_tie_multiplicity"]
        )
        all_raw.append(merged.sort_values("pathway", kind="mergesort"))

    pathway_raw = pd.concat(all_raw, ignore_index=True)
    pathway_raw_path = output_dir / "pathway_level_raw.tsv"
    pathway_raw.to_csv(
        pathway_raw_path,
        sep="\t",
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    timing_path = output_dir / "runtime_memory.tsv"
    pd.DataFrame(timing_rows).to_csv(
        timing_path,
        sep="\t",
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    output_files["pathway_raw"] = pathway_raw_path
    output_files["runtime_memory"] = timing_path

    verify_git_unchanged(REPO_ROOT, initial_git)
    receipt_path = output_dir / "lane_receipt.json"
    input_records: dict[str, Any] = {
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "scenarios": {},
    }
    for scenario_name in SCENARIOS:
        input_records["scenarios"][scenario_name] = {
            label: {
                "path": str(input_paths[scenario_name][label]),
                "sha256": sha256_file(input_paths[scenario_name][label]),
            }
            for label in ("ranks", "pathways")
        }
    receipt = {
        "schema_version": 1,
        "kind": "figure1_lane_receipt",
        "suite_version": SUITE_VERSION,
        "lane": lane,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "invocation": [str(item) for item in sys.argv],
        "git": initial_git,
        "scripts": {
            "run_lane": file_record(Path(__file__)),
            "common": file_record(SCRIPT_DIR / "common.py"),
            "prepare_inputs": file_record(SCRIPT_DIR / "prepare_inputs.py"),
            "r_helper": file_record(R_HELPER),
        },
        "python_environment": {
            "executable": str(Path(sys.executable).resolve()),
            "version": platform.python_version(),
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "psutil": psutil.__version__,
            "thread_environment": {
                variable: os.environ[variable]
                for variable in (
                    "RAYON_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        },
        "package_identity": package_identity,
        "lane_contract": contract,
        "gsea_parameters": dict(GSEA_PARAMETERS),
        "python_call_arguments": python_arguments,
        "r_commands": r_commands,
        "inputs": input_records,
        "upstream_receipts": {
            "artifact": artifact_binding,
            "reference_oci": reference_binding,
        },
        "outputs": {
            label: file_record(path, relative_to=output_dir)
            for label, path in sorted(output_files.items())
        },
        "metric_policy": {
            "pathway_level_raw_is_only_metric_source": True,
            "pvalue_log_transform": "-log10(max(pvalue, 1e-300))",
            "manual_metric_overrides_permitted": False,
        },
        "result_scope": {
            "legacy_lane_identity": "official PyPI 0.1.4 artifact lane",
            "legacy_native_core_source_reproducible": False,
            "ties_scope": "same recorded Python/platform environment sensitivity only",
            "ties_cross_platform_equivalence_claimed": False,
            "reference_runtime": (
                "recorded linux/amd64 OCI profile with all six "
                "/opt/reference output hashes checked"
            ),
        },
    }
    write_json(receipt_path, receipt)
    return receipt_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", required=True, choices=sorted(LANE_CONTRACTS))
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--expected-git-tag", required=True)
    parser.add_argument("--artifact-receipt", required=True, type=Path)
    parser.add_argument("--reference-receipt", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = run_lane(args)
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
