"""Shared, dependency-light helpers for the Figure 1 evidence pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


SUITE_VERSION = "figure1-dual-lane-v3"
PUBLICATION_SOURCE_COMMIT = "5cedd4abbc8d399221c741256ec5f3839861686d"
LEGACY_PYFGSEA_COMMIT = "67f89433d9c1b76bb6aeb29a9a7c33c1360405e5"
LEGACY_PYFGSEA_TREE = "aa139f506bd9951485a4e397cc0ba13ef760425f"
LEGACY_PYPI_WHEEL_SHA256 = {
    "pyfgsea-0.1.4-cp38-abi3-macosx_11_0_arm64.whl": (
        "15161c02cdc45acf80c87f611e3e70cec323e7adab72ab3cb39b2c5c0dcdaecb"
    ),
    "pyfgsea-0.1.4-cp38-abi3-manylinux_2_34_x86_64.whl": (
        "bd55fe8f99ba204b1f5ee98d3f8a97f240e3cdb76a6e4b826ffdfae0260fd0dc"
    ),
    "pyfgsea-0.1.4-cp38-abi3-win_amd64.whl": (
        "0fc12a73efba4bfbf6a1e1bc9446c885f796d62b9c3b82c1d2b6f8031c64360a"
    ),
}
SCENARIOS = ("publication_main", "ties_predeclared")
EPS = 1e-50
LOG10_FLOOR = 1e-300

LANE_CONTRACTS: dict[str, dict[str, str | None]] = {
    "legacy": {
        "pyfgsea_distribution_version": "0.1.4",
        # The v0.1.4 tag and distribution metadata say 0.1.4, but the shipped
        # module and Cargo crate still declare 0.1.3.  Preserving and checking
        # that discrepancy is part of reproducing the historical artifact.
        "pyfgsea_module_version": "0.1.3",
        "algorithm_revision": None,
        "fgsea_version": "1.32.2",
        "r_version": "4.4.3",
        "bioconductor_version": "3.20",
    },
    "current": {
        "pyfgsea_distribution_version": "0.2.0rc6",
        "pyfgsea_module_version": "0.2.0rc6",
        "algorithm_revision": "fgsea-1.38-pr178-v1",
        "fgsea_version": "1.38.0",
        "r_version": "4.6.0",
        "bioconductor_version": "3.23",
    },
}

REFERENCE_ARTIFACT_CONTRACTS: dict[str, dict[str, str]] = {
    "legacy": {
        "profile": "legacy-publication",
        "reference_id": "legacy-publication",
        "dockerfile": "Dockerfile.reference-fgsea-1.32.2",
        "base_image_digest": (
            "sha256:089317f336a61255bb35f1efd799820cef37136d2acf2a76ba4abb74af51d4a3"
        ),
        "fgsea_tarball_sha256": (
            "2eb41ffb00af5ba3a1eb121d85a04028e57a023e8af3578e21440af62ff231a4"
        ),
    },
    "current": {
        "profile": "current-conformance",
        "reference_id": "current-conformance",
        "dockerfile": "Dockerfile.reference-fgsea-1.38.0",
        "base_image_digest": (
            "sha256:dc3c818ba1a6a58cbc5c450878bb4d9a6385feef60bbbfa5f8d62e54f544e566"
        ),
        "fgsea_tarball_sha256": (
            "223eede5c1c1c8f8a5979e51a40d7f373f0476aefc5c3525c323af5cf871798f"
        ),
    },
}

GSEA_PARAMETERS: dict[str, int | float] = {
    "min_size": 15,
    "max_size": 500,
    "sample_size": 101,
    "python_seed": 1,
    "nperm_nes": 1800,
    "r_seed": 314,
    "eps": EPS,
}

def sha256_file(path: Path) -> str:
    """Hash an existing regular file, failing when evidence is absent."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"required evidence file is missing: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: str, *, label: str) -> str:
    """Validate a caller-supplied SHA-256 rather than accepting a vague ID."""

    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        char not in "0123456789abcdef" for char in normalized
    ):
        raise ValueError(f"{label} must be a 64-character lowercase SHA-256")
    return normalized


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable, UTF-8 JSON with a trailing newline."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and reject other top-level shapes."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"required JSON file is missing: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {source}")
    return payload


def ensure_empty_output_dir(path: Path) -> Path:
    """Create an evidence directory only when no prior result can be overwritten."""

    target = Path(path).resolve()
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"output directory must be empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def git_value(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {message}")
    return completed.stdout.strip()


def verify_clean_git_checkout(
    repo_root: Path, *, expected_commit: str, expected_tag: str
) -> dict[str, Any]:
    """Require the evidence scripts themselves to come from one clean tagged tree."""

    commit = git_value(repo_root, "rev-parse", "HEAD")
    expected = expected_commit.strip().lower()
    if len(expected) != 40 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("--expected-git-commit must be a full 40-character commit SHA")
    if commit.lower() != expected:
        raise RuntimeError(
            f"evidence checkout mismatch: expected {expected}, found {commit}"
        )
    status = git_value(repo_root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError("evidence checkout is not clean:\n" + status)
    tags = set(git_value(repo_root, "tag", "--points-at", "HEAD").splitlines())
    if expected_tag not in tags:
        raise RuntimeError(
            f"expected tag {expected_tag!r} does not point at evidence commit {commit}"
        )
    tag_ref = f"refs/tags/{expected_tag}"
    tag_object_type = git_value(repo_root, "cat-file", "-t", tag_ref)
    tag_object = git_value(repo_root, "rev-parse", tag_ref).lower()
    tag_commit = git_value(repo_root, "rev-parse", f"{tag_ref}^{{commit}}").lower()
    if tag_commit != commit.lower():
        raise RuntimeError(
            f"expected tag {expected_tag!r} does not peel to evidence commit {commit}"
        )
    if (
        re.fullmatch(r"v0\.2\.0(?:-rc[1-9][0-9]*)?", expected_tag)
        and tag_object_type != "tag"
    ):
        raise RuntimeError("0.2.0 release evidence requires an annotated release tag")
    return {
        "commit": commit,
        "tree": git_value(repo_root, "rev-parse", "HEAD^{tree}"),
        "tag": expected_tag,
        "tag_object": tag_object,
        "tag_object_type": tag_object_type,
        "clean": True,
    }


def verify_git_unchanged(repo_root: Path, initial: Mapping[str, Any]) -> None:
    """Fail if commit, tree, tag, or working-copy cleanliness changed mid-run."""

    if git_value(repo_root, "rev-parse", "HEAD") != initial["commit"]:
        raise RuntimeError("Git commit changed during evidence generation")
    if git_value(repo_root, "rev-parse", "HEAD^{tree}") != initial["tree"]:
        raise RuntimeError("Git tree changed during evidence generation")
    status = git_value(repo_root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError("evidence checkout became dirty during the run:\n" + status)
    tags = set(git_value(repo_root, "tag", "--points-at", "HEAD").splitlines())
    if initial.get("tag") not in tags:
        raise RuntimeError("expected evidence tag no longer points at HEAD")
    tag_ref = f"refs/tags/{initial['tag']}"
    if git_value(repo_root, "cat-file", "-t", tag_ref) != initial.get(
        "tag_object_type"
    ) or git_value(repo_root, "rev-parse", tag_ref).lower() != initial.get(
        "tag_object"
    ):
        raise RuntimeError("evidence tag object changed during the run")


def file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, str]:
    """Return a portable path/hash record for an evidence file."""

    resolved = Path(path).resolve()
    if relative_to is None:
        display = str(resolved)
    else:
        display = resolved.relative_to(Path(relative_to).resolve()).as_posix()
    return {"path": display, "sha256": sha256_file(resolved)}


def verify_file_record(base: Path, record: Mapping[str, Any], *, label: str) -> Path:
    """Resolve and verify one path/hash record from a portable receipt."""

    raw_path = record.get("path")
    raw_hash = record.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(raw_hash, str):
        raise ValueError(f"{label} has an invalid path/hash record")
    source = Path(raw_path)
    if not source.is_absolute():
        source = Path(base).resolve() / source
    source = source.resolve()
    expected = require_sha256(raw_hash, label=f"{label} sha256")
    actual = sha256_file(source)
    if actual != expected:
        raise RuntimeError(
            f"{label} hash mismatch: expected {expected}, found {actual}"
        )
    return source


def assert_finite_range(
    values: Sequence[float],
    *,
    label: str,
    lower: float | None = None,
    upper: float | None = None,
) -> None:
    """Fail closed on NaN/Inf or impossible probability ranges."""

    for index, raw in enumerate(values):
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"{label}[{index}] is not finite: {value}")
        if lower is not None and value < lower:
            raise ValueError(f"{label}[{index}] is below {lower}: {value}")
        if upper is not None and value > upper:
            raise ValueError(f"{label}[{index}] is above {upper}: {value}")
