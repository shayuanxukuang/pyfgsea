#!/usr/bin/env python3
"""Check a public PyFgsea 0.1.4 wheel against the v0.1.4 source tag."""

from __future__ import annotations

import argparse
import base64
import csv
import email.parser
import hashlib
import io
import re
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

try:
    from .common import (
        LEGACY_PYFGSEA_COMMIT,
        LEGACY_PYFGSEA_TREE,
        LEGACY_PYPI_WHEEL_SHA256,
        SUITE_VERSION,
        ensure_empty_output_dir,
        file_record,
        sha256_file,
        verify_clean_git_checkout,
        verify_git_unchanged,
        write_json,
    )
except ImportError:  # pragma: no cover - direct script execution
    from common import (  # type: ignore
        LEGACY_PYFGSEA_COMMIT,
        LEGACY_PYFGSEA_TREE,
        LEGACY_PYPI_WHEEL_SHA256,
        SUITE_VERSION,
        ensure_empty_output_dir,
        file_record,
        sha256_file,
        verify_clean_git_checkout,
        verify_git_unchanged,
        write_json,
    )


def _git_bytes(repo: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {message}")
    return completed.stdout


def _canonical_text(data: bytes) -> bytes:
    """Normalize only checkout-dependent line endings for source comparison."""

    return data.replace(b"\r\n", b"\n")


def _wheel_contents(wheel: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if member.is_dir():
                continue
            if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
                raise RuntimeError(f"unsafe wheel member: {member.filename!r}")
            name = path.as_posix()
            if name in result:
                raise RuntimeError(f"duplicate wheel member: {name}")
            result[name] = archive.read(member)
    if not result:
        raise RuntimeError("legacy wheel is empty")
    return result


def _source_manifest(repo: Path) -> tuple[list[str], dict[str, bytes]]:
    names = (
        _git_bytes(
            repo,
            "ls-tree",
            "-r",
            "--name-only",
            LEGACY_PYFGSEA_COMMIT,
            "--",
            "pyfgsea",
        )
        .decode("utf-8", errors="strict")
        .splitlines()
    )
    if not names:
        raise RuntimeError("legacy source tag contains no pyfgsea package files")
    blobs = {
        name: _git_bytes(repo, "show", f"{LEGACY_PYFGSEA_COMMIT}:{name}")
        for name in names
    }
    return names, blobs


def _verify_record(contents: dict[str, bytes], record_name: str) -> None:
    rows = list(csv.reader(io.StringIO(contents[record_name].decode("utf-8"))))
    normalized_rows: list[tuple[str, str, str]] = []
    recorded_names: set[str] = set()
    for row in rows:
        if len(row) != 3:
            raise RuntimeError("legacy wheel RECORD has a malformed row")
        raw_name, digest_field, size_field = row
        slash_name = raw_name.replace("\\", "/")
        path = PurePosixPath(slash_name)
        name = path.as_posix()
        if (
            not raw_name
            or path.is_absolute()
            or ".." in path.parts
            or re.match(r"^[A-Za-z]:", slash_name)
            or name != slash_name
        ):
            raise RuntimeError(f"legacy wheel RECORD has an unsafe path: {raw_name!r}")
        if name in recorded_names:
            raise RuntimeError(
                f"legacy wheel RECORD has a normalized-path collision: {raw_name!r}"
            )
        recorded_names.add(name)
        normalized_rows.append((name, digest_field, size_field))

    if len(rows) != len(contents) or recorded_names != set(contents):
        raise RuntimeError("legacy wheel RECORD file set differs from archive members")
    for name, digest_field, size_field in normalized_rows:
        if name == record_name:
            if digest_field or size_field:
                raise RuntimeError(
                    "legacy wheel RECORD must leave its own hash/size empty"
                )
            continue
        if not digest_field.startswith("sha256=") or not size_field.isdigit():
            raise RuntimeError(f"legacy wheel RECORD lacks SHA-256/size for {name}")
        expected = (
            base64.urlsafe_b64encode(hashlib.sha256(contents[name]).digest())
            .rstrip(b"=")
            .decode("ascii")
        )
        if digest_field.split("=", 1)[1] != expected:
            raise RuntimeError(f"legacy wheel RECORD hash mismatch for {name}")
        if int(size_field) != len(contents[name]):
            raise RuntimeError(f"legacy wheel RECORD size mismatch for {name}")


def verify_legacy_artifact(args: argparse.Namespace) -> Path:
    repo = Path(args.legacy_repo).resolve()
    evidence_repo = Path(args.evidence_repo).resolve()
    wheel = Path(args.wheel).resolve()
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise FileNotFoundError(f"legacy wheel is missing: {wheel}")
    output_dir = ensure_empty_output_dir(args.output_dir)
    receipt_path = Path(args.receipt).resolve()
    if receipt_path.parent != output_dir:
        raise ValueError("--receipt must be directly inside --output-dir")
    git = verify_clean_git_checkout(
        repo,
        expected_commit=LEGACY_PYFGSEA_COMMIT,
        expected_tag="v0.1.4",
    )
    if git["tree"] != LEGACY_PYFGSEA_TREE:
        raise RuntimeError("legacy v0.1.4 source tree differs from the frozen contract")
    evidence_git = verify_clean_git_checkout(
        evidence_repo,
        expected_commit=args.expected_evidence_commit,
        expected_tag=args.expected_evidence_tag,
    )

    authoritative_hash = LEGACY_PYPI_WHEEL_SHA256.get(wheel.name)
    if authoritative_hash is None:
        raise RuntimeError(
            f"wheel filename is not published for PyPI 0.1.4: {wheel.name}"
        )
    if sha256_file(wheel) != authoritative_hash:
        raise RuntimeError(
            "legacy wheel bytes differ from the official PyPI 0.1.4 hash"
        )

    contents = _wheel_contents(wheel)
    metadata_members = [
        name for name in contents if name.endswith(".dist-info/METADATA")
    ]
    if len(metadata_members) != 1:
        raise RuntimeError("legacy wheel must contain exactly one METADATA file")
    metadata = email.parser.BytesParser().parsebytes(contents[metadata_members[0]])
    if (
        metadata.get("Name", "").lower() != "pyfgsea"
        or metadata.get("Version") != "0.1.4"
    ):
        raise RuntimeError("legacy wheel metadata is not PyFgsea 0.1.4")
    dist_info = metadata_members[0].rsplit("/", 1)[0]
    for required in (f"{dist_info}/WHEEL", f"{dist_info}/RECORD"):
        if required not in contents:
            raise RuntimeError(f"legacy wheel is missing {required}")
    _verify_record(contents, f"{dist_info}/RECORD")

    core_members = [
        name
        for name in contents
        if PurePosixPath(name).parent.as_posix() == "pyfgsea"
        and PurePosixPath(name).name.lower().startswith("_core.")
        and PurePosixPath(name).suffix.lower() in {".pyd", ".so", ".dylib"}
    ]
    if len(core_members) != 1:
        raise RuntimeError("legacy wheel must contain exactly one native PyFgsea core")
    core_member = core_members[0]

    tracked_names, source_blobs = _source_manifest(repo)
    wheel_sources = {
        name for name in contents if name.startswith("pyfgsea/") and name != core_member
    }
    if wheel_sources != set(tracked_names):
        missing = sorted(set(tracked_names) - wheel_sources)
        extra = sorted(wheel_sources - set(tracked_names))
        raise RuntimeError(
            f"legacy wheel source set differs from v0.1.4; missing={missing}, extra={extra}"
        )
    source_hashes: dict[str, dict[str, str]] = {}
    for name in tracked_names:
        git_source = _canonical_text(source_blobs[name])
        wheel_source = _canonical_text(contents[name])
        if wheel_source != git_source:
            raise RuntimeError(f"legacy wheel source differs from v0.1.4: {name}")
        source_hashes[name] = {
            "canonical_git_sha256": hashlib.sha256(git_source).hexdigest(),
            "canonical_wheel_sha256": hashlib.sha256(wheel_source).hexdigest(),
        }

    init_source = _canonical_text(contents["pyfgsea/__init__.py"]).decode(
        "utf-8", errors="strict"
    )
    version_match = re.search(
        r'^__version__\s*=\s*"([^"]+)"', init_source, re.MULTILINE
    )
    if version_match is None or version_match.group(1) != "0.1.3":
        raise RuntimeError(
            "legacy wheel does not preserve the historical module 0.1.3 label"
        )

    verify_git_unchanged(repo, git)
    verify_git_unchanged(evidence_repo, evidence_git)
    write_json(
        receipt_path,
        {
            "schema_version": 1,
            "kind": "figure1_legacy_artifact_receipt",
            "suite_version": SUITE_VERSION,
            "status": "passed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "invocation": [str(item) for item in sys.argv],
            "git": git,
            "evidence_git": evidence_git,
            "expected": {
                "distribution_version": "0.1.4",
                "module_declared_version": "0.1.3",
                "algorithm_revision_contract": "legacy-no-revision-api",
            },
            "wheel": {
                **file_record(wheel),
                "filename": wheel.name,
                "authoritative_source": "https://pypi.org/pypi/pyfgsea/0.1.4/json",
                "authoritative_pypi_sha256": authoritative_hash,
                "core_member": core_member,
                "core_sha256": hashlib.sha256(contents[core_member]).hexdigest(),
                "source_set_exact": True,
                "source_bytes_equal_after_crlf_normalization": True,
                "source_manifest": source_hashes,
                "record_hashes_and_sizes_valid": True,
            },
            "script": file_record(Path(__file__)),
            "all_legacy_artifact_gates_passed": True,
            "limitation": (
                "The published wheel is bound to every v0.1.4 Python source file and "
                "to its native-core bytes; the historical release did not publish a "
                "reproducible sdist-to-wheel build receipt."
            ),
        },
    )
    return receipt_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-repo", required=True, type=Path)
    parser.add_argument("--evidence-repo", required=True, type=Path)
    parser.add_argument("--expected-evidence-commit", required=True)
    parser.add_argument("--expected-evidence-tag", required=True)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    receipt = verify_legacy_artifact(parse_args(argv))
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
