#!/usr/bin/env python3
"""Build and verify the PyFgsea release artifact chain.

The verifier deliberately owns the complete chain:

    clean Git commit -> sdist -> wheel built from that sdist -> fresh venv install

A success receipt is written only after every check passes.  On failure the
program exits non-zero and leaves no receipt that could be mistaken for release
evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import email.parser
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_VERSION = "0.2.0"
EXPECTED_ALGORITHM_REVISION = "fgsea-1.38-pr178-v1"
RELEASE_TAG_PATTERN = re.compile(r"v0\.2\.0(?:-rc[1-9][0-9]*)?")
REQUIRED_COMMIT_PATHS = (
    ".cargo/config.toml",
    "Cargo.toml",
    "Cargo.lock",
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "rust-toolchain.toml",
    "src/lib.rs",
)
NATIVE_SUFFIXES = (
    ".a",
    ".dll",
    ".dylib",
    ".exe",
    ".lib",
    ".o",
    ".obj",
    ".pyd",
    ".pyc",
    ".so",
)
NATIVE_MAGICS = (
    b"\x7fELF",
    b"MZ",
    b"!<arch>\n",
    b"\xfe\xed\xfa\xce",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",
)
PROBE_MARKER = "__PYFGSEA_ARTIFACT_EVIDENCE__="


class VerificationError(RuntimeError):
    """A release invariant was not satisfied."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_command(command: Sequence[str]) -> list[str]:
    return [str(part) for part in command]


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    commands: list[dict[str, Any]],
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    command = [str(part) for part in command]
    started = _utc_now()
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=None if env is None else dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    commands.append(
        {
            "argv": _display_command(command),
            "cwd": str(cwd.resolve()),
            "started_at_utc": started,
            "finished_at_utc": _utc_now(),
            "returncode": completed.returncode,
        }
    )
    if completed.returncode != 0:
        stdout = completed.stdout.decode("utf-8", errors="replace")[-4000:]
        stderr = completed.stderr.decode("utf-8", errors="replace")[-4000:]
        raise VerificationError(
            "command failed with exit code "
            f"{completed.returncode}: {command!r}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
    return completed


def _git_bytes(repo: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise VerificationError(
            "git command failed: "
            f"git -C {repo} {' '.join(args)}\n"
            + completed.stderr.decode("utf-8", errors="replace")[-4000:]
        )
    return completed.stdout


def _git_text(repo: Path, *args: str) -> str:
    return _git_bytes(repo, *args).decode("utf-8", errors="strict").strip()


def _canonical_commit(repo: Path, requested: str) -> str:
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", requested):
        raise VerificationError("--commit must be a full hexadecimal commit SHA")
    resolved = _git_text(repo, "rev-parse", f"{requested}^{{commit}}")
    if resolved.lower() != requested.lower():
        raise VerificationError(
            f"requested commit {requested!r} did not resolve to the same full SHA: {resolved}"
        )
    return resolved.lower()


def _verify_release_tag(repo: Path, release_tag: str, commit: str) -> dict[str, Any]:
    if RELEASE_TAG_PATTERN.fullmatch(release_tag) is None:
        raise VerificationError(
            "--release-tag must be v0.2.0 or an annotated v0.2.0-rcN tag"
        )
    tag_ref = f"refs/tags/{release_tag}"
    object_type = _git_text(repo, "cat-file", "-t", tag_ref)
    if object_type != "tag":
        raise VerificationError(
            f"release ref {tag_ref!r} is not an annotated tag object"
        )
    tag_object = _git_text(repo, "rev-parse", tag_ref).lower()
    peeled_commit = _git_text(repo, "rev-parse", f"{tag_ref}^{{commit}}").lower()
    if peeled_commit != commit:
        raise VerificationError(
            f"release tag {release_tag!r} points to {peeled_commit}, expected {commit}"
        )
    return {
        "name": release_tag,
        "annotated": True,
        "tag_object": tag_object,
        "peeled_commit": peeled_commit,
    }


def _git_status(repo: Path) -> list[str]:
    raw = _git_bytes(
        repo,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    return [
        entry.decode("utf-8", errors="replace") for entry in raw.split(b"\0") if entry
    ]


def _require_clean_head(repo: Path, commit: str) -> None:
    head = _git_text(repo, "rev-parse", "HEAD").lower()
    if head != commit:
        raise VerificationError(f"repository HEAD is {head}, expected {commit}")
    dirty = _git_status(repo)
    if dirty:
        preview = "\n".join(dirty[:30])
        raise VerificationError(f"repository is not clean:\n{preview}")


def _tracked_package_paths(repo: Path, commit: str) -> list[str]:
    raw = _git_bytes(
        repo,
        "ls-tree",
        "-r",
        "--name-only",
        "-z",
        commit,
        "--",
        "pyfgsea",
    )
    paths = [item.decode("utf-8", errors="strict") for item in raw.split(b"\0") if item]
    if not paths:
        raise VerificationError("the commit contains no tracked pyfgsea package files")
    if any(not path.startswith("pyfgsea/") for path in paths):
        raise VerificationError("git returned an unexpected path outside pyfgsea/")
    _require_case_unique(paths, context="Git package source manifest")
    return sorted(paths)


def _git_source_manifest(repo: Path, commit: str) -> dict[str, bytes]:
    paths = [*REQUIRED_COMMIT_PATHS, *_tracked_package_paths(repo, commit)]
    manifest: dict[str, bytes] = {}
    for path in paths:
        try:
            manifest[path] = _git_bytes(repo, "show", f"{commit}:{path}")
        except VerificationError as exc:
            raise VerificationError(
                f"required source is absent from commit: {path}"
            ) from exc
    return manifest


def _source_hash_manifest(sources: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    return {
        path: {"sha256": _sha256_bytes(data), "size": len(data)}
        for path, data in sorted(sources.items())
    }


def _source_manifest_sha256(sources: Mapping[str, bytes]) -> str:
    encoded = json.dumps(
        _source_hash_manifest(sources),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _require_case_unique(paths: Iterable[str], *, context: str) -> None:
    seen: dict[str, str] = {}
    for path in paths:
        folded = path.casefold()
        previous = seen.get(folded)
        if previous is not None and previous != path:
            raise VerificationError(
                f"{context} contains case-colliding paths: {previous!r} and {path!r}"
            )
        seen[folded] = path


def _normal_archive_path(name: str, *, archive: str) -> PurePosixPath:
    if "\\" in name:
        raise VerificationError(f"{archive} member uses a backslash path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts:
        raise VerificationError(f"{archive} contains an unsafe member path: {name!r}")
    if any(part in ("", ".", "..") for part in path.parts):
        raise VerificationError(f"{archive} contains an unsafe member path: {name!r}")
    return path


def _looks_native_binary(path: str, data: bytes) -> bool:
    lower = path.lower()
    if lower.endswith(NATIVE_SUFFIXES):
        return True
    return any(data.startswith(magic) for magic in NATIVE_MAGICS)


def _read_sdist(path: Path) -> tuple[str, dict[str, bytes]]:
    if not path.is_file():
        raise VerificationError(f"sdist does not exist: {path}")
    roots: set[str] = set()
    contents: dict[str, bytes] = {}
    with tarfile.open(path, mode="r:gz") as archive:
        for member in archive.getmembers():
            member_path = _normal_archive_path(member.name, archive="sdist")
            roots.add(member_path.parts[0])
            if member.isdir():
                continue
            if not member.isfile():
                raise VerificationError(
                    f"sdist contains a non-regular member: {member.name!r}"
                )
            if len(member_path.parts) < 2:
                raise VerificationError(
                    f"sdist file is outside its root: {member.name!r}"
                )
            relative = PurePosixPath(*member_path.parts[1:]).as_posix()
            if relative in contents:
                raise VerificationError(f"sdist contains duplicate member {relative!r}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise VerificationError(f"could not read sdist member {member.name!r}")
            contents[relative] = extracted.read()
    if len(roots) != 1:
        raise VerificationError(
            f"sdist must have exactly one top-level root, found {sorted(roots)!r}"
        )
    _require_case_unique(contents, context="sdist")
    return next(iter(roots)), contents


def _cargo_version(cargo_toml: bytes) -> str:
    text = cargo_toml.decode("utf-8", errors="strict")
    package_match = re.search(r"(?ms)^\[package\]\s*(.*?)(?=^\[|\Z)", text)
    if package_match is None:
        raise VerificationError("Cargo.toml has no [package] table")
    version_match = re.search(
        r'(?m)^\s*version\s*=\s*"([^"]+)"\s*$', package_match.group(1)
    )
    if version_match is None:
        raise VerificationError("Cargo.toml [package] has no literal version")
    return version_match.group(1)


def _metadata_name_version(data: bytes, *, source: str) -> tuple[str, str]:
    message = email.parser.BytesParser().parsebytes(data)
    name = message.get("Name")
    version = message.get("Version")
    if not name or not version:
        raise VerificationError(f"{source} metadata lacks Name or Version")
    return name, version


def _canonical_project_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _verify_sdist(
    path: Path,
    expected_sources: Mapping[str, bytes],
    *,
    expected_version: str,
) -> dict[str, Any]:
    root, contents = _read_sdist(path)
    native_members = [
        name for name, data in contents.items() if _looks_native_binary(name, data)
    ]
    if native_members:
        raise VerificationError(
            "sdist contains forbidden native or compiled files: "
            + ", ".join(sorted(native_members))
        )

    expected_members = set(expected_sources) | {"PKG-INFO"}
    actual_members = set(contents)
    if actual_members != expected_members:
        missing = sorted(expected_members - actual_members)
        extra = sorted(actual_members - expected_members)
        raise VerificationError(
            "sdist source boundary differs from the committed release inputs; "
            f"missing={missing!r}, extra={extra!r}"
        )

    for source_path, expected in expected_sources.items():
        actual = contents.get(source_path)
        if actual is None:
            raise VerificationError(f"sdist is missing commit source {source_path!r}")
        if actual != expected:
            raise VerificationError(
                f"sdist source differs from commit for {source_path!r}: "
                f"expected {_sha256_bytes(expected)}, got {_sha256_bytes(actual)}"
            )

    expected_package = {
        name for name in expected_sources if name.startswith("pyfgsea/")
    }
    actual_package = {name for name in contents if name.startswith("pyfgsea/")}
    if actual_package != expected_package:
        missing = sorted(expected_package - actual_package)
        extra = sorted(actual_package - expected_package)
        raise VerificationError(
            f"sdist pyfgsea source set differs from Git; missing={missing!r}, extra={extra!r}"
        )

    cargo_version = _cargo_version(contents["Cargo.toml"])
    if cargo_version != expected_version:
        raise VerificationError(
            f"sdist Cargo version is {cargo_version!r}, expected {expected_version!r}"
        )
    pkg_info = contents.get("PKG-INFO")
    if pkg_info is None:
        raise VerificationError("sdist is missing PKG-INFO")
    project_name, metadata_version = _metadata_name_version(pkg_info, source="sdist")
    if _canonical_project_name(project_name) != "pyfgsea":
        raise VerificationError(
            f"sdist project name is {project_name!r}, expected 'pyfgsea'"
        )
    if metadata_version != expected_version:
        raise VerificationError(
            f"sdist metadata version is {metadata_version!r}, expected {expected_version!r}"
        )

    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "sha256": _sha256_file(path),
        "size": path.stat().st_size,
        "top_level_root": root,
        "cargo_version": cargo_version,
        "metadata_version": metadata_version,
        "native_binary_count": 0,
        "verified_commit_source_count": len(expected_sources),
        "verified_source_manifest_sha256": _source_manifest_sha256(expected_sources),
        "pyfgsea_source_set_exact": True,
    }


def _safe_extract_sdist(path: Path, destination: Path) -> Path:
    root, contents = _read_sdist(path)
    destination.mkdir(parents=True, exist_ok=False)
    root_path = destination / root
    for relative, data in contents.items():
        target = root_path.joinpath(*PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return root_path


def _read_wheel(path: Path) -> dict[str, bytes]:
    if not path.is_file():
        raise VerificationError(f"wheel does not exist: {path}")
    contents: dict[str, bytes] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            member_path = _normal_archive_path(info.filename, archive="wheel")
            if info.is_dir():
                continue
            relative = member_path.as_posix()
            if relative in contents:
                raise VerificationError(f"wheel contains duplicate member {relative!r}")
            contents[relative] = archive.read(info)
    _require_case_unique(contents, context="wheel")
    return contents


def _core_wheel_members(contents: Mapping[str, bytes]) -> list[str]:
    candidates = []
    for name in contents:
        path = PurePosixPath(name)
        if path.parent.as_posix() != "pyfgsea":
            continue
        lower = path.name.lower()
        if lower.startswith("_core.") and lower.endswith((".pyd", ".so", ".dylib")):
            candidates.append(name)
    return sorted(candidates)


def _verify_wheel(
    path: Path,
    expected_sources: Mapping[str, bytes],
    *,
    expected_version: str,
) -> dict[str, Any]:
    contents = _read_wheel(path)
    metadata_members = [
        name for name in contents if name.endswith(".dist-info/METADATA")
    ]
    if len(metadata_members) != 1:
        raise VerificationError(
            f"wheel must contain one METADATA file, found {metadata_members!r}"
        )
    project_name, metadata_version = _metadata_name_version(
        contents[metadata_members[0]], source="wheel"
    )
    if _canonical_project_name(project_name) != "pyfgsea":
        raise VerificationError(
            f"wheel project name is {project_name!r}, expected 'pyfgsea'"
        )
    if metadata_version != expected_version:
        raise VerificationError(
            f"wheel metadata version is {metadata_version!r}, expected {expected_version!r}"
        )

    dist_info = metadata_members[0].rsplit("/", 1)[0]
    for required in (f"{dist_info}/WHEEL", f"{dist_info}/RECORD"):
        if required not in contents:
            raise VerificationError(f"wheel is missing {required!r}")

    unexpected_members = sorted(
        name
        for name in contents
        if not name.startswith("pyfgsea/") and not name.startswith(f"{dist_info}/")
    )
    if unexpected_members:
        raise VerificationError(
            "wheel contains members outside the verified pyfgsea package and its "
            f"single dist-info directory: {unexpected_members!r}"
        )

    core_members = _core_wheel_members(contents)
    if len(core_members) != 1:
        raise VerificationError(
            f"wheel must contain exactly one pyfgsea native core, found {core_members!r}"
        )
    core_member = core_members[0]

    expected_package = {
        name for name in expected_sources if name.startswith("pyfgsea/")
    }
    actual_package_sources = {
        name for name in contents if name.startswith("pyfgsea/") and name != core_member
    }
    if actual_package_sources != expected_package:
        missing = sorted(expected_package - actual_package_sources)
        extra = sorted(actual_package_sources - expected_package)
        raise VerificationError(
            f"wheel pyfgsea source set differs from verified sdist; "
            f"missing={missing!r}, extra={extra!r}"
        )
    for source_path in sorted(expected_package):
        expected = expected_sources[source_path]
        actual = contents[source_path]
        if actual != expected:
            raise VerificationError(
                f"wheel source differs from verified sdist for {source_path!r}: "
                f"expected {_sha256_bytes(expected)}, got {_sha256_bytes(actual)}"
            )

    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "sha256": _sha256_file(path),
        "size": path.stat().st_size,
        "metadata_version": metadata_version,
        "core_member": core_member,
        "core_sha256": _sha256_bytes(contents[core_member]),
        "core_size": len(contents[core_member]),
        "verified_source_manifest_sha256": _source_manifest_sha256(expected_sources),
        "pyfgsea_source_set_exact": True,
        "wheel_member_boundary_exact": True,
    }


def _venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _is_within(path: Path, directory: Path) -> bool:
    try:
        return os.path.commonpath(
            [
                os.path.normcase(str(path.resolve())),
                os.path.normcase(str(directory.resolve())),
            ]
        ) == os.path.normcase(str(directory.resolve()))
    except ValueError:
        return False


def _require_external_receipt(receipt_path: Path, repo: Path) -> None:
    if _is_within(receipt_path, repo):
        raise VerificationError(
            "receipt must be outside the verified Git worktree so writing success "
            "evidence cannot dirty or hide files inside the source repository"
        )


def _direct_url_wheel_path(direct_url: Mapping[str, Any]) -> Path:
    url = direct_url.get("url")
    if not isinstance(url, str):
        raise VerificationError("direct_url.json has no string url")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise VerificationError(
            f"direct_url.json does not identify a local wheel: {url!r}"
        )
    local = urllib.request.url2pathname(urllib.parse.unquote(parsed.path))
    return Path(local).resolve()


def _direct_url_sha256(direct_url: Mapping[str, Any]) -> str:
    archive_info = direct_url.get("archive_info")
    if not isinstance(archive_info, Mapping):
        raise VerificationError("direct_url.json has no archive_info object")
    hashes = archive_info.get("hashes")
    if isinstance(hashes, Mapping) and isinstance(hashes.get("sha256"), str):
        return str(hashes["sha256"]).lower()
    archive_hash = archive_info.get("hash")
    if isinstance(archive_hash, str) and archive_hash.lower().startswith("sha256="):
        return archive_hash.split("=", 1)[1].lower()
    raise VerificationError("direct_url.json has no SHA-256 archive hash")


def _verify_installed_probe(
    probe: Mapping[str, Any],
    *,
    venv: Path,
    wheel: Path,
    wheel_evidence: Mapping[str, Any],
    expected_version: str,
    expected_algorithm_revision: str,
) -> dict[str, Any]:
    if probe.get("pyfgsea_version") != expected_version:
        raise VerificationError(
            f"installed pyfgsea.__version__ is {probe.get('pyfgsea_version')!r}, "
            f"expected {expected_version!r}"
        )
    if probe.get("distribution_version") != expected_version:
        raise VerificationError(
            f"installed distribution version is {probe.get('distribution_version')!r}, "
            f"expected {expected_version!r}"
        )
    if probe.get("algorithm_revision") != expected_algorithm_revision:
        raise VerificationError(
            f"installed algorithm revision is {probe.get('algorithm_revision')!r}, "
            f"expected {expected_algorithm_revision!r}"
        )

    prefix = Path(str(probe.get("sys_prefix", ""))).resolve()
    if prefix != venv.resolve():
        raise VerificationError(
            f"probe sys.prefix is {prefix}, expected venv {venv.resolve()}"
        )
    base_prefix = Path(str(probe.get("base_prefix", ""))).resolve()
    if base_prefix == prefix:
        raise VerificationError("installed probe is not running in a Python venv")
    executable = Path(str(probe.get("sys_executable", ""))).resolve()
    package_file = Path(str(probe.get("package_file", ""))).resolve()
    core_file = Path(str(probe.get("core_file", ""))).resolve()
    for label, installed_path in (
        ("Python executable", executable),
        ("pyfgsea package", package_file),
        ("pyfgsea native core", core_file),
    ):
        if not _is_within(installed_path, venv):
            raise VerificationError(
                f"{label} is outside the fresh venv: {installed_path}"
            )

    installed_core_sha = str(probe.get("core_sha256", "")).lower()
    if installed_core_sha != str(wheel_evidence["core_sha256"]).lower():
        raise VerificationError(
            "installed native core hash differs from wheel core: "
            f"{installed_core_sha} != {wheel_evidence['core_sha256']}"
        )

    direct_url = probe.get("direct_url")
    if not isinstance(direct_url, Mapping):
        raise VerificationError("installed distribution has no valid direct_url.json")
    direct_path = _direct_url_wheel_path(direct_url)
    if os.path.normcase(str(direct_path)) != os.path.normcase(str(wheel.resolve())):
        raise VerificationError(
            f"direct_url.json points to {direct_path}, expected {wheel.resolve()}"
        )
    direct_sha = _direct_url_sha256(direct_url)
    if direct_sha != str(wheel_evidence["sha256"]).lower():
        raise VerificationError(
            f"direct_url wheel SHA-256 is {direct_sha}, expected {wheel_evidence['sha256']}"
        )

    return {
        "venv": str(venv.resolve()),
        "python_executable": str(executable),
        "sys_prefix": str(prefix),
        "base_prefix": str(base_prefix),
        "package_file": str(package_file),
        "core_file": str(core_file),
        "core_sha256": installed_core_sha,
        "pyfgsea_version": probe["pyfgsea_version"],
        "distribution_version": probe["distribution_version"],
        "algorithm_revision": probe["algorithm_revision"],
        "direct_url": direct_url,
        "direct_url_wheel_sha256": direct_sha,
        "package_and_core_inside_venv": True,
    }


INSTALL_PROBE = r"""
import hashlib
import importlib
import importlib.metadata
import json
import pathlib
import sys

import pyfgsea

core = importlib.import_module("pyfgsea._core")
distribution = importlib.metadata.distribution("pyfgsea")
direct_url_text = distribution.read_text("direct_url.json")
if direct_url_text is None:
    raise RuntimeError("pyfgsea direct_url.json is missing")
core_file = pathlib.Path(core.__file__).resolve()
payload = {
    "sys_executable": str(pathlib.Path(sys.executable).resolve()),
    "sys_prefix": str(pathlib.Path(sys.prefix).resolve()),
    "base_prefix": str(pathlib.Path(getattr(sys, "base_prefix", sys.prefix)).resolve()),
    "package_file": str(pathlib.Path(pyfgsea.__file__).resolve()),
    "core_file": str(core_file),
    "core_sha256": hashlib.sha256(core_file.read_bytes()).hexdigest(),
    "pyfgsea_version": pyfgsea.__version__,
    "distribution_version": distribution.version,
    "algorithm_revision": core.algorithm_revision(),
    "direct_url": json.loads(direct_url_text),
}
print("__PYFGSEA_ARTIFACT_EVIDENCE__=" + json.dumps(payload, sort_keys=True))
"""


def _installed_probe(
    python: Path,
    *,
    cwd: Path,
    commands: list[dict[str, Any]],
) -> dict[str, Any]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    completed = _run(
        [str(python), "-I", "-c", INSTALL_PROBE],
        cwd=cwd,
        commands=commands,
        env=env,
    )
    stdout = completed.stdout.decode("utf-8", errors="replace")
    marked = [line for line in stdout.splitlines() if line.startswith(PROBE_MARKER)]
    if len(marked) != 1:
        raise VerificationError(
            f"installed probe emitted {len(marked)} evidence records, expected one"
        )
    try:
        payload = json.loads(marked[0][len(PROBE_MARKER) :])
    except json.JSONDecodeError as exc:
        raise VerificationError("installed probe emitted invalid JSON") from exc
    if not isinstance(payload, dict):
        raise VerificationError("installed probe payload is not an object")
    return payload


def _single_artifact(directory: Path, pattern: str, *, label: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise VerificationError(
            f"expected exactly one {label} matching {pattern!r} in {directory}, "
            f"found {[item.name for item in matches]!r}"
        )
    return matches[0]


def _require_empty_or_absent(path: Path, *, label: str) -> None:
    if path.exists():
        if not path.is_dir():
            raise VerificationError(f"{label} exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise VerificationError(
                f"{label} must be empty before verification: {path}"
            )
    else:
        path.mkdir(parents=True)


def _write_success_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise VerificationError(
            f"receipt already exists; refusing to overwrite: {path}"
        )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Linking a fully written temporary file is atomic and refuses to
        # replace an existing receipt on both POSIX and NTFS.
        os.link(temporary_name, path)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _tool_version(
    command: Sequence[str], *, cwd: Path, commands: list[dict[str, Any]]
) -> str:
    completed = _run(command, cwd=cwd, commands=commands)
    combined = completed.stdout.decode("utf-8", errors="replace").strip()
    if not combined:
        combined = completed.stderr.decode("utf-8", errors="replace").strip()
    return combined.splitlines()[0] if combined else "unknown"


def _build_and_verify(args: argparse.Namespace) -> dict[str, Any]:
    started = _utc_now()
    repo = args.repo.resolve()
    output_dir = args.output_dir.resolve()
    venv = args.venv.resolve()
    receipt_path = args.receipt.resolve()
    builder_python = args.python.resolve()
    commands: list[dict[str, Any]] = []

    if not (repo / ".git").exists() and not _git_text(repo, "rev-parse", "--git-dir"):
        raise VerificationError(f"not a Git worktree: {repo}")
    if not builder_python.is_file():
        raise VerificationError(f"builder Python does not exist: {builder_python}")
    if receipt_path.exists():
        raise VerificationError(
            f"receipt already exists; refusing to overwrite: {receipt_path}"
        )
    _require_external_receipt(receipt_path, repo)
    _require_empty_or_absent(output_dir, label="output directory")
    if venv.exists():
        raise VerificationError(f"fresh venv path must not exist: {venv}")

    commit = _canonical_commit(repo, args.commit)
    _require_clean_head(repo, commit)
    release_tag = _verify_release_tag(repo, args.release_tag, commit)
    tree = _git_text(repo, "rev-parse", f"{commit}^{{tree}}")
    expected_sources = _git_source_manifest(repo, commit)
    cargo_version = _cargo_version(expected_sources["Cargo.toml"])
    if cargo_version != args.expected_version:
        raise VerificationError(
            f"commit Cargo version is {cargo_version!r}, expected {args.expected_version!r}"
        )

    tool_versions = {
        "python": _tool_version(
            [str(builder_python), "--version"], cwd=repo, commands=commands
        ),
        "maturin": _tool_version(
            [str(builder_python), "-m", "maturin", "--version"],
            cwd=repo,
            commands=commands,
        ),
        "git": _tool_version(["git", "--version"], cwd=repo, commands=commands),
        "cargo": _tool_version(["cargo", "--version"], cwd=repo, commands=commands),
        "rustc": _tool_version(["rustc", "--version"], cwd=repo, commands=commands),
    }

    _run(
        [str(builder_python), "-m", "maturin", "sdist", "--out", str(output_dir)],
        cwd=repo,
        commands=commands,
    )
    sdist = _single_artifact(output_dir, "*.tar.gz", label="sdist")
    sdist_evidence = _verify_sdist(
        sdist, expected_sources, expected_version=args.expected_version
    )

    temporary_context: tempfile.TemporaryDirectory[str] | None = None
    if args.work_dir is None:
        temporary_context = tempfile.TemporaryDirectory(
            prefix="pyfgsea-artifact-chain-"
        )
        work_dir = Path(temporary_context.name).resolve()
    else:
        work_dir = args.work_dir.resolve()
        _require_empty_or_absent(work_dir, label="work directory")
    try:
        extracted = _safe_extract_sdist(sdist, work_dir / "sdist-source")
        build_command = [
            str(builder_python),
            "-m",
            "maturin",
            "build",
            "--release",
            "--locked",
            "--out",
            str(output_dir),
            "--interpreter",
            str(builder_python),
        ]
        if args.offline:
            build_command.append("--offline")
        _run(build_command, cwd=extracted, commands=commands)

        # Re-read the source after the build.  This closes the gap where a build
        # hook could mutate a verified input before compiling the extension.
        for source_path, expected in expected_sources.items():
            built_source = extracted.joinpath(*PurePosixPath(source_path).parts)
            if not built_source.is_file() or built_source.read_bytes() != expected:
                raise VerificationError(
                    f"sdist source changed during wheel build: {source_path!r}"
                )

        wheel = _single_artifact(output_dir, "*.whl", label="wheel")
        wheel_evidence = _verify_wheel(
            wheel, expected_sources, expected_version=args.expected_version
        )
        wheel_evidence["build_input_sdist_sha256"] = sdist_evidence["sha256"]
        wheel_evidence["wheel_built_from_verified_sdist"] = True

        _run(
            [str(builder_python), "-m", "venv", str(venv)],
            cwd=work_dir,
            commands=commands,
        )
        venv_python = _venv_python(venv)
        if not venv_python.is_file():
            raise VerificationError(f"venv Python was not created: {venv_python}")
        install_command = [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
        ]
        if args.wheelhouse is not None:
            install_command.extend(
                ["--no-index", "--find-links", str(args.wheelhouse.resolve())]
            )
        install_command.append(str(wheel.resolve()))
        _run(install_command, cwd=work_dir, commands=commands)

        probe = _installed_probe(venv_python, cwd=work_dir, commands=commands)
        installed_evidence = _verify_installed_probe(
            probe,
            venv=venv,
            wheel=wheel,
            wheel_evidence=wheel_evidence,
            expected_version=args.expected_version,
            expected_algorithm_revision=args.expected_algorithm_revision,
        )
    finally:
        if temporary_context is not None:
            temporary_context.cleanup()

    _require_clean_head(repo, commit)
    return {
        "schema_version": 1,
        "status": "passed",
        "started_at_utc": started,
        "finished_at_utc": _utc_now(),
        "expected": {
            "pyfgsea_version": args.expected_version,
            "algorithm_revision": args.expected_algorithm_revision,
        },
        "git": {
            "repository": str(repo),
            "commit": commit,
            "tree": tree,
            "release_tag": release_tag,
            "head_matches_commit": True,
            "clean_before_and_after": True,
            "source_manifest": _source_hash_manifest(expected_sources),
        },
        "artifact_chain": {
            "sdist": sdist_evidence,
            "wheel": wheel_evidence,
            "installed": installed_evidence,
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
            "tool_versions": tool_versions,
            "offline_cargo_build": bool(args.offline),
            "wheelhouse": None
            if args.wheelhouse is None
            else str(args.wheelhouse.resolve()),
        },
        "commands": commands,
        "all_release_gates_passed": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a PyFgsea sdist from a clean commit, build its wheel, install "
            "that wheel into a fresh venv, and emit a fail-closed JSON receipt."
        )
    )
    parser.add_argument("--repo", type=Path, required=True, help="Clean Git worktree")
    parser.add_argument(
        "--commit",
        required=True,
        help="Full commit SHA that must equal repository HEAD",
    )
    parser.add_argument(
        "--release-tag",
        required=True,
        help="Annotated v0.2.0-rcN or v0.2.0 tag that must peel to --commit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Absent or empty directory for the generated sdist and wheel",
    )
    parser.add_argument(
        "--venv",
        type=Path,
        required=True,
        help="Nonexistent path for the clean install venv",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        required=True,
        help="New JSON path; existing files are never overwritten",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Builder Python with maturin installed (default: current interpreter)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Optional absent/empty retained work directory (default: temporary)",
    )
    parser.add_argument(
        "--wheelhouse",
        type=Path,
        help="Install dependencies only from this wheelhouse (--no-index)",
    )
    parser.add_argument(
        "--offline", action="store_true", help="Pass --offline to the Cargo wheel build"
    )
    parser.add_argument(
        "--expected-version", default=EXPECTED_VERSION, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--expected-algorithm-revision",
        default=EXPECTED_ALGORITHM_REVISION,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = _build_and_verify(args)
        _write_success_receipt(args.receipt.resolve(), receipt)
    except (OSError, VerificationError, tarfile.TarError, zipfile.BadZipFile) as exc:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "finished_at_utc": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "receipt_written": False,
        }
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print(str(args.receipt.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
