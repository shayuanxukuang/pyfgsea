"""Bind one reference profile to a clean commit and complete evidence receipt.

The checked-in manifest deliberately leaves run-specific fields unbound.  This
command fills them only after the working tree is clean and the caller provides
the primary dataset, exact parameters (including seeds), immutable image
digests, a resolved lockfile, and concrete result files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "reference_manifest.json"
IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SOURCE_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
PLATFORM_RE = re.compile(r"^[a-z0-9_]+/[a-z0-9_]+(?:/[a-z0-9_.-]+)?$")


def run_git(source_root: Path, *args: str, binary: bool = False):
    completed = subprocess.run(
        ["git", *args],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=not binary,
    )
    return completed.stdout


def clean_commit_and_tree_hash(source_root: Path) -> Tuple[str, str]:
    status = run_git(
        source_root, "status", "--porcelain", "--untracked-files=normal"
    )
    if status.strip():
        raise RuntimeError(
            "Refusing to bind reference evidence to a dirty working tree. "
            "Commit the exact source and rerun this command."
        )
    commit = run_git(source_root, "rev-parse", "HEAD").strip()
    archive = run_git(source_root, "archive", "--format=tar", "HEAD", binary=True)
    tree_sha256 = hashlib.sha256(archive).hexdigest()
    return commit, tree_sha256


def source_version(source_root: Path) -> str:
    pyproject = source_root / "pyproject.toml"
    if pyproject.is_file():
        project = tomllib.loads(pyproject.read_text(encoding="utf-8")).get("project", {})
        version = project.get("version")
        if isinstance(version, str) and version.strip():
            return version.strip()
    cargo = source_root / "Cargo.toml"
    if cargo.is_file():
        package = tomllib.loads(cargo.read_text(encoding="utf-8")).get("package", {})
        version = package.get("version")
        if isinstance(version, str) and version.strip():
            version = version.strip()
            match = re.fullmatch(r"(\d+\.\d+\.\d+)-rc(\d+)", version)
            if match is not None:
                return f"{match.group(1)}rc{match.group(2)}"
            return version
    raise RuntimeError(
        f"Cannot determine the PyFgsea version in source checkout {source_root}"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_named_paths(values: Iterable[str]) -> Dict[str, Path]:
    parsed: Dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name.strip() or not raw_path.strip():
            raise ValueError(f"Expected NAME=PATH, got: {value}")
        key = name.strip()
        if key in parsed:
            raise ValueError(f"Duplicate artifact name: {key}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        parsed[key] = path
    return parsed


def reject_json_constant(constant: str):
    raise ValueError(f"non-finite JSON constant {constant}")


def parse_named_json(values: Iterable[str]) -> Dict[str, object]:
    parsed: Dict[str, object] = {}
    for value in values:
        name, separator, raw_value = value.partition("=")
        key = name.strip()
        if not separator or not key or not raw_value.strip():
            raise ValueError(f"Expected NAME=JSON_VALUE, got: {value}")
        if key in parsed:
            raise ValueError(f"Duplicate parameter name: {key}")
        try:
            parsed[key] = json.loads(
                raw_value,
                parse_constant=reject_json_constant,
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Parameter {key!r} must use a JSON value: {raw_value}"
            ) from error
    return parsed


def artifact_record(path: Path) -> Dict[str, object]:
    try:
        display_path = path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        display_path = str(path)
    return {
        "path": display_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        required=True,
        choices=["legacy_publication", "current_conformance"],
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--pyfgsea-source",
        type=Path,
        default=REPO_ROOT,
        help=(
            "Clean source checkout that generated the result. Use a separately "
            "verified 0.1.4 checkout for the legacy profile."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Primary dataset/input whose SHA-256 is recorded explicitly.",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Input artifact to hash; may be repeated.",
    )
    parser.add_argument(
        "--result",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Result artifact to hash; at least one is required.",
    )
    parser.add_argument(
        "--parameter",
        action="append",
        required=True,
        metavar="NAME=JSON_VALUE",
        help="Exact run parameter or seed; may be repeated.",
    )
    parser.add_argument("--fgsea-commit", required=True)
    parser.add_argument("--resolved-lock", type=Path, required=True)
    parser.add_argument("--docker-base-image-digest", required=True)
    parser.add_argument("--docker-image-digest", required=True)
    parser.add_argument(
        "--docker-platform",
        required=True,
        help="Concrete OCI platform, for example linux/amd64.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the updated manifest instead of writing it.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    dataset_path = args.dataset.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    input_paths = parse_named_paths(args.input)
    if "dataset" in input_paths:
        raise ValueError("The input name 'dataset' is reserved for --dataset.")
    result_paths = parse_named_paths(args.result)
    if not result_paths:
        raise RuntimeError("At least one --result NAME=PATH is required.")
    parameters = parse_named_json(args.parameter)
    seed_parameters = {
        name: value
        for name, value in parameters.items()
        if name == "seed" or name.endswith("_seed")
    }
    if not seed_parameters:
        raise RuntimeError("At least one explicit seed parameter is required.")
    for name, value in seed_parameters.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= (2**64 - 1)
        ):
            raise ValueError(f"Seed parameter {name!r} must be a uint64 integer.")

    for option, digest in (
        ("--docker-base-image-digest", args.docker_base_image_digest),
        ("--docker-image-digest", args.docker_image_digest),
    ):
        if not IMAGE_DIGEST_RE.fullmatch(digest):
            raise ValueError(f"{option} must be sha256:<64 lowercase hex>.")
    if not PLATFORM_RE.fullmatch(args.docker_platform):
        raise ValueError("--docker-platform must look like linux/amd64.")
    if not SOURCE_COMMIT_RE.fullmatch(args.fgsea_commit):
        raise ValueError("--fgsea-commit must be a 7-40 character lowercase hex ID.")

    resolved_lock = args.resolved_lock.expanduser().resolve()
    if not resolved_lock.is_file():
        raise FileNotFoundError(resolved_lock)

    profile = manifest["profiles"][args.profile]
    declared_base_digest = profile.get("docker_base_image_digest")
    if declared_base_digest and declared_base_digest != args.docker_base_image_digest:
        raise RuntimeError(
            "base image digest mismatch for "
            f"{args.profile}: manifest declares {declared_base_digest!r}, "
            f"caller supplied {args.docker_base_image_digest!r}."
        )
    declared_platform = profile.get("docker_platform")
    if declared_platform and declared_platform != args.docker_platform:
        raise RuntimeError(
            "OCI platform mismatch for "
            f"{args.profile}: manifest declares {declared_platform!r}, "
            f"caller supplied {args.docker_platform!r}."
        )
    source_root = args.pyfgsea_source.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    observed_pyfgsea_version = source_version(source_root)
    expected_pyfgsea_version = profile["pyfgsea_version"]
    if observed_pyfgsea_version != expected_pyfgsea_version:
        raise RuntimeError(
            "PyFgsea source version mismatch for "
            f"{args.profile}: expected {expected_pyfgsea_version}, "
            f"found {observed_pyfgsea_version} in {source_root}."
        )
    commit, tree_sha256 = clean_commit_and_tree_hash(source_root)
    declared_fgsea_commit = profile.get("fgsea_commit")
    if declared_fgsea_commit != args.fgsea_commit:
        raise RuntimeError(
            "fgsea source commit mismatch for "
            f"{args.profile}: manifest declares {declared_fgsea_commit!r}, "
            f"caller supplied {args.fgsea_commit!r}."
        )

    dataset_record = artifact_record(dataset_path)
    profile["pyfgsea_commit"] = commit
    profile["pyfgsea_source_tree_sha256"] = tree_sha256
    profile["dataset_sha256"] = dataset_record["sha256"]
    profile["parameters"] = dict(sorted(parameters.items()))
    profile["input_artifacts"] = {"dataset": dataset_record}
    profile["input_artifacts"].update(
        {
            name: artifact_record(path)
            for name, path in sorted(input_paths.items())
        }
    )
    profile["result_artifacts"] = {
        name: artifact_record(path) for name, path in sorted(result_paths.items())
    }
    profile["resolved_lockfile_sha256"] = sha256_file(resolved_lock)
    profile["docker_base_image_digest"] = args.docker_base_image_digest
    profile["docker_image_digest"] = args.docker_image_digest
    profile["docker_platform"] = args.docker_platform
    profile["recomputation_status"] = "artifacts-recorded-pending-validation"
    profile["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()

    manifest["generated_results"] = any(
        item.get("recomputation_status") != "not-run"
        for item in manifest["profiles"].values()
    )
    manifest["configuration_status"] = "artifacts-recorded-pending-validation"

    rendered = json.dumps(manifest, indent=2, sort_keys=False) + "\n"
    if args.dry_run:
        sys.stdout.write(rendered)
    else:
        manifest_path.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
