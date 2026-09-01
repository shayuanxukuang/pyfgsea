"""Small, fail-closed provenance receipts for reproduction artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_PYFGSEA_VERSIONS = {"0.1.4", "0.2.0"}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of an existing regular file."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Evidence file is missing: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*arguments: str) -> Optional[str]:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _hashed_paths(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    result = {}
    for name, path in sorted(paths.items()):
        resolved = Path(path).resolve()
        result[name] = {
            "path": str(resolved),
            "sha256": sha256_file(resolved),
        }
    return result


def capture_git_state() -> dict[str, object]:
    """Capture repository identity before a run creates evidence artifacts."""

    status = _git_value("status", "--porcelain", "--untracked-files=all")
    return {
        "commit": _git_value("rev-parse", "HEAD"),
        "tree": _git_value("rev-parse", "HEAD^{tree}"),
        "clean": status == "",
        "status_porcelain": status,
    }


def verify_pyfgsea_installation() -> dict[str, object]:
    """Verify and describe the explicitly selected installed PyFgsea lane."""

    expected_version = os.environ.get("PYFGSEA_EXPECTED_VERSION", "").strip()
    if expected_version not in SUPPORTED_PYFGSEA_VERSIONS:
        allowed = ", ".join(sorted(SUPPORTED_PYFGSEA_VERSIONS))
        raise RuntimeError(
            "Set PYFGSEA_EXPECTED_VERSION explicitly before generating evidence; "
            f"supported values are {allowed}."
        )

    try:
        import pyfgsea
        from pyfgsea import wrapper as pyfgsea_wrapper
    except ImportError as exc:
        raise RuntimeError(
            "A loadable PyFgsea installation is required for an evidence receipt"
        ) from exc

    package_version = str(getattr(pyfgsea, "__version__", "0+unknown"))
    if package_version != expected_version:
        raise RuntimeError(
            "PyFgsea lane mismatch: "
            f"expected {expected_version}, loaded {package_version} from {pyfgsea.__file__}"
        )
    core_path = Path(pyfgsea_wrapper._ext.__file__).resolve()
    revision_reader = getattr(pyfgsea_wrapper, "_algorithm_revision", None)
    algorithm_revision = revision_reader() if callable(revision_reader) else None
    if package_version == "0.2.0" and algorithm_revision in {
        None,
        "unknown-unverified-extension",
    }:
        raise RuntimeError(
            "PyFgsea 0.2.0 Rust algorithm revision is unavailable or unverified"
        )
    return {
        "expected_version": expected_version,
        "version": package_version,
        "module_path": str(Path(pyfgsea.__file__).resolve()),
        "core_path": str(core_path),
        "core_sha256": sha256_file(core_path),
        "algorithm_revision": algorithm_revision,
    }


def write_evidence_receipt(
    path: Path,
    *,
    script: Path,
    parameters: Mapping[str, object],
    inputs: Mapping[str, Path],
    outputs: Mapping[str, Path],
    git_state: Optional[Mapping[str, object]] = None,
    extra: Optional[Mapping[str, object]] = None,
) -> None:
    """Write provenance only after every declared input/output can be hashed."""

    script_path = Path(script).resolve()
    recorded_git = dict(git_state) if git_state is not None else capture_git_state()
    if not recorded_git.get("commit") or not recorded_git.get("tree"):
        raise RuntimeError("Git commit and tree identity are required for evidence")
    current_git = capture_git_state()
    if (
        current_git.get("commit") != recorded_git.get("commit")
        or current_git.get("tree") != recorded_git.get("tree")
    ):
        raise RuntimeError("Git commit or tree changed while the evidence run was active")
    package = verify_pyfgsea_installation()

    payload: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [str(item) for item in sys.argv],
        "working_directory": str(Path.cwd().resolve()),
        "script": {
            "path": str(script_path),
            "sha256": sha256_file(script_path),
        },
        "git": recorded_git,
        "python": {
            "version": platform.python_version(),
            "executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(),
        },
        "pyfgsea": package,
        "parameters": dict(parameters),
        "inputs": _hashed_paths(inputs),
        "outputs": _hashed_paths(outputs),
    }
    if extra:
        payload["extra"] = dict(extra)

    receipt_path = Path(path)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
