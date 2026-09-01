from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_pyfgsea_artifacts.py"
SPEC = importlib.util.spec_from_file_location("verify_pyfgsea_artifacts", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


def _metadata(version: str = "0.2.0") -> bytes:
    return (f"Metadata-Version: 2.3\nName: pyfgsea\nVersion: {version}\n\n").encode()


def _sources() -> dict[str, bytes]:
    return {
        ".cargo/config.toml": b'[registries.crates-io]\nprotocol = "sparse"\n',
        "Cargo.toml": b'[package]\nname = "pyfgsea"\nversion = "0.2.0"\n',
        "Cargo.lock": b"# lock\n",
        "LICENSE": b"MIT\n",
        "README.md": b"# PyFgsea\n",
        "pyproject.toml": b"[build-system]\n",
        "rust-toolchain.toml": b'[toolchain]\nchannel = "1.92.0"\n',
        "src/lib.rs": b"pub fn example() {}\n",
        "pyfgsea/__init__.py": b'__version__ = "0.2.0"\n',
        "pyfgsea/schemas/example.json": b"{}\n",
    }


def _write_sdist(path: Path, sources: dict[str, bytes], **extra: bytes) -> None:
    members = {**sources, "PKG-INFO": _metadata(), **extra}
    with tarfile.open(path, "w:gz") as archive:
        for relative, data in members.items():
            info = tarfile.TarInfo(f"pyfgsea-0.2.0/{relative}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def _write_wheel(path: Path, sources: dict[str, bytes]) -> bytes:
    core = b"MZ-test-native-core"
    with zipfile.ZipFile(path, "w") as archive:
        for relative, data in sources.items():
            if relative.startswith("pyfgsea/"):
                archive.writestr(relative, data)
        archive.writestr("pyfgsea/_core.pyd", core)
        archive.writestr("pyfgsea-0.2.0.dist-info/METADATA", _metadata())
        archive.writestr("pyfgsea-0.2.0.dist-info/WHEEL", "Wheel-Version: 1.0\n")
        archive.writestr("pyfgsea-0.2.0.dist-info/RECORD", "")
    return core


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_sdist_sources_must_match_and_contain_no_native_binary(tmp_path: Path) -> None:
    sources = _sources()
    good = tmp_path / "pyfgsea-0.2.0.tar.gz"
    _write_sdist(good, sources)
    evidence = verify._verify_sdist(good, sources, expected_version="0.2.0")
    assert evidence["native_binary_count"] == 0
    assert evidence["pyfgsea_source_set_exact"] is True

    bad = tmp_path / "pyfgsea-0.2.0-native.tar.gz"
    _write_sdist(bad, sources, **{"pyfgsea/_core.pyd": b"MZbad"})
    with pytest.raises(verify.VerificationError, match="forbidden native"):
        verify._verify_sdist(bad, sources, expected_version="0.2.0")


def test_sdist_rejects_uncommitted_package_source(tmp_path: Path) -> None:
    sources = _sources()
    path = tmp_path / "pyfgsea-0.2.0.tar.gz"
    _write_sdist(path, sources, **{"pyfgsea/untracked.py": b"unexpected = True\n"})
    with pytest.raises(verify.VerificationError, match="source boundary differs"):
        verify._verify_sdist(path, sources, expected_version="0.2.0")


def test_sdist_rejects_extra_nonpackage_release_input(tmp_path: Path) -> None:
    sources = _sources()
    path = tmp_path / "pyfgsea-0.2.0.tar.gz"
    _write_sdist(path, sources, **{"build.rs": b"fn main() {}\n"})
    with pytest.raises(verify.VerificationError, match="source boundary differs"):
        verify._verify_sdist(path, sources, expected_version="0.2.0")


def test_wheel_sources_and_native_core_are_hashed(tmp_path: Path) -> None:
    sources = _sources()
    path = tmp_path / "pyfgsea-0.2.0-cp38-abi3-win_amd64.whl"
    core = _write_wheel(path, sources)
    evidence = verify._verify_wheel(path, sources, expected_version="0.2.0")
    assert evidence["core_member"] == "pyfgsea/_core.pyd"
    assert evidence["core_sha256"] == verify._sha256_bytes(core)
    assert evidence["pyfgsea_source_set_exact"] is True
    assert evidence["wheel_member_boundary_exact"] is True


@pytest.mark.parametrize("extra_member", ["injected.pth", "otherpkg/__init__.py"])
def test_wheel_rejects_unexpected_top_level_members(
    tmp_path: Path, extra_member: str
) -> None:
    sources = _sources()
    path = tmp_path / "pyfgsea-0.2.0-cp38-abi3-win_amd64.whl"
    _write_wheel(path, sources)
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr(extra_member, b"unexpected\n")
    with pytest.raises(verify.VerificationError, match="outside the verified pyfgsea"):
        verify._verify_wheel(path, sources, expected_version="0.2.0")


def test_receipt_must_be_outside_verified_repository(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(verify.VerificationError, match="receipt must be outside"):
        verify._require_external_receipt(repo / "ignored" / "receipt.json", repo)
    verify._require_external_receipt(tmp_path / "evidence" / "receipt.json", repo)


def test_release_tag_must_be_annotated_and_peel_to_commit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "PyFgsea release test")
    _git(repo, "config", "user.email", "release-test@example.invalid")
    (repo / "tracked.txt").write_text("release source\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "release source")
    commit = _git(repo, "rev-parse", "HEAD").lower()

    _git(repo, "tag", "-a", "v0.2.0-rc1", "-m", "candidate")
    evidence = verify._verify_release_tag(repo, "v0.2.0-rc1", commit)
    assert evidence["annotated"] is True
    assert evidence["peeled_commit"] == commit

    _git(repo, "tag", "v0.2.0-rc2")
    with pytest.raises(verify.VerificationError, match="not an annotated tag"):
        verify._verify_release_tag(repo, "v0.2.0-rc2", commit)

    with pytest.raises(verify.VerificationError, match="must be v0.2.0"):
        verify._verify_release_tag(repo, "v0.2.1-rc1", commit)


def test_installed_probe_requires_venv_paths_and_direct_url_hash(
    tmp_path: Path,
) -> None:
    venv = tmp_path / "venv"
    package = venv / "Lib" / "site-packages" / "pyfgsea" / "__init__.py"
    core = package.parent / "_core.pyd"
    python = venv / "Scripts" / "python.exe"
    package.parent.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    package.write_text("", encoding="utf-8")
    core.write_bytes(b"core")
    python.write_bytes(b"")
    wheel = tmp_path / "pyfgsea.whl"
    wheel.write_bytes(b"wheel")
    wheel_sha = verify._sha256_file(wheel)
    core_sha = verify._sha256_file(core)
    probe = {
        "sys_executable": str(python),
        "sys_prefix": str(venv),
        "base_prefix": str(tmp_path / "base-python"),
        "package_file": str(package),
        "core_file": str(core),
        "core_sha256": core_sha,
        "pyfgsea_version": "0.2.0",
        "distribution_version": "0.2.0",
        "algorithm_revision": "fgsea-1.38-pr178-v1",
        "direct_url": {
            "url": wheel.resolve().as_uri(),
            "archive_info": {"hashes": {"sha256": wheel_sha}},
        },
    }
    evidence = verify._verify_installed_probe(
        probe,
        venv=venv,
        wheel=wheel,
        wheel_evidence={"sha256": wheel_sha, "core_sha256": core_sha},
        expected_version="0.2.0",
        expected_algorithm_revision="fgsea-1.38-pr178-v1",
    )
    assert evidence["package_and_core_inside_venv"] is True
    assert evidence["direct_url_wheel_sha256"] == wheel_sha

    probe["direct_url"] = json.loads(json.dumps(probe["direct_url"]))
    probe["direct_url"]["archive_info"]["hashes"]["sha256"] = "0" * 64
    with pytest.raises(verify.VerificationError, match="direct_url wheel SHA-256"):
        verify._verify_installed_probe(
            probe,
            venv=venv,
            wheel=wheel,
            wheel_evidence={"sha256": wheel_sha, "core_sha256": core_sha},
            expected_version="0.2.0",
            expected_algorithm_revision="fgsea-1.38-pr178-v1",
        )
