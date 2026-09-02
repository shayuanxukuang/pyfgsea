from __future__ import annotations

import importlib.util
import io
import json
import os
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
        "src/lib.rs": (
            b'const ALGORITHM_REVISION: &str = "fgsea-test-revision";\n'
            b"pub fn example() {}\n"
        ),
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


def _probe_payload(
    *,
    venv: Path,
    executable: Path,
    package: Path,
    core: Path,
    wheel: Path,
) -> tuple[dict[str, object], dict[str, str]]:
    wheel_sha = verify._sha256_file(wheel)
    core_sha = verify._sha256_file(core)
    probe: dict[str, object] = {
        "sys_executable": str(executable),
        "sys_prefix": str(venv),
        "base_prefix": str(venv.parent / "base-python"),
        "package_file": str(package),
        "core_file": str(core),
        "core_sha256": core_sha,
        "pyfgsea_version": "0.2.0",
        "distribution_version": "0.2.0",
        "algorithm_revision": "fgsea-test-revision",
        "direct_url": {
            "url": wheel.resolve().as_uri(),
            "archive_info": {"hashes": {"sha256": wheel_sha}},
        },
    }
    return probe, {"sha256": wheel_sha, "core_sha256": core_sha}


def test_algorithm_revision_is_derived_from_committed_rust_source() -> None:
    source = _sources()["src/lib.rs"]
    assert verify._committed_algorithm_revision(source) == "fgsea-test-revision"
    with pytest.raises(verify.VerificationError, match="exactly one"):
        verify._committed_algorithm_revision(b"pub fn no_revision() {}\n")


def test_cargo_rc_version_maps_to_python_distribution_version() -> None:
    assert verify._distribution_version("0.2.0-rc3") == "0.2.0rc3"
    assert verify._distribution_version("1.0.0") == "1.0.0"


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
    with pytest.raises(verify.VerificationError, match="sdist sources differ"):
        verify._verify_sdist(path, sources, expected_version="0.2.0")


def test_sdist_rejects_extra_nonpackage_release_input(tmp_path: Path) -> None:
    sources = _sources()
    path = tmp_path / "pyfgsea-0.2.0.tar.gz"
    _write_sdist(path, sources, **{"build.rs": b"fn main() {}\n"})
    with pytest.raises(verify.VerificationError, match="sdist sources differ"):
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


def test_artifact_bundle_layout_is_relocatable(tmp_path: Path) -> None:
    bundle = tmp_path / "downloaded-bundle"
    output_dir = bundle / "dist"
    receipt = bundle / "evidence" / "receipt.json"

    assert verify._require_portable_bundle_layout(output_dir, receipt) == bundle.resolve()

    with pytest.raises(verify.VerificationError, match="portable <bundle>/dist"):
        verify._require_portable_bundle_layout(bundle / "other-dist", receipt)
    with pytest.raises(verify.VerificationError, match="portable <bundle>/evidence"):
        verify._require_portable_bundle_layout(output_dir, bundle / "receipt.json")


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

    _git(repo, "tag", "-a", "v1.2.3-rc4", "-m", "candidate")
    evidence = verify._verify_release_tag(repo, "v1.2.3-rc4", commit)
    assert evidence["annotated"] is True
    assert evidence["peeled_commit"] == commit
    assert evidence["cargo_version"] == "1.2.3-rc4"
    assert evidence["base_version"] == "1.2.3"

    _git(repo, "tag", "-a", "v2.0.0", "-m", "generic release")
    generic = verify._verify_release_tag(repo, "v2.0.0", commit)
    assert generic["cargo_version"] == "2.0.0"

    _git(repo, "tag", "v1.2.3-rc5")
    with pytest.raises(verify.VerificationError, match="not an annotated tag"):
        verify._verify_release_tag(repo, "v1.2.3-rc5", commit)

    with pytest.raises(verify.VerificationError, match="vMAJOR.MINOR.PATCH"):
        verify._verify_release_tag(repo, "release-1.2.3", commit)


def test_installed_probe_requires_venv_paths_and_direct_url_hash(
    tmp_path: Path,
) -> None:
    venv = tmp_path / "venv"
    package = venv / "Lib" / "site-packages" / "pyfgsea" / "__init__.py"
    core = package.parent / "_core.pyd"
    python = verify._venv_python(venv)
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
        "algorithm_revision": "fgsea-test-revision",
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
        expected_algorithm_revision="fgsea-test-revision",
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
            expected_algorithm_revision="fgsea-test-revision",
        )


@pytest.mark.skipif(os.name == "nt", reason="POSIX venv launcher symlink semantics")
def test_installed_probe_accepts_posix_venv_launcher_symlink(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    package = venv / "lib" / "python3.11" / "site-packages" / "pyfgsea" / "__init__.py"
    core = package.parent / "_core.so"
    package.parent.mkdir(parents=True)
    package.write_text("", encoding="utf-8")
    core.write_bytes(b"core")

    base_python = tmp_path / "base-python" / "bin" / "python3.11"
    base_python.parent.mkdir(parents=True)
    base_python.write_bytes(b"base interpreter")
    launcher = venv / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(base_python)

    wheel = tmp_path / "pyfgsea.whl"
    wheel.write_bytes(b"wheel")
    probe, wheel_evidence = _probe_payload(
        venv=venv,
        executable=base_python.resolve(),
        package=package,
        core=core,
        wheel=wheel,
    )
    evidence = verify._verify_installed_probe(
        probe,
        venv=venv,
        wheel=wheel,
        wheel_evidence=wheel_evidence,
        expected_version="0.2.0",
        expected_algorithm_revision="fgsea-test-revision",
    )
    assert evidence["python_executable"] == str(base_python.resolve())
    assert evidence["venv_python_launcher"] == str(launcher.absolute())


@pytest.mark.skipif(os.name == "nt", reason="POSIX venv launcher symlink semantics")
def test_installed_probe_rejects_wrong_resolved_python_and_core_escape(
    tmp_path: Path,
) -> None:
    venv = tmp_path / "venv"
    package = venv / "lib" / "python3.11" / "site-packages" / "pyfgsea" / "__init__.py"
    core = package.parent / "_core.so"
    package.parent.mkdir(parents=True)
    package.write_text("", encoding="utf-8")
    core.write_bytes(b"core")

    base_python = tmp_path / "base-python" / "bin" / "python3.11"
    base_python.parent.mkdir(parents=True)
    base_python.write_bytes(b"base interpreter")
    launcher = venv / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(base_python)

    wheel = tmp_path / "pyfgsea.whl"
    wheel.write_bytes(b"wheel")
    probe, wheel_evidence = _probe_payload(
        venv=venv,
        executable=base_python.resolve(),
        package=package,
        core=core,
        wheel=wheel,
    )

    other_python = tmp_path / "other-python"
    other_python.write_bytes(b"wrong interpreter")
    wrong_python_probe = dict(probe)
    wrong_python_probe["sys_executable"] = str(other_python.resolve())
    with pytest.raises(verify.VerificationError, match="fresh venv launcher"):
        verify._verify_installed_probe(
            wrong_python_probe,
            venv=venv,
            wheel=wheel,
            wheel_evidence=wheel_evidence,
            expected_version="0.2.0",
            expected_algorithm_revision="fgsea-test-revision",
        )

    outside_core = tmp_path / "outside" / "_core.so"
    outside_core.parent.mkdir()
    outside_core.write_bytes(b"outside core")
    escaped_core_probe = dict(probe)
    escaped_core_probe["core_file"] = str(outside_core)
    escaped_core_probe["core_sha256"] = verify._sha256_file(outside_core)
    escaped_core_evidence = {
        "sha256": wheel_evidence["sha256"],
        "core_sha256": verify._sha256_file(outside_core),
    }
    with pytest.raises(verify.VerificationError, match="native core.*outside"):
        verify._verify_installed_probe(
            escaped_core_probe,
            venv=venv,
            wheel=wheel,
            wheel_evidence=escaped_core_evidence,
            expected_version="0.2.0",
            expected_algorithm_revision="fgsea-test-revision",
        )


def test_installed_test_manifest_is_bound_to_commit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    pipeline = repo / "repro" / "figure1_dual_lane" / "test_pipeline.py"
    pipeline.parent.mkdir(parents=True)
    (repo / "tests" / "test_example.py").write_text(
        "def test_example():\n    assert True\n", encoding="utf-8"
    )
    pipeline.write_text("def test_pipeline():\n    assert True\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.name", "PyFgsea release test")
    _git(repo, "config", "user.email", "release-test@example.invalid")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "test sources")
    commit = _git(repo, "rev-parse", "HEAD")

    manifest = verify._git_installed_test_manifest(repo, commit)
    assert sorted(manifest) == [
        "repro/figure1_dual_lane/test_pipeline.py",
        "tests/test_example.py",
    ]

    (repo / "tests" / "test_example.py").write_text(
        "def test_example():\n    assert False\n", encoding="utf-8"
    )
    with pytest.raises(verify.VerificationError, match="differs from commit"):
        verify._git_installed_test_manifest(repo, commit)


def test_junit_counts_parse_passed_skipped_failed_and_errors(tmp_path: Path) -> None:
    junit = tmp_path / "installed-tests.junit.xml"
    junit.write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        '<testsuites><testsuite tests="11" failures="2" errors="1" skipped="3" />'
        "</testsuites>",
        encoding="utf-8",
    )
    assert verify._junit_counts(junit) == {
        "passed": 5,
        "total": 11,
        "failed": 2,
        "errors": 1,
        "skipped": 3,
    }


def test_installed_test_evidence_binds_junit_tests_commit_and_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    bundle = tmp_path / "bundle"
    junit = bundle / "evidence" / "installed-tests.junit.xml"
    cwd = bundle / "installed-test-work"
    python = tmp_path / "venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    committed_tests = {
        "tests/test_example.py": b"def test_example():\n    assert True\n",
        "repro/figure1_dual_lane/test_pipeline.py": (
            b"def test_pipeline():\n    assert True\n"
        ),
    }
    monkeypatch.setattr(
        verify,
        "_git_installed_test_manifest",
        lambda _repo, _commit: committed_tests,
    )

    observed_commands: list[list[str]] = []

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        commands: list[dict[str, object]],
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[bytes]:
        del cwd, commands
        observed_commands.append(command)
        if "import pytest; print(pytest.__version__)" in command:
            return subprocess.CompletedProcess(command, 0, b"8.4.2\n", b"")
        junit_arg = next(item for item in command if item.startswith("--junitxml="))
        junit_path = Path(junit_arg.split("=", 1)[1])
        junit_path.parent.mkdir(parents=True, exist_ok=True)
        junit_path.write_text(
            '<testsuites><testsuite tests="7" failures="0" errors="0" '
            'skipped="2" /></testsuites>',
            encoding="utf-8",
        )
        assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(verify, "_run", fake_run)
    evidence = verify._run_installed_tests(
        python,
        repo=repo,
        commit="a" * 40,
        cwd=cwd,
        junit_path=junit,
        bundle_root=bundle,
        wheel_sha256="b" * 64,
        commands=[],
    )

    pytest_command = observed_commands[-1]
    assert pytest_command[1:3] == ["-I", "-c"]
    assert pytest_command[3] == verify.INSTALLED_TEST_BOOTSTRAP
    assert pytest_command[4:6] == [str(repo.resolve()), "-q"]
    assert "--import-mode=importlib" in pytest_command
    assert str(repo / "tests") in pytest_command
    assert str(repo / "repro" / "figure1_dual_lane" / "test_pipeline.py") in pytest_command
    assert evidence["git_commit"] == "a" * 40
    assert evidence["wheel_sha256"] == "b" * 64
    assert evidence["artifact_import_preloaded_before_test_support_path"] is True
    assert evidence["test_support_path_appended_after_site_packages"] is True
    assert evidence["junit"]["bundle_path"] == "evidence/installed-tests.junit.xml"
    assert evidence["junit"]["sha256"] == verify._sha256_file(junit)
    assert evidence["counts"] == {
        "passed": 5,
        "total": 7,
        "failed": 0,
        "errors": 0,
        "skipped": 2,
    }
    assert sorted(evidence["test_source_manifest"]) == sorted(committed_tests)


def test_installed_test_failure_cannot_write_success_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "bundle" / "evidence" / "receipt.json"

    def fail_verification(_args: object) -> dict[str, object]:
        raise verify.VerificationError("installed test suite failed")

    monkeypatch.setattr(verify, "_build_and_verify", fail_verification)
    result = verify.main(
        [
            "--repo",
            str(tmp_path / "repo"),
            "--commit",
            "a" * 40,
            "--release-tag",
            "v0.2.0-rc3",
            "--output-dir",
            str(tmp_path / "bundle" / "dist"),
            "--venv",
            str(tmp_path / "bundle" / "venv"),
            "--receipt",
            str(receipt),
        ]
    )
    assert result == 1
    assert not receipt.exists()
