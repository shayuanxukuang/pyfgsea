from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_gse155254_0_2_0_figure2_candidate.py"
)
SPEC = importlib.util.spec_from_file_location("figure2_reproduction", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
figure2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(figure2)


def test_release_tag_version() -> None:
    assert figure2.RELEASE_TAG_PATTERN.fullmatch("v0.2.0-rc7")
    assert not figure2.RELEASE_TAG_PATTERN.fullmatch("v0.2.0-rc6")
    assert not figure2.RELEASE_TAG_PATTERN.fullmatch("v0.2.0-rc5")
    assert not figure2.RELEASE_TAG_PATTERN.fullmatch("v0.2.0-rc4")
    assert not figure2.RELEASE_TAG_PATTERN.fullmatch("v0.2.0")
    assert not figure2.RELEASE_TAG_PATTERN.fullmatch("v0.2.0-rc3")


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_parameter_file_matches_runner() -> None:
    evidence = figure2._load_figure2_parameters()
    record = evidence["record"]
    assert set(record) == {"schema_version", "dataset", "recorded_on", "parameters"}
    assert record["schema_version"] == 1
    assert record["dataset"] == "GSE155254"
    assert record["recorded_on"] == "2026-09-01"
    assert record["parameters"] == figure2.RECORDED_PARAMETERS
    assert record["parameters"]["mode"] == "aligned"
    assert record["parameters"]["bin_width"] == 0
    assert record["parameters"]["use_nes_cache"] is False
    assert record["parameters"]["pathway_size_policy"] == "exact"
    assert "pathway_size_policy" not in figure2.PARAMETERS


def test_fgsea_reference() -> None:
    contract = figure2._load_reference_contract()
    assert contract["profile"] == "current_conformance"
    assert contract["fgsea_alignment_target_version"] == "1.38.0"
    assert contract["r_reference_run_performed"] is False
    assert "does not execute R fgsea" in contract["interpretation"]


def test_output_directory(tmp_path: Path) -> None:
    with pytest.raises(figure2.Figure2Error, match="outside"):
        figure2._require_external_output(figure2.REPO_ROOT / "results" / "figure2")
    external = tmp_path / "figure2"
    assert figure2._require_external_output(external) == external.resolve()
    external.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        figure2._require_external_output(external)


def test_clean_annotated_tag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "PyFgsea Figure 2 test")
    _git(repo, "config", "user.email", "figure2-test@example.invalid")
    (repo / "tracked.txt").write_text("release\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "release")
    commit = _git(repo, "rev-parse", "HEAD").lower()
    _git(repo, "tag", "-a", "v0.2.0-rc7", "-m", "RC7")
    monkeypatch.setattr(figure2, "REPO_ROOT", repo)

    state = figure2._capture_release_git_state(commit, "v0.2.0-rc7")
    assert state["clean"] is True
    assert state["release_tag"]["annotated"] is True
    assert state["release_tag"]["peeled_commit"] == commit

    (repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(figure2.Figure2Error, match="checkout contains"):
        figure2._capture_release_git_state(commit, "v0.2.0-rc7")


def test_result_validation() -> None:
    pathways = [f"Pathway {index:02d}" for index in range(figure2.EXPECTED_N_PATHWAYS)]
    rows = []
    for window_id in range(figure2.EXPECTED_N_WINDOWS):
        for pathway_index, pathway in enumerate(pathways, start=1):
            rows.append(
                {
                    "Pathway": pathway,
                    "NES": float(pathway_index),
                    "P-value": pathway_index / 1000.0,
                    "padj": figure2.EXPECTED_N_PATHWAYS / 1000.0,
                    "status": "resolved",
                    "window_id": window_id,
                    "pt_mid": window_id / 100.0,
                }
            )
    results = pd.DataFrame(rows)
    validation = figure2._validate_results(results)
    assert validation["n_rows"] == 2666
    assert validation["complete_grid"] is True
    assert validation["duplicate_key_count"] == 0
    assert validation["status_counts"] == {"resolved": 2666}
    assert validation["bh"]["max_absolute_difference"] <= figure2.BH_ATOL
    assert figure2._audit_results is figure2._validate_results
    assert figure2.BindingError is figure2.Figure2Error

    unresolved = results.copy()
    unresolved.loc[0, "status"] = "eps_floor"
    with pytest.raises(figure2.Figure2Error, match="unresolved rows"):
        figure2._validate_results(unresolved)

    incomplete = results.iloc[:-1].copy()
    with pytest.raises(figure2.Figure2Error, match="not 2666"):
        figure2._validate_results(incomplete)

    invalid_p = results.copy()
    invalid_p.loc[0, "P-value"] = 1.01
    with pytest.raises(figure2.Figure2Error, match=r"p-value is in \(0, 1\]"):
        figure2._validate_results(invalid_p)

    invalid_padj = results.copy()
    invalid_padj.loc[0, "padj"] = -0.01
    with pytest.raises(figure2.Figure2Error, match=r"padj is in \[0, 1\]"):
        figure2._validate_results(invalid_padj)


def test_artifact_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = tmp_path / "downloaded-bundle"
    dist = bundle / "dist"
    evidence_dir = bundle / "evidence"
    dist.mkdir(parents=True)
    evidence_dir.mkdir()
    sdist = dist / "pyfgsea-0.2.0rc7.tar.gz"
    sdist.write_bytes(b"sdist")
    core = b"MZ-figure2-core"
    wheel = dist / "pyfgsea-0.2.0rc7-cp38-abi3-win_amd64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("pyfgsea/_core.pyd", core)
    junit = evidence_dir / "installed-tests.junit.xml"
    junit.write_text(
        '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0" />'
        "</testsuites>",
        encoding="utf-8",
    )
    core_sha = hashlib.sha256(core).hexdigest()
    commit = "1" * 40
    tree = "2" * 40
    tag_object = "3" * 40
    tag = {
        "name": "v0.2.0-rc7",
        "annotated": True,
        "tag_object": tag_object,
        "peeled_commit": commit,
    }
    payload = {
        "schema_version": 1,
        "status": "passed",
        "all_artifact_chain_gates_passed": True,
        "expected": {
            "cargo_version": "0.2.0-rc7",
            "pyfgsea_version": "0.2.0rc7",
            "algorithm_revision": "fgsea-1.38-pr178-v1",
        },
        "artifact_bundle": {
            "schema_version": 1,
            "layout": "receipt-parent-parent-v1",
            "bundle_root_relative_to_receipt": "..",
            "sdist": f"dist/{sdist.name}",
            "wheel": f"dist/{wheel.name}",
            "installed_tests_junit": "evidence/installed-tests.junit.xml",
        },
        "git": {
            "commit": commit,
            "tree": tree,
            "clean_before_and_after": True,
            "release_tag": tag,
            "source_manifest": {},
        },
        "artifact_chain": {
            "sdist": {
                "path": "/expired/runner/path/pyfgsea-0.2.0rc7.tar.gz",
                "bundle_path": f"dist/{sdist.name}",
                "sha256": _sha256(sdist),
                "verified_source_manifest_sha256": "4" * 64,
                "pyfgsea_source_set_exact": True,
                "native_binary_count": 0,
                "cargo_version": "0.2.0-rc7",
                "metadata_version": "0.2.0rc7",
            },
            "wheel": {
                "path": "/expired/runner/path/pyfgsea-0.2.0rc7.whl",
                "bundle_path": f"dist/{wheel.name}",
                "sha256": _sha256(wheel),
                "build_input_sdist_sha256": _sha256(sdist),
                "wheel_built_from_verified_sdist": True,
                "wheel_member_boundary_exact": True,
                "pyfgsea_source_set_exact": True,
                "metadata_version": "0.2.0rc7",
                "verified_source_manifest_sha256": "4" * 64,
                "core_member": "pyfgsea/_core.pyd",
                "core_sha256": core_sha,
            },
            "installed": {
                "core_sha256": core_sha,
                "direct_url_wheel_sha256": _sha256(wheel),
                "pyfgsea_version": "0.2.0rc7",
                "distribution_version": "0.2.0rc7",
                "algorithm_revision": "fgsea-1.38-pr178-v1",
                "package_and_core_inside_venv": True,
            },
        },
        "installed_tests": {
            "status": "passed",
            "git_commit": commit,
            "test_paths": ["tests", "repro/figure1_dual_lane/test_pipeline.py"],
            "test_source_manifest": {
                "tests/test_example.py": {"sha256": "5" * 64, "size": 1}
            },
            "test_source_manifest_sha256": "6" * 64,
            "pytest_version": "8.4.2",
            "trajectory_extra_installed": True,
            "isolated_python": True,
            "import_mode": "importlib",
            "cwd": "/expired/runner/test-work",
            "cwd_outside_worktree": True,
            "wheel_sha256": _sha256(wheel),
            "junit": {
                "bundle_path": "evidence/installed-tests.junit.xml",
                "bytes": junit.stat().st_size,
                "sha256": _sha256(junit),
            },
            "counts": {
                "passed": 1,
                "total": 1,
                "failed": 0,
                "errors": 0,
                "skipped": 0,
            },
        },
    }
    receipt = evidence_dir / "receipt.json"
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    git_state = {"commit": commit, "tree": tree, "release_tag": tag}
    monkeypatch.setattr(
        figure2,
        "_reverify_release_artifacts",
        lambda *args: {
            "source_manifest_sha256": "4" * 64,
            "source_file_count": 0,
            "source_manifest_matches_commit": True,
            "sdist_reverified": True,
            "wheel_reverified": True,
        },
    )
    evidence = figure2._verify_artifact_receipt(receipt, git_state)
    assert evidence["wheel"]["core_sha256"] == core_sha
    assert evidence["wheel"]["path"] == str(wheel.resolve())
    assert evidence["installed_tests"]["counts"]["passed"] == 1

    failed_tests = json.loads(json.dumps(payload))
    failed_tests["installed_tests"]["counts"]["passed"] = 0
    failed_tests["installed_tests"]["counts"]["failed"] = 1
    receipt.write_text(json.dumps(failed_tests), encoding="utf-8")
    with pytest.raises(figure2.Figure2Error, match="did not pass"):
        figure2._verify_artifact_receipt(receipt, git_state)

    escaped_bundle = json.loads(json.dumps(payload))
    escaped_bundle["artifact_chain"]["wheel"]["bundle_path"] = "../outside.whl"
    escaped_bundle["artifact_bundle"]["wheel"] = "../outside.whl"
    receipt.write_text(json.dumps(escaped_bundle), encoding="utf-8")
    with pytest.raises(figure2.Figure2Error, match="bundle_path.*leaves"):
        figure2._verify_artifact_receipt(receipt, git_state)

    missing_manifest = json.loads(json.dumps(payload))
    del missing_manifest["git"]["source_manifest"]
    receipt.write_text(json.dumps(missing_manifest), encoding="utf-8")
    with pytest.raises(figure2.Figure2Error, match="source manifest"):
        figure2._verify_artifact_receipt(receipt, git_state)

    false_exact = json.loads(json.dumps(payload))
    false_exact["artifact_chain"]["wheel"]["pyfgsea_source_set_exact"] = False
    receipt.write_text(json.dumps(false_exact), encoding="utf-8")
    with pytest.raises(figure2.Figure2Error, match="wheel PyFgsea file list"):
        figure2._verify_artifact_receipt(receipt, git_state)

    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("changed.txt", "changed")
    with pytest.raises(figure2.Figure2Error, match="wheel is missing"):
        figure2._verify_artifact_receipt(receipt, git_state)


def test_atomic_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    staging = tmp_path / ".figure2.incomplete-test"
    output = tmp_path / "figure2"
    staging.mkdir()
    (staging / "trajectory_results.csv").write_text("result\n", encoding="utf-8")
    original_replace = figure2.os.replace

    def fail_manifest_replace(source: str | Path, destination: str | Path) -> None:
        if str(source).endswith(".run_manifest.json.pending"):
            raise OSError("simulated manifest write failure")
        original_replace(source, destination)

    monkeypatch.setattr(figure2.os, "replace", fail_manifest_replace)
    with pytest.raises(OSError, match="simulated manifest write failure"):
        figure2._publish_evidence(staging, output, {"verification_status": "verified"})
    assert not staging.exists()
    assert not output.exists()
    assert not (output / "run_manifest.json").exists()
