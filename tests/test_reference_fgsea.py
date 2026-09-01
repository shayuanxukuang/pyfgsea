"""Frozen-reference metadata plus an optional R fgsea 1.38.0 smoke lane."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from pyfgsea import run_gsea


ROOT = Path(__file__).resolve().parents[1]


def test_reference_manifest_freezes_legacy_and_current_lanes():
    manifest = json.loads((ROOT / "reference_manifest.json").read_text("utf-8"))
    legacy = manifest["profiles"]["legacy_publication"]
    current = manifest["profiles"]["current_conformance"]

    assert manifest["schema_version"] == 2
    assert (legacy["r_version"], legacy["fgsea_version"], legacy["pyfgsea_version"]) == (
        "4.4.3",
        "1.32.2",
        "0.1.4",
    )
    assert (
        current["r_version"],
        current["bioconductor_version"],
        current["fgsea_version"],
        current["pyfgsea_version"],
    ) == ("4.6.0", "3.23", "1.38.0", "0.2.0rc4")
    assert current["pyfgsea_release_target"] == "0.2.0"
    assert current["upstream_alignment_basis"]["merge_commit"] == (
        "9a06694dfc7b54a0a698061a97db15945ede725c"
    )
    assert legacy["fgsea_commit"] == "4620281"
    assert current["fgsea_commit"] == "1fe4644"
    for profile in (legacy, current):
        assert "pyfgsea_commit" in profile
        assert "dataset_sha256" in profile
        assert "parameters" in profile
    assert len(current["fgsea_source"]["sha256"]) == 64
    assert len(legacy["fgsea_source"]["sha256"]) == 64

    host = manifest["host_validation"]
    assert host["docker_profiles_static_validation_only"] is True
    local = host["isolated_local_validation"]["current_conformance"]
    assert local["status"] == "not-run-for-clean-rc4"
    assert local["artifact_binding"] is False


def _reference_rscript():
    configured = os.environ.get("PYFGSEA_REFERENCE_RSCRIPT")
    if configured and not Path(configured).is_file():
        pytest.fail(f"configured PYFGSEA_REFERENCE_RSCRIPT does not exist: {configured}")
    executable = configured or shutil.which("Rscript")
    if not executable:
        pytest.skip("Rscript is not available; frozen R conformance lane was not requested")
    version = subprocess.run(
        [
            executable,
            "--vanilla",
            "-e",
            "cat(as.character(getRversion()), '|', "
            "as.character(packageVersion('fgsea')), '|', "
            "as.character(BiocManager::version()), sep='')",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    expected = "4.6.0|1.38.0|3.23"
    if version.returncode != 0 or version.stdout.strip() != expected:
        message = version.stderr.strip() or version.stdout.strip()
        if configured:
            pytest.fail(
                "configured current reference is not exact R 4.6.0 / "
                f"Bioconductor 3.23 / fgsea 1.38.0: {message}"
            )
        pytest.skip("exact R 4.6.0 / Bioconductor 3.23 / fgsea 1.38.0 is unavailable")
    return executable


def test_optional_r_fgsea_138_es_conformance(tmp_path):
    rscript = _reference_rscript()
    genes = [f"G{index:03d}" for index in range(200)]
    scores = np.cos(np.arange(200) * 0.11) + np.linspace(2.0, -2.0, 200)
    pathways = {
        "top": genes[:20],
        "middle": genes[80:105],
        "bottom": genes[-20:],
    }
    stats_path = tmp_path / "stats.tsv"
    sets_path = tmp_path / "sets.tsv"
    output_path = tmp_path / "fgsea.tsv"
    pd.DataFrame({"gene": genes, "score": scores}).to_csv(
        stats_path, sep="\t", index=False
    )
    pd.DataFrame(
        [(name, gene) for name, members in pathways.items() for gene in members],
        columns=["pathway", "gene"],
    ).to_csv(sets_path, sep="\t", index=False)

    expression = (
        "s<-read.delim(commandArgs(TRUE)[1]);"
        "m<-read.delim(commandArgs(TRUE)[2]);"
        "stats<-setNames(s$score,s$gene);"
        "paths<-split(m$gene,m$pathway);"
        "set.seed(138);"
        "z<-fgsea::fgseaMultilevel(paths,stats,minSize=1,maxSize=199,"
        "sampleSize=21,nPermSimple=1001,eps=0,scoreType='std');"
        "write.table(z[,.(pathway,ES,NES,pval,log2err)],commandArgs(TRUE)[3],"
        "sep='\\t',row.names=FALSE,quote=FALSE)"
    )
    subprocess.run(
        [rscript, "--vanilla", "-e", expression, str(stats_path), str(sets_path), str(output_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    reference = pd.read_csv(output_path, sep="\t").set_index("pathway")
    observed = run_gsea(
        pd.DataFrame({"gene": genes, "score": scores}),
        pathways,
        gene_col="gene",
        score_col="score",
        min_size=1,
        max_size=199,
        sample_size=21,
        nperm_simple=1001,
        nperm_nes=1000,
        calculate_nes=True,
        eps=0.0,
        seed=138,
    ).set_index("Pathway")

    np.testing.assert_allclose(
        observed.loc[reference.index, "ES"], reference["ES"], rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        observed.loc[reference.index, "NES"],
        reference["NES"],
        rtol=0.12,
        atol=0.05,
    )
    assert np.isfinite(reference["pval"]).all()
    assert np.isfinite(observed.loc[reference.index, "P-value"]).all()
    assert np.isfinite(reference["log2err"]).all()
    assert np.isfinite(observed.loc[reference.index, "log2err"]).all()
    log2_delta = np.abs(
        np.log2(observed.loc[reference.index, "P-value"].to_numpy())
        - np.log2(reference["pval"].to_numpy())
    )
    combined_error = np.sqrt(
        np.square(observed.loc[reference.index, "log2err"].to_numpy())
        + np.square(reference["log2err"].to_numpy())
    )
    assert np.all(log2_delta <= 2.0 * combined_error)
    np.testing.assert_allclose(
        observed.loc[reference.index, "log2err"],
        reference["log2err"],
        rtol=0.25,
        atol=0.2,
    )
