from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_trajectory_example_produces_real_outputs(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "trajectory_demo.py"),
            "--adata",
            str(ROOT / "repro" / "data" / "toy_trajectory.h5ad"),
            "--gmt",
            str(ROOT / "repro" / "data" / "toy_pathways.gmt"),
            "--outdir",
            str(tmp_path),
            "--window-size",
            "100",
            "--step",
            "50",
            "--min-size",
            "5",
            "--max-size",
            "30",
            "--sample-size",
            "21",
            "--nperm-nes",
            "20",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    table_path = tmp_path / "trajectory_gsea_table.tsv"
    figure_path = tmp_path / "trajectory_demo.png"
    result = pd.read_csv(table_path, sep="\t")
    assert not result.empty
    assert {"Pathway", "NES", "window_id", "pt_mid"}.issubset(result.columns)
    assert set(result["Pathway"]) == {
        "EARLY_RESPONSE",
        "MIDDLE_RESPONSE",
        "LATE_RESPONSE",
    }
    assert result["window_id"].nunique() == 9
    assert figure_path.stat().st_size > 0
    assert "Results:" in completed.stdout
