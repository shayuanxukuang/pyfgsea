import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from repro.evidence_receipt import (  # noqa: E402
    capture_git_state,
    sha256_file,
    verify_pyfgsea_installation,
    write_evidence_receipt,
)


def plot_bland_altman(ax, x, y, title="Bland-Altman Plot"):
    """Generates a Bland-Altman plot."""
    mean = (x + y) / 2
    diff = y - x
    md = np.mean(diff)
    sd = np.std(diff, axis=0)

    # Scatter
    ax.scatter(mean, diff, alpha=0.3, s=10, edgecolors="none", color="tab:blue")

    # Lines
    ax.axhline(md, color="red", linestyle="-", label=f"Mean Bias: {md:.4f}")
    ax.axhline(
        md + 1.96 * sd,
        color="gray",
        linestyle="--",
        label=f"+1.96 SD: {md + 1.96 * sd:.4f}",
    )
    ax.axhline(
        md - 1.96 * sd,
        color="gray",
        linestyle="--",
        label=f"-1.96 SD: {md - 1.96 * sd:.4f}",
    )

    # Equivalence margins
    ax.axhline(
        0.1, color="green", linestyle=":", alpha=0.5, label="Equivalence Margin (0.1)"
    )
    ax.axhline(-0.1, color="green", linestyle=":", alpha=0.5)

    # Labels
    ax.set_xlabel("Mean NES (PyFgsea & R)")
    ax.set_ylabel("Difference (PyFgsea - R)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True, alpha=0.3)

    return md, sd


def run_equivalence_test(x, y, margin=0.1):
    """TOST equivalence test."""
    diff = y - x
    # Test against lower bound
    _, p1 = ttest_1samp(diff, -margin, alternative="greater")
    # Test against upper bound
    _, p2 = ttest_1samp(diff, margin, alternative="less")

    return max(p1, p2)


def _load_agreement_values(input_path: Optional[Path] = None) -> pd.DataFrame:
    candidates = []
    if input_path is not None:
        candidates.append(input_path)
    env_path = os.environ.get("PYFGSEA_AGREEMENT_VALUES", "").strip()
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            Path("results/reference/figure1_agreement_values.csv"),
            Path("bioinfor0208/revision/data/figure1_agreement_values.csv"),
        ]
    )

    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        checked = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "Bland-Altman evidence requires a real Figure 1 agreement table. "
            f"No input was found; checked: {checked}"
        )

    frame = pd.read_csv(source)
    required = {"NES_Py", "NES_R"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"{source} contains missing NES comparison values")
    if frame.empty:
        raise ValueError(f"{source} contains no NES comparison rows")
    values = frame[["NES_Py", "NES_R"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{source} contains non-finite NES comparison values")
    frame[["NES_Py", "NES_R"]] = values
    upstream_receipt = source.with_suffix(".receipt.json")
    if not upstream_receipt.is_file():
        raise FileNotFoundError(
            f"Figure 1 agreement receipt is missing: {upstream_receipt}"
        )
    upstream = json.loads(upstream_receipt.read_text(encoding="utf-8"))
    if upstream.get("git", {}).get("clean") is not True:
        raise ValueError("Figure 1 agreement receipt was not captured from a clean Git tree")
    source_hash = sha256_file(source)
    output_hashes = {
        value.get("sha256")
        for value in upstream.get("outputs", {}).values()
        if isinstance(value, dict)
    }
    if source_hash not in output_hashes:
        raise ValueError(
            f"{source} is not bound as an output of {upstream_receipt}"
        )
    frame.attrs["source_path"] = str(source.resolve())
    frame.attrs["source_receipt_path"] = str(upstream_receipt.resolve())
    return frame


def generate_bland_altman_supplement(input_path: Optional[Path] = None):
    initial_git_state = capture_git_state()
    pyfgsea_identity = verify_pyfgsea_installation()
    print("Generating Bland-Altman Plots...")
    df = _load_agreement_values(input_path)
    upstream = json.loads(
        Path(df.attrs["source_receipt_path"]).read_text(encoding="utf-8")
    )
    if upstream.get("pyfgsea", {}).get("version") != pyfgsea_identity["version"]:
        raise ValueError("Figure 1 agreement input belongs to a different PyFgsea lane")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    md, sd = plot_bland_altman(ax, df["NES_R"], df["NES_Py"])

    # Equivalence Test
    p_tost = run_equivalence_test(df["NES_R"], df["NES_Py"], margin=0.1)

    # Add stats
    stats_text = (
        f"Mean Bias: {md:.4f}\n"
        f"SD of Diff: {sd:.4f}\n"
        f"Equivalence (0.1 NES): p={p_tost:.2e}\n"
        f"N={len(df)}"
    )
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    out_dir = Path("supplementary_figures")
    out_dir.mkdir(exist_ok=True)
    save_path = out_dir / "Supp_Fig1_Bland_Altman.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    stats_path = out_dir / "Supp_Fig1_Bland_Altman.stats.csv"
    pd.DataFrame(
        [
            {
                "n": len(df),
                "mean_bias": md,
                "sd_difference": sd,
                "equivalence_margin": 0.1,
                "tost_pvalue": p_tost,
            }
        ]
    ).to_csv(stats_path, index=False)
    write_evidence_receipt(
        save_path.with_suffix(".receipt.json"),
        script=Path(__file__),
        parameters={"equivalence_margin_nes": 0.1, "limits_of_agreement_sd": 1.96},
        inputs={
            "figure1_agreement_values": Path(df.attrs["source_path"]),
            "figure1_agreement_receipt": Path(df.attrs["source_receipt_path"]),
        },
        outputs={"figure": save_path, "statistics": stats_path},
        git_state=initial_git_state,
    )
    print(f"Saved {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    arguments = parser.parse_args()
    generate_bland_altman_supplement(arguments.input)
