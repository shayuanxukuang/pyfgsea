import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from repro.evidence_receipt import (  # noqa: E402
    capture_git_state,
    sha256_file,
    verify_pyfgsea_installation,
    write_evidence_receipt,
)


def plot_thread_scaling_supplement():
    """Generates thread scaling speedup curve from benchmark results."""
    initial_git_state = capture_git_state()
    pyfgsea_identity = verify_pyfgsea_installation()
    print("Generating Thread Scaling Speedup Curve...")

    # Locate data
    data_path = Path("results/benchmark/benchmark_thread_scaling.csv")

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Thread-scaling evidence is missing: {data_path}. "
            "Run the benchmark before generating this figure."
        )
    upstream_receipt_path = data_path.with_name("benchmark_thread_scaling.receipt.json")
    if not upstream_receipt_path.is_file():
        raise FileNotFoundError(
            f"Thread benchmark receipt is missing: {upstream_receipt_path}"
        )
    upstream = json.loads(upstream_receipt_path.read_text(encoding="utf-8"))
    if upstream.get("git", {}).get("clean") is not True:
        raise ValueError("Thread benchmark receipt was not captured from a clean Git tree")
    if upstream.get("pyfgsea", {}).get("version") != pyfgsea_identity["version"]:
        raise ValueError("Thread benchmark belongs to a different PyFgsea lane")
    recorded_hash = (
        upstream.get("outputs", {}).get("measurements", {}).get("sha256")
    )
    if recorded_hash != sha256_file(data_path):
        raise ValueError("Thread benchmark table does not match its receipt")
    df = pd.read_csv(data_path)
    required = {"Threads", "Time", "Memory"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{data_path} is missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"{data_path} contains no benchmark rows")
    numeric = df[["Threads", "Time", "Memory"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"{data_path} contains non-finite benchmark values")
    df[["Threads", "Time", "Memory"]] = numeric
    if df["Threads"].duplicated().any() or (df["Threads"] <= 0).any():
        raise ValueError(f"{data_path} must contain unique positive thread counts")
    if (df["Time"] <= 0).any() or (df["Memory"] < 0).any():
        raise ValueError(f"{data_path} contains invalid time or memory values")
    if 1 not in df["Threads"].values:
        raise ValueError(f"{data_path} must contain a one-thread baseline")

    # Calculate Speedup
    t1 = df.loc[df["Threads"] == 1, "Time"].values[0]
    df["Speedup"] = t1 / df["Time"]

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Speedup Curve
    color = "tab:blue"
    ax1.set_xlabel("Number of Threads")
    ax1.set_ylabel("Speedup (vs 1 Thread)", color=color)
    ax1.plot(
        df["Threads"],
        df["Speedup"],
        marker="o",
        linestyle="-",
        color=color,
        linewidth=2,
        label="Speedup",
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_xticks(df["Threads"])

    # Ideal Linear Speedup (Reference)
    ax1.plot(
        [1, 16], [1, 16], linestyle="--", color="gray", alpha=0.5, label="Ideal Linear"
    )

    # Memory Curve (Secondary Axis)
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Peak RSS (MB)", color=color)
    ax2.plot(
        df["Threads"],
        df["Memory"],
        marker="s",
        linestyle="--",
        color=color,
        linewidth=2,
        label="Memory",
    )
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Thread Scaling: Speedup & Memory Stability")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.grid(True, alpha=0.3)

    # Annotation for >4 threads saturation (if applicable)
    if 8 in df["Threads"].values:
        speedup_8 = df.loc[df["Threads"] == 8, "Speedup"].values[0]
        # Only annotate if speedup is significantly less than ideal
        if speedup_8 < 6:
            ax1.annotate(
                "Diminishing Returns",
                xy=(8, speedup_8),
                xytext=(8, speedup_8 - 1.5),
                arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=8),
                fontsize=9,
            )

    plt.tight_layout()

    out_dir = Path("supplementary_figures")
    out_dir.mkdir(exist_ok=True)
    save_path = out_dir / "Supp_Fig3_Thread_Scaling.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    write_evidence_receipt(
        save_path.with_suffix(".receipt.json"),
        script=Path(__file__),
        parameters={"speedup_baseline_threads": 1},
        inputs={
            "thread_scaling_table": data_path,
            "thread_scaling_receipt": upstream_receipt_path,
        },
        outputs={"figure": save_path},
        git_state=initial_git_state,
    )
    print(f"Saved {save_path}")


if __name__ == "__main__":
    plot_thread_scaling_supplement()
