import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_thread_scaling_supplement():
    """Generates thread scaling speedup curve from benchmark results."""
    print("Generating Thread Scaling Speedup Curve...")

    # Locate data
    data_path = Path("results/benchmark/benchmark_thread_scaling.csv")

    if data_path.exists():
        df = pd.read_csv(data_path)
    else:
        print(f"Warning: {data_path} not found. Using dummy data for visualization.")
        df = pd.DataFrame(
            {
                "Threads": [1, 2, 4, 8, 16],
                "Time": [1.17, 0.64, 0.41, 0.27, 0.25],
                "Memory": [7.2, 12.7, 13.9, 16.2, 18.4],
            }
        )

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
    print(f"Saved {save_path}")


if __name__ == "__main__":
    plot_thread_scaling_supplement()
