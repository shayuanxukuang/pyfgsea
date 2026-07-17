from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

from scp1064_utils import GLOBAL_FIGURES, GLOBAL_TABLES, RESULTS


plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def clean_label(value: object) -> str:
    return str(value).replace("_", " ")


def draw_card(ax, xy, width, height, title, body, face, edge="#4c5a65", title_color="#1b1f23", fontsize=9):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.0,
        facecolor=face,
        edgecolor=edge,
    )
    ax.add_patch(box)
    ax.text(x + 0.035 * width, y + height - 0.28 * height, title, fontsize=fontsize + 1, fontweight="bold", color=title_color)
    ax.text(x + 0.035 * width, y + height - 0.48 * height, fill(body, 34), fontsize=fontsize, color="#2f3437", va="top")


def main() -> None:
    outcome = read_tsv(RESULTS / "scp1064_outcome_alignment_summary.tsv")
    author = read_tsv(RESULTS / "scp1064_author_effect_summary.tsv")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    ax = axes[0]
    if not outcome.empty and "spearman" in outcome:
        plot = outcome.dropna(subset=["spearman"]).copy()
        plot["label"] = (plot["level"].astype(str) + ": " + plot["axis"].map(clean_label)).map(lambda value: fill(value, 42))
        ax.barh(plot["label"].head(18), plot["spearman"].head(18), color="#4C78A8")
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("RNA event/protein Spearman")
    ax.set_title("SCP1064 outcome alignment")
    ax = axes[1]
    if not author.empty and "max_abs_spearman" in author:
        author_labels = author["axis"].map(clean_label).map(lambda value: fill(value, 34))
        ax.barh(author_labels, author["max_abs_spearman"], color="#F58518")
    ax.set_xlabel("Author reference max |rho|")
    ax.set_title("Author effect concordance")
    fig.suptitle("SCP1064 validation: E1-V1 partial", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.5)
    GLOBAL_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(GLOBAL_FIGURES / "supplementary_figure_scp1064.png", dpi=180, bbox_inches="tight")
    fig.savefig(GLOBAL_FIGURES / "supplementary_figure_scp1064.pdf", bbox_inches="tight")
    plt.close(fig)

    claims = read_tsv(GLOBAL_TABLES / "ted_dataset_level_claim_boundary.tsv")
    gse153 = read_tsv(GLOBAL_TABLES / "gse153056_pdl1_outcome_alignment.tsv")
    gse937 = read_tsv(GLOBAL_TABLES / "gse93735_reversal_index.tsv")
    scp = read_tsv(RESULTS / "scp1064_outcome_alignment_summary.tsv")
    shuffle = read_tsv(RESULTS / "scp1064_lightweight_shuffle_summary.tsv")

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    ax = axes[0, 0]
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("A. Frozen validation registry", loc="left", fontweight="bold")
    registry_cards = [
        ("GSE153056", "known source -> RNA event -> PD-L1 protein outcome", "#e8f1fb"),
        ("GSE93735", "LPS inflammatory event -> dexamethasone reversal", "#e8f5e5"),
        ("SCP1064", "CRISPR guide source -> RNA event -> protein readout", "#fff0df"),
    ]
    y = 0.74
    for title, body, face in registry_cards:
        draw_card(ax, (0.05, y), 0.88, 0.18, title, body, face=face, fontsize=9)
        y -= 0.23
    ax.text(0.06, 0.07, "Gates fixed before expression-level interpretation", fontsize=9, color="#5f6368")

    ax = axes[0, 1]
    if not gse153.empty:
        ax.scatter(gse153["event_effect_size"], gse153["pdl1_protein_effect_size"], s=18, alpha=0.75)
        rho = pd.to_numeric(gse153["outcome_correlation"], errors="coerce").dropna()
        direction = pd.to_numeric(gse153["event_outcome_direction_match"], errors="coerce")
        direction_match = direction.mean() if not direction.empty else float("nan")
        if not rho.empty:
            ax.text(
                0.04,
                0.93,
                f"Spearman = {rho.iloc[0]:.3f}\ndirection match = {direction_match:.3f}",
                transform=ax.transAxes,
                va="top",
                bbox=dict(facecolor="white", edgecolor="#c8d0d8", alpha=0.9),
            )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("TED IFNG/PD-L1 RNA event effect")
    ax.set_ylabel("PD-L1 protein outcome effect")
    ax.set_title("B. GSE153056 RNA/protein alignment", loc="left", fontweight="bold")

    ax = axes[1, 0]
    if not gse937.empty:
        numeric = gse937.copy()
        numeric["reversal_fraction"] = pd.to_numeric(numeric["reversal_fraction"], errors="coerce")
        primary = numeric[numeric["role"].eq("primary_positive_reversal_axis")]["reversal_fraction"].max()
        control_max = numeric[numeric["role"].eq("negative_control_axis")]["reversal_fraction"].max()
        secondary = numeric[numeric["role"].eq("secondary_positive_reversal_axis")]["reversal_fraction"].max()
        values = [primary, control_max]
        labels = ["LPS inflammatory event", "matched negative-control max"]
        colors = ["#4C78A8", "#A6A6A6"]
        ax.bar(labels, values, color=colors, width=0.55)
        ax.scatter(["LPS inflammatory event"], [secondary], s=60, marker="D", color="#54A24B", label="secondary cytokine axis")
        ax.axhline(control_max, color="#666666", linestyle="--", linewidth=1.1)
        ax.set_ylim(0, max(values + [secondary]) * 1.35)
        ax.set_ylabel("Positive reversal fraction")
        ax.set_title("C. GSE93735 dexamethasone reversal", loc="left", fontweight="bold")
        ax.text(
            0.03,
            0.92,
            f"primary = {primary:.4f}\ncontrol max = {control_max:.4f}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(facecolor="white", edgecolor="#c8d0d8", alpha=0.9),
        )
        ax.text(
            0.53,
            0.16,
            "Glucocorticoid engagement is evaluated separately;\nit is not plotted as a reversal fraction.",
            transform=ax.transAxes,
            fontsize=8,
            color="#5f6368",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
        )
        ax.legend(loc="upper right", frameon=False, fontsize=8)

    ax = axes[1, 1]
    if not scp.empty:
        plot = scp.dropna(subset=["spearman"]).copy()
        plot = plot[plot["alignment_pass"].astype(str).str.lower().eq("true")].head(8)
        labels = plot["level"].astype(str) + " | " + plot["axis"].map(clean_label) + " | " + plot["protein_name"].astype(str)
        labels = [fill(label, 34) for label in labels]
        ax.barh(labels, plot["spearman"], color="#54A24B")
        if not shuffle.empty:
            gate = shuffle[shuffle["evaluated_for_gate"].astype(str).str.lower().eq("true")]
            passed = int(gate["shuffle_gate_pass"].astype(str).str.lower().eq("true").sum())
            total = int(len(gate))
            ax.text(
                0.03,
                0.08,
                f"lightweight shuffle gate: {passed}/{total} pass",
                transform=ax.transAxes,
                bbox=dict(facecolor="white", edgecolor="#c8d0d8", alpha=0.9),
                fontsize=8,
            )
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("RNA event / protein Spearman rho")
    ax.set_title("D. SCP1064 source-to-protein readouts", loc="left", fontweight="bold")
    fig.suptitle("Public known-source benchmarks validated outcome and reversal boundaries", fontsize=16, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(GLOBAL_FIGURES / "figure2_known_source_validation.png", dpi=180, bbox_inches="tight")
    fig.savefig(GLOBAL_FIGURES / "figure2_known_source_validation.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {GLOBAL_FIGURES / 'supplementary_figure_scp1064.png'}")


if __name__ == "__main__":
    main()
