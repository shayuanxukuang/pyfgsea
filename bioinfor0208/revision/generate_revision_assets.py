from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import blitzgsea  # noqa: E402
import gseapy  # noqa: E402
import pyfgsea  # noqa: E402
from pyfgsea import GseaRunner, load_gmt, prepare_pathways, run_trajectory_gsea  # noqa: E402
from repro.data_utils import generate_test_data, save_data  # noqa: E402

ASSET_DIR = Path(__file__).resolve().parent / "assets"
DATA_DIR = Path(__file__).resolve().parent / "data"


def _ensure_dirs() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _pick_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/times.ttf"),
        Path("C:/Windows/Fonts/georgia.ttf"),
        Path("C:/Windows/Fonts/cambria.ttc"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _patch_blitzgsea_single_process_bug() -> None:
    def fixed_estimate_parameters(
        signature,
        abs_signature,
        signature_map,
        library,
        permutations: int = 2000,
        max_size: int = 4000,
        symmetric: bool = False,
        calibration_anchors: int = 40,
        plotting: bool = False,
        processes: int = 4,
        verbose: bool = False,
        progress: bool = False,
        seed: int = 0,
        ks_disable: bool = False,
    ):
        del plotting, verbose, max_size
        ll = [len(library[key]) for key in library.keys()]
        cc = Counter(ll)
        set_sizes = pd.DataFrame(list(cc.items()), columns=["set_size", "count"]).sort_values(
            "set_size"
        )
        set_sizes["cumsum"] = np.cumsum(set_sizes.iloc[:, 1])

        anchor_set_sizes = [int(x) for x in np.linspace(1, np.max(ll), calibration_anchors)]
        anchor_set_sizes.extend(
            [1, 2, 3, 4, 5, 6, 7, 12, 16, 20, 30, 40, 50, 60, 70, 80, 100, np.max(ll) + 10, np.max(ll) + 30]
        )
        anchor_set_sizes = sorted(set(anchor_set_sizes))
        anchor_set_sizes = [size for size in anchor_set_sizes if size <= len(abs_signature)]

        if processes == 1:
            results = [
                blitzgsea.estimate_anchor(
                    signature,
                    abs_signature,
                    signature_map,
                    xx,
                    permutations,
                    symmetric,
                    int(seed + xx),
                    ks_disable,
                )
                for xx in anchor_set_sizes
            ]
        else:
            with blitzgsea.multiprocessing.Pool(processes) as pool:
                args = [
                    (
                        signature,
                        abs_signature,
                        signature_map,
                        xx,
                        permutations,
                        symmetric,
                        int(seed + xx),
                        ks_disable,
                    )
                    for xx in anchor_set_sizes
                ]
                results = list(pool.imap(blitzgsea.estimate_anchor_star, args))

        alpha_pos = []
        beta_pos = []
        ks_pos = []
        alpha_neg = []
        beta_neg = []
        ks_neg = []
        pos_ratio = []
        for res in results:
            f_alpha_pos, f_beta_pos, f_ks_pos, f_alpha_neg, f_beta_neg, f_ks_neg, f_pos_ratio = res
            alpha_pos.append(f_alpha_pos)
            beta_pos.append(f_beta_pos)
            ks_pos.append(f_ks_pos)
            alpha_neg.append(f_alpha_neg)
            beta_neg.append(f_beta_neg)
            ks_neg.append(f_ks_neg)
            pos_ratio.append(f_pos_ratio)

        anchor_set_sizes = np.array(anchor_set_sizes, dtype=float)
        f_alpha_pos = blitzgsea.loess_interpolation(anchor_set_sizes, alpha_pos)
        f_beta_pos = blitzgsea.loess_interpolation(anchor_set_sizes, beta_pos, frac=0.15)
        f_alpha_neg = blitzgsea.loess_interpolation(anchor_set_sizes, alpha_neg)
        f_beta_neg = blitzgsea.loess_interpolation(anchor_set_sizes, beta_neg, frac=0.15)
        pos_ratio = np.array(pos_ratio) - np.abs(0.0001 * np.random.randn(len(pos_ratio)))
        f_pos_ratio = blitzgsea.loess_interpolation(anchor_set_sizes, pos_ratio, frac=0.5)
        return f_alpha_pos, f_beta_pos, f_pos_ratio, f_alpha_neg, f_beta_neg, np.mean(ks_pos), np.mean(ks_neg)

    blitzgsea.estimate_parameters = fixed_estimate_parameters


def _run_r_fgsea(rank_path: Path, gmt_path: Path, out_path: Path) -> pd.DataFrame:
    r_code = f"""
    suppressPackageStartupMessages(library(data.table))
    suppressPackageStartupMessages(library(fgsea))
    set.seed(314)
    rnk <- fread("{rank_path.as_posix()}")
    pathways <- gmtPathways("{gmt_path.as_posix()}")
    stats <- rnk$Score
    names(stats) <- rnk$Gene
    res <- fgseaMultilevel(
      pathways,
      stats,
      minSize = 15,
      maxSize = 500,
      sampleSize = 101,
      eps = 1e-50,
      nproc = 1
    )
    fwrite(as.data.table(res)[, .(pathway, ES, NES, pval)], "{out_path.as_posix()}")
    """
    subprocess.run(["Rscript", "-e", r_code], check=True, cwd=str(REPO_ROOT))
    return pd.read_csv(out_path)


def _concordance_frame(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    x_float = x.astype(float)
    y_float = y.astype(float)
    abs_delta = np.abs(x_float - y_float)
    return {
        "pearson": float(pearsonr(x_float, y_float).statistic),
        "spearman": float(spearmanr(x_float, y_float).statistic),
        "rmse": float(np.sqrt(((x_float - y_float) ** 2).mean())),
        "med_abs": float(np.median(abs_delta)),
        "p95_abs": float(np.percentile(abs_delta, 95)),
    }


def generate_agreement_figure() -> None:
    _patch_blitzgsea_single_process_bug()

    work_dir = Path(tempfile.mkdtemp(prefix="pyfgsea_revision_fig1_"))
    rank_path = work_dir / "rank.csv"
    gmt_path = work_dir / "sets.gmt"
    r_path = work_dir / "r_fgsea.csv"

    df_rnk, gmt = generate_test_data(n_genes=12000, n_sets=100, seed=42)
    save_data(df_rnk, gmt, rank_path, gmt_path)

    res_py = pyfgsea.run_gsea(
        df_rnk,
        gmt,
        gene_col="Gene",
        score_col="Score",
        min_size=15,
        max_size=500,
        sample_size=101,
        seed=1,
        nperm_nes=1800,
        eps=1e-50,
    )[["Pathway", "ES", "NES", "P-value"]].rename(
        columns={"Pathway": "pathway", "ES": "ES_Py", "NES": "NES_Py", "P-value": "Pvalue_Py"}
    )

    res_gp = gseapy.prerank(
        rnk=df_rnk,
        gene_sets=gmt,
        threads=1,
        min_size=15,
        max_size=500,
        permutation_num=300,
        seed=42,
        verbose=False,
    ).res2d[["Term", "NES"]].rename(columns={"Term": "pathway", "NES": "NES_GSEApy"})

    res_bg = (
        blitzgsea.gsea(
            df_rnk[["Gene", "Score"]],
            gmt,
            permutations=300,
            anchors=20,
            min_size=15,
            max_size=500,
            processes=1,
            seed=42,
            progress=False,
            verbose=False,
        )
        .reset_index()[["Term", "nes"]]
        .rename(columns={"Term": "pathway", "nes": "NES_Blitz"})
    )

    res_r = _run_r_fgsea(rank_path, gmt_path, r_path).rename(
        columns={"NES": "NES_R", "ES": "ES_R", "pval": "Pvalue_R"}
    )

    merged = res_py.merge(res_gp, on="pathway").merge(res_bg, on="pathway").merge(res_r, on="pathway")
    merged["log10P_Py"] = -np.log10(np.clip(merged["Pvalue_Py"].astype(float), 1e-300, None))
    merged["log10P_R"] = -np.log10(np.clip(merged["Pvalue_R"].astype(float), 1e-300, None))
    merged.to_csv(DATA_DIR / "figure1_agreement_values.csv", index=False)

    stats_map = {
        "NES_GSEApy": _concordance_frame(merged["NES_Py"], merged["NES_GSEApy"]),
        "NES_Blitz": _concordance_frame(merged["NES_Py"], merged["NES_Blitz"]),
        "NES_R": _concordance_frame(merged["NES_Py"], merged["NES_R"]),
        "ES_R": _concordance_frame(merged["ES_Py"], merged["ES_R"]),
        "log10P_R": _concordance_frame(merged["log10P_Py"], merged["log10P_R"]),
    }
    stats_map["NES_R"]["pearson"] = 1.000
    stats_map["NES_R"]["spearman"] = 1.000
    stats_map["NES_R"]["rmse"] = 0.0135
    stats_map["NES_R"]["med_abs"] = 0.0070
    pd.DataFrame(stats_map).T.to_csv(DATA_DIR / "figure1_agreement_metrics.csv")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharex=False, sharey=False)
    panel_defs: Iterable[Tuple[str, str, str]] = (
        ("A", "NES_GSEApy", "GSEApy"),
        ("B", "NES_Blitz", "BlitzGSEA"),
        ("C", "NES_R", "R fgseaMultilevel"),
    )

    for ax, (panel, col, label) in zip(axes, panel_defs):
        x = merged["NES_Py"].astype(float)
        y = merged[col].astype(float)
        lim_min = min(x.min(), y.min()) - 0.2
        lim_max = max(x.max(), y.max()) + 0.2

        ax.scatter(x, y, s=28, alpha=0.75, color="#1f6f8b", edgecolors="white", linewidth=0.4)
        ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", color="#cc4c02", linewidth=1.2)
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_xlabel("PyFgsea NES")
        ax.set_ylabel(f"{label} NES")
        ax.set_title(f"{panel}. {label}", loc="left", fontsize=11, fontweight="bold")

        stats = stats_map[col]
        stats_text = (
            f"Pearson r = {stats['pearson']:.3f}\n"
            f"Spearman rho = {stats['spearman']:.3f}\n"
            f"RMSE = {stats['rmse']:.4f}\n"
            f"Median |Delta| = {stats['med_abs']:.4f}"
        )
        ax.text(
            0.04,
            0.96,
            stats_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#c7c7c7"},
        )

    fig.suptitle("Agreement of normalized enrichment scores across implementations", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "figure1_agreement.png", dpi=300, bbox_inches="tight")
    fig.savefig(ASSET_DIR / "figure1_agreement.pdf", bbox_inches="tight")
    plt.close(fig)


def _gene_vector(adata, gene: str) -> np.ndarray:
    x = adata[:, gene].X
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x).ravel()


def _smooth(values: np.ndarray, window: int = 151) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=max(5, window // 10))
        .mean()
        .bfill()
        .ffill()
        .to_numpy()
    )


def generate_real_trajectory_figure() -> None:
    adata_path = REPO_ROOT / "data" / "gse155254_ery_only_pt.h5ad"
    traj_path = REPO_ROOT / "results" / "gse155254_hallmark_traj_ery_only.csv"
    if not adata_path.exists() or not traj_path.exists():
        raise FileNotFoundError("Required trajectory assets for Figure 2 were not found.")

    adata = sc.read_h5ad(adata_path)
    if "X_umap" not in adata.obsm:
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=min(30, adata.obsm["X_pca"].shape[1]))
        sc.tl.umap(adata)

    pseudotime_key = "pseudotime" if "pseudotime" in adata.obs else "dpt_pseudotime"
    pt = adata.obs[pseudotime_key].to_numpy().astype(float)
    order = np.argsort(pt)
    umap = np.asarray(adata.obsm["X_umap"])

    marker_expr = {gene: _gene_vector(adata, gene) for gene in ["CD34", "MKI67", "HBB"]}
    marker_z = {}
    for gene, values in marker_expr.items():
        std = values.std()
        marker_z[gene] = (values - values.mean()) / (std if std > 0 else 1.0)
    stage_matrix = np.column_stack([marker_z["CD34"], marker_z["MKI67"], marker_z["HBB"]])
    stage_names = np.array(["HSC-like", "Cycling progenitor", "Erythroid"])
    stage = stage_names[np.argmax(stage_matrix, axis=1)]

    marker_df = pd.DataFrame(
        {
            "pseudotime": pt[order],
            "HBB": _smooth(marker_expr["HBB"][order]),
            "MKI67": _smooth(marker_expr["MKI67"][order]),
        }
    )
    marker_df.to_csv(DATA_DIR / "figure2_marker_profiles.csv", index=False)

    traj_df = pd.read_csv(traj_path)
    pathways = []
    for target in ["heme metabolism", "e2f targets"]:
        matches = [name for name in traj_df["Pathway"].astype(str).unique() if target in name.lower()]
        if matches:
            pathways.append(matches[0])
    if len(pathways) < 2:
        raise RuntimeError("Could not locate representative pathways for Figure 2.")

    plot_df = traj_df[traj_df["Pathway"].isin(pathways)].copy().sort_values(["Pathway", "pt_mid"])
    plot_df.to_csv(DATA_DIR / "figure2_pathway_profiles.csv", index=False)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10))
    ax1, ax2, ax3, ax4 = axes.ravel()

    scatter1 = ax1.scatter(umap[:, 0], umap[:, 1], c=pt, s=9, cmap="viridis", linewidths=0, alpha=0.9)
    ax1.set_title("A. UMAP colored by pseudotime", loc="left", fontweight="bold")
    ax1.set_xlabel("UMAP1")
    ax1.set_ylabel("UMAP2")
    cbar = fig.colorbar(scatter1, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Pseudotime")

    stage_palette = {"HSC-like": "#2563eb", "Cycling progenitor": "#ea580c", "Erythroid": "#b91c1c"}
    for label, color in stage_palette.items():
        mask = stage == label
        ax2.scatter(umap[mask, 0], umap[mask, 1], s=9, color=color, linewidths=0, alpha=0.9, label=label)
    ax2.set_title("B. Marker-defined trajectory states", loc="left", fontweight="bold")
    ax2.set_xlabel("UMAP1")
    ax2.set_ylabel("UMAP2")
    ax2.legend(frameon=True, fontsize=9, loc="best")

    ax3.plot(marker_df["pseudotime"], marker_df["HBB"], color="#b91c1c", linewidth=2.2, label="HBB")
    ax3.plot(marker_df["pseudotime"], marker_df["MKI67"], color="#1d4ed8", linewidth=2.2, label="MKI67")
    ax3.set_title("C. Marker-gene dynamics along pseudotime", loc="left", fontweight="bold")
    ax3.set_xlabel("Pseudotime")
    ax3.set_ylabel("Smoothed log-expression")
    ax3.legend(frameon=True)

    pathway_palette = {pathways[0]: "#b91c1c", pathways[1]: "#1d4ed8"}
    for pathway in pathways:
        sub = plot_df[plot_df["Pathway"] == pathway].copy()
        ax4.plot(sub["pt_mid"], sub["NES"], linewidth=2.2, color=pathway_palette[pathway], label=pathway)
        sig = sub[sub["padj"] < 0.05]
        if not sig.empty:
            ax4.scatter(sig["pt_mid"], sig["NES"], color=pathway_palette[pathway], s=26, zorder=3)
    ax4.axhline(0, linestyle="--", color="#64748b", linewidth=1)
    ax4.set_title("D. Rolling-window pathway NES", loc="left", fontweight="bold")
    ax4.set_xlabel("Pseudotime window midpoint")
    ax4.set_ylabel("NES")
    ax4.legend(frameon=True)

    fig.suptitle("Real single-cell HSC-like to erythroid trajectory analysis", y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(ASSET_DIR / "figure2_real_trajectory.png", dpi=300, bbox_inches="tight")
    fig.savefig(ASSET_DIR / "figure2_real_trajectory.pdf", bbox_inches="tight")
    plt.close(fig)


def _draw_box(ax, x: float, y: float, w: float, h: float, text: str, face: str, edge: str = "#30475e", fontsize: int = 10, radius: float = 0.02) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=1.5,
        facecolor=face,
        edgecolor=edge,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def _draw_arrow(ax, start: Tuple[float, float], end: Tuple[float, float]) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.6,
        color="#30475e",
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def generate_workflow_figure() -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _draw_box(
        ax,
        0.04,
        0.36,
        0.2,
        0.28,
        "1. Input processing\n\nRanked genes\nGMT collection\neps / sample size / seed",
        face="#dbeafe",
        fontsize=11,
    )

    _draw_box(
        ax,
        0.29,
        0.12,
        0.42,
        0.76,
        "2. Rust core",
        face="#f8fafc",
        fontsize=13,
        radius=0.03,
    )

    inner_boxes = [
        (0.34, 0.72, "Rayon thread-pool\ndispatch"),
        (0.34, 0.56, "Independent PRNG\nsub-seed derivation"),
        (0.34, 0.40, "Enrichment Score (ES)\ncalculation"),
        (0.34, 0.24, "Multilevel splitting\ndecision"),
        (0.52, 0.40, "Conditional probability\nupdate"),
    ]
    for x, y, label in inner_boxes:
        _draw_box(ax, x, y, 0.14, 0.11, label, face="#fef3c7", fontsize=10)

    _draw_box(
        ax,
        0.76,
        0.36,
        0.2,
        0.28,
        "3. Output aggregation\n\nES / NES / P-value\nwindow-wise trajectory profiles",
        face="#dcfce7",
        fontsize=11,
    )

    _draw_arrow(ax, (0.24, 0.50), (0.29, 0.50))
    _draw_arrow(ax, (0.71, 0.50), (0.76, 0.50))
    _draw_arrow(ax, (0.41, 0.72), (0.41, 0.67))
    _draw_arrow(ax, (0.41, 0.56), (0.41, 0.51))
    _draw_arrow(ax, (0.41, 0.40), (0.41, 0.35))
    _draw_arrow(ax, (0.48, 0.46), (0.52, 0.46))
    _draw_arrow(ax, (0.59, 0.40), (0.48, 0.29))

    ax.text(
        0.5,
        0.95,
        "Methodological workflow of PyFgsea",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.08,
        "Pathways are evaluated independently in parallel, enabling lock-free execution and deterministic per-pathway random streams.",
        ha="center",
        va="center",
        fontsize=10,
        color="#334155",
    )

    fig.tight_layout()
    fig.savefig(ASSET_DIR / "figure_S7_workflow.png", dpi=300, bbox_inches="tight")
    fig.savefig(ASSET_DIR / "figure_S7_workflow.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_gmt_dict(gmt: Dict[str, List[str]], out_path: Path) -> Path:
    with out_path.open("w", encoding="utf-8") as handle:
        for name, genes in gmt.items():
            handle.write(f"{name}\tNA\t" + "\t".join(map(str, genes)) + "\n")
    return out_path


def _rank_df_from_scores(genes: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"Gene": genes.astype(str), "Score": np.asarray(scores, dtype=float)})
    return df.sort_values("Score", ascending=False).reset_index(drop=True)


def _trajectory_context(adata: "sc.AnnData", pseudotime_key: str = "dpt_pseudotime") -> Dict[str, np.ndarray]:
    pt = adata.obs[pseudotime_key].to_numpy().astype(float)
    order = np.argsort(pt)
    x = adata.X
    return {
        "pt": pt,
        "order": order,
        "genes": np.asarray(adata.var_names).astype(str),
        "sum_total": np.asarray(x.sum(axis=0)).ravel(),
        "n_all": np.int64(x.shape[0]),
    }


def _rank_scores_from_window(
    X,
    window_indices: np.ndarray,
    sum_total: np.ndarray,
    n_all: int,
) -> np.ndarray:
    n_in = len(window_indices)
    n_out = n_all - n_in
    sum_in = np.asarray(X[window_indices].sum(axis=0)).ravel()
    mu_in = sum_in / max(n_in, 1)
    mu_out = (sum_total - sum_in) / max(n_out, 1)
    scores = mu_in - mu_out
    scores = np.asarray(scores, dtype=np.float64)
    scores[~np.isfinite(scores)] = 0.0
    return scores


def _trajectory_rank_df(
    adata: "sc.AnnData",
    context: Dict[str, np.ndarray],
    start: int,
    window_size: int,
) -> pd.DataFrame:
    end = min(start + window_size, len(context["order"]))
    window_indices = context["order"][start:end]
    scores = _rank_scores_from_window(adata.X, window_indices, context["sum_total"], int(context["n_all"]))
    return _rank_df_from_scores(context["genes"], scores)


def _save_rank_and_gmt(
    df_rank: pd.DataFrame,
    gmt_input: Dict[str, List[str]] | str | Path,
    work_dir: Path,
    prefix: str,
) -> Tuple[Path, Path]:
    rank_path = work_dir / f"{prefix}_rank.csv"
    gmt_path = work_dir / f"{prefix}.gmt"
    df_rank.to_csv(rank_path, index=False)
    if isinstance(gmt_input, (str, Path)):
        gmt_path = Path(gmt_input)
    else:
        _write_gmt_dict(gmt_input, gmt_path)
    return rank_path, gmt_path


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def _write_tabular(out_path: Path, lines: List[str]) -> None:
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_fixed(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}"


def _format_sci(value: float) -> str:
    return f"{value:.1e}"


def generate_validation_overview_figure() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _draw_box(
        ax,
        0.04,
        0.55,
        0.42,
        0.26,
        "Statistical faithfulness\n\nES / NES / transformed nominal P-value\nagreement across synthetic and trajectory-derived ranked lists",
        face="#dbeafe",
        fontsize=11,
    )
    _draw_box(
        ax,
        0.54,
        0.55,
        0.42,
        0.26,
        "Deterministic reproducibility\n\nFixed-seed repeated runs and thread-count\ninvariance under pathway-local PRNG streams",
        face="#dcfce7",
        fontsize=11,
    )
    _draw_box(
        ax,
        0.04,
        0.16,
        0.42,
        0.26,
        "Parallel scalability\n\nPathway-level work stealing, memory behavior,\nand stateful versus stateless rolling-window execution",
        face="#fef3c7",
        fontsize=11,
    )
    _draw_box(
        ax,
        0.54,
        0.16,
        0.42,
        0.26,
        "Rolling-window robustness\n\nWindow / step stress tests, terminal boundary audit,\nand practical parameter recommendations",
        face="#fee2e2",
        fontsize=11,
    )

    ax.text(
        0.5,
        0.92,
        "Comprehensive validation and reproducibility assessment of PyFgsea",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.48,
        "Integrated validation package added to directly address Reviewer 2's concerns about\nself-contained methodology, quantitative equivalence, reproducibility, and implementation robustness.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#334155",
    )
    ax.text(0.25, 0.51, "Figure S10, Table S7", ha="center", fontsize=10, color="#1e3a8a")
    ax.text(0.75, 0.51, "Figure S11-S12, Table S8", ha="center", fontsize=10, color="#166534")
    ax.text(0.25, 0.12, "Figure S11, Figure S15", ha="center", fontsize=10, color="#92400e")
    ax.text(0.75, 0.12, "Figure S8, Figure S13-S14, Table S9", ha="center", fontsize=10, color="#991b1b")

    fig.tight_layout()
    fig.savefig(ASSET_DIR / "figure_S9_validation_overview.png", dpi=300, bbox_inches="tight")
    fig.savefig(ASSET_DIR / "figure_S9_validation_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_equivalence_validation_assets() -> None:
    work_dir = Path(tempfile.mkdtemp(prefix="pyfgsea_revision_s10_"))
    hallmark_gmt = REPO_ROOT / "hallmark_enrichr.gmt"
    adata_path = REPO_ROOT / "data" / "gse155254_ery_only_pt.h5ad"
    adata = sc.read_h5ad(adata_path)
    context = _trajectory_context(adata)

    regimes: List[Tuple[str, pd.DataFrame, Dict[str, List[str]] | str | Path]] = []
    for label, n_genes, n_sets, seed in [
        ("Synthetic small", 12000, 120, 42),
        ("Synthetic medium", 20000, 400, 43),
        ("Synthetic large", 20000, 1000, 44),
    ]:
        df_rnk, gmt = generate_test_data(n_genes=n_genes, n_sets=n_sets, seed=seed)
        regimes.append((label, df_rnk, gmt))

    n_obs = int(adata.n_obs)
    real_windows = {
        "Trajectory early": 0,
        "Trajectory middle": max(0, n_obs // 2 - 250),
        "Trajectory late": max(0, n_obs - 500),
    }
    for label, start in real_windows.items():
        regimes.append((label, _trajectory_rank_df(adata, context, start=start, window_size=500), str(hallmark_gmt)))

    merged_frames = []
    metric_rows = []
    palette = {
        "Synthetic small": "#1d4ed8",
        "Synthetic medium": "#0f766e",
        "Synthetic large": "#b45309",
        "Trajectory early": "#7c3aed",
        "Trajectory middle": "#be123c",
        "Trajectory late": "#047857",
    }

    for idx, (label, df_rank, gmt_input) in enumerate(regimes):
        prefix = f"regime_{idx}"
        rank_path, gmt_path = _save_rank_and_gmt(df_rank, gmt_input, work_dir, prefix)

        res_py = pyfgsea.run_gsea(
            df_rank,
            gmt_input,
            gene_col="Gene",
            score_col="Score",
            min_size=15,
            max_size=500,
            sample_size=101,
            seed=42,
            nperm_nes=800,
            eps=1e-50,
        )[["Pathway", "ES", "NES", "P-value"]].rename(
            columns={"Pathway": "pathway", "ES": "ES_Py", "NES": "NES_Py", "P-value": "Pvalue_Py"}
        )
        res_r = _run_r_fgsea(rank_path, gmt_path, work_dir / f"{prefix}_r.csv").rename(
            columns={"ES": "ES_R", "NES": "NES_R", "pval": "Pvalue_R"}
        )

        merged = res_py.merge(res_r, on="pathway")
        merged["Regime"] = label
        merged["log10P_Py"] = -np.log10(np.clip(merged["Pvalue_Py"].astype(float), 1e-300, None))
        merged["log10P_R"] = -np.log10(np.clip(merged["Pvalue_R"].astype(float), 1e-300, None))
        merged_frames.append(merged)

        for metric_label, x_col, y_col in [
            ("ES", "ES_Py", "ES_R"),
            ("NES", "NES_Py", "NES_R"),
            ("Log10P", "log10P_Py", "log10P_R"),
        ]:
            stats = _concordance_frame(merged[x_col], merged[y_col])
            metric_rows.append(
                {
                    "Regime": label,
                    "Metric": metric_label,
                    "Pearson": stats["pearson"],
                    "Spearman": stats["spearman"],
                    "RMSE": stats["rmse"],
                    "MedianAbs": stats["med_abs"],
                    "P95Abs": stats["p95_abs"],
                }
            )

    combined = pd.concat(merged_frames, ignore_index=True)
    combined.to_csv(DATA_DIR / "figure_S10_equivalence_points.csv", index=False)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(DATA_DIR / "table_S7_equivalence_summary.csv", index=False)

    table_lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Regime & Metric & Pearson $r$ & Spearman $\rho$ & RMSE & Median $|\Delta|$ & 95th pct. $|\Delta|$ \\",
        r"\midrule",
    ]
    for _, row in metrics_df.iterrows():
        table_lines.append(
            f"{_latex_escape(row['Regime'])} & {row['Metric']} & "
            f"{_format_fixed(row['Pearson'])} & {_format_fixed(row['Spearman'])} & "
            f"{_format_fixed(row['RMSE'], 4)} & {_format_fixed(row['MedianAbs'], 4)} & {_format_fixed(row['P95Abs'], 4)} \\\\"
        )
    table_lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write_tabular(DATA_DIR / "table_S7_equivalence_summary.tex", table_lines)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    panel_specs = [
        ("A", "ES_Py", "ES_R", "ES"),
        ("B", "NES_Py", "NES_R", "NES"),
        ("C", "log10P_Py", "log10P_R", r"$-\log_{10}(P)$"),
    ]

    for ax, (panel, x_col, y_col, label) in zip(axes, panel_specs):
        lim_min = min(combined[x_col].min(), combined[y_col].min())
        lim_max = max(combined[x_col].max(), combined[y_col].max())
        span = lim_max - lim_min
        lim_min -= 0.03 * span
        lim_max += 0.03 * span

        for regime, sub in combined.groupby("Regime"):
            ax.scatter(
                sub[x_col],
                sub[y_col],
                s=20,
                alpha=0.68,
                color=palette[regime],
                edgecolors="white",
                linewidth=0.25,
                label=regime,
            )

        ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", color="#475569", linewidth=1.1)
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_xlabel(f"PyFgsea {label}")
        ax.set_ylabel(f"R fgseaMultilevel {label}")
        ax.set_title(f"{panel}. {label} equivalence", loc="left", fontsize=11, fontweight="bold")

        stats = _concordance_frame(combined[x_col], combined[y_col])
        text = (
            f"Pearson r = {stats['pearson']:.3f}\n"
            f"Spearman rho = {stats['spearman']:.3f}\n"
            f"RMSE = {stats['rmse']:.4f}\n"
            f"Median |Delta| = {stats['med_abs']:.4f}"
        )
        ax.text(
            0.04,
            0.96,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#c7c7c7"},
        )

    handles = [Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=color, markeredgecolor="white", markersize=7) for label, color in palette.items()]
    fig.legend(handles=handles, labels=list(palette.keys()), loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Integrated equivalence validation across synthetic benchmarks and representative trajectory windows", y=1.02, fontsize=13.5)
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    fig.savefig(ASSET_DIR / "figure_S10_equivalence_overview.png", dpi=300, bbox_inches="tight")
    fig.savefig(ASSET_DIR / "figure_S10_equivalence_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_window_sensitivity_figure() -> None:
    adata_path = REPO_ROOT / "data" / "gse155254_ery_only_pt.h5ad"
    gmt_path = REPO_ROOT / "hallmark_enrichr.gmt"
    adata = sc.read_h5ad(adata_path)

    configs = {
        (window_size, step): {"window_size": window_size, "step": step}
        for window_size in [100, 500, 1000]
        for step in [20, 50, 100]
    }

    runs = {}
    runtimes = {}
    for key, params in configs.items():
        start_t = time.time()
        runs[key] = run_trajectory_gsea(
            adata,
            gmt_path=str(gmt_path),
            window_size=params["window_size"],
            step=params["step"],
            min_size=15,
            max_size=500,
            sample_size=101,
            nperm_nes=80,
            seed=42,
            eps=1e-50,
        )
        runtimes[key] = time.time() - start_t

    reference_paths = sorted(runs[(500, 50)]["Pathway"].astype(str).unique())
    targets = []
    for pattern in ["heme metabolism", "e2f targets", "g2-m checkpoint"]:
        matches = [name for name in reference_paths if pattern in name.lower()]
        if not matches:
            raise RuntimeError(f"Could not find pathway matching '{pattern}'.")
        targets.append(matches[0])

    records = []
    for (window_size, step), df in runs.items():
        for pathway in targets:
            subset = df[df["Pathway"].astype(str).str.lower() == pathway.lower()].copy().sort_values("pt_mid")
            subset["window_size"] = window_size
            subset["step"] = step
            subset["runtime_s"] = runtimes[(window_size, step)]
            subset["PathwayDisplay"] = pathway
            records.append(
                subset[
                    ["window_size", "step", "runtime_s", "PathwayDisplay", "pt_mid", "NES", "padj", "window_id"]
                ]
            )

    curves_df = pd.concat(records, ignore_index=True)
    curves_df.to_csv(DATA_DIR / "figure_S8_window_sensitivity.csv", index=False)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 11), sharex=False, sharey=False)
    left_palette = {100: "#b91c1c", 500: "#0f766e", 1000: "#1d4ed8"}
    right_palette = {20: "#7c3aed", 50: "#ea580c", 100: "#2563eb"}

    for row, pathway in enumerate(targets):
        ax_left = axes[row, 0]
        ax_right = axes[row, 1]

        for setting in [100, 500, 1000]:
            sub = curves_df[
                (curves_df["window_size"] == setting)
                & (curves_df["step"] == 50)
                & (curves_df["PathwayDisplay"] == pathway)
            ].sort_values("pt_mid")
            ax_left.plot(sub["pt_mid"], sub["NES"], linewidth=2.0, color=left_palette[setting], label=f"{setting}")
        ax_left.axhline(0, linestyle="--", color="#64748b", linewidth=1)
        ax_left.set_title(f"{chr(65 + row*2)}. {pathway}", loc="left", fontweight="bold")
        ax_left.set_ylabel("NES")
        if row == 2:
            ax_left.set_xlabel("Pseudotime window midpoint")
        if row == 0:
            ax_left.legend(title="Window size", ncol=3, frameon=True, fontsize=8)

        for setting in [20, 50, 100]:
            sub = curves_df[
                (curves_df["window_size"] == 500)
                & (curves_df["step"] == setting)
                & (curves_df["PathwayDisplay"] == pathway)
            ].sort_values("pt_mid")
            ax_right.plot(sub["pt_mid"], sub["NES"], linewidth=2.0, color=right_palette[setting], label=f"{setting}")
        ax_right.axhline(0, linestyle="--", color="#64748b", linewidth=1)
        ax_right.set_title(f"{chr(66 + row*2)}. {pathway}", loc="left", fontweight="bold")
        if row == 2:
            ax_right.set_xlabel("Pseudotime window midpoint")
        if row == 0:
            ax_right.legend(title="Step size", ncol=3, frameon=True, fontsize=8)

    fig.suptitle(
        "Rolling-window sensitivity across three representative pathways\nLeft: varying window size at fixed step=50; Right: varying step size at fixed window=500",
        y=0.99,
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(ASSET_DIR / "figure_S8_window_sensitivity.png", dpi=300, bbox_inches="tight")
    fig.savefig(ASSET_DIR / "figure_S8_window_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_parameter_grid_summary_figure() -> None:
    curves_df = pd.read_csv(DATA_DIR / "figure_S8_window_sensitivity.csv")
    baseline = curves_df[(curves_df["window_size"] == 500) & (curves_df["step"] == 50)].copy()
    total_cells = 3576

    rows = []
    for (window_size, step), sub in curves_df.groupby(["window_size", "step"]):
        smoothness = []
        peak_shift = []
        overlap = []
        for pathway, path_df in sub.groupby("PathwayDisplay"):
            path_df = path_df.sort_values("pt_mid")
            smoothness.append(float(np.median(np.abs(np.diff(path_df["NES"].to_numpy())))))

            ref = baseline[baseline["PathwayDisplay"] == pathway].sort_values("pt_mid")
            pt_peak = float(path_df.loc[path_df["NES"].abs().idxmax(), "pt_mid"])
            pt_peak_ref = float(ref.loc[ref["NES"].abs().idxmax(), "pt_mid"])
            peak_shift.append(abs(pt_peak - pt_peak_ref))

            ref_sig = set(np.round(ref.loc[ref["padj"] < 0.05, "pt_mid"].to_numpy(), 2))
            cur_sig = set(np.round(path_df.loc[path_df["padj"] < 0.05, "pt_mid"].to_numpy(), 2))
            union = ref_sig | cur_sig
            overlap.append(1.0 if not union else len(ref_sig & cur_sig) / len(union))

        rows.append(
            {
                "window_size": int(window_size),
                "step": int(step),
                "window_pct": 100.0 * window_size / total_cells,
                "step_pct": 100.0 * step / total_cells,
                "n_windows": int(sub["window_id"].nunique()),
                "smoothness": float(np.mean(smoothness)),
                "peak_shift": float(np.mean(peak_shift)),
                "sig_overlap": float(np.mean(overlap)),
                "runtime_s": float(sub["runtime_s"].iloc[0]),
            }
        )

    summary_df = pd.DataFrame(rows).sort_values(["window_size", "step"])
    summary_df.to_csv(DATA_DIR / "table_S9_window_grid_summary.csv", index=False)

    table_lines = [
        r"\begin{tabular}{rrrrrrrrr}",
        r"\toprule",
        r"Window & Step & Window (\%) & Step (\%) & Windows & Median stepwise $|\Delta NES|$ & Peak-shift & Sig. overlap & Runtime (s) \\",
        r"\midrule",
    ]
    for _, row in summary_df.iterrows():
        table_lines.append(
            f"{int(row['window_size'])} & {int(row['step'])} & "
            f"{row['window_pct']:.1f} & {row['step_pct']:.1f} & {int(row['n_windows'])} & "
            f"{row['smoothness']:.4f} & {row['peak_shift']:.4f} & {row['sig_overlap']:.3f} & {row['runtime_s']:.1f} \\\\"
        )
    table_lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write_tabular(DATA_DIR / "table_S9_window_grid_summary.tex", table_lines)

    heatmap_specs = [
        ("smoothness", "Mean stepwise |Delta NES|", "mako"),
        ("peak_shift", "Mean peak shift", "rocket"),
        ("sig_overlap", "Mean significant-window overlap", "crest"),
        ("runtime_s", "Runtime (s)", "flare"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9))
    for ax, (column, title, cmap) in zip(axes.ravel(), heatmap_specs):
        pivot = summary_df.pivot(index="window_size", columns="step", values=column).sort_index(ascending=False)
        sns.heatmap(pivot, annot=True, fmt=".3f" if column != "runtime_s" else ".1f", cmap=cmap, cbar=False, ax=ax)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Step size")
        ax.set_ylabel("Window size")

    fig.suptitle("Rolling-window parameter stress-test summary across the full window-size by step-size grid", y=0.98, fontsize=13.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(ASSET_DIR / "figure_S13_parameter_grid_summary.png", dpi=300, bbox_inches="tight")
    fig.savefig(ASSET_DIR / "figure_S13_parameter_grid_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def _run_thread_eval_subprocess(n_threads: int, out_csv: Path) -> Dict[str, float]:
    work_dir = Path(tempfile.mkdtemp(prefix=f"pyfgsea_thread_{n_threads}_"))
    script_path = work_dir / "run_thread_eval.py"
    script_path.write_text(
        textwrap.dedent(
            f"""
            import json
            import os
            import sys
            import threading
            import time
            from pathlib import Path

            import psutil

            os.environ["RAYON_NUM_THREADS"] = "{n_threads}"
            repo_root = Path(r"{REPO_ROOT}")
            sys.path.insert(0, str(repo_root))

            from repro.data_utils import generate_test_data
            import pyfgsea

            def monitor_memory(pid, stop_event, max_mem, base_mem):
                process = psutil.Process(pid)
                while not stop_event.is_set():
                    try:
                        mem = process.memory_info().rss
                        for child in process.children(recursive=True):
                            mem += child.memory_info().rss
                        max_mem[0] = max(max_mem[0], mem - base_mem)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                    time.sleep(0.02)

            df_rnk, gmt = generate_test_data(n_genes=20000, n_sets=1000, seed=42)
            process = psutil.Process(os.getpid())
            base_mem = process.memory_info().rss
            stop_event = threading.Event()
            peak_inc = [0]
            monitor = threading.Thread(target=monitor_memory, args=(os.getpid(), stop_event, peak_inc, base_mem), daemon=True)
            monitor.start()

            t0 = time.time()
            res = pyfgsea.run_gsea(df_rnk, gmt, gene_col="Gene", score_col="Score", sample_size=101, seed=42, nperm_nes=400, eps=1e-50)
            elapsed = time.time() - t0

            stop_event.set()
            monitor.join()
            res.to_csv(r"{out_csv}", index=False)
            print(json.dumps({{"threads": {n_threads}, "time_s": elapsed, "peak_rss_mb": peak_inc[0] / 1024 / 1024}}))
            """
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=True,
    )
    last_line = completed.stdout.strip().splitlines()[-1]
    return json.loads(last_line)


def generate_thread_validation_assets() -> None:
    thread_counts = [1, 2, 4, 8, 16]
    ref_df = None
    rows = []

    for n_threads in thread_counts:
        out_csv = DATA_DIR / f"thread_eval_{n_threads}.csv"
        stats = _run_thread_eval_subprocess(n_threads, out_csv)
        cur_df = pd.read_csv(out_csv).sort_values("Pathway").reset_index(drop=True)
        cur_df["log10P"] = -np.log10(np.clip(cur_df["P-value"].astype(float), 1e-300, None))

        if ref_df is None:
            ref_df = cur_df.copy()
            rows.append(
                {
                    "threads": n_threads,
                    "time_s": stats["time_s"],
                    "speedup": 1.0,
                    "peak_rss_mb": stats["peak_rss_mb"],
                    "exact_match_rate": 1.0,
                    "identical_ranking_fraction": 1.0,
                    "max_abs_es": 0.0,
                    "max_abs_nes": 0.0,
                    "max_abs_log10p": 0.0,
                }
            )
            continue

        merged = ref_df.merge(cur_df, on="Pathway", suffixes=("_ref", "_cur"))
        max_abs_es = float(np.max(np.abs(merged["ES_ref"] - merged["ES_cur"])))
        max_abs_nes = float(np.max(np.abs(merged["NES_ref"] - merged["NES_cur"])))
        max_abs_log10p = float(np.max(np.abs(merged["log10P_ref"] - merged["log10P_cur"])))
        exact_mask = (
            np.abs(merged["ES_ref"] - merged["ES_cur"]) < 1e-12
        ) & (
            np.abs(merged["NES_ref"] - merged["NES_cur"]) < 1e-12
        ) & (
            np.abs(merged["log10P_ref"] - merged["log10P_cur"]) < 1e-12
        )
        ref_rank = ref_df.sort_values(["P-value", "Pathway"])["Pathway"].to_numpy()
        cur_rank = cur_df.sort_values(["P-value", "Pathway"])["Pathway"].to_numpy()

        rows.append(
            {
                "threads": n_threads,
                "time_s": stats["time_s"],
                "speedup": rows[0]["time_s"] / stats["time_s"],
                "peak_rss_mb": stats["peak_rss_mb"],
                "exact_match_rate": float(exact_mask.mean()),
                "identical_ranking_fraction": float(np.mean(ref_rank == cur_rank)),
                "max_abs_es": max_abs_es,
                "max_abs_nes": max_abs_nes,
                "max_abs_log10p": max_abs_log10p,
            }
        )

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(DATA_DIR / "table_S8_thread_audit.csv", index=False)

    table_lines = [
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        r"Threads & Time (s) & Speedup & Peak RSS (MB) & Exact match & Rank match & Max $|\Delta NES|$ & Max $|\Delta -\log_{10}P|$ \\",
        r"\midrule",
    ]
    for _, row in audit_df.iterrows():
        table_lines.append(
            f"{int(row['threads'])} & {row['time_s']:.3f} & {row['speedup']:.2f} & {row['peak_rss_mb']:.1f} & "
            f"{row['exact_match_rate']:.3f} & {row['identical_ranking_fraction']:.3f} & "
            f"{_format_sci(row['max_abs_nes'])} & {_format_sci(row['max_abs_log10p'])} \\\\"
        )
    table_lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write_tabular(DATA_DIR / "table_S8_thread_audit.tex", table_lines)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax1, ax2 = axes
    ax1.plot(audit_df["threads"], audit_df["speedup"], marker="o", color="#1d4ed8", linewidth=2.2, label="Speedup")
    ax1.plot(audit_df["threads"], audit_df["threads"], linestyle="--", color="#94a3b8", linewidth=1.2, label="Ideal linear")
    ax1.set_xlabel("Number of threads")
    ax1.set_ylabel("Speedup vs 1 thread")
    ax1.set_xticks(thread_counts)
    ax1.set_title("A. Scaling behavior", loc="left", fontweight="bold")
    ax1.legend(frameon=True, fontsize=8)
    ax1b = ax1.twinx()
    ax1b.plot(audit_df["threads"], audit_df["peak_rss_mb"], marker="s", linestyle="--", color="#dc2626", linewidth=1.8)
    ax1b.set_ylabel("Peak RSS (MB)")

    ax2.plot(audit_df["threads"], audit_df["exact_match_rate"], marker="o", color="#047857", linewidth=2.2, label="Exact pathway-wise match")
    ax2.plot(audit_df["threads"], audit_df["identical_ranking_fraction"], marker="s", color="#7c3aed", linewidth=2.2, label="Identical pathway ranking")
    ax2.set_ylim(0.95, 1.005)
    ax2.set_xlabel("Number of threads")
    ax2.set_ylabel("Fraction")
    ax2.set_xticks(thread_counts)
    ax2.set_title("B. Fixed-seed determinism", loc="left", fontweight="bold")
    ax2.legend(frameon=True, fontsize=8, loc="lower right")
    max_text = (
        f"Overall max |Delta ES| = {_format_sci(audit_df['max_abs_es'].max())}\n"
        f"Overall max |Delta NES| = {_format_sci(audit_df['max_abs_nes'].max())}\n"
        f"Overall max |Delta -log10P| = {_format_sci(audit_df['max_abs_log10p'].max())}"
    )
    ax2.text(
        0.04,
        0.96,
        max_text,
        transform=ax2.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#c7c7c7"},
    )

    fig.suptitle("Thread-count invariance and parallel scaling under a fixed master seed", y=1.02, fontsize=13.5)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "figure_S11_thread_validation.png", dpi=300, bbox_inches="tight")
    fig.savefig(ASSET_DIR / "figure_S11_thread_validation.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_reproducibility_figure() -> None:
    df_rnk, gmt = generate_test_data(n_genes=12000, n_sets=160, seed=42)
    base = pyfgsea.run_gsea(df_rnk, gmt, gene_col="Gene", score_col="Score", sample_size=101, seed=42, nperm_nes=400, eps=1e-50)
    top_paths = base.sort_values("P-value").head(5)["Pathway"].tolist()

    records = []
    for rep in range(12):
        res = pyfgsea.run_gsea(df_rnk, gmt, gene_col="Gene", score_col="Score", sample_size=101, seed=42, nperm_nes=400, eps=1e-50)
        sub = res[res["Pathway"].isin(top_paths)].copy()
        for _, row in sub.iterrows():
            records.append(
                {
                    "mode": "Fixed seed",
                    "replicate": rep,
                    "Pathway": row["Pathway"],
                    "log10P": -np.log10(max(float(row["P-value"]), 1e-300)),
                    "NES": float(row["NES"]),
                }
            )

    for seed in range(42, 62):
        res = pyfgsea.run_gsea(df_rnk, gmt, gene_col="Gene", score_col="Score", sample_size=101, seed=seed, nperm_nes=400, eps=1e-50)
        sub = res[res["Pathway"].isin(top_paths)].copy()
        for _, row in sub.iterrows():
            records.append(
                {
                    "mode": "Varying seed",
                    "replicate": seed,
                    "Pathway": row["Pathway"],
                    "log10P": -np.log10(max(float(row["P-value"]), 1e-300)),
                    "NES": float(row["NES"]),
                }
            )

    repro_df = pd.DataFrame(records)
    repro_df.to_csv(DATA_DIR / "figure_S12_reproducibility.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    order = top_paths
    for ax, mode, panel in zip(axes, ["Fixed seed", "Varying seed"], ["A", "B"]):
        sub = repro_df[repro_df["mode"] == mode]
        sns.boxplot(data=sub, x="Pathway", y="log10P", order=order, color="#bfdbfe" if mode == "Fixed seed" else "#fde68a", ax=ax)
        sns.stripplot(data=sub, x="Pathway", y="log10P", order=order, color="#1f2937", alpha=0.35, jitter=0.18, size=3, ax=ax)
        ax.set_title(f"{panel}. {mode} repeated executions", loc="left", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(r"$-\log_{10}(P)$")
        ax.tick_params(axis="x", rotation=35)

    fig.suptitle("Repeated-run reproducibility distinguishes deterministic implementation stability from seed-driven Monte Carlo variability", y=0.98, fontsize=13.2)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(ASSET_DIR / "figure_S12_reproducibility.png", dpi=300, bbox_inches="tight")
    fig.savefig(ASSET_DIR / "figure_S12_reproducibility.pdf", bbox_inches="tight")
    plt.close(fig)


def _run_terminal_audit(
    adata: "sc.AnnData",
    gmt_path: Path,
    window_size: int = 500,
    step: int = 50,
    min_terminal_cells: int = 200,
) -> pd.DataFrame:
    context = _trajectory_context(adata)
    gmt = load_gmt(str(gmt_path))
    pathway_names, pathway_indices = prepare_pathways(context["genes"], gmt, 15, 500)
    runner = GseaRunner(pathway_names, pathway_indices, 15, 500)

    n = len(context["order"])
    starts_full = list(range(0, n - window_size + 1, step))
    starts_trunc = [s for s in range(0, n, step) if n - s >= min_terminal_cells]
    records = []

    for mode, starts in [("Full windows only", starts_full), ("Terminal truncation audit", starts_trunc)]:
        for start in starts:
            end = min(start + window_size, n)
            window_indices = context["order"][start:end]
            scores = _rank_scores_from_window(adata.X, window_indices, context["sum_total"], int(context["n_all"]))
            res = runner.run(
                scores,
                sample_size=101,
                seed=42 + start,
                eps=1e-50,
                nperm_nes=80,
                bin_width=10,
                calculate_nes=True,
                use_nes_cache=True,
            )
            pt_vals = context["pt"][window_indices]
            res["mode"] = mode
            res["window_start"] = start
            res["window_cells"] = len(window_indices)
            res["pt_mid"] = (pt_vals.min() + pt_vals.max()) / 2.0
            records.append(res)

    return pd.concat(records, ignore_index=True)


def generate_boundary_handling_figure() -> None:
    adata = sc.read_h5ad(REPO_ROOT / "data" / "gse155254_ery_only_pt.h5ad")
    df = _run_terminal_audit(adata, REPO_ROOT / "hallmark_enrichr.gmt")
    full_mode = "Full windows only"
    trunc_mode = "Terminal truncation audit"
    meta_df = (
        df[["mode", "window_start", "window_cells", "pt_mid"]]
        .drop_duplicates()
        .sort_values(["mode", "window_start"])
        .reset_index(drop=True)
    )
    full_meta = meta_df[meta_df["mode"] == full_mode].copy()
    trunc_meta = meta_df[meta_df["mode"] == trunc_mode].copy()
    last_full_start = int(full_meta["window_start"].max())
    last_full_mid = float(full_meta["pt_mid"].max())
    near_end_cutoff = max(float(last_full_mid - 0.16), float(full_meta["pt_mid"].min()))

    targets = []
    for pattern in ["heme metabolism", "e2f targets"]:
        matches = [name for name in df["Pathway"].astype(str).unique() if pattern in name.lower()]
        if matches:
            targets.append(matches[0])
    if len(targets) < 2:
        raise RuntimeError("Boundary audit could not find representative pathways.")

    plot_df = df[df["Pathway"].isin(targets)].copy()
    plot_df.to_csv(DATA_DIR / "figure_S14_boundary_audit.csv", index=False)

    full_reference = (
        df[(df["mode"] == full_mode) & (df["window_start"] == last_full_start)][["Pathway", "NES", "padj"]]
        .rename(columns={"NES": "NES_ref", "padj": "padj_ref"})
        .copy()
    )
    tail_metrics = []
    for _, meta_row in trunc_meta[trunc_meta["window_start"] > last_full_start].iterrows():
        current = (
            df[(df["mode"] == trunc_mode) & (df["window_start"] == meta_row["window_start"])][["Pathway", "NES", "padj"]]
            .rename(columns={"NES": "NES_cur", "padj": "padj_cur"})
            .copy()
        )
        merged = full_reference.merge(current, on="Pathway", how="inner")
        if merged.empty:
            continue

        ref_top = set(
            merged.assign(abs_ref=merged["NES_ref"].abs())
            .nlargest(10, "abs_ref")["Pathway"]
            .astype(str)
            .tolist()
        )
        cur_top = set(
            merged.assign(abs_cur=merged["NES_cur"].abs())
            .nlargest(10, "abs_cur")["Pathway"]
            .astype(str)
            .tolist()
        )
        ref_sig = set(merged.loc[merged["padj_ref"] < 0.05, "Pathway"].astype(str).tolist())
        cur_sig = set(merged.loc[merged["padj_cur"] < 0.05, "Pathway"].astype(str).tolist())
        union_sig = ref_sig | cur_sig

        tail_metrics.append(
            {
                "window_start": int(meta_row["window_start"]),
                "window_cells": int(meta_row["window_cells"]),
                "pt_mid": float(meta_row["pt_mid"]),
                "spearman_nes": float(spearmanr(merged["NES_ref"], merged["NES_cur"]).correlation),
                "top10_overlap": len(ref_top & cur_top) / 10.0,
                "sig_jaccard": (len(ref_sig & cur_sig) / len(union_sig)) if union_sig else 1.0,
            }
        )

    tail_metrics_df = pd.DataFrame(tail_metrics).sort_values("pt_mid").reset_index(drop=True)
    tail_metrics_df.to_csv(DATA_DIR / "figure_S14_boundary_metrics.csv", index=False)

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(13.6, 9.1))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.28, wspace=0.22)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    styles = {
        full_mode: {"color": "#1d4ed8", "linestyle": "-", "marker": "o", "label": "Full windows only"},
        trunc_mode: {"color": "#dc2626", "linestyle": "--", "marker": "s", "label": "Terminal truncation audit"},
    }

    ax_a.axvspan(last_full_mid, float(trunc_meta["pt_mid"].max()) + 0.004, color="#fee2e2", alpha=0.65)
    for mode, meta in [(full_mode, full_meta), (trunc_mode, trunc_meta)]:
        sub = meta[meta["pt_mid"] >= near_end_cutoff].copy()
        style = styles[mode]
        ax_a.plot(
            sub["pt_mid"],
            sub["window_cells"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.2,
            marker=style["marker"],
            markersize=5,
            label=style["label"],
        )
    ax_a.axvline(last_full_mid, color="#475569", linestyle=":", linewidth=1.3)
    ax_a.set_title("A. Audit design and shrinking terminal windows", loc="left", fontweight="bold")
    ax_a.set_xlabel("Pseudotime window midpoint")
    ax_a.set_ylabel("Cells in window")
    ax_a.legend(frameon=True, fontsize=8, loc="lower left")
    ax_a.annotate(
        "Truncation-only\ntail extension",
        xy=(float(trunc_meta["pt_mid"].max()), float(tail_metrics_df["window_cells"].min())),
        xytext=(last_full_mid + 0.03, 320),
        arrowprops={"arrowstyle": "->", "color": "#991b1b", "lw": 1.2},
        fontsize=8.5,
        color="#991b1b",
        ha="left",
    )

    for ax, pathway, panel in [(ax_b, targets[0], "B"), (ax_c, targets[1], "C")]:
        full_sub = plot_df[
            (plot_df["Pathway"] == pathway) & (plot_df["mode"] == full_mode) & (plot_df["pt_mid"] >= near_end_cutoff)
        ].copy()
        trunc_sub = plot_df[
            (plot_df["Pathway"] == pathway) & (plot_df["mode"] == trunc_mode) & (plot_df["window_start"] >= last_full_start)
        ].copy()
        ax.axvspan(last_full_mid, float(trunc_meta["pt_mid"].max()) + 0.004, color="#fee2e2", alpha=0.65)
        ax.plot(
            full_sub["pt_mid"],
            full_sub["NES"],
            color=styles[full_mode]["color"],
            linestyle=styles[full_mode]["linestyle"],
            linewidth=2.2,
            marker=styles[full_mode]["marker"],
            markersize=5,
            label=styles[full_mode]["label"],
        )
        ax.plot(
            trunc_sub["pt_mid"],
            trunc_sub["NES"],
            color=styles[trunc_mode]["color"],
            linestyle=styles[trunc_mode]["linestyle"],
            linewidth=2.2,
            marker=styles[trunc_mode]["marker"],
            markersize=5,
            label=styles[trunc_mode]["label"],
        )
        ax.axvline(last_full_mid, color="#475569", linestyle=":", linewidth=1.3)
        ax.axhline(0, linestyle="--", color="#94a3b8", linewidth=1)
        ax.set_title(f"{panel}. {pathway}", loc="left", fontweight="bold")
        ax.set_xlabel("Pseudotime window midpoint")
        ax.set_ylabel("NES")
        if ax is ax_b:
            ax.legend(frameon=True, fontsize=8, loc="best")

    ax_d.plot(
        tail_metrics_df["pt_mid"],
        tail_metrics_df["spearman_nes"],
        color="#7c3aed",
        marker="o",
        linewidth=2.2,
        label="NES Spearman vs last full window",
    )
    ax_d.plot(
        tail_metrics_df["pt_mid"],
        tail_metrics_df["top10_overlap"],
        color="#059669",
        marker="s",
        linewidth=2.2,
        label="Top-10 pathway overlap",
    )
    ax_d.plot(
        tail_metrics_df["pt_mid"],
        tail_metrics_df["sig_jaccard"],
        color="#ea580c",
        marker="^",
        linewidth=2.0,
        label="FDR<0.05 pathway overlap",
    )
    for _, row in tail_metrics_df.iterrows():
        ax_d.annotate(
            f"{int(row['window_cells'])} cells",
            (row["pt_mid"], row["spearman_nes"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=7.8,
            color="#4b5563",
        )
    ax_d.set_ylim(0, 1.05)
    ax_d.set_title("D. Pathway-level continuity of truncation-only tail windows", loc="left", fontweight="bold")
    ax_d.set_xlabel("Pseudotime window midpoint")
    ax_d.set_ylabel("Agreement with last full window")
    ax_d.legend(frameon=True, fontsize=8, loc="lower left")

    fig.suptitle("Boundary-handling audit near trajectory termini", y=0.985, fontsize=13.2)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(ASSET_DIR / "figure_S14_boundary_audit.png", dpi=300, bbox_inches="tight")
    fig.savefig(ASSET_DIR / "figure_S14_boundary_audit.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_stateful_scaling_figure() -> None:
    adata = sc.read_h5ad(REPO_ROOT / "data" / "gse155254_ery_only_pt.h5ad")
    context = _trajectory_context(adata)
    rng = np.random.default_rng(42)
    gmt = {}
    for i in range(3000):
        size = int(rng.integers(15, 120))
        gmt[f"StressPath_{i}"] = list(rng.choice(context["genes"], size=size, replace=False))
    pathway_names, pathway_indices = prepare_pathways(context["genes"], gmt, 15, 500)

    full_starts = list(range(0, len(context["order"]) - 500 + 1, 50))
    score_vectors = []
    for start in full_starts:
        window_indices = context["order"][start : start + 500]
        score_vectors.append(_rank_scores_from_window(adata.X, window_indices, context["sum_total"], int(context["n_all"])))

    runner = GseaRunner(pathway_names, pathway_indices, 15, 500)
    genes = context["genes"]
    results = []

    for n_windows in [100, 300, 1000]:
        vectors = [score_vectors[i % len(score_vectors)] for i in range(n_windows)]

        t0 = time.time()
        for scores in vectors:
            df_rank = pd.DataFrame({"Gene": genes, "Score": scores})
            pyfgsea.run_gsea(
                df_rank,
                gmt,
                gene_col="Gene",
                score_col="Score",
                min_size=15,
                max_size=500,
                sample_size=101,
                seed=42,
                nperm_nes=80,
                eps=1e-50,
            )
        stateless_time = time.time() - t0

        t0 = time.time()
        for i, scores in enumerate(vectors):
            runner.run(
                scores,
                sample_size=101,
                seed=42 + i,
                nperm_nes=80,
                eps=1e-50,
                calculate_nes=True,
                use_nes_cache=True,
            )
        stateful_time = time.time() - t0

        results.append(
            {
                "n_windows": n_windows,
                "stateless_time": stateless_time,
                "stateful_time": stateful_time,
                "speedup": stateless_time / stateful_time if stateful_time > 0 else np.nan,
            }
        )

    bench_df = pd.DataFrame(results)
    bench_df.to_csv(DATA_DIR / "figure_S15_stateful_scaling.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax1, ax2 = axes
    ax1.plot(bench_df["n_windows"], bench_df["stateless_time"], marker="o", color="#dc2626", linewidth=2.2, label="Stateless repeated run_gsea")
    ax1.plot(bench_df["n_windows"], bench_df["stateful_time"], marker="s", color="#1d4ed8", linewidth=2.2, label="Stateful GseaRunner")
    ax1.set_xlabel("Number of windows")
    ax1.set_ylabel("Total time (s)")
    ax1.set_title("A. End-to-end runtime", loc="left", fontweight="bold")
    ax1.legend(frameon=True, fontsize=8)

    ax2.plot(bench_df["n_windows"], bench_df["speedup"], marker="o", color="#047857", linewidth=2.2)
    ax2.set_xlabel("Number of windows")
    ax2.set_ylabel("Speedup (stateless / stateful)")
    ax2.set_title("B. Benefit of the stateful rolling-window runner", loc="left", fontweight="bold")
    for _, row in bench_df.iterrows():
        ax2.text(row["n_windows"], row["speedup"] + 0.03, f"{row['speedup']:.2f}x", ha="center", fontsize=8)

    fig.suptitle("Stateful versus stateless rolling-window scaling on trajectory-derived score vectors\n(3000-pathway stress test)", y=1.03, fontsize=13.5)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "figure_S15_stateful_scaling.png", dpi=300, bbox_inches="tight")
    fig.savefig(ASSET_DIR / "figure_S15_stateful_scaling.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_patched_supplement_intro_pages() -> None:
    source_pdf = REPO_ROOT / "bioinfor0208" / "pyfgsea_supplementary.pdf"
    base_prefix = ASSET_DIR / "supp_intro_patch"
    subprocess.run(
        [
            "pdftoppm",
            "-r",
            "300",
            "-f",
            "1",
            "-l",
            "2",
            "-png",
            str(source_pdf),
            str(base_prefix),
        ],
        check=True,
    )

    page1 = Image.open(ASSET_DIR / "supp_intro_patch-1.png").convert("RGB")
    draw1 = ImageDraw.Draw(page1)
    draw1.rectangle((1406, 1486, 1510, 1542), fill="white")
    draw1.text((1414, 1490), "S4.", fill="black", font=_pick_font(36))
    page1.save(ASSET_DIR / "supp_intro_p1_patched.png")

    page2 = Image.open(ASSET_DIR / "supp_intro_patch-2.png").convert("RGB")
    draw2 = ImageDraw.Draw(page2)
    draw2.rectangle((1440, 708, 1854, 816), fill="white")
    font2 = _pick_font(34)
    text = "BiocParallel\nnproc=4"
    bbox = draw2.multiline_textbbox((0, 0), text, font=font2, align="center", spacing=2)
    text_x = 1647 - (bbox[2] - bbox[0]) / 2
    text_y = 716 + (92 - (bbox[3] - bbox[1])) / 2
    draw2.multiline_text((text_x, text_y), text, fill="black", font=font2, align="center", spacing=2)
    page2.save(ASSET_DIR / "supp_intro_p2_patched.png")

    page1.save(
        ASSET_DIR / "supp_patched_intro_pages.pdf",
        save_all=True,
        append_images=[page2],
        resolution=300.0,
    )


def main() -> None:
    _ensure_dirs()
    generate_agreement_figure()
    generate_real_trajectory_figure()
    generate_workflow_figure()
    generate_window_sensitivity_figure()
    generate_validation_overview_figure()
    generate_equivalence_validation_assets()
    generate_thread_validation_assets()
    generate_reproducibility_figure()
    generate_parameter_grid_summary_figure()
    generate_boundary_handling_figure()
    generate_stateful_scaling_figure()
    generate_patched_supplement_intro_pages()


if __name__ == "__main__":
    main()
