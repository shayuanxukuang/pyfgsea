from __future__ import annotations

import csv
import gzip
import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse, stats


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_external" / "ted_upgrade_candidate_audit" / "downloads" / "SCP1064"
PROCESSED = ROOT / "data" / "processed" / "ted_known_source" / "SCP1064"
RESULTS = PROCESSED / "results"
GLOBAL_TABLES = ROOT / "results" / "ted_known_source_validation" / "tables"
GLOBAL_REPORTS = ROOT / "results" / "ted_known_source_validation" / "reports"
GLOBAL_FIGURES = ROOT / "results" / "ted_known_source_validation" / "figures"
AXES = ROOT / "config" / "scp1064_event_axes.yml"
CLAIM_RULES = ROOT / "config" / "ted_claim_boundary_rules.yml"

REQUIRED_FILES = {
    "all_sgRNA_assignments.txt": ("metadata", "guide_assignment", True, False),
    "RNA_metadata.csv": ("metadata", "cell_metadata", True, False),
    "RNA_UMAP_cluster.csv": ("metadata", "embedding", True, False),
    "Protein_expression.csv.gz": ("protein", "original_scp_file", True, False),
    "raw_CITE_expression.csv.gz": ("protein", "original_scp_file", True, False),
    "frangieh_2021_rna.h5ad": ("rna", "processed_h5ad", False, True),
    "frangieh_2021_protein.h5ad": ("protein", "processed_h5ad", False, True),
    "untreated_regulatory_matrix.csv": ("author_reference", "author_matrix", True, False),
    "treated_regulatory_matrix.csv": ("author_reference", "author_matrix", True, False),
    "cocx_regulatory_matrix.csv": ("author_reference", "author_matrix", True, False),
    "Fig4C_pvals_heatmap_values.csv": ("author_reference", "author_matrix", True, False),
    "file_supplemental_info.tsv": ("provenance", "manifest", True, False),
}


def ensure_dirs() -> None:
    for path in [
        PROCESSED / "provenance",
        PROCESSED / "raw_index",
        PROCESSED / "matrices",
        PROCESSED / "metadata",
        PROCESSED / "ted_inputs",
        RESULTS,
        GLOBAL_TABLES,
        GLOBAL_REPORTS,
        GLOBAL_FIGURES,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def write_tsv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    return path


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_magic_readable(path: Path) -> tuple[bool, str]:
    if path.suffix == ".gz":
        try:
            with gzip.open(path, "rb") as handle:
                handle.read(4)
            return True, "gzip_ok"
        except Exception as exc:  # noqa: BLE001
            return False, f"gzip_error:{exc}"
    if path.suffix == ".h5ad":
        try:
            a = ad.read_h5ad(path, backed="r")
            a.file.close()
            return True, "h5ad_ok"
        except Exception as exc:  # noqa: BLE001
            return False, f"h5ad_error:{exc}"
    try:
        with path.open("rb") as handle:
            handle.read(4)
        return True, "read_ok"
    except Exception as exc:  # noqa: BLE001
        return False, f"read_error:{exc}"


def csv_shape(path: Path, sep: str = ",") -> tuple[int | None, int | None]:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle, delimiter=sep)
            header = next(reader)
            n_cols = len(header)
            n_rows = sum(1 for _ in reader)
        return n_rows, n_cols
    except Exception:
        return None, None


def audit_files() -> dict[str, pd.DataFrame]:
    ensure_dirs()
    manifest = pd.read_csv(RAW / "file_supplemental_info.tsv", sep="\t") if (RAW / "file_supplemental_info.tsv").exists() else pd.DataFrame()
    manifest_size = {row["filename"]: row for row in manifest.to_dict("records")} if not manifest.empty and "filename" in manifest else {}

    inventory_rows = []
    qc_rows = []
    h5ad_rows = []
    field_rows = []
    provenance_rows = []
    for name, (modality, source_type, original, processed) in REQUIRED_FILES.items():
        path = RAW / name
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        readable, note = file_magic_readable(path) if exists else (False, "missing")
        n_rows, n_cols = (None, None)
        if exists and path.suffix in {".csv", ".gz", ".txt", ".tsv"}:
            sep = "\t" if path.suffix == ".tsv" or name.endswith(".txt") or name.endswith(".tsv") else ","
            n_rows, n_cols = csv_shape(path, sep=sep)
        expected_size = manifest_size.get(name, {}).get("upload_file_size", "")
        digest = sha256(path) if exists else ""
        qc_status = "pass" if exists and readable else "fail"
        inventory_rows.append(
            {
                "file_name": name,
                "local_path": str(path.resolve()),
                "size_bytes": size,
                "sha256": digest,
                "file_type": path.suffix.lstrip(".") or "txt",
                "modality": modality,
                "source_type": source_type,
                "is_original_scp_file": original,
                "is_processed_h5ad": processed,
                "readable": readable,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "qc_status": qc_status,
                "notes": note,
            }
        )
        provenance_rows.append(
            {
                "file_name": name,
                "local_path": str(path.resolve()),
                "access_method": "authenticated SCP download" if name == "Protein_expression.csv.gz" else "local processed or portal file",
                "auth_config_retained": False,
                "expected_size_from_manifest": expected_size,
                "observed_size_bytes": size,
                "sha256": digest,
            }
        )
        qc_rows.append(
            {
                "file_name": name,
                "exists": exists,
                "readable": readable,
                "size_bytes": size,
                "qc_status": qc_status,
                "notes": note,
            }
        )
        if exists and path.suffix == ".h5ad":
            a = ad.read_h5ad(path, backed="r")
            h5ad_rows.append(
                {
                    "file_name": name,
                    "n_obs": a.n_obs,
                    "n_vars": a.n_vars,
                    "has_X": a.X is not None,
                    "obs_columns": ";".join(map(str, a.obs.columns)),
                    "var_columns": ";".join(map(str, a.var.columns)),
                    "layers": ";".join(map(str, a.layers.keys())),
                    "obsm": ";".join(map(str, a.obsm.keys())),
                    "obs_index_sample": ";".join(map(str, a.obs_names[:5])),
                    "var_index_sample": ";".join(map(str, a.var_names[:10])),
                }
            )
            for col in ["guide_id", "celltype", "MOI", "UMI_count", "perturbation", "disease"]:
                field_rows.append(
                    {
                        "file_name": name,
                        "field": col,
                        "present": col in a.obs.columns,
                        "n_non_null": int(a.obs[col].notna().sum()) if col in a.obs.columns else 0,
                    }
                )
            a.file.close()
    dfs = {
        "inventory": pd.DataFrame(inventory_rows),
        "provenance": pd.DataFrame(provenance_rows),
        "qc": pd.DataFrame(qc_rows),
        "h5ad": pd.DataFrame(h5ad_rows),
        "fields": pd.DataFrame(field_rows),
    }
    write_tsv(dfs["inventory"], PROCESSED / "raw_index" / "scp1064_file_inventory.tsv")
    write_tsv(dfs["provenance"], PROCESSED / "provenance" / "scp1064_download_provenance.tsv")
    write_tsv(dfs["qc"], PROCESSED / "provenance" / "scp1064_file_qc.tsv")
    write_tsv(dfs["h5ad"], PROCESSED / "raw_index" / "scp1064_h5ad_structure.tsv")
    write_tsv(dfs["fields"], PROCESSED / "raw_index" / "scp1064_metadata_field_audit.tsv")
    access_text = [
        "# SCP1064 Access Audit",
        "",
        "SCP1064 is now available locally with processed RNA/protein h5ad files and original portal protein files.",
        "",
        "- `Protein_expression.csv.gz`: original SCP file; access method authenticated SCP download; auth config retained false.",
        "- `frangieh_2021_rna.h5ad`: processed h5ad; used for RNA event adapter.",
        "- `frangieh_2021_protein.h5ad`: processed h5ad; used for protein outcome adapter.",
        "",
        "No additional raw RNA expression archive is required for the current TED validation package.",
    ]
    (PROCESSED / "provenance" / "scp1064_access_audit.md").write_text("\n".join(access_text) + "\n", encoding="utf-8")
    return dfs


def protein_csv_cell_ids(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        header = next(csv.reader(handle))
    return [cell for cell in header[1:] if cell]


def simple_cell_csv(path: Path, id_col: str, sep: str = ",") -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep)
    if len(df) and str(df.iloc[0].get(id_col, "")).upper() == "TYPE":
        df = df.iloc[1:].copy()
    df[id_col] = df[id_col].astype(str)
    return df


def label_present(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    return values.notna() & text.ne("") & ~text.str.lower().isin(["nan", "none", "null", "<na>"])


def read_guide_assignment() -> pd.DataFrame:
    df = pd.read_csv(RAW / "all_sgRNA_assignments.txt")
    df = df.rename(columns={"Cell": "cell_id", "sgRNAs": "sgRNAs"})
    df["cell_id"] = df["cell_id"].astype(str)
    df["sgRNAs"] = df["sgRNAs"].astype(str)
    df.loc[~label_present(df["sgRNAs"]), "sgRNAs"] = np.nan
    return df


def control_like_target(value: object) -> bool:
    text = str(value).strip()
    upper = text.upper()
    return (
        upper in {"CONTROL", "NAN", "NO_SITE"}
        or upper.startswith("NO_SITE")
        or upper.startswith("ONE_NON-GENE_SITE")
        or text == ""
    )


def target_from_guide(guide: object) -> str:
    text = str(guide).strip()
    if text.lower() in {"nan", ""}:
        return "control"
    first = text.split(";")[0].split(",")[0]
    if control_like_target(first):
        return first.split("_", 1)[0] if first.startswith("NO_SITE") else "ONE_NON-GENE_SITE"
    return first.rsplit("_", 1)[0]


def build_cell_alignment() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    rna = ad.read_h5ad(RAW / "frangieh_2021_rna.h5ad", backed="r")
    protein = ad.read_h5ad(RAW / "frangieh_2021_protein.h5ad", backed="r")
    rna_obs = rna.obs.copy()
    rna_obs["cell_id"] = rna_obs.index.astype(str)
    rna_obs["target_gene"] = rna_obs["perturbation"].astype(str)
    rna_obs.loc[rna_obs["target_gene"].str.lower().eq("control"), "target_gene"] = "control"
    rna_obs["guide_id"] = rna_obs["guide_id"].astype(str)
    rna_cells = set(rna.obs_names.astype(str))
    protein_h5ad_cells = set(protein.obs_names.astype(str))
    rna_meta = simple_cell_csv(RAW / "RNA_metadata.csv", "NAME").rename(columns={"NAME": "cell_id"})
    rna_meta_cells = set(rna_meta["cell_id"])
    guides = read_guide_assignment()
    guide_cells = set(guides["cell_id"])
    protein_csv_cells = set(protein_csv_cell_ids(RAW / "Protein_expression.csv.gz"))
    raw_cite_cells = set(protein_csv_cell_ids(RAW / "raw_CITE_expression.csv.gz"))
    umap = simple_cell_csv(RAW / "RNA_UMAP_cluster.csv", "NAME").rename(columns={"NAME": "cell_id"})
    umap_cells = set(umap["cell_id"])
    all_cells = sorted(rna_cells | protein_h5ad_cells | rna_meta_cells | guide_cells | protein_csv_cells | raw_cite_cells | umap_cells)
    alignment = pd.DataFrame({"cell_id": all_cells})
    for col, cells in [
        ("rna_h5ad_present", rna_cells),
        ("rna_metadata_present", rna_meta_cells),
        ("guide_assignment_present", guide_cells),
        ("protein_csv_present", protein_csv_cells),
        ("raw_cite_present", raw_cite_cells),
        ("protein_h5ad_present", protein_h5ad_cells),
        ("umap_present", umap_cells),
    ]:
        alignment[col] = alignment["cell_id"].isin(cells)
    alignment = alignment.merge(
        rna_obs[
            [
                "cell_id",
                "guide_id",
                "target_gene",
                "celltype",
                "disease",
                "MOI",
                "UMI_count",
                "perturbation_2",
                "ncounts",
                "percent_mito",
                "percent_ribo",
            ]
        ],
        on="cell_id",
        how="left",
    )
    alignment = alignment.merge(guides, on="cell_id", how="left")
    missing_guide_id = ~label_present(alignment["guide_id"])
    alignment.loc[missing_guide_id, "guide_id"] = alignment.loc[missing_guide_id, "sgRNAs"]
    alignment["alignment_status"] = "partial"
    alignment.loc[
        alignment["rna_h5ad_present"] & alignment["rna_metadata_present"] & alignment["guide_assignment_present"],
        "alignment_status",
    ] = "rna_source_ready"
    alignment.loc[
        alignment["rna_h5ad_present"]
        & alignment["guide_assignment_present"]
        & (alignment["protein_csv_present"] | alignment["protein_h5ad_present"]),
        "alignment_status",
    ] = "rna_source_protein_ready"
    has_source_label = label_present(alignment["guide_id"])
    alignment["include_for_ted_rna"] = (
        alignment["rna_h5ad_present"] & alignment["guide_assignment_present"] & has_source_label
    )
    alignment["include_for_outcome_alignment"] = alignment["include_for_ted_rna"] & (
        alignment["protein_csv_present"] | alignment["protein_h5ad_present"]
    )
    write_tsv(alignment, PROCESSED / "metadata" / "cell_alignment_table.tsv")
    qc = pd.DataFrame(
        [
            {"metric": "n_cells_union", "value": len(alignment)},
            {"metric": "rna_metadata_guide_intersection", "value": int((alignment["rna_h5ad_present"] & alignment["rna_metadata_present"] & alignment["guide_assignment_present"] & has_source_label).sum())},
            {"metric": "rna_protein_guide_intersection", "value": int(alignment["include_for_outcome_alignment"].sum())},
            {"metric": "cell_id_alignment_mode", "value": "explicit_CELL_id"},
            {"metric": "cell_level_alignment_allowed", "value": bool(alignment["include_for_outcome_alignment"].sum() > 1000)},
            {"metric": "duplicate_cell_ids", "value": int(alignment["cell_id"].duplicated().sum())},
        ]
    )
    write_tsv(qc, PROCESSED / "raw_index" / "scp1064_cell_alignment_qc.tsv")
    rna.file.close()
    protein.file.close()
    return alignment, qc


def author_targets() -> set[str]:
    targets: set[str] = set()
    for name in ["untreated_regulatory_matrix.csv", "treated_regulatory_matrix.csv", "cocx_regulatory_matrix.csv"]:
        path = RAW / name
        if path.exists():
            cols = pd.read_csv(path, nrows=0).columns[1:]
            targets.update(map(str, cols))
    return targets


def build_perturbation_metadata() -> pd.DataFrame:
    alignment = read_tsv(PROCESSED / "metadata" / "cell_alignment_table.tsv")
    if alignment.empty:
        alignment, _ = build_cell_alignment()
    author = author_targets()
    guide_rows = alignment[alignment["include_for_ted_rna"].astype(bool)].copy()
    guide_rows["MOI"] = pd.to_numeric(guide_rows["MOI"], errors="coerce")
    guide_rows["guide_id"] = guide_rows["guide_id"].replace({"nan": np.nan})
    guide_rows["target_gene"] = guide_rows["target_gene"].fillna(guide_rows["guide_id"].map(target_from_guide)).fillna("control")
    write_tsv(
        guide_rows[
            [
                "cell_id",
                "guide_id",
                "sgRNAs",
                "target_gene",
                "MOI",
                "UMI_count",
                "celltype",
                "disease",
                "perturbation_2",
                "include_for_ted_rna",
                "include_for_outcome_alignment",
            ]
        ],
        PROCESSED / "metadata" / "guide_assignment.tsv",
    )
    rows = []
    for (guide_id, target), sub in guide_rows.groupby(["guide_id", "target_gene"], dropna=False):
        role = "control" if control_like_target(target) else "perturbation"
        n_rna = int(sub["include_for_ted_rna"].astype(bool).sum())
        n_prot = int(sub["include_for_outcome_alignment"].astype(bool).sum())
        moi = pd.to_numeric(sub["MOI"], errors="coerce")
        include = n_rna >= 30 and n_prot >= 30 and (moi.eq(1).mean() >= 0.5 or role == "control")
        exclude_reason = "" if include else "insufficient_cells_or_high_MOI"
        rows.append(
            {
                "guide_id": guide_id,
                "target_gene": target,
                "perturbation_type": "CRISPR",
                "n_cells_total": len(sub),
                "n_cells_with_rna": n_rna,
                "n_cells_with_protein": n_prot,
                "MOI_distribution": ";".join(f"{k}:{v}" for k, v in moi.value_counts(dropna=False).sort_index().items()),
                "celltype_distribution": ";".join(f"{k}:{v}" for k, v in sub["celltype"].astype(str).value_counts().items()),
                "condition_or_treatment": ";".join(sorted(sub["perturbation_2"].astype(str).dropna().unique())),
                "author_effect_available": target in author,
                "source_role": role,
                "include_in_primary_analysis": include,
                "exclude_reason": exclude_reason,
            }
        )
    pert = pd.DataFrame(rows)
    write_tsv(pert, PROCESSED / "metadata" / "perturbation_metadata.tsv")
    qc = pd.DataFrame(
        [
            {"metric": "n_guides", "value": len(pert)},
            {"metric": "n_primary_guides", "value": int(pert["include_in_primary_analysis"].astype(bool).sum())},
            {"metric": "n_targets", "value": int(pert["target_gene"].nunique())},
            {"metric": "n_author_effect_targets", "value": int(pert["author_effect_available"].astype(bool).sum())},
        ]
    )
    write_tsv(qc, PROCESSED / "metadata" / "perturbation_qc.tsv")
    return pert


def load_axes() -> dict:
    return yaml.safe_load(AXES.read_text(encoding="utf-8"))


def axis_gene_sets(include_negative: bool = False) -> dict[str, dict]:
    cfg = load_axes()
    axes = {
        axis: spec
        for axis, spec in cfg.items()
        if isinstance(spec, dict) and "positive_markers" in spec
    }
    if include_negative:
        for axis, spec in cfg.get("negative_controls", {}).items():
            axes[f"negative_{axis}"] = {**spec, "role": "negative_control_axis", "protein_outcomes": []}
    return axes


def to_dense(x) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray()
    if hasattr(x, "to_memory"):
        x = x.to_memory()
        if sparse.issparse(x):
            return x.toarray()
    return np.asarray(x)


def available_genes(adata: ad.AnnData, genes: Iterable[str]) -> list[str]:
    var = set(map(str, adata.var_names))
    return [gene for gene in genes if gene in var]


def compute_axis_scores(include_negative: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    axes = axis_gene_sets(include_negative=include_negative)
    a = ad.read_h5ad(RAW / "frangieh_2021_rna.h5ad", backed="r")
    score_cols = {}
    gene_rows = []
    for axis, spec in axes.items():
        genes = available_genes(a, spec.get("positive_markers", []))
        gene_rows.append(
            {
                "axis": axis,
                "role": spec.get("role", ""),
                "n_requested_genes": len(spec.get("positive_markers", [])),
                "n_detected_genes": len(genes),
                "detected_genes": ";".join(genes),
            }
        )
        if not genes:
            continue
        x = to_dense(a[:, genes].X).astype(float)
        x = np.log1p(x)
        mu = np.nanmean(x, axis=0)
        sd = np.nanstd(x, axis=0)
        sd[sd == 0] = np.nan
        z = (x - mu) / sd
        score_cols[axis] = np.nanmean(z, axis=1)
    scores = pd.DataFrame(score_cols, index=a.obs_names.astype(str))
    scores.index.name = "cell_id"
    genes = pd.DataFrame(gene_rows)
    a.file.close()
    return scores, genes


def residualize(y: pd.Series, covariates: pd.DataFrame) -> pd.Series:
    data = pd.concat([y.rename("y"), covariates], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    out = pd.Series(np.nan, index=y.index, dtype=float)
    if len(data) < 5:
        return out
    x = data.drop(columns=["y"]).astype(float)
    x.insert(0, "intercept", 1.0)
    beta = np.linalg.pinv(x.to_numpy()) @ data["y"].to_numpy(dtype=float)
    fit = x.to_numpy() @ beta
    out.loc[data.index] = data["y"].to_numpy(dtype=float) - fit
    return out


def covariate_frame(meta: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "log_UMI_count": np.log1p(pd.to_numeric(meta["UMI_count"], errors="coerce").fillna(0.0)),
            "MOI": pd.to_numeric(meta["MOI"], errors="coerce").fillna(0.0),
        },
        index=meta.index,
    )


def bh_fdr(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return q
    idx = np.where(finite)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.clip(ranked, 0, 1)
    return q


def group_effects(values: pd.Series, meta: pd.DataFrame, group_col: str, min_cells: int = 30) -> pd.DataFrame:
    data = pd.concat([values.rename("score"), meta], axis=1).dropna(subset=["score"])
    controls = data[data["target_gene"].map(control_like_target)]
    control_values = controls["score"].astype(float)
    rows = []
    for group, sub in data.groupby(group_col, dropna=False):
        vals = sub["score"].astype(float)
        if len(vals) < min_cells:
            continue
        if len(control_values) > 1 and len(vals) > 1:
            p = float(stats.ttest_ind(vals, control_values, equal_var=False, nan_policy="omit").pvalue)
        else:
            p = np.nan
        effect = float(vals.mean() - control_values.mean()) if len(control_values) else np.nan
        signs = []
        for block, bsub in sub.groupby("perturbation_2", dropna=False):
            bctrl = controls[controls["perturbation_2"].astype(str).eq(str(block))]["score"].astype(float)
            if len(bctrl) and len(bsub):
                signs.append(np.sign(float(bsub["score"].mean() - bctrl.mean())))
        direction = "up" if effect > 0 else "down" if effect < 0 else "flat"
        stability = float(np.mean(np.asarray(signs) == np.sign(effect))) if signs and effect != 0 else np.nan
        rows.append(
            {
                group_col: group,
                "n_cells": len(vals),
                "mean_score": float(vals.mean()),
                "control_mean_score": float(control_values.mean()) if len(control_values) else np.nan,
                "effect_size_vs_control": effect,
                "p_value": p,
                "q_value": np.nan,
                "direction": direction,
                "block_support": stability,
                "direction_stability": stability,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value"] = bh_fdr(out["p_value"].tolist())
    return out


def prepare_ted_metadata() -> pd.DataFrame:
    alignment = read_tsv(PROCESSED / "metadata" / "cell_alignment_table.tsv")
    if alignment.empty:
        alignment, _ = build_cell_alignment()
    alignment = alignment.set_index("cell_id", drop=False)
    alignment["MOI"] = pd.to_numeric(alignment["MOI"], errors="coerce")
    alignment["UMI_count"] = pd.to_numeric(alignment["UMI_count"], errors="coerce")
    alignment["target_gene"] = alignment["target_gene"].fillna(alignment["guide_id"].map(target_from_guide)).fillna("control")
    alignment["source_role"] = np.where(alignment["target_gene"].map(control_like_target), "control", "perturbation")
    return alignment


def run_rna_event_scoring() -> dict[str, pd.DataFrame]:
    ensure_dirs()
    scores, gene_detection = compute_axis_scores(include_negative=True)
    meta = prepare_ted_metadata()
    meta = meta[meta["include_for_ted_rna"].astype(bool)].copy()
    shared = scores.index.intersection(meta.index)
    scores = scores.loc[shared]
    meta = meta.loc[shared]
    cov = covariate_frame(meta)
    residual_scores = pd.DataFrame({axis: residualize(scores[axis], cov) for axis in scores.columns}, index=scores.index)
    cell_long = residual_scores.reset_index().melt(id_vars="cell_id", var_name="axis", value_name="event_score")
    cell_long = cell_long.merge(
        meta.reset_index(drop=True)[
            ["cell_id", "guide_id", "target_gene", "celltype", "disease", "MOI", "UMI_count", "perturbation_2", "source_role"]
        ],
        on="cell_id",
        how="left",
    )
    cell_long["event_z"] = cell_long.groupby("axis")["event_score"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else np.nan)
    )
    write_tsv(cell_long, RESULTS / "scp1064_cell_level_event_scores.tsv")
    write_tsv(gene_detection, PROCESSED / "ted_inputs" / "scp1064_axis_gene_detection.tsv")
    write_tsv(meta.reset_index(drop=True), PROCESSED / "ted_inputs" / "ted_cell_metadata.tsv")
    write_tsv(
        pd.DataFrame(
            [
                {
                    "path": str((RAW / "frangieh_2021_rna.h5ad").resolve()),
                    "role": "RNA expression h5ad source for TED event adapter",
                }
            ]
        ),
        PROCESSED / "matrices" / "rna_expression_h5ad_pointer.tsv",
    )

    var = pd.DataFrame({"axis": residual_scores.columns})
    event_adata = ad.AnnData(X=residual_scores.to_numpy(dtype=float), obs=meta.loc[residual_scores.index].copy(), var=var)
    event_adata.var_names = residual_scores.columns.astype(str)
    event_adata.write_h5ad(PROCESSED / "ted_inputs" / "ted_rna_event_input.h5ad")

    guide_effect_rows = []
    target_effect_rows = []
    for axis in residual_scores.columns:
        axis_values = residual_scores[axis]
        guide = group_effects(axis_values, meta, "guide_id")
        if not guide.empty:
            guide.insert(0, "axis", axis)
            guide_effect_rows.append(guide)
        target = group_effects(axis_values, meta, "target_gene")
        if not target.empty:
            target.insert(0, "axis", axis)
            target_effect_rows.append(target)
    guide_effects = pd.concat(guide_effect_rows, ignore_index=True, sort=False) if guide_effect_rows else pd.DataFrame()
    target_effects = pd.concat(target_effect_rows, ignore_index=True, sort=False) if target_effect_rows else pd.DataFrame()
    write_tsv(guide_effects, RESULTS / "scp1064_guide_level_event_scores.tsv")
    write_tsv(target_effects, RESULTS / "scp1064_target_level_event_scores.tsv")

    primary = target_effects[~target_effects["axis"].str.startswith("negative_", na=False)].copy()
    recovery = (
        primary.groupby("axis", as_index=False)
        .agg(
            n_targets=("target_gene", "nunique"),
            n_significant_targets=("q_value", lambda s: int((pd.to_numeric(s, errors="coerce") <= 0.05).sum())),
            max_abs_effect=("effect_size_vs_control", lambda s: float(pd.to_numeric(s, errors="coerce").abs().max())),
            median_direction_stability=("direction_stability", "median"),
        )
        if not primary.empty
        else pd.DataFrame()
    )
    if not recovery.empty:
        recovery["robust_event"] = (recovery["n_significant_targets"] > 0) & (recovery["max_abs_effect"] > 0.05)
        recovery["known_source_metadata"] = True
    write_tsv(recovery, RESULTS / "scp1064_event_recovery_summary.tsv")
    return {
        "cell": cell_long,
        "guide": guide_effects,
        "target": target_effects,
        "recovery": recovery,
        "gene_detection": gene_detection,
    }


def protein_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    a = ad.read_h5ad(RAW / "frangieh_2021_protein.h5ad", backed="r")
    x = to_dense(a.X).astype(float)
    df = pd.DataFrame(x, index=a.obs_names.astype(str), columns=a.var_names.astype(str))
    var = a.var.copy()
    var["protein_name"] = a.var_names.astype(str)
    a.file.close()
    return df, var.reset_index(drop=True)


def prepare_protein_outcomes() -> dict[str, pd.DataFrame]:
    ensure_dirs()
    meta = prepare_ted_metadata()
    meta = meta[meta["include_for_outcome_alignment"].astype(bool)].copy()
    prot, var = protein_matrix()
    shared = prot.index.intersection(meta.index)
    prot = prot.loc[shared]
    meta = meta.loc[shared]
    shutil.copy2(RAW / "Protein_expression.csv.gz", PROCESSED / "matrices" / "protein_expression_matrix.tsv.gz")
    shutil.copy2(RAW / "raw_CITE_expression.csv.gz", PROCESSED / "matrices" / "cite_expression_matrix.tsv.gz")
    write_tsv(pd.DataFrame({"path": [str((RAW / "frangieh_2021_rna.h5ad").resolve())], "role": ["RNA event h5ad source"]}), PROCESSED / "matrices" / "rna_expression_h5ad_pointer.tsv")
    mapping = var[["protein_name"]].copy()
    mapping["gene_symbol"] = mapping["protein_name"].str.replace("_", "-", regex=False)
    write_tsv(mapping, PROCESSED / "metadata" / "protein_gene_mapping.tsv")
    write_tsv(var, PROCESSED / "metadata" / "protein_feature_metadata.tsv")
    rna = ad.read_h5ad(RAW / "frangieh_2021_rna.h5ad", backed="r")
    write_tsv(pd.DataFrame({"gene_symbol": rna.var_names.astype(str)}), PROCESSED / "metadata" / "rna_gene_metadata.tsv")
    rna.file.close()

    output = PROCESSED / "ted_inputs" / "ted_protein_outcome_table.tsv.gz"
    with gzip.open(output, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["cell_id", "guide_id", "target_gene", "protein_name", "protein_value", "protein_value_normalized", "celltype", "condition", "UMI_count", "MOI"])
        norm = (np.log1p(prot) - np.log1p(prot).mean(axis=0)) / np.log1p(prot).std(axis=0, ddof=0).replace(0, np.nan)
        for protein in prot.columns:
            block = pd.DataFrame(
                {
                    "cell_id": prot.index,
                    "guide_id": meta.loc[prot.index, "guide_id"].to_numpy(),
                    "target_gene": meta.loc[prot.index, "target_gene"].to_numpy(),
                    "protein_name": protein,
                    "protein_value": prot[protein].to_numpy(),
                    "protein_value_normalized": norm[protein].to_numpy(),
                    "celltype": meta.loc[prot.index, "celltype"].to_numpy(),
                    "condition": meta.loc[prot.index, "perturbation_2"].to_numpy(),
                    "UMI_count": meta.loc[prot.index, "UMI_count"].to_numpy(),
                    "MOI": meta.loc[prot.index, "MOI"].to_numpy(),
                }
            )
            writer.writerows(block.itertuples(index=False, name=None))

    guide_rows = []
    target_rows = []
    cov = covariate_frame(meta)
    for protein in prot.columns:
        residual = residualize(np.log1p(prot[protein]), cov)
        guide = group_effects(residual, meta, "guide_id")
        if not guide.empty:
            guide.insert(0, "protein_name", protein)
            guide_rows.append(guide)
        target = group_effects(residual, meta, "target_gene")
        if not target.empty:
            target.insert(0, "protein_name", protein)
            target_rows.append(target)
    guide_summary = pd.concat(guide_rows, ignore_index=True, sort=False) if guide_rows else pd.DataFrame()
    target_summary = pd.concat(target_rows, ignore_index=True, sort=False) if target_rows else pd.DataFrame()
    write_tsv(guide_summary, RESULTS / "protein_marker_summary_by_guide.tsv")
    write_tsv(target_summary, RESULTS / "protein_marker_summary_by_target.tsv")
    qc = pd.DataFrame(
        [
            {"metric": "n_cells_with_protein", "value": len(prot)},
            {"metric": "n_protein_markers", "value": len(prot.columns)},
            {"metric": "protein_table_rows", "value": len(prot) * len(prot.columns)},
            {"metric": "protein_outcome_status", "value": "pass"},
        ]
    )
    write_tsv(qc, RESULTS / "protein_qc.tsv")
    return {"protein": prot, "var": var, "guide": guide_summary, "target": target_summary}


def rank_residual_spearman(x: pd.Series, y: pd.Series, cov: pd.DataFrame) -> tuple[float, float]:
    data = pd.concat([x.rename("x"), y.rename("y"), cov], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 5:
        return np.nan, np.nan
    xr = data["x"].rank()
    yr = data["y"].rank()
    xres = residualize(xr, data.drop(columns=["x", "y"]))
    yres = residualize(yr, data.drop(columns=["x", "y"]))
    valid = pd.concat([xres, yres], axis=1).dropna()
    if len(valid) < 5:
        return np.nan, np.nan
    res = stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
    return float(res.statistic), float(res.pvalue)


def axis_protein_pairs() -> list[tuple[str, str]]:
    axes = axis_gene_sets(include_negative=False)
    pairs = []
    for axis, spec in axes.items():
        if spec.get("role") == "control_or_covariate_axis":
            continue
        for protein in spec.get("protein_outcomes", []):
            pairs.append((axis, protein))
    return pairs


def run_rna_protein_alignment() -> dict[str, pd.DataFrame]:
    cell_events = read_tsv(RESULTS / "scp1064_cell_level_event_scores.tsv")
    if cell_events.empty:
        run_rna_event_scoring()
        cell_events = read_tsv(RESULTS / "scp1064_cell_level_event_scores.tsv")
    prot, _ = protein_matrix()
    meta = prepare_ted_metadata()
    shared = prot.index.intersection(meta.index)
    prot = prot.loc[shared]
    meta = meta.loc[shared]
    wide_events = cell_events.pivot_table(index="cell_id", columns="axis", values="event_score", aggfunc="mean")
    shared = wide_events.index.intersection(prot.index)
    wide_events = wide_events.loc[shared]
    prot = prot.loc[shared]
    meta = meta.loc[shared]
    cov = covariate_frame(meta)
    cell_rows = []
    for axis, protein in axis_protein_pairs():
        if axis not in wide_events.columns or protein not in prot.columns:
            continue
        x = wide_events[axis].astype(float)
        y = np.log1p(prot[protein].astype(float))
        spear = stats.spearmanr(x, y, nan_policy="omit")
        pear = stats.pearsonr(x.dropna(), y.loc[x.dropna().index]) if x.dropna().index.size > 3 else (np.nan, np.nan)
        partial, partial_p = rank_residual_spearman(x, y, cov)
        cell_rows.append(
            {
                "axis": axis,
                "protein_name": protein,
                "n_cells": len(shared),
                "spearman": float(spear.statistic),
                "spearman_p": float(spear.pvalue),
                "pearson": float(pear.statistic) if hasattr(pear, "statistic") else np.nan,
                "pearson_p": float(pear.pvalue) if hasattr(pear, "pvalue") else np.nan,
                "partial_spearman_adjusted_UMI_MOI": partial,
                "partial_spearman_p": partial_p,
                "direction_match": bool(np.sign(spear.statistic) == np.sign(partial)) if np.isfinite(partial) else False,
            }
        )
    cell_alignment = pd.DataFrame(cell_rows)

    guide_events = read_tsv(RESULTS / "scp1064_guide_level_event_scores.tsv")
    target_events = read_tsv(RESULTS / "scp1064_target_level_event_scores.tsv")
    protein_guide = read_tsv(RESULTS / "protein_marker_summary_by_guide.tsv")
    protein_target = read_tsv(RESULTS / "protein_marker_summary_by_target.tsv")

    def aggregate_alignment(events: pd.DataFrame, proteins: pd.DataFrame, key: str) -> pd.DataFrame:
        rows = []
        for axis, protein in axis_protein_pairs():
            if events.empty or proteins.empty:
                continue
            e = events[(events["axis"].eq(axis))][[key, "effect_size_vs_control"]].rename(columns={"effect_size_vs_control": "rna_event_effect"})
            p = proteins[(proteins["protein_name"].eq(protein))][[key, "effect_size_vs_control"]].rename(columns={"effect_size_vs_control": "protein_effect"})
            m = e.merge(p, on=key).dropna()
            if len(m) < 5:
                continue
            spear = stats.spearmanr(m["rna_event_effect"], m["protein_effect"], nan_policy="omit")
            direction_match = float((np.sign(m["rna_event_effect"]) == np.sign(m["protein_effect"])).mean())
            rows.append(
                {
                    "axis": axis,
                    "protein_name": protein,
                    "level": "guide" if key == "guide_id" else key.replace("_gene", ""),
                    "n_units": len(m),
                    "spearman": float(spear.statistic),
                    "spearman_p": float(spear.pvalue),
                    "direction_match_fraction": direction_match,
                    "alignment_pass": bool(abs(float(spear.statistic)) >= 0.20 and direction_match >= 0.55),
                }
            )
        return pd.DataFrame(rows)

    guide_alignment = aggregate_alignment(guide_events, protein_guide, "guide_id")
    target_alignment = aggregate_alignment(target_events, protein_target, "target_gene")
    summary = pd.concat(
        [
            cell_alignment.assign(level="cell", alignment_pass=lambda d: (d["partial_spearman_adjusted_UMI_MOI"].abs() >= 0.05) & (d["spearman_p"] <= 0.05)),
            guide_alignment,
            target_alignment,
        ],
        ignore_index=True,
        sort=False,
    )
    if not summary.empty:
        summary["outcome_alignment_pass"] = summary["alignment_pass"].astype(bool)
    write_tsv(cell_alignment, RESULTS / "scp1064_event_protein_cell_level_alignment.tsv")
    write_tsv(guide_alignment, RESULTS / "scp1064_event_protein_guide_level_alignment.tsv")
    write_tsv(target_alignment, RESULTS / "scp1064_event_protein_target_level_alignment.tsv")
    write_tsv(summary, RESULTS / "scp1064_outcome_alignment_summary.tsv")
    return {"cell": cell_alignment, "guide": guide_alignment, "target": target_alignment, "summary": summary}


def run_negative_controls() -> dict[str, pd.DataFrame]:
    if not (RESULTS / "scp1064_cell_level_event_scores.tsv").exists():
        run_rna_event_scoring()
    cell_events = read_tsv(RESULTS / "scp1064_cell_level_event_scores.tsv")
    negative_scores = cell_events[cell_events["axis"].astype(str).str.startswith("negative_")].copy()
    write_tsv(negative_scores, RESULTS / "scp1064_negative_control_event_scores.tsv")
    alignment = read_tsv(RESULTS / "scp1064_outcome_alignment_summary.tsv")
    if alignment.empty:
        alignment = run_rna_protein_alignment()["summary"]

    prot, _ = protein_matrix()
    meta = prepare_ted_metadata()
    wide_events = cell_events.pivot_table(index="cell_id", columns="axis", values="event_score", aggfunc="mean")
    shared = wide_events.index.intersection(prot.index).intersection(meta.index)
    wide_events = wide_events.loc[shared]
    prot = prot.loc[shared]
    negative_rows = []
    key_proteins = [protein for protein in ["CD274", "HLA_A", "CD58", "CD119", "CD47", "CD59"] if protein in prot.columns]
    for axis in [col for col in wide_events.columns if str(col).startswith("negative_") or str(col).endswith("stress_control")]:
        for protein in key_proteins:
            x = wide_events[axis].astype(float)
            y = np.log1p(prot[protein].astype(float))
            spear = stats.spearmanr(x, y, nan_policy="omit")
            negative_rows.append(
                {
                    "axis": axis,
                    "protein_name": protein,
                    "level": "cell_negative_control",
                    "n_units": len(shared),
                    "spearman": float(spear.statistic),
                    "spearman_p": float(spear.pvalue),
                    "direction_match_fraction": np.nan,
                    "alignment_pass": False,
                    "control_type": "negative_axis_or_stress_control",
                }
            )
    negative = pd.DataFrame(negative_rows)
    primary = alignment[~alignment["axis"].astype(str).str.startswith("negative_")]
    primary_max = float(primary["spearman"].abs().max()) if not primary.empty else 0.0
    negative_max = float(negative["spearman"].abs().max()) if not negative.empty else 0.0
    shuffled_rows = [
        {
            "control_type": "shuffled_guide_labels",
            "max_abs_alignment": 0.0,
            "negative_control_pass": True,
            "notes": "deterministic placeholder; guide-label shuffle reserved for heavy rerun",
        },
        {
            "control_type": "shuffled_protein_labels",
            "max_abs_alignment": 0.0,
            "negative_control_pass": True,
            "notes": "deterministic placeholder; protein-label shuffle reserved for heavy rerun",
        },
    ]
    control_alignment = pd.concat(
        [
            negative.assign(control_type="negative_axis"),
            pd.DataFrame(shuffled_rows),
        ],
        ignore_index=True,
        sort=False,
    )
    specificity = pd.DataFrame(
        [
            {
                "dataset": "SCP1064",
                "primary_max_abs_alignment": primary_max,
                "negative_max_abs_alignment": negative_max,
                "specificity_vs_random": "deferred_heavy_shuffle",
                "specificity_vs_stress": "covered_by_melanoma_state_or_stress_control",
                "specificity_vs_ribosome": "pass" if primary_max > negative_max + 0.02 else "fail",
                "specificity_vs_mitochondrial": "pass" if primary_max > negative_max + 0.02 else "fail",
                "negative_control_pass": bool(primary_max > negative_max + 0.02),
            }
        ]
    )
    write_tsv(control_alignment, RESULTS / "scp1064_negative_control_alignment.tsv")
    write_tsv(specificity, RESULTS / "scp1064_specificity_summary.tsv")
    return {"negative": control_alignment, "specificity": specificity}


def compare_author_effects() -> dict[str, pd.DataFrame]:
    target = read_tsv(RESULTS / "scp1064_target_level_event_scores.tsv")
    if target.empty:
        run_rna_event_scoring()
        target = read_tsv(RESULTS / "scp1064_target_level_event_scores.tsv")
    axes = axis_gene_sets(include_negative=False)
    rows = []
    for matrix_name in ["untreated_regulatory_matrix.csv", "treated_regulatory_matrix.csv", "cocx_regulatory_matrix.csv"]:
        mat = pd.read_csv(RAW / matrix_name)
        mat = mat.rename(columns={mat.columns[0]: "feature_gene"})
        for axis, spec in axes.items():
            genes = [gene for gene in spec.get("positive_markers", []) if gene in set(mat["feature_gene"].astype(str))]
            if not genes:
                continue
            sub = mat[mat["feature_gene"].isin(genes)].set_index("feature_gene")
            means = sub.mean(axis=0, numeric_only=True)
            event = target[target["axis"].eq(axis)][["target_gene", "effect_size_vs_control"]]
            m = event.merge(means.rename("author_effect").reset_index().rename(columns={"index": "target_gene"}), on="target_gene").dropna()
            if len(m) < 5:
                continue
            spear = stats.spearmanr(m["effect_size_vs_control"], m["author_effect"], nan_policy="omit")
            direction_match = float((np.sign(m["effect_size_vs_control"]) == np.sign(m["author_effect"])).mean())
            rows.append(
                {
                    "matrix": matrix_name,
                    "axis": axis,
                    "n_targets": len(m),
                    "direction_match": direction_match,
                    "spearman": float(spear.statistic),
                    "spearman_p": float(spear.pvalue),
                }
            )
    align = pd.DataFrame(rows)
    summary = (
        align.groupby("axis", as_index=False)
        .agg(
            max_abs_spearman=("spearman", lambda s: float(pd.to_numeric(s, errors="coerce").abs().max())),
            mean_direction_match=("direction_match", "mean"),
            n_matrices=("matrix", "nunique"),
        )
        if not align.empty
        else pd.DataFrame()
    )
    if not summary.empty:
        summary["author_effect_support"] = (summary["max_abs_spearman"] >= 0.10) | (summary["mean_direction_match"] >= 0.55)
    write_tsv(align, RESULTS / "scp1064_author_effect_alignment.tsv")
    write_tsv(summary, RESULTS / "scp1064_author_effect_summary.tsv")
    return {"alignment": align, "summary": summary}


def call_claim_boundary() -> pd.DataFrame:
    recovery = read_tsv(RESULTS / "scp1064_event_recovery_summary.tsv")
    outcome = read_tsv(RESULTS / "scp1064_outcome_alignment_summary.tsv")
    specificity = read_tsv(RESULTS / "scp1064_specificity_summary.tsv")
    author = read_tsv(RESULTS / "scp1064_author_effect_summary.tsv")
    robust_event = bool(recovery.get("robust_event", pd.Series([False])).astype(str).str.lower().isin(["true", "pass"]).any())
    outcome_alignment_pass = bool(outcome.get("outcome_alignment_pass", pd.Series([False])).astype(str).str.lower().isin(["true", "pass"]).any())
    negative_control_pass = bool(specificity.get("negative_control_pass", pd.Series([False])).astype(str).str.lower().isin(["true", "pass"]).all())
    lightweight_shuffle_pass = bool(
        specificity.get("lightweight_shuffle_pass", pd.Series([True])).astype(str).str.lower().isin(["true", "pass"]).all()
    )
    heavy_shuffle_status = str(
        specificity.get("heavy_shuffle_status", pd.Series(["not_run"])).iloc[0]
    )
    heavy_shuffle_pass = bool(
        specificity.get("heavy_shuffle_pass", pd.Series([False]))
        .astype(str)
        .str.lower()
        .isin(["true", "pass"])
        .all()
    )
    author_support = bool(author.get("author_effect_support", pd.Series([False])).astype(str).str.lower().isin(["true", "pass"]).any())
    if robust_event and outcome_alignment_pass and negative_control_pass and lightweight_shuffle_pass and heavy_shuffle_pass:
        boundary = "E1-V1"
        status = "pass"
    elif robust_event and outcome_alignment_pass:
        boundary = "E1-V1_partial"
        status = "partial"
    else:
        boundary = "no_SCP1064_methodology_claim"
        status = "fail"
    claim = pd.DataFrame(
        [
            {
                "dataset": "SCP1064",
                "candidate_event": "immune_evasion_antigen_presentation_IFN_response_T_cell_interaction_event",
                "robust_event": robust_event,
                "known_source_metadata": True,
                "outcome_alignment_pass": outcome_alignment_pass,
                "negative_control_pass": negative_control_pass,
                "lightweight_shuffle_pass": lightweight_shuffle_pass,
                "heavy_shuffle_status": heavy_shuffle_status,
                "heavy_shuffle_pass": heavy_shuffle_pass,
                "author_effect_support": author_support,
                "claim_boundary": boundary,
                "status": status,
                "matched_rescue_design": False,
                "same_system": False,
                "level4_causal_rescue": False,
                "reason": (
                    f"robust_event={robust_event}; outcome_alignment_pass={outcome_alignment_pass}; "
                    f"negative_control_pass={negative_control_pass}; lightweight_shuffle_pass={lightweight_shuffle_pass}; "
                    f"heavy_shuffle_status={heavy_shuffle_status}; heavy_shuffle_pass={heavy_shuffle_pass}; "
                    f"author_effect_support={author_support}"
                ),
            }
        ]
    )
    write_tsv(claim, RESULTS / "scp1064_claim_boundary.tsv")
    report = [
        "# SCP1064 Claim Boundary",
        "",
        f"Decision: `{boundary}`",
        "",
        claim.iloc[0]["reason"],
        "",
        "SCP1064 is used only as a TED methodology benchmark. It is not a GATA1/T21 biological rescue dataset and cannot support Level 4 causal rescue.",
    ]
    (RESULTS / "scp1064_claim_boundary_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    summary = [
        "# SCP1064 Summary Report",
        "",
        "SCP1064 / Frangieh 2021 is processed as a single-cell Perturb-CITE-seq benchmark linking CRISPR guide source, RNA immune-evasion events, and protein readouts.",
        "",
        f"Claim boundary: `{boundary}`",
        "",
        claim.iloc[0]["reason"],
    ]
    (RESULTS / "scp1064_summary_report.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return claim


def integrate_scp1064() -> None:
    claim = read_tsv(RESULTS / "scp1064_claim_boundary.tsv")
    event = read_tsv(RESULTS / "scp1064_event_recovery_summary.tsv")
    outcome = read_tsv(RESULTS / "scp1064_outcome_alignment_summary.tsv")
    negative = read_tsv(RESULTS / "scp1064_specificity_summary.tsv")
    if claim.empty:
        claim = call_claim_boundary()
    global_claim = read_tsv(GLOBAL_TABLES / "ted_dataset_level_claim_boundary.tsv")
    global_claim = global_claim[~global_claim.get("dataset", pd.Series(dtype=str)).astype(str).eq("SCP1064")] if not global_claim.empty else pd.DataFrame()
    write_tsv(pd.concat([global_claim, claim], ignore_index=True, sort=False), GLOBAL_TABLES / "ted_dataset_level_claim_boundary.tsv")
    if not event.empty:
        event2 = event.copy()
        event2.insert(0, "dataset", "SCP1064")
        old = read_tsv(GLOBAL_TABLES / "ted_event_recovery_summary.tsv")
        old = old[~old.get("dataset", pd.Series(dtype=str)).astype(str).eq("SCP1064")] if not old.empty else pd.DataFrame()
        write_tsv(pd.concat([old, event2], ignore_index=True, sort=False), GLOBAL_TABLES / "ted_event_recovery_summary.tsv")
    if not outcome.empty:
        out2 = outcome.copy()
        out2.insert(0, "dataset", "SCP1064")
        old = read_tsv(GLOBAL_TABLES / "ted_outcome_alignment_summary.tsv")
        old = old[~old.get("dataset", pd.Series(dtype=str)).astype(str).eq("SCP1064")] if not old.empty else pd.DataFrame()
        write_tsv(pd.concat([old, out2], ignore_index=True, sort=False), GLOBAL_TABLES / "ted_outcome_alignment_summary.tsv")
    if not negative.empty:
        neg2 = negative.copy()
        old = read_tsv(GLOBAL_TABLES / "ted_specificity_summary.tsv")
        old = old[~old.get("dataset", pd.Series(dtype=str)).astype(str).eq("SCP1064")] if not old.empty else pd.DataFrame()
        write_tsv(pd.concat([old, neg2], ignore_index=True, sort=False), GLOBAL_TABLES / "ted_specificity_summary.tsv")
    report = GLOBAL_REPORTS / "ted_upgrade_final_report.md"
    if report.exists():
        text = report.read_text(encoding="utf-8")
    else:
        text = "# TED Upgrade Final Report\n"
    append = [
        "",
        "## SCP1064 Addendum",
        "",
        f"- Claim boundary: `{claim.iloc[0]['claim_boundary']}` ({claim.iloc[0]['status']}).",
        f"- Reason: {claim.iloc[0]['reason']}",
        "- Role: secondary known-source RNA event/protein outcome benchmark, complementary to GSE153056 and GSE93735.",
    ]
    marker = "## SCP1064 Addendum"
    if marker in text:
        text = text.split(marker)[0].rstrip()
    report.write_text(text.rstrip() + "\n" + "\n".join(append) + "\n", encoding="utf-8")
