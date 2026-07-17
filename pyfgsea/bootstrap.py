from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from .trajectory import run_trajectory_gsea
from .trajectory_events import summarize_events


_EVENT_INTERVAL_METRICS = (
    "activation_onset",
    "peak_time",
    "duration",
    "AUC",
)


def _bootstrap_cell_indices(n_obs: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n_obs, size=n_obs)


def _bootstrap_sample_indices(
    obs: pd.DataFrame,
    sample_key: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if sample_key not in obs:
        raise ValueError(f"sample_key '{sample_key}' not found in adata.obs")
    samples = pd.Series(obs[sample_key]).dropna().astype(str).unique()
    if len(samples) == 0:
        raise ValueError(f"sample_key '{sample_key}' has no non-null samples")

    sampled = rng.choice(samples, size=len(samples), replace=True)
    indices = []
    sample_values = pd.Series(obs[sample_key]).astype(str).to_numpy()
    for sample in sampled:
        indices.extend(np.where(sample_values == str(sample))[0].tolist())
    if not indices:
        raise ValueError("sample bootstrap produced no cells")
    return np.asarray(indices, dtype=int)


def _subset_with_unique_obs_names(adata, indices: np.ndarray, boot_id: int):
    # Sampling with replacement necessarily creates duplicate source names.
    # They are replaced immediately below, so suppress AnnData's transient
    # duplicate-name warning during materialization.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Observation names are not unique",
            category=UserWarning,
        )
        boot = adata[indices].copy()
    boot.obs_names = [
        f"{name}__boot{boot_id}_{idx}"
        for idx, name in enumerate(map(str, boot.obs_names))
    ]
    return boot


def _quantile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(np.quantile(values, q))


def _validate_pseudotime_draws(adata, pseudotime_draws_key: str) -> np.ndarray:
    if pseudotime_draws_key not in adata.obsm:
        raise ValueError(
            f"pseudotime_draws_key '{pseudotime_draws_key}' not found in adata.obsm"
        )

    raw_draws = adata.obsm[pseudotime_draws_key]
    if hasattr(raw_draws, "toarray"):
        raw_draws = raw_draws.toarray()
    try:
        draws = np.asarray(raw_draws, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"adata.obsm['{pseudotime_draws_key}'] must contain numeric pseudotime draws"
        ) from exc

    if draws.ndim != 2:
        raise ValueError(
            f"adata.obsm['{pseudotime_draws_key}'] must be a two-dimensional "
            "n_cells-by-n_draws matrix"
        )
    if draws.shape[0] != adata.n_obs:
        raise ValueError(
            f"adata.obsm['{pseudotime_draws_key}'] has {draws.shape[0]} rows; "
            f"expected adata.n_obs={adata.n_obs}"
        )
    if draws.shape[1] < 1:
        raise ValueError(
            f"adata.obsm['{pseudotime_draws_key}'] must contain at least one draw"
        )
    if not np.isfinite(draws).all():
        raise ValueError(
            f"adata.obsm['{pseudotime_draws_key}'] contains non-finite values"
        )
    return draws


def _event_detection_mask(group: pd.DataFrame) -> pd.Series:
    if "significant_window_count" in group.columns:
        count = pd.to_numeric(group["significant_window_count"], errors="coerce")
        return count.fillna(0).gt(0)
    if "event_window_count" in group.columns:
        count = pd.to_numeric(group["event_window_count"], errors="coerce")
        return count.fillna(0).gt(0)
    if "event_label" in group.columns:
        return ~group["event_label"].fillna("").astype(str).str.lower().eq("no clear event")
    return pd.Series(True, index=group.index, dtype=bool)


def _make_event_intervals(
    boot_events: pd.DataFrame,
    n_boot: int,
    lower_q: float,
    upper_q: float,
    pathway_col: str = "Pathway",
) -> pd.DataFrame:
    columns = [
        pathway_col,
        "n_boot_total",
        "n_boot_with_event_summary",
        "n_boot_detected",
        "detection_support",
        "ci_lower_quantile",
        "ci_upper_quantile",
    ]
    for metric in _EVENT_INTERVAL_METRICS:
        columns.extend(
            [
                f"{metric}_median",
                f"{metric}_lower",
                f"{metric}_upper",
            ]
        )

    if boot_events is None or boot_events.empty:
        return pd.DataFrame(columns=columns)
    required = {pathway_col, "boot_id"}
    missing = required - set(boot_events.columns)
    if missing:
        raise ValueError(f"boot_events is missing required columns: {sorted(missing)}")

    rows = []
    for pathway, group in boot_events.groupby(pathway_col, sort=False):
        group = group.copy()
        group["__detected"] = _event_detection_mask(group).to_numpy(dtype=bool)
        detected_boot_ids = group.loc[group["__detected"], "boot_id"].dropna().unique()
        detected = group[group["__detected"]].copy()
        row = {
            pathway_col: pathway,
            "n_boot_total": int(n_boot),
            "n_boot_with_event_summary": int(group["boot_id"].nunique()),
            "n_boot_detected": int(len(detected_boot_ids)),
            "detection_support": float(len(detected_boot_ids) / max(int(n_boot), 1)),
            "ci_lower_quantile": float(lower_q),
            "ci_upper_quantile": float(upper_q),
        }
        for metric in _EVENT_INTERVAL_METRICS:
            if metric not in detected.columns:
                row[f"{metric}_median"] = np.nan
                row[f"{metric}_lower"] = np.nan
                row[f"{metric}_upper"] = np.nan
                continue
            values = pd.to_numeric(detected[metric], errors="coerce")
            row[f"{metric}_median"] = _quantile(values, 0.5)
            row[f"{metric}_lower"] = _quantile(values, lower_q)
            row[f"{metric}_upper"] = _quantile(values, upper_q)
        rows.append(row)

    return pd.DataFrame(rows, columns=columns).sort_values(pathway_col).reset_index(drop=True)


def _make_bands(
    boot_results: pd.DataFrame,
    lower_q: float,
    upper_q: float,
    pathway_col: str = "Pathway",
) -> pd.DataFrame:
    if boot_results is None or boot_results.empty:
        return pd.DataFrame()
    required = {pathway_col, "window_id", "NES"}
    missing = required - set(boot_results.columns)
    if missing:
        raise ValueError(f"boot_results is missing required columns: {sorted(missing)}")

    rows = []
    for keys, group in boot_results.groupby([pathway_col, "window_id"], sort=False):
        pathway, window_id = keys
        row = {
            pathway_col: pathway,
            "window_id": window_id,
            "n_boot": int(group["boot_id"].nunique()) if "boot_id" in group else len(group),
            "NES_mean": float(pd.to_numeric(group["NES"], errors="coerce").mean()),
            "NES_lower": _quantile(group["NES"], lower_q),
            "NES_upper": _quantile(group["NES"], upper_q),
        }
        if "ES" in group:
            row["ES_mean"] = float(pd.to_numeric(group["ES"], errors="coerce").mean())
            row["ES_lower"] = _quantile(group["ES"], lower_q)
            row["ES_upper"] = _quantile(group["ES"], upper_q)
        if "padj" in group:
            row["padj_median"] = _quantile(group["padj"], 0.5)
        if "pt_mid" in group:
            row["pt_mid_median"] = _quantile(group["pt_mid"], 0.5)
            row["pt_mid_lower"] = _quantile(group["pt_mid"], lower_q)
            row["pt_mid_upper"] = _quantile(group["pt_mid"], upper_q)
        if "n_cells" in group:
            row["n_cells_median"] = _quantile(group["n_cells"], 0.5)
        rows.append(row)

    return pd.DataFrame(rows).sort_values([pathway_col, "window_id"]).reset_index(drop=True)


def bootstrap_trajectory_gsea(
    adata,
    gmt_path,
    pseudotime_key: str = "dpt_pseudotime",
    n_boot: int = 100,
    resample: str = "cells_within_windows",
    sample_key: Optional[str] = None,
    seed: int = 42,
    ci: tuple[float, float] = (0.025, 0.975),
    event_kwargs: Optional[dict] = None,
    pseudotime_draws_key: Optional[str] = None,
    **trajectory_kwargs,
) -> pd.DataFrame:
    """
    Bootstrap rolling-window trajectory GSEA and return NES confidence bands.

    ``resample="samples"`` resamples biological samples using ``sample_key``.
    Cell-level modes are useful for curve stability diagnostics, while
    sample-level resampling is preferred when biological replicates exist.

    When ``pseudotime_draws_key`` is provided, ``adata.obsm[key]`` must be a
    finite ``n_cells x n_draws`` matrix. One posterior draw is selected
    uniformly with replacement for every bootstrap replicate and is assigned
    to ``pseudotime_key`` before trajectory GSEA is rerun.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    if len(ci) != 2 or not (0 <= ci[0] < ci[1] <= 1):
        raise ValueError("ci must be a two-value tuple inside [0, 1]")
    if pseudotime_key not in adata.obs and pseudotime_draws_key is None:
        raise ValueError(f"pseudotime_key '{pseudotime_key}' not found in adata.obs")

    pseudotime_draws = (
        _validate_pseudotime_draws(adata, pseudotime_draws_key)
        if pseudotime_draws_key is not None
        else None
    )

    resample = resample.lower().replace("-", "_")
    if resample not in {"cells", "cells_within_windows", "samples"}:
        raise ValueError("resample must be 'cells', 'cells_within_windows', or 'samples'")
    if resample == "samples" and sample_key is None:
        raise ValueError("sample_key is required when resample='samples'")

    event_kwargs = {} if event_kwargs is None else dict(event_kwargs)
    rng = np.random.default_rng(seed)
    draw_rng = (
        np.random.default_rng(np.random.SeedSequence([int(seed), 1729]))
        if pseudotime_draws is not None
        else None
    )
    result_frames = []
    event_frames = []
    draw_records = []
    gene_set_index = trajectory_kwargs.pop("gene_set_index", None)
    for boot_id in range(n_boot):
        if resample == "samples":
            indices = _bootstrap_sample_indices(adata.obs, sample_key, rng)
        else:
            indices = _bootstrap_cell_indices(adata.n_obs, rng)

        boot = _subset_with_unique_obs_names(adata, indices, boot_id)
        draw_id = None
        if pseudotime_draws is not None:
            draw_id = int(draw_rng.integers(0, pseudotime_draws.shape[1]))
            boot.obs[pseudotime_key] = pseudotime_draws[indices, draw_id]
            draw_records.append({"boot_id": int(boot_id), "draw_id": draw_id})
        res = run_trajectory_gsea(
            boot,
            gmt_path=gmt_path,
            pseudotime_key=pseudotime_key,
            seed=seed + boot_id,
            gene_set_index=gene_set_index,
            **trajectory_kwargs,
        )
        if res is None or res.empty:
            continue
        gene_set_index = gene_set_index or res.attrs.get("gene_set_index")
        res = res.copy()
        res["boot_id"] = boot_id
        res["bootstrap_resample"] = resample
        if draw_id is not None:
            res["draw_id"] = draw_id
        result_frames.append(res)

        events = summarize_events(res, **event_kwargs)
        if not events.empty:
            events = events.copy()
            events["boot_id"] = boot_id
            events["bootstrap_resample"] = resample
            if draw_id is not None:
                events["draw_id"] = draw_id
            event_frames.append(events)

    boot_results = (
        pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    )
    boot_events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    bands = _make_bands(boot_results, ci[0], ci[1])
    event_intervals = _make_event_intervals(
        boot_events,
        n_boot=n_boot,
        lower_q=ci[0],
        upper_q=ci[1],
    )
    bands.attrs["boot_results"] = boot_results
    bands.attrs["boot_events"] = boot_events
    bands.attrs["event_intervals"] = event_intervals
    bands.attrs["bootstrap_draws"] = pd.DataFrame(
        draw_records,
        columns=["boot_id", "draw_id"],
    )
    bands.attrs["bootstrap"] = {
        "n_boot": int(n_boot),
        "resample": resample,
        "sample_key": sample_key,
        "seed": int(seed),
        "ci": tuple(ci),
        "pseudotime_source": (
            "posterior_draws" if pseudotime_draws is not None else "obs"
        ),
        "pseudotime_draws_key": pseudotime_draws_key,
        "n_pseudotime_draws": (
            int(pseudotime_draws.shape[1]) if pseudotime_draws is not None else 0
        ),
        "pseudotime_draw_sampling": (
            "uniform_with_replacement" if pseudotime_draws is not None else "none"
        ),
        "event_interval_metrics": tuple(_EVENT_INTERVAL_METRICS),
        "event_detection_definition": "significant_window_count > 0",
    }
    return bands
