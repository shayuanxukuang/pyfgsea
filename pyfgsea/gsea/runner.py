from ..trajectory import run_trajectory_gsea
import pandas as pd
from .smooth import smooth_nes


def run_core(
    adata,
    gmt_path,
    out_csv=None,
    pseudotime_key="dpt_pseudotime",
    window_size=500,
    step=50,
    nperm=2000,
    smooth=True,
    min_size=15,
    max_size=500,
    sample_size=101,
    seed=42,
    eps=1e-50,
    bin_width=0,
    calculate_nes=True,
    use_nes_cache=False,
    mode="aligned",
    score_type="std",
    tie_policy="gene_id",
    nperm_simple=None,
    max_levels=None,
    precheck_n=None,
    precheck_eps=None,
):
    print(
        f"[Core] Running Trajectory GSEA (Window={window_size}, Step={step}, MinSize={min_size})..."
    )

    df = run_trajectory_gsea(
        adata,
        gmt_path=gmt_path,
        root_gene=None,
        window_size=window_size,
        step=step,
        out_csv=out_csv,
        nperm_nes=nperm,
        pseudotime_key=pseudotime_key,
        min_size=min_size,
        max_size=max_size,
        sample_size=sample_size,
        seed=seed,
        eps=eps,
        bin_width=bin_width,
        calculate_nes=calculate_nes,
        use_nes_cache=use_nes_cache,
        mode=mode,
        score_type=score_type,
        tie_policy=tie_policy,
        nperm_simple=nperm_simple,
        max_levels=max_levels,
        precheck_n=precheck_n,
        precheck_eps=precheck_eps,
    )

    if df is None or df.empty:
        print("  [Error] No results returned.")
        return pd.DataFrame()

    if smooth:
        print("  - Smoothing NES curves...")
        df = smooth_nes(df)

    return df
