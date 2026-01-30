from ..trajectory import run_trajectory_gsea
import pandas as pd
from .smooth import smooth_nes


def run_core(
    adata,
    gmt_path,
    out_csv=None,
    pseudotime_key="dpt_pseudotime",
    window_size=800,
    step=50,
    nperm=1000,
    smooth=True,
    min_size=15,
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
    )

    if df is None or df.empty:
        print("  [Error] No results returned.")
        return pd.DataFrame()

    if smooth:
        print("  - Smoothing NES curves...")
        df = smooth_nes(df)

    return df
