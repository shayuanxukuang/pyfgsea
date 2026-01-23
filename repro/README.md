
# Paper Reproduction Scripts

This directory contains scripts to reproduce the figures and tables presented in the PyFgsea paper.

## Main Figures & Tables

| Script | Output | Description |
|--------|--------|-------------|
| `fig1_table1_performance.py` | `results/benchmark/performance_table.csv` | Reproduces Table 1 (Runtime & Peak Memory) benchmarks. |
| `fig_ablation_tail.py` | `results/ablation_tail/tail_summary.csv` | Reproduces Deep Tail Precision analysis. |
| `fig_stability.py` | `figures/fig_stability_boxplot.png` | Reproduces P-value Stability plots. |

## Supplementary Figures

| Script | Output | Description |
|--------|--------|-------------|
| `fig_supp_tail_consistency.py` | `figures/supp_tail_consistency.png` | **Figure Sx**: Deep tail consistency check vs R-fgsea inter-seed variance. |
| `fig_supp_bland_altman.py` | `supplementary_figures/Supp_Fig1_Bland_Altman.png` | Bland-Altman plot comparing PyFgsea and R-fgsea NES. |
| `fig_supp_thread_scaling.py` | `supplementary_figures/Supp_Fig3_Thread_Scaling.png` | Thread scaling efficiency and memory stability. |
| `fig_supp_myeloid_traj.py` | `supplementary_figures/Supp_Fig4a_Myeloid_Trajectory.png` | HSC->Myeloid trajectory validation (Paul15 dataset). |
| `fig_supp_window_sensitivity.py` | `supplementary_figures/Supp_Fig4b_Window_Sensitivity.png` | Sensitivity analysis of rolling window size. |
| `fig_supp_null_calibration.py` | `supplementary_figures/Supp_Fig5_Null_Calibration_Multi.png` | Null calibration QQ and ECDF plots. |

## Running Instructions

Ensure you have installed the package and dependencies (see root README).

You can use the `Makefile` in the root directory:

```bash
# Run all benchmarks
make benchmarks

# Run all main figures
make figures

# Run all supplementary figures
make supp
```

Or run individual scripts:

```bash
python repro/fig1_table1_performance.py
python repro/fig_supp_tail_consistency.py
```
