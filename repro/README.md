
# Paper Reproduction Scripts

This directory contains scripts to reproduce the figures and tables presented in the PyFgsea paper.

## 0.2.0 RC evidence status

The directory is a mixed reproduction toolkit, not a single aggregate receipt.
The fail-closed `figure1_dual_lane/` pipeline now defines the full dual-lane
Figure 1 agreement audit (ES, NES, p-value/deep-tail, overlaps, ties,
runtime/memory, and legacy/current changes) for 0.1.4/fgsea 1.32.2 and
0.2.0/fgsea 1.38.0. It has not yet been executed as a frozen release receipt.
`fig1_table1_performance.py` remains a performance benchmark only.

Evidence-producing scripts must exit nonzero on missing inputs and write a
JSON receipt containing the exact parameters, Git/package identity, and SHA-256
hashes of every declared input and output. The following scripts satisfy that
mechanical receipt gate, but their results still require scientific review:

- `figure1_dual_lane/prepare_inputs.py`, `run_lane.py`, and `adjudicate.py`
- `fig_ablation_tail.py` and `fig_supp_tail_consistency.py`
- `fig_supp_null_calibration.py` and `fig_stability.py`
- `benchmark_threads.py`, `fig_supp_thread_scaling.py`, and
  `fig1_table1_performance.py`
- `fig_supp_bland_altman.py` once a real, lane-bound Figure 1 agreement table
  is supplied

The following remain exploratory and must not be cited as frozen 0.2.0 RC
evidence:

- `fig_supp_myeloid_traj.py` and `fig_supp_window_sensitivity.py`: Paul15 is
  still obtained through Scanpy rather than a predeclared input artifact, and
  the scripts use `nperm_nes=100`, not the proposed Figure 2 contract.
- `ablation_root_cause.py` and `benchmark_calibration.py`: diagnostic utilities
  without complete artifact receipts.

Missing GMT files, missing R seed results, absent target pathways, and absent
window identifiers are errors. The scripts do not substitute zeros, unrelated
pathways, generated window identifiers, or a newly downloaded replacement GMT.

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

Scripts that compare against R require an explicit reference lane. Set
`FGSEA_REFERENCE_VERSION=1.32.2` for historical publication reproduction or
`FGSEA_REFERENCE_VERSION=1.38.0` for current conformance. The installed R,
Bioconductor, and fgsea versions are checked before results are written; a
missing or mismatched reference stops the run. Supplementary plots likewise
stop when their real benchmark inputs are absent.

All receipt-producing scripts also require an explicit Python lane:
`PYFGSEA_EXPECTED_VERSION=0.1.4` for the historical implementation or
`PYFGSEA_EXPECTED_VERSION=0.2.0` for the RC/current implementation. The loaded
module path, Rust extension path and SHA-256, package version, and (for 0.2.0)
algorithm revision are verified before the long analysis begins.

The root `Makefile` provides package checks but does not define aggregate paper
targets. Run the evidence-producing scripts explicitly so that failures are not
hidden:

```bash
python repro/fig1_table1_performance.py
python repro/fig_supp_tail_consistency.py
```
