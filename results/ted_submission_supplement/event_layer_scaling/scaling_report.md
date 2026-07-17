## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-07-15
- Verification Status: ANALYZED
- Version Label: ted_event_layer_scaling_v1

# TED runtime and memory scaling

Upstream module scoring and the downstream TED event layer are timed separately.
The one-million-cell cases use predeclared synthetic modules, biological-block aggregation, window summaries, and block-label permutation; single cells are not treated as independent inferential units.
This benchmark does not equate the Rust PyFgsea core speed with end-to-end TED speed.
Parallel scaling is reported separately using benchmark_ted_performance.py because this synthetic event-layer kernel is deliberately single-worker.

```text
profile   cells  repeats  median_upstream_seconds  iqr_low_upstream_seconds  iqr_high_upstream_seconds  median_event_layer_seconds  iqr_low_event_layer_seconds  iqr_high_event_layer_seconds  median_total_seconds  iqr_low_total_seconds  iqr_high_total_seconds  median_peak_rss_mb  iqr_low_peak_rss_mb  iqr_high_peak_rss_mb
   full   10000        5                 0.036437                  0.036331                   0.037143                    0.009191                     0.009071                      0.009419              0.045579               0.045570                0.046632           84.093750            84.093750             84.093750
   full   50000        5                 0.187956                  0.183820                   0.188831                    0.026010                     0.025427                      0.027277              0.214330               0.211841                0.215295           94.812500            94.804688             95.136719
   full  100000        5                 0.389396                  0.389146                   0.393799                    0.051685                     0.050312                      0.080853              0.445548               0.445394                0.470313          108.449219           107.316406            108.449219
   full  500000        5                 1.988842                  1.977063                   2.000001                    0.416058                     0.414509                      0.417487              2.406397               2.403379                2.416123          210.257812           210.257812            210.257812
   full 1000000        5                 3.935074                  3.919823                   3.981162                    0.869713                     0.813655                      0.883799              4.804853               4.803688                4.890303          339.175781           339.003906            339.175781
```
