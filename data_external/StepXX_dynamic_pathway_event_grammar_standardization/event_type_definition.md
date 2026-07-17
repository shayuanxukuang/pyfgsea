# Dynamic pathway event type definitions

This file defines the standardized event grammar used by `dynamic_pathway_event_table.tsv`.
The goal is to convert pathway or module activity curves into auditable pathway-event rows.

## Core event types

| event_type | Definition | Required evidence | Unsupported escalation |
|---|---|---|---|
| onset | A pathway begins activation or suppression at an ordered time, pseudotime, branch or spatial window. | ordered coordinate, effect direction, event_FDR or window-level support | causal initiation mechanism |
| peak | A pathway reaches maximal activity in a trajectory, time, branch or spatial window. | peak coordinate, effect size, comparator or null support | mechanism inferred from a curve maximum alone |
| shutdown | A pathway enters suppression, trough or persistent loss after a prior active or expected state. | suppression/trough coordinate or event-loss family, robustness or contrast support | irreversible biological termination without validation |
| branch-specific | A pathway event differs between branches, fates, genotypes or lineages. | branch labels or fate scaffold, branch-specific effect, null or expected-program support | fate-determining causal mechanism |
| spatial-localized | A pathway event is localized to a spatial region, tissue section or spatial trajectory window. | spatial/bin/region context, dynamic signal, spatial/null support | direct spatial causal mechanism from enrichment alone |

## Extension event types

| event_type | Definition | Boundary |
|---|---|---|
| rescued-extension | A perturbed event reverses in a matched rescue or intervention design. | Requires matched rescue/intervention contrast and negative controls. |
| failed-rescue-extension | TED predicts rescue axes or observes incomplete reversal, but matched rescue validation is absent or fails. | Report as prediction or failed public gate, not completed functional validation. |

## Required row fields

Each standardized event row includes `event_FDR` when a numeric event-level FDR is available,
`event_FDR_available`, `event_FDR_reason_if_missing`, `robustness_score`,
`claim_boundary`, `supported_interpretation` and `unsupported_interpretation`.
Prediction-only rescue extensions additionally report `prediction_robustness_score`,
`validation_robustness_score` and `validation_status`, so rescue predictions are not
mistaken for matched functional validation.
