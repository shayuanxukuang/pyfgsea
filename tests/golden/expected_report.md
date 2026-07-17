# Mechanism Claim Card

## 1. Candidate TED Event

erythroid_event_001

## 2. Leading Hypothesis

**H3_GATA1_regulatory GATA1 regulatory impairment** with posterior **0.944**.

## 3. Competing Hypotheses

| hypothesis                   | label                                    |   prior |   posterior |   log_likelihood_ratio_total |
|:-----------------------------|:-----------------------------------------|--------:|------------:|-----------------------------:|
| H5_T21_chromatin_interaction | T21-specific chromatin/GATA1 interaction |    0.1  |  0.0221301  |                     0.552942 |
| H4_downstream_heme           | downstream heme/maturation mechanism     |    0.15 |  0.0108918  |                    -0.561449 |
| H2_proliferation_confounded  | proliferation-confounded mechanism       |    0.15 |  0.00954781 |                    -0.693147 |
| H1_state_composition         | state/composition artifact               |    0.13 |  0.00689566 |                    -0.875466 |
| H0_noise_batch               | noise/batch artifact                     |    0.12 |  0.00624013 |                    -0.895315 |

## 4. Posterior Distribution

| hypothesis                   | label                                    |   prior |   posterior |   log_likelihood_ratio_total |
|:-----------------------------|:-----------------------------------------|--------:|------------:|-----------------------------:|
| H3_GATA1_regulatory          | GATA1 regulatory impairment              |    0.35 |  0.944295   |                     3.05368  |
| H5_T21_chromatin_interaction | T21-specific chromatin/GATA1 interaction |    0.1  |  0.0221301  |                     0.552942 |
| H4_downstream_heme           | downstream heme/maturation mechanism     |    0.15 |  0.0108918  |                    -0.561449 |
| H2_proliferation_confounded  | proliferation-confounded mechanism       |    0.15 |  0.00954781 |                    -0.693147 |
| H1_state_composition         | state/composition artifact               |    0.13 |  0.00689566 |                    -0.875466 |
| H0_noise_batch               | noise/batch artifact                     |    0.12 |  0.00624013 |                    -0.895315 |

## 5. Evidence-Family Contribution

Supporting leading hypothesis:

| evidence_family                  |   log_likelihood_ratio |   likelihood_ratio |
|:---------------------------------|-----------------------:|-------------------:|
| counterfactual_ot                |               0.635175 |            1.88735 |
| day_stratified_timing            |               0.618634 |            1.85639 |
| family_block_robustness          |               0.561449 |            1.75321 |
| external_GATA1_KD                |               0.485203 |            1.6245  |
| proliferation_adjusted_mediation |               0.452856 |            1.5728  |
| negative_mediator_controls       |               0.300364 |            1.35035 |

Evidence against leading hypothesis:

_None._

## 6. Evidence Dependency Warning

Evidence is aggregated by `evidence_family` and `dependency_group`; rows within a dependency group do not simply get multiplied together.

| evidence_family                  |   hypothesis |
|:---------------------------------|-------------:|
| counterfactual_ot                |            6 |
| day_stratified_timing            |            6 |
| external_GATA1_KD                |            6 |
| family_block_robustness          |            6 |
| negative_mediator_controls       |            6 |
| proliferation_adjusted_mediation |            6 |

_Naive fusion comparison was not requested for this run._

## 7. Current Claim Ceiling

**L3.5 mechanism-prioritized event**: computationally adjudicated, rescue-ready mechanism model.

## 8. Why The Claim Cannot Be Higher

The current data support a computationally adjudicated, rescue-ready mechanism model, but not a rescue-supported mechanism claim.

Missing evidence for next level: pre-registered matched rescue result.

## 9. Next Best Experiment

**A1_GATA1_FL_D7_rescue Full-length GATA1 rescue at D7**

Utility: 0.948; EIG: 0.081; expected claim delta: 0.472; falsification score: 1.000.

## 10. Minimal Readout Panel

Required:
- D9_regulatory_module
- D11_maturation_module
- D11_hemoglobinization
- TED_event_score

## 11. Expected Result Pattern

| hypothesis                   | readout                  | expected_pattern         |
|:-----------------------------|:-------------------------|:-------------------------|
| H3_GATA1_regulatory          | D9_regulatory_module     | strong_rescue            |
| H3_GATA1_regulatory          | D11_maturation_module    | partial_rescue           |
| H3_GATA1_regulatory          | D11_hemoglobinization    | partial_to_strong_rescue |
| H4_downstream_heme           | D9_regulatory_module     | weak_rescue              |
| H4_downstream_heme           | D11_hemoglobinization    | strong_rescue            |
| H5_T21_chromatin_interaction | D9_regulatory_module     | partial_rescue           |
| H5_T21_chromatin_interaction | chromatin_linked_targets | incomplete_rescue        |

## 12. Falsification Rule

### A1_GATA1_FL_D7_rescue Full-length GATA1 rescue at D7

```json
[
  {
    "hypothesis": "H3_GATA1_regulatory",
    "rule": "If full-length GATA1 is restored but D9 regulatory module and TED event score do not rescue, H3 should drop substantially."
  }
]
```

### A4_hemin_rescue Hemin rescue

```json
[
  {
    "hypothesis": "H4_downstream_heme",
    "rule": "If heme readouts do not improve despite adequate exposure, H4 loses support."
  }
]
```

## 13. Ambiguous-Result Handling

If the result pattern splits across competing hypotheses, do not upgrade the claim ceiling.
Update the posterior with `ted-mad update`, then choose the highest-ranked remaining contrast experiment:
- A1_GATA1_FL_D7_rescue Full-length GATA1 rescue at D7
- A4_hemin_rescue Hemin rescue

## 14. Sensitivity Summary

Posterior and claim robustness:

_No sensitivity analysis was requested for this run._

Active-design robustness:

_No cost/risk ranking sensitivity analysis was requested for this run._

## 15. Reviewer Defense Notes

- Competing mechanisms are explicit; the report does not only score the favored model.
- Evidence is fused at evidence-family/dependency-group level to reduce double counting.
- The claim ceiling is capped when direct rescue or orthogonal perturbation evidence is missing.
- The next experiment is selected for information gain, claim upgrade, falsification value, cost, and risk.
- Falsification rules are reported before rescue data are observed.

Provenance:

- TED-MAD version: `0.1.0`
- pyfgsea version: `0.1.4`
- git commit: `abc1234`
- run timestamp: `2026-05-13T00:00:00+00:00`
- random seed: `20260513`
- evidence: `tests/golden/example_evidence.yaml` sha256 `sha-e`
- experiments: `tests/golden/example_experiments.yaml` sha256 `sha-x`
- hypotheses: `tests/golden/example_hypotheses.yaml` sha256 `sha-h`
