# TED validation demo

The deterministic validation demo converts a block-aware activity table into
validated TSV and HTML event reports. It completes in seconds on a normal
workstation and uses controlled data only; it is not an external truth source.

## Run

```powershell
python scripts/run_ted_validation_demo.py --outdir results/ted_validation_demo
ted validate results/ted_validation_demo/demo_activity.tsv --kind activity --report results/ted_validation_demo/activity_cli_validation.tsv
ted validate results/ted_validation_demo/demo_events_v2.tsv --kind event --schema-version v2 --report results/ted_validation_demo/event_cli_validation.tsv
```

The fixed random seed is `20260715`. The demo contains six biological blocks
and four prespecified pathway behaviors. Cells are not used as independent
inferential replicates. Event schema v2 is auto-detected when
`--schema-version` is omitted because the table contains E/V fields.

## Outputs

| File | Purpose |
| --- | --- |
| `demo_activity.tsv` | Input activity table with dataset, block, trajectory, time, pathway, activity and weight fields. |
| `demo_events_v2.tsv` | Canonical v2 event report with event support (E), E0 reason code, validation provenance (V), evidence boundary and interpretation fields. |
| `demo_events.tsv` | Byte-equivalent compatibility alias. |
| `demo_validation.tsv` | Combined activity/event schema validation report. |
| `demo_report.html` | Human-readable event table for validation inspection. |
| `demo_manifest.json` | Seed, file roles, schema version and evidence-boundary statement. |

The demo retains deprecated `evidence_tier`, `claim_ceiling` and
`matched_functional_rescue` columns to exercise transition compatibility.
They do not determine a v2 evidence boundary.

## Fail-closed evidence gates

For v2, `ted validate` requires `evidence_boundary` to match
`event_support_code` and `validation_provenance_code`. Every E0 row must carry
one of the five stable `e0_reason_code` values; E1 and E2 rows leave it null.
E1 and E2 require a numeric event q value, E2 must be identifiable, and V3 is
rejected unless `matched_functional_rescue=true`. A null event q is accepted
only for `E0_not_estimable`; other E0 reasons require a numeric q. E0 is a
support/design outcome and does not mean that no event exists. V0 means
computational evidence only.
