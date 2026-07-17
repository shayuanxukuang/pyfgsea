from __future__ import annotations

"""Build conservative E/V audit rows from immutable legacy analysis outputs.

The source files retain their historical column names.  These derived tables
make the inferential unit and current formal E/V boundary explicit without
rewriting the underlying analyses.
"""

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "bib_manuscript_revision"

GSE271_RAW = (
    ROOT
    / "data_external"
    / "deliverables_all_ted_rounds"
    / "GSE271399_T21_GATA1s"
    / "gse271399_family_block_permutation_fdr.tsv"
)
GSE271_ESTIMABILITY = (
    ROOT
    / "results"
    / "ted_submission_supplement"
    / "zscape_leave_one_embryo_full_refit"
    / "gse271399_estimability_audit.tsv"
)
GSE937_REVERSAL = (
    ROOT
    / "results"
    / "ted_known_source_validation"
    / "tables"
    / "gse93735_reversal_index.tsv"
)
GSE937_NEGATIVE = (
    ROOT
    / "results"
    / "ted_known_source_validation"
    / "tables"
    / "gse93735_negative_control_results.tsv"
)
def require(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source files: " + ", ".join(map(str, missing)))


def build_gse271() -> pd.DataFrame:
    raw = pd.read_csv(GSE271_RAW, sep="\t")
    estimability = pd.read_csv(GSE271_ESTIMABILITY, sep="\t")
    row = raw.loc[
        raw["family_id"].eq("ERYTHROID_EVENT_LOSS_FAMILY")
        & raw["trajectory"].eq("erythroid")
        & raw["contrast"].eq("T21_GATA1s_vs_T21_wtGATA1")
    ]
    if len(row) != 1:
        raise ValueError(f"Expected one primary GSE271399 row, found {len(row)}")
    row = row.iloc[0]
    donor = estimability.loc[estimability["requested_refit"].eq("leave_one_donor_out")]
    if len(donor) != 1 or donor.iloc[0]["status"] != "not_estimable":
        raise ValueError("GSE271399 donor estimability audit is not the expected fail-closed row")
    return pd.DataFrame(
        [
            {
                "dataset": "GSE271399",
                "event_family": row["family_id"],
                "trajectory": row["trajectory"],
                "contrast": row["contrast"],
                "design_stratum_definition": "day x pseudotime bin x coarse state",
                "n_design_strata": int(row["n_blocks"]),
                "independent_biological_unit": "not estimable from public metadata",
                "n_independent_donor_units": int(donor.iloc[0]["n_independent_units"]),
                "donor_leave_out_status": donor.iloc[0]["status"],
                "design_stratum_delta_auc": float(row["observed_family_delta_auc"]),
                "design_stratum_permutations": int(row["n_perm"]),
                "design_stratum_permutation_p": float(row["block_perm_p"]),
                "design_stratum_permutation_q": float(row["block_perm_q"]),
                "design_stratum_direction_stability": float(row["direction_stability"]),
                "event_test_status": "run_supported",
                "event_q_missing_reason": None,
                "event_support_code": "E1",
                "e0_reason_code": None,
                "validation_provenance_code": "V0",
                "evidence_boundary": "E1-V0",
                "e2_assigned": False,
                "boundary_reason": (
                    "design strata are not independent biological replicates; "
                    "donor leave-out is not estimable"
                ),
                "matched_same_system_rescue": False,
                "source_analysis": GSE271_RAW.relative_to(ROOT).as_posix(),
                "source_estimability": GSE271_ESTIMABILITY.relative_to(ROOT).as_posix(),
            }
        ]
    )


def build_gse937() -> pd.DataFrame:
    reversal = pd.read_csv(GSE937_REVERSAL, sep="\t")
    negative = pd.read_csv(GSE937_NEGATIVE, sep="\t")
    if reversal.empty or negative.empty:
        raise ValueError("GSE93735 reversal evidence tables must not be empty")
    primary = reversal.iloc[0]
    return pd.DataFrame(
        [
            {
                "dataset": "GSE93735",
                "event_family": "inflammatory reversal",
                "analysis_unit": "sample-level contrast; two samples per group",
                "event_test_status": "not_run",
                "event_q": None,
                "event_q_missing_reason": "insufficient_blocks",
                "event_support_code": "E0",
                "e0_reason_code": "E0_not_estimable",
                "validation_provenance_code": "V2",
                "evidence_boundary": "E0-V2",
                "primary_recovery_fraction": float(primary["reversal_fraction"]),
                "maximum_negative_control_recovery": float(
                    negative["reversal_fraction"].max()
                ),
                "supported_interpretation": (
                    "intervention-reversal readout is present, but primary event support "
                    "is not estimable under the available design"
                ),
                "unsupported_interpretation_current_evidence": (
                    "statistically supported primary event or matched GATA1/T21 rescue"
                ),
                "source_reversal": GSE937_REVERSAL.relative_to(ROOT).as_posix(),
                "source_negative_controls": GSE937_NEGATIVE.relative_to(ROOT).as_posix(),
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    require(
        [
            GSE271_RAW,
            GSE271_ESTIMABILITY,
            GSE937_REVERSAL,
            GSE937_NEGATIVE,
        ]
    )
    args.out.mkdir(parents=True, exist_ok=True)
    gse271 = args.out / "gse271399_design_stratum_audit.tsv"
    gse937 = args.out / "gse93735_ev_boundary.tsv"
    build_gse271().to_csv(gse271, sep="\t", index=False)
    build_gse937().to_csv(gse937, sep="\t", index=False)
    print(f"Wrote {gse271.relative_to(ROOT)}")
    print(f"Wrote {gse937.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
