#!/usr/bin/env python3
"""Compare the two Figure 1 reference runs and write the result tables."""

from __future__ import annotations

import argparse
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

try:
    from .common import (
        GSEA_PARAMETERS,
        LANE_CONTRACTS,
        LOG10_FLOOR,
        SCENARIOS,
        SUITE_VERSION,
        assert_finite_range,
        ensure_empty_output_dir,
        file_record,
        read_json,
        require_sha256,
        sha256_file,
        verify_clean_git_checkout,
        verify_file_record,
        verify_git_unchanged,
        write_json,
    )
except ImportError:  # pragma: no cover - direct script execution
    from common import (  # type: ignore
        GSEA_PARAMETERS,
        LANE_CONTRACTS,
        LOG10_FLOOR,
        SCENARIOS,
        SUITE_VERSION,
        assert_finite_range,
        ensure_empty_output_dir,
        file_record,
        read_json,
        require_sha256,
        sha256_file,
        verify_clean_git_checkout,
        verify_file_record,
        verify_git_unchanged,
        write_json,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
RAW_REQUIRED_COLUMNS = {
    "lane",
    "scenario",
    "pathway",
    "py_es",
    "r_es",
    "es_difference",
    "py_nes",
    "r_nes",
    "nes_difference",
    "py_pval",
    "r_pval",
    "py_padj",
    "r_padj",
    "py_neg_log10_pval",
    "r_neg_log10_pval",
    "neg_log10_pval_difference",
    "input_tied_score_group_count",
    "input_tied_gene_count",
    "input_maximum_tie_multiplicity",
}


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2:
        raise ValueError("correlation requires at least two pathway rows")
    if np.std(left) == 0 or np.std(right) == 0:
        raise ValueError("correlation is undefined for a constant metric")
    value = float(np.corrcoef(left, right)[0, 1])
    if not math.isfinite(value):
        raise ValueError("correlation is not finite")
    return value


def concordance_metrics(left: pd.Series, right: pd.Series) -> dict[str, float | int]:
    """Compute the complete agreement vector without any override hook."""

    x = pd.to_numeric(left, errors="raise").to_numpy(dtype=float)
    y = pd.to_numeric(right, errors="raise").to_numpy(dtype=float)
    assert_finite_range(x, label="metric left")
    assert_finite_range(y, label="metric right")
    difference = x - y
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return {
        "n": int(len(x)),
        "pearson": _correlation(x, y),
        "spearman": _correlation(x_rank, y_rank),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "median_absolute_difference": float(np.median(np.abs(difference))),
        "p95_absolute_difference": float(np.percentile(np.abs(difference), 95)),
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
        "mean_signed_difference": float(np.mean(difference)),
    }


def _validate_receipt(
    receipt_path: Path, lane: str
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, Path, Path]:
    receipt = read_json(receipt_path)
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind") != "figure1_lane_receipt"
    ):
        raise ValueError(f"{lane} input is not a Figure 1 lane receipt")
    if receipt.get("suite_version") != SUITE_VERSION or receipt.get("lane") != lane:
        raise ValueError(f"{lane} receipt has the wrong suite or lane")
    if receipt.get("lane_contract") != LANE_CONTRACTS[lane]:
        raise ValueError(f"{lane} run settings differ from the comparison settings")
    if receipt.get("gsea_parameters") != GSEA_PARAMETERS:
        raise ValueError(f"{lane} receipt uses a different GSEA parameter contract")
    git = receipt.get("git")
    if not isinstance(git, Mapping) or git.get("clean") is not True:
        raise ValueError(f"{lane} receipt was not captured from a clean Git tree")
    policy = receipt.get("metric_policy")
    if not isinstance(policy, Mapping):
        raise ValueError(f"{lane} receipt lacks a metric policy")
    if policy.get("pathway_level_raw_is_only_metric_source") is not True:
        raise ValueError(f"{lane} receipt does not designate raw rows as authoritative")
    if policy.get("manual_metric_overrides_permitted") is not False:
        raise ValueError(f"{lane} receipt permits manual metric overrides")
    if policy.get("pvalue_log_transform") != "-log10(max(pvalue, 1e-300))":
        raise ValueError(f"{lane} receipt uses a different p-value transform")

    identity = receipt.get("package_identity")
    if not isinstance(identity, Mapping):
        raise ValueError(f"{lane} receipt lacks installed package identity")
    contract = LANE_CONTRACTS[lane]
    identity_expectations = {
        "lane": lane,
        "distribution_version": contract["pyfgsea_distribution_version"],
        "module_declared_version": contract["pyfgsea_module_version"],
        "algorithm_revision": contract["algorithm_revision"],
        "algorithm_revision_contract": (
            "legacy-no-revision-api"
            if lane == "legacy"
            else contract["algorithm_revision"]
        ),
    }
    for key, expected in identity_expectations.items():
        if identity.get(key) != expected:
            raise ValueError(f"{lane} installed identity mismatch for {key}")
    require_sha256(str(identity.get("wheel_sha256", "")), label=f"{lane} wheel hash")
    require_sha256(str(identity.get("core_sha256", "")), label=f"{lane} core hash")

    outputs = receipt.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError(f"{lane} receipt lacks output hashes")
    expected_output_labels = {
        "artifact_receipt",
        "installed_identity",
        "reference_oci_receipt",
        "pathway_raw",
        "runtime_memory",
    }
    for scenario in SCENARIOS:
        expected_output_labels.update(
            {
                f"pyfgsea_{scenario}",
                f"r_fgsea_{scenario}",
                f"r_environment_{scenario}",
                f"r_session_{scenario}",
                f"r_stdout_{scenario}",
                f"r_stderr_{scenario}",
            }
        )
    if set(outputs) != expected_output_labels:
        raise ValueError(f"{lane} receipt output label set is incomplete or unexpected")
    for label, record in outputs.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"{lane}/{label} output record is invalid")
        verify_file_record(receipt_path.parent, record, label=f"{lane}/{label}")
    identity_record = outputs.get("installed_identity")
    if not isinstance(identity_record, Mapping):
        raise ValueError(f"{lane} receipt lacks installed_identity output")
    identity_path = verify_file_record(
        receipt_path.parent, identity_record, label=f"{lane} installed identity"
    )
    if read_json(identity_path) != dict(identity):
        raise ValueError(f"{lane} installed identity sidecar differs from receipt")
    artifact_binding = identity.get("artifact_binding")
    reference_binding = identity.get("reference_binding")
    if not isinstance(artifact_binding, Mapping) or not isinstance(
        reference_binding, Mapping
    ):
        raise ValueError(f"{lane} run record lacks package or reference records")
    artifact_copy = verify_file_record(
        receipt_path.parent,
        outputs["artifact_receipt"],
        label=f"{lane} artifact receipt copy",
    )
    reference_copy = verify_file_record(
        receipt_path.parent,
        outputs["reference_oci_receipt"],
        label=f"{lane} reference receipt copy",
    )
    if sha256_file(artifact_copy) != artifact_binding.get(
        "receipt_sha256"
    ) or sha256_file(reference_copy) != reference_binding.get("receipt_sha256"):
        raise ValueError(
            f"{lane} package or reference records do not match the run identity"
        )
    for scenario in SCENARIOS:
        environment_record = outputs.get(f"r_environment_{scenario}")
        session_record = outputs.get(f"r_session_{scenario}")
        if not isinstance(environment_record, Mapping) or not isinstance(
            session_record, Mapping
        ):
            raise ValueError(f"{lane}/{scenario} lacks R environment evidence")
        environment_path = verify_file_record(
            receipt_path.parent,
            environment_record,
            label=f"{lane}/{scenario} R environment",
        )
        session_path = verify_file_record(
            receipt_path.parent,
            session_record,
            label=f"{lane}/{scenario} R sessionInfo",
        )
        if not session_path.read_text(encoding="utf-8").strip():
            raise ValueError(f"{lane}/{scenario} R sessionInfo is empty")
        environment = pd.read_csv(environment_path, sep="\t", dtype=str)
        if list(environment.columns) != ["key", "value"]:
            raise ValueError(f"{lane}/{scenario} R environment table is invalid")
        details = dict(zip(environment["key"], environment["value"]))
        for key, expected in {
            "r_version": contract["r_version"],
            "bioconductor_version": contract["bioconductor_version"],
            "fgsea_version": contract["fgsea_version"],
            "r_seed": str(GSEA_PARAMETERS["r_seed"]),
            "score_type": "std",
        }.items():
            if details.get(key) != expected:
                raise ValueError(f"{lane}/{scenario} R environment mismatch for {key}")
    raw_record = outputs.get("pathway_raw")
    timing_record = outputs.get("runtime_memory")
    if not isinstance(raw_record, Mapping) or not isinstance(timing_record, Mapping):
        raise ValueError(f"{lane} receipt lacks raw pathway or timing output")
    raw_path = verify_file_record(receipt_path.parent, raw_record, label=f"{lane} raw")
    timing_path = verify_file_record(
        receipt_path.parent, timing_record, label=f"{lane} timing"
    )
    raw = pd.read_csv(raw_path, sep="\t")
    missing = sorted(RAW_REQUIRED_COLUMNS.difference(raw.columns))
    if missing:
        raise ValueError(f"{lane} raw table is missing columns: {missing}")
    if set(raw["lane"].astype(str)) != {lane}:
        raise ValueError(f"{lane} raw table contains a different lane label")
    if set(raw["scenario"].astype(str)) != set(SCENARIOS):
        raise ValueError(f"{lane} raw table does not contain exactly {SCENARIOS}")
    expected_counts = {"publication_main": 100, "ties_predeclared": 60}
    if raw.groupby("scenario").size().to_dict() != expected_counts:
        raise ValueError(f"{lane} raw table has the wrong per-scenario row counts")
    if raw.duplicated(["scenario", "pathway"]).any():
        raise ValueError(f"{lane} raw table contains duplicate pathway rows")
    for prefix in ("py", "r"):
        for metric in ("es", "nes", "pval", "padj", "neg_log10_pval"):
            values = raw[f"{prefix}_{metric}"]
            lower, upper = (0.0, 1.0) if metric in {"pval", "padj"} else (None, None)
            assert_finite_range(
                values, label=f"{lane} {prefix}_{metric}", lower=lower, upper=upper
            )
    recomputed = {
        "es_difference": raw["py_es"] - raw["r_es"],
        "nes_difference": raw["py_nes"] - raw["r_nes"],
        "neg_log10_pval_difference": (
            raw["py_neg_log10_pval"] - raw["r_neg_log10_pval"]
        ),
    }
    for prefix in ("py", "r"):
        expected_log = -np.log10(
            np.maximum(raw[f"{prefix}_pval"].to_numpy(dtype=float), LOG10_FLOOR)
        )
        if not np.allclose(
            raw[f"{prefix}_neg_log10_pval"].to_numpy(dtype=float),
            expected_log,
            rtol=1e-14,
            atol=1e-14,
        ):
            raise ValueError(
                f"{lane} stored {prefix} -log10P is not derived from p-value"
            )
    for column, expected in recomputed.items():
        if not np.allclose(
            raw[column].to_numpy(dtype=float),
            expected.to_numpy(dtype=float),
            rtol=1e-14,
            atol=1e-15,
        ):
            raise ValueError(f"{lane} raw {column} is inconsistent with engine columns")

    timing = pd.read_csv(timing_path, sep="\t")
    required_timing = {
        "lane",
        "scenario",
        "engine",
        "measurement_scope",
        "engine_elapsed_seconds",
        "elapsed_seconds",
        "peak_rss_bytes",
        "peak_increment_bytes",
    }
    if not required_timing.issubset(timing.columns):
        raise ValueError(f"{lane} timing table lacks required columns")
    if len(timing) != len(SCENARIOS) * 2:
        raise ValueError(f"{lane} timing table has the wrong number of rows")
    if timing.duplicated(["scenario", "engine"]).any():
        raise ValueError(f"{lane} timing table contains duplicates")
    expected_lattice = {
        (scenario, engine)
        for scenario in SCENARIOS
        for engine in ("pyfgsea", "r_fgsea")
    }
    actual_lattice = set(zip(timing["scenario"], timing["engine"]))
    if actual_lattice != expected_lattice or set(timing["lane"]) != {lane}:
        raise ValueError(
            f"{lane} timing table has the wrong lane/scenario/engine lattice"
        )
    expected_scopes = {
        "pyfgsea": "run_gsea_call_only",
        "r_fgsea": "Rscript_process_and_internal_fgsea",
    }
    if any(
        row.measurement_scope != expected_scopes[row.engine]
        for row in timing.itertuples(index=False)
    ):
        raise ValueError(f"{lane} timing measurement scopes differ from the contract")
    assert_finite_range(timing["elapsed_seconds"], label=f"{lane} elapsed", lower=0.0)
    assert_finite_range(
        timing["engine_elapsed_seconds"], label=f"{lane} engine elapsed", lower=0.0
    )
    assert_finite_range(timing["peak_rss_bytes"], label=f"{lane} peak RSS", lower=0.0)

    call_contract = receipt.get("python_call_arguments")
    if not isinstance(call_contract, Mapping):
        raise ValueError(f"{lane} receipt lacks the Python call contract")
    passed = call_contract.get("passed_arguments")
    effective = call_contract.get("effective_arguments")
    if not isinstance(passed, Mapping) or not isinstance(effective, Mapping):
        raise ValueError(f"{lane} Python call contract is malformed")
    for argument, expected in {
        "min_size": 15,
        "max_size": 500,
        "sample_size": 101,
        "seed": 1,
        "nperm_nes": 1800,
        "eps": 1e-50,
    }.items():
        if passed.get(argument) != expected or effective.get(argument) != expected:
            raise ValueError(f"{lane} Python call differs for {argument}")
    expected_score_type = "two_sided_abs" if lane == "legacy" else "std"
    if effective.get("score_type") != expected_score_type:
        raise ValueError(f"{lane} effective Python score_type differs")
    if lane == "current" and any(
        passed.get(key) != value
        for key, value in {
            "mode": "aligned",
            "tie_policy": "gene_id",
            "bin_width": 0,
            "nperm_simple": 1000,
        }.items()
    ):
        raise ValueError("current Python aligned-mode contract differs")
    r_commands = receipt.get("r_commands")
    if not isinstance(r_commands, list) or len(r_commands) != len(SCENARIOS):
        raise ValueError(f"{lane} receipt lacks two R commands")
    for scenario, command in zip(SCENARIOS, r_commands):
        if not isinstance(command, list) or len(command) != 15:
            raise ValueError(f"{lane}/{scenario} R command is malformed")
        expected_tail = [
            str(LANE_CONTRACTS[lane]["fgsea_version"]),
            str(LANE_CONTRACTS[lane]["r_version"]),
            str(LANE_CONTRACTS[lane]["bioconductor_version"]),
            "314",
            "15",
            "500",
            "101",
        ]
        if command[1] != "--vanilla" or command[8:] != expected_tail:
            raise ValueError(f"{lane}/{scenario} R command contract differs")
        if scenario not in command[3] or scenario not in command[4]:
            raise ValueError(f"{lane}/{scenario} R command used the wrong inputs")
    return receipt, raw, timing, raw_path, timing_path


def _same_input_contract(legacy: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    legacy_inputs = legacy.get("inputs")
    current_inputs = current.get("inputs")
    if not isinstance(legacy_inputs, Mapping) or not isinstance(
        current_inputs, Mapping
    ):
        raise ValueError("lane receipt is missing inputs")
    if legacy_inputs.get("manifest", {}).get("sha256") != current_inputs.get(
        "manifest", {}
    ).get("sha256"):
        raise ValueError("legacy/current lanes used different input manifests")
    for scenario in SCENARIOS:
        for label in ("ranks", "pathways"):
            legacy_hash = legacy_inputs["scenarios"][scenario][label]["sha256"]
            current_hash = current_inputs["scenarios"][scenario][label]["sha256"]
            if legacy_hash != current_hash:
                raise ValueError(f"legacy/current {scenario}/{label} hashes differ")
    if legacy.get("gsea_parameters") != current.get("gsea_parameters"):
        raise ValueError("legacy/current fixed GSEA parameter contracts differ")
    legacy_python = legacy.get("python_environment")
    current_python = current.get("python_environment")
    if not isinstance(legacy_python, Mapping) or not isinstance(
        current_python, Mapping
    ):
        raise ValueError("lane receipt is missing its Python environment")
    for key in (
        "version",
        "system",
        "machine",
        "implementation",
        "numpy",
        "pandas",
        "psutil",
        "thread_environment",
    ):
        if legacy_python.get(key) != current_python.get(key):
            raise ValueError(
                f"legacy/current Python analysis environments differ for {key}"
            )
    for script_name in ("run_lane", "common", "prepare_inputs", "r_helper"):
        legacy_hash = legacy["scripts"][script_name]["sha256"]
        current_hash = current["scripts"][script_name]["sha256"]
        if legacy_hash != current_hash:
            raise ValueError(f"lane scripts differ for {script_name}")


def _verify_common_input_bundle(
    manifest_path: Path,
    legacy_receipt: Mapping[str, Any],
    current_receipt: Mapping[str, Any],
) -> None:
    manifest_hash = sha256_file(manifest_path)
    for lane, receipt in (("legacy", legacy_receipt), ("current", current_receipt)):
        recorded = receipt["inputs"]["manifest"].get("sha256")
        if recorded != manifest_hash:
            raise ValueError(
                f"{lane} run record does not match the supplied input manifest"
            )
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema_version") != 2
        or manifest.get("kind") != "figure1_input_manifest"
    ):
        raise ValueError("supplied input manifest has the wrong schema or kind")
    if manifest.get("suite_version") != SUITE_VERSION:
        raise ValueError("supplied input manifest belongs to a different suite")
    generator = manifest.get("generator")
    if not isinstance(generator, Mapping) or generator.get(
        "script_sha256"
    ) != sha256_file(SCRIPT_DIR / "prepare_inputs.py"):
        raise ValueError("supplied input manifest used a different generator script")
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, Mapping) or set(scenarios) != set(SCENARIOS):
        raise ValueError(f"supplied input manifest must contain exactly {SCENARIOS}")
    for scenario in SCENARIOS:
        record = scenarios[scenario]
        if not isinstance(record, Mapping):
            raise ValueError(f"input scenario is invalid: {scenario}")
        for label in ("ranks", "pathways"):
            file_hash = sha256_file(
                verify_file_record(
                    manifest_path.parent,
                    record[label],
                    label=f"input bundle {scenario}/{label}",
                )
            )
            for lane, receipt in (
                ("legacy", legacy_receipt),
                ("current", current_receipt),
            ):
                recorded = receipt["inputs"]["scenarios"][scenario][label]["sha256"]
                if recorded != file_hash:
                    raise ValueError(
                        f"{lane} run record does not match input {scenario}/{label}"
                    )


def _metric_rows(raw: pd.DataFrame, raw_hash: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pairs = {
        "ES": ("py_es", "r_es"),
        "NES": ("py_nes", "r_nes"),
        "pvalue": ("py_pval", "r_pval"),
        "neg_log10_pvalue": ("py_neg_log10_pval", "r_neg_log10_pval"),
    }
    for (lane, scenario), group in raw.groupby(["lane", "scenario"], sort=True):
        for metric, (left, right) in pairs.items():
            rows.append(
                {
                    "comparison": "pyfgsea_vs_r_fgsea",
                    "lane": lane,
                    "scenario": scenario,
                    "metric": metric,
                    "left_column": left,
                    "right_column": right,
                    "raw_source_sha256": raw_hash,
                    **concordance_metrics(group[left], group[right]),
                }
            )
    return pd.DataFrame(rows)


def _overlap_rows(raw: pd.DataFrame, raw_hash: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (lane, scenario), group in raw.groupby(["lane", "scenario"], sort=True):
        py_ranked = group.assign(abs_nes=group["py_nes"].abs()).sort_values(
            ["abs_nes", "pathway"],
            ascending=[False, True],
            kind="mergesort",
        )
        r_ranked = group.assign(abs_nes=group["r_nes"].abs()).sort_values(
            ["abs_nes", "pathway"],
            ascending=[False, True],
            kind="mergesort",
        )
        top_count = min(10, len(group))
        py_top = set(py_ranked.head(top_count)["pathway"])
        r_top = set(r_ranked.head(top_count)["pathway"])
        top_intersection = py_top & r_top
        top_union = py_top | r_top
        rows.append(
            {
                "lane": lane,
                "scenario": scenario,
                "overlap_type": "top10_by_absolute_NES",
                "threshold": top_count,
                "py_count": len(py_top),
                "r_count": len(r_top),
                "intersection_count": len(top_intersection),
                "union_count": len(top_union),
                "overlap_fraction": len(top_intersection) / top_count,
                "jaccard": len(top_intersection) / len(top_union),
                "empty_union": False,
                "raw_source_sha256": raw_hash,
            }
        )
        py_fdr = set(group.loc[group["py_padj"] < 0.05, "pathway"])
        r_fdr = set(group.loc[group["r_padj"] < 0.05, "pathway"])
        fdr_intersection = py_fdr & r_fdr
        fdr_union = py_fdr | r_fdr
        empty_union = not fdr_union
        rows.append(
            {
                "lane": lane,
                "scenario": scenario,
                "overlap_type": "FDR_pathway_set",
                "threshold": 0.05,
                "py_count": len(py_fdr),
                "r_count": len(r_fdr),
                "intersection_count": len(fdr_intersection),
                "union_count": len(fdr_union),
                "overlap_fraction": (
                    len(fdr_intersection) / max(len(py_fdr), len(r_fdr))
                    if max(len(py_fdr), len(r_fdr))
                    else 1.0
                ),
                "jaccard": len(fdr_intersection) / len(fdr_union) if fdr_union else 1.0,
                "empty_union": empty_union,
                "raw_source_sha256": raw_hash,
            }
        )
    return pd.DataFrame(rows)


def _extreme_rows(raw: pd.DataFrame, raw_hash: str, count: int) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for (_, _), group in raw.groupby(["lane", "scenario"], sort=True):
        ranked = group.assign(
            tail_depth=group[["py_neg_log10_pval", "r_neg_log10_pval"]].max(axis=1),
            absolute_tail_difference=group["neg_log10_pval_difference"].abs(),
        ).sort_values(
            ["tail_depth", "absolute_tail_difference", "pathway"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        chosen = ranked.head(min(count, len(ranked))).copy()
        chosen["selection_rank"] = np.arange(1, len(chosen) + 1)
        chosen["selection_rule"] = (
            "descending max(Py -log10P,R -log10P), then absolute difference, then pathway"
        )
        chosen["raw_source_sha256"] = raw_hash
        selected.append(chosen)
    columns = [
        "lane",
        "scenario",
        "selection_rank",
        "selection_rule",
        "pathway",
        "py_es",
        "r_es",
        "es_difference",
        "py_nes",
        "r_nes",
        "nes_difference",
        "py_pval",
        "r_pval",
        "py_neg_log10_pval",
        "r_neg_log10_pval",
        "neg_log10_pval_difference",
        "tail_depth",
        "absolute_tail_difference",
        "raw_source_sha256",
    ]
    return pd.concat(selected, ignore_index=True)[columns]


def _cross_lane_rows(raw: pd.DataFrame, combined_raw_hash: str) -> pd.DataFrame:
    legacy = raw.loc[raw["lane"] == "legacy"].drop(columns="lane")
    current = raw.loc[raw["lane"] == "current"].drop(columns="lane")
    paired = legacy.merge(
        current,
        on=["scenario", "pathway"],
        suffixes=("_legacy", "_current"),
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != len(legacy) or len(paired) != len(current):
        raise ValueError("legacy/current pathway rows do not align one-to-one")
    rows: list[dict[str, Any]] = []
    for scenario, group in paired.groupby("scenario", sort=True):
        for engine in ("py", "r"):
            for metric in ("es", "nes", "pval", "neg_log10_pval"):
                legacy_column = f"{engine}_{metric}_legacy"
                current_column = f"{engine}_{metric}_current"
                rows.append(
                    {
                        "comparison": "current_vs_legacy",
                        "engine": "pyfgsea" if engine == "py" else "r_fgsea",
                        "scenario": scenario,
                        "metric": metric,
                        "left": "current",
                        "right": "legacy",
                        "raw_source_sha256": combined_raw_hash,
                        **concordance_metrics(
                            group[current_column], group[legacy_column]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _draw_figure(raw: pd.DataFrame, metrics: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for axis, lane, scenario in zip(
        axes.flat,
        ("legacy", "current", "legacy", "current"),
        (
            "publication_main",
            "publication_main",
            "ties_predeclared",
            "ties_predeclared",
        ),
    ):
        group = raw.loc[(raw["lane"] == lane) & (raw["scenario"] == scenario)]
        x = group["py_nes"].to_numpy(dtype=float)
        y = group["r_nes"].to_numpy(dtype=float)
        lower = min(float(x.min()), float(y.min())) - 0.1
        upper = max(float(x.max()), float(y.max())) + 0.1
        axis.scatter(x, y, s=22, alpha=0.72, color="#176d8c", edgecolors="none")
        axis.plot([lower, upper], [lower, upper], "--", color="#b33c1a", linewidth=1.0)
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_xlabel("PyFgsea NES")
        axis.set_ylabel("R fgsea NES")
        lane_label = (
            "0.1.4 / fgsea 1.32.2"
            if lane == "legacy"
            else "0.2.0rc7 / fgsea 1.38.0"
        )
        scenario_label = (
            "publication input"
            if scenario == "publication_main"
            else "predeclared ties"
        )
        axis.set_title(f"{lane_label}\n{scenario_label}")
        row = metrics.loc[
            (metrics["lane"] == lane)
            & (metrics["scenario"] == scenario)
            & (metrics["metric"] == "NES")
        ].iloc[0]
        axis.text(
            0.03,
            0.97,
            (
                f"Pearson={row['pearson']:.6f}\n"
                f"Spearman={row['spearman']:.6f}\n"
                f"RMSE={row['rmse']:.6g}\n"
                f"median |difference|={row['median_absolute_difference']:.6g}"
            ),
            transform=axis.transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
        )
    figure.suptitle(
        "Figure 1 dual-lane PyFgsea/R-fgsea agreement (derived from raw rows)"
    )
    figure.savefig(
        output,
        dpi=300,
        metadata={"Software": SUITE_VERSION, "Title": "Figure 1 dual-lane agreement"},
    )
    plt.close(figure)


def compare_results(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).resolve()
    if _is_within(output_dir, REPO_ROOT):
        raise ValueError("Choose an output directory outside the repository")
    output_dir = ensure_empty_output_dir(output_dir)
    initial_git = verify_clean_git_checkout(
        REPO_ROOT,
        expected_commit=args.expected_git_commit,
        expected_tag=args.expected_git_tag,
    )

    legacy_path = Path(args.legacy_receipt).resolve()
    current_path = Path(args.current_receipt).resolve()
    legacy_receipt, legacy_raw, legacy_timing, legacy_raw_path, legacy_timing_path = (
        _validate_receipt(legacy_path, "legacy")
    )
    (
        current_receipt,
        current_raw,
        current_timing,
        current_raw_path,
        current_timing_path,
    ) = _validate_receipt(current_path, "current")
    _same_input_contract(legacy_receipt, current_receipt)
    input_manifest_path = Path(args.input_manifest).resolve()
    _verify_common_input_bundle(input_manifest_path, legacy_receipt, current_receipt)
    if legacy_receipt["git"] != current_receipt["git"]:
        raise ValueError(
            "legacy/current lanes were not run from the same evidence checkout"
        )
    if legacy_receipt["git"]["commit"] != initial_git["commit"]:
        raise ValueError("lane receipts were produced by a different evidence commit")
    current_script_hashes = {
        "run_lane": sha256_file(SCRIPT_DIR / "run_lane.py"),
        "common": sha256_file(SCRIPT_DIR / "common.py"),
        "prepare_inputs": sha256_file(SCRIPT_DIR / "prepare_inputs.py"),
        "r_helper": sha256_file(SCRIPT_DIR / "run_reference.R"),
    }
    for script_name, expected_hash in current_script_hashes.items():
        if legacy_receipt["scripts"][script_name]["sha256"] != expected_hash:
            raise ValueError(
                f"lane receipts were generated by a different {script_name} script"
            )

    combined_raw = pd.concat([legacy_raw, current_raw], ignore_index=True)
    combined_raw = combined_raw.sort_values(
        ["scenario", "lane", "pathway"], kind="mergesort"
    ).reset_index(drop=True)
    combined_raw_path = output_dir / "figure1_pathway_level_raw.tsv"
    combined_raw.to_csv(
        combined_raw_path,
        sep="\t",
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    combined_raw_hash = sha256_file(combined_raw_path)

    # All downstream calculations intentionally start from the just-written
    # pathway-level artifact.  The in-memory lane frames are no longer used as
    # a metric source after this point.
    combined_raw = pd.read_csv(combined_raw_path, sep="\t")

    metrics = _metric_rows(combined_raw, combined_raw_hash)
    metrics_path = output_dir / "figure1_agreement_metrics.tsv"
    metrics.to_csv(
        metrics_path, sep="\t", index=False, float_format="%.17g", lineterminator="\n"
    )
    overlap = _overlap_rows(combined_raw, combined_raw_hash)
    overlap_path = output_dir / "figure1_pathway_overlap.tsv"
    overlap.to_csv(
        overlap_path, sep="\t", index=False, float_format="%.17g", lineterminator="\n"
    )
    extreme = _extreme_rows(combined_raw, combined_raw_hash, args.extreme_count)
    extreme_path = output_dir / "figure1_extreme_tail_cases.tsv"
    extreme.to_csv(
        extreme_path, sep="\t", index=False, float_format="%.17g", lineterminator="\n"
    )
    cross_lane = _cross_lane_rows(combined_raw, combined_raw_hash)
    cross_lane_path = output_dir / "figure1_legacy_current_change.tsv"
    cross_lane.to_csv(
        cross_lane_path,
        sep="\t",
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    timing = pd.concat([legacy_timing, current_timing], ignore_index=True).sort_values(
        ["scenario", "lane", "engine"], kind="mergesort"
    )
    timing_path = output_dir / "figure1_runtime_memory.tsv"
    timing.to_csv(
        timing_path, sep="\t", index=False, float_format="%.17g", lineterminator="\n"
    )
    figure_path = output_dir / "figure1_dual_lane_agreement.png"
    _draw_figure(combined_raw, metrics, figure_path)

    output_files = {
        "pathway_level_raw": combined_raw_path,
        "agreement_metrics": metrics_path,
        "pathway_overlap": overlap_path,
        "extreme_tail_cases": extreme_path,
        "legacy_current_change": cross_lane_path,
        "runtime_memory": timing_path,
        "figure": figure_path,
    }
    verify_git_unchanged(REPO_ROOT, initial_git)
    receipt_path = output_dir / "adjudication_receipt.json"
    write_json(
        receipt_path,
        {
            "schema_version": 1,
            "kind": "figure1_adjudication_receipt",
            "suite_version": SUITE_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "invocation": [str(item) for item in sys.argv],
            "git": initial_git,
            "script": file_record(Path(__file__)),
            "python_environment": {
                "executable": str(Path(sys.executable).resolve()),
                "version": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "matplotlib": matplotlib.__version__,
            },
            "inputs": {
                "input_manifest": file_record(input_manifest_path),
                "legacy_receipt": file_record(legacy_path),
                "legacy_raw": file_record(legacy_raw_path),
                "legacy_timing": file_record(legacy_timing_path),
                "current_receipt": file_record(current_path),
                "current_raw": file_record(current_raw_path),
                "current_timing": file_record(current_timing_path),
            },
            "outputs": {
                label: file_record(path, relative_to=output_dir)
                for label, path in sorted(output_files.items())
            },
            "derivation_policy": {
                "only_metric_source": "figure1_pathway_level_raw.tsv",
                "manual_overrides": False,
                "ties_scenario_is_publication_input": False,
                "extreme_tail_count_per_lane_and_scenario": args.extreme_count,
                "fdr_cutoff": 0.05,
                "top_pathway_count": 10,
            },
            "result_scope": {
                "legacy_lane_identity": "official PyPI 0.1.4 artifact lane",
                "legacy_native_core_source_reproducible": False,
                "ties_scope": "same recorded environment sensitivity evidence",
                "ties_cross_platform_equivalence_claimed": False,
            },
        },
    )
    return receipt_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-receipt", required=True, type=Path)
    parser.add_argument("--current-receipt", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--expected-git-tag", required=True)
    parser.add_argument("--extreme-count", type=int, default=10)
    args = parser.parse_args(argv)
    if args.extreme_count < 1:
        parser.error("--extreme-count must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = compare_results(args)
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
