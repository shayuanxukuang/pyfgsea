#!/usr/bin/env python3
"""Materialize the immutable publication and predeclared-ties inputs once."""

from __future__ import annotations

import argparse
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .common import (
        EXPECTED_INPUT_SHA256,
        PUBLICATION_SOURCE_COMMIT,
        SCENARIOS,
        SUITE_VERSION,
        ensure_empty_output_dir,
        file_record,
        sha256_file,
        write_json,
    )
except ImportError:  # pragma: no cover - direct script execution
    from common import (  # type: ignore
        EXPECTED_INPUT_SHA256,
        PUBLICATION_SOURCE_COMMIT,
        SCENARIOS,
        SUITE_VERSION,
        ensure_empty_output_dir,
        file_record,
        sha256_file,
        write_json,
    )


PUBLICATION_PARAMETERS = {"n_genes": 12000, "n_sets": 100, "seed": 42}
PUBLICATION_SCORE_SIGNIFICANT_DIGITS = 12
TIES_PARAMETERS = {
    "n_genes": 4000,
    "n_sets": 60,
    "seed": 4242,
    "score_round_decimals": 1,
}


def generate_test_data(
    *, n_genes: int, n_sets: int, seed: int
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Transcribe the exact historical ``repro.data_utils`` generator logic."""

    if n_sets < 20:
        raise ValueError("n_sets must be at least 20")
    numpy_rng = np.random.RandomState(seed)
    python_rng = random.Random(seed)
    genes = [f"GENE_{index:05d}" for index in range(n_genes)]

    scores = numpy_rng.randn(n_genes)
    scores[:500] += 2.0
    scores[-500:] -= 2.0
    ranks = pd.DataFrame({"Gene": genes, "Score": scores})
    ranks = ranks.sort_values("Score", ascending=False)

    pathways: dict[str, list[str]] = {}
    for index in range(n_sets - 20):
        size = python_rng.randint(15, 200)
        pathways[f"NULL_PATH_{index}"] = python_rng.sample(genes, size)

    top_genes = genes[:1000]
    for index in range(10):
        size = python_rng.randint(20, 50)
        pathways[f"POS_PATH_{index}"] = python_rng.sample(top_genes, size)

    bottom_genes = genes[-1000:]
    for index in range(10):
        size = python_rng.randint(20, 50)
        pathways[f"NEG_PATH_{index}"] = python_rng.sample(bottom_genes, size)

    return ranks, pathways


def _write_scenario(
    root: Path,
    name: str,
    ranks: pd.DataFrame,
    pathways: dict[str, list[str]],
    parameters: dict[str, int],
    *,
    score_transform: str,
    float_significant_digits: int,
) -> dict[str, Any]:
    scenario_dir = root / name
    scenario_dir.mkdir()
    ranks_path = scenario_dir / "ranks.csv"
    pathways_path = scenario_dir / "pathways.gmt"

    ordered = ranks.sort_values(
        ["Score", "Gene"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    if ordered["Gene"].duplicated().any():
        raise RuntimeError(f"{name} contains duplicate gene identifiers")
    if not np.isfinite(ordered["Score"].to_numpy(dtype=float)).all():
        raise RuntimeError(f"{name} contains non-finite scores")
    if len(pathways) != parameters["n_sets"]:
        raise RuntimeError(f"{name} pathway count does not match its contract")
    if len(set(pathways)) != len(pathways):
        raise RuntimeError(f"{name} contains duplicate pathway names")
    sizes = [len(set(members)) for members in pathways.values()]
    if min(sizes) < 15 or max(sizes) > 500:
        raise RuntimeError(f"{name} contains a pathway outside [15, 500]")

    # The publication scenario is explicitly canonicalized before this write.
    # The ties scenario is already rounded to one decimal place.
    ordered.to_csv(
        ranks_path,
        index=False,
        float_format=f"%.{float_significant_digits}g",
        lineterminator="\n",
    )
    with pathways_path.open("w", encoding="utf-8", newline="\n") as handle:
        for pathway, members in pathways.items():
            if len(set(members)) != len(members):
                raise RuntimeError(f"{name}/{pathway} contains duplicate members")
            handle.write(f"{pathway}\tNA\t" + "\t".join(members) + "\n")

    counts = ordered["Score"].value_counts()
    tied_groups = counts[counts > 1]
    return {
        "parameters": parameters,
        "score_transform": score_transform,
        "ranks": file_record(ranks_path, relative_to=root),
        "pathways": file_record(pathways_path, relative_to=root),
        "invariants": {
            "gene_count": int(len(ordered)),
            "pathway_count": int(len(pathways)),
            "minimum_pathway_size": int(min(sizes)),
            "maximum_pathway_size": int(max(sizes)),
            "tied_score_group_count": int(len(tied_groups)),
            "tied_gene_count": int(tied_groups.sum()) if len(tied_groups) else 0,
            "maximum_tie_multiplicity": (
                int(tied_groups.max()) if len(tied_groups) else 1
            ),
        },
    }


def prepare_inputs(output_dir: Path) -> Path:
    """Generate both scenarios and return the portable manifest path."""

    root = ensure_empty_output_dir(output_dir)
    publication_ranks, publication_pathways = generate_test_data(
        **PUBLICATION_PARAMETERS
    )
    publication_ranks = publication_ranks.copy()
    publication_ranks["Score"] = publication_ranks["Score"].map(
        lambda value: float(
            format(float(value), f".{PUBLICATION_SCORE_SIGNIFICANT_DIGITS}g")
        )
    )
    ties_ranks, ties_pathways = generate_test_data(
        n_genes=TIES_PARAMETERS["n_genes"],
        n_sets=TIES_PARAMETERS["n_sets"],
        seed=TIES_PARAMETERS["seed"],
    )
    ties_ranks = ties_ranks.copy()
    ties_ranks["Score"] = ties_ranks["Score"].round(
        TIES_PARAMETERS["score_round_decimals"]
    )

    scenario_records = {
        "publication_main": _write_scenario(
            root,
            "publication_main",
            publication_ranks,
            publication_pathways,
            {
                **PUBLICATION_PARAMETERS,
                "score_significant_digits": PUBLICATION_SCORE_SIGNIFICANT_DIGITS,
            },
            score_transform="canonicalize_to_12_significant_decimal_digits",
            float_significant_digits=PUBLICATION_SCORE_SIGNIFICANT_DIGITS,
        ),
        "ties_predeclared": _write_scenario(
            root,
            "ties_predeclared",
            ties_ranks,
            ties_pathways,
            dict(TIES_PARAMETERS),
            score_transform="round_binary64_to_1_decimal",
            float_significant_digits=17,
        ),
    }
    if tuple(scenario_records) != SCENARIOS:
        raise AssertionError("scenario order differs from the suite contract")
    for scenario, expected_files in EXPECTED_INPUT_SHA256.items():
        for label, expected_hash in expected_files.items():
            actual_hash = scenario_records[scenario][label]["sha256"]
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"generated {scenario}/{label} hash drifted: "
                    f"expected {expected_hash}, found {actual_hash}"
                )

    manifest_path = root / "input_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "kind": "figure1_input_manifest",
            "suite_version": SUITE_VERSION,
            "historical_generator": {
                "source_commit": PUBLICATION_SOURCE_COMMIT,
                "source_path": "bioinfor0208/revision/generate_revision_assets.py",
                "generator_call": "generate_test_data(n_genes=12000,n_sets=100,seed=42)",
                "note": (
                    "publication_main runs a direct transcription of the historical "
                    "generator, then canonicalizes platform-dependent binary64 tails "
                    "to 12 significant decimal digits; ties_predeclared is a separate, "
                    "prospectively labelled quantized-score stress scenario and is not "
                    "a paper input"
                ),
            },
            "generator": {
                "command": [str(item) for item in sys.argv],
                "script_sha256": sha256_file(Path(__file__)),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "scenarios": scenario_records,
        },
    )
    return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new or empty directory that will contain the immutable input bundle",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = prepare_inputs(args.output_dir)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
