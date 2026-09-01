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
FROZEN_INPUT_ROOT = Path(__file__).resolve().with_name("frozen_inputs")
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


def _read_frozen_pathways(path: Path) -> dict[str, list[str]]:
    pathways: dict[str, list[str]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.split("\t")
        if len(fields) < 3 or fields[1] != "NA":
            raise RuntimeError(f"invalid frozen GMT row at {path}:{line_number}")
        pathway, members = fields[0], fields[2:]
        if not pathway or pathway in pathways:
            raise RuntimeError(f"duplicate or empty pathway at {path}:{line_number}")
        pathways[pathway] = members
    return pathways


def _materialize_frozen_scenario(
    root: Path,
    name: str,
    parameters: dict[str, int],
    *,
    score_transform: str,
    score_transform_parameters: dict[str, int],
) -> dict[str, Any]:
    source_dir = FROZEN_INPUT_ROOT / name
    scenario_dir = root / name
    scenario_dir.mkdir()
    source_ranks_path = source_dir / "ranks.csv"
    source_pathways_path = source_dir / "pathways.gmt"
    ranks_path = scenario_dir / "ranks.csv"
    pathways_path = scenario_dir / "pathways.gmt"

    for label, source, target in (
        ("ranks", source_ranks_path, ranks_path),
        ("pathways", source_pathways_path, pathways_path),
    ):
        source_bytes = source.read_bytes()
        target.write_bytes(source_bytes)
        if target.read_bytes() != source_bytes:
            raise RuntimeError(f"materialized frozen {name}/{label} bytes differ")

    ranks = pd.read_csv(ranks_path)
    if list(ranks.columns) != ["Gene", "Score"]:
        raise RuntimeError(f"{name} ranks must contain exactly Gene and Score")
    ordered = ranks.sort_values(
        ["Score", "Gene"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    if not ordered.equals(ranks.reset_index(drop=True)):
        raise RuntimeError(f"{name} committed ranks are not canonically ordered")
    if ordered["Gene"].duplicated().any():
        raise RuntimeError(f"{name} contains duplicate gene identifiers")
    if not np.isfinite(ordered["Score"].to_numpy(dtype=float)).all():
        raise RuntimeError(f"{name} contains non-finite scores")

    pathways = _read_frozen_pathways(pathways_path)
    if len(pathways) != parameters["n_sets"]:
        raise RuntimeError(f"{name} pathway count does not match its contract")
    if len(set(pathways)) != len(pathways):
        raise RuntimeError(f"{name} contains duplicate pathway names")
    sizes = [len(set(members)) for members in pathways.values()]
    if min(sizes) < 15 or max(sizes) > 500:
        raise RuntimeError(f"{name} contains a pathway outside [15, 500]")

    for pathway, members in pathways.items():
        if len(set(members)) != len(members):
            raise RuntimeError(f"{name}/{pathway} contains duplicate members")

    counts = ordered["Score"].value_counts()
    tied_groups = counts[counts > 1]
    return {
        "parameters": parameters,
        "score_transform": score_transform,
        "score_transform_parameters": score_transform_parameters,
        "materialization": "copy_commit_bound_frozen_bytes",
        "frozen_source": {
            "ranks": file_record(source_ranks_path, relative_to=FROZEN_INPUT_ROOT),
            "pathways": file_record(
                source_pathways_path, relative_to=FROZEN_INPUT_ROOT
            ),
        },
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
    """Materialize both frozen scenarios and return the portable manifest path."""

    root = ensure_empty_output_dir(output_dir)
    scenario_records = {
        "publication_main": _materialize_frozen_scenario(
            root,
            "publication_main",
            dict(PUBLICATION_PARAMETERS),
            score_transform=(
                "frozen_bytes_canonicalized_to_12_significant_decimal_digits"
            ),
            score_transform_parameters={
                "significant_decimal_digits": PUBLICATION_SCORE_SIGNIFICANT_DIGITS
            },
        ),
        "ties_predeclared": _materialize_frozen_scenario(
            root,
            "ties_predeclared",
            dict(TIES_PARAMETERS),
            score_transform="frozen_bytes_round_binary64_to_1_decimal",
            score_transform_parameters={
                "round_decimal_places": TIES_PARAMETERS["score_round_decimals"]
            },
        ),
    }
    if tuple(scenario_records) != SCENARIOS:
        raise AssertionError("scenario order differs from the suite contract")
    manifest_path = root / "input_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": 2,
            "kind": "figure1_input_manifest",
            "suite_version": SUITE_VERSION,
            "historical_generator": {
                "source_commit": PUBLICATION_SOURCE_COMMIT,
                "source_path": "bioinfor0208/revision/generate_revision_assets.py",
                "generator_call": "generate_test_data(n_genes=12000,n_sets=100,seed=42)",
                "note": (
                    "publication_main is copied from committed frozen bytes produced "
                    "once by the directly transcribed historical generator and "
                    "canonicalized to 12 significant decimal digits; evidence runs do "
                    "not regenerate platform-dependent normal scores. ties_predeclared "
                    "is a separate, prospectively labelled quantized-score stress "
                    "scenario and is not a paper input"
                ),
            },
            "generator": {
                "mode": "copy_commit_bound_frozen_bytes",
                "command": [str(item) for item in sys.argv],
                "script_sha256": sha256_file(Path(__file__)),
                "frozen_input_root": "repro/figure1_dual_lane/frozen_inputs",
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
