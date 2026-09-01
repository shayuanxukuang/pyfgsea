"""Fail-closed validation for R/fgsea reference runs.

Reference comparisons must opt into one of the two declared fgsea versions via
``FGSEA_REFERENCE_VERSION``.  A locally installed, otherwise usable fgsea is
not accepted implicitly.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "verify_fgsea_reference.R"
SUPPORTED_FGSEA_VERSIONS = {"1.32.2", "1.38.0"}
REFERENCE_CONTRACTS = {
    "1.32.2": {"r_version": "4.4.3", "bioconductor_version": "3.20"},
    "1.38.0": {"r_version": "4.6.0", "bioconductor_version": "3.23"},
}


@dataclass(frozen=True)
class RReferenceEnvironment:
    """A verified R executable and its declared reference metadata."""

    rscript: str
    expected_fgsea_version: str
    details: Dict[str, str]

    @property
    def r_guard(self) -> str:
        """Return an in-process guard for an embedded R script."""

        expected = json.dumps(self.expected_fgsea_version)
        contract = REFERENCE_CONTRACTS[self.expected_fgsea_version]
        expected_r = json.dumps(contract["r_version"])
        expected_bioc = json.dumps(contract["bioconductor_version"])
        return f"""
        expected_fgsea <- {expected}
        expected_r <- {expected_r}
        expected_bioc <- {expected_bioc}
        actual_fgsea <- as.character(utils::packageVersion("fgsea"))
        actual_r <- as.character(getRversion())
        actual_bioc <- as.character(BiocManager::version())
        if (!identical(actual_fgsea, expected_fgsea)) {{
            stop(sprintf(
                "fgsea reference mismatch: expected %s, found %s",
                expected_fgsea,
                actual_fgsea
            ), call. = FALSE)
        }}
        if (!identical(actual_r, expected_r)) {{
            stop(sprintf(
                "R reference mismatch: expected %s, found %s",
                expected_r,
                actual_r
            ), call. = FALSE)
        }}
        if (!identical(actual_bioc, expected_bioc)) {{
            stop(sprintf(
                "Bioconductor reference mismatch: expected %s, found %s",
                expected_bioc,
                actual_bioc
            ), call. = FALSE)
        }}
        """


def verify_r_fgsea_reference() -> RReferenceEnvironment:
    """Validate the explicitly selected R/fgsea reference environment.

    The absence of ``FGSEA_REFERENCE_VERSION`` is an error.  This prevents a
    workstation's arbitrary fgsea installation from becoming reference data.
    """

    expected = os.environ.get("FGSEA_REFERENCE_VERSION", "").strip()
    if expected not in SUPPORTED_FGSEA_VERSIONS:
        allowed = ", ".join(sorted(SUPPORTED_FGSEA_VERSIONS))
        raise RuntimeError(
            "Set FGSEA_REFERENCE_VERSION explicitly before generating R "
            f"reference results; supported values are {allowed}."
        )

    configured_rscript = os.environ.get("PYFGSEA_REFERENCE_RSCRIPT", "").strip()
    if configured_rscript:
        configured_path = Path(configured_rscript).expanduser().resolve()
        if not configured_path.is_file():
            raise RuntimeError(
                "PYFGSEA_REFERENCE_RSCRIPT does not name a file: "
                f"{configured_path}"
            )
        rscript = str(configured_path)
    else:
        rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError(
            "Rscript is required for an R/fgsea reference run. Use one of the "
            "declared reference Dockerfiles or install the exact environment."
        )
    if not VERIFY_SCRIPT.is_file():
        raise RuntimeError(f"Reference verifier is missing: {VERIFY_SCRIPT}")

    completed = subprocess.run(
        [rscript, str(VERIFY_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"R/fgsea reference verification failed: {message}")

    details: Dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            details[key.strip().lower()] = value.strip()

    contract = REFERENCE_CONTRACTS[expected]
    if details.get("r_version") != contract["r_version"]:
        raise RuntimeError("R verifier did not return the declared lane version")
    if details.get("bioconductor_version") != contract["bioconductor_version"]:
        raise RuntimeError("R verifier did not return the declared Bioconductor version")
    if details.get("fgsea_version") != expected:
        raise RuntimeError("R verifier did not return the declared fgsea version")

    return RReferenceEnvironment(
        rscript=rscript,
        expected_fgsea_version=expected,
        details=details,
    )


def write_reference_environment(
    path: Path,
    reference: RReferenceEnvironment,
    extra: Optional[Mapping[str, object]] = None,
) -> None:
    """Write a per-run environment sidecar after successful verification."""

    payload: Dict[str, object] = {
        "schema_version": 1,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "rscript": reference.rscript,
        "expected_fgsea_version": reference.expected_fgsea_version,
        "environment": dict(sorted(reference.details.items())),
    }
    if extra:
        payload["run"] = dict(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
