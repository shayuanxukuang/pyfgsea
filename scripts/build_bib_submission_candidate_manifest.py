from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "BIB_submission_candidate_2026-07-16"
OUTPUT = CANDIDATE / "submission_candidate_manifest.tsv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files = sorted(path for path in CANDIDATE.rglob("*") if path.is_file() and path != OUTPUT)
    rows = [
        {
            "path": path.relative_to(CANDIDATE).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    pd.DataFrame(rows).to_csv(OUTPUT, sep="\t", index=False)
    print(f"Wrote {len(rows)} candidate-package entries to {OUTPUT}")


if __name__ == "__main__":
    main()
