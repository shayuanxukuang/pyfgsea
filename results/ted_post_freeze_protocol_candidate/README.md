# TED post-freeze external validation protocol candidate

Status: **DRAFT - NOT ACTIVATED - NOT EXTERNALLY TIMESTAMPED**

This directory is a ready-to-complete protocol package. It is deliberately not
described as a post-freeze validation because the repository is dirty, the
immutable TED release commit does not yet exist, no dataset/outcome custodian has
been assigned, and no Zenodo or OSF timestamp has been created.

Activation order:

1. Create the immutable neutral TED release commit and insert its SHA in
   `protocol.json`.
2. Select one dataset using metadata-only screening. Record the accession,
   inclusion decision, biological blocks, primary contrast and file-level RNA vs
   outcome separation without inspecting outcome values.
3. Freeze event families, gene-set file hashes, thresholds, negative controls,
   exclusions and all-result reporting rules.
4. Deposit this complete directory on Zenodo or OSF and record the immutable URL,
   DOI/version and timestamp.
5. Only after steps 1-4, download/open the expression matrix. Keep the external
   outcome file unopened and checksum-locked until TED outputs are finalized.
6. Run TED once, freeze all success and failure outputs, then unlock outcome data.

The current local Git HEAD (`2b047dc557604018b28d82fa3da9ab496e1955a4`)
is recorded only as provenance. It is not the freeze commit because the worktree
contains extensive uncommitted work.

