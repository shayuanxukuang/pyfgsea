# PyFgsea documentation style

Public documentation answers four questions:

1. What does PyFgsea do?
2. How do I run it?
3. What provenance makes the result reproducible?
4. What is unsupported or incomplete?

## Writing

- Start with behavior, commands, or outputs.
- Use short sentences and concrete file, parameter, and version names.
- Put prerequisites before commands and limitations next to the relevant use.
- Separate requirements from examples.
- Omit process metadata that does not change behavior, commands, results, or
  limitations.
- Avoid promotional claims, speculation, and unsupported conclusions.
- Keep candidate details in `docs/releases/0.2.0-rc-history.md`.

## Versions

- `Cargo.toml` is the single version source. `pyproject.toml` keeps `version`
  dynamic.
- Write Python prereleases as `0.2.0rcN` and tags as `v0.2.0-rcN`.
- Name deprecated and approximate interfaces explicitly.
- State defaults that affect numerical interpretation.

## Reproducibility

- Keep fgsea 1.32.2 publication results separate from fgsea 1.38.0 current
  comparisons.
- State score type, mode, pathway-size policy, seeds, sample size, epsilon, and
  thread count.
- Keep unresolved and failure states visible.
- Distinguish exploratory source runs from packaged release checks.

Release and paper records include:

- a clean worktree, full commit and tree, and annotated tag;
- source and Rust tests;
- source, sdist, wheel built from that sdist, and clean installation;
- installed version, native-core, and installed-wheel tests;
- exact commands, parameters, and reference environments; and
- SHA-256 values for files, inputs, and outputs.

## Release notes

Use `What changed`, `Compatibility with 0.1.4`, `Migration`, `Reproducing the
paper`, and `Known limitations`. Keep candidate history in the separate table.
