#!/usr/bin/env Rscript

fail <- function(message) {
  stop(message, call. = FALSE)
}

required_env <- function(name) {
  value <- trimws(Sys.getenv(name, unset = ""))
  if (!nzchar(value)) {
    fail(sprintf("%s must be set for a reference run", name))
  }
  value
}

expected_fgsea <- required_env("FGSEA_REFERENCE_VERSION")
if (!(expected_fgsea %in% c("1.32.2", "1.38.0"))) {
  fail(sprintf("unsupported FGSEA_REFERENCE_VERSION: %s", expected_fgsea))
}
contract <- switch(
  expected_fgsea,
  "1.32.2" = list(r = "4.4.3", bioc = "3.20"),
  "1.38.0" = list(r = "4.6.0", bioc = "3.23")
)

if (!requireNamespace("fgsea", quietly = TRUE)) {
  fail("the fgsea package is not installed")
}
actual_fgsea <- as.character(utils::packageVersion("fgsea"))
if (!identical(actual_fgsea, expected_fgsea)) {
  fail(sprintf(
    "fgsea reference mismatch: expected %s, found %s",
    expected_fgsea,
    actual_fgsea
  ))
}

declared_r <- trimws(Sys.getenv("FGSEA_REFERENCE_R_VERSION", unset = ""))
if (nzchar(declared_r) && !identical(declared_r, contract$r)) {
  fail(sprintf(
    "declared R version %s conflicts with the %s lane contract (%s)",
    declared_r,
    expected_fgsea,
    contract$r
  ))
}
expected_r <- contract$r
actual_r <- as.character(getRversion())
if (!identical(actual_r, expected_r)) {
  fail(sprintf("R reference mismatch: expected %s, found %s", expected_r, actual_r))
}

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  fail("BiocManager is required to verify the Bioconductor release")
}
declared_bioc <- trimws(Sys.getenv("FGSEA_REFERENCE_BIOC_VERSION", unset = ""))
if (nzchar(declared_bioc) && !identical(declared_bioc, contract$bioc)) {
  fail(sprintf(
    "declared Bioconductor version %s conflicts with the %s lane contract (%s)",
    declared_bioc,
    expected_fgsea,
    contract$bioc
  ))
}
expected_bioc <- contract$bioc
actual_bioc <- as.character(BiocManager::version())
if (!identical(actual_bioc, expected_bioc)) {
  fail(sprintf(
    "Bioconductor reference mismatch: expected %s, found %s",
    expected_bioc,
    actual_bioc
  ))
}

cat(sprintf("REFERENCE_ID=%s\n", Sys.getenv("FGSEA_REFERENCE_ID", unset = "unspecified")))
cat(sprintf("R_VERSION=%s\n", actual_r))
cat(sprintf("BIOCONDUCTOR_VERSION=%s\n", actual_bioc))
cat(sprintf("FGSEA_VERSION=%s\n", actual_fgsea))
cat(sprintf("FGSEA_PATH=%s\n", normalizePath(find.package("fgsea"), winslash = "/")))
