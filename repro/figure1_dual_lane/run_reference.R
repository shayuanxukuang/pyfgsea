#!/usr/bin/env Rscript

fail <- function(message) {
  stop(message, call. = FALSE)
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 12L) {
  fail(paste(
    "usage: run_reference.R RANKS GMT RESULT ENV SESSION FGSEA R BIOC",
    "R_SEED MIN_SIZE MAX_SIZE SAMPLE_SIZE (EPS is supplied by environment)"
  ))
}

ranks_path <- args[[1L]]
gmt_path <- args[[2L]]
result_path <- args[[3L]]
environment_path <- args[[4L]]
session_path <- args[[5L]]
expected_fgsea <- args[[6L]]
expected_r <- args[[7L]]
expected_bioc <- args[[8L]]
r_seed <- as.integer(args[[9L]])
min_size <- as.integer(args[[10L]])
max_size <- as.integer(args[[11L]])
sample_size <- as.integer(args[[12L]])
eps_text <- Sys.getenv("PYFGSEA_FIGURE1_EPS", unset = "")
if (!nzchar(eps_text)) {
  fail("PYFGSEA_FIGURE1_EPS is required")
}
eps <- as.numeric(eps_text)

if (!file.exists(ranks_path) || !file.exists(gmt_path)) {
  fail("rank or GMT input is missing")
}
for (package_name in c("BiocManager", "data.table", "fgsea")) {
  if (!requireNamespace(package_name, quietly = TRUE)) {
    fail(sprintf("required R package is missing: %s", package_name))
  }
}

actual_fgsea <- as.character(utils::packageVersion("fgsea"))
actual_r <- as.character(getRversion())
actual_bioc <- as.character(BiocManager::version())
if (!identical(actual_fgsea, expected_fgsea)) {
  fail(sprintf("fgsea mismatch: expected %s, found %s", expected_fgsea, actual_fgsea))
}
if (!identical(actual_r, expected_r)) {
  fail(sprintf("R mismatch: expected %s, found %s", expected_r, actual_r))
}
if (!identical(actual_bioc, expected_bioc)) {
  fail(sprintf(
    "Bioconductor mismatch: expected %s, found %s",
    expected_bioc,
    actual_bioc
  ))
}
if (is.na(r_seed) || is.na(min_size) || is.na(max_size) || is.na(sample_size)) {
  fail("integer parameters could not be parsed")
}
if (!is.finite(eps) || eps <= 0) {
  fail("eps must be finite and positive")
}

ranks <- data.table::fread(ranks_path)
if (!identical(names(ranks), c("Gene", "Score"))) {
  fail("ranks must have exactly the columns Gene,Score")
}
if (anyNA(ranks$Gene) || anyDuplicated(ranks$Gene)) {
  fail("ranked gene identifiers must be nonmissing and unique")
}
if (any(!is.finite(ranks$Score))) {
  fail("rank scores must all be finite")
}

# This explicit secondary key makes the predeclared ties scenario independent
# of a platform's sort implementation.  It is a no-op for the publication
# scenario, whose generated scores are unique.
data.table::setorder(ranks, -Score, Gene)
stats <- ranks$Score
names(stats) <- ranks$Gene
pathways <- fgsea::gmtPathways(gmt_path)
if (length(pathways) == 0L || anyDuplicated(names(pathways))) {
  fail("GMT must contain one or more uniquely named pathways")
}

set.seed(r_seed)
timing <- system.time({
  result <- fgsea::fgseaMultilevel(
    pathways = pathways,
    stats = stats,
    minSize = min_size,
    maxSize = max_size,
    sampleSize = sample_size,
    eps = eps,
    scoreType = "std",
    nproc = 1
  )
})

required <- c("pathway", "ES", "NES", "pval", "padj", "size")
if (!all(required %in% names(result))) {
  fail(sprintf(
    "fgsea result is missing columns: %s",
    paste(setdiff(required, names(result)), collapse = ",")
  ))
}
result <- data.table::as.data.table(result)[, ..required]
data.table::setorder(result, pathway)
if (nrow(result) == 0L) {
  fail("fgsea returned no pathways")
}
numeric_columns <- c("ES", "NES", "pval", "padj", "size")
if (any(!vapply(result[, ..numeric_columns], function(x) all(is.finite(x)), logical(1)))) {
  fail("fgsea returned a non-finite required value")
}
if (any(result$pval < 0 | result$pval > 1 | result$padj < 0 | result$padj > 1)) {
  fail("fgsea returned a probability outside [0,1]")
}

data.table::fwrite(result, result_path, sep = "\t", quote = FALSE)
environment <- data.table::as.data.table(list(
  key = c(
    "r_version",
    "bioconductor_version",
    "fgsea_version",
    "fgsea_library_path",
    "data_table_version",
    "r_seed",
    "score_type",
    "elapsed_seconds",
    "user_seconds",
    "system_seconds"
  ),
  value = c(
    actual_r,
    actual_bioc,
    actual_fgsea,
    normalizePath(find.package("fgsea"), winslash = "/", mustWork = TRUE),
    as.character(utils::packageVersion("data.table")),
    as.character(r_seed),
    "std",
    sprintf("%.17g", unname(timing[["elapsed"]])),
    sprintf("%.17g", unname(timing[["user.self"]])),
    sprintf("%.17g", unname(timing[["sys.self"]]))
  )
))
data.table::fwrite(environment, environment_path, sep = "\t", quote = FALSE)
session <- capture.output(sessionInfo())
writeLines(session, session_path, useBytes = TRUE)
