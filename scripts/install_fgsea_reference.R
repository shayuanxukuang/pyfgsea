#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("usage: install_fgsea_reference.R /path/to/fgsea.tar.gz", call. = FALSE)
}

tarball <- normalizePath(args[[1]], mustWork = TRUE)
bioc_version <- trimws(Sys.getenv("FGSEA_REFERENCE_BIOC_VERSION", unset = ""))
fgsea_version <- trimws(Sys.getenv("FGSEA_REFERENCE_VERSION", unset = ""))
if (!nzchar(bioc_version) || !nzchar(fgsea_version)) {
  stop(
    "FGSEA_REFERENCE_BIOC_VERSION and FGSEA_REFERENCE_VERSION are required",
    call. = FALSE
  )
}
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  stop("BiocManager must be installed before the reference packages", call. = FALSE)
}

options(repos = BiocManager::repositories(version = bioc_version))
dependencies <- c(
  "Rcpp", "data.table", "BiocParallel", "ggplot2", "cowplot",
  "fastmatch", "Matrix", "scales", "BH"
)
BiocManager::install(
  dependencies,
  version = bioc_version,
  ask = FALSE,
  update = FALSE,
  Ncpus = max(1L, parallel::detectCores(logical = FALSE), na.rm = TRUE)
)

utils::install.packages(tarball, repos = NULL, type = "source")
actual <- as.character(utils::packageVersion("fgsea"))
if (!identical(actual, fgsea_version)) {
  stop(sprintf("installed fgsea %s, expected %s", actual, fgsea_version), call. = FALSE)
}
