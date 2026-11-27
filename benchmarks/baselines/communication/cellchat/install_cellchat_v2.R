options(repos = c(CRAN = "https://cloud.r-project.org"))
Sys.setenv(R_REMOTES_NO_ERRORS_FROM_WARNINGS = "true")

lib_path <- Sys.getenv("CELLCHAT_R_LIB", unset = "")
if (nzchar(lib_path)) {
  dir.create(lib_path, recursive = TRUE, showWarnings = FALSE)
  .libPaths(c(normalizePath(lib_path, winslash = "/", mustWork = FALSE), .libPaths()))
}

install_if_missing <- function(pkgs) {
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    install.packages(missing, dependencies = TRUE)
  }
}

install_if_missing(c("remotes", "BiocManager"))
install_if_missing(c("NMF", "future"))

bioc_pkgs <- c("Biobase", "BiocNeighbors")
missing_bioc <- bioc_pkgs[!vapply(bioc_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_bioc) > 0) {
  BiocManager::install(missing_bioc, ask = FALSE, update = FALSE)
}

if (!requireNamespace("circlize", quietly = TRUE)) {
  remotes::install_github("jokergoo/circlize", dependencies = TRUE, upgrade = "never")
}

if (!requireNamespace("ComplexHeatmap", quietly = TRUE)) {
  remotes::install_github("jokergoo/ComplexHeatmap", dependencies = TRUE, upgrade = "never")
}

if (!requireNamespace("CellChat", quietly = TRUE)) {
  remotes::install_github("jinworks/CellChat", dependencies = TRUE, upgrade = "never")
}

cat("CellChat v2 installation finished.\n")
cat("Library paths:\n")
print(.libPaths())
