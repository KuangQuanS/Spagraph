args <- commandArgs(trailingOnly = TRUE)

parse_args <- function(x) {
  out <- list(
    input_dir = NULL,
    output_dir = NULL,
    random_iter = 30,
    min_spots_per_group = 1,
    spatial_method = "Delaunay",
    knn_k = 6
  )
  i <- 1
  while (i <= length(x)) {
    key <- x[[i]]
    if (i == length(x)) stop(paste("Missing value for argument", key))
    value <- x[[i + 1]]
    if (key == "--input-dir") out$input_dir <- value
    if (key == "--output-dir") out$output_dir <- value
    if (key == "--random-iter") out$random_iter <- as.integer(value)
    if (key == "--min-spots-per-group") out$min_spots_per_group <- as.integer(value)
    if (key == "--spatial-method") out$spatial_method <- value
    if (key == "--knn-k") out$knn_k <- as.integer(value)
    i <- i + 2
  }
  if (is.null(out$input_dir) || is.null(out$output_dir)) {
    stop("Usage: Rscript run_giotto_spatial_ccc.R --input-dir <dir> --output-dir <dir>")
  }
  out
}

opt <- parse_args(args)
dir.create(opt$output_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(Giotto)
  library(Matrix)
})

expr <- readMM(file.path(opt$input_dir, "expression_raw.mtx"))
expr <- as(expr, "CsparseMatrix")
genes <- read.delim(file.path(opt$input_dir, "genes.tsv"), header = FALSE, stringsAsFactors = FALSE)[[1]]
spots <- read.delim(file.path(opt$input_dir, "spots.tsv"), header = FALSE, stringsAsFactors = FALSE)[[1]]
meta <- read.csv(file.path(opt$input_dir, "meta.csv"), stringsAsFactors = FALSE)
coords <- read.csv(file.path(opt$input_dir, "spatial_locs.csv"), stringsAsFactors = FALSE)
pairs <- read.csv(file.path(opt$input_dir, "shared_simple_pairs.csv"), stringsAsFactors = FALSE)
pairs$LR_comb <- paste(pairs$ligand, pairs$receptor, sep = "-")

rownames(expr) <- genes
colnames(expr) <- spots

meta <- meta[meta$cell_ID %in% spots, , drop = FALSE]
coords <- coords[coords$cell_ID %in% spots, , drop = FALSE]
meta <- meta[match(spots, meta$cell_ID), , drop = FALSE]
coords <- coords[match(spots, coords$cell_ID), , drop = FALSE]

group_sizes <- sort(table(meta$labels), decreasing = TRUE)
valid_groups <- names(group_sizes[group_sizes >= opt$min_spots_per_group])
keep <- meta$labels %in% valid_groups

expr <- expr[, keep, drop = FALSE]
spots <- spots[keep]
meta <- meta[keep, , drop = FALSE]
coords <- coords[keep, , drop = FALSE]

gobject <- createGiottoObject(
  expression = expr,
  spatial_locs = coords[, c("cell_ID", "sdimx", "sdimy")],
  cell_metadata = meta,
  verbose = FALSE
)

gobject <- normalizeGiotto(
  gobject,
  scalefactor = 6000,
  log_norm = TRUE,
  scale_feats = FALSE,
  scale_cells = FALSE,
  verbose = FALSE
)

if (opt$spatial_method == "kNN") {
  gobject <- createSpatialNetwork(
    gobject,
    method = "kNN",
    k = opt$knn_k,
    verbose = FALSE
  )
} else {
  gobject <- createSpatialNetwork(
    gobject,
    method = "Delaunay",
    verbose = FALSE
  )
}

ligands <- pairs$ligand
receptors <- pairs$receptor

exprCC <- exprCellCellcom(
  gobject = gobject,
  cluster_column = "labels",
  random_iter = opt$random_iter,
  feat_set_1 = ligands,
  feat_set_2 = receptors,
  set_seed = TRUE,
  seed_number = 1234,
  verbose = FALSE
)

spatCC <- spatCellCellcom(
  gobject = gobject,
  cluster_column = "labels",
  random_iter = opt$random_iter,
  feat_set_1 = ligands,
  feat_set_2 = receptors,
  min_observations = 1,
  do_parallel = FALSE,
  set_seed = TRUE,
  seed_number = 1234,
  verbose = "none"
)

combCC <- combCCcom(
  spatialCC = spatCC,
  exprCC = exprCC,
  min_lig_nr = 1,
  min_rec_nr = 1,
  min_padj_value = 1,
  min_log2fc = 0,
  min_av_diff = 0
)

write.csv(exprCC, file.path(opt$output_dir, "giotto_exprCC.csv"), row.names = FALSE)
write.csv(spatCC, file.path(opt$output_dir, "giotto_spatCC.csv"), row.names = FALSE)
write.csv(combCC, file.path(opt$output_dir, "giotto_combCC.csv"), row.names = FALSE)

shared_pairs <- unique(pairs$LR_comb)
combCC$interaction_name <- combCC$LR_comb
combCC_filtered <- combCC[combCC$LR_comb %in% shared_pairs, , drop = FALSE]
write.csv(combCC_filtered, file.path(opt$output_dir, "giotto_combCC_shared_pairs.csv"), row.names = FALSE)

pair_summary <- aggregate(
  cbind(PI_spat, LR_expr_spat, p.adj_spat, PI, LR_expr) ~ LR_comb,
  data = combCC_filtered,
  FUN = max
)
colnames(pair_summary)[1] <- "LR_comb"
pair_summary <- merge(pair_summary, pairs, by = "LR_comb", all.x = TRUE)
pair_summary$interaction_name <- pair_summary$interaction_name
pair_summary <- pair_summary[order(-pair_summary$PI_spat, -pair_summary$LR_expr_spat), ]
pair_summary$giotto_spatial_rank <- rank(-pair_summary$PI_spat, ties.method = "min")
pair_summary$giotto_expr_rank <- rank(-pair_summary$PI, ties.method = "min")
write.csv(pair_summary, file.path(opt$output_dir, "giotto_pair_summary.csv"), row.names = FALSE)

config <- data.frame(
  random_iter = opt$random_iter,
  min_spots_per_group = opt$min_spots_per_group,
  spatial_method = opt$spatial_method,
  knn_k = opt$knn_k,
  n_spots_used = ncol(expr),
  n_groups_used = length(unique(meta$labels)),
  n_input_pairs = length(shared_pairs)
)
write.csv(config, file.path(opt$output_dir, "run_config.csv"), row.names = FALSE)
capture.output(sessionInfo(), file = file.path(opt$output_dir, "sessionInfo.txt"))
