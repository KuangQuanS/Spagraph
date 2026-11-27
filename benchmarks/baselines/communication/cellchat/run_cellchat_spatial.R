args <- commandArgs(trailingOnly = TRUE)

lib_path <- Sys.getenv("CELLCHAT_R_LIB", unset = "")
if (nzchar(lib_path)) {
  dir.create(lib_path, recursive = TRUE, showWarnings = FALSE)
  .libPaths(c(normalizePath(lib_path, winslash = "/", mustWork = FALSE), .libPaths()))
}

parse_args <- function(x) {
  out <- list(
    input_dir = NULL,
    output_dir = NULL,
    min_dominant_fraction = 0,
    min_spots_per_group = 10,
    ratio = NA_real_,
    tol = NA_real_,
    interaction_range = NA_real_,
    contact_range = NA_real_,
    scale_distance = NA_real_,
    nboot = 100,
    workers = 4,
    avg_type = "truncatedMean",
    trim = 0.1,
    contact_dependent = TRUE,
    db_mode = "default"
  )

  i <- 1
  while (i <= length(x)) {
    key <- x[[i]]
    if (i == length(x)) {
      stop(paste("Missing value for argument", key))
    }
    value <- x[[i + 1]]
    if (key == "--input-dir") out$input_dir <- value
    if (key == "--output-dir") out$output_dir <- value
    if (key == "--min-dominant-fraction") out$min_dominant_fraction <- as.numeric(value)
    if (key == "--min-spots-per-group") out$min_spots_per_group <- as.integer(value)
    if (key == "--ratio") out$ratio <- as.numeric(value)
    if (key == "--tol") out$tol <- as.numeric(value)
    if (key == "--interaction-range") out$interaction_range <- as.numeric(value)
    if (key == "--contact-range") out$contact_range <- as.numeric(value)
    if (key == "--scale-distance") out$scale_distance <- as.numeric(value)
    if (key == "--nboot") out$nboot <- as.integer(value)
    if (key == "--workers") out$workers <- as.integer(value)
    if (key == "--avg-type") out$avg_type <- value
    if (key == "--trim") out$trim <- as.numeric(value)
    if (key == "--contact-dependent") out$contact_dependent <- tolower(value) == "true"
    if (key == "--db-mode") out$db_mode <- value
    i <- i + 2
  }

  if (is.null(out$input_dir) || is.null(out$output_dir)) {
    stop(
      paste(
        "Usage: Rscript run_cellchat_spatial.R",
        "--input-dir <dir> --output-dir <dir>",
        "[--min-dominant-fraction 0.6] [--min-spots-per-group 10]",
        "[--ratio 0.38] [--tol 32.5] [--interaction-range 250]",
        "[--contact-range 100] [--scale-distance 0.01]",
        "[--nboot 100] [--workers 4] [--db-mode default|secreted]"
      )
    )
  }

  out
}

opt <- parse_args(args)

suppressPackageStartupMessages({
  library(CellChat)
  library(Matrix)
  library(future)
})

dir.create(opt$output_dir, recursive = TRUE, showWarnings = FALSE)

expr <- readMM(file.path(opt$input_dir, "expression_log1p.mtx"))
expr <- as(expr, "CsparseMatrix")
genes <- read.delim(
  file.path(opt$input_dir, "genes.tsv"),
  header = FALSE,
  stringsAsFactors = FALSE
)[[1]]
spots <- read.delim(
  file.path(opt$input_dir, "spots.tsv"),
  header = FALSE,
  stringsAsFactors = FALSE
)[[1]]
meta <- read.delim(
  file.path(opt$input_dir, "meta.tsv"),
  row.names = 1,
  check.names = FALSE,
  stringsAsFactors = FALSE
)
coords <- read.delim(
  file.path(opt$input_dir, "coordinates.tsv"),
  row.names = 1,
  check.names = FALSE,
  stringsAsFactors = FALSE
)
spatial_factors <- read.delim(
  file.path(opt$input_dir, "spatial_factors.tsv"),
  check.names = FALSE,
  stringsAsFactors = FALSE
)

rownames(expr) <- genes
colnames(expr) <- spots

ratio <- ifelse(is.na(opt$ratio), spatial_factors$ratio[[1]], opt$ratio)
tol <- ifelse(is.na(opt$tol), spatial_factors$tol[[1]], opt$tol)
interaction_range <- ifelse(
  is.na(opt$interaction_range),
  spatial_factors$interaction_range[[1]],
  opt$interaction_range
)
contact_range <- ifelse(
  is.na(opt$contact_range),
  spatial_factors$contact_range[[1]],
  opt$contact_range
)
scale_distance <- ifelse(
  is.na(opt$scale_distance),
  spatial_factors$scale_distance[[1]],
  opt$scale_distance
)

keep <- meta$dominant_fraction >= opt$min_dominant_fraction
meta <- meta[keep, , drop = FALSE]
coords <- coords[rownames(meta), c("x", "y"), drop = FALSE]
expr <- expr[, rownames(meta), drop = FALSE]

group_sizes <- sort(table(meta$labels), decreasing = TRUE)
valid_groups <- names(group_sizes[group_sizes >= opt$min_spots_per_group])
meta <- meta[meta$labels %in% valid_groups, , drop = FALSE]
coords <- coords[rownames(meta), , drop = FALSE]
expr <- expr[, rownames(meta), drop = FALSE]

meta$labels <- factor(meta$labels)
meta$samples <- factor(meta$samples)

if (length(unique(meta$labels)) < 2) {
  stop("Fewer than two cell groups remain after filtering.")
}

db <- CellChatDB.human
if (opt$db_mode == "secreted") {
  db <- subsetDB(db, search = "Secreted Signaling", key = "annotation")
} else {
  db <- subsetDB(db)
}

future::plan("multisession", workers = opt$workers)

cellchat <- createCellChat(
  object = expr,
  meta = meta,
  group.by = "labels",
  datatype = "spatial",
  coordinates = coords,
  spatial.factors = data.frame(ratio = ratio, tol = tol)
)
cellchat@DB <- db
cellchat <- subsetData(cellchat)
cellchat <- identifyOverExpressedGenes(cellchat)
cellchat <- identifyOverExpressedInteractions(cellchat, variable.both = FALSE)
cellchat <- computeCommunProb(
  cellchat,
  type = opt$avg_type,
  trim = opt$trim,
  distance.use = TRUE,
  interaction.range = interaction_range,
  scale.distance = scale_distance,
  contact.dependent = opt$contact_dependent,
  contact.range = contact_range,
  nboot = opt$nboot
)
cellchat <- filterCommunication(cellchat, min.cells = opt$min_spots_per_group)
cellchat <- computeCommunProbPathway(cellchat)
cellchat <- aggregateNet(cellchat)

lr_df <- subsetCommunication(cellchat)
pathway_df <- subsetCommunication(cellchat, slot.name = "netP")

write.csv(
  lr_df,
  file.path(opt$output_dir, "cellchat_lr_communications.csv"),
  quote = TRUE,
  row.names = FALSE
)
write.csv(
  pathway_df,
  file.path(opt$output_dir, "cellchat_pathway_communications.csv"),
  quote = TRUE,
  row.names = FALSE
)
write.csv(
  data.frame(label = names(sort(table(meta$labels), decreasing = TRUE)),
             spots = as.integer(sort(table(meta$labels), decreasing = TRUE))),
  file.path(opt$output_dir, "cellchat_group_sizes.csv"),
  quote = TRUE,
  row.names = FALSE
)
write.csv(
  meta,
  file.path(opt$output_dir, "cellchat_meta_used.csv"),
  quote = TRUE
)
write.csv(
  data.frame(
    ratio = ratio,
    tol = tol,
    interaction_range = interaction_range,
    contact_range = contact_range,
    scale_distance = scale_distance,
    min_dominant_fraction = opt$min_dominant_fraction,
    min_spots_per_group = opt$min_spots_per_group,
    nboot = opt$nboot,
    workers = opt$workers,
    avg_type = opt$avg_type,
    trim = opt$trim,
    contact_dependent = opt$contact_dependent,
    db_mode = opt$db_mode,
    n_spots_used = nrow(meta),
    n_groups_used = length(unique(meta$labels))
  ),
  file.path(opt$output_dir, "run_config.csv"),
  quote = TRUE,
  row.names = FALSE
)
capture.output(sessionInfo(), file = file.path(opt$output_dir, "sessionInfo.txt"))
saveRDS(cellchat, file = file.path(opt$output_dir, "cellchat_object.rds"))
