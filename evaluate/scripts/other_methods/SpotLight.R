# ==============================================================================
# 0. 环境设置与包加载
# ==============================================================================
library(anndata)
library(SPOTlight)
library(Seurat)
library(SingleCellExperiment)
library(Matrix)
library(dplyr)

basedir <- "D:/ST_Graduation_Project_data/database/SimualtedSpatalData"
DATASET_IDS <- 6:32

for (i in DATASET_IDS) {
  cat("========== Data", i, "==========\n")
  
  output_dir <- paste0("D:/ST_Graduation_Project_data/evaluate/Data", i)
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  }
  
  # 文件路径
  sc_file <- paste0(basedir, "/dataset", i, "/scRNA.h5ad")
  st_file <- paste0(basedir, "/dataset", i, "/Spatial.h5ad")
  
  # ==============================================================================
  # 1. 准备单细胞数据 (Reference)
  # ==============================================================================
  cat(">>> 正在加载单细胞数据...\n")
  ad_sc <- read_h5ad(sc_file)
  
  # Cells x Genes -> Genes x Cells
  if ("counts" %in% names(ad_sc$layers)) {
    counts_sc <- t(ad_sc$layers[["counts"]])
  } else {
    counts_sc <- t(ad_sc$X)
  }
  
  sc_meta <- ad_sc$obs
  rownames(sc_meta) <- colnames(counts_sc)
  
  seu_sc <- CreateSeuratObject(counts = counts_sc, meta.data = sc_meta)
  Idents(seu_sc) <- "celltype_final"  # 这里用你自己的列名
  
  seu_sc <- NormalizeData(seu_sc, verbose = FALSE)
  seu_sc <- FindVariableFeatures(seu_sc, selection.method = "vst",
                                 nfeatures = 3000, verbose = FALSE)
  seu_sc <- ScaleData(seu_sc, verbose = FALSE)
  cat(">>> 正在计算 Marker Genes...\n")
  markers <- FindAllMarkers(
    seu_sc,
    only.pos       = TRUE,
    min.pct        = 0.25,
    logfc.threshold = 0.25,
    verbose        = FALSE
  )
  
  sce_sc <- as.SingleCellExperiment(seu_sc)
  
  # ==============================================================================
  # 2. 准备空间数据 (Spatial)
  # ==============================================================================
  cat(">>> 正在加载空间数据...\n")
  ad_st <- read_h5ad(st_file)
  
  if ("counts" %in% names(ad_st$layers)) {
    counts_st <- t(ad_st$layers[["counts"]])
  } else {
    counts_st <- t(ad_st$X)
  }
  counts_st_mat <- as.matrix(counts_st)
  st_genes <- rownames(ad_st$var)
  
  # 如果索引是数字（比如 "0", "1"），那肯定不对，需要去 var 的列里找基因名
  # (假设你知道基因名列叫 'gene_name' 或 'symbol'，这里需要根据你的数据情况调整)
  # if (is.numeric(type.convert(st_genes, as.is=TRUE))) {
  #    if ("gene_name" %in% colnames(ad_st$var)) st_genes <- ad_st$var$gene_name
  # }
  
  # 赋值 (这就保证了 trainNMF 绝对能拿到基因名)
  rownames(counts_st_mat) <- st_genes
  colnames(counts_st_mat) <- rownames(ad_st$obs)
  # ==============================================================================
  # 3. 运行 SPOTlight 反卷积
  # ==============================================================================
  cat(">>> 正在运行 SPOTlight...\n")
  
  valid_types <- unique(markers$cluster)
  sc_types <- unique(as.character(colData(sce_sc)$cell_type))
  missing_types <- setdiff(sc_types, valid_types)
  
  # 只保留在 markers 中出现过的 cell_type
  keep_cells <- as.character(colData(sce_sc)$cell_type) %in% valid_types
  sce_sc_subset <- sce_sc[, keep_cells]
  groups_subset <- as.character(colData(sce_sc_subset)$cell_type)
  
  # 建议 hvg 用基因名向量，如果你现在这样也能跑可先不改
  # hvg_genes <- VariableFeatures(seu_sc)
  
  spotlight_ls <- SPOTlight(
    x      = sce_sc_subset,
    y      = counts_st_mat,
    groups = groups_subset,
    mgs    = markers,
    hvg    = 3000,              # 或者改成 hvg = hvg_genes
    weight_id = "avg_log2FC",
    group_id  = "cluster",
    gene_id   = "gene"
  )
  
  # ==============================================================================
  # 4. 提取结果并保存
  # ==============================================================================
  cat(">>> 正在提取并保存结果...\n")
  
  prop_mat <- spotlight_ls$mat
  
  if (is.null(prop_mat)) {
    print(names(spotlight_ls))
    stop("无法找到结果矩阵，请检查 spotlight_ls 的结构")
  }
  
  write.csv(prop_mat, paste0(output_dir, "/SPOTlight.csv"))
  
  # ==============================================================================
  # 5. 手动释放内存
  # ==============================================================================
  cat(">>> 清理内存...\n")
  rm(
    ad_sc, ad_st,
    counts_sc, counts_st, counts_st_mat,
    sc_meta,
    seu_sc, sce_sc, sce_sc_subset,
    markers,
    valid_types, sc_types, missing_types, keep_cells, groups_subset,
    spotlight_ls, prop_mat
  )
  gc()  # 触发垃圾回收
  
  cat(">>> Data", i, "完成\n\n")
}

#################################################################
library(anndata)
library(SPOTlight)
library(Seurat)
library(SingleCellExperiment)
library(Matrix)
library(dplyr)

basedir <- "D:/ST_Graduation_Project_data/database/STARmap"

# 文件路径
sc_file <- paste0(basedir, "/starmap_sc_rna.h5ad")
st_file <- paste0(basedir, "/STARmap_SP.h5ad")
output_dir <- basedir
# ==============================================================================
# 1. 准备单细胞数据 (Reference)
# ==============================================================================
cat(">>> 正在加载单细胞数据...\n")
ad_sc <- read_h5ad(sc_file)

# Cells x Genes -> Genes x Cells
if ("counts" %in% names(ad_sc$layers)) {
  counts_sc <- t(ad_sc$layers[["counts"]])
} else {
  counts_sc <- t(ad_sc$X)
}

sc_meta <- ad_sc$obs
rownames(sc_meta) <- colnames(counts_sc)

seu_sc <- CreateSeuratObject(counts = counts_sc, meta.data = sc_meta)
Idents(seu_sc) <- "celltype"  # 这里用你自己的列名

seu_sc <- NormalizeData(seu_sc, verbose = FALSE)
seu_sc <- FindVariableFeatures(seu_sc, selection.method = "vst",
                               nfeatures = 3000, verbose = FALSE)
seu_sc <- ScaleData(seu_sc, verbose = FALSE)
cat(">>> 正在计算 Marker Genes...\n")
markers <- FindAllMarkers(
  seu_sc,
  only.pos       = TRUE,
  min.pct        = 0.25,
  logfc.threshold = 0.25,
  verbose        = FALSE
)

sce_sc <- as.SingleCellExperiment(seu_sc)

# ==============================================================================
# 2. 准备空间数据 (Spatial)
# ==============================================================================
cat(">>> 正在加载空间数据...\n")
ad_st <- read_h5ad(st_file)

if ("counts" %in% names(ad_st$layers)) {
  counts_st <- t(ad_st$layers[["counts"]])
} else {
  counts_st <- t(ad_st$X)
}
counts_st_mat <- as.matrix(counts_st)
st_genes <- rownames(ad_st$var)

rownames(counts_st_mat) <- st_genes
colnames(counts_st_mat) <- rownames(ad_st$obs)
# ==============================================================================
# 3. 运行 SPOTlight 反卷积
# ==============================================================================
cat(">>> 正在运行 SPOTlight...\n")

valid_types <- unique(markers$cluster)
sc_types <- unique(as.character(colData(sce_sc)$celltype)) #要改
missing_types <- setdiff(sc_types, valid_types)

# 只保留在 markers 中出现过的 cell_type
keep_cells <- as.character(colData(sce_sc)$celltype) %in% valid_types #要改
sce_sc_subset <- sce_sc[, keep_cells]
groups_subset <- as.character(colData(sce_sc_subset)$celltype) #要改

# 建议 hvg 用基因名向量，如果你现在这样也能跑可先不改
# hvg_genes <- VariableFeatures(seu_sc)

spotlight_ls <- SPOTlight(
  x      = sce_sc_subset,
  y      = counts_st_mat,
  groups = groups_subset,
  mgs    = markers,
  hvg    = 3000,              # 或者改成 hvg = hvg_genes
  weight_id = "avg_log2FC",
  group_id  = "cluster",
  gene_id   = "gene"
)

# ==============================================================================
# 4. 提取结果并保存
# ==============================================================================
cat(">>> 正在提取并保存结果...\n")

prop_mat <- spotlight_ls$mat

if (is.null(prop_mat)) {
  print(names(spotlight_ls))
  stop("无法找到结果矩阵，请检查 spotlight_ls 的结构")
}

write.csv(prop_mat, paste0(output_dir, "/SPOTlight.csv"))

# ==============================================================================
# 5. 手动释放内存
# ==============================================================================
cat(">>> 清理内存...\n")
rm(
  ad_sc, ad_st,
  counts_sc, counts_st, counts_st_mat,
  sc_meta,
  seu_sc, sce_sc, sce_sc_subset,
  markers,
  valid_types, sc_types, missing_types, keep_cells, groups_subset,
  spotlight_ls, prop_mat
)
gc()  # 触发垃圾回收

# 查看单细胞数据的细胞数量
cat("细胞数量 (ncol x): ", ncol(sce_sc_subset), "\n")

# 查看标签向量的长度
cat("标签数量 (length groups): ", length(groups_subset), "\n")