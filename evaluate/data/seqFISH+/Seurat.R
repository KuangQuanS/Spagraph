# ==============================================================================
# 0. 环境准备与配置
# ==============================================================================
library(Seurat)
library(dplyr)
library(anndata)
library(Matrix)

# --- 【用户配置区域】请根据实际情况修改这里 ---

# 1. 指定文件路径 (注意：Windows路径要把 \ 改为 /)
SC_H5AD_PATH <- "F:\\ST_Graduation_Project\\spagraph_data\\database\\seqFISH+\\scRNA.h5ad"
ST_H5AD_PATH <- "F:\\ST_Graduation_Project\\spagraph_data\\database\\seqFISH+\\Spatial.h5ad"
OUT_FILE     <- "F:/ST_Graduation_Project/spagraph_data/evaluate/seqFISH+/seurat/Imputed_Expression_Igsf21_Rprm.csv"

# 2. 单细胞数据中，存储细胞类型的列名
CELLTYPE_COL <- "celltype"

# 3. 指定需要重建的基因列表 (注意大小写要和数据中完全一致)
TARGET_GENES <- c("Igsf21", "Rprm")

# ==============================================================================
# 1. 辅助函数：将 h5ad 转换为 Seurat 对象
# ==============================================================================
h5ad_to_seurat <- function(h5ad_file, type_name) {
  message(paste0("   [", type_name, "] 正在读取: ", basename(h5ad_file)))
  
  ad <- read_h5ad(h5ad_file)
  
  # 提取矩阵 (优先找 raw counts)
  if ("counts" %in% names(ad$layers)) {
    counts_matrix <- ad$layers[["counts"]]
  } else {
    counts_matrix <- ad$X
  }
  
  # 关键：转置矩阵 (Python: Cells x Genes -> R: Genes x Cells)
  counts_matrix <- t(counts_matrix)
  
  # 对齐行名
  colnames(counts_matrix) <- rownames(ad$obs)
  
  # 创建对象
  obj <- CreateSeuratObject(counts = counts_matrix, meta.data = ad$obs)
  return(obj)
}

# ==============================================================================
# 2. 主流程：基因表达重建
# ==============================================================================

message("\n========================================")
message("🚀 开始执行：单细胞-空间 基因表达重建")
message("========================================")

# --- A. 准备单细胞数据 ---
sc_rna <- h5ad_to_seurat(SC_H5AD_PATH, "scRNA")

# 检查基因是否存在于单细胞数据中
missing_genes <- TARGET_GENES[!TARGET_GENES %in% rownames(sc_rna)]
if (length(missing_genes) > 0) {
  stop(paste("错误：在单细胞数据中找不到以下基因（请检查大小写）:", paste(missing_genes, collapse = ", ")))
}

sc_rna <- SCTransform(sc_rna, verbose = FALSE)

# --- B. 准备空间数据 ---
spatial <- h5ad_to_seurat(ST_H5AD_PATH, "Spatial")
spatial <- SCTransform(spatial, verbose = FALSE)

# --- C. 寻找锚点 (Integration) ---
message("   >>> 正在寻找锚点 (FindTransferAnchors)...")
anchors <- FindTransferAnchors(
  reference = sc_rna,
  query = spatial,
  normalization.method = "SCT",
  dims = 1:30,
  recompute.residuals = FALSE # 节省内存
)

# --- D. 重建指定基因的表达量 ---
message("   >>> 正在重建指定基因表达 (TransferData)...")

# 提取单细胞中这两个基因的表达矩阵作为参考
ref_expression <- GetAssayData(sc_rna, assay = "SCT", slot = "data")[TARGET_GENES, , drop=FALSE]

imputed_expression <- TransferData(
  anchorset = anchors,
  refdata = ref_expression,
  dims = 1:30
)

# --- E. 整理与保存结果 ---

# 提取重建后的矩阵
imputed_matrix <- imputed_expression@data

# 转置矩阵：使行名为 Spot ID，列名为 基因名
final_df <- t(as.matrix(imputed_matrix))

# 保存为 CSV
dir.create(dirname(OUT_FILE), recursive = TRUE, showWarnings = FALSE)
write.csv(final_df, OUT_FILE, row.names = TRUE)

message(paste("   ✅ 成功保存重建结果到:", OUT_FILE))

# ==============================================================================
# 3. 内存清理
# ==============================================================================
rm(sc_rna, spatial, anchors, imputed_expression, ref_expression)
gc()

message("\n🎉 任务处理完毕！")