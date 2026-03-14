# ==============================================================================
# 0. 环境准备与配置
# ==============================================================================
library(Seurat)
library(dplyr)
library(anndata)
library(Matrix)

# --- 【用户配置区域】请根据实际情况修改这里 ---

# 1. 数据根目录 (注意：Windows路径要把 \ 改为 / )
BASE_DIR <- "D:/ST_Graduation_Project_data/database/SimualtedSpatalData"

# 2. 数据集范围
DATASET_IDS <- 1:32

# 3. 单细胞数据中，存储细胞类型的列名 (非常重要！请确认你的 h5ad 里这一列叫什么)
# 常见的名字: "celltype", "cell_type", "annotation", "Cluster"
CELLTYPE_COL <- "celltype_final" 

# ==============================================================================
# 1. 辅助函数：将 h5ad 转换为 Seurat 对象
# ==============================================================================
h5ad_to_seurat <- function(h5ad_file, type_name) {
  message(paste0("   [", type_name, "] 正在读取: ", basename(h5ad_file)))
  
  # 读取 h5ad
  ad <- read_h5ad(h5ad_file)
  
  # 提取矩阵 (优先找 raw counts)
  if ("counts" %in% names(ad$layers)) {
    counts_matrix <- ad$layers[["counts"]]
  } else {
    counts_matrix <- ad$X
  }
  
  # 关键：转置矩阵 (Python: Cells x Genes -> R: Genes x Cells)
  counts_matrix <- t(counts_matrix)
  
  # 提取元数据
  meta_data <- ad$obs
  
  # 对齐行名
  colnames(counts_matrix) <- rownames(meta_data)
  
  # 创建对象
  obj <- CreateSeuratObject(counts = counts_matrix, meta.data = meta_data)
  return(obj)
}

# ==============================================================================
# 2. 开始批量循环
# ==============================================================================

for (i in DATASET_IDS) {
  
  # 动态构建当前数据集的路径
  dataset_name <- paste0("dataset", i)
  current_dir <- file.path(BASE_DIR, dataset_name)
  
  data_name <- paste0("Data", i)
  out_dir <- file.path("D:/ST_Graduation_Project_data/evaluate", data_name)
  
  sc_path <- file.path(current_dir, "scRNA.h5ad")
  st_path <- file.path(current_dir, "Spatial.h5ad")
  out_file <- file.path(out_dir, "Seurat.csv")
  
  message(paste("\n========================================"))
  message(paste("正在处理:", dataset_name, "(", i, "/ 32 )"))
  message(paste("目录:", current_dir))
  
  # 检查文件是否存在
  if (!file.exists(sc_path) || !file.exists(st_path)) {
    warning(paste("跳过: 文件缺失 ->", dataset_name))
    next
  }
  
  # 使用 tryCatch 捕获错误，防止单个数据报错中断整个循环
  tryCatch({
    
    # --- A. 准备单细胞数据 ---
    sc_rna <- h5ad_to_seurat(sc_path, "scRNA")
    
    # 检查细胞类型列是否存在
    if (!CELLTYPE_COL %in% colnames(sc_rna@meta.data)) {
      stop(paste0("错误：在单细胞数据中找不到列名 '", CELLTYPE_COL, "'。请检查配置区域！"))
    }
    
    sc_rna <- SCTransform(sc_rna, verbose = FALSE)

    # --- B. 准备空间数据 ---
    spatial <- h5ad_to_seurat(st_path, "Spatial")
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
    
    # --- D. 标签转移 (Prediction) ---
    message("   >>> 正在转移标签 (TransferData)...")
    predictions <- TransferData(
      anchorset = anchors,
      refdata = sc_rna@meta.data[, CELLTYPE_COL],
      dims = 1:30
    )
    
    # --- E. 保存结果 (修正版) ---
    
    # 1. 提取所有打分列 (此时包含 max)
    score_cols <- grep("^prediction\\.score\\.", colnames(predictions), value = TRUE)
    
    # 2. 【修正点1】手动剔除 "prediction.score.max" 这一列
    score_cols <- setdiff(score_cols, "prediction.score.max")
    
    # 3. 提取矩阵
    result_matrix <- predictions[, score_cols]
    
    # 4. 【修正点2】修复列名 (处理空格变点的问题)
    # 方法：先去掉前缀，然后尝试把连续的点(.)变回空格，或者根据你的实际情况调整
    # Seurat 内部把 "T cells" 变成了 "T.cells"，"B-cells" 变成了 "B.cells"
    
    # 第一步：去掉固定的前缀 "prediction.score."
    clean_names <- gsub("^prediction\\.score\\.", "", colnames(result_matrix))
    
    # 第二步：如果你确定原始名字里是空格而不是点，可以在这里把点替换回空格
    # (注意：如果原始名字里本来就有点，这步可能会误伤，视情况而定)
    # clean_names <- gsub("\\.", " ", clean_names) 
    
    # 赋值回矩阵
    colnames(result_matrix) <- clean_names
    
    # 5. 保存 (row.names = TRUE 保留 Spot ID)
    write.csv(result_matrix, out_file, row.names = TRUE)
    message(paste("   ✅ 成功保存:", out_file))
    
  }, error = function(e) {
    # 报错处理模块
    message(paste("   ❌ 处理失败:", dataset_name))
    message(paste("   错误信息:", e$message))
  })
  
  # ==============================================================================
  # 3. 内存清理 (非常重要)
  # ==============================================================================
  # 删除本轮循环的大对象
  if (exists("sc_rna")) rm(sc_rna)
  if (exists("spatial")) rm(spatial)
  if (exists("anchors")) rm(anchors)
  if (exists("predictions")) rm(predictions)
  
  # 强制进行垃圾回收
  gc()
}

message("\n🎉 所有任务处理完毕！")

######################################################################################
basedir <- "D:/ST_Graduation_Project_data/database/STARmap"

# 文件路径
sc_path <- paste0(basedir, "/starmap_sc_rna.h5ad")
st_path <- paste0(basedir, "/STARmap_SP.h5ad")
out_file <- paste0(basedir, "Seurat.csv")
CELLTYPE_COL <- "celltype"
# --- A. 准备单细胞数据 ---
sc_rna <- h5ad_to_seurat(sc_path, "scRNA")

# 检查细胞类型列是否存在
if (!CELLTYPE_COL %in% colnames(sc_rna@meta.data)) {
  stop(paste0("错误：在单细胞数据中找不到列名 '", CELLTYPE_COL, "'。请检查配置区域！"))
}

sc_rna <- SCTransform(sc_rna, verbose = FALSE)

# --- B. 准备空间数据 ---
spatial <- h5ad_to_seurat(st_path, "Spatial")
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

# --- D. 标签转移 (Prediction) ---
message("   >>> 正在转移标签 (TransferData)...")
predictions <- TransferData(
  anchorset = anchors,
  refdata = sc_rna@meta.data[, CELLTYPE_COL],
  dims = 1:30
)

# --- E. 保存结果 (修正版) ---

# 1. 提取所有打分列 (此时包含 max)
score_cols <- grep("^prediction\\.score\\.", colnames(predictions), value = TRUE)

# 2. 【修正点1】手动剔除 "prediction.score.max" 这一列
score_cols <- setdiff(score_cols, "prediction.score.max")

# 3. 提取矩阵
result_matrix <- predictions[, score_cols]

# 4. 【修正点2】修复列名 (处理空格变点的问题)
# 方法：先去掉前缀，然后尝试把连续的点(.)变回空格，或者根据你的实际情况调整
# Seurat 内部把 "T cells" 变成了 "T.cells"，"B-cells" 变成了 "B.cells"

# 第一步：去掉固定的前缀 "prediction.score."
clean_names <- gsub("^prediction\\.score\\.", "", colnames(result_matrix))

# 第二步：如果你确定原始名字里是空格而不是点，可以在这里把点替换回空格
# (注意：如果原始名字里本来就有点，这步可能会误伤，视情况而定)
# clean_names <- gsub("\\.", " ", clean_names) 

# 赋值回矩阵
colnames(result_matrix) <- clean_names

# 5. 保存 (row.names = TRUE 保留 Spot ID)
write.csv(result_matrix, "D:/ST_Graduation_Project_data/evaluate/Seurat.csv", row.names = TRUE)

# ==============================================================================
# 3. 内存清理 (非常重要)
# ==============================================================================
# 删除本轮循环的大对象
if (exists("sc_rna")) rm(sc_rna)
if (exists("spatial")) rm(spatial)
if (exists("anchors")) rm(anchors)
if (exists("predictions")) rm(predictions)

# 强制进行垃圾回收
gc()