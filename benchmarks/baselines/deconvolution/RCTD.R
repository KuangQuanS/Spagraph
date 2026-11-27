library(anndata)
library(RCTD)
library(Matrix)

# 1. 设置路径
#sc_file <- "./SimualtedSpatalData/dataset1/scRNA.h5ad"
#st_file <- "./SimualtedSpatalData/dataset1/Spatial.h5ad"

sc_file <- "./seqFISH+/starmap_sc_rna.h5ad"
st_file <- "./seqFISH+/seqFISH_SP.h5ad"

#sc_file <- "./GSE144236/GSE144236_P2_SC.h5ad"
#st_file <- "./GSE144236/GSE144236_P2_ST.h5ad"

# ================= 准备单细胞数据 =================
ad_sc <- read_h5ad(sc_file)

# 转置并确保是稀疏矩阵
if ("counts" %in% names(ad_sc$layers)) {
  counts_sc <- t(ad_sc$layers[["counts"]])
} else {
  counts_sc <- t(ad_sc$X)
}


# 提取细胞类型
cell_types <- setNames(ad_sc$obs$celltype, rownames(ad_sc$obs))

# ---【关键补充】过滤少于 25 个细胞的类型 (来自代码 A 的逻辑) ---
cell_type_counts <- table(cell_types)
valid_types <- names(cell_type_counts)[cell_type_counts >= 25]
cells_to_keep <- names(cell_types)[cell_types %in% valid_types]

# 更新过滤后的数据
counts_sc <- counts_sc[, cells_to_keep]
cell_types <- cell_types[cells_to_keep]
cell_types <- as.factor(cell_types)

levels(cell_types) <- gsub("/", "_", levels(cell_types))

# 如果还有其它不安全字符，也可以一起替换（可选）
levels(cell_types) <- gsub("\\s+", "_", levels(cell_types))   # 空格 -> _
levels(cell_types) <- gsub("[^A-Za-z0-9_.-]", "_", levels(cell_types))  # 其它异常字符 -> _

# 再次清理空 level
cell_types <- droplevels(cell_types)

counts_sc <- as(counts_sc, "dgCMatrix")
#counts_sc <- as(as.matrix(counts_sc), "dgCMatrix")
cell_types <- droplevels(as.factor(cell_types))
reference <- Reference(counts = counts_sc, cell_types = cell_types)

# ================= 准备空间数据 =================
ad_st <- read_h5ad(st_file)

if ("counts" %in% names(ad_st$layers)) {
  counts_st <- t(ad_st$layers[["counts"]])
} else {
  counts_st <- t(ad_st$X)
}
# 强制转为稀疏矩阵
# counts_st <- as(counts_st, "dgCMatrix")

# ---【插入这行修复代码】---
# 强制将矩阵转换为 RCTD 唯一认可的标准稀疏矩阵格式
counts_st <- as(counts_st, "dgCMatrix")

coords_mat <- ad_st$obsm[["spatial"]]
coords <- as.data.frame(coords_mat)
rownames(coords) <- rownames(ad_st$obs)
# 创建 Spatial 对象
puck <- SpatialRNA(counts = counts_st, coords = coords)

# ================= 运行 RCTD =================
myRCTD <- create.RCTD(puck, reference, max_cores = 4)

# 【重要修改】使用 'full' 模式 (来自代码 A，适合 Visium/ST)
# 如果你是 Slide-seq，才用 'doublet'
myRCTD <- run.RCTD(myRCTD, doublet_mode = 'full')

# ================= 结果输出 =================
results <- myRCTD@results
# 归一化
norm_weights <- sweep(results$weights, 1, rowSums(results$weights), '/')

# 保存
write.csv(as.matrix(norm_weights), "seqFISH_RCTD.csv")

###########################################################################
###########################################################################
library(anndata)
library(RCTD)
library(Matrix)

# ================= 循环设置 =================
base_dir <- "./SimualtedSpatalData"
# 循环范围：从 1 到 32
dataset_ids <- 1:32

for (i in dataset_ids) {
  
  # 1. 动态构建路径
  dataset_name <- paste0("dataset", i)
  current_dir <- file.path(base_dir, dataset_name)
  
  sc_file <- file.path(current_dir, "scRNA.h5ad")
  st_file <- file.path(current_dir, "Spatial.h5ad")
  
  # 输出文件名 (存放在各自的文件夹下，或者你可以指定统一目录)
  output_file <- file.path(current_dir, "RCTD_results.csv")
  
  message(paste("\n========================================"))
  message(paste("正在处理:", dataset_name, "(", i, "/", length(dataset_ids), ")"))
  message(paste("SC路径:", sc_file))
  message(paste("ST路径:", st_file))
  
  # 检查文件是否存在，不存在则跳过
  if (!file.exists(sc_file) || !file.exists(st_file)) {
    warning(paste("跳过: 文件缺失 ->", dataset_name))
    next
  }
  
  # 使用 tryCatch 捕获可能的错误，防止单个数据报错导致循环中断
  tryCatch({
    
    # ================= 准备单细胞数据 =================
    ad_sc <- read_h5ad(sc_file)
    
    # 转置并确保是稀疏矩阵
    if ("counts" %in% names(ad_sc$layers)) {
      counts_sc <- t(ad_sc$layers[["counts"]])
    } else {
      counts_sc <- t(ad_sc$X)
    }
    
    # 提取细胞类型
    cell_types <- setNames(ad_sc$obs$celltype_final, rownames(ad_sc$obs))
    
    # --- 过滤少于 25 个细胞的类型 ---
    cell_type_counts <- table(cell_types)
    valid_types <- names(cell_type_counts)[cell_type_counts >= 25]
    
    # 如果过滤后没有剩下的类型，报错跳过
    if (length(valid_types) == 0) stop("所有细胞类型的细胞数都少于25，无法继续。")
    
    cells_to_keep <- names(cell_types)[cell_types %in% valid_types]
    
    # 更新过滤后的数据
    counts_sc <- counts_sc[, cells_to_keep]
    cell_types <- cell_types[cells_to_keep]
    cell_types <- as.factor(cell_types)
    cell_types <- droplevels(as.factor(cell_types))
    
    # 构建 Reference
    reference <- Reference(counts = counts_sc, cell_types = cell_types)
    
    # ================= 准备空间数据 =================
    ad_st <- read_h5ad(st_file)
    
    if ("counts" %in% names(ad_st$layers)) {
      counts_st <- t(ad_st$layers[["counts"]])
    } else {
      counts_st <- t(ad_st$X)
    }
    
    # 构造伪坐标 (针对模拟数据)
    n_spots <- ncol(counts_st)
    fake_coords <- data.frame(x = 1:n_spots, y = rep(1, n_spots))
    rownames(fake_coords) <- colnames(counts_st)
    counts_st <- as(counts_st, "dgCMatrix")
    # 创建 Spatial 对象
    puck <- SpatialRNA(counts = counts_st, coords = fake_coords)
    
    # ================= 运行 RCTD =================
    # max_cores 可以根据你服务器的配置调整
    myRCTD <- create.RCTD(puck, reference, max_cores = 4)
    
    # 使用 full 模式
    myRCTD <- run.RCTD(myRCTD, doublet_mode = 'full')
    
    # ================= 结果输出 =================
    results <- myRCTD@results
    # 归一化权重
    norm_weights <- sweep(results$weights, 1, rowSums(results$weights), '/')
    
    # 保存结果到 csv
    write.csv(as.matrix(norm_weights), output_file)
    message(paste("成功保存:", output_file))
    
  }, error = function(e) {
    # 如果出错，打印错误信息但不停止循环
    message(paste("❌ 错误发生在", dataset_name, ":", e$message))
  })
  
  # ================= 内存清理 =================
  # R 处理大量循环时内存容易爆，手动清理很重要
  rm(ad_sc, counts_sc, reference, ad_st, counts_st, puck, myRCTD, results, norm_weights)
  gc() 
}

message("\n所有任务处理完毕！")