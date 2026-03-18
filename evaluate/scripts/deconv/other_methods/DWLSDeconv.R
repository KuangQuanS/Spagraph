# ==============================================================================
# 0. 加载包与配置
# ==============================================================================
library(Giotto)
library(anndata)
library(Matrix)
library(data.table)

# --- 配置区域 ---
BASE_DIR       <- "D:/ST_Graduation_Project_data/database/SimualtedSpatalData"
CELLTYPE_COL   <- "celltype_final"
MY_PYTHON_PATH <- "E:/python/python.exe"

# ✅ 循环范围: 1 到 31
DATASET_IDS    <- 24:31

# ==============================================================================
# 1. 初始化 Giotto (只运行一次)
# ==============================================================================
instrs <- createGiottoInstructions(
  python_path = MY_PYTHON_PATH,
  show_plot   = FALSE,
  return_plot = FALSE,
  save_plot   = FALSE
)

# ==============================================================================
# 2. 批量循环
# ==============================================================================

for (i in DATASET_IDS) {
  
  # --- 动态生成路径 ---
  dataset_name <- paste0("dataset", i)
  current_dir  <- file.path(BASE_DIR, dataset_name)
  
  sc_path  <- file.path(current_dir, "scRNA.h5ad")
  st_path  <- file.path(current_dir, "Spatial.h5ad")
  out_file <- file.path(current_dir, "SpatialDWLS_result.csv")
  
  message(paste("\n========================================"))
  message(paste(">>> 正在处理:", dataset_name, "(", i, "/ 31 )"))
  message(paste(">>> 目录:", current_dir))
  
  # 检查文件
  if (!file.exists(sc_path) || !file.exists(st_path)) {
    warning(paste("跳过:", dataset_name, "文件缺失"))
    next
  }
  
  # 使用 tryCatch 防止报错中断
  tryCatch({
    
    # ==========================================================================
    # [1/5] 读取 Spatial 数据 (ST)
    # ==========================================================================
    message("   >>> [1/5] 读取 Spatial 数据...")
    ad_st <- read_h5ad(st_path)
    
    if ("counts" %in% names(ad_st$layers)) {
      st_counts <- ad_st$layers[["counts"]]
    } else {
      st_counts <- ad_st$X
    }
    st_counts <- t(st_counts)
    
    st_meta <- as.data.frame(ad_st$obs)
    st_meta$cell_ID <- rownames(st_meta)
    
    if ("spatial" %in% names(ad_st$obsm)) {
      st_locs <- as.data.frame(ad_st$obsm[["spatial"]])
      colnames(st_locs) <- c("sdimx", "sdimy")
      st_locs$cell_ID <- rownames(st_meta)
    } else {
      st_locs <- data.frame(
        sdimx = runif(ncol(st_counts), 0, 100),
        sdimy = runif(ncol(st_counts), 0, 100),
        cell_ID = colnames(st_counts)
      )
    }
    
    st_data <- createGiottoObject(
      raw_exprs     = st_counts,
      cell_metadata = st_meta,
      spatial_locs  = st_locs,
      instructions  = instrs
    )
    
    st_data <- normalizeGiotto(st_data)
    st_data <- calculateHVF(st_data, show_plot=FALSE, return_plot=FALSE)
    
    gm_st <- getFeatureMetadata(st_data, output = "data.table")
    hvf_col <- intersect(c("hvf", "HVG", "hvg"), colnames(gm_st))[1]
    feat_col <- intersect(c("feat_ID", "gene_ID", "feats"), colnames(gm_st))[1]
    
    if (!is.na(hvf_col)) {
      featgenes_st <- gm_st[get(hvf_col) %in% c("yes", TRUE)][[feat_col]]
    } else {
      featgenes_st <- gm_st[[feat_col]][1:2000]
    }
    
    st_data <- runPCA(st_data, feats_to_use = featgenes_st, scale_unit = FALSE)
    st_data <- runUMAP(st_data, dimensions_to_use = 1:10)
    st_data <- createNearestNetwork(st_data, dimensions_to_use = 1:10, k = 15)
    st_data <- doLeidenCluster(st_data, resolution = 0.4, n_iterations = 1000, name = "leiden_clus")
    
    # ==========================================================================
    # [2/5] 准备 scRNA 数据 (SC)
    # ==========================================================================
    message("   >>> [2/5] 读取 scRNA 数据...")
    ad_sc <- read_h5ad(sc_path)
    
    if ("counts" %in% names(ad_sc$layers)) {
      sc_counts <- ad_sc$layers[["counts"]]
    } else {
      sc_counts <- ad_sc$X
    }
    sc_counts <- t(sc_counts)
    
    sc_meta <- as.data.frame(ad_sc$obs)
    sc_meta$cell_ID <- rownames(sc_meta)
    
    sc_locs <- data.frame(
      sdimx = runif(ncol(sc_counts), 0, 100),
      sdimy = runif(ncol(sc_counts), 0, 100),
      cell_ID = colnames(sc_counts)
    )
    
    sc_data <- createGiottoObject(
      raw_exprs     = sc_counts,
      cell_metadata = sc_meta,
      spatial_locs  = sc_locs,
      instructions  = instrs
    )
    
    sc_data <- normalizeGiotto(sc_data)
    sc_data <- calculateHVF(sc_data, show_plot=FALSE, return_plot=FALSE)
    
    gm_sc <- getFeatureMetadata(sc_data, output = "data.table")
    hvf_col_sc <- intersect(c("hvf", "HVG", "hvg"), colnames(gm_sc))[1]
    feat_col_sc <- intersect(c("feat_ID", "gene_ID", "feats"), colnames(gm_sc))[1]
    
    if (!is.na(hvf_col_sc)) {
      featgenes_sc <- gm_sc[get(hvf_col_sc) %in% c("yes", TRUE)][[feat_col_sc]]
    } else {
      featgenes_sc <- gm_sc[[feat_col_sc]][1:2000]
    }
    
    sc_data <- runPCA(sc_data, feats_to_use = featgenes_sc, scale_unit = FALSE)
    
    # ==========================================================================
    # [3/5] 构建 Signature Matrix
    # ==========================================================================
    message("   >>> [3/5] 构建签名矩阵 (Signature Matrix)...")
    
    if (!CELLTYPE_COL %in% colnames(sc_meta)) {
      stop(paste("报错: scRNA 数据里找不到列:", CELLTYPE_COL))
    }
    
    sc_data <- addCellMetadata(
      sc_data,
      new_metadata = as.character(sc_meta[[CELLTYPE_COL]]),
      vector_name  = "leiden_clus" 
    )
    
    message("       正在计算 scran markers...")
    scran_markers <- findMarkers_one_vs_all(
      gobject           = sc_data,
      method            = "scran", # ✅ 保持你的 scran
      expression_values = "normalized",
      cluster_column    = "leiden_clus",
      min_feats         = 5,
      verbose           = FALSE
    )
    
    dt_markers <- as.data.table(scran_markers)
    gene_col_mk <- intersect(c("feats", "genes", "feat_ID"), colnames(dt_markers))[1]
    top_dt <- dt_markers[, head(.SD, 100), by = "cluster"] # ✅ 保持你的 Top 100
    Sig_scran <- unique(top_dt[[gene_col_mk]])
    
    message("       正在计算均值矩阵...")
    expr_mat <- getExpression(sc_data, values = "normalized", output = "matrix")
    Sig_scran <- intersect(Sig_scran, rownames(expr_mat))
    
    ExprSubset <- expr_mat[Sig_scran, , drop = FALSE]
    linear_subset <- 2^(ExprSubset) - 1
    
    id_vec <- getCellMetadata(sc_data)$leiden_clus
    unique_ids <- unique(id_vec)
    
    Sig_exp <- matrix(NA, nrow = length(Sig_scran), ncol = length(unique_ids))
    rownames(Sig_exp) <- Sig_scran
    colnames(Sig_exp) <- unique_ids
    
    # ✅ 保持你原本的循环逻辑 (含 drop fix)
    for (uid in unique_ids) {
      idx <- which(id_vec == uid)
      if (length(idx) > 1) {
        Sig_exp[, uid] <- rowMeans(linear_subset[, idx, drop = FALSE])
      } else {
        Sig_exp[, uid] <- as.vector(linear_subset[, idx, drop = TRUE])
      }
    }
    
    # ==========================================================================
    # [4/5] 运行 DWLS 反卷积
    # ==========================================================================
    message("   >>> [4/5] 运行 DWLS 反卷积...")
    
    st_data <- runDWLSDeconv(
      gobject     = st_data,
      sign_matrix = Sig_exp,
      n_cell      = 20,
      name        = "DWLS"
    )
    
    # ==========================================================================
    # [5/5] 保存结果
    # ==========================================================================
    message("   >>> [5/5] 保存结果...")
    
    res <- getSpatialEnrichment(
      gobject = st_data,
      name    = "DWLS",
      output  = "data.table"
    )
    
    # ✅ 保持 row.names = FALSE
    write.csv(res, out_file, row.names = FALSE)
    message(paste("   ✅ 成功保存:", out_file))
    
  }, error = function(e) {
    message(paste("   ❌ 失败:", dataset_name))
    message(paste("   错误:", e$message))
  })
  
  # 内存清理 (加在循环末尾是必须的，否则跑几个就崩了)
  rm(list = c("st_data", "sc_data", "ad_st", "ad_sc", "st_counts", "sc_counts", "expr_mat", "Sig_exp"))
  gc()
}

message("\n🎉 1-31 所有任务全部完成！")

##################################################################################

basedir <- "D:/ST_Graduation_Project_data/database/STARmap"

# 文件路径
sc_path <- paste0(basedir, "/starmap_sc_rna.h5ad")
st_path <- paste0(basedir, "/STARmap_SP.h5ad")
out_file <- paste0(basedir, "SpatialDWLS_result.csv")
CELLTYPE_COL <- "celltype"
message("   >>> [1/5] 读取 Spatial 数据...")
ad_st <- read_h5ad(st_path)

if ("counts" %in% names(ad_st$layers)) {
  st_counts <- ad_st$layers[["counts"]]
} else {
  st_counts <- ad_st$X
}
st_counts <- t(st_counts)

st_meta <- as.data.frame(ad_st$obs)
st_meta$cell_ID <- rownames(st_meta)

if ("spatial" %in% names(ad_st$obsm)) {
  st_locs <- as.data.frame(ad_st$obsm[["spatial"]])
  colnames(st_locs) <- c("sdimx", "sdimy")
  st_locs$cell_ID <- rownames(st_meta)
} else {
  st_locs <- data.frame(
    sdimx = runif(ncol(st_counts), 0, 100),
    sdimy = runif(ncol(st_counts), 0, 100),
    cell_ID = colnames(st_counts)
  )
}
rm(ad_st)
st_data <- createGiottoObject(
  raw_exprs     = st_counts,
  cell_metadata = st_meta,
  spatial_locs  = st_locs,
  instructions  = instrs
)

st_data <- normalizeGiotto(st_data)
st_data <- calculateHVF(st_data, show_plot=FALSE, return_plot=FALSE)

gm_st <- getFeatureMetadata(st_data, output = "data.table")
hvf_col <- intersect(c("hvf", "HVG", "hvg"), colnames(gm_st))[1]
feat_col <- intersect(c("feat_ID", "gene_ID", "feats"), colnames(gm_st))[1]

if (!is.na(hvf_col)) {
  featgenes_st <- gm_st[get(hvf_col) %in% c("yes", TRUE)][[feat_col]]
} else {
  featgenes_st <- gm_st[[feat_col]][1:2000]
}

st_data <- runPCA(st_data, feats_to_use = featgenes_st, scale_unit = FALSE)
st_data <- runUMAP(st_data, dimensions_to_use = 1:10)
st_data <- createNearestNetwork(st_data, dimensions_to_use = 1:10, k = 15)
st_data <- doLeidenCluster(st_data, resolution = 0.4, n_iterations = 1000, name = "leiden_clus")

# ==========================================================================
# [2/5] 准备 scRNA 数据 (SC)
# ==========================================================================
message("   >>> [2/5] 读取 scRNA 数据...")
ad_sc <- read_h5ad(sc_path)

if ("counts" %in% names(ad_sc$layers)) {
  sc_counts <- ad_sc$layers[["counts"]]
} else {
  sc_counts <- ad_sc$X
}
sc_counts <- t(sc_counts)

sc_meta <- as.data.frame(ad_sc$obs)
sc_meta$cell_ID <- rownames(sc_meta)
rm(ad_sc)
sc_locs <- data.frame(
  sdimx = runif(ncol(sc_counts), 0, 100),
  sdimy = runif(ncol(sc_counts), 0, 100),
  cell_ID = colnames(sc_counts)
)

sc_data <- createGiottoObject(
  raw_exprs     = sc_counts,
  cell_metadata = sc_meta,
  spatial_locs  = sc_locs,
  instructions  = instrs
)

sc_data <- normalizeGiotto(sc_data)
sc_data <- calculateHVF(sc_data, show_plot=FALSE, return_plot=FALSE)

gm_sc <- getFeatureMetadata(sc_data, output = "data.table")
hvf_col_sc <- intersect(c("hvf", "HVG", "hvg"), colnames(gm_sc))[1]
feat_col_sc <- intersect(c("feat_ID", "gene_ID", "feats"), colnames(gm_sc))[1]

if (!is.na(hvf_col_sc)) {
  featgenes_sc <- gm_sc[get(hvf_col_sc) %in% c("yes", TRUE)][[feat_col_sc]]
} else {
  featgenes_sc <- gm_sc[[feat_col_sc]][1:2000]
}

sc_data <- runPCA(sc_data, feats_to_use = featgenes_sc, scale_unit = FALSE)

# ==========================================================================
# [3/5] 构建 Signature Matrix
# ==========================================================================
message("   >>> [3/5] 构建签名矩阵 (Signature Matrix)...")

if (!CELLTYPE_COL %in% colnames(sc_meta)) {
  stop(paste("报错: scRNA 数据里找不到列:", CELLTYPE_COL))
}

sc_data <- addCellMetadata(
  sc_data,
  new_metadata = as.character(sc_meta[[CELLTYPE_COL]]),
  vector_name  = "leiden_clus" 
)

message("       正在计算 scran markers...")
scran_markers <- findMarkers_one_vs_all(
  gobject           = sc_data,
  method            = "scran", # ✅ 保持你的 scran
  expression_values = "normalized",
  cluster_column    = "leiden_clus",
  min_feats         = 5,
  verbose           = FALSE
)

dt_markers <- as.data.table(scran_markers)
gene_col_mk <- intersect(c("feats", "genes", "feat_ID"), colnames(dt_markers))[1]
top_dt <- dt_markers[, head(.SD, 100), by = "cluster"] # ✅ 保持你的 Top 100
Sig_scran <- unique(top_dt[[gene_col_mk]])

message("       正在计算均值矩阵...")
expr_mat <- getExpression(sc_data, values = "normalized", output = "matrix")
Sig_scran <- intersect(Sig_scran, rownames(expr_mat))

ExprSubset <- expr_mat[Sig_scran, , drop = FALSE]
linear_subset <- 2^(ExprSubset) - 1

id_vec <- getCellMetadata(sc_data)$leiden_clus
unique_ids <- unique(id_vec)

Sig_exp <- matrix(NA, nrow = length(Sig_scran), ncol = length(unique_ids))
rownames(Sig_exp) <- Sig_scran
colnames(Sig_exp) <- unique_ids

# ✅ 保持你原本的循环逻辑 (含 drop fix)
for (uid in unique_ids) {
  idx <- which(id_vec == uid)
  if (length(idx) > 1) {
    Sig_exp[, uid] <- rowMeans(linear_subset[, idx, drop = FALSE])
  } else {
    Sig_exp[, uid] <- as.vector(linear_subset[, idx, drop = TRUE])
  }
}

# ==========================================================================
# [4/5] 运行 DWLS 反卷积
# ==========================================================================
message("   >>> [4/5] 运行 DWLS 反卷积...")

st_data <- runDWLSDeconv(
  gobject     = st_data,
  sign_matrix = Sig_exp,
  n_cell      = 20,
  name        = "DWLS"
)

# ==========================================================================
# [5/5] 保存结果
# ==========================================================================
message("   >>> [5/5] 保存结果...")

res <- getSpatialEnrichment(
  gobject = st_data,
  name    = "DWLS",
  output  = "data.table"
)


write.csv(res, out_file, row.names = FALSE)

# 内存清理 (加在循环末尾是必须的，否则跑几个就崩了)
rm(list = c("st_data", "sc_data", "ad_st", "ad_sc", "st_counts", "sc_counts", "expr_mat", "Sig_exp"))
gc()