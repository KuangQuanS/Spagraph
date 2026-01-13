#!/usr/bin/env python3
"""
GSE211956 Dataset Analysis Pipeline
Run on server with: python run_GSE211956.py
"""

import spagraph as spg
import torch
import gc

def clear_gpu():
    """清理GPU显存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # 重置峰值内存统计（用于监控）
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    print("✅ GPU Memory Cleared!")

# ============================================================
# P2 Sample
# ============================================================
print("\n" + "="*60)
print("Processing P2 Sample")
print("="*60 + "\n")

sc_file = "/home/maweicheng/ST_data/GSE211956/GSE211956_SC.h5ad"
st_file = "/home/maweicheng/ST_data/GSE211956/GSE211956_ST_P2.h5ad"
output_dir = "/home/maweicheng/ST_data/GSE211956/evaluate/P2"

# Stage 1: VAE Training
art = spg.vae(
    sc_file=sc_file, 
    st_file=st_file,
    top_n_per_type=100, 
    output_dir=output_dir,
    device="cuda:1"
)

clear_gpu()  # Clearn GPU memory

# Stage 2: Deconvolution
res = spg.deconv(
    vae=art, 
    output_dir=output_dir, 
    k_cells_per_cluster=15, 
    k_celltype=[20, 30, 40], 
    scale_basis="all", 
    save_reconstructed_genes=True,
    device="cuda:1"
)

clear_gpu()  # Clearn GPU memory

# Stage 3: Cell Communication
spg.cellcom(
    deconv_dir=output_dir,
    st_h5ad=st_file, 
    output_dir=output_dir,
    ligand_expr_threshold=3,
    receptor_expr_threshold=3,
    allow_same_celltype_comm=True,
    n_spot_neighbors=8,
    epochs=200,
    batch_size=128,
    seed=42,
    device="cuda:1"
)

clear_gpu()  # Clearn GPU memory
print("\n" + "="*60)
print("P2 Sample Completed!")
print("="*60 + "\n")


# ============================================================
# P3 Sample
# ============================================================
print("\n" + "="*60)
print("Processing P3 Sample")
print("="*60 + "\n")

st_file = "/home/maweicheng/ST_data/GSE211956/GSE211956_ST_P3.h5ad"
output_dir = "/home/maweicheng/ST_data/GSE211956/evaluate/P3"

# Stage 1: VAE Training
art = spg.vae(
    sc_file=sc_file, 
    st_file=st_file,
    top_n_per_type=100, 
    output_dir=output_dir,
    device="cuda:1"
)

clear_gpu()  # Clearn GPU memory

# Stage 2: Deconvolution
res = spg.deconv(
    vae=art, 
    output_dir=output_dir, 
    k_cells_per_cluster=15, 
    k_celltype=[20, 30, 40], 
    scale_basis="all", 
    save_reconstructed_genes=True,
    device="cuda:1"
)

clear_gpu()  # Clearn GPU memory

# Stage 3: Cell Communication
spg.cellcom(
    deconv_dir=output_dir,
    st_h5ad=st_file, 
    output_dir=output_dir,
    ligand_expr_threshold=3,
    receptor_expr_threshold=3,
    allow_same_celltype_comm=True,
    n_spot_neighbors=8,
    epochs=200,
    batch_size=128,
    seed=42,
    device="cuda:1"
)

clear_gpu()  # Clearn GPU memory
print("\n" + "="*60)
print("P3 Sample Completed!")
print("="*60 + "\n")

print("\n" + "="*60)
print("All Samples Completed Successfully!")
print("="*60 + "\n")
