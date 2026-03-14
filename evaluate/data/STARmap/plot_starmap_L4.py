import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import warnings
import os

# 忽略警告
warnings.filterwarnings('ignore')

# 设置matplotlib样式
plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

# ==========================================
# 配置部分
# ==========================================
st_h5ad_path_1 = "f:/ST_Graduation_Project/spagraph_data/database/STARmap/STARmap_SP_1523.h5ad"
st_h5ad_path_2 = "f:/ST_Graduation_Project/spagraph_data/database/STARmap/Spatial.h5ad"
cell_composition_csv = "F:\ST_Graduation_Project\spagraph_data\evaluate\STARmap\文章数据\SPOTlight_result.txt"
output_dir = Path("./STARmap/figures")
target_cell_type = "Excitatory L4"
pcc_value = 0.79
method_name = "SPOTlight"  # 方法名称，用作文件名前缀 

# 手动设定颜色
colors = {
    'ExcitatoryL4': 'red',
    'ExcitatoryL2and3': 'blue',
    'ExcitatoryL6': 'green',
    'ExcitatoryL5': 'orange',
    'Olig': 'purple',
    'Endo': 'brown',
    'Micro': 'pink',
    'Astro': 'gray',
    'other': 'black',
    'HPC': 'cyan',
    'Vip': 'magenta',
    'Pvalb': 'yellow',
    'Sst': 'lime',
    'Npy': 'navy',
    'Smc': 'olive'
}

# ==========================================
# 数据加载
# ==========================================
adata = sc.read_h5ad(st_h5ad_path_1)
print(adata)

# ==========================================
# 第一个图：细胞类型空间分布
# ==========================================
# 获取空间坐标
spatial_coords = adata.obsm['spatial']
cell_types = adata.obs['celltype']

# 获取唯一的细胞类型
unique_types = cell_types.unique()
unique_types = list(unique_types)

# 把'other'移到最后
if 'other' in unique_types:
    unique_types.remove('other')
    unique_types.append('other')

n_types = len(unique_types)

# 创建图表
fig, ax = plt.subplots(figsize=(6, 8))

# 为每个细胞类型绘制点
for cell_type in unique_types:
    mask = cell_types == cell_type
    coords = spatial_coords[mask]
    ax.scatter(coords[:, 0], coords[:, 1], 
               c=colors.get(cell_type, 'gray'),  # 如果没有指定颜色，用灰色
               s=35,  # 点大小
               alpha=0.8, 
               label=cell_type, 
               edgecolors='none')

# 设置标题和标签
ax.set_title('Cell Type Spatial Distribution', fontsize=16, fontweight='bold')
ax.set_xlabel('X Coordinate', fontsize=14)
ax.set_ylabel('Y Coordinate', fontsize=14)

# 移除坐标轴数字
ax.set_xticks([])
ax.set_yticks([])

# 添加图例
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, frameon=False)

# 移除轴脊
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 保存图表
import os
os.makedirs('./STARmap/figures', exist_ok=True)
plt.tight_layout()
plt.savefig('./STARmap/figures/STARmap_celltype_spatial.pdf', dpi=300, bbox_inches='tight')
# plt.show()

# ==========================================
# 第二个图：特定细胞类型的像素化图
# ==========================================
output_dir.mkdir(parents=True, exist_ok=True)
# 输出文件名
output_file = output_dir / f'{method_name}_Excitatory_L4.pdf'

# ==========================================
# 2. 数据加载与对齐
# ==========================================
print("1. Loading and Aligning Data...")
# 重新加载第二个 h5ad 文件用于 L4 像素化图
adata = sc.read_h5ad(st_h5ad_path_2)
cell_composition = pd.read_csv(cell_composition_csv, sep=',', index_col=0)

# 索引清洗与对齐
adata.obs_names = adata.obs_names.astype(str).str.strip()
cell_composition.index = cell_composition.index.astype(str).str.strip()
print(adata.obs_names)
print(cell_composition.index)
shared_spots = adata.obs_names.intersection(cell_composition.index)

# 自动修复后缀匹配问题
if len(shared_spots) == 0:
    print("   ⚠️  Trying to fix index suffixes...")
    adata_clean = adata.obs_names.str.replace(r'-\d+$', '', regex=True)
    csv_clean = cell_composition.index.str.replace(r'-\d+$', '', regex=True)
    if len(adata_clean.intersection(cell_composition.index)) > 0:
        adata.obs_names = adata_clean
        shared_spots = adata.obs_names.intersection(cell_composition.index)
    elif len(adata.obs_names.intersection(csv_clean)) > 0:
        cell_composition.index = csv_clean
        shared_spots = adata.obs_names.intersection(cell_composition.index)
    else:
        raise ValueError("Index alignment failed.")

# 合并数据
adata_subset = adata[shared_spots].copy()
cell_composition_aligned = cell_composition.loc[shared_spots]
cell_composition_aligned.columns = [f"cell_comp_{c}" for c in cell_composition_aligned.columns]
adata_subset.obs = pd.concat([adata_subset.obs, cell_composition_aligned], axis=1)

# ==========================================
# 3. 指定目标与坐标处理
# ==========================================
# --- 指定细胞类型 ---
target_col = f"cell_comp_{target_cell_type}"

if target_col not in adata_subset.obs.columns:
    raise ValueError(f"Cell type '{target_cell_type}' not found.")

# 坐标处理
if 'spatial' in adata_subset.obsm:
    coords = adata_subset.obsm['spatial']
else:
    raise ValueError("obsm['spatial'] not found.")

# 修复 1D 坐标
if coords.ndim == 1 or coords.shape[1] == 1:
    coords = coords.reshape(-1, 1) if coords.ndim==1 else coords
    coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])

x_raw, y_raw = coords[:, 0], coords[:, 1]

# ==========================================
# 4. 绘图 (极简 + 0值同色背景)
# ==========================================
print(f"Plotting {target_cell_type} (Clean Mode)...")

# 1. 设置色盘
cmap = plt.get_cmap('viridis')

# 2. 【关键步骤】获取色盘中数值为 0 的颜色
# 这样背景色就和 value=0 的点颜色完全一样了
bg_color = cmap(0.0) 

# 3. 创建画布
# figsize 可以随意设置，因为后面会由 bbox_inches='tight' 裁切
# 但保持一定的长宽比有助于保持像素的方形
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

# 4. 设置画布和坐标轴背景色
fig.patch.set_facecolor(bg_color)
ax.set_facecolor(bg_color)

# 5. 绘制散点
values = adata_subset.obs[target_col].values
marker_size = 500 # 根据需要微调

scatter = ax.scatter(
    x_raw, y_raw, 
    c=values, 
    s=marker_size, 
    marker='s', 
    cmap=cmap,
    vmin=0, 
    vmax=values.max() * 1.1, 
    alpha=1.0, 
    edgecolors='none', 
    zorder=10
)

# 6. 【极简处理】移除所有装饰
ax.axis('off')                      # 关掉坐标轴
# ax.set_title(...)                 # 不写标题
# plt.colorbar(...)                 # 不画色条

# 7. 【全屏填充】强制 Axes 占满整个 Figure（底部留空间给图注）
# 左下角(0,0.08)，宽1.0，高0.92，底部留8%空间给图注
ax.set_position([0.0, 0.08, 1.0, 0.92])
ax.set_aspect('equal', 'box') # 保持比例

# 8. 添加图注显示PCC值（放在底部空白区域）
fig.text(0.5, 0.04, f'PCC: {pcc_value:.4f}', ha='center', va='center', 
         fontsize=14, color='black', fontweight='bold')

# ==========================================
# 5. 保存 (无边框)
# ==========================================
print(f"Saving to: {output_file}")

# pad_inches=0 配合 bbox_inches='tight' 切除所有白边
# facecolor=bg_color 确保即便有缝隙也是背景色
fig.savefig(
    output_file, 
    dpi=300, 
    bbox_inches='tight', 
    pad_inches=0, 
    facecolor=bg_color
)

# 不显示图片，直接保存，防止 show() 重置布局
plt.close(fig) 
print("Done!")