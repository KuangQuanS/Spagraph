import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

THIS_FILE = Path(__file__).resolve()
DATASET_DIR = THIS_FILE.parent
EVALUATE_DIR = next(parent for parent in THIS_FILE.parents if parent.name == "evaluate")
REPO_ROOT = EVALUATE_DIR.parent

# ------------------------------
# 文件路径
# ------------------------------
cell_composition_csv = DATASET_DIR / "GSM7782699_ST_composition.csv"
st_h5ad_path = REPO_ROOT / "spagraph_data" / "database" / "GSE243275" / "GSM7782699_ST.h5ad"
output_dir = DATASET_DIR / "deconv_fig"
cosine_csv = ""
spot_size = 1.5
# 创建 deconv_fig 文件夹
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading ST data...")
adata = sc.read_h5ad(st_h5ad_path)

# 🚀 **关键：检查是否已有空间坐标**
if "spatial" not in adata.obsm:
    raise ValueError("❌ 你的 h5ad 文件缺少 adata.obsm['spatial']，无法绘图")

# ----------------------------------------------------
# 自动生成 lowres 图片 (如果不存在) 以防止 MemoryError
# ----------------------------------------------------
try:
    library_id = list(adata.uns['spatial'].keys())[0]
    spatial_dict = adata.uns['spatial'][library_id]
    
    if 'lowres' not in spatial_dict['images']:
        print("\n[Info] 'lowres' image key missing. Generating from available image...")
        
        # 寻找可用源图片
        available_keys = list(spatial_dict['images'].keys())
        if not available_keys:
            print("[Warning] No images found in adata.uns['spatial']. Plotting explicitly without image.")
        else:
            # 优先用 hires, 否则用第一个
            src_key = 'hires' if 'hires' in available_keys else available_keys[0]
            src_img = spatial_dict['images'][src_key]
            
            # 计算缩放比例，目标宽度 ~2000px
            h, w = src_img.shape[:2]
            target_w = 2000
            if w > target_w:
                factor = int(np.ceil(w / target_w))
                print(f"[Info] Downsampling {src_key} ({w}x{h}) by factor {factor}...")
                
                # 使用切片进行下采样 (最快且不需要额外库)
                lowres_img = src_img[::factor, ::factor].copy()
                spatial_dict['images']['lowres'] = lowres_img
                
                # 计算对应的 scalefactor
                # 寻找源图片的 scale factor
                src_scale_key = f'tissue_{src_key}_scalef'
                src_scale = spatial_dict['scalefactors'].get(src_scale_key, 1.0)
                
                # lowres scale = src scale / factor
                spatial_dict['scalefactors']['tissue_lowres_scalef'] = src_scale / factor
                print(f"[Info] Created 'lowres' image. New scale factor: {spatial_dict['scalefactors']['tissue_lowres_scalef']:.6f}")
            else:
                print(f"[Info] Image is small enough ({w}x{h}). Copying as lowres.")
                spatial_dict['images']['lowres'] = src_img.copy()
                # 复制对应的 scale factor
                src_scale_key = f'tissue_{src_key}_scalef'
                spatial_dict['scalefactors']['tissue_lowres_scalef'] = spatial_dict['scalefactors'].get(src_scale_key, 1.0)
                
except Exception as e:
    print(f"[Warning] Failed to generate lowres image: {e}")


print("\nLoading cell composition...")
cell_comp = pd.read_csv(cell_composition_csv, index_col=0)

print("\nAligning indices...")
shared = adata.obs_names.intersection(cell_comp.index)
print("共有 spots:", len(shared))

adata_subset = adata[shared].copy()
cell_comp = cell_comp.loc[shared]

print("\nAdding cell composition to adata.obs...")
for ct in cell_comp.columns:
    adata_subset.obs[f"cell_comp_{ct}"] = cell_comp[ct].values

print("\nComputing dominant cell type...")
cell_comp_values = cell_comp.copy()
dominant_types = cell_comp_values.idxmax(axis=1)
dominant_vals = cell_comp_values.max(axis=1)

adata_subset.obs["dominant_cell_type"] = dominant_types
adata_subset.obs["dominant_cell_value"] = dominant_vals

# ------------------------------
# 🚀 第一段：空间可视化 dominant cell type
# ------------------------------
print("\nPlotting spatial dominant cell type...")
fig, ax = plt.subplots(figsize=(8, 8))
sc.pl.spatial(
    adata_subset,
    color="dominant_cell_type",
    size=spot_size,
    title=None,
    ax=ax,
    show=False,
    img_key="lowres"  # 使用低分辨率图以避免内存溢出
)
ax.set_axis_off()
pdf_path = output_dir / "dominant_cell_type.pdf"
plt.savefig(pdf_path, bbox_inches='tight', transparent=True)
print(f"✅ Saved PDF: {pdf_path}")
plt.close()

import scanpy as sc
import numpy as np

print("\n======================")
print("第二段：绘制每个细胞类型的空间分布")
print("======================")

# 找出之前加入的 cell_comp 列
cell_comp_cols = [c for c in adata_subset.obs.columns if c.startswith("cell_comp_")]

if len(cell_comp_cols) == 0:
    raise ValueError("❌ 没找到 cell_comp_ 开头的列，请检查你的 adata_subset.obs")

# 提取真实的 cell type 名称
cell_types = [c.replace("cell_comp_", "") for c in cell_comp_cols]

# ============================
# 每个 cell type 单独绘制空间图
# ============================
for col, ct in zip(cell_comp_cols, cell_types):
    print(f"\n绘制 {ct} ...")
    
    # 使用临时列，将值低于 0.01 的数据替换为 NaN，使其不显示（不修改原始数据）
    temp_col = f"{col}_plot"
    adata_subset.obs[temp_col] = adata_subset.obs[col].where(adata_subset.obs[col] >= 0.05, np.nan)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    raw_vmax = np.nanquantile(adata_subset.obs[temp_col], 0.98)
    step = 0.01
    vmax = np.ceil(max(raw_vmax, 0) / step) * step
    if vmax <= 0:
        vmax = step
    sc.pl.spatial(
        adata_subset,
        color=temp_col,
        size=spot_size,               # 显示在 notebook 比较合适
        cmap="viridis",
        vmin=0,
        vmax=vmax,
        title="",
        ax=ax,
        show=False,
        img_key="lowres"  # 使用低分辨率图以避免 MemoryError
    )
    ax.set_title("")
    ax.set_axis_off()
    cbar = ax.collections[0].colorbar if ax.collections else None
    if cbar is not None:
        from matplotlib.ticker import FormatStrFormatter
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        cbar.ax.tick_params(labelsize=16)
    pdf_path = output_dir / f"spatial_distribution_{ct}.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', transparent=True)
    print(f"✅ Saved PDF: {pdf_path}")
    plt.close()

# 第三段：在切片上绘制细胞组成饼图 - 带正确的坐标缩放
from matplotlib.patches import Wedge, Patch
from sklearn.neighbors import NearestNeighbors
import numpy as np

print("="*70)
print("第三段：在切片上绘制 spot 细胞组成饼图（带坐标缩放）")
print("="*70)

# 准备数据
comp_cols = [c for c in adata_subset.obs.columns if c.startswith('cell_comp_')]
comp_df = adata_subset.obs[comp_cols].copy()
comp_df.columns = [c.replace('cell_comp_', '') for c in comp_df.columns]

# 空间坐标（原始坐标）
coords = np.array(adata_subset.obsm['spatial'])
x_raw, y_raw = coords[:, 0], coords[:, 1]

# 获取scanpy的background image和缩放因子
print("\n正在获取组织切片图像...")
img_array = None
scale_factor = 1.0

if adata_subset.uns.get('spatial'):
    sample_key = list(adata_subset.uns['spatial'].keys())[0]
    spatial_data = adata_subset.uns['spatial'][sample_key]
    
    # 获取图像和对应的缩放因子
    if 'images' in spatial_data:
        images = spatial_data['images']
        
        # 优先使用hires，否则使用lowres
        if 'hires' in images:
            img_array = np.array(images['hires'])
            scale_factor = spatial_data['scalefactors'].get('tissue_hires_scalef', 1.0)
        elif 'lowres' in images:
            img_array = np.array(images['lowres'])
            scale_factor = spatial_data['scalefactors'].get('tissue_lowres_scalef', 1.0)
        else:
            img_key = list(images.keys())[0]
            img_array = np.array(images[img_key])
            scale_factor = spatial_data['scalefactors'].get('tissue_lowres_scalef', 1.0)

# 使用缩放因子调整坐标
x_scaled = x_raw * scale_factor
y_scaled = y_raw * scale_factor

# 计算饼图半径（使用缩放后的坐标）
nbrs = NearestNeighbors(n_neighbors=2).fit(coords)
dists, _ = nbrs.kneighbors(coords)
median_nn = np.median(dists[:, 1])
pie_radius = median_nn * scale_factor * 0.35

# 颜色映射 - 使用更柔和的配色 (Nature/Science 风格)
cell_types = comp_df.columns.tolist()
# 自定义配色表 (15色)
custom_colors = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", 
    "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
    "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"
]
if len(cell_types) <= len(custom_colors):
    color_map = {ct: custom_colors[i] for i, ct in enumerate(cell_types)}
else:
    # 如果类型太多，回退到 tab20
    cmap = plt.cm.get_cmap('tab20', max(len(cell_types), 3))
    color_map = {ct: cmap(i) for i, ct in enumerate(cell_types)}

# 裁剪图像到 spot 区域
print("\n正在裁剪图像...")
if img_array is not None:
    h_orig, w_orig = img_array.shape[:2]
    
    # 计算饼图半径（先计算一下用于边距）
    nbrs_temp = NearestNeighbors(n_neighbors=2).fit(coords)
    dists_temp, _ = nbrs_temp.kneighbors(coords)
    median_nn_temp = np.median(dists_temp[:, 1])
    pie_radius_temp = median_nn_temp * scale_factor * 0.35
    
    # 计算裁剪边界（添加边距）
    margin = pie_radius_temp * 2.5
    x_min = max(0, int(x_scaled.min() - margin))
    x_max = min(w_orig, int(x_scaled.max() + margin))
    y_min = max(0, int(y_scaled.min() - margin))
    y_max = min(h_orig, int(y_scaled.max() + margin))
    
    # 裁剪图像
    img_cropped = img_array[y_min:y_max, x_min:x_max]
    
    # 调整坐标到裁剪后的图像坐标系
    x_plot = x_scaled - x_min
    y_plot = y_scaled - y_min
else:
    img_cropped = None
    x_plot = x_scaled
    y_plot = y_scaled

# 创建图形
fig, ax = plt.subplots(figsize=(18, 16), dpi=600)

# 显示背景切片图（裁剪后的）
if img_cropped is not None:
    h_crop, w_crop = img_cropped.shape[:2]
    ax.imshow(img_cropped, extent=[0, w_crop, h_crop, 0], origin='upper', alpha=0.3, zorder=1)

# 绘制spot的饼图
print("\n绘制饼图...")

# 计算最终的饼图半径
nbrs = NearestNeighbors(n_neighbors=2).fit(coords)
dists, _ = nbrs.kneighbors(coords)
median_nn = np.median(dists[:, 1])
pie_radius = median_nn * scale_factor * 0.35

successful_spots = 0
# ------------------------------
# 优化：添加进度条提示，防止用户以为卡死
# ------------------------------
total_spots = len(comp_df)
print(f"Total spots to process: {total_spots}")

for spot_idx in range(total_spots):
    if spot_idx % 100 == 0:
        print(f"Processing spot {spot_idx}/{total_spots}...", end='\r')
    vals = np.array(comp_df.iloc[spot_idx].values, dtype=float)
    
    # 归一化
    if vals.sum() > 0:
        vals = vals / vals.sum()
    else:
        continue
    
    # 跳过全零的spot
    if vals.max() < 0.01:
        continue
    
    # 检查spot是否在裁剪范围内
    if img_cropped is not None:
        if x_plot[spot_idx] < 0 or x_plot[spot_idx] >= w_crop or y_plot[spot_idx] < 0 or y_plot[spot_idx] >= h_crop:
            continue
    
    # 获取spot坐标（裁剪后图像坐标系中）
    cx, cy = x_plot[spot_idx], y_plot[spot_idx]
    
    # 绘制这个spot的饼图
    start_angle = 0.0
    for val, cell_type in zip(vals, cell_types):
        if val <= 0.001:
            continue
        
        theta_start = start_angle
        theta_end = start_angle + val * 360.0
        
        # 创建wedge
        wedge = Wedge((cx, cy), pie_radius, theta_start, theta_end,
                 facecolor=color_map[cell_type], edgecolor='black', 
                 linewidth=0.8, zorder=10, alpha=1.0)
        ax.add_patch(wedge)
        
        start_angle = theta_end
    
    successful_spots += 1

print(f"✓ 绘制了 {successful_spots} 个 spot 的饼图")

# 设置坐标轴（匹配裁剪后的图像范围）
if img_cropped is not None:
    h_crop, w_crop = img_cropped.shape[:2]
    ax.set_xlim(0, w_crop)
    ax.set_ylim(h_crop, 0)
else:
    ax.set_xlim(x_plot.min() - pie_radius * 3, x_plot.max() + pie_radius * 3)
    ax.set_ylim(y_plot.max() + pie_radius * 3, y_plot.min() - pie_radius * 3)

ax.set_aspect('equal')
ax.axis('off')  # 移除坐标轴和脊线
ax.set_title('')  # 移除标题

# 添加图例
from matplotlib.patches import Patch
legend_patches = [Patch(facecolor=color_map[ct], edgecolor='black', linewidth=0.5, label=ct) for ct in cell_types]
ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.02, 0.5), 
          fontsize=10, framealpha=0.95, title='Cell Type', title_fontsize=12, ncol=1)

plt.tight_layout()

# 保存图像为PDF
output_file = output_dir / 'spot_pie_charts_on_tissue.pdf'
fig.savefig(output_file, bbox_inches='tight', transparent=True)
print(f"\n✅ Saved PDF: {output_file}")

plt.close()

print("\n" + "="*70)
print("完成！所有图已保存为PDF到 deconv_fig 文件夹")
print("="*70)

# 第四段：绘制 Cosine Similarity 热图
print("="*70)
print("第四段：绘制 Spot Cosine Similarity 热图")
print("="*70)

try:
    # 加载 cosine similarity CSV
    print(f"Loading cosine similarity from: {cosine_csv}")
    cos_df = pd.read_csv(cosine_csv)
    
    # 解析 spot_id 获取 x, y 坐标
    # spot_id 格式: "XxY" (例如 "10x10")
    cos_df['x_coord'] = cos_df['spot_id'].str.split('x').str[0].astype(int)
    cos_df['y_coord'] = cos_df['spot_id'].str.split('x').str[1].astype(int)
    
    # 创建透视表 (pivot table) 用于热图
    # 行=Y坐标, 列=X坐标, 值=cosine_similarity
    pivot_data = cos_df.pivot(index='y_coord', columns='x_coord', values='cosine_similarity')
    
    # 创建热图 - 使用圆形散点图
    fig, ax = plt.subplots(figsize=(14, 12), dpi=120)
    
    # 获取 x, y 坐标和相似度值
    x_coords = cos_df['x_coord'].values
    y_coords = cos_df['y_coord'].values
    similarity_values = cos_df['cosine_similarity'].values
    
    # 计算spot大小（基于坐标密度）
    x_range = x_coords.max() - x_coords.min()
    y_range = y_coords.max() - y_coords.min()
    spot_size = min(x_range, y_range) / max(len(np.unique(x_coords)), len(np.unique(y_coords))) * 100
    
    # 使用scatter绘制圆形spot
    scatter = ax.scatter(
        x_coords, 
        y_coords, 
        c=similarity_values,
        s=spot_size**1,  # 点的大小
        cmap='Blues',
        vmin=0,
        vmax=1,
        alpha=0.9,
        edgecolors='black',
        linewidths=0.5
    )
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cosine Similarity', fontsize=11)
    
    # 设置标题和标签
    ax.set_title('Spot Cosine Similarity Heatmap (Reconstructed vs True Expression)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    
    # 设置坐标轴范围（添加边距）
    margin = spot_size * 0.1
    ax.set_xlim(x_coords.min() - margin, x_coords.max() + margin)
    ax.set_ylim(y_coords.max() + margin, y_coords.min() - margin)
    
    # 翻转y轴使其从上到下递增
    ax.invert_yaxis()
    
    # 设置等比例坐标轴
    ax.set_aspect('equal')
    
    # 添加统计信息
    mean_cos = cos_df['cosine_similarity'].mean()
    median_cos = cos_df['cosine_similarity'].median()
    min_cos = cos_df['cosine_similarity'].min()
    max_cos = cos_df['cosine_similarity'].max()
    
    stats_text = f'Mean: {mean_cos:.4f}\nMedian: {median_cos:.4f}\nMin: {min_cos:.4f}\nMax: {max_cos:.4f}'
    ax.text(1.15, 0.5, stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    
    # 保存图像为PDF
    output_file = output_dir / 'cosine_similarity_heatmap.pdf'
    fig.savefig(output_file, bbox_inches='tight', transparent=True)
    print(f"\n✅ Saved PDF: {output_file}")
    
    plt.close()
    
except FileNotFoundError:
    print(f"⚠️  Cosine similarity file not found: {cosine_csv}. Skipping this section.")
except Exception as e:
    print(f"⚠️  Error loading cosine similarity file: {e}. Skipping this section.")

print("\n" + "="*70)
print("完成！所有图已保存为PDF到 deconv_fig 文件夹")
print("="*70)
