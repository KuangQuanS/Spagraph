import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import matplotlib
matplotlib.use("Agg") # Headless mode
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import pandas as pd
import numpy as np
import scanpy as sc
from pathlib import Path
import gc


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATE_DIR = SCRIPT_DIR.parent.parent
DATA_ROOT = EVALUATE_DIR / "data"
REPO_ROOT = EVALUATE_DIR.parent
DATABASE_ROOT = REPO_ROOT / "spagraph_data" / "database"


def _require_columns(df: pd.DataFrame, required: list[str], *, df_name: str = "df") -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {df_name}: {missing}")


def _require_existing(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")

#================= 配置区域 =================
#GSE243275
DATA_DIR = Path("D:\Spagraph\evaluate\data\GSE243275")
ST_H5AD_PATH = Path("D:\Spagraph\spagraph_data\database\GSE243275\GSM7782699_ST.h5ad")
TARGET_LR_PAIRS = ["DLL4_NOTCH3", "VEGFC_KDR", "PDGFD_PDGFRB", "FN1_CD44"]
SPOT_DISPLAY_SIZE = 50      # Spot显示大小：控制空间图上点的大小（用于空间中心性图等）
TOP_PAIRS = 15            # Top配体对数量：柱状图中显示频率/注意力最高的前N个LR配体对
TOP_SPOTS = 200           # Top Spots数量：空间图上高亮显示得分最高的前N个spots
TOP_EDGES_PER_PAIR = 100  # 每个配体对的连线数：空间连线图中每个LR对最多显示N条得分最高的边，防止图像过于混乱
offset_range = 20.0       # 抖动范围：为重叠的点添加随机偏移量，避免空间连线图中的点完全重叠

# CID44971
# DATA_DIR = Path("./evaluate/CID44971")
# ST_H5AD_PATH = Path("./database/Wu/CID44971/CID44971_ST.h5ad")
# TARGET_LR_PAIRS = ["CCL19_CCR7", "PDGFD_PDGFRB", "MDK_SDC1", "GZMA_PARD3"]
# SPOT_DISPLAY_SIZE = 50      # Spot显示大小：控制空间图上点的大小（用于空间中心性图等）
# TOP_PAIRS = 15            # Top配体对数量：柱状图中显示频率/注意力最高的前N个LR配体对
# TOP_SPOTS = 200           # Top Spots数量：空间图上高亮显示得分最高的前N个spots
# TOP_EDGES_PER_PAIR = 100  # 每个配体对的连线数：空间连线图中每个LR对最多显示N条得分最高的边，防止图像过于混乱
# offset_range = 20.0       # 抖动范围：为重叠的点添加随机偏移量，避免空间连线图中的点完全重叠

# # GSE144236
# DATA_DIR = DATA_ROOT / "GSE144236"
# ST_H5AD_PATH = DATABASE_ROOT / "GSE144240" / "GSE144236_P2_ST.h5ad"
# #TARGET_LR_PAIRS = ["LAMB3_CD44", "ANGPTL4_SDC1", "LAMB3_ITGA6_ITGB1"]
# TARGET_LR_PAIRS = ["ANXA1"]
# SPOT_DISPLAY_SIZE = 150
# TOP_SPOTS = 50
# TOP_PAIRS = 15            # Top配体对数量：柱状图中显示频率/注意力最高的前N个LR配体对
# TOP_SPOTS = 50           # Top Spots数量：空间图上高亮显示得分最高的前N个spots
# TOP_EDGES_PER_PAIR = 50  # 每个配体对的连线数：空间连线图中每个LR对最多显示N条得分最高的边，防止图像过于混乱
# offset_range = 20.0       # 抖动范围：为重叠的点添加随机偏移量，避免空间连线图中的点完全重叠

# GSE211956_P2
# DATA_DIR = Path("./evaluate/GSE211956/P2")
# ST_H5AD_PATH = Path("./database/GSE211956/GSE211956_ST_P2.h5ad")
# TARGET_LR_PAIRS = ["PDGFA_PDGFRB", "ANGPT2_ITGA5_ITGB1", "PDGFD_PDGFRB" ,"COL4A2_SDC1", "COL4A2_CD44", "WNT5A_FZD1"]
# SPOT_DISPLAY_SIZE = 50      # Spot显示大小：控制空间图上点的大小（用于空间中心性图等）
# TOP_PAIRS = 15            # Top配体对数量：柱状图中显示频率/注意力最高的前N个LR配体对
# TOP_SPOTS = 200           # Top Spots数量：空间图上高亮显示得分最高的前N个spots
# TOP_EDGES_PER_PAIR = 100  # 每个配体对的连线数：空间连线图中每个LR对最多显示N条得分最高的边，防止图像过于混乱
# offset_range = 30.0       # 抖动范围：为重叠的点添加随机偏移量，避免空间连线图中的点完全重叠
# coord_exchange = True
# MIN_EVENT_COUNT = 10 # 最小事件数阈值：只保留某个lr_pair总出现次数 >= 该阈值的交互

# GSE211956_P3
# DATA_DIR = Path("./evaluate/GSE211956/P3")
# ST_H5AD_PATH = Path("./database/GSE211956/GSE211956_ST_P3.h5ad")
# TARGET_LR_PAIRS = ["CXCL12_CXCR4", "PROS1_AXL", "PDGFD_PDGFRB", "IGF1_ITGA6_ITGB4"]
# SPOT_DISPLAY_SIZE = 50      # Spot显示大小：控制空间图上点的大小（用于空间中心性图等）
# TOP_PAIRS = 15            # Top配体对数量：柱状图中显示频率/注意力最高的前N个LR配体对
# TOP_SPOTS = 200           # Top Spots数量：空间图上高亮显示得分最高的前N个spots
# TOP_EDGES_PER_PAIR = 100  # 每个配体对的连线数：空间连线图中每个LR对最多显示N条得分最高的边，防止图像过于混乱
# offset_range = 30.0       # 抖动范围：为重叠的点添加随机偏移量，避免空间连线图中的点完全重叠

coord_exchange = False
MIN_EVENT_COUNT = 10 # 最小事件数阈值：只保留某个lr_pair总出现次数 >= 该阈值的交互

# ================= 分割线 =================
LR_COMM_PATH = DATA_DIR / "lr_communication.csv"  # default filtered communication CSV
OUTPUT_DIR = DATA_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 从“圆形图（Circle Plot）”开始使用的过滤阈值：只保留 attention_score > 该阈值 的交互
ATTENTION_SCORE_THRESHOLD = 0.8

# 样式设置
plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["font.size"] = 14
sns.set_style("whitegrid")

# ================= 1. 数据加载 =================
print("Loading data...")
_require_existing(LR_COMM_PATH)
_require_existing(ST_H5AD_PATH)
df = pd.read_csv(LR_COMM_PATH)

_require_columns(
    df,
    [
        "lr_pair",
        "original_lr_score",
        "attention_score",
        "source_cell",
        "target_cell",
        "src_spot_barcode",
        "dst_spot_barcode",
    ],
    df_name=str(LR_COMM_PATH),
)

df["original_lr_score"] = pd.to_numeric(df["original_lr_score"], errors="coerce").fillna(0.0)
df["attention_score"] = pd.to_numeric(df["attention_score"], errors="coerce").fillna(0.0)

# Normalize barcodes to string for consistent matching with adata.obs_names
df["src_spot_barcode"] = df["src_spot_barcode"].astype(str)
df["dst_spot_barcode"] = df["dst_spot_barcode"].astype(str)

# 加载空间数据 (一次性加载)
adata = sc.read_h5ad(ST_H5AD_PATH)
if "spatial" not in getattr(adata, "obsm", {}):
    raise ValueError(f"adata.obsm['spatial'] not found in {ST_H5AD_PATH}")
coords = pd.DataFrame(adata.obsm["spatial"], index=adata.obs_names, columns=["x", "y"])
if coord_exchange:
    coords[["x", "y"]] = coords[["y", "x"]].values
# 获取缩放因子 (如果有的话)
sf = 1.0
if "spatial" in adata.uns:
    keys = list(adata.uns['spatial'].keys())
    sf = adata.uns['spatial'][keys[0]]['scalefactors'].get('tissue_hires_scalef', 1.0)

print(f"Loaded {len(df)} interactions. Scale factor: {sf}")

# ================= 2. 过滤数据 =================
# 先统计每个lr_pair的出现次数，并过滤掉发生次数过少的LR对
lr_pair_counts = df["lr_pair"].value_counts()
valid_lr_pairs = lr_pair_counts[lr_pair_counts >= MIN_EVENT_COUNT].index

# 创建过滤mask（避免多次复制）
event_mask = df["lr_pair"].isin(valid_lr_pairs)
print(
    f"Event-filtered data: {event_mask.sum()} interactions "
    f"(from {len(valid_lr_pairs)} LR pairs with min_event_count >= {MIN_EVENT_COUNT})"
)

# 应用双重过滤：(1) attention_score阈值 (2) lr_pair最小出现次数
# 使用单次过滤+复制，而不是两次
df_event_filtered = df[event_mask]  # 仅保留视图供统计使用
attention_mask = df_event_filtered["attention_score"] > float(ATTENTION_SCORE_THRESHOLD)
df_filt = df_event_filtered[attention_mask].copy()  # 只在最后复制一次

print(
    f"Filtered interactions for downstream plots: {len(df_filt)} "
    f"(attention_score > {ATTENTION_SCORE_THRESHOLD}, min_event_count >= {MIN_EVENT_COUNT})"
)

# ================= 2. 基础统计图 (垂直布局 - 大字号版) =================
print("Plotting Separate Stats Barplots (Large Font)...")

# 1. 准备数据（使用发生次数过滤后的数据）
top_freq = df_event_filtered["lr_pair"].value_counts().head(TOP_PAIRS)
top_att = df_event_filtered.groupby("lr_pair")["attention_score"].mean().sort_values(ascending=False).head(TOP_PAIRS)

# -------------------------------------------------------------------------
# 图一：Frequency (Event Count)
# -------------------------------------------------------------------------
# 创建单独的画布，尺寸给够 (10, 10) 保证15个柱子不挤
fig, ax = plt.subplots(figsize=(12, 8)) 

sns.barplot(x=top_freq.values, y=top_freq.index, color="#4c72b0", ax=ax, edgecolor="white", linewidth=0.8)

# 格式化 X 轴
def millions_formatter(x, pos):
    return f'{x*1e-6:.1f}'

ax.xaxis.set_major_formatter(FuncFormatter(millions_formatter))

# 设置标签和标题 (大字号)
ax.set_xlabel("Event Count ($x10^6$)", fontsize=18, fontweight='bold', labelpad=10)
ax.set_title("Top Ligand-Receptor Pairs\n(by Event Frequency)", fontsize=22, fontweight="heavy", pad=20)
ax.set_ylabel("")

# 设置刻度字号
ax.tick_params(axis='y', labelsize=15) 
ax.tick_params(axis='x', labelsize=14)

# 样式微调
ax.grid(axis='x', linestyle='--', alpha=0.5)
sns.despine(ax=ax, left=True, bottom=False)

# 标注数值
for i, v in enumerate(top_freq.values):
    ax.text(v, i, f" {v*1e-6:.2f}M", va='center', fontsize=13, color='black', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_lr_pairs_freq.pdf", dpi=300, bbox_inches="tight")
plt.close() # 关掉画布，释放内存

# -------------------------------------------------------------------------
# 图二：Attention (Interaction Strength)
# -------------------------------------------------------------------------
# 再次创建新的独立画布
fig, ax = plt.subplots(figsize=(12, 8))

sns.barplot(x=top_att.values, y=top_att.index, color="#dd8452", ax=ax, edgecolor="white", linewidth=0.8)

# 设置标签和标题 (大字号)
ax.set_xlabel("Attention Score", fontsize=18, fontweight='bold', labelpad=10)
ax.set_title("Top Ligand-Receptor Pairs\n(by Avg Attention)", fontsize=22, fontweight="heavy", pad=20)
ax.set_ylabel("")
####################################################################################################### 这儿调范围
ax.set_xlim(0, 3)
# 设置刻度字号
ax.tick_params(axis='y', labelsize=15)
ax.tick_params(axis='x', labelsize=14)

# 样式微调
ax.grid(axis='x', linestyle='--', alpha=0.5)
sns.despine(ax=ax, left=True, bottom=False)

# 标注数值
for i, v in enumerate(top_att.values):
    ax.text(v, i, f" {v:.2f}", va='center', fontsize=13, color='black')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_lr_pairs_attention.pdf", dpi=300, bbox_inches="tight")
plt.close()

print("Saved separated plots: top_lr_pairs_freq.pdf & top_lr_pairs_attention.pdf")
# ================= 3. 圆形图 (Circle Plot - 美化版) =================
print("Plotting Circle Plot...")

# 建议设置在 50-100 之间
CIRCLE_MAX_EDGES = 200 

def draw_circle_panel(ax, metric_col, title, max_edges=CIRCLE_MAX_EDGES):
    # --- 1. 准备数据 ---
    df_use = df_filt
    if metric_col == "count":
        grp = df_use.groupby(["source_cell", "target_cell"]).size()
    else:
        grp = df_use.groupby(["source_cell", "target_cell"])[metric_col].sum()
    
    cells = sorted(set(df_use["source_cell"]) | set(df_use["target_cell"]))
    n = len(cells)
    if n == 0:
        ax.axis("off")
        return
    c_idx = {c: i for i, c in enumerate(cells)}
    
    # --- 2. 计算节点大小 (更大的节点) ---
    cell_totals = {c: 0.0 for c in cells}
    for (src, dst), val in grp.items():
        cell_totals[src] += val
        cell_totals[dst] += val
    
    max_total = max(cell_totals.values()) if cell_totals else 1.0
    # 增大节点尺寸范围：500-2500 (原 300-1600)
    node_sizes = [500 + 2000 * (cell_totals[c] / max_total) for c in cells]

    # --- 3. 坐标与颜色 ---
    angles = np.linspace(np.pi/2, 2.5*np.pi, n, endpoint=False)
    xs, ys = np.cos(angles), np.sin(angles)
    # 使用更鲜艳的配色
    colors = sns.color_palette("husl", n)
    
    # --- 4. 智能筛选边 ---
    all_edges = []
    for (src, dst), val in grp.items():
        if val > 0:
            all_edges.append(((src, dst), val))
    
    selected_edges = set()
    
    # (A) 确保每个细胞至少有一条最强连线
    for cell in cells:
        related_edges = [e for e in all_edges if e[0][0] == cell or e[0][1] == cell]
        if related_edges:
            best_edge = max(related_edges, key=lambda x: x[1])
            selected_edges.add(best_edge)
    
    # (B) 填充 Top N
    sorted_all_edges = sorted(all_edges, key=lambda x: x[1], reverse=True)
    for edge in sorted_all_edges:
        if len(selected_edges) >= max_edges:
            break
        selected_edges.add(edge)
        
    # (C) 按权重从小到大排序 (细线在下，粗线在上)
    edges_to_draw = sorted(list(selected_edges), key=lambda x: x[1])
    
    # --- 5. 绘制边 (线宽差异更大、颜色渐变) ---
    max_val = grp.max()
    min_val = min(e[1] for e in edges_to_draw) if edges_to_draw else 0
    val_range = max_val - min_val if max_val > min_val else 1
    
    # 使用渐变色表示强弱：弱连线偏灰蓝，强连线偏红橙
    from matplotlib.colors import LinearSegmentedColormap
    edge_cmap = plt.cm.get_cmap('coolwarm')

    for (src, dst), val in edges_to_draw:
        si, di = c_idx[src], c_idx[dst]
        
        # 归一化权重 [0, 1]
        norm_val = (val - min_val) / val_range if val_range > 0 else 0.5
        
        # 线宽：范围 1.0 - 6.0 (原 0.5 - 2.5)，差异更明显
        lw = 1.0 + norm_val * 5.0
        
        # 颜色：使用渐变色，弱连接用冷色，强连接用暖色
        edge_color = edge_cmap(norm_val)
        
        # 透明度：弱连接更透明
        alpha = 0.4 + norm_val * 0.5
        
        # 绘制路径
        if si == di:  # 自环
            patch = FancyArrowPatch(
                (xs[si]+0.06, ys[si]+0.06), (xs[si]-0.06, ys[si]+0.06),
                connectionstyle="arc3,rad=0.8", arrowstyle="-|>", 
                color=colors[si], lw=lw, alpha=alpha, mutation_scale=15, zorder=1
            )
        else:  # 弧线 - 减小弧度让线更直
            rad = 0.15 if (di - si) % n < n/2 else -0.15
            patch = FancyArrowPatch(
                (xs[si], ys[si]), (xs[di], ys[di]),
                connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", 
                color=edge_color, lw=lw, alpha=alpha, mutation_scale=15, zorder=1
            )
        ax.add_patch(patch)

    # --- 6. 绘制节点与标签 (更大字体) ---
    ax.scatter(xs, ys, s=node_sizes, c=colors, edgecolors="k", linewidths=2, zorder=10)

    for i, name in enumerate(cells):
        # 根据角度决定对齐方式
        angle_deg = np.degrees(angles[i]) % 360
        
        if 45 < angle_deg < 135:  # 上方
            ha, va = "center", "bottom"
        elif 225 < angle_deg < 315:  # 下方
            ha, va = "center", "top"
        elif angle_deg <= 45 or angle_deg >= 315:  # 右侧
            ha, va = "left", "center"
        else:  # 左侧
            ha, va = "right", "center"
        
        # 标签位置更靠近圆边缘
        label_radius = 1.15
        ax.text(
            xs[i] * label_radius, ys[i] * label_radius, name, 
            ha=ha, va=va, fontsize=20, fontweight="bold", zorder=11
        )

    # --- 7. 设置坐标轴 (减少空白、确保正圆) ---
    ax.set_xlim(-1.45, 1.45)  # 缩小范围，减少中心空白
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect('equal')  # 【关键】强制等比例，确保是正圆
    ax.axis("off")
    
    # 标题更大
    ax.set_title(title, fontsize=22, fontweight="bold", pad=15)

# 创建画布：正方形比例，确保两个子图都是正圆
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
draw_circle_panel(axes[0], "count", "Number of Interactions", max_edges=80)
draw_circle_panel(axes[1], "original_lr_score", "Interaction Strength", max_edges=80)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "circle_plot.pdf", bbox_inches="tight", dpi=300)
plt.close()

# ================= 3 细胞类型输入/输出统计 (In/Out Degree) =================
print("Plotting Cell Type In/Out Counts...")

# 1. 准备数据
# 分别统计作为起点(Outgoing)和终点(Incoming)的次数
df_use = df_filt
c_out = df_use["source_cell"].value_counts().rename("Outgoing")
c_in = df_use["target_cell"].value_counts().rename("Incoming")

# 合并、填充0、转整数
degree_df = pd.concat([c_out, c_in], axis=1).fillna(0).astype(int)

# 排序：按 Outgoing 从小到大排 (因为 barh 是从下往上画，这样最大的会在最上面)
degree_df = degree_df.sort_values("Outgoing", ascending=True)

# 2. 绘图
# 高度动态调整：根据细胞类型数量，保证每行有足够空间
h = max(6, len(degree_df) * 0.4) 
fig, ax = plt.subplots(figsize=(9, 14))

# 直接用 pandas 的 plot 接口画双色条形图，非常方便
degree_df.plot(kind="barh", color=["#4c72b0", "#dd8452"], ax=ax, width=0.8, edgecolor="white")

# 3. 美化
ax.set_title("Incoming vs Outgoing Events per Cell Type", fontsize=18, fontweight="bold", pad=5)
ax.set_ylabel("") # 细胞名就在轴上，不需要 Label
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)

# 4. 智能格式化 X 轴 (自动选 k 或 M)
max_val = degree_df.max().max()
if max_val >= 1e6:
    # 如果超过百万，显示 M
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x*1e-6:.1f}M'))
    ax.set_xlabel("Event Count (Millions)", fontsize=16, fontweight='bold')
else:
    # 否则显示 k
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x/1000:.0f}k'))
    ax.set_xlabel("Event Count (Thousands)", fontsize=16, fontweight='bold')

# 5. 样式微调
ax.grid(axis='x', linestyle='--', alpha=0.5)
ax.legend(fontsize=14) # 图例
sns.despine(ax=ax, left=True, bottom=False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "celltype_in_out.pdf", dpi=300, bbox_inches="tight")
plt.close()

# ================= 3.1 细胞类型输入/输出统计 (横向画布版/Vertical Bars) =================
print("Plotting Cell Type In/Out Counts (Vertical Version)...")

# 1. 数据准备 (重新排序：从大到小，因为是从左往右画)
degree_df_vert = degree_df.sort_values("Outgoing", ascending=False)

# 2. 绘图 (宽画布)
# 宽度动态调整：根据细胞类型数量，保证每个柱子有足够宽度
w = max(10, len(degree_df_vert) * 0.5)
fig, ax = plt.subplots(figsize=(w, 4))

# kind="bar" (竖向)
degree_df_vert.plot(kind="bar", color=["#4c72b0", "#dd8452"], ax=ax, width=0.8, edgecolor="white")

# 3. 美化
ax.set_title("Incoming vs Outgoing Events per Cell Type", fontsize=18, fontweight="bold", pad=20)
ax.set_xlabel("") # 标签在底下，不用额外写 Title
ax.tick_params(axis='y', labelsize=14)
# X轴标签旋转 45 度，防止重叠
ax.tick_params(axis='x', labelsize=14)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor")

# 4. 智能格式化 Y 轴
max_val = degree_df_vert.max().max()
if max_val >= 1e6:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x*1e-6:.1f}M'))
    ax.set_ylabel("Event Count (Millions)", fontsize=16, fontweight='bold')
else:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x/1000:.0f}k'))
    ax.set_ylabel("Event Count (Thousands)", fontsize=16, fontweight='bold')

# 5. 样式
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(fontsize=14)
sns.despine(ax=ax, top=True, right=True)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "celltype_in_out_vertical.pdf", dpi=300, bbox_inches="tight")
plt.close()

# ================= 4. 空间中心性 (Spatial Centrality) - 复刻原版画风 =================
print("Plotting Spatial Centrality (Strength & Degree)...")

# 1. 预加载空间数据 (一次性准备好，不用每次函数都读一遍)
adata = sc.read_h5ad(ST_H5AD_PATH)
coords = pd.DataFrame(adata.obsm["spatial"], index=adata.obs_names, columns=["x", "y"]).astype(float)
if coord_exchange:
    coords[["x", "y"]] = coords[["y", "x"]].values

# 获取缩放因子 (为了复刻原图的对其逻辑)
sf = 1.0
if "spatial" in adata.uns:
    keys = list(adata.uns['spatial'].keys())
    sf = adata.uns['spatial'][keys[0]]['scalefactors'].get('tissue_hires_scalef', 1.0)

def plot_spatial_map(score_series, fname, title, spot_size=SPOT_DISPLAY_SIZE):
    """
    完全复刻你提供的 _plot_top_spots_on_tissue 画图风格
    """
    # 准备数据：排序取 Top 50
    valid_scores = score_series.reindex(adata.obs_names).fillna(0)
    top_spots = valid_scores.sort_values(ascending=False).head(TOP_SPOTS).index
    
    # 开始画图
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 1. 画背景组织图 (Scanpy 原生)
    # 保持你原来的 alpha_img=1.0 (清晰背景)
    sc.pl.spatial(adata, color=None, alpha_img=1.0, show=False, ax=ax)
    
    # 2. 准备坐标 (应用缩放因子)
    coords_disp = coords.copy()
    coords_disp[['x', 'y']] = coords_disp[['x', 'y']] * sf
    
    # 3. 画所有点的背景 (灰色) - 保持原来的参数
    # s=bg_size logic: top_size * 0.50
    top_size = int(spot_size)
    bg_size = max(1, int(top_size * 0.50))
    
    ax.scatter(coords_disp['x'], coords_disp['y'], 
               s=bg_size, c='lightgray', alpha=0.75, linewidth=0, zorder=3)
    
    # 4. 画 Top Spots (黄色高亮) - 保持原来的参数
    ax.scatter(coords_disp.loc[top_spots, 'x'], coords_disp.loc[top_spots, 'y'], 
               s=top_size, c='#FFD400', edgecolor='k', linewidth=0.35, alpha=0.95, zorder=5)

    # 5. 调整坐标轴
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.axis("off") # 去掉坐标轴
    
    # 保存
    save_path = OUTPUT_DIR / fname.replace('.png', '.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

# --- 主循环逻辑：计算 Strength ---
metrics = [
    ("original_lr_score", "LR Score"), 
    ("attention_score", "Attention")
]

for col, col_name in metrics:
    df_use = df_filt
    # Logic: Strength (强度/总和) -> 用 .sum()
    # 含义：这里信号最强
    s_sum = df_use.groupby("src_spot_barcode")[col].sum().add(
        df_use.groupby("dst_spot_barcode")[col].sum(), fill_value=0
    )
    plot_spatial_map(s_sum, f"top_spots_{col}_Strength.png", f"Top {TOP_SPOTS} Spots by {col_name} Strength (Sum)")


# ================= 5. 空间连线 (Edge Visualization) =================
print("Plotting Specific LR Edges...")

# 全局细胞类型到marker的映射（跨图统一）
global_cell_to_marker = {}
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'H', '*']  # 10种marker

def plot_edges(sub_df, lr_name, score_col):
    if sub_df.empty:
        return

    # ---- 1. 数据准备与坐标映射 ----
    edges = sub_df.sort_values(score_col, ascending=False).head(TOP_EDGES_PER_PAIR).copy()

    coords_idx = coords.copy()
    coords_idx.index = coords_idx.index.astype(str)

    edges["src_spot_barcode"] = edges["src_spot_barcode"].astype(str)
    edges["dst_spot_barcode"] = edges["dst_spot_barcode"].astype(str)
    edges["src_x"] = edges["src_spot_barcode"].map(coords_idx["x"])
    edges["src_y"] = edges["src_spot_barcode"].map(coords_idx["y"])
    edges["dst_x"] = edges["dst_spot_barcode"].map(coords_idx["x"])
    edges["dst_y"] = edges["dst_spot_barcode"].map(coords_idx["y"])
    edges = edges.dropna(subset=["src_x", "src_y", "dst_x", "dst_y"])
    if edges.empty:
        print(f"Skip {lr_name} ({score_col}): no edges with valid spatial coordinates")
        return

    # ---- 2. 计算线宽（归一化分数） ----
    scores = edges[score_col].astype(float).to_numpy()
    s_min = float(np.nanmin(scores))
    s_max = float(np.nanmax(scores))
    if not np.isfinite(s_min) or not np.isfinite(s_max):
        print(f"Skip {lr_name} ({score_col}): invalid scores")
        return

    if s_max > s_min:
        norm_scores = (scores - s_min) / (s_max - s_min + 1e-12)
    else:
        norm_scores = np.zeros_like(scores, dtype=float)
    # 线宽范围：2.0 到 4.5，差异更明显
    widths = 2.0 + 2.5 * norm_scores

    # ---- 3. 开始绘图 ----
    fig, ax = plt.subplots(figsize=(10, 10))

    # 背景图变暗（H&E底图）
    sc.pl.spatial(adata, color=None, alpha_img=0.4, size=0.1, show=False, ax=ax)

    # 使用与 scanpy 显示一致的缩放
    coords_plot = coords_idx.astype(float).copy()
    coords_plot[["x", "y"]] = coords_plot[["x", "y"]] * sf

    # 灰色 spot 背景
    ax.scatter(coords_plot["x"], coords_plot["y"], s=16, c="lightgray", alpha=0.15, linewidth=0, zorder=2)

    # 预计算缩放后的边坐标
    edges["src_x_plot"] = edges["src_x"].astype(float) * sf
    edges["src_y_plot"] = edges["src_y"].astype(float) * sf
    edges["dst_x_plot"] = edges["dst_x"].astype(float) * sf
    edges["dst_y_plot"] = edges["dst_y"].astype(float) * sf

    # 过滤掉起点终点几乎重合的边
    d2 = (edges["src_x_plot"] - edges["dst_x_plot"]) ** 2 + (edges["src_y_plot"] - edges["dst_y_plot"]) ** 2
    edges = edges.loc[d2 > 1e-12].copy()
    if edges.empty:
        print(f"Skip {lr_name} ({score_col}): all selected edges are self-loops")
        plt.close(fig)
        return

    # 颜色定义
    spot_color = "#00FFFF" # 青色 (Source)
    dst_color = "#FF00FF"  # 洋红色 (Destination)
    line_color = "#32CD32" # 荧光绿 (Edge)
    # 描边特效
    outline_effect = [pe.Stroke(linewidth=3.0, foreground="black"), pe.Normal()]

    # ---- 4. 绘制点和线 ----
    # 为不同细胞类型分配不同的marker形状（全局映射，跨图统一）
    global global_cell_to_marker
    unique_cells = sorted(set(edges['source_cell']) | set(edges['target_cell']))
    for cell_type in unique_cells:
        if cell_type not in global_cell_to_marker:
            # 分配新的marker
            idx = len(global_cell_to_marker)
            global_cell_to_marker[cell_type] = markers[idx % len(markers)]
    
    # zip 的时候要注意可能 edges 被上面的 d2 过滤变短了
    valid_widths = widths[:len(edges)]
    
    # 添加随机种子以保证可重复性，但每次调用时不同
    np.random.seed(hash(lr_name + score_col) % (2**32))
    # 随机偏移范围（防止重叠）
    
    
    for row, lw in zip(edges.itertuples(index=False), valid_widths):
        # 添加小的随机偏移防止点重叠
        src_x_jitter = row.src_x_plot + np.random.uniform(-offset_range, offset_range)
        src_y_jitter = row.src_y_plot + np.random.uniform(-offset_range, offset_range)
        dst_x_jitter = row.dst_x_plot + np.random.uniform(-offset_range, offset_range)
        dst_y_jitter = row.dst_y_plot + np.random.uniform(-offset_range, offset_range)
        
        # 绘制起点和终点 - 使用细胞类型对应的marker
        src_marker = global_cell_to_marker[row.source_cell]
        dst_marker = global_cell_to_marker[row.target_cell]
        ax.scatter(src_x_jitter, src_y_jitter, s=40, color=spot_color, marker=src_marker, edgecolor="black", linewidth=0.5, zorder=6)
        ax.scatter(dst_x_jitter, dst_y_jitter, s=40, color=dst_color, marker=dst_marker, edgecolor="black", linewidth=0.5, zorder=6)

        # 【关键修改】动态计算弧度 (rad)
        # 如果起点在终点左边 (dx > sx)，线向上弯 (rad正)
        # 如果起点在终点右边 (dx < sx)，线向下弯 (rad负)
        # 这样会让所有线看起来往同一个"方向"拱，比较整齐好看
        rad = 0.2 if dst_x_jitter > src_x_jitter else -0.2

        # 绘制弧线 (无箭头)
        patch = FancyArrowPatch(
            (src_x_jitter, src_y_jitter),
            (dst_x_jitter, dst_y_jitter),
            connectionstyle=f"arc3,rad={rad}",  # 使用动态弧度
            arrowstyle="-",                      # 【关键修改】去箭头
            # mutation_scale 不需要了，因为没箭头了
            linewidth=float(lw),
            color=line_color,
            alpha=0.85,                          # 稍微透明一点点，更有层次感
            shrinkA=0.0, shrinkB=0.0,            # 不缩进，直接连到点中心
            zorder=5,
        )
        patch.set_path_effects(outline_effect)
        ax.add_patch(patch)

    if not ax.yaxis_inverted():
        ax.invert_yaxis()

    ax.set_title(f"{lr_name}\n({score_col})", fontsize=20, fontweight="bold", pad=20)
    ax.axis("off")
    
    # ---- 5. 添加图例 ----
    # 只显示当前图中实际出现的细胞类型
    current_cells = sorted(set(edges['source_cell']) | set(edges['target_cell']))
    legend_elements = []
    for cell_type in current_cells:
        marker = global_cell_to_marker[cell_type]
        # 使用Line2D来绘制实际的marker，颜色用灰色突出marker形状
        legend_elements.append(Line2D([0], [0], marker=marker, color='w', markerfacecolor='gray', 
                                       markeredgecolor='black', markersize=20, linewidth=0, label=cell_type))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='center left', fontsize=18, framealpha=0.95, 
                 title='Cell Type', title_fontsize=18, ncol=1, bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    safe_name = lr_name.replace("/", "_")
    plt.savefig(OUTPUT_DIR / f"edge_{safe_name}_{score_col}.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    gc.collect()

# 只要 TARGET_LR_PAIRS 不为空，就自动画
if not TARGET_LR_PAIRS:
    # 如果没有指定，默认画 Attention 最高的那个
    top_1 = df.groupby("lr_pair")["attention_score"].sum().idxmax()
    TARGET_LR_PAIRS = [top_1]
    print(f"No target pairs specified, plotting top 1: {top_1}")

for pair in TARGET_LR_PAIRS:
    sub = df_filt[df_filt["lr_pair"] == pair]
    plot_edges(sub, pair, "attention_score")
    plot_edges(sub, pair, "original_lr_score")
    gc.collect()

print("Done! All plots saved to", OUTPUT_DIR)
