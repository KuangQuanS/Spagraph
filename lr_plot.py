import matplotlib
matplotlib.use("Agg") # Headless mode
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import pandas as pd
import numpy as np
import scanpy as sc
from pathlib import Path


def _require_columns(df: pd.DataFrame, required: list[str], *, df_name: str = "df") -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {df_name}: {missing}")

# ================= 配置区域 =================
# GSE24375
# DATA_DIR = Path("/mnt/d/ST_Graduation_Project_data/evaluate/GSE243275")
# ST_H5AD_PATH = Path("/mnt/d/ST_Graduation_Project_data/database/GSE243275/GSM7782699_ST.h5ad")
# TARGET_LR_PAIRS = ["ANGPTL4_SDC1", "PDGFD_PDGFRB"]

# CID44971
DATA_DIR = Path("/mnt/d/ST_Graduation_Project_data/evaluate/CID44971")
ST_H5AD_PATH = Path("/mnt/d/ST_Graduation_Project_data/database/Wu/CID44971/CID44971_ST.h5ad")
TARGET_LR_PAIRS = ["CCL19_CCR7", "ANGPTL4_SDC2"]

# GSE144236
# DATA_DIR = Path("/mnt/d/ST_Graduation_Project_data/evaluate/GSE144236")
# ST_H5AD_PATH = Path("/mnt/d/ST_Graduation_Project_data/database/GSE144240/GSE144236_P2_ST.h5ad")
# TARGET_LR_PAIRS = ["PGF_NRP1", "PGF_NRP2", "ANGPTL4_SDC1"]

# GSE211956
# DATA_DIR = Path("/mnt/d/ST_Graduation_Project_data/evaluate/GSE211956/P2")
# ST_H5AD_PATH = Path("/mnt/d/ST_Graduation_Project_data/database/GSE211956/GSE211956_ST_P2.h5ad")
# TARGET_LR_PAIRS = ["ANGPTL4_SDC1", "PDGFD_PDGFRB"]

# ================= 分割线 =================
LR_COMM_PATH = DATA_DIR / "lr_communication.csv"  # default filtered communication CSV
OUTPUT_DIR = DATA_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 绘图参数
TOP_PAIRS = 15
TOP_SPOTS = 200
TOP_EDGES_PER_PAIR = 200
SPOT_DISPLAY_SIZE = 50
# 从“圆形图（Circle Plot）”开始使用的过滤阈值：只保留 attention_score > 该阈值 的交互
ATTENTION_SCORE_THRESHOLD = 0.8
# 样式设置
plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["font.size"] = 14
sns.set_style("whitegrid")

# ================= 1. 数据加载 =================
print("Loading data...")
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

# 获取缩放因子 (如果有的话)
sf = 1.0
if "spatial" in adata.uns:
    keys = list(adata.uns['spatial'].keys())
    sf = adata.uns['spatial'][keys[0]]['scalefactors'].get('tissue_hires_scalef', 1.0)

print(f"Loaded {len(df)} interactions. Scale factor: {sf}")

# ================= 2.5. 过滤数据（从圆形图开始使用） =================
df_filt = df.loc[df["attention_score"] > float(ATTENTION_SCORE_THRESHOLD)].copy()
print(
    f"Filtered interactions for downstream plots: {len(df_filt)} "
    f"(attention_score > {ATTENTION_SCORE_THRESHOLD})"
)

# ================= 2. 基础统计图 (垂直布局 - 大字号版) =================
print("Plotting Separate Stats Barplots (Large Font)...")

# 1. 准备数据
top_freq = df["lr_pair"].value_counts().head(TOP_PAIRS)
top_att = df.groupby("lr_pair")["attention_score"].mean().sort_values(ascending=False).head(TOP_PAIRS)

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
plt.savefig(OUTPUT_DIR / "top_lr_pairs_freq.png", dpi=300, bbox_inches="tight")
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
ax.set_xlim(0, 1.5)
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
plt.savefig(OUTPUT_DIR / "top_lr_pairs_attention.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved separated plots: top_lr_pairs_freq.png & top_lr_pairs_attention.png")
# ================= 3. 圆形图 (Circle Plot - 连通性保障版) =================
print("Plotting Circle Plot...")

# 建议设置在 50-100 之间
CIRCLE_MAX_EDGES = 200 

def draw_circle_panel(ax, metric_col, title, max_edges=CIRCLE_MAX_EDGES):
    # --- 1. 准备数据 ---
    # 从此处开始，统一用过滤后的 df_filt
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
    
    # --- 2. 计算节点大小 ---
    cell_totals = {c: 0.0 for c in cells}
    for (src, dst), val in grp.items():
        cell_totals[src] += val
        cell_totals[dst] += val
    
    max_total = max(cell_totals.values()) if cell_totals else 1.0
    node_sizes = [300 + 1300 * (cell_totals[c] / max_total) for c in cells]

    # --- 3. 坐标与颜色 ---
    angles = np.linspace(np.pi/2, 2.5*np.pi, n, endpoint=False)
    xs, ys = np.cos(angles), np.sin(angles)
    colors = sns.color_palette("tab20", n)
    
    # --- 4. 智能筛选边 (核心修改) ---
    # 逻辑：先保证每个细胞至少有一条边，然后再填充 Top N
    
    all_edges = []
    for (src, dst), val in grp.items():
        if val > 0:
            all_edges.append(((src, dst), val))
    
    # 用集合存储最终要画的边，自动去重
    selected_edges = set()
    
    # (A) 【保底阶段】：确保每个细胞至少有一条最强连线
    for cell in cells:
        # 找出所有涉及该细胞的边 (作为起点或终点)
        related_edges = [e for e in all_edges if e[0][0] == cell or e[0][1] == cell]
        if related_edges:
            # 选出权重最大的一条
            best_edge = max(related_edges, key=lambda x: x[1])
            selected_edges.add(best_edge)
    
    # (B) 【填充阶段】：剩下的名额给全局最强边
    # 先按权重从大到小排序所有边
    sorted_all_edges = sorted(all_edges, key=lambda x: x[1], reverse=True)
    
    for edge in sorted_all_edges:
        if len(selected_edges) >= max_edges:
            break
        selected_edges.add(edge)
        
    # (C) 转回列表并按从小到大排序 (为了画图时细线在下，粗线在上)
    edges_to_draw = sorted(list(selected_edges), key=lambda x: x[1])
    
    # --- 5. 绘图 ---
    max_val = grp.max() # 线宽基准还是用全局最大值，保持比例真实

    for (src, dst), val in edges_to_draw:
        si, di = c_idx[src], c_idx[dst]
        
        # 线宽
        lw = 0.5 + (np.log1p(val) / np.log1p(max_val)) * 2.0
        
        # 绘制路径
        if si == di: # 自环
            patch = FancyArrowPatch((xs[si]+0.05, ys[si]+0.05), (xs[si]-0.05, ys[si]+0.05),
                                   connectionstyle="arc3,rad=0.8", arrowstyle="-|>", 
                                   color=colors[si], lw=lw, alpha=0.6, mutation_scale=10, zorder=1)
        else: # 弧线
            rad = 0.2 if (di - si) % n < n/2 else -0.2
            patch = FancyArrowPatch((xs[si], ys[si]), (xs[di], ys[di]),
                                   connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", 
                                   color=colors[si], lw=lw, alpha=0.5, mutation_scale=10, zorder=1)
        ax.add_patch(patch)

    # --- 6. 绘制节点与标签 ---
    ax.scatter(xs, ys, s=node_sizes, c=colors, edgecolors="k", zorder=10)

    for i, name in enumerate(cells):
        ha = "left" if xs[i] > 0 else "right"
        va = "center"
        if ys[i] > 0.8: va = "bottom"
        if ys[i] < -0.8: va = "top"
        ax.text(xs[i]*1.18, ys[i]*1.18, name, ha=ha, va=va, fontsize=12, fontweight="bold", zorder=11)

    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.9, 1.9)
    ax.axis("off")
    ax.text(0, 1.65, title, ha="center", va="bottom", fontsize=16, fontweight="bold")

# 创建画布并调用
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
draw_circle_panel(axes[0], "count", "Number of Interactions", max_edges=80)
draw_circle_panel(axes[1], "original_lr_score", "Interaction Strength", max_edges=80)
plt.savefig(OUTPUT_DIR / "circle_plot.png", bbox_inches="tight")
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
ax.tick_params(axis='y', labelsize=13)
ax.tick_params(axis='x', labelsize=12)

# 4. 智能格式化 X 轴 (自动选 k 或 M)
max_val = degree_df.max().max()
if max_val >= 1e6:
    # 如果超过百万，显示 M
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x*1e-6:.1f}M'))
    ax.set_xlabel("Event Count (Millions)", fontsize=14, fontweight='bold')
else:
    # 否则显示 k
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x/1000:.0f}k'))
    ax.set_xlabel("Event Count (Thousands)", fontsize=14, fontweight='bold')

# 5. 样式微调
ax.grid(axis='x', linestyle='--', alpha=0.5)
ax.legend(fontsize=12) # 图例
sns.despine(ax=ax, left=True, bottom=False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "celltype_in_out.png", dpi=300, bbox_inches="tight")
plt.close()

# ================= 4. 空间中心性 (Spatial Centrality) - 复刻原版画风 =================
print("Plotting Spatial Centrality (Strength & Degree)...")

# 1. 预加载空间数据 (一次性准备好，不用每次函数都读一遍)
adata = sc.read_h5ad(ST_H5AD_PATH)
coords = pd.DataFrame(adata.obsm["spatial"], index=adata.obs_names, columns=["x", "y"]).astype(float)

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
    save_path = OUTPUT_DIR / fname
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

# --- 主循环逻辑：计算 Strength 和 Degree ---
metrics = [
    ("original_lr_score", "LR Score"), 
    ("attention_score", "Attention")
]

for col, col_name in metrics:
    df_use = df_filt
    # Logic 1: Strength (强度/总和) -> 用 .sum()
    # 含义：这里信号最强
    s_sum = df_use.groupby("src_spot_barcode")[col].sum().add(
        df_use.groupby("dst_spot_barcode")[col].sum(), fill_value=0
    )
    plot_spatial_map(s_sum, f"top_spots_{col}_Strength.png", f"Top {TOP_SPOTS} Spots by {col_name} Strength (Sum)")
    
    # Logic 2: Degree (唯一邻居数 / Unique-neighbor degree)
    # 含义：这里连接最广 (Hub) —— 统计每个 spot 的唯一相连 spot 数（出邻居 + 入邻居）
    out_deg = df_use.groupby("src_spot_barcode")["dst_spot_barcode"].nunique()
    in_deg = df_use.groupby("dst_spot_barcode")["src_spot_barcode"].nunique()
    s_deg = out_deg.add(in_deg, fill_value=0)
    plot_spatial_map(s_deg, f"top_spots_{col}_Degree.png", f"Top {TOP_SPOTS} Spots by {col_name} Degree (Unique Neighbors)")


# ================= 5. 空间连线 (Edge Visualization) =================
print("Plotting Specific LR Edges...")

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
    # zip 的时候要注意可能 edges 被上面的 d2 过滤变短了
    valid_widths = widths[:len(edges)]
    
    for row, lw in zip(edges.itertuples(index=False), valid_widths):
        # 绘制起点和终点
        # 点稍微大一点 (s=40)，描边细一点 (linewidth=0.5)
        ax.scatter(row.src_x_plot, row.src_y_plot, s=40, color=spot_color, edgecolor="black", linewidth=0.5, zorder=6)
        ax.scatter(row.dst_x_plot, row.dst_y_plot, s=40, color=dst_color, edgecolor="black", linewidth=0.5, zorder=6)

        # 【关键修改】动态计算弧度 (rad)
        # 如果起点在终点左边 (dx > sx)，线向上弯 (rad正)
        # 如果起点在终点右边 (dx < sx)，线向下弯 (rad负)
        # 这样会让所有线看起来往同一个“方向”拱，比较整齐好看
        rad = 0.2 if row.dst_x_plot > row.src_x_plot else -0.2

        # 绘制弧线 (无箭头)
        patch = FancyArrowPatch(
            (row.src_x_plot, row.src_y_plot),
            (row.dst_x_plot, row.dst_y_plot),
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
    plt.tight_layout()
    safe_name = lr_name.replace("/", "_")
    plt.savefig(OUTPUT_DIR / f"edge_{safe_name}_{score_col}.png", dpi=300, bbox_inches="tight")
    plt.close()

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

print("Done! All plots saved to", OUTPUT_DIR)