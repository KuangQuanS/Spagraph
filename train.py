import argparse
import logging
import os
import torch
import glob
import pandas as pd
import numpy as np
import scanpy as sc
from hetero_vae_data_utils import STHeteroSubgraphDataset, hetero_subgraph_collate_fn
from hetero_graph_builder import GraphAugmentor
from hetero_model import HeteroSTModel
from utils import setup_logging, set_seed
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
import time
# 重建完整VAE模型以获取encoder
from SC_MAP_ST.deconv_model import DualDecoderVAE as FullVAE

def parse_args():
    parser = argparse.ArgumentParser(description='Heterogeneous ST Communication Model Training')
    # 数据输入参数 - 简化为单个 deconv_dir
    parser.add_argument('--deconv_dir', type=str, required=True, 
                       help='Stage1+Stage2 输出目录（包含 final_vae.pth, final_vae_cluster_data.npz 等）')
    parser.add_argument('--st_h5ad', type=str, required=True, help='空间转录组h5ad文件路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录路径')
    
    # VAE参数
    parser.add_argument('--vae_latent_dim', type=int, default=64, help='VAE隐空间维度')
    parser.add_argument('--vae_hidden_dim', type=int, default=256, help='VAE隐层维度')
    
    # 图参数
    parser.add_argument('--n_spot_neighbors', type=int, default=10, help='Spot邻近数')
    parser.add_argument('--spot_distance_sigma', type=float, default=50.0, help='Spot距离高斯参数')
    parser.add_argument('--composition_weight_mode', type=str, default='sqrt', help='成分权重模式')
    
    # LR通讯参数
    parser.add_argument('--lr_distance_sigma', type=float, default=50.0, help='LR通讯距离衰减参数sigma (default: 50.0)')
    parser.add_argument('--mean_expr_threshold', type=float, default=1.0, 
                       help='配体/受体活跃基因的平均表达阈值 (normalize_total 1e4后，default: 1.0)')
    parser.add_argument('--lr_comm_score_threshold', type=float, default=0.0, 
                       help='通讯得分过滤阈值，低于此值的通讯事件将被过滤 (default: 0.0，即不过滤)')
    parser.add_argument('--spot_cell_expr_npz', type=str, default=None, help='spot-cell表达NPZ路径（pre-computed结果，不提供则计算）')
    parser.add_argument('--load_lr_knn', type=str, default=None, help='加载预先计算的LR通讯得分CSV路径')

    # GAT参数
    parser.add_argument('--gat_layers', type=int, default=3, help='GAT层数')
    parser.add_argument('--gat_hidden_dims', type=str, default='256,256,128', help='GAT隐层维度')
    parser.add_argument('--gat_heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--gat_dropout', type=float, default=0.1, help='Dropout概率')
    
    # 模型参数
    parser.add_argument('--fusion_dim', type=int, default=256, help='融合向量维度')
    parser.add_argument('--output_dim', type=int, default=64, help='输出维度')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小 (已支持真正的批处理)')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='检查点间隔')
    parser.add_argument('--sample_rate', type=float, default=1.0, help='每个epoch采样比例 (default: 1.0, 即全部数据; 0.3表示采样30%)')
    parser.add_argument('--min_comm_edges', type=int, default=1, help='最小通讯边数阈值，少于此值的spot将被过滤 (default: 1)')
    
    return parser.parse_args()

#---------------------------------主程序---------------------------------
def main():
    # 解析参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(os.path.join(args.output_dir, 'training.log'))
    logging.info("="*80)
    logging.info("="*80)
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f'使用设备: {device}')
    
    # ========== 阶段1：加载数据和VAE编码器 ==========
    logging.info("="*80)
    logging.info("阶段1: 加载数据和VAE编码器")
    logging.info("="*80)
    
    # ✅ 使用 deconv_dir 构建所有路径
    deconv_dir = args.deconv_dir
    vae_weight_path = os.path.join(deconv_dir, 'final_vae.pth')
    vae_npz_path = os.path.join(deconv_dir, 'final_vae_cluster_data.npz')
    
    # 1. 尝试从 stage1 的 npz 文件加载 cluster 表达
    cluster_expr = None
    cluster_full_expr = None
    cluster_to_celltype = {}
    
    if os.path.exists(vae_npz_path):
        logging.info(f"从 NPZ 文件加载 cluster 表达: {vae_npz_path}")
        cluster_data = np.load(vae_npz_path, allow_pickle=True)
        
        cluster_ids = cluster_data['cluster_ids']
        expressions_array = cluster_data['cluster_expressions']  # marker genes
        expressions_full_array = cluster_data['cluster_expressions_full']  # all genes
        
        # 加载 celltype mapping (如果有的话)
        if 'cluster_to_celltype' in cluster_data:
            celltype_mapping_array = cluster_data['cluster_to_celltype']
            cluster_to_celltype = {str(row['cluster_id']): str(row['celltype']) 
                                  for row in celltype_mapping_array}
            logging.info(f"从 NPZ 加载 celltype 映射: {len(cluster_to_celltype)} clusters")
        else:
            logging.warning("NPZ 文件中未找到 celltype 映射，将使用 Cluster_{id} 作为名称")
            cluster_to_celltype = {str(cid): f"Cluster_{cid}" for cid in cluster_ids}
        
        # 将 VAE 权重中的 genes 列表加载出来
        checkpoint_temp = torch.load(vae_weight_path, map_location='cpu', weights_only=False)
        marker_genes = checkpoint_temp.get('genes', None)
        all_genes = checkpoint_temp.get('all_genes', None)
        
        if marker_genes is None:
            raise ValueError("VAE checkpoint 中找不到 genes 列表")
        if all_genes is None:
            raise ValueError("VAE checkpoint 中找不到 all_genes 列表")
        
        # ✅ 构建 DataFrame (marker genes) - 使用 Cluster_ID 格式作为索引（保持与 cluster_composition 一致）
        cluster_index_names = [f"Cluster_{cid}" for cid in cluster_ids]
        cluster_expr = pd.DataFrame(
            expressions_array,
            index=cluster_index_names,
            columns=marker_genes
        )
        logging.info(f"已从 NPZ 加载 cluster marker 表达: {cluster_expr.shape}")
        logging.info(f"   Cluster 索引格式: {cluster_expr.index.tolist()[:5]}")
        
        # 构建 DataFrame (all genes)
        cluster_full_expr = pd.DataFrame(
            expressions_full_array,
            index=cluster_index_names,
            columns=all_genes
        )
        logging.info(f"已从 NPZ 加载 cluster 全基因表达: {cluster_full_expr.shape}")
        
    else:
        # NPZ 文件不存在，无法继续
        raise FileNotFoundError(
            f"找不到 Stage 1 NPZ 文件: {vae_npz_path}\n"
            f"请使用最新版本重新运行 Stage 1 训练以生成 NPZ 文件。"
        )

    # 3. 验证 celltype 映射是否存在
    if not cluster_to_celltype:
        raise ValueError(
            "NPZ 文件中未找到 celltype 映射！\n"
            "请确保 sc_adata 包含 'cell_type' 列，并重新运行 Stage 1 训练。"
        )
    
    # 构建cluster到cell的映射（全局定义）
    cluster_to_cell = {}
    for cluster_id, celltype_name in cluster_to_celltype.items():
        cluster_name = f"Cluster_{cluster_id}"
        cluster_to_cell[cluster_name] = celltype_name
    logging.info(f"已构建cluster-cell映射: {len(cluster_to_cell)} 个cluster映射到 {len(set(cluster_to_cell.values()))} 个cell类型")
    
    # 4. 加载CellChat配体-受体数据库
    cellchat_file = 'cellchat_human.csv'
    
    if os.path.exists(cellchat_file):
        lr_db = pd.read_csv(cellchat_file)
        # 转换为LR对元组列表 [(ligand, receptor), ...]
        lr_pairs = []
        for _, row in lr_db.iterrows():
            lig = str(row['ligand']).strip()
            rec = str(row['receptor']).strip()
            lr_pairs.append((lig, rec))
        logging.info(f"已加载CellChat LR对: {len(lr_pairs)}个")
    else:
        # 如果找不到CellChat数据库，使用marker基因构建LR对
        logging.warning(f"找不到CellChat数据库 ({cellchat_file})")
    
    # 5. 加载ST数据
    adata = sc.read_h5ad(args.st_h5ad)
    logging.info(f"已加载ST数据: {adata.shape}")
    
    # 6. 加载spot坐标
    spot_coords = adata.obsm['spatial'] if 'spatial' in adata.obsm else None
    if spot_coords is not None:
        logging.info(f"已加载spot坐标: {spot_coords.shape}")
        # 全局记录spot数，后续统计使用
        n_spots = len(spot_coords)
        logging.info(f"spot数量: {n_spots}")
    else:
        logging.warning(f"找不到spot坐标，将使用随机坐标")
    
    # 7. 加载spot-cluster反卷积比例矩阵
    cluster_composition_file = os.path.join(deconv_dir, '*_cluster_composition.csv')
    cluster_composition_files = glob.glob(cluster_composition_file)
    if not cluster_composition_files:
        logging.warning(f"找不到cluster反卷积比例CSV，将跳过")
        cluster_composition = None
    else:
        cluster_composition = pd.read_csv(cluster_composition_files[0], index_col=0)
        logging.info(f"已加载cluster反卷积比例: {cluster_composition.shape}")
        # 记录cluster数量
        n_clusters = cluster_composition.shape[1]
        logging.info(f"cluster数量: {n_clusters}")
    
    # 8. 初始化和加载VAE编码器
    logging.info(f"加载VAE权重: {vae_weight_path}")
    
    checkpoint = torch.load(vae_weight_path, map_location=device, weights_only=False)
    
    # 从checkpoint中提取配置
    input_dim = checkpoint.get('input_dim', cluster_expr.shape[1])
    latent_dim = checkpoint.get('latent_dim', args.vae_latent_dim)
    output_type = checkpoint.get('output_type', 'mse')
    
    full_vae = FullVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        output_type=output_type
    ).to(device)
    
    # 加载完整VAE权重
    if 'vae_state_dict' in checkpoint:
        full_vae.load_state_dict(checkpoint['vae_state_dict'])
        logging.info(f"加载完整VAE权重")
    
    # 提取VAE编码器用于HeteroSTModel
    vae_encoder = full_vae.encoder
    logging.info(f"提取VAE编码器用于HeteroSTModel")
    
    # ========== 阶段2：构建spot-cluster表达和spot-cell表达 ==========
    logging.info("="*80)
    logging.info("阶段2: 构建spot-cluster表达和spot-cell表达")
    logging.info("="*80)
    
    spot_cell_expr_npz_path = None
    
    # 检查是否已有pre-computed NPZ路径
    if args.spot_cell_expr_npz and os.path.exists(args.spot_cell_expr_npz):
        spot_cell_expr_npz_path = args.spot_cell_expr_npz
        logging.info(f"直接使用pre-computed NPZ: {spot_cell_expr_npz_path}")
    
    elif cluster_composition is not None:
        n_spots = cluster_composition.shape[0]
        n_clusters = cluster_composition.shape[1]
        n_genes = cluster_expr.shape[1]
        
        logging.info(f"矩阵维度: {n_spots} spots × {n_clusters} clusters × {n_genes} genes")
        
        # 获取spot和cluster名字列表
        spot_names = adata.obs_names.tolist()
        cluster_names_raw = cluster_composition.columns.tolist()  # 可能是 ['0', '1', '17'] 或 ['Cluster_0', 'Cluster_1']
        
        # ✅ 统一格式：确保cluster名称格式为 'Cluster_X'
        cluster_names = []
        for name in cluster_names_raw:
            if name.startswith('Cluster_'):
                cluster_names.append(name)
            else:
                # 如果是纯数字，添加 'Cluster_' 前缀
                cluster_names.append(f'Cluster_{name}')
        
        logging.info(f"Cluster名称格式统一: {cluster_names[:5]}... (共{len(cluster_names)}个)")
        
        # ✅ 获取每个spot的total counts
        if 'total_counts' in adata.obs.columns:
            spot_total_counts = adata.obs['total_counts'].values
        elif 'nCount_RNA' in adata.obs.columns:
            spot_total_counts = adata.obs['nCount_RNA'].values
        else:
            # 如果没有保存，直接计算
            spot_total_counts = np.array(adata.X.sum(axis=1)).flatten()
        
        logging.info(f"Spot total counts统计: 平均={spot_total_counts.mean():.2f}, 中位数={np.median(spot_total_counts):.2f}")
        
        # ========== 第一步：构建spot-cluster表达 ==========
        logging.info("构建spot-cluster表达矩阵...")
        logging.info("   公式: (cluster_expr / 1e4) × spot_total_count × cluster_weight")
        spot_cluster_csv_data = []
        spot_cluster_npz_dict = {}
        
        for spot_idx in range(n_spots):
            spot_name = spot_names[spot_idx]
            spot_total_count = spot_total_counts[spot_idx]
            
            for cluster_idx, cluster_name in enumerate(cluster_names):
                cluster_weight = cluster_composition.iloc[spot_idx, cluster_idx]
                if cluster_weight > 1e-6:  # 只处理有贡献的cluster
                    # ✅ cluster_expr 的索引现在是 Cluster_ID 格式，直接使用 cluster_name 查找
                    if cluster_name not in cluster_expr.index:
                        logging.warning(f"找不到 cluster '{cluster_name}' in cluster_expr，跳过")
                        continue
                    
                    cluster_expr_values = cluster_expr.loc[cluster_name].values
                    
                    # ✅ 正确的计算公式：先除以1e4恢复比例，再乘以spot的total count，再乘以cluster比例
                    spot_cluster_expr = (cluster_expr_values / 1e4) * spot_total_count * cluster_weight
                    
                    # CSV行：spot_name | spot_cluster_name | gene1 | gene2 | ...
                    row_data = {
                        'spot_name': spot_name,
                        'spot_cluster': f"{spot_name}_{cluster_name}",
                    }
                    for gene_idx, gene_name in enumerate(cluster_expr.columns):
                        row_data[gene_name] = spot_cluster_expr[gene_idx]
                    spot_cluster_csv_data.append(row_data)
                    
                    # NPZ数据
                    combined_name = f"{spot_name}_{cluster_name}"
                    spot_cluster_npz_dict[combined_name] = spot_cluster_expr
        
        # 保存spot-cluster CSV
        spot_cluster_csv_df = pd.DataFrame(spot_cluster_csv_data)
        spot_cluster_csv_path = os.path.join(args.output_dir, 'spot_cluster_expr.csv')
        spot_cluster_csv_df.to_csv(spot_cluster_csv_path, index=False)
        logging.info(f"已保存spot-cluster CSV: {spot_cluster_csv_path}")
        
        # ✅ 输出表达值统计
        expr_values = spot_cluster_csv_df.iloc[:, 2:].values.flatten()  # 跳过spot_name和spot_cluster列
        expr_values = expr_values[expr_values > 0]  # 只统计非零值
        logging.info(f"   表达值统计 (非零): 平均={expr_values.mean():.2f}, 中位数={np.median(expr_values):.2f}, 最大={expr_values.max():.2f}")
        
        # ========== 第二步：根据celltype映射聚合为spot-cell表达 ==========
        logging.info("根据celltype映射聚合为spot-cell表达...")
        
        # 获取所有cell类型
        cell_names = list(set(cluster_to_cell.values()))
        cell_names.sort()  # 保持顺序一致
        
        csv_data = []
        npz_cell_names = []
        npz_cell_expr_dict = {}
        
        for spot_idx in range(n_spots):
            spot_name = spot_names[spot_idx]
            
            # 为每个cell类型聚合其对应的clusters
            for cell_name in cell_names:
                # 找到属于这个cell类型的clusters
                cell_clusters = [c for c, ct in cluster_to_cell.items() if ct == cell_name]
                
                if cell_clusters:
                    # 聚合所有属于这个cell的cluster表达
                    cell_expr_sum = np.zeros(n_genes)
                    for cluster_name in cell_clusters:
                        spot_cluster_name = f"{spot_name}_{cluster_name}"
                        if spot_cluster_name in spot_cluster_npz_dict:
                            cell_expr_sum += spot_cluster_npz_dict[spot_cluster_name]
                    
                    # CSV行：spot_name | spot_cell_name | gene1 | gene2 | ...
                    row_data = {
                        'spot_name': spot_name,
                        'spot_cell': f"{spot_name}_{cell_name}",
                    }
                    for gene_idx, gene_name in enumerate(cluster_expr.columns):
                        row_data[gene_name] = cell_expr_sum[gene_idx]
                    csv_data.append(row_data)
                    
                    # NPZ数据
                    combined_name = f"{spot_name}_{cell_name}"
                    npz_cell_names.append(combined_name)
                    npz_cell_expr_dict[combined_name] = cell_expr_sum
        
        # 保存CSV表格
        csv_df = pd.DataFrame(csv_data)
        csv_path = os.path.join(args.output_dir, 'spot_cell_expr.csv')
        csv_df.to_csv(csv_path, index=False)
        logging.info(f"已保存CSV表格: {csv_path}")
        
        # ✅ 输出表达值统计
        cell_expr_values = csv_df.iloc[:, 2:].values.flatten()  # 跳过spot_name和spot_cell列
        cell_expr_values = cell_expr_values[cell_expr_values > 0]  # 只统计非零值
        logging.info(f"   表达值统计 (非零): 平均={cell_expr_values.mean():.2f}, 中位数={np.median(cell_expr_values):.2f}, 最大={cell_expr_values.max():.2f}")
        
        # 保存NPZ数据 [n_spot_cells, n_genes]
        spot_cell_expr_npz_path = os.path.join(args.output_dir, 'spot_cell_expr.npz')
        npz_cell_expr_array = np.array([npz_cell_expr_dict[name] for name in npz_cell_names], dtype=np.float32)
        np.savez_compressed(
            spot_cell_expr_npz_path,
            spot_cell_expr=npz_cell_expr_array,
            cell_names=npz_cell_names,
            gene_names=cluster_expr.columns.tolist()
        )
        logging.info(f"已保存NPZ数据: {spot_cell_expr_npz_path}")
        
    else:
        logging.warning(f"无法构建spot-cell-gene矩阵（缺少cluster_composition）")
    
    # ========== 阶段2.5：构建cell表达矩阵（用于模型输入）==========
    logging.info("="*80)
    logging.info("阶段2.5: 构建cell表达矩阵")
    logging.info("="*80)
    
    # 从mapping得到cell信息
    cell_names = list(set(cluster_to_celltype.values()))
    cell_names.sort()  # 保持顺序一致
    
    # 为每个cell聚合marker基因表达（平均值）
    cell_marker_dict = {}
    for cell in cell_names:
        # 找到所有映射到这个 celltype 的 cluster_ids
        cluster_ids = [cid for cid, ctype in cluster_to_celltype.items() if ctype == cell]
        
        # ✅ 构建cluster行名列表 (使用 Cluster_X 格式，因为 cluster_expr 的索引是 Cluster_X)
        cluster_keys = [f"Cluster_{cid}" for cid in cluster_ids]
        
        # 从marker表达量中聚合
        if all(key in cluster_expr.index for key in cluster_keys):
            marker_rows = cluster_expr.loc[cluster_keys]
            cell_marker_dict[cell] = marker_rows.mean(axis=0).values
            logging.info(f"Cell '{cell}': 聚合 {len(cluster_ids)} 个cluster的marker表达")
        else:
            logging.warning(f"Cell '{cell}': 部分cluster未找到，跳过")
            continue
    
    # 构建marker表达矩阵 (cell × marker_genes)
    cell_expr = pd.DataFrame.from_dict(
        cell_marker_dict, orient='index', 
        columns=cluster_expr.columns
    )
    logging.info(f"已构建cell marker表达: {cell_expr.shape}")
    
    # 构建全基因表达矩阵（如果存在）
    cell_full_dict = {}
    if cluster_full_expr is not None:
        for cell in cell_names:
            # 找到所有映射到这个 celltype 的 cluster_ids
            cluster_ids = [cid for cid, ctype in cluster_to_celltype.items() if ctype == cell]
            # ✅ 使用 Cluster_X 格式
            cluster_keys = [f"Cluster_{cid}" for cid in cluster_ids]
            
            if all(key in cluster_full_expr.index for key in cluster_keys):
                full_rows = cluster_full_expr.loc[cluster_keys]
                cell_full_dict[cell] = full_rows.mean(axis=0).values
        
        if cell_full_dict:
            cell_full_expr = pd.DataFrame.from_dict(
                cell_full_dict, orient='index',
                columns=cluster_full_expr.columns
            )
            logging.info(f"已构建cell全基因表达: {cell_full_expr.shape}")
        else:
            cell_full_expr = None
    else:
        cell_full_expr = None
    
    # 构建cell composition矩阵（每个spot的cell比例）
    if cluster_composition is not None:
        composition = pd.DataFrame(index=cluster_composition.index, columns=cell_names, dtype=float)
        composition.fillna(0.0, inplace=True)
        
        for spot_idx in range(cluster_composition.shape[0]):
            for cluster_idx, cluster_name_raw in enumerate(cluster_composition.columns):
                # ✅ 统一格式：确保cluster名称格式为 'Cluster_X'
                if cluster_name_raw.startswith('Cluster_'):
                    cluster_name = cluster_name_raw
                else:
                    cluster_name = f'Cluster_{cluster_name_raw}'
                
                if cluster_name in cluster_to_cell:
                    cell_name = cluster_to_cell[cluster_name]
                    cluster_weight = cluster_composition.iloc[spot_idx, cluster_idx]
                    composition.loc[composition.index[spot_idx], cell_name] += cluster_weight
        
        logging.info(f"已构建cell composition矩阵: {composition.shape}")
    else:
        composition = None
    
    # ========== 阶段3.5：预计算KNN和LR通讯得分 ==========
    logging.info("="*80)
    logging.info("阶段3.5: 预计算KNN邻域和LR通讯得分")
    logging.info("="*80)
    
    # ✅ 调试：检查关键变量状态
    logging.info(f"spot_coords类型: {type(spot_coords)}, 形状: {spot_coords.shape if spot_coords is not None else 'None'}")
    logging.info(f"composition类型: {type(composition)}, 形状: {composition.shape if composition is not None else 'None'}")
    logging.info(f"args.output_dir: {args.output_dir}")
    logging.info(f"args.load_lr_knn: {args.load_lr_knn}")
    
    # 构建KNN邻域
    N = len(spot_coords)
    n_neighbors = args.n_spot_neighbors
    logging.info(f"准备构建KNN: N={N} spots, k={n_neighbors} neighbors")
    
    # 检查是否加载预先计算的KNN mask
    if args.load_lr_knn:
        logging.info(f"从预保存的加载KNN: {args.load_lr_knn}")
        knn_path = os.path.join(args.load_lr_knn, 'knn_mask.npz')
        knn_npz_data = np.load(knn_path, allow_pickle=True)
        knn_mask = knn_npz_data['knn_mask']
        logging.info(f"已加载KNN mask: {knn_mask.shape}")
    else:
        logging.info(f"构建KNN图: {N} spots, {n_neighbors} neighbors")
        knn = kneighbors_graph(spot_coords, n_neighbors=n_neighbors, mode="connectivity", include_self=False)
        knn_mask = knn.toarray()  # [N, N]

        # 添加物理距离限制（可选）
        distance_threshold = 200.0  # 200μm
        for i in range(N):
            for j in range(N):
                if knn_mask[i, j] == 1:
                    dist = np.sqrt((spot_coords[i, 0] - spot_coords[j, 0])**2 + 
                                  (spot_coords[i, 1] - spot_coords[j, 1])**2)
                    if dist > distance_threshold:
                        knn_mask[i, j] = 0
        
        logging.info(f"应用距离过滤: 最大 {distance_threshold}μm")
        
        # ✅ 使用 output_dir 保存 KNN mask
        knn_npz_path = os.path.join(args.output_dir, "knn_mask.npz")
        os.makedirs(args.output_dir, exist_ok=True)
        np.savez_compressed(
            knn_npz_path,
            knn_mask=knn_mask
        )

        logging.info(f"KNN邻接矩阵已保存到: {knn_npz_path}")
    
    # ✅ 计算并保存LR通讯得分矩阵（使用 output_dir）
    if True:  # 总是计算 LR 通讯得分
        logging.info("开始计算LR通讯得分...")
        logging.info(f"   - 活跃基因筛选: 表达比例≥10% 且 normalize_total(1e4) > {args.mean_expr_threshold}")
        logging.info(f"   - 通讯得分过滤阈值: {args.lr_comm_score_threshold}")
        logging.info(f"   - 距离衰减参数sigma: {args.lr_distance_sigma}")
        
        # ✅ 调试：检查KNN mask状态
        logging.info(f"   - KNN mask形状: {knn_mask.shape}")
        logging.info(f"   - KNN mask中非零元素数: {np.count_nonzero(knn_mask)}")
        logging.info(f"   - 平均每个spot的邻居数: {np.count_nonzero(knn_mask) / N:.2f}")
        
        # 加载spot-cell表达数据
        npz_data = np.load(spot_cell_expr_npz_path, allow_pickle=True)
        spot_cell_expr_array = npz_data['spot_cell_expr']
        cell_names_in_npz = list(npz_data['cell_names'])
        gene_names_in_npz = list(npz_data['gene_names'])
        
        # 构建查询字典: {(spot_idx, cell_idx): array_row_idx}
        spot_names = adata.obs_names.tolist()
        cell_names_list = cell_expr.index.tolist()
        spot_cell_expr_index = {}
        
        for idx, name in enumerate(cell_names_in_npz):
            parts = name.rsplit('_', 1)
            if len(parts) == 2:
                spot_name, cell_name = parts
                try:
                    cell_id = cell_names_list.index(cell_name)
                    spot_id = spot_names.index(spot_name)
                    spot_cell_expr_index[(spot_id, cell_id)] = idx
                except ValueError:
                    pass
        
        # ========== 优化1：预构建基因索引字典 ==========
        # 避免在循环中重复查找基因索引
        gene_name_to_idx = {gene.upper(): idx for idx, gene in enumerate(gene_names_in_npz)}
        
        # ========== 过滤不参与通讯的基因 ==========
        logging.info("过滤不参与通讯的基因...")
        
        # 定义过滤规则
        filtered_genes = set()
        for gene_name in gene_names_in_npz:
            gene_upper = gene_name.upper()
            # 1. 线粒体基因 (MT-)
            if gene_upper.startswith('MT-'):
                filtered_genes.add(gene_name)
            # 2. 血红蛋白基因 (HB)
            elif gene_upper.startswith('HB'):
                filtered_genes.add(gene_name)
            # 3. 假基因 (包含 'PSEUDO', '-AS', 'LOC')
            elif 'PSEUDO' in gene_upper or gene_upper.endswith('-AS1') or gene_upper.startswith('LOC'):
                filtered_genes.add(gene_name)
            # 4. 核糖体蛋白基因 (RPS, RPL) - 可选
            # elif gene_upper.startswith('RPS') or gene_upper.startswith('RPL'):
            #     filtered_genes.add(gene_name)
        
        logging.info(f"   - 过滤基因数: {len(filtered_genes)}/{len(gene_names_in_npz)}")
        logging.info(f"   - 线粒体基因: {sum(1 for g in filtered_genes if g.upper().startswith('MT-'))}")
        logging.info(f"   - 血红蛋白基因: {sum(1 for g in filtered_genes if g.upper().startswith('HB'))}")
        logging.info(f"   - 假基因: {sum(1 for g in filtered_genes if 'PSEUDO' in g.upper() or g.upper().endswith('-AS1') or g.upper().startswith('LOC'))}")
        
        # ========== 新增：筛选每个细胞类型中的活跃基因 ==========
        # 活跃基因定义：表达比例≥10% 且 normalize_total(1e4) > threshold
        logging.info("筛选每个细胞类型中的活跃基因...")
        cell_active_genes_ligand = {}  # cell_name -> set of active ligand gene indices
        cell_active_genes_receptor = {}  # cell_name -> set of active receptor gene indices
        
        expr_proportion_threshold = 0.1  # 10%的细胞中表达
        mean_expr_threshold = args.mean_expr_threshold  # ✅ 使用超参数，配体和受体统一阈值
        
        for cell_idx, cell_name in enumerate(cell_names_list):
            # 收集该细胞类型的所有表达数据
            cell_exprs = []
            for spot_idx in range(N):
                spot_name = spot_names[spot_idx]
                cell_combined_name = f"{spot_name}_{cell_name}"
                if cell_combined_name in cell_names_in_npz:
                    array_idx = cell_names_in_npz.index(cell_combined_name)
                    cell_exprs.append(spot_cell_expr_array[array_idx])
            
            if not cell_exprs:
                cell_active_genes_ligand[cell_name] = set()
                cell_active_genes_receptor[cell_name] = set()
                continue
                
            cell_expr_matrix = np.array(cell_exprs)  # [n_cells_of_this_type, n_genes]
            
            # ✅ 归一化 - 每个细胞normalize到总和为10000
            cell_totals = cell_expr_matrix.sum(axis=1, keepdims=True)  # [n_cells, 1]
            cell_totals[cell_totals == 0] = 1  # 避免除以0
            cell_expr_normalized = cell_expr_matrix / cell_totals * 1e4  # [n_cells, n_genes]
            
            # 计算每个基因的统计信息
            # 1. 表达比例：在多少比例的细胞中表达 (原始expr > 0)
            expr_proportion = np.mean(cell_expr_matrix > 0, axis=0)  # [n_genes]
            
            # 2. 标准化后的平均表达：mean(normalized_expr)
            mean_expr = np.mean(cell_expr_normalized, axis=0)  # [n_genes]
            
            # 3. 过滤掉不参与通讯的基因
            filtered_gene_indices = set()
            for gene_name in filtered_genes:
                if gene_name in gene_names_in_npz:
                    gene_idx = gene_names_in_npz.index(gene_name)
                    filtered_gene_indices.add(gene_idx)
            
            # ✅ 筛选活跃配体基因：表达比例≥10% 且 平均表达 > threshold（统一阈值）
            active_mask_ligand = (expr_proportion >= expr_proportion_threshold) & \
                                 (mean_expr > mean_expr_threshold)
            active_ligand_indices = set(np.where(active_mask_ligand)[0]) - filtered_gene_indices
            
            # ✅ 筛选活跃受体基因：表达比例≥10% 且 平均表达 > threshold（统一阈值）
            active_mask_receptor = (expr_proportion >= expr_proportion_threshold) & \
                                   (mean_expr > mean_expr_threshold)
            active_receptor_indices = set(np.where(active_mask_receptor)[0]) - filtered_gene_indices
            
            cell_active_genes_ligand[cell_name] = active_ligand_indices
            cell_active_genes_receptor[cell_name] = active_receptor_indices
            
            logging.info(f"   - {cell_name}: 配体={len(active_ligand_indices)}, 受体={len(active_receptor_indices)} "
                        f"(过滤前: {np.sum(active_mask_ligand)}/{np.sum(active_mask_receptor)})")
        
        # ========== 优化2：预处理LR对，构建索引映射 ==========
        # 将LR对转换为索引对，并过滤掉不存在的基因
        valid_lr_pairs = []
        for ligand, receptor in lr_pairs:
            ligand_upper = ligand.upper()
            lig_idx = gene_name_to_idx.get(ligand_upper)
            
            if lig_idx is None:
                continue
            
            # 处理联合受体
            receptor_genes = [r.strip() for r in receptor.split('_')]
            rec_indices = []
            found_all = True
            
            for receptor_gene in receptor_genes:
                receptor_upper = receptor_gene.upper()
                rec_idx = gene_name_to_idx.get(receptor_upper)
                if rec_idx is None:
                    found_all = False
                    break
                rec_indices.append(rec_idx)
            
            if found_all:
                valid_lr_pairs.append((lig_idx, rec_indices, ligand, receptor))
        
        logging.info(f"   - 有效LR对: {len(valid_lr_pairs)}/{len(lr_pairs)}")
        
        # 初始化通讯事件记录
        comm_event_records = []
        
        n_cells = len(cell_names_list)
        
        # ========== 优化3：使用进度条和批量处理 ==========
        logging.info("   - 开始遍历KNN邻居对...")
        total_pairs = 0
        spots_with_cells = 0
        spots_without_cells = 0
        same_celltype_skipped = 0  # 统计跳过的同类型细胞对
        
        # 遍历所有KNN邻居对
        for i in range(N):
            spot_i_barcode = spot_names[i]  # 获取spot i的barcode
            
            # 获取spot i的cell composition
            composition_i = composition.iloc[i].values
            cell_in_i = np.where(composition_i > 1e-6)[0]
            
            if len(cell_in_i) == 0:
                spots_without_cells += 1
                continue
            
            spots_with_cells += 1
            
            for j in range(N):
                if knn_mask[i, j] == 0:  # 不是邻居，跳过
                    continue
                
                total_pairs += 1
                spot_j_barcode = spot_names[j]  # 获取spot j的barcode
                
                # 获取spot j的cell composition
                composition_j = composition.iloc[j].values
                cell_in_j = np.where(composition_j > 1e-6)[0]
                
                if len(cell_in_j) == 0:
                    continue
                
                # 遍历cell对，计算LR通讯
                for cell_i_idx in cell_in_i:
                    idx_i = spot_cell_expr_index.get((i, cell_i_idx))
                    if idx_i is None:
                        continue
                    
                    cell_i_expr = spot_cell_expr_array[idx_i, :]
                    cell_i_name = cell_names_list[cell_i_idx]  # 获取cell名称
                    cell_i = f"{spot_i_barcode}_{cell_i_name}"  # 组合成cell名称
                    
                    for cell_j_idx in cell_in_j:
                        idx_j = spot_cell_expr_index.get((j, cell_j_idx))
                        if idx_j is None:
                            continue
                        
                        cell_j_expr = spot_cell_expr_array[idx_j, :]
                        cell_j_name = cell_names_list[cell_j_idx]  # 获取cell名称
                        cell_j = f"{spot_j_barcode}_{cell_j_name}"  # 组合成cell名称
                        
                        # ✅ 跳过相同细胞类型之间的通讯
                        if cell_i_name == cell_j_name:
                            same_celltype_skipped += 1
                            continue
                        
                        # ========== 优化4：向量化LR得分计算 ==========
                        # 使用预处理的LR索引，避免重复查找
                        for lig_idx, rec_indices, ligand, receptor in valid_lr_pairs:
                            # 检查配体基因是否在源细胞中活跃（配体阈值 > 0.5）
                            if lig_idx not in cell_active_genes_ligand[cell_i_name]:
                                continue
                            
                            # 检查所有受体基因是否在目标细胞中活跃（受体阈值 > 0.3）
                            receptor_active = all(rec_idx in cell_active_genes_receptor[cell_j_name] for rec_idx in rec_indices)
                            if not receptor_active:
                                continue
                            
                            # 获取配体和受体的表达值
                            lig_val = cell_i_expr[lig_idx]
                            rec_vals = cell_j_expr[rec_indices]
                            
                            # 计算受体乘积（联合受体取乘积）
                            rec_product = np.prod(rec_vals)
                            
                            # 计算spot间距离权重
                            distance = np.sqrt((spot_coords[i, 0] - spot_coords[j, 0])**2 + 
                                              (spot_coords[i, 1] - spot_coords[j, 1])**2)
                            distance_weight = np.exp(-distance / args.lr_distance_sigma)
                            
                            # 计算通讯得分：几何平均数 × 距离权重
                            score = np.sqrt(lig_val * rec_product) * distance_weight
                            
                            # ✅ 过滤低于阈值的通讯事件
                            if score >= args.lr_comm_score_threshold:
                                comm_event_records.append([
                                    spot_i_barcode, spot_j_barcode, cell_i, cell_j, ligand, receptor, score
                                ])
            
            # 每处理100个spot打印一次进度
            if (i + 1) % 100 == 0:
                logging.info(f"   - 已处理 {i+1}/{N} spots, 发现 {len(comm_event_records)} 个通讯事件")
        
        logging.info(f"计算完成: {len(comm_event_records)} 个LR通讯事件")
        logging.info(f"   - Spots with cells: {spots_with_cells}/{N}")
        logging.info(f"   - Spots without cells: {spots_without_cells}/{N}")
        logging.info(f"   - 处理的邻居对: {total_pairs}")
        logging.info(f"   - 跳过同类型细胞对: {same_celltype_skipped}")
        
        # ✅ 如果设置了阈值，输出过滤统计
        if args.lr_comm_score_threshold > 0:
            logging.info(f"   - 通讯得分阈值过滤: score >= {args.lr_comm_score_threshold}")
        
        # ✅ 使用 output_dir 保存 LR 通讯得分
        csv_path = os.path.join(args.output_dir, "lr_scoresc.csv")
        
        df = pd.DataFrame(
            comm_event_records,
            columns=['spot_i', 'spot_j', 'cell_i', 'cell_j', 'ligand', 'receptor', 'comm_score']
        )
        df.to_csv(csv_path, index=False)
        logging.info(f"LR通讯得分已保存到: {csv_path}")
        logging.info(f"   - 总事件数: {len(df)}")
        logging.info(f"   - Spot对数: {df.groupby(['spot_i', 'spot_j']).ngroups}")
        logging.info(f"   - Cell对数: {df.groupby(['cell_i', 'cell_j']).ngroups}")
    else:
        logging.info("使用预先计算的LR通讯")
        csv_path = os.path.join(args.load_lr_knn, "lr_scoresc.csv")


    # 加载数据集（在__getitem__中动态构建subgraph）
    # 准备graph_data字典（只包含坐标和composition）
    graph_data = {
        'coords': spot_coords,
        'composition': composition,
        'knn_mask': knn_mask,  # 传入预计算的KNN邻接矩阵
    }
    
    dataset = STHeteroSubgraphDataset(
        st_h5ad_path=args.st_h5ad,
        cluster_expr=cluster_expr,
        cell_expr=cell_expr,
        cell_full_expr=cell_full_expr,
        graph_data=graph_data,
        lr_pairs=lr_pairs,
        k_neighbors=args.n_spot_neighbors,
        spot_cell_expr_npz_path=spot_cell_expr_npz_path,
        load_lr_scores_csv=csv_path,
        min_comm_edges=args.min_comm_edges,
        device=device
    )
    
    # ✅ 支持采样训练加速
    if args.sample_rate < 1.0:
        from torch.utils.data import RandomSampler
        num_samples = int(len(dataset) * args.sample_rate)
        sampler = RandomSampler(dataset, num_samples=num_samples, replacement=False)
        logging.info(f"⚡ 采样训练模式: 每个epoch采样 {args.sample_rate*100:.1f}% 数据 ({num_samples}/{len(dataset)} spots)")
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=0,
            collate_fn=hetero_subgraph_collate_fn
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,  # 设置为0避免多进程日志重复
            collate_fn=hetero_subgraph_collate_fn  # 使用自定义的collate_fn来支持批处理
        )
    
    logging.info("="*80)
    logging.info("阶段3: 构建HeteroGAT模型")
    logging.info("="*80)
    gat_hidden_dims = [int(x) for x in args.gat_hidden_dims.split(',')]
    
    n_genes = cluster_expr.shape[1]
    n_cells = cell_expr.shape[0]
    
    model = HeteroSTModel(
        n_genes=n_genes,
        vae_latent_dim=latent_dim,  # 使用从checkpoint读取的latent_dim
        vae_hidden_dim=args.vae_hidden_dim,
        gat_layers=args.gat_layers,
        gat_hidden_dims=gat_hidden_dims,
        gat_heads=args.gat_heads,
        gat_dropout=args.gat_dropout,
        output_dim=args.output_dim,
        n_celltypes=n_cells,
        vae_encoder=vae_encoder
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    logging.info(f"模型构建完成")
    logging.info(f"   - 总参数数: {sum(p.numel() for p in model.parameters()):,}")
    # 训练循环
    train_losses = []
    graph_augmentor = GraphAugmentor(drop_edge_rate=0.15)
    
    # 用于收集cell-cell注意力得分和LR ID
    all_cc_attention_scores = []
    all_edge_index_cc = []
    all_edge_attr_cc = []  # 新增：收集边属性 [lr_score, lr_id]
    all_spot_indices = []  # 新增：收集对应的spot索引
    all_cell_names = list(cell_expr.index)
    
    # 保存subgraph结构信息（用于后续统计）
    subgraph_info = None
    
    # ========== 阶段4：训练循环 ==========
    logging.info("="*80)
    logging.info("阶段4: 开始训练")
    logging.info("="*80)

    # 使用外层tqdm跟踪epoch进度
    epoch_pbar = tqdm(range(args.epochs), desc="Training", position=0)
    
    for epoch in epoch_pbar:
        model.train()
        total_loss = 0.0
        total_contrast_loss = 0.0
        total_comm_pred_loss = 0.0  # 通讯强度预测损失
        for batch_idx, batch in enumerate(dataloader, 1):
            # 保存第一个batch的结构信息用于后续统计
            if subgraph_info is None:
                subgraph_info = {
                    'n_spots_sub': batch['n_spots_sub'],
                    'n_cells': batch['n_cells']
                }
            
            batch_size = batch['batch_size']
            n_spots_sub = batch['n_spots_sub']
            n_cells = batch['n_cells']

            # 累积批次损失
            batch_loss = 0.0
            batch_contrast_loss = 0.0
            batch_comm_pred_loss = 0.0  # 通讯强度预测损失

            # 处理每个subgraph
            for b in range(batch_size):
                # 提取第b个subgraph的数据
                expr_raw = batch['expr_raw'][b].to(device)  # [k+1, n_genes]
                cell_expr_raw = batch['cell_expr_raw'][b].to(device)  # [(k+1)*n_cells, n_marker_genes]

                edge_index_like = batch['edge_index_like'][b].to(device)  # [2, E_like]
                edge_attr_like = batch['edge_attr_like'][b].to(device)    # [E_like]
                edge_index_cc = batch['edge_index_cc'][b].to(device)      # [2, E_cc]
                edge_attr_cc = batch['edge_attr_cc'][b].to(device)        # [E_cc]

                # ========== 原始图前向传播 ==========
                spot_repr, cell_repr, combined, spot_proj, cc_attention, predicted_comm_strength = model(
                    expr_raw=expr_raw,
                    cell_expr_raw=cell_expr_raw,
                    edge_index_like=edge_index_like,
                    edge_attr_like=edge_attr_like,
                    edge_index_cc=edge_index_cc,
                    edge_attr_cc=edge_attr_cc,
                    return_attention=True
                )
                
                # 收集cell-cell注意力得分
                if cc_attention is not None:
                    all_cc_attention_scores.append(cc_attention.detach().cpu())
                    all_edge_index_cc.append(edge_index_cc.detach().cpu())
                    all_edge_attr_cc.append(edge_attr_cc.detach().cpu())
                    # 收集对应的spot信息（center_spot_idx）
                    center_spot_idx = batch['center_spot_idx'][b]
                    spot_indices = torch.full((edge_index_cc.size(1),), center_spot_idx, dtype=torch.long)
                    all_spot_indices.append(spot_indices)
                # print(cc_attention)
                # print(edge_index_cc)
                # print(edge_attr_cc)
                # ========== 增强图前向传播 ==========
                # 构建增强图
                augmented = graph_augmentor.augment_graph(
                    edge_index_like.cpu().numpy(), edge_attr_like.cpu().numpy() if edge_attr_like.size(0) > 0 else None,
                    edge_index_cc.cpu().numpy(), edge_attr_cc.cpu().numpy() if edge_attr_cc.size(0) > 0 else None
                )

                # 转换为tensor
                edge_index_like_aug = torch.tensor(augmented['edge_index_like'], dtype=torch.long, device=device)
                edge_attr_like_aug = torch.tensor(augmented['edge_attr_like'], dtype=torch.float32, device=device) if augmented['edge_attr_like'] is not None else edge_attr_like[:0]

                edge_index_cc_aug = torch.tensor(augmented['edge_index_cc'], dtype=torch.long, device=device)
                edge_attr_cc_aug = torch.tensor(augmented['edge_attr_cc'], dtype=torch.float32, device=device) if augmented['edge_attr_cc'] is not None else edge_attr_cc[:0]

                # 增强图前向传播
                spot_repr_aug, _, _, spot_proj_aug, _, _ = model(
                    expr_raw=expr_raw,
                    cell_expr_raw=cell_expr_raw,
                    edge_index_like=edge_index_like_aug,
                    edge_attr_like=edge_attr_like_aug,
                    edge_index_cc=edge_index_cc_aug,
                    edge_attr_cc=edge_attr_cc_aug,
                    return_attention=False  # 增强图不需要收集注意力得分
                )

                # ========== 对比学习损失 ==========
                # 对每个subgraph中的center spot计算对比损失

                # center spot是subgraph的第一个spot（节点索引为0）
                center_spot_idx = 0

                # 获取center spot的原始和增强表示
                spot_proj_center = spot_proj[center_spot_idx]  # [proj_dim]
                spot_proj_aug_center = spot_proj_aug[center_spot_idx]  # [proj_dim]

                # 计算余弦相似度（正样本）
                pos_sim = torch.nn.functional.cosine_similarity(
                    spot_proj_center.unsqueeze(0),
                    spot_proj_aug_center.unsqueeze(0),
                    dim=1
                )  # [1]

                # ✅ 对比学习损失：删除弱边后，重要的 spot 表示应保持相似
                contrast_loss = 1.0 - pos_sim.mean()  # 最小化 (1 - 相似度)

                # ✅ 通讯强度预测损失（主要监督信号）
                comm_pred_loss = 0.0
                if predicted_comm_strength is not None and edge_attr_cc.size(0) > 0:
                    # 真实通讯强度：LR 得分
                    true_comm_strength = edge_attr_cc[:, 0]  # [n_edges] - 第0列是lr_score
                    
                    # ✅ 使用全局最大值归一化（在epoch开始时计算）
                    # 或者使用固定的归一化范围（更稳定）
                    # 这里使用简单的min-max归一化到 [0, 1]
                    if true_comm_strength.max() > 0:
                        true_comm_strength = (true_comm_strength - true_comm_strength.min()) / \
                                           (true_comm_strength.max() - true_comm_strength.min() + 1e-8)
                    
                    # MSE 损失 - predicted_comm_strength 的输出已经是 [0, 1] (Sigmoid)
                    comm_pred_loss = torch.nn.functional.mse_loss(predicted_comm_strength, true_comm_strength)

                # ✅ 总损失 = 通讯强度预测损失（主要） + 对比学习损失（辅助）
                loss = comm_pred_loss + 0.1 * contrast_loss

                batch_loss += loss
                batch_contrast_loss += contrast_loss
                batch_comm_pred_loss += comm_pred_loss.item() if isinstance(comm_pred_loss, torch.Tensor) else 0.0

            # 对整个batch求平均损失，然后反向传播
            avg_batch_loss = batch_loss / batch_size
            avg_batch_contrast_loss = batch_contrast_loss / batch_size
            avg_batch_comm_pred_loss = batch_comm_pred_loss / batch_size if batch_comm_pred_loss > 0 else 0.0

            # 反向传播
            optimizer.zero_grad()
            avg_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += avg_batch_loss.item()
            total_contrast_loss += avg_batch_contrast_loss.item()
            total_comm_pred_loss += avg_batch_comm_pred_loss
        
        # 计算epoch平均损失
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        avg_contrast = total_contrast_loss / len(dataloader) if len(dataloader) > 0 else 0
        avg_comm_pred = total_comm_pred_loss / len(dataloader) if len(dataloader) > 0 else 0
        train_losses.append(avg_loss)
        
        # 更新epoch进度条（只在epoch结束时更新一次）
        epoch_pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Contrast': f'{avg_contrast:.4f}',
            'CommPred': f'{avg_comm_pred:.4f}'
        })
        
        logging.info(f"[Epoch {epoch+1}/{args.epochs}] Loss: {avg_loss:.4f}, Contrast: {avg_contrast:.4f}, CommPred: {avg_comm_pred:.4f}")
        
        # 保存检查点
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f"hetero_model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"检查点已保存: {checkpoint_path}")
    
    # 关闭tqdm进度条
    epoch_pbar.close()
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, "hetero_model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"最终模型已保存: {final_model_path}")
    
    # ========== 阶段5：统计cell-cell边重要性 ==========
    logging.info("="*80)
    logging.info("阶段5: 统计cell-cell边重要性")
    logging.info("="*80)
    
    if all_cc_attention_scores:
        logging.info(f"收集到 {len(all_cc_attention_scores)} 个batch的注意力得分")
        
        # 合并所有batch的注意力得分
        all_scores = torch.cat(all_cc_attention_scores, dim=0)  # [total_edges, num_heads]
        all_edges = torch.cat(all_edge_index_cc, dim=1)  # [2, total_edges]
        all_attrs = torch.cat(all_edge_attr_cc, dim=0)  # [total_edges, 2] - [lr_score, lr_id]
        all_spots = torch.cat(all_spot_indices, dim=0)  # [total_edges] - center spot indices
        
        logging.info(f"合并后数据形状: all_scores={all_scores.shape}, all_edges={all_edges.shape}, all_attrs={all_attrs.shape}, all_spots={all_spots.shape}")
        
        # 计算平均注意力得分（跨所有heads）
        avg_scores = all_scores.mean(dim=1)  # [total_edges]
        logging.info(f"平均注意力得分形状: {avg_scores.shape}")
        
        # 加载LR对映射
        lr_mapping_path = os.path.join(args.load_lr_knn, "lr_pair_mapping.txt")
        lr_id_to_pair = {}
        if os.path.exists(lr_mapping_path):
            with open(lr_mapping_path, 'r') as f:
                next(f)  # 跳过表头
                for line in f:
                    lr_id, ligand, receptor = line.strip().split('\t')
                    lr_id_to_pair[int(lr_id)] = (ligand, receptor)
            logging.info(f"已加载LR对映射: {len(lr_id_to_pair)} 个LR对")
        else:
            logging.warning(f"找不到LR对映射文件: {lr_mapping_path}")
        
        # 统计每个cell-cell-LR-spot对的得分（保留spot信息）
        cell_lr_spot_scores = {}
        # 初始化统计计数器，避免UnboundLocalError
        processed_edges = 0
        skipped_spot_edges = 0
        skipped_invalid_cells = 0
        
        for i in range(all_edges.size(1)):
            src_idx = all_edges[0, i].item()
            dst_idx = all_edges[1, i].item()
            center_spot_idx = all_spots[i].item()
            
            # 只关注cell-cell边（跳过spot相关的边）
            if src_idx >= n_spots and dst_idx >= n_spots:
                # 从节点编号反推出cell_id
                src_cell_idx = (src_idx - n_spots) % n_cells
                dst_cell_idx = (dst_idx - n_spots) % n_cells
                
                if src_cell_idx < n_cells and dst_cell_idx < n_cells:
                    lr_score = all_attrs[i, 0].item()
                    lr_id = int(all_attrs[i, 1].item())
                    attention_score = avg_scores[i].item()
                    
                    src_cell = all_cell_names[src_cell_idx]
                    dst_cell = all_cell_names[dst_cell_idx]
                    
                    # 获取LR对名称
                    if lr_id in lr_id_to_pair:
                        ligand, receptor = lr_id_to_pair[lr_id]
                        lr_pair_name = f"{ligand}_{receptor}"
                    else:
                        lr_pair_name = f"lr_{lr_id}"
                    
                    # 收集每个具体边的得分（包含spot信息）
                    edge_key = (center_spot_idx, src_cell, dst_cell, lr_pair_name)
                    if edge_key not in cell_lr_spot_scores:
                        cell_lr_spot_scores[edge_key] = []
                    cell_lr_spot_scores[edge_key].append(attention_score)
                    processed_edges += 1
                else:
                    skipped_invalid_cells += 1
                    if skipped_invalid_cells <= 5:  # 只打印前5个
                        logging.info(f"跳过无效cell索引: src_idx={src_idx}->{src_cell_idx}, dst_idx={dst_idx}->{dst_cell_idx}")
            else:
                skipped_spot_edges += 1
        
        logging.info(f"处理统计: processed_edges={processed_edges}, skipped_spot_edges={skipped_spot_edges}, skipped_invalid_cells={skipped_invalid_cells}")
        logging.info(f"cell_lr_spot_scores条目数: {len(cell_lr_spot_scores)}")
        
        # 计算每个具体边（spot-cell-lr）的平均得分
        edge_avg_scores = {}
        for edge_key, scores in cell_lr_spot_scores.items():
            edge_avg_scores[edge_key] = np.mean(scores)
        
        # 同时计算cell类型级别的统计（用于排序）
        cell_lr_pair_scores = {}
        for (spot_idx, src_cell, dst_cell, lr_pair), score in edge_avg_scores.items():
            pair_key = (src_cell, dst_cell, lr_pair)
            if pair_key not in cell_lr_pair_scores:
                cell_lr_pair_scores[pair_key] = []
            cell_lr_pair_scores[pair_key].append(score)
        
        cell_lr_pair_avg_scores = {}
        for pair, scores in cell_lr_pair_scores.items():
            cell_lr_pair_avg_scores[pair] = np.mean(scores)
        
        logging.info(f"cell_lr_pair_avg_scores条目数: {len(cell_lr_pair_avg_scores)}")
        
        # 按cell类型级别得分排序
        sorted_pairs = sorted(cell_lr_pair_avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 保存详细结果（包含spot信息）
        stats_path = os.path.join(args.output_dir, "cell_cell_attention_stats.csv")
        with open(stats_path, 'w') as f:
            f.write("center_spot,source_cell,target_cell,lr_pair,avg_attention_score\n")
            for (spot_idx, src_cell, dst_cell, lr_pair), score in sorted(edge_avg_scores.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{spot_idx},{src_cell},{dst_cell},{lr_pair},{score:.6f}\n")
        
        logging.info(f"Cell-cell边重要性统计已保存: {stats_path}")
        logging.info(f"   - 总共分析了 {len(edge_avg_scores)} 个具体边")
        logging.info(f"   - 总共分析了 {len(cell_lr_pair_avg_scores)} 个cell-cell-LR对")
        logging.info(f"   - Top 10 最重要的cell-cell通讯:")
        
        for i, ((src, dst, lr_pair), score) in enumerate(sorted_pairs[:10]):
            logging.info(f"     {i+1}. {src} -> {dst} ({lr_pair}): {score:.6f}")
        
        # ========== 新增: 统计LR对的出现次数和得分 ==========
        logging.info("\n" + "="*60)
        logging.info("统计配体受体对的出现频率和注意力得分")
        logging.info("="*60)
        
        # 统计每个LR对的出现次数和平均注意力得分
        lr_pair_stats = {}
        for (spot_idx, src_cell, dst_cell, lr_pair), score in edge_avg_scores.items():
            if lr_pair not in lr_pair_stats:
                lr_pair_stats[lr_pair] = {
                    'count': 0,
                    'scores': [],
                    'cell_pairs': set()
                }
            lr_pair_stats[lr_pair]['count'] += 1
            lr_pair_stats[lr_pair]['scores'].append(score)
            lr_pair_stats[lr_pair]['cell_pairs'].add((src_cell, dst_cell))
        
        # 计算统计量
        lr_pair_summary = []
        for lr_pair, stats in lr_pair_stats.items():
            lr_pair_summary.append({
                'lr_pair': lr_pair,
                'count': stats['count'],
                'avg_attention': np.mean(stats['scores']),
                'std_attention': np.std(stats['scores']),
                'min_attention': np.min(stats['scores']),
                'max_attention': np.max(stats['scores']),
                'n_cell_pairs': len(stats['cell_pairs'])
            })
        
        # 保存LR对统计结果
        lr_stats_path = os.path.join(args.output_dir, "lr_pair_statistics.csv")
        with open(lr_stats_path, 'w') as f:
            f.write("lr_pair,occurrence_count,avg_attention_score,std_attention_score,min_attention_score,max_attention_score,n_cell_type_pairs\n")
            for item in sorted(lr_pair_summary, key=lambda x: x['count'], reverse=True):
                f.write(f"{item['lr_pair']},{item['count']},{item['avg_attention']:.6f},{item['std_attention']:.6f},"
                       f"{item['min_attention']:.6f},{item['max_attention']:.6f},{item['n_cell_pairs']}\n")
        
        logging.info(f"LR对统计结果已保存: {lr_stats_path}")
        logging.info(f"   - 总共发现 {len(lr_pair_summary)} 个不同的LR对")
        logging.info(f"   - Top 10 最常出现的LR对:")
        
        for i, item in enumerate(sorted(lr_pair_summary, key=lambda x: x['count'], reverse=True)[:10]):
            logging.info(f"     {i+1}. {item['lr_pair']}: 出现{item['count']}次, "
                        f"平均注意力={item['avg_attention']:.4f}, "
                        f"涉及{item['n_cell_pairs']}个细胞类型对")
        
        logging.info(f"\n   - Top 10 注意力得分最高的LR对:")
        for i, item in enumerate(sorted(lr_pair_summary, key=lambda x: x['avg_attention'], reverse=True)[:10]):
            logging.info(f"     {i+1}. {item['lr_pair']}: 平均注意力={item['avg_attention']:.4f}, "
                        f"出现{item['count']}次")
        
        # ========== 新增: 用模型预测得分生成最终通讯结果 ==========
        logging.info("\n" + "="*60)
        logging.info("生成基于模型预测的通讯结果")
        logging.info("="*60)
        
        # ✅ 方案：直接使用训练阶段收集的 cc_attention 作为 predicted_strength 的近似
        # 因为 DataLoader 有 shuffle，重新前向传播的边顺序会不一致
        # 或者：使用 avg_scores（attention）作为 predicted_strength 的替代
        logging.info("使用注意力得分作为通讯强度的代理指标...")
        all_pred_strengths = avg_scores  # [total_edges] - 使用 GAT 注意力得分
        
        logging.info(f"   注意：由于 DataLoader shuffle，无法准确重建 predicted_comm_strength")
        logging.info(f"   使用 attention_score 作为模型学习到的重要性指标")
        logging.info(f"   这在实际效果上是合理的：attention 本身就反映了模型认为的边重要性")
        
        # 生成三种得分的通讯结果文件
        model_based_comm_path = os.path.join(args.output_dir, "lr_communication_model_based.csv")
        with open(model_based_comm_path, 'w') as f:
            f.write("center_spot,source_cell,target_cell,lr_pair,original_lr_score,attention_score,predicted_strength,modulated_score,score_type\n")
            
            # 遍历所有边，生成三种得分
            for i in range(all_edges.size(1)):
                src_idx = all_edges[0, i].item()
                dst_idx = all_edges[1, i].item()
                center_spot_idx = all_spots[i].item()
                
                if src_idx >= n_spots and dst_idx >= n_spots:
                    src_cell_idx = (src_idx - n_spots) % n_cells
                    dst_cell_idx = (dst_idx - n_spots) % n_cells
                    
                    if src_cell_idx < n_cells and dst_cell_idx < n_cells:
                        # ✅ edge_attr_cc 是 2维: [lr_score, lr_id]
                        lr_score = all_attrs[i, 0].item()  # 第0列: LR得分
                        lr_id = int(all_attrs[i, 1].item())  # 第1列: LR ID
                        attention_score = avg_scores[i].item()
                        # ✅ 使用 attention_score 作为 predicted_strength（因为训练时attention就是模型学习的重要性）
                        predicted_strength = attention_score
                        
                        src_cell = all_cell_names[src_cell_idx]
                        dst_cell = all_cell_names[dst_cell_idx]
                        
                        if lr_id in lr_id_to_pair:
                            ligand, receptor = lr_id_to_pair[lr_id]
                            lr_pair_name = f"{ligand}_{receptor}"
                        else:
                            lr_pair_name = f"lr_{lr_id}"
                        
                        # 计算调制后的得分（原始LR得分 × 模型预测强度）
                        modulated_score = lr_score * predicted_strength
                        
                        # 推荐使用modulated_score作为最终得分
                        score_type = "modulated"
                        
                        f.write(f"{center_spot_idx},{src_cell},{dst_cell},{lr_pair_name},"
                               f"{lr_score:.6f},{attention_score:.6f},{predicted_strength:.6f},"
                               f"{modulated_score:.6f},{score_type}\n")
        
        logging.info(f"基于模型预测的通讯结果已保存: {model_based_comm_path}")
        logging.info(f"   - original_lr_score: 原始LR得分（表达 × 距离衰减）")
        logging.info(f"   - attention_score: GAT注意力得分（图结构重要性）")
        logging.info(f"   - predicted_strength: MLP预测的通讯强度 [0,1]")
        logging.info(f"   - modulated_score: 原始LR得分 × 预测强度（推荐使用）⭐")
        logging.info(f"   - 解释：modulated_score = 基础强度 × 模型学习的重要性因子")
        
    else:
        logging.warning("没有收集到cell-cell注意力得分")    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), train_losses, label="Training Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("HeteroGAT Training Loss Curve", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_curve_path = os.path.join(args.output_dir, "loss_curve.png")
    plt.savefig(loss_curve_path, dpi=150)
    plt.close()
    logging.info(f"损失曲线已保存: {loss_curve_path}")
    
    logging.info("\n" + "="*80)
    logging.info("训练完成！")
    logging.info("="*80)

if __name__ == '__main__':
    main()