import argparse
import logging
import os
import torch
import glob
import pandas as pd
import numpy as np
import scanpy as sc
from hetero_vae_data_utils import STHeteroSubgraphDataset, hetero_subgraph_collate_fn,setup_logging, set_seed, EarlyStopping
from hetero_model import HeteroSTModel
from evaluate import evaluate_cell_communication, plot_training_loss
from calculate_lr_scores import calculate_lr_scores
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from SC_MAP_ST.deconv_model import DualDecoderVAE as FullVAE
from dgi_pretrain_model import DGIPretrainModel, dgi_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Heterogeneous ST Communication Model Training')
    parser.add_argument('--deconv_dir', type=str, required=True, 
                       help='Stage1+Stage2 输出目录（包含 final_vae.pth, final_vae_cluster_data.npz 等）')
    parser.add_argument('--st_h5ad', type=str, required=True, help='空间转录组h5ad文件路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录路径')
    
    # VAE参数
    parser.add_argument('--vae_latent_dim', type=int, default=64, help='VAE隐空间维度')
    parser.add_argument('--vae_hidden_dim', type=int, default=256, help='VAE隐层维度')
    
    # 图参数
    parser.add_argument('--n_spot_neighbors', type=int, default=10, help='Spot邻近数')
    
    # LR通讯参数
    parser.add_argument('--lr_distance_sigma', type=float, default=50.0, help='LR通讯距离衰减参数sigma')
    parser.add_argument('--mean_expr_threshold', type=float, default=1.0, 
                       help='配体/受体活跃基因的平均表达阈值 (normalize_total 1e4后，default: 1.0)')
    parser.add_argument('--lr_comm_score_threshold', type=float, default=0.0, 
                       help='通讯得分过滤阈值，低于此值的通讯事件将被过滤 (default: 0.0，即不过滤)')
    parser.add_argument('--spot_cell_expr_npz', type=str, default=None, help='spot-cell表达NPZ路径（pre-computed结果，不提供则计算）')

    # GAT参数
    parser.add_argument('--gat_layers', type=int, default=3, help='GAT层数')
    parser.add_argument('--gat_hidden_dims', type=str, default='256,256,128', help='GAT隐层维度')
    parser.add_argument('--gat_heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--gat_dropout', type=float, default=0.1, help='Dropout概率')
    
    # 模型参数
    parser.add_argument('--output_dim', type=int, default=64, help='输出维度')
    
    # DGI自监督预训练参数
    parser.add_argument('--use_dgi_pretrain', action='store_true', help='是否使用DGI自监督预训练')
    parser.add_argument('--dgi_epochs', type=int, default=30, help='DGI预训练epoch数')
    parser.add_argument('--dgi_lr', type=float, default=1e-3, help='DGI预训练学习率')
    parser.add_argument('--corruption_mode', type=str, default='feature_mask', 
                       choices=['feature_mask', 'gaussian_noise', 'shuffle'],
                       help='DGI特征破坏模式')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='特征mask比例 (建议0.4-0.6，增强对比学习难度)')
    parser.add_argument('--noise_std', type=float, default=0.2, help='高斯噪声标准差 (建议0.1-0.3)')
    parser.add_argument('--edge_drop_rate', type=float, default=0.3, help='边丢弃比例 (建议0.2-0.4，增强图结构扰动)')
    parser.add_argument('--readout_mode', type=str, default='mean', 
                       choices=['mean', 'sum', 'gated'],
                       help='图readout模式')
    
    parser.add_argument('--pretrained_encoder', type=str, default=None, 
                       help='DGI预训练的编码器权重路径')
    parser.add_argument('--freeze_encoder', type=str, default='false',
                       choices=['true', 'false'],
                       help='是否冻结预训练的编码器（true=只训练预测头，false=微调整个模型）')
    
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
    parser.add_argument('--early_stop_patience', type=int, default=10, help='早停patience，0表示不使用早停 (default: 0)')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.001, help='早停最小改善阈值 (default: 0.0001)')
    
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
        cluster_data = np.load(vae_npz_path, allow_pickle=True)
        
        cluster_ids = cluster_data['cluster_ids']
        expressions_array = cluster_data['cluster_expressions']  # marker genes
        expressions_full_array = cluster_data['cluster_expressions_full']  # all genes
        
        celltype_mapping_array = cluster_data['cluster_to_celltype']
        cluster_to_celltype = {str(row['cluster_id']): str(row['celltype']) 
                                for row in celltype_mapping_array}
        
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
        spot_cluster_csv_data = []
        spot_cluster_npz_dict = {}
        
        for spot_idx in range(n_spots):
            spot_name = spot_names[spot_idx]
            spot_total_count = spot_total_counts[spot_idx]
            
            for cluster_idx, cluster_name in enumerate(cluster_names):
                cluster_weight = cluster_composition.iloc[spot_idx, cluster_idx]
                if cluster_name not in cluster_expr.index:
                    logging.warning(f"找不到 cluster '{cluster_name}' in cluster_expr，跳过")
                    continue
                
                cluster_expr_values = cluster_expr.loc[cluster_name].values
                
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
        
        # ✅ 跳过保存spot-cluster CSV（中间结果，不需要保存）
        logging.info(f"已构建spot-cluster表达矩阵 (中间结果，不保存)")
        
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
        
        # ========== 过滤稀有细胞类型 ==========
        logging.info("="*60)
        logging.info("过滤稀有细胞类型（出现频率 < 10%）")
        logging.info("="*60)
        
        # 计算每个细胞类型的出现频率（在多少比例的 spots 中存在）
        cell_occurrence = {}
        n_spots = composition.shape[0]
        occurrence_threshold = 0.10  # 10% 阈值
        
        for cell_name in composition.columns:
            # 统计在多少个 spots 中该细胞类型的比例 > 0
            n_spots_with_cell = (composition[cell_name] > 1e-6).sum()
            occurrence_rate = n_spots_with_cell / n_spots
            cell_occurrence[cell_name] = occurrence_rate
        
        # 找出需要保留的细胞类型
        cells_to_keep = [cell for cell, rate in cell_occurrence.items() if rate >= occurrence_threshold]
        cells_to_remove = [cell for cell, rate in cell_occurrence.items() if rate < occurrence_threshold]
        
        logging.info(f"细胞类型过滤统计:")
        logging.info(f"   - 原始细胞类型数: {len(composition.columns)}")
        logging.info(f"   - 保留细胞类型数: {len(cells_to_keep)} (出现率 ≥ {occurrence_threshold*100}%)")
        logging.info(f"   - 移除细胞类型数: {len(cells_to_remove)} (出现率 < {occurrence_threshold*100}%)")
        
        if cells_to_remove:
            logging.info(f"   - 被移除的细胞类型:")
            for cell in cells_to_remove:
                logging.info(f"     * {cell}: 出现率={cell_occurrence[cell]*100:.2f}% ({int(cell_occurrence[cell]*n_spots)}/{n_spots} spots)")
        
        # 过滤 composition 矩阵
        composition = composition[cells_to_keep]
        logging.info(f"过滤后 composition 形状: {composition.shape}")
        
        # 过滤 cell_expr 和 cell_full_expr
        cell_expr = cell_expr.loc[cells_to_keep]
        logging.info(f"过滤后 cell_expr 形状: {cell_expr.shape}")
        
        if cell_full_expr is not None:
            cell_full_expr = cell_full_expr.loc[cells_to_keep]
            logging.info(f"过滤后 cell_full_expr 形状: {cell_full_expr.shape}")
        
        # 更新 cell_names 列表
        cell_names = cells_to_keep
        logging.info(f"保留的细胞类型: {cell_names}")
        
        # ✅ 保存过滤后的cell表达矩阵（最终版本）
        cell_expr_csv_path = os.path.join(args.output_dir, 'cell_marker_expr.csv')
        cell_expr.to_csv(cell_expr_csv_path)
        logging.info(f"已保存过滤后的cell marker表达: {cell_expr_csv_path}")
        
        if cell_full_expr is not None:
            cell_full_expr_csv_path = os.path.join(args.output_dir, 'cell_full_expr.csv')
            cell_full_expr.to_csv(cell_full_expr_csv_path)
            logging.info(f"已保存过滤后的cell全基因表达: {cell_full_expr_csv_path}")
        
    else:
        composition = None
    
    # ========== 阶段3.5：预计算KNN邻域和LR通讯得分 ==========
    logging.info("="*80)
    logging.info("阶段3.5: 预计算KNN邻域和LR通讯得分")
    logging.info("="*80)
    
    # 调用calculate_lr_scores函数
    knn_mask, csv_path, graph_data = calculate_lr_scores(
        spot_coords=spot_coords,
        composition=composition,
        args=args,
        lr_pairs=lr_pairs,
        spot_cell_expr_npz_path=spot_cell_expr_npz_path,
        adata=adata,
        cell_expr=cell_expr,
        output_dir=args.output_dir
    )

    
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
        valid_cell_types=cell_names if composition is not None else None,  # 传递过滤后的细胞类型列表
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
    
    # 加载DGI预训练的编码器（如果提供）
    if args.pretrained_encoder is not None and os.path.exists(args.pretrained_encoder):
        logging.info(f"加载DGI预训练的编码器: {args.pretrained_encoder}")
        pretrained_checkpoint = torch.load(args.pretrained_encoder, map_location=device)
        
        # 加载编码器权重到gat_spatial和gat_comm
        encoder_state_dict = pretrained_checkpoint['encoder_state_dict']
        
        # 尝试加载到gat_spatial
        try:
            model.gat_spatial.load_state_dict(encoder_state_dict)
            logging.info("   - gat_spatial编码器权重加载成功")
        except Exception as e:
            logging.warning(f"   - gat_spatial编码器权重加载失败: {e}")
        
        # 尝试加载到gat_comm
        try:
            model.gat_comm.load_state_dict(encoder_state_dict)
            logging.info("   - gat_comm编码器权重加载成功")
        except Exception as e:
            logging.warning(f"   - gat_comm编码器权重加载失败: {e}")
        
        # 是否冻结编码器
        freeze_encoder = args.freeze_encoder.lower() == 'true'
        if freeze_encoder:
            logging.info("   - 冻结编码器权重（只训练预测头）")
            for param in model.gat_spatial.parameters():
                param.requires_grad = False
            for param in model.gat_comm.parameters():
                param.requires_grad = False
        else:
            logging.info("   - 微调编码器（训练整个模型）")
    else:
        if args.pretrained_encoder is not None:
            logging.warning(f"找不到预训练编码器文件: {args.pretrained_encoder}")
        logging.info("从头开始训练模型（无预训练）")
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    logging.info(f"模型构建完成")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"   - 总参数数: {total_params:,}")
    logging.info(f"   - 可训练参数数: {trainable_params:,}")
    
    # ========== 阶段3.5: DGI自监督预训练（可选）==========
    if args.use_dgi_pretrain:
        logging.info("="*80)
        logging.info("阶段3.5: DGI自监督预训练")
        logging.info("="*80)
        logging.info(f"DGI预训练配置:")
        logging.info(f"   - Epochs: {args.dgi_epochs}")
        logging.info(f"   - Learning rate: {args.dgi_lr}")
        logging.info(f"   - Corruption mode: {args.corruption_mode}")
        logging.info(f"   - Edge drop rate: {args.edge_drop_rate}")
        logging.info(f"   - Readout mode: {args.readout_mode}")
        
        # 构建DGI模型（共享编码器）
        dgi_model = DGIPretrainModel(
            vae_encoder=vae_encoder,
            vae_latent_dim=latent_dim,
            gat_hidden_dims=gat_hidden_dims,
            gat_heads=args.gat_heads,
            gat_dropout=args.gat_dropout,
            readout_mode=args.readout_mode,
            corruption_mode=args.corruption_mode,
            mask_ratio=args.mask_ratio,
            noise_std=args.noise_std
        ).to(device)
        
        # 将主模型的编码器权重复制到DGI模型
        dgi_model.encoder.load_state_dict(model.gat_spatial.state_dict())
        
        dgi_optimizer = torch.optim.Adam(dgi_model.parameters(), lr=args.dgi_lr)
        dgi_scheduler = CosineAnnealingLR(dgi_optimizer, T_max=args.dgi_epochs, eta_min=1e-6)
        
        logging.info("开始DGI预训练...")
        dgi_losses = []
        dgi_pbar = tqdm(range(args.dgi_epochs), desc="DGI Pretrain", leave=False, disable=False)
        
        for epoch in dgi_pbar:
            dgi_model.train()
            epoch_dgi_loss = 0.0
            
            for batch_idx, batch in enumerate(dataloader):
                batch_size = batch['batch_size']
                # === 打印DGI图结构信息 (仅第1个epoch的前1个batch) ===
                if batch_idx < 1 and epoch == 0:
                    batch_size_dgi = batch['batch_size']
                    n_spots_sub_dgi_list = batch['n_spots_sub']  # ✅ 现在是列表
                    n_cells_dgi_list = batch['n_cells']  # ✅ 现在是列表
                    
                    logging.info(f"\n{'='*60}")
                    logging.info(f"[DGI] Batch {batch_idx} Graph Structure:")
                    logging.info(f"{'='*60}")
                    for b in range(batch_size_dgi):
                        n_spots_this = n_spots_sub_dgi_list[b]  # ✅ 从列表获取
                        n_cells_this = n_cells_dgi_list[b]  # ✅ 从列表获取
                        
                        edge_index_like_b = batch['edge_index_like'][b]
                        n_like_edges_total = edge_index_like_b.size(1)
                        
                        ss_mask = (edge_index_like_b[0] < n_spots_this) & (edge_index_like_b[1] < n_spots_this)
                        n_ss_edges = ss_mask.sum().item()
                        
                        sc_mask = ((edge_index_like_b[0] < n_spots_this) & (edge_index_like_b[1] >= n_spots_this)) | \
                                  ((edge_index_like_b[0] >= n_spots_this) & (edge_index_like_b[1] < n_spots_this))
                        n_sc_edges = sc_mask.sum().item()
                        
                        n_cc_edges = batch['edge_index_cc'][b].size(1)
                        
                        logging.info(f"  Subgraph {b}:")
                        logging.info(f"    节点: {n_spots_this} spots + {n_cells_this} cells = {n_spots_this + n_cells_this} total")
                        logging.info(f"    边: SS={n_ss_edges}, SC={n_sc_edges}, CC={n_cc_edges}, total={n_ss_edges + n_sc_edges + n_cc_edges}")
                    logging.info(f"{'='*60}\n")
                
                batch_loss = 0.0
                
                for b in range(batch_size):
                    expr_raw = batch['expr_raw'][b].to(device)
                    cell_expr_raw = batch['cell_expr_raw'][b].to(device)
                    edge_index_like = batch['edge_index_like'][b].to(device)
                    edge_attr_like = batch['edge_attr_like'][b].to(device)
                    edge_index_cc = batch['edge_index_cc'][b].to(device)
                    edge_attr_cc = batch['edge_attr_cc'][b].to(device)
                    
                    # 合并边（DGI使用完整的异构图）
                    edge_index = torch.cat([edge_index_like, edge_index_cc], dim=1)
                    edge_attr = torch.cat([edge_attr_like, edge_attr_cc[:, 0] if edge_attr_cc.size(0) > 0 else edge_attr_like[:0]], dim=0)
                    
                    # DGI前向传播（内部会进行特征破坏和边丢弃）
                    pos_scores, neg_scores, summary = dgi_model(
                        expr_raw, cell_expr_raw, edge_index, edge_attr
                    )
                    
                    # 计算DGI损失
                    loss = dgi_loss(pos_scores, neg_scores)
                    
                    batch_loss += loss
                
                avg_loss = batch_loss / batch_size
                dgi_optimizer.zero_grad()
                avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(dgi_model.parameters(), 1.0)
                dgi_optimizer.step()
                
                epoch_dgi_loss += avg_loss.item()
            
            dgi_scheduler.step()
            avg_epoch_loss = epoch_dgi_loss / len(dataloader)
            dgi_losses.append(avg_epoch_loss)
            dgi_pbar.set_postfix({'Loss': f'{avg_epoch_loss:.4f}'})
            dgi_pbar.update(1)
            logging.info(f"[DGI Epoch {epoch+1}/{args.dgi_epochs}] Loss: {avg_epoch_loss:.4f}")
        
        dgi_pbar.close()
        
        # 将预训练的编码器权重迁移到主模型
        model.gat_spatial.load_state_dict(dgi_model.encoder.state_dict())
        model.gat_comm.load_state_dict(dgi_model.encoder.state_dict())
        
        # 保存DGI预训练权重
        dgi_checkpoint_path = os.path.join(args.output_dir, "dgi_pretrained_encoder.pth")
        torch.save({
            'encoder_state_dict': dgi_model.encoder.state_dict(),
            'dgi_losses': dgi_losses,
            'config': {
                'corruption_mode': args.corruption_mode,
                'readout_mode': args.readout_mode,
                'mask_ratio': args.mask_ratio,
                'edge_drop_rate': args.edge_drop_rate
            }
        }, dgi_checkpoint_path)
        logging.info(f"DGI预训练权重已保存: {dgi_checkpoint_path}")
        logging.info(f"DGI预训练完成！最终损失: {dgi_losses[-1]:.4f}")
        
        del dgi_model, dgi_optimizer, dgi_scheduler
        torch.cuda.empty_cache()
    
    # 训练循环
    train_losses = []
    # ✅ 移除 GraphAugmentor：第二阶段不需要图增强，DGI已完成对比学习
    
    # 用于收集cell-cell注意力得分和LR ID
    all_cc_attention_scores = []
    all_edge_index_cc = []
    all_edge_attr_cc = []  # 收集边属性 [lr_score, lr_id]
    all_spot_indices = []  # 收集对应的spot索引
    all_cell_names = list(cell_expr.index)
    
    # ========== 阶段4：训练循环 ==========
    logging.info("="*80)
    logging.info("阶段4: 开始训练")
    logging.info("="*80)

    # 初始化早停
    early_stopping = None
    if args.early_stop_patience > 0:
        early_stopping = EarlyStopping(
            patience=args.early_stop_patience,
            min_delta=args.early_stop_min_delta,
            verbose=True
        )
        logging.info(f"已启用早停机制: patience={args.early_stop_patience}, min_delta={args.early_stop_min_delta}")
    else:
        logging.info("未启用早停机制")

    # 使用外层tqdm跟踪epoch进度
    # 移除position参数避免Jupyter中重复显示，添加leave=True保持最终状态
    epoch_pbar = tqdm(range(args.epochs), desc="Training", leave=True, dynamic_ncols=True)
    
    for epoch in epoch_pbar:
        model.train()
        total_loss = 0.0
        total_comm_pred_loss = 0.0  # ✅ 边存在性预测损失（BCE，唯一的损失）
        for batch_idx, batch in enumerate(dataloader, 1):
            batch_size = batch['batch_size']
            n_spots_sub_list = batch['n_spots_sub']  # ✅ 现在是列表
            n_cells_list = batch['n_cells']  # ✅ 现在是列表

            # === 打印图结构信息 (仅前1个batch) ===
            if batch_idx < 1 and epoch == 0:
                logging.info(f"\n{'='*60}")
                logging.info(f"Batch {batch_idx} Graph Structure:")
                logging.info(f"{'='*60}")
                for b in range(batch_size):
                    n_spots_this = n_spots_sub_list[b]  # ✅ 从列表获取
                    n_cells_this = n_cells_list[b]  # ✅ 从列表获取
                    
                    # 统计边类型：edge_index_like包含Spot-Spot和Spot-Cell两种边
                    edge_index_like_b = batch['edge_index_like'][b]
                    n_like_edges_total = edge_index_like_b.size(1)
                    
                    # Spot-Spot边：两个端点都 < n_spots_this
                    ss_mask = (edge_index_like_b[0] < n_spots_this) & (edge_index_like_b[1] < n_spots_this)
                    n_ss_edges = ss_mask.sum().item()
                    
                    # Spot-Cell边：一个端点 < n_spots_this，另一个 >= n_spots_this
                    sc_mask = ((edge_index_like_b[0] < n_spots_this) & (edge_index_like_b[1] >= n_spots_this)) | \
                              ((edge_index_like_b[0] >= n_spots_this) & (edge_index_like_b[1] < n_spots_this))
                    n_sc_edges = sc_mask.sum().item()
                    
                    # Cell-Cell边
                    n_cc_edges = batch['edge_index_cc'][b].size(1)
                    
                    logging.info(f"  Subgraph {b}:")
                    logging.info(f"    异构图节点:")
                    logging.info(f"      - Spot 节点数: {n_spots_this}")
                    logging.info(f"      - Cell 节点数: {n_cells_this} (动态分配，仅composition>1e-6)")
                    logging.info(f"      - 总节点数: {n_spots_this + n_cells_this}")
                    logging.info(f"    异构图边:")
                    logging.info(f"      - Spot-Spot 边数: {n_ss_edges} (空间相似性)")
                    logging.info(f"      - Spot-Cell 边数: {n_sc_edges} (composition权重)")
                    logging.info(f"      - Cell-Cell 边数: {n_cc_edges} (LR通讯)")
                    logging.info(f"      - 总边数: {n_ss_edges + n_sc_edges + n_cc_edges}")
                logging.info(f"{'='*60}\n")

            # 累积批次损失
            batch_loss = 0.0
            batch_comm_pred_loss = 0.0  # ✅ 边存在性预测损失（BCE）

            # 处理每个subgraph
            for b in range(batch_size):
                # 提取第b个subgraph的数据
                expr_raw = batch['expr_raw'][b].to(device)  # [k+1, n_genes]
                cell_expr_raw = batch['cell_expr_raw'][b].to(device)  # [(k+1)*n_cells, n_marker_genes]

                edge_index_like = batch['edge_index_like'][b].to(device)  # [2, E_like]
                edge_attr_like = batch['edge_attr_like'][b].to(device)    # [E_like]
                edge_index_cc = batch['edge_index_cc'][b].to(device)      # [2, E_cc]
                edge_attr_cc = batch['edge_attr_cc'][b].to(device)        # [E_cc]

                # ========== 前向传播（只用原始图，不需要增强） ==========
                spot_repr, cell_repr, combined, spot_proj, cc_attention, predicted_comm_strength = model(
                    expr_raw=expr_raw,
                    cell_expr_raw=cell_expr_raw,
                    edge_index_like=edge_index_like,
                    edge_attr_like=edge_attr_like,
                    edge_index_cc=edge_index_cc,
                    edge_attr_cc=edge_attr_cc,
                    return_attention=True
                )
                
                # 收集cell-cell注意力得分（用于后续分析）
                if cc_attention is not None:
                    all_cc_attention_scores.append(cc_attention.detach().cpu())
                    all_edge_index_cc.append(edge_index_cc.detach().cpu())
                    all_edge_attr_cc.append(edge_attr_cc.detach().cpu())
                    # 收集对应的spot信息（center_spot_idx）
                    center_spot_idx = batch['center_spot_idx'][b]
                    spot_indices = torch.full((edge_index_cc.size(1),), center_spot_idx, dtype=torch.long)
                    all_spot_indices.append(spot_indices)

                # ========== 边存在性预测损失（二分类任务） ==========
                comm_pred_loss = 0.0
                if predicted_comm_strength is not None and edge_attr_cc.size(0) > 0:
                    # ✅ 改为二分类任务：基于LR得分阈值生成标签
                    true_lr_scores = edge_attr_cc[:, 0]  # [n_edges] - 第0列是lr_score
                    
                    # 计算全局中位数作为阈值（动态阈值，适应不同batch）
                    threshold = true_lr_scores.median()
                    
                    # 生成二分类标签：高于中位数=1（边存在），低于=0（边不存在）
                    binary_labels = (true_lr_scores > threshold).float()
                    
                    # BCE 损失 - predicted_comm_strength 的输出已经是 [0, 1] (Sigmoid)
                    comm_pred_loss = torch.nn.functional.binary_cross_entropy(
                        predicted_comm_strength, 
                        binary_labels,
                        reduction='mean'
                    )

                # ✅ 总损失 = 边存在性预测损失（DGI已经完成了对比学习）
                loss = comm_pred_loss

                batch_loss += loss
                batch_comm_pred_loss += comm_pred_loss.item() if isinstance(comm_pred_loss, torch.Tensor) else 0.0

            # 对整个batch求平均损失，然后反向传播
            avg_batch_loss = batch_loss / batch_size
            avg_batch_comm_pred_loss = batch_comm_pred_loss / batch_size if batch_comm_pred_loss > 0 else 0.0

            # 反向传播
            optimizer.zero_grad()
            avg_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += avg_batch_loss.item()
            total_comm_pred_loss += avg_batch_comm_pred_loss
        
        # 计算epoch平均损失
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        avg_comm_pred = total_comm_pred_loss / len(dataloader) if len(dataloader) > 0 else 0
        train_losses.append(avg_loss)
        
        # 更新epoch进度条（只在epoch结束时更新一次）
        epoch_pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'BCE': f'{avg_comm_pred:.4f}'  # ✅ BCE = Binary Cross Entropy
        })
        
        # ✅ 在第一个epoch输出任务说明
        if epoch == 0:
            logging.info(f"[训练任务] 边存在性预测（二分类）：预测Cell-Cell边是否为重要通讯边")
            logging.info(f"   - 标签生成：基于LR得分中位数动态阈值")
            logging.info(f"   - 损失函数：Binary Cross Entropy (BCE)")
        
        logging.info(f"[Epoch {epoch+1}/{args.epochs}] Loss: {avg_loss:.4f}, BCE: {avg_comm_pred:.4f}")
        
        # 保存检查点
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f"hetero_model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"检查点已保存: {checkpoint_path}")
        
        # 早停检查
        if early_stopping is not None:
            if early_stopping(epoch, avg_loss):
                logging.info(f"早停触发！在epoch {epoch+1}停止训练")
                logging.info(f"最佳损失: {early_stopping.best_loss:.6f} (epoch {early_stopping.best_epoch+1})")
                # 保存早停时的模型
                early_stop_path = os.path.join(args.output_dir, f"hetero_model_early_stop_epoch{epoch+1}.pth")
                torch.save(model.state_dict(), early_stop_path)
                logging.info(f"早停模型已保存: {early_stop_path}")
                break
    
    # 关闭tqdm进度条
    epoch_pbar.close()
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, "hetero_model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"最终模型已保存: {final_model_path}")
    
    # ========== 阶段5：统计cell-cell边重要性 ==========
    evaluate_cell_communication(
        all_cc_attention_scores=all_cc_attention_scores,
        all_edge_index_cc=all_edge_index_cc,
        all_edge_attr_cc=all_edge_attr_cc,
        all_spot_indices=all_spot_indices,
        all_cell_names=all_cell_names,
        output_dir=args.output_dir,
        n_spots=n_spots,
        n_cells=n_cells
    )
    
    # 绘制损失曲线
    plot_training_loss(train_losses, args.output_dir, args.epochs)
    
    logging.info("\n" + "="*80)
    logging.info("训练完成！")
    logging.info("="*80)

if __name__ == '__main__':
    main()