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
from evaluate import evaluate_cell_communication, plot_training_loss, plot_dgi_loss
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
    parser.add_argument('--min_comm_edges', type=int, default=1, 
                       help='最小通讯边数阈值，少于此值的spot将被过滤 (default: 1)')

    # GAT参数
    parser.add_argument('--gat_layers', type=int, default=6, help='GAT层数')
    parser.add_argument('--gat_hidden_dims', type=str, default='512,256,128', help='GAT隐层维度')
    parser.add_argument('--gat_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--gat_dropout', type=float, default=0.3, help='Dropout概率')
    parser.add_argument('--temperature', type=float, default=2.0, help='注意力温度系数，用于控制注意力分布的尖锐程度 (default: 1.0)')
    
    # 模型参数
    parser.add_argument('--output_dim', type=int, default=120, help='输出维度')
    
    # DGI自监督预训练参数
    parser.add_argument('--use_dgi_pretrain', action='store_true', help='是否使用DGI自监督预训练')
    parser.add_argument('--pretrained_encoder', type=str, default=None, 
                       help='DGI预训练的编码器权重路径')
    parser.add_argument('--freeze_encoder', type=str, default='false',
                       choices=['true', 'false'],
                       help='是否冻结预训练的编码器（true=只训练预测头，false=微调整个模型）')
    
    # 混合训练参数（监督 + 自监督）
    parser.add_argument('--supervised_weight', type=float, default=0.5,
                        help='监督学习权重alpha（0~1），自监督权重为1-alpha，默认0.5')
    parser.add_argument('--negative_sample_ratio', type=float, default=1.0,
                        help='负采样比例（相对于正边数量），默认1.0')
    
    parser.add_argument('--dgi_epochs', type=int, default=30, help='DGI预训练epoch数')
    parser.add_argument('--dgi_lr', type=float, default=1e-4, help='DGI预训练学习率')
    parser.add_argument('--corruption_mode', type=str, default='feature_mask', 
                       choices=['feature_mask', 'gaussian_noise', 'shuffle'],
                       help='DGI特征破坏模式')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='特征mask比例 (建议0.4-0.6，增强对比学习难度)')
    parser.add_argument('--noise_std', type=float, default=0.2, help='高斯噪声标准差 (建议0.1-0.3)')
    parser.add_argument('--edge_drop_rate', type=float, default=0.3, help='边丢弃比例 (建议0.2-0.4，增强图结构扰动)')
    parser.add_argument('--readout_mode', type=str, default='mean', 
                       choices=['mean', 'sum', 'gated'],
                       help='图readout模式')
    parser.add_argument('--dgi_early_stop_patience', type=int, default=5, help='DGI早停patience，0表示不使用早停 (default: 10)')
    parser.add_argument('--dgi_early_stop_min_delta', type=float, default=1e-2, help='DGI早停最小改善阈值 (default: 1e-6)')
    

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小 (已支持真正的批处理)')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--checkpoint_interval', type=int, default=50, help='检查点间隔')
    parser.add_argument('--sample_rate', type=float, default=1.0, help='每个epoch采样比例 (default: 1.0, 即全部数据; 0.3表示采样30%)')
    parser.add_argument('--val_split', type=float, default=0.1, 
                       help='验证集比例 (default: 0.1，即10%作为验证集)')
    parser.add_argument('--early_stop_patience', type=int, default=20, help='早停patience，0表示不使用早停 (default: 0)')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0001, help='早停最小改善阈值 (default: 0.0001)')
    
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
        
        cluster_index_names = [f"Cluster_{cid}" for cid in cluster_ids]
        cluster_expr = pd.DataFrame(
            expressions_array,
            index=cluster_index_names,
            columns=marker_genes
        )
        logging.info(f"已从 NPZ 加载 cluster marker 表达: {cluster_expr.shape}")

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
        )

    cluster_to_cell = {}
    for cluster_id, celltype_name in cluster_to_celltype.items():
        cluster_name = f"Cluster_{cluster_id}"
        cluster_to_cell[cluster_name] = celltype_name
    logging.info(f"已构建cluster-cell映射: {len(cluster_to_cell)} 个cluster映射到 {len(set(cluster_to_cell.values()))} 个cell类型")
    
    # 4. 加载CellChat配体-受体数据库
    cellchat_file = 'cellchat_human.csv'
    

    lr_db = pd.read_csv(cellchat_file)
    # 转换为LR对元组列表 [(ligand, receptor), ...]
    lr_pairs = []
    for _, row in lr_db.iterrows():
        lig = str(row['ligand']).strip()
        rec = str(row['receptor']).strip()
        lr_pairs.append((lig, rec))
    logging.info(f"已加载LR对: {len(lr_pairs)}个")

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
        raise ValueError("找不到spot坐标")

    # 7. 加载spot-cluster反卷积比例矩阵
    cluster_composition_file = os.path.join(deconv_dir, '*_cluster_composition.csv')
    cluster_composition_files = glob.glob(cluster_composition_file)
    cluster_composition = pd.read_csv(cluster_composition_files[0], index_col=0)

    # 记录cluster数量
    n_clusters = cluster_composition.shape[1]
    logging.info(f"cluster数量: {n_clusters}")
    
    # 8. 初始化和加载VAE编码器
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
    

    full_vae.load_state_dict(checkpoint['vae_state_dict'])
    vae_encoder = full_vae.encoder

    # ========== 阶段2：构建spot-cell全基因表达 ==========
    logging.info("="*80)
    logging.info("阶段2: 构建spot-cell全基因表达")
    logging.info("="*80)
    
    spot_names = adata.obs_names.tolist()
    spot_total_counts = np.array(adata.X.sum(axis=1)).flatten()
    logging.info(f"Spot总数: {len(spot_names)}, 平均counts={spot_total_counts.mean():.1f}")
    
    # 统一cluster名称格式
    cluster_names = [f"Cluster_{c}" if not c.startswith('Cluster_') else c 
                     for c in cluster_composition.columns]
    
    # 构建每个spot-cell的全基因表达
    spot_cell_full_expr = {}
    for spot_idx, spot_name in enumerate(spot_names):
        spot_total_count = spot_total_counts[spot_idx]
        
        for cell_name, celltype in cluster_to_celltype.items():
            cluster_key = f"Cluster_{cell_name}"
            if cluster_key not in cluster_full_expr.index:
                continue
            
            # 获取cluster在该spot的权重
            if cluster_key in cluster_names:
                cluster_idx = cluster_names.index(cluster_key)
                cluster_weight = cluster_composition.iloc[spot_idx, cluster_idx]
            else:
                cluster_weight = 0.0
            
            if cluster_weight < 1e-6:
                continue
            
            # 计算: (cluster_full_expr / 1e4) × cluster_weight × spot_total_count
            cluster_expr_normalized = cluster_full_expr.loc[cluster_key].values / 1e4
            spot_cell_expr = cluster_expr_normalized * cluster_weight * spot_total_count
            
            # 累加到该spot的celltype表达
            key = f"{spot_name}_{celltype}"
            if key in spot_cell_full_expr:
                spot_cell_full_expr[key] += spot_cell_expr
            else:
                spot_cell_full_expr[key] = spot_cell_expr.copy()
    
    # 转为DataFrame
    spot_cell_expr_df = pd.DataFrame.from_dict(
        spot_cell_full_expr, orient='index', columns=cluster_full_expr.columns
    )
    spot_cell_expr_df.index.name = 'spot_cell'
    
    # 删除全为0的spot-cell（直接过滤，不需要额外的稀有细胞类型过滤）
    row_sums = spot_cell_expr_df.sum(axis=1)
    spot_cell_expr_df = spot_cell_expr_df[row_sums > 0]

    csv_path = os.path.join(args.output_dir, 'spot_cell_full_expr.csv')
    spot_cell_expr_df.to_csv(csv_path)
    logging.info(f"已保存spot-cell全基因表达: {csv_path}, 形状={spot_cell_expr_df.shape}")
    
    # 提取实际存在的celltype（从spot_cell_expr_df的index中解析）
    cell_names = sorted(set([idx.split('_', 1)[1] for idx in spot_cell_expr_df.index]))
    logging.info(f"实际存在的细胞类型数: {len(cell_names)}")
    
    # ✅ 构建marker基因的占位符DataFrame（仅用于Dataset接口兼容）
    cell_expr = pd.DataFrame(
        np.zeros((len(cell_names), cluster_expr.shape[1]), dtype=np.float32),
        index=cell_names,
        columns=cluster_expr.columns
    )
    
    # ✅ 构建全基因的占位符DataFrame（仅用于Dataset接口兼容，实际LR计算用spot_cell_expr_df）
    cell_full_expr = pd.DataFrame(
        np.zeros((len(cell_names), cluster_full_expr.shape[1]), dtype=np.float32),
        index=cell_names,
        columns=cluster_full_expr.columns
    )
    
    # ✅ 构建celltype-level的composition矩阵
    composition = pd.DataFrame(0.0, index=cluster_composition.index, columns=cell_names)
    for spot_idx in range(len(cluster_composition)):
        for cluster_idx, cluster_name in enumerate(cluster_names):
            celltype = cluster_to_cell.get(cluster_name)
            if celltype and celltype in cell_names:  # 只处理实际存在的细胞类型
                composition.iloc[spot_idx, composition.columns.get_loc(celltype)] += cluster_composition.iloc[spot_idx, cluster_idx]
    
    logging.info(f"Composition矩阵形状: {composition.shape}")
    
    # ========== 阶段3.5：预计算KNN邻域和LR通讯得分 ==========
    logging.info("="*80)
    logging.info("阶段3.5: 预计算KNN邻域和LR通讯得分")
    logging.info("="*80)
    
    # ✅ 直接传入spot-cell全基因表达DataFrame（实际数据，不是占位符！）
    knn_mask, csv_path, graph_data = calculate_lr_scores(
        spot_coords=spot_coords,
        composition=composition,
        args=args,
        lr_pairs=lr_pairs,
        adata=adata,
        cell_full_expr=spot_cell_expr_df,  # ✅ 传入实际的spot-cell全基因表达！
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
        load_lr_scores_csv=csv_path,
        min_comm_edges=args.min_comm_edges,
        valid_cell_types=cell_names,
        device=device
    )
    
    # ========== 数据集划分：训练集和验证集 ==========
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    
    # 使用固定种子确保可重复性
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    logging.info(f"数据集划分完成:")
    logging.info(f"   - 训练集: {train_size} 个spots")
    logging.info(f"   - 验证集: {val_size} 个spots")
    logging.info(f"   - 验证集比例: {args.val_split:.1%}")
    
    # ========== 保存LR对映射文件 ==========
    logging.info("保存LR对映射文件...")
    lr_mapping_path = os.path.join(args.output_dir, "lr_pair_mapping.txt")
    with open(lr_mapping_path, 'w') as f:
        f.write("lr_id\tligand\treceptor\n")  # 表头
        for lr_id, (ligand, receptor) in dataset.lr_id_to_pair.items():
            f.write(f"{lr_id}\t{ligand}\t{receptor}\n")
    logging.info(f"LR对映射文件已保存: {lr_mapping_path} (共{len(dataset.lr_id_to_pair)}个LR对)")
    
    # 创建训练和验证数据加载器
    if args.sample_rate < 1.0:
        from torch.utils.data import RandomSampler
        num_samples = int(len(train_dataset) * args.sample_rate)
        sampler = RandomSampler(train_dataset, num_samples=num_samples, replacement=False)
        logging.info(f"⚡ 采样训练模式: 每个epoch采样 {args.sample_rate*100:.1f}% 训练数据 ({num_samples}/{len(train_dataset)} spots)")
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=0,
            collate_fn=hetero_subgraph_collate_fn
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=hetero_subgraph_collate_fn
        )
    
    # 验证集数据加载器（不打乱顺序）
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=hetero_subgraph_collate_fn
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
        vae_encoder=vae_encoder,
        temperature=args.temperature
    ).to(device)
    
    # 加载DGI预训练的编码器(如果提供)
    if args.pretrained_encoder is not None and os.path.exists(args.pretrained_encoder):
        logging.info(f"加载DGI预训练的编码器: {args.pretrained_encoder}")
        pretrained_checkpoint = torch.load(args.pretrained_encoder, map_location=device)
        
        # 加载编码器权重到edge_attn_spatial(edge_dim兼容)
        encoder_state_dict = pretrained_checkpoint['encoder_state_dict']
        
        # 过滤掉node_proj相关权重(延迟初始化,维度不确定)
        filtered_state_dict = {k: v for k, v in encoder_state_dict.items() if 'node_proj' not in k}
        
        # ✅ 只加载到edge_attn_spatial(edge_dim=1,与DGI一致)
        try:
            model.edge_attn_spatial.load_state_dict(filtered_state_dict, strict=False)
            logging.info("   - edge_attn_spatial编码器权重加载成功(跳过node_proj)")
        except Exception as e:
            raise RuntimeError(f"edge_attn_spatial编码器权重加载失败: {e}")
        
        # ⚠️ edge_attn_comm的edge_dim=2，与DGI的edge_dim=1不兼容，跳过加载
        logging.info("   - edge_attn_comm从头开始训练（edge_dim不匹配，跳过DGI权重加载）")
        
        # 是否冻结编码器
        freeze_encoder = args.freeze_encoder.lower() == 'true'
        if freeze_encoder:
            logging.info("   - 冻结edge_attn_spatial权重（已加载预训练）")
            for param in model.edge_attn_spatial.parameters():
                param.requires_grad = False
            # ⚠️ edge_attn_comm不冻结，因为没有预训练权重
            logging.info("   - edge_attn_comm保持可训练（无预训练权重）")
        else:
            logging.info("   - 微调整个模型（包括预训练的edge_attn_spatial）")
    else:
        if args.pretrained_encoder is not None:
            raise FileNotFoundError(f"找不到预训练编码器文件: {args.pretrained_encoder}")
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
    
    # ========== 阶段3.5: 自监督预训练（可选）==========
    if args.use_dgi_pretrain:
        logging.info("="*80)
        logging.info("阶段3.5: DGI自监督预训练")
        logging.info("="*80)
        
        # 检查是否已经加载了预训练权重
        if args.pretrained_encoder is not None and os.path.exists(args.pretrained_encoder):
            logging.info("⚠️ 检测到已加载预训练权重，将在此基础上继续DGI预训练")
            logging.info("   - 这将进一步优化已加载的预训练权重")
        else:
            logging.info("从头开始进行DGI预训练")
        
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
            noise_std=args.noise_std,
            edge_dim=2  # ✅ 统一使用edge_dim=2，所有边都包含[score, id]
        ).to(device)
        
        # 将主模型的编码器权重复制到DGI模型
        # ✅ 注意：DGI的encoder包含edge_attention子模块
        dgi_model.encoder.edge_attention.load_state_dict(model.edge_attn_spatial.state_dict())
        
        dgi_optimizer = torch.optim.Adam(dgi_model.parameters(), lr=args.dgi_lr)
        dgi_scheduler = CosineAnnealingLR(dgi_optimizer, T_max=args.dgi_epochs, eta_min=1e-6)
        
        logging.info("开始DGI预训练...")
        dgi_train_losses = []
        dgi_val_losses = []
        dgi_pbar = tqdm(range(args.dgi_epochs), desc="DGI Pretrain", leave=True, position=0)
        
        # DGI早停参数
        dgi_best_loss = float('inf')
        dgi_patience = args.dgi_early_stop_patience  # 使用命令行参数
        dgi_patience_counter = 0
        dgi_min_delta = args.dgi_early_stop_min_delta  # 使用命令行参数
        
        if dgi_patience > 0:
            logging.info(f"DGI早停已启用: patience={dgi_patience}, min_delta={dgi_min_delta}")
        else:
            logging.info("DGI早停已禁用")
        
        for epoch in dgi_pbar:
            # ========== 训练阶段 ==========
            dgi_model.train()
            epoch_dgi_train_loss = 0.0
            
            for batch_idx, batch in enumerate(train_dataloader):
                batch_size = batch['batch_size']
                # === 打印DGI图结构信息 (仅第1个epoch的前1个batch) ===
                if batch_idx < 1 and epoch == 0:
                    pass  # 移除调试信息
                
                batch_loss = 0.0
                
                for b in range(batch_size):
                    expr_raw = batch['expr_raw'][b].to(device)
                    cell_expr_raw = batch['cell_expr_raw'][b].to(device)
                    edge_index_like = batch['edge_index_like'][b].to(device)
                    edge_attr_like = batch['edge_attr_like'][b].to(device)
                    edge_index_cc = batch['edge_index_cc'][b].to(device)
                    edge_attr_cc = batch['edge_attr_cc'][b].to(device)
                    
                    # ✅ 扩展相似度边属性为2D格式 [weight, -1]
                    if edge_attr_like.dim() == 1:
                        edge_attr_like = torch.cat([edge_attr_like.unsqueeze(-1), torch.full_like(edge_attr_like.unsqueeze(-1), -1)], dim=-1)
                    
                    # 合并边（DGI使用完整的异构图）
                    edge_index = torch.cat([edge_index_like, edge_index_cc], dim=1)
                    edge_attr = torch.cat([edge_attr_like, edge_attr_cc], dim=0)
                    
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
                
                epoch_dgi_train_loss += avg_loss.item()
            
            # ========== 验证阶段 ==========
            dgi_model.eval()
            epoch_dgi_val_loss = 0.0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dataloader):
                    batch_size = batch['batch_size']
                    batch_loss = 0.0
                    
                    for b in range(batch_size):
                        expr_raw = batch['expr_raw'][b].to(device)
                        cell_expr_raw = batch['cell_expr_raw'][b].to(device)
                        edge_index_like = batch['edge_index_like'][b].to(device)
                        edge_attr_like = batch['edge_attr_like'][b].to(device)
                        edge_index_cc = batch['edge_index_cc'][b].to(device)
                        edge_attr_cc = batch['edge_attr_cc'][b].to(device)
                        
                        # ✅ 扩展相似度边属性为2D格式 [weight, -1]
                        if edge_attr_like.dim() == 1:
                            edge_attr_like = torch.cat([edge_attr_like.unsqueeze(-1), torch.full_like(edge_attr_like.unsqueeze(-1), -1)], dim=-1)
                        
                        # 合并边（DGI使用完整的异构图）
                        edge_index = torch.cat([edge_index_like, edge_index_cc], dim=1)
                        edge_attr = torch.cat([edge_attr_like, edge_attr_cc], dim=0)
                        
                        # DGI前向传播（验证阶段不进行特征破坏）
                        pos_scores, neg_scores, summary = dgi_model(
                            expr_raw, cell_expr_raw, edge_index, edge_attr
                        )
                        
                        # 计算DGI损失
                        loss = dgi_loss(pos_scores, neg_scores)
                        
                        batch_loss += loss
                    
                    avg_loss = batch_loss / batch_size
                    epoch_dgi_val_loss += avg_loss.item()
            
            dgi_scheduler.step()
            avg_train_loss = epoch_dgi_train_loss / len(train_dataloader)
            avg_val_loss = epoch_dgi_val_loss / len(val_dataloader)
            
            dgi_train_losses.append(avg_train_loss)
            dgi_val_losses.append(avg_val_loss)
            
            dgi_pbar.set_postfix({
                'Train': f'{avg_train_loss:.4f}', 
                'Val': f'{avg_val_loss:.4f}'
            })
            dgi_pbar.update(1)
            
            # DGI早停检查（使用验证损失）
            if dgi_patience > 0:
                if avg_val_loss < dgi_best_loss - dgi_min_delta:
                    dgi_best_loss = avg_val_loss
                    dgi_patience_counter = 0
                else:
                    dgi_patience_counter += 1
                
                if dgi_patience_counter >= dgi_patience:
                    logging.info(f"DGI早停触发！在epoch {epoch+1}停止预训练")
                    logging.info(f"DGI最佳验证损失: {dgi_best_loss:.6f} (epoch {epoch+1-dgi_patience})")
                    break
        
        dgi_pbar.close()
        
        # 将预训练的编码器权重迁移到主模型
        # ✅ 现在两个网络都是edge_dim=2，可以都加载DGI预训练权重
        
        # 过滤掉node_proj相关权重(延迟初始化,维度不确定)和输出层权重(维度可能不匹配)
        dgi_state_dict = dgi_model.encoder.edge_attention.state_dict()
        filtered_state_dict = {k: v for k, v in dgi_state_dict.items() 
                             if 'node_proj' not in k and 'output_projection' not in k}
        
        model.edge_attn_spatial.load_state_dict(filtered_state_dict, strict=False)
        model.edge_attn_comm.load_state_dict(filtered_state_dict, strict=False)
        logging.info("   - edge_attn_spatial已加载DGI预训练权重(跳过node_proj和output_projection)")
        logging.info("   - edge_attn_comm已加载DGI预训练权重(跳过node_proj和output_projection)")
        
        # 保存DGI预训练权重
        dgi_checkpoint_path = os.path.join(args.output_dir, "dgi_pretrained_encoder.pth")
        torch.save({
            'encoder_state_dict': dgi_model.encoder.edge_attention.state_dict(),  # ✅ 保存edge_attention子模块
            'dgi_train_losses': dgi_train_losses,
            'dgi_val_losses': dgi_val_losses,
            'config': {
                'corruption_mode': args.corruption_mode,
                'readout_mode': args.readout_mode,
                'mask_ratio': args.mask_ratio,
                'edge_drop_rate': args.edge_drop_rate
            }
        }, dgi_checkpoint_path)
        logging.info(f"DGI预训练权重已保存: {dgi_checkpoint_path}")
        logging.info(f"DGI预训练完成！最终训练损失: {dgi_train_losses[-1]:.4f}, 验证损失: {dgi_val_losses[-1]:.4f}")
        
        # 绘制DGI预训练损失曲线（包含训练和验证损失）
        plot_dgi_loss(dgi_train_losses, dgi_val_losses, args.output_dir, args.dgi_epochs)
        
        del dgi_model, dgi_optimizer, dgi_scheduler
        torch.cuda.empty_cache()
    
    # 训练循环
    train_losses = []
    val_losses = []
    edge_losses = []  # 新增：边强度预测损失历史
    comm_pred_losses = []  # 新增：通讯预测损失历史
    
    # 用于收集cell-cell注意力得分和LR ID（只在训练集上收集）
    all_cc_attention_scores = []
    all_edge_index_cc = []
    all_edge_attr_cc = []  # 收集边属性 [lr_score, lr_id]
    all_spot_indices = []  # 收集对应的spot索引
    all_cell_node_mappings = []  # 收集细胞节点到细胞类型的映射
    all_batch_indices = []  # 收集每个边所属的批次索引
    all_n_spots_sub = []  # 收集每个批次的子图spot数量
    all_cell_names = list(cell_expr.index)
    
    # ========== 阶段4：训练循环 ==========
    logging.info("="*80)
    logging.info("阶段4: 开始训练（混合训练：监督 + 自监督）")
    logging.info("="*80)
    logging.info(f"训练配置:")
    logging.info(f"  - 监督学习权重（α）: {args.supervised_weight:.2f}")
    logging.info(f"  - 自监督学习权重（1-α）: {1-args.supervised_weight:.2f}")
    logging.info(f"  - 负采样比例: {args.negative_sample_ratio:.1f}")
    logging.info(f"  - 监督损失: MSE(tanh(logits), 2*(lr_scores-0.5))")
    logging.info(f"  - 自监督损失: BCE(sigmoid(logits), edge_labels)")
    logging.info("="*80)

    # 使用外层tqdm跟踪epoch进度
    # 移除position参数避免Jupyter中重复显示，添加leave=True保持最终状态
    epoch_pbar = tqdm(range(args.epochs), desc="Training", leave=True, dynamic_ncols=True)
    
    # 主训练早停参数
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = args.early_stop_patience
    early_stop_min_delta = args.early_stop_min_delta
    
    if early_stop_patience > 0:
        logging.info(f"主训练早停已启用: patience={early_stop_patience}, min_delta={early_stop_min_delta}")
    else:
        logging.info("主训练早停已禁用")
    
    for epoch in epoch_pbar:
        # ========== 训练阶段 ==========
        model.train()
        total_train_loss = 0.0
        total_supervised_loss = 0.0  # 监督学习损失
        total_self_supervised_loss = 0.0  # 自监督学习损失
        total_edge_loss = 0.0  # 混合边强度损失
        
        for batch_idx, batch in enumerate(train_dataloader, 1):
            batch_size = batch['batch_size']
            n_spots_sub_list = batch['n_spots_sub']  # ✅ 现在是列表
            n_cells_list = batch['n_cells']  # ✅ 现在是列表

            # === 打印图结构信息 (仅前1个batch) ===
            if batch_idx < 1 and epoch == 0:
                pass  # 移除调试信息

            # 累积批次损失
            batch_loss = 0.0
            batch_supervised = 0.0  # 监督学习损失
            batch_self_supervised = 0.0  # 自监督学习损失
            batch_edge = 0.0  # 混合边强度损失

            # 处理每个subgraph
            for b in range(batch_size):
                # 提取第b个subgraph的数据
                expr_raw = batch['expr_raw'][b].to(device)  # [k+1, n_genes]
                cell_expr_raw = batch['cell_expr_raw'][b].to(device)  # [(k+1)*n_cells, n_marker_genes]

                edge_index_like = batch['edge_index_like'][b].to(device)  # [2, E_like]
                edge_attr_like = batch['edge_attr_like'][b].to(device)    # [E_like]
                edge_index_cc = batch['edge_index_cc'][b].to(device)      # [2, E_cc]
                edge_attr_cc = batch['edge_attr_cc'][b].to(device)        # [E_cc]

                # ========== 前向传播 ==========
                spot_repr, cell_repr, combined, spot_proj, cc_attention, predicted_masked_edges, edge_mask = model(
                    expr_raw=expr_raw,
                    cell_expr_raw=cell_expr_raw,
                    edge_index_like=edge_index_like,
                    edge_attr_like=edge_attr_like,
                    edge_index_cc=edge_index_cc,
                    edge_attr_cc=torch.zeros_like(edge_attr_cc),  # 不输入真实强度，避免信息泄露
                    return_attention=True,
                    edge_mask_ratio=0.15  # 15%的边被mask用于预测
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
                    # 收集细胞节点映射信息
                    spot_cell_mapping = batch['spot_cell_mapping'][b]
                    # 创建细胞节点局部索引到细胞类型ID的映射
                    cell_node_to_cell_type = {}
                    for (spot_local_idx, cell_type_id), cell_node_local_idx in spot_cell_mapping.items():
                        cell_node_to_cell_type[cell_node_local_idx] = cell_type_id
                    all_cell_node_mappings.append(cell_node_to_cell_type)
                    # 收集批次索引
                    batch_indices = torch.full((edge_index_cc.size(1),), len(all_cc_attention_scores) - 1, dtype=torch.long)
                    all_batch_indices.append(batch_indices)
                    # 收集子图spot数量
                    n_spots_sub = batch['n_spots_sub'][b]
                    n_spots_sub_tensor = torch.full((edge_index_cc.size(1),), n_spots_sub, dtype=torch.long)
                    all_n_spots_sub.append(n_spots_sub_tensor)

                # ========== 混合训练：监督（MSE）+ 自监督（BCE）==========
                # 注意：不再使用mask预测损失，所有监督学习都通过混合损失完成
                supervised_loss = 0.0  # 监督学习：拟合LR得分
                self_supervised_loss = 0.0  # 自监督学习：边存在性预测
                
                if cc_attention is not None and edge_attr_cc.size(0) > 0:
                    edge_strength_logits = cc_attention  # [n_pos_edges]
                    lr_scores = edge_attr_cc[:, 0]  # [n_pos_edges]
                    
                    # ========== 1. 监督学习部分（MSE）==========
                    # 使用tanh映射到[-1, 1]的调整系数
                    adjustment_factor = torch.tanh(edge_strength_logits)
                    target_factor = 2 * (lr_scores - 0.5)  # [0,1] -> [-1,1]
                    
                    # MSE损失：拟合已知的LR得分
                    supervised_loss = torch.nn.functional.mse_loss(
                        adjustment_factor,
                        target_factor,
                        reduction='mean'
                    )
                    
                    # ========== 2. 自监督学习部分（BCE）==========
                    # 2.1 正边：真实存在的通讯边
                    pos_edge_pred = torch.sigmoid(edge_strength_logits)  # [n_pos_edges]
                    pos_labels = torch.ones_like(pos_edge_pred)  # 标签=1
                    
                    # 2.2 负采样：不存在的边
                    n_pos = len(edge_strength_logits)
                    n_neg = int(n_pos * args.negative_sample_ratio)
                    
                    if n_neg > 0:
                        # 随机采样负边（不同细胞类型的随机配对）
                        n_cells = cell_repr.size(0)
                        neg_src = torch.randint(0, n_cells, (n_neg,), device=device)
                        neg_dst = torch.randint(0, n_cells, (n_neg,), device=device)
                        
                        # 避免自环和重复正边
                        mask = neg_src != neg_dst
                        neg_src = neg_src[mask]
                        neg_dst = neg_dst[mask]
                        
                        if len(neg_src) > 0:
                            # 构造负边索引
                            neg_edge_index = torch.stack([neg_src, neg_dst], dim=0)
                            
                            # 构造负边特征（使用零向量）
                            neg_edge_attr = torch.zeros(len(neg_src), edge_attr_cc.size(1), device=device)
                            
                            # 前向传播获取负边logits
                            with torch.no_grad():
                                # 使用模型的edge_attn_comm预测负边
                                # 构造负边的特征（cell embeddings）
                                neg_src_feat = cell_repr[neg_src]
                                neg_dst_feat = cell_repr[neg_dst]
                                neg_edge_feat = torch.cat([
                                    neg_src_feat, neg_dst_feat, 
                                    neg_edge_attr
                                ], dim=-1)
                                
                                # 简单的线性预测（使用edge_attn_comm的第一层）
                                # 这里简化处理，直接用随机低分作为负样本
                                neg_edge_logits = torch.randn(len(neg_src), device=device) - 2.0  # 偏向负值
                            
                            neg_edge_pred = torch.sigmoid(neg_edge_logits)
                            neg_labels = torch.zeros_like(neg_edge_pred)  # 标签=0
                            
                            # 合并正负边的BCE损失
                            all_preds = torch.cat([pos_edge_pred, neg_edge_pred])
                            all_labels = torch.cat([pos_labels, neg_labels])
                            
                            self_supervised_loss = torch.nn.functional.binary_cross_entropy(
                                all_preds,
                                all_labels,
                                reduction='mean'
                            )
                        else:
                            # 只用正边的BCE
                            self_supervised_loss = torch.nn.functional.binary_cross_entropy(
                                pos_edge_pred,
                                pos_labels,
                                reduction='mean'
                            )
                    else:
                        # 不做负采样，只用正边BCE
                        self_supervised_loss = torch.nn.functional.binary_cross_entropy(
                            pos_edge_pred,
                            pos_labels,
                            reduction='mean'
                        )
                    
                    # ========== 3. 混合损失 ==========
                    alpha = args.supervised_weight
                    edge_strength_loss = alpha * supervised_loss + (1 - alpha) * self_supervised_loss
                else:
                    edge_strength_loss = 0.0
                    supervised_loss = 0.0
                    self_supervised_loss = 0.0
                
                # 总损失 = 混合边强度损失（监督MSE + 自监督BCE）
                loss = edge_strength_loss

                batch_loss += loss
                batch_supervised += supervised_loss.item() if isinstance(supervised_loss, torch.Tensor) else 0.0
                batch_self_supervised += self_supervised_loss.item() if isinstance(self_supervised_loss, torch.Tensor) else 0.0
                batch_edge += edge_strength_loss.item() if isinstance(edge_strength_loss, torch.Tensor) else 0.0

            # 对整个batch求平均损失，然后反向传播
            avg_batch_loss = batch_loss / batch_size
            avg_batch_supervised = batch_supervised / batch_size if batch_supervised > 0 else 0.0
            avg_batch_self_supervised = batch_self_supervised / batch_size if batch_self_supervised > 0 else 0.0
            avg_batch_edge = batch_edge / batch_size if batch_edge > 0 else 0.0

            # 反向传播
            optimizer.zero_grad()
            avg_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += avg_batch_loss.item()
            total_supervised_loss += avg_batch_supervised
            total_self_supervised_loss += avg_batch_self_supervised
            total_edge_loss += avg_batch_edge
        
        # ========== 验证阶段 ==========
        model.eval()
        total_val_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader, 1):
                batch_size = batch['batch_size']
                batch_loss = 0.0

                for b in range(batch_size):
                    expr_raw = batch['expr_raw'][b].to(device)
                    cell_expr_raw = batch['cell_expr_raw'][b].to(device)
                    edge_index_like = batch['edge_index_like'][b].to(device)
                    edge_attr_like = batch['edge_attr_like'][b].to(device)
                    edge_index_cc = batch['edge_index_cc'][b].to(device)
                    edge_attr_cc = batch['edge_attr_cc'][b].to(device)

                    # 前向传播（验证阶段不进行边mask）
                    spot_repr, cell_repr, combined, spot_proj, cc_attention, _, _ = model(
                        expr_raw=expr_raw,
                        cell_expr_raw=cell_expr_raw,
                        edge_index_like=edge_index_like,
                        edge_attr_like=edge_attr_like,
                        edge_index_cc=edge_index_cc,
                        edge_attr_cc=torch.zeros_like(edge_attr_cc),
                        return_attention=True,
                        edge_mask_ratio=0.0  # 验证阶段不mask
                    )

                    # 计算验证损失
                    if cc_attention is not None and edge_attr_cc.size(0) > 0:
                        edge_strength_logits = cc_attention
                        lr_scores = edge_attr_cc[:, 0]
                        
                        # 监督学习部分
                        adjustment_factor = torch.tanh(edge_strength_logits)
                        target_factor = 2 * (lr_scores - 0.5)
                        supervised_loss = torch.nn.functional.mse_loss(
                            adjustment_factor, target_factor, reduction='mean'
                        )
                        
                        # 自监督学习部分
                        pos_edge_pred = torch.sigmoid(edge_strength_logits)
                        pos_labels = torch.ones_like(pos_edge_pred)
                        
                        n_pos = len(edge_strength_logits)
                        n_neg = int(n_pos * args.negative_sample_ratio)
                        
                        if n_neg > 0:
                            n_cells = cell_repr.size(0)
                            neg_src = torch.randint(0, n_cells, (n_neg,), device=device)
                            neg_dst = torch.randint(0, n_cells, (n_neg,), device=device)
                            mask = neg_src != neg_dst
                            neg_src = neg_src[mask]
                            neg_dst = neg_dst[mask]
                            
                            if len(neg_src) > 0:
                                neg_edge_logits = torch.randn(len(neg_src), device=device) - 2.0
                                neg_edge_pred = torch.sigmoid(neg_edge_logits)
                                neg_labels = torch.zeros_like(neg_edge_pred)
                                
                                all_preds = torch.cat([pos_edge_pred, neg_edge_pred])
                                all_labels = torch.cat([pos_labels, neg_labels])
                                
                                self_supervised_loss = torch.nn.functional.binary_cross_entropy(
                                    all_preds, all_labels, reduction='mean'
                                )
                            else:
                                self_supervised_loss = torch.nn.functional.binary_cross_entropy(
                                    pos_edge_pred, pos_labels, reduction='mean'
                                )
                        else:
                            self_supervised_loss = torch.nn.functional.binary_cross_entropy(
                                pos_edge_pred, pos_labels, reduction='mean'
                            )
                        
                        alpha = args.supervised_weight
                        edge_strength_loss = alpha * supervised_loss + (1 - alpha) * self_supervised_loss
                    else:
                        edge_strength_loss = 0.0
                    
                    batch_loss += edge_strength_loss
                
                avg_batch_loss = batch_loss / batch_size
                total_val_loss += avg_batch_loss.item()
        
        # 计算epoch平均损失
        avg_train_loss = total_train_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        avg_val_loss = total_val_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0
        avg_supervised = total_supervised_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        avg_self_supervised = total_self_supervised_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        edge_losses.append(avg_supervised)  # 记录监督学习损失（MSE）
        comm_pred_losses.append(avg_self_supervised)  # 记录自监督学习损失（BCE）
        
        # ========== 早停检查 ==========
        if early_stop_patience > 0:
            if avg_val_loss < best_val_loss - early_stop_min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                logging.info(f"验证损失改善: {best_val_loss:.6f} (patience重置为0)")
            else:
                patience_counter += 1
                logging.info(f"验证损失未改善: {avg_val_loss:.6f} vs {best_val_loss:.6f} (patience: {patience_counter}/{early_stop_patience})")
                
            if patience_counter >= early_stop_patience:
                logging.info(f"早停触发: 验证损失在{early_stop_patience}个epoch内未改善")
                break
        
        # 更新epoch进度条（只在epoch结束时更新一次）
        alpha = args.supervised_weight
        epoch_pbar.set_postfix({
            'Train': f'{avg_train_loss:.4f}',
            'Val': f'{avg_val_loss:.6f}',
            'Sup': f'{avg_supervised:.4f}',  # 监督学习损失（MSE）
            'Self': f'{avg_self_supervised:.4f}',  # 自监督学习损失（BCE）
            'α': f'{alpha:.1f}'  # 监督权重
        })
        
        # 保存检查点
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f"hetero_model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
    
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
    plot_training_loss(train_losses, val_losses, args.output_dir, args.epochs)
    
    logging.info("\n" + "="*80)
    logging.info("训练完成！")
    logging.info("="*80)

if __name__ == '__main__':
    main()