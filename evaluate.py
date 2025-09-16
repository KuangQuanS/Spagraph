import argparse
import os
import torch
import pandas as pd
import numpy as np
from data_utils import ST_COMMDataset, comm_collate_fn
from model import ST_COMM
from tokenizer import GeneTokenizer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import networkx as nx
import glob
# 添加聚类相关导入
import scanpy as sc
import anndata as ad
from sklearn.neighbors import kneighbors_graph
import seaborn as sns
from PIL import Image
from matplotlib.colors import ListedColormap

def load_lr_id_mapping(csv_path):
    """加载lr_id到ligand_receptor名称的映射"""
    try:
        df = pd.read_csv(csv_path)
        # 创建从lr_id到ligand_receptor的映射字典
        id_to_name = dict(zip(df['lr_id'], df['ligand_receptor']))
        print(f"✅ 加载了 {len(id_to_name)} 个配体受体对映射")
        return id_to_name
    except Exception as e:
        print(f"⚠️ 无法加载配体受体映射文件 {csv_path}: {e}")
        return {}

def parse_args():
    parser = argparse.ArgumentParser(description='ST_COMM Model Evaluation')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录路径')
    parser.add_argument('--vocab_file', type=str, required=True, help='词汇表文件路径')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='用于评估的模型检查点路径')
    parser.add_argument('--bert_pretrained_model', type=str, required=True, help='BERT预训练模型路径')
    parser.add_argument('--vit_pretrained_model', type=str, help='ViT预训练模型路径')
    parser.add_argument('--lr_mapping_file', type=str, default='./ligand_receptor_labeled.csv', help='配体受体映射文件路径')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='评估结果输出目录')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--max_length', type=int, default=2048, help='最大序列长度')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--top_k_edges', type=int, default=20, help='显示Top-K注意力边数量')
    parser.add_argument('--save_detailed', action='store_true', help='保存详细的评估结果')
    parser.add_argument('--max_graph_edges', type=int, default=50, help='网络图中显示的最大边数量')
    parser.add_argument('--min_node_labels', type=int, default=30, help='显示节点标签的最大节点数量')
    parser.add_argument('--attention_threshold', type=float, default=0.0, help='注意力权重阈值，仅保存大于此值的边')
    parser.add_argument('--filter_no_lr', action='store_true', help='是否过滤掉没有配体受体对的边')
    
    # 添加聚类相关参数
    parser.add_argument('--enable_clustering', action='store_true', help='启用基于节点embedding的聚类分析')
    parser.add_argument('--clustering_resolution', type=float, default=0.5, help='Leiden聚类分辨率参数')
    parser.add_argument('--clustering_n_neighbors', type=int, default=15, help='聚类时构建邻接图的邻居数量')
    parser.add_argument('--clustering_point_size', type=int, default=100, help='聚类可视化中点的大小')
    parser.add_argument('--original_image_path', type=str, help='原始H&E图像路径（用于聚类可视化）')
    
    return parser.parse_args()

def safe_extract(data, batch_idx, default_shape=None):
    """Extract the returned data."""
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        if data.dim() >= 2 and data.shape[0] > batch_idx:
            return data[batch_idx].cpu().numpy()
        elif data.dim() == 1 or data.shape[0] <= batch_idx:
            return data.cpu().numpy()
    elif isinstance(data, (list, tuple)):
        if len(data) > batch_idx and data[batch_idx] is not None:
            item = data[batch_idx]
            if isinstance(item, torch.Tensor):
                return item.cpu().numpy()
            else:
                return np.array(item)
        elif len(data) > 0:
            item = data[0]
            if isinstance(item, torch.Tensor):
                return item.cpu().numpy()
            else:
                return np.array(item)
    elif isinstance(data, np.ndarray):
        if len(data) > batch_idx:
            return data[batch_idx]
        else:
            return data[0] if len(data) > 0 else None

    return np.arange(default_shape) if default_shape else None

def evaluate_model(model, dataloader, device, save_dir=None, detailed=False, lr_id_mapping=None, attention_threshold=0.0, filter_no_lr=False):
    """评估模型"""
    model.eval()
    all_results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # 获取 batch 数据
            input_ids = batch['input_ids'].to(device)
            images = batch['patches'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            coords = batch['coords'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_attr = batch['edge_attr'].to(device)
            spot_ids = batch.get('spot_ids', None)
            knn_mask = batch.get('knn_mask', None)
            
            # 模型前向传播
            node_emb, attn_scores, attn_logistic, text_emb, image_emb, fusion_emb = model(
                input_ids, attention_mask, images, edge_index, edge_attr, coords
            )
            
            B, N = coords.shape[:2]
            
            # 处理每个样本
            for b in range(B):
                # 处理attention_logistic特殊情况
                attn_logistic_b = None
                if attn_logistic is not None:
                    last_layer = attn_logistic[-1]
                    if last_layer.dim() >= 2 and last_layer.shape[0] > b:
                        attn_logistic_b = last_layer[b].cpu().numpy()
                    else:
                        attn_logistic_b = last_layer.cpu().numpy()

                batch_result = {
                    'batch_idx': batch_idx,
                    'sample_idx': b,
                    'spot_ids': safe_extract(spot_ids, b, N),
                    'coords': safe_extract(coords, b, N),
                    'edge_index': safe_extract(edge_index, b, None),
                    'edge_attr': safe_extract(edge_attr, b, None),
                    'attention_logistic': attn_logistic_b,
                    'knn_mask': safe_extract(knn_mask, b, None)
                }
                
                # 如果需要详细信息，保存更多数据
                if detailed:
                    batch_result.update({
                        'node_embeddings': safe_extract(node_emb, b, None),
                        'attention_scores': safe_extract(attn_scores, b, None) if attn_scores is not None else None,
                        'text_embeddings': safe_extract(text_emb, b, None),
                        'image_embeddings': safe_extract(image_emb, b, None),
                        'fusion_embeddings': safe_extract(fusion_emb, b, None),
                    })
                
                # 分析边的注意力权重
                edge_analysis = analyze_edge_attention(
                    safe_extract(edge_index, b, None),
                    safe_extract(edge_attr, b, None),
                    attn_logistic_b,
                    safe_extract(spot_ids, b, N),
                    safe_extract(coords, b, N),  # 添加坐标参数
                    lr_id_mapping
                )

                batch_result['edge_attention_analysis'] = edge_analysis
                
                all_results.append(batch_result)
    
    # 保存结果
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_edge_analysis_csv(all_results, save_dir, attention_threshold, filter_no_lr)
        
        if detailed:
            save_path = os.path.join(save_dir, 'detailed_evaluation_results.npz')
            np.savez_compressed(save_path, results=all_results)
            print(f"✅ Detailed results saved to {save_path}")
        
        create_attention_statistics_plot(all_results, save_dir)
        print(f"✅ Evaluation results saved to {save_dir}")
    
    return all_results

def analyze_edge_attention(edge_index, edge_attr, attn_logistic, spot_ids, coords=None, lr_id_mapping=None):
    """分析每条边的注意力权重（包含坐标信息）"""
    if attn_logistic is None or edge_index is None:
        return None
        
    edge_analysis = []
    
    if edge_index.ndim == 2 and edge_index.shape[0] >= 2:
        num_edges = edge_index.shape[1]
        for i in range(num_edges):
            src_node = edge_index[0, i]
            tgt_node = edge_index[1, i]
            
            # 安全获取注意力权重
            if attn_logistic.ndim == 2:
                attention_weight = float(attn_logistic[src_node, tgt_node]) if (
                    src_node < attn_logistic.shape[0] and tgt_node < attn_logistic.shape[1]
                ) else 0.0
            else:
                attention_weight = float(attn_logistic[i]) if i < len(attn_logistic) else 0.0
            
            # 提取edge_attr信息 (edge_attr格式: [score, lr_id])
            edge_attr_info = edge_attr[i].tolist() if edge_attr is not None and i < len(edge_attr) else None
            score = edge_attr_info[0] if edge_attr_info and len(edge_attr_info) >= 1 else None
            lr_id = int(edge_attr_info[1]) if edge_attr_info and len(edge_attr_info) >= 2 else None
            
            # 映射lr_id到配体受体名称
            lr_name = None
            if lr_id is not None and lr_id != -1 and lr_id_mapping:
                lr_name = lr_id_mapping.get(lr_id, f"unknown_lr_{lr_id}")
            elif lr_id == -1:
                lr_name = "no_ligand_receptor"
            
            # 获取源和目标节点的坐标
            src_x, src_y, tgt_x, tgt_y = None, None, None, None
            if coords is not None:
                if src_node < len(coords) and coords[src_node] is not None:
                    if hasattr(coords[src_node], '__len__') and len(coords[src_node]) >= 2:
                        src_x, src_y = float(coords[src_node][0]), float(coords[src_node][1])
                if tgt_node < len(coords) and coords[tgt_node] is not None:
                    if hasattr(coords[tgt_node], '__len__') and len(coords[tgt_node]) >= 2:
                        tgt_x, tgt_y = float(coords[tgt_node][0]), float(coords[tgt_node][1])
            
            edge_info = {
                'edge_idx': i,
                'src_node': int(src_node),
                'tgt_node': int(tgt_node),
                'src_spot_id': spot_ids[src_node] if spot_ids is not None and src_node < len(spot_ids) else f"node_{src_node}",
                'tgt_spot_id': spot_ids[tgt_node] if spot_ids is not None and tgt_node < len(spot_ids) else f"node_{tgt_node}",
                'src_x': src_x,
                'src_y': src_y,
                'tgt_x': tgt_x,
                'tgt_y': tgt_y,
                'edge_attr': edge_attr_info,
                'score': score,
                'lr_id': lr_id,
                'lr_name': lr_name,
                'attention_weight': attention_weight
            }
            edge_analysis.append(edge_info)
    
    edge_analysis.sort(key=lambda x: x['attention_weight'], reverse=True)
    return edge_analysis

def save_edge_analysis_csv(results, save_dir, attention_threshold=0.0, filter_no_lr=False):
    """将边的注意力分析结果保存为CSV文件（包含坐标信息）"""
    all_edges_data = []
    filtered_edges_data = []
    
    for result in results:
        if result['edge_attention_analysis'] is not None:
            sample_info = f"batch_{result['batch_idx']}_sample_{result['sample_idx']}"
            
            for edge_info in result['edge_attention_analysis']:
                edge_data = {
                    'sample': sample_info,
                    'edge_idx': edge_info['edge_idx'],
                    'src_node': edge_info['src_node'],
                    'tgt_node': edge_info['tgt_node'],
                    'source_spot': edge_info['src_spot_id'],  # 改名以便可视化使用
                    'target_spot': edge_info['tgt_spot_id'],
                    'src_x': edge_info.get('src_x', None),
                    'src_y': edge_info.get('src_y', None),
                    'tgt_x': edge_info.get('tgt_x', None),
                    'tgt_y': edge_info.get('tgt_y', None),
                    'attention_weight': edge_info['attention_weight'],
                    'score': edge_info.get('score', None),
                    'lr_id': edge_info.get('lr_id', None),
                    'lr_name': edge_info.get('lr_name', None),
                    'edge_attr_mean': np.mean(edge_info['edge_attr']) if edge_info['edge_attr'] is not None else None,
                }
                all_edges_data.append(edge_data)
                
                # 应用过滤条件
                is_valid_attention = edge_info['attention_weight'] > attention_threshold
                is_valid_lr = not filter_no_lr or (edge_info.get('lr_name') != 'no_ligand_receptor' and edge_info.get('lr_name') is not None)
                
                if is_valid_attention and is_valid_lr:
                    filtered_edges_data.append(edge_data)
    
    # 保存完整版本
    df_full = pd.DataFrame(all_edges_data)
    csv_path_full = os.path.join(save_dir, 'edge_attention_analysis_full.csv')
    df_full.to_csv(csv_path_full, index=False)
    print(f"✅ 完整边分析CSV保存到 {csv_path_full} (共{len(all_edges_data)}条边)")
    
    # 保存过滤版本
    if len(filtered_edges_data) > 0:
        df_filtered = pd.DataFrame(filtered_edges_data)
        filter_info = f"attention>{attention_threshold}"
        if filter_no_lr:
            filter_info += "_with_LR"
        csv_path_filtered = os.path.join(save_dir, f'edge_attention_analysis_filtered_{filter_info}.csv')
        df_filtered.to_csv(csv_path_filtered, index=False)
        print(f"✅ 过滤后边分析CSV保存到 {csv_path_filtered} (共{len(filtered_edges_data)}条边)")
        print(f"📊 过滤比例: {len(filtered_edges_data)}/{len(all_edges_data)} ({len(filtered_edges_data)/len(all_edges_data)*100:.1f}%)")
    else:
        print("⚠️ 过滤后没有边满足条件")
    
    return df_full, df_filtered if len(filtered_edges_data) > 0 else None

def create_attention_statistics_plot(results, save_dir):
    """创建注意力权重的统计图表"""
    all_attention_weights = []
    
    for result in results:
        if result['edge_attention_analysis'] is not None:
            for edge_info in result['edge_attention_analysis']:
                all_attention_weights.append(edge_info['attention_weight'])
    
    if len(all_attention_weights) == 0:
        print("⚠️ No attention weights found for statistics")
        return
    
    plt.figure(figsize=(12, 8))
    
    # 子图1: 直方图
    plt.subplot(2, 2, 1)
    plt.hist(all_attention_weights, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    plt.xlabel('Attention Weight')
    plt.ylabel('Frequency')
    plt.title('Distribution of Edge Attention Weights')
    plt.grid(True, alpha=0.3)
    
    # 子图2: Top-K 注意力权重
    plt.subplot(2, 2, 2)
    top_k = min(20, len(all_attention_weights))
    top_weights = sorted(all_attention_weights, reverse=True)[:top_k]
    plt.bar(range(top_k), top_weights, color='lightcoral')
    plt.xlabel('Edge Rank')
    plt.ylabel('Attention Weight')
    plt.title(f'Top-{top_k} Attention Weights')
    plt.grid(True, alpha=0.3)
    
    # 子图3: 箱线图
    plt.subplot(2, 2, 3)
    plt.boxplot(all_attention_weights, vert=True)
    plt.ylabel('Attention Weight')
    plt.title('Box Plot of Attention Weights')
    plt.grid(True, alpha=0.3)
    
    # 子图4: 统计信息
    plt.subplot(2, 2, 4)
    stats = f"""Statistics:
Total Edges: {len(all_attention_weights):,}
Mean: {np.mean(all_attention_weights):.6f}
Std: {np.std(all_attention_weights):.6f}
Min: {np.min(all_attention_weights):.6f}
Max: {np.max(all_attention_weights):.6f}
Median: {np.median(all_attention_weights):.6f}
95th: {np.percentile(all_attention_weights, 95):.6f}
99th: {np.percentile(all_attention_weights, 99):.6f}"""
    
    plt.text(0.05, 0.95, stats, fontsize=10, verticalalignment='top', 
             transform=plt.gca().transAxes, fontfamily='monospace')
    plt.axis('off')
    plt.title('Attention Weight Statistics')
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'attention_statistics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Attention statistics plot saved to {save_path}")

def print_top_attention_edges(results, top_k=10):
    """打印注意力权重最高的边信息"""
    all_edges = []
    
    for result in results:
        if result['edge_attention_analysis'] is not None:
            sample_info = f"B{result['batch_idx']}_S{result['sample_idx']}"
            for edge_info in result['edge_attention_analysis']:
                edge_info_copy = edge_info.copy()
                edge_info_copy['sample'] = sample_info
                all_edges.append(edge_info_copy)
    
    all_edges.sort(key=lambda x: x['attention_weight'], reverse=True)
    
    print(f"\n🔍 Top-{top_k} Edges with Highest Attention Weights:")
    print("=" * 120)
    print(f"{'Rank':<6} {'Sample':<12} {'Src Spot':<15} {'Tgt Spot':<15} {'Attention':<15} {'Edge Attr':<25}")
    print("=" * 120)
    
    for i, edge in enumerate(all_edges[:top_k]):
        src_spot = str(edge['src_spot_id'])[:18]
        tgt_spot = str(edge['tgt_spot_id'])[:18]
        attention = edge['attention_weight']
        edge_attr = edge['edge_attr']
        sample = edge['sample']
        
        if isinstance(edge_attr, (list, np.ndarray)):
            edge_attr_str = f"mean:{np.mean(edge_attr):.3f}" if len(edge_attr) > 0 else "[]"
        else:
            edge_attr_str = f"{edge_attr:.3f}" if edge_attr is not None else "None"
        
        print(f"{i+1:<6} {sample:<12} {src_spot:<20} {tgt_spot:<20} {attention:<15.8f} {edge_attr_str:<25}")
    
    print("=" * 120)

# ==================== 聚类分析模块 ====================

def extract_node_embeddings_for_clustering(model_path, data_path, device, bert_model_path, vocab_file, max_length=2048):
    """从训练好的模型中提取节点embedding用于聚类"""
    print("=== 提取节点embedding用于聚类 ===")
    
    # 加载模型
    model = ST_COMM(
        bert_model=bert_model_path,
        vit_depth=6, 
        vit_heads=6, 
        hidden_size=384, 
        vit_mlp_dim=1536
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    # 使用 strict=False 来兼容不同层数的模型
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    
    if missing_keys:
        print(f"⚠️ Missing keys in clustering model: {len(missing_keys)} parameters not loaded")
    if unexpected_keys:
        print(f"⚠️ Unexpected keys in clustering model: {len(unexpected_keys)} parameters ignored")
    
    model.eval()
    print(f"✅ 聚类模型加载完成: {model_path}")
    
    # 加载数据
    tokenizer = GeneTokenizer(vocab_file=vocab_file, max_length=max_length)
    
    # 获取NPZ文件
    npz_files = glob.glob(os.path.join(data_path, "*.npz"))
    npz_files = [file for file in npz_files if not file.endswith("_graph_data.npz")]
    
    if len(npz_files) == 0:
        raise ValueError(f"No .npz files found in {data_path}")
    
    npz_file = npz_files[0]  # 使用第一个样本
    dataset = ST_COMMDataset(
        token_patch_npz_path=npz_file,
        graph_npz_path=npz_file.replace(".npz", "_graph_data.npz"),
        event_csv_path=npz_file.replace(".npz", "_lr.csv")
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # 使用batch_size=1以便处理
        shuffle=False,
        collate_fn=lambda batch: comm_collate_fn(batch, tokenizer=tokenizer)
    )
    
    all_node_embeddings = []
    all_coordinates = []
    all_spot_ids = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # print(f"处理batch {batch_idx + 1}/{len(dataloader)}")
            
            # 获取batch数据
            input_ids = batch['input_ids'].to(device)
            images = batch['patches'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            coords = batch['coords'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_attr = batch['edge_attr'].to(device)
            spot_ids = batch.get('spot_ids', None)
            
            # 模型前向传播
            node_emb, attn_scores, attn_logistic, text_emb, image_emb, fusion_emb = model(
                input_ids, attention_mask, images, edge_index, edge_attr, coords
            )
            
            # 提取每个样本的节点embedding和坐标
            B, N = coords.shape[:2]
            for b in range(B):
                # 提取节点embedding（使用center node，即第一个节点）
                if node_emb is not None:
                    if node_emb.dim() >= 2 and node_emb.shape[0] > b:
                        node_embedding_b = node_emb[b].cpu().numpy()  # [N, hidden_size]
                        # 只使用center node (第一个节点)
                        center_embedding = node_embedding_b[0:1]  # [1, hidden_size]
                        all_node_embeddings.append(center_embedding)
                
                # 提取坐标
                coords_b = coords[b].cpu().numpy()  # [N, 2]
                center_coord = coords_b[0:1]  # [1, 2]
                all_coordinates.append(center_coord)
                
                # 提取spot_ids
                if spot_ids is not None:
                    spot_ids_b = spot_ids[b] if isinstance(spot_ids, list) else spot_ids
                    center_spot_id = spot_ids_b[0] if hasattr(spot_ids_b, '__getitem__') else f"spot_{batch_idx}_{b}"
                    all_spot_ids.append(center_spot_id)
                else:
                    all_spot_ids.append(f"spot_{batch_idx}_{b}")
    
    # 合并所有数据
    if len(all_node_embeddings) > 0:
        node_embeddings = np.vstack(all_node_embeddings)  # [total_spots, hidden_size]
        coordinates = np.vstack(all_coordinates)  # [total_spots, 2]
        
        print(f"✅ 提取完成:")
        print(f"  - 节点embedding形状: {node_embeddings.shape}")
        print(f"  - 坐标形状: {coordinates.shape}")
        print(f"  - spot数量: {len(all_spot_ids)}")
        print(f"  - 坐标范围: X[{coordinates[:, 0].min():.1f}, {coordinates[:, 0].max():.1f}], Y[{coordinates[:, 1].min():.1f}, {coordinates[:, 1].max():.1f}]")
        
        return node_embeddings, coordinates, all_spot_ids
    else:
        raise ValueError("未能提取到节点embedding")

def create_anndata_from_node_embeddings(embeddings, coordinates, spot_ids):
    """从节点embedding创建AnnData对象用于scanpy分析"""
    # 创建AnnData对象
    adata = ad.AnnData(X=embeddings)
    
    # 添加obs信息
    adata.obs['spot_id'] = spot_ids
    adata.obs['x_coord'] = coordinates[:, 0]
    adata.obs['y_coord'] = coordinates[:, 1]
    adata.obs_names = [str(sid) for sid in spot_ids]
    
    # 添加var信息
    adata.var_names = [f'node_feature_{i}' for i in range(embeddings.shape[1])]
    
    print(f"✅ 创建AnnData对象: {adata.n_obs} spots × {adata.n_vars} features")
    
    return adata

def perform_leiden_clustering_on_embeddings(embeddings, coordinates, spot_ids, resolution=0.5, n_neighbors=15, output_dir='clustering_results'):
    """对节点embedding进行Leiden聚类"""
    print(f"=== 执行Leiden聚类 (resolution={resolution}, n_neighbors={n_neighbors}) ===")
    
    # 创建AnnData对象
    adata = create_anndata_from_node_embeddings(embeddings, coordinates, spot_ids)
    
    # 计算邻居图
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
    
    # 进行Leiden聚类
    sc.tl.leiden(adata, resolution=resolution, key_added='leiden')
    
    # 获取聚类结果
    clusters = adata.obs['leiden'].astype(int)
    n_clusters = len(np.unique(clusters))
    
    print(f"✅ 聚类完成，发现 {n_clusters} 个聚类")
    print("聚类分布:")
    cluster_counts = clusters.value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        print(f"  聚类 {cluster_id}: {count} 个spots")
    
    # 保存聚类结果
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存AnnData对象
    adata_path = os.path.join(output_dir, 'fusion_embeddings_clustered.h5ad')
    adata.write(adata_path)
    print(f"✅ AnnData对象已保存: {adata_path}")
    
    # 保存CSV格式结果
    results_df = pd.DataFrame({
        'spot_id': spot_ids,
        'x_coord': coordinates[:, 0],
        'y_coord': coordinates[:, 1],
        'leiden_cluster': clusters
    })
    
    csv_path = os.path.join(output_dir, 'leiden_clustering_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"✅ 聚类结果已保存: {csv_path}")
    
    return adata

def visualize_clusters_on_tissue(adata, point_size=100, alpha=0.8, output_dir='clustering_results', image_path=None):
    """在组织图像上可视化聚类结果"""
    print("=== 生成聚类可视化图像 ===")
    
    coordinates = adata.obs[['x_coord', 'y_coord']].values
    clusters = adata.obs['leiden'].astype(int).values
    
    # 获取聚类数量和颜色
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    
    # 创建图形
    if image_path and os.path.exists(image_path):
        print(f"使用背景图像: {image_path}")
        image = Image.open(image_path)
        image_array = np.array(image)
        
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.imshow(image_array)
        ax.set_xlim(0, image.width)
        ax.set_ylim(image.height, 0)  # 翻转Y轴
    else:
        print("未提供背景图像，使用空白背景")
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_xlim(coordinates[:, 0].min() - 50, coordinates[:, 0].max() + 50)
        ax.set_ylim(coordinates[:, 1].min() - 50, coordinates[:, 1].max() + 50)
    
    # 使用高对比度颜色
    if n_clusters <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    elif n_clusters <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    else:
        colors = plt.cm.hsv(np.linspace(0, 1, n_clusters))
    
    # 绘制每个聚类
    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = clusters == cluster_id
        cluster_coords = coordinates[cluster_mask]
        
        ax.scatter(cluster_coords[:, 0], cluster_coords[:, 1], 
                  c=[colors[i]], s=point_size, alpha=alpha, 
                  label=f'Cluster {cluster_id} ({np.sum(cluster_mask)} spots)',
                  edgecolors='white', linewidth=0.5)
    
    # 设置标题和标签
    ax.set_title(f'Node Embedding Leiden Clustering\n({n_clusters} clusters, {len(coordinates)} spots)', 
                fontsize=16)
    ax.axis('off')
    
    # 添加图例
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    legend.set_frame_on(True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.8)
    
    # 保存图像
    plt.tight_layout()
    cluster_vis_path = os.path.join(output_dir, 'fusion_embedding_leiden_clusters.png')
    plt.savefig(cluster_vis_path, dpi=300, bbox_inches='tight')
    print(f"✅ 聚类可视化已保存: {cluster_vis_path}")
    
    plt.show()
    
    return fig

def plot_cluster_statistics_for_embeddings(adata, output_dir='clustering_results'):
    """绘制聚类统计信息"""
    clusters = adata.obs['leiden'].astype(int).values
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 聚类大小分布
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    ax1.bar(cluster_counts.index, cluster_counts.values, alpha=0.7)
    ax1.set_xlabel('Cluster ID')
    ax1.set_ylabel('Number of Spots')
    ax1.set_title('Cluster Size Distribution')
    ax1.grid(True, alpha=0.3)
    
    # 聚类大小饼图
    ax2.pie(cluster_counts.values, labels=[f'C{i}' for i in cluster_counts.index], 
           autopct='%1.1f%%', startangle=90)
    ax2.set_title('Cluster Proportions')
    
    plt.tight_layout()
    stats_path = os.path.join(output_dir, 'cluster_statistics.png')
    plt.savefig(stats_path, dpi=300, bbox_inches='tight')
    print(f"✅ 聚类统计图已保存: {stats_path}")
    
    plt.show()

def run_clustering_analysis(model_path, data_path, device, bert_model_path, vocab_file, 
                           resolution=0.5, n_neighbors=15, point_size=100, 
                           output_dir='clustering_results', image_path=None):
    """运行完整的聚类分析流程"""
    print("🔬 开始节点embedding聚类分析...")
    
    try:
        # 1. 提取节点embedding
        embeddings, coordinates, spot_ids = extract_node_embeddings_for_clustering(
            model_path, data_path, device, bert_model_path, vocab_file
        )
        
        # 2. 执行聚类
        adata = perform_leiden_clustering_on_embeddings(
            embeddings, coordinates, spot_ids, resolution, n_neighbors, output_dir
        )
        
        # 3. 可视化
        visualize_clusters_on_tissue(
            adata, point_size, output_dir=output_dir, image_path=image_path
        )
        
        # 4. 统计图
        plot_cluster_statistics_for_embeddings(adata, output_dir)
        
        print(f"✅ 聚类分析完成！结果保存在 {output_dir}")
        return adata
        
    except Exception as e:
        print(f"❌ 聚类分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==================== 原有评估模块 ====================

def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    tokenizer = GeneTokenizer(vocab_file=args.vocab_file, max_length=args.max_length)
    
    # 加载配体受体映射
    lr_id_mapping = load_lr_id_mapping(args.lr_mapping_file)
    
    model = ST_COMM(
        bert_model=args.bert_pretrained_model, 
        vit_depth=6, 
        vit_heads=6, 
        hidden_size=384, 
        vit_mlp_dim=1536
    ).to(device)
    
    # 加载ViT权重
    if args.vit_pretrained_model and os.path.exists(args.vit_pretrained_model):
        print(f"✅ Loading ViT weights from {args.vit_pretrained_model}")
        vit_ckpt = torch.load(args.vit_pretrained_model, map_location=device, weights_only=True)
        new_vit_ckpt = {}
        prefix_to_remove = 'vit.'
        for key, value in vit_ckpt.items():
            if key.startswith(prefix_to_remove):
                new_key = key.removeprefix(prefix_to_remove)
                new_vit_ckpt[new_key] = value
        model.vit.load_state_dict(new_vit_ckpt)
        print(f"✅ Loaded pretrained ViT")
    
    # 加载模型检查点
    checkpoint = torch.load(args.model_checkpoint, map_location=device, weights_only=True)
    # 使用 strict=False 来兼容不同层数的模型
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    
    if missing_keys:
        print(f"⚠️ Missing keys: {len(missing_keys)} parameters not loaded")
    if unexpected_keys:
        print(f"⚠️ Unexpected keys: {len(unexpected_keys)} parameters ignored (likely from different layer configuration)")
    
    print(f"✅ Loaded model checkpoint from {args.model_checkpoint}")
    
    # 获取数据文件
    npz_files = glob.glob(os.path.join(args.data_dir, "*.npz"))
    npz_files = [file for file in npz_files if not file.endswith("_graph_data.npz")]
    
    if len(npz_files) == 0:
        raise ValueError(f"No .npz files found in {args.data_dir}")
    
    print(f"Found {len(npz_files)} samples for evaluation")
    
    npz_file = npz_files[0]
    dataset = ST_COMMDataset(
        token_patch_npz_path=npz_file,
        graph_npz_path=npz_file.replace(".npz", "_graph_data.npz"),
        event_csv_path=npz_file.replace(".npz", "_lr.csv")
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: comm_collate_fn(batch, tokenizer=tokenizer)
    )
    
    print("🔬 Starting model evaluation...")
    eval_results = evaluate_model(model, dataloader, device, args.output_dir, args.save_detailed, lr_id_mapping, args.attention_threshold, args.filter_no_lr)
    
    print_top_attention_edges(eval_results, top_k=args.top_k_edges)
    
    # 如果启用聚类分析，运行聚类流程
    if args.enable_clustering:
        print("\n" + "="*60)
        print("🔬 启动聚类分析...")
        print("="*60)
        
        # 使用第一个NPZ文件作为示例
        model_path = args.model_checkpoint
        data_path = args.data_dir
        
        # 确定输出目录
        clustering_output_dir = os.path.join(args.output_dir, 'clustering_results')
        
        # 运行聚类分析
        clustering_result = run_clustering_analysis(
            model_path=model_path,
            data_path=data_path,
            device=device,
            bert_model_path=args.bert_pretrained_model,
            vocab_file=args.vocab_file,
            resolution=args.clustering_resolution,
            n_neighbors=args.clustering_n_neighbors,
            point_size=args.clustering_point_size,
            output_dir=clustering_output_dir,
            image_path=args.original_image_path
        )
        
        if clustering_result is not None:
            print("✅ 聚类分析成功完成！")
        else:
            print("❌ 聚类分析失败")
    
    print("✅ Evaluation completed!")

if __name__ == '__main__':
    main()
