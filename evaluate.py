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
            print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
            
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
        
        visualize_edge_attention(all_results, save_dir, max_edges=50, min_nodes_for_labels=30)
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

def visualize_edge_attention(results, save_dir, max_edges=50, min_nodes_for_labels=30):
    """绘制网络图，显示节点和边的注意力权重"""
    if len(results) == 0:
        print("⚠️ No results found for visualization")
        return
    
    # 取第一个样本进行可视化
    result = results[0]
    if result['edge_attention_analysis'] is None:
        print("⚠️ No edge attention analysis found for visualization")
        return
    
    # 获取数据
    edge_analysis = result['edge_attention_analysis']
    coords = result['coords']
    spot_ids = result['spot_ids']
    
    # 创建网络图
    G = nx.Graph()
    
    # 添加节点
    for i, spot_id in enumerate(spot_ids):
        if coords is not None and i < len(coords):
            pos = (coords[i][0], coords[i][1]) if coords[i].ndim > 0 else (i, 0)
        else:
            pos = (i, 0)
        G.add_node(spot_id, pos=pos)
    
    # 添加边，按注意力权重排序，只取Top-K
    top_k_edges = min(max_edges, len(edge_analysis))  # 限制显示的边数量
    top_edges = edge_analysis[:top_k_edges]
    
    edge_weights = []
    for edge_info in top_edges:
        src_spot = edge_info['src_spot_id']
        tgt_spot = edge_info['tgt_spot_id']
        weight = edge_info['attention_weight']
        
        if src_spot in G.nodes and tgt_spot in G.nodes:
            G.add_edge(src_spot, tgt_spot, weight=weight)
            edge_weights.append(weight)
    
    if len(edge_weights) == 0:
        print("⚠️ No valid edges found for visualization")
        return
    
    # 设置图形大小
    plt.figure(figsize=(16, 12))
    ax = plt.gca()  # 获取当前axes
    
    # 获取节点位置
    pos = nx.get_node_attributes(G, 'pos')
    if not pos:  # 如果没有位置信息，使用spring layout
        pos = nx.spring_layout(G, k=1, iterations=50)
    
    # 归一化边权重用于可视化
    min_weight = min(edge_weights)
    max_weight = max(edge_weights)
    
    # 绘制节点
    node_sizes = [100] * len(G.nodes())  # 所有节点大小相同
    nx.draw_networkx_nodes(G, pos, 
                          node_size=node_sizes,
                          node_color='lightblue',
                          alpha=0.8,
                          edgecolors='black',
                          linewidths=0.5,
                          ax=ax)
    
    # 绘制边，边的粗细和颜色反映注意力权重
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    # 归一化权重用于边的粗细
    normalized_weights = [(w - min_weight) / (max_weight - min_weight) * 5 + 0.5 for w in weights]
    
    # 绘制边
    nx.draw_networkx_edges(G, pos,
                          width=normalized_weights,
                          edge_color=weights,
                          edge_cmap=plt.cm.Reds,
                          alpha=0.7,
                          ax=ax)
    
    # 添加节点标签（只显示部分以避免拥挤）
    if len(G.nodes()) <= min_nodes_for_labels:  # 只有节点数量不多时才显示标签
        labels = {node: str(node)[:8] for node in G.nodes()}  # 截断长标签
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax)
    
    plt.title(f'Attention Network Graph\n(Top-{top_k_edges} edges, Sample: {result["batch_idx"]}_{result["sample_idx"]})', 
              fontsize=14, fontweight='bold')
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, 
                              norm=plt.Normalize(vmin=min_weight, vmax=max_weight))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Attention Weight', rotation=270, labelpad=15)
    
    plt.axis('off')
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(save_dir, 'attention_network_graph.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Network graph saved to {save_path}")
    
    # 创建第二个图：显示所有样本的统计信息
    create_attention_statistics_plot(results, save_dir)

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
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
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
    
    print("✅ Evaluation completed!")

if __name__ == '__main__':
    main()
