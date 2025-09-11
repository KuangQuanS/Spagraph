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
import glob

def parse_args():
    parser = argparse.ArgumentParser(description='ST_COMM Model Evaluation')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录路径')
    parser.add_argument('--vocab_file', type=str, required=True, help='词汇表文件路径')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='用于评估的模型检查点路径')
    parser.add_argument('--bert_pretrained_model', type=str, required=True, help='BERT预训练模型路径')
    parser.add_argument('--vit_pretrained_model', type=str, help='ViT预训练模型路径')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='评估结果输出目录')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--max_length', type=int, default=2048, help='最大序列长度')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--top_k_edges', type=int, default=20, help='显示Top-K注意力边数量')
    parser.add_argument('--save_detailed', action='store_true', help='保存详细的评估结果')
    
    return parser.parse_args()

def safe_extract(data, batch_idx, default_shape=None):
    """安全地提取数据"""
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

def evaluate_model(model, dataloader, device, save_dir=None, detailed=False):
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
                    if isinstance(attn_logistic, (list, tuple)) and len(attn_logistic) > 0:
                        last_layer = attn_logistic[-1]
                        if isinstance(last_layer, torch.Tensor):
                            if last_layer.dim() >= 2 and last_layer.shape[0] > b:
                                attn_logistic_b = last_layer[b].cpu().numpy()
                            else:
                                attn_logistic_b = last_layer.cpu().numpy()
                    elif isinstance(attn_logistic, torch.Tensor):
                        if attn_logistic.dim() >= 2 and attn_logistic.shape[0] > b:
                            attn_logistic_b = attn_logistic[b].cpu().numpy()
                        else:
                            attn_logistic_b = attn_logistic.cpu().numpy()

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
                    safe_extract(spot_ids, b, N)
                )
                batch_result['edge_attention_analysis'] = edge_analysis
                
                all_results.append(batch_result)
    
    # 保存结果
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_edge_analysis_csv(all_results, save_dir)
        
        if detailed:
            save_path = os.path.join(save_dir, 'detailed_evaluation_results.npz')
            np.savez_compressed(save_path, results=all_results)
            print(f"✅ Detailed results saved to {save_path}")
        
        visualize_edge_attention(all_results, save_dir)
        print(f"✅ Evaluation results saved to {save_dir}")
    
    return all_results

def analyze_edge_attention(edge_index, edge_attr, attn_logistic, spot_ids):
    """分析每条边的注意力权重"""
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
            
            edge_info = {
                'edge_idx': i,
                'src_node': int(src_node),
                'tgt_node': int(tgt_node),
                'src_spot_id': spot_ids[src_node] if spot_ids is not None and src_node < len(spot_ids) else f"node_{src_node}",
                'tgt_spot_id': spot_ids[tgt_node] if spot_ids is not None and tgt_node < len(spot_ids) else f"node_{tgt_node}",
                'edge_attr': edge_attr[i].tolist() if edge_attr is not None and i < len(edge_attr) else None,
                'attention_weight': attention_weight
            }
            edge_analysis.append(edge_info)
    
    edge_analysis.sort(key=lambda x: x['attention_weight'], reverse=True)
    return edge_analysis

def save_edge_analysis_csv(results, save_dir):
    """将边的注意力分析结果保存为CSV文件"""
    all_edges_data = []
    
    for result in results:
        if result['edge_attention_analysis'] is not None:
            sample_info = f"batch_{result['batch_idx']}_sample_{result['sample_idx']}"
            
            for edge_info in result['edge_attention_analysis']:
                edge_data = {
                    'sample': sample_info,
                    'edge_idx': edge_info['edge_idx'],
                    'src_node': edge_info['src_node'],
                    'tgt_node': edge_info['tgt_node'],
                    'src_spot_id': edge_info['src_spot_id'],
                    'tgt_spot_id': edge_info['tgt_spot_id'],
                    'attention_weight': edge_info['attention_weight'],
                    'edge_attr_mean': np.mean(edge_info['edge_attr']) if edge_info['edge_attr'] is not None else None,
                }
                all_edges_data.append(edge_data)
    
    df = pd.DataFrame(all_edges_data)
    csv_path = os.path.join(save_dir, 'edge_attention_analysis.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ Edge analysis CSV saved to {csv_path}")
    return df

def visualize_edge_attention(results, save_dir):
    """可视化边的注意力权重分布"""
    all_attention_weights = []
    
    for result in results:
        if result['edge_attention_analysis'] is not None:
            for edge_info in result['edge_attention_analysis']:
                all_attention_weights.append(edge_info['attention_weight'])
    
    if len(all_attention_weights) == 0:
        print("⚠️ No attention weights found for visualization")
        return
    
    plt.figure(figsize=(15, 10))
    
    # 子图1: 直方图
    plt.subplot(2, 3, 1)
    plt.hist(all_attention_weights, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    plt.xlabel('Attention Weight')
    plt.ylabel('Frequency')
    plt.title('Distribution of Edge Attention Weights')
    plt.grid(True, alpha=0.3)
    
    # 子图2: 累积分布
    plt.subplot(2, 3, 2)
    sorted_weights = np.sort(all_attention_weights)
    cumulative = np.arange(1, len(sorted_weights) + 1) / len(sorted_weights)
    plt.plot(sorted_weights, cumulative, color='orange', linewidth=2)
    plt.xlabel('Attention Weight')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution of Attention Weights')
    plt.grid(True, alpha=0.3)
    
    # 子图3: Top-K 注意力权重
    plt.subplot(2, 3, 3)
    top_k = min(20, len(all_attention_weights))
    top_weights = sorted(all_attention_weights, reverse=True)[:top_k]
    plt.bar(range(top_k), top_weights, color='lightcoral')
    plt.xlabel('Edge Rank')
    plt.ylabel('Attention Weight')
    plt.title(f'Top-{top_k} Attention Weights')
    plt.xticks(range(0, top_k, max(1, top_k//10)))
    plt.grid(True, alpha=0.3)
    
    # 子图4: 箱线图
    plt.subplot(2, 3, 4)
    plt.boxplot(all_attention_weights, vert=True)
    plt.ylabel('Attention Weight')
    plt.title('Box Plot of Attention Weights')
    plt.grid(True, alpha=0.3)
    
    # 子图5: 对数直方图
    plt.subplot(2, 3, 5)
    log_weights = np.log10(np.array(all_attention_weights) + 1e-10)
    plt.hist(log_weights, bins=50, alpha=0.7, edgecolor='black', color='lightgreen')
    plt.xlabel('log10(Attention Weight)')
    plt.ylabel('Frequency')
    plt.title('Log-scale Distribution of Attention Weights')
    plt.grid(True, alpha=0.3)
    
    # 子图6: 统计信息
    plt.subplot(2, 3, 6)
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
    
    save_path = os.path.join(save_dir, 'edge_attention_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Edge attention visualization saved to {save_path}")

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
    print(f"{'Rank':<6} {'Sample':<12} {'Src Spot':<20} {'Tgt Spot':<20} {'Attention':<15} {'Edge Attr':<25}")
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
    eval_results = evaluate_model(model, dataloader, device, args.output_dir, args.save_detailed)
    
    print_top_attention_edges(eval_results, top_k=args.top_k_edges)
    
    print("✅ Evaluation completed!")

if __name__ == '__main__':
    main()
