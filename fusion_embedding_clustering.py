"""
Fusion Embedding Clustering and Visualization
基于fusion embedding进行Leiden聚类并在原图上可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import scanpy as sc
import anndata as ad
from sklearn.neighbors import kneighbors_graph
import seaborn as sns
import os
import argparse
from matplotlib.colors import ListedColormap

def load_fusion_embeddings(npz_path, use_center_only=True):
    """从NPZ文件中加载fusion embeddings和相关信息
    
    Args:
        npz_path: NPZ文件路径
        use_center_only: 如果为True，只使用每个batch的第一个embedding（假设是中心节点）
    """
    print(f"Loading NPZ file: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    results = data['results']
    
    # 收集所有样本的fusion embeddings和坐标
    all_embeddings = []
    all_coords = []
    all_spot_ids = []
    all_sample_indices = []
    
    for i, result in enumerate(results):
        if isinstance(result, dict) and 'fusion_embeddings' in result:
            embeddings = result['fusion_embeddings']
            coords = result['coords']
            spot_ids = result['spot_ids']
            sample_idx = result.get('sample_idx', i)
            
            if use_center_only:
                # 只使用第一个embedding和对应的坐标（假设是中心节点）
                center_embedding = embeddings[0:1]  # 保持2D形状
                center_coord = coords[0:1]
                center_spot_id = spot_ids[0]
                
                all_embeddings.append(center_embedding)
                all_coords.append(center_coord)
                all_spot_ids.append(center_spot_id)
                all_sample_indices.append(sample_idx)
                
                if i < 3:  # 打印前几个batch的信息用于调试
                    print(f"Batch {i}: center spot {center_spot_id}, coord {center_coord[0]}")
            else:
                # 使用所有embeddings（包括邻居）
                all_embeddings.append(embeddings)
                all_coords.append(coords)
                all_spot_ids.extend(spot_ids)
                all_sample_indices.extend([sample_idx] * len(spot_ids))
            
            if i < 3:  # 打印前几个batch的信息
                print(f"Sample {sample_idx}: {len(spot_ids)} spots, embedding shape: {embeddings.shape}")
    
    # 合并所有数据
    fusion_embeddings = np.vstack(all_embeddings)
    coordinates = np.vstack(all_coords)
    
    print(f"Total: {len(all_spot_ids)} spots, embedding shape: {fusion_embeddings.shape}")
    print(f"Coordinate shape: {coordinates.shape}")
    print(f"Coordinate range: X[{coordinates[:, 0].min():.1f}, {coordinates[:, 0].max():.1f}], "
          f"Y[{coordinates[:, 1].min():.1f}, {coordinates[:, 1].max():.1f}]")
    
    return fusion_embeddings, coordinates, all_spot_ids, all_sample_indices

def create_anndata_object(embeddings, coordinates, spot_ids, sample_indices):
    """创建AnnData对象用于scanpy分析"""
    # 创建AnnData对象
    adata = ad.AnnData(X=embeddings)
    
    # 添加obs信息
    adata.obs['spot_id'] = spot_ids
    adata.obs['sample_idx'] = sample_indices
    adata.obs['x_coord'] = coordinates[:, 0]
    adata.obs['y_coord'] = coordinates[:, 1]
    adata.obs_names = spot_ids
    
    # 添加var信息
    adata.var_names = [f'feature_{i}' for i in range(embeddings.shape[1])]
    
    print(f"Created AnnData object: {adata.n_obs} spots × {adata.n_vars} features")
    
    return adata

def perform_leiden_clustering(adata, resolution=0.5, n_neighbors=15):
    """使用Leiden算法进行聚类"""
    print(f"Performing Leiden clustering with resolution={resolution}, n_neighbors={n_neighbors}")
    
    # 计算邻居图
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
    
    # 进行Leiden聚类
    sc.tl.leiden(adata, resolution=resolution, key_added='leiden')
    
    # 获取聚类结果
    clusters = adata.obs['leiden'].astype(int)
    n_clusters = len(np.unique(clusters))
    
    print(f"Found {n_clusters} clusters")
    print("Cluster distribution:")
    cluster_counts = clusters.value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        print(f"  Cluster {cluster_id}: {count} spots")
    
    return clusters, n_clusters

def visualize_clusters_on_image(coordinates, clusters, image_path, output_path, 
                                figsize=(16, 12), point_size=100, alpha=0.8):
    """在原图上可视化聚类结果"""
    print(f"Loading original image: {image_path}")
    image = Image.open(image_path)
    image_array = np.array(image)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image_array)
    
    # 获取聚类数量和颜色
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    
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
    ax.set_title(f'Fusion Embedding Leiden Clustering\n({n_clusters} clusters, {len(coordinates)} spots)', 
                fontsize=16)
    ax.set_xlim(0, image.width)
    ax.set_ylim(image.height, 0)  # 翻转Y轴
    ax.axis('off')
    
    # 添加图例
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    legend.set_frame_on(True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.8)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Clustering visualization saved to: {output_path}")
    
    plt.show()
    
    return fig

def save_clustering_results(coordinates, clusters, spot_ids, sample_indices, output_csv):
    """保存聚类结果到CSV文件"""
    results_df = pd.DataFrame({
        'spot_id': spot_ids,
        'sample_idx': sample_indices,
        'x_coord': coordinates[:, 0],
        'y_coord': coordinates[:, 1],
        'leiden_cluster': clusters
    })
    
    results_df.to_csv(output_csv, index=False)
    print(f"Clustering results saved to: {output_csv}")
    
    return results_df

def plot_cluster_statistics(clusters, output_path):
    """绘制聚类统计信息"""
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
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Cluster statistics saved to: {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Fusion Embedding Leiden Clustering')
    parser.add_argument('--npz_path', type=str, 
                       default='/home/maweicheng/ST_Graduation_Project/evaluation_results/detailed_evaluation_results.npz',
                       help='Path to NPZ file containing fusion embeddings')
    parser.add_argument('--image_path', type=str,
                       default='/home/maweicheng/ST_Graduation_Project/database/GSM6177601/GSE203612_GSM6177601.png',
                       help='Path to original H&E image')
    parser.add_argument('--output_dir', type=str,
                       default='/home/maweicheng/ST_Graduation_Project/clustering_results',
                       help='Output directory for results')
    parser.add_argument('--resolution', type=float, default=0.5,
                       help='Leiden clustering resolution')
    parser.add_argument('--n_neighbors', type=int, default=15,
                       help='Number of neighbors for building the graph')
    parser.add_argument('--use_center_only', action='store_true', default=True,
                       help='Only use center node embedding from each batch (default: True)')
    parser.add_argument('--use_all_nodes', action='store_true', 
                       help='Use all node embeddings including neighbors')
    parser.add_argument('--point_size', type=int, default=100,
                       help='Point size for visualization')
    
    args = parser.parse_args()
    
    # 确定使用哪种模式
    use_center_only = args.use_center_only and not args.use_all_nodes
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== Fusion Embedding Leiden Clustering ===")
    print(f"Mode: {'Center nodes only' if use_center_only else 'All nodes (including neighbors)'}")
    
    # 1. 加载fusion embeddings
    print("\n1. Loading fusion embeddings...")
    embeddings, coordinates, spot_ids, sample_indices = load_fusion_embeddings(args.npz_path, use_center_only)
    
    # 2. 创建AnnData对象
    print("\n2. Creating AnnData object...")
    adata = create_anndata_object(embeddings, coordinates, spot_ids, sample_indices)
    
    # 3. 进行Leiden聚类
    print("\n3. Performing Leiden clustering...")
    clusters, n_clusters = perform_leiden_clustering(adata, 
                                                    resolution=args.resolution, 
                                                    n_neighbors=args.n_neighbors)
    
    # 4. 在原图上可视化聚类结果
    print("\n4. Visualizing clusters on original image...")
    cluster_vis_path = os.path.join(args.output_dir, 'fusion_embedding_leiden_clusters.png')
    visualize_clusters_on_image(coordinates, clusters, args.image_path, cluster_vis_path,
                               point_size=args.point_size)
    
    # 5. 保存聚类结果
    print("\n5. Saving clustering results...")
    results_csv = os.path.join(args.output_dir, 'leiden_clustering_results.csv')
    results_df = save_clustering_results(coordinates, clusters, spot_ids, sample_indices, results_csv)
    
    # 6. 绘制聚类统计信息
    print("\n6. Creating cluster statistics...")
    stats_path = os.path.join(args.output_dir, 'cluster_statistics.png')
    plot_cluster_statistics(clusters, stats_path)
    
    # 7. 保存AnnData对象（可选）
    adata_path = os.path.join(args.output_dir, 'fusion_embeddings_clustered.h5ad')
    adata.write(adata_path)
    print(f"AnnData object saved to: {adata_path}")
    
    print(f"\n=== Clustering Analysis Complete ===")
    print(f"Results saved in: {args.output_dir}")
    print(f"- Found {n_clusters} clusters")
    print(f"- Processed {len(spot_ids)} spots")
    print(f"- Embedding dimension: {embeddings.shape[1]}")

if __name__ == '__main__':
    main()
