"""
Stage 1 Utility Functions
Includes marker gene loading, clustering, and cluster aggregation methods
"""

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from typing import List, Tuple, Dict, Optional
import torch
import scipy.sparse as sp


def load_marker_genes_from_file(file_path: str) -> List[str]:
    """
    Load marker genes from a text file
    
    Args:
        file_path: Path to the text file containing marker genes (one gene per line)
    
    Returns:
        marker_genes: List of marker genes
    """
    print(f"Loading marker genes from file: {file_path}")

    import os
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Marker genes file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        marker_genes = [line.strip() for line in f if line.strip()]
    
    print(f"   Loaded {len(marker_genes)} marker genes")
    return marker_genes


def compute_clusters_and_marker_genes(adata, 
                                     top_n: int = 100, 
                                     min_fold_change: float = 1.5, 
                                     resolution: float = 0.5, 
                                     save_path: Optional[str] = None,
                                     marker_selection_method: str = 'l1',
                                     min_cells_per_cluster: int = 2) -> Tuple[List[str], pd.Series, sc.AnnData]:
    """
    Compute clusters and extract top marker genes for each cluster
    
    Args:
        adata: AnnData object
        top_n: Number of top marker genes per cluster
        min_fold_change: Minimum fold change for marker selection
        resolution: Leiden clustering resolution
        save_path: Path to save marker genes (optional)
        marker_selection_method: Method for final marker gene selection
            - 'l1': Use L1-regularized logistic regression (Lasso)
            - 'variance': Use variance threshold filtering
            - 'correlation': Use correlation-based filtering
    
    Returns:
        marker_genes: List of marker genes
        sc_clusters: Series of cluster labels
        adata_full: Clustered AnnData object
    """
    adata_backup = adata.copy()
    
    # Preprocessing: normalization and PCA
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    
    # PCA
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    
    # Build neighbor graph
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    
    # Leiden clustering
    sc.tl.leiden(adata, resolution=resolution)

    # Restore to original gene set for marker analysis
    adata_full = adata_backup.copy()
    sc.pp.normalize_total(adata_full, target_sum=1e4)
    sc.pp.log1p(adata_full)
    
    # Transfer clustering results to full dataset
    adata_full.obs['leiden'] = adata.obs['leiden'].copy()

    # Drop clusters with too few cells
    counts = adata_full.obs['leiden'].value_counts()
    small_clusters = counts[counts < min_cells_per_cluster].index.tolist()
    if small_clusters:
        keep_mask = ~adata_full.obs['leiden'].isin(small_clusters)

        adata_full = adata_full[keep_mask].copy()
        if hasattr(adata_full.obs['leiden'], 'cat'):
            adata_full.obs['leiden'] = adata_full.obs['leiden'].cat.remove_unused_categories()
    
    # Compute marker genes for each cluster
    sc.tl.rank_genes_groups(
        adata_full, 
        'leiden', 
        method='wilcoxon',
        key_added='rank_genes_groups',
        n_genes=top_n * 2
    )
    
    # Extract marker genes
    marker_genes = set()
    result = adata_full.uns['rank_genes_groups']
    
    lasso_selected = {}
    for cluster in sorted(adata_full.obs['leiden'].unique()):
        if cluster in result['names'].dtype.names:
            genes = result['names'][cluster]
            scores = result['scores'][cluster]
            pvals = result['pvals_adj'][cluster]
            logfoldchanges = result['logfoldchanges'][cluster]
            
            selected_genes = []
            for i in range(len(genes)):
                if (pvals[i] < 0.05 and 
                    scores[i] > 0 and 
                    logfoldchanges[i] >= np.log2(min_fold_change)):
                    selected_genes.append(genes[i])
                    
                if len(selected_genes) >= top_n:
                    break
            
            # Apply marker selection method
            if len(selected_genes) > 0:
                if marker_selection_method == 'l1':
                    # Apply Lasso regression for further selection
                    sub_adata = adata_full[:, selected_genes].copy()
                    y = (adata_full.obs['leiden'] == cluster).astype(int)
                    X = sub_adata.X
                    # 保持稀疏输入以加速 saga（支持 CSR）
                    if not sp.issparse(X):
                        X = sp.csr_matrix(X)
                    clf = LogisticRegression(
                        C=1,
                        penalty='l1',
                        solver='saga',
                        class_weight='balanced',
                        max_iter=2000,
                        n_jobs=-1
                    )

                    clf.fit(X, y)
                    coef = clf.coef_.ravel()

                    gene_to_coef = {g: abs(c) for g, c in zip(selected_genes, coef) if abs(c) > 1e-5}
                    final_selected_genes = sorted(gene_to_coef, key=gene_to_coef.get, reverse=True)
    
                elif marker_selection_method == 'variance':
                    # Apply variance threshold filtering
                    sub_adata = adata_full[:, selected_genes].copy()
                    X = sub_adata.X
                    if hasattr(X, 'toarray'):
                        X = X.toarray()
                    
                    # Calculate variance for each gene
                    variances = np.var(X, axis=0)
                    # Select top genes by variance
                    top_indices = np.argsort(variances)[-min(len(selected_genes), top_n):]
                    final_selected_genes = [selected_genes[i] for i in top_indices]
  
                elif marker_selection_method == 'correlation':
                    # Apply correlation-based filtering
                    sub_adata = adata_full[:, selected_genes].copy()
                    X = sub_adata.X
                    if hasattr(X, 'toarray'):
                        X = X.toarray()

                    # Vectorized Pearson correlation with cluster membership
                    y = (adata_full.obs['leiden'] == cluster).values.astype(np.float64)
                    X_f = X.astype(np.float64)
                    X_centered = X_f - X_f.mean(axis=0)
                    y_centered = y - y.mean()
                    numer = (X_centered * y_centered[:, None]).sum(axis=0)
                    denom = np.sqrt((X_centered ** 2).sum(axis=0) * (y_centered ** 2).sum() + 1e-12)
                    correlations = np.abs(numer / denom)
                    correlations = np.nan_to_num(correlations, nan=0.0)

                    # Select genes with highest absolute correlation
                    top_indices = np.argsort(correlations)[-min(len(selected_genes), top_n):]
                    final_selected_genes = [selected_genes[i] for i in top_indices]
                    
                else:
                    raise ValueError(f"Unknown marker_selection_method: {marker_selection_method}. "
                                   f"Choose from 'l1', 'variance', or 'correlation'")
                
                lasso_selected[cluster] = final_selected_genes
                marker_genes.update(final_selected_genes)
            else:
                lasso_selected[cluster] = []
                print(f"   {cluster}: 0 genes")
    
    print(f"Total: {len(marker_genes)} marker genes")
    print(f"   Number of clusters: {len(adata_full.obs['leiden'].unique())}")
    
    clusters_to_drop = [cluster for cluster, genes in lasso_selected.items() if len(genes) == 0]
    if clusters_to_drop:

        keep_mask = ~adata_full.obs['leiden'].isin(clusters_to_drop)

        adata_full = adata_full[keep_mask].copy()
        if hasattr(adata_full.obs['leiden'], 'cat'):
            adata_full.obs['leiden'] = adata_full.obs['leiden'].cat.remove_unused_categories()
    
    # Return clustering info, marker genes, and full adata for annotation
    sc_clusters = adata_full.obs['leiden'].copy()
    if hasattr(sc_clusters, 'cat'):
        sc_clusters = sc_clusters.cat.remove_unused_categories()
    
    return sorted(list(marker_genes)), sc_clusters, adata_full


def compute_cluster_centers_and_expressions(
    embeddings: np.ndarray,
    sc_train_data: np.ndarray,
    sc_train_labels: np.ndarray,
    sc_X_full_train_count: Optional[np.ndarray] = None,
    aggregation_method: str = 'weighted'
) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Compute cluster centers and expressions using different aggregation methods
    
    Args:
        embeddings: Cell embeddings in latent space [n_cells, latent_dim]
        sc_train_data: Marker gene expression [n_cells, n_marker_genes]
        sc_train_labels: Cluster labels [n_cells]
        sc_X_full_train_count: Full gene expression [n_cells, n_all_genes] (可选，如果为None则不计算全基因聚合表达)
        aggregation_method: 'mean', 'median', or 'weighted'
            - 'mean': Simple average
            - 'median': Median aggregation
            - 'weighted': Weighted average with UMI, representativeness, and marker activity
    
    Returns:
        cluster_prototypes: Dict of cluster centers in latent space
        cluster_expressions: Dict of cluster expressions (marker genes)
        cluster_expressions_full_count: Dict of cluster expressions (all genes, 如果sc_X_full_train_count=None则为空dict)
        cluster_cell_weights: Dict of cell weights per cluster (None for mean/median)
    """

    cluster_prototypes = {}
    cluster_expressions = {}
    cluster_expressions_full_count = {}
    cluster_cell_weights = {}
    
    for cluster_id in np.unique(sc_train_labels):
        cluster_mask = sc_train_labels == cluster_id
        cluster_cells = np.where(cluster_mask)[0]
        n_cells = len(cluster_cells)
        
        if n_cells == 0:
            continue
        
        # Get cluster data
        cluster_embeddings = embeddings[cluster_mask]
        cluster_data = sc_train_data[cluster_mask]
        cluster_full_data = sc_X_full_train_count[cluster_mask] if sc_X_full_train_count is not None else None
        
        if aggregation_method == 'mean':
            # Simple mean aggregation
            cluster_center = np.mean(cluster_embeddings, axis=0)
            cluster_expression = np.mean(cluster_data, axis=0)
            cluster_expr_full = np.mean(cluster_full_data, axis=0) if cluster_full_data is not None else None
            cluster_cell_weights[cluster_id] = None

            
        elif aggregation_method == 'median':
            # Median aggregation
            cluster_center = np.median(cluster_embeddings, axis=0)
            cluster_expression = np.median(cluster_data, axis=0)
            cluster_expr_full = np.median(cluster_full_data, axis=0) if cluster_full_data is not None else None
            cluster_cell_weights[cluster_id] = None

            
        elif aggregation_method == 'weighted':
            # Weighted aggregation
            # 1. UMI weight (如果没有全基因数据，使用marker基因数据)
            umi_data = cluster_full_data if cluster_full_data is not None else cluster_data
            cell_umi = umi_data.sum(axis=1)
            w_umi = np.log1p(cell_umi) / np.mean(np.log1p(cell_umi))
            w_umi = np.clip(w_umi, 0.1, 10.0)
            
            # 2. Representativeness weight (cosine similarity to centroid)
            cluster_centroid = np.mean(cluster_embeddings, axis=0)
            similarities = np.dot(cluster_embeddings, cluster_centroid) / (
                np.linalg.norm(cluster_embeddings, axis=1) * np.linalg.norm(cluster_centroid) + 1e-8
            )
            w_rep = (similarities - similarities.min()) / (similarities.max() - similarities.min() + 1e-8)
            w_rep = np.clip(w_rep, 0.01, 1.0)
            
            # 3. Marker activity weight
            marker_expr_log = np.log1p(cluster_data)
            marker_activity = marker_expr_log.mean(axis=1)
            if marker_activity.max() > marker_activity.min():
                w_marker = (marker_activity - marker_activity.min()) / (marker_activity.max() - marker_activity.min() + 1e-8)
            else:
                w_marker = np.ones(n_cells)
            w_marker = np.clip(w_marker, 0.01, 1.0)
            
            # 4. Combine weights
            alpha, beta, gamma = 0.4, 0.3, 0.3
            w_combined = alpha * w_umi + beta * w_rep + gamma * w_marker
            w_combined = w_combined / (w_combined.sum() + 1e-8)
            
            # Store weights
            cluster_cell_weights[cluster_id] = w_combined
            
            # 5. Compute weighted aggregates
            cluster_center = np.sum(cluster_embeddings * w_combined[:, np.newaxis], axis=0)
            cluster_expression = np.sum(cluster_data * w_combined[:, np.newaxis], axis=0)
            cluster_expr_full = np.sum(cluster_full_data * w_combined[:, np.newaxis], axis=0) if cluster_full_data is not None else None
            
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}. "
                           f"Choose from 'mean', 'median', or 'weighted'")
        
        # Save results
        cluster_prototypes[cluster_id] = cluster_center
        cluster_expressions[cluster_id] = cluster_expression
        if cluster_expr_full is not None:
            cluster_expressions_full_count[cluster_id] = cluster_expr_full
 
    return cluster_prototypes, cluster_expressions, cluster_expressions_full_count, cluster_cell_weights


def print_weight_statistics(sc_X_full_train_count: np.ndarray):
    """
    Print UMI and gene count statistics for weight computation
    
    Args:
        sc_X_full_train_count: Full gene expression matrix [n_cells, n_genes]
    """
    print("   Computing cell weight statistics...")
    cell_umi = sc_X_full_train_count.sum(axis=1)
    cell_n_genes = (sc_X_full_train_count > 0).sum(axis=1)
    
    # UMI weight
    w_umi = np.log1p(cell_umi) / np.mean(np.log1p(cell_umi))
    w_umi = np.clip(w_umi, 0.1, 10.0)
    
    # n_genes weight
    w_ngenes = np.log1p(cell_n_genes) / np.mean(np.log1p(cell_n_genes))
    w_ngenes = np.clip(w_ngenes, 0.1, 10.0)
    
    print(f"      UMI weight: mean={w_umi.mean():.3f}, min={w_umi.min():.3f}, max={w_umi.max():.3f}")
    print(f"      n_genes weight: mean={w_ngenes.mean():.3f}, min={w_ngenes.min():.3f}, max={w_ngenes.max():.3f}")
