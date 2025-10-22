import os
import scanpy as sc
import anndata as ad
import numpy as np
from typing import Tuple, List

def load_data(data_dir: str) -> Tuple[ad.AnnData, ad.AnnData, List[str]]:
    """
    Load SC and ST data from specified directory
    
    Data structure requirement:
    data_dir/
    ├── sample1/
    │   ├── sample1_SC.h5ad
    │   └── sample1_ST.h5ad
    ├── sample2/
    │   ├── sample2_SC.h5ad
    │   └── sample2_ST.h5ad
    └── ...
    """
    print(f"Loading data from {data_dir}...")
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Get all subdirectories (samples)
    sample_dirs = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    sample_dirs.sort()
    
    if not sample_dirs:
        raise ValueError(f"No sample directories found in: {data_dir}")
    
    print(f"Found samples: {sample_dirs}")
    
    sc_data_list = []
    st_data_list = []
    valid_samples = []
    
    for sample in sample_dirs:
        sample_dir = os.path.join(data_dir, sample)
        
        # Find *_SC.h5ad and *_ST.h5ad files
        sc_files = [f for f in os.listdir(sample_dir) if f.endswith('SC.h5ad')]
        st_files = [f for f in os.listdir(sample_dir) if f.endswith('ST.h5ad')]
        
        if not sc_files or not st_files:
            print(f"[{sample}] Warning: Missing SC.h5ad or ST.h5ad file, skipping")
            continue
        
        sc_file = os.path.join(sample_dir, sc_files[0])
        st_file = os.path.join(sample_dir, st_files[0])
        
        try:
            print(f"[{sample}] Loading...")
            
            # Load SC data
            sc_adata = sc.read_h5ad(sc_file)
            sc_adata.obs['sample'] = sample
            sc_adata.obs['modality'] = 'SC'
            
            # Load ST data
            st_adata = sc.read_h5ad(st_file)
            st_adata.obs['sample'] = sample
            st_adata.obs['modality'] = 'ST'
            
            print(f"  SC shape: {sc_adata.shape}")
            print(f"  ST shape: {st_adata.shape}")
            
            sc_data_list.append(sc_adata)
            st_data_list.append(st_adata)
            valid_samples.append(sample)
            
        except Exception as e:
            print(f"[{sample}] Error loading: {str(e)}")
            continue
    
    if not valid_samples:
        raise ValueError(f"No valid samples could be loaded")
    
    # Merge SC data - use inner join to keep only common genes
    print(f"Merging {len(sc_data_list)} SC samples...")
    combined_sc = ad.concat(sc_data_list, axis=0, join='inner', 
                            keys=valid_samples, index_unique='-')
    
    # Merge ST data - use inner join to keep only common genes  
    print(f"Merging {len(st_data_list)} ST samples...")
    combined_st = ad.concat(st_data_list, axis=0, join='inner', 
                            keys=valid_samples, index_unique='-')
    
    print(f"Data loading completed!")
    print(f"  SC total shape: {combined_sc.shape}")
    print(f"  ST total shape: {combined_st.shape}")
    
    return combined_sc, combined_st, valid_samples

def compute_clusters_and_marker_genes(adata, top_n=100, min_fold_change=1.5, resolution=1, save_path=None):
    """
    Compute clusters and extract top marker genes for each cluster
    """
    print("Starting clustering analysis...")
    
    # Backup original data
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
    
    # Build neighborhood graph
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    
    # Leiden clustering
    sc.tl.leiden(adata, resolution=resolution)
    
    n_clusters = len(adata.obs['leiden'].unique())
    print(f"Clustering result: {n_clusters} clusters")
    for cluster in sorted(adata.obs['leiden'].unique()):
        count = (adata.obs['leiden'] == cluster).sum()
        print(f"  Cluster {cluster}: {count} cells")
    
    # Restore to original gene set for marker analysis
    adata_full = adata_backup.copy()
    sc.pp.normalize_total(adata_full, target_sum=1e4)
    sc.pp.log1p(adata_full)
    
    # Transfer clustering results to full data
    adata_full.obs['leiden'] = adata.obs['leiden'].copy()
    
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
            
            marker_genes.update(selected_genes)

    
    total_genes = len(marker_genes)
    print(f"Total: {total_genes} marker genes")
    
    # Return clustering info and marker genes
    return sorted(list(marker_genes)), adata_full.obs['leiden'].copy()