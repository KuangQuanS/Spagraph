import os
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import argparse
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Import unified model definitions
from .deconv_model import VAE, HeterogeneousGATDeconvolution, SpatialDeconvolutionLoss
from .deconv_initialization import compute_signature_initialization

class SpatialDataset(Dataset):
    """Spatial transcriptomics dataset
    
    Stores both normalized (for embedding) and raw (for loss) data
    """
    def __init__(self, st_data_normalized, st_data_raw, spatial_coords, spot_ids):
        self.st_data_normalized = torch.FloatTensor(st_data_normalized)
        self.st_data_raw = torch.FloatTensor(st_data_raw)
        self.spatial_coords = torch.FloatTensor(spatial_coords)
        self.spot_ids = spot_ids
        
    def __len__(self):
        return len(self.st_data_normalized)
    
    def __getitem__(self, idx):
        return {
            'expression_normalized': self.st_data_normalized[idx],  # For VAE embedding
            'expression_raw': self.st_data_raw[idx],                # For loss
            'coords': self.spatial_coords[idx],
            'spot_id': self.spot_ids[idx],
            'index': idx  # Add index for spot_total_counts lookup
        }

class GATDeconvolution:
    """Stage 2: GAT deconvolution trainer
    
    Supports two modes:
    1. File mode: Load via stage1_model_path
    2. Memory mode: Pass directly via stage1_artifacts
    """
    def __init__(self, stage1_model_path: str = None, output_dir: str = "./stage2_results/", 
                 device: str = None, weight_threshold: float = 0.01, stage1_artifacts=None, seed: int = 42,
                 use_ols_scaling: bool = False, library_size: float = 1.0):

        # Set random seed first
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.stage1_model_path = stage1_model_path
        self.stage1_artifacts = stage1_artifacts
        self.output_dir = output_dir
        self.seed = seed
        # Don't create directory on init; create only when saving files
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Weight threshold for sparsification
        self.weight_threshold = weight_threshold
        
        # Output options
        self.save_reconstructed_genes = False  # Whether to save reconstructed full gene expression
        self.use_ols_scaling = use_ols_scaling  # Whether to use OLS (Ordinary Least Squares) scaling (default False uses sum-based)
        self.library_size = library_size  # Manual library size factor, multiplied on top of scale (default 1.0, no change)

        # Model components
        self.vae_encoder = None
        self.gat_model = None
        self.loss_fn = None
        self.label_encoder = None
        self.marker_genes = None
        self.celltype_prototypes = None
        self.celltype_expressions = None
        self.cluster_to_celltype = None  # Mapping from cluster index to celltype name
        self.celltype_key = None  # Track which celltype column was used
        self.use_embedding_knn = False  # If True, build spot KNN with embeddings when spatial coords missing
        self.initial_weights = None
        self.restore_best_state = True
        self.full_graph_training = True
        
        # Graph construction parameters
        self.k_spatial = 20
        self.k_celltype = 10
    
    def load_from_artifacts(self, artifacts):
        """Load all necessary components from Stage1Artifacts object (pure memory mode)"""
        # Load VAE encoder
        if artifacts.vae_encoder is not None:
            # Directly use the loaded encoder
            self.vae_encoder = artifacts.vae_encoder
            if hasattr(self.vae_encoder, 'to'):
                self.vae_encoder = self.vae_encoder.to(self.device)
            self.vae_encoder.eval()
            self.latent_dim = self.vae_encoder.fc_mu.out_features
        elif artifacts.vae_state_dict is not None:
            # Rebuild encoder from state_dict
            from .deconv_model import DualDecoderVAE
            input_dim = artifacts.input_dim
            latent_dim = artifacts.latent_dim
            output_type = artifacts.output_type or 'mse'
            
            full_vae = DualDecoderVAE(input_dim=input_dim, latent_dim=latent_dim, output_type=output_type).to(self.device)
            full_vae.load_state_dict(artifacts.vae_state_dict)
            
            self.vae_encoder = full_vae.encoder
            self.vae_encoder.eval()
            self.latent_dim = latent_dim
        else:
            raise ValueError("Stage1Artifacts must contain vae_encoder or vae_state_dict")
        
        # Load other components
        self.label_encoder = artifacts.label_encoder
        self.marker_genes = artifacts.marker_genes
        self.genes = artifacts.genes
        self.sc_clusters = artifacts.sc_clusters
        self.resolution = artifacts.resolution or 0.5
        self.celltype_key = artifacts.celltype_key
        self.cluster_to_celltype = artifacts.cluster_to_celltype
        self.all_genes = artifacts.all_genes
        self.hvg_genes_union = artifacts.hvg_genes_union
        self.auto_library_size = getattr(artifacts, 'auto_library_size', 1.0)  # Load auto-computed library_size
        
        # Load cluster centers and expressions
        if artifacts.celltype_prototypes is not None:
            if isinstance(artifacts.celltype_prototypes, torch.Tensor):
                self.celltype_prototypes = artifacts.celltype_prototypes.to(self.device)
            else:
                self.celltype_prototypes = torch.FloatTensor(artifacts.celltype_prototypes).to(self.device)
        
        if artifacts.celltype_expressions is not None:
            if isinstance(artifacts.celltype_expressions, torch.Tensor):
                self.celltype_expressions = artifacts.celltype_expressions.to(self.device)
            else:
                self.celltype_expressions = torch.FloatTensor(artifacts.celltype_expressions).to(self.device)
        
        if artifacts.celltype_expressions_full is not None:
            self.celltype_expressions_full = artifacts.celltype_expressions_full
        
        # Freeze encoder parameters
        for param in self.vae_encoder.parameters():
            param.requires_grad = False
        
    def load_vae_encoder(self):
        """Load Stage 1 VAE components
        
        Prioritize loading from stage1_artifacts (memory mode), otherwise load from file
        """
        # If artifacts exist and in memory mode, use directly
        if self.stage1_artifacts is not None and self.stage1_artifacts.is_memory_mode():
            self.load_from_artifacts(self.stage1_artifacts)
            return
        
        # Otherwise load from file
        if self.stage1_model_path is None:
            raise ValueError("Must provide stage1_model_path or stage1_artifacts containing memory data")
        
        checkpoint = torch.load(self.stage1_model_path, map_location=self.device, weights_only=False)
        
        # Rebuild VAE
        input_dim = checkpoint['input_dim']
        latent_dim = checkpoint['latent_dim']
        output_type = checkpoint.get('output_type', 'mse')  # Get output_type from checkpoint
        
        # Detect if it's a dual decoder architecture
        state_dict = checkpoint['vae_state_dict']

        from .deconv_model import DualDecoderVAE
        full_vae = DualDecoderVAE(input_dim=input_dim, latent_dim=latent_dim, output_type=output_type).to(self.device)

        full_vae.load_state_dict(state_dict)

        self.vae_encoder = full_vae.encoder
        self.vae_encoder.eval()  # Freeze encoder
        
        # Get latent dimension from the loaded VAE model
        self.latent_dim = full_vae.encoder.fc_mu.out_features
        
        # Other information
        self.label_encoder = checkpoint['label_encoder']
        self.marker_genes = checkpoint['marker_genes']
        self.genes = checkpoint['genes']
        self.sc_clusters = checkpoint.get('sc_clusters', None)

        self.resolution = checkpoint.get('resolution', 0.5)
        self.celltype_key = checkpoint.get('celltype_key', None)  # Get celltype mode info
        
        # Load cluster-to-celltype mapping if available
        self.cluster_to_celltype = None
        if self.celltype_key is not None:
            # If using celltype mode, create mapping from label_encoder
            # label_encoder.classes_ contains celltype names
            self.cluster_to_celltype = {i: ct for i, ct in enumerate(self.label_encoder.classes_)}

        # Load cluster data from npz file
        npz_filepath = self.stage1_model_path.replace('.pth', '_cluster_data.npz')
        
        if not os.path.exists(npz_filepath):
            raise FileNotFoundError(
                f"Cluster data file not found: {npz_filepath}\n"
                f"Please retrain Stage 1 with the latest version to generate the npz file."
            )
        
        cluster_data = np.load(npz_filepath, allow_pickle=True)
        
        cluster_ids = cluster_data['cluster_ids']
        prototypes_array = cluster_data['cluster_prototypes']
        expressions_array = cluster_data['cluster_expressions']
        expressions_full_array = cluster_data['cluster_expressions_full']
        
        # Verify cluster IDs match label_encoder
        n_clusters = len(self.label_encoder.classes_)
        if len(cluster_ids) != n_clusters:
            raise ValueError(
                f"Cluster count mismatch! NPZ has {len(cluster_ids)} clusters, "
                f"but label_encoder has {n_clusters} clusters"
            )
        
        # Convert to tensor
        self.celltype_prototypes = torch.FloatTensor(prototypes_array).to(self.device)
        self.celltype_expressions = torch.FloatTensor(expressions_array).to(self.device)
        
        if expressions_full_array.ndim == 0:
            # It's a 0-d object array containing a list
            expressions_full_list = expressions_full_array.item()
        else:
            # It's a 1-d object array, each element is an array
            expressions_full_list = [expressions_full_array[i] for i in range(len(cluster_ids))]
        
        self.celltype_expressions_full = expressions_full_list

        # Optional: load HVG intersection (SC/ST 3000 HVGs) if available
        if 'hvg_genes_union' in cluster_data:
            try:
                hvg_union = cluster_data['hvg_genes_union']
                # hvg_union may be object array; convert to Python list of str
                if isinstance(hvg_union, np.ndarray):
                    hvg_union = list(hvg_union.tolist())
                self.hvg_genes_union = [str(g) for g in hvg_union]
            except Exception as e:
                self.hvg_genes_union = None
        else:
            self.hvg_genes_union = None
        
        # Load celltype mapping if available
        if 'cluster_to_celltype' in cluster_data:
            celltype_mapping_array = cluster_data['cluster_to_celltype']
            self.cluster_to_celltype = {str(row['cluster_id']): str(row['celltype']) 
                                       for row in celltype_mapping_array}
        else:
            self.cluster_to_celltype = None
        
        # Load average cell counts
        self.avg_cell_counts = checkpoint.get('avg_cell_counts', None)
        
        # Load all marker genes list
        all_genes = checkpoint.get('all_genes', None)
        if all_genes is not None:
            self.all_genes = all_genes
        else:
            self.all_genes = None

        # Freeze encoder parameters
        for param in self.vae_encoder.parameters():
            param.requires_grad = False
       
    def build_gat_model(self, n_cell_types: int, gat_hidden_dim=64, gat_layers=3, 
                       gat_heads=4, dropout=0.1, loss_lambda_pearson=1.0, loss_lambda_mse=1.0,
                       loss_lambda_cosine=1.0, loss_lambda_gene_pearson=0.0, loss_lambda_gene_cosine=0.0,
                       loss_lambda_reg=0.5, loss_lambda_sparse=0.01,
                       loss_lambda_proportion=1.0, spot_total_counts=None,
                       use_dynamic_cluster_repr=False, k_cells_per_cluster=10, 
                       sc_cell_expressions=None,
                       signature_prior_strength: float = 0.0,
                       lambda_poisson: float = 0.0,
                       lambda_spatial: float = 0.0,
                       spatial_temperature: float = 1.0,
                       heldout_gene_fraction: float = 0.0):
        """Build GAT deconvolution model
        
        Args:
            spot_total_counts: Array of total counts for each spot (shape: [n_spots])
                              Used for scaling: reconstructed = s_i × Σ(w_ic × R_c)
            use_dynamic_cluster_repr: Whether to enable dynamic cluster representation
            k_cells_per_cluster: Number of nearest cells to use per cluster
            sc_cell_expressions: [n_cells, n_all_genes] Single-cell full gene raw counts (required for dynamic mode)
        """
        # Get embedding dimension from VAE encoder
        embedding_dim = self.latent_dim
        if spot_total_counts is not None:
            self.spot_total_counts = spot_total_counts
        else:
            self.spot_total_counts = None
        self.gat_model = HeterogeneousGATDeconvolution(
            embedding_dim=embedding_dim,  # Use actual VAE latent dimension
            n_cell_types=n_cell_types,
            gat_hidden_dim=gat_hidden_dim,
            gat_layers=gat_layers,
            gat_heads=gat_heads,
            dropout=dropout,
            k_spatial=self.k_spatial,
            k_celltype=self.k_celltype,
            celltype_prototypes=self.celltype_prototypes,  # Initialize with Stage 1 celltype prototypes
            # Dynamic cluster parameters
            use_dynamic_cluster_repr=use_dynamic_cluster_repr,
            k_cells_per_cluster=k_cells_per_cluster,
            sc_cell_expressions=sc_cell_expressions,  # [n_cells, n_all_genes] full genes
            signature_prior_strength=signature_prior_strength,
        ).to(self.device)
        
        # Compute cluster proportions in single-cell data
        sc_celltype_proportions = None
        if hasattr(self, 'sc_clusters') and self.sc_clusters is not None:
            # sc_clusters are single-cell cluster labels loaded from stage1
            cluster_counts = {}
            for cluster_id in self.sc_clusters:
                cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
            
            total_cells = len(self.sc_clusters)
            # Build proportion array in cluster ID order
            proportions = []
            for i in range(n_cell_types):
                count = cluster_counts.get(str(i), 0)  # cluster ID may be string
                if count == 0:
                    count = cluster_counts.get(i, 0)  # Try integer
                proportion = count / total_cells if total_cells > 0 else 1.0 / n_cell_types
                proportions.append(proportion)
            
            sc_celltype_proportions = proportions
        else:
            pass
        
        celltype_expr_full = np.vstack([np.asarray(expr, dtype=np.float64) for expr in self.celltype_expressions_full])
        # Compute marker gene indices in all genes
        marker_gene_indices = [self.all_genes.index(g) for g in self.genes]
        # Compute HVG intersection indices in all genes (if available)
        hvg_gene_indices = None
        if getattr(self, "hvg_genes_union", None) is not None:
            try:
                hvg_gene_indices = [self.all_genes.index(g) for g in self.hvg_genes_union if g in self.all_genes]
            except Exception as e:
                hvg_gene_indices = None
        
        self.loss_fn = SpatialDeconvolutionLoss(
            lambda_pearson=loss_lambda_pearson,
            lambda_mse=loss_lambda_mse,
            lambda_cosine=loss_lambda_cosine,
            lambda_gene_pearson=loss_lambda_gene_pearson,
            lambda_gene_cosine=loss_lambda_gene_cosine,
            lambda_reg=loss_lambda_reg,
            lambda_sparse=loss_lambda_sparse,
            lambda_proportion=loss_lambda_proportion,
            sc_celltype_proportions=sc_celltype_proportions,
            spot_total_counts=self.spot_total_counts,
            celltype_expressions_full=celltype_expr_full,
            marker_gene_indices=marker_gene_indices,
            hvg_gene_indices=hvg_gene_indices,
            scale_basis=getattr(self, "scale_basis", "hvg"),
            library_size=self.library_size,
            lambda_poisson=lambda_poisson,
            lambda_spatial=lambda_spatial,
            spatial_temperature=spatial_temperature,
            heldout_gene_fraction=heldout_gene_fraction,
            split_seed=self.seed,
        ).to(self.device)  # Move loss function to device
        
    def train_epoch_batched(self, 
                           dataloader: DataLoader,
                           optimizer) -> Dict[str, float]:
        """Train one epoch with batching"""
        self.gat_model.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'pearson_loss': 0.0,
            'mse_loss': 0.0,
            'cosine_loss': 0.0,
            'gene_pearson_loss': 0.0,
            'gene_cosine_loss': 0.0,
            'weight_reg': 0.0,
            'sparsity_loss': 0.0,
            'proportion_loss': 0.0,
            'poisson_loss': 0.0,
            'spatial_loss': 0.0,
            'heldout_loss': 0.0,
        }
        
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            # Extract batch data
            batch_st_normalized = batch['expression_normalized'].to(self.device)  # For embedding
            batch_st_raw = batch['expression_raw'].to(self.device)                # For loss
            batch_spatial_coords = batch['coords'].to(self.device)
            batch_indices = batch['index']  # Get batch indices
            
            # Get spot_total_counts for this batch (convert to tensor)
            if self.spot_total_counts is not None:
                batch_spot_total_counts = torch.FloatTensor(
                    self.spot_total_counts[batch_indices]
                ).to(self.device)
            else:
                batch_spot_total_counts = None
            
            # Get dynamic cluster data (if enabled)
            batch_knn_indices = None
            if self.knn_cell_indices is not None:
                # Use non_blocking to accelerate data transfer
                batch_knn_indices = torch.LongTensor(
                    self.knn_cell_indices[batch_indices]
                ).to(self.device, non_blocking=True)
            
            # Compute spot embeddings (using normalized data, consistent with VAE training)
            with torch.no_grad():
                mu, log_var = self.vae_encoder(batch_st_normalized)
                spot_embeddings = mu
            
            # GAT forward pass
            gat_outputs = self.gat_model(
                spot_embeddings=spot_embeddings,
                spatial_coords=batch_spatial_coords,
                celltype_prototypes=self.celltype_prototypes,
                use_embedding_knn=self.use_embedding_knn,
                initial_weights=(
                    torch.as_tensor(
                        self.initial_weights[np.asarray(batch_indices)],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    if self.initial_weights is not None else None
                ),
            )
            
            # Compute loss (using raw count data)
            loss_outputs = self.loss_fn(
                attention_weights=gat_outputs['deconv_weights'],
                celltype_expression=self.celltype_expressions,
                true_spot_expression=batch_st_raw,  # Use raw counts for loss
                spot_embedding=spot_embeddings,  # Use original VAE embeddings for dynamic cluster MLP
                celltype_embedding=gat_outputs['celltype_features'],
                edge_index=gat_outputs['edge_index'],
                batch_spot_total_counts=batch_spot_total_counts,  # Pass batch-specific counts
                # Dynamic cluster parameters
                knn_cell_indices=batch_knn_indices,
                sc_cell_embeddings=self.sc_cell_embeddings_tensor,  # Use pre-converted tensor
                sc_cell_expressions=self.gat_model.sc_cell_expressions if hasattr(self.gat_model, 'sc_cell_expressions') else None,
                gat_model=self.gat_model
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss_outputs['total_loss'].backward()
            optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses.keys():
                if key in loss_outputs:
                    epoch_losses[key] += loss_outputs[key].item()
        
        # Compute average losses
        for key in epoch_losses.keys():
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def train_gat_deconvolution(self, 
                               st_data_normalized: np.ndarray,  # For VAE embedding (match Stage 1 input; now raw counts)
                               st_data_raw: np.ndarray,         # For loss calculation (raw counts)
                               spatial_coords: np.ndarray,
                               sample_name: str,
                               st_adata=None,
                               n_epochs: int = 50,
                               lr: float = 1e-3,
                               batch_size: int = 512,
                               print_every: int = 50,
                               knn_cell_indices=None,
                               sc_cell_embeddings=None,
                               initial_weights=None,
                               full_graph_training: bool = True,
                               restore_best_state: bool = True):
        """Train GAT deconvolution model
        
        Args:
            st_data_normalized: Normalized ST data (sum=1) for VAE embedding
            st_data_raw: Raw count ST data for loss calculation
            print_every: Print loss every N epochs (default: 50)
            knn_cell_indices: [n_spots, n_cell_types, k] Pre-computed k-nearest cell indices (dynamic mode)
            sc_cell_embeddings: [n_cells, embedding_dim] Single-cell embeddings (dynamic mode)
        """
        # Save st_adata for later use
        self.st_adata = st_adata
        
        # Save dynamic cluster data
        self.knn_cell_indices = knn_cell_indices
        self.sc_cell_embeddings = sc_cell_embeddings
        self.initial_weights = initial_weights
        self.full_graph_training = bool(full_graph_training)
        self.restore_best_state = bool(restore_best_state)
        
        # Pre-convert to GPU tensor to avoid repeated conversion per batch
        if self.sc_cell_embeddings is not None:
            self.sc_cell_embeddings_tensor = torch.FloatTensor(self.sc_cell_embeddings).to(self.device)
        else:
            self.sc_cell_embeddings_tensor = None
 
        spatial_tensor = torch.FloatTensor(spatial_coords).to(self.device)
        
        # Create dataset and dataloader (use normalized data for embedding)
        spot_ids = list(range(len(st_data_normalized)))
        dataset = SpatialDataset(st_data_normalized, st_data_raw, spatial_coords, spot_ids)
        
        # DataLoader configuration (num_workers=0 for Windows/Jupyter compatibility)
        effective_batch_size = len(dataset) if self.full_graph_training else batch_size
        dataloader = DataLoader(
            dataset, 
            batch_size=effective_batch_size,
            shuffle=not self.full_graph_training,
            drop_last=False,
            num_workers=0,  # Use 0 for Windows and Jupyter notebook compatibility (single-process)
            pin_memory=True if torch.cuda.is_available() else False  # Only use pin_memory with CUDA
        )
        
        # Optimizer
        optimizer = torch.optim.Adam(self.gat_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=20, factor=0.5
        )
        
        # Training history
        train_losses = []
        pearson_losses = []
        mse_losses = []
        cos_losses = []
        gene_pearson_losses = []
        gene_cosine_losses = []
        weight_regs = []
        sparsity_regs = []
        diversity_losses = []
        hetero_losses = []
        proportion_losses = []
        
        # Early stopping strategy: Monitor absolute improvement of Pearson and Cosine
        best_pearson = float('inf')
        best_mse = float('inf')
        best_cosine = float('inf')
        best_gene_pearson = float('inf')
        best_gene_cosine = float('inf')
        patience_counter = 0
        patience = 20  # Early stopping patience
        min_delta = 0.001  # Absolute improvement threshold (Pearson/Cosine needs to decrease by at least this much)
        best_monitor = float('inf')
        best_state = None
        best_epoch = 0
        selected_metrics = None
        
        for epoch in range(n_epochs):
            # Train one epoch
            epoch_losses = self.train_epoch_batched(
                dataloader=dataloader,
                optimizer=optimizer
            )
            
            # Record average loss
            avg_total_loss = epoch_losses['total_loss']
            train_losses.append(avg_total_loss)
            pearson_losses.append(epoch_losses['pearson_loss'])
            mse_losses.append(epoch_losses['mse_loss'])
            cos_losses.append(epoch_losses['cosine_loss'])
            gene_pearson_losses.append(epoch_losses.get('gene_pearson_loss', 0.0))
            gene_cosine_losses.append(epoch_losses.get('gene_cosine_loss', 0.0))
            weight_regs.append(epoch_losses['weight_reg'])
            sparsity_regs.append(epoch_losses.get('sparsity_loss', 0.0))
            proportion_losses.append(epoch_losses.get('proportion_loss', 0.0))
            
            # Learning rate schedule (based on total loss)
            scheduler.step(avg_total_loss)

            current_pearson = epoch_losses['pearson_loss']
            current_mse = epoch_losses['mse_loss']
            current_cosine = epoch_losses['cosine_loss']
            current_gene_pearson = epoch_losses.get('gene_pearson_loss', 0.0)
            current_gene_cosine = epoch_losses.get('gene_cosine_loss', 0.0)
            
            if current_pearson < best_pearson:
                best_pearson = current_pearson
            if current_mse < best_mse:
                best_mse = current_mse
            if current_cosine < best_cosine:
                best_cosine = current_cosine
            if current_gene_pearson < best_gene_pearson:
                best_gene_pearson = current_gene_pearson
            if current_gene_cosine < best_gene_cosine:
                best_gene_cosine = current_gene_cosine

            # The exported model must correspond to the selected epoch.  The
            # legacy implementation recorded best losses but evaluated the
            # final state, which made auto-k scores and outputs inconsistent.
            monitor_loss = (
                epoch_losses['heldout_loss']
                if self.loss_fn.heldout_marker_positions.numel() > 0
                else avg_total_loss
            )
            if best_monitor - monitor_loss > min_delta or best_state is None:
                best_monitor = monitor_loss
                best_epoch = epoch + 1
                selected_metrics = {
                    'pearson': current_pearson,
                    'mse': current_mse,
                    'cosine': current_cosine,
                    'gene_pearson': current_gene_pearson,
                    'gene_cosine': current_gene_cosine,
                }
                if self.restore_best_state:
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.gat_model.state_dict().items()
                    }
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print every N epochs (print_every=9999 in grid search won't trigger)
            if (epoch) % print_every == 0:
                print(f"  Epoch {epoch}/{n_epochs}: Total={avg_total_loss:.4f}, MSE={current_mse:.4f}, Pearson={current_pearson:.4f}, Cosine={current_cosine:.4f}, Gene_Pearson={current_gene_pearson:.4f}, Gene_Cosine={current_gene_cosine:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                if print_every != 9999:  # Don't print during grid search
                    print(f"Early stopping at epoch {epoch+1}/{n_epochs}")
                    # Only print best values if they are not inf
                    if best_pearson != float('inf') and best_mse != float('inf') and best_cosine != float('inf'):
                        print(f"  Best: Pearson={best_pearson:.4f}, MSE={best_mse:.4f}, Cosine={best_cosine:.4f}")
                break

        if self.restore_best_state and best_state is not None:
            self.gat_model.load_state_dict(best_state)
        if selected_metrics is not None:
            best_pearson = selected_metrics['pearson']
            best_mse = selected_metrics['mse']
            best_cosine = selected_metrics['cosine']
            best_gene_pearson = selected_metrics['gene_pearson']
            best_gene_cosine = selected_metrics['gene_cosine']
        
        # Plot training curves (only if output_dir exists)
        if self.output_dir:
            self.plot_training_curves(train_losses, pearson_losses, mse_losses,
                                     cos_losses, gene_pearson_losses, gene_cosine_losses,
                                     weight_regs, sparsity_regs, diversity_losses, hetero_losses, 
                                     proportion_losses, sample_name)
        
        # Note: Best model already saved during training loop as best_gat_model.pth
        # No need to save final_gat_model.pth again to avoid duplication
        
        # Evaluate and visualize results (use normalized data for embedding)
        eval_outputs = self.evaluate_and_visualize(
            st_data_normalized, 
            self.st_adata, 
            spatial_tensor, 
            sample_name,
            knn_cell_indices=self.knn_cell_indices,
            sc_cell_embeddings=self.sc_cell_embeddings
        )
        
        return {
            'best_pearson': best_pearson,
            'best_mse': best_mse,
            'best_cosine': best_cosine,
            'best_gene_pearson': best_gene_pearson,
            'best_gene_cosine': best_gene_cosine,
            'train_losses': train_losses,
            'sample_name': sample_name,
            'best_epoch': best_epoch,
            'best_total_loss': best_monitor,
            **(eval_outputs or {})
        }
    
    def evaluate_and_visualize(self, 
                             st_data: np.ndarray,
                             st_adata,
                             spatial_coords: torch.Tensor,
                             sample_name: str,
                             knn_cell_indices=None,
                             sc_cell_embeddings=None):
        """Evaluate model and visualize results, generate deconvolution matrices
        
        Note: No longer needs cells_per_spot parameter.
        Uses spot_total_counts stored during build_gat_model.
        
        Args:
            knn_cell_indices: [n_spots, n_cell_types, k] 预计算的k-nearest cell索引（动态模式）
            sc_cell_embeddings: [n_cells, embedding_dim] 单细胞embeddings（动态模式）
        """
        self.gat_model.eval()
        
        st_tensor = torch.FloatTensor(st_data).to(self.device)
        full_expr_file = None
        cosine_csv = None
        
        with torch.no_grad():
            # Compute spot embeddings
            mu, log_var = self.vae_encoder(st_tensor)
            spot_embeddings = mu
            
            # GAT forward pass
            scale_basis = getattr(self, "scale_basis", "all")
            normalize_attention = (scale_basis != "none")
            gat_outputs = self.gat_model(
                spot_embeddings=spot_embeddings,
                spatial_coords=spatial_coords,
                celltype_prototypes=self.celltype_prototypes,
                use_embedding_knn=self.use_embedding_knn,
                normalize_attention=normalize_attention,
                initial_weights=(
                    torch.as_tensor(self.initial_weights, dtype=torch.float32, device=self.device)
                    if self.initial_weights is not None else None
                ),
            )
            
            # Get prediction results
            deconv_weights = gat_outputs['deconv_weights'].detach().cpu().numpy()
            attention_scores = gat_outputs['attention_scores'].detach().cpu().numpy()
        
        original_nonzero = np.count_nonzero(deconv_weights)
        deconv_weights[deconv_weights < self.weight_threshold] = 0
        new_nonzero = np.count_nonzero(deconv_weights)
 
        # Renormalize weights to sum to 1 per spot (after thresholding)
        row_sums = deconv_weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        deconv_weights = deconv_weights / row_sums


        n_spots = deconv_weights.shape[0]

        # Get spot barcodes
        spot_barcodes = list(st_adata.obs.index)
        
        # Full gene expression matrix (use count version with spot_total_counts)
        reconstructed_full_expr = None
        if self.celltype_expressions_full is not None and all(expr is not None for expr in self.celltype_expressions_full):

            celltype_expr_full = np.array(self.celltype_expressions_full)  # [n_clusters, n_all_genes]
            
            # Check if using dynamic cluster representation for reconstruction
            if (self.gat_model.use_dynamic_cluster_repr and 
                knn_cell_indices is not None and 
                hasattr(self.gat_model, 'sc_cell_expressions') and
                self.gat_model.sc_cell_expressions is not None):
                
                # ========== Dynamic cluster reconstruction ==========
                # 1. 转换为tensor
                deconv_weights_tensor = torch.FloatTensor(deconv_weights).to(self.device)  # [n_spots, n_clusters]
                knn_indices_tensor = torch.LongTensor(knn_cell_indices).to(self.device)  # [n_spots, n_clusters, k]
                spot_embeddings_tensor = spot_embeddings  # 已经在GPU上
                
                # 2. 计算动态cluster权重（每个spot的每个cluster的k个cell的权重）
                # 使用均匀权重
                n_spots, n_cell_types, k = knn_indices_tensor.shape
                dynamic_weights = torch.ones(n_spots, n_cell_types, k, device=self.device) / k
                
                # 3. 处理padding（将-1索引对应的权重置零）
                padding_mask = (knn_indices_tensor == -1)
                dynamic_weights = dynamic_weights.masked_fill(padding_mask, 0.0)
                # 重新归一化
                weight_sums = dynamic_weights.sum(dim=-1, keepdim=True)
                dynamic_weights = dynamic_weights / (weight_sums + 1e-8)
                
                # 4. 使用动态权重计算混合表达（全基因）
                # self.gat_model.sc_cell_expressions: [n_cells, n_all_genes]（全基因原始count）
                # 直接使用compute_dynamic_mixed_expression进行全基因重建
                if hasattr(self.loss_fn, 'compute_dynamic_mixed_expression'):
                    mixed_expr_full_tensor = self.loss_fn.compute_dynamic_mixed_expression(
                        attention_weights=deconv_weights_tensor,
                        dynamic_cluster_weights=dynamic_weights,
                        dynamic_cluster_indices=knn_indices_tensor,
                        sc_cell_expressions=self.gat_model.sc_cell_expressions  # [n_cells, n_all_genes]
                    )  # [n_spots, n_all_genes]
                    mixed_expr_full = mixed_expr_full_tensor.detach().cpu().numpy()
                else:
                    # If loss_fn doesn't have this method, fall back to static method
                    mixed_expr_full = np.dot(deconv_weights, celltype_expr_full)  # [n_spots, n_all_genes]
            else:
                # ========== Static cluster reconstruction (original logic) ==========
                mixed_expr_full = np.dot(deconv_weights, celltype_expr_full)  # [n_spots, n_all_genes]
            
            scale_basis = getattr(self, "scale_basis", "all")
            if scale_basis == "none":
                reconstructed_full_expr = mixed_expr_full
                scale = np.ones((n_spots, 1))
            elif scale_basis == "fixed_10":
                # Fixed scale factor 10: proportion × 10 = cell count
                reconstructed_full_expr = mixed_expr_full * 10.0
                scale = np.full((n_spots, 1), 10.0)
            else:
                # Use spot_total_counts for scaling (consistent with deconv_model.py)
                if self.spot_total_counts is None:
                    raise ValueError("spot_total_counts not set; cannot compute scale factor.")
                spot_counts = self.spot_total_counts[:len(spot_barcodes)]
                
                # Get gene subset (for computing scale)
                if scale_basis == "all":
                    # Use all genes
                    basis_indices = list(range(len(self.all_genes)))
                elif scale_basis == "hvg" and self.hvg_genes_union is not None:
                    # Use HVG intersection
                    basis_indices = [i for i, g in enumerate(self.all_genes) if g in self.hvg_genes_union]
                else:
                    # Default: use marker subset
                    basis_indices = [i for i, g in enumerate(self.all_genes) if g in self.genes]
                
                # Extract subset for computing scale
                mixed_basis = mixed_expr_full[:, basis_indices]  # [n_spots, n_basis_genes]
                
                # Get true observations (for OLS)
                if scale_basis == "all":
                    # Ensure obs_basis_raw aligns perfectly with mixed_basis genes
                    # mixed_basis is based on self.all_genes
                    # obs_basis_raw must also be based on self.all_genes
                    
                    # 1. Find which genes in self.all_genes exist in st_adata
                    valid_gene_indices = [i for i, g in enumerate(self.all_genes) if g in st_adata.var_names]
                    valid_genes = [self.all_genes[i] for i in valid_gene_indices]
                    
                    # 2. Extract genes that exist in ST data
                    st_subset = st_adata[:, valid_genes]
                    obs_basis_subset = st_subset.X.toarray() if hasattr(st_subset.X, 'toarray') else st_subset.X
                    
                    # 3. Extract corresponding columns in mixed_basis
                    mixed_basis_subset = mixed_basis[:, valid_gene_indices]
                    
                    # 4. Use aligned subset for computation
                    obs_basis_raw = obs_basis_subset
                    mixed_basis = mixed_basis_subset
                    
                elif scale_basis == "hvg" and self.hvg_genes_union is not None:
                    # Similar processing for HVG
                    valid_hvg_indices = [i for i, g in enumerate(self.hvg_genes_union) if g in st_adata.var_names]
                    valid_hvgs = [self.hvg_genes_union[i] for i in valid_hvg_indices]
                    
                    # Extract ST
                    st_subset = st_adata[:, valid_hvgs]
                    obs_basis_raw = st_subset.X.toarray() if hasattr(st_subset.X, 'toarray') else st_subset.X
                    
                    # Extract mixed (note mixed_basis is already hvg subset, needs alignment again)
                    # mixed_basis columns now correspond to self.hvg_genes_union
                    mixed_basis = mixed_basis[:, valid_hvg_indices]
                    
                else:
                    # Marker mode usually has all genes, but add check for safety
                    valid_marker_indices = [i for i, g in enumerate(self.genes) if g in st_adata.var_names]
                    valid_markers = [self.genes[i] for i in valid_marker_indices]
                    
                    st_subset = st_adata[:, valid_markers]
                    obs_basis_raw = st_subset.X.toarray() if hasattr(st_subset.X, 'toarray') else st_subset.X
                    
                    # mixed_basis corresponds to self.genes
                    mixed_basis = mixed_basis[:, valid_marker_indices]
                
                # ========== Choose scaling method ==========
                if self.use_ols_scaling:
                    # OLS (Ordinary Least Squares) scaling: s = (m·y) / (m·m)
                    print("Using OLS (Ordinary Least Squares) scaling")
                    eps = 1e-8
                    numer = (mixed_basis * obs_basis_raw).sum(axis=1)  # [n_spots]
                    denom = (mixed_basis * mixed_basis).sum(axis=1) + eps  # [n_spots]
                    scale = numer / denom  # [n_spots]
                    
                    # Numerical robustness: clip to reasonable range
                    scale_min, scale_max = 0.01, 100.0
                    n_clipped_low = (scale < scale_min).sum()
                    n_clipped_high = (scale > scale_max).sum()
                    scale = np.clip(scale, scale_min, scale_max)
                    
                    if n_clipped_low > 0 or n_clipped_high > 0:
                        print(f"Warning: OLS scale clipped: {n_clipped_low} spots < {scale_min}, {n_clipped_high} spots > {scale_max}")
                    
                    scale = scale[:, np.newaxis]  # [n_spots, 1]
                else:
                    # Original sum-based scaling
                    print("Using sum-based scaling")
                    mixed_basis_totals = mixed_basis.sum(axis=1, keepdims=True)  # [n_spots, 1]
                    scale = spot_counts[:, np.newaxis] / (mixed_basis_totals + 1e-8)
                
                # Output scale statistics (for diagnosis)
                scale_vals = scale.ravel()
                print(f"Scale statistics: mean={scale_vals.mean():.4f}, median={np.median(scale_vals):.4f}, "
                           f"min={scale_vals.min():.4f}, max={scale_vals.max():.4f}, "
                           f"Q25={np.percentile(scale_vals, 25):.4f}, Q75={np.percentile(scale_vals, 75):.4f}")
                
                # Apply library_size factor (after scale)
                reconstructed_full_expr = mixed_expr_full * scale * self.library_size
                if self.library_size != 1.0:
                    print(f"Library size factor: {self.library_size}")
            
            # Get full gene names
            if self.all_genes is not None:
                all_gene_names = self.all_genes
            else:
                all_gene_names = [f"Gene_{i}" for i in range(celltype_expr_full.shape[1])]
            
            # Only save when enabled
            if self.save_reconstructed_genes and self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
                
                # 1. Save spot-level reconstructed expression (original feature)
                full_expr_df = pd.DataFrame(
                    reconstructed_full_expr,
                    columns=all_gene_names,
                    index=spot_barcodes
                )
                full_expr_file = f"{self.output_dir}/{sample_name}_reconstructed.csv"
                full_expr_df.to_csv(full_expr_file)
                
                # 2. New: Save spot-cell level dynamic expression (for Stage 3 cellcom)
                if (self.gat_model.use_dynamic_cluster_repr and 
                    knn_cell_indices is not None and 
                    hasattr(self.gat_model, 'sc_cell_expressions') and
                    self.gat_model.sc_cell_expressions is not None):
                    
                    print("Computing spot-cell level dynamic expression...")
                    spot_cell_expr_dict = {}
                    
                    # deconv_weights: [n_spots, n_clusters]
                    # dynamic_weights: [n_spots, n_clusters, k]
                    # knn_indices_tensor: [n_spots, n_clusters, k]
                    n_spots, n_clusters, k = dynamic_weights.shape
                    
                    # cluster到celltype的映射
                    cluster_to_celltype_map = {}
                    if self.cluster_to_celltype:
                        cluster_to_celltype_map = {str(k): str(v) for k, v in self.cluster_to_celltype.items()}
                    cluster_list = list(self.label_encoder.classes_)
                    
                    for spot_idx in range(n_spots):
                        spot_name = spot_barcodes[spot_idx]
                        
                        for cluster_idx, cluster_id in enumerate(cluster_list):
                            # Get celltype name
                            celltype_name = cluster_to_celltype_map.get(str(cluster_id), f"Cluster_{cluster_id}")
                            
                            # This cluster's deconv weight (proportion) in this spot
                            cluster_proportion = deconv_weights[spot_idx, cluster_idx]
                            
                            if cluster_proportion < 1e-6:
                                continue
                            
                            # Get indices and weights of k nearest cells for this cluster
                            cell_indices = knn_indices_tensor[spot_idx, cluster_idx]  # [k]
                            cell_weights = dynamic_weights[spot_idx, cluster_idx]  # [k]
                            
                            # Filter out padding (index -1)
                            valid_mask = (cell_indices != -1).cpu()
                            cell_indices = cell_indices[valid_mask]
                            cell_weights = cell_weights[valid_mask]
                            
                            if len(cell_indices) == 0:
                                continue
                            
                            # Re-normalize weights
                            cell_weights = cell_weights / (cell_weights.sum() + 1e-8)
                            
                            # Compute spot-cell expression: weighted average of k cells' expression
                            # sc_cell_expressions: [n_cells, n_all_genes]
                            cell_exprs = self.gat_model.sc_cell_expressions[cell_indices]  # [k, n_genes]
                            weighted_expr = (cell_exprs * cell_weights.unsqueeze(1)).sum(dim=0)  # [n_genes]
                            
                            # Scaling: use globally computed scale factor and library_size factor, consistent with reconstructed_full_expr
                            # scale: [n_spots, 1]
                            spot_scale = scale[spot_idx, 0]
                            scaled_expr = weighted_expr.cpu().numpy() * cluster_proportion * spot_scale * self.library_size
                            
                            # Accumulate to this spot-celltype
                            key = f"{spot_name}_{celltype_name}"
                            if key in spot_cell_expr_dict:
                                spot_cell_expr_dict[key] += scaled_expr
                            else:
                                spot_cell_expr_dict[key] = scaled_expr.copy()
                    
                    # Convert to DataFrame
                    spot_cell_expr_df = pd.DataFrame.from_dict(
                        spot_cell_expr_dict, orient='index', columns=all_gene_names
                    )
                    spot_cell_expr_df.index.name = 'spot_cell'
                    
                    # Filter rows that are all zeros
                    row_sums = spot_cell_expr_df.sum(axis=1)
                    spot_cell_expr_df = spot_cell_expr_df[row_sums > 0]
                    
                    # Save
                    spot_cell_file = f"{self.output_dir}/{sample_name}_spot_cell_expr.csv"
                    spot_cell_expr_df.to_csv(spot_cell_file)
                    print(f"Saved spot-cell dynamic expression: {spot_cell_file}, shape={spot_cell_expr_df.shape}")
        else:
            pass

        # 3. Cell type composition matrix (spot × cluster/celltype)

        cluster_list = list(self.label_encoder.classes_)

        # Get cluster-to-celltype mapping only from Stage 1 checkpoint/npz
        checkpoint_cluster_to_celltype = {}
        if self.cluster_to_celltype:
            checkpoint_cluster_to_celltype = {str(k): str(v) for k, v in self.cluster_to_celltype.items()}
        
        # Map cluster columns to celltype names
        celltype_columns = []
        cluster_columns = []
        for cluster_id in cluster_list:
            cluster_columns.append(str(cluster_id))
            # Use the mapping if available, otherwise use cluster ID as is
            celltype_name = checkpoint_cluster_to_celltype.get(str(cluster_id), f"Cluster_{cluster_id}")
            celltype_columns.append(celltype_name)

        # Create DataFrame with possibly duplicate column names (multiple clusters -> same celltype)
        composition_df = pd.DataFrame(
            deconv_weights,
            columns=celltype_columns,
            index=spot_barcodes
        )

        # If duplicate celltype names exist, merge corresponding columns (sum by column) and log
        dup_names = [name for name in set(celltype_columns) if celltype_columns.count(name) > 1]
        if len(dup_names) > 0:
            composition_by_celltype = composition_df.groupby(by=composition_df.columns, axis=1).sum()
        else:
            composition_by_celltype = composition_df

        # Save aggregated celltype composition (only if output_dir exists)
        composition_file = None
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            composition_file = f"{self.output_dir}/{sample_name}_composition.csv"
            composition_by_celltype.to_csv(composition_file)

        # Skip cluster-level composition to reduce file clutter

        # ============ Compute reconstruction quality (Cosine Similarity) ============
        marker_indices = None
        if reconstructed_full_expr is not None and self.all_genes is not None:
            try:
                marker_indices = [self.all_genes.index(g) for g in self.genes]
            except ValueError:
                marker_indices = None

        if reconstructed_full_expr is not None and marker_indices is not None:
            reconstructed_marker_expr = reconstructed_full_expr[:, marker_indices]

            st_marker_subset = st_adata[:, self.genes].X
            true_expr = st_marker_subset.toarray() if hasattr(st_marker_subset, 'toarray') else st_marker_subset

            # Ensure pure numpy float arrays to avoid object-dtype issues
            reconstructed_marker_expr = np.asarray(reconstructed_marker_expr, dtype=np.float64)
            true_expr = np.asarray(true_expr, dtype=np.float64)

            # Compute cosine similarity per spot (log-normalized space)
            reconstructed_log = np.log1p(reconstructed_marker_expr)
            true_log = np.log1p(true_expr)
            
            cosine_similarities = []
            for i in range(n_spots):
                rec = reconstructed_log[i]
                true = true_log[i]
                
                # Cosine similarity
                cos_sim = np.dot(rec, true) / (np.linalg.norm(rec) * np.linalg.norm(true) + 1e-8)
                cosine_similarities.append(cos_sim)
            
            cosine_similarities = np.array(cosine_similarities)
            
            # Skip saving cosine similarities to reduce file clutter
            # (still available in metrics)

            # Plot reconstruction quality curve (sorted by similarity, only if output_dir exists)
            if self.output_dir:
                self.plot_reconstruction_quality_curve(cosine_similarities, sample_name)
        
        # 缓存到对象并返回，便于上层直接拿到矩阵
        self.last_deconv = composition_by_celltype
        self.last_deconv_path = composition_file
        self.last_reconstructed_expr_path = full_expr_file
        self.last_cosine_path = cosine_csv

        return {
            'composition_df': composition_by_celltype,
            'composition_path': composition_file,
            'reconstructed_expr_path': full_expr_file,
            'cosine_path': cosine_csv,
            'deconv_weights_raw': deconv_weights  # 未合并的 cluster-level 权重 [n_spots, n_clusters]
        }
    
    def plot_reconstruction_quality_curve(self, cosine_similarities, sample_name):
        """Plot reconstruction quality curve (sorted by cosine similarity)
        
        Args:
            cosine_similarities: Array of cosine similarities per spot [n_spots]
            sample_name: Sample name for saving
        """
        n_spots = len(cosine_similarities)
        
        # Sort cosine similarities in ascending order
        sorted_similarities = np.sort(cosine_similarities)
        n_spots = len(sorted_similarities)
        spot_indices = np.arange(1, n_spots + 1)  # 1-based indexing
        
        # Compute statistics
        mean_sim = np.mean(cosine_similarities)
        median_sim = np.median(cosine_similarities)
        q25_sim = np.percentile(cosine_similarities, 25)
        q75_sim = np.percentile(cosine_similarities, 75)
        min_sim = np.min(cosine_similarities)
        max_sim = np.max(cosine_similarities)
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot curve
        ax.plot(spot_indices, sorted_similarities, linewidth=2.5, color='#1f77b4', label='Cosine Similarity')
        
        # Add reference lines
        ax.axhline(mean_sim, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean: {mean_sim:.3f}')
        ax.axhline(median_sim, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Median: {median_sim:.3f}')
        ax.axhline(0.8, color='gray', linestyle=':', linewidth=1, alpha=0.5)  # Reference line at 0.8
        
        # Styling
        ax.set_xlabel('Spot Index (sorted by similarity)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cosine Similarity', fontsize=12, fontweight='bold')
        ax.set_title(f'Reconstruction Quality Distribution - {sample_name}', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, 1.0])
        ax.set_xlim([0, n_spots])
        
        # Add text box with statistics
        textstr = '\n'.join([
            f'Total spots: {n_spots}',
            f'Min: {min_sim:.3f}',
            f'Q25: {q25_sim:.3f}',
            f'Median: {median_sim:.3f}',
            f'Q75: {q75_sim:.3f}',
            f'Max: {max_sim:.3f}',
            f'Mean: {mean_sim:.3f}'
        ])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save figure
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            output_file = f"{self.output_dir}/{sample_name}_reconstruction_quality_curve.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_curves(self, train_losses, pearson_losses, mse_losses,
                           cos_losses, gene_pearson_losses, gene_cosine_losses,
                           weight_regs, sparsity_regs, diversity_losses, hetero_losses, 
                           proportion_losses, sample_name):
        """Plot training curves (single panel, excluding total loss)."""
        epochs = range(1, len(pearson_losses) + 1)
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(epochs, cos_losses, color='#d62728', label='Cosine', linewidth=2.2)
        ax.plot(epochs, pearson_losses, color='#ff7f0e', label='Pearson', linewidth=2.2)
        ax.plot(epochs, mse_losses, color='#2ca02c', label='MSE (log space)', linewidth=2.0)
        ax.plot(epochs, proportion_losses, color='#bcbd22', label='Proportion', linewidth=1.8)

        ax.set_title(f'GAT Deconvolution Losses - {sample_name}', fontsize=15, fontweight='bold')
        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10, ncol=2)

        plt.tight_layout()
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            plt.savefig(f"{self.output_dir}/gat_training_curves_{sample_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_model(self, filepath: str):
        """Save model (weights only)"""
        torch.save({
            'gat_state_dict': self.gat_model.state_dict()
        }, filepath)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Stage 2: GAT Deconvolution for Spatial Transcriptomics')
    # Model and data arguments
    parser.add_argument('--stage1_model_path', type=str, 
                       default="./stage1_results/final_vae.pth",
                       help='Stage 1 VAE model path')
    parser.add_argument('--st_file', type=str, required=True,
                       help='Spatial transcriptomics data file path (.h5ad)')
    parser.add_argument('--output_dir', type=str, 
                       default="./stage2_results",
                       help='Output directory path')

    # GAT model arguments
    parser.add_argument('--gat_hidden_dim', type=int, default=512,
                       help='GAT hidden layer dimension')
    parser.add_argument('--gat_layers', type=int, default=4,
                       help='Number of GAT layers')
    parser.add_argument('--gat_heads', type=int, default=4,
                       help='Number of GAT attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Graph construction arguments
    parser.add_argument('--k_spatial', type=int, default=5,
                       help='Number of spatial neighbors (KNN)')
    parser.add_argument('--k_celltype', type=int, default=30,
                       help='Number of nearest celltypes per spot (KNN)')
    
    # Training arguments
    parser.add_argument('--n_epochs', type=int, default=250,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-3,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    
    # Loss function arguments
    parser.add_argument('--loss_lambda_mse', type=float, default=0.1,
                       help='MSE reconstruction loss weight')
    parser.add_argument('--loss_lambda_pearson', type=float, default=5,
                       help='Pearson correlation loss weight')
    parser.add_argument('--loss_lambda_cosine', type=float, default=5,
                       help='Cosine similarity loss weight')
    parser.add_argument('--loss_lambda_gene_pearson', type=float, default=1,
                       help='Gene-level Pearson loss weight (across spots)')
    parser.add_argument('--loss_lambda_gene_cosine', type=float, default=1,
                       help='Gene-level Cosine loss weight (across spots)')
    parser.add_argument('--loss_lambda_reg', type=float, default=0.1,
                       help='Weight regularization weight')
    parser.add_argument('--loss_lambda_sparse', type=float, default=1,
                       help='Sparsity regularization weight (Shannon entropy)')
    parser.add_argument('--loss_lambda_proportion', type=float, default=0.1,
                       help='Global cell type proportion consistency loss weight (matches SC cluster distribution)')
    # Spot composition argument
    parser.add_argument('--cells_per_spot', type=float, default=10,
                       help='Average number of cells per spot (default: auto-calculate from data, or 10.0 for Visium if auto-calc fails)')
    
    # Weight thresholding argument
    parser.add_argument('--weight_threshold', type=float, default=0.001,
                       help='Weight threshold for sparsification (default 0.01, i.e., 1%)')
    
    # Scaling basis argument
    parser.add_argument('--scale_basis', type=str, default='all',
                       choices=['marker', 'hvg', 'all', 'none'],
                       help='Gene set used to compute per-spot scaling factor for final reconstruction: '
                            'marker (marker genes only), hvg (SC/ST HVG intersection), all (all shared genes), '
                            'or none (no scaling, direct weighted mixture). '
                            'Default: all. Must match the spot_total_counts calculation method.')
    
    # Device argument
    parser.add_argument('--device', type=str, default=None,
                       help='Computing device (cuda/cpu, None for auto-select)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.st_file):
        raise FileNotFoundError(f"ST data file not found: {args.st_file}")
    
    # Extract sample name
    sample_name = os.path.splitext(os.path.basename(args.st_file))[0]
    if sample_name.endswith('_ST'):
        sample_name = sample_name[:-3]
    
    # Initialize trainer
    trainer = GATDeconvolution(
        stage1_model_path=args.stage1_model_path,
        output_dir=args.output_dir,
        device=args.device,
        weight_threshold=args.weight_threshold
    )
    
    # Set graph construction parameters
    trainer.k_spatial = args.k_spatial
    trainer.k_celltype = args.k_celltype
    # Set scaling basis for reconstruction
    trainer.scale_basis = args.scale_basis
    
    # Load VAE Encoder
    trainer.load_vae_encoder()

    if trainer.sc_clusters is None:
        raise ValueError("Stage 1 model missing cluster information! Please retrain with new version.")
    
    print(f"Loaded {len(trainer.label_encoder.classes_)} clusters")
    
    # Check if pretrained cluster data available
    has_prototypes = hasattr(trainer, 'celltype_prototypes') and trainer.celltype_prototypes is not None
    has_expressions = hasattr(trainer, 'celltype_expressions') and trainer.celltype_expressions is not None
    
    if has_prototypes and has_expressions:
        n_clusters = trainer.celltype_prototypes.shape[0]
    else:
        raise ValueError("Stage 1 model missing cluster centers or expressions!")
    
    st_adata = sc.read_h5ad(args.st_file)
    st_adata.var_names_make_unique()  # Handle duplicate gene names
    # Preserve raw counts before normalization/log
    st_raw_all = st_adata.X.toarray() if hasattr(st_adata.X, "toarray") else st_adata.X
    st_proc = st_adata.copy()
    sc.pp.normalize_total(st_proc, target_sum=1e4)
    sc.pp.log1p(st_proc)
    # Check spatial coordinates
    if 'spatial' not in st_adata.obsm:
        print(f"Warning: ST data file {args.st_file} missing spatial coordinates. Falling back to embedding-based KNN for spot graph.")
        spatial_coords = np.zeros((st_adata.n_obs, 2), dtype=float)
        trainer.use_embedding_knn = True
    else:
        spatial_coords = st_adata.obsm['spatial']
        trainer.use_embedding_knn = False
    
    # Extract ST marker genes (raw for loss, log1p norm for embedding)
    st_subset_raw = st_adata[:, trainer.genes].copy()
    st_subset_norm = st_proc[:, trainer.genes].copy()

    st_X_raw = st_subset_raw.X.toarray() if hasattr(st_subset_raw.X, 'toarray') else st_subset_raw.X
    st_X_embed = st_subset_norm.X.toarray() if hasattr(st_subset_norm.X, 'toarray') else st_subset_norm.X

    # Select counts based on scale_basis
    if args.scale_basis == 'none':
        spot_total_counts = None
    elif args.scale_basis == 'all':
        spot_total_counts_all = np.asarray(st_raw_all.sum(axis=1)).ravel()
        spot_total_counts = spot_total_counts_all
    elif args.scale_basis == 'hvg':
        spot_total_counts_hvg = None
        if hasattr(trainer, 'hvg_genes_union') and trainer.hvg_genes_union is not None:
            hvg_in_st = [g for g in trainer.hvg_genes_union if g in st_adata.var_names]
            if len(hvg_in_st) > 0:
                st_hvg = st_adata[:, hvg_in_st]
                st_hvg_raw = st_hvg.X.toarray() if hasattr(st_hvg.X, 'toarray') else st_hvg.X
                spot_total_counts_hvg = np.asarray(st_hvg_raw.sum(axis=1)).ravel()
        spot_total_counts = spot_total_counts_hvg
    else:
        spot_total_counts_marker = np.asarray(st_X_raw.sum(axis=1)).ravel()
        trainer.scale_basis = 'marker'
        spot_total_counts = spot_total_counts_marker
     
    trainer.build_gat_model(
        n_cell_types=n_clusters,
        gat_hidden_dim=args.gat_hidden_dim,
        gat_layers=args.gat_layers,
        gat_heads=args.gat_heads,
        dropout=args.dropout,
        loss_lambda_pearson=args.loss_lambda_pearson,
        loss_lambda_mse=args.loss_lambda_mse,
        loss_lambda_cosine=args.loss_lambda_cosine,
        loss_lambda_gene_pearson=args.loss_lambda_gene_pearson,
        loss_lambda_gene_cosine=args.loss_lambda_gene_cosine,
        loss_lambda_reg=args.loss_lambda_reg,
        loss_lambda_sparse=args.loss_lambda_sparse,
        loss_lambda_proportion=args.loss_lambda_proportion,
        spot_total_counts=spot_total_counts  # Pass spot total counts instead of cells_per_spot
    )
    
    trainer.train_gat_deconvolution(
        st_data_normalized=st_X_embed,  # For VAE embedding (now raw)
        st_data_raw=st_X_raw,           # For loss calculation
        spatial_coords=spatial_coords,
        sample_name=sample_name,
        st_adata=st_adata,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
