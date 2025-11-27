# """
# VAE Training and Evaluation Utilities for Stage 1
# """

# import torch
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from deconv_model import vae_loss_function, zinb_loss_function, compute_mmd


# class SimpleDataset(Dataset):
#     """Simple dataset for VAE training"""
#     def __init__(self, X, modality):
#         self.X = torch.FloatTensor(X)
#         self.modality = torch.LongTensor(modality)
    
#     def __len__(self):
#         return len(self.X)
    
#     def __getitem__(self, idx):
#         return self.X[idx], self.modality[idx]


# def train_vae_epoch(vae, train_loader, optimizer, device, loss_type='mse', beta=1.0, lambda_mmd=0.0):
#     """Train VAE for one epoch
    
#     Args:
#         vae: VAE model (can be VAE or DualDecoderVAE)
#         train_loader: Training data loader
#         optimizer: Optimizer
#         device: Device (cuda/cpu)
#         loss_type: 'mse' or 'zinb'
#         beta: KL divergence weight
#         lambda_mmd: MMD loss weight
    
#     Returns:
#         avg_loss, avg_recon, avg_kl, avg_mmd
#     """
#     vae.train()
#     epoch_loss = 0.0
#     epoch_recon = 0.0
#     epoch_kl = 0.0
#     epoch_mmd = 0.0
    
#     # 检查是否是双解码器架构
#     is_dual_decoder = hasattr(vae, 'decoder_sc') and hasattr(vae, 'decoder_st')
    
#     for batch_data, batch_modality in train_loader:
#         batch_data = batch_data.to(device)
#         batch_modality = batch_modality.to(device)
        
#         optimizer.zero_grad()
        
#         # VAE forward pass
#         if is_dual_decoder:
#             # 双解码器: 需要传入modality参数
#             if loss_type == 'zinb':
#                 mean, disp, pi, mu, log_var, z = vae(batch_data, batch_modality)
#                 total_loss, recon_loss, kl_div = zinb_loss_function(
#                     mean, disp, pi, batch_data, mu, log_var, beta=beta
#                 )
#             else:
#                 recon_data, mu, log_var, z = vae(batch_data, batch_modality)
#                 total_loss, recon_loss, kl_div = vae_loss_function(
#                     recon_data, batch_data, mu, log_var, beta=beta
#                 )
#         else:
#             # 单解码器: 不需要modality参数
#             if loss_type == 'zinb':
#                 mean, disp, pi, mu, log_var, z = vae(batch_data)
#                 total_loss, recon_loss, kl_div = zinb_loss_function(
#                     mean, disp, pi, batch_data, mu, log_var, beta=beta
#                 )
#             else:
#                 recon_data, mu, log_var, z = vae(batch_data)
#                 total_loss, recon_loss, kl_div = vae_loss_function(
#                     recon_data, batch_data, mu, log_var, beta=beta
#                 )
        
#         # Compute MMD loss for modality alignment
#         mmd_loss = torch.tensor(0.0, device=device)
#         if lambda_mmd > 0:
#             # Separate SC and ST embeddings in this batch
#             sc_mask = batch_modality == 0
#             st_mask = batch_modality == 1
            
#             # Only compute MMD if both modalities present in batch
#             if sc_mask.sum() > 0 and st_mask.sum() > 0:
#                 sc_embeddings = z[sc_mask]
#                 st_embeddings = z[st_mask]
#                 mmd_loss = compute_mmd(sc_embeddings, st_embeddings, kernel='rbf')
        
#         # Total loss with MMD
#         total_loss = total_loss + lambda_mmd * mmd_loss
        
#         # Normalize loss
#         total_loss = total_loss / len(batch_data)
#         recon_loss = recon_loss / len(batch_data)
#         kl_div = kl_div / len(batch_data)
        
#         total_loss.backward()
#         optimizer.step()
        
#         epoch_loss += total_loss.item()
#         epoch_recon += recon_loss.item()
#         epoch_kl += kl_div.item()
#         epoch_mmd += mmd_loss.item() if lambda_mmd > 0 else 0.0
    
#     avg_loss = epoch_loss / len(train_loader)
#     avg_recon = epoch_recon / len(train_loader)
#     avg_kl = epoch_kl / len(train_loader)
#     avg_mmd = epoch_mmd / len(train_loader)
    
#     return avg_loss, avg_recon, avg_kl, avg_mmd


# def evaluate_vae(vae, test_loader, device, loss_type='mse', beta=1.0):
#     """Evaluate VAE
    
#     Args:
#         vae: VAE model (can be VAE or DualDecoderVAE)
#         test_loader: Test data loader
#         device: Device (cuda/cpu)
#         loss_type: 'mse' or 'zinb'
#         beta: KL divergence weight
    
#     Returns:
#         test_loss
#     """
#     vae.eval()
#     total_loss = 0.0
    
#     # 检查是否是双解码器架构
#     is_dual_decoder = hasattr(vae, 'decoder_sc') and hasattr(vae, 'decoder_st')
    
#     with torch.no_grad():
#         for batch_data, batch_modality in test_loader:
#             batch_data = batch_data.to(device)
#             batch_modality = batch_modality.to(device)
            
#             if is_dual_decoder:
#                 # 双解码器: 需要传入modality参数
#                 if loss_type == 'zinb':
#                     mean, disp, pi, mu, log_var, z = vae(batch_data, batch_modality)
#                     loss, _, _ = zinb_loss_function(mean, disp, pi, batch_data, mu, log_var, beta)
#                 else:
#                     recon_data, mu, log_var, z = vae(batch_data, batch_modality)
#                     loss, _, _ = vae_loss_function(recon_data, batch_data, mu, log_var, beta)
#             else:
#                 # 单解码器: 不需要modality参数
#                 if loss_type == 'zinb':
#                     mean, disp, pi, mu, log_var, z = vae(batch_data)
#                     loss, _, _ = zinb_loss_function(mean, disp, pi, batch_data, mu, log_var, beta)
#                 else:
#                     recon_data, mu, log_var, z = vae(batch_data)
#                     loss, _, _ = vae_loss_function(recon_data, batch_data, mu, log_var, beta)
                
#             total_loss += loss.item() / len(batch_data)
    
#     return total_loss / len(test_loader)


# def train_vae(vae, train_X, test_X, train_modality, test_modality, device,
#               batch_size=256, n_epochs=100, lr=1e-3, beta=1.0, loss_type='mse', 
#               lambda_mmd=1.0, output_dir="./stage1_results"):
#     """Train VAE with optional MMD loss for modality alignment
    
#     Args:
#         vae: VAE model
#         train_X: Training data
#         test_X: Test data
#         train_modality: Training modality labels
#         test_modality: Test modality labels
#         device: Device (cuda/cpu)
#         batch_size: Batch size
#         n_epochs: Number of epochs
#         lr: Learning rate
#         beta: KL divergence weight
#         loss_type: 'mse' or 'zinb'
#         lambda_mmd: MMD loss weight
#         output_dir: Output directory for saving plots
    
#     Returns:
#         best_loss
#     """
#     print("="*60)
#     print("Starting VAE training...")
#     print(f"   Train data: {train_X.shape} (SC: {sum(train_modality==0)}, ST: {sum(train_modality==1)})")
#     print(f"   Test data: {test_X.shape} (SC: {sum(test_modality==0)}, ST: {sum(test_modality==1)})")
#     print(f"   Loss type: {loss_type.upper()}")
#     print(f"   MMD weight: {lambda_mmd}")

#     # Data loader
#     train_dataset = SimpleDataset(train_X, train_modality)
#     test_dataset = SimpleDataset(test_X, test_modality)
    
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
#     # Optimizer
#     optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', patience=10, factor=0.5
#     )
    
#     # Training history
#     train_losses = []
#     test_losses = []
#     recon_losses = []
#     kl_losses = []
#     mmd_losses = []
    
#     best_loss = float('inf')
#     patience_counter = 0
#     patience = 15
    
#     pbar = tqdm(range(n_epochs), desc="VAE Training", unit="epoch")
#     for epoch in pbar:
#         # Training
#         avg_loss, avg_recon, avg_kl, avg_mmd = train_vae_epoch(
#             vae, train_loader, optimizer, device, loss_type, beta, lambda_mmd
#         )
        
#         train_losses.append(avg_loss)
#         recon_losses.append(avg_recon)
#         kl_losses.append(avg_kl)
#         mmd_losses.append(avg_mmd)
        
#         # Evaluate
#         if (epoch + 1) % 5 == 0:
#             test_loss = evaluate_vae(vae, test_loader, device, loss_type, beta)
#             test_losses.append(test_loss)
            
#             scheduler.step(test_loss)
            
#             # Update progress bar
#             if lambda_mmd > 0:
#                 pbar.set_postfix({'Train': f'{avg_loss:.4f}', 'Recon': f'{avg_recon:.4f}', 
#                                  'KL': f'{avg_kl:.4f}', 'MMD': f'{avg_mmd:.4f}', 'Test': f'{test_loss:.4f}'})
#             else:
#                 pbar.set_postfix({'Train': f'{avg_loss:.4f}', 'Recon': f'{avg_recon:.4f}', 
#                                  'KL': f'{avg_kl:.4f}', 'Test': f'{test_loss:.4f}'})
            
#             # Save best model
#             if test_loss < best_loss:
#                 best_loss = test_loss
#                 patience_counter = 0
#             else:
#                 patience_counter += 1
                
#             # Early stopping
#             if patience_counter >= patience:
#                 pbar.close()
#                 break
    
#     # Plot training curves
#     plot_vae_training_curves(train_losses, test_losses, recon_losses, kl_losses, mmd_losses, output_dir)
    
#     return best_loss


# def plot_vae_training_curves(train_losses, test_losses, recon_losses, kl_losses, mmd_losses=None, output_dir="./stage1_results"):
#     """Plot VAE training curves"""
#     # Determine if we need to plot MMD
#     has_mmd = mmd_losses is not None and len(mmd_losses) > 0 and max(mmd_losses) > 0
    
#     if has_mmd:
#         fig, axes = plt.subplots(2, 3, figsize=(22, 10))
#         ((ax1, ax2, ax3), (ax4, ax5, ax6)) = axes
#     else:
#         fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
#     # Total loss
#     ax1.plot(train_losses, label='Train')
#     if len(test_losses) > 0:
#         test_epochs = range(5, len(train_losses)+1, 5)
#         if len(test_epochs) == len(test_losses):
#             ax1.plot(test_epochs, test_losses, label='Test')
#     ax1.set_title('Total Loss')
#     ax1.set_xlabel('Epochs')
#     ax1.set_ylabel('Loss')
#     ax1.legend()
#     ax1.grid(True)
    
#     # Reconstruction loss
#     ax2.plot(recon_losses, 'g-')
#     ax2.set_title('Reconstruction Loss')
#     ax2.set_xlabel('Epochs')
#     ax2.set_ylabel('Loss')
#     ax2.grid(True)
    
#     # KL divergence
#     ax3.plot(kl_losses, 'r-')
#     ax3.set_title('KL Divergence')
#     ax3.set_xlabel('Epochs')
#     ax3.set_ylabel('KL Div')
#     ax3.grid(True)
    
#     # Loss components comparison
#     ax4.plot(recon_losses, label='Reconstruction', color='green')
#     ax4.plot(kl_losses, label='KL Divergence', color='red')
#     if has_mmd:
#         ax4.plot(mmd_losses, label='MMD', color='purple')
#     ax4.set_title('Loss Components')
#     ax4.set_xlabel('Epochs')
#     ax4.set_ylabel('Loss')
#     ax4.legend()
#     ax4.grid(True)
    
#     if has_mmd:
#         # MMD loss
#         ax5.plot(mmd_losses, 'purple')
#         ax5.set_title('MMD Loss (Modality Alignment)')
#         ax5.set_xlabel('Epochs')
#         ax5.set_ylabel('MMD')
#         ax5.grid(True)
        
#         # All components normalized
#         ax6.plot(np.array(recon_losses) / (max(recon_losses) + 1e-8), label='Recon (norm)', color='green')
#         ax6.plot(np.array(kl_losses) / (max(kl_losses) + 1e-8), label='KL (norm)', color='red')
#         ax6.plot(np.array(mmd_losses) / (max(mmd_losses) + 1e-8), label='MMD (norm)', color='purple')
#         ax6.set_title('Normalized Loss Components')
#         ax6.set_xlabel('Epochs')
#         ax6.set_ylabel('Normalized Loss')
#         ax6.legend()
#         ax6.grid(True)
    
#     plt.tight_layout()
#     plt.savefig(f"{output_dir}/vae_training_curves.png", dpi=300, bbox_inches='tight')
#     plt.close()

# """
# VAE Model I/O (Save/Load) Utilities for Stage 1
# """

# import torch
# import os
# from model import VAE


# def save_vae_checkpoint(vae, label_encoder, marker_genes, genes, all_genes,
#                        sc_clusters, resolution, filepath, avg_cell_counts=None):
#     """Save VAE model checkpoint (weights + basic metadata)"""
#     print("="*60)
#     print(f"Saving model to: {filepath}")
#     if avg_cell_counts is not None:
#         print(f"   Average cell counts: {avg_cell_counts:.1f} (for Stage 2 scale factor)")
    
#     torch.save({
#         'vae_state_dict': vae.state_dict(),
#         'label_encoder': label_encoder,
#         'marker_genes': marker_genes,
#         'genes': genes,
#         'input_dim': len(genes),
#         'latent_dim': vae.latent_dim,
#         'output_type': vae.output_type,
#         'sc_clusters': sc_clusters,
#         'resolution': resolution,
#         'all_genes': all_genes,
#         'avg_cell_counts': avg_cell_counts
#     }, filepath)
    
#     print(f"   Saved successfully!")


# def load_vae_for_inference(filepath, device):
#     """Load VAE model for inference (basic loading)
    
#     Args:
#         filepath: Path to checkpoint
#         device: Device (cuda/cpu)
    
#     Returns:
#         vae, label_encoder, marker_genes, genes
#     """
#     checkpoint = torch.load(filepath, map_location=device)
    
#     input_dim = checkpoint['input_dim']
#     latent_dim = checkpoint['latent_dim']
#     output_type = checkpoint.get('output_type', 'mse')
    
#     vae = VAE(input_dim=input_dim, latent_dim=latent_dim, output_type=output_type).to(device)
#     vae.load_state_dict(checkpoint['vae_state_dict'])
    
#     label_encoder = checkpoint['label_encoder']
#     marker_genes = checkpoint['marker_genes']
#     genes = checkpoint['genes']
    
#     print(f"VAE model loaded: {filepath}")
    
#     return vae, label_encoder, marker_genes, genes


# def load_vae_pretrained(filepath, device):
#     """Load pretrained VAE weights for continued training
    
#     Args:
#         filepath: Path to checkpoint
#         device: Device (cuda/cpu)
    
#     Returns:
#         Tuple of (vae, components_dict, output_type, latent_dim)
#         where components_dict contains all other checkpoint components
#     """
#     print("="*60)
#     print(f"Loading pretrained weights from: {filepath}")
    
#     if not os.path.exists(filepath):
#         raise FileNotFoundError(f"Pretrained model not found: {filepath}")
    
#     checkpoint = torch.load(filepath, map_location=device)
    
#     # Load model architecture info
#     input_dim = checkpoint['input_dim']
#     latent_dim = checkpoint['latent_dim']
#     output_type = checkpoint.get('output_type', 'mse')
    
#     print(f"   Input dim: {input_dim}")
#     print(f"   Latent dim: {latent_dim}")
#     print(f"   Output type: {output_type}")
    
#     # Build VAE model with same architecture
#     vae = VAE(input_dim=input_dim, latent_dim=latent_dim, output_type=output_type).to(device)
#     vae.load_state_dict(checkpoint['vae_state_dict'])
    
#     # Extract other components
#     components = {
#         'label_encoder': checkpoint.get('label_encoder', None),
#         'marker_genes': checkpoint.get('marker_genes', None),
#         'genes': checkpoint.get('genes', None),
#         'all_genes': checkpoint.get('all_genes', None),
#         'sc_clusters': checkpoint.get('sc_clusters', None),
#         'resolution': checkpoint.get('resolution', 0.5)
#     }
    
#     print("   Pretrained weights loaded successfully!")
#     print("="*60)
    
#     return vae, components, output_type, latent_dim

# """
# VAE Visualization Utilities (UMAP, etc.) for Stage 1
# """

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import umap


# def plot_modality_alignment_umap(vae, train_X, train_modality, device, y_train=None, output_dir="./stage1_results"):
#     """
#     Plot UMAP visualization of SC and ST modality alignment
    
#     Args:
#         vae: Trained VAE model
#         train_X: Training data (combined SC + ST)
#         train_modality: Modality labels (0=SC, 1=ST)
#         device: Device (cuda/cpu)
#         y_train: Optional cluster labels for SC samples
#         output_dir: Output directory
#     """
#     print("="*60)
#     print("Generating UMAP visualization for modality alignment...")
    
#     # Get embeddings from trained VAE
#     vae.eval()
#     with torch.no_grad():
#         batch_size = 1000
#         all_embeddings = []
        
#         for i in range(0, len(train_X), batch_size):
#             batch_data = train_X[i:i+batch_size]
#             batch_tensor = torch.FloatTensor(batch_data).to(device)
#             mu, log_var = vae.encoder(batch_tensor)
#             all_embeddings.append(mu.cpu().numpy())
        
#         embeddings = np.vstack(all_embeddings)
    
#     print(f"   Computing UMAP on {embeddings.shape[0]} samples with {embeddings.shape[1]} dims...")
    
#     # Compute UMAP
#     reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric='euclidean', random_state=42)
#     umap_coords = reducer.fit_transform(embeddings)
    
#     # Create figure with subplots
#     fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
#     # Plot 1: Color by modality (SC vs ST)
#     ax1 = axes[0]
#     sc_mask = train_modality == 0
#     st_mask = train_modality == 1
    
#     ax1.scatter(umap_coords[sc_mask, 0], umap_coords[sc_mask, 1], 
#                c='#1f77b4', s=20, alpha=0.6, label=f'SC (n={sum(sc_mask)})', edgecolors='none')
#     ax1.scatter(umap_coords[st_mask, 0], umap_coords[st_mask, 1], 
#                c='#ff7f0e', s=20, alpha=0.6, label=f'ST (n={sum(st_mask)})', edgecolors='none')
    
#     ax1.set_title('UMAP: SC vs ST Modality Alignment', fontsize=14, fontweight='bold')
#     ax1.set_xlabel('UMAP 1', fontsize=12)
#     ax1.set_ylabel('UMAP 2', fontsize=12)
#     ax1.legend(fontsize=11, markerscale=2)
#     ax1.grid(True, alpha=0.3)
    
#     # Plot 2: Color by cluster (SC only) + ST
#     ax2 = axes[1]
    
#     if y_train is not None:
#         # Get SC data with cluster labels
#         sc_clusters = y_train
#         n_clusters = len(np.unique(sc_clusters))
        
#         # Use a colormap for clusters
#         cmap = plt.cm.get_cmap('tab20', n_clusters)
        
#         # Plot each cluster
#         for cluster_id in np.unique(sc_clusters):
#             cluster_mask_in_sc = sc_clusters == cluster_id
#             # Convert to global index (all train_X)
#             sc_indices = np.where(sc_mask)[0]
#             cluster_global_mask = np.zeros(len(train_X), dtype=bool)
#             cluster_global_mask[sc_indices[cluster_mask_in_sc]] = True
            
#             ax2.scatter(umap_coords[cluster_global_mask, 0], 
#                        umap_coords[cluster_global_mask, 1],
#                        c=[cmap(cluster_id)], s=20, alpha=0.6, 
#                        label=f'Cluster {cluster_id}', edgecolors='none')
        
#         # Plot ST in gray
#         ax2.scatter(umap_coords[st_mask, 0], umap_coords[st_mask, 1], 
#                    c='lightgray', s=20, alpha=0.4, label=f'ST (n={sum(st_mask)})', edgecolors='none')
        
#         ax2.set_title(f'UMAP: SC Clusters (n={n_clusters}) + ST', fontsize=14, fontweight='bold')
#     else:
#         # If no cluster labels, just plot SC and ST
#         ax2.scatter(umap_coords[sc_mask, 0], umap_coords[sc_mask, 1], 
#                    c='#1f77b4', s=20, alpha=0.6, label=f'SC', edgecolors='none')
#         ax2.scatter(umap_coords[st_mask, 0], umap_coords[st_mask, 1], 
#                    c='#ff7f0e', s=20, alpha=0.6, label=f'ST', edgecolors='none')
#         ax2.set_title('UMAP: SC + ST', fontsize=14, fontweight='bold')
    
#     ax2.set_xlabel('UMAP 1', fontsize=12)
#     ax2.set_ylabel('UMAP 2', fontsize=12)
#     ax2.legend(fontsize=9, markerscale=2, ncol=2, loc='upper right')
#     ax2.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(f"{output_dir}/modality_alignment_umap.png", dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"   UMAP visualization saved to: {output_dir}/modality_alignment_umap.png")
