"""
CSV-based Spatial Transcriptomics Attention Visualization
Directly using CSV files containing coordinate information for visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import argparse
import os

def plot_attention_on_image(csv_path, image_path, output_path, max_edges=500, min_attention=None):
    """Plot attention network on original image"""
    
    # Load data
    print(f"Loading CSV data: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"CSV data shape: {df.shape}")
    print(f"Column names: {list(df.columns)}")
    
    # Load original image
    print(f"Loading original image: {image_path}")
    image = Image.open(image_path)
    image_array = np.array(image)
    print(f"Image size: {image.size}")
    
    # Check if coordinate information exists
    coord_cols = ['src_x', 'src_y', 'tgt_x', 'tgt_y']
    missing_coords = [col for col in coord_cols if col not in df.columns]
    if missing_coords:
        print(f"Warning: Missing coordinate columns {missing_coords}")
        return
    
    # Filter out edges without coordinates
    df_valid = df.dropna(subset=coord_cols)
    print(f"Valid edges (with coordinates): {len(df_valid)}")
    
    if len(df_valid) == 0:
        print("Error: No valid coordinate data")
        return
    
    # Limit edge count
    if len(df_valid) > max_edges:
        df_valid = df_valid.nlargest(max_edges, 'attention_weight')
        print(f"Showing top {max_edges} edges with highest attention")
    
    # Set attention range
    if min_attention is None:
        min_attention = df_valid['attention_weight'].min()
    max_attention = df_valid['attention_weight'].max()
    
    print(f"Attention weight range: {min_attention:.4f} - {max_attention:.4f}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(image_array)
    
    # Create color mapping
    colors = ['#2E8B57', '#32CD32', '#FFFF00', '#FF8C00', '#FF4500', '#DC143C']
    cmap = LinearSegmentedColormap.from_list('attention', colors, N=256)
    
    # Draw edges
    print("Drawing attention edges...")
    for _, row in df_valid.iterrows():
        x1, y1 = row['src_x'], row['src_y']
        x2, y2 = row['tgt_x'], row['tgt_y']
        attention = row['attention_weight']
        
        # Normalize attention weight
        norm_attention = (attention - min_attention) / (max_attention - min_attention)
        color = cmap(norm_attention)
        
        # Adjust line width based on attention
        linewidth = 0.5 + 2.0 * norm_attention
        
        # Draw edge
        ax.plot([x1, x2], [y1, y2], 
               color=color, linewidth=linewidth, alpha=0.7, solid_capstyle='round')
    
    # Collect all spot positions
    all_spots = set()
    spot_coords = {}
    
    for _, row in df_valid.iterrows():
        src_spot = row['source_spot']
        tgt_spot = row['target_spot']
        
        all_spots.add(src_spot)
        all_spots.add(tgt_spot)
        
        spot_coords[src_spot] = (row['src_x'], row['src_y'])
        spot_coords[tgt_spot] = (row['tgt_x'], row['tgt_y'])
    
    # Draw spot points
    print(f"Drawing {len(all_spots)} spots...")
    for spot_id, (x, y) in spot_coords.items():
        ax.scatter(x, y, s=25, c='white', edgecolors='black', linewidth=0.8, alpha=0.9, zorder=10)
    
    # Set title and labels
    ax.set_title(f'Spatial Transcriptomics Attention Network\n({len(df_valid)} edges, {len(all_spots)} spots)', fontsize=14)
    ax.set_xlim(0, image.width)
    ax.set_ylim(image.height, 0)  # Flip Y axis
    
    # Add color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_attention, vmax=max_attention))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=30)
    cbar.set_label('Attention Weight', fontsize=12)
    
    # Add statistical information
    if 'lr_name' in df_valid.columns:
        lr_counts = df_valid['lr_name'].value_counts().head(5)
        stats_text = "Top 5 LR pairs:\n" + "\n".join([f"{lr}: {count}" for lr, count in lr_counts.items()])
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save image
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization result saved to: {output_path}")
    
    plt.show()

def plot_lr_categories(csv_path, image_path, output_dir, top_n=6):
    """Visualize by LR categories"""
    
    df = pd.read_csv(csv_path)
    image = Image.open(image_path)
    image_array = np.array(image)
    
    # Check coordinate columns
    coord_cols = ['src_x', 'src_y', 'tgt_x', 'tgt_y']
    df_valid = df.dropna(subset=coord_cols)
    
    if 'lr_name' not in df_valid.columns:
        print("Error: No lr_name column in CSV")
        return
    
    # Get most active LR pairs
    lr_counts = df_valid['lr_name'].value_counts().head(top_n)
    
    # Create subplots
    n_cols = 3
    n_rows = (len(lr_counts) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif len(lr_counts) == 1:
        axes = np.array([[axes]])
    
    for idx, (lr_name, count) in enumerate(lr_counts.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Display original image
        ax.imshow(image_array)
        
        # Filter edges for this LR pair
        lr_df = df_valid[df_valid['lr_name'] == lr_name]
        
        # Set color
        color = plt.cm.tab10(idx)
        
        # Draw edges
        for _, row_data in lr_df.iterrows():
            x1, y1 = row_data['src_x'], row_data['src_y']
            x2, y2 = row_data['tgt_x'], row_data['tgt_y']
            
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=1.5, alpha=0.7)
        
        # Draw spots
        all_x = list(lr_df['src_x']) + list(lr_df['tgt_x'])
        all_y = list(lr_df['src_y']) + list(lr_df['tgt_y'])
        ax.scatter(all_x, all_y, s=20, c='white', edgecolors=color, linewidth=1, alpha=0.9)
        
        avg_attention = lr_df['attention_weight'].mean()
        ax.set_title(f'{lr_name}\n{count} edges, avg attention: {avg_attention:.3f}', fontsize=10)
        ax.set_xlim(0, image.width)
        ax.set_ylim(image.height, 0)
        ax.axis('off')
    
    # Hide extra subplots
    for idx in range(len(lr_counts), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('Attention Networks by Ligand-Receptor Categories', fontsize=16)
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, 'lr_categories_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"LR category visualization saved to: {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='CSV-based Spatial Transcriptomics Visualization')
    parser.add_argument('--csv_path', type=str, 
                       default='/home/maweicheng/ST_Graduation_Project/evaluation_results_filtered/edge_attention_analysis_filtered_attention>5.0_with_LR.csv',
                       help='CSV file path containing coordinates')
    parser.add_argument('--image_path', type=str,
                       default='/home/maweicheng/ST_Graduation_Project/database/GSM6177601/GSE203612_GSM6177601.png',
                       help='Original image path')
    parser.add_argument('--output_dir', type=str,
                       default='/home/maweicheng/ST_Graduation_Project/visualization_results',
                       help='Output directory')
    parser.add_argument('--max_edges', type=int, default=300,
                       help='Maximum number of edges to display')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== CSV-based Spatial Transcriptomics Visualization ===")
    
    # Main visualization
    print("\n1. Creating main attention network visualization...")
    main_output = os.path.join(args.output_dir, 'attention_network_main.png')
    plot_attention_on_image(args.csv_path, args.image_path, main_output, args.max_edges)
    
    # LR category visualization
    print("\n2. Creating LR category visualization...")
    plot_lr_categories(args.csv_path, args.image_path, args.output_dir, top_n=6)
    
    print("\nVisualization complete!")

if __name__ == '__main__':
    main()
