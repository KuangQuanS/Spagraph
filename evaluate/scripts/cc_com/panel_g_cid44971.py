import gc
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # Headless backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyArrowPatch
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors

sc.settings.verbosity = 0

# ─────────────────── CONFIGURATION ─────────────────────────────────────────

DATA_DIR = Path(r"d:\Spagraph\evaluate\data\CID44971")
OUTPUT_DIR = DATA_DIR / "figures" / "panel_g"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

H5AD_PATH = Path(r"d:\Spagraph\spagraph_data\database\Wu\CID44971\CID44971_ST.h5ad")
COMPOSITION_PATH = DATA_DIR / "CID44971_ST_composition.csv"
LR_CSV_PATH = DATA_DIR / "lr_communication.csv"

# Target LR pairs from user request
LR_PAIRS = [
    "THBS2_SDC1",
    "CCL19_CCR7",
    "THBS1_SDC4",
]

# ─────────────────── visual parameters ───────────────────────────────
TOP_EDGES_PER_PAIR = 150     # Reduced from 500 to show top most robust connections
MAX_EDGES_PER_SPOT_PAIR = 2  # Reduced to prevent dense hairballs
OFFSET_RANGE = 20.0
PATHOLOGY_REGION_MAX_COMPONENTS = 2

COLOR_MAP = {
    "B-cells": "#1f77b4", "CAFs": "#ff7f0e", "Cancer Epithelial": "#2ca02c", 
    "Endothelial": "#d62728", "Myeloid": "#9467bd", "Normal Epithelial": "#8c564b", 
    "PVL": "#e377c2", "Plasmablasts": "#7f7f7f", "T-cells": "#bcbd22"
}

MARKER_MAP = {
    "B-cells": "o", "CAFs": "s", "Cancer Epithelial": "^", 
    "Endothelial": "v", "Myeloid": "D", "Normal Epithelial": "<", 
    "PVL": "p", "Plasmablasts": "P", "T-cells": "*"
}

def load_data():
    print("Loading data ...")
    adata = sc.read_h5ad(H5AD_PATH)
    try:
        try:
            comp_df = pd.read_csv(COMPOSITION_PATH, index_col=0)
        except FileNotFoundError:
            comp_df = pd.read_csv(DATA_DIR / "CID44971_ST_cell_composition.csv", index_col=0)
    except FileNotFoundError:
        print("Warning: Cell composition not found, will not draw interface.")
        comp_df = None

    lr_df = pd.read_csv(LR_CSV_PATH)
    
    # Optional: Build interface spots if comp_df exists
    interface_spots = set()
    if comp_df is not None:
        try:
            # For CID44971
            # Imm: B-cells, Myeloid, Plasmablasts, T-cells
            # Ep: Cancer Epithelial, Normal Epithelial
            # Fib: CAFs
            comp_df["imm"] = comp_df[["B-cells", "Myeloid", "Plasmablasts", "T-cells"]].sum(axis=1)
            comp_df["ep"] = comp_df[["Cancer Epithelial", "Normal Epithelial"]].sum(axis=1)
            comp_df["fib"] = comp_df["CAFs"]
            
            spot_types = comp_df[["imm", "ep", "fib"]].idxmax(axis=1)
            
            coords = adata.obsm["spatial"]
            points = pd.DataFrame(coords, index=adata.obs_names, columns=["x", "y"])
            
            nbrs = NearestNeighbors(n_neighbors=6, algorithm="ball_tree").fit(coords)
            distances, indices = nbrs.kneighbors(coords)
            
            for i, spot in enumerate(adata.obs_names):
                curr_type = spot_types.iloc[i]
                neighbor_types = set(spot_types.iloc[indices[i][1:]])
                if curr_type == "ep" and ("fib" in neighbor_types or "imm" in neighbor_types):
                    interface_spots.add(spot)
                elif curr_type in ["fib", "imm"] and "ep" in neighbor_types:
                    interface_spots.add(spot)
        except Exception as e:
            print(f"Failed to calculate interface spots: {e}")

    return adata, comp_df, lr_df, interface_spots

def jitter_coords(coord, random_state):
    return coord + random_state.uniform(-OFFSET_RANGE, OFFSET_RANGE, size=2)

def plot_all_pairs_1x3(adata, lr_df, comp_df, interface_spots, score_col="original_lr_score", label="score"):
    print(f"── Panel G: {label} ──")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    
    legend_entries = {}
    found_any = False
    
    rng = np.random.RandomState(42)

    for ax, lr_pair in zip(axes, LR_PAIRS):
        sub_df = lr_df[lr_df["lr_pair"] == lr_pair].copy()
        if sub_df.empty:
            ax.set_title(f"{lr_pair}\n(No interactions found)")
            ax.axis('off')
            continue
            
        found_any = True
        sub_df = sub_df.sort_values(score_col, ascending=False).head(TOP_EDGES_PER_PAIR)
        
        # limit edges per spot pair cache
        edge_counts = {}
        filtered_indices = []
        for idx, row in sub_df.iterrows():
            pair_key = tuple(sorted([row['src_spot_barcode'], row['dst_spot_barcode']]))
            edge_counts[pair_key] = edge_counts.get(pair_key, 0) + 1
            if edge_counts[pair_key] <= MAX_EDGES_PER_SPOT_PAIR:
                filtered_indices.append(idx)
        sub_df = sub_df.loc[filtered_indices]
        
        try:
            img = adata.uns["spatial"][list(adata.uns["spatial"].keys())[0]]["images"]["hires"]
            scale_fac = adata.uns["spatial"][list(adata.uns["spatial"].keys())[0]]["scalefactors"]["tissue_hires_scalef"]
            ax.imshow(img, alpha=0.4 if interface_spots else 0.5)
            
            coords_scaled = adata.obsm["spatial"] * scale_fac
            x_min_data, x_max_data = coords_scaled[:, 0].min(), coords_scaled[:, 0].max()
            y_min_data, y_max_data = coords_scaled[:, 1].min(), coords_scaled[:, 1].max()
            pad_x = (x_max_data - x_min_data) * 0.05
            pad_y = (y_max_data - y_min_data) * 0.05
            
            ax.set_xlim(max(0, x_min_data - pad_x), min(img.shape[1], x_max_data + pad_x))
            ax.set_ylim(min(img.shape[0], y_max_data + pad_y), max(0, y_min_data - pad_y))
        except:
            scale_fac = 1.0

        # Interface contour
        if interface_spots:
            interface_coords = adata[list(interface_spots)].obsm["spatial"] * scale_fac
            if len(interface_coords) > 3:
                try:
                    tri = Delaunay(interface_coords)
                    edges = set()
                    for simplex in tri.simplices:
                        edges.add(frozenset([simplex[0], simplex[1]]))
                        edges.add(frozenset([simplex[1], simplex[2]]))
                        edges.add(frozenset([simplex[2], simplex[0]]))
                    
                    line_segments = []
                    for i, j in edges:
                        p1 = interface_coords[i]
                        p2 = interface_coords[j]
                        dist = np.linalg.norm(p1 - p2)
                        if dist < 100:  
                            line_segments.append([p1, p2])
                            
                    for seg in line_segments:
                        ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]],
                                color="#888888", linewidth=2.0, alpha=0.5, zorder=1)
                except Exception as e:
                    print(f"Delaunay error for {lr_pair}: {e}")

        used_spots = set(sub_df["src_spot_barcode"]).union(set(sub_df["dst_spot_barcode"]))
        
        # Plot edges
        vmax = sub_df[score_col].max()
        vmin = sub_df[score_col].min()
        for _, row in sub_df.iterrows():
            l_spot, r_spot = row["src_spot_barcode"], row["dst_spot_barcode"]
            score = row[score_col]
            if l_spot not in adata.obs_names or r_spot not in adata.obs_names:
                continue
                
            l_pos = adata.obsm["spatial"][adata.obs_names.get_loc(l_spot)] * scale_fac
            r_pos = adata.obsm["spatial"][adata.obs_names.get_loc(r_spot)] * scale_fac
            
            l_pos_j = jitter_coords(l_pos, rng)
            r_pos_j = jitter_coords(r_pos, rng)
            
            lw = (score / vmax) * 2.0 if vmax > 0 else 1.0
            norm_score = (score - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            edge_color = plt.cm.coolwarm(norm_score)
            
            rad = 0.15 if r_pos_j[0] > l_pos_j[0] else -0.15
            patch = FancyArrowPatch(
                (l_pos_j[0], l_pos_j[1]), 
                (r_pos_j[0], r_pos_j[1]),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>", 
                mutation_scale=5.0,
                linewidth=lw, 
                color=edge_color,
                alpha=0.85, 
                shrinkA=2.0, 
                shrinkB=2.0, 
                zorder=2
            )
            ax.add_patch(patch)

            
            # Nodes
            l_ct = row["source_cell"]
            r_ct = row["target_cell"]
            
            if l_ct in COLOR_MAP: legend_entries[l_ct] = True
            if r_ct in COLOR_MAP: legend_entries[r_ct] = True
            
            l_color = COLOR_MAP.get(l_ct, "gray")
            r_color = COLOR_MAP.get(r_ct, "gray")
            
            l_marker = MARKER_MAP.get(l_ct, "o")
            r_marker = MARKER_MAP.get(r_ct, "o")

            ax.scatter(l_pos_j[0], l_pos_j[1], s=50, c="#4285F4", marker=l_marker, edgecolor='white', linewidth=0.5, zorder=3)
            ax.scatter(r_pos_j[0], r_pos_j[1], s=50, c="#FA3355", marker=r_marker, edgecolor='white', linewidth=0.5, zorder=4)

        title_str = lr_pair.replace("_", " – ")
        ax.set_title(title_str, fontsize=14, pad=15, fontweight="bold")
        ax.axis("off")

    if not found_any:
        print("No interactions found for given LR pairs.")
        return

    # Build Legend
    handles = []
    # Cell Types
    ct_order = [ct for ct in COLOR_MAP.keys() if ct in legend_entries]
    for ct in ct_order:
        marker = MARKER_MAP[ct]
        # use gray for the cell type shapes to match previous plot aesthetic for the legend
        handles.append(Line2D([0], [0], marker=marker, color='w', markerfacecolor='#adb5bd',
                              markersize=10, label=ct))
    
    handles.append(Line2D([0], [0], color=plt.cm.coolwarm(0.8), lw=2, label="Interaction"))

    # Roles
    handles.append(Patch(color='#4285F4', label='Ligand'))
    handles.append(Patch(color='#FA3355', label='Receptor'))

    n_items = len(handles)
    columns_count = (n_items + 1) // 2
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.08),  
        ncol=columns_count,
        frameon=False,
        prop={"size": 9, "weight": "bold"},
        handletextpad=0.4,
        columnspacing=1.2,
    )
    plt.subplots_adjust(bottom=0.15, left=0.01, right=0.99, wspace=0.0)

    pdf = OUTPUT_DIR / f"panel_g_{label}.pdf"
    png = OUTPUT_DIR / f"panel_g_{label}.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    gc.collect()
    print(f"  → {png}")

if __name__ == "__main__":
    adata, comp_df, lr_df, interface_spots = load_data()
    plot_all_pairs_1x3(adata, lr_df, comp_df, interface_spots, score_col="original_lr_score", label="original_score")
    plot_all_pairs_1x3(adata, lr_df, comp_df, interface_spots, score_col="attention_score", label="attention")
    print("Done!")
