import gc
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors

sc.settings.verbosity = 0

# ─────────────────── CONFIGURATION ─────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATE_DIR = SCRIPT_DIR.parent.parent
DATA_ROOT = EVALUATE_DIR / "data"
REPO_ROOT = EVALUATE_DIR.parent
DATABASE_ROOT = REPO_ROOT / "spagraph_data" / "database"

DATA_DIR = DATA_ROOT / "GSE211956" / "P3"
ST_H5AD_PATH = DATABASE_ROOT / "GSE211956" / "GSE211956_ST_P3.h5ad"
LR_COMM_PATH = DATA_DIR / "lr_communication.csv"
if not LR_COMM_PATH.exists():
    LR_COMM_PATH = DATA_DIR / "ccc_analysis" / "_observed_run" / "cellcom" / "lr_communication.csv"
OUTPUT_DIR = DATA_DIR / "figures" / "panel_g"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMPOSITION_PATH = DATA_DIR / "GSE211956_ST_P3_cell_composition.csv"

# Target LR pairs from user request
LR_PAIRS = [
    "PDGFD_PDGFRB",
    "CXCL12_CXCR4",
    "IGF1_ITGA6_ITGB4",
]

# ─────────────────── visual parameters ───────────────────────────────
TOP_EDGES_PER_PAIR = 200     # Reduced from 500 to show top most robust connections
MAX_EDGES_PER_SPOT_PAIR = 2  # Reduced to prevent dense hairballs
OFFSET_RANGE = 20.0

cell_types = ['B cells', 'Endothelial cells', 'Fibroblasts', 'Mast cells', 'Myeloid cells', 'Plasmablasts', 'T cells', 'Tumour cells']
colors = plt.cm.tab20.colors
markers = ["o", "v", "^", "<", ">", "s", "p", "*", "h", "H", "D", "d", "P", "X"]

COLOR_MAP = {ct: matplotlib.colors.to_hex(colors[i % len(colors)]) for i, ct in enumerate(cell_types)}
MARKER_MAP = {ct: markers[i % len(markers)] for i, ct in enumerate(cell_types)}


def load_data():
    print("Loading data ...")
    adata = sc.read_h5ad(ST_H5AD_PATH)
    try:
        comp_df = pd.read_csv(COMPOSITION_PATH, index_col=0)
    except FileNotFoundError:
        print("Warning: Cell composition not found, will not draw interface.")
        comp_df = None

    lr_df = pd.read_csv(LR_COMM_PATH)
    
    # Check if df has old column names
    if "ligand_spot" in lr_df.columns:
        lr_df = lr_df.rename(columns={"ligand_spot": "src_spot_barcode", "receptor_spot": "dst_spot_barcode",
                                      "ligand_dominant_cell_type": "source_cell", "receptor_dominant_cell_type": "target_cell"})
    
    # Optional: Build interface spots if comp_df exists
    interface_spots = set()
    if comp_df is not None:
        try:
            comp_df["tumor"] = comp_df["Tumour cells"]
            
            fib_cols = [c for c in comp_df.columns if "Fibro" in c or c == "Myofibroblasts"]
            if fib_cols:
                comp_df["stromal"] = comp_df[fib_cols].sum(axis=1)
            else:
                comp_df["stromal"] = 0
                
            imm_cols = [c for c in comp_df.columns if "Macrophage" in c or "T cell" in c or "Myeloid" in c]
            if imm_cols:
                comp_df["imm"] = comp_df[imm_cols].sum(axis=1)
            else:
                comp_df["imm"] = 0
                
            spot_types = comp_df[["tumor", "stromal", "imm"]].idxmax(axis=1)
            
            coords = adata.obsm["spatial"]
            points = pd.DataFrame(coords, index=adata.obs_names, columns=["x", "y"])
            
            nbrs = NearestNeighbors(n_neighbors=6, algorithm="ball_tree").fit(coords)
            distances, indices = nbrs.kneighbors(coords)
            
            for i, spot in enumerate(adata.obs_names):
                curr_type = spot_types.iloc[i]
                neighbor_types = set(spot_types.iloc[indices[i][1:]])
                if curr_type == "tumor" and ("stromal" in neighbor_types or "imm" in neighbor_types):
                    interface_spots.add(spot)
                elif curr_type in ["stromal", "imm"] and "tumor" in neighbor_types:
                    interface_spots.add(spot)
        except Exception as e:
            import traceback
            traceback.print_exc()
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
            pair_key = tuple(sorted([str(row['src_spot_barcode']), str(row['dst_spot_barcode'])]))
            edge_counts[pair_key] = edge_counts.get(pair_key, 0) + 1
            if edge_counts[pair_key] <= MAX_EDGES_PER_SPOT_PAIR:
                filtered_indices.append(idx)
        sub_df = sub_df.loc[filtered_indices]
        
        try:
            img = adata.uns["spatial"][list(adata.uns["spatial"].keys())[0]]["images"]["hires"]
            scale_fac = adata.uns["spatial"][list(adata.uns["spatial"].keys())[0]]["scalefactors"]["tissue_hires_scalef"]
            ax.imshow(img, alpha=0.65 if interface_spots else 0.8)
            
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
            valid_interface_spots = [s for s in interface_spots if s in adata.obs_names]
            if len(valid_interface_spots) > 3:
                interface_coords = adata[valid_interface_spots].obsm["spatial"] * scale_fac
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
                                color="#888888", linewidth=1.5, alpha=0.6, zorder=1)
                except Exception as e:
                    print(f"Delaunay error for {lr_pair}: {e}")

        # Plot edges
        vmax = sub_df[score_col].max()
        vmin = sub_df[score_col].min()
        for _, row in sub_df.iterrows():
            l_spot, r_spot = str(row["src_spot_barcode"]), str(row["dst_spot_barcode"])
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
            
            if l_ct not in COLOR_MAP:
                COLOR_MAP[l_ct] = matplotlib.colors.to_hex(colors[len(COLOR_MAP) % len(colors)])
                MARKER_MAP[l_ct] = markers[len(MARKER_MAP) % len(markers)]
                
            if r_ct not in COLOR_MAP:
                COLOR_MAP[r_ct] = matplotlib.colors.to_hex(colors[len(COLOR_MAP) % len(colors)])
                MARKER_MAP[r_ct] = markers[len(MARKER_MAP) % len(markers)]

            legend_entries[l_ct] = True
            legend_entries[r_ct] = True
            
            l_marker = MARKER_MAP.get(l_ct, "o")
            r_marker = MARKER_MAP.get(r_ct, "o")

            ax.scatter(l_pos_j[0], l_pos_j[1], s=50, c="#4285F4", marker=l_marker, edgecolor='white', linewidth=0.25, zorder=3)
            ax.scatter(r_pos_j[0], r_pos_j[1], s=50, c="#FA3355", marker=r_marker, edgecolor='white', linewidth=0.25, zorder=4)

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
    import matplotlib.patches as mpatches
    handles.append(mpatches.Patch(color='#4285F4', label='Ligand'))
    handles.append(mpatches.Patch(color='#FA3355', label='Receptor'))

    n_items = len(handles)
    columns_count = min(6, (n_items + 1) // 2)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.12),  
        ncol=columns_count,
        frameon=False,
        prop={"size": 9, "weight": "bold"},
        handletextpad=0.4,
        columnspacing=1.2,
    )
    plt.subplots_adjust(bottom=0.20, left=0.01, right=0.99, wspace=0.0)

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
