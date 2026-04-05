"""
Panel G — Three selected LR pairs in a 1×3 row with a two-line shared legend.
Attention-score figure and frequency-score figure are saved separately.

Output:
    figures/panel_g/panel_g_attention.pdf/.png
    figures/panel_g/panel_g_frequency.pdf/.png
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import gc
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from scipy.spatial import Delaunay
from sklearn.neighbors import kneighbors_graph

# ─────────────────────────────── paths ───────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATE_DIR = SCRIPT_DIR.parent.parent
DATA_ROOT = EVALUATE_DIR / "data"
REPO_ROOT = EVALUATE_DIR.parent
DATABASE_ROOT = REPO_ROOT / "spagraph_data" / "database"

DATA_DIR = DATA_ROOT / "GSE144236"
ST_H5AD_PATH = DATABASE_ROOT / "GSE144240" / "GSE144236_P2_ST.h5ad"
LR_COMM_PATH = DATA_DIR / "lr_communication.csv"
OUTPUT_DIR = DATA_DIR / "figures" / "panel_g"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────── selected LR pairs ────────────────────────────
SELECTED_LR_PAIRS = [
    "TNC_SDC1",
    "TNXB_SDC1",
    "THBS1_CD47",
]

# ─────────────────── visual parameters ───────────────────────────────
TOP_EDGES_PER_PAIR = 500
MAX_EDGES_PER_SPOT_PAIR = 5
OFFSET_RANGE = 20.0
PATHOLOGY_REGION_MAX_COMPONENTS = 2

# Colors — muted, elegant, publication-grade
LINE_COLOR = "#6B7280"  # warm gray for edges
SRC_COLOR = "#3B82F6"  # mid blue (ligand)
DST_COLOR = "#F43F5E"  # rose-pink (receptor)
KEY_REGION_COLOR = "#DC2626"  # red contour for the key region
KEY_REGION_ALPHA = 0.18
KEY_REGION_MAX_LEN = 110.0
KEY_REGION_DOT_SIZE = 20

MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "H", "*"]

# ──────────────────────── load data ──────────────────────────────────
print("Loading data ...")

df = pd.read_csv(LR_COMM_PATH)
df["original_lr_score"] = pd.to_numeric(df["original_lr_score"], errors="coerce").fillna(0.0)
df["attention_score"]   = pd.to_numeric(df["attention_score"],   errors="coerce").fillna(0.0)
df["src_spot_barcode"]  = df["src_spot_barcode"].astype(str)
df["dst_spot_barcode"]  = df["dst_spot_barcode"].astype(str)

adata = sc.read_h5ad(ST_H5AD_PATH)
coords = pd.DataFrame(
    adata.obsm["spatial"],
    index=adata.obs_names.astype(str),
    columns=["x", "y"],
).astype(float)

scale_factor = 1.0
if "spatial" in adata.uns:
    keys = list(adata.uns["spatial"].keys())
    scale_factor = adata.uns["spatial"][keys[0]]["scalefactors"].get(
        "tissue_hires_scalef", 1.0
    )

coords_plot = coords.copy()
coords_plot[["x", "y"]] = coords_plot[["x", "y"]] * scale_factor


# ────────────── interface overlay (relaxed threshold) ────────────────
def _build_interface_spots() -> set[str]:
    """Return set of interface-spot barcodes with a *relaxed* threshold
    (80th percentile instead of 95th) to show more interface spots."""
    composition_path = DATA_DIR / "Spatial_composition.csv"
    if not composition_path.exists():
        return set()
    composition = pd.read_csv(composition_path, index_col=0)
    required = ["Epithelial", "Fibroblast", "Mac", "CD1C", "Tcell"]
    if any(c not in composition.columns for c in required):
        return set()
    composition.index = composition.index.astype(str)
    common = composition.index.intersection(coords.index.astype(str))
    if len(common) == 0:
        return set()
    composition = composition.loc[common]

    ep  = composition["Epithelial"].astype(float)
    fib = composition["Fibroblast"].astype(float)
    imm = (composition["Mac"].astype(float)
           + composition["CD1C"].astype(float)
           + composition["Tcell"].astype(float))
    score = ep * fib * np.maximum(imm, 0.05)
    if float(score.max()) <= float(score.min()):
        return set()
    scaled = (score - score.min()) / (score.max() - score.min())

    # ↓ relaxed: 80th → ~3× more interface spots than 95th
    threshold = float(np.percentile(scaled.to_numpy(), 80))
    mask = scaled >= threshold

    # keep largest two connected components
    local_xy = coords.loc[common, ["x", "y"]].to_numpy(dtype=float)
    adj = kneighbors_graph(
        local_xy,
        n_neighbors=min(8, max(1, len(common) - 1)),
        mode="connectivity",
        include_self=False,
    ).tolil()

    selected = set(common[mask])
    idx_map = {s: i for i, s in enumerate(common)}
    visited: set[str] = set()
    components: list[list[str]] = []
    for s in common:
        if s not in selected or s in visited:
            continue
        stack = [s]
        comp: list[str] = []
        visited.add(s)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for ni in adj.rows[idx_map[cur]]:
                ns = common[ni]
                if ns in selected and ns not in visited:
                    visited.add(ns)
                    stack.append(ns)
        components.append(comp)
    components.sort(key=len, reverse=True)
    kept = set()
    for c in components[:PATHOLOGY_REGION_MAX_COMPONENTS]:
        kept.update(c)
    print(f"  Interface spots (relaxed): {len(kept)}")
    return kept


IFACE_SPOTS = _build_interface_spots()


def _stable_seed(*parts: str) -> int:
    joined = "||".join(parts).encode("utf-8")
    digest = hashlib.sha256(joined).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)


def _format_lr_title(lr_name: str) -> str:
    return lr_name.replace("_", "-")


def _draw_key_region(ax: plt.Axes) -> bool:
    iface_visible = sorted(IFACE_SPOTS.intersection(coords_plot.index.astype(str)))
    if not iface_visible:
        return False

    key_region = coords_plot.loc[iface_visible, ["x", "y"]]
    points = key_region.to_numpy(dtype=float)
    if len(points) < 4:
        ax.scatter(
            key_region["x"],
            key_region["y"],
            s=KEY_REGION_DOT_SIZE,
            c=KEY_REGION_COLOR,
            alpha=KEY_REGION_ALPHA,
            edgecolors="none",
            zorder=3,
            rasterized=True,
        )
        return True

    try:
        tri = Delaunay(points)
    except Exception:
        ax.scatter(
            key_region["x"],
            key_region["y"],
            s=KEY_REGION_DOT_SIZE,
            c=KEY_REGION_COLOR,
            alpha=KEY_REGION_ALPHA,
            edgecolors="none",
            zorder=3,
            rasterized=True,
        )
        return True

    segments_drawn = False
    edges: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        simplex = list(simplex)
        for start, end in ((0, 1), (1, 2), (2, 0)):
            i, j = sorted((simplex[start], simplex[end]))
            edges.add((i, j))

    for i, j in edges:
        p1 = points[i]
        p2 = points[j]
        if np.linalg.norm(p1 - p2) > KEY_REGION_MAX_LEN:
            continue
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color=KEY_REGION_COLOR,
            linewidth=2.0,
            alpha=KEY_REGION_ALPHA,
            zorder=3,
        )
        segments_drawn = True

    if not segments_drawn:
        ax.scatter(
            key_region["x"],
            key_region["y"],
            s=KEY_REGION_DOT_SIZE,
            c=KEY_REGION_COLOR,
            alpha=KEY_REGION_ALPHA,
            edgecolors="none",
            zorder=3,
            rasterized=True,
        )
    return True


# ──────────── cell-type → marker mapping  (global) ───────────────────
global_cell_to_marker: dict[str, str] = {}


# ────────────────── draw one subplot ─────────────────────────────────
def _draw_one(
    ax: plt.Axes,
    lr_name: str,
    score_col: str,
) -> list[str]:
    """Draw a single LR-pair subplot into *ax*.  Returns list of cell types."""
    pair_df = df[df["lr_pair"] == lr_name].copy()
    if pair_df.empty:
        ax.set_title(_format_lr_title(lr_name), fontsize=9, fontweight="bold")
        ax.axis("off")
        return []

    edges = (
        pair_df.sort_values(score_col, ascending=False)
        .groupby(["src_spot_barcode", "dst_spot_barcode"], as_index=False, sort=False)
        .head(MAX_EDGES_PER_SPOT_PAIR)
        .head(TOP_EDGES_PER_PAIR)
        .copy()
    )
    edges["src_x"] = edges["src_spot_barcode"].map(coords["x"])
    edges["src_y"] = edges["src_spot_barcode"].map(coords["y"])
    edges["dst_x"] = edges["dst_spot_barcode"].map(coords["x"])
    edges["dst_y"] = edges["dst_spot_barcode"].map(coords["y"])
    edges = edges.dropna(subset=["src_x", "src_y", "dst_x", "dst_y"])
    if edges.empty:
        ax.set_title(_format_lr_title(lr_name), fontsize=9, fontweight="bold")
        ax.axis("off")
        return []

    # normalise scores → edge widths
    scores = edges[score_col].astype(float).to_numpy()
    smin, smax = float(np.nanmin(scores)), float(np.nanmax(scores))
    if np.isfinite(smin) and np.isfinite(smax) and smax > smin:
        norm = (scores - smin) / (smax - smin + 1e-12)
    else:
        norm = np.zeros_like(scores, dtype=float)
    widths = 0.4 + 0.8 * norm

    # H&E background — very faint
    sc.pl.spatial(adata, color=None, alpha_img=0.15, size=0, show=False, ax=ax)

    # Key region overlay styled like panel_g_cid44971.py
    _draw_key_region(ax)

    # scale edges
    edges["sx"] = edges["src_x"] * scale_factor
    edges["sy"] = edges["src_y"] * scale_factor
    edges["dx"] = edges["dst_x"] * scale_factor
    edges["dy"] = edges["dst_y"] * scale_factor
    dist_sq = (edges["sx"] - edges["dx"])**2 + (edges["sy"] - edges["dy"])**2
    edges = edges.loc[dist_sq > 1e-12].copy()
    if edges.empty:
        ax.set_title(_format_lr_title(lr_name), fontsize=9, fontweight="bold")
        ax.axis("off")
        return []

    current_cells = sorted(set(edges["source_cell"]) | set(edges["target_cell"]))
    for ct in current_cells:
        if ct not in global_cell_to_marker:
            global_cell_to_marker[ct] = MARKERS[len(global_cell_to_marker) % len(MARKERS)]

    valid_widths = widths[:len(edges)]
    rng = np.random.default_rng(_stable_seed(lr_name, score_col))

    for row, w in zip(edges.itertuples(index=False), valid_widths):
        sx = row.sx + rng.uniform(-OFFSET_RANGE, OFFSET_RANGE)
        sy = row.sy + rng.uniform(-OFFSET_RANGE, OFFSET_RANGE)
        dx = row.dx + rng.uniform(-OFFSET_RANGE, OFFSET_RANGE)
        dy = row.dy + rng.uniform(-OFFSET_RANGE, OFFSET_RANGE)

        # edge
        rad = 0.15 if dx > sx else -0.15
        patch = FancyArrowPatch(
            (sx, sy), (dx, dy),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", mutation_scale=4.0,
            linewidth=float(w), color=LINE_COLOR,
            alpha=0.45, shrinkA=2.5, shrinkB=2.5, zorder=5,
        )
        ax.add_patch(patch)

        # ligand node
        ax.scatter(sx, sy, s=14, color=SRC_COLOR,
                   marker=global_cell_to_marker[row.source_cell],
                   edgecolor="white", linewidth=0.25, zorder=6)
        # receptor node
        ax.scatter(dx, dy, s=14, color=DST_COLOR,
                   marker=global_cell_to_marker[row.target_cell],
                   edgecolor="white", linewidth=0.25, zorder=6)

    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")

    # Title — just the LR pair name, small & bold
    ax.set_title(_format_lr_title(lr_name), fontsize=9.5, fontweight="bold", pad=6)

    return current_cells


# ────────────────── compose the 1×3 figure ───────────────────────────
def make_panel_g(score_col: str, label: str) -> None:
    """Create a 1×3 figure for *score_col* ('attention_score' or 'original_lr_score')."""

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })

    # By reducing the width to ~13 (for a height of ~5), each subplot gets a squarer box.
    # This prevents matplotlib from forcing empty horizontal space to maintain 'equal' aspect.
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.0))
    fig.subplots_adjust(wspace=0.0)

    all_cells: set[str] = set()
    for ax, lr in zip(axes, SELECTED_LR_PAIRS):
        cells = _draw_one(ax, lr, score_col)
        all_cells.update(cells)

    # ── shared legend at the bottom, wrapped into two lines ───────────
    all_cells_sorted = sorted(all_cells)
    handles: list[Line2D] = []

    # separator heading — cell types
    for ct in all_cells_sorted:
        handles.append(Line2D(
            [0], [0], marker=global_cell_to_marker.get(ct, "o"),
            color="w", markerfacecolor="#9CA3AF",
            markeredgecolor="white", markeredgewidth=0.3,
            markersize=7, linewidth=0, label=ct,
        ))

    # overlay items
    handles.append(Line2D([0], [0], color=LINE_COLOR, linewidth=1.5,
                          label="Interaction"))
    handles.append(Line2D(
        [0], [0], marker="o", color="w",
        markerfacecolor=SRC_COLOR, markeredgecolor="white",
        markersize=7, linewidth=0, label="Ligand",
    ))
    handles.append(Line2D(
        [0], [0], marker="o", color="w",
        markerfacecolor=DST_COLOR, markeredgecolor="white",
        markersize=7, linewidth=0, label="Receptor",
    ))
    if IFACE_SPOTS:
        handles.append(Line2D(
            [0], [0], color=KEY_REGION_COLOR, linewidth=2.0,
            alpha=0.6, label="Key region",
        ))

    n_items = len(handles)
    columns_count = max(1, (n_items + 1) // 2)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=columns_count,
        frameon=False,
        prop={"size": 8.5, "weight": "bold"},
        handletextpad=0.3,
        columnspacing=1.0,
    )
    plt.subplots_adjust(bottom=0.16, left=0.01, right=0.99, wspace=0.0)

    pdf = OUTPUT_DIR / f"panel_g_{label}.pdf"
    png = OUTPUT_DIR / f"panel_g_{label}.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    gc.collect()
    print(f"  → {pdf}")
    print(f"  → {png}")


# ────────────────────── main ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n── Panel G: attention_score ──")
    make_panel_g("attention_score", "attention")

    print("\n── Panel G: original_lr_score (frequency) ──")
    make_panel_g("original_lr_score", "frequency")

    print("\nDone!")
