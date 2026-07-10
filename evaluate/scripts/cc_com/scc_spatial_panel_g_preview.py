from __future__ import annotations

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


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATE_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = EVALUATE_DIR / "data" / "GSE144236"
DATABASE_ROOT = EVALUATE_DIR.parent / "spagraph_data" / "database"

ST_H5AD_PATH = DATABASE_ROOT / "GSE144240" / "GSE144236_P2_ST.h5ad"
SPAGRAPH_LR_PATH = DATA_DIR / "lr_communication.csv"
COMPOSITION_PATH = DATA_DIR / "Spatial_composition.csv"
CELLCHAT_SHARED_PATH = DATA_DIR / "cellchat_baseline_figures" / "cellchat_spagraph_shared_pair_ranks.csv"
COMMOT_PATH = DATA_DIR / "commot_baseline_perm20_min5" / "commot_pair_summary_cross_group.csv"
GIOTTO_PATH = DATA_DIR / "giotto_baseline_iter20" / "giotto_pair_summary.csv"
OUTPUT_DIR = DATA_DIR / "cellchat_baseline_figures"

SPATIAL_PAIRS = [
    ("TNC_SDC1", "shared anchor"),
    ("TNXB_SDC1", "Spagraph-promoted"),
    ("LGALS9_CD44", "relative deprioritized"),
]

LINE_COLOR = "#B8BEC7"
SRC_COLOR = "#3B82F6"
DST_COLOR = "#F43F5E"
INTERFACE_COLOR = "#F59E0B"
BACKGROUND_SPOT = "#D6D9DE"
TOP_EDGES_PER_PAIR = 280
MAX_EDGES_PER_SPOT_PAIR = 3
OFFSET_RANGE = 18.0
PATHOLOGY_REGION_MAX_COMPONENTS = 2


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("||".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)


def format_pair(lr_pair: str) -> str:
    return lr_pair.replace("_", "-")


def load_inputs():
    spagraph_df = pd.read_csv(SPAGRAPH_LR_PATH)
    spagraph_df["lr_pair"] = spagraph_df["lr_pair"].astype(str)
    spagraph_df["src_spot_barcode"] = spagraph_df["src_spot_barcode"].astype(str)
    spagraph_df["dst_spot_barcode"] = spagraph_df["dst_spot_barcode"].astype(str)
    spagraph_df["attention_score"] = pd.to_numeric(spagraph_df["attention_score"], errors="coerce")
    spagraph_df = spagraph_df.dropna(subset=["attention_score"]).copy()

    adata = sc.read_h5ad(ST_H5AD_PATH)
    adata.obs_names = adata.obs_names.astype(str)
    coords = pd.DataFrame(adata.obsm["spatial"], index=adata.obs_names, columns=["x", "y"]).astype(float)

    composition = pd.read_csv(COMPOSITION_PATH, index_col=0)
    composition.index = composition.index.astype(str)
    composition = composition.loc[coords.index.intersection(composition.index)].copy()

    ranks = pd.read_csv(CELLCHAT_SHARED_PATH)
    commot = pd.read_csv(COMMOT_PATH)[["interaction_name", "commot_rank"]].rename(columns={"interaction_name": "lr_pair"})
    giotto = pd.read_csv(GIOTTO_PATH)[["interaction_name", "giotto_spatial_rank"]].rename(columns={"interaction_name": "lr_pair"})
    rank_df = ranks.merge(commot, on="lr_pair", how="left").merge(giotto, on="lr_pair", how="left")
    return spagraph_df, adata, coords, composition, rank_df


def build_interface_spots(composition: pd.DataFrame, coords: pd.DataFrame) -> set[str]:
    required = ["Epithelial", "Fibroblast", "Mac", "CD1C", "Tcell"]
    if any(col not in composition.columns for col in required):
        return set()

    common = composition.index.intersection(coords.index)
    comp = composition.loc[common]
    ep = comp["Epithelial"].astype(float)
    fib = comp["Fibroblast"].astype(float)
    imm = comp["Mac"].astype(float) + comp["CD1C"].astype(float) + comp["Tcell"].astype(float)
    score = ep * fib * np.maximum(imm, 0.05)
    if float(score.max()) <= float(score.min()):
        return set()
    scaled = (score - score.min()) / (score.max() - score.min())
    threshold = float(np.percentile(scaled.to_numpy(), 80))
    selected = set(common[scaled >= threshold])

    local_xy = coords.loc[common, ["x", "y"]].to_numpy(dtype=float)
    from sklearn.neighbors import kneighbors_graph
    adjacency = kneighbors_graph(local_xy, n_neighbors=min(8, max(1, len(common) - 1)), mode="connectivity", include_self=False).tolil()
    idx_map = {spot: i for i, spot in enumerate(common)}
    visited: set[str] = set()
    components: list[list[str]] = []
    for spot in common:
        if spot not in selected or spot in visited:
            continue
        stack = [spot]
        component: list[str] = []
        visited.add(spot)
        while stack:
            cur = stack.pop()
            component.append(cur)
            for neighbor_idx in adjacency.rows[idx_map[cur]]:
                neighbor_spot = common[neighbor_idx]
                if neighbor_spot in selected and neighbor_spot not in visited:
                    visited.add(neighbor_spot)
                    stack.append(neighbor_spot)
        components.append(component)
    components.sort(key=len, reverse=True)
    kept = set()
    for component in components[:PATHOLOGY_REGION_MAX_COMPONENTS]:
        kept.update(component)
    return kept


def draw_interface_overlay(ax: plt.Axes, coords_plot: pd.DataFrame, interface_spots: set[str]) -> None:
    visible = sorted(interface_spots.intersection(coords_plot.index))
    if not visible:
        return
    points = coords_plot.loc[visible, ["x", "y"]].to_numpy(dtype=float)
    ax.scatter(points[:, 0], points[:, 1], s=22, c=INTERFACE_COLOR, alpha=0.28, edgecolors="white", linewidths=0.25, zorder=2, rasterized=True)
    if len(points) < 4:
        return
    try:
        tri = Delaunay(points)
    except Exception:
        return
    edges: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        for start, end in ((0, 1), (1, 2), (2, 0)):
            i, j = sorted((simplex[start], simplex[end]))
            edges.add((i, j))
    for i, j in edges:
        p1 = points[i]
        p2 = points[j]
        if np.linalg.norm(p1 - p2) > 110.0:
            continue
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=INTERFACE_COLOR, linewidth=2.2, alpha=0.58, zorder=3)


def draw_spatial_pair(ax, pair_name, role_text, rank_df, spagraph_df, adata, coords, interface_spots):
    pair_df = spagraph_df[spagraph_df["lr_pair"] == pair_name].copy()
    pair_df = (
        pair_df.sort_values("attention_score", ascending=False)
        .groupby(["src_spot_barcode", "dst_spot_barcode"], as_index=False, sort=False)
        .head(MAX_EDGES_PER_SPOT_PAIR)
        .head(TOP_EDGES_PER_PAIR)
        .copy()
    )

    scale_factor = 1.0
    if "spatial" in adata.uns:
        library_key = list(adata.uns["spatial"].keys())[0]
        scale_factor = adata.uns["spatial"][library_key]["scalefactors"].get("tissue_hires_scalef", 1.0)

    sc.pl.spatial(adata, color=None, alpha_img=0.16, size=0, show=False, ax=ax)

    coords_plot = coords.copy()
    coords_plot[["x", "y"]] = coords_plot[["x", "y"]] * scale_factor
    ax.scatter(coords_plot["x"], coords_plot["y"], s=8, c=BACKGROUND_SPOT, alpha=0.45, edgecolors="none", zorder=1, rasterized=True)
    draw_interface_overlay(ax, coords_plot, interface_spots)

    pair_df["src_x"] = pair_df["src_spot_barcode"].map(coords_plot["x"])
    pair_df["src_y"] = pair_df["src_spot_barcode"].map(coords_plot["y"])
    pair_df["dst_x"] = pair_df["dst_spot_barcode"].map(coords_plot["x"])
    pair_df["dst_y"] = pair_df["dst_spot_barcode"].map(coords_plot["y"])
    pair_df = pair_df.dropna(subset=["src_x", "src_y", "dst_x", "dst_y"]).copy()

    scores = pair_df["attention_score"].to_numpy(dtype=float)
    if len(scores):
        smin = float(np.nanmin(scores))
        smax = float(np.nanmax(scores))
        norm = (scores - smin) / (smax - smin + 1e-12) if smax > smin else np.zeros_like(scores)
        widths = 0.45 + 1.15 * norm
    else:
        widths = np.array([])

    rng = np.random.default_rng(stable_seed(pair_name))
    for row, width in zip(pair_df.itertuples(index=False), widths):
        sx = row.src_x + rng.uniform(-OFFSET_RANGE, OFFSET_RANGE)
        sy = row.src_y + rng.uniform(-OFFSET_RANGE, OFFSET_RANGE)
        dx = row.dst_x + rng.uniform(-OFFSET_RANGE, OFFSET_RANGE)
        dy = row.dst_y + rng.uniform(-OFFSET_RANGE, OFFSET_RANGE)
        rad = 0.15 if dx > sx else -0.15
        patch = FancyArrowPatch((sx, sy), (dx, dy), connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=4.0, linewidth=float(width), color=LINE_COLOR, alpha=0.42, shrinkA=2.0, shrinkB=2.0, zorder=4)
        ax.add_patch(patch)
        ax.scatter(sx, sy, s=18, color=SRC_COLOR, marker="o", edgecolor="white", linewidth=0.3, zorder=5)
        ax.scatter(dx, dy, s=18, color=DST_COLOR, marker="o", edgecolor="white", linewidth=0.3, zorder=6)

    row = rank_df.loc[rank_df["lr_pair"] == pair_name].iloc[0]
    ax.text(0.5, 1.075, f"{format_pair(pair_name)}  ({role_text})", transform=ax.transAxes, ha="center", va="bottom", fontsize=12, fontweight="bold", color="#111827")
    subtitle1 = f"Spagraph #{int(row.spagraph_rank)} | CellChat #{int(row.cellchat_rank)}"
    subtitle2 = f"COMMOT #{int(row.commot_rank)} | Giotto #{int(row.giotto_spatial_rank)}"
    ax.text(0.5, 1.032, subtitle1, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.8, color="#6B7280")
    ax.text(0.5, 1.001, subtitle2, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.8, color="#6B7280")

    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "pdf.fonttype": 42, "ps.fonttype": 42})
    spagraph_df, adata, coords, composition, rank_df = load_inputs()
    interface_spots = build_interface_spots(composition, coords)

    fig, axes = plt.subplots(1, len(SPATIAL_PAIRS), figsize=(5.35 * len(SPATIAL_PAIRS), 5.45))
    for ax, (pair_name, role_text) in zip(axes, SPATIAL_PAIRS):
        draw_spatial_pair(ax, pair_name, role_text, rank_df, spagraph_df, adata, coords, interface_spots)

    handles = [
        Line2D([0], [0], color=LINE_COLOR, linewidth=1.6, label="Top attention-ranked interactions"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SRC_COLOR, markeredgecolor="white", markersize=8, label="Ligand"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=DST_COLOR, markeredgecolor="white", markersize=8, label="Receptor"),
        Line2D([0], [0], color=INTERFACE_COLOR, linewidth=2.0, alpha=0.5, label="Tumor-stroma interface"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False, fontsize=9.2, columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(OUTPUT_DIR / "figureG_spatial_triptych_preview.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "figureG_spatial_triptych_preview.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
