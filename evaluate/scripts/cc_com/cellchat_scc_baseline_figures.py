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
from sklearn.neighbors import kneighbors_graph


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATE_DIR = SCRIPT_DIR.parent.parent
DATA_ROOT = EVALUATE_DIR / "data"
REPO_ROOT = EVALUATE_DIR.parent
DATABASE_ROOT = REPO_ROOT / "spagraph_data" / "database"

DATA_DIR = DATA_ROOT / "GSE144236"
ST_H5AD_PATH = DATABASE_ROOT / "GSE144240" / "GSE144236_P2_ST.h5ad"
SPAGRAPH_LR_PATH = DATA_DIR / "lr_communication.csv"
CELLCHAT_LR_PATH = DATA_DIR / "cellchat_baseline_allspots_nboot5" / "cellchat_lr_communications.csv"
COMPOSITION_PATH = DATA_DIR / "Spatial_composition.csv"
OUTPUT_DIR = DATA_DIR / "cellchat_baseline_figures"

SELECTED_PAIRS = [
    ("TNC_SDC1", "anchor"),
    ("TNXB_SDC1", "promoted"),
    ("THBS1_CD47", "promoted"),
    ("FN1_CD44", "deprioritized"),
    ("LGALS9_PTPRC", "deprioritized"),
    ("LGALS9_CD44", "deprioritized"),
    ("CXCL9_CXCR3", "deprioritized"),
]
SPATIAL_PAIRS = ["TNC_SDC1", "TNXB_SDC1", "LGALS9_CD44"]

PAIR_AXIS_GROUPS = [
    ("ECM remodeling", ["TNC_SDC1", "TNXB_SDC1", "THBS1_CD47"]),
    ("Immunosuppressive", ["LGALS9_PTPRC", "LGALS9_CD44"]),
    ("Broadly distributed", ["FN1_CD44", "CXCL9_CXCR3"]),
]
CELLCHAT_ALIAS = {
    "LGALS9_PTPRC": "LGALS9_CD45",
}

PAIR_CLASS_COLORS = {
    "anchor": "#4C72B0",
    "promoted": "#2E8B57",
    "deprioritized": "#C44E52",
}

LINE_COLOR = "#B8BEC7"
CELLCHAT_POINT = "#7F8A99"
SRC_COLOR = "#3B82F6"
DST_COLOR = "#F43F5E"
INTERFACE_COLOR = "#F59E0B"
BACKGROUND_SPOT = "#D6D9DE"

MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "H", "*", "X", "P"]
TOP_EDGES_PER_PAIR = 280
MAX_EDGES_PER_SPOT_PAIR = 3
OFFSET_RANGE = 18.0
PATHOLOGY_REGION_MAX_COMPONENTS = 2


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("||".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)


def format_pair(lr_pair: str) -> str:
    return lr_pair.replace("_", "-")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, sc.AnnData, pd.DataFrame, pd.DataFrame]:
    spagraph_df = pd.read_csv(SPAGRAPH_LR_PATH)
    spagraph_df["lr_pair"] = spagraph_df["lr_pair"].astype(str)
    spagraph_df["src_spot_barcode"] = spagraph_df["src_spot_barcode"].astype(str)
    spagraph_df["dst_spot_barcode"] = spagraph_df["dst_spot_barcode"].astype(str)
    spagraph_df["source_cell"] = spagraph_df["source_cell"].astype(str)
    spagraph_df["target_cell"] = spagraph_df["target_cell"].astype(str)
    spagraph_df["attention_score"] = pd.to_numeric(spagraph_df["attention_score"], errors="coerce")
    spagraph_df = spagraph_df.dropna(subset=["attention_score"]).copy()

    cellchat_df = pd.read_csv(CELLCHAT_LR_PATH)
    cellchat_df["interaction_name"] = cellchat_df["interaction_name"].astype(str)
    cellchat_df["prob"] = pd.to_numeric(cellchat_df["prob"], errors="coerce")
    cellchat_df = cellchat_df.dropna(subset=["prob"]).copy()

    adata = sc.read_h5ad(ST_H5AD_PATH)
    adata.obs_names = adata.obs_names.astype(str)

    coords = pd.DataFrame(
        adata.obsm["spatial"],
        index=adata.obs_names,
        columns=["x", "y"],
    ).astype(float)

    composition = pd.read_csv(COMPOSITION_PATH, index_col=0)
    composition.index = composition.index.astype(str)
    composition = composition.loc[coords.index.intersection(composition.index)].copy()

    return spagraph_df, cellchat_df, adata, coords, composition


def build_pair_rank_table(spagraph_df: pd.DataFrame, cellchat_df: pd.DataFrame) -> pd.DataFrame:
    spagraph_pair = (
        spagraph_df.groupby("lr_pair", as_index=False)
        .agg(
            spagraph_score=("attention_score", "mean"),
            spagraph_edge_count=("lr_pair", "size"),
        )
    )
    cellchat_pair = (
        cellchat_df.groupby("interaction_name", as_index=False)
        .agg(
            cellchat_prob=("prob", "max"),
            cellchat_group_count=("interaction_name", "size"),
        )
        .rename(columns={"interaction_name": "cellchat_name"})
    )
    cellchat_lookup = cellchat_pair.set_index("cellchat_name")

    rows = []
    for row in spagraph_pair.itertuples(index=False):
        cellchat_name = CELLCHAT_ALIAS.get(row.lr_pair, row.lr_pair)
        if cellchat_name not in cellchat_lookup.index:
            continue
        cellchat_row = cellchat_lookup.loc[cellchat_name]
        rows.append(
            {
                "lr_pair": row.lr_pair,
                "cellchat_name": cellchat_name,
                "spagraph_score": float(row.spagraph_score),
                "spagraph_edge_count": int(row.spagraph_edge_count),
                "cellchat_prob": float(cellchat_row["cellchat_prob"]),
                "cellchat_group_count": int(cellchat_row["cellchat_group_count"]),
            }
        )

    merged = pd.DataFrame(rows)
    merged["spagraph_rank"] = merged["spagraph_score"].rank(ascending=False, method="min")
    merged["cellchat_rank"] = merged["cellchat_prob"].rank(ascending=False, method="min")
    denom = max(len(merged) - 1, 1)
    merged["spagraph_rank_pct"] = 100.0 * (1.0 - (merged["spagraph_rank"] - 1.0) / denom)
    merged["cellchat_rank_pct"] = 100.0 * (1.0 - (merged["cellchat_rank"] - 1.0) / denom)
    merged["rank_shift"] = merged["spagraph_rank_pct"] - merged["cellchat_rank_pct"]
    merged["pair_label"] = merged["lr_pair"].map(format_pair)
    return merged.sort_values("spagraph_rank").reset_index(drop=True)


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
    adjacency = kneighbors_graph(
        local_xy,
        n_neighbors=min(8, max(1, len(common) - 1)),
        mode="connectivity",
        include_self=False,
    ).tolil()
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


def save_rank_tables(pair_rank_df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pair_rank_df.to_csv(OUTPUT_DIR / "cellchat_spagraph_shared_pair_ranks.csv", index=False)
    selected = pair_rank_df[pair_rank_df["lr_pair"].isin([pair for pair, _ in SELECTED_PAIRS])].copy()
    selected["role"] = selected["lr_pair"].map(dict(SELECTED_PAIRS))
    selected.to_csv(OUTPUT_DIR / "cellchat_spagraph_selected_pair_ranks.csv", index=False)


def _ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{['th','st','nd','rd'][n % 10] if n % 10 < 4 else 'th'}"


# Light background tints for axis groups
_GROUP_BAND_COLORS = ["#EEF2FF", "#FFF7ED", "#F0FDF4"]


def draw_rank_shift(pair_rank_df: pd.DataFrame) -> None:
    n_shared = len(pair_rank_df)
    selected = pair_rank_df[pair_rank_df["lr_pair"].isin([p for p, _ in SELECTED_PAIRS])].copy()
    selected["role"] = selected["lr_pair"].map(dict(SELECTED_PAIRS))
    selected["pair_order"] = selected["lr_pair"].map(
        {pair: i for i, (pair, _) in enumerate(SELECTED_PAIRS)}
    )
    selected = selected.sort_values("pair_order").reset_index(drop=True)
    selected["y"] = np.arange(len(selected))[::-1]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "text.antialiased": True,
        }
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.4), dpi=150)

    # --- coloured background bands per group ---
    pair_to_y = dict(zip(selected["lr_pair"], selected["y"]))
    for idx, (group_name, group_pairs) in enumerate(PAIR_AXIS_GROUPS):
        ys = [pair_to_y[p] for p in group_pairs if p in pair_to_y]
        if not ys:
            continue
        band_lo = min(ys) - 0.45
        band_hi = max(ys) + 0.45
        ax.axhspan(band_lo, band_hi, color=_GROUP_BAND_COLORS[idx % len(_GROUP_BAND_COLORS)],
                    zorder=0, linewidth=0)
        # Group label on the right edge, inside the band
        ax.text(
            101.5, max(ys) + 0.35, group_name,
            ha="right", va="top", fontsize=8, fontstyle="italic",
            color="#B0B5BE", zorder=0,
        )

    # --- median reference ---
    ax.axvline(x=50, color="#D1D5DB", linewidth=0.9, linestyle=":", zorder=1)

    # --- dumbbell lines and dots ---
    for row in selected.itertuples(index=False):
        color = PAIR_CLASS_COLORS[row.role]
        ax.plot(
            [row.cellchat_rank_pct, row.spagraph_rank_pct],
            [row.y, row.y],
            color=color, linewidth=2.6, alpha=0.65,
            solid_capstyle="round", zorder=2,
        )
        ax.scatter(row.cellchat_rank_pct, row.y, s=58, color=CELLCHAT_POINT,
                   edgecolors="white", linewidths=0.8, zorder=3)
        ax.scatter(row.spagraph_rank_pct, row.y, s=76, color=color,
                   edgecolors="white", linewidths=0.9, zorder=4)
        # Rank annotations — dodge when dots are close
        sp_rank_txt = f"#{int(row.spagraph_rank)}"
        cc_rank_txt = f"#{int(row.cellchat_rank)}"
        gap = abs(row.spagraph_rank_pct - row.cellchat_rank_pct)
        if gap < 12:
            # Spagraph label above, CellChat label below
            ax.text(row.spagraph_rank_pct, row.y + 0.33, sp_rank_txt,
                    ha="center", va="bottom", fontsize=8.5,
                    color=color, fontweight="bold", zorder=5)
            ax.text(row.cellchat_rank_pct, row.y - 0.33, cc_rank_txt,
                    ha="center", va="top", fontsize=8.5,
                    color=CELLCHAT_POINT, fontweight="bold", zorder=5)
        else:
            ax.text(row.spagraph_rank_pct, row.y + 0.33, sp_rank_txt,
                    ha="center", va="bottom", fontsize=8.5,
                    color=color, fontweight="bold", zorder=5)
            ax.text(row.cellchat_rank_pct, row.y + 0.33, cc_rank_txt,
                    ha="center", va="bottom", fontsize=8.5,
                    color=CELLCHAT_POINT, fontweight="bold", zorder=5)

    ax.set_yticks(selected["y"])
    ax.set_yticklabels(
        [format_pair(p) for p in selected["lr_pair"]],
        fontsize=11.5, fontweight="bold",
    )
    ax.set_xlim(-2, 102)
    ax.set_ylim(-0.6, len(selected) - 0.4)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(
        f"Rank Percentile  (shared LR universe, n = {n_shared})",
        fontsize=11, fontweight="bold",
    )
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CELLCHAT_POINT,
               markeredgecolor="white", markersize=8, label="CellChat rank"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PAIR_CLASS_COLORS["anchor"],
               markeredgecolor="white", markersize=9, label="Shared anchor"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PAIR_CLASS_COLORS["promoted"],
               markeredgecolor="white", markersize=9, label="Attention-promoted"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PAIR_CLASS_COLORS["deprioritized"],
               markeredgecolor="white", markersize=9, label="Attention-deprioritized"),
    ]
    ax.legend(
        handles=legend_handles, loc="upper center",
        bbox_to_anchor=(0.55, -0.08), frameon=False, ncol=2,
        fontsize=9.5, columnspacing=1.0, handletextpad=0.4,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUTPUT_DIR / "figureA_cellchat_rank_shift_dumbbell.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "figureA_cellchat_rank_shift_dumbbell.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


def draw_interface_overlay(ax: plt.Axes, coords_plot: pd.DataFrame, interface_spots: set[str]) -> None:
    visible = sorted(interface_spots.intersection(coords_plot.index))
    if not visible:
        return
    points = coords_plot.loc[visible, ["x", "y"]].to_numpy(dtype=float)
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=22,
        c=INTERFACE_COLOR,
        alpha=0.28,
        edgecolors="white",
        linewidths=0.25,
        zorder=2,
        rasterized=True,
    )
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


def draw_spatial_pair(
    ax: plt.Axes,
    pair_name: str,
    pair_rank_df: pd.DataFrame,
    spagraph_df: pd.DataFrame,
    adata: sc.AnnData,
    coords: pd.DataFrame,
    interface_spots: set[str],
) -> None:
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
    ax.scatter(
        coords_plot["x"],
        coords_plot["y"],
        s=8,
        c=BACKGROUND_SPOT,
        alpha=0.45,
        edgecolors="none",
        zorder=1,
        rasterized=True,
    )
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
        patch = FancyArrowPatch(
            (sx, sy),
            (dx, dy),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=4.0,
            linewidth=float(width),
            color=LINE_COLOR,
            alpha=0.42,
            shrinkA=2.0,
            shrinkB=2.0,
            zorder=4,
        )
        ax.add_patch(patch)
        ax.scatter(
            sx,
            sy,
            s=18,
            color=SRC_COLOR,
            marker="o",
            edgecolor="white",
            linewidth=0.3,
            zorder=5,
        )
        ax.scatter(
            dx,
            dy,
            s=18,
            color=DST_COLOR,
            marker="o",
            edgecolor="white",
            linewidth=0.3,
            zorder=6,
        )

    rank_row = pair_rank_df.loc[pair_rank_df["lr_pair"] == pair_name].iloc[0]
    n_shared = len(pair_rank_df)
    role = dict(SELECTED_PAIRS).get(pair_name, "")
    role_labels = {"anchor": "shared anchor", "promoted": "attention-promoted",
                   "deprioritized": "attention-deprioritized"}
    role_text = role_labels.get(role, "")
    title = format_pair(pair_name)
    if role_text:
        title += f"  ({role_text})"
    ax.text(
        0.5, 1.075, title,
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=12, fontweight="bold", color="#111827",
    )
    subtitle = (
        f"Spagraph #{int(rank_row.spagraph_rank)}"
        f" / CellChat #{int(rank_row.cellchat_rank)}"
        f"  (of {n_shared} shared pairs)"
    )
    ax.text(
        0.5, 1.03, subtitle,
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=9, color="#6B7280",
    )
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")


def draw_spatial_montage(
    pair_rank_df: pd.DataFrame,
    spagraph_df: pd.DataFrame,
    adata: sc.AnnData,
    coords: pd.DataFrame,
    interface_spots: set[str],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, len(SPATIAL_PAIRS), figsize=(5.2 * len(SPATIAL_PAIRS), 5.2))
    for ax, pair_name in zip(axes, SPATIAL_PAIRS):
        draw_spatial_pair(ax, pair_name, pair_rank_df, spagraph_df, adata, coords, interface_spots)

    handles = [
        Line2D([0], [0], color=LINE_COLOR, linewidth=1.6, label="Top attention-ranked interactions"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SRC_COLOR, markeredgecolor="white", markersize=8, label="Ligand"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=DST_COLOR, markeredgecolor="white", markersize=8, label="Receptor"),
        Line2D([0], [0], color=INTERFACE_COLOR, linewidth=2.0, alpha=0.5, label="Tumor-stroma interface"),
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=False,
        fontsize=9.2,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(OUTPUT_DIR / "figureB_spatial_pair_montage.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "figureB_spatial_pair_montage.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spagraph_df, cellchat_df, adata, coords, composition = load_inputs()
    pair_rank_df = build_pair_rank_table(spagraph_df, cellchat_df)
    save_rank_tables(pair_rank_df)
    draw_rank_shift(pair_rank_df)
    interface_spots = build_interface_spots(composition, coords)
    draw_spatial_montage(pair_rank_df, spagraph_df, adata, coords, interface_spots)
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
