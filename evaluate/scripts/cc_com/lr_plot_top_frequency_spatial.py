import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import gc
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from scipy.spatial import distance_matrix
from sklearn.neighbors import kneighbors_graph


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATE_DIR = SCRIPT_DIR.parent.parent
DATA_ROOT = EVALUATE_DIR / "data"
REPO_ROOT = EVALUATE_DIR.parent
DATABASE_ROOT = REPO_ROOT / "spagraph_data" / "database"


def _require_existing(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")


def _require_columns(df: pd.DataFrame, required: list[str], *, df_name: str = "df") -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {df_name}: {missing}")


def _stable_seed(*parts: str) -> int:
    joined = "||".join(parts).encode("utf-8")
    digest = hashlib.sha256(joined).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)


# ================= Dataset config =================
# GSE243275
# DATA_DIR = DATA_ROOT / "GSE243275"
# ST_H5AD_PATH = DATABASE_ROOT / "GSE243275" / "GSM7782699_ST.h5ad"
# coord_exchange = False

# GSE144236
DATA_DIR = DATA_ROOT / "GSE144236"
ST_H5AD_PATH = DATABASE_ROOT / "GSE144240" / "GSE144236_P2_ST.h5ad"
coord_exchange = False

# GSE211956_P2
# DATA_DIR = DATA_ROOT / "GSE211956" / "P2"
# ST_H5AD_PATH = DATABASE_ROOT / "GSE211956" / "GSE211956_ST_P2.h5ad"
# coord_exchange = True

# GSE211956_P3
# DATA_DIR = DATA_ROOT / "GSE211956" / "P3"
# ST_H5AD_PATH = DATABASE_ROOT / "GSE211956" / "GSE211956_ST_P3.h5ad"
# coord_exchange = False

LR_COMM_PATH = DATA_DIR / "lr_communication.csv"
FIGURE_ROOT = DATA_DIR / "figures"
ATTENTION_OUTPUT_DIR = FIGURE_ROOT / "top_attention_lr_pairs"
FREQUENCY_OUTPUT_DIR = FIGURE_ROOT / "top_frequency_lr_pairs"
INTERFACE_MAP_OUTPUT_DIR = FIGURE_ROOT / "pathology_interface"
ATTENTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FREQUENCY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INTERFACE_MAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_PAIRS_PER_GROUP = 5
TOP_EDGES_PER_PAIR = 500
MIN_EVENT_COUNT = 10
OFFSET_RANGE = 20.0
SCORE_COLUMNS = ("original_lr_score", "attention_score")
PATHOLOGY_REGION_NAME = "Epithelial-Stromal Interface"
PATHOLOGY_REGION_FILL = "#FFD166"
PATHOLOGY_REGION_EDGE = "#C47F00"
PATHOLOGY_REGION_MAX_COMPONENTS = 2
PATHOLOGY_REGION_POINT_SIZE = 72
MAX_EDGES_PER_SPOT_PAIR = 5
INTERFACE_MAP_FIGSIZE = (6.6, 5.8)
INTERFACE_CLASS_COLORS = {
    "Fibroblast-rich": "#E0E0E0",
    "Epithelial-rich": "#4C2A85",
    "Epithelial-Stromal Interface": "#FFD700",
}
FIG_SIZE = (9.4, 6.2)
TITLE_FONT_SIZE = 16
TITLE_PAD = 12
LEGEND_FONT_SIZE = 10
LEGEND_TITLE_FONT_SIZE = 11
LEGEND_MARKER_SIZE = 9
OVERLAY_LEGEND_FONT_SIZE = 9


print("Loading data...")
_require_existing(LR_COMM_PATH)
_require_existing(ST_H5AD_PATH)

df = pd.read_csv(LR_COMM_PATH)
_require_columns(
    df,
    [
        "lr_pair",
        "original_lr_score",
        "attention_score",
        "source_cell",
        "target_cell",
        "src_spot_barcode",
        "dst_spot_barcode",
    ],
    df_name=str(LR_COMM_PATH),
)

df["original_lr_score"] = pd.to_numeric(df["original_lr_score"], errors="coerce").fillna(0.0)
df["attention_score"] = pd.to_numeric(df["attention_score"], errors="coerce").fillna(0.0)
df["src_spot_barcode"] = df["src_spot_barcode"].astype(str)
df["dst_spot_barcode"] = df["dst_spot_barcode"].astype(str)

adata = sc.read_h5ad(ST_H5AD_PATH)
if "spatial" not in getattr(adata, "obsm", {}):
    raise ValueError(f"adata.obsm['spatial'] not found in {ST_H5AD_PATH}")

coords = pd.DataFrame(
    adata.obsm["spatial"],
    index=adata.obs_names.astype(str),
    columns=["x", "y"],
).astype(float)
if coord_exchange:
    coords[["x", "y"]] = coords[["y", "x"]].values

scale_factor = 1.0
if "spatial" in adata.uns:
    keys = list(adata.uns["spatial"].keys())
    scale_factor = adata.uns["spatial"][keys[0]]["scalefactors"].get(
        "tissue_hires_scalef",
        1.0,
    )


def _build_pathology_region_overlay() -> tuple[str | None, list[dict[str, set[str]]], pd.DataFrame | None]:
    if DATA_DIR.name != "GSE144236":
        return None, [], None

    composition_path = DATA_DIR / "Spatial_composition.csv"
    if not composition_path.exists():
        return None, [], None

    composition = pd.read_csv(composition_path, index_col=0)
    required = ["Epithelial", "Fibroblast", "Mac", "CD1C", "Tcell"]
    missing = [col for col in required if col not in composition.columns]
    if missing:
        print(f"Skip pathology region overlay: missing composition columns {missing}")
        return None, [], None

    composition.index = composition.index.astype(str)
    common_spots = composition.index.intersection(coords.index.astype(str))
    if len(common_spots) == 0:
        return None, [], None
    composition = composition.loc[common_spots].copy()

    epithelial = composition["Epithelial"].astype(float)
    fibroblast = composition["Fibroblast"].astype(float)
    immune = (
        composition["Mac"].astype(float)
        + composition["CD1C"].astype(float)
        + composition["Tcell"].astype(float)
    )

    interface_score = epithelial * fibroblast * np.maximum(immune, 0.05)
    if float(interface_score.max()) > float(interface_score.min()):
        interface_scaled = (interface_score - interface_score.min()) / (interface_score.max() - interface_score.min())
    else:
        interface_scaled = pd.Series(np.zeros(len(interface_score), dtype=float), index=interface_score.index)

    boundary_threshold = float(np.percentile(interface_scaled.to_numpy(), 95))
    interface_mask = interface_scaled >= boundary_threshold
    if not interface_mask.any():
        boundary_threshold = float(np.percentile(interface_scaled.to_numpy(), 90))
        interface_mask = interface_scaled >= boundary_threshold

    local_coords = coords.loc[common_spots, ["x", "y"]].to_numpy(dtype=float)
    adjacency = kneighbors_graph(
        local_coords,
        n_neighbors=min(8, max(1, len(common_spots) - 1)),
        mode="connectivity",
        include_self=False,
    ).tolil()

    ep_values = epithelial.to_numpy()
    fib_values = fibroblast.to_numpy()
    interface_values = interface_mask.to_numpy()

    boundary_spots = composition.index[interface_mask].astype(str)
    boundary_coords = coords.loc[boundary_spots, ["x", "y"]].to_numpy(dtype=float)
    if boundary_coords.size == 0:
        signed_distance = np.where(ep_values >= fib_values, -1.0, 1.0)
    else:
        min_distances = distance_matrix(local_coords, boundary_coords).min(axis=1)
        direction = np.where(ep_values >= fib_values, -1.0, 1.0)
        signed_distance = min_distances * direction
        signed_distance[interface_values] = 0.0

    selected_spots = set(boundary_spots)
    selected_index_lookup = {spot: idx for idx, spot in enumerate(common_spots)}
    visited: set[str] = set()
    components: list[list[str]] = []
    for spot in common_spots:
        if spot not in selected_spots or spot in visited:
            continue
        stack = [spot]
        component: list[str] = []
        visited.add(spot)
        while stack:
            current = stack.pop()
            component.append(current)
            current_idx = selected_index_lookup[current]
            for neighbor_idx in adjacency.rows[current_idx]:
                neighbor_spot = common_spots[neighbor_idx]
                if neighbor_spot in selected_spots and neighbor_spot not in visited:
                    visited.add(neighbor_spot)
                    stack.append(neighbor_spot)
        components.append(component)

    components.sort(key=len, reverse=True)
    kept_components = components[:PATHOLOGY_REGION_MAX_COMPONENTS]
    region_components: list[dict[str, set[str]]] = []
    for component in kept_components:
        component_set = set(component)
        region_components.append({"spots": component_set, "boundary_spots": component_set})

    kept_spots = {spot for item in region_components for spot in item["spots"]}
    is_interface = np.array([spot in kept_spots for spot in common_spots], dtype=bool)
    is_epithelial_rich = (~is_interface) & (signed_distance < 0)
    is_fibroblast_rich = (~is_interface) & (signed_distance > 0)

    print(
        f"Pathology region overlay ({PATHOLOGY_REGION_NAME}): "
        f"{len(kept_spots)} interface spots across {len(region_components)} components "
        f"[boundary_threshold={boundary_threshold:.3f}]"
    )
    region_df = pd.DataFrame(
        {
            "spot": common_spots,
            "x": coords.loc[common_spots, "x"].astype(float).to_numpy() * scale_factor,
            "y": coords.loc[common_spots, "y"].astype(float).to_numpy() * scale_factor,
            "is_epithelial_rich": is_epithelial_rich,
            "is_fibroblast_rich": is_fibroblast_rich,
            "is_interface": is_interface,
            "signed_distance": signed_distance,
            "interface_score_scaled": interface_scaled.to_numpy(),
        }
    )
    return PATHOLOGY_REGION_NAME, region_components, region_df


PATHOLOGY_REGION_LABEL, PATHOLOGY_REGION_COMPONENTS, PATHOLOGY_REGION_DF = _build_pathology_region_overlay()


def plot_pathology_interface_map() -> None:
    if PATHOLOGY_REGION_LABEL is None or PATHOLOGY_REGION_DF is None:
        return

    fig, ax = plt.subplots(figsize=INTERFACE_MAP_FIGSIZE)
    sc.pl.spatial(adata, color=None, alpha_img=0.14, size=0.1, show=False, ax=ax)

    region_df = PATHOLOGY_REGION_DF.copy()
    interface_df = region_df.loc[region_df["is_interface"]].copy()
    epithelial_df = region_df.loc[region_df["is_epithelial_rich"] & ~region_df["is_interface"]].copy()
    fibroblast_df = region_df.loc[region_df["is_fibroblast_rich"] & ~region_df["is_interface"]].copy()

    if not fibroblast_df.empty:
        ax.scatter(
            fibroblast_df["x"],
            fibroblast_df["y"],
            s=68,
            c=INTERFACE_CLASS_COLORS["Fibroblast-rich"],
            edgecolors="none",
            alpha=0.95,
            zorder=3,
        )
    if not epithelial_df.empty:
        ax.scatter(
            epithelial_df["x"],
            epithelial_df["y"],
            s=68,
            c=INTERFACE_CLASS_COLORS["Epithelial-rich"],
            edgecolors="none",
            alpha=0.98,
            zorder=4,
        )
    if not interface_df.empty:
        ax.scatter(
            interface_df["x"],
            interface_df["y"],
            s=78,
            c=INTERFACE_CLASS_COLORS["Epithelial-Stromal Interface"],
            edgecolors="#A66B00",
            linewidths=0.5,
            alpha=0.98,
            zorder=5,
        )

    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.axis("off")
    ax.set_title("Spatial Mapping of the Epithelial-Stromal Interface", fontsize=15, pad=10)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=INTERFACE_CLASS_COLORS["Fibroblast-rich"], markeredgecolor="none", markersize=8, linewidth=0, label="Fibroblast-rich"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=INTERFACE_CLASS_COLORS["Epithelial-rich"], markeredgecolor="none", markersize=8, linewidth=0, label="Epithelial-rich"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=INTERFACE_CLASS_COLORS["Epithelial-Stromal Interface"], markeredgecolor="#A66B00", markersize=8, linewidth=0, label="Interface"),
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
        framealpha=0.92,
        edgecolor="#D0D0D0",
        fontsize=10,
        handletextpad=0.4,
        labelspacing=0.4,
    )

    save_path = INTERFACE_MAP_OUTPUT_DIR / "epithelial_stromal_interface_map.pdf"
    plt.tight_layout(pad=0.4)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    print(f"Saved: {save_path}")


def _select_nearest_pathology_boundary(
    edges: pd.DataFrame,
) -> set[str]:
    if not PATHOLOGY_REGION_COMPONENTS or edges.empty:
        return set()

    event_points = np.concatenate(
        [
            edges[["src_x", "src_y"]].to_numpy(dtype=float),
            edges[["dst_x", "dst_y"]].to_numpy(dtype=float),
        ],
        axis=0,
    )
    event_center = np.nanmean(event_points, axis=0)

    best_boundary: set[str] = set()
    best_distance = float("inf")
    for component in PATHOLOGY_REGION_COMPONENTS:
        component_spots = sorted(component["spots"])
        component_points = coords.loc[component_spots, ["x", "y"]].to_numpy(dtype=float)
        component_center = np.nanmean(component_points, axis=0)
        distance = float(np.linalg.norm(component_center - event_center))
        if distance < best_distance:
            best_distance = distance
            best_boundary = component["boundary_spots"]
    return best_boundary

pair_stats = (
    df.groupby("lr_pair", as_index=False)
    .agg(
        occurrence_count=("lr_pair", "size"),
        attention_mean=("attention_score", "mean"),
        attention_sum=("attention_score", "sum"),
        original_lr_sum=("original_lr_score", "sum"),
    )
    .sort_values("lr_pair")
    .reset_index(drop=True)
)
eligible_pairs = pair_stats.loc[pair_stats["occurrence_count"] >= MIN_EVENT_COUNT].copy()
if eligible_pairs.empty:
    raise ValueError(
        f"No LR pairs satisfy occurrence_count >= {MIN_EVENT_COUNT} in {LR_COMM_PATH}."
    )

df_event_filtered = df[df["lr_pair"].isin(eligible_pairs["lr_pair"])].copy()


def _select_top_pairs(stats_df: pd.DataFrame, ranking_type: str) -> pd.DataFrame:
    if ranking_type == "attention":
        ranked = stats_df.sort_values(
            ["attention_mean", "occurrence_count", "attention_sum", "lr_pair"],
            ascending=[False, False, False, True],
        )
    elif ranking_type == "frequency":
        ranked = stats_df.sort_values(
            ["occurrence_count", "attention_mean", "attention_sum", "lr_pair"],
            ascending=[False, False, False, True],
        )
    else:
        raise ValueError(f"Unknown ranking_type: {ranking_type}")
    result = ranked.head(TOP_PAIRS_PER_GROUP).copy().reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    result.insert(0, "ranking_type", ranking_type)
    return result


attention_pairs = _select_top_pairs(eligible_pairs, "attention")
frequency_pairs = _select_top_pairs(eligible_pairs, "frequency")
attention_pairs.to_csv(ATTENTION_OUTPUT_DIR / "top_attention_lr_pairs.csv", index=False)
frequency_pairs.to_csv(FREQUENCY_OUTPUT_DIR / "top_frequency_lr_pairs.csv", index=False)

print(
    f"Selected {len(attention_pairs)} top-attention pairs and "
    f"{len(frequency_pairs)} top-frequency pairs "
    f"(min_event_count >= {MIN_EVENT_COUNT})."
)

global_cell_to_marker: dict[str, str] = {}
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "H", "*"]


def plot_edges(
    sub_df: pd.DataFrame,
    lr_name: str,
    score_col: str,
    *,
    output_dir: Path,
    rank: int,
) -> None:
    if sub_df.empty:
        print(f"Skip {lr_name} ({score_col}): no events found")
        return

    edges = (
        sub_df.sort_values(score_col, ascending=False)
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
        print(f"Skip {lr_name} ({score_col}): no edges with valid spatial coordinates")
        return

    scores = edges[score_col].astype(float).to_numpy()
    score_min = float(np.nanmin(scores))
    score_max = float(np.nanmax(scores))
    if not np.isfinite(score_min) or not np.isfinite(score_max):
        print(f"Skip {lr_name} ({score_col}): invalid scores")
        return

    if score_max > score_min:
        norm_scores = (scores - score_min) / (score_max - score_min + 1e-12)
    else:
        norm_scores = np.zeros_like(scores, dtype=float)
    widths = 1.0 + 1.8 * norm_scores

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sc.pl.spatial(adata, color=None, alpha_img=0.4, size=0.1, show=False, ax=ax)

    coords_plot = coords.copy()
    coords_plot[["x", "y"]] = coords_plot[["x", "y"]] * scale_factor
    ax.scatter(coords_plot["x"], coords_plot["y"], s=16, c="lightgray", alpha=0.15, linewidth=0, zorder=2)

    show_pathology_region = (
        score_col == "attention_score"
        and PATHOLOGY_REGION_LABEL is not None
        and bool(PATHOLOGY_REGION_COMPONENTS)
    )
    if show_pathology_region:
        region_boundary_spots = _select_nearest_pathology_boundary(edges)
        region_spots = sorted(region_boundary_spots.intersection(coords_plot.index.astype(str)))
        if region_spots:
            region_coords = coords_plot.loc[region_spots]
            ax.scatter(
                region_coords["x"],
                region_coords["y"],
                s=PATHOLOGY_REGION_POINT_SIZE,
                c=PATHOLOGY_REGION_FILL,
                alpha=0.30,
                edgecolors=PATHOLOGY_REGION_EDGE,
                linewidths=0.55,
                zorder=3,
            )

    edges["src_x_plot"] = edges["src_x"] * scale_factor
    edges["src_y_plot"] = edges["src_y"] * scale_factor
    edges["dst_x_plot"] = edges["dst_x"] * scale_factor
    edges["dst_y_plot"] = edges["dst_y"] * scale_factor

    distance_sq = (
        (edges["src_x_plot"] - edges["dst_x_plot"]) ** 2
        + (edges["src_y_plot"] - edges["dst_y_plot"]) ** 2
    )
    edges = edges.loc[distance_sq > 1e-12].copy()
    if edges.empty:
        print(f"Skip {lr_name} ({score_col}): all selected edges are self-loops")
        plt.close(fig)
        return

    current_cells = sorted(set(edges["source_cell"]) | set(edges["target_cell"]))
    for cell_type in current_cells:
        if cell_type not in global_cell_to_marker:
            global_cell_to_marker[cell_type] = MARKERS[len(global_cell_to_marker) % len(MARKERS)]

    line_color = "#2E8B57"
    src_color = "#24C7D9"
    dst_color = "#E048C8"
    outline_effect = [pe.Stroke(linewidth=2.1, foreground="black"), pe.Normal()]

    valid_widths = widths[: len(edges)]
    rng = np.random.default_rng(_stable_seed(str(output_dir), lr_name))

    for row, width in zip(edges.itertuples(index=False), valid_widths):
        src_x_jitter = row.src_x_plot + rng.uniform(-OFFSET_RANGE, OFFSET_RANGE)
        src_y_jitter = row.src_y_plot + rng.uniform(-OFFSET_RANGE, OFFSET_RANGE)
        dst_x_jitter = row.dst_x_plot + rng.uniform(-OFFSET_RANGE, OFFSET_RANGE)
        dst_y_jitter = row.dst_y_plot + rng.uniform(-OFFSET_RANGE, OFFSET_RANGE)

        src_marker = global_cell_to_marker[row.source_cell]
        dst_marker = global_cell_to_marker[row.target_cell]
        ax.scatter(
            src_x_jitter,
            src_y_jitter,
            s=32,
            color=src_color,
            marker=src_marker,
            edgecolor="black",
            linewidth=0.35,
            zorder=6,
        )
        ax.scatter(
            dst_x_jitter,
            dst_y_jitter,
            s=32,
            color=dst_color,
            marker=dst_marker,
            edgecolor="black",
            linewidth=0.35,
            zorder=6,
        )

        rad = 0.2 if dst_x_jitter > src_x_jitter else -0.2
        patch = FancyArrowPatch(
            (src_x_jitter, src_y_jitter),
            (dst_x_jitter, dst_y_jitter),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-",
            linewidth=float(width),
            color=line_color,
            alpha=0.72,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=5,
        )
        patch.set_path_effects(outline_effect)
        ax.add_patch(patch)

    if not ax.yaxis_inverted():
        ax.invert_yaxis()

    ax.set_title(
        f"{lr_name}\n({score_col})",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        pad=TITLE_PAD,
    )
    ax.axis("off")

    celltype_elements = [
        Line2D(
            [0],
            [0],
            marker=global_cell_to_marker[cell_type],
            color="w",
            markerfacecolor="#B8B8B8",
            markeredgecolor="black",
            markersize=LEGEND_MARKER_SIZE,
            linewidth=0,
            label=cell_type,
        )
        for cell_type in current_cells
    ]
    if celltype_elements:
        cell_legend = ax.legend(
            handles=celltype_elements,
            loc="upper left",
            bbox_to_anchor=(1.02, 0.98),
            framealpha=0.92,
            fancybox=True,
            edgecolor="#D0D0D0",
            fontsize=LEGEND_FONT_SIZE,
            title="Cell Type",
            title_fontsize=LEGEND_TITLE_FONT_SIZE,
            labelspacing=0.35,
            handletextpad=0.45,
            borderpad=0.45,
        )
        ax.add_artist(cell_legend)

    overlay_elements = [
        Line2D([0], [0], color=line_color, linewidth=2.0, label="Interaction edge"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=src_color,
            markeredgecolor="black",
            markersize=LEGEND_MARKER_SIZE - 1,
            linewidth=0,
            label="Ligand side",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=dst_color,
            markeredgecolor="black",
            markersize=LEGEND_MARKER_SIZE - 1,
            linewidth=0,
            label="Receptor side",
        ),
    ]
    if show_pathology_region:
        overlay_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=PATHOLOGY_REGION_FILL,
                markeredgecolor=PATHOLOGY_REGION_EDGE,
                markersize=LEGEND_MARKER_SIZE,
                linewidth=0,
                label=PATHOLOGY_REGION_LABEL,
            )
        )

    overlay_legend = ax.legend(
        handles=overlay_elements,
        loc="lower left",
        bbox_to_anchor=(1.02, 0.04),
        framealpha=0.90,
        fancybox=True,
        edgecolor="#D0D0D0",
        fontsize=OVERLAY_LEGEND_FONT_SIZE,
        labelspacing=0.30,
        handletextpad=0.45,
        borderpad=0.40,
    )
    ax.add_artist(overlay_legend)

    safe_name = lr_name.replace("/", "_")
    save_path = output_dir / f"{rank:02d}_{safe_name}_{score_col}.pdf"
    plt.tight_layout(pad=0.5, rect=(0.0, 0.0, 0.82, 1.0))
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    print(f"Saved: {save_path}")


def _plot_group(selection_df: pd.DataFrame, output_dir: Path, group_name: str) -> None:
    print(f"\nPlotting {group_name} LR pairs...")
    for row in selection_df.itertuples(index=False):
        pair_name = row.lr_pair
        rank = int(row.rank)
        pair_df = df_event_filtered[df_event_filtered["lr_pair"] == pair_name].copy()
        print(f"  {rank}. {pair_name} ({len(pair_df)} events)")
        for score_column in SCORE_COLUMNS:
            plot_edges(pair_df, pair_name, score_column, output_dir=output_dir, rank=rank)


plot_pathology_interface_map()
_plot_group(attention_pairs, ATTENTION_OUTPUT_DIR, "top attention")
_plot_group(frequency_pairs, FREQUENCY_OUTPUT_DIR, "top frequency")

print(f"Done! Top-attention LR pair plots saved to {ATTENTION_OUTPUT_DIR}")
print(f"Done! Top-frequency LR pair plots saved to {FREQUENCY_OUTPUT_DIR}")
