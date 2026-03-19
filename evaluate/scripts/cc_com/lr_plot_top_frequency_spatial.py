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
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.spatial import ConvexHull, Delaunay
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


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
DATA_DIR = DATA_ROOT / "GSE243275"
ST_H5AD_PATH = DATABASE_ROOT / "GSE243275" / "GSM7782699_ST.h5ad"
coord_exchange = False

# GSE144236
# DATA_DIR = DATA_ROOT / "GSE144236"
# ST_H5AD_PATH = DATABASE_ROOT / "GSE144240" / "GSE144236_P2_ST.h5ad"
# coord_exchange = False

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
ATTENTION_REGION_OUTPUT_DIR = FIGURE_ROOT / "top_attention_lr_pair_regions"
FREQUENCY_REGION_OUTPUT_DIR = FIGURE_ROOT / "top_frequency_lr_pair_regions"
ATTENTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FREQUENCY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ATTENTION_REGION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FREQUENCY_REGION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_PAIRS_PER_GROUP = 5
TOP_EDGES_PER_PAIR = 5000
MIN_EVENT_COUNT = 10
OFFSET_RANGE = 20.0
SCORE_COLUMNS = ("original_lr_score", "attention_score")
REGION_SCORE_COLUMNS = ("attention_score", "original_lr_score")
HOTSPOT_QUANTILE = 0.90
MIN_HOTSPOT_SPOTS = 12
DBSCAN_MIN_SAMPLES = 3
DBSCAN_EPS_FACTOR = 1.75
ALPHA_RADIUS_FACTOR = 2.50
REPRESENTATIVE_EDGES = 5
MAX_REGION_CLUSTERS = 2
ZOOM_MARGIN_RATIO = 0.18
ATTENTION_REGION_COLOR = "#228B22"
ORIGINAL_REGION_COLOR = "#F4A300"


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


def _build_spot_activity_map(sub_df: pd.DataFrame, score_col: str) -> pd.Series:
    src_scores = sub_df.groupby("src_spot_barcode")[score_col].sum()
    dst_scores = sub_df.groupby("dst_spot_barcode")[score_col].sum()
    spot_scores = src_scores.add(dst_scores, fill_value=0.0)
    return spot_scores.reindex(coords.index.astype(str), fill_value=0.0).astype(float)


def _estimate_local_scale(points: np.ndarray) -> float:
    if len(points) <= 1:
        return 1.0
    n_neighbors = min(3, len(points))
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(points)
    distances, _ = nn.kneighbors(points)
    neighbor_idx = 1 if distances.shape[1] > 1 else 0
    positive_distances = distances[:, neighbor_idx]
    positive_distances = positive_distances[positive_distances > 0]
    if positive_distances.size == 0:
        return 1.0
    return float(np.median(positive_distances))


def _select_hotspot_spots(spot_scores: pd.Series) -> pd.Series:
    positive_scores = spot_scores[spot_scores > 0].sort_values(ascending=False)
    if positive_scores.empty:
        return positive_scores

    threshold = float(positive_scores.quantile(HOTSPOT_QUANTILE))
    selected = positive_scores[positive_scores >= threshold]
    if len(selected) < MIN_HOTSPOT_SPOTS:
        selected = positive_scores.head(min(MIN_HOTSPOT_SPOTS, len(positive_scores)))
    return selected


def _cluster_hotspot_spots(selected_scores: pd.Series) -> list[dict[str, object]]:
    if selected_scores.empty:
        return []

    point_df = coords.loc[selected_scores.index].copy()
    point_df["score"] = selected_scores.values
    point_df[["x", "y"]] = point_df[["x", "y"]] * scale_factor

    points = point_df[["x", "y"]].to_numpy(dtype=float)
    if len(points) < DBSCAN_MIN_SAMPLES:
        return [
            {
                "spot_ids": point_df.index.to_list(),
                "points": points,
                "score_sum": float(point_df["score"].sum()),
                "bbox": (
                    float(point_df["x"].min()),
                    float(point_df["x"].max()),
                    float(point_df["y"].min()),
                    float(point_df["y"].max()),
                ),
            }
        ]

    local_scale = _estimate_local_scale(points)
    eps = max(local_scale * DBSCAN_EPS_FACTOR, 1.0)
    labels = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(points)
    if np.all(labels == -1):
        labels = np.zeros(len(points), dtype=int)

    clusters: list[dict[str, object]] = []
    for label in sorted(set(labels)):
        mask = labels == label if label != -1 else labels == -1
        cluster_df = point_df.loc[mask].copy()
        if cluster_df.empty:
            continue
        clusters.append(
            {
                "spot_ids": cluster_df.index.to_list(),
                "points": cluster_df[["x", "y"]].to_numpy(dtype=float),
                "score_sum": float(cluster_df["score"].sum()),
                "bbox": (
                    float(cluster_df["x"].min()),
                    float(cluster_df["x"].max()),
                    float(cluster_df["y"].min()),
                    float(cluster_df["y"].max()),
                ),
            }
        )

    clusters.sort(key=lambda item: (item["score_sum"], len(item["spot_ids"])), reverse=True)
    return clusters[:MAX_REGION_CLUSTERS]


def _triangle_circumradius(triangle: np.ndarray) -> float:
    side_a = np.linalg.norm(triangle[1] - triangle[0])
    side_b = np.linalg.norm(triangle[2] - triangle[1])
    side_c = np.linalg.norm(triangle[0] - triangle[2])
    area_twice = abs(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
    if area_twice <= 1e-12:
        return float("inf")
    area = area_twice / 2.0
    return float(side_a * side_b * side_c / (4.0 * area))


def _build_region_geometry(cluster_points: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if len(cluster_points) < 3:
        return [], []

    fill_polygons: list[np.ndarray] = []
    boundary_segments: list[np.ndarray] = []

    if len(cluster_points) == 3:
        fill_polygons = [cluster_points]
        boundary_segments = [
            cluster_points[[0, 1]],
            cluster_points[[1, 2]],
            cluster_points[[2, 0]],
        ]
        return fill_polygons, boundary_segments

    local_scale = _estimate_local_scale(cluster_points)
    alpha_radius = max(local_scale * ALPHA_RADIUS_FACTOR, 1.0)
    try:
        triangulation = Delaunay(cluster_points)
    except Exception:
        hull = ConvexHull(cluster_points)
        hull_points = cluster_points[hull.vertices]
        fill_polygons = [hull_points]
        for start_idx in range(len(hull_points)):
            end_idx = (start_idx + 1) % len(hull_points)
            boundary_segments.append(hull_points[[start_idx, end_idx]])
        return fill_polygons, boundary_segments

    kept_simplices: list[np.ndarray] = []
    edge_counts: dict[tuple[int, int], int] = {}
    for simplex in triangulation.simplices:
        triangle = cluster_points[simplex]
        if _triangle_circumradius(triangle) > alpha_radius:
            continue
        kept_simplices.append(simplex)
        for i, j in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(simplex[i]), int(simplex[j]))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    if kept_simplices:
        fill_polygons = [cluster_points[simplex] for simplex in kept_simplices]
        boundary_segments = [cluster_points[list(edge)] for edge, count in edge_counts.items() if count == 1]
        return fill_polygons, boundary_segments

    hull = ConvexHull(cluster_points)
    hull_points = cluster_points[hull.vertices]
    fill_polygons = [hull_points]
    for start_idx in range(len(hull_points)):
        end_idx = (start_idx + 1) % len(hull_points)
        boundary_segments.append(hull_points[[start_idx, end_idx]])
    return fill_polygons, boundary_segments


def _build_region_bundle(sub_df: pd.DataFrame, score_col: str) -> dict[str, object]:
    spot_scores = _build_spot_activity_map(sub_df, score_col)
    selected_scores = _select_hotspot_spots(spot_scores)
    clusters = _cluster_hotspot_spots(selected_scores)
    region_clusters: list[dict[str, object]] = []

    for cluster in clusters:
        fill_polygons, boundary_segments = _build_region_geometry(cluster["points"])
        region_clusters.append(
            {
                **cluster,
                "fill_polygons": fill_polygons,
                "boundary_segments": boundary_segments,
            }
        )

    return {
        "score_col": score_col,
        "spot_scores": spot_scores,
        "selected_scores": selected_scores,
        "clusters": region_clusters,
    }


def _select_representative_edges(
    sub_df: pd.DataFrame,
    score_col: str,
    focus_spots: set[str],
) -> pd.DataFrame:
    candidate_edges = sub_df.copy()
    if focus_spots:
        both_inside = candidate_edges["src_spot_barcode"].isin(focus_spots) & candidate_edges["dst_spot_barcode"].isin(
            focus_spots
        )
        inside_edges = candidate_edges.loc[both_inside].copy()
        if len(inside_edges) >= REPRESENTATIVE_EDGES:
            candidate_edges = inside_edges
        else:
            either_inside = candidate_edges["src_spot_barcode"].isin(focus_spots) | candidate_edges["dst_spot_barcode"].isin(
                focus_spots
            )
            candidate_edges = candidate_edges.loc[either_inside].copy()

    candidate_edges = candidate_edges.sort_values(score_col, ascending=False).head(REPRESENTATIVE_EDGES).copy()
    candidate_edges["src_x"] = candidate_edges["src_spot_barcode"].map(coords["x"]) * scale_factor
    candidate_edges["src_y"] = candidate_edges["src_spot_barcode"].map(coords["y"]) * scale_factor
    candidate_edges["dst_x"] = candidate_edges["dst_spot_barcode"].map(coords["x"]) * scale_factor
    candidate_edges["dst_y"] = candidate_edges["dst_spot_barcode"].map(coords["y"]) * scale_factor
    candidate_edges = candidate_edges.dropna(subset=["src_x", "src_y", "dst_x", "dst_y"])
    candidate_edges = candidate_edges.loc[
        ((candidate_edges["src_x"] - candidate_edges["dst_x"]) ** 2 + (candidate_edges["src_y"] - candidate_edges["dst_y"]) ** 2)
        > 1e-12
    ].copy()
    return candidate_edges


def _draw_region_bundle(
    ax: plt.Axes,
    region_bundle: dict[str, object],
    *,
    line_color: str,
    line_style: str,
    fill_alpha: float,
) -> None:
    for cluster in region_bundle["clusters"]:
        fill_polygons = cluster["fill_polygons"]
        boundary_segments = cluster["boundary_segments"]
        if fill_polygons:
            collection = PolyCollection(
                fill_polygons,
                facecolors=line_color,
                edgecolors="none",
                alpha=fill_alpha,
                zorder=4,
            )
            ax.add_collection(collection)
        if boundary_segments:
            collection = LineCollection(
                boundary_segments,
                colors=line_color,
                linewidths=2.8,
                linestyles=line_style,
                zorder=5,
            )
            collection.set_path_effects([pe.Stroke(linewidth=4.0, foreground="white"), pe.Normal()])
            ax.add_collection(collection)


def _draw_representative_edges(
    ax: plt.Axes,
    edges: pd.DataFrame,
    *,
    edge_color: str,
) -> None:
    if edges.empty:
        return

    rng = np.random.default_rng(_stable_seed(edge_color, str(len(edges))))
    for row in edges.itertuples(index=False):
        src_x = float(row.src_x) + rng.uniform(-OFFSET_RANGE * 0.15, OFFSET_RANGE * 0.15)
        src_y = float(row.src_y) + rng.uniform(-OFFSET_RANGE * 0.15, OFFSET_RANGE * 0.15)
        dst_x = float(row.dst_x) + rng.uniform(-OFFSET_RANGE * 0.15, OFFSET_RANGE * 0.15)
        dst_y = float(row.dst_y) + rng.uniform(-OFFSET_RANGE * 0.15, OFFSET_RANGE * 0.15)
        patch = FancyArrowPatch(
            (src_x, src_y),
            (dst_x, dst_y),
            arrowstyle="->",
            mutation_scale=12,
            linewidth=2.0,
            color=edge_color,
            alpha=0.9,
            shrinkA=2.0,
            shrinkB=2.0,
            zorder=6,
        )
        patch.set_path_effects([pe.Stroke(linewidth=3.0, foreground="white"), pe.Normal()])
        ax.add_patch(patch)
        ax.scatter([src_x, dst_x], [src_y, dst_y], s=18, c=edge_color, alpha=0.9, zorder=7, linewidth=0)


def _apply_background(ax: plt.Axes) -> None:
    sc.pl.spatial(adata, color=None, alpha_img=0.4, size=0.1, show=False, ax=ax)
    coords_plot = coords.copy()
    coords_plot[["x", "y"]] = coords_plot[["x", "y"]] * scale_factor
    ax.scatter(coords_plot["x"], coords_plot["y"], s=14, c="lightgray", alpha=0.12, linewidth=0, zorder=2)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.axis("off")


def _set_zoom(ax: plt.Axes, bbox: tuple[float, float, float, float]) -> None:
    x_min, x_max, y_min, y_max = bbox
    x_margin = max((x_max - x_min) * ZOOM_MARGIN_RATIO, 40.0)
    y_margin = max((y_max - y_min) * ZOOM_MARGIN_RATIO, 40.0)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_max + y_margin, y_min - y_margin)


def plot_region_prototype(
    sub_df: pd.DataFrame,
    lr_name: str,
    *,
    output_dir: Path,
    rank: int,
) -> None:
    if sub_df.empty:
        print(f"Skip region prototype for {lr_name}: no events found")
        return

    attention_bundle = _build_region_bundle(sub_df, "attention_score")
    original_bundle = _build_region_bundle(sub_df, "original_lr_score")
    if not attention_bundle["clusters"] and not original_bundle["clusters"]:
        print(f"Skip region prototype for {lr_name}: no hotspot regions found")
        return

    attention_focus_spots = set(attention_bundle["clusters"][0]["spot_ids"]) if attention_bundle["clusters"] else set()
    original_focus_spots = set(original_bundle["clusters"][0]["spot_ids"]) if original_bundle["clusters"] else set()

    attention_edges = _select_representative_edges(sub_df, "attention_score", attention_focus_spots)
    original_edges = _select_representative_edges(sub_df, "original_lr_score", original_focus_spots)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ["Attention Region", "Original LR Region", "Zoom-In View"]
    for ax, title in zip(axes, titles):
        _apply_background(ax)
        ax.set_title(title, fontsize=18, fontweight="bold", pad=14)

    _draw_region_bundle(
        axes[0],
        attention_bundle,
        line_color=ATTENTION_REGION_COLOR,
        line_style="solid",
        fill_alpha=0.22,
    )
    _draw_representative_edges(axes[0], attention_edges, edge_color=ATTENTION_REGION_COLOR)

    _draw_region_bundle(
        axes[1],
        original_bundle,
        line_color=ORIGINAL_REGION_COLOR,
        line_style="dashed",
        fill_alpha=0.18,
    )
    _draw_representative_edges(axes[1], original_edges, edge_color=ORIGINAL_REGION_COLOR)

    _draw_region_bundle(
        axes[2],
        attention_bundle,
        line_color=ATTENTION_REGION_COLOR,
        line_style="solid",
        fill_alpha=0.22,
    )
    _draw_region_bundle(
        axes[2],
        original_bundle,
        line_color=ORIGINAL_REGION_COLOR,
        line_style="dashed",
        fill_alpha=0.10,
    )
    _draw_representative_edges(axes[2], attention_edges.head(3), edge_color=ATTENTION_REGION_COLOR)
    _draw_representative_edges(axes[2], original_edges.head(3), edge_color=ORIGINAL_REGION_COLOR)

    zoom_bbox = None
    if attention_bundle["clusters"]:
        zoom_bbox = attention_bundle["clusters"][0]["bbox"]
    elif original_bundle["clusters"]:
        zoom_bbox = original_bundle["clusters"][0]["bbox"]
    if zoom_bbox is not None:
        _set_zoom(axes[2], zoom_bbox)
        x_min, x_max, y_min, y_max = zoom_bbox
        x_margin = max((x_max - x_min) * ZOOM_MARGIN_RATIO, 40.0)
        y_margin = max((y_max - y_min) * ZOOM_MARGIN_RATIO, 40.0)
        rect = Rectangle(
            (x_min - x_margin, y_min - y_margin),
            (x_max - x_min) + 2 * x_margin,
            (y_max - y_min) + 2 * y_margin,
            linewidth=1.6,
            edgecolor="#4D4D4D",
            facecolor="none",
            zorder=8,
        )
        axes[0].add_patch(rect)

    handles = [
        Line2D([0], [0], color=ATTENTION_REGION_COLOR, linewidth=3.0, linestyle="-", label="Attention Region"),
        Line2D([0], [0], color=ORIGINAL_REGION_COLOR, linewidth=3.0, linestyle="--", label="Original LR Region"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=True, fontsize=14, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(lr_name, fontsize=24, fontweight="bold", y=0.98)

    safe_name = lr_name.replace("/", "_")
    save_path = output_dir / f"{rank:02d}_{safe_name}_region_prototype.pdf"
    plt.tight_layout(rect=(0, 0.04, 1, 0.94))
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    print(f"Saved region prototype: {save_path}")


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

    edges = sub_df.sort_values(score_col, ascending=False).head(TOP_EDGES_PER_PAIR).copy()
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
    widths = 2.0 + 2.5 * norm_scores

    fig, ax = plt.subplots(figsize=(10, 10))
    sc.pl.spatial(adata, color=None, alpha_img=0.4, size=0.1, show=False, ax=ax)

    coords_plot = coords.copy()
    coords_plot[["x", "y"]] = coords_plot[["x", "y"]] * scale_factor
    ax.scatter(coords_plot["x"], coords_plot["y"], s=16, c="lightgray", alpha=0.15, linewidth=0, zorder=2)

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

    line_color = "#32CD32"
    src_color = "#00FFFF"
    dst_color = "#FF00FF"
    outline_effect = [pe.Stroke(linewidth=3.0, foreground="black"), pe.Normal()]

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
            s=40,
            color=src_color,
            marker=src_marker,
            edgecolor="black",
            linewidth=0.5,
            zorder=6,
        )
        ax.scatter(
            dst_x_jitter,
            dst_y_jitter,
            s=40,
            color=dst_color,
            marker=dst_marker,
            edgecolor="black",
            linewidth=0.5,
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
            alpha=0.85,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=5,
        )
        patch.set_path_effects(outline_effect)
        ax.add_patch(patch)

    if not ax.yaxis_inverted():
        ax.invert_yaxis()

    ax.set_title(f"{lr_name}\n({score_col})", fontsize=20, fontweight="bold", pad=20)
    ax.axis("off")

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker=global_cell_to_marker[cell_type],
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=20,
            linewidth=0,
            label=cell_type,
        )
        for cell_type in current_cells
    ]
    if legend_elements:
        ax.legend(
            handles=legend_elements,
            loc="center left",
            fontsize=18,
            framealpha=0.95,
            title="Cell Type",
            title_fontsize=18,
            ncol=1,
            bbox_to_anchor=(1, 0.5),
        )

    safe_name = lr_name.replace("/", "_")
    save_path = output_dir / f"{rank:02d}_{safe_name}_{score_col}.pdf"
    plt.tight_layout()
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


def _plot_region_group(selection_df: pd.DataFrame, output_dir: Path, group_name: str) -> None:
    print(f"\nPlotting {group_name} region prototypes...")
    for row in selection_df.itertuples(index=False):
        pair_name = row.lr_pair
        rank = int(row.rank)
        pair_df = df_event_filtered[df_event_filtered["lr_pair"] == pair_name].copy()
        plot_region_prototype(pair_df, pair_name, output_dir=output_dir, rank=rank)


_plot_group(attention_pairs, ATTENTION_OUTPUT_DIR, "top attention")
_plot_group(frequency_pairs, FREQUENCY_OUTPUT_DIR, "top frequency")
_plot_region_group(attention_pairs, ATTENTION_REGION_OUTPUT_DIR, "top attention")
_plot_region_group(frequency_pairs, FREQUENCY_REGION_OUTPUT_DIR, "top frequency")

print(f"Done! Top-attention LR pair plots saved to {ATTENTION_OUTPUT_DIR}")
print(f"Done! Top-frequency LR pair plots saved to {FREQUENCY_OUTPUT_DIR}")
print(f"Done! Top-attention region prototypes saved to {ATTENTION_REGION_OUTPUT_DIR}")
print(f"Done! Top-frequency region prototypes saved to {FREQUENCY_REGION_OUTPUT_DIR}")
