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
ATTENTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FREQUENCY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_PAIRS_PER_GROUP = 5
TOP_EDGES_PER_PAIR = 5000
MIN_EVENT_COUNT = 10
OFFSET_RANGE = 20.0
SCORE_COLUMNS = ("original_lr_score", "attention_score")
PATHOLOGY_REGION_NAME = "Invasive Front"
PATHOLOGY_REGION_FILL = "#FFD166"
PATHOLOGY_REGION_EDGE = "#C47F00"


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


def _build_pathology_region_overlay() -> tuple[str | None, set[str]]:
    if DATA_DIR.name != "GSE144236":
        return None, set()

    composition_path = DATA_DIR / "Spatial_composition.csv"
    if not composition_path.exists():
        return None, set()

    composition = pd.read_csv(composition_path, index_col=0)
    required = ["Epithelial", "Fibroblast", "Mac", "CD1C", "Tcell"]
    missing = [col for col in required if col not in composition.columns]
    if missing:
        print(f"Skip pathology region overlay: missing composition columns {missing}")
        return None, set()

    composition.index = composition.index.astype(str)
    common_spots = composition.index.intersection(coords.index.astype(str))
    if len(common_spots) == 0:
        return None, set()
    composition = composition.loc[common_spots].copy()

    epithelial = composition["Epithelial"].astype(float)
    fibroblast = composition["Fibroblast"].astype(float)
    immune = (
        composition["Mac"].astype(float)
        + composition["CD1C"].astype(float)
        + composition["Tcell"].astype(float)
    )

    ep_thr = max(0.35, float(epithelial.quantile(0.65)))
    fib_thr = max(0.15, float(fibroblast.quantile(0.65)))
    immune_thr = max(0.05, float(immune.quantile(0.70)))

    ep_mask = epithelial >= ep_thr
    fib_mask = fibroblast >= fib_thr
    immune_mask = immune >= immune_thr

    local_coords = coords.loc[common_spots, ["x", "y"]].to_numpy(dtype=float)
    adjacency = kneighbors_graph(
        local_coords,
        n_neighbors=min(8, max(1, len(common_spots) - 1)),
        mode="connectivity",
        include_self=False,
    ).tolil()

    interface_mask = np.zeros(len(common_spots), dtype=bool)
    ep_values = ep_mask.to_numpy()
    fib_values = fib_mask.to_numpy()
    for idx in range(len(common_spots)):
        neighbors = adjacency.rows[idx]
        if not neighbors:
            continue
        if ep_values[idx] and fib_values[neighbors].any():
            interface_mask[idx] = True
        elif fib_values[idx] and ep_values[neighbors].any():
            interface_mask[idx] = True

    interface_series = pd.Series(interface_mask, index=common_spots)
    expanded = interface_series.copy()
    for idx, spot in enumerate(common_spots):
        neighbors = adjacency.rows[idx]
        if interface_mask[idx]:
            continue
        if neighbors and interface_mask[neighbors].any():
            if ep_values[idx] or fib_values[idx] or immune_mask.iloc[idx]:
                expanded.iloc[idx] = True

    selected_spots = set(expanded.index[expanded])
    print(
        f"Pathology region overlay ({PATHOLOGY_REGION_NAME}): "
        f"{len(selected_spots)} spots "
        f"[ep_thr={ep_thr:.3f}, fib_thr={fib_thr:.3f}, immune_thr={immune_thr:.3f}]"
    )
    return PATHOLOGY_REGION_NAME, selected_spots


PATHOLOGY_REGION_LABEL, PATHOLOGY_REGION_SPOTS = _build_pathology_region_overlay()

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

    show_pathology_region = (
        score_col == "attention_score"
        and PATHOLOGY_REGION_LABEL is not None
        and bool(PATHOLOGY_REGION_SPOTS)
    )
    if show_pathology_region:
        region_spots = sorted(PATHOLOGY_REGION_SPOTS.intersection(coords_plot.index.astype(str)))
        if region_spots:
            region_coords = coords_plot.loc[region_spots]
            ax.scatter(
                region_coords["x"],
                region_coords["y"],
                s=360,
                c=PATHOLOGY_REGION_FILL,
                alpha=0.22,
                edgecolors=PATHOLOGY_REGION_EDGE,
                linewidths=0.8,
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
    if show_pathology_region:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=PATHOLOGY_REGION_FILL,
                markeredgecolor=PATHOLOGY_REGION_EDGE,
                markersize=16,
                linewidth=0,
                label=PATHOLOGY_REGION_LABEL,
            )
        )
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


_plot_group(attention_pairs, ATTENTION_OUTPUT_DIR, "top attention")
_plot_group(frequency_pairs, FREQUENCY_OUTPUT_DIR, "top frequency")

print(f"Done! Top-attention LR pair plots saved to {ATTENTION_OUTPUT_DIR}")
print(f"Done! Top-frequency LR pair plots saved to {FREQUENCY_OUTPUT_DIR}")
