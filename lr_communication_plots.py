"""Generate ligand-receptor communication visualizations for CID44971.

Replicates the notebook `lr_communication_plots.ipynb` as a script. Figures
are written to `results/CID44971/figures`.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless execution

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12
sns.set_style("whitegrid")


# Paths and toggles
DATA_DIR = Path("results/CID44971")
LR_SCORES_PATH = DATA_DIR / "lr_scores.csv"
LR_COMM_PATH = DATA_DIR / "lr_communication_filtered_1.csv"
EXPR_PATH = DATA_DIR / "spot_cell_full_expr.csv"  # optional expression matrix
OUTPUT_DIR = DATA_DIR / "figures"

# Spatial inputs
ENABLE_SPATIAL_OVERLAY = True  # set False to skip spatial overlays
ENABLE_SPATIAL_EXPR = True  # set False to skip spatial gene plots
ST_H5AD_PATH = Path("../ST_Graduation_Project_data/database/Wu/CID44971/CID44971_ST.h5ad")
COMPOSITION_PATH = Path("SC_MAP_ST/deconv_results/CID44971/CID44971_cell_composition.csv")
RECON_EXPR_PATH = Path("SC_MAP_ST/deconv_results/CID44971/CID44971_reconstructed_all_genes.csv")
TOP_LR_PER_CELLTYPE = 20
TOP_EDGES_PER_COMM = 20


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load LR scores and communication tables, add helper columns."""
    lr_scores = pd.read_csv(LR_SCORES_PATH)
    lr_comm = pd.read_csv(LR_COMM_PATH)

    lr_scores["lr_pair"] = lr_scores["ligand"].astype(str) + "_" + lr_scores["receptor"].astype(str)
    if "lr_pair" not in lr_comm.columns:
        lr_comm["lr_pair"] = lr_comm["source_cell"].astype(str) + "_" + lr_comm["target_cell"].astype(str)

    lr_comm[["lr_ligand", "lr_receptor_combined"]] = lr_comm["lr_pair"].str.split("_", n=1, expand=True)

    # Degree-scaled attention: boost attention by source out-degree
    src_degree = lr_comm.groupby("src_spot_barcode").size().rename("src_degree")
    lr_comm = lr_comm.merge(src_degree, left_on="src_spot_barcode", right_index=True, how="left")
    lr_comm["degree_scaled_attention"] = lr_comm["attention_score"] * lr_comm["src_degree"].clip(lower=1)

    return lr_scores, lr_comm


def collect_top_items_per_cell(
    lr_comm: pd.DataFrame, cell_col: str, value_col: str, top_n: int, score_col: Optional[str] = None
) -> List[str]:
    """Return ordered unique items appearing in top_n per cell type."""
    items: List[str] = []
    if cell_col not in lr_comm.columns or value_col not in lr_comm.columns:
        return items
    for cell_type, group in lr_comm.groupby(cell_col):
        if score_col and score_col in group.columns:
            top_values = (
                group.groupby(value_col)[score_col]
                .max()
                .sort_values(ascending=False)
                .head(top_n)
                .index.tolist()
            )
        else:
            top_values = group[value_col].value_counts().head(top_n).index.tolist()
        items.extend(top_values)
    # Preserve order while removing duplicates
    return list(dict.fromkeys(items))


def plot_celltype_heatmaps(lr_comm: pd.DataFrame) -> None:
    """Plot cell-type communication matrices."""
    cell_pair_df = (
        lr_comm.groupby(["source_cell", "target_cell"])
        .agg(event_count=("lr_pair", "size"), mean_attention=("attention_score", "mean"))
        .reset_index()
    )

    cell_matrix = cell_pair_df.pivot(index="source_cell", columns="target_cell", values="event_count").fillna(0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cell_matrix, annot=True, fmt=".0f", cmap="PuBu", cbar_kws={"label": "Event count"})
    plt.title("Cell-type communication frequency")
    plt.xlabel("Target cell type")
    plt.ylabel("Source cell type")
    plt.tight_layout()
    fig_path = OUTPUT_DIR / "celltype_heatmap.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", fig_path)

    cell_out = lr_comm.groupby("source_cell").size().rename("outgoing")
    cell_in = lr_comm.groupby("target_cell").size().rename("incoming")
    cell_degree = pd.concat([cell_out, cell_in], axis=1).fillna(0).astype(int).sort_values("outgoing", ascending=False)

    cell_degree.plot(kind="bar", figsize=(12, 6), color=["#4c72b0", "#dd8452"])
    plt.title("Cell-type incoming/outgoing events")
    plt.xlabel("Cell type")
    plt.ylabel("Event count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig_path = OUTPUT_DIR / "celltype_in_out.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", fig_path)


def build_top_pairs(lr_comm: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Aggregate LR events per spot pair; keep top edges per communication ID using score_col."""
    if score_col not in lr_comm.columns:
        raise ValueError(f"Score column '{score_col}' not found in lr_comm.")
    ranked = lr_comm.copy()
    ranked["rank_within_pair"] = ranked.groupby("lr_pair")[score_col].rank(method="first", ascending=False)
    ranked = ranked[ranked["rank_within_pair"] <= TOP_EDGES_PER_COMM]

    pair_df = (
        ranked.groupby(["src_spot_barcode", "dst_spot_barcode", "lr_pair"])
        .agg(event_count=("lr_pair", "size"), mean_score=(score_col, "mean"), max_score=(score_col, "max"))
        .reset_index()
    )
    pair_df = pair_df.merge(
        ranked[
            ["src_spot_barcode", "dst_spot_barcode", "lr_pair", "source_cell", "target_cell"]
        ].drop_duplicates(),
        on=["src_spot_barcode", "dst_spot_barcode", "lr_pair"],
    )
    return pair_df.sort_values(["lr_pair", "mean_score"], ascending=[True, False])


def prepare_top_attention_edges(lr_comm: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Return the top-N individual edges ranked by attention_score."""
    if "attention_score" not in lr_comm.columns:
        raise ValueError("attention_score column is required for top attention edges.")
    return lr_comm.sort_values("attention_score", ascending=False).head(top_n).copy()


def plot_spatial_overlay(
    lr_comm: pd.DataFrame,
    top_pairs: pd.DataFrame,
    *,
    fig_name: str = "top_pairs_spatial.png",
    title: str = "Top spot pairs on tissue (spatial overlay)",
) -> None:
    """Plot top spot pairs on tissue space using spatial coordinates."""
    if not ENABLE_SPATIAL_OVERLAY:
        print("Spatial overlay disabled; skipping.")
        return
    if not ST_H5AD_PATH.exists():
        print(f"Spatial overlay skipped: missing {ST_H5AD_PATH}")
        return
    try:
        import scanpy as sc
    except ImportError:
        print("Spatial overlay skipped: scanpy not installed.")
        return

    adata = sc.read_h5ad(ST_H5AD_PATH)
    coords = pd.DataFrame(adata.obsm["spatial"], index=adata.obs_names, columns=["x", "y"])

    deg_out = lr_comm.groupby("src_spot_barcode").size().rename("out")
    deg_in = lr_comm.groupby("dst_spot_barcode").size().rename("in")
    deg = pd.concat([deg_out, deg_in], axis=1).fillna(0)
    deg["total"] = deg["out"] + deg["in"]

    spot_celltype = None
    if COMPOSITION_PATH.exists():
        comp_df = pd.read_csv(COMPOSITION_PATH, index_col=0)
        spot_celltype = comp_df.idxmax(axis=1)

    if spot_celltype is not None:
        top_spots_per_ct = []
        for ct in spot_celltype.unique():
            spots_for_ct = spot_celltype[spot_celltype == ct].index
            deg_ct = deg.loc[deg.index.isin(spots_for_ct)]
            top_for_ct = deg_ct.sort_values("total", ascending=False).head(10)
            top_spots_per_ct.append(top_for_ct)
        top_spots = pd.concat(top_spots_per_ct)
    else:
        top_spots = deg.sort_values("total", ascending=False).head(15)

    arrows = top_pairs.copy()
    arrows["src_x"] = arrows["src_spot_barcode"].map(coords["x"])
    arrows["src_y"] = arrows["src_spot_barcode"].map(coords["y"])
    arrows["dst_x"] = arrows["dst_spot_barcode"].map(coords["x"])
    arrows["dst_y"] = arrows["dst_spot_barcode"].map(coords["y"])
    arrows = arrows.dropna(subset=["src_x", "src_y", "dst_x", "dst_y"])

    lr_unique = arrows["lr_pair"].unique()
    lr_colors = dict(zip(lr_unique, sns.color_palette("tab20", len(lr_unique))))
    arrow_colors = arrows["lr_pair"].map(lr_colors)

    if len(arrows) > 0:
        s_min, s_max = arrows["mean_score"].min(), arrows["mean_score"].max()
        line_widths = 1.0 + (arrows["mean_score"] - s_min) / (s_max - s_min + 1e-6) * 2.0
    else:
        line_widths = []

    if spot_celltype is not None:
        cell_types = spot_celltype.loc[top_spots.index].fillna("Unknown").unique()
        ct_colors: Dict[str, Iterable[float]] = dict(zip(cell_types, sns.color_palette("tab10", len(cell_types))))
        top_colors = spot_celltype.loc[top_spots.index].map(ct_colors)
    else:
        ct_colors = None
        top_colors = "crimson"

    plt.figure(figsize=(10, 10))
    plt.scatter(coords["x"], coords["y"], s=20, c="lightgray", alpha=0.25, label="All spots")
    plt.scatter(
        coords.loc[top_spots.index, "x"],
        coords.loc[top_spots.index, "y"],
        s=40,
        c=top_colors,
        alpha=0.9,
        label="Top-degree spots",
        edgecolor="k",
        linewidth=0.4,
    )

    for (_, row), color, lw in zip(arrows.iterrows(), arrow_colors, line_widths):
        plt.arrow(
            row.src_x,
            row.src_y,
            row.dst_x - row.src_x,
            row.dst_y - row.src_y,
            color=color,
            alpha=0.6,
            linewidth=lw,
            length_includes_head=True,
            head_width=15,
        )

    lr_handles = [plt.Line2D([0], [0], color=clr, lw=2, label=lp) for lp, clr in lr_colors.items()]
    if ct_colors:
        ct_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=clr,
                markeredgecolor="k",
                lw=0,
                label=ct,
            )
            for ct, clr in ct_colors.items()
        ]
    else:
        ct_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="crimson",
                markeredgecolor="k",
                lw=0,
                label="Top-degree spots",
            )
        ]

    plt.gca().invert_yaxis()
    plt.legend(handles=ct_handles + lr_handles, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    fig_path = OUTPUT_DIR / fig_name
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", fig_path)


def plot_attention_overlay(lr_comm: pd.DataFrame, top_edges: pd.DataFrame, top_n: int = 50) -> None:
    """Plot top-N edges by attention score without hotspot highlighting."""
    if not ENABLE_SPATIAL_OVERLAY:
        print("Spatial overlay disabled; skipping attention overlay.")
        return
    if not ST_H5AD_PATH.exists():
        print(f"Attention overlay skipped: missing {ST_H5AD_PATH}")
        return
    try:
        import scanpy as sc
    except ImportError:
        print("Attention overlay skipped: scanpy not installed.")
        return

    adata = sc.read_h5ad(ST_H5AD_PATH)
    coords = pd.DataFrame(adata.obsm["spatial"], index=adata.obs_names, columns=["x", "y"])

    edges = top_edges.copy()
    edges["src_x"] = edges["src_spot_barcode"].map(coords["x"])
    edges["src_y"] = edges["src_spot_barcode"].map(coords["y"])
    edges["dst_x"] = edges["dst_spot_barcode"].map(coords["x"])
    edges["dst_y"] = edges["dst_spot_barcode"].map(coords["y"])
    edges = edges.dropna(subset=["src_x", "src_y", "dst_x", "dst_y"])

    if edges.empty:
        print("No edges available for attention overlay.")
        return

    scores = edges["attention_score"]
    s_min, s_max = scores.min(), scores.max()
    line_widths = 1.0 + (scores - s_min) / (s_max - s_min + 1e-6) * 2.0

    plt.figure(figsize=(10, 10))
    plt.scatter(coords["x"], coords["y"], s=20, c="lightgray", alpha=0.2, linewidth=0)

    for (_, row), lw in zip(edges.iterrows(), line_widths):
        plt.arrow(
            row.src_x,
            row.src_y,
            row.dst_x - row.src_x,
            row.dst_y - row.src_y,
            color="#d62728",
            alpha=0.7,
            linewidth=lw,
            length_includes_head=True,
            head_width=15,
        )

    plt.gca().invert_yaxis()
    plt.title(f"Top {top_n} edges by attention score")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    fig_path = OUTPUT_DIR / "top_attention_spatial.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", fig_path)


def plot_expression_heatmaps(lr_comm: pd.DataFrame) -> None:
    """Plot ligand/receptor expression heatmaps if expression matrix exists."""
    if not EXPR_PATH.exists():
        print(f"Expression file not found at {EXPR_PATH}; skipping heatmaps.")
        return

    expr_df = pd.read_csv(EXPR_PATH)
    expr_df["cell_type"] = expr_df["spot_cell"].apply(lambda x: str(x).rsplit("_", 1)[-1])

    source_top = collect_top_items_per_cell(
        lr_comm, "source_cell", "lr_pair", TOP_LR_PER_CELLTYPE, score_col="original_lr_score"
    )
    target_top = collect_top_items_per_cell(
        lr_comm, "target_cell", "lr_pair", TOP_LR_PER_CELLTYPE, score_col="original_lr_score"
    )
    top_pairs = list(dict.fromkeys(source_top + target_top))
    if not top_pairs:
        print("No LR pairs available for expression heatmaps.")
        return
    ligands = lr_comm[lr_comm["lr_pair"].isin(top_pairs)]["lr_ligand"].unique()
    receptor_genes = set()
    for rec_combo in lr_comm[lr_comm["lr_pair"].isin(top_pairs)]["lr_receptor_combined"].dropna():
        receptor_genes.update(rec_combo.split("_"))
    ligands = [g for g in ligands if g in expr_df.columns]
    receptor_genes = [g for g in receptor_genes if g in expr_df.columns]

    ligand_expr = expr_df.groupby("cell_type")[ligands].mean().sort_index()
    receptor_expr = expr_df.groupby("cell_type")[receptor_genes].mean().sort_index()

    plt.figure(figsize=(12, 6 + 0.3 * len(ligands)))
    sns.heatmap(ligand_expr, cmap="Reds", cbar_kws={"label": "Mean expression"})
    plt.title("Ligand expression across cell types (top LR pairs)")
    plt.xlabel("Ligand gene")
    plt.ylabel("Cell type")
    plt.tight_layout()
    fig_path = OUTPUT_DIR / "ligand_expression_heatmap.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", fig_path)

    plt.figure(figsize=(12, 6 + 0.3 * len(receptor_genes)))
    sns.heatmap(receptor_expr, cmap="Blues", cbar_kws={"label": "Mean expression"})
    plt.title("Receptor subunit expression across cell types (top LR pairs)")
    plt.xlabel("Receptor gene")
    plt.ylabel("Cell type")
    plt.tight_layout()
    fig_path = OUTPUT_DIR / "receptor_expression_heatmap.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", fig_path)


def plot_spatial_expression(lr_comm: pd.DataFrame) -> None:
    """Plot spatial expression for top ligands/receptors if h5ad exists."""
    if not ENABLE_SPATIAL_EXPR:
        print("Spatial expression disabled; skipping.")
        return
    if not ST_H5AD_PATH.exists():
        print(f"Spatial expression skipped: missing {ST_H5AD_PATH}")
        return
    if not RECON_EXPR_PATH.exists():
        print(f"Spatial expression skipped: missing {RECON_EXPR_PATH}")
        return
    try:
        import scanpy as sc
    except ImportError:
        print("Spatial expression skipped: scanpy not installed.")
        return

    if "lr_ligand" not in lr_comm.columns:
        lr_comm[["lr_ligand", "lr_receptor_combined"]] = lr_comm["lr_pair"].str.split("_", n=1, expand=True)

    top_pairs_source = collect_top_items_per_cell(
        lr_comm, "source_cell", "lr_pair", TOP_LR_PER_CELLTYPE, score_col="original_lr_score"
    )
    top_pairs_target = collect_top_items_per_cell(
        lr_comm, "target_cell", "lr_pair", TOP_LR_PER_CELLTYPE, score_col="original_lr_score"
    )
    top_pairs = list(dict.fromkeys(top_pairs_source + top_pairs_target))
    if not top_pairs:
        print("No LR pairs available for spatial expression plots.")
        return

    adata = sc.read_h5ad(ST_H5AD_PATH)
    coords = np.array(adata.obsm["spatial"])
    spot_index = list(adata.obs_names)

    recon_expr = pd.read_csv(RECON_EXPR_PATH, index_col=0)
    recon_expr = recon_expr.reindex(spot_index)

    def fetch_expr(gene: str) -> np.ndarray | None:
        if gene not in recon_expr.columns:
            return None
        values = recon_expr[gene].to_numpy(dtype=float)
        if np.all(np.isnan(values)):
            return None
        values = np.nan_to_num(values, nan=0.0)
        values = np.clip(values, a_min=0.0, a_max=None)
        max_val = values.max()
        if max_val > 0:
            values = values / max_val
        return np.log1p(values)

    def plot_pair(pair_name: str, ligand_gene: Optional[str], receptor_combo: Optional[str]) -> None:
        ligand_expr = fetch_expr(ligand_gene) if ligand_gene else None
        receptor_expr = None
        if receptor_combo:
            expr_list = []
            for part in [p for p in str(receptor_combo).split("_") if p]:
                val = fetch_expr(part)
                if val is not None:
                    expr_list.append(val)
            if expr_list:
                receptor_expr = np.mean(expr_list, axis=0)

        if ligand_expr is None and receptor_expr is None:
            print(f"[pair] skip {pair_name}: no expression data")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
        axes = np.atleast_1d(axes)

        def draw(ax, values: Optional[np.ndarray], title: str) -> None:
            if values is None or not np.isfinite(values).any():
                ax.text(0.5, 0.5, "not available", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.invert_yaxis()
                ax.set_aspect("equal", adjustable="box")
                ax.set_title(title)
                return
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap="viridis", s=10, linewidth=0)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="expression")
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(title)

        draw(axes[0], ligand_expr, f"Ligand: {ligand_gene or 'N/A'}")
        draw(axes[1], receptor_expr, f"Receptor: {receptor_combo or 'N/A'}")
        plt.suptitle(pair_name)
        plt.tight_layout()
        safe_name = pair_name.replace("/", "_").replace(" ", "_")
        out_path = OUTPUT_DIR / f"spatial_pair_{safe_name}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print("Saved", out_path)

    pair_info = (
        lr_comm[lr_comm["lr_pair"].isin(top_pairs)][["lr_pair", "lr_ligand", "lr_receptor_combined"]]
        .drop_duplicates()
        .itertuples(index=False)
    )

    for pair_name, ligand_gene, receptor_combo in pair_info:
        plot_pair(pair_name, ligand_gene, receptor_combo)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lr_scores, lr_comm = load_data()
    print(f"lr_scores: {lr_scores.shape}, lr_communication: {lr_comm.shape}")
    print("Figures will be saved to", OUTPUT_DIR)

    plot_celltype_heatmaps(lr_comm)

    top_pairs_score = build_top_pairs(lr_comm, score_col="original_lr_score")
    plot_spatial_overlay(lr_comm, top_pairs_score)

    top_attention_edges = prepare_top_attention_edges(lr_comm, top_n=50)
    plot_attention_overlay(lr_comm, top_attention_edges, top_n=50)

    plot_expression_heatmaps(lr_comm)
    plot_spatial_expression(lr_comm)


if __name__ == "__main__":
    main()
