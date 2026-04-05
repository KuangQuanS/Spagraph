"""
Pathology interface spatial mapping — publication figure.

Identifies invadopodia-like interface spots via composition scoring
and plots a clean spatial overlay suitable for thesis Figure C.
"""
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.spatial import distance_matrix
from sklearn.neighbors import kneighbors_graph

# ─── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATE_DIR = SCRIPT_DIR.parent.parent
DATA_ROOT = EVALUATE_DIR / "data"
REPO_ROOT = EVALUATE_DIR.parent
DATABASE_ROOT = REPO_ROOT / "spagraph_data" / "database"

DATA_DIR = DATA_ROOT / "GSE144236"
ST_H5AD_PATH = DATABASE_ROOT / "GSE144240" / "GSE144236_P2_ST.h5ad"
OUTPUT_DIR = DATA_DIR / "figures" / "pathology_interface"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Parameters ───────────────────────────────────────────────────────
PATHOLOGY_REGION_MAX_COMPONENTS = 2


def run() -> None:
    print("Loading data ...")
    adata = sc.read_h5ad(ST_H5AD_PATH)
    coords_raw = pd.DataFrame(
        adata.obsm["spatial"],
        index=adata.obs_names.astype(str),
        columns=["x", "y"],
    ).astype(float)

    scale_factor = 1.0
    library_key = list(adata.uns["spatial"].keys())[0]
    scale_factor = adata.uns["spatial"][library_key]["scalefactors"].get(
        "tissue_hires_scalef", 1.0
    )

    # ── Composition ───────────────────────────────────────────────────
    composition_path = DATA_DIR / "Spatial_composition.csv"
    if not composition_path.exists():
        raise FileNotFoundError(f"Missing: {composition_path}")

    composition = pd.read_csv(composition_path, index_col=0)
    composition.index = composition.index.astype(str)
    common_spots = composition.index.intersection(coords_raw.index.astype(str))
    if len(common_spots) == 0:
        raise ValueError("No overlapping spots between composition and adata.")
    composition = composition.loc[common_spots].copy()

    epithelial = composition["Epithelial"].astype(float)
    fibroblast = composition["Fibroblast"].astype(float)
    immune = (
        composition["Mac"].astype(float)
        + composition["CD1C"].astype(float)
        + composition["Tcell"].astype(float)
    )

    # invadopodia-like interface score
    interface_score = epithelial * fibroblast * np.maximum(immune, 0.05)
    if float(interface_score.max()) > float(interface_score.min()):
        interface_scaled = (interface_score - interface_score.min()) / (
            interface_score.max() - interface_score.min()
        )
    else:
        interface_scaled = pd.Series(
            np.zeros(len(interface_score), dtype=float), index=interface_score.index
        )

    # Relaxed threshold (80th percentile) to show more interface spots
    boundary_threshold = float(np.percentile(interface_scaled.to_numpy(), 80))
    interface_mask = interface_scaled >= boundary_threshold
    if not interface_mask.any():
        boundary_threshold = float(np.percentile(interface_scaled.to_numpy(), 70))
        interface_mask = interface_scaled >= boundary_threshold

    # signed distance for epithelial / fibroblast classification
    local_coords = coords_raw.loc[common_spots, ["x", "y"]].to_numpy(dtype=float)
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
    boundary_coords = coords_raw.loc[boundary_spots, ["x", "y"]].to_numpy(dtype=float)
    if boundary_coords.size == 0:
        signed_distance = np.where(ep_values >= fib_values, -1.0, 1.0)
    else:
        min_distances = distance_matrix(local_coords, boundary_coords).min(axis=1)
        direction = np.where(ep_values >= fib_values, -1.0, 1.0)
        signed_distance = min_distances * direction
        signed_distance[interface_values] = 0.0

    # connected-component filtering
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
    kept_components = components[: PATHOLOGY_REGION_MAX_COMPONENTS]
    kept_spots = {spot for comp in kept_components for spot in comp}

    is_interface = np.array([spot in kept_spots for spot in common_spots], dtype=bool)
    is_epithelial_rich = (~is_interface) & (signed_distance < 0)
    is_fibroblast_rich = (~is_interface) & (signed_distance > 0)

    # pixel coords
    px = coords_raw.loc[common_spots, "x"].to_numpy(dtype=float) * scale_factor
    py = coords_raw.loc[common_spots, "y"].to_numpy(dtype=float) * scale_factor

    n_iface = int(is_interface.sum())
    n_ep = int(is_epithelial_rich.sum())
    n_fib = int(is_fibroblast_rich.sum())
    print(
        f"  {n_ep} epithelial-rich, {n_fib} fibroblast-rich, {n_iface} interface spots"
    )

    # ── Plot ──────────────────────────────────────────────────────────
    _plot(adata, px, py, is_epithelial_rich, is_fibroblast_rich, is_interface)


# =====================================================================
#                         PLOTTING
# =====================================================================

# Publication palette — muted tones that won't fight the H&E
_C_FIBRO = "#B0B7C3"        # cool gray
_C_TUMOR = "#5B4A9E"        # muted purple (matches H&E stain)
_C_IFACE = "#E8963E"        # warm orange — focal point

# Marker sizes — calibrated for 646-spot Visium at 6-inch width
_S_BASE = 38                # background spots
_S_IFACE = 52               # interface spots (slightly larger to pop)


def _plot(
    adata: sc.AnnData,
    px: np.ndarray,
    py: np.ndarray,
    is_ep: np.ndarray,
    is_fib: np.ndarray,
    is_iface: np.ndarray,
) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,       # editable text in Illustrator
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
        "savefig.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(5.8, 5.4))

    # No H&E background to create contrast with other panels
    # sc.pl.spatial(adata, color=None, alpha_img=0.18, size=0, show=False, ax=ax)

    # Draw layers bottom → top: fibroblast → epithelial → interface
    if is_fib.any():
        ax.scatter(
            px[is_fib], py[is_fib],
            s=_S_BASE, c=_C_FIBRO,
            edgecolors="white", linewidths=0.25,
            alpha=0.82, zorder=3,
            rasterized=True,
        )
    if is_ep.any():
        ax.scatter(
            px[is_ep], py[is_ep],
            s=_S_BASE, c=_C_TUMOR,
            edgecolors="white", linewidths=0.25,
            alpha=0.82, zorder=4,
            rasterized=True,
        )
    if is_iface.any():
        ax.scatter(
            px[is_iface], py[is_iface],
            s=_S_IFACE, c=_C_IFACE,
            edgecolors="white", linewidths=0.5,
            alpha=0.95, zorder=5,
            rasterized=True,
        )

    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Legend — compact, below the plot ──────────────────────────
    handles = [
        Line2D(
            [], [], marker="o", linestyle="None",
            markerfacecolor=_C_FIBRO, markeredgecolor="white",
            markeredgewidth=0.4, markersize=7,
            label="Fibroblast-rich",
        ),
        Line2D(
            [], [], marker="o", linestyle="None",
            markerfacecolor=_C_TUMOR, markeredgecolor="white",
            markeredgewidth=0.4, markersize=7,
            label="Epithelial-rich",
        ),
        Line2D(
            [], [], marker="o", linestyle="None",
            markerfacecolor=_C_IFACE, markeredgecolor="white",
            markeredgewidth=0.6, markersize=8,
            label="Interface",
        ),
    ]
    # ── Legend — right side ───────────────────────────────────────────────
    leg = ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        frameon=False,
        prop={"size": 10, "weight": "bold"},
        handletextpad=0.3,
        labelspacing=0.8,
    )

    # ── Save ──────────────────────────────────────────────────────
    pdf_path = OUTPUT_DIR / "epithelial_stromal_interface.pdf"
    png_path = OUTPUT_DIR / "epithelial_stromal_interface.png"

    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    print(f"  PDF → {pdf_path}")
    print(f"  PNG → {png_path}")


if __name__ == "__main__":
    run()
