#!/usr/bin/env python3
"""Run a Scanpy UMAP workflow treating each cell row as a spot observation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a CSV expression matrix (rows=cells/spots, columns=genes), "
            "normalize + log1p, select highly variable genes, then compute UMAP "
            "embeddings and Leiden clusters using Scanpy."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "SC_MAP_ST/deconv_results/CID44971/CID44971_reconstructed_all_genes.csv"
        ),
        help="Path to the expression CSV where the first column stores spot/cell ids.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("SC_MAP_ST/deconv_results/CID44971"),
        help="Directory for all outputs (h5ad, cluster table, UMAP figure).",
    )
    parser.add_argument(
        "--n-top-genes",
        type=int,
        default=2000,
        help="Number of highly-variable genes to keep before downstream analysis.",
    )
    parser.add_argument(
        "--n-pcs",
        type=int,
        default=50,
        help="Number of principal components used for neighbor graph construction.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=40,
        help="Number of neighbors used in the kNN graph for clustering/UMAP.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.5,
        help="Leiden resolution parameter (higher -> more clusters).",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.5,
        help="UMAP minimum distance parameter controlling cluster tightness.",
    )
    parser.add_argument(
        "--h5ad-name",
        type=str,
        default="spot_cell_scanpy.h5ad",
        help="Output file name for the AnnData object.",
    )
    parser.add_argument(
        "--clusters-name",
        type=str,
        default="spot_cell_leiden.csv",
        help="Output CSV storing Leiden cluster labels.",
    )
    parser.add_argument(
        "--umap-name",
        type=str,
        default="spot_cell_umap.png",
        help="Output PNG containing the UMAP colored by Leiden clusters.",
    )
    parser.add_argument(
        "--st-h5ad",
        type=Path,
        default=Path(
            "../ST_Graduation_Project_data/database/Wu/CID44971/CID44971_ST.h5ad"
        ),
        help="Optional path to the Visium/ST h5ad containing spatial coordinates + tissue image.",
    )
    parser.add_argument(
        "--library-id",
        type=str,
        default=None,
        help="Library ID key inside adata.uns['spatial']; defaults to the first available key.",
    )
    parser.add_argument(
        "--image-key",
        type=str,
        default="hires",
        choices=("hires", "lowres"),
        help="Resolution key from the spatial image dict to use as tissue background.",
    )
    parser.add_argument(
        "--tissue-spot-size",
        type=float,
        default=0.85,
        help="Spot size parameter passed to scanpy.pl.spatial when drawing the tissue overlay.",
    )
    parser.add_argument(
        "--tissue-overlay-name",
        type=str,
        default="spot_cell_tissue.png",
        help="Output PNG showing the Leiden clusters overlaid on the tissue image.",
    )
    parser.add_argument(
        "--run-original",
        action="store_true",
        help="Also run the pipeline on the original ST h5ad expression matrix.",
    )
    parser.add_argument(
        "--original-h5ad-name",
        type=str,
        default="original_st_scanpy.h5ad",
        help="AnnData file for the original ST analysis (when --run-original is enabled).",
    )
    parser.add_argument(
        "--original-clusters-name",
        type=str,
        default="original_st_leiden.csv",
        help="Cluster CSV for the original ST analysis (when --run-original is enabled).",
    )
    parser.add_argument(
        "--original-umap-name",
        type=str,
        default="original_st_umap.png",
        help="UMAP figure for the original ST analysis (when --run-original is enabled).",
    )
    parser.add_argument(
        "--original-tissue-name",
        type=str,
        default="original_st_tissue.png",
        help="Tissue overlay for the original ST analysis (when --run-original is enabled).",
    )
    return parser.parse_args()


def run_scanpy_workflow(
    adata: sc.AnnData,
    args: argparse.Namespace,
    key_prefix: str,
    h5ad_path: Path,
    clusters_path: Path,
    umap_path: Path,
) -> tuple[sc.AnnData, str]:
    adata = adata.copy()
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    adata.layers["counts"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=args.n_top_genes,
        subset=True,
        flavor="seurat_v3",
    )
    sc.pp.scale(adata, max_value=10)

    sc.tl.pca(adata, n_comps=args.n_pcs, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, n_pcs=args.n_pcs)
    sc.tl.umap(adata, min_dist=args.min_dist)

    leiden_key = f"{key_prefix}_leiden"
    sc.tl.leiden(adata, resolution=args.resolution, key_added=leiden_key)

    adata.write(h5ad_path, compression="gzip")
    adata.obs[[leiden_key]].to_csv(clusters_path)

    fig = sc.pl.umap(
        adata,
        color=leiden_key,
        legend_loc="on data",
        size=50,
        show=False,
        return_fig=True,
    )
    fig.savefig(umap_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return adata, leiden_key


def plot_clusters_on_tissue(
    adata: sc.AnnData,
    cluster_key: str,
    st_h5ad_path: Path,
    output_path: Path,
    image_key: str,
    spot_size: float,
    library_id: str | None = None,
) -> None:
    st_h5ad_path = Path(st_h5ad_path)
    if not st_h5ad_path.exists():
        print(f"[tissue overlay] {st_h5ad_path} not found; skipping tissue plot.")
        return

    st_adata = sc.read_h5ad(st_h5ad_path)
    if "spatial" not in st_adata.obsm:
        raise ValueError(f"No 'spatial' coordinates found in {st_h5ad_path}")

    cluster_series = adata.obs[cluster_key]
    common = st_adata.obs_names.intersection(cluster_series.index)
    if len(common) == 0:
        raise ValueError(
            "No overlapping spot IDs between expression CSV and ST h5ad for overlay."
        )

    st_subset = st_adata[common].copy()
    aligned_clusters = cluster_series.loc[common]
    st_subset.obs[cluster_key] = pd.Categorical(
        aligned_clusters,
        categories=cluster_series.cat.categories,
    )
    color_key = f"{cluster_key}_colors"
    if color_key in adata.uns:
        st_subset.uns[color_key] = adata.uns[color_key]

    spatial_info = st_subset.uns.get("spatial")
    if not spatial_info:
        raise ValueError(f"No spatial metadata found in {st_h5ad_path}")
    resolved_library = library_id or next(iter(spatial_info))
    if resolved_library not in spatial_info:
        available = ", ".join(spatial_info.keys())
        raise ValueError(
            f"Library '{resolved_library}' not found. Available libraries: {available}"
        )
    library_data = spatial_info[resolved_library]
    if image_key not in library_data["images"]:
        available_images = ", ".join(library_data["images"].keys())
        raise ValueError(
            f"Image key '{image_key}' missing. Available: {available_images}"
        )

    fig = sc.pl.spatial(
        st_subset,
        color=cluster_key,
        img_key=image_key,
        library_id=resolved_library,
        spot_size=spot_size,
        show=False,
        return_fig=True,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_clusters_on_existing_adata(
    adata: sc.AnnData,
    cluster_key: str,
    output_path: Path,
    image_key: str,
    spot_size: float,
    library_id: str | None = None,
) -> None:
    if "spatial" not in adata.uns:
        raise ValueError("AnnData object lacks spatial metadata for plotting.")

    spatial_info = adata.uns.get("spatial")
    resolved_library = library_id or next(iter(spatial_info))
    if resolved_library not in spatial_info:
        available = ", ".join(spatial_info.keys())
        raise ValueError(
            f"Library '{resolved_library}' not found. Available libraries: {available}"
        )

    fig = sc.pl.spatial(
        adata,
        color=cluster_key,
        img_key=image_key,
        library_id=resolved_library,
        spot_size=spot_size,
        show=False,
        return_fig=True,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    outdir = args.outdir.expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    expr = pd.read_csv(input_path, index_col=0)
    if expr.empty:
        raise ValueError(f"No expression values found in {input_path}")

    recon_adata = sc.AnnData(expr)
    recon_h5ad = outdir / args.h5ad_name
    recon_clusters = outdir / args.clusters_name
    recon_umap = outdir / args.umap_name

    recon_adata, recon_key = run_scanpy_workflow(
        recon_adata,
        args,
        key_prefix="spot",
        h5ad_path=recon_h5ad,
        clusters_path=recon_clusters,
        umap_path=recon_umap,
    )

    if args.st_h5ad:
        overlay_path = outdir / args.tissue_overlay_name
        plot_clusters_on_tissue(
            recon_adata,
            recon_key,
            args.st_h5ad,
            overlay_path,
            image_key=args.image_key,
            spot_size=args.tissue_spot_size,
            library_id=args.library_id,
        )
        print(f"Tissue overlay saved to {overlay_path}")

    print(
        f"[Reconstructed] Saved AnnData to {recon_h5ad}, clusters to {recon_clusters}, "
        f"and UMAP figure to {recon_umap}"
    )

    if args.run_original:
        if args.st_h5ad is None:
            raise ValueError(
                "--run-original requires --st-h5ad pointing to the raw ST h5ad file."
            )
        original_adata = sc.read_h5ad(args.st_h5ad)
        orig_h5ad = outdir / args.original_h5ad_name
        orig_clusters = outdir / args.original_clusters_name
        orig_umap = outdir / args.original_umap_name

        original_adata, original_key = run_scanpy_workflow(
            original_adata,
            args,
            key_prefix="orig",
            h5ad_path=orig_h5ad,
            clusters_path=orig_clusters,
            umap_path=orig_umap,
        )

        orig_tissue_path = outdir / args.original_tissue_name
        plot_clusters_on_existing_adata(
            original_adata,
            original_key,
            orig_tissue_path,
            image_key=args.image_key,
            spot_size=args.tissue_spot_size,
            library_id=args.library_id,
        )
        print(
            f"[Original] Saved AnnData to {orig_h5ad}, clusters to {orig_clusters}, "
            f"UMAP to {orig_umap}, and tissue overlay to {orig_tissue_path}"
        )


if __name__ == "__main__":
    main()
