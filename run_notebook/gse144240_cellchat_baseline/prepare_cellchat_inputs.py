from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import io, sparse
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare GSE144240 CellChat spatial baseline inputs."
    )
    parser.add_argument("--st-h5ad", type=Path, required=True)
    parser.add_argument("--composition-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--sample-name", default="sample1")
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--visium-neighbor-distance-um", type=float, default=100.0)
    parser.add_argument("--spot-diameter-um", type=float, default=65.0)
    return parser.parse_args()


def normalize_log1p_counts(x: sparse.spmatrix, target_sum: float) -> sparse.csr_matrix:
    if not sparse.issparse(x):
        x = sparse.csr_matrix(x)
    x = x.tocsr(copy=True).astype(np.float64)
    library_sizes = np.asarray(x.sum(axis=1)).ravel()
    scale = np.divide(
        target_sum,
        library_sizes,
        out=np.zeros_like(library_sizes, dtype=np.float64),
        where=library_sizes > 0,
    )
    x = sparse.diags(scale).dot(x).tocsr()
    x.data = np.log1p(x.data)
    return x


def estimate_ratio(coords: np.ndarray, target_neighbor_distance_um: float) -> float:
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)
    nearest = distances[:, 1]
    return float(target_neighbor_distance_um / np.median(nearest))


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    st = ad.read_h5ad(args.st_h5ad)
    comp = pd.read_csv(args.composition_csv, index_col=0)
    comp = comp.loc[st.obs_names]

    coords = np.asarray(st.obsm["spatial"])
    ratio = estimate_ratio(coords, args.visium_neighbor_distance_um)
    tol = args.spot_diameter_um / 2.0

    dominant_label = comp.idxmax(axis=1)
    dominant_fraction = comp.max(axis=1)

    meta = pd.DataFrame(
        {
            "labels": dominant_label,
            "dominant_fraction": dominant_fraction,
            "samples": args.sample_name,
        },
        index=st.obs_names,
    )

    coord_df = pd.DataFrame(coords, index=st.obs_names, columns=["x", "y"])

    expr = normalize_log1p_counts(st.X, target_sum=args.target_sum).transpose().tocoo()
    io.mmwrite(str(args.outdir / "expression_log1p.mtx"), expr)

    pd.Series(st.var_names, name="gene").to_csv(
        args.outdir / "genes.tsv", sep="\t", index=False, header=False
    )
    pd.Series(st.obs_names, name="spot").to_csv(
        args.outdir / "spots.tsv", sep="\t", index=False, header=False
    )
    meta.to_csv(args.outdir / "meta.tsv", sep="\t")
    coord_df.to_csv(args.outdir / "coordinates.tsv", sep="\t")
    comp.to_csv(args.outdir / "composition.tsv", sep="\t")

    spatial_factors = pd.DataFrame(
        [
            {
                "ratio": ratio,
                "tol": tol,
                "interaction_range": 250.0,
                "contact_range": 100.0,
                "scale_distance": 0.01,
                "raw_nn_distance_median": float(
                    np.median(cKDTree(coords).query(coords, k=2)[0][:, 1])
                ),
            }
        ]
    )
    spatial_factors.to_csv(args.outdir / "spatial_factors.tsv", sep="\t", index=False)

    label_counts = meta["labels"].value_counts().rename_axis("label").reset_index(name="spots")
    label_counts.to_csv(args.outdir / "label_counts.tsv", sep="\t", index=False)

    summary_lines = [
        f"st_h5ad={args.st_h5ad}",
        f"composition_csv={args.composition_csv}",
        f"n_spots={st.n_obs}",
        f"n_genes={st.n_vars}",
        f"sample_name={args.sample_name}",
        f"suggested_ratio={ratio:.6f}",
        f"suggested_tol={tol:.2f}",
        f"suggested_interaction_range=250.0",
        f"suggested_contact_range=100.0",
        f"median_raw_neighbor_distance={spatial_factors.loc[0, 'raw_nn_distance_median']:.6f}",
        "",
        "dominant_label_counts:",
    ]
    summary_lines.extend(
        f"  {row.label}: {row.spots}" for row in label_counts.itertuples(index=False)
    )
    (args.outdir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
