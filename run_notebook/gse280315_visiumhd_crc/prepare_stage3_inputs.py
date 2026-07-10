from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


def split_complex(value: str) -> list[str]:
    return [part.strip().upper() for part in str(value).replace("+", "_").split("_") if part.strip()]


def read_10x_h5(path: Path) -> tuple[sp.csc_matrix, list[str], list[str]]:
    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        shape = tuple(int(v) for v in matrix["shape"][:])
        x = sp.csc_matrix(
            (matrix["data"][:], matrix["indices"][:], matrix["indptr"][:]),
            shape=shape,
        )
        genes = [g.decode() for g in matrix["features/name"][:]]
        barcodes = [b.decode() for b in matrix["barcodes"][:]]
    return x, genes, barcodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare aggregated Stage 3 inputs for GSE280315 Visium HD CRC.")
    parser.add_argument("--matrix-h5", required=True)
    parser.add_argument("--metadata-parquet", required=True)
    parser.add_argument("--positions-parquet", required=True)
    parser.add_argument("--cellchat", default="cellchat_human.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="GSM8594567_P1CRC_128um")
    parser.add_argument("--bin-size-um", type=int, default=128)
    parser.add_argument("--source-bin-um", type=int, default=8)
    parser.add_argument("--max-genes", type=int, default=0, help="Optional cap after CellChat gene selection; 0 keeps all LR genes.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_gene_by_barcode, genes, barcodes = read_10x_h5(Path(args.matrix_h5))
    gene_upper = pd.Index([g.upper() for g in genes])

    cellchat = pd.read_csv(args.cellchat)
    lr_genes: set[str] = set()
    for _, row in cellchat.iterrows():
        lr_genes.update(split_complex(row["ligand"]))
        lr_genes.update(split_complex(row["receptor"]))
    selected_gene_mask = gene_upper.isin(sorted(lr_genes))
    selected_gene_idx = np.flatnonzero(selected_gene_mask)
    if args.max_genes and len(selected_gene_idx) > args.max_genes:
        selected_gene_idx = selected_gene_idx[: args.max_genes]
    selected_genes = [genes[i] for i in selected_gene_idx]

    metadata = pd.read_parquet(args.metadata_parquet)
    positions = pd.read_parquet(args.positions_parquet)
    obs = positions.merge(metadata, on="barcode", how="inner", suffixes=("", "_meta"))
    barcode_to_idx = pd.Series(np.arange(len(barcodes)), index=barcodes)
    obs["barcode_idx"] = obs["barcode"].map(barcode_to_idx)
    obs = obs[
        (obs["in_tissue"] == 1)
        & (obs["barcode_idx"].notna())
        & (obs["DeconvolutionClass"] == "singlet")
        & (obs["DeconvolutionLabel1"].notna())
    ].copy()
    obs["barcode_idx"] = obs["barcode_idx"].astype(int)
    factor = args.bin_size_um // args.source_bin_um
    obs["coarse_row"] = (obs["array_row"].astype(int) // factor).astype(str)
    obs["coarse_col"] = (obs["array_col"].astype(int) // factor).astype(str)
    obs["spot"] = args.prefix + "_r" + obs["coarse_row"] + "_c" + obs["coarse_col"]
    obs["celltype"] = obs["DeconvolutionLabel1"].astype(str).str.replace(r"\s+", "_", regex=True)
    obs = obs.sort_values(["spot", "celltype", "barcode_idx"]).reset_index(drop=True)

    x_barcode_by_gene = x_gene_by_barcode[selected_gene_idx, :][:, obs["barcode_idx"].to_numpy()].T.tocsr()

    spot_codes, spot_names = pd.factorize(obs["spot"], sort=True)
    type_codes, celltypes = pd.factorize(obs["celltype"], sort=True)
    spot_cell_keys = obs["spot"].astype(str) + "_" + obs["celltype"].astype(str)
    spot_cell_codes, spot_cell_names = pd.factorize(spot_cell_keys, sort=True)

    membership = sp.csr_matrix(
        (np.ones(len(obs), dtype=np.float32), (spot_cell_codes, np.arange(len(obs)))),
        shape=(len(spot_cell_names), len(obs)),
    )
    spot_cell_expr = membership @ x_barcode_by_gene
    spot_cell_df = pd.DataFrame.sparse.from_spmatrix(
        spot_cell_expr,
        index=pd.Index(spot_cell_names, name="spot_cell"),
        columns=selected_genes,
    )
    spot_cell_df = spot_cell_df.sparse.to_dense()

    count_table = pd.crosstab(
        pd.Categorical(obs["spot"], categories=spot_names),
        pd.Categorical(obs["celltype"], categories=celltypes),
    )
    composition = count_table.div(count_table.sum(axis=1), axis=0).fillna(0.0)
    composition.index.name = None

    spot_membership = sp.csr_matrix(
        (np.ones(len(obs), dtype=np.float32), (spot_codes, np.arange(len(obs)))),
        shape=(len(spot_names), len(obs)),
    )
    spot_expr = spot_membership @ x_barcode_by_gene
    coords = obs.groupby("spot", sort=True)[["pxl_col_in_fullres", "pxl_row_in_fullres"]].mean()
    coords = coords.reindex(spot_names)
    adata = ad.AnnData(
        X=spot_expr.tocsr(),
        obs=pd.DataFrame(index=pd.Index(spot_names, name="spot")),
        var=pd.DataFrame(index=pd.Index(selected_genes, name="gene")),
    )
    adata.obsm["spatial"] = coords[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=np.float32)

    expr_path = output_dir / f"{args.prefix}_spot_cell_expr.csv"
    comp_path = output_dir / f"{args.prefix}_composition.csv"
    h5ad_path = output_dir / f"{args.prefix}.h5ad"
    spot_cell_df.to_csv(expr_path)
    composition.to_csv(comp_path)
    adata.write_h5ad(h5ad_path)

    print(f"selected_genes={len(selected_genes)}")
    print(f"source_bins={len(obs)}")
    print(f"spots={adata.n_obs}")
    print(f"celltypes={len(celltypes)}")
    print(f"spot_cells={spot_cell_df.shape[0]}")
    print(f"wrote={expr_path}")
    print(f"wrote={comp_path}")
    print(f"wrote={h5ad_path}")


if __name__ == "__main__":
    main()
