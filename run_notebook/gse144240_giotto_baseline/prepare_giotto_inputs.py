from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import pandas as pd
from scipy import io, sparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SCC inputs for Giotto spatial CCC baseline.")
    parser.add_argument("--st-h5ad", type=Path, required=True)
    parser.add_argument("--composition-csv", type=Path, required=True)
    parser.add_argument("--cellchat-db-csv", type=Path, required=True)
    parser.add_argument("--spagraph-lr-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--sample-name", default="GSE144240_P2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    st = ad.read_h5ad(args.st_h5ad)
    st.obs_names = st.obs_names.astype(str)

    comp = pd.read_csv(args.composition_csv, index_col=0)
    comp.index = comp.index.astype(str)
    comp = comp.loc[st.obs_names]

    labels = comp.idxmax(axis=1)
    dominant_fraction = comp.max(axis=1)
    meta = pd.DataFrame(
        {
            "cell_ID": st.obs_names,
            "labels": labels.to_numpy(),
            "dominant_fraction": dominant_fraction.to_numpy(),
            "sample": args.sample_name,
        }
    )
    coords = pd.DataFrame(st.obsm["spatial"], index=st.obs_names, columns=["sdimx", "sdimy"]).reset_index()
    coords = coords.rename(columns={"index": "cell_ID"})

    db = pd.read_csv(args.cellchat_db_csv, usecols=["interaction_name", "ligand", "receptor"])
    spg_pairs = pd.read_csv(args.spagraph_lr_csv, usecols=["lr_pair"]).drop_duplicates()
    st_genes = set(map(str, st.var_names))
    db_simple = db[
        (~db["ligand"].astype(str).str.contains("_", regex=False))
        & (~db["receptor"].astype(str).str.contains("_", regex=False))
    ].copy()
    db_simple = db_simple[
        db_simple["ligand"].isin(st_genes)
        & db_simple["receptor"].isin(st_genes)
    ].copy()
    shared = db_simple[db_simple["interaction_name"].isin(set(spg_pairs["lr_pair"]))].copy()
    shared = shared.drop_duplicates(subset=["interaction_name"]).sort_values("interaction_name").reset_index(drop=True)

    expr = st.X
    if not sparse.issparse(expr):
        expr = sparse.csr_matrix(expr)
    expr = expr.tocoo()
    io.mmwrite(str(args.outdir / "expression_raw.mtx"), expr.transpose())
    pd.Series(st.var_names, name="gene").to_csv(args.outdir / "genes.tsv", sep="\t", index=False, header=False)
    pd.Series(st.obs_names, name="spot").to_csv(args.outdir / "spots.tsv", sep="\t", index=False, header=False)
    meta.to_csv(args.outdir / "meta.csv", index=False)
    coords.to_csv(args.outdir / "spatial_locs.csv", index=False)
    shared.to_csv(args.outdir / "shared_simple_pairs.csv", index=False)

    summary = [
        f"n_spots={st.n_obs}",
        f"n_genes={st.n_vars}",
        f"n_shared_simple_pairs={len(shared)}",
        "dominant_label_counts:",
    ]
    summary.extend(f"  {label}: {count}" for label, count in labels.value_counts().items())
    (args.outdir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
