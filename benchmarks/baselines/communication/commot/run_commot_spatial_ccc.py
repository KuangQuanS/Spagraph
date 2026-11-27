from __future__ import annotations

import argparse
import importlib
import site
import sys
import types
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run COMMOT spatial CCC baseline for GSE144240/GSE144236.")
    parser.add_argument("--st-h5ad", type=Path, required=True)
    parser.add_argument("--composition-csv", type=Path, required=True)
    parser.add_argument("--cellchat-db-csv", type=Path, required=True)
    parser.add_argument("--spagraph-lr-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--database-name", default="scc_shared")
    parser.add_argument("--clustering-column", default="labels")
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--visium-neighbor-distance-um", type=float, default=100.0)
    parser.add_argument("--interaction-range-um", type=float, default=250.0)
    parser.add_argument("--min-spots-per-group", type=int, default=1)
    parser.add_argument("--cot-nitermax", type=int, default=200)
    parser.add_argument("--cot-rho", type=float, default=10.0)
    parser.add_argument("--cot-eps-p", type=float, default=0.1)
    parser.add_argument("--n-permutations", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--max-pairs", type=int, default=0, help="Optional cap for smoke tests; 0 means use all shared pairs.")
    return parser.parse_args()


def load_commot_functions():
    # COMMOT 0.0.3 imports a broad dependency tree in package __init__.
    # For this baseline we only need the spatial CCC functions.
    np.Inf = np.inf
    site_packages = [Path(p) for p in site.getsitepackages()]
    candidates = [p / "commot" for p in site_packages]
    matches = [p for p in candidates if p.exists()]
    if not matches:
        raise FileNotFoundError("Could not find installed COMMOT package directory in site-packages.")
    base = matches[0]
    pkg = types.ModuleType("commot")
    pkg.__path__ = [str(base)]
    sys.modules["commot"] = pkg

    tools_pkg = types.ModuleType("commot.tools")
    tools_pkg.__path__ = [str(base / "tools")]
    sys.modules["commot.tools"] = tools_pkg

    mod = importlib.import_module("commot.tools._spatial_communication")
    return mod.spatial_communication, mod.cluster_communication


def normalize_log1p(adata: ad.AnnData, target_sum: float) -> ad.AnnData:
    x = adata.X
    if not sparse.issparse(x):
        x = sparse.csr_matrix(x)
    x = x.tocsr(copy=True).astype(np.float64)
    out = ad.AnnData(X=x, obs=adata.obs.copy(), var=adata.var.copy(), obsm={"spatial": np.asarray(adata.obsm["spatial"])})
    sc.pp.normalize_total(out, target_sum=target_sum, inplace=True)
    sc.pp.log1p(out)
    return out


def gene_set_has_all(lr_name: str, genes: set[str], delimiter: str = "_") -> bool:
    return set(str(lr_name).split(delimiter)).issubset(genes)


def build_shared_ligrec(st: ad.AnnData, cellchat_db_csv: Path, spagraph_lr_csv: Path) -> pd.DataFrame:
    db = pd.read_csv(cellchat_db_csv)
    spg = pd.read_csv(spagraph_lr_csv, usecols=["lr_pair"]).drop_duplicates()
    st_genes = set(map(str, st.var_names))

    pathway_col = "pathway_name" if "pathway_name" in db.columns else "pathway_name_2"
    if pathway_col not in db.columns:
        pathway_col = "annotation"

    shared = db[db["interaction_name"].isin(set(spg["lr_pair"]))].copy()
    shared = shared[shared["ligand"].map(lambda x: gene_set_has_all(x, st_genes))]
    shared = shared[shared["receptor"].map(lambda x: gene_set_has_all(x, st_genes))]
    shared = shared[["interaction_name", "ligand", "receptor", pathway_col]].copy()
    shared.columns = ["interaction_name", "ligand", "receptor", "pathway"]
    shared = shared.drop_duplicates(subset=["interaction_name"]).sort_values("interaction_name").reset_index(drop=True)
    return shared


def summarize_pair_scores(
    adata: ad.AnnData,
    shared_ligrec: pd.DataFrame,
    database_name: str,
    clustering_column: str,
    n_permutations: int,
    random_seed: int,
    cluster_communication,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, object]] = []

    for row in shared_ligrec.itertuples(index=False):
        cluster_communication(
            adata,
            database_name=database_name,
            lr_pair=(row.ligand, row.receptor),
            clustering=clustering_column,
            n_permutations=n_permutations,
            random_seed=random_seed,
            copy=False,
        )
        key = f"commot_cluster-{clustering_column}-{database_name}-{row.ligand}-{row.receptor}"
        result = adata.uns[key]
        score_mat = result["communication_matrix"]
        pval_mat = result["communication_pvalue"]
        for source in score_mat.index:
            for target in score_mat.columns:
                detail_rows.append(
                    {
                        "interaction_name": row.interaction_name,
                        "ligand": row.ligand,
                        "receptor": row.receptor,
                        "pathway": row.pathway,
                        "source": source,
                        "target": target,
                        "commot_score": float(score_mat.loc[source, target]),
                        "commot_pvalue": float(pval_mat.loc[source, target]),
                    }
                )

    detail = pd.DataFrame(detail_rows)
    summary = (
        detail.sort_values(["interaction_name", "commot_score"], ascending=[True, False])
        .groupby("interaction_name", as_index=False)
        .first()
    )
    summary = summary.rename(columns={"source": "best_source", "target": "best_target"})
    summary["commot_rank"] = summary["commot_score"].rank(method="min", ascending=False).astype(int)
    summary["commot_percentile"] = (
        100.0 * (1.0 - (summary["commot_rank"] - 1) / np.maximum(len(summary) - 1, 1))
    )
    summary = summary.sort_values(["commot_rank", "interaction_name"]).reset_index(drop=True)

    cross = detail[detail["source"] != detail["target"]].copy()
    summary_cross = (
        cross.sort_values(["interaction_name", "commot_score"], ascending=[True, False])
        .groupby("interaction_name", as_index=False)
        .first()
    )
    summary_cross = summary_cross.rename(columns={"source": "best_source", "target": "best_target"})
    summary_cross["commot_rank"] = summary_cross["commot_score"].rank(method="min", ascending=False).astype(int)
    summary_cross["commot_percentile"] = (
        100.0 * (1.0 - (summary_cross["commot_rank"] - 1) / np.maximum(len(summary_cross) - 1, 1))
    )
    summary_cross = summary_cross.sort_values(["commot_rank", "interaction_name"]).reset_index(drop=True)
    return detail, summary, summary_cross


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    spatial_communication, cluster_communication = load_commot_functions()

    st = ad.read_h5ad(args.st_h5ad)
    st.obs_names = st.obs_names.astype(str)

    comp = pd.read_csv(args.composition_csv, index_col=0)
    comp.index = comp.index.astype(str)
    comp = comp.loc[st.obs_names]

    labels = comp.idxmax(axis=1)
    dominant_fraction = comp.max(axis=1)
    group_sizes_full = labels.value_counts()
    valid_groups = group_sizes_full[group_sizes_full >= args.min_spots_per_group].index
    keep = labels.isin(valid_groups)

    st = st[keep.to_numpy(), :].copy()
    comp = comp.loc[st.obs_names]
    labels = labels.loc[st.obs_names]
    dominant_fraction = dominant_fraction.loc[st.obs_names]

    shared_ligrec = build_shared_ligrec(st, args.cellchat_db_csv, args.spagraph_lr_csv)
    if args.max_pairs > 0:
        shared_ligrec = shared_ligrec.head(args.max_pairs).copy()

    adata = normalize_log1p(st, target_sum=args.target_sum)
    adata.obs[args.clustering_column] = labels.astype(str).values
    adata.obs["dominant_fraction"] = dominant_fraction.values

    raw_nn = float(np.median(cKDTree(np.asarray(adata.obsm["spatial"])).query(np.asarray(adata.obsm["spatial"]), k=2)[0][:, 1]))
    ratio = args.visium_neighbor_distance_um / raw_nn
    dis_thr_raw = args.interaction_range_um / ratio

    ligrec = shared_ligrec[["ligand", "receptor", "pathway"]].copy()
    spatial_communication(
        adata,
        database_name=args.database_name,
        df_ligrec=ligrec,
        pathway_sum=False,
        heteromeric=True,
        heteromeric_rule="min",
        heteromeric_delimiter="_",
        dis_thr=dis_thr_raw,
        cot_eps_p=args.cot_eps_p,
        cot_rho=args.cot_rho,
        cot_nitermax=args.cot_nitermax,
        copy=False,
    )

    detail, summary, summary_cross = summarize_pair_scores(
        adata=adata,
        shared_ligrec=shared_ligrec,
        database_name=args.database_name,
        clustering_column=args.clustering_column,
        n_permutations=args.n_permutations,
        random_seed=args.random_seed,
        cluster_communication=cluster_communication,
    )

    group_sizes = labels.value_counts().rename_axis("label").reset_index(name="spots")
    run_config = pd.DataFrame(
        [
            {
                "database_name": args.database_name,
                "n_spots": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "n_shared_pairs": int(len(shared_ligrec)),
                "min_spots_per_group": args.min_spots_per_group,
                "target_sum": args.target_sum,
                "visium_neighbor_distance_um": args.visium_neighbor_distance_um,
                "interaction_range_um": args.interaction_range_um,
                "raw_nn_distance_median": raw_nn,
                "ratio_um_per_raw_unit": ratio,
                "dis_thr_raw": dis_thr_raw,
                "cot_eps_p": args.cot_eps_p,
                "cot_rho": args.cot_rho,
                "cot_nitermax": args.cot_nitermax,
                "n_permutations": args.n_permutations,
                "random_seed": args.random_seed,
                "max_pairs": args.max_pairs,
            }
        ]
    )

    shared_ligrec.to_csv(args.outdir / "commot_shared_ligrec.csv", index=False)
    detail.to_csv(args.outdir / "commot_cluster_pair_scores.csv", index=False)
    summary.to_csv(args.outdir / "commot_pair_summary.csv", index=False)
    summary_cross.to_csv(args.outdir / "commot_pair_summary_cross_group.csv", index=False)
    group_sizes.to_csv(args.outdir / "commot_group_sizes.csv", index=False)
    run_config.to_csv(args.outdir / "run_config.csv", index=False)

    summary_lines = [
        f"st_h5ad={args.st_h5ad}",
        f"composition_csv={args.composition_csv}",
        f"cellchat_db_csv={args.cellchat_db_csv}",
        f"spagraph_lr_csv={args.spagraph_lr_csv}",
        f"min_spots_per_group={args.min_spots_per_group}",
        f"n_shared_pairs={len(shared_ligrec)}",
        f"raw_nn_distance_median={raw_nn:.6f}",
        f"ratio_um_per_raw_unit={ratio:.6f}",
        f"dis_thr_raw={dis_thr_raw:.6f}",
        f"cot_nitermax={args.cot_nitermax}",
        f"n_permutations={args.n_permutations}",
        "",
        "group_sizes:",
    ]
    summary_lines.extend(f"  {row.label}: {row.spots}" for row in group_sizes.itertuples(index=False))
    (args.outdir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
