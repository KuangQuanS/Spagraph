# basic imports
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import tangram as tg


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
OUTPUT_DIR = REPO_ROOT / "evaluate" / "data" / "seqFISH+" / "tangram_seqFISH+"

sc_file_path = REPO_ROOT / "spagraph_data" / "database" / "seqFISH+" / "scRNA.h5ad"
spatial_file_path = REPO_ROOT / "spagraph_data" / "database" / "seqFISH+" / "Spatial.h5ad"
celltype_key = "celltype"
output_file_path = OUTPUT_DIR / "composition.csv"
gene_expression_output = OUTPUT_DIR / "trangram_expression.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ad_sc = sc.read_h5ad(sc_file_path)
ad_sp = sc.read_h5ad(spatial_file_path)

# use raw count both of scrna and spatial
sc.pp.normalize_total(ad_sc)
celltype_counts = ad_sc.obs[celltype_key].value_counts()
celltype_drop = celltype_counts.index[celltype_counts < 2]
print(f"Drop celltype {list(celltype_drop)} contain less 2 sample")
ad_sc = ad_sc[~ad_sc.obs[celltype_key].isin(celltype_drop),].copy()
sc.tl.rank_genes_groups(ad_sc, groupby=celltype_key, use_raw=False)
markers_df = pd.DataFrame(ad_sc.uns["rank_genes_groups"]["names"]).iloc[0:200, :]
print(markers_df)
genes_sc = np.unique(markers_df.melt().value.values)
print(genes_sc)
genes_st = ad_sp.var_names.values
genes = list(set(genes_sc).intersection(set(genes_st)))

tg.pp_adatas(ad_sc, ad_sp, genes=genes)

ad_map = tg.map_cells_to_space(
    ad_sc,
    ad_sp,
    mode="clusters",
    cluster_label=celltype_key,
)

tg.project_cell_annotations(ad_map, ad_sp, annotation=celltype_key)

celltype_names = ad_map.obs_names.tolist()
print(f"Cell types: {celltype_names}")
print(f"ad_map shape: {ad_map.shape}")
print(f"ad_map.obs columns: {ad_map.obs.columns.tolist()}")

cell_composition_tangram = pd.DataFrame(
    ad_map.X.T,
    index=ad_sp.obs_names,
    columns=celltype_names,
)

print(f"\nCell composition matrix shape: {cell_composition_tangram.shape}")
print(f"Column names: {cell_composition_tangram.columns.tolist()}")
print(f"\nFirst few rows:\n{cell_composition_tangram.head()}")

row_sums = cell_composition_tangram.sum(axis=1)
print(f"\nRow sums - min: {row_sums.min():.4f}, max: {row_sums.max():.4f}, mean: {row_sums.mean():.4f}")

print("\n" + "=" * 60)
print("NORMALIZING TANGRAM RESULTS")
print("=" * 60)

row_sums = cell_composition_tangram.sum(axis=1)
print(f"\nBEFORE normalization:")
print(f"   Row sums - min: {row_sums.min():.6f}, max: {row_sums.max():.6f}, mean: {row_sums.mean():.6f}")
print(f"   Example: first row sum = {row_sums.iloc[0]:.6f}")

cell_composition_tangram = cell_composition_tangram.div(row_sums, axis=0)

row_sums_after = cell_composition_tangram.sum(axis=1)
print(f"\nAFTER normalization:")
print(
    f"   Row sums - min: {row_sums_after.min():.6f}, "
    f"max: {row_sums_after.max():.6f}, mean: {row_sums_after.mean():.6f}"
)
print(f"   Example: first row sum = {row_sums_after.iloc[0]:.6f}")

print(f"\nFirst row values (now as percentages):")
print(cell_composition_tangram.head())

cell_composition_tangram.to_csv(output_file_path)
print(f"\nSaved normalized results to {output_file_path}")
print("=" * 60)

ad_ge = tg.project_genes(
    adata_map=ad_map,
    adata_sc=ad_sc,
    cluster_label=celltype_key,
)

print("\nGene projection completed!")
df_expression = ad_ge.to_df()
df_expression.columns = df_expression.columns.str.upper()
df_expression.to_csv(gene_expression_output)
print(f"Saved gene expression to {gene_expression_output}")
