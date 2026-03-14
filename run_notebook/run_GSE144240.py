from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import spagraph as spg

sc_file = f"/home/maweicheng/ST_data/GSE144240/scRNA.h5ad"
st_file = f"/home/maweicheng/ST_data/GSE144240/Spatial.h5ad"
output_dir = f"/home/maweicheng/ST_data/GSE144240/evaluate"
# marker_selection_method="l1"
# art = spg.vae(sc_file=sc_file, st_file=st_file, resolution=4, top_n_per_type=100, output_dir=output_dir, precomputed_marker_file=f"{output_dir}/final_genes.txt")
art = spg.vae(sc_file=sc_file,
              st_file=st_file,
              top_n_per_type=100,
              output_dir=output_dir
              )

res = spg.deconv(vae=art,
                 output_dir=output_dir,
                 k_cells_per_cluster=15,
                 k_celltype=40,
                 scale_basis="all",
                 save_reconstructed_genes=True
                 )

spg.cellcom(
    deconv_dir=output_dir,
    st_h5ad=st_file,
    output_dir=output_dir,
    ligand_expr_threshold=3,
    receptor_expr_threshold=3,
    n_spot_neighbors=8,
    epochs=200,
    batch_size=128,
    seed=42,
    device="cuda:1"
)
