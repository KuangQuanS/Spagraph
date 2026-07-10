from __future__ import annotations

import spagraph


spagraph.cellcom(
    deconv_dir="results/gse280315_visiumhd_crc_p5_128um_inputs",
    st_h5ad="results/gse280315_visiumhd_crc_p5_128um_inputs/GSM8594569_P5CRC_128um.h5ad",
    output_dir="results/gse280315_visiumhd_crc_p5_128um_cellcom_smoke",
    n_spot_neighbors=8,
    ligand_expr_threshold=3.0,
    receptor_expr_threshold=1.0,
    lr_score_threshold=1.0,
    min_comm_edges=1,
    epochs=10,
    batch_size=4,
    device="cuda",
    save_lr_scores_csv=False,
    export_unified_csv=False,
    export_filtered_csv=True,
    seed=42,
)
