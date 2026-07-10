# GSE144240 CellChat Baseline

This bundle prepares and runs a CellChat v2 spatial baseline for the `GSE144240` Visium sample currently stored in the repo as `GSE144236_P2`.

## Files

- `prepare_cellchat_inputs.py`: builds the CellChat input folder from `ST.h5ad` and the deconvolution composition csv.
- `install_cellchat_v2.R`: installs CellChat v2 and the common packages that usually block GitHub installation.
- `run_cellchat_spatial.R`: runs CellChat v2 in spatial mode and exports ligand-receptor and pathway tables.
- `run_server_example.sh`: minimal Linux entrypoint.
- `input/`: prepared files ready to upload.

## Expected Input Logic

Because CellChat v2 expects one discrete label per spot, the prepared `meta.tsv` uses the dominant cell type from `Spatial_composition.csv` as the baseline label and keeps the dominant fraction in a separate column for optional filtering.

## Upload-To-Server Workflow

1. Upload this whole folder.
2. Enter the folder on the server.
3. Run:

```bash
chmod +x run_server_example.sh
./run_server_example.sh
```

## Manual Run

```bash
export CELLCHAT_R_LIB="$(pwd)/.r_libs/cellchat"

Rscript install_cellchat_v2.R

Rscript run_cellchat_spatial.R \
  --input-dir ./input \
  --output-dir ./output \
  --min-dominant-fraction 0.0 \
  --min-spots-per-group 10 \
  --nboot 100 \
  --workers 4
```

## Important Parameters

- `--min-dominant-fraction`: drop mixed spots below this dominant-label confidence.
- `--min-spots-per-group`: groups smaller than this are removed before inference.
- `--ratio`, `--tol`, `--interaction-range`, `--contact-range`, `--scale-distance`: spatial parameters. If omitted, the script uses the values stored in `input/spatial_factors.tsv`.

## Outputs

- `cellchat_lr_communications.tsv`
- `cellchat_pathway_communications.tsv`
- `cellchat_group_sizes.tsv`
- `cellchat_meta_used.tsv`
- `run_config.tsv`
- `sessionInfo.txt`
- `cellchat_object.rds`
