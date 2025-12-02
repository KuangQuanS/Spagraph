import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt


def load_and_align(pred_csv: str, gt_h5ad: str, target_gene: str):
    pred_df = pd.read_csv(pred_csv, index_col=0)
    gt_adata = sc.read_h5ad(gt_h5ad)

    if "spatial" not in gt_adata.obsm:
        raise ValueError("Ground truth h5ad missing obsm['spatial'] coordinates.")

    shared_spots = pred_df.index.intersection(gt_adata.obs_names)
    if len(shared_spots) == 0:
        raise ValueError("No overlapping spots between predicted CSV and ground truth h5ad.")

    if target_gene not in pred_df.columns:
        raise ValueError(f"Target gene '{target_gene}' not found in predicted CSV columns.")
    if target_gene not in gt_adata.var_names:
        raise ValueError(f"Target gene '{target_gene}' not found in ground truth h5ad genes.")

    pred_vec = pred_df.loc[shared_spots, target_gene].astype(float)
    gt_slice = gt_adata[shared_spots, target_gene]
    gt_mat = gt_slice.X.toarray() if hasattr(gt_slice.X, "toarray") else gt_slice.X
    gt_vec = np.asarray(gt_mat).ravel().astype(float)

    coords = gt_adata.obsm["spatial"]
    coords = coords[[gt_adata.obs_names.get_loc(s) for s in shared_spots], :]

    return coords, gt_vec, pred_vec.values, shared_spots


def plot_heatmaps(coords, gt_vec, pred_vec, target_gene: str, output_png: str, log1p: bool, clip_percentile: float):
    if log1p:
        gt_vec = np.log1p(gt_vec)
        pred_vec = np.log1p(pred_vec)

    all_vals = np.concatenate([gt_vec, pred_vec])
    vmax = np.percentile(all_vals, clip_percentile)
    vmin = np.percentile(all_vals, 100 - clip_percentile) if log1p else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    titles = ["Ground truth", "Prediction"]
    data = [gt_vec, pred_vec]
    for ax, vals, title in zip(axes, data, titles):
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, cmap="viridis", s=8, vmin=vmin, vmax=vmax)
        ax.set_title(f"{title} - {target_gene}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="log1p(expr)" if log1p else "expr")

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmaps to {output_png}")


def main():
    parser = argparse.ArgumentParser(description="Plot target gene heatmaps for ground truth and prediction.")
    parser.add_argument("--pred_csv", required=True, help="Predicted expression CSV (spots x genes).")
    parser.add_argument("--ground_truth_h5ad", required=True, help="Ground truth ST h5ad.")
    parser.add_argument("--target_gene", required=True, help="Target gene to plot.")
    parser.add_argument("--output_png", default=None, help="Output PNG path (default: <target_gene>_heatmap.png).")
    parser.add_argument("--log1p", action="store_true", help="Apply log1p before plotting.")
    parser.add_argument("--clip_percentile", type=float, default=99.0,
                        help="Percentile for color scaling (default: 99).")
    args = parser.parse_args()

    coords, gt_vec, pred_vec, _ = load_and_align(args.pred_csv, args.ground_truth_h5ad, args.target_gene)
    output_png = args.output_png or f"{args.target_gene}_heatmap.png"
    plot_heatmaps(coords, gt_vec, pred_vec, args.target_gene, output_png, args.log1p, args.clip_percentile)


if __name__ == "__main__":
    main()
