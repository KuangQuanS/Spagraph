import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from typing import List


def align_data(recon_df: pd.DataFrame, gt_adata):
    """Align spots and genes between reconstructed csv and ground truth h5ad."""
    gt_df = pd.DataFrame(gt_adata.X.toarray() if hasattr(gt_adata.X, "toarray") else gt_adata.X,
                         index=gt_adata.obs_names,
                         columns=gt_adata.var_names)

    shared_spots = recon_df.index.intersection(gt_df.index)
    if len(shared_spots) == 0:
        raise ValueError("No overlapping spots between reconstructed matrix and ground truth h5ad.")

    recon_df = recon_df.loc[shared_spots]
    gt_df = gt_df.loc[shared_spots]

    shared_genes = recon_df.columns.intersection(gt_df.columns)
    if len(shared_genes) == 0:
        raise ValueError("No overlapping genes between reconstructed matrix and ground truth h5ad.")

    recon_df = recon_df[shared_genes].astype(float)
    gt_df = gt_df[shared_genes].astype(float)

    # Sanitize inf/nan to avoid failures in downstream HVG selection/metrics
    recon_df = recon_df.replace([np.inf, -np.inf], np.nan)
    gt_df = gt_df.replace([np.inf, -np.inf], np.nan)
    recon_df = recon_df.fillna(0)
    gt_df = gt_df.fillna(0)

    return gt_df, recon_df, list(shared_genes)


def select_hvg(gt_df: pd.DataFrame, top_n: int = 1000):
    """Select top-N highly variable genes from ground truth (shared genes only).
    
    Uses standard ST pipeline: library-size normalize -> log1p -> HVG (Seurat flavor).
    """
    adata = sc.AnnData(gt_df.values, obs=gt_df.index.to_frame(), var=pd.DataFrame(index=gt_df.columns))
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=min(top_n, gt_df.shape[1]), flavor="seurat")
    hvg_genes = list(adata.var.index[adata.var["highly_variable"]])
    return hvg_genes


def load_gene_list(path: str, shared_genes: List[str]) -> List[str]:
    """Load user-specified genes and keep intersection with shared genes, preserving order."""
    with open(path, "r") as f:
        raw_genes = [line.strip() for line in f if line.strip()]
    seen = set()
    raw_genes = [g for g in raw_genes if not (g in seen or seen.add(g))]

    shared_set = set(shared_genes)
    selected = [g for g in raw_genes if g in shared_set]
    return selected


def compute_metrics(gt_mat: np.ndarray, pred_mat: np.ndarray, genes):
    """Compute PCC, SSIM, RMSE, JS per gene."""
    eps = 1e-8
    C1 = 0.01
    C2 = 0.03

    records = []
    for idx, gene in enumerate(genes):
        true_vec = np.asarray(gt_mat[:, idx], dtype=float)
        pred_vec = np.asarray(pred_mat[:, idx], dtype=float)

        finite_mask = np.isfinite(true_vec) & np.isfinite(pred_vec)
        if finite_mask.sum() == 0:
            records.append({"gene": gene, "pcc": np.nan, "ssim": np.nan, "rmse": np.nan, "js": np.nan})
            continue
        true_vec = true_vec[finite_mask]
        pred_vec = pred_vec[finite_mask]

        # PCC
        t_mean = true_vec.mean()
        p_mean = pred_vec.mean()
        t_std = true_vec.std()
        p_std = pred_vec.std()
        if t_std < eps or p_std < eps:
            pcc = np.nan
        else:
            pcc = np.mean((true_vec - t_mean) * (pred_vec - p_mean)) / (t_std * p_std)

        # SSIM (per gene, scaled to [0,1])
        t_max = true_vec.max()
        p_max = pred_vec.max()
        t_scaled = true_vec / (t_max + eps)
        p_scaled = pred_vec / (p_max + eps)

        t_mu = t_scaled.mean()
        p_mu = p_scaled.mean()
        t_var = t_scaled.var()
        p_var = p_scaled.var()
        cov = np.mean((t_scaled - t_mu) * (p_scaled - p_mu))

        ssim_num = (2 * p_mu * t_mu + C1**2) * (2 * cov + C2**2)
        ssim_den = (p_mu**2 + t_mu**2 + C1**2) * (p_var + t_var + C2**2)
        ssim = ssim_num / ssim_den if ssim_den > eps else np.nan

        # RMSE on z-scores
        t_std_z = t_std if t_std >= eps else 1.0
        p_std_z = p_std if p_std >= eps else 1.0
        t_z = (true_vec - t_mean) / t_std_z
        p_z = (pred_vec - p_mean) / p_std_z
        rmse = np.sqrt(np.mean((p_z - t_z) ** 2))

        # Jensen-Shannon divergence
        t_sum = true_vec.sum()
        p_sum = pred_vec.sum()
        t_prob = true_vec / (t_sum + eps)
        p_prob = pred_vec / (p_sum + eps)
        m_prob = 0.5 * (t_prob + p_prob)

        def kl(a, b):
            return np.sum(a * np.log((a + eps) / (b + eps)))

        js = 0.5 * kl(t_prob, m_prob) + 0.5 * kl(p_prob, m_prob)

        records.append({
            "gene": gene,
            "pcc": pcc,
            "ssim": ssim,
            "rmse": rmse,
            "js": js
        })

    return pd.DataFrame(records)


def save_boxplots(metrics_df: pd.DataFrame, output_csv: str, title: str = "Metrics Boxplot"):
    """Save boxplots for available metric columns."""
    metric_cols = [col for col in ['pcc', 'ssim', 'rmse', 'js'] if col in metrics_df.columns]
    if not metric_cols:
        return
    data = [metrics_df[col].dropna() for col in metric_cols]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(data, labels=metric_cols, showfliers=False)
    ax.set_ylabel("Score")
    ax.set_title(title)
    plt.tight_layout()
    png_path = output_csv.rsplit('.', 1)[0] + "_boxplot.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Boxplot saved to {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark metrics for reconstructed expression.")
    parser.add_argument("--reconstructed_csv", required=False,
                        help="Path to reconstructed all-genes CSV (spots x genes).")
    parser.add_argument("--ground_truth_h5ad", required=False,
                        help="Path to ground truth ST h5ad.")
    parser.add_argument("--output_csv", default="benchmark_metrics.csv",
                        help="Path to save per-gene metrics CSV.")
    parser.add_argument("--top_hvg", type=int, default=1000,
                        help="Number of highly variable genes to evaluate (default: 1000).")
    parser.add_argument("--gene_list", type=str, default=None,
                        help="Optional path to a gene list (one gene per line). If provided, use these genes (intersected with shared genes) instead of HVG selection.")
    parser.add_argument("--composition_pred_csv", type=str, default=None,
                        help="Predicted cell composition CSV (spots x celltypes). If provided with --composition_true_csv, compute metrics on compositions instead of gene expression.")
    parser.add_argument("--composition_true_csv", type=str, default=None,
                        help="Ground truth cell composition CSV (spots x celltypes).")
    args = parser.parse_args()

    # ============ Composition mode ============
    if args.composition_pred_csv and args.composition_true_csv:
        pred_comp = pd.read_csv(args.composition_pred_csv, index_col=0)
        true_comp = pd.read_csv(args.composition_true_csv, index_col=0)

        shared_spots = pred_comp.index.intersection(true_comp.index)
        if len(shared_spots) == 0:
            raise ValueError("No overlapping spots between composition CSVs.")
        pred_comp = pred_comp.loc[shared_spots]
        true_comp = true_comp.loc[shared_spots]

        shared_cols = [c for c in true_comp.columns if c in pred_comp.columns]
        if len(shared_cols) == 0:
            raise ValueError("No overlapping celltype columns between composition CSVs.")
        pred_comp = pred_comp[shared_cols].astype(float)
        true_comp = true_comp[shared_cols].astype(float)

        pred_comp = pred_comp.replace([np.inf, -np.inf], np.nan).fillna(0)
        true_comp = true_comp.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Row-normalize true composition to proportions (handle count-style inputs)
        true_row_sum = true_comp.sum(axis=1)
        true_row_sum[true_row_sum == 0] = 1.0
        true_comp = true_comp.div(true_row_sum, axis=0)

        # Pred is assumed already normalized; warn if not close to 1
        pred_row_sum = pred_comp.sum(axis=1)
        if not np.allclose(pred_row_sum.values, 1.0, atol=1e-3):
            print("⚠️ Warning: predicted composition rows do not sum to 1 (max diff "
                  f"{np.max(np.abs(pred_row_sum.values - 1.0)):.4f}). Using provided values without renorm.")

        metrics_df = compute_metrics(true_comp.values, pred_comp.values, list(shared_cols))
        metrics_df.to_csv(args.output_csv, index=False)

        print(f"[Composition mode] Spots aligned: {pred_comp.shape[0]}, Celltypes evaluated: {len(shared_cols)}")
        print(f"Mean PCC: {np.nanmean(metrics_df['pcc']):.4f}")
        print(f"Mean SSIM: {np.nanmean(metrics_df['ssim']):.4f}")
        print(f"Mean RMSE: {metrics_df['rmse'].mean():.4f}")
        print(f"Mean JS: {metrics_df['js'].mean():.4f}")
        print(f"Per-celltype metrics saved to {args.output_csv}")
        save_boxplots(metrics_df, args.output_csv, title="Composition Metrics")
        return

    # ============ Expression mode ============
    if not args.reconstructed_csv or not args.ground_truth_h5ad:
        raise ValueError("Provide --reconstructed_csv and --ground_truth_h5ad for expression evaluation, or both composition CSVs for composition mode.")

    recon_df = pd.read_csv(args.reconstructed_csv, index_col=0)
    gt_adata = sc.read_h5ad(args.ground_truth_h5ad)

    gt_df, pred_df, genes = align_data(recon_df, gt_adata)

    # Select genes: prefer user-provided list; otherwise HVG
    if args.gene_list:
        selected_genes = load_gene_list(args.gene_list, genes)
        if len(selected_genes) == 0:
            raise ValueError("No genes from gene_list found in shared genes between datasets.")
        print(f"Using user gene list: {len(selected_genes)} genes (from {args.gene_list})")
    else:
        selected_genes = select_hvg(gt_df, top_n=args.top_hvg)
        if len(selected_genes) == 0:
            raise ValueError("No highly variable genes selected. Check input data.")
        if len(selected_genes) < args.top_hvg:
            print(f"Warning: only {len(selected_genes)} HVGs available (requested {args.top_hvg}).")
        print(f"Using top {len(selected_genes)} HVGs.")

    gt_mat = gt_df[selected_genes].values
    pred_mat = pred_df[selected_genes].values

    metrics_df = compute_metrics(gt_mat, pred_mat, selected_genes)
    metrics_df.to_csv(args.output_csv, index=False)

    print(f"Spots aligned: {gt_mat.shape[0]}, Genes evaluated: {len(selected_genes)}")
    print(f"Mean PCC: {np.nanmean(metrics_df['pcc']):.4f}")
    print(f"Mean SSIM: {np.nanmean(metrics_df['ssim']):.4f}")
    print(f"Mean RMSE: {metrics_df['rmse'].mean():.4f}")
    print(f"Mean JS: {metrics_df['js'].mean():.4f}")
    print(f"Per-gene metrics saved to {args.output_csv}")
    save_boxplots(metrics_df, args.output_csv, title="Expression Metrics")


if __name__ == "__main__":
    main()
