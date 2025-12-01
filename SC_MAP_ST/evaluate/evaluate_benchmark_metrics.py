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
    if all(len(d) <= 1 for d in data):
        # Degenerate case: single value per metric -> use bars
        heights = [d.iloc[0] if len(d) else np.nan for d in data]
        ax.bar(metric_cols, heights, color="#1f77b4")
        ax.set_ylabel("Score")
        for i, h in enumerate(heights):
            if np.isfinite(h):
                ax.text(i, h, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    else:
        ax.boxplot(data, tick_labels=metric_cols, showfliers=False)
        ax.set_ylabel("Score")
    ax.set_title(title)
    plt.tight_layout()
    png_path = output_csv.rsplit('.', 1)[0] + "_boxplot.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Boxplot saved to {png_path}")


def save_compare_boxplots(metrics_a: pd.DataFrame, metrics_b: pd.DataFrame,
                          labels: List[str], output_csv: str, title: str = "Composition Metrics Comparison"):
    """Save side-by-side boxplots comparing two methods for each metric."""
    metric_cols = [col for col in ['pcc', 'ssim', 'rmse', 'js'] if col in metrics_a.columns and col in metrics_b.columns]
    if not metric_cols:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e']
    data_a = [metrics_a[col].dropna() for col in metric_cols]
    data_b = [metrics_b[col].dropna() for col in metric_cols]
    if all(len(d) <= 1 for d in data_a + data_b):
        # Use grouped bars when only single values
        width = 0.35
        x = np.arange(len(metric_cols))
        heights_a = [d.iloc[0] if len(d) else np.nan for d in data_a]
        heights_b = [d.iloc[0] if len(d) else np.nan for d in data_b]
        ax.bar(x - width/2, heights_a, width, label=labels[0], color=colors[0])
        ax.bar(x + width/2, heights_b, width, label=labels[1], color=colors[1])
        ax.set_xticks(x)
        ax.set_xticklabels(metric_cols)
        ax.set_ylabel("Score")
        for xi, h in zip(x - width/2, heights_a):
            if np.isfinite(h):
                ax.text(xi, h, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
        for xi, h in zip(x + width/2, heights_b):
            if np.isfinite(h):
                ax.text(xi, h, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    else:
        positions = []
        box_data = []
        for i, (da, db) in enumerate(zip(data_a, data_b)):
            positions.extend([i * 3 + 1, i * 3 + 2])
            box_data.extend([da, db])
        bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True, showfliers=False)
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[i % 2])
        centers = [(i * 3 + 1.5) for i in range(len(metric_cols))]
        ax.set_xticks(centers)
        ax.set_xticklabels(metric_cols)
        ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=colors[0]),
                       plt.Rectangle((0, 0), 1, 1, color=colors[1])],
              labels=labels, loc='best')
    plt.tight_layout()
    png_path = output_csv.rsplit('.', 1)[0] + "_compare_boxplot.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Comparison boxplot saved to {png_path}")


def save_multi_boxplots(metrics_list: List[pd.DataFrame], labels: List[str],
                        output_csv: str, title: str = "Composition Metrics Comparison"):
    """Save grouped boxplots for multiple methods."""
    if len(metrics_list) == 0:
        return
    metric_cols = [col for col in ['pcc', 'ssim', 'rmse', 'js'] if all(col in m.columns for m in metrics_list)]
    if not metric_cols:
        return
    # Colorblind-friendly palette
    palette = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442']
    fig, ax = plt.subplots(figsize=(10, 6))
    data_per_metric = [[m[col].dropna() for m in metrics_list] for col in metric_cols]
    if all(all(len(d) <= 1 for d in per_metric) for per_metric in data_per_metric):
        # Use grouped bars when only single values
        width = 0.8 / max(len(metrics_list), 1)
        x = np.arange(len(metric_cols))
        for j, label in enumerate(labels):
            heights = [per_metric[j].iloc[0] if len(per_metric[j]) else np.nan for per_metric in data_per_metric]
            ax.bar(x + (j - (len(metrics_list)-1)/2)*width, heights, width,
                   label=label, color=palette[j % len(palette)])
            for xi, h in zip(x + (j - (len(metrics_list)-1)/2)*width, heights):
                if np.isfinite(h):
                    ax.text(xi, h, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_cols)
        ax.set_ylabel("Score")
    else:
        positions = []
        box_data = []
        for i, per_metric in enumerate(data_per_metric):
            for j, d in enumerate(per_metric):
                positions.append(i * (len(metrics_list) + 1) + j + 1)
                box_data.append(d)
        bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True, showfliers=False)
        for i, patch in enumerate(bp['boxes']):
            method_idx = i % len(metrics_list)
            patch.set_facecolor(palette[method_idx % len(palette)])
        group_width = len(metrics_list) + 1
        centers = [i * group_width + (len(metrics_list)+1)/2 for i in range(len(metric_cols))]
        ax.set_xticks(centers)
        ax.set_xticklabels(metric_cols)
        ax.set_ylabel("Score")
    ax.set_title(title)
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=palette[i % len(palette)]) for i in range(len(labels))]
    ax.legend(handles=legend_handles, labels=labels, loc='best')
    plt.tight_layout()
    png_path = output_csv.rsplit('.', 1)[0] + "_multi_boxplot.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Multi-method boxplot saved to {png_path}")


def bootstrap_single_celltype(gt_mat: np.ndarray, pred_mat: np.ndarray, genes, n_boot: int = 200,
                              random_state: int = 0) -> pd.DataFrame:
    """Bootstrap metrics for a single cell type to enable boxplot visualization."""
    rng = np.random.default_rng(random_state)
    records = []
    n_spots = gt_mat.shape[0]
    for _ in range(n_boot):
        idx = rng.choice(n_spots, size=n_spots, replace=True)
        boot_gt = gt_mat[idx]
        boot_pred = pred_mat[idx]
        m = compute_metrics(boot_gt, boot_pred, genes)
        records.append(m.iloc[0].to_dict())
    return pd.DataFrame(records)


def save_spot_diff_boxplots(true_vec: pd.Series, pred_dfs: List[pd.DataFrame],
                            labels: List[str], output_csv: str, celltype: str):
    """Boxplot of per-spot absolute differences for a single cell type across methods."""
    abs_diffs = []
    for df in pred_dfs:
        pred_vec = df.iloc[:, 0]
        abs_diffs.append(np.abs(pred_vec.values - true_vec.values))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(abs_diffs, tick_labels=labels, showfliers=False)
    ax.set_ylabel(f"|pred - true| ({celltype})")
    ax.set_title(f"{celltype} Per-spot Absolute Difference")
    plt.tight_layout()
    png_path = output_csv.rsplit('.', 1)[0] + "_spot_diff_boxplot.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Per-spot abs diff boxplot saved to {png_path}")


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
    parser.add_argument("--composition_pred_csv2", type=str, default=None,
                        help="Second predicted cell composition CSV for comparison (optional).")
    parser.add_argument("--composition_pred_csv3", type=str, default=None,
                        help="Third predicted cell composition CSV for comparison (optional).")
    parser.add_argument("--composition_pred_csv4", type=str, default=None,
                        help="Fourth predicted cell composition CSV for comparison (optional).")
    parser.add_argument("--celltype", type=str, default=None,
                        help="If set (composition mode), evaluate only this cell type.")
    parser.add_argument("--spot_diff_csv", type=str, default=None,
                        help="If set with --celltype, save per-spot true/pred/diff values to this CSV "
                             "(default: <output_csv> with _spot_diffs suffix).")
    parser.add_argument("--celltype_bootstrap", type=int, default=200,
                        help="Number of bootstrap samples for single-celltype plots (default: 200).")
    parser.add_argument("--method1_name", type=str, default="Method1",
                        help="Label for first composition prediction.")
    parser.add_argument("--method2_name", type=str, default="Method2",
                        help="Label for second composition prediction.")
    parser.add_argument("--method3_name", type=str, default="Method3",
                        help="Label for third composition prediction.")
    parser.add_argument("--method4_name", type=str, default="Method4",
                        help="Label for fourth composition prediction.")
    args = parser.parse_args()

    # ============ Composition mode ============
    if args.composition_pred_csv and args.composition_true_csv:
        # Load predictions (up to 4) and true
        pred_paths = [args.composition_pred_csv, args.composition_pred_csv2, args.composition_pred_csv3, args.composition_pred_csv4]
        pred_names = [args.method1_name, args.method2_name, args.method3_name, args.method4_name]
        pred_dfs = []
        pred_labels = []
        for p, name in zip(pred_paths, pred_names):
            if p:
                pred_dfs.append(pd.read_csv(p, index_col=0))
                pred_labels.append(name)
        true_comp = pd.read_csv(args.composition_true_csv, index_col=0)

        shared_spots = true_comp.index
        for df in pred_dfs:
            shared_spots = shared_spots.intersection(df.index)
        if len(shared_spots) == 0:
            raise ValueError("No overlapping spots between composition CSVs.")
        true_comp = true_comp.loc[shared_spots]
        pred_dfs = [df.loc[shared_spots] for df in pred_dfs]

        expected_cols = list(true_comp.columns)
        expected_set = set(expected_cols)
        for df, name in zip(pred_dfs, pred_labels):
            pred_set = set(df.columns)
            missing = expected_set - pred_set
            extra = pred_set - expected_set
            if missing or extra:
                missing_msg = f"missing columns {sorted(missing)}" if missing else ""
                extra_msg = f"extra columns {sorted(extra)}" if extra else ""
                connector = "; " if missing and extra else ""
                raise ValueError(f"Column mismatch for {name}: {missing_msg}{connector}{extra_msg}. "
                                 "Prediction and ground truth must share identical column names.")

        if args.celltype:
            if args.celltype not in expected_set:
                raise ValueError(f"Cell type '{args.celltype}' not found in ground truth columns.")
            expected_cols = [args.celltype]

        # Float conversion / cleanup before normalization
        true_comp = true_comp.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)
        pred_dfs = [df.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0) for df in pred_dfs]
        pred_dfs_for_check = [df.copy() for df in pred_dfs]

        # Row-normalize true composition using all celltypes (handle count-style inputs)
        true_row_sum = true_comp.sum(axis=1)
        true_row_sum[true_row_sum == 0] = 1.0
        true_comp = true_comp.div(true_row_sum, axis=0)

        # If only one cell type is requested, subset after normalization
        true_comp = true_comp[expected_cols]
        pred_dfs = [df[expected_cols] for df in pred_dfs]

        # Pred is assumed already normalized; warnings removed to avoid noise in single-celltype mode

        metrics_list = []
        out_prefix = args.output_csv.rsplit('.', 1)[0]
        # Method1 metrics to main csv
        metrics_df = compute_metrics(true_comp.values, pred_dfs[0].values, list(expected_cols))
        metrics_df.to_csv(args.output_csv, index=False)
        metrics_list.append(metrics_df)
        print(f"[Composition mode] Spots aligned: {pred_dfs[0].shape[0]}, Celltypes evaluated: {len(expected_cols)}")
        print(f"{pred_labels[0]} Mean PCC: {np.nanmean(metrics_df['pcc']):.4f}")
        print(f"{pred_labels[0]} Mean SSIM: {np.nanmean(metrics_df['ssim']):.4f}")
        print(f"{pred_labels[0]} Mean RMSE: {metrics_df['rmse'].mean():.4f}")
        print(f"{pred_labels[0]} Mean JS: {metrics_df['js'].mean():.4f}")
        print(f"Per-celltype metrics saved to {args.output_csv}")
        # For single celltype, generate bootstrap metrics to enable boxplots with spread
        if args.celltype:
            bootstrap_metrics = [bootstrap_single_celltype(true_comp.values, pred_dfs[0].values,
                                                           list(expected_cols), n_boot=args.celltype_bootstrap)]
            save_boxplots(bootstrap_metrics[0], args.output_csv, title=f"{pred_labels[0]} Composition Metrics")
        else:
            save_boxplots(metrics_df, args.output_csv, title=f"{pred_labels[0]} Composition Metrics")

        # Additional methods
        for idx in range(1, len(pred_dfs)):
            alt_metrics = compute_metrics(true_comp.values, pred_dfs[idx].values, list(expected_cols))
            alt_csv = f"{out_prefix}_method{idx+1}.csv"
            alt_metrics.to_csv(alt_csv, index=False)
            metrics_list.append(alt_metrics)
            print(f"{pred_labels[idx]} metrics saved to {alt_csv}")
            if args.celltype:
                boot = bootstrap_single_celltype(true_comp.values, pred_dfs[idx].values,
                                                 list(expected_cols), n_boot=args.celltype_bootstrap)
                save_boxplots(boot, alt_csv, title=f"{pred_labels[idx]} Composition Metrics")
                bootstrap_metrics.append(boot)
            else:
                save_boxplots(alt_metrics, alt_csv, title=f"{pred_labels[idx]} Composition Metrics")

        if len(metrics_list) > 1:
            labels = pred_labels[:len(metrics_list)]
            if args.celltype:
                save_multi_boxplots(bootstrap_metrics, labels, output_csv=args.output_csv,
                                    title="Composition Metrics Comparison")
            else:
                save_multi_boxplots(metrics_list, labels, output_csv=args.output_csv,
                                    title="Composition Metrics Comparison")

        # Save per-spot diffs for single cell type
        if args.celltype:
            out_spot = args.spot_diff_csv or f"{out_prefix}_spot_diffs.csv"
            true_vec = true_comp.iloc[:, 0]
            spot_df = pd.DataFrame({"spot": true_vec.index, f"true_{args.celltype}": true_vec.values})
            for name, df in zip(pred_labels, pred_dfs):
                pred_vec = df.iloc[:, 0]
                spot_df[f"pred_{name}"] = pred_vec.values
                spot_df[f"diff_{name}"] = pred_vec.values - true_vec.values
                spot_df[f"abs_diff_{name}"] = np.abs(pred_vec.values - true_vec.values)
            spot_df.to_csv(out_spot, index=False)
            print(f"Per-spot values and diffs saved to {out_spot}")
            save_spot_diff_boxplots(true_vec, pred_dfs, pred_labels[:len(pred_dfs)],
                                    output_csv=args.output_csv, celltype=args.celltype)
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
    #metrics_df.to_csv(args.output_csv, index=False)

    print(f"Spots aligned: {gt_mat.shape[0]}, Genes evaluated: {len(selected_genes)}")
    print(f"Mean PCC: {np.nanmean(metrics_df['pcc']):.4f}")
    print(f"Mean SSIM: {np.nanmean(metrics_df['ssim']):.4f}")
    print(f"Mean RMSE: {metrics_df['rmse'].mean():.4f}")
    print(f"Mean JS: {metrics_df['js'].mean():.4f}")
    print(f"Per-gene metrics saved to {args.output_csv}")
    save_boxplots(metrics_df, args.output_csv, title="Expression Metrics")


if __name__ == "__main__":
    main()
