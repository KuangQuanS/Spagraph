import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import re
from typing import List


def clean_column_names(df):
    """Clean column names by replacing spaces and symbols (especially .) with -."""
    df.columns = [re.sub(r'[^\w]+', '-', col) for col in df.columns]
    return df


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


def align_csv_pair(pred_df: pd.DataFrame, gt_df: pd.DataFrame, target_gene: str):
    """Align two CSV matrices on spots and target gene; return vectors for PCC."""
    shared_spots = pred_df.index.intersection(gt_df.index)
    if len(shared_spots) == 0:
        raise ValueError("No overlapping spots between predicted and ground truth CSVs.")

    pred_df = pred_df.loc[shared_spots]
    gt_df = gt_df.loc[shared_spots]

    shared_genes = pred_df.columns.intersection(gt_df.columns)
    if target_gene not in shared_genes:
        raise ValueError(f"Target gene '{target_gene}' not found in overlapping genes.")

    pred_vec = pred_df[target_gene].astype(float)
    gt_vec = gt_df[target_gene].astype(float)

    pred_vec = pred_vec.replace([np.inf, -np.inf], np.nan).fillna(0)
    gt_vec = gt_vec.replace([np.inf, -np.inf], np.nan).fillna(0)

    return gt_vec.values, pred_vec.values


def load_and_align_for_heatmap(pred_csv: str, gt_h5ad: str, target_gene: str):
    """Load and align data for heatmap plotting."""
    pred_df = pd.read_csv(pred_csv, index_col=0)
    pred_df = clean_column_names(pred_df)
    pred_df.index = pred_df.index.astype(str)
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


def plot_gene_heatmaps(coords, gt_vec, pred_vec, target_gene: str, output_png: str, log1p: bool = False, clip_percentile: float = 99.0):
    """Plot heatmaps for ground truth and prediction of a target gene."""
    if log1p:
        gt_vec = np.log1p(gt_vec)
        pred_vec = np.log1p(pred_vec)

    # Basic stats
    def summarize(name, arr):
        return f"{name}: mean={arr.mean():.4f}, median={np.median(arr):.4f}, min={arr.min():.4f}, max={arr.max():.4f}"
    print(summarize("GT", gt_vec))
    print(summarize("Pred", pred_vec))

    # 为 Ground truth 与 Prediction 分别绘制两张干净的图片（无坐标轴）
    base = output_png.rsplit('.', 1)[0]

    # 绘制单张的帮助函数
    def _plot_single(vals, title, outpath):
        vmax = np.percentile(vals, clip_percentile)
        vmin = np.percentile(vals, 100 - clip_percentile) if log1p else 0.0
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, cmap="viridis", s=16, vmin=vmin, vmax=vmax, edgecolors='none')
        ax.set_title(f"{title} - {target_gene}", fontsize=12, fontweight='bold')
        # 移除刻度与脊线，保留色条
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="log1p(expr)" if log1p else "expr")
        cbar.ax.tick_params(labelsize=8)
        plt.tight_layout()
        fig.savefig(outpath, dpi=300, bbox_inches='tight', pad_inches=0.0)
        plt.close(fig)
        print(f"Saved {title} heatmap to {outpath}")

    gt_path = f"{base}_{target_gene}_GT.png"
    pred_path = f"{base}_{target_gene}_Pred.png"
    _plot_single(gt_vec, "Ground truth", gt_path)
    _plot_single(pred_vec, "Prediction", pred_path)


def compute_pcc(true_vec: np.ndarray, pred_vec: np.ndarray) -> float:
    """Compute Pearson correlation for two 1D arrays; returns nan if zero variance."""
    eps = 1e-8
    finite_mask = np.isfinite(true_vec) & np.isfinite(pred_vec)
    if finite_mask.sum() == 0:
        return np.nan
    true_vec = true_vec[finite_mask]
    pred_vec = pred_vec[finite_mask]

    t_mean = true_vec.mean()
    p_mean = pred_vec.mean()
    t_std = true_vec.std()
    p_std = pred_vec.std()
    if t_std < eps or p_std < eps:
        return np.nan
    return np.mean((true_vec - t_mean) * (pred_vec - p_mean)) / (t_std * p_std)


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
        
        # Check if both vectors are effectively zero
        if t_sum < eps and p_sum < eps:
            js = np.nan
        else:
            t_prob = true_vec / (t_sum + eps)
            p_prob = pred_vec / (p_sum + eps)
            m_prob = 0.5 * (t_prob + p_prob)

            def kl(a, b):
                # Suppress log warnings for expected zero cases
                with np.errstate(divide='ignore', invalid='ignore'):
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


def compute_ars(metrics_list: List[pd.DataFrame]) -> pd.DataFrame:
    """Compute Average Ranking Score (ARS) for multiple methods.
    
    ARS aggregates PCC, SSIM, RMSE, and JSD to evaluate relative accuracy.
    Ranking rules:
    - PCC, SSIM: ascending order (smallest value gets rank 1)
    - RMSE, JS: descending order (largest value gets rank 1)
    - ARS: higher is better
    
    Args:
        metrics_list: List of DataFrames, each containing metrics for one method
        
    Returns:
        DataFrame with method rankings and ARS scores
    """
    if len(metrics_list) == 0:
        return pd.DataFrame()
    
    n_methods = len(metrics_list)
    
    # Aggregate mean metrics per method
    method_stats = []
    for i, df in enumerate(metrics_list):
        method_stats.append({
            'method_id': i,
            'mean_pcc': np.nanmean(df['pcc']),
            'mean_ssim': np.nanmean(df['ssim']),
            'mean_rmse': df['rmse'].mean(),
            'mean_js': df['js'].mean()
        })
    
    stats_df = pd.DataFrame(method_stats)
    
    # Rank methods
    # PCC/SSIM: ascending=True → smallest value gets rank 1
    # RMSE/JS: ascending=False → largest value gets rank 1
    stats_df['rank_pcc'] = stats_df['mean_pcc'].rank(ascending=True, method='average')
    stats_df['rank_ssim'] = stats_df['mean_ssim'].rank(ascending=True, method='average')
    stats_df['rank_rmse'] = stats_df['mean_rmse'].rank(ascending=False, method='average')
    stats_df['rank_js'] = stats_df['mean_js'].rank(ascending=False, method='average')
    
    # Compute ARS as sum of ranks divided by (4 * number of methods)
    # This normalizes ARS to be between 0 and 1, with 1 being the best
    stats_df['ARS'] = (stats_df['rank_pcc'] + stats_df['rank_ssim'] + 
                       stats_df['rank_rmse'] + stats_df['rank_js']) / (4.0 * n_methods)
    
    return stats_df[['method_id', 'mean_pcc', 'mean_ssim', 'mean_rmse', 'mean_js', 
                     'rank_pcc', 'rank_ssim', 'rank_rmse', 'rank_js', 'ARS']]


def save_ars_barplot(ars_df: pd.DataFrame, labels: List[str], output_csv: str):
    """Save horizontal bar plot for ARS scores with consistent colors as boxplot."""
    if len(ars_df) == 0:
        return
    
    # Use same color palette as multi_boxplots
    palette = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442']
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.8)))
    
    # Create a mapping of method_id to original order for consistent coloring
    plot_data = ars_df.copy()
    plot_data['method_name'] = [labels[int(i)] for i in plot_data['method_id']]
    
    # Assign colors based on original method_id (not sorted order)
    method_colors = [palette[int(row['method_id']) % len(palette)] for _, row in plot_data.iterrows()]
    
    # Sort by ARS for display (higher is better)
    plot_data = plot_data.sort_values('ARS', ascending=True)  # ascending for horizontal bar
    
    # Reorder colors to match sorted data
    color_mapping = dict(zip(ars_df['method_id'], [palette[int(i) % len(palette)] for i in ars_df['method_id']]))
    sorted_colors = [color_mapping[int(row['method_id'])] for _, row in plot_data.iterrows()]
    
    bars = ax.barh(plot_data['method_name'], plot_data['ARS'], color=sorted_colors)
    
    ax.set_xlabel('ARS', fontsize=12, fontweight='bold')
    ax.set_title('Average Ranking Score Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    png_path = output_csv.rsplit('.', 1)[0] + "_ARS_barplot.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"ARS bar plot saved to {png_path}")


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
    # png_path = output_csv.rsplit('.', 1)[0] + "_boxplot.png"
    # plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    # print(f"Boxplot saved to {png_path}")


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
    # png_path = output_csv.rsplit('.', 1)[0] + "_compare_boxplot.png"
    # plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    # print(f"Comparison boxplot saved to {png_path}")


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
        # showfliers=True to display outliers, whis=[0,100] extends whiskers to min/max
        bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True, 
                        whis=[0, 100], showfliers=True,
                        showmeans=True, meanprops=dict(marker='D', markerfacecolor='#FF8C00', markeredgecolor='#FF8C00', markersize=4),
                        flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5))
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
    # png_path = output_csv.rsplit('.', 1)[0] + "_spot_diff_boxplot.png"
    # plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    # print(f"Per-spot abs diff boxplot saved to {png_path}")


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
    parser.add_argument("--target_gene", type=str, default=None,
                        help="If set (expression mode), evaluate only this gene (overrides HVG/gene_list).")
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
    parser.add_argument("--composition_pred_csv5", type=str, default=None,
                        help="Fifth predicted cell composition CSV for comparison (optional).")
    parser.add_argument("--composition_pred_csv6", type=str, default=None,
                        help="Sixth predicted cell composition CSV for comparison (optional).")
    parser.add_argument("--composition_pred_csv7", type=str, default=None,
                        help="Seventh predicted cell composition CSV for comparison (optional).")
    parser.add_argument("--composition_pred_csv8", type=str, default=None,
                        help="Eighth predicted cell composition CSV for comparison (optional).")
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
    parser.add_argument("--method5_name", type=str, default="Method5",
                        help="Label for fifth composition prediction.")
    parser.add_argument("--method6_name", type=str, default="Method6",
                        help="Label for sixth composition prediction.")
    parser.add_argument("--method7_name", type=str, default="Method7",
                        help="Label for seventh composition prediction.")
    parser.add_argument("--method8_name", type=str, default="Method8",
                        help="Label for eighth composition prediction.")
    parser.add_argument("--use_intersection", action="store_true",
                        help="Use intersection of column names across all CSVs instead of requiring exact match.")
    parser.add_argument("--use_union", action="store_true",
                        help="Use union of column names across all CSVs, filling missing columns with zeros.")
    parser.add_argument("--output_png", type=str, default=None,
                        help="Output PNG path for gene heatmap (when using --target_gene). Default: <target_gene>_heatmap.png")
    parser.add_argument("--heatmap_log1p", action="store_true",
                        help="Apply log1p transform before plotting gene heatmap.")
    parser.add_argument("--heatmap_clip_percentile", type=float, default=99.0,
                        help="Percentile for heatmap color scaling (default: 99.0).")
    args = parser.parse_args()

    # Check for conflicting column mode arguments
    if args.use_intersection and args.use_union:
        raise ValueError("Cannot specify both --use_intersection and --use_union. Choose one column matching mode.")

    # ============ Composition mode ============
    if args.composition_pred_csv and args.composition_true_csv:
        # Load predictions (up to 8) and true
        pred_paths = [args.composition_pred_csv, args.composition_pred_csv2, args.composition_pred_csv3, args.composition_pred_csv4, args.composition_pred_csv5, args.composition_pred_csv6, args.composition_pred_csv7, args.composition_pred_csv8]
        pred_names = [args.method1_name, args.method2_name, args.method3_name, args.method4_name, args.method5_name, args.method6_name, args.method7_name, args.method8_name]
        pred_dfs = []
        pred_labels = []
        for p, name in zip(pred_paths, pred_names):
            if p:
                df = pd.read_csv(p, index_col=0)
                df = clean_column_names(df)
                pred_dfs.append(df)
                pred_labels.append(name)
        true_comp = pd.read_csv(args.composition_true_csv, index_col=0)
        true_comp = clean_column_names(true_comp)

        shared_spots = true_comp.index
        for df in pred_dfs:
            shared_spots = shared_spots.intersection(df.index)
        if len(shared_spots) == 0:
            raise ValueError("No overlapping spots between composition CSVs.")
        true_comp = true_comp.loc[shared_spots]
        pred_dfs = [df.loc[shared_spots] for df in pred_dfs]

        # Check columns: either exact match, use intersection, or use union
        if args.use_intersection:
            # Use intersection of all columns
            expected_set = set(true_comp.columns)
            for df in pred_dfs:
                expected_set = expected_set.intersection(set(df.columns))
            if len(expected_set) == 0:
                print(f"[Error] No overlapping columns found between ground truth and predictions!")
                print(f"Ground truth columns ({len(true_comp.columns)}): {sorted(true_comp.columns)[:5]}...")
                for i, df in enumerate(pred_dfs):
                    print(f"Prediction {i+1} columns ({len(df.columns)}): {sorted(df.columns)[:5]}...")
                raise ValueError("No overlapping columns between CSVs.")
            expected_cols = sorted(expected_set)
            print(f"[Info] Using {len(expected_cols)} shared cell types (intersection mode).")
        elif args.use_union:
            # Use union of all columns, fill missing with zeros
            all_cols = set(true_comp.columns)
            for df in pred_dfs:
                all_cols = all_cols.union(set(df.columns))
            expected_cols = sorted(all_cols)
            
            # Fill missing columns with zeros in all DataFrames
            for i, df in enumerate([true_comp] + pred_dfs):
                missing_cols = all_cols - set(df.columns)
                for col in missing_cols:
                    df[col] = 0.0
                # Ensure consistent column order
                df = df[expected_cols]
                if i == 0:
                    true_comp = df
                else:
                    pred_dfs[i-1] = df
            
            print(f"[Info] Using {len(expected_cols)} total cell types (union mode, missing filled with zeros).")
        else:
            # Require exact match (original behavior)
            true_cols = set(true_comp.columns)
            for i, df in enumerate(pred_dfs):
                pred_cols = set(df.columns)
                if true_cols != pred_cols:
                    missing_in_pred = true_cols - pred_cols
                    missing_in_true = pred_cols - true_cols
                    print(f"[Error] Column mismatch between ground truth and prediction {i+1}!")
                    print(f"Ground truth has {len(true_comp.columns)} columns")
                    print(f"Prediction {i+1} has {len(df.columns)} columns")
                    if missing_in_pred:
                        print(f"Missing in prediction {i+1} (first 10): {sorted(missing_in_pred)[:10]}")
                    if missing_in_true:
                        print(f"Missing in ground truth (first 10): {sorted(missing_in_true)[:10]}")
                    raise ValueError(f"Column names must match exactly between ground truth and all predictions!")
            expected_cols = sorted(true_cols)
            print(f"[Info] All {len(expected_cols)} cell types matched successfully.")

        if args.celltype:
            cleaned_celltype = re.sub(r'[^\w]+', '-', args.celltype)
            if cleaned_celltype not in true_cols:
                raise ValueError(f"Cell type '{args.celltype}' (cleaned: '{cleaned_celltype}') not found in ground truth columns.")
            expected_cols = [cleaned_celltype]

        # Float conversion / cleanup before normalization
        true_comp = true_comp.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)
        pred_dfs = [df.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0) for df in pred_dfs]
        pred_dfs_for_check = [df.copy() for df in pred_dfs]

        # Check and normalize pred_dfs if row sums are not 1.0
        for i in range(len(pred_dfs)):
            df = pred_dfs[i]
            row_sums = df.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-6):
                print(f"[Info] Normalizing {pred_labels[i]} as row sums are not 1.0 (mean: {row_sums.mean():.4f})")
                row_sums[row_sums == 0] = 1.0
                pred_dfs[i] = df.div(row_sums, axis=0)
            else:
                print(f"[Info] {pred_labels[i]} is already normalized.")

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
        # metrics_df.to_csv(args.output_csv, index=False)
        metrics_list.append(metrics_df)
        
        # Only print detailed metrics if single method (no comparison)
        if len(pred_dfs) == 1:
            print(f"[Composition mode] Spots aligned: {pred_dfs[0].shape[0]}, Celltypes evaluated: {len(expected_cols)}")
            print(f"{pred_labels[0]} Mean PCC: {np.nanmean(metrics_df['pcc']):.4f}")
            print(f"{pred_labels[0]} Mean SSIM: {np.nanmean(metrics_df['ssim']):.4f}")
            print(f"{pred_labels[0]} Mean RMSE: {metrics_df['rmse'].mean():.4f}")
            print(f"{pred_labels[0]} Mean JS: {metrics_df['js'].mean():.4f}")
            print(f"Per-celltype metrics saved to {args.output_csv}")
        else:
            print(f"[Composition mode] Spots aligned: {pred_dfs[0].shape[0]}, Celltypes evaluated: {len(expected_cols)}")
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
            # alt_metrics.to_csv(alt_csv, index=False)
            metrics_list.append(alt_metrics)
            # Don't print individual method stats in comparison mode
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
            
            # Compute and display ARS for multiple methods
            print("\n" + "="*60)
            print("Average Ranking Score (ARS) - Based on PCC, SSIM, RMSE, JS")
            print("="*60)
            ars_df = compute_ars(metrics_list)
            for idx, row in ars_df.iterrows():
                method_name = labels[int(row['method_id'])]
                print(f"\n{method_name}:")
                print(f"  Mean PCC: {row['mean_pcc']:.4f} (Rank: {row['rank_pcc']:.1f})")
                print(f"  Mean SSIM: {row['mean_ssim']:.4f} (Rank: {row['rank_ssim']:.1f})")
                print(f"  Mean RMSE: {row['mean_rmse']:.4f} (Rank: {row['rank_rmse']:.1f})")
                print(f"  Mean JS: {row['mean_js']:.4f} (Rank: {row['rank_js']:.1f})")
                print(f"  >>> ARS: {row['ARS']:.4f} (Higher is better) <<<")
            
            # Save ARS to CSV
            ars_output = f"{out_prefix}_ARS.csv"
            ars_df['method_name'] = [labels[int(i)] for i in ars_df['method_id']]
            ars_df[['method_name', 'mean_pcc', 'mean_ssim', 'mean_rmse', 'mean_js', 
                    'rank_pcc', 'rank_ssim', 'rank_rmse', 'rank_js', 'ARS']].to_csv(ars_output, index=False)
            print(f"\nARS results saved to {ars_output}")
            
            # Save ARS bar plot
            save_ars_barplot(ars_df, labels, args.output_csv)
            print("="*60 + "\n")

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
            # spot_df.to_csv(out_spot, index=False)
            # print(f"Per-spot values and diffs saved to {out_spot}")
            save_spot_diff_boxplots(true_vec, pred_dfs, pred_labels[:len(pred_dfs)],
                                    output_csv=args.output_csv, celltype=args.celltype)
        return

    # ============ Expression mode ============
    if not args.reconstructed_csv or not args.ground_truth_h5ad:
        raise ValueError("Provide --reconstructed_csv and --ground_truth_h5ad for expression evaluation, or both composition CSVs for composition mode.")

    recon_df = pd.read_csv(args.reconstructed_csv, index_col=0)
    # Ensure index is string type
    recon_df.index = recon_df.index.astype(str)
    recon_df = clean_column_names(recon_df)

    # Special case: single gene PCC with two CSVs (or h5ad as GT)
    if args.target_gene:
        gt_path = args.ground_truth_h5ad
        if gt_path.lower().endswith(".csv"):
            gt_df = pd.read_csv(gt_path, index_col=0)
            gt_df = clean_column_names(gt_df)
        else:
            gt_adata = sc.read_h5ad(gt_path)
            gt_df = pd.DataFrame(gt_adata.X.toarray() if hasattr(gt_adata.X, "toarray") else gt_adata.X,
                                 index=gt_adata.obs_names,
                                 columns=gt_adata.var_names)
            # Ensure index types match (convert to string if needed)
            gt_df.index = gt_df.index.astype(str)
        true_vec, pred_vec = align_csv_pair(recon_df, gt_df, args.target_gene)
        pcc = compute_pcc(true_vec, pred_vec)
        print(f"Target gene '{args.target_gene}' PCC: {pcc:.6f}" if np.isfinite(pcc) else
              f"Target gene '{args.target_gene}' PCC: nan (zero variance or invalid values)")

        # --- 新增: 计算并打印 GT 与预测（重建）的总测序深度信息 ---
        # 对齐后按共有 spots 及共有 genes 计算总深度（sum over spots x genes）以及每 spot 的 mean/median
        shared_spots = recon_df.index.intersection(gt_df.index)
        if len(shared_spots) > 0:
            recon_sub = recon_df.loc[shared_spots].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)
            gt_sub = gt_df.loc[shared_spots].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)

            # 若有基因列不一致，取交集以便公平比较
            shared_genes = recon_sub.columns.intersection(gt_sub.columns)
            if len(shared_genes) == 0:
                print("Warning: No overlapping genes found between reconstructed and ground truth when computing depths.")
            else:
                recon_sub = recon_sub[shared_genes]
                gt_sub = gt_sub[shared_genes]

            gt_total = float(gt_sub.values.sum())
            pred_total = float(recon_sub.values.sum())
            gt_per_spot = gt_sub.sum(axis=1)
            pred_per_spot = recon_sub.sum(axis=1)

            print(f"GT total sequencing depth (sum over {len(shared_spots)} spots and {len(gt_sub.columns)} genes): {gt_total:.3f}")
            print(f"Pred total sequencing depth (sum over aligned spots/genes): {pred_total:.3f}")
            print(f"GT per-spot mean/median: {gt_per_spot.mean():.3f}/{gt_per_spot.median():.3f}")
            print(f"Pred per-spot mean/median: {pred_per_spot.mean():.3f}/{pred_per_spot.median():.3f}")
        else:
            print("Warning: no overlapping spots to compute total sequencing depth.")

        # Generate heatmap if ground truth is h5ad (has spatial coordinates)
        if not gt_path.lower().endswith(".csv"):
            output_png = args.output_png or f"{args.target_gene}_heatmap.png"
            try:
                coords, gt_vec_heatmap, pred_vec_heatmap, _ = load_and_align_for_heatmap(
                    args.reconstructed_csv, gt_path, args.target_gene)
                plot_gene_heatmaps(coords, gt_vec_heatmap, pred_vec_heatmap, args.target_gene,
                                 output_png, args.heatmap_log1p, args.heatmap_clip_percentile)
            except Exception as e:
                print(f"Warning: Could not generate heatmap: {e}")

        return

    gt_adata = sc.read_h5ad(args.ground_truth_h5ad)

    gt_df, pred_df, genes = align_data(recon_df, gt_adata)

    # Select genes: target_gene > gene_list > HVG
    if args.target_gene:
        if args.target_gene not in genes:
            raise ValueError(f"Target gene '{args.target_gene}' not found in shared genes.")
        selected_genes = [args.target_gene]
        print(f"Using target gene: {args.target_gene}")
    elif args.gene_list:
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
