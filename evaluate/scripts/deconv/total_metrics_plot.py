import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATE_DIR = SCRIPT_DIR.parent
DATA_ROOT = EVALUATE_DIR / "data"
COMBINED_PLOTS_DIR = DATA_ROOT / "combined_plots"

# 设置matplotlib样式，让图表更美观
plt.style.use('default')  # 使用默认样式，也可以试试 'seaborn' 如果安装了seaborn
plt.rcParams['font.family'] = 'DejaVu Sans'  # 设置字体
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['savefig.dpi'] = 400

# 读取 Data1 到 Data32 的 metrics_ARS.csv
COMBINED_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
data_dict = {}
for i in range(1, 33):
    try:
        df = pd.read_csv(DATA_ROOT / f'Data{i}' / 'metrics_ARS.csv')
        data_dict[i] = df
    except FileNotFoundError:
        print(f"Warning: Data{i}/metrics_ARS.csv not found, skipping...")

# 准备数据：为每个方法收集跨数据集的指标
methods = [ 'Seurat','DestVI','SPOTlight', 'SpatialDWLS', 'Stereoscope', 'RCTD', 'Tangram','Spagraph']
metrics = ['mean_pcc', 'mean_ssim', 'mean_rmse', 'mean_js', 'ARS']

# 好看的颜色方案（色盲友好）
colors = {
    'Spagraph': '#0072B2',    # 蓝色
    'Tangram': '#D55E00',     # 橙色
    'RCTD': '#009E73',        # 绿色
    'Seurat': '#CC79A7',      # 粉色
    'SpatialDWLS': '#F0E442',  # 黄色
    'SPOTlight': '#56B4E9',    # 浅蓝色
    'DestVI': '#E69F00',       # 橙黄色
    'Stereoscope': '#999999'   # 灰色
}

# 为每个指标创建单独的图表
for idx, metric in enumerate(metrics):
    
    # 为每个方法收集该指标在所有数据集上的值
    box_data = []
    box_labels = []
    box_colors = []
    
    for method in methods:
        metric_values = []
        
        for data_id in sorted(data_dict.keys()):
            df = data_dict[data_id]
            method_row = df[df['method_name'] == method]
            if not method_row.empty and metric in method_row.columns:
                metric_values.append(method_row[metric].values[0])
        
        if metric_values:
            box_data.append(metric_values)
            box_labels.append(method)
            box_colors.append(colors[method])
    
    if metric == 'ARS':
        fig, ax = plt.subplots(figsize=(6, 10))  # y轴长度是x轴的两倍
        # 为ARS绘制横向柱状图（带最小值和最大值）
        means = [np.mean(vals) for vals in box_data]
        mins = [np.min(vals) for vals in box_data]
        maxs = [np.max(vals) for vals in box_data]
        
        y_pos = np.arange(len(box_labels))
        bars = ax.barh(y_pos, means, color=box_colors, alpha=0.8, height=0.7, edgecolor='black', linewidth=0.5)
        
        # 添加误差条显示最小值和最大值
        ax.errorbar(means, y_pos, xerr=[np.array(means) - np.array(mins), np.array(maxs) - np.array(means)], 
                    fmt='none', ecolor='black', capsize=6, elinewidth=2, capthick=1)
        
        # 添加数值标签
        for bar, mean_val in zip(bars, means):
            ax.text(mean_val + 0.01, bar.get_y() + bar.get_height()/2 + 0.05, 
                    f'{mean_val:.3f}', ha='left', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(box_labels, fontsize=12)
        ax.set_xlabel(metric.replace('_', ' ').upper(), fontsize=14)
        ax.set_title(f'{metric.replace("_", " ").upper()} Distribution (n={len(data_dict)})', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        output_path = COMBINED_PLOTS_DIR / f'metrics_{metric}_barh.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"{metric} horizontal bar plot saved to {output_path}")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))  # y轴长度是x轴的两倍
        # 为其他指标绘制横向箱线图
        if box_data:
            bp = ax.boxplot(box_data, vert=False, labels=box_labels, patch_artist=True, 
                            whis=[5, 95], showfliers=False,  # 调整胡须和异常值
                            showmeans=True, 
                            meanprops=dict(marker='D', markerfacecolor='#FF8C00', markeredgecolor='black', markersize=6, markeredgewidth=1),
                            medianprops=dict(color='black', linewidth=2),
                            whiskerprops=dict(color='black', linewidth=1.5),
                            capprops=dict(color='black', linewidth=1.5),
                            boxprops=dict(linewidth=1.5))
            
            # 给每个箱子上色
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
                patch.set_edgecolor('black')
                patch.set_linewidth(1)
        
        ax.set_xlabel(metric.replace('_', ' ').upper(), fontsize=14)
        ax.set_title(f'{metric.replace("_", " ").upper()} Distribution (n={len(data_dict)})', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')
        ax.tick_params(axis='y', labelsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        output_path = COMBINED_PLOTS_DIR / f'metrics_{metric}_boxplot.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"{metric} horizontal boxplot saved to {output_path}")
    
    plt.tight_layout()
    plt.close()

print(f"Successfully processed {len(data_dict)} datasets")

###################################然后是单个数据的整合#####################################
"""
Generate combined plots for all 32 datasets - 8 columns x 4 rows layout
Computes metrics from raw CSV files and saves outputs as PDF files
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import sys
import warnings
import re
warnings.filterwarnings('ignore')

# Import functions from evaluate_benchmark_metrics
from evaluate_benchmark_metrics import (
    clean_column_names, 
    compute_metrics, 
    compute_ars
)


def compute_dataset_metrics(base_dir, dataset_num, method_names):
    """
    Compute metrics for one dataset by loading ground truth and all method predictions.
    Returns: dict with method_name -> metrics_df mapping, plus ARS results
    """
    data_path = Path(base_dir) / f"Data{dataset_num}"
    
    # Load ground truth
    gt_csv = data_path / f"dataset{dataset_num}_density.csv"
    if not gt_csv.exists():
        print(f"  ✗ Ground truth not found: {gt_csv}")
        return None
    
    true_comp = pd.read_csv(gt_csv, index_col=0)
    true_comp = clean_column_names(true_comp)
    
    # Define prediction file mapping
    pred_files = {
        'Spagraph': 'Spatial_cell_composition.csv',
        'Tangram': 'tangram.csv',
        'RCTD': 'RCTD_results.csv',
        'Seurat': 'Seurat.csv',
        'SpatialDWLS': 'SpatialDWLS_result.csv',
        'SPOTlight': 'SPOTlight.csv',
        'DestVI': 'DestVI.csv',
        'Stereoscope': 'Stereoscope.csv'
    }
    
    # Load all predictions
    pred_dfs = {}
    for method_name in method_names:
        pred_file = data_path / pred_files[method_name]
        if not pred_file.exists():
            print(f"  ✗ {method_name} prediction not found: {pred_file}")
            continue
        df = pd.read_csv(pred_file, index_col=0)
        df = clean_column_names(df)
        pred_dfs[method_name] = df
    
    if len(pred_dfs) == 0:
        return None
    
    # Align spots (intersection)
    shared_spots = true_comp.index
    for df in pred_dfs.values():
        shared_spots = shared_spots.intersection(df.index)
    
    if len(shared_spots) == 0:
        print(f"  ✗ No overlapping spots for Data{dataset_num}")
        return None
    
    true_comp = true_comp.loc[shared_spots]
    for method in pred_dfs:
        pred_dfs[method] = pred_dfs[method].loc[shared_spots]
    
    # Use union of all columns (cell types), fill missing with zeros
    all_cols = set(true_comp.columns)
    for df in pred_dfs.values():
        all_cols = all_cols.union(set(df.columns))
    expected_cols = sorted(all_cols)
    
    # Fill missing columns with zeros
    for col in expected_cols:
        if col not in true_comp.columns:
            true_comp[col] = 0.0
    true_comp = true_comp[expected_cols]
    
    for method in pred_dfs:
        df = pred_dfs[method]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0.0
        pred_dfs[method] = df[expected_cols]
    
    # Clean and normalize
    true_comp = true_comp.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)
    for method in pred_dfs:
        pred_dfs[method] = pred_dfs[method].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Normalize predictions
    for method in pred_dfs:
        df = pred_dfs[method]
        row_sums = df.sum(axis=1)
        row_sums[row_sums == 0] = 1.0
        pred_dfs[method] = df.div(row_sums, axis=0)
    
    # Normalize ground truth
    true_row_sum = true_comp.sum(axis=1)
    true_row_sum[true_row_sum == 0] = 1.0
    true_comp = true_comp.div(true_row_sum, axis=0)
    
    # Compute metrics for each method
    results = {}
    metrics_list = []
    for method in method_names:
        if method not in pred_dfs:
            continue
        metrics_df = compute_metrics(true_comp.values, pred_dfs[method].values, expected_cols)
        results[method] = metrics_df
        metrics_list.append(metrics_df)
    
    # Compute ARS
    if len(metrics_list) > 0:
        ars_df = compute_ars(metrics_list)
        ars_df['method_name'] = [method_names[int(i)] for i in ars_df['method_id'] if int(i) < len(method_names)]
        results['ARS'] = ars_df
    
    return results


def get_method_colors():
    """Return colorblind-friendly palette for methods."""
    return {
        'Spagraph': '#0072B2',
        'Tangram': '#D55E00',
        'RCTD': '#009E73',
        'Seurat': '#CC79A7',
        'SpatialDWLS': '#F0E442',
        'SPOTlight': '#56B4E9',
        'DestVI': '#E69F00',
        'Stereoscope': '#999999'
    }


def get_method_display_name(method_name):
    """Keep method names unchanged for compact grid plots."""
    return method_name


def style_method_ticklabels(ax, fontsize):
    """Make method labels larger and bold without changing layout semantics."""
    for label in ax.get_yticklabels():
        label.set_fontsize(fontsize)
        label.set_fontweight('bold')
        label.set_horizontalalignment('right')


def save_plot_outputs(fig, output_pdf, png_dpi=600):
    """Save a vector PDF and a high-resolution PNG for documents/slides."""
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_pdf) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
        print(f"Saved {output_pdf}")

    output_png = output_pdf.with_suffix('.png')
    fig.savefig(output_png, dpi=png_dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved {output_png}")


def plot_single_dataset_boxplot(ax, metrics_dict, metric, methods, colors, dataset_id):
    """Plot a single dataset's boxplot for one metric on given axis."""
    box_data = []
    box_colors = []
    valid_labels = []
    
    for method in methods:
        if method not in metrics_dict:
            continue
        metrics_df = metrics_dict[method]
        if metric in metrics_df.columns:
            # Get all values for this metric (one per cell type)
            values = metrics_df[metric].dropna().values
            if len(values) > 0:
                box_data.append(values)
                box_colors.append(colors.get(method, '#0072B2'))
                valid_labels.append(get_method_display_name(method))
    
    if box_data and len(box_data) > 0:
        # Create boxplot
        bp = ax.boxplot(box_data, vert=False, labels=valid_labels, patch_artist=True,
                       whis=[5, 95], showfliers=False,
                       showmeans=True,
                       meanprops=dict(marker='D', markerfacecolor='#FF8C00', markeredgecolor='black', 
                                     markersize=4, markeredgewidth=0.8),
                       medianprops=dict(color='black', linewidth=1.5),
                       whiskerprops=dict(color='black', linewidth=1),
                       capprops=dict(color='black', linewidth=1),
                       boxprops=dict(linewidth=1))
        
        # Color boxes
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            patch.set_edgecolor('black')
            patch.set_linewidth(0.8)
        
        ax.set_title(f"Dataset {dataset_id}", fontsize=11, fontweight='bold', pad=2)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=13, pad=2, length=0)
        style_method_ticklabels(ax, fontsize=13)
        ax.grid(True, alpha=0.2, linestyle='--', axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_yaxis()
        
        # # Set fixed x-axis ranges for each metric
        # if metric == 'pcc':
        #     ax.set_xlim([0, 1])
        # elif metric == 'ssim':
        #     ax.set_xlim([0, 1])
        # elif metric == 'rmse':
        #     ax.set_xlim([0, 1.4])
        # elif metric == 'js':
        #     ax.set_xlim([0, 0.8])
    else:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])


def plot_single_dataset_ars(ax, ars_df, methods, colors, dataset_id):
    """Plot a single dataset's ARS as horizontal bar plot with value labels."""
    if ars_df is None or len(ars_df) == 0:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    
    bar_colors = []
    ars_values = []
    method_labels = []
    
    for method in methods:
        method_row = ars_df[ars_df['method_name'] == method]
        if not method_row.empty and 'ARS' in method_row.columns:
            ars_values.append(method_row['ARS'].values[0])
            bar_colors.append(colors.get(method, '#0072B2'))
            method_labels.append(method)
    
    if ars_values:
        y_pos = np.arange(len(method_labels))
        bars = ax.barh(y_pos, ars_values, color=bar_colors, alpha=0.8, height=0.7,
                      edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, ars_values):
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', 
                   ha='left', va='center', fontsize=8.5, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='none', alpha=0.7))
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([get_method_display_name(name) for name in method_labels], fontsize=13)
        ax.set_title(f"Dataset {dataset_id}", fontsize=11, fontweight='bold', pad=2)
        ax.tick_params(axis='x', labelsize=8.5)
        ax.tick_params(axis='y', labelsize=13, pad=2, length=0)
        style_method_ticklabels(ax, fontsize=13)
        ax.grid(True, alpha=0.2, linestyle='--', axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_yaxis()
        ax.set_xlim([0, 1.08])
    else:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])


def create_combined_metric_plot(all_results, metric, methods, colors, output_pdf):
    """Create 4x8 grid plot for one metric across all datasets."""
    n_datasets = len(all_results)
    n_cols = 4
    n_rows = 8
    
    # Create figure with larger size for better readability
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 28), squeeze=False)
    
    # Main title
    metric_name = metric.replace('mean_', '').upper() if 'mean_' in metric else metric.upper()
    fig.suptitle(f'{metric_name} Metrics (n={n_datasets} datasets)', 
                 fontsize=24, fontweight='bold', y=0.985)
    
    # Create subplots
    for idx, dataset_id in enumerate(sorted(all_results.keys())):
        if idx >= n_rows * n_cols:
            break
        
        row = idx // n_cols
        col = idx % n_cols
        
        # Create subplot
        ax = axes[row, col]
        
        # Plot data
        metrics_dict = all_results[dataset_id]
        plot_single_dataset_boxplot(ax, metrics_dict, metric, methods, colors, dataset_id)
        
        # Only show y-labels on leftmost column
        if col != 0:
            ax.tick_params(axis='y', labelleft=False)
        
        # Add x-label only on bottom row
        if row == n_rows - 1:
            ax.set_xlabel(metric_name, fontsize=11)
    
    # Remove empty subplots
    for idx in range(n_datasets, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    fig.subplots_adjust(left=0.12, right=0.992, top=0.965, bottom=0.04, wspace=0.10, hspace=0.18)
    save_plot_outputs(fig, output_pdf)
    plt.close(fig)
    return
    
    fig.subplots_adjust(left=0.14, right=0.985, top=0.972, bottom=0.04, wspace=0.32, hspace=0.48)
    
    # Save to PDF
    with PdfPages(output_pdf) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        print(f"✓ Saved {output_pdf}")
    
    plt.close(fig)


def create_combined_ars_plot(all_results, methods, colors, output_pdf):
    """Create 4x8 grid plot for ARS across all datasets."""
    n_datasets = len(all_results)
    n_cols = 4
    n_rows = 8
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 28), squeeze=False)
    
    # Main title
    fig.suptitle(f'ARS (Average Ranking Score)  n={n_datasets} datasets', 
                 fontsize=24, fontweight='bold', y=0.985)
    
    # Create subplots
    for idx, dataset_id in enumerate(sorted(all_results.keys())):
        if idx >= n_rows * n_cols:
            break
        
        row = idx // n_cols
        col = idx % n_cols
        
        # Create subplot
        ax = axes[row, col]
        
        # Get ARS data
        metrics_dict = all_results[dataset_id]
        ars_df = metrics_dict.get('ARS', None)
        
        # Plot ARS
        plot_single_dataset_ars(ax, ars_df, methods, colors, dataset_id)
        
        # Only show y-labels on leftmost column
        if col != 0:
            ax.tick_params(axis='y', labelleft=False)
        
        # Add x-label only on bottom row
        if row == n_rows - 1:
            ax.set_xlabel('ARS', fontsize=11)
    
    # Remove empty subplots
    for idx in range(n_datasets, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    fig.subplots_adjust(left=0.12, right=0.992, top=0.965, bottom=0.04, wspace=0.10, hspace=0.18)
    save_plot_outputs(fig, output_pdf)
    plt.close(fig)
    return
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save to PDF
    with PdfPages(output_pdf) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        print(f"✓ Saved {output_pdf}")
    
    plt.close(fig)


def main():
    # Set default parameters
    base_dir = DATA_ROOT
    n_datasets = 32
    output_dir = COMBINED_PLOTS_DIR
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define methods
    methods = ['Spagraph', 'Tangram', 'RCTD', 'Seurat', 'SpatialDWLS', 'SPOTlight', 'DestVI', 'Stereoscope']
    colors = get_method_colors()
    
    # Compute metrics for all datasets
    print(f"\n{'='*60}")
    print(f"Computing metrics for {n_datasets} datasets from {base_dir}")
    print(f"{'='*60}\n")
    
    all_results = {}
    for i in range(1, n_datasets + 1):
        print(f"Processing Data{i}...")
        results = compute_dataset_metrics(base_dir, i, methods)
        if results:
            all_results[i] = results
            print(f"  ✓ Data{i} completed")
        else:
            print(f"  ✗ Data{i} skipped")
    
    if len(all_results) == 0:
        print("❌ No datasets were successfully processed!")
        return
    
    print(f"\n✓ Successfully processed {len(all_results)} datasets\n")
    
    # Generate plots for each metric
    metrics = ['pcc', 'ssim', 'rmse', 'js']
    
    print(f"{'='*60}")
    print(f"Generating combined plots (8x4 grid)")
    print(f"{'='*60}\n")
    
    for metric in metrics:
        output_pdf = output_dir / f"combined_{metric}.pdf"
        print(f"Creating {metric} plot...")
        create_combined_metric_plot(all_results, metric, methods, colors, output_pdf)
    
    # Generate ARS plot
    print(f"Creating ARS plot...")
    ars_pdf = output_dir / "combined_ARS.pdf"
    create_combined_ars_plot(all_results, methods, colors, ars_pdf)
    
    print(f"\n{'='*60}")
    print(f"✅ All plots saved to: {output_dir.absolute()}")
    print(f"{'='*60}\n")
    print(f"Generated files:")
    for metric in metrics:
        print(f"  - combined_{metric}.pdf")
    print(f"  - combined_ARS.pdf")


if __name__ == "__main__":
    main()
