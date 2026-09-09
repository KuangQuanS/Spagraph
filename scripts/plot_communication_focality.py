"""Render Fig3e with the original boxplot styling and corrected group values.

Structural adaptation of the project's CCC boxplot panel: no statistics are
recomputed here. All 15 LR pairs in each precomputed group are displayed.
Boxes show quartiles, median and 1.5-IQR whiskers; points show every LR pair.
The two metrics describe spatial concentration and cell-type breadth, not
independent biological replication. Outputs are editable PDF/SVG plus PNG.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    assert data.lr_pair.is_unique
    selected = data[data.group.isin(['attention', 'frequency'])]
    assert selected.groupby('group').size().to_dict() == {'attention': 15, 'frequency': 15}
    metrics = [('edge_spatial_focality', 'Edge spatial focality'),
               ('celltype_pair_count', 'Cell-type-pair count')]
    assert np.isfinite(selected[[m for m, _ in metrics]]).all().all()
    plt.rcParams.update({'font.family': 'serif',
                         'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                         'font.size': 13, 'font.weight': 'semibold',
                         'axes.labelsize': 16, 'axes.labelweight': 'bold',
                         'axes.titleweight': 'bold', 'axes.linewidth': 0.9,
                         'xtick.labelsize': 13, 'ytick.labelsize': 13,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    colors = ('#C44E52', '#4C72B0')
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.6))
    for ax, (metric, label) in zip(axes, metrics):
        values = [selected.loc[selected.group.eq(g), metric].to_numpy()
                  for g in ['attention', 'frequency']]
        box = ax.boxplot(values,
                         tick_labels=['Attention-only\nLR pairs (n=15)',
                                      'Frequency-only\nLR pairs (n=15)'],
                         patch_artist=True, widths=0.52, showfliers=False,
                         medianprops={'color': '#111111', 'linewidth': 1.4},
                         whiskerprops={'color': '#666666', 'linewidth': 1.0},
                         capprops={'color': '#666666', 'linewidth': 1.0},
                         boxprops={'edgecolor': '#666666', 'linewidth': 1.0})
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
        rng = np.random.default_rng(42)  # Jitter only; no simulated measurements.
        for pos, (series, color) in enumerate(zip(values, colors), start=1):
            ax.scatter(pos + rng.normal(0, 0.045, len(series)), series,
                       s=20, color=color, alpha=0.82, linewidths=0.3,
                       edgecolors='white', zorder=3)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#666666')
        ax.tick_params(colors='#333333')
        ax.grid(axis='y', color='#EAEAEA', linewidth=0.9)
        ax.set_axisbelow(True)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight('bold')
        ax.set_ylabel(label)
        ax.set_title(label, pad=8, fontsize=11)
        ax.margins(y=0.12)
    # Nominal tests belong in the caption, not a biological-significance badge.
    fig.tight_layout(w_pad=2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / 'fig3e.pdf', bbox_inches='tight')
    fig.savefig(args.output_dir / 'fig3e.svg', bbox_inches='tight')
    fig.savefig(args.output_dir / 'fig3e.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
