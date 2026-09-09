"""Reuse the manuscript rank-strip renderer with a common LR comparison set."""
from pathlib import Path
import argparse
import importlib.util
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ensemble', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / 'evaluate/scripts/cc_com/scc_multimethod_rank_panel_f.py'
    spec = importlib.util.spec_from_file_location('rank_strip', source)
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    plot.OUTPUT_DIR = args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sp = pd.read_csv(args.ensemble)
    if 'eligible_for_ranking' in sp:
        sp = sp[sp.eligible_for_ranking.astype(str).str.lower().isin(['true', '1'])]
    score = 'mean_attention_percentile' if 'mean_attention_percentile' in sp else 'mean_percentile'
    cc = pd.read_csv(args.data_dir / 'cellchat_baseline_allspots_nboot5/cellchat_lr_communications.csv')
    cc['interaction_name'] = cc.interaction_name.replace({'LGALS9_CD45': 'LGALS9_PTPRC'})
    cm = pd.read_csv(args.data_dir / 'commot_baseline_perm20_min5/commot_pair_summary_cross_group.csv')
    gi = pd.read_csv(args.data_dir / 'giotto_baseline_iter20/giotto_pair_summary.csv')
    series = {'spagraph': sp.set_index('lr_pair')[score],
              'cellchat': cc.groupby('interaction_name').prob.max(),
              'commot': cm.set_index('interaction_name').commot_score,
              'giotto': gi.set_index('interaction_name').PI_spat}
    for name, values in series.items():
        if not values.index.is_unique or not np.isfinite(values).all():
            raise ValueError(f'Invalid or duplicate scores: {name}')
    common = sorted(set.intersection(*(set(v.index) for v in series.values())))
    if len(common) < 2:
        raise ValueError('At least two common LR pairs are required')
    table = pd.DataFrame({'lr_pair': common}).set_index('lr_pair')
    for name, values in series.items():
        table[f'{name}_score'] = values.reindex(common)
        rank = values.reindex(common).rank(ascending=False, method='min')
        table[f'{name}_rank'] = rank
        table[f'{name}_rank_pct'] = 100 * (1 - (rank - 1)/(len(common)-1))
    table.to_csv(args.output_dir / 'common_pair_ranks.csv')
    pd.DataFrame({'method': list(series), 'eligible_pairs': [len(v) for v in series.values()],
                  'common_pairs': len(common)}).to_csv(args.output_dir / 'comparison_universe.csv', index=False)
    selected = [p for p, _ in plot.PAIRS]
    panel = table.loc[selected].reset_index()
    panel['commot_percentile'] = panel.commot_rank_pct
    panel['giotto_percentile'] = panel.giotto_rank_pct
    panel['giotto_spatial_rank'] = panel.giotto_rank
    panel['pair_label'] = panel.lr_pair.map(plot.format_pair)
    panel['y'] = np.arange(len(panel))[::-1]
    plot.PAIR_GROUPS = [('', selected)]
    plot.GROUP_BAND_COLORS = ['#F6F7FA']
    plot.draw_panel_f(panel, axis_label=f'Rank percentile among {len(common)} shared LR pairs (%)')
    panel.to_csv(args.output_dir / 'fig3d_values.csv', index=False)


if __name__ == '__main__':
    main()
