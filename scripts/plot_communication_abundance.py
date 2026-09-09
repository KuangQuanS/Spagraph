"""Adapt the original CCC abundance scatter to the retained consensus ranking."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair-tables', type=Path, nargs='+', required=True)
    parser.add_argument('--consensus', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    tables = [pd.read_csv(p).set_index('lr_pair') for p in args.pair_tables]
    consensus = pd.read_csv(args.consensus).set_index('lr_pair')
    pairs = consensus.index
    abundance = tables[0].loc[pairs, 'total_lr_score']
    count = tables[0].loc[pairs, 'supporting_unique_edges']
    for table in tables:
        assert table.index.is_unique
        assert np.allclose(table.loc[pairs, 'total_lr_score'], abundance)
        assert table.loc[pairs, 'supporting_unique_edges'].eq(count).all()
    d = pd.DataFrame({'abundance': abundance, 'event_count': count,
                      'consensus_percentile': consensus.mean_percentile,
                      'consensus_rank': consensus['rank']})
    assert np.isfinite(d.to_numpy()).all() and d.abundance.ge(0).all()
    # Preserve the original top-15 selection rule: attention versus event frequency.
    frequency = d.sort_values(['event_count', 'consensus_percentile'], ascending=False).head(15).index
    ranked = d.sort_values(['consensus_rank'], kind='stable').head(15).index
    d['group'] = 'other'
    d.loc[ranked, 'group'] = 'attention'
    d.loc[frequency, 'group'] = 'frequency'
    d.loc[ranked.intersection(frequency), 'group'] = 'both'
    rho = d.abundance.rank().corr(d.consensus_percentile.rank())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.output_dir / 'fig3c_values.csv')
    pd.DataFrame([{'n_pairs': len(d), 'n_training_seeds': len(tables), 'spearman_rho': rho}]).to_csv(args.output_dir / 'fig3c_summary.csv', index=False)
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial','DejaVu Sans'],
                         'pdf.fonttype':42, 'svg.fonttype':'none', 'font.size':14.5})
    fig, ax = plt.subplots(figsize=(14,8), dpi=300)
    styles = [('other','#D9D9D9',38,'Other'), ('attention','#C44E52',108,'Consensus top 15'),
              ('frequency','#4C72B0',108,'Frequency top 15'), ('both','#7A5195',128,'Both')]
    for group,color,size,label in styles:
        sub=d[d.group.eq(group)]
        if sub.empty:
            continue
        ax.scatter(np.log10(sub.abundance+1),sub.consensus_percentile,s=size,c=color,
                   alpha=.72 if group=='other' else .96, edgecolors='white',linewidths=.9,label=label)
    ax.text(.03,.97,f'Spearman ρ = {rho:.3f}\nn = {len(d)} LR pairs',transform=ax.transAxes,
            va='top',fontsize=15.5)
    ax.set_xlabel('log10(LR abundance + 1)',fontsize=18)
    ax.set_ylabel('Mean within-run attention percentile',fontsize=18)
    ax.set_ylim(0,1.04)
    ax.grid(color='#EAEAEA',linewidth=.9); ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False,loc='center left',bbox_to_anchor=(1.01,.5))
    fig.tight_layout(rect=(0,0,.85,1))
    fig.savefig(args.output_dir/'fig3c.pdf',bbox_inches='tight')
    fig.savefig(args.output_dir/'fig3c.svg',bbox_inches='tight')
    fig.savefig(args.output_dir/'fig3c.png',dpi=300,bbox_inches='tight')
    plt.close(fig)
    print(f'n={len(d)}, rho={rho:.9f}, top15 intersection={len(ranked.intersection(frequency))}')


if __name__ == '__main__':
    main()
