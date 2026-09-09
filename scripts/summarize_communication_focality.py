"""Recompute original focality using complete historical LR edge support."""
import argparse
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--spatial',type=Path,required=True)
    p.add_argument('--groups',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    keys=['src_spot_barcode','dst_spot_barcode','source_cell','target_cell']
    frames=[]
    for seed in [11,23,42,67,101]:
        d=pd.read_csv(args.run_root/f'seed_{seed}/communication_edge_statistics.csv')
        if d.duplicated(keys).any(): raise ValueError('Duplicate aggregate edges')
        frames.append(d.set_index(keys).sort_index())
    base=frames[0].copy()
    for d in frames:
        assert base.index.equals(d.index)
        assert base.supporting_lr_pairs.equals(d.supporting_lr_pairs)
    base['edge_attention']=sum(d.edge_attention for d in frames)/len(frames)
    a=ad.read_h5ad(args.spatial,backed='r')
    coords=pd.DataFrame(np.asarray(a.obsm['spatial']),index=a.obs_names)
    a.file.close()
    diagonal=np.linalg.norm(np.ptp(coords.to_numpy(),axis=0))
    assert diagonal>0 and coords.index.is_unique
    d=base.reset_index()
    d['lr_pair']=d.supporting_lr_pairs.str.split(';')
    events=d.explode('lr_pair')
    groups=pd.read_csv(args.groups).set_index('lr_pair')
    if not groups.index.is_unique:
        raise ValueError('Grouping table must contain unique LR identities')
    if not groups['group'].isin(['other','attention','frequency','both']).all():
        raise ValueError('Unknown selection group')
    rows=[]
    for pair,g in events.groupby('lr_pair'):
        if pair not in groups.index: continue
        if len(g) != int(groups.loc[pair, 'event_count']):
            raise ValueError(f'Edge support count disagrees with ranking input: {pair}')
        source=coords.loc[g.src_spot_barcode].to_numpy()
        target=coords.loc[g.dst_spot_barcode].to_numpy()
        mid=(source+target)/2
        w=g.edge_attention.clip(lower=0).to_numpy()
        if not w.sum(): w=np.ones(len(w))
        center=np.average(mid,axis=0,weights=w)
        spread=np.average(np.linalg.norm(mid-center,axis=1),weights=w)
        rows.append({'lr_pair':pair,'group':groups.loc[pair,'group'],
                     'edge_spatial_focality':np.clip(1-spread/diagonal,0,1),
                     'celltype_pair_count':len(g[['source_cell','target_cell']].drop_duplicates()),
                     'supporting_unique_edges':len(g)})
    result=pd.DataFrame(rows)
    assert len(result)==len(groups)
    if not np.isfinite(result[['edge_spatial_focality','celltype_pair_count']]).all().all():
        raise ValueError('Spatial metrics contain non-finite values')
    summary=[]
    for metric in ['edge_spatial_focality','celltype_pair_count']:
        x=result.loc[result.group.eq('attention'),metric]
        y=result.loc[result.group.eq('frequency'),metric]
        if len(x)<2 or len(y)<2:
            raise ValueError('At least two LR pairs per disjoint group are required')
        test=mannwhitneyu(x,y,alternative='two-sided',method='asymptotic')
        summary.append({'metric':metric,'n_attention':len(x),'n_frequency':len(y),
                        'median_attention':x.median(),'median_frequency':y.median(),
                        'U':test.statistic,'nominal_p':test.pvalue,
                        'rank_biserial':2*test.statistic/(len(x)*len(y))-1})
    s=pd.DataFrame(summary)
    order=np.argsort(s.nominal_p.to_numpy())
    adjusted=np.maximum.accumulate(s.nominal_p.to_numpy()[order]*np.array([2,1])).clip(0,1)
    s['nominal_p_holm']=np.nan
    s.loc[order,'nominal_p_holm']=adjusted
    args.output.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output/'fig3e_values.csv',index=False)
    s.to_csv(args.output/'fig3e_summary.csv',index=False)
    print(s.to_string(index=False))
    print('Caution: shared LR support violates independent biological replication; P values are descriptive nominal tests.')


if __name__=='__main__': main()
