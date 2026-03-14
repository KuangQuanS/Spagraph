import scanpy as sc
import pandas as pd
from pathlib import Path

# 创建输出文件夹
output_dir = Path("./seqFISH+/gene_prediction_fig")
output_dir.mkdir(parents=True, exist_ok=True)

# 设置 scanpy 的输出目录
sc.settings.figdir = str(output_dir)


counts = pd.read_csv('f:/ST_Graduation_Project/spagraph_data/database/seqFISH+/Spatial_count.txt',sep='\t')
print(counts)
location = pd.read_csv('f:/ST_Graduation_Project/spagraph_data/database/seqFISH+/Locations.txt',sep='\t')
print(location)

adata = sc.AnnData(counts.values)
print(adata.obs_names)
adata.obsm['spatial'] = location.values
# 反转 y 轴
adata.obsm['spatial'][:, 1] = -adata.obsm['spatial'][:, 1]

meta = pd.read_csv('f:/ST_Graduation_Project/spagraph_data/database/seqFISH+/Spatial_annotate.txt', sep='\t',index_col=0)
print(meta)

adata.obs = meta
print(adata.obs)
print(adata.obs['celltype'].value_counts())
# 把 celltype 设成 category（画图更好看）
adata.obs['celltype'] = adata.obs['celltype'].astype('category')

sc.pl.embedding(
    adata,
    basis='spatial',
    color='celltype',
    s=50,           # 点大小
    alpha=0.8,      # 透明度
    show=False,
    save='.pdf'
)