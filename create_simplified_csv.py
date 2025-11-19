import pandas as pd

# 读取CSV文件
df = pd.read_csv('./results/CID44971/lr_communication_model_based.csv')

# 过滤掉score_type，只保留每组的original, logits, adjusted
df_pivot = df.pivot_table(
    index=['center_spot', 'source_cell', 'target_cell', 'lr_pair'],
    columns='score_type',
    values=['original_lr_score', 'edge_logits', 'adjusted_score']
).reset_index()

# 展平列名
df_pivot.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_pivot.columns]

# 重命名列
df_pivot = df_pivot.rename(columns={
    'original_lr_score_original': 'original_lr_score',
    'edge_logits_logits': 'edge_logits',
    'adjusted_score_adjusted': 'adjusted_score'
})

# 添加spot barcode列
df_pivot['center_spot_barcode'] = df_pivot['center_spot'].astype(str)
df_pivot['target_spot_barcode'] = df_pivot['center_spot'].astype(str)  # intra-spot, so same

# 重新排列列顺序
columns_order = [
    'center_spot_barcode', 'target_spot_barcode', 'source_cell', 'target_cell',
    'lr_pair', 'original_lr_score', 'edge_logits', 'adjusted_score'
]

df_final = df_pivot[columns_order]

# 保存到新CSV文件
df_final.to_csv('./results/CID44971/simplified_lr_communication.csv', index=False)

print("Simplified CSV created successfully!")
print(f"Total edges: {len(df_final)}")