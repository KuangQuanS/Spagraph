# import pandas as pd
# from sklearn.preprocessing import OneHotEncoder

# # === 配置 ===
# input_csv_path = "cellchat_human.csv"           # 输入文件，包含 ligand 和 receptor 两列
# output_csv_path = "ligand_receptor_encoded.csv"  # 输出文件：包含 ligand_receptor 和 one-hot 字符串

# # === 读取并拼接 ligand 和 receptor，统一大写 ===
# df = pd.read_csv(input_csv_path)
# df["ligand_receptor"] = df["ligand"].str.upper() + "_" + df["receptor"].str.upper()

# # === 获取唯一组合并进行 one-hot 编码 ===
# lr_unique = df[["ligand_receptor"]].drop_duplicates().reset_index(drop=True)
# encoder = OneHotEncoder(sparse_output=False, dtype=int)
# onehot = encoder.fit_transform(lr_unique)

# # === 把 one-hot 转为字符串形式，例如：'010000100000' ===
# onehot_str = [''.join(map(str, row)) for row in onehot]

# # === 组合保存 ===
# output_df = pd.DataFrame({
#     "ligand_receptor": lr_unique["ligand_receptor"],
#     "onehot": onehot_str
# })
# output_df.to_csv(output_csv_path, index=False)

# print(f"✅ 编码完成，已保存至：{output_csv_path}")

from sklearn.preprocessing import LabelEncoder
import pandas as pd

# 读取数据
df = pd.read_csv("cellchat_human.csv")
df["ligand_receptor"] = df["ligand"].str.upper() + "_" + df["receptor"].str.upper()

# Label 编码
le = LabelEncoder()
df["lr_id"] = le.fit_transform(df["ligand_receptor"])

# 保存映射表
mapping_df = pd.DataFrame({
    "ligand_receptor": le.classes_,
    "lr_id": range(len(le.classes_))
})
mapping_df.to_csv("ligand_receptor_label_mapping.csv", index=False)

# 保存编码结果
df[["ligand_receptor", "lr_id"]].to_csv("ligand_receptor_labeled.csv", index=False)

print("✅ 编码完成，建议模型中使用 nn.Embedding(1939, emb_dim)")
