# 第二阶段与第三阶段数据集成改进

## 问题背景

### 原始实现的问题
1. **第二阶段（反卷积）**：使用动态cluster表示（k-nearest cells）重建spot级别的全基因表达
2. **第三阶段（细胞通讯）**：使用静态cluster表达（从NPZ加载的平均表达）+ 反卷积比例矩阵来构建spot-cell表达
3. **不一致性**：第三阶段没有利用第二阶段的动态重建结果，导致数据不一致

### 改进方案
✅ **在第二阶段直接生成spot-cell级别的动态表达，第三阶段直接使用**

---

## 实现细节

### 1. 第二阶段改进 (`stage2.py`)

在 `evaluate_and_visualize()` 函数中，当 `save_reconstructed_genes=True` 时：

```python
# 原有功能：保存spot级别的重构表达
{sample_name}_reconstructed.csv  # [n_spots, n_all_genes]

# 新增功能：保存spot-cell级别的动态表达
{sample_name}_spot_cell_expr.csv  # [n_spot_cells, n_all_genes]
```

#### Spot-Cell表达计算公式

对于每个 `(spot, celltype)` 对：

```
spot_cell_expr = Σ_k (cell_k_expr * weight_k) * cluster_proportion * spot_total_count
```

其中：
- `cell_k_expr`: 第k个nearest cell的全基因表达（normalized to 1e4）
- `weight_k`: 该cell的权重（均匀权重 = 1/k）
- `cluster_proportion`: 该cluster在该spot的反卷积比例
- `spot_total_count`: 该spot的原始总count

### 2. 第三阶段改进 (`cellcom.py`)

优先级加载策略：

```python
# 优先级1: 第二阶段自动生成的动态表达（推荐）
deconv_dir/*_spot_cell_expr.csv

# 优先级2: 用户手动指定的CSV文件
--spot_cell_expr_csv <path>

# 优先级3: 回退到静态cluster构建（不推荐，会有警告）
使用 cluster_full_expr × cluster_composition
```

---

## 使用方法

### 基本流程

```python
import spagraph as spg

# 第一阶段：VAE训练
spg.train_vae(
    sc_h5ad="data/sc.h5ad",
    output_dir="output/deconv/",
    n_clusters=15
)

# 第二阶段：GAT反卷积（✅ 自动生成spot-cell动态表达）
spg.deconv(
    deconv_dir="output/deconv/",
    st_h5ad="data/st.h5ad",
    output_dir="output/deconv/",
    save_reconstructed_genes=True,  # ✅ 必须启用才会生成spot-cell表达
    k_celltype=10  # 动态cluster的k值
)

# 第三阶段：细胞通讯（✅ 自动使用第二阶段生成的动态表达）
spg.cellcom(
    deconv_dir="output/deconv/",  # 自动查找 *_spot_cell_expr.csv
    st_h5ad="data/st.h5ad",
    output_dir="output/cellcom/",
    epochs=100
)
```

### 输出文件

第二阶段输出：
```
output/deconv/
├── final_vae.pth
├── final_vae_cluster_data.npz
├── {sample}_cluster_composition.csv       # spot × cluster 比例矩阵
├── {sample}_reconstructed.csv             # spot级别重构表达
└── {sample}_spot_cell_expr.csv            # ✅ 新增：spot-cell动态表达
```

第三阶段输入：
- 自动检测并使用 `{sample}_spot_cell_expr.csv`
- 如果找不到，会有警告并回退到静态构建

---

## 技术优势

### 1. 数据一致性
- 第三阶段使用的spot-cell表达与第二阶段的重建逻辑完全一致
- 都基于动态cluster表示（k-nearest cells）

### 2. 计算效率
- 第三阶段不需要重新计算spot-cell表达
- 直接加载CSV，节省计算时间

### 3. 可复现性
- spot-cell表达保存为文件，可以用于后续分析
- 避免了第三阶段重复计算带来的数值差异

### 4. 向后兼容
- 如果第二阶段没有生成spot-cell文件（旧版本），第三阶段仍然可以回退到静态构建
- 用户仍然可以手动指定 `--spot_cell_expr_csv`

---

## 注意事项

1. **必须启用 `save_reconstructed_genes=True`**
   - 只有启用此选项，第二阶段才会生成spot-cell表达文件

2. **动态模式要求**
   - 第二阶段必须启用动态cluster模式（默认已启用）
   - 需要提供 `k_celltype` 参数（表示每个cluster取k个nearest cells）

3. **文件命名约定**
   - 第二阶段生成的文件名：`{sample_name}_spot_cell_expr.csv`
   - 第三阶段使用glob模式自动查找：`*_spot_cell_expr.csv`

4. **静态构建警告**
   - 如果第三阶段找不到动态表达文件，会输出警告：
   ```
   ⚠️ 未找到第二阶段生成的spot-cell动态表达文件，使用静态cluster表达构建（可能不一致）
   ```

---

## 测试验证

检查是否正确集成：

```python
import os
import glob

deconv_dir = "output/deconv/"

# 检查第二阶段输出
spot_cell_files = glob.glob(os.path.join(deconv_dir, "*_spot_cell_expr.csv"))
if spot_cell_files:
    print(f"✅ 找到spot-cell动态表达文件: {spot_cell_files[0]}")
else:
    print("❌ 未找到spot-cell动态表达文件，请检查 save_reconstructed_genes=True")

# 运行第三阶段并检查日志
# 应该看到：✅ 检测到第二阶段生成的spot-cell动态表达文件: ...
```

---

## 常见问题

**Q: 第三阶段仍然显示"使用静态cluster表达构建"？**

A: 检查以下几点：
1. 第二阶段是否设置了 `save_reconstructed_genes=True`
2. deconv_dir 路径是否正确
3. 第二阶段是否成功运行完成
4. 检查 deconv_dir 中是否存在 `*_spot_cell_expr.csv` 文件

**Q: 能否手动指定spot-cell表达文件？**

A: 可以，使用 `spot_cell_expr_csv` 参数：
```python
spg.cellcom(
    deconv_dir="output/deconv/",
    st_h5ad="data/st.h5ad",
    spot_cell_expr_csv="custom_spot_cell.csv",  # 手动指定
    output_dir="output/cellcom/"
)
```
注意：手动指定的文件优先级低于第二阶段自动生成的文件。

**Q: 静态构建和动态构建的差异有多大？**

A: 差异取决于数据的异质性：
- 如果cluster内部细胞表达高度一致，差异较小
- 如果cluster内部存在明显亚群，动态构建能更准确地反映局部异质性
- 推荐使用动态构建以保证数据一致性
