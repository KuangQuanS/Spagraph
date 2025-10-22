# SC-MAP-ST 反卷积结果输出说明

## 📋 概述

经过 Stage2 GAT 解卷积训练，系统会自动生成以下反卷积结果文件，这些文件保存在 `--output_dir` 指定的目录中。

## 📁 输出文件详细说明

### 1. 反卷积权重和注意力分数
**文件名**: `{sample_name}_deconv_weights.npz`

**内容**:
- `deconv_weights`: 形状 [n_spots, n_clusters]，每行和为1.0的权重矩阵
  - 表示每个 spot 中各 celltype 的比例
- `attention_scores`: 形状 [n_spots, n_clusters]，GAT 原始注意力分数
- `clusters`: 聚类列表

**使用场景**: 直接获取空间解卷积结果，用于后续分析

---

### 2. Marker 基因重建表达矩阵
**文件名**: `{sample_name}_reconstructed_marker_genes.csv`

**内容**:
- 行: spot (Spot_0, Spot_1, ...)
- 列: marker 基因 (GeneA, GeneB, ...)
- 值: 重建的基因表达值

**计算方式**:
```
reconstructed_marker_expr[i, j] = sum(deconv_weights[i, k] * celltype_expr_marker[k, j])
                                 = 对每个 celltype 加权求和
```

**大小**: n_spots × n_marker_genes (通常 1000-3000 × 100)

**使用场景**:
- 验证解卷积质量
- 与原始 spot 表达对比
- Marker 基因的空间表达模式分析

---

### 3. 全基因重建表达矩阵
**文件名**: `{sample_name}_reconstructed_all_genes.csv`

**内容**:
- 行: spot (Spot_0, Spot_1, ...)
- 列: 所有基因 (Gene_0, Gene_1, ...)
- 值: 重建的全基因表达值

**计算方式**:
```
reconstructed_full_expr[i, j] = sum(deconv_weights[i, k] * celltype_expr_full[k, j])
```

**大小**: n_spots × n_all_genes (通常 1000-3000 × 10000-20000)

**前提条件**: 需要在 Stage1 中计算并保存每个 celltype 的全基因平均表达

**使用场景**:
- 获得完整的基因表达预测
- 进行基因水平的空间表达分析
- 与单细胞数据整合

---

### 4. 细胞类型组成矩阵
**文件名**: `{sample_name}_cell_composition.csv`

**内容**:
- 行: spot (Spot_0, Spot_1, ...)
- 列: celltype (Cluster_0, Cluster_1, ...)
- 值: 每个 spot 中各 celltype 的比例 (0-1，行和为1)

**大小**: n_spots × n_clusters (通常 1000-3000 × 25)

**使用场景**:
- 空间细胞类型组成可视化
- 细胞类型空间分布分析
- 比较不同区域的细胞类型差异

---

### 5. 完整结果摘要
**文件名**: `{sample_name}_deconvolution_results.npz`

**内容**: 综合了所有关键结果的压缩文件，包括：
- deconv_weights
- attention_scores
- clusters
- marker_genes
- n_spots
- n_clusters

**使用场景**: Python 中快速加载所有结果用于进一步分析

---

## 🔍 结果验证

在训练结束时，系统会自动打印以下验证信息：

```
📈 解卷积权重统计 (聚类比例):
   Cluster 0: 0.123 ± 0.045
   Cluster 1: 0.234 ± 0.067
   ...

🔍 解卷积权重和验证:
   权重和: 1.000000 ± 0.000001 (应该等于1.0)
```

**检查项目**:
- ✅ 权重和应等于 1.0
- ✅ 没有负数权重
- ✅ 各聚类的平均比例合理

---

## 📊 如何使用输出结果

### Python 加载
```python
import pandas as pd
import numpy as np

# 加载重建表达
marker_expr = pd.read_csv('sample_reconstructed_marker_genes.csv', index_col=0)
full_expr = pd.read_csv('sample_reconstructed_all_genes.csv', index_col=0)
composition = pd.read_csv('sample_cell_composition.csv', index_col=0)

# 加载权重
data = np.load('sample_deconv_weights.npz')
deconv_weights = data['deconv_weights']
```

### R 加载
```r
# 加载为 data.frame
marker_expr <- read.csv('sample_reconstructed_marker_genes.csv', row.names=1)
composition <- read.csv('sample_cell_composition.csv', row.names=1)
```

---

## 💡 常见问题

**Q: 为什么全基因表达矩阵的基因名是 Gene_0, Gene_1, ...？**
A: 因为需要保留 Stage1 中计算的全基因顺序。如果需要真实基因名，可以从原始单细胞数据中获取基因映射。

**Q: 如何确保反卷积结果的准确性？**
A: 检查以下几点：
1. Pearson 相关系数和 Cosine 相似度都在下降
2. Alignment Loss 也在下降
3. 权重和精确等于 1.0
4. 各聚类的比例符合生物学常识

**Q: 可以直接将重建表达用于下游分析吗？**
A: 可以，但建议：
1. 先与原始表达进行对比验证
2. 了解模型的局限性（只使用了 marker 基因训练）
3. 在必要时进行归一化

---

## 📐 文件大小参考

| 文件 | 大小(典型) | 压缩率 |
|------|-----------|--------|
| marker_genes.csv | 1-5 MB | 无压缩 |
| all_genes.csv | 50-200 MB | 无压缩 |
| cell_composition.csv | 50-500 KB | 无压缩 |
| deconv_weights.npz | 1-2 MB | 压缩 |

建议定期备份重要的结果文件。
