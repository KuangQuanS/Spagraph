# 动态Cluster表示功能使用说明

## 功能概述

新增的动态cluster表示功能允许模型不使用固定的cluster平均表达，而是为每个spot动态选择最近的k个细胞进行加权组合。

## 核心思想

**传统方法（静态）**:
```
Spot表达 = Σ(w_c × cluster_c的平均表达)  [raw counts]
```

**新方法（动态）**:
```
Spot表达 = Σ(w_c × Σ(α_ci × cell_ci的原始count表达))
其中:
- cell_ci 是cluster_c中距离spot最近的k个细胞（基于embedding余弦相似度）
- α_ci 是可学习的权重（通过MLP计算，基于spot和cell的embedding，softmax归一化，Σα_ci=1）
- cell_ci的表达使用原始count（不做normalize/log1p，只有VAE阶段做normalize+log1p）
```

**图结构**:
```
Spot节点 → Cluster节点 → Cell节点
         ↑ GAT学习     ↑ MLP学习，归一化为百分比
         w_c权重       α_ci百分比（和为1）
```

## 使用方法

### 1. Stage 1 保持不变

```python
art = spg.vae(sc_file=sc_file, st_file=st_file)
```

### 2. Stage 2 启用动态模式

需要在 `stage2.py` 的 `build_gat_model` 和训练循环中添加参数：

#### 修改 `build_gat_model` 调用

```python
trainer.build_gat_model(
    n_cell_types=n_clusters,
    ...
    # 新增参数
    use_dynamic_cluster_repr=True,  # 启用动态模式
    k_cells_per_cluster=10,         # 每个cluster选择10个最近细胞
    sc_cell_embeddings=sc_embeddings,  # [n_cells, embedding_dim] VAE输出
    sc_cell_expressions=sc_expr_full,  # [n_cells, n_all_genes] ⚠️ 必须是原始count（raw counts）
    sc_cell_labels=sc_labels           # [n_cells] cluster标签
)
```

**⚠️ 重要**: `sc_cell_expressions` 必须是原始count矩阵（不要做normalize或log1p），因为Loss函数需要用原始count进行缩放计算。

#### 修改训练循环

在 `train_epoch_batched` 中传递动态cluster权重：

```python
# Forward GAT
gat_outputs = self.gat_model(
    spot_embeddings=spot_embeddings,
    spatial_coords=batch_coords,
    celltype_prototypes=self.celltype_prototypes,
    use_embedding_knn=self.use_embedding_knn
)

# 提取动态权重（如果启用）
dynamic_weights = gat_outputs.get('dynamic_cluster_weights')
dynamic_indices = gat_outputs.get('dynamic_cluster_indices')

# 计算loss
loss_dict = self.loss_fn(
    attention_weights=deconv_weights,
    celltype_expression=None,
    true_spot_expression=batch_raw,
    ...
    # 传递动态参数
    dynamic_cluster_weights=dynamic_weights,
    dynamic_cluster_indices=dynamic_indices,
    sc_cell_expressions=self.gat_model.sc_cell_expressions  # 如果启用动态模式
)
```

## 参数说明

- **use_dynamic_cluster_repr** (bool): 是否启用动态cluster表示，默认 `False`
- **k_cells_per_cluster** (int): 每个cluster选择多少个最近细胞，默认 `10`
- **sc_cell_embeddings** (array): 单细胞的VAE embedding，形状 `[n_cells, embedding_dim]`
- **sc_cell_expressions** (array): 单细胞的全基因表达，形状 `[n_cells, n_all_genes]` 
  - ⚠️ **必须是原始count矩阵（raw counts）**，不要做normalize或log1p
  - 只有VAE训练阶段使用normalize+log1p，其他阶段都用raw counts
- **sc_cell_labels** (array): 单细胞的cluster标签，形状 `[n_cells]`

## 优势

1. **更精细建模**: 不同spot可以使用不同的细胞组合表示同一个cluster
2. **自适应性**: 通过可学习的MLP自动调整细胞权重（边权重归一化为百分比）
3. **可解释性**: 可以追踪每个spot具体使用了哪些细胞及其百分比

## 技术细节

### 1. 边权重归一化
- Cell → Cluster的边权重通过softmax归一化，每个cluster的k个cell权重和为1（百分比矩阵）
- 这确保了cluster的动态表达是其成员细胞的加权平均（权重为百分比）

### 2. 原始Count使用
- **VAE阶段**: 输入做normalize+log1p（sc.pp.normalize_total + sc.pp.log1p）
- **GAT阶段**: 全部使用原始count（raw counts）
  - `sc_cell_expressions`: 原始count
  - `celltype_expressions_full`: 原始count（cluster平均）
  - Loss函数的缩放逻辑依赖原始count
  
### 3. 表达计算公式
```python
# 每个cluster的动态表达
cluster_expr = Σ_i(α_i × cell_i_raw_expr)  # α_i是百分比，和为1

# Spot的混合表达
spot_expr = Σ_c(w_c × cluster_c_dynamic_expr)  # w_c是GAT学到的权重

# 最终缩放
spot_expr_scaled = spot_expr × (spot_total_counts / mixed_basis_totals)
```

## 注意事项

1. 需要保存Stage 1的单细胞数据（embedding和表达）
2. 计算开销略高于静态模式
3. 需要足够的GPU内存存储单细胞数据

## 下一步

需要完成以下修改才能使用：

1. ✅ 修改 `HeterogeneousGATDeconvolution` 添加动态模式参数
2. ✅ 实现 `compute_dynamic_cluster_weights` 方法
3. ✅ 修改 `SpatialDeconvolutionLoss` 支持动态表达计算
4. ⏳ 修改 `stage2.py` 的 `build_gat_model` 传递新参数
5. ⏳ 修改 `stage2.py` 的训练循环传递动态权重
6. ⏳ 修改 `stage1.py` 保存单细胞embeddings和表达
7. ⏳ 修改 `deconv.py` 的API支持新参数

当前已完成模型层面的修改，需要继续完成训练流程的集成。
