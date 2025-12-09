# 动态Cluster表示 - 正确实现方案

## 核心理解（正确版本）

### 第一阶段（VAE）- 预计算阶段
✅ **已经可以获得所有需要的信息**：
- `sc_cell_embeddings` [n_cells, embedding_dim]
- `spot_embeddings` [n_spots, embedding_dim]  
- `sc_cell_labels` [n_cells] - cluster标签
- `sc_cell_expressions` [n_cells, n_all_genes] - 原始count

✅ **可以预计算k-nearest cells**：
```python
# 在第一阶段结束时，对每个spot和每个cluster
for spot_idx in range(n_spots):
    for cluster_id in range(n_clusters):
        # 找到该cluster中距离spot最近的k个细胞（基于embedding余弦相似度）
        cluster_cells = sc_cells[sc_cell_labels == cluster_id]
        similarities = cosine_similarity(spot_embeddings[spot_idx], cluster_cells_embeddings)
        top_k_indices = argsort(similarities)[-k:]  # k个最近细胞的全局索引
        
        # 保存索引，供第二阶段使用
        knn_cell_indices[spot_idx, cluster_id] = top_k_indices
```

✅ **Cluster平均表达仍然有用**：
- 作为GAT的 `celltype_prototypes` 初始化
- 在不启用动态模式时，仍然使用cluster平均表达

### 第二阶段（GAT）- 只需小改动

✅ **GAT forward保持不变**：
```python
# GAT仍然计算每个spot对每个cluster的注意力权重
deconv_weights = gat_model(spot_embeddings, celltype_prototypes, spatial_coords)
# deconv_weights: [n_spots, n_cell_types]
```

✅ **只在Loss计算中使用动态表达**：
```python
# 静态模式（原来的方式）
mixed_expr = deconv_weights @ celltype_expr_mean  # [n_spots, n_genes]

# 动态模式（新方式）
for cluster_id in range(n_cell_types):
    # 获取预计算的k个最近细胞索引
    knn_indices = knn_cell_indices[:, cluster_id, :]  # [n_spots, k]
    
    # 获取这k个细胞的表达（原始count）
    cell_exprs = sc_cell_expressions[knn_indices]  # [n_spots, k, n_genes]
    
    # 计算这k个细胞的权重（可学习的MLP）
    cell_weights = mlp(spot_embeddings, cell_embeddings[knn_indices])  # [n_spots, k]
    cell_weights = softmax(cell_weights, dim=1)  # 归一化为百分比
    
    # Cluster的动态表达 = k个细胞的加权平均
    cluster_dynamic_expr = (cell_weights @ cell_exprs).squeeze(1)  # [n_spots, n_genes]
    
    # 乘以spot对该cluster的注意力权重
    mixed_expr += deconv_weights[:, cluster_id:cluster_id+1] * cluster_dynamic_expr
```

## 需要修改的地方

### 1. 第一阶段（spagraph/training/deconv.py）
```python
def run_deconv(...):
    # ... 现有VAE训练代码 ...
    
    # 新增：如果启用动态模式，预计算k-nearest cells
    if use_dynamic_cluster_repr:
        knn_cell_indices = precompute_knn_cells(
            spot_embeddings=st_data_emb,
            sc_cell_embeddings=sc_data_emb,
            sc_cell_labels=sc_labels,
            k_cells_per_cluster=k_cells_per_cluster
        )  # [n_spots, n_clusters, k]
        
        # 保存到artifacts
        stage1_artifacts.knn_cell_indices = knn_cell_indices
        stage1_artifacts.sc_cell_expressions_raw = sc_adata.X  # 原始count
```

### 2. 第二阶段（spagraph/models/deconv_model.py）

#### HeterogeneousGATDeconvolution 简化
```python
def __init__(self, ..., use_dynamic_cluster_repr=False, 
             sc_cell_expressions=None):
    # 只需要保存细胞表达（原始count）
    # 不需要保存embeddings和labels（已在stage1预计算好索引）
    if use_dynamic_cluster_repr:
        self.register_buffer('sc_cell_expressions', 
                           torch.FloatTensor(sc_cell_expressions))
```

#### forward 保持简单
```python
def forward(self, spot_embeddings, spatial_coords, celltype_prototypes, 
            knn_cell_indices=None):
    # GAT计算注意力权重（和原来一样）
    deconv_weights = self.gat_forward(...)  # [n_spots, n_cell_types]
    
    # 如果传入了knn_cell_indices，返回它（供Loss使用）
    return {
        'deconv_weights': deconv_weights,
        'knn_cell_indices': knn_cell_indices  # 直接传递
    }
```

#### Loss 中添加动态表达计算
```python
def forward(self, ..., knn_cell_indices=None):
    if knn_cell_indices is not None:
        # 使用动态表达
        mixed_expr = self.compute_dynamic_mixed_expression(
            attention_weights=deconv_weights,
            knn_cell_indices=knn_cell_indices,
            sc_cell_expressions=self.sc_cell_expressions
        )
    else:
        # 使用静态cluster平均表达
        mixed_expr = deconv_weights @ celltype_expressions_full
```

### 3. 训练循环（spagraph/models/stage2.py）
```python
def train_epoch_batched(...):
    # 从artifacts获取预计算的索引
    batch_knn_indices = knn_cell_indices[batch_spot_indices]  # 切片batch
    
    # Forward GAT
    gat_outputs = self.gat_model(
        spot_embeddings=batch_spot_emb,
        spatial_coords=batch_coords,
        celltype_prototypes=self.celltype_prototypes,
        knn_cell_indices=batch_knn_indices  # 传入预计算的索引
    )
    
    # 计算loss（Loss内部会用到knn_cell_indices）
    loss_dict = self.loss_fn(
        attention_weights=gat_outputs['deconv_weights'],
        ...,
        knn_cell_indices=batch_knn_indices  # 传给Loss
    )
```

## 优势

✅ **分离关注点**：
- 第一阶段：预计算k-nearest cells（基于embedding）
- 第二阶段：只学习这k个细胞的权重（MLP）

✅ **高效**：
- k-nearest搜索只做一次（第一阶段）
- 第二阶段只需学习权重，不需要重复计算相似度

✅ **灵活**：
- Cluster平均表达仍然可用（作为prototypes）
- 可以随时切换静态/动态模式

✅ **清晰**：
- GAT forward逻辑不变
- 所有动态逻辑集中在Loss中
- 预计算结果明确保存在artifacts中

## 总结

你的理解完全正确！核心改动：
1. **第一阶段末尾**：预计算每个spot对每个cluster的k个最近细胞索引
2. **第二阶段Loss**：使用这些索引获取细胞表达，学习权重，计算动态cluster表达
3. **其他部分**：基本不需要改动

这样实现更加清晰、高效、易维护！
