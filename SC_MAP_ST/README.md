# ST_Graduation_Project

## 项目结构

```
ST_Graduation_Project/
├── SC_MAP_ST/                    # 主要项目目录
│   ├── model.py                  # 核心模型定义 (VAE, GAT, 损失函数)
│   ├── stage1.py                 # Stage 1: VAE训练 (SC-ST整合)
│   ├── stage2.py                 # Stage 2: GAT解卷积 (空间细胞类型预测)
│   ├── cell_composition_visualization.ipynb  # 细胞组成可视化
│   ├── run.ipynb                 # 训练脚本和参数配置
│   ├── requirements.txt          # 项目依赖
│   ├── stage1_results/           # Stage 1训练结果
│   └── stage2_results/           # Stage 2训练结果
├── STEncoder/                    # ST编码器相关代码
├── STimage-1K4M/                 # ST图像数据
├── database/                     # 数据集存储
├── checkpoint/                   # 模型检查点
├── results/                      # 实验结果
├── visualization_results/        # 可视化结果
├── notebook/                     # Jupyter notebooks
├── requirements.txt              # 全局依赖
└── *.py                         # 工具脚本和数据处理
```

## 核心算法

### Stage 1: 多模态VAE整合

#### VAE架构设计

**编码器 (VAEEncoder)**:
- 输入: 基因表达向量 `[n_cells/genes, n_features]`
- 隐藏层: MLP + LayerNorm + ReLU + Dropout
- 输出: 均值向量 `μ` 和对数方差向量 `log σ²`
- LayerNorm确保多模态数据兼容性

**解码器 (VAEDecoder)**:
- 支持两种输出模式:
  - MSE模式: 单输出层重建连续表达值
  - ZINB模式: 三输出层 (mean, dispersion, dropout probability)

**VAE完整流程**:
```
输入基因表达 → 编码器 → (μ, log σ²) → 重参数化采样 → 解码器 → 重建表达
```

#### 损失函数详解

**ZINB负对数似然损失**:
```
ZINB(x; μ, θ, π) = π·δ₀(x) + (1-π)·NB(x; μ, θ)
L_zinb = -∑ log ZINB(x_i; μ_i, θ_i, π_i)
```

**KL散度正则化**:
```
D_KL(q(z|x)||p(z)) = -0.5 ∑ (1 + log σ² - μ² - σ²)
```

**MMD模态对齐损失**:
- RBF核函数: `K(x,y) = exp(-γ||x-y||²)`
- 中位数启发式确定γ: `γ = 1/(2·median²)`
- MMD² = E[K(x,x')] + E[K(y,y')] - 2E[K(x,y)]

**总损失**:
```
L_total = L_reconstruction + β·L_KL + λ_mmd·L_MMD
```

#### 训练策略
- 重参数化技巧: `z = μ + σ·ε, ε~N(0,I)`
- 批处理训练，支持SC和ST数据的联合学习
- 动态β调度防止KL消失问题

### Stage 2: 异构GAT解卷积

#### 异构图构建

**节点类型**:
- Spot节点: VAE编码的embedding投影到GAT空间
- CellType节点: 可学习的原型向量

**边构建策略**:

**空间边 (Spot-Spot)**:
- 基于欧几里得距离的KNN
- 权重计算: `w = exp(-d/σ)` (σ为距离标准差)
- 双向边确保空间对称性

**语义边 (Spot-CellType)**:
- 基于余弦相似度的KNN选择
- 每个Spot连接最相似的k个CellType
- 相似度: `sim = cos(spot_emb, celltype_emb)`

**异构图结构**:
```
节点: [Spot_0, Spot_1, ..., Spot_N, CellType_0, ..., CellType_M]
边: 空间邻接 + 语义相似度
```

#### GAT架构设计

**多层注意力机制**:
```python
# 第一层: 异构输入处理
GATConv(in_channels=hidden_dim, out_channels=hidden_dim//heads, heads=heads, concat=True)

# 中间层: 特征变换
GATConv(in_channels=hidden_dim, out_channels=hidden_dim//heads, heads=heads, concat=True)

# 输出层: 最终表示
GATConv(in_channels=hidden_dim, out_channels=hidden_dim, heads=1, concat=False)
```

**注意力权重计算**:
- 节点对特征拼接: `[spot_feat, celltype_feat]`
- MLP注意力评分: `score = MLP([spot||celltype])`
- 稀疏掩码: 只保留图中连接的节点对
- Softmax归一化: `weights = softmax(scores · mask)`

#### 解卷积权重计算

**向量化解卷积**:
```python
# 扩展维度进行批处理
spot_expanded = spot_features.unsqueeze(1)          # [n_spots, 1, hidden_dim]
cell_expanded = celltype_features.unsqueeze(0)      # [1, n_celltypes, hidden_dim]

# 拼接特征
combined = torch.cat([spot_expanded, cell_expanded], dim=-1)  # [n_spots, n_celltypes, 2*hidden_dim]

# 计算注意力分数
attention_scores = attention_mlp(combined).squeeze(-1)  # [n_spots, n_celltypes]

# 应用稀疏掩码
attention_scores[~sparse_mask] = -inf

# 归一化得到解卷积权重
deconv_weights = softmax(attention_scores, dim=1)
```

**细胞组成重建**:
```
spot_expression = deconv_weights @ celltype_expression
```

### 多损失函数优化

#### 皮尔逊相关性损失
```python
# 中心化
pred_centered = pred - pred.mean(dim=1, keepdim=True)
target_centered = target - target.mean(dim=1, keepdim=True)

# 皮尔逊系数
numerator = (pred_centered * target_centered).sum(dim=1)
denominator = sqrt((pred_centered²).sum(dim=1) * (target_centered²).sum(dim=1))
pearson_corr = numerator / denominator

L_pearson = 1.0 - pearson_corr.mean()
```

#### MSE重建损失
```
L_mse = ||reconstructed_spot - true_spot_expression||²
```

#### 余弦相似度损失
```
cos_sim = cos(reconstructed_spot, true_spot_expression)
L_cosine = 1.0 - cos_sim.mean()
```

#### 模态对齐损失 (Embedding层面)
```
recon_embedding = deconv_weights @ celltype_embedding
cos_sim_align = cos(spot_embedding, recon_embedding)
L_align = 1.0 - cos_sim_align.mean()
```

#### 权重正则化
```
L_reg = ||attention_weights.sum(dim=1) - 1||²
```

#### 稀疏性正则化
```
L_sparse = -attention_weights · log(attention_weights + ε)
```

#### 总损失
```
L_total = λ_pearson·L_pearson + λ_mse·L_mse + λ_cosine·L_cosine +
          λ_align·L_align + λ_reg·L_reg + λ_sparse·L_sparse
```

## 配置参数

### Stage 1参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_epochs` | 50 | 训练轮数 |
| `latent_dim` | 128 | 潜在空间维度 |
| `loss_type` | 'zinb' | 损失类型 ('mse' 或 'zinb') |
| `beta` | 0.1 | KL散度权重 |
| `lambda_mmd` | 1.0 | MMD模态对齐权重 |
| `top_n_per_type` | 200 | 每个簇的标记基因数 |
| `resolution` | 0.5 | Leiden聚类分辨率 |
| `precomputed_marker_file` | None | 预计算标记基因文件路径 (可选，用于跳过Lasso筛选) |

### Stage 2参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `gat_hidden_dim` | 256 | GAT隐藏层维度 |
| `gat_layers` | 6 | GAT层数 |
| `gat_heads` | 4 | 注意力头数 |
| `k_spatial` | 10 | 空间邻域大小 |
| `k_celltype` | 10 | 细胞类型邻域大小 |
| `loss_lambda_pearson` | 1.0 | 皮尔逊相关性损失权重 |
| `loss_lambda_mse` | 1.0 | MSE重建损失权重 |
| `loss_lambda_cosine` | 1.0 | 余弦相似度损失权重 |
| `loss_lambda_align` | 0.5 | 模态对齐损失权重 |

## 使用预计算的Marker Genes

为了节省时间并跳过耗时的Lasso回归筛选过程，你可以使用之前训练中生成的marker genes文件：

### 步骤

1. **首次训练** (包含Lasso筛选):
```bash
python stage1.py \
    --sc_file "path/to/sc_data.h5ad" \
    --st_file "path/to/st_data.h5ad" \
    --output_dir ./stage1_results/dataset1 \
    --top_n_per_type 200 \
    --latent_dim 256 \
    --loss_type zinb \
    --beta 0.1 \
    --lambda_mmd 1.0
```

2. **使用预计算的marker genes** (跳过Lasso筛选):
```bash
python stage1.py \
    --sc_file "path/to/sc_data.h5ad" \
    --st_file "path/to/st_data.h5ad" \
    --precomputed_marker_file "./stage1_results/dataset1/final_genes.txt" \
    --output_dir ./stage1_results/dataset1_fast \
    --latent_dim 256 \
    --loss_type zinb \
    --beta 0.1 \
    --lambda_mmd 1.0
```

### 优势

- **时间节省**: 跳过Lasso回归计算，通常可节省30-50%的训练时间
- **一致性**: 使用相同的marker genes集合确保结果可重现
- **灵活性**: 可在不同参数配置下重复使用相同的marker genes

### 注意事项

- 确保预计算的marker genes文件 (`final_genes.txt`) 存在于指定的路径
- 预计算的marker genes应该与当前数据集的基因名匹配
- 如果使用不同的数据集，需要重新计算marker genes

### 输出文件

#### Stage 1输出
- `final_vae.pth`: 训练好的VAE模型
- `sc_adata_clustered.h5ad`: 聚类后的单细胞数据
- `marker_genes.txt`: 每个簇的标记基因
- 训练曲线和损失图表

#### Stage 2输出
- `final_gat_model.pth`: 训练好的GAT模型
- `*_cell_composition.csv`: 细胞类型组成矩阵
- `*_reconstructed_*.csv`: 重建的基因表达矩阵
- `*_celltype_cluster_mapping.txt`: 簇到细胞类型的映射
- 训练曲线和可视化结果

### 评估指标
- **重建精度**: MSE, 余弦相似度, 皮尔逊相关性
- **聚类质量**: ARI, NMI, 轮廓系数
- **空间一致性**: 空间自相关分析
- **生物学解释性**: 标记基因富集分析

## 故障排除

### 常见问题

1. **内存不足**
   - 减小 `batch_size`
   - 使用更小的 `latent_dim`
   - 增加 `k_spatial` 和 `k_celltype` 的步长

2. **训练不收敛**
   - 调整学习率 `lr`
   - 修改损失权重
   - 增加训练轮数 `n_epochs`

3. **GPU内存不足**
   - 减小 `gat_hidden_dim`
   - 使用CPU训练: `--device cpu`

4. **数据格式错误**
   - 确保 `.h5ad` 文件包含空间坐标
   - 检查基因名是否匹配
   - 验证细胞类型标注格式

### 调试技巧

```python
# 检查数据完整性
print(f"SC data shape: {sc_adata.shape}")
print(f"ST data shape: {st_adata.shape}")
print(f"Common genes: {len(common_genes)}")
print(f"Spatial coords shape: {st_adata.obsm['spatial'].shape}")

# 验证模型输出
print(f"VAE latent dim: {vae.latent_dim}")
print(f"GAT output shape: {gat_output.shape}")
```