# 双头边过滤实现方案

## 方案概述
实现"边存在性判别器 + 边强度回归器"的双头架构，专门识别和过滤假阳性边。

## 架构设计

### 1. 伪标签生成（Label 方案 1：结构腐蚀）
- **数据源**：`knn_mask` (KNN邻域图) 作为腐蚀图
- **标签逻辑**：
  ```
  is_important = 1 if edge exists in both original_graph AND knn_graph
               = 0 if edge only in original_graph (假阳性候选)
  ```
- **实现位置**：`calculate_lr_scores.py` - 在生成 LR 通讯边时添加 `is_important` 列

### 2. 数据流更新
- **edge_attr_cc 维度**：从 `[n_edges, 2]` 扩展到 `[n_edges, 3]`
  - `[:,0]`: `lr_score` (归一化的LR通讯得分)
  - `[:,1]`: `lr_id` (LR对编号)
  - `[:,2]`: `is_important` (二值标签: 1=真边, 0=假阳性候选)

### 3. 模型架构修改
在 `HeteroSTModel` 中添加：
```python
# 边存在性判别器 (existence head)
self.edge_exist_head = nn.Sequential(
    nn.Linear(gat_hidden_dims[-1], gat_hidden_dims[-1] // 2),
    nn.ReLU(),
    nn.Dropout(gat_dropout),
    nn.Linear(gat_hidden_dims[-1] // 2, 1)
)

# 边强度回归器 (strength head)
self.edge_rate_head = nn.Sequential(
    nn.Linear(gat_hidden_dims[-1], gat_hidden_dims[-1] // 2),
    nn.ReLU(),
    nn.Dropout(gat_dropout),
    nn.Linear(gat_hidden_dims[-1] // 2, 1)
)
```

### 4. 前向传播修改
在 `edge_attn_comm` 之后：
```python
# 获取边表示 (从 comm_repr 的源/目标节点特征)
src_repr = comm_repr[edge_index_cc[0]]
dst_repr = comm_repr[edge_index_cc[1]]
edge_repr = torch.cat([src_repr, dst_repr], dim=-1)  # [n_edges, hidden_dim]

# 双头预测
exist_logits = self.edge_exist_head(edge_repr).squeeze(-1)  # [n_edges]
rate_pred = torch.nn.functional.softplus(
    self.edge_rate_head(edge_repr).squeeze(-1)
)  # [n_edges], 确保非负
```

### 5. 损失函数重写
```python
# 提取标签
lr_scores = edge_attr_cc[:, 0]  # [n_edges]
exist_labels = edge_attr_cc[:, 2]  # [n_edges], 0/1

# 边存在性损失 (BCE)
loss_exist = F.binary_cross_entropy_with_logits(
    exist_logits, exist_labels, reduction='mean'
)

# 边强度回归损失 (MSE 或 Huber)
# 可选：只对 is_important=1 的边计算回归损失
mask_important = (exist_labels == 1)
if mask_important.sum() > 0:
    loss_rate = F.mse_loss(
        rate_pred[mask_important], 
        lr_scores[mask_important], 
        reduction='mean'
    )
else:
    loss_rate = 0.0

# 总损失
lambda_exist = 0.5
lambda_rate = 0.4
lambda_contrast = 0.1  # 可选，保留原有对比学习项

loss = lambda_exist * loss_exist + lambda_rate * loss_rate
```

### 6. 推理过滤策略
在 `evaluate_cell_communication` 中：
```python
# 计算边存在概率
p_exist = torch.sigmoid(exist_logits)  # [n_edges]

# 策略1: 阈值过滤
keep_mask = p_exist > 0.5

# 策略2: Top-K 过滤（推荐）
# 对每个源节点（或每个 spot），保留 top-k 边
for src_node in unique_sources:
    src_edges = (edge_index_cc[0] == src_node)
    k = min(5, src_edges.sum())  # 每个源节点最多保留5条边
    topk_indices = torch.topk(p_exist[src_edges], k).indices
    keep_mask[src_edges][topk_indices] = True

# 过滤边
filtered_edges = edge_index_cc[:, keep_mask]
filtered_scores = rate_pred[keep_mask]
filtered_p_exist = p_exist[keep_mask]
```

### 7. 输出文件更新
在 CSV 输出中添加列：
- `p_exist`: 边存在概率
- `rate_pred`: 预测的边强度
- `is_important_label`: 原始伪标签

## 命令行参数新增
```python
parser.add_argument('--lambda_exist', type=float, default=0.5,
                   help='边存在性损失权重')
parser.add_argument('--lambda_rate', type=float, default=0.4,
                   help='边强度回归损失权重')
parser.add_argument('--edge_topk', type=int, default=5,
                   help='每个源节点保留的最大边数 (default: 5)')
parser.add_argument('--edge_exist_threshold', type=float, default=0.5,
                   help='边存在性概率阈值 (default: 0.5)')
```

## 实现文件清单
1. ✅ `calculate_lr_scores.py` - 添加伪标签生成逻辑
2. ✅ `hetero_graph_builder.py` - 更新 `edge_attr_cc` 维度和加载逻辑
3. ✅ `hetero_model.py` - 添加双头架构
4. ✅ `train.py` - 重写损失函数（lines 791-895）
5. ✅ `evaluate.py` - 添加 top-k 过滤和清洁图输出

## 预期效果
- **训练阶段**：`loss_exist` 应快速下降至 0.3-0.5，表示模型学会区分真边/假边
- **验证阶段**：`p_exist` 分布应呈双峰（接近0和接近1）
- **推理阶段**：自动过滤 50%-70% 的低置信度边，保留高质量通讯边
