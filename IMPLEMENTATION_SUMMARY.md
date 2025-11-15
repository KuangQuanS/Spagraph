# 双头边过滤实现完成报告

## 实施总结
已完成"边存在性判别器 + 边强度回归器"的完整双头架构，用于系统性识别和去除图模型中的假阳性边。

---

## ✅ 完成的修改

### 1. 伪标签生成 (`calculate_lr_scores.py`)
**修改内容：**
- 基于 KNN mask（腐蚀图）生成二值伪标签 `is_important`
  - `is_important = 1`: 边同时存在于原图和KNN图（真边）
  - `is_important = 0`: 边仅存在于原图（假阳性候选）
- 更新CSV输出，新增 `is_important` 列
- 添加统计日志：真边/假阳性候选的数量和比例

**关键代码：**
```python
is_important = 1 if knn_mask[i, j] == 1 else 0
comm_event_records.append([..., score, is_important])
```

---

### 2. 数据流更新 (`hetero_graph_builder.py`)
**修改内容：**
- `lr_scores_dict` 改为存储 `(comm_score, is_important)` 元组
- `edge_attr_cc` 维度从 `[n_edges, 2]` 扩展到 `[n_edges, 3]`
  - 列0: `lr_score` (归一化LR通讯得分)
  - 列1: `lr_id` (LR对编号)
  - 列2: `is_important` (二值标签)
- 对聚合边取 `is_important` 的最大值（只要有一条真边就标记为真）

**关键代码：**
```python
comm_score, is_important = self.lr_scores_dict[key]
is_important_final = max(is_important_labels)
edge_attr_cc_list.append([total_lr_score, lr_id, is_important_final])
```

---

### 3. 模型架构扩展 (`hetero_model.py`)
**修改内容：**
- 新增 `edge_exist_head`（边存在性判别器）
  - 输入：源节点+目标节点的拼接表示 `[hidden_dim*2]`
  - 输出：边存在性logits `[1]`
- 新增 `edge_rate_head`（边强度回归器）
  - 输入：源节点+目标节点的拼接表示 `[hidden_dim*2]`
  - 输出：边强度预测 `[1]`（经过softplus确保非负）
- 前向传播返回 `exist_logits` 和 `rate_pred`

**关键代码：**
```python
self.edge_exist_head = nn.Sequential(...)
self.edge_rate_head = nn.Sequential(...)

edge_repr = torch.cat([src_repr, dst_repr], dim=-1)
exist_logits = self.edge_exist_head(edge_repr).squeeze(-1)
rate_pred = F.softplus(self.edge_rate_head(edge_repr).squeeze(-1))
```

---

### 4. 损失函数重写 (`train.py`, lines 791-895)
**修改内容：**
- **边存在性损失（BCE）**：
  ```python
  loss_exist = F.binary_cross_entropy_with_logits(
      exist_logits, exist_labels, reduction='mean'
  )
  ```
- **边强度回归损失（Smooth L1）**：
  ```python
  mask_important = (exist_labels == 1)
  loss_rate = F.smooth_l1_loss(
      rate_pred[mask_important], 
      lr_scores[mask_important], 
      reduction='mean'
  )
  ```
  - 注意：只在 `is_important=1` 的边上计算回归损失
- **总损失**：
  ```python
  loss = lambda_exist * loss_exist + lambda_rate * loss_rate
  ```
- 移除旧的负采样BCE逻辑，简化训练流程

**新增命令行参数：**
```bash
--lambda_exist 0.5       # 边存在性损失权重
--lambda_rate 0.4        # 边强度回归损失权重
--edge_topk 5           # 推理时每个源节点保留的最大边数
--edge_exist_threshold 0.5  # 边存在性概率阈值
```

---

### 5. 推理过滤策略 (`evaluate.py`)
**修改内容：**
- 提取双头预测：`p_exist = sigmoid(exist_logits)`, `rate_pred`
- **Top-K 过滤**：对每个源节点，按 `p_exist` 保留top-5边
  ```python
  for src_node in unique_sources:
      src_p_exist = p_exist[src_edges_mask]
      topk_indices = torch.topk(src_p_exist, k).indices
      keep_mask[src_global_indices[topk_indices]] = True
  ```
- 输出清洁图CSV：`lr_communication_edge_attention.csv`
  - 包含列：`lr_score`, `is_important_label`, `p_exist`, `rate_pred`, `attention_score`
- 日志统计：过滤前后边数、保留比例、`p_exist` 分布

---

## 📊 预期效果

### 训练阶段
- `loss_exist` 应快速下降至 0.3-0.5（BCE标准水平）
- `loss_rate` 应稳定在较低值（0.1-0.3，取决于数据归一化）
- 监控 `Exist` 和 `Rate` 损失的比例，确保两者平衡

### 验证阶段
- `p_exist` 分布应呈**双峰**（接近0和接近1）
  - 真边（is_important=1）的 `p_exist` 集中在 >0.7
  - 假阳性（is_important=0）的 `p_exist` 集中在 <0.3
- 如果分布不明显分离，调整 `lambda_exist` 和 `lambda_rate` 比例

### 推理阶段
- 自动过滤 50%-70% 的低置信度边
- 保留的边为高质量通讯边（`p_exist > 0.7`）
- CSV输出可用于下游可视化和分析

---

## 🚀 使用方法

### 训练命令示例
```bash
python train.py \
  --deconv_dir SC_MAP_ST/deconv_results/CID44971 \
  --st_h5ad database/Wu/CID44971/CID44971_ST.h5ad \
  --output_dir results/CID44971 \
  --epochs 100 \
  --lambda_exist 0.5 \
  --lambda_rate 0.4 \
  --edge_topk 5 \
  --batch_size 4 \
  --learning_rate 1e-4
```

### 关键参数调优建议
1. **如果假阳性过多**（`p_exist` 分布不清晰）：
   - 增大 `--lambda_exist` (e.g., 0.6-0.7)
   - 减小 `--lambda_rate` (e.g., 0.3)
   
2. **如果真边回归精度不足**：
   - 增大 `--lambda_rate` (e.g., 0.5-0.6)
   - 减小 `--lambda_exist` (e.g., 0.3-0.4)

3. **推理过滤强度**：
   - 减小 `--edge_topk` (e.g., 3) → 更激进的过滤
   - 增大 `--edge_topk` (e.g., 10) → 保留更多边

---

## 📁 输出文件

### 训练阶段
- `results/*/lr_scores.csv` - 包含 `is_important` 列的LR通讯得分
- `results/*/knn_mask.npz` - KNN邻域图（腐蚀图）
- `results/*/hetero_model_final.pth` - 训练好的双头模型

### 评估阶段
- `results/*/lr_communication_edge_attention.csv` - **清洁图边结果**（核心输出）
  - 列：`center_spot`, `source_cell`, `target_cell`, `lr_pair`, `lr_score`, `is_important_label`, **`p_exist`**, **`rate_pred`**, `attention_score`
- `results/*/lr_pair_statistics.csv` - LR对统计（按出现频率和注意力得分排序）

---

## 🔍 调试检查点

### 1. 检查伪标签生成
```bash
# 查看 lr_scores.csv 中 is_important 分布
python -c "import pandas as pd; df = pd.read_csv('results/*/lr_scores.csv'); print(df['is_important'].value_counts())"
```
预期输出：0和1的数量应该相对平衡（20%-80%之间）

### 2. 检查模型输出
```python
# 在训练日志中查看
grep "边存在性概率分布" results/*/training.log
```
预期输出：`mean=0.5-0.7`，`min` 和 `max` 接近0和1

### 3. 检查过滤效果
```bash
# 比较过滤前后边数
wc -l results/*/lr_scores.csv
wc -l results/*/lr_communication_edge_attention.csv
```
预期：过滤后应保留30%-50%的边

---

## ✅ 实现完整性验证

- [x] 伪标签生成（基于KNN mask）
- [x] 数据流更新（edge_attr_cc扩展到3列）
- [x] 双头架构（exist_head + rate_head）
- [x] 损失函数重写（BCE + Smooth L1）
- [x] 推理过滤（Top-K策略）
- [x] CSV输出（包含p_exist和rate_pred）
- [x] 命令行参数（lambda_exist, lambda_rate, edge_topk）
- [x] 日志统计（分布、过滤比例）

---

## 📚 相关文档

- **实现方案**：`DUAL_HEAD_EDGE_FILTERING.md` - 详细架构设计
- **训练日志**：`results/*/training.log` - 实时训练进度
- **模型检查点**：`results/*/hetero_model_final.pth` - 最终模型

---

## 🎯 下一步（可选）

1. **超参数网格搜索**：
   - 尝试不同的 `lambda_exist` 和 `lambda_rate` 组合
   - 使用验证集的 `p_exist` 分布清晰度作为评价指标

2. **替代伪标签方案**（如果当前效果不理想）：
   - 方案2：基于LR score百分位数（75%）的二值化
   - 方案3：双图一致性（原图vs腐蚀图预测差异）

3. **可视化分析**：
   - 绘制 `p_exist` 直方图（按is_important分组）
   - 散点图：`rate_pred` vs `lr_score`（验证回归精度）

4. **下游应用**：
   - 使用 `lr_communication_edge_attention.csv` 进行空间通讯可视化
   - 基于 `p_exist` 和 `rate_pred` 进行细胞通讯网络分析
