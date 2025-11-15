# 双头边过滤 - 快速开始指南

## 核心思想
用**两个独立的神经网络头**来预测每条边的：
1. **存在性**（这条边是真的还是假阳性？）→ `p_exist`
2. **强度**（如果是真边，通讯有多强？）→ `rate_pred`

然后根据 `p_exist` 自动过滤掉假阳性边。

---

## 一键运行

### 训练（推荐参数）
```bash
python train.py \
  --deconv_dir SC_MAP_ST/deconv_results/CID44971 \
  --st_h5ad database/Wu/CID44971/CID44971_ST.h5ad \
  --output_dir results/CID44971_dual_head \
  --epochs 100 \
  --lambda_exist 0.5 \
  --lambda_rate 0.4 \
  --edge_topk 5 \
  --batch_size 4
```

### 快速验证效果
```bash
# 1. 查看过滤统计
tail -100 results/*/training.log | grep "边存在性概率"

# 2. 对比边数变化
wc -l results/*/lr_scores.csv                           # 原始边数
wc -l results/*/lr_communication_edge_attention.csv     # 过滤后边数

# 3. 查看双头预测分布
python -c "
import pandas as pd
df = pd.read_csv('results/CID44971_dual_head/lr_communication_edge_attention.csv')
print('p_exist 分布:')
print(df['p_exist'].describe())
print('\nrate_pred vs lr_score 相关性:', df[['rate_pred', 'lr_score']].corr().iloc[0,1])
"
```

---

## 关键输出文件

| 文件 | 用途 | 关键列 |
|------|------|--------|
| `lr_scores.csv` | 原始LR通讯数据（含伪标签） | `is_important` |
| `lr_communication_edge_attention.csv` | **清洁图边**（过滤后） | `p_exist`, `rate_pred` |
| `lr_pair_statistics.csv` | LR对统计 | `occurrence_count`, `avg_attention_score` |

---

## 参数调优速查表

| 问题 | 调整参数 | 推荐值 |
|------|----------|--------|
| 假阳性太多（p_exist分布不清晰） | `--lambda_exist` ↑ | 0.6-0.7 |
| 回归精度不足 | `--lambda_rate` ↑ | 0.5-0.6 |
| 需要更激进的过滤 | `--edge_topk` ↓ | 3 |
| 保留更多边 | `--edge_topk` ↑ | 10 |

---

## 验证双头是否工作

### ✅ 良好的训练效果：
```
Exist: 0.35  Rate: 0.12  λ_E: 0.5  λ_R: 0.4
边存在性概率分布: min=0.02, mean=0.68, max=0.98
边存在性概率 > 0.5: 8234/12000 (68.6%)
过滤前边数: 12000
过滤后边数: 6000 (50.0% 保留)
```

### ❌ 需要调整的信号：
```
# p_exist 分布太平坦（没学到区分）
边存在性概率分布: min=0.42, mean=0.51, max=0.59
→ 增大 --lambda_exist

# 过滤太少或太多
过滤后边数: 11500 (95.8% 保留)  → edge_topk太大，或模型过拟合
过滤后边数: 1200 (10.0% 保留)   → edge_topk太小，或模型欠拟合
```

---

## 可视化（Jupyter Notebook）

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载结果
df = pd.read_csv('results/CID44971_dual_head/lr_communication_edge_attention.csv')

# 1. p_exist 分布（按伪标签分组）
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
df[df['is_important_label']==1]['p_exist'].hist(bins=50, alpha=0.7, label='True edges', ax=ax[0])
df[df['is_important_label']==0]['p_exist'].hist(bins=50, alpha=0.7, label='False positives', ax=ax[0])
ax[0].set_xlabel('p_exist')
ax[0].set_title('Edge Existence Probability Distribution')
ax[0].legend()

# 2. rate_pred vs lr_score
ax[1].scatter(df['lr_score'], df['rate_pred'], alpha=0.5, s=10)
ax[1].plot([0, 1], [0, 1], 'r--', label='y=x')
ax[1].set_xlabel('lr_score (ground truth)')
ax[1].set_ylabel('rate_pred (predicted)')
ax[1].set_title('Regression Accuracy')
ax[1].legend()

plt.tight_layout()
plt.savefig('dual_head_analysis.png', dpi=150)
plt.show()
```

---

## 常见问题

### Q: 为什么p_exist和attention_score不一样？
A: 
- `attention_score`：GAT注意力机制的输出（训练时的中间变量）
- `p_exist`：专门的分类头输出（判断边是否是假阳性）
- 它们互补：attention关注"相对重要性"，p_exist关注"绝对存在性"

### Q: 为什么只在is_important=1的边上计算rate回归损失？
A: 因为假阳性边（is_important=0）本身就不应该存在，让模型学习它们的"强度"是没有意义的。我们希望模型专注于优化真边的强度预测。

### Q: Top-K过滤会不会丢失重要信息？
A: Top-K是per-source过滤（每个源节点保留K条边），不是全局过滤。这确保了每个细胞类型都能保留其最重要的几个通讯伙伴，而不是只保留全局最强的边。

---

## 文件修改清单

如果需要手动回滚或检查更改：

1. `calculate_lr_scores.py` - 添加is_important伪标签
2. `hetero_graph_builder.py` - edge_attr_cc扩展到3列
3. `hetero_model.py` - 添加双头架构
4. `train.py` - 重写损失函数（lines 791-895）+ 新增命令行参数
5. `evaluate.py` - 添加Top-K过滤 + 保存清洁图CSV

所有改动都已完成，可以直接运行！
