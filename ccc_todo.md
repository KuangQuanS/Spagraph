# CCC 模块待做清单（收束版）

> 目的：只补最必要、最直接支撑论文 claim 的分析。
> 原则：不把 degree correction 当创新点，不为它单独设计验证。
> 当前要证明的不是“发现了真实通讯机制”，而是：
>
> **attention-based prioritization 不是随机噪声，也不只是高表达/高频背景；它筛出的候选通讯轴更 cell-type-specific，注意力分数与表达量解耦，而 frequency 排名更倾向选出全局高丰度的管家交互。**

## 核心叙事

GSE243275 现有结果已经表明：

- frequency top-20 的全局 Moran's I 显著高于 attention top-20
- 这并不等于 attention 失败
- 更合理的解释是：frequency 更偏向全局连续、广泛存在的 housekeeping-like pair

因此正文叙事应收束为：

1. Frequency 排名选出全局高丰度、低特异性的管家交互
2. Attention 排名筛出的 pair 更 cell-type-specific
3. Attention 分数与表达量解耦，说明模型学到的不是简单 abundance
4. 空间图作为定性证据，说明 attention 会把原始广泛分布收敛为更局部的病理相关 niche

参考写法：

> Frequency-ranked pairs show high global spatial autocorrelation, but this largely reflects ubiquitous housekeeping interactions. In contrast, attention-ranked pairs exhibit stronger cell-type specificity and their attention scores are decorrelated from expression abundance, suggesting that the model prioritizes spatially contextual candidate communication programs beyond simple expression level.

---

## 本轮只做这 4 件事

1. Moran's I
   作为“frequency 更 global / housekeeping-like”的量化参照。

2. Cell-type specificity
   作为 attention 更 cell-type-specific 的硬证据。

3. Expression vs Attention 解耦定量
   证明 attention 不是简单抄表达。

4. 空间图定性展示
   作为“attention 将原始广泛分布收敛为局部病理相关 niche”的图像证据。

---

# 0. 分析总设定

## 比较对象
每个数据集固定比较两组：

- top attention LR pairs
- top frequency LR pairs

## 推荐设置

- 每个数据集取 top 20 个 LR pairs
- 分析统一基于同一版 observed Stage 3 rerun
- Moran / specificity / expression-attention 都使用同一份 `lr_communication.csv`

## 数据集优先级

优先做你论文里最适合讲故事的 3 组：

- SCC
- DCIS
- ovarian cancer

本轮先以 `GSE243275` 跑通最小实现。

---

# 1. Moran's I

## 1.1 目的
定量衡量 LR pair 活跃模式的全局空间自相关。

## 1.2 在本轮叙事中的角色
Moran's I 不再是 attention 的胜负判据，而是一个参照指标。

如果 frequency 更高，合理解释是：

- frequency pairs 更像全局连续分布的 housekeeping-like interaction
- 它们在组织中“广泛且平滑”
- 但不一定更有信息量

## 1.3 pair 的 spot-level score
每个 LR pair 在每个 spot 上的得分定义为：

- 该 spot 作为 sender 的 `original_lr_score` 总和
- 加上该 spot 作为 receiver 的 `original_lr_score` 总和

即：

`pair_score(spot i) = sum(original_lr_score of all edges touching spot i for this LR pair)`

## 1.4 具体步骤

1. 取 top attention LR pairs
2. 取 top frequency LR pairs
3. 为每个 pair 构建 spot-level score map
4. 基于原始空间邻接图计算 Moran's I
5. 比较两组 Moran's I 分布

## 1.5 统计比较

- attention vs frequency
- Mann-Whitney U test

## 1.6 预期解释

- 若 frequency > attention，不算坏结果
- 这说明 frequency 更偏向全局广泛分布的背景交互
- 后续由 specificity 和 expression-attention 解耦补足 attention 的价值

## 1.7 建议出图

### Panel A

- x 轴：attention / frequency
- y 轴：Moran's I
- 图型：箱线图或小提琴图

---

# 2. Cell-type Specificity

## 2.1 目的
证明 attention 排名前列的 LR pairs 涉及更少、更特异的 sender-receiver cell type 组合。

## 2.2 核心逻辑

- frequency 高的 pair 往往在很多 cell type 组合间无差别活跃
- attention 高的 pair 如果更 restricted，就说明模型过滤掉了背景交互

## 2.3 具体步骤

对每个 LR pair 计算：

- Shannon entropy of cell-type pair distribution
- unique cell-type pair count

再比较：

- top attention pairs
- top frequency pairs

## 2.4 你最想看到什么

- attention 的 entropy 更低
- attention 的 unique pair count 更低
- 差异显著

## 2.5 建议出图

### Panel B

- x 轴：attention / frequency
- y 轴：cell-type entropy 或 unique cell-type pair count
- 图型：箱线图或小提琴图

## 2.6 推荐写法

> Attention-prioritized interactions involved fewer sender-receiver cell-type combinations than frequency-ranked pairs, indicating higher cell-type specificity. In contrast, frequency-ranked pairs were dominated by broadly active interactions present across many cell-type combinations.

---

# 3. Expression vs Attention 解耦

## 3.1 目的
证明 attention score 并不是简单复现 LR pair 的表达丰度。

## 3.2 核心逻辑

如果 attention 只是在学 abundance，那么：

- `attention_mean` 应与 `original_lr_sum` 强正相关

如果两者弱相关，则说明：

- attention 学到的是超越表达量的空间上下文信息

## 3.3 主图设置

- X 轴：`log10(original_lr_sum + 1)`
- Y 轴：`attention_mean`
- 点：每个 LR pair 一个点
- 颜色：
  - top attention = 红色
  - top frequency = 蓝色
  - 其他 = 灰色

## 3.4 统计

- 计算 Spearman rho
- 输出 p-value

## 3.5 你最想看到什么

- 相关性较弱
- top attention pairs 更容易位于“中低表达但高注意力”的区域
- top frequency pairs 更偏“高表达但注意力不突出”

## 3.6 建议出图

### Panel C

- x 轴：`log10(original_lr_sum + 1)`
- y 轴：`attention_mean`
- 标注 Spearman rho 和 p-value
- 可选标注 attention top pairs 的名字

## 3.7 推荐写法

> Attention scores were only weakly correlated with raw interaction abundance, indicating that the graph attention mechanism does not simply amplify highly expressed ligand-receptor pairs. Instead, attention prioritization appears to capture spatially contextual information beyond expression level alone.

---

# 4. 空间图（定性证据）

## 4.1 角色
空间图不承担主定量结论，而是作为定性证据支撑下面这句话：

**attention 会把原始较广泛的分布收敛为更局部、更有结构、更像病理相关 niche 的一部分。**

## 4.2 图中要表达什么

- 上排：`original_lr_score`
- 下排：`attention_score`
- 左右列分别选 attention 代表性 pair 和 frequency 代表性 pair

## 4.3 图的解读方式

- 若某 pair 原始得分分布较广，但 attention 后只保留少数局部区域
  - 说明模型在做 spatial refinement
- 若某 pair 原始得分高、attention 也高
  - 说明它在 abundance 和 topology 两个维度都重要

---

# 5. 结果写法（统一降调）

## 5.1 这轮不要写

- proved
- demonstrated the mechanism
- reconstructed the trajectory

## 5.2 这轮推荐写

- suggested
- supported a model in which ...
- was consistent with prior reports
- nominated ... as a candidate signaling axis

## 5.3 可直接放文中的总结句

> Collectively, these analyses support the use of attention as a topology-aware prioritization strategy that filters high-abundance background interactions and highlights more cell-type-specific candidate signaling axes.

更保守一点也可以写：

> Collectively, these results suggest that attention-based prioritization can help distinguish spatially contextual candidate communication programs from globally abundant background interactions.

---

# 6. 最终交付物

## 主图

- Panel A：attention vs frequency 的 Moran's I
- Panel B：attention vs frequency 的 cell-type specificity
- Panel C：Expression vs Attention 解耦散点图

## 定性图

- attention / frequency 代表性 LR pair 的空间图
- 每个 pair 同时展示 `original_lr_score` 和 `attention_score`

## 补充表

每个数据集至少输出：

- LR pair 名称
- ranking type（attention / frequency）
- Moran's I
- cell-type entropy
- unique cell-type pair count
- occurrence_count
- original_lr_sum
- attention_mean

---

# 7. 明确不做的事

这轮先不做：

- degree correction 单独验证
- 过多外部 CCC 方法对比
- Geary's C / Local Moran's I 等额外空间统计
- 复杂 immune gap 自动检测
- curated LR list enrichment
- coordinate permutation
- boundary enrichment 主分析

---

# 8. 备选方案（审稿人要求时再做）

如果审稿人明确要求负对照，优先考虑：

- 随机抽样 ranking 对照

即：

- 不打乱坐标
- 不重跑 Stage 3
- 从候选 LR pairs 中随机抽 K 个，重复多次
- 比较 attention top-K 是否显著偏离随机

注意：这一项**不属于当前主实现**。

---

## 2026-03-18 implementation note

- 当前最小实现目标：`GSE243275`
- 主脚本只保留 observed-only rerun
- 主定量结果固定为：
  - Moran's I
  - cell-type specificity
  - Expression vs Attention decoupling
