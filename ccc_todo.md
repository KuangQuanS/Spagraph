# CCC 模块待做清单（精简版）

> 目的：只补最必要、最直接支撑论文 claim 的分析。  
> 原则：**不把 degree correction 当创新点，不单独为它设计验证任务。**  
> 当前要证明的不是“发现了真实通讯机制”，而是：
>
> **attention-based prioritization 不是随机噪声，也不只是高表达/高频背景；它更倾向于筛出空间上更有组织、与病理区域更一致的候选通讯轴。**

---

## 本轮只做这 4 件事

1. **负对照：打乱空间坐标**  
   证明高 attention 依赖真实空间结构，不是随机噪声。

2. **空间一致性定量：Moran's I**  
   比较 top attention LR 对 vs top frequency LR 对，谁更空间聚集。

3. **Boundary / niche enrichment**  
   比较 top attention LR 对 vs top frequency LR 对，谁更富集在病理上有意义的区域。

4. **结果写法调整**  
   补文献支持，整体降调，不再声称“证明真实机制”。

---

# 0. 分析总设定（先固定，后面都按这个跑）

## 比较对象
对每个数据集，固定比较两组：
- **top attention LR pairs**
- **top frequency LR pairs**

## 推荐设置
- 每个数据集取 `top 20` 个 LR pairs
- 所有后续分析都基于同一版：
  - deconvolution 结果
  - LR 数据库
  - spot 坐标
  - neighbor graph / KNN 参数
  - candidate edges

## 数据集建议
优先做你文章里最适合讲边界/生态位的 3 个：
- **SCC**
- **DCIS**
- **ovarian cancer**

> 这轮不要扩太多数据集，不然工作量会膨胀。

---

# 1. 负对照：打乱空间坐标

## 1.1 目的
证明 attention 排名依赖真实组织空间，而不是表达背景或随机图结构。

## 1.2 核心思路
**不要随机生成新坐标。**  
正确做法是：

### spot-coordinate permutation
保留原始坐标集合不变，只随机打乱“哪个 spot 对应哪个坐标”。

原来：
- `spot_i -> (x_i, y_i)`

打乱后：
- `spot_i -> (x_perm(i), y_perm(i))`

这样做的好处：
- 保留组织整体几何范围
- 保留 spot 密度分布
- 保留采样尺度
- 只破坏“表达状态 <-> 空间位置”的真实对应关系

## 1.3 具体步骤
对每个数据集分别做：

1. 固定原始表达矩阵、细胞类型比例、LR 数据库不变。
2. 提取全部 spot 的原始坐标表。
3. 随机打乱坐标顺序，把打乱后的坐标重新赋给 spot。
4. 用打乱后的坐标重建空间邻接图 / candidate graph。
5. 重新计算：
   - frequency ranking
   - attention ranking
6. 记录 top 20 LR pairs 的后续指标（见下文）。
7. 重复 `100 次`。

## 1.4 要输出什么
每次 permutation 后，至少保存：
- top attention LR pairs 名单
- top frequency LR pairs 名单
- 这两组 pairs 的 Moran's I 均值
- 这两组 pairs 的 boundary enrichment 均值

## 1.5 你最终想看到什么
在真实坐标下：
- attention 的 Moran's I 更高
- attention 的 boundary enrichment 更高

在坐标打乱后：
- attention 的这些优势明显下降
- 原本很漂亮的空间热点变散

## 1.6 建议出图
### 图 1A：permutation null distribution
每个数据集做一个 panel：
- 横轴：`permuted` / `observed`
- 纵轴：Moran's I 或 boundary enrichment
- 灰色：100 次 permutation 的分布
- 红点：真实 attention
- 蓝点：真实 frequency

## 1.7 可直接写进 Methods 的描述
> To test whether prioritized communication patterns depended on authentic tissue geometry, we performed coordinate permutation controls. Specifically, the original coordinate set was preserved while spot identities were randomly reassigned to coordinates, thereby disrupting the coupling between molecular state and spatial position without altering tissue density or sampling geometry.

---

# 2. 空间一致性定量：Moran's I

## 2.1 目的
把“attention 看起来更集中”变成一个定量结论。

## 2.2 为什么只用 Moran's I
这轮先别堆太多指标。  
**Moran's I 一个主指标就够了。**

原因：
- 直观
- 常用
- 好解释
- 审稿人容易接受

## 2.3 先定义 pair 的 spot-level score
对每个 LR pair，在每个 spot 上定义一个总分数，用来表示这个 pair 在该 spot 的活跃程度。

推荐定义：

- 某个 spot 作为 sender 的该 LR 边权总和
- 加上该 spot 作为 receiver 的该 LR 边权总和

写成概念公式就是：

`pair_score(spot i) = 所有与 spot i 相连、且属于该 LR pair 的边权之和`

然后分别对：
- attention score
- frequency score

都构建一张 pair-specific spatial map。

## 2.4 具体步骤
对每个数据集：

1. 取 top 20 attention LR pairs。
2. 取 top 20 frequency LR pairs。
3. 对每个 pair 构建 spot-level score map。
4. 基于原始空间邻接图，计算该 pair 的 Moran's I。
5. 最终得到两组分布：
   - 20 个 top attention pairs 的 Moran's I
   - 20 个 top frequency pairs 的 Moran's I

## 2.5 统计比较
每个数据集单独比较：
- attention vs frequency

建议：
- `Mann–Whitney U test` 或 `Wilcoxon rank-sum test`

## 2.6 你最终想看到什么
- top attention LR pairs 的 Moran's I 整体高于 top frequency LR pairs
- 说明 attention 排名更倾向于选择空间上成团、成域、成边界的通讯对

## 2.7 建议出图
### 图 1B：Moran's I 分布图
每个数据集一个 panel：
- x 轴：attention / frequency
- y 轴：Moran's I
- 图型：箱线图或小提琴图

## 2.8 结果段可直接参考的写法
> Across multiple datasets, ligand–receptor pairs prioritized by attention exhibited higher spatial autocorrelation than those ranked by interaction frequency, indicating that attention-based prioritization preferentially captures communication programs organized into coherent tissue domains rather than globally abundant background interactions.

---

# 3. Boundary / niche enrichment

## 3.1 目的
证明 attention 排名前列的 LR pairs 更容易落在**病理上有意义的区域**，而不是全组织到处都有的背景信号。

## 3.2 最重要原则
**区域必须先定义，再看 CCC 结果。**  
不能根据 attention 热点反过来定义边界，不然会变成循环论证。

## 3.3 每个数据集建议怎么定义区域

---

## SCC
### 区域：tumor–stroma boundary

### 推荐定义
根据 deconvolution 结果：
- tumor-dominant spots：tumor/epithelial 占比高于阈值
- stromal-dominant spots：fibroblast/stroma 占比高于阈值

边界定义：
- 若某 tumor-dominant spot 至少有一个 stromal-dominant 邻居，则该 spot 属于 boundary
- 若某 stromal-dominant spot 至少有一个 tumor-dominant 邻居，则该 spot 也属于 boundary
- 最终 boundary = 两侧边界 spots 的并集

---

## DCIS
### 区域：myoepithelial-associated boundary

### 推荐定义
根据 deconvolution 或 marker：
- 先找 ACTA2+/KRT15+ 高的 myoepithelial spots
- 再找与 DCIS epithelial spots 邻接的这些 spots
- 这部分区域定义为 myoepithelial boundary / transition zone

> 这轮建议先不要硬做 immune-excluded gap，容易复杂化。

---

## Ovarian cancer
### 区域：tumor–stroma interface 或 fibroblast-rich immune-excluded zone

### 推荐定义（优先简单版本）
先做 tumor–stroma interface：
- tumor-rich spots
- fibroblast-rich spots
- 两者相邻区域定义为 interface

如果后面顺利，再加 fibroblast-rich immune-excluded zone：
- Fibro5 高
- immune 低
- 且靠近 tumor 区

---

## 3.4 enrichment 怎么算
对每个 LR pair，都已经有一张 spot-level score map。

定义区域 `R` 后，计算：

`enrichment = mean(score in R) / mean(score outside R)`

更推荐记录 log2 形式：

`log2_enrichment = log2( (mean in R + eps) / (mean out R + eps) )`

这样更稳，也更容易画图。

## 3.5 具体步骤
对每个数据集：

1. 先独立定义 boundary / niche mask。
2. 对 top 20 attention pairs 逐个计算 enrichment。
3. 对 top 20 frequency pairs 逐个计算 enrichment。
4. 比较两组 enrichment 分布。

## 3.6 你最终想看到什么
- top attention pairs 的 enrichment 明显更高
- 说明 attention 更倾向于把病理边界/关键生态位中的 LR 轴排到前面
- frequency 更容易给到全局高丰度但区域性不强的 pair

## 3.7 建议出图
### 图 1C：boundary enrichment 分布图
每个数据集一个 panel：
- x 轴：attention / frequency
- y 轴：log2 enrichment
- 图型：箱线图或小提琴图

## 3.8 结果段可直接参考的写法
> Compared with frequency-ranked interactions, attention-prioritized ligand–receptor pairs showed stronger enrichment within independently defined pathological niches, including tumor–stroma boundaries in SCC, myoepithelial-associated transition zones in DCIS, and stromal interfaces in ovarian cancer.

---

# 4. 文献支持 + 降调写法

## 4.1 这轮的写作目标
不是去写：
- “证明了真实通讯机制”
- “重建了病理演进过程”
- “揭示了决定性因果程序”

而是写成：
- 与已有知识一致
- 提出了候选通讯轴
- 支持一种空间上有组织的 signaling model

## 4.2 写法原则
对每个重点 LR pair，都按下面四步写：

### 第一步：先写现象
- 哪个 pair 在哪个区域富集
- attention 排名高于 frequency
- 空间上是否成团、成边界、成特定 niche

### 第二步：再写已知知识
补一句文献支持，例如：
- This is consistent with prior studies showing that ...
- This pathway has been implicated in ...

### 第三步：最后降调
不要写：
- proves
- demonstrates a mechanism
- reconstructs the trajectory

改写成：
- suggests
- supports the possibility that
- is consistent with
- nominates ... as a candidate signaling axis

## 4.3 建议替换表达

### 不建议
- revealed the mechanism
- reconstructed the pathological trajectory
- demonstrated that ... is driven by ...

### 建议改成
- suggested
- supported a model in which ...
- was consistent with prior reports
- nominated ... as a candidate signaling axis

## 4.4 可直接放进文中的总结句
> Collectively, these analyses support the use of attention as a topology-aware prioritization strategy that filters high-abundance background interactions and highlights spatially constrained candidate signaling axes.

或者更保守一点：

> Collectively, these results suggest that attention-based prioritization can help distinguish spatially organized candidate communication programs from globally abundant background interactions.

---

# 5. 最终交付物（这轮做完后至少应该有）

## 主图（建议作为一个总图）
### Panel A
真实坐标 vs permutation null distribution

### Panel B
attention vs frequency 的 Moran's I 对比

### Panel C
attention vs frequency 的 boundary enrichment 对比

## 补充表
每个数据集整理一个表，至少包含：
- LR pair 名称
- ranking type（attention / frequency）
- Moran's I
- boundary enrichment
- literature support（1 句话）

---

# 6. 明确不做的事（防止任务膨胀）

这轮**先不做**：
- 不把 degree correction 当创新点来证明
- 不扩展到太多外部 CCC 方法对比
- 不上太多空间统计指标（先不做 Geary's C / Local Moran's I）
- 不做太复杂的 immune gap 自动检测
- 不追求“真实 ground truth CCC”

---

# 7. 一句话版任务摘要（给别的 AI 看）

当前 CCC 模块补强的目标不是证明真实分子机制，而是证明 attention-based prioritization 具备以下性质：

1. **不是随机噪声**：通过坐标 permutation 负对照验证；
2. **更具空间组织性**：通过 top attention vs top frequency 的 Moran's I 比较验证；
3. **更贴近病理有意义区域**：通过 boundary / niche enrichment 验证；
4. **叙述方式降调**：以“candidate signaling axis / consistent with prior studies / suggests”取代机制性定论。

如果要继续扩展，优先顺序应当是：
- 先把这 4 件事做完；
- 再考虑是否加更多数据集或更多对照。

## 2026-03-12 implementation note
- First minimal implementation target: `GSE243275`
- Reason: this dataset already has `lr_communication.csv`, `*_composition.csv`, `*_spot_cell_expr.csv`, and a matching ST `.h5ad`
- `GSE144236` and `GSE211956/P3` can reuse the same workflow later, but they still need the Stage 2 `*_spot_cell_expr.csv` input for the same rerun/permutation path
