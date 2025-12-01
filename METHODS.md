# 方法

## 整体框架概述

本课题针对单细胞 RNA 测序（single-cell RNA‑seq, SC）与空间转录组（spatial transcriptomics, ST）联合分析，构建了一个三阶段的统一框架：  

1. **阶段一：SC–ST 共享表示学习与参考构建**  
   - 在共享潜在空间中联合建模 SC 与 ST 的表达分布；  
   - 基于单细胞聚类与标记基因构建细胞类型（或簇）级别的参考表达谱。  
2. **阶段二：基于异构图注意力网络的空间解卷积**  
   - 以阶段一得到的参考表达谱作为“字典”，在 ST 空间上构建 spot 与细胞类型的异构图；  
   - 使用图注意力网络预测每个空间点的细胞组成，并在计数空间重建基因表达。  
3. **阶段三：基于配体–受体和异构图的细胞通讯建模**  
   - 结合反卷积结果与配体–受体数据库，构建 spot–细胞类型–配体/受体的通信图；  
   - 使用基于边特征的注意力网络学习通信边的重要性，得到细胞–细胞及配体–受体对的通讯强度，并在空间上可视化。  

整体流程自下而上以基因表达为基础，先通过 VAE/GAT 建立“细胞–空间点–细胞通讯”的三层图结构，再通过一系列重构与正则损失进行端到端优化，最后使用统一的评估框架对反卷积和通讯结果进行定量分析。

---

## 数据与预处理

### 数据类型

- **单细胞数据（SC）**：h5ad 格式的单细胞表达矩阵及细胞注释信息。  
- **空间转录组数据（ST）**：h5ad 格式的空间表达矩阵及 spot 坐标（`obsm["spatial"]`），部分数据集另含组织切片图像。  
- **配体–受体数据库**：整理自文献与 CellChat 等资源的配体–受体对列表，保存在 `ligand_receptor_labeled.csv`。  

### 基因对齐与表达变换

1. 在 SC 和 ST 之间按基因名取交集，只保留共同基因。  
2. 对用于 VAE 与 GAT 的表达特征，统一采用：
   - 先按细胞／spot 进行文库大小归一化（`scanpy.pp.normalize_total(target_sum=1e4)`）；  
   - 再进行 `log1p` 变换，得到 log‑归一化表达矩阵。  
3. 为 Stage 2 和通讯分析保留原始 UMI 计数，用于在计数空间上重建表达及计算通信得分。

---

## 阶段一：SC–ST 共享潜在空间与参考构建（`stage1.py`, `stage1_utils.py`, `deconv_model.py`）

### 1. 单细胞聚类与标记基因筛选

在 `stage1_utils.compute_clusters_and_marker_genes` 中对单细胞数据执行以下步骤：

1. **预处理与聚类**  
   - 对 SC 数据进行 `normalize_total` + `log1p`，筛选高度变异基因（HVG）；  
   - 基于 HVG 做 PCA、邻接图构建，并使用 Leiden 算法进行聚类。  
2. **小簇过滤**  
   - 统计每个簇包含的细胞数，对细胞数 **少于 2** 的簇直接剔除；相应细胞从数据中移除并更新聚类标签。  
3. **差异表达分析与候选标记基因**  
   - 在完整基因集合上再次 `normalize_total` + `log1p`；  
   - 使用 `scanpy.tl.rank_genes_groups`（Wilcoxon 检验）计算每个簇的差异表达基因；  
   - 以校正 p 值、统计量和 log fold change 为标准，筛选显著上调基因，取最多 `top_n_per_type` 个作为候选。  
4. **二次筛选与空簇删除**  
   - 在候选基因上根据配置选择三种策略之一：
     - **L1 逻辑回归（默认）**：以“该簇 vs 其他细胞”为标签，使用 L1 正则逻辑回归筛除系数接近 0 的基因；  
     - **方差筛选**：按全局表达方差排序；  
     - **相关性筛选**：按与簇指示变量的相关系数绝对值排序。  
   - 对于没有任何标记基因的簇，直接删除该簇及其细胞。  

最终得到：  
（1）每个细胞的聚类标签；（2）全局标记基因集合；（3）仅包含有效细胞的单细胞 AnnData。

### 2. VAE 模型与训练数据构建

1. **训练特征**  
   - 从过滤后的单细胞数据中提取所有保留细胞在标记基因上的 `log1p` 归一化表达；  
   - 对 ST 数据做相同预处理，提取在同一标记基因集合上的表达；  
   - 将 SC 与 ST 样本按行拼接，构成训练矩阵 `X`，并构建模态标签向量 `m`（SC=0, ST=1）。  
2. **训练/验证划分**  
   - 对单细胞样本按细胞随机划分训练集与验证集；  
   - 空间样本全部并入训练集，并在验证集上同步监控损失。  

### 3. 双解码器 VAE 结构

VAE 结构在 `deconv_model.DualDecoderVAE` 中实现：

- **编码器**：多层线性层 + LayerNorm + ReLU + Dropout，将输入维度 `G` 映射到潜在维度 `d_z`（默认 128），输出均值 `μ` 和对数方差 `log σ²`。  
- **重参数化**：使用标准正态噪声生成潜在向量 `z`。  
- **解码器**：为 SC 和 ST 分别设计两个对称解码器（结构相同，权重不同），从 `z` 预测标记基因表达。  
- **输出头**：使用线性层输出与输入同维度的向量，并配合均方误差（MSE）重构损失；当前实验中关闭了 ZINB 头，统一使用 `output_type='mse'`。  

### 4. Stage 1 损失函数（MSE + KL）

对第 `i` 个样本（细胞或 spot），记输入为 `x_i`，重建为 `\hat{x}_i`，后验分布为 `q(z_i|x_i)`，先验 `p(z)=N(0,I)`。总损失为：

\[
\mathcal{L}_{\text{VAE}}
= \frac{1}{N}\sum_{i=1}^{N} \Big(
  \|x_i - \hat{x}_i\|_2^2
  + \beta\, \mathrm{KL}\big(q(z_i|x_i)\,\|\,p(z)\big)
\Big)
\]

其中 `β` 为 KL 权重（命令行默认 0.1）。MMD 对齐项虽已在代码中实现，但在最终实验中将其权重 `λ_{\text{MMD}}` 设为 0，不额外约束 SC/ST 潜在分布，以避免过强对齐带来的负面影响。

训练使用 Adam 优化器（学习率 5×10^-4），配合 `ReduceLROnPlateau` 调度及基于验证集损失的早停策略。

### 5. 聚类原型与参考表达谱

Stage 1 训练完成后，利用 VAE 编码器和原始计数矩阵构建参考：

1. **潜在空间聚类原型**  
   - 使用训练好的编码器对所有保留单细胞的标记基因表达编码，得到潜在向量 `z_i`；  
   - 对每个簇求其所有细胞 `z_i` 的平均，作为该簇的潜在原型向量 `p_c`。  
2. **原始计数空间的参考表达**  
   - 从保存的原始计数矩阵中提取每个细胞的全基因计数；  
   - 对每个簇，在原始计数空间上按配置的聚合策略（均值、median 或加权平均）得到簇级别的参考表达谱：
     - 标记基因子集（用于可视化与 Stage 2 对齐）；  
     - 全基因集合（用于 Stage 2 中的 spot 表达重建）。  
3. **簇–细胞类型映射**  
   - 若 SC 注释中包含 `cell_type` 或 `celltype` 列，则统计每个聚类中该列的主导细胞类型，将簇 ID 映射到细胞类型名称，用于后续解卷积与通讯分析。

---

## 阶段二：基于异构图的空间解卷积（`stage2.py`, `deconv_model.py`）

Stage 2 的目标是利用 Stage 1 的簇原型和参考表达谱，对 ST 数据进行空间解卷积，预测每个 spot 的细胞类型组成，并在原始计数空间重建表达。

### 1. ST 嵌入与 spot 总计数

1. 对 ST 数据从原始计数矩阵出发，保存全基因计数 `X^{\text{raw}}`；  
2. 同时在标记基因集合上复制 Stage 1 的预处理：`normalize_total` + `log1p`，得到用于编码器输入的特征矩阵；  
3. 使用 Stage 1 训练好的编码器对每个 spot 的标记基因表达进行编码，得到潜在表示 `z_i`；  
4. 在原始计数空间上，利用 Stage 1 的 `all_genes` 列表对 ST 进行子集化，计算每个 spot 的 UMI 总数 `s_i`，作为后续重建的缩放因子。

### 2. 异构图构建与 GAT 解卷积模型

在 `deconv_model.HeterogeneousGATDeconvolution` 中实现的解卷积模型使用两类节点：

- **Spot 节点**：特征为 VAE 编码器输出的潜在表示，经线性层投影到 GAT 隐空间。  
- **Celltype 节点**：特征为 Stage 1 的簇原型向量，经线性层投影并作为可学习参数。  

图中包含两类主要边：

1. **Spot–Spot 边**  
   - 根据 spot 的空间坐标构建 KNN 图，每个点连向最近 `k_spatial` 个邻居；  
   - 边权为高斯核权重 `\exp(-d^2 / (2\sigma^2))`，或基于潜在表示的余弦相似度。  
2. **Spot–Celltype 边**  
   - 使用 Stage 1 的解卷积分数（或随后 Stage 2 的预测）作为初始权重，表示某细胞类型在该 spot 中的贡献；  
   - 仅保留权重大于阈值的连接，以保持图稀疏。  

GAT 部分由多层 GATConv 组成，在每一层中对上述异构边进行信息聚合，并使用残差连接保持与原始潜在表示的一致性。最终，模型输出对每个 spot–celltype 对的注意力权重，通过 softmax 归一化为解卷积权重矩阵 `W`。

### 3. 基于参考表达的计数空间重建

记 `R_c` 为第 `c` 个簇在全基因上的参考计数向量，`W_{ic}` 为 GAT 预测的 spot `i` 对簇 `c` 的权重（行和为 1）。首先在参考表达空间线性组合得到未缩放的表达：

\[
X_i^{\text{mix}} = \sum_{c} W_{ic} \, R_c .
\]

随后使用该 spot 的总 UMI 数 `s_i` 对混合表达进行缩放，使得重建的总计数与真实文库大小匹配：

\[
\tilde{X}_i^{\text{full}} = s_i \cdot 
\frac{X_i^{\text{mix}}}{\sum_g X_{i,g}^{\text{mix}} + \varepsilon} .
\]

只在标记基因集合上从 `\tilde{X}_i^{\text{full}}` 与 ST 的原始计数进行损失计算。

### 4. 空间解卷积分损失（`SpatialDeconvolutionLoss`）

空间解卷积阶段的总损失为多项重建与正则项的线性组合：

1. **基因维度 Pearson 损失** `L_pearson`  
   - 对每个 spot，在 `log1p` 空间中计算重建表达与真实表达的 Pearson 相关系数 `r_i`，损失定义为 `1 - r_i` 的平均。  
2. **MSE 重建损失** `L_mse`  
   - 在原始计数空间中，对标记基因的表达计算均方误差。  
3. **Cosine 相似度损失** `L_cosine`  
   - 在 `log1p` 空间中计算余弦相似度，损失为 `1 - cos` 的平均。  
4. **基因层面 Pearson / Cosine 损失** `L_gene-pearson`, `L_gene-cosine`  
   - 将表达矩阵转置，视为“基因 × spot”，从基因角度评估重建表达在所有 spot 上的趋势一致性。  
5. **权重正则与稀疏性**  
   - `L_reg`：约束每个 spot 的权重之和接近 1（软约束，虽然 softmax 已保证），稳定训练；  
   - `L_sparse`：基于注意力分布熵的稀疏性项，鼓励每个 spot 主要由少数细胞类型解释。  
6. **全局细胞类型比例一致性** `L_proportion`  
   - 计算 ST 中预测的全局细胞类型权重（`W` 在 spot 维度上的平均）与单细胞数据中簇比例之间的 KL 散度，约束整体分布一致。  

最终损失：

\[
\mathcal{L}_{\text{deconv}}
= \lambda_{\text{P}} L_{\text{pearson}}
+ \lambda_{\text{MSE}} L_{\text{mse}}
+ \lambda_{\text{C}} L_{\text{cosine}}
+ \lambda_{\text{gene-P}} L_{\text{gene-pearson}}
+ \lambda_{\text{gene-C}} L_{\text{gene-cosine}}
+ \lambda_{\text{reg}} L_{\text{reg}}
+ \lambda_{\text{sparse}} L_{\text{sparse}}
+ \lambda_{\text{prop}} L_{\text{proportion}} .
\]

损失权重在不同数据集上通过验证集调参。训练使用 Adam 优化器与 `ReduceLROnPlateau` 调度，并以 Pearson/MSE/Cosine 三项核心重建损失的联合早停作为停止条件。

---

## 阶段三：基于异构图的细胞–细胞通讯建模（`train.py`, `hetero_graph_builder.py`, `hetero_model.py`, `calculate_lr_scores.py`, `evaluate.py`）

### 1. 从反卷积结果构建 spot–cell 表达

细胞通讯部分使用 Stage 2 的反卷积结果作为输入：

1. 从 Stage 2 的 `*_cluster_composition.csv` 以及簇→细胞类型映射中得到每个 spot 的簇比例并聚合为细胞类型比例矩阵 `composition`。  
2. 使用 Stage 1 的簇级全基因参考表达 `cluster_full_expr` 与 spot 的总 UMI 数 `s_i`，按
   \[
   \text{expr}_{i,c} \approx \frac{R_c}{10^4} \times w_{ic} \times s_i
   \]
   构建 `spot_cell_full_expr`：每个条目代表空间点 `i` 中细胞类型 `c` 的全基因表达近似。  
3. 对 `spot_cell_full_expr` 做每细胞的 `normalize_total(1e4)`，并以设定阈值 `mean_expr_threshold` 选出“激活基因”；  
4. 按细胞类型聚合，得到每种细胞在激活基因集合与全基因集合上的平均表达，用于后续 MLP 编码与配体–受体计算。

### 2. 配体–受体得分与空间邻域（`calculate_lr_scores.py`）

1. **空间邻域**  
   - 使用 spot 坐标构建 KNN 图（邻居数 `n_neighbors`），得到邻接矩阵 `knn_mask`，用于限定潜在通信的 spot 对。  
   - 同时通过距离阈值（例如 150 μm）构建更宽松的通信掩码 `lr_comm_mask`，允许在一定空间范围内的潜在通信。  
2. **基因过滤与激活基因**  
   - 从 `spot_cell_full_expr` 中剔除线粒体基因（如 `MT-` 开头）以及低表达基因；  
   - 使用 `normalize_total(1e4)` 后大于阈值的表达筛选活跃基因集合。  
3. **配体–受体对索引化与得分计算**  
   - 将配体–受体对 `(ligand, receptor)` 转换为基因索引对，支持联合受体（多个受体基因乘积）；  
   - 在 KNN 限定的 spot 对 `(i, j)` 之间，遍历所有出现在 `composition` 中的细胞类型对 `(cell_i, cell_j)`，只保留类型不同的细胞对；  
   - 对每个实际存在的 spot–cell 对，取其在 `spot_cell_full_expr` 中的表达，并基于配体–受体表达乘积（如配体表达 × 受体表达几何平均）计算 LR 通讯得分；  
   - 对于长尾分布的得分，使用 `sqrt` 度量与 `log1p` 变换进行压缩；  
   - 将所有非零通讯事件保存为 `lr_scores.csv`，并构建 spot–spot 级的 LR 得分矩阵和 KNN 掩码，以供异构图构建与训练使用。

### 3. 异构图构建与图增强（`hetero_graph_builder.py`）

细胞通讯阶段的异构图包含三类边：

1. **Spot–Spot 边**：与 Stage 2 类似，使用 KNN + 高斯权重构建空间邻域。  
2. **Spot–Celltype 边**：使用 stage2 的细胞组成 `composition` 作为权重，支持开方（`sqrt`）、`log1p` 等变换以减弱极端值影响。  
3. **Celltype–Celltype 边**：基于细胞类型平均表达和配体–受体对列表，计算每对细胞类型之间的 LR 得分并建立有向边。  

此外，`GraphAugmentor` 提供简单的图增强策略：  
对 spot–spot / spot–cell 边进行随机丢弃，对 cell–cell 通讯边按得分删除较弱边，用于训练时的腐败图构建与鲁棒性提升。

### 4. HeteroSTModel：基于边特征的注意力网络（`hetero_model.py`）

HeteroSTModel 以激活基因上的细胞类型表达为输入特征，使用 MLP 代替 VAE 编码器，并结合两个 EdgeAttentionNetwork 处理空间相似度边与通讯边。

1. **MLP 编码器**  
   - 对每个节点（spot 或细胞类型）的激活基因表达使用多层感知机编码到低维潜在空间（默认 64 维），并作为后续边注意力模块的节点特征。  
2. **边注意力层（EdgeAttentionLayer）**  
   - 对每条边的特征（相似度边：[weight, -1]；通讯边：[lr_score, lr_id_emb]）使用小型 MLP 编码并生成注意力 logits；  
   - 以目标节点为归一化单位进行边级 softmax，得到多头注意力权重；  
   - 通过加权聚合边更新到目标节点，实现基于边特征驱动的消息传递，并显式预测边强度 logits 以便监督和分析。  
3. **空间与通讯通路**  
   - 模型分别为空间相似度图和通讯图构建两套 EdgeAttentionNetwork，输出两套节点表示；  
   - 通过可训练的融合层融合空间与通讯信息，再通过线性层得到最终节点表示和用于对比学习的投影向量。  
4. **节点重构头与通讯预测头**  
   - 节点重构头：从融合后的节点表示重建基因表达，用于节点特征重构损失；  
   - 通讯预测头：在 mask 部分通讯边的得分后，只利用剩余边的注意力与节点表示预测被 mask 边的 LR 得分，用于边重构损失。

### 5. 自监督训练目标（`train.py`）

训练阶段采用纯自监督目标，不依赖外部通讯标注：

1. **边掩蔽重构损失**  
   - 在每个子图中随机 mask 一部分通讯边的得分（保留边 ID 与拓扑）；  
   - 使用通讯通路输出的节点表示与注意力得分预测被 mask 边的 LR 得分，并对比原始得分计算 MSE：
     \[
     L_{\text{mask}} = \mathrm{MSE}(\hat{s}_e, s_e) .
     \]
2. **节点特征掩蔽重构损失**  
   - 随机 mask 部分节点的输入基因特征，在输出端重建这些被 mask 的特征，计算 MSE：
     \[
     L_{\text{node}} = \mathrm{MSE}(\hat{x}_{v,\text{mask}}, x_{v,\text{mask}}) .
     \]
3. **总损失**  
   - 当前实验中，总损失为
     \[
     \mathcal{L}_{\text{comm}}
       = \lambda_{\text{mask}} L_{\text{mask}}
       + \lambda_{\text{node}} L_{\text{node}} ,
     \]
     其中 `λ_mask` 与 `λ_node` 分别为边重构和节点重构的权重。  
   - 训练使用 Adam 优化器、余弦退火学习率调度以及基于验证集损失的早停策略。

### 6. 通讯结果汇总与可视化（`evaluate.py`, `lr_communication_plots.py`）

1. **注意力得分汇总**  
   - 在验证或测试阶段，从所有 batch 中收集 cell–cell 通讯边的注意力得分和 LR ID，合并为全局边集合；  
   - 以每个 spot–LR 对为单位，对所有相应边的注意力得分求平均，得到该 spot 上该配体–受体对的综合通讯强度；  
   - 进一步在 LR 维度上汇总，统计每个配体–受体对的出现次数、平均注意力、覆盖的 spot 数量等，保存为 `lr_pair_statistics.csv`。  
2. **统一通讯结果表**  
   - 生成统一的 `lr_communication.csv`，包含源/靶 spot、源/靶细胞类型、LR 对名称、原始 LR 得分及模型注意力得分，用于下游可视化。  
3. **可视化**  
   - `lr_communication_plots.py` 基于上述结果绘制：  
     - 细胞类型之间的通讯频次和注意力矩阵热图；  
     - 每种细胞类型的入度/出度柱状图；  
     - 在空间坐标上的 top 通讯边叠加图（按 LR 对着色、按注意力强度调节线宽）；  
     - 重要配体和受体在空间中的表达分布图等。

---

## 反卷积与通讯结果的评估指标（`SC_MAP_ST/evaluate/*`）

为系统评估不同方法与本方法的反卷积性能，我们实现了统一的评估脚本 `evaluate_benchmark_metrics.py`，可在两类数据上计算指标：

1. **基因表达重建（Expression mode）**  
   - 对于重建的 ST 表达矩阵与“真值” h5ad（或高可信参考）之间，首先对 spot 与基因做交集对齐；  
   - 在 ground truth 上选取高度变异基因（或给定基因列表），计算 per‑gene PCC、SSIM、RMSE、JS，并汇总为四个指标的分布和均值。  
2. **细胞组成预测（Composition mode）**  
   - 在真实和预测的细胞组成矩阵之间按 spot 与细胞类型对齐；  
   - 对每个细胞类型的空间组成向量计算 PCC、SSIM、RMSE、JS，得到 per‑celltype 指标；  
   - 支持多方法对比（例如本方法 vs Tangram vs RCTD vs SPOTlight），并绘制多方法箱线图。  
3. **单细胞类型分析**  
   - 通过 `--celltype` 参数可以只针对单一细胞类型进行评估，并输出每个 spot 的预测值、真值及差值；  
   - 评估脚本还可根据 per‑spot 绝对误差绘制箱线图，直观展示误差分布。  

对于细胞通讯结果，`evaluate.py` 生成的 `lr_communication.csv` 与 `lr_pair_statistics.csv` 可用于统计 top LR 对、top 细胞对及其空间分布，构成定性与半定量的评估基础。

---

## 实现与训练细节

- 所有模型使用 **PyTorch** 与 **PyTorch Geometric** 实现，图结构构建基于 `scikit-learn` 的 KNN 与 `scipy` 的距离函数。  
- 单细胞与空间预处理、差异表达分析和聚类均基于 **Scanpy**。  
- 训练曲线（VAE 与 GAT）以及反卷积重建质量曲线、UMAP 模态对齐可视化均在 `SC_MAP_ST/deconv_results/*` 目录下自动保存。  
- 细胞通讯训练与分析相关中间结果（如 `spot_cell_full_expr.csv`、`lr_scores.csv`、`lr_communication.csv`）及图像存放于 `results/<样本名>/` 目录，便于复现与进一步探索。  

该整体方法在多个公开 ST 数据集和 STARmap 数据上进行了评估，从细胞组成预测、基因表达重建以及配体–受体通讯模式三个层面系统验证了模型的有效性。该 Markdown 文件可直接作为论文或毕业设计中的「方法」章节基础文本。 

