我想生成一个用于空间转录组+细胞通讯的 Heterogeneous Graph Attention Network (HeteroGAT) 训练代码。要求如下：

1️⃣ 节点设计：
- Spot 节点：
  - 特征 = ResNet 提取的图像 embedding + VAE 编码的 marker 基因表达 embedding。
- Celltype 节点：
  - 特征 = marker 基因表达 embedding + 在该 spot 内的比例（可以乘上比例，例如 sqrt(w)）。

2️⃣ 边设计：
- Spot–Spot 边：
  - 根据物理坐标，使用 kNN 或半径 r 建边。
  - 权重 = 距离衰减（例如 exp(-d^2/(2*sigma^2))）。
  - 这些边在 GAT 中不必更新，也可固定，主要用于提供空间上下文。
- Spot–Celltype 边：
  - 权重 = celltype 在该 spot 中的比例（可 sqrt 或 log1p）。
- Celltype–Celltype 边：
  - 不区分同 spot / 邻近 spot，统一使用 ligand–receptor (LR) score。
  - 每条边附带通讯 ID，用作 edge feature 或 hetero edge type。
  - 可以在子图中构建，子图包含中心 spot + 邻近 spot + 对应 celltype 节点。

3️⃣ 图模型：
- 使用 Heterogeneous Graph Attention Network (HeteroGAT)。
- Spot–Spot 边可固定，不参与注意力更新。
- 主要关注 Celltype–Celltype 边的 attention 输出，可作为细胞通讯强度 proxy。

4️⃣ 训练目标：
- 自监督训练：
  - 重建 spot 层面的基因表达（或 marker 基因表达）。
  - 可加 LR 边预测损失，对 edge attention 或 LRscore 做拟合。
- 输出：
  - 节点 embedding（Spot & Celltype）
  - Celltype–Celltype 边 attention，用于表示局部细胞通讯强度。

5️⃣ 其他要求：
- 支持子图采样训练，保证大图可处理。
- 可在 PyTorch Geometric 或 DGL 实现。
- 提供清晰的数据结构设计示例（节点特征、边列表、边权、edge type）。
- 可直接运行，包含模型定义 + 前向传播 + loss 计算。

请生成符合以上要求的 Python 代码示例。
