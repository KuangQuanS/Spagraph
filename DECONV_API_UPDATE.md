# Deconv API 更新：统一 k_celltype 参数

## 变更说明

### 旧 API（已废弃）
```python
# 单次运行
spg.deconv(
    vae=vae,
    st_h5ad="data/st.h5ad",
    k_celltype=20,
    k_celltype_range=None  # 或者 []
)

# 网格搜索
spg.deconv(
    vae=vae,
    st_h5ad="data/st.h5ad",
    k_celltype=20,  # 被忽略
    k_celltype_range=[20, 25, 30, 35, 40]  # 实际使用这个
)
```

**问题**：
- 有 `k_celltype` 和 `k_celltype_range` 两个参数，容易混淆
- 网格搜索时 `k_celltype` 会被忽略
- 参数语义不清晰

---

### 新 API（推荐）

现在只有一个参数 `k_celltype`，根据类型自动判断：

```python
# 单次运行：传递整数
spg.deconv(
    vae=vae,
    st_h5ad="data/st.h5ad",
    k_celltype=20  # 整数 → 单次运行
)

# 网格搜索：传递列表
spg.deconv(
    vae=vae,
    st_h5ad="data/st.h5ad",
    k_celltype=[20, 25, 30, 35, 40]  # 列表 → 网格搜索
)
```

**优势**：
- ✅ 只有一个参数，语义清晰
- ✅ 类型即语义：整数 = 单次，列表 = 网格搜索
- ✅ 与常见框架的设计一致（如 scikit-learn 的 param_grid）

---

## 参数详细说明

### `k_celltype` 参数

**类型**: `int` 或 `list`

**功能**: 动态 cluster 表示的 k 值（每个 cluster 使用 k 个最近细胞）

**行为**:

1. **整数**（如 `k_celltype=20`）:
   - 单次运行
   - 每个 cluster 使用 20 个最近的单细胞
   
2. **列表**（如 `k_celltype=[20, 25, 30]`）:
   - 网格搜索
   - 遍历所有候选值，选择评分最优的 k
   - 只保存最优结果到磁盘（内存高效）

3. **单元素列表**（如 `k_celltype=[20]`）:
   - 等价于 `k_celltype=20`
   - 自动提取为整数

---

## 完整示例

### 示例 1: 单次运行

```python
import spagraph as spg

# 第一阶段：VAE 训练
vae = spg.train_vae(
    sc_h5ad="data/sc.h5ad",
    output_dir="output/vae/",
    n_clusters=15
)

# 第二阶段：GAT 反卷积（单个 k）
results = spg.deconv(
    vae=vae,
    st_h5ad="data/st.h5ad",
    output_dir="output/deconv/",
    k_celltype=20,  # 单次运行
    save_reconstructed_genes=True
)

print(f"Deconv shape: {results['deconv'].shape}")
print(f"Metrics: {results['metrics']}")
```

### 示例 2: 网格搜索

```python
import spagraph as spg

# 第一阶段：VAE 训练
vae = spg.train_vae(
    sc_h5ad="data/sc.h5ad",
    output_dir="output/vae/",
    n_clusters=15
)

# 第二阶段：GAT 反卷积（网格搜索 k）
results = spg.deconv(
    vae=vae,
    st_h5ad="data/st.h5ad",
    output_dir="output/deconv/",
    k_celltype=[15, 20, 25, 30, 35],  # 网格搜索
    save_reconstructed_genes=True,
    save_all_trials=False  # 不保存所有试验的矩阵（节省空间）
)

print(f"Best k: {results['best_k']}")
print(f"Best score: {results['best_score']:.4f}")
print(f"All trials: {results['all_trials']}")
```

### 示例 3: 自定义网格搜索范围

```python
# 粗网格搜索
results_coarse = spg.deconv(
    vae=vae,
    st_h5ad="data/st.h5ad",
    k_celltype=[10, 20, 30, 40, 50]  # 大步长
)

# 精细网格搜索（基于粗搜索结果）
best_k_coarse = results_coarse['best_k']
results_fine = spg.deconv(
    vae=vae,
    st_h5ad="data/st.h5ad",
    output_dir="output/deconv/",
    k_celltype=[best_k_coarse - 5, best_k_coarse, best_k_coarse + 5],  # 小步长
    save_reconstructed_genes=True
)

print(f"Final best k: {results_fine['best_k']}")
```

---

## 网格搜索输出

### 返回字典包含

```python
{
    'best_k': 25,  # 最优 k 值
    'best_score': 0.1234,  # 最优评分（gene_cosine + cosine）
    'all_trials': [  # 所有试验摘要
        {
            'k': 15,
            'pearson': 0.85,
            'cosine': 0.12,
            'mse': 0.03,
            'gene_pearson': 0.88,
            'gene_cosine': 0.10,
            'score': 0.22  # gene_cosine + cosine
        },
        {
            'k': 20,
            'pearson': 0.87,
            'cosine': 0.10,
            'mse': 0.025,
            'gene_pearson': 0.90,
            'gene_cosine': 0.08,
            'score': 0.18
        },
        # ... 其他试验
    ],
    'deconv': DataFrame,  # 最优 k 的 deconv 矩阵
    'metrics': {...},  # 最优 k 的评估指标
    'sample_name': 'dataset1',
    # ... 其他字段
}
```

### 输出文件（使用最优 k）

```
output/deconv/
├── dataset1_cluster_composition.csv    # deconv 矩阵（最优 k）
├── dataset1_reconstructed.csv          # spot 级别重构表达
├── dataset1_spot_cell_expr.csv         # spot-cell 级别动态表达
└── config_deconv.txt                   # 配置文件（记录最优 k）
```

如果启用 `save_all_trials=True`，还会额外保存：
```
output/deconv/
├── dataset1_cluster_composition_k15.csv
├── dataset1_cluster_composition_k20.csv
├── dataset1_cluster_composition_k25.csv
└── ...
```

---

## 常见问题

**Q1: 如何选择初始的 k 值范围？**

A: 推荐策略：
- 小数据集（< 1000 spots）：`[10, 15, 20, 25, 30]`
- 中等数据集（1000-5000 spots）：`[15, 20, 25, 30, 35]`
- 大数据集（> 5000 spots）：`[20, 25, 30, 35, 40]`

经验规则：k 应小于 cluster 内平均细胞数的 1/2

**Q2: 网格搜索会很慢吗？**

A: 网格搜索的时间复杂度：
- 单次训练时间 × k 候选值数量
- 例如：单次 5 分钟 × 5 个候选值 = 25 分钟
- 所有试验都在内存中进行，不写磁盘（除非启用 `save_all_trials=True`）

**Q3: 如何理解评分指标？**

A: 网格搜索使用的评分：
```python
score = gene_cosine + cosine  # 越小越好
```
- `gene_cosine`: 基因级别的余弦距离（衡量重建质量）
- `cosine`: spot 级别的余弦距离（衡量比例分布）

**Q4: 单元素列表和整数有区别吗？**

A: 没有区别，自动转换：
```python
k_celltype=[20]  # 自动提取为 20
# 等价于
k_celltype=20
```

**Q5: 空列表会怎样？**

A: 自动使用默认值 20：
```python
k_celltype=[]  # 使用默认值 20
```

---

## 迁移指南

### 从旧 API 迁移到新 API

#### 单次运行
```python
# 旧代码
spg.deconv(k_celltype=20, k_celltype_range=None)

# 新代码（只需删除 k_celltype_range）
spg.deconv(k_celltype=20)
```

#### 网格搜索
```python
# 旧代码
spg.deconv(k_celltype=20, k_celltype_range=[20, 25, 30])

# 新代码（将列表移到 k_celltype）
spg.deconv(k_celltype=[20, 25, 30])
```

#### 向后兼容性

**注意**: 旧参数 `k_celltype_range` 已被移除，但不会导致错误：
- 如果代码中仍包含 `k_celltype_range`，会被作为 `**kwargs` 传递
- 不会影响功能，但建议更新代码

---

## 最佳实践

### 推荐工作流

```python
import spagraph as spg

# Step 1: VAE 训练
vae = spg.train_vae(
    sc_h5ad="data/sc.h5ad",
    output_dir="output/vae/",
    n_clusters=15,
    epochs=500
)

# Step 2: 粗网格搜索（快速定位范围）
coarse_search = spg.deconv(
    vae=vae,
    st_h5ad="data/st.h5ad",
    k_celltype=[10, 20, 30, 40, 50],  # 大步长
    n_epochs=200  # 减少训练轮数
)
print(f"Coarse best k: {coarse_search['best_k']}")

# Step 3: 精细网格搜索（优化最优值）
best_k = coarse_search['best_k']
fine_search = spg.deconv(
    vae=vae,
    st_h5ad="data/st.h5ad",
    output_dir="output/deconv/",
    k_celltype=[best_k - 5, best_k, best_k + 5],  # 小步长
    n_epochs=300,  # 完整训练
    save_reconstructed_genes=True
)
print(f"Final best k: {fine_search['best_k']}")

# Step 4: 细胞通讯分析
spg.cellcom(
    deconv_dir="output/deconv/",
    st_h5ad="data/st.h5ad",
    output_dir="output/cellcom/",
    epochs=100
)
```

### 调试建议

```python
# 快速测试（不保存文件）
test_result = spg.deconv(
    vae=vae,
    st_h5ad="data/st.h5ad",
    output_dir=None,  # 不保存
    k_celltype=20,
    n_epochs=50  # 少量轮数
)

# 检查指标
print(test_result['metrics'])
```

---

## 技术细节

### 动态 Cluster 表示

`k_celltype` 控制的是动态 cluster 表示的精细度：

- **k 较小**（如 10）：
  - 使用每个 cluster 最近的 10 个细胞
  - 更能捕捉局部异质性
  - 可能受噪声影响较大

- **k 较大**（如 40）：
  - 使用每个 cluster 最近的 40 个细胞
  - 更稳定，噪声鲁棒性强
  - 可能过度平滑

### 网格搜索算法

```python
for k in k_celltype:
    # 1. 训练模型（纯内存，不写文件）
    result = run_deconv(k_celltype=k, output_dir=None)
    
    # 2. 计算评分
    score = result['metrics']['gene_cosine'] + result['metrics']['cosine']
    
    # 3. 更新最优
    if score < best_score:
        best_k = k
        best_score = score
        best_result = result

# 4. 重新运行最优 k（这次写文件）
final_result = run_deconv(k_celltype=best_k, output_dir=output_dir)
```

**内存优化**：
- 每次试验只保存评估指标，不保存完整矩阵
- 找到最优 k 后，重新运行一次以生成所有输出文件
- 如果启用 `save_all_trials=True`，会在内存中保存所有矩阵

---

## 总结

### 核心变更
- ❌ 删除 `k_celltype_range` 参数
- ✅ 统一为 `k_celltype` 参数（支持整数或列表）

### 使用方式
```python
# 单次运行
k_celltype=20

# 网格搜索
k_celltype=[20, 25, 30]
```

### 优势
- 语义清晰：类型即语义
- API 简洁：一个参数解决两种需求
- 易于理解：符合直觉的设计
