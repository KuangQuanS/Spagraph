# SC_MAP_ST - 简化版

简化的单细胞到空转映射系统，专注于核心功能。

## 🎯 第一阶段：Encoder + 细胞分类器训练

### 功能
- 自动加载Wu数据集下所有样本的SC数据
- 训练细胞encoder + 分类器
- 支持8种细胞类型分类

### 使用方法

```bash
# 运行第一阶段训练
cd /home/maweicheng/ST_Graduation_Project/SC_MAP_ST
python simple_stage1.py
```

### 数据要求
- 数据目录: `/home/maweicheng/ST_Graduation_Project/database/Wu/`
- 每个样本包含: `{sample}_SC.h5ad` 和 `{sample}_ST.h5ad`
- SC数据的obs中需要有'cell_type'列

### 输出结果
- `./simple_sc_results/final_model.pth` - 训练好的模型
- `./simple_sc_results/processed_sc_data.h5ad` - 预处理后的数据
- `./simple_sc_results/confusion_matrix.png` - 混淆矩阵
- `./simple_sc_results/training_curves.png` - 训练曲线

### 细胞类型
支持8种细胞类型：
- B-cells
- CAFs  
- Cancer Epithelial
- Endothelial
- Myeloid
- Normal Epithelial
- PVL
- T-cells

## 🏗️ 模型架构

### Encoder
- Input: 基因表达 (n_genes维)
- Hidden: [1024, 512, 256] 
- Output: 128维潜在表示

### Classifier  
- Input: 128维潜在表示
- Hidden: [64]
- Output: 8类细胞类型

## 📊 训练设置
- 高变基因数: 3000
- 批次大小: 256
- 学习率: 1e-3
- 训练轮数: 100
- 优化器: Adam
- 学习率调度: ReduceLROnPlateau



## 📊 输出结果

Pipeline运行后会在输出目录生成以下文件：

```
results/
├── integrated_sc_data.h5ad           # 整合后的单细胞数据
├── mapped_st_data.h5ad               # 映射后的空转数据
├── cell_classifier.pth               # 训练好的分类器模型
├── batch_integration.png             # 批次整合效果图
├── umap_integration.png              # UMAP整合可视化
├── classifier_evaluation.png         # 分类器评估结果
├── mapping_visualization.png         # 映射结果可视化
└── validation_results.png            # 验证结果图表
```

## 🎨 可视化功能

### 批次整合效果
- PCA和Harmony校正前后对比
- UMAP可视化批次和细胞类型分布

### 分类器性能
- 训练损失和准确率曲线
- 混淆矩阵热图
- 置信度分布

### 空间映射结果
- 空间细胞类型分布
- 预测置信度映射
- 空间一致性评估

### 验证评估
- 组成恢复相关性
- Marker基因保守性
- 空间一致性分布
- 嵌入质量指标

## ⚙️ 参数配置

### 数据预处理参数
- `target_genes`: 目标基因数量 (默认: 3000)
- `min_genes_per_cell`: 每细胞最少基因数 (默认: 200)
- `max_genes_per_cell`: 每细胞最多基因数 (默认: 5000)
- `mt_threshold`: 线粒体基因阈值 (默认: 20.0)

### 批次整合参数
- `n_top_genes`: 高变基因数量 (默认: 2000)
- `n_pcs`: PCA主成分数 (默认: 50)
- `n_harmony_pcs`: Harmony主成分数 (默认: 30)

### 分类器参数
- `hidden_dims`: 隐藏层维度 (默认: [256, 128])
- `dropout`: Dropout率 (默认: 0.3)
- `learning_rate`: 学习率 (默认: 1e-3)
- `n_epochs`: 训练轮数 (默认: 100)

### 映射参数
- `n_neighbors`: k近邻数量 (默认: 10)
- `similarity_metric`: 相似性度量 (默认: 'cosine')
- `latent_dim`: 潜在空间维度 (默认: 128)

## 🔍 验证指标

### 细胞类型准确性
- 总体准确率
- 按置信度分层准确率
- 按细胞类型准确率

### 组成恢复质量
- 组成向量相关性
- 平均绝对误差(MAE)

### Marker基因保守性
- 每种细胞类型的marker基因表达相关性
- 总体保守性得分

### 空间一致性
- 邻居细胞类型一致性
- 空间聚集程度

### 嵌入质量
- 轮廓系数 (Silhouette Score)
- Calinski-Harabasz指数
- 批次整合质量

## 🛠️ 高级用法

### 单独使用各模块

```python
# 只进行批次整合
from batch_integration import BatchIntegrator
integrator = BatchIntegrator()
integrated_data = integrator.integration_pipeline(sc_adata_list)

# 只训练分类器
from supervised_classifier import SupervisedClassifier
classifier = SupervisedClassifier()
classifier.train(X_train, y_train)

# 只进行空间映射
from spatial_mapper import SpatialMapper
mapper = SpatialMapper()
results = mapper.mapping_pipeline(sc_adata, st_adata)

# 只进行验证
from validation_evaluator import MappingValidator
validator = MappingValidator()
validation_results = validator.comprehensive_validation(sc_adata, st_adata, mapping_results)
```

### 自定义参数

```python
# 创建自定义配置的映射器
mapper = SCToSTMapper(output_dir='./custom_results')

# 修改预处理参数
processed_sc_list, processed_st_adata = mapper.preprocess_data(
    sc_adata_list, st_adata,
    target_genes=5000,
    min_genes_per_cell=500,
    mt_threshold=15.0
)

# 修改整合参数
adata_integrated = mapper.integrate_batches(
    processed_sc_list, batch_names,
    n_top_genes=3000
)
```

## 📋 数据格式要求

### 单细胞数据 (h5ad格式)
- `adata.X`: 基因表达矩阵 (细胞 × 基因)
- `adata.obs['cell_type']`: 细胞类型标注
- `adata.var.index`: 基因名称

### 空转数据 (h5ad格式)
- `adata.X`: 基因表达矩阵 (spots × 基因)
- `adata.obsm['spatial']`: 空间坐标 (可选)
- `adata.var.index`: 基因名称

## 🤝 贡献指南

欢迎提交issues和pull requests来改进这个项目！

## 📄 许可证

MIT License

## 📞 联系方式

如有问题，请提交issue或联系开发团队。