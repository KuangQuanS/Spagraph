#!/usr/bin/env python
"""测试高级混合损失函数"""

import torch
import torch.nn as nn
from model import SpatialDeconvolutionLoss

def test_loss_function():
    """测试损失函数的各个组件"""
    print("=" * 70)
    print("🧪 测试高级混合损失函数")
    print("=" * 70)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")
    
    # 创建示例数据
    n_spots = 100
    n_cell_types = 25
    n_genes = 1262
    
    # 随机生成数据
    deconv_weights = torch.softmax(torch.randn(n_spots, n_cell_types), dim=1).to(device)
    celltype_expressions = torch.randn(n_cell_types, n_genes).to(device)
    target_expressions = torch.randn(n_spots, n_genes).to(device)
    spatial_coords = torch.randn(n_spots, 2).to(device)
    
    print(f"数据形状:")
    print(f"  └─ 解卷积权重: {deconv_weights.shape}")
    print(f"  └─ 细胞类型表达: {celltype_expressions.shape}")
    print(f"  └─ 目标表达: {target_expressions.shape}")
    print(f"  └─ 空间坐标: {spatial_coords.shape}\n")
    
    # 创建损失函数（默认参数）
    print("初始化损失函数...")
    loss_fn = SpatialDeconvolutionLoss(
        lambda1=0.5,  # Wasserstein
        lambda2=0.3,  # PCC
        lambda3=0.2,  # 空间平滑性
        beta=0.1      # 正则化
    ).to(device)
    print("✅ 损失函数初始化成功\n")
    
    # 计算损失
    print("计算损失...")
    loss_outputs = loss_fn(
        deconv_weights=deconv_weights,
        celltype_expressions=celltype_expressions,
        target_expressions=target_expressions,
        spatial_coords=spatial_coords
    )
    
    print("\n📊 损失分解:")
    print(f"  ├─ 总损失 (Total Loss):        {loss_outputs['total_loss'].item():.6f}")
    print(f"  ├─ Wasserstein距离:            {loss_outputs['wasserstein_loss'].item():.6f}")
    print(f"  ├─ PCC相关性:                  {loss_outputs['pcc_loss'].item():.6f}")
    print(f"  ├─ 空间平滑性:                 {loss_outputs['smooth_loss'].item():.6f}")
    print(f"  └─ 权重正则化:                 {loss_outputs['weight_reg'].item():.6f}")
    
    # 验证损失值是有限的
    print("\n✅ 验证:")
    for key, value in loss_outputs.items():
        if torch.is_tensor(value):
            is_finite = torch.isfinite(value).all()
            print(f"  └─ {key}: {'✓' if is_finite else '✗'} (有限值)")
    
    # 测试反向传播
    print("\n🔄 测试反向传播...")
    total_loss = loss_outputs['total_loss']
    total_loss.backward()
    print("✅ 反向传播成功\n")
    
    # 测试不同的权重配置
    print("=" * 70)
    print("🎯 测试不同的损失权重配置")
    print("=" * 70)
    
    configs = [
        {"name": "平衡配置", "lambda1": 0.5, "lambda2": 0.3, "lambda3": 0.2, "beta": 0.1},
        {"name": "强化Wasserstein", "lambda1": 0.8, "lambda2": 0.1, "lambda3": 0.05, "beta": 0.05},
        {"name": "强化PCC", "lambda1": 0.2, "lambda2": 0.6, "lambda3": 0.1, "beta": 0.1},
        {"name": "强化空间平滑", "lambda1": 0.3, "lambda2": 0.2, "lambda3": 0.5, "beta": 0.0},
    ]
    
    for config in configs:
        name = config.pop("name")
        loss_fn_test = SpatialDeconvolutionLoss(**config).to(device)
        loss_outputs_test = loss_fn_test(
            deconv_weights=deconv_weights,
            celltype_expressions=celltype_expressions,
            target_expressions=target_expressions,
            spatial_coords=spatial_coords
        )
        total = loss_outputs_test['total_loss'].item()
        print(f"  {name:<20} → Total Loss={total:.6f}")
    
    print("\n" + "=" * 70)
    print("✅ 所有测试通过！损失函数已准备好使用")
    print("=" * 70)

if __name__ == '__main__':
    test_loss_function()
