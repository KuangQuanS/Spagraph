from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """CLIP模型配置"""
    # 模型架构
    embed_dim: int = 512  # 特征嵌入维度
    image_encoder: str = "resnet50"  # 图像编码器类型
    text_encoder: str = "bert-base-uncased"  # 文本编码器类型
    temperature: float = 0.07  # 温度参数
    
    # 图像相关
    image_size: int = 224  # 输入图像大小
    
    # 文本相关
    max_text_length: int = 77  # 最大文本长度

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基本训练参数
    batch_size: int = 32  # 批次大小
    num_epochs: int = 30  # 训练轮数
    learning_rate: float = 3e-4  # 学习率
    weight_decay: float = 0.2  # 权重衰减
    warmup_steps: int = 500  # 预热步数
    
    # 优化器参数
    beta1: float = 0.9  # Adam优化器参数
    beta2: float = 0.98  # Adam优化器参数
    eps: float = 1e-6  # Adam优化器参数
    
    # 学习率调度
    lr_scheduler: str = "cosine"  # 学习率调度器类型
    
    # 数据加载
    num_workers: int = 4  # 数据加载线程数
    train_val_split: float = 0.9  # 训练集比例
    
    # 保存和日志
    save_every: int = 1  # 每多少个epoch保存一次模型
    log_every: int = 100  # 每多少个步骤记录一次日志
    
    # 设备
    device: str = "cuda"  # 训练设备
    
    # 混合精度训练
    use_amp: bool = True  # 是否使用混合精度训练
    
    # 分布式训练
    distributed: bool = False  # 是否使用分布式训练
    world_size: int = 1  # 分布式训练的世界大小
    local_rank: int = 0  # 本地排名

@dataclass
class Config:
    """总配置"""
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    
    # 数据路径
    train_data: str = "data/train.json"  # 训练数据路径
    val_data: Optional[str] = "data/val.json"  # 验证数据路径
    
    # 模型保存路径
    output_dir: str = "outputs"  # 输出目录
    
    # 预训练模型路径（如果有）
    pretrained_model: Optional[str] = None  # 预训练模型路径

# 默认配置
default_config = Config()