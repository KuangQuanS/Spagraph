import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from typing import Optional, Tuple
import json
import os
from model import ViT  # 使用您的ViT实现
from tokenizer import GeneTokenizer  # 使用您的tokenizer

# 简单的进度条替代
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", **kwargs):
        print(f"{desc}")
        for i, item in enumerate(iterable):
            if i % 10 == 0:  # 每10次打印一次进度
                print(f"Progress: {i}/{len(iterable) if hasattr(iterable, '__len__') else '?'}")
            yield item

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerDecoder(nn.Module):
    """Transformer Decoder模块"""
    def __init__(self, d_model: int, nhead: int, num_layers: int, vocab_size: int, max_seq_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """生成自回归掩码"""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
    
    def forward(self, memory: torch.Tensor, tgt: Optional[torch.Tensor] = None, 
                max_length: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            memory: ViT encoder的输出 [batch_size, patch_num, d_model]
            tgt: 目标token序列 [batch_size, seq_len] (训练时使用)
            max_length: 生成的最大长度 (推理时使用)
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        if tgt is not None:
            # 训练模式：teacher forcing
            return self._forward_train(memory, tgt)
        else:
            # 推理模式：自回归生成
            return self._forward_inference(memory, max_length or self.max_seq_len)
    
    def _forward_train(self, memory: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """训练模式的前向传播"""
        batch_size, seq_len = tgt.shape
        device = tgt.device
        
        # Token embedding + 位置编码
        tgt_emb = self.token_embedding(tgt) * math.sqrt(self.d_model)  # [batch_size, seq_len, d_model]
        tgt_emb = self.pos_encoding(tgt_emb.transpose(0, 1)).transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # 生成自回归掩码
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(device)
        
        # Transformer decoder
        output = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask
        )  # [batch_size, seq_len, d_model]
        
        # 输出投影
        logits = self.output_projection(output)  # [batch_size, seq_len, vocab_size]
        
        return logits
    
    def _forward_inference(self, memory: torch.Tensor, max_length: int) -> torch.Tensor:
        """推理模式的前向传播（自回归生成）"""
        batch_size = memory.shape[0]
        device = memory.device
        
        # 初始化输出序列，从<START>开始 (假设vocab中0是START token)
        generated = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        for _ in range(max_length - 1):
            # 当前序列的embedding
            tgt_emb = self.token_embedding(generated) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoding(tgt_emb.transpose(0, 1)).transpose(0, 1)
            
            # 生成掩码
            seq_len = generated.shape[1]
            tgt_mask = self.generate_square_subsequent_mask(seq_len).to(device)
            
            # Decoder前向传播
            output = self.transformer_decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask
            )
            
            # 预测下一个token
            logits = self.output_projection(output[:, -1:, :])  # [batch_size, 1, vocab_size]
            next_token = torch.argmax(logits, dim=-1)  # [batch_size, 1]
            
            # 添加到序列中
            generated = torch.cat([generated, next_token], dim=1)
            
            # 如果所有序列都生成了结束符，可以提前停止
            # (这里假设vocab中1是END token)
            if torch.all(next_token == 1):
                break
        
        return generated

class STDecoder(nn.Module):
    """空间转录组解码器：从patch预测token序列"""
    
    def __init__(self, 
                 image_size: int = 32,
                 patch_size: int = 4,
                 image_channels: int = 3,
                 vit_dim: int = 384,
                 vit_depth: int = 12,
                 vit_heads: int = 12,
                 vit_mlp_dim: int = 1536,
                 vocab_size: int = 20000,
                 decoder_layers: int = 6,
                 decoder_heads: int = 8,
                 max_seq_len: int = 2048):
        """
        Args:
            image_size: 图像尺寸
            patch_size: patch大小
            image_channels: 图像通道数
            vit_dim: ViT隐层维度
            vit_depth: ViT深度
            vit_heads: ViT注意力头数
            vit_mlp_dim: ViT MLP维度
            vocab_size: 词汇表大小
            decoder_layers: decoder层数
            decoder_heads: 多头注意力头数
            max_seq_len: 最大序列长度
        """
        super().__init__()
        
        # 使用您的ViT实现
        self.vit = ViT(
            image_size=image_size,
            patch_size=patch_size,
            channels=image_channels,
            dim=vit_dim,
            depth=vit_depth,
            heads=vit_heads,
            mlp_dim=vit_mlp_dim,
            dropout=0.1,
            emb_dropout=0.1
        )
        
        # Transformer Decoder
        self.decoder = TransformerDecoder(
            d_model=vit_dim,
            nhead=decoder_heads,
            num_layers=decoder_layers,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len
        )
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.vit_dim = vit_dim
    
    def forward(self, 
                images: torch.Tensor, 
                target_tokens: Optional[torch.Tensor] = None,
                max_length: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            images: 图像张量 [batch_size, 3, image_size, image_size] 或 [batch_size, 1, 3, image_size, image_size]
            target_tokens: 目标token序列 [batch_size, seq_len] (训练时)
            max_length: 生成最大长度 (推理时)
        
        Returns:
            训练时返回logits: [batch_size, seq_len, vocab_size]
            推理时返回生成的token序列: [batch_size, generated_len]
        """
        # 处理输入维度
        if len(images.shape) == 4:
            # [batch_size, 3, H, W] -> [batch_size, 1, 3, H, W]
            images = images.unsqueeze(1)
        
        # ViT编码
        # 您的ViT期望输入 [B, N, C, H, W]，对于单个图像，N=1
        patch_embeddings = self.vit(images)  # [batch_size, 1, vit_dim]
        
        # 去掉中间的维度，因为我们只有一个图像per batch
        memory = patch_embeddings.squeeze(1)  # [batch_size, vit_dim]
        
        # 方案1: 添加可学习的位置嵌入来扩展memory
        # 这样每个位置的memory都有不同的信息
        batch_size = memory.shape[0]
        num_memory_tokens = 16  # 可以调整这个数量
        device = memory.device  # 获取memory的设备
        
        # 创建可学习的位置嵌入
        if not hasattr(self, 'memory_pos_embedding'):
            self.memory_pos_embedding = nn.Parameter(
                torch.randn(num_memory_tokens, self.vit_dim) * 0.02
            )
        
        # 确保位置嵌入在正确的设备上
        pos_embedding = self.memory_pos_embedding.to(device)
        
        # 扩展memory并添加位置信息
        expanded_memory = memory.unsqueeze(1).repeat(1, num_memory_tokens, 1)  # [batch_size, 16, vit_dim]
        pos_emb = pos_embedding.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, 16, vit_dim]
        memory = expanded_memory + pos_emb  # [batch_size, 16, vit_dim] - 每个位置都有不同信息
        
        # Decoder解码
        output = self.decoder(memory=memory, tgt=target_tokens, max_length=max_length)
        
        return output

class STDecoderDataset(Dataset):
    """ST Decoder数据集"""
    
    def __init__(self, npz_path: str, tokenizer_vocab_path: str = None):
        """
        Args:
            npz_path: NPZ文件路径
            tokenizer_vocab_path: tokenizer词汇表路径
        """
        self.data = np.load(npz_path, allow_pickle=True)
        
        # 加载数据
        self.patches = self.data['patch']  # [num_spots, 3, 32, 32] 根据您的ViT配置
        self.tokens = self.data['tokens']    # [num_spots, seq_len] - 基因名称数组
        self.coords = self.data['coords']    # [num_spots, 2]
        self.spot_ids = self.data['spot_ids']  # [num_spots]
        
        # 初始化tokenizer
        self.tokenizer = None
        if tokenizer_vocab_path and os.path.exists(tokenizer_vocab_path):
            self.tokenizer = GeneTokenizer(tokenizer_vocab_path, max_length=2048)
        
        print(f"Loaded {len(self.patches)} spots")
        print(f"Patch shape: {self.patches.shape}")
        print(f"Token shape: {self.tokens.shape}")
        print(f"Token dtype: {self.tokens.dtype}")   
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = torch.FloatTensor(self.patches[idx])  # [3, 32, 32]
        
        # 处理基因名称tokens
        tokens_data = self.tokens[idx]
        
        if self.tokenizer is not None:
            # 如果有tokenizer，将基因名称转换为ID
            if isinstance(tokens_data, np.ndarray) and tokens_data.dtype == object:
                # 处理object数组，提取基因名称
                gene_names = []
                for token in tokens_data:
                    # 清理numpy string格式
                    gene_name = str(token)
                    if gene_name.startswith("np.str_('") and gene_name.endswith("')"):
                        gene_name = gene_name[9:-2]  # 移除 "np.str_('" 和 "')"
                    elif gene_name.startswith("'") and gene_name.endswith("'"):
                        gene_name = gene_name[1:-1]  # 移除引号
                    
                    # 过滤掉空字符串和无效的基因名
                    if gene_name and gene_name != "nan" and len(gene_name) > 0:
                        gene_names.append(gene_name)
                
                # 限制基因数量以避免序列过长
                # if len(gene_names) > 2048:  # 改为2000个基因以内，接近你想要的2048
                #     gene_names = gene_names[:2048]
            else:
                gene_names = tokens_data.tolist() if hasattr(tokens_data, 'tolist') else tokens_data
            
            # 使用tokenizer编码
            try:
                encoded = self.tokenizer.encode(gene_names, add_special_tokens=True)
                tokens = encoded['input_ids']
            except Exception as e:
                print(f"Error encoding genes at idx {idx}: {e}")
                print(f"Gene names sample: {gene_names[:5] if len(gene_names) > 5 else gene_names}")
                # 返回默认的token序列
                tokens = torch.LongTensor([self.tokenizer.cls_token_id, self.tokenizer.sep_token_id])
        else:
            # 如果没有tokenizer，尝试直接转换
            try:
                if isinstance(tokens_data, np.ndarray) and tokens_data.dtype == object:
                    # 跳过object类型的复杂处理，返回简单的序列
                    tokens = torch.LongTensor([0, 1, 2])  # 简单的占位符
                else:
                    tokens = torch.LongTensor(tokens_data)
            except:
                tokens = torch.LongTensor([0, 1, 2])  # 默认值
            
        coord = torch.FloatTensor(self.coords[idx])   # [2]
        spot_id = self.spot_ids[idx]  # 标识符
        
        return {
            'images': patch, 
            'tokens': tokens,
            'coords': coord,
            'spot_id': spot_id
        }

def collate_fn(batch):
    """自定义collate函数来处理不同长度的token序列"""
    images = torch.stack([item['images'] for item in batch])
    coords = torch.stack([item['coords'] for item in batch])
    spot_ids = [item['spot_id'] for item in batch]
    
    # 处理不同长度的tokens
    tokens_list = [item['tokens'] for item in batch]
    
    # 找到最大长度
    max_len = max(len(tokens) for tokens in tokens_list)
    
    # Pad所有序列到相同长度
    padded_tokens = []
    for tokens in tokens_list:
        if len(tokens) < max_len:
            # 用-100填充（CrossEntropyLoss的ignore_index）
            pad_length = max_len - len(tokens)
            padded = torch.cat([tokens, torch.full((pad_length,), -100, dtype=torch.long)])
        else:
            padded = tokens
        padded_tokens.append(padded)
    
    tokens = torch.stack(padded_tokens)
    
    return {
        'images': images,
        'tokens': tokens,
        'coords': coords,
        'spot_id': spot_ids
    }

def train_st_decoder(model: STDecoder, 
                    dataloader: DataLoader, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    num_epochs: int = 10):
    """训练ST Decoder"""
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # 忽略padding token
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            images = batch['images'].to(device)    # [B, 3, 32, 32]
            tokens = batch['tokens'].to(device)    # [B, seq_len]
            
            # 准备输入和目标
            input_tokens = tokens[:, :-1]    # 输入：除了最后一个token
            target_tokens = tokens[:, 1:]    # 目标：除了第一个token
            
            # 前向传播
            logits = model(images=images, target_tokens=input_tokens)
            
            # 计算损失
            loss = criterion(logits.reshape(-1, model.vocab_size), target_tokens.reshape(-1))
            
            # 计算准确率 - 只计算非padding token
            predictions = torch.argmax(logits, dim=-1)
            # 创建mask：排除padding token (通常是0)和ignore_index (-100)
            valid_mask = (target_tokens != -100) & (target_tokens != 0)  # 排除padding token 0
            correct = (predictions == target_tokens) & valid_mask
            
            total_correct += correct.sum().item()
            total_tokens += valid_mask.sum().item()
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 计算当前batch的准确率 - 只计算非padding token
            batch_accuracy = correct.sum().item() / valid_mask.sum().item() if valid_mask.sum().item() > 0 else 0.0
            progress_bar.set_postfix({
                'loss': loss.item(), 
                'accuracy': f"{batch_accuracy:.3f}",
                'valid_tokens': valid_mask.sum().item()
            })
        
        avg_loss = total_loss / len(dataloader)
        epoch_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

def generate_tokens(model: STDecoder, 
                   images: torch.Tensor, 
                   device: torch.device,
                   max_length: int = 2048) -> torch.Tensor:
    """生成token序列"""
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        generated_tokens = model(images=images, max_length=max_length)
    return generated_tokens

def main():
    """主函数示例"""
    # 设备设置
    device = torch.device('cuda:1')
    print(f"Using device: {device}")
    
    # 数据加载
    npz_path = "/home/maweicheng/ST_Graduation_Project/database/GSM6177601/GSE203612_GSM6177601.npz"  # 替换为您的数据路径
    tokenizer_vocab_path = "STEncoder/gene_tokenizer_vocab_nolab.json"  # 替换为您的词汇表路径
    
    # 首先创建dataset以获取真实的vocab_size
    dataset = STDecoderDataset(npz_path, tokenizer_vocab_path)
    
    # 从tokenizer获取实际的vocab_size
    if dataset.tokenizer is not None:
        vocab_size = dataset.tokenizer.get_vocab_size()
        print(f"Using vocab_size from tokenizer: {vocab_size}")
    else:
        vocab_size = 20000  # 默认值
        print(f"Using default vocab_size: {vocab_size}")
    
    # 创建模型（使用真实的vocab_size）
    model = STDecoder(
        image_size=32,      # 您的patch尺寸
        patch_size=4,       # patch分割大小
        image_channels=3,   # RGB图像
        vit_dim=384,        # 匹配您model.py中的dim
        vit_depth=12,       # ViT深度
        vit_heads=12,       # 注意力头数
        vit_mlp_dim=1536,   # MLP维度
        vocab_size=vocab_size,
        decoder_layers=6,
        decoder_heads=8,
        max_seq_len=2048
    ).to(device)
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # 训练
    print("Starting training...")
    train_st_decoder(model, dataloader, optimizer, device, num_epochs=500)
    
    # 保存模型
    torch.save(model.state_dict(), "st_decoder_checkpoint.pth")
    print("Model saved!")
    
    # 推理示例
    model.eval()
    sample_batch = next(iter(dataloader))
    images = sample_batch['images'][:1]  # 取第一个样本
    
    generated_tokens = generate_tokens(model, images, device)
    print(f"Generated tokens shape: {generated_tokens.shape}")
    print(f"Generated tokens: {generated_tokens[0]}")

if __name__ == "__main__":
    main()