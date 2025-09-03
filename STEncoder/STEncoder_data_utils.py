import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Union
import logging
import random
from sklearn.model_selection import train_test_split
import glob

class STDataProcessor:
    def load_npz_data(
        self,
        npz_input: Union[str, List[str]],
        max_tokens_per_cell: int = 1024
    ) -> Dict[str, List]:
        # 判断是目录还是单文件或文件列表
        if isinstance(npz_input, str):
            if os.path.isdir(npz_input):
                npz_files = glob.glob(os.path.join(npz_input, "*.npz"))
            elif npz_input.endswith(".npz") and os.path.isfile(npz_input):
                npz_files = [npz_input]
            else:
                raise ValueError(f"Invalid npz_input: {npz_input}")
        elif isinstance(npz_input, list):
            npz_files = [f for f in npz_input if f.endswith(".npz") and os.path.exists(f)]
        else:
            raise TypeError("npz_input must be str or list of .npz file paths")
        if not npz_files:
            raise ValueError(f"No valid .npz files found in {npz_input}")
        all_tokens, all_patches = [], []
        for path in npz_files:
            try:
                data = np.load(path, allow_pickle=True)
                if "tokens" in data:
                    tokens = data["tokens"]
                    # 限制每个spot的最大token数
                    tokens = [t[:max_tokens_per_cell] for t in tokens]
                    all_tokens.extend(tokens)
                if "patch" in data:
                    patches = data["patch"]
                    all_patches.extend(patches)
            except Exception as e:
                print(f"[Warning] Failed to load {path}: {e}")
                continue
        return {
            "tokens": all_tokens,
            "patch": all_patches
        }
    

#-------------------------------BERT---------------------------------
class MaskedLanguageModelingDataset(Dataset):
    """用于BERT掩码语言模型预训练的数据集"""
    
    def __init__(self, tokens: List[List[str]], mask_prob: float = 0.15, 
                 max_tokens_per_cell: int = 2048):
        self.tokens = tokens
        self.mask_prob = mask_prob
        self.max_tokens_per_cell = max_tokens_per_cell
    
    def __len__(self) -> int:
        return len(self.tokens)
    
    def __getitem__(self, idx: int) -> Dict:
        tokens = self.tokens[idx]
        # 处理tokens长度
        if len(tokens) > self.max_tokens_per_cell:
            tokens = tokens[:self.max_tokens_per_cell]
        
        return {
            'tokens': tokens
        }

def mlm_collate_fn(batch: List[Dict], tokenizer) -> Dict[str, torch.Tensor]:
    """
    用于BERT式的Mask Language Modeling训练的collate_fn
    """
    tokens = [item["tokens"].tolist() if isinstance(item["tokens"], (np.ndarray)) else item["tokens"] for item in batch]
    encoded = tokenizer.batch_encode(tokens)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # 克隆为masked输入
    masked_input_ids = input_ids.clone()
    labels = torch.full_like(input_ids, fill_value=-100)  # 初始化为忽略值

    vocab_size = tokenizer.get_vocab_size()
    special_token_ids = {
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.mask_token_id,
    }

    for i in range(input_ids.size(0)):
        valid_indices = (input_ids[i] != tokenizer.pad_token_id).nonzero(as_tuple=True)[0].tolist()

        if not valid_indices:
            continue

        num_to_mask = max(1, int(0.15 * len(valid_indices)))
        mask_indices = random.sample(valid_indices, num_to_mask)

        for idx in mask_indices:
            original_token_id = input_ids[i, idx].item()
            # 设定label为原始token_id
            labels[i, idx] = original_token_id

            p = random.random()
            if p < 0.8:
                # 80% 替换为 [MASK]
                masked_input_ids[i, idx] = tokenizer.mask_token_id
            elif p < 0.9:
                # 10% 替换为随机非特殊 token
                while True:
                    random_id = random.randint(0, vocab_size - 1)
                    if random_id not in special_token_ids:
                        masked_input_ids[i, idx] = random_id
                        break
                # 10% 保持原样，不改 masked_input_ids[i, idx]

    return {
        "input_ids": masked_input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

#-------------------------------VIT---------------------------------
class MaskedImageModelingDataset(Dataset):
    """用于VIT掩码图像建模预训练的数据集"""
    
    def __init__(self, patches: List[np.ndarray], mask_prob: float = 0.4):
        self.patches = patches
        self.mask_prob = mask_prob
    
    def __len__(self) -> int:
        return len(self.patches)
    
    def __getitem__(self, idx: int) -> Dict:
        patch = self.patches[idx]
        
        # 确保patch是正确的形状
        # if len(patch.shape) == 3:  # [H, W, C]
        #     patch = np.transpose(patch, (2, 0, 1))  # [C, H, W]
        #     continue
        # elif len(patch.shape) == 2:  # [H, W]
        #     patch = patch[np.newaxis, :, :]  # [1, H, W]
        
        return {
            'patch': patch
        }

def mim_collate_fn(batch: List[Dict]) -> Dict:
    """VIT掩码图像建模的collate函数"""
    patches = [item['patch'] for item in batch]
    patches_array = np.array(patches)
    
    # 确保patches是[B, C, H, W]格式
    if len(patches_array.shape) == 3:  # [B, H, W]
        patches_array = patches_array[:, np.newaxis, :, :]
    elif len(patches_array.shape) == 4 and patches_array.shape[1] not in [1, 3]:
        patches_array = np.transpose(patches_array, (0, 3, 1, 2))
    
    # 转换为tensor
    patches_tensor = torch.tensor(patches_array, dtype=torch.float32)
    
    # 归一化
    if patches_tensor.max() > 1.0:
        patches_tensor = patches_tensor / 255.0
    
    # 创建原始图像的副本作为重建目标
    original_patches = patches_tensor.clone()
    
    # 创建掩码图像
    masked_patches = patches_tensor.clone()
    
    # 创建掩码矩阵，用于跟踪哪些patch被掩码
    batch_size, channels, height, width = patches_tensor.shape
    patch_size = 4  # 假设patch_size为4
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    
    # 创建掩码矩阵 [B, num_patches_h, num_patches_w]
    mask = torch.zeros(batch_size, num_patches_h, num_patches_w, dtype=torch.bool)
    
    # 对每个图像应用掩码
    for i in range(batch_size):
        # 随机选择40%的patch进行掩码
        num_patches = num_patches_h * num_patches_w
        num_to_mask = int(num_patches * 0.4)
        
        # 随机选择patch索引
        flat_indices = random.sample(range(num_patches), num_to_mask)
        
        # 转换为2D索引
        for idx in flat_indices:
            h_idx = idx // num_patches_w
            w_idx = idx % num_patches_w
            
            # 设置掩码
            mask[i, h_idx, w_idx] = True
            
            # 掩码对应的图像区域
            h_start = h_idx * patch_size
            h_end = (h_idx + 1) * patch_size
            w_start = w_idx * patch_size
            w_end = (w_idx + 1) * patch_size
            
            # 将patch替换为0（黑色）
            masked_patches[i, :, h_start:h_end, w_start:w_end] = 0
    
    return {
        'masked_patches': masked_patches,
        'original_patches': original_patches,
        'mask': mask
    }
