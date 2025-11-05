import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Union, Optional
import logging
from sklearn.model_selection import train_test_split
import glob
import csv
# --------------------------------自定义Dataset-----------------------------------
class STNpzDataset(Dataset):
    def __init__(self, tokens: List[List[str]], images: Optional[List[np.ndarray]] = None, coords: Optional[List[np.ndarray]] = None, spot_id: Optional[List[str]] = None):
        self.tokens = tokens
        self.images = images
        self.coords = coords
        self.spot_id = spot_id
        self.has_images = images is not None
        self.has_coords = coords is not None
        self.has_spot_id = spot_id is not None
        
        if self.has_images:
            assert len(self.tokens) == len(self.images), "tokens和images长度不一致" # type: ignore
        if self.has_coords:
            assert len(self.tokens) == len(self.coords), "tokens和coords长度不一致" # type: ignore
    
    def __len__(self) -> int:
        return len(self.tokens)
    
    def __getitem__(self, idx: int) -> Dict:
        item = {
            'tokens': self.tokens[idx]
        }
        
        if self.has_images:
            item['image'] = self.images[idx] # type: ignore
        if self.has_coords:
            item['coords'] = self.coords[idx] # type: ignore
        if self.has_spot_id:
            item['spot_id'] = self.spot_id[idx] # type: ignore
            
        return item

# --------------------------------自定义Dataloader-----------------------------------
def collate_fn(batch: List[Dict], tokenizer=None) -> Dict:
    # 取出所有 tokens
    tokens = [item['tokens'] for item in batch]   # [['CD3D','CD8A'], ['ACTB',...]]
    
    # 如果有 tokenizer 就编码，否则直接返回原始 tokens
    if tokenizer is not None:
        encoded = tokenizer.batch_encode(tokens)
        input_ids = encoded['input_ids']
    else:
        input_ids = tokens  

    batch_dict = {'tokens': input_ids}

    # 处理图像
    if 'image' in batch[0]:
        images = [item['image'] for item in batch]
        images_array = np.array(images)
        if images_array.ndim == 3:  # [B,H,W] → [B,1,H,W]
            images_array = images_array[:, np.newaxis, :, :]
        elif images_array.shape[-1] in [1,3]:  # [B,H,W,C] → [B,C,H,W]
            images_array = np.transpose(images_array, (0, 3, 1, 2))
        batch_dict['images'] = torch.tensor(images_array, dtype=torch.float32)

    # 处理坐标
    if 'coords' in batch[0]:
        coords = [item['coords'] for item in batch]   # [[x,y], [x,y]...]
        batch_dict['coords'] = torch.tensor(coords, dtype=torch.float32)

    if 'spot_id' in batch[0]:
        batch_dict['spot_ids'] = [item['spot_id'] for item in batch]

    return batch_dict
# --------------------------------数据处理器-----------------------------------
class STDataProcessor:
    def load_npz_data(self, npz_dir: str, test_size: float = 0.1, val_size: float = 0.1, 
                      random_state: int = 42, max_tokens_per_cell: int = 1024) -> Dict:
        npz_files = glob.glob(os.path.join(npz_dir, "*.npz"))
        if not npz_files:
            raise ValueError(f"No npz files found in {npz_dir}")
        logging.info(f"Found {len(npz_files)} npz files in {npz_dir}")
        
        all_tokens, all_images, all_coords, all_spot_ids = [], [], [], []
        has_images, has_coords, has_spot_ids = False, False, False

        for npz_file in npz_files:
            data = np.load(npz_file, allow_pickle=True)
            tokens_array = data['tokens']

            # spot_ids
            if 'spot_ids' in data:
                has_spot_ids = True
                spot_ids_array = data['spot_ids']
                if len(spot_ids_array) == len(tokens_array):
                    all_spot_ids.extend(spot_ids_array)
                else:
                    logging.warning(f"Spot_ids and tokens length mismatch in {npz_file}, skipping spot_ids")
                    has_spot_ids = False

            # 图片
            if 'patch' in data:
                has_images = True
                images = data['patch']
                if len(images) == len(tokens_array):
                    all_images.extend(images)
                else:
                    logging.warning(f"Images and tokens length mismatch in {npz_file}, skipping images")
                    has_images = False

            # 坐标
            if 'coords' in data:
                has_coords = True
                coords_array = data['coords']
                if len(coords_array) == len(tokens_array):
                    all_coords.extend(coords_array)
                else:
                    logging.warning(f"Coords and tokens length mismatch in {npz_file}, skipping coords")
                    has_coords = False

            # 遍历 tokens
            for i in range(len(tokens_array)):
                tokens = tokens_array[i]
                if not isinstance(tokens, list) and hasattr(tokens, 'tolist'):
                    tokens = tokens.tolist()
                elif not isinstance(tokens, list):
                    tokens = [str(tokens)]

                if len(tokens) > max_tokens_per_cell:
                    tokens = tokens[:max_tokens_per_cell]
                all_tokens.append(tokens)

        logging.info(f"Loaded {len(all_tokens)} cell samples")

        # 划分数据集
        datasets = self._split_datasets(all_tokens, all_images, all_coords, all_spot_ids, test_size, val_size, random_state, has_images, has_coords, has_spot_ids)
        
        return datasets
    
    def _split_datasets(self, all_tokens, all_images, all_coords, all_spot_ids,
                        test_size, val_size, random_state, has_images, has_coords, has_spot_ids):

        if has_images and has_coords and has_spot_ids:
            train_tokens, val_tokens, train_images, val_images, train_coords, val_coords, train_spot_ids, val_spot_ids = train_test_split(
                all_tokens, all_images, all_coords, all_spot_ids,
                test_size=val_size,
                random_state=random_state
            )
        elif has_coords and has_spot_ids:
            train_tokens, val_tokens, train_coords, val_coords, train_spot_ids, val_spot_ids = train_test_split(
                all_tokens, all_coords, all_spot_ids,
                test_size=val_size,
                random_state=random_state
            )
            train_images = val_images = None
        elif has_images and has_spot_ids:
            train_tokens, val_tokens, train_images, val_images, train_spot_ids, val_spot_ids = train_test_split(
                all_tokens, all_images, all_spot_ids,
                test_size=val_size,
                random_state=random_state
            )
            train_coords = val_coords = None
        elif has_spot_ids:
            train_tokens, val_tokens, train_spot_ids, val_spot_ids = train_test_split(
                all_tokens, all_spot_ids,
                test_size=val_size,
                random_state=random_state
            )
            train_images = val_images = None
            train_coords = val_coords = None
        else:
            # 没有 spot_ids 就用 dummy idx
            train_tokens, val_tokens = train_test_split(all_tokens, test_size=val_size, random_state=random_state)
            train_spot_ids = list(range(len(train_tokens)))
            val_spot_ids   = list(range(len(val_tokens)))
            train_images = val_images = None
            train_coords = val_coords = None

        return {
            "train": STNpzDataset(train_tokens, train_images, train_coords, train_spot_ids), # type: ignore
            "val":   STNpzDataset(val_tokens,   val_images,   val_coords, val_spot_ids) # type: ignore
        }


#--------------------------------------------这个是为了算单个npz里面的细胞通讯得分构造的dataset-----------------------------------
class Process_CC_Dataset(Dataset):
    def __init__(self, npz_files, use_lr_score=True):  # 添加布尔参数，默认启用
        self.npz_files = npz_files
        data = np.load(npz_files, allow_pickle=True)
        self.tokens = data["tokens"]
        self.images = data["patch"]
        self.coords = data["coords"]
        self.spot_ids = data["spot_ids"]
        self.sample_name = os.path.basename(npz_files).replace(".npz", "")
        self.use_lr_score = use_lr_score  # 保存参数供getitem使用

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        # 构建返回字典的基础部分
        result = {
            "coords": self.coords[idx],
            "tokens": self.tokens[idx],
            "images": self.images[idx],
            "spot_ids": self.spot_ids[idx],
            "sample_name": self.sample_name
        }
        
        # 只有当use_lr_score为True时，才加载并添加lr_score_mat
        if self.use_lr_score:
            lr_score_mat_path = self.npz_files.replace(".npz", "_lr_score_mat.npy")
            lr_score_mat = np.load(lr_score_mat_path)
            result["lr_score_mat"] = lr_score_mat
        
        return result

def create_cc_batch_collate(batch, tokenizer, use_lr_score=False):
    """
    一个能正确处理批次的 collate_fn。
    它会遍历批次中的每个样本，然后将它们堆叠成一个大的批处理张量。
    """
    
    # 初始化几个列表，用来收集批次中每个样本的数据
    list_coords = []
    list_images = []
    list_spot_ids = []
    list_sample_names = []
    if use_lr_score:
        list_lr_score_mat = []
    list_of_sequences = []

    # `batch` 是一个样本字典的列表。我们遍历其中每一个样本。
    # 如果 batch_size=4, 这个循环会执行4次。
    for sample in batch:
        # 1. 处理非文本数据并添加到列表中
        list_coords.append(torch.tensor(sample["coords"], dtype=torch.float32))
        list_images.append(torch.tensor(sample["images"], dtype=torch.float32))
        list_spot_ids.append(sample["spot_ids"]) 
        if use_lr_score:
            list_lr_score_mat.append(torch.tensor(sample["lr_score_mat"], dtype=torch.float32))
        list_sample_names.append(sample["sample_name"])
        # 2. 准备文本数据
        tokens_list = sample["tokens"]
        single_long_sequence = " ".join(tokens_list)
        list_of_sequences.append(single_long_sequence)

    # 3. 对整个批次的文本数据进行一次性编码（最高效的方式）
    # tokenizer.batch_encode 接收一个字符串列表，返回批处理好的张量
    encoded = tokenizer.batch_encode(list_of_sequences)
    
    # `encoded` 里的张量已经是 [batch_size, sequence_length] 形状
    batch_input_ids = encoded["input_ids"]
    batch_attention_mask = encoded["attention_mask"]

    # 4. 将列表中的张量堆叠成一个批处理张量
    # torch.stack 会在第0维增加一个新的“批次”维度
    batch_coords = torch.stack(list_coords, dim=0)
    batch_images = torch.stack(list_images, dim=0)
    if use_lr_score:
        batch_lr_score_mat = torch.stack(list_lr_score_mat, dim=0)
    # 5. 返回最终的批处理字典
    if use_lr_score:
        return {
            "coords": batch_coords,             # shape: [B, num_spots, 2]
            "images": batch_images,             # shape: [B, num_spots, H, W, C]
            "spot_ids": list_spot_ids,         
            "sample_name": list_sample_names,   # 这是一个字符串列表，长度为 B
            "input_ids": batch_input_ids,       # shape: [B, 2048]
            "lr_score_mat": batch_lr_score_mat,
            "attention_mask": batch_attention_mask # shape: [B, 2048]
        }
    else:
        return {
            "coords": batch_coords,             # shape: [B, num_spots, 2]
            "images": batch_images,             # shape: [B, num_spots, H, W, C]
            "spot_ids": list_spot_ids,         
            "sample_name": list_sample_names,   # 这是一个字符串列表，长度为 B
            "input_ids": batch_input_ids,       # shape: [B, 2048]
            "attention_mask": batch_attention_mask # shape: [B, 2048]
        }

# ---------------------------------ST_COMM用的---------------------------------
def convert_label_2_id(label, label_dict):
    return label_dict.get(label.upper(), -1)  # 未知标签返回 -1

class ST_COMMDataset(Dataset):
    def __init__(self, token_patch_npz_path, graph_npz_path, event_csv_path, k_neighbors=10):
        self.data = np.load(token_patch_npz_path, allow_pickle=True)
        print(f"process data from: {token_patch_npz_path}")
        self.tokens = self.data['tokens']          # [N, 2048]
        self.patches = self.data['patch']        # [N, patch_dim]
        self.spot_ids = self.data['spot_ids']      # [N]
        self.coords = self.data['coords']          # [N, 2]
        graph_data = np.load(graph_npz_path, allow_pickle=True)
        self.lr_score_mat = graph_data['lr_score_mat']  # [N, N]
        self.knn_mask = graph_data['knn_mask']          # [N, K]
        self.graph_spot_ids = graph_data['spot_ids']    # [N]
        if list(self.spot_ids) == list(self.graph_spot_ids):
            print("✅ spot_ids 和 graph_spot_ids 顺序一致")
        else:
            print("❌ 顺序不一致")

        self.spot_id_to_index = {sid: i for i, sid in enumerate(self.graph_spot_ids)}

        # 通信事件
        self.comm_event_dict = {}
        with open(event_csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                i, j = row['spot_i'], row['spot_j']
                key = (i, j)
                self.comm_event_dict[key] = {
                    'ligand_receptor': row['ligand_receptor'], 
                    'comm_score': float(row['comm_score']),
                }

        lr_label_csv = pd.read_csv("ligand_receptor_labeled.csv")
        self.label_dict = {row['ligand_receptor']: int(row['lr_id']) for _, row in lr_label_csv.iterrows()}
        self.k = k_neighbors

    def __len__(self):
        return len(self.spot_ids)

    def __getitem__(self, index):
        center_spot_id = self.spot_ids[index]
        try:
            center_graph_idx = self.spot_id_to_index[center_spot_id]
        except:
            raise ValueError(f"Spot ID {center_spot_id} not found in graph spot IDs")

        # 获取 knn 邻居索引
        knn_indices = [np.where(row == 1)[0] for row in self.knn_mask]
        knn_indices = np.array(knn_indices)  # shape: (N, K)
        neighbors = knn_indices[center_graph_idx]  # [K]
        # 聚合中心 + 邻居的所有索引
        all_indices = [center_graph_idx] + list(neighbors)
        tokens = self.tokens[all_indices]         # [K+1, 2048]
        patches = self.patches[all_indices]       # [K+1, patch_dim]
        coords = self.coords[all_indices]        # [K+1, 2]
        # spot_ids 包括了所有邻居，以及中心本身
        spot_ids = self.graph_spot_ids[all_indices]  # [K+1]
        # 构建图边
        num_nodes = len(all_indices)
        edge_index = []
        edge_attr = []

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                src_id = spot_ids[i]
                tgt_id = spot_ids[j]
                edge_index.append([i, j])

                # --- 1. 优先使用通信事件 CSV ---
                if self.comm_event_dict is not None:
                    key = (src_id, tgt_id)  # 注意类型一致性
                    if key in self.comm_event_dict:
                        score = self.comm_event_dict[key]['comm_score']
                        lr_label = self.comm_event_dict[key]['ligand_receptor']
                        lr_id = convert_label_2_id(lr_label, self.label_dict)
                    else:
                        score = 0.0  # 没有通信事件，默认无交互
                        lr_id = -1 
                # --- 2. 否则使用矩阵 ---
                elif self.lr_score_mat is not None:
                    idx_i = self.spot_id_to_index[src_id]
                    idx_j = self.spot_id_to_index[tgt_id]
                    score = self.lr_score_mat[idx_i, idx_j]
                    lr_id = -1 
                else:
                    raise ValueError("Neither comm_event_dict nor lr_score_mat is available")

                edge_attr.append([score, lr_id])

        edge_index = torch.tensor(edge_index, dtype=torch.long).T  # [2, E]
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)   # [E, 1]

        return {
            'tokens': tokens,        # [K+1, 2048]
            'patches': torch.tensor(patches, dtype=torch.float32),      # [K+1, patch_dim]
            'coords': torch.tensor(coords, dtype=torch.float32),        # [K+1, 2]
            'spot_ids': spot_ids,       # [K+1]
            'edge_index': edge_index,
            'edge_attr': edge_attr
        }

def comm_collate_fn(batch, tokenizer):
    """
    自定义 collate_fn，处理批量数据并对 tokens 进行分词。
    batch: 列表，每个元素是 Dataset 返回的样本字典。
    tokenizer: 分词器（如 BERTTokenizer 或自定义的 GeneTokenizer）。
    """
    # 1. 收集批次中所有字段的数据
    list_tokens = []          # 收集所有样本的原始 tokens（基因名字符串）
    list_patches = []         # 收集 patches
    list_spot_ids = []        # 收集 spot_ids
    list_edge_index = []      # 收集 edge_index
    list_edge_attr = []       # 收集 edge_attr
    list_coords = []

    for sample in batch:
        list_tokens.append(sample['tokens'])
        list_patches.append(sample['patches'])
        list_spot_ids.append(sample['spot_ids'])
        list_edge_index.append(sample['edge_index'])
        list_edge_attr.append(sample['edge_attr'])
        list_coords.append(sample['coords'])
    # 2. 处理 tokens：批量分词
    # 2.1 将每个样本的 tokens（基因列表）转换为字符串（用空格拼接）
    # 例如：[[gene1, gene2], [gene3, ...]] → ["gene1 gene2", "gene3 ..."]
    batch_sequences = []
    for tokens in list_tokens:
        # tokens 是 [K+1, M] 形状的基因列表（每个节点的基因）
        # 需将每个节点的基因拼接成字符串，再整体作为序列
        node_sequences = [" ".join(node_genes) for node_genes in tokens]
        batch_sequences.append(node_sequences)  # 形状：[B, K+1]，B是批次大小

    # 2.2 批量编码（注意：需要展平后编码，再恢复形状）
    # 展平：[B, K+1] → [B*(K+1)]（便于分词器批量处理）
    flattened_sequences = [seq for node_seqs in batch_sequences for seq in node_seqs]
    encoded = tokenizer.batch_encode(flattened_sequences)
    # 2.3 恢复形状：[B*(K+1), max_length] → [B, K+1, max_length]
    B = len(batch_sequences)
    K_plus_1 = len(batch_sequences[0])  # 每个样本的节点数（K+1）
    input_ids = encoded["input_ids"].view(B, K_plus_1, -1)  # [B, K+1, max_length]
    attention_mask = encoded["attention_mask"].view(B, K_plus_1, -1)  # [B, K+1, max_length]

    # 3. 处理其他字段（堆叠为批量张量）
    batch_patches = torch.stack(list_patches, dim=0)  # [B, K+1, patch_dim]
    # spot_ids 若为字符串，保持列表；若为整数可堆叠（根据实际类型调整）
    # batch_spot_ids = torch.stack(list_spot_ids, dim=0)  # 若为整数
    batch_edge_index = torch.stack(list_edge_index, dim=0)  # [B, 2, E]
    batch_edge_attr = torch.stack(list_edge_attr, dim=0)    # [B, E, 2]
    batch_coords = torch.stack(list_coords, dim=0)  # [B, K+1, 2]

    # 4. 返回批量数据
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "patches": batch_patches,
        "coords": batch_coords,  
        "spot_ids": list_spot_ids,  # 若为字符串则保持列表
        "edge_index": batch_edge_index,
        "edge_attr": batch_edge_attr
    }