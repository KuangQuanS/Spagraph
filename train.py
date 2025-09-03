import argparse
import logging
import os
import csv
import torch
import json
import glob
import pandas as pd
from data_utils import STSampleDataset,create_batch_collate, ST_COMMDataset,comm_collate_fn
from model import ST_COMM, BERTEncoder, clip_loss, info_nce_loss
from utils import setup_logging, set_seed, build_graph_data_for_contrastive_learning
from tokenizer import GeneTokenizer
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
from safetensors.torch import load_file
from transformers import BertModel, BertConfig
from torch_geometric.utils import k_hop_subgraph

def parse_args():
    parser = argparse.ArgumentParser(description='Gene Distribution Prediction Training')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录路径')
    parser.add_argument('--vocab_file', type=str, required=True, help='词汇表文件路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录路径')
    parser.add_argument('--vit_pretrained_model', type=str, default=None, help='VIT预训练模型路径或名称')
    parser.add_argument('--bert_pretrained_model', type=str, default=None, help='BERT预训练模型路径或名称')
    parser.add_argument('--pre_work', action='store_true',help='是否生成注意力得分以及配体受体')
    parser.add_argument('--contrast_loss', action='store_true',help='是否启用对比学习')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率')
    parser.add_argument('--max_length', type=int, default=2048, help='最大序列长度')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    
    return parser.parse_args()

def augment_graph(edge_index, edge_attr, drop_prob=0.2, perturb_prob=0.1, attr_noise_std=0.1):
    """
    图增强（保持边数不变）
    输入:
        edge_index: [2, E]
        edge_attr: [E, D]
    输出:
        new_edge_index: [2, E]
        new_edge_attr: [E, D]
    """
    E = edge_index.size(1)
    D = edge_attr.size(1)

    # 1. 随机删除一部分边
    keep_mask = torch.rand(E, device=edge_index.device) > drop_prob
    ei_keep = edge_index[:, keep_mask]
    ea_keep = edge_attr[keep_mask]

    # 2. 随机加边（保持总边数不变）
    num_nodes = int(edge_index.max().item()) + 1
    num_add = E - ei_keep.size(1)
    if num_add > 0:
        added_edges = set()
        while len(added_edges) < num_add:
            i = random.randint(0, num_nodes - 1)
            j = random.randint(0, num_nodes - 1)
            if i != j:
                added_edges.add((i, j))
        added_edge_index = torch.tensor(list(added_edges), dtype=torch.long, device=edge_index.device).T
        added_edge_attr = torch.randn(num_add, D, device=edge_attr.device) * attr_noise_std
        ei_new = torch.cat([ei_keep, added_edge_index], dim=1)
        ea_new = torch.cat([ea_keep, added_edge_attr], dim=0)
    else:
        ei_new, ea_new = ei_keep, ea_keep

    # 3. 边扰动概率
    if perturb_prob > 0:
        num_perturb = int(E * perturb_prob)
        for _ in range(num_perturb):
            idx = random.randint(0, E - 1)
            src = random.randint(0, num_nodes - 1)
            dst = random.randint(0, num_nodes - 1)
            if src != dst:
                ei_new[:, idx] = torch.tensor([src, dst], device=edge_index.device)

    # 4. 属性加噪
    ea_new += torch.randn_like(ea_new) * attr_noise_std

    # 确保输出形状一致
    assert ei_new.size(1) == E, f"Edge count changed: {ei_new.size(1)} vs {E}"
    assert ea_new.size(0) == E, f"Edge attr count changed: {ea_new.size(0)} vs {E}"

    return ei_new, ea_new



import random
def augment_rw(edge_index, edge_attr, num_nodes, walk_start_ratio=0.2, walk_len=3):
    """
    以随机起始点进行多次随机游走，并取k-hop子图
    """
    device = edge_index.device

    num_walks = int(num_nodes * walk_start_ratio)
    start_nodes = torch.randperm(num_nodes)[:num_walks].tolist()

    # 收集访问到的节点
    visited = set(start_nodes)
    adj = [[] for _ in range(num_nodes)]
    for i in range(edge_index.size(1)):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        adj[u].append(v)
        adj[v].append(u)  # 无向图

    for start in start_nodes:
        curr = start
        for _ in range(walk_len):
            neighbors = adj[curr]
            if len(neighbors) == 0:
                break
            curr = random.choice(neighbors)
            visited.add(curr)

    # 子图节点索引
    visited = list(visited)
    visited_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    visited_mask[visited] = True

    # 从原始图中提取子图
    node_idx, new_edge_index, mapping, edge_mask = k_hop_subgraph(
        visited, num_hops=1, edge_index=edge_index, relabel_nodes=True
    )
    new_edge_attr = edge_attr[edge_mask]

    return new_edge_index, new_edge_attr

#---------------------------------ST_COMM模型-----------------------------------
def train_one_sample(model, dataloader, optimizer, scheduler, device,
                    lambda_graph=0.7, lambda_clip=0.3):
    model.train()
    total_loss, total_infonce, total_contrast = 0, 0, 0

    pbar = tqdm(dataloader, desc="Training batches")
    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        images = batch['patches'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        coords = batch['coords'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_attr = batch['edge_attr'].to(device)

        ei1_list, ea1_list = [], []
        ei2_list, ea2_list = [], []

        B = edge_index.size(0)
        num_nodes = coords.size(1)

        for b in range(B):
            ei_b1, ea_b1 = augment_graph(edge_index[b], edge_attr[b])
            ei1_list.append(ei_b1)
            ea1_list.append(ea_b1)

            ei_b2, ea_b2 = augment_rw(edge_index[b], edge_attr[b], num_nodes)
            ei2_list.append(ei_b2)
            ea2_list.append(ea_b2)

        ei1 = torch.stack(ei1_list, dim=0)
        ea1 = torch.stack(ea1_list, dim=0)
        ei2 = torch.stack(ei2_list, dim=0)
        ea2 = torch.stack(ea2_list, dim=0)

        node_emb1, attn_scores1, text_emb, image_emb, fusion_emb = model(input_ids, attention_mask, images, ei1, ea1, coords)
        node_emb2, attn_scores2, text_emb, image_emb, fusion_emb = model(input_ids, attention_mask, images, ei2, ea2, coords)
        infonce_loss = info_nce_loss(node_emb1, node_emb2)

        B, N, D = text_emb.shape
        text_emb_flat = text_emb.reshape(B*N, D)
        image_emb_flat = image_emb.reshape(B*N, D)

        logits_per_image = torch.matmul(image_emb_flat, text_emb_flat.T)
        logits_per_text = logits_per_image.T

        c_loss = clip_loss(logits_per_image, logits_per_text)
        loss = lambda_graph * infonce_loss + lambda_clip * c_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_infonce += infonce_loss.item()
        total_contrast += c_loss.item()

        # 在 tqdm 进度条上显示当前 loss
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "InfoNCE": f"{infonce_loss.item():.4f}",
            "CLIP": f"{c_loss.item():.4f}"
        })

        # 你还可以保留logging，如果需要的话
        if (batch_idx + 1) % 10 == 0:
            logging.info(f"Batch {batch_idx+1} - Loss: {loss.item():.4f}, InfoNCE: {infonce_loss.item():.4f}, CLIP: {c_loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    avg_infonce = total_infonce / len(dataloader)
    avg_contrast = total_contrast / len(dataloader)

    logging.info(f"Epoch summary - Avg Loss: {avg_loss:.4f}, Avg InfoNCE: {avg_infonce:.4f}, Avg CLIP: {avg_contrast:.4f}")

    return avg_loss, avg_infonce, avg_contrast

def process_single_npz(npz_path, save_json=False, attention_threshold=0.2, lr_dict=None, tokenizer=None, model=None, device=None):
    """
    针对单个 npz 样本：
    1. 逐spot跑 BERT 得注意力
    2. 计算 knn 通信矩阵
    3. 存 lr_score_mat.npy (+ 可选 json)
    """
    # 1.先建 dataset + dataloader
    lr_dict=lr_dict
    tokenizer=tokenizer
    model=model
    device=device
    print(f"Processing {npz_path}...")
    dataset = STSampleDataset(npz_path, use_lr_score=False)  # 返回 spot_id, tokens, patch, coords
    print(len(dataset), "spots in this sample")
    dataloader = DataLoader(
        dataset,
        batch_size=4,  # 这里可以小一点
        shuffle=False,
        collate_fn=lambda batch: create_batch_collate(batch, tokenizer, use_lr_score=False)
    )
    # 第一波处理不需要 lr_score_mat，所以 use_lr_score=False
    # 读取所有 spot_id + coords
    spot_ids = dataset.spot_ids
    coords = dataset.coords # (N(spot), 2)
    # 暂存所有注意力结果
    attn_dict = {}  # {spot_id -> {gene_name: attn_score}}
    
    # 2.逐batch跑 BERT
    for batch in tqdm(dataloader, desc=f"Extracting Attention for {os.path.basename(npz_path)}"):
        batch_spot_ids = batch["spot_ids"]
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        attentions = outputs.attentions
        attn_last = attentions[-1].mean(dim=1)  # (B, seq, seq)
        token_attn_score = attn_last[:, 0, :]   # (B, seq_len)
        
        # decode 回 gene_name
        for i, sid in enumerate(batch_spot_ids):
            ids = batch["input_ids"][i].cpu().tolist()
            decoded_string = tokenizer.decode(ids)
            gene_names_list = decoded_string.split()
            scores = token_attn_score[i].cpu().tolist()
            attn_dict[sid] = {g: s for g, s in zip(gene_names_list, scores)}

    # 可选保存 json，但json占的空间有点大
    if save_json:
        json_path = npz_path.replace(".npz", "_attention.json")
        with open(json_path, "w") as f:
            json.dump([
                {"spot_id": sid, "genes": [{"gene": g, "score": s} for g, s in attn_dict[sid].items()]}
                for sid in spot_ids
            ], f, indent=2)
        print(f"✅ Saved attention JSON → {json_path}")
    
    # 3.计算 knn 图 & 通信矩阵
    N = len(spot_ids)
    lr_score_mat = np.zeros((N, N), dtype=np.float32)
    knn = kneighbors_graph(coords, n_neighbors=10, mode="connectivity", include_self=False)
    # 上面得到一个 (N, N) 形状的稀疏矩阵，也就是KNN邻接矩阵，为1的说明是邻居
    knn_mask = knn.toarray() # toarray() 方法转换为稠密矩阵
    # 注意力得分阈值
    atten_thre = attention_threshold
    comm_event_records = [] # 记录配体受体名字
    for i in range(N): # 逐步处理单个npz里的每个spot
        sid_i = spot_ids[i] # 储存spot id
        attn_i = attn_dict.get(sid_i, {}) # 根据id从attn_dict里取出注意力分数，存到attn_i里。如果没找到，就用空字典{}代替
        for j in range(N): # 对于当前的i点，再逐个检查其他所有点j
            if knn_mask[i, j] == 0: # 如果不是邻居，跳过当前点
                continue
            sid_j = spot_ids[j] # 得到邻居的spot id
            attn_j = attn_dict.get(sid_j, {}) # 得到邻居的注意力分散
            comm_score = 0.0 # 初始化一个通信分数为0，后面会累加计算这个分数
            for lig, rec_list in lr_dict.items(): # 逐个遍历所有配体和它们对应的受体列表
                lig_score = attn_i.get(lig, 0) # 从i点的注意力分数attn_i里，取出 “配体lig” 的分数。如果i点没有这个配体，就取0
                if lig_score < atten_thre : # 如果这个配体的注意力分数不够阈值（不是调控基因），跳过这个配体
                    continue
                if lig_score > atten_thre and len(rec_list)==0: # 没有受体也跳过，但是应该不太可能
                    continue
                for rec in rec_list: # 对受体列表进行遍历
                    rec_genes = rec.split("_")
                    if all(attn_j.get(gene, 0) >= atten_thre for gene in rec_genes):
                            rec_score = np.mean([attn_j.get(gene, 0) for gene in rec_genes])
                            # 替代乘法，用几何平均得到通讯得分
                            score = np.sqrt(lig_score * rec_score)
                            comm_score += score
                            # 记录当前两个spot之间的单个配体受体通信事件，comm_score是记录总的通讯得分
                            comm_event_records.append([sid_i, sid_j, f"{lig}_{rec}", score])
            lr_score_mat[i, j] = comm_score # 记录当前邻居spot与目标spot的通信分数
    # 最后得到的lr_score_mat为N x N的矩阵，表示每个spot之间的通信分数，也就是通讯矩阵

    output_npz_path = npz_path.replace(".npz", "_graph_data.npz")
    np.savez_compressed(
        output_npz_path,
        lr_score_mat=lr_score_mat,
        spot_ids=np.array(spot_ids),
        knn_mask=np.array(knn_mask)
    )
    print(f"✅ Saved graph_data npz → {output_npz_path}")
    # 保存索引及其对应的spot id,KNN_MASK
    # 保存通信事件ID
    csv_path = npz_path.replace(".npz", "_lr.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["spot_i", "spot_j", "ligand_receptor", "comm_score"])
        writer.writerows(comm_event_records)
    print(f"✅ Saved ligand-receptor events → {csv_path}")

def topk_gene_set(attn_dict, k=50):
    return set(sorted(attn_dict.items(), key=lambda x: -x[1])[:k])

def main():
    # 解析参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志
    setup_logging(os.path.join(args.output_dir, 'training.log'))
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # 创建分词器
    tokenizer = GeneTokenizer(vocab_file=args.vocab_file, max_length=args.max_length)
    
    #---------------------------------BERTEncoder模型---------------------------------
    BERT_model = BertModel.from_pretrained(args.bert_pretrained_model, local_files_only=True)
    BERT_model.to(device)
    BERT_model.eval()
    #---------------------------------预加载 CellChat 配体-受体数据库---------------------------------
    lr_db = pd.read_csv("cellchat_human.csv")
    lr_dict = {}
    for _, row in lr_db.iterrows():
        lig = str(row["ligand"]).upper()
        rec = str(row["receptor"]).upper()
        if lig not in lr_dict:
            lr_dict[lig] = []
        lr_dict[lig].append(rec)
    print(f"Loaded {len(lr_dict)} ligand entries")
    # 上面加载 CellChat 配体-受体数据库，并得到了一个字典lr_dict，格式类似
    # 'TGFB1': ['TGFBR1_R2', 'ACVR1B_TGFBR2', 'ACVR1C_TGFBR2', 'ACVR1_TGFBR']
    # TGFB1 是配体，后面是该配体对应的受体列表。其中TGFBR1_TGFBR2带下划线的这种表示复合配体，必须两个基因TGFBR1和TGFBR2都表达才能形成配体-受体对。
    # 配体受体对的ID就用cellchat_human.csv中一样的，受体_配体这样写就OK 
    # ---------------------------------前期准备工作，批量跑所有样本生成注意力得分和通讯得分---------------------------------
    # 获取所有.npz文件
    npz_files = glob.glob(os.path.join(args.data_dir, "*.npz"))

    # 过滤掉尾部包含_graph_data.npz的文件
    npz_files = [file for file in npz_files if not file.endswith("_graph_data.npz")]
    print(f"Found {len(npz_files)} samples")
    if args.pre_work:
        for npz_path in npz_files:
            process_single_npz(npz_path, 
                               save_json=False, 
                               attention_threshold=0.005, 
                               lr_dict=lr_dict,
                               tokenizer=tokenizer, 
                               model=BERT_model, 
                               device=device)
    # 注意力阈值的话，2048个token，每个token的平均注意力应该是1/2048约等于0.005，设置为0.02感觉差不多吧，效果不行的话取top-k也许
    # 后面把注意力得分当基因表达值都OK
    # 上面这一步得到两个，一个csv，记录了spot之间的通讯情况，另一个.npy，保存了通讯得分矩阵（N,N）

    #---------------------------------准备训练ST_COMM模型---------------------------------
    # graphormer需要输入节点特征（node features）：融合embedding
    # 边信息（edge list / edge features）：edge_index用通讯得分矩阵，edge_features用score和ligand-receptor id
    # 位置编码、结构编码（如 shortest path distance）:坐标
    model = ST_COMM(bert_model=args.bert_pretrained_model, vit_depth=6, vit_heads=6, hidden_size=384, vit_mlp_dim=1536).to(device)
    print(f"✅ Loaded pretrained BERT from {args.bert_pretrained_model}")

    # 加载 ViT 预训练权重
    vit_ckpt = torch.load(args.vit_pretrained_model, map_location=device, weights_only=True)

    new_vit_ckpt = {}
    prefix_to_remove = 'vit.'
    # 遍历加载的 state_dict，移除前缀
    for key, value in vit_ckpt.items():
        if key.startswith(prefix_to_remove):
            new_key = key.removeprefix(prefix_to_remove)
            new_vit_ckpt[new_key] = value

    model.vit.load_state_dict(new_vit_ckpt)
    print(f"✅ Loaded pretrained ViT from {args.vit_pretrained_model}")

    # 冻结 BERT 前 4 层，只微调后面2层
    for name, param in model.bert.named_parameters():
    # 冻结 embedding + 前4层
        if "embeddings" in name or any(f"encoder.layer.{i}." in name for i in range(4)):
            param.requires_grad = False


    # 冻结 ViT 前 4 层
    for p in model.vit.to_patch_embedding.parameters():
        p.requires_grad = False

    for i, block in enumerate(model.vit.transformer.layers):
        if i < 4 :
            for p in block.parameters():
                p.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 记录所有 epoch 的损失
    epoch_train_loss, epoch_infonce_loss, epoch_contrast_loss = [], [], []

    for epoch in range(args.epochs):
        logging.info(f'Processing Epoch {epoch+1}/{args.epochs}')
        
        total_loss, total_infonce, total_contrast = 0.0, 0.0, 0.0
        total_batches = 0

        # tqdm 包裹 npz_files，显示样本处理进度
        with tqdm(npz_files, desc=f"Processing Epoch {epoch+1}") as pbar:
            for npz_path in pbar:
                dataset = ST_COMMDataset(
                    token_patch_npz_path=npz_path,
                    graph_npz_path=npz_path.replace(".npz", "_graph_data.npz"),
                    event_csv_path=npz_path.replace(".npz", "_lr.csv")
                )
                dataloader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=lambda batch: comm_collate_fn(batch, tokenizer=tokenizer)
                )

                train_loss, infonce_loss, contrast_loss = train_one_sample(
                    model, dataloader, optimizer, scheduler, device
                )

                total_loss += train_loss
                total_infonce += infonce_loss
                total_contrast += contrast_loss
                total_batches += 1

                # 更新进度条后缀，显示当前平均loss
                pbar.set_postfix({
                    "avg_loss": f"{total_loss / total_batches:.4f}",
                    "avg_infoNCE": f"{total_infonce / total_batches:.4f}",
                    "avg_contrast": f"{total_contrast / total_batches:.4f}"
                })

        avg_loss = total_loss / total_batches
        avg_infonce = total_infonce / total_batches
        avg_contrast = total_contrast / total_batches

        logging.info(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}, InfoNCE: {avg_infonce:.4f}, Contrast: {avg_contrast:.4f}")

        epoch_train_loss.append(avg_loss)
        epoch_infonce_loss.append(avg_infonce)
        epoch_contrast_loss.append(avg_contrast)

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{args.output_dir}/ST_COMM_epoch{epoch+1}.pth")

    plt.figure(figsize=(8,6))
    plt.plot(range(1, args.epochs+1), epoch_train_loss, label="Total Loss")
    plt.plot(range(1, args.epochs+1), epoch_infonce_loss, label="InfoNCE Loss")
    plt.plot(range(1, args.epochs+1), epoch_contrast_loss, label="Contrast Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ST_COMM Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("st_comm_loss_curve.png", dpi=150)
    plt.close()
    print("✅ Training Done! Loss curve saved -> st_comm_loss_curve.png")

if __name__ == '__main__':
    main()