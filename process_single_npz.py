import json
import csv
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import Process_CC_Dataset,create_cc_batch_collate
from sklearn.neighbors import kneighbors_graph

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
    dataset = Process_CC_Dataset(npz_path, use_lr_score=False)  # 返回 spot_id, tokens, patch, coords
    print(len(dataset), "spots in this sample")
    dataloader = DataLoader(
        dataset,
        batch_size=4,  # 这里可以小一点
        shuffle=False,
        collate_fn=lambda batch: create_cc_batch_collate(batch, tokenizer, use_lr_score=False)
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

        # 直接取最后一层的 raw logits
        attentions = outputs.attentions
        attn_last = attentions[-1].mean(dim=1) # (B, seq, seq) token_attn_score = attn_last[:, 0, :]
        token_attn_score = attn_last[:, 0, :]
        #------------注释隔离------------
        # raw_scores = model.encoder.layer[-1].attention.self.last_attention_scores  
        # # (B, num_heads, seq_len, seq_len)

        # # 平均 head，再取 CLS token 对其他 token 的得分
        # raw_last = raw_scores.mean(dim=1)      # (B, seq_len, seq_len)
        # token_raw_score = raw_last[:, 0, :]    # (B, seq_len)

        # # 取 [CLS] 对每个 token 的注意力强度
        # token_attn_score = token_raw_score  # shape [B, seq_len]
        #------------注释隔离------------

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