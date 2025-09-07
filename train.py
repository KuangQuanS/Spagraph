import argparse
import logging
import os
import torch
import glob
import pandas as pd
from data_utils import ST_COMMDataset,comm_collate_fn
from model import ST_COMM, clip_loss, info_nce_loss
from utils import setup_logging, set_seed
from tokenizer import GeneTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from transformers import BertModel, BertConfig
from rw_graph import augment_graph, augment_rw
from process_single_npz import process_single_npz
#from BertLogitsAttention import BertSelfAttentionWithLogits

def parse_args():
    parser = argparse.ArgumentParser(description='Gene Distribution Prediction Training')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录路径')
    parser.add_argument('--vocab_file', type=str, required=True, help='词汇表文件路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录路径')
    parser.add_argument('--vit_pretrained_model', type=str, default=None, help='VIT预训练模型路径或名称')
    parser.add_argument('--bert_pretrained_model', type=str, default=None, help='BERT预训练模型路径或名称')
    parser.add_argument('--pre_work', action='store_true',help='是否生成注意力得分以及配体受体')
    parser.add_argument('--save_json', action='store_true',help='是否生成注意力得分以及配体受体')
    parser.add_argument('--contrast_loss', action='store_true',help='是否启用对比学习')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率')
    parser.add_argument('--max_length', type=int, default=2048, help='最大序列长度')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--npz_file', type=str, help='单个npz文件路径')
    
    return parser.parse_args()

#---------------------------------ST_COMM模型-----------------------------------
# TODO:考虑加入融合后的向量之间的余弦相似度来使不同类之间的spot通讯更重要
def train_one_sample(model, batch, optimizer, scheduler, device,
                     lambda_graph=0.7, lambda_clip=0.3):
    """
    处理单个 batch 的逻辑
    """
    model.train()

    # 获取 batch 数据
    input_ids = batch['input_ids'].to(device)
    images = batch['patches'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    coords = batch['coords'].to(device)
    edge_index = batch['edge_index'].to(device)
    edge_attr = batch['edge_attr'].to(device)

    # 图增强
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

    # 模型前向传播
    node_emb1, attn_scores1, attn_logistic1, text_emb, image_emb, fusion_emb = model(input_ids, attention_mask, images, ei1, ea1, coords)
    node_emb2, attn_scores2, attn_logistic2, text_emb, image_emb, fusion_emb = model(input_ids, attention_mask, images, ei2, ea2, coords)

    # 计算损失
    infonce_loss = info_nce_loss(node_emb1, node_emb2)

    B, N, D = text_emb.shape
    text_emb_flat = text_emb.reshape(B * N, D)
    image_emb_flat = image_emb.reshape(B * N, D)
    logits_per_image = torch.matmul(image_emb_flat, text_emb_flat.T)
    logits_per_text = logits_per_image.T

    c_loss = clip_loss(logits_per_image, logits_per_text)
    loss = lambda_graph * infonce_loss + lambda_clip * c_loss

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss.item(), infonce_loss.item(), c_loss.item()

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
    #-------------注释隔离------------
    # for i in range(BERT_model.config.num_hidden_layers):
    #     BERT_model.encoder.layer[i].attention.self = BertSelfAttentionWithLogits(BERT_model.config)
    #-------------注释隔离------------
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
                               save_json=args.save_json, 
                               attention_threshold=0.0004, 
                               lr_dict=lr_dict,
                               tokenizer=tokenizer, 
                               model=BERT_model, 
                               device=device)
    # 注意力阈值的话，2048个token，每个token的平均注意力应该是1/2048约等于0.000488
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
    npz_file = npz_files[0]
    # 加载指定的单个 npz 文件
    dataset = ST_COMMDataset(
        token_patch_npz_path=npz_file,
        graph_npz_path=npz_file.replace(".npz", "_graph_data.npz"),
        event_csv_path=npz_file.replace(".npz", "_lr.csv")
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: comm_collate_fn(batch, tokenizer=tokenizer)
    )

    # 记录所有 epoch 的损失
    epoch_train_loss, epoch_infonce_loss, epoch_contrast_loss = [], [], []

    # 使用 tqdm 包裹 epochs 循环，显示训练进度
    for epoch in tqdm(range(args.epochs), desc="Epoch Progress"):
        total_loss, total_infonce, total_contrast = 0.0, 0.0, 0.0
        total_batches = 0

        for batch in dataloader:
            train_loss, infonce_loss, contrast_loss = train_one_sample(
                model, batch, optimizer, scheduler, device
            )

            total_loss += train_loss
            total_infonce += infonce_loss
            total_contrast += contrast_loss
            total_batches += 1

        avg_loss = total_loss / total_batches
        avg_infonce = total_infonce / total_batches
        avg_contrast = total_contrast / total_batches

        logging.info(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}, InfoNCE: {avg_infonce:.4f}, Contrast: {avg_contrast:.4f}")

        epoch_train_loss.append(avg_loss)
        epoch_infonce_loss.append(avg_infonce)
        epoch_contrast_loss.append(avg_contrast)

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{args.output_dir}/ST_COMM_epoch{epoch+1}.pth")

    # 绘制损失曲线
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, args.epochs + 1), epoch_train_loss, label="Total Loss")
    plt.plot(range(1, args.epochs + 1), epoch_infonce_loss, label="InfoNCE Loss")
    plt.plot(range(1, args.epochs + 1), epoch_contrast_loss, label="Contrast Loss")
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