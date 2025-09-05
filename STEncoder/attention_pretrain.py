import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import logging
import random
from tqdm import tqdm
from transformers import BertConfig, BertForMaskedLM
import glob
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from model import ViT
from tokenizer import GeneTokenizer
from accelerate import Accelerator
import matplotlib.pyplot as plt
from STEncoder_data_utils import STDataProcessor, MaskedLanguageModelingDataset, MaskedImageModelingDataset, mlm_collate_fn, mim_collate_fn

# 设置日志
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 创建一个文件处理器（FileHandler），并将日志写入文件
file_handler = logging.FileHandler('/home/maweicheng/ST_Graduation_Project/STEncoder/train.log')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_formatter)

# 创建一个流处理器（StreamHandler），并将日志输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(console_formatter)

# 将处理器添加到日志记录器
logger.addHandler(file_handler)
logger.addHandler(console_handler)

#=============================BERT============================
class STBertPretrainer(nn.Module):
    """BERT掩码语言模型预训练器"""
    
    def __init__(self, vocab_size=None, hidden_size=768, num_hidden_layers=12, 
                 num_attention_heads=12, intermediate_size=3072, hidden_dropout_prob=0.1, 
                 attention_probs_dropout_prob=0.1, max_position_embeddings=1024):
        super(STBertPretrainer, self).__init__()
        
        # 创建BERT配置
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=1
        )
        
        # 创建BERT掩码语言模型
        self.bert_mlm = BertForMaskedLM(config)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.bert_mlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def save_pretrained(self, save_path):
        """保存预训练模型"""
        os.makedirs(save_path, exist_ok=True)
        self.bert_mlm.save_pretrained(save_path)
        logging.info(f"BERT MLM model saved to {save_path}")

def train_bert_mlm(model, train_loader, accelerator, args):
    if accelerator.is_main_process:
        logging.info("Starting BERT MLM training...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.bert_lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.bert_epochs
    )

    # 让 Accelerator 管理模型/优化器/数据
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    best_train_loss = float('inf')
    train_loss_list = []
    for epoch in range(args.bert_epochs):
        # ----------- 1. Train ----------
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch+1}/{args.bert_epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            optimizer.zero_grad()
            with accelerator.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
            accelerator.backward(loss) 
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        train_loss /= len(train_loader)

        scheduler.step()

        train_loss_list.append(train_loss)

        if accelerator.is_main_process and (epoch + 1) % 5 == 0:
            logging.info(f"Epoch {epoch+1}/{args.bert_epochs} - Train Loss: {train_loss:.4f}")

        # ✅ 只在 rank=0 保存
        if accelerator.is_main_process and train_loss < best_train_loss:
            best_train_loss = train_loss
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(os.path.join(args.output_dir, f"bert_best"))
            logging.info(f"Best model saved with train loss: {best_train_loss:.4f}")

        if accelerator.is_main_process and (epoch + 1) % 20 == 0:
            unwrapped_model = accelerator.unwrap_model(model)
            save_path = os.path.join(args.output_dir, f"bert_mlm_epoch_{epoch+1}-train_{train_loss:.4f}")
            unwrapped_model.save_pretrained(save_path)
            logging.info(f"✅ Epoch {epoch+1}: checkpoint saved")

    # ✅ 只在主进程画图 & 保存
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(os.path.join(args.output_dir, "bert_mlm_final"))

        plt.figure(figsize=(8,6))
        plt.plot(range(1, args.bert_epochs+1), train_loss_list, label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("BERT MLM Training Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, "bert_mlm_loss_curve.png"), dpi=150)
        plt.close()
        logging.info(f"✅ Loss curve saved -> {os.path.join(args.output_dir, 'bert_mlm_loss_curve.png')}")
#=============================VIT============================
class STVisionPretrainer(nn.Module):
    """VIT掩码图像建模预训练器"""
    def __init__(self, image_size=32, patch_size=4, dim=768, depth=12, 
                 heads=12, mlp_dim=3072, channels=3, dim_head=64, 
                 dropout=0.1, emb_dropout=0.1):
        super(STVisionPretrainer, self).__init__()
        
        # 创建ViT编码器
        self.vit = ViT(
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=channels,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout
        )
        
        # 创建patch重建器
        self.patch_reconstruction = PatchReconstruction(dim, patch_size, channels)
        
        # 保存参数
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
    
    def forward(self, masked_patches):
        # 使用ViT编码掩码图像
        features = self.vit(masked_patches)
        
        # 去掉CLS token
        patch_features = features[:, 1:]
        
        # 重建patch
        reconstructed_patches = self.patch_reconstruction(patch_features)
        
        # 重组为完整图像
        batch_size = masked_patches.shape[0]
        num_patches_h = num_patches_w = self.image_size // self.patch_size
        
        # 将重建的patch重组为图像
        reconstructed_images = torch.zeros(
            batch_size, self.channels, self.image_size, self.image_size,
            device=masked_patches.device
        )
        
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                patch_idx = i * num_patches_w + j
                h_start = i * self.patch_size
                h_end = (i + 1) * self.patch_size
                w_start = j * self.patch_size
                w_end = (j + 1) * self.patch_size
                
                reconstructed_images[:, :, h_start:h_end, w_start:w_end] = reconstructed_patches[:, patch_idx]
        
        return reconstructed_images
    
    def save_pretrained(self, save_path):
        """保存预训练模型"""
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_path, "vit_mim.pt"))
        logging.info(f"ViT MIM model saved to {save_path}")

class PatchReconstruction(nn.Module):
    """用于重建被掩码的图像patch的模块"""
    
    def __init__(self, dim, patch_size, channels=3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(dim, patch_size * patch_size * channels),
            Rearrange('b n (p1 p2 c) -> b n c p1 p2', p1=patch_size, p2=patch_size, c=channels)
        )
    
    def forward(self, x):
        return self.decoder(x)

def train_vit_mim(model, train_loader, accelerator, args):
    """训练ViT掩码图像建模（支持Accelerate）"""
    if accelerator.is_main_process:
        logging.info("🚀 Starting ViT MIM training...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.vit_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.vit_epochs)

    # 让 Accelerator 管理模型/优化器/数据
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    criterion = torch.nn.MSELoss()
    best_train_loss = float("inf")

    train_loss_list = []

    for epoch in range(args.vit_epochs):
        # ---------------- 1. Training ----------------
        model.train()
        train_loss = 0.0
        if accelerator.is_local_main_process:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.vit_epochs}")
        else:
            progress_bar = train_loader  # 非主进程不显示

        for batch in progress_bar:
            masked_patches = batch["masked_patches"]
            original_patches = batch["original_patches"]
            mask = batch["mask"]

            reconstructed_images = model(masked_patches)

            loss = 0
            bsz, channels, height, width = original_patches.shape
            patch_size = args.patch_size
            num_patches_h = height // patch_size
            num_patches_w = width // patch_size

            # 逐 patch 计算损失（mask=1的才算）
            for i in range(bsz):
                for h in range(num_patches_h):
                    for w in range(num_patches_w):
                        if mask[i, h, w]:
                            hs, he = h * patch_size, (h + 1) * patch_size
                            ws, we = w * patch_size, (w + 1) * patch_size
                            orig_patch = original_patches[i, :, hs:he, ws:we]
                            recon_patch = reconstructed_images[i, :, hs:he, ws:we]
                            loss += criterion(recon_patch, orig_patch)

            num_masked = mask.sum().item()
            if num_masked > 0:
                loss /= num_masked

            optimizer.zero_grad()
            accelerator.backward(loss) 
            optimizer.step()

            train_loss += loss.item()
            if accelerator.is_local_main_process:
                progress_bar.set_postfix({"loss": loss.item()})

        train_loss /= len(train_loader)

        scheduler.step()
        train_loss_list.append(train_loss)
        if accelerator.is_main_process:
            logging.info(f"[Epoch {epoch+1}/{args.vit_epochs}] Train: {train_loss:.4f}")

        # ✅ 保存最好模型（只在主进程）
        if accelerator.is_main_process and train_loss < best_train_loss:
            best_train_loss = train_loss
            model_dir = os.path.join(args.output_dir, f"vit_best")
            logging.info(f"best train loss: {best_train_loss} Saving")
            accelerator.unwrap_model(model).save_pretrained(model_dir)

        # ✅ 定期保存 checkpoint
        if accelerator.is_main_process and (epoch + 1) % 20 == 0:
            ckpt_dir = os.path.join(args.output_dir, f"vit_epoch_{epoch+1}_train_{train_loss:.4f}")
            accelerator.unwrap_model(model).save_pretrained(ckpt_dir)
            logging.info(f"Epoch:{epoch + 1 } Saved")

    # ✅ 训练完成后保存最终模型
    if accelerator.is_main_process:
        logging.info("🎉 Training Done! Saved final model.")

        # ✅ 画 Loss 曲线（只在主进程）
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, args.vit_epochs + 1), train_loss_list, label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("ViT MIM Training Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, "vit_mim_loss_curve.png"), dpi=150)
        plt.close()
        logging.info("✅ Loss curve saved.")

def main():
    parser = argparse.ArgumentParser(description="ST Attention Pretraining")
    
    # 通用参数
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录路径")
    parser.add_argument("--model_type", type=str, default="both", choices=["bert", "vit", "both"], 
                        help="要预训练的模型类型")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器的工作线程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--vocab_file", type=str, help="词典路径")

    # BERT MLM参数
    parser.add_argument("--bert_lr", type=float, default=5e-5, help="BERT学习率")
    parser.add_argument("--bert_epochs", type=int, default=10, help="BERT训练轮数")
    parser.add_argument("--bert_hidden_size", type=int, default=384, help="BERT隐藏层大小")
    parser.add_argument("--bert_num_hidden_layers", type=int, default=6, help="BERT隐藏层数量")
    parser.add_argument("--bert_num_attention_heads", type=int, default=6, help="BERT注意力头数量")
    parser.add_argument("--bert_intermediate_size", type=int, default=1536, help="BERT中间层大小")
    parser.add_argument("--max_position_embeddings", type=int, default=2048, help="最大位置嵌入数")
    
    # VIT MIM参数
    parser.add_argument("--vit_lr", type=float, default=1e-4, help="VIT学习率")
    parser.add_argument("--vit_epochs", type=int, default=10, help="VIT训练轮数")
    parser.add_argument("--image_size", type=int, default=32, help="图像大小")
    parser.add_argument("--patch_size", type=int, default=4, help="patch大小")
    parser.add_argument("--vit_hidden_size", type=int, default=384, help="VIT隐藏层大小")
    parser.add_argument("--vit_num_hidden_layers", type=int, default=6, help="VIT隐藏层数量")
    parser.add_argument("--vit_num_attention_heads", type=int, default=6, help="VIT注意力头数量")
    parser.add_argument("--vit_mlp_dim", type=int, default=1536, help="VIT MLP维度")
    parser.add_argument("--image_channels", type=int, default=3, help="图像通道数")
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    logging.info("Loading data...")
    data_processor = STDataProcessor()

    # 一次性加载整个目录
    data = data_processor.load_npz_data(args.data_dir, max_tokens_per_cell=args.max_position_embeddings)

    all_tokens = data.get("tokens", [])
    all_patches = data.get("patch", [])

    logging.info(f"Loaded {len(all_tokens)} token sequences and {len(all_patches)} patches")

    train_tokens = all_tokens
    train_patches = all_patches
    
    accelerator = Accelerator()
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO)
        
    # 创建 tokenizer
    tokenizer = GeneTokenizer(vocab_file=args.vocab_file, max_length=args.max_position_embeddings)
    
    # 训练 BERT MLM
    if args.model_type in ["bert", "both"]:
        logging.info("Preparing BERT MLM training...")

        train_dataset = MaskedLanguageModelingDataset(train_tokens)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=lambda batch: mlm_collate_fn(batch, tokenizer)
        )
        # 创建模型
        bert_model = STBertPretrainer(
            vocab_size=tokenizer.get_vocab_size(),
            hidden_size=args.bert_hidden_size,
            num_hidden_layers=args.bert_num_hidden_layers,
            num_attention_heads=args.bert_num_attention_heads,
            intermediate_size=args.bert_intermediate_size,
            max_position_embeddings=args.max_position_embeddings
        )

        # 训练模型
        train_bert_mlm(bert_model, train_loader, accelerator, args)
    
    # 训练VIT MIM
    if args.model_type in ["vit", "both"]:
        logging.info("Preparing ViT MIM training...")
        
        # 创建数据集
        train_dataset = MaskedImageModelingDataset(train_patches)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=mim_collate_fn
        )
        
        # 创建模型
        vit_model = STVisionPretrainer(
            image_size=args.image_size,
            patch_size=args.patch_size,
            dim=args.vit_hidden_size,
            depth=args.vit_num_hidden_layers,
            heads=args.vit_num_attention_heads,
            mlp_dim=args.vit_mlp_dim,
            channels=args.image_channels
        )
        
        # 训练模型
        train_vit_mim(vit_model, train_loader, accelerator, args)
    
    logging.info("Pretraining completed!")

if __name__ == "__main__":
    main()