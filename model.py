import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from transformers import BertModel, BertConfig
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torchvision.models as models
# -----------------------------------ResNet50-----------------------------------
class ResNet50Encoder(nn.Module):
    def __init__(self, input_channels=3, output_dim=384, pretrained=True):
        super(ResNet50Encoder, self).__init__()
        
        # 加载预训练的ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # 如果输入通道数不是3，修改第一层
        if input_channels != 3:
            self.resnet.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # 移除原来的分类层
        self.resnet.fc = nn.Identity()
        
        # 添加适配层，将ResNet的2048维输出映射到指定维度
        self.adaptor = nn.Sequential(
            nn.Linear(2048, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.output_dim = output_dim
        
    def forward(self, img):
        """
        img: [B, N, C, H, W] - 批次中每个样本有N个图像
        返回: [B, N, output_dim] - 每个图像的特征表示
        """
        B, N, C, H, W = img.shape
        
        # 重塑为 [B*N, C, H, W] 以便批量处理
        img = img.view(B * N, C, H, W)
        
        # 通过ResNet提取特征
        features = self.resnet(img)  # [B*N, 2048]
        
        # 通过适配层
        features = self.adaptor(features)  # [B*N, output_dim]
        
        # 重塑回 [B, N, output_dim]
        features = features.view(B, N, self.output_dim)
        
        return features

# -----------------------------------保留的辅助类-----------------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# -----------------------------------BERT-----------------------------------
class BERTEncoder(nn.Module):
    def __init__(self,
                 bert_model=None,
                 vocab_size=None,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=1024):
        super(BERTEncoder, self).__init__()

        if bert_model:  
            self.bert = BertModel.from_pretrained(bert_model, local_files_only=True)
            if self.bert.config.max_position_embeddings < max_position_embeddings:
                self.bert.resize_position_embeddings(max_position_embeddings)
        else:
            config = BertConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=hidden_dropout_prob,
                max_position_embeddings=max_position_embeddings,
                type_vocab_size=1
            )
            self.bert = BertModel(config)

    def forward(self, input_ids, attention_mask=None, output_attentions=False):
        B, N, L = input_ids.shape  # B=16, N=11, L=2048
        input_ids = input_ids.view(B * N, L)  # -> [16*11, 2048]
        attention_mask = attention_mask.view(B * N, L)
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        text_emb = outputs.last_hidden_state  # [B*N, L, H]
        text_emb = text_emb[:, 0, :]  # 如果你只取 [CLS] 表达，得到 [B*N, H]
        text_emb = text_emb.view(B, N, -1)  # reshape 回 [B, N, H]
        return text_emb
 
# -----------------------------------融合模块-----------------------------------
class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context):
        q = self.to_q(x)
        kv = self.to_kv(context).chunk(2, dim = -1)
        k, v = kv
        
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class MultiModalFusion(nn.Module):
    def __init__(self, dim=384, heads=6, dim_head=64, dropout=0.1, depth=2):
        super().__init__()
        self.depth = depth
        self.dim = dim
        
        # 共享的层归一化
        self.norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth * 2)])
        
        # 文本到图像和图像到文本的交叉注意力
        self.cross_attention = nn.ModuleList([
            CrossAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            for _ in range(depth * 2)
        ])
        
        # 共享的前馈网络
        self.ffn = nn.ModuleList([
            FeedForward(dim, dim*4, dropout=dropout)
            for _ in range(depth * 2)
        ])
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, text_features, image_features):
        text_out = text_features
        image_out = image_features
        
        for i in range(self.depth):
            # 文本到图像的注意力
            text_out = self.norm[i*2](text_out)
            text_attended = self.cross_attention[i*2](text_out, image_out)
            text_out = text_out + text_attended
            text_out = text_out + self.ffn[i*2](self.norm[i*2+1](text_out))
            
            # 图像到文本的注意力
            image_out = self.norm[i*2](image_out)
            image_attended = self.cross_attention[i*2+1](image_out, text_out)
            image_out = image_out + image_attended
            image_out = image_out + self.ffn[i*2+1](self.norm[i*2+1](image_out))
        
        # 融合特征
        fused_features = self.fusion(torch.cat([text_out, image_out], dim=-1))
        return fused_features
    
# -----------------------------------Graphformer-----------------------------------
class GraphomerEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=256, num_heads=4, num_layers=4):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, num_heads)

        self.layers = nn.ModuleList([
            GraphomerLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_feats, edge_index, edge_attr):
        """
        node_feats: [B, N, D] (e.g. 16, 11, 512)
        edge_index: [B, 2, E] (e.g. 16, 2, 110)
        edge_attr: [B, E, A] (e.g. 16, 110, 2)
        """
        B, N, _ = node_feats.shape
        node_feats = self.node_proj(node_feats)  # [B, N, H]
        edge_bias = self.edge_proj(edge_attr)    # [B, E, num_heads]

        attn_scores = [] 
        attn_logistic = []
        for layer in self.layers:
            node_feats, attn, attn_raw = layer(node_feats, edge_index, edge_bias)
            attn_scores.append(attn)
            attn_logistic.append(attn_raw)

        return self.norm(node_feats), attn_scores, attn_logistic

class GraphomerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_bias):
        """
        x: [B, N, H]
        edge_index: [B, 2, E]
        edge_bias: [B, E, num_heads]
        """
        B, N, H = x.shape
        E = edge_index.shape[-1]
        qkv = self.qkv_proj(x)  # [B, N, 3H]
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [B, N, H]

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, N, d]
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, N, d]
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, N, d]

        out = torch.zeros_like(v)  # [B, h, N, d]

        for b in range(B):
            src, tgt = edge_index[b]  # Each: [E]
            q_b = q[b]  # [h, N, d]
            k_b = k[b]
            v_b = v[b]

            attn_score = (q_b[:, tgt] * k_b[:, src]).sum(-1) / (self.head_dim ** 0.5)  # [h, E]
            attn_score = attn_score + edge_bias[b].transpose(0, 1)  # [h, E]
            attn_weight = F.softmax(attn_score, dim=-1)  # [h, E]

            for i in range(E):
                h_idx = attn_weight[:, i].unsqueeze(-1)  # [h, 1]
                v_msg = v_b[:, src[i]] * h_idx  # [h, d]
                out[b, :, tgt[i]] += v_msg

        out = out.transpose(1, 2).contiguous().view(B, N, H)
        out = self.out_proj(out)
        return self.norm(x + self.dropout(out)),attn_weight,attn_score
# -----------------------------------总模型-----------------------------------
class ST_COMM(nn.Module):
    def __init__(self,
                 bert_model=None,
                 vocab_size=None,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=1024,
                 image_channels=3,
                 resnet_pretrained=True,
                 graphormer_layers=2,
                 graphormer_heads=4):
        super(ST_COMM, self).__init__()

        # ==== 1) BERT编码器 (基因表达序列) ====
        self.bert = BERTEncoder(bert_model=bert_model,
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_dropout_prob=hidden_dropout_prob,
                max_position_embeddings=max_position_embeddings)

        # ==== 2) ResNet50编码器 (图像特征提取) ====
        self.resnet = ResNet50Encoder(
            input_channels=image_channels,
            output_dim=hidden_size,
            pretrained=resnet_pretrained
        )

        # ==== 3) 多模态融合 (Cross-Attention + CLS Concat) ====
        self.fusion_module = MultiModalFusion(
            dim=hidden_size,
            heads=num_attention_heads,
            dim_head=hidden_size // num_attention_heads,
            dropout=hidden_dropout_prob
        )

        # ==== 4) Graphormer 图模块 ====
        self.graphormer = GraphomerEncoder(
            node_dim=hidden_size, 
            edge_dim=2,
            hidden_dim=hidden_size, 
            num_heads=graphormer_heads,
            num_layers=graphormer_layers
        )

        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids, attention_mask, images, edge_index, edge_attr, num_nodes=None):
        # ==== Step 1: BERT编码 ====
        text_features = self.bert(input_ids, attention_mask)
        # text_features shape: torch.Size([16, 11, 384])
        
        # ==== Step 2: ResNet50编码 ====
        image_features = self.resnet(images)
        # image_features shape: torch.Size([16, 11, 384])
        
        # ==== Step 3: Cross-Attention + 融合 ====
        fusion_emb = self.fusion_module(text_features, image_features)
        # fusion_emb shape: torch.Size([16, 11, 384])
        
        # ==== Step 4: Graphormer 编码图结构 ====
        node_emb, attn_scores, attn_logistic = self.graphormer(
            fusion_emb,
            edge_index,
            edge_attr
        )

        node_emb = self.final_norm(node_emb)
        
        return node_emb, attn_scores, attn_logistic, text_features, image_features, fusion_emb

def clip_loss(text_features, image_features, temperature=0.07):
    """
    节点级别的图像-文本对齐损失
    让每个spot的图像特征和文本特征对齐，不涉及邻居
    text_features: [B, N, D] - 每个spot的文本特征
    image_features: [B, N, D] - 每个spot的图像特征
    """
    B, N, D = text_features.shape
    
    # L2归一化
    text_features = F.normalize(text_features, dim=-1)  # [B, N, D]
    image_features = F.normalize(image_features, dim=-1)  # [B, N, D]
    
    # 计算每个spot内部的相似度（只对齐自己的图像和文本）
    # 重塑为 [B*N, D] 来批量计算
    text_flat = text_features.view(B*N, D)  # [B*N, D]
    image_flat = image_features.view(B*N, D)  # [B*N, D]
    
    # 计算相似度矩阵 [B*N, B*N]
    logits_per_text = torch.matmul(text_flat, image_flat.T) / temperature
    logits_per_image = logits_per_text.T
    
    # 标签：每个spot只与自己对应的模态特征对齐
    labels = torch.arange(B*N, device=text_features.device)
    
    loss_text = F.cross_entropy(logits_per_text, labels)
    loss_image = F.cross_entropy(logits_per_image, labels)
    
    return (loss_text + loss_image) / 2

def info_nce_loss(graph_emb_before, graph_emb_after, temperature=0.2):
    """
    图级别的对比学习损失
    比较图神经网络前后的图级别表示，增强图结构学习
    graph_emb_before: [B, D] - 图神经网络前的图级别特征
    graph_emb_after: [B, D] - 图神经网络后的图级别特征
    """
    B, D = graph_emb_before.shape
    
    # L2归一化
    z1 = F.normalize(graph_emb_before, dim=-1)  # [B, D]
    z2 = F.normalize(graph_emb_after, dim=-1)   # [B, D]
    
    # 计算相似度矩阵 [B, B]
    sim_matrix = torch.matmul(z1, z2.T) / temperature
    
    # 标签：每个图只与自己的增强版本对齐
    labels = torch.arange(B, device=z1.device)
    
    # 双向损失
    loss_12 = F.cross_entropy(sim_matrix, labels)
    loss_21 = F.cross_entropy(sim_matrix.T, labels)
    
    return (loss_12 + loss_21) / 2