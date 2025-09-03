import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from transformers import BertModel, BertConfig
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# -----------------------------------VIT-----------------------------------
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, *args, **kwargs):
        args = list(args)
        args[0] = self.norm(args[0])
        return self.fn(*args, **kwargs)

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

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size=32, patch_size=4, num_classes=None, dim=384, depth=12, 
                 heads=12, mlp_dim=1536, channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = (image_size, image_size)
        patch_height, patch_width = (patch_size, patch_size)
        
        assert image_height % patch_height == 0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                     p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, img):
        B, N, C, H, W = img.shape  # B=16, N=11, C=3, H=W=32
        img = img.view(B * N, C, H, W)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = x[:, 0]   
        x = x.view(B, N, -1) 
        return x

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
    """多模态融合模块，使用交叉注意力机制融合文本和图像特征"""
    
    def __init__(self, dim=384, heads=6, dim_head=64, dropout=0.1, depth=2):
        super().__init__()
        # 文本到图像的交叉注意力
        self.text_to_image_layers = nn.ModuleList([])
        for _ in range(depth):
            self.text_to_image_layers.append(nn.ModuleList([
                PreNorm(dim, CrossAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, dim*4, dropout=dropout))
            ]))
        
        # 图像到文本的交叉注意力
        self.image_to_text_layers = nn.ModuleList([])
        for _ in range(depth):
            self.image_to_text_layers.append(nn.ModuleList([
                PreNorm(dim, CrossAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, dim*4, dropout=dropout))
            ]))
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, text_features, image_features):
        # 文本到图像的交叉注意力
        text_attended = text_features
        for cross_attn, ff in self.text_to_image_layers:
            text_attended = cross_attn(text_attended, image_features) + text_attended
            text_attended = ff(text_attended) + text_attended
        
        # 图像到文本的交叉注意力
        image_attended = image_features
        for cross_attn, ff in self.image_to_text_layers:
            image_attended = cross_attn(image_attended, text_features) + image_attended
            image_attended = ff(image_attended) + image_attended
        # text_attended shape: torch.Size([16, 11, 384])
        # image_attended shape: torch.Size([16, 11, 384])
        # 融合特征
        fused_features = torch.cat([text_attended, image_attended], dim=2)
        # fused_features shape before fusion: torch.Size([16, 11, 768])
        fused_features = self.fusion(fused_features)
        # fused_features shape: torch.Size([16, 11, 384])
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
        for layer in self.layers:
            node_feats, attn = layer(node_feats, edge_index, edge_bias)
            attn_scores.append(attn)

        return self.norm(node_feats), attn_scores

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
        return self.norm(x + self.dropout(out)),attn_weight
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
                 image_size=32,
                 patch_size=4,
                 image_channels=3,
                 vit_depth=12,
                 vit_heads=12,
                 vit_mlp_dim=3072,
                 graphormer_layers=3,
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

        # ==== 2) ViT编码器 (图像 patch) ====
        self.vit = ViT(
            image_size=image_size,
            patch_size=patch_size,
            dim=hidden_size,
            depth=vit_depth,
            heads=vit_heads,
            mlp_dim=vit_mlp_dim,
            channels=image_channels,
            dropout=hidden_dropout_prob,
            emb_dropout=hidden_dropout_prob
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

    def forward(self, input_ids, attention_mask, images, edge_index, edge_attr, num_nodes=None):
        # ==== Step 1: BERT编码 ====
        text_features = self.bert(input_ids, attention_mask)
        # text_features shape: torch.Size([16, 11, 384])
        # ==== Step 2: ViT编码 ====
        image_features = self.vit(images)
        # image_features shape: torch.Size([16, 11, 384])
        # ==== Step 3: Cross-Attention + 融合 ====
        fusion_emb = self.fusion_module(text_features, image_features)
        # fusion_emb shape: torch.Size([16, 11, 384])
        # ==== Step 4: Graphormer 编码图结构 ====
        node_emb, attn_scores = self.graphormer(
            fusion_emb,
            edge_index,
            edge_attr
        )

        return node_emb, attn_scores, text_features, image_features, fusion_emb

def clip_loss(logits_per_image, logits_per_text):
    labels = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
    loss_img = F.cross_entropy(logits_per_image, labels)
    loss_txt = F.cross_entropy(logits_per_text, labels)
    return (loss_img + loss_txt) / 2

def info_nce_loss(z1, z2, temperature=0.2):
    """
    z1, z2: Tensor of shape [B, N, D]
    对每个 batch 单独计算 N x N 相似度矩阵，并计算 infoNCE
    
    返回：batch 上的平均损失
    """
    B, N, D = z1.size()
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    losses = []
    for b in range(B):
        z1_b = z1[b]  # [N, D]
        z2_b = z2[b]  # [N, D]
        sim_matrix = torch.matmul(z1_b, z2_b.T)  # [N, N]

        pos_sim = torch.diag(sim_matrix)  # 正样本对相似度 [N]

        logits = sim_matrix / temperature

        labels = torch.arange(N, device=z1.device)
        loss = F.cross_entropy(logits, labels)
        losses.append(loss)

    return torch.stack(losses).mean()