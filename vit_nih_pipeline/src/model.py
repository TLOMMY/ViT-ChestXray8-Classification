"""
Vision Transformer (ViT) model implementation
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Image -> Patch Embedding"""

    def __init__(self, img_size=64, patch_size=16, num_channels=1, embed_dim=32):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            num_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        # (B, C, H, W) -> (B, embed_dim, H//patch, W//patch) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    """Transformer Encoder Block"""

    def __init__(self, embed_dim=32, num_heads=4, mlp_dim=64, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Attention + Residual
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out

        # MLP + Residual
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer for NIH Chest X-ray binary classification"""

    def __init__(self, config):
        super().__init__()
        m_cfg = config.model

        # 1. Patch Embedding
        self.patch_embedding = PatchEmbedding(
            m_cfg.img_size, m_cfg.patch_size, 
            m_cfg.num_channels, m_cfg.embed_dim
        )

        # 2. CLS Token & Positional Embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, m_cfg.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.randn(1, m_cfg.num_patches + 1, m_cfg.embed_dim)
        )
        self.dropout = nn.Dropout(m_cfg.dropout)

        # 3. Transformer Blocks
        self.transformer = nn.Sequential(*[
            TransformerBlock(m_cfg.embed_dim, m_cfg.num_heads, m_cfg.mlp_dim, m_cfg.dropout)
            for _ in range(m_cfg.transformer_units)
        ])

        # 4. Classification Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(m_cfg.embed_dim),
            nn.Linear(m_cfg.embed_dim, m_cfg.num_classes)
        )

    def forward(self, x):
        B = x.size(0)

        # Patch embedding
        x = self.patch_embedding(x)  # (B, num_patches, embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer
        x = self.transformer(x)

        # CLS token classification
        x = x[:, 0]
        x = self.mlp_head(x)
        return x


def build_model(config):
    """Build model and move to device"""
    device = torch.device(config.train.device if torch.cuda.is_available() else "cpu")
    model = VisionTransformer(config).to(device)
    return model, device
