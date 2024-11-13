import torch
import torch.nn as nn
import math
from .diff_ssm import DiffSSM
from .dynamic_encoder import DynamicEncodingLayer
from .pooling import BidirectionalPooling


class FastCamoDiff(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Initial processing
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, config.hidden_dim, kernel_size=4, stride=4),
            nn.SiLU(),
            nn.Conv2d(config.hidden_dim, config.hidden_dim, kernel_size=1)
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, (config.img_size // 4) ** 2, config.hidden_dim)
        )

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            DiffSSM(config) for _ in range(5)
        ])

        # Dynamic encoding layers for encoder
        self.encoder_dynamic = nn.ModuleList([
            DynamicEncodingLayer(config) for _ in range(2)
        ])

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            DiffSSM(config) for _ in range(8)
        ])

        # Dynamic encoding layers for decoder
        self.decoder_dynamic = nn.ModuleList([
            DynamicEncodingLayer(config) for _ in range(2)
        ])

        # Bidirectional pooling layers
        self.pool_layers = nn.ModuleList([
            BidirectionalPooling(config) for _ in range(4)
        ])

        # Final prediction head
        self.pred_head = nn.Sequential(
            nn.ConvTranspose2d(config.hidden_dim, config.hidden_dim,
                               kernel_size=4, stride=4),
            nn.SiLU(),
            nn.Conv2d(config.hidden_dim, 1, kernel_size=1)
        )

    def forward(self, x, t, beta1=None, beta2=None):
        """
        Args:
            x: Input image [B, C, H, W]
            t: Timestep
            beta1: Noise level for object region (optional)
            beta2: Noise level for background region (optional)
        """
        # Default noise levels if not provided
        if beta1 is None:
            beta1 = torch.ones_like(t) * self.config.beta1_start
        if beta2 is None:
            beta2 = torch.ones_like(t) * self.config.beta2_start

        # Initial processing
        B = x.shape[0]
        x = self.patch_embed(x)

        # Reshape to sequence
        x = x.flatten(2).transpose(1, 2)
        pe = self.pos_embed.expand(B, -1, -1)

        # Encoder
        skip_connections = []
        for i, (block, pool) in enumerate(zip(self.encoder_blocks, self.pool_layers)):
            if i < 2:  # Apply dynamic encoding for first two blocks
                x, pe = self.encoder_dynamic[i](x, pe, t)
            x = block(x, t, beta1, beta2)
            x = pool(x, pe)
            skip_connections.append(x)

        # Decoder
        for i, (block, pool) in enumerate(zip(self.decoder_blocks, self.pool_layers)):
            if i < 2:  # Apply dynamic encoding for first two blocks
                x, pe = self.decoder_dynamic[i](x, pe, t, skip_connections[-i - 1])
            x = block(x, t, beta1, beta2)
            x = pool(x, pe)
            if i < len(skip_connections):
                x = x + skip_connections[-i - 1]

        # Final prediction
        x = x.transpose(1, 2).reshape(B, self.config.hidden_dim,
                                      self.config.img_size // 4,
                                      self.config.img_size // 4)
        x = self.pred_head(x)

        return torch.sigmoid(x)

    def _get_timestep_embedding(self, timesteps, embedding_dim):
        """Create sinusoidal timestep embeddings"""
        assert len(timesteps.shape) == 1
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb