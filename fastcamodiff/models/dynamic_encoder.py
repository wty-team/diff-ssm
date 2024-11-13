import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DynamicEncodingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.expanded_dim = config.expanded_dim

        # Projections for dynamic encoding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, config.img_size * config.img_size // 16, self.hidden_dim)
        )

        # Recalibration networks
        self.recal_network = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # Combination network
        self.combine_network = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

    def quad_split(self, x, pe):
        """Split patches into quadrants and recalibrate positional encodings"""
        B, N, D = x.shape
        H = W = int(math.sqrt(N))

        # Reshape to spatial dimensions
        x = x.view(B, H, W, D)
        pe = pe.view(B, H, W, D)

        # Split into quadrants
        h_mid = H // 2
        w_mid = W // 2

        # Create sub-patches
        quadrants = [
            (x[:, :h_mid, :w_mid], pe[:, :h_mid, :w_mid]),  # Top-left
            (x[:, :h_mid, w_mid:], pe[:, :h_mid, w_mid:]),  # Top-right
            (x[:, h_mid:, :w_mid], pe[:, h_mid:, :w_mid]),  # Bottom-left
            (x[:, h_mid:, w_mid:], pe[:, h_mid:, w_mid:])  # Bottom-right
        ]

        # Recalibrate positional encodings for each quadrant
        recalibrated = []
        for i, (quad_x, quad_pe) in enumerate(quadrants):
            # Create quadrant indicator
            quad_indicator = torch.full_like(quad_pe[:, :, :, :1], i / 4)

            # Concatenate with original PE
            combined = torch.cat([quad_pe, quad_indicator.expand(-1, -1, -1, D)], dim=-1)

            # Recalibrate
            new_pe = self.recal_network(combined)
            recalibrated.append((quad_x.reshape(B, -1, D), new_pe.reshape(B, -1, D)))

        return recalibrated

    def forward(self, z, pe, t, direct=None):
        """
        Args:
            z: Input features [B, N, D]
            pe: Positional encodings [B, N, D]
            t: Timestep
            direct: Direct features for concatenation (optional)
        """
        B, N, D = z.shape

        # Split into quadrants and recalibrate
        quad_splits = self.quad_split(z, pe)
        forward_features = []
        backward_features = []

        for (quad_x, quad_pe) in quad_splits:
            # Forward pass with Diff-SSM
            fwd = self.diff_ssm_forward(quad_x, t)
            forward_features.append(fwd)

            # Backward pass with Diff-SSM
            bwd = self.diff_ssm_backward(quad_x, t)
            backward_features.append(bwd)

        # Combine forward and backward features
        forward_combined = torch.cat(forward_features, dim=1)
        backward_combined = torch.cat(backward_features, dim=1)

        # If direct features are provided, include them in combination
        if direct is not None:
            combined = self.combine_network(
                torch.cat([forward_combined, backward_combined, direct], dim=-1)
            )
        else:
            combined = self.combine_network(
                torch.cat([forward_combined, backward_combined], dim=-1)
            )

        # Create new positional encoding
        new_pe = self.recalibrate_pe(pe, t)

        return combined, new_pe

    def recalibrate_pe(self, pe, t):
        """Recalibrate positional encoding based on timestep"""
        t_embed = self._get_timestep_embedding(t, self.hidden_dim)
        t_embed = t_embed.unsqueeze(1).expand(-1, pe.size(1), -1)

        combined = torch.cat([pe, t_embed], dim=-1)
        new_pe = self.recal_network(combined)

        return new_pe

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
