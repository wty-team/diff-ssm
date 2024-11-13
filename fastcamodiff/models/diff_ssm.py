import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SSMKernel(nn.Module):
    def __init__(self, h_dim, ssm_dim):
        super().__init__()
        self.h_dim = h_dim
        self.ssm_dim = ssm_dim

        # Learnable parameters
        self.A = nn.Parameter(torch.randn(ssm_dim, ssm_dim))
        self.B = nn.Parameter(torch.randn(ssm_dim, 1))
        self.C = nn.Parameter(torch.randn(1, ssm_dim))

        self.D = nn.Parameter(torch.randn(1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.A, std=0.01)
        nn.init.normal_(self.B, std=0.01)
        nn.init.normal_(self.C, std=0.01)
        nn.init.normal_(self.D, std=0.01)

    def forward(self, L):
        # Compute SSM kernel
        dA = torch.diag_embed(torch.exp(self.A))  # Make A diagonal
        dB = self.B.unsqueeze(-1)
        dC = self.C.unsqueeze(0)

        # Compute power of A
        powers = torch.arange(L).unsqueeze(-1).unsqueeze(-1).to(dA.device)
        power_matrices = torch.matrix_power(dA, powers)

        # Compute kernel
        kernel = torch.einsum('ijk,kl,l->ij', power_matrices, dB, dC.squeeze())
        kernel = kernel + self.D

        return kernel


class DiffSSM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.ssm_dim = config.ssm_dim

        # SSM kernels for forward and backward passes
        self.forward_kernel = SSMKernel(self.hidden_dim, self.ssm_dim)
        self.backward_kernel = SSMKernel(self.hidden_dim, self.ssm_dim)

        # Linear layers
        self.input_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # 1D convolution layers
        self.conv1 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1)

        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

    def forward(self, x, t, beta1, beta2):
        """
        Args:
            x: Input tensor [B, L, D]
            t: Timestep
            beta1: Noise level for object region
            beta2: Noise level for background region
        """
        B, L, D = x.shape

        # Input projection and normalization
        h = self.input_proj(x)
        h = self.norm1(h)

        # 1D convolution branch
        conv_out = self.conv1(h.transpose(1, 2))
        conv_out = F.silu(conv_out)
        conv_out = self.conv2(conv_out)
        conv_out = conv_out.transpose(1, 2)

        # SSM branch
        # Forward pass
        forward_kernel = self.forward_kernel(L)
        h_forward = torch.conv1d(
            h.transpose(1, 2),
            forward_kernel.unsqueeze(0).unsqueeze(0),
            padding=L - 1
        )[:, :, :L].transpose(1, 2)

        # Backward pass
        backward_kernel = self.backward_kernel(L)
        h_backward = torch.conv1d(
            h.transpose(1, 2).flip(dims=[2]),
            backward_kernel.unsqueeze(0).unsqueeze(0),
            padding=L - 1
        )[:, :, :L].flip(dims=[2]).transpose(1, 2)

        # Combine branches with noise level weighting
        t_embed = self._get_timestep_embedding(t, self.hidden_dim)
        noise_scale = torch.sigmoid(t_embed)
        h = h_forward * (beta1 * noise_scale) + h_backward * (beta2 * noise_scale) + conv_out

        # Output projection and residual connection
        h = self.output_proj(h)
        h = self.norm2(h)

        return x + h

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