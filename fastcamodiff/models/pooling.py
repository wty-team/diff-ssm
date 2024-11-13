import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalPooling(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim

        # Prediction networks
        self.predict_net = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1)
        )

        # Update networks
        self.update_net = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1)
        )

        # Merge network for combining features
        self.merge_net = nn.Sequential(
            nn.Conv1d(self.hidden_dim * 2, self.hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        )

    def split_sequence(self, x, x_loc):
        """Split sequence into two subsequences"""
        B, L, D = x.shape
        mid = L // 2

        return (x[:, :mid], x[:, mid:]), (x_loc[:, :mid], x_loc[:, mid:])

    def forward(self, x, x_loc=None):
        """
        Args:
            x: Input tensor [B, L, D]
            x_loc: Location encoding [B, L, D]
        """
        # Handle input without location encoding
        if x_loc is None:
            x_loc = torch.zeros_like(x)

        # Convert to channel-first format for convolutions
        x = x.transpose(1, 2)
        x_loc = x_loc.transpose(1, 2)

        # Split sequences
        (x_t, x_0), (x_loc_t, x_loc_0) = self.split_sequence(x, x_loc)

        # Predict step
        pred = self.predict_net(x_t)
        diff = x_0 - pred + (x_loc_t - x_loc_0)

        # Update step
        update = self.update_net(diff)
        s = x_t + update

        # Merge features
        merged = self.merge_net(torch.cat([s, diff], dim=1))

        # Convert back to sequence format
        return merged.transpose(1, 2)

    def inverse(self, s, d):
        """Inverse transform for reconstruction"""
        # Convert to channel-first format
        s = s.transpose(1, 2)
        d = d.transpose(1, 2)

        # Predict and update steps
        pred = self.predict_net(s)
        update = self.update_net(d)

        # Reconstruct original
        x_t = s - update
        x_0 = pred + d

        # Concatenate and merge
        x = torch.cat([x_t, x_0], dim=2)

        return x.transpose(1, 2)
