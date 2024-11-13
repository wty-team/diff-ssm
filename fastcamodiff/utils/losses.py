import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred_noise, target_noise, mask, beta1, beta2):
        """
        Calculate diffusion loss with region-specific weighting
        Args:
            pred_noise: Predicted noise
            target_noise: Target noise
            mask: Binary mask indicating object regions
            beta1: Noise level for object region
            beta2: Noise level for background region
        """
        # Separate object and background regions
        object_loss = self.mse(pred_noise * mask, target_noise * mask)
        background_loss = self.mse(pred_noise * (1 - mask),
                                   target_noise * (1 - mask))

        # Weight losses by respective noise levels
        loss = beta1 * object_loss + beta2 * background_loss

        return loss


def kl_divergence(dist1, dist2):
    """Calculate KL divergence between two normal distributions"""
    mean1, std1 = dist1
    mean2, std2 = dist2

    return torch.log(std2 / std1) + (std1 ** 2 + (mean1 - mean2) ** 2) / (2 * std2 ** 2) - 0.5


class TotalLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.diffusion_loss = DiffusionLoss()
        self.config = config

    def forward(self, pred_noise, target_noise, mask, beta1, beta2, x_T):
        """
        Calculate total loss including sampling loss, VLB loss and difference loss
        """
        # Sampling loss
        sample_loss = self.diffusion_loss(pred_noise, target_noise, mask,
                                          beta1, beta2)

        # VLB loss (KL divergence)
        vlb_loss = kl_divergence(
            (x_T, torch.sqrt(beta1 + beta2)),
            (torch.zeros_like(x_T), torch.ones_like(x_T))
        ).mean()

        # Difference loss (inverse KL divergence between object and background)
        obj_dist = (pred_noise * mask, torch.sqrt(beta1))
        bg_dist = (pred_noise * (1 - mask), torch.sqrt(beta2))
        diff_loss = 1.0 / (kl_divergence(obj_dist, bg_dist).mean() + 1e-6)

        # Total loss
        total_loss = sample_loss + vlb_loss + diff_loss

        return total_loss, {
            'sample_loss': sample_loss.item(),
            'vlb_loss': vlb_loss.item(),
            'diff_loss': diff_loss.item()
        }