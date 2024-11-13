import torch
import numpy as np


def calculate_s_measure(pred, gt, alpha=0.5):
    """Calculate S-measure (Structure measure)"""
    y = gt.mean()
    if y == 0:
        x = pred.mean()
        Q = 1.0 - x
    elif y == 1:
        x = pred.mean()
        Q = x
    else:
        # Object region
        gt_obj = gt == 1
        pred_obj = pred * gt_obj
        O_region = object_region(pred_obj, gt_obj)

        # Background region
        gt_bg = gt == 0
        pred_bg = pred * gt_bg
        B_region = background_region(pred_bg, gt_bg)

        Q = alpha * O_region + (1 - alpha) * B_region

    return Q


def object_region(pred_obj, gt_obj):
    """Calculate object region score"""
    x = pred_obj.sum() / (gt_obj.sum() + 1e-8)
    sigma_x = pred_obj.std()
    score = 2 * x / (x * x + 1 + sigma_x + 1e-8)
    return score


def background_region(pred_bg, gt_bg):
    """Calculate background region score"""
    x = pred_bg.sum() / (gt_bg.sum() + 1e-8)
    sigma_x = pred_bg.std()
    score = 2 * (1 - x) / ((1 - x) * (1 - x) + 1 + sigma_x + 1e-8)
    return score


def calculate_f_measure(pred, gt, beta2=0.3):
    """Calculate F-measure"""
    # Calculate precision and recall
    pred_binary = (pred > 0.5).float()
    tp = (pred_binary * gt).sum()
    fp = (pred_binary * (1 - gt)).sum()
    fn = ((1 - pred_binary) * gt).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    # Calculate F-measure
    f_measure = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)

    return f_measure


def calculate_mae(pred, gt):
    """Calculate Mean Absolute Error"""
    return torch.abs(pred - gt).mean()


def calculate_metrics(pred, gt):
    """Calculate all metrics"""
    return {
        's_measure': calculate_s_measure(pred, gt).item(),
        'f_measure': calculate_f_measure(pred, gt).item(),
        'mae': calculate_mae(pred, gt).item()
    }