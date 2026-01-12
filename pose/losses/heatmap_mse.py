# pose/losses/heatmap_mse.py

import torch
import torch.nn as nn
from pose.registry import register_loss


@register_loss("heatmap_mse")
class HeatmapMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Keep a mean reduction for the default path (no sample weights).
        self.criterion = nn.MSELoss(reduction="mean")

    def forward(self, preds, targets, visible=None, sample_weight=None):
        """
        preds: (B, K, H, W)
        targets: (B, K, H, W)
        visible: (B, K) or None
        sample_weight: (B,) or None
        """
        if sample_weight is None:
            if visible is not None:
                vis = visible.view(visible.size(0), visible.size(1), 1, 1)
                preds = preds * vis
                targets = targets * vis
            return self.criterion(preds, targets)

        # Weighted path: compute per-sample mean MSE then aggregate.
        if visible is not None:
            vis = visible.view(visible.size(0), visible.size(1), 1, 1)
            preds = preds * vis
            targets = targets * vis

        err = (preds - targets) ** 2
        per_sample = err.view(err.size(0), -1).mean(dim=1)
        sw = sample_weight.to(device=per_sample.device, dtype=per_sample.dtype).view(-1)
        denom = sw.sum().clamp(min=1e-6)
        return (per_sample * sw).sum() / denom
