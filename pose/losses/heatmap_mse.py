# pose/losses/heatmap_mse.py

import torch
import torch.nn as nn
from pose.registry import register_loss


@register_loss("heatmap_mse")
class HeatmapMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(reduction="mean")

    def forward(self, preds, targets, visible=None):
        """
        preds: (B, K, H, W)
        targets: (B, K, H, W)
        visible: (B, K) or None
        """
        if visible is not None:
            vis = visible.view(visible.size(0), visible.size(1), 1, 1)
            preds = preds * vis
            targets = targets * vis
        return self.criterion(preds, targets)
