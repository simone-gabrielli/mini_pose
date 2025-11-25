# pose/losses/depth_mse.py
import torch
import torch.nn as nn
from pose.registry import register_loss

@register_loss("depth_mse")
class DepthMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(reduction="mean")

    def forward(self, preds, targets):
        """
        preds: (B, 1, H, W)
        targets: (B, 1, H, W)
        """
        return self.criterion(preds, targets)
