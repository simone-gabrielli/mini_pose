import torch
import torch.nn as nn

from pose.registry import register_loss


@register_loss("heatmap_3d_mse")
class Heatmap3DMSELoss(nn.Module):
    """MSE loss for volumetric 3D heatmaps.

    Expects predictions and targets shaped as (B, K, D, H, W).
    Optional visibility mask of shape (B, K) can zero-out invisible keypoints.
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(reduction="mean")

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, visible=None):
        if visible is not None:
            vis = visible.view(visible.size(0), visible.size(1), 1, 1, 1)
            preds = preds * vis
            targets = targets * vis
        return self.criterion(preds, targets)
