from typing import Optional

import torch
import torch.nn as nn

from pose.registry import register_loss


@register_loss("pose_reprojection")
class ReprojectionLoss(nn.Module):
    """Landmark reprojection loss for direct pose regression.

    Computes a robust per-point discrepancy between predicted and
    ground-truth 2D landmark locations.

    Args:
        robust: 'l2' (squared L2) or 'charbonnier'.
        epsilon: small constant for Charbonnier.
    """

    def __init__(self, robust: str = "l2", epsilon: float = 1e-3) -> None:
        super().__init__()
        self.robust = robust
        self.epsilon = float(epsilon)

    def _per_point_error(self, diff: torch.Tensor) -> torch.Tensor:
        # diff: (B, N, 2)
        if self.robust == "l2":
            return (diff ** 2).sum(dim=-1)  # (B, N)
        elif self.robust == "charbonnier":
            return torch.sqrt((diff ** 2).sum(dim=-1) + self.epsilon ** 2)
        else:
            raise ValueError(f"Unsupported robust mode: {self.robust}")

    def forward(
        self,
        preds_2d: torch.Tensor,
        targets_2d: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute reprojection loss.

        Args:
            preds_2d: (B, N, 2) predicted pixel coordinates.
            targets_2d: (B, N, 2) ground-truth pixel coordinates.
            weights: optional (B, N) visibility/importance weights in [0,1].
        """
        if preds_2d.shape != targets_2d.shape:
            raise ValueError(
                f"preds_2d and targets_2d must have same shape; got {preds_2d.shape} vs {targets_2d.shape}"
            )

        diff = preds_2d - targets_2d
        per_point = self._per_point_error(diff)  # (B, N)

        if weights is not None:
            if weights.shape != per_point.shape:
                raise ValueError(
                    f"weights must have shape {per_point.shape}; got {weights.shape}"
                )
            per_point = per_point * weights
            denom = weights.sum().clamp(min=1e-6)
        else:
            denom = torch.tensor(per_point.numel(), dtype=per_point.dtype, device=per_point.device)

        loss = per_point.sum() / denom
        return loss
