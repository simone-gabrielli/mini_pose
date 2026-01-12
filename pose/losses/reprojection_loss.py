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
        sample_weight: Optional[torch.Tensor] = None,
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
            per_sample_denom = weights.sum(dim=1).clamp(min=1e-6)  # (B,)
        else:
            per_sample_denom = torch.full(
                (per_point.size(0),),
                float(per_point.size(1)),
                dtype=per_point.dtype,
                device=per_point.device,
            )

        per_sample_num = per_point.sum(dim=1)  # (B,)

        # Preserve previous behavior when no sample_weight is provided:
        # global sum / global denom.
        if sample_weight is None:
            return per_sample_num.sum() / per_sample_denom.sum().clamp(min=1e-6)

        sw = sample_weight.to(device=per_point.device, dtype=per_point.dtype).view(-1)
        num = (per_sample_num * sw).sum()
        denom = (per_sample_denom * sw).sum().clamp(min=1e-6)
        return num / denom
