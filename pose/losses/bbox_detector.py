# pose/losses/bbox_detector.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from pose.registry import register_loss


@register_loss("bbox_detector")
class BBoxDetectorLoss(nn.Module):
    """Loss for TinyFaceDetector-style single-box detection.

    Targets are expected as (B,5): [conf, x1, y1, x2, y2] with bbox coords normalized to [0,1].

    Model outputs are expected as (B,5): [conf_logit, x1_raw, y1_raw, x2_raw, y2_raw].
    The bbox raws are passed through sigmoid before regression.
    """

    def __init__(self, conf_weight: float = 1.0, bbox_weight: float = 1.0, bbox_loss: str = "smooth_l1"):
        super().__init__()
        self.conf_weight = float(conf_weight)
        self.bbox_weight = float(bbox_weight)
        self.bbox_loss = str(bbox_loss).lower()

        if self.bbox_loss not in {"smooth_l1", "l1", "mse"}:
            raise ValueError("bbox_loss must be one of: smooth_l1, l1, mse")

    def forward(self, preds, targets, sample_weight=None):
        """Compute weighted loss.

        preds: Tensor (B,5)
        targets: Tensor (B,5)
        sample_weight: Tensor (B,) or None
        """
        if preds is None:
            raise ValueError("preds is None")

        if isinstance(preds, (tuple, list)):
            # Some models may return (out, aux). Use the first entry.
            preds = preds[0]

        if preds.dim() != 2 or preds.size(-1) != 5:
            raise ValueError(f"Expected preds shape (B,5), got {tuple(preds.shape)}")
        if targets.dim() != 2 or targets.size(-1) != 5:
            raise ValueError(f"Expected targets shape (B,5), got {tuple(targets.shape)}")

        preds = preds.float()
        targets = targets.float()

        conf_t = targets[:, 0].clamp(0.0, 1.0)
        bbox_t = targets[:, 1:5].clamp(0.0, 1.0)

        conf_logit = preds[:, 0]
        conf_loss_per = F.binary_cross_entropy_with_logits(conf_logit, conf_t, reduction="none")

        # bbox regression only for positive samples
        pos = (conf_t > 0.5).float()  # (B,)
        bbox_p = torch.sigmoid(preds[:, 1:5])

        if self.bbox_loss == "smooth_l1":
            bbox_err = F.smooth_l1_loss(bbox_p, bbox_t, reduction="none")
        elif self.bbox_loss == "l1":
            bbox_err = (bbox_p - bbox_t).abs()
        else:  # mse
            bbox_err = (bbox_p - bbox_t) ** 2

        bbox_loss_per = bbox_err.mean(dim=1) * pos

        total_per = self.conf_weight * conf_loss_per + self.bbox_weight * bbox_loss_per

        if sample_weight is None:
            return total_per.mean()

        sw = sample_weight.to(device=total_per.device, dtype=total_per.dtype).view(-1)
        denom = sw.sum().clamp(min=1e-6)
        return (total_per * sw).sum() / denom
