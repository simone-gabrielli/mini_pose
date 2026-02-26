# pose/losses/bbox_detector.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pose.registry import register_loss


def _compute_iou_variants(bbox_p: torch.Tensor, bbox_t: torch.Tensor, mode: str = "ciou"):
    """Compute IoU-family losses between predicted and target bboxes.

    Args:
        bbox_p: (B, 4) predicted [x1, y1, x2, y2] in [0, 1].
        bbox_t: (B, 4) target [x1, y1, x2, y2] in [0, 1].
        mode: one of 'iou', 'giou', 'diou', 'ciou'.

    Returns:
        loss_per: (B,) loss values (1 - IoU_variant), always >= 0.
    """
    eps = 1e-7

    # Ensure x1 < x2, y1 < y2 for both pred and gt
    px1 = torch.min(bbox_p[:, 0], bbox_p[:, 2])
    py1 = torch.min(bbox_p[:, 1], bbox_p[:, 3])
    px2 = torch.max(bbox_p[:, 0], bbox_p[:, 2])
    py2 = torch.max(bbox_p[:, 1], bbox_p[:, 3])

    gx1, gy1, gx2, gy2 = bbox_t[:, 0], bbox_t[:, 1], bbox_t[:, 2], bbox_t[:, 3]

    # Intersection
    ix1 = torch.max(px1, gx1)
    iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2)
    iy2 = torch.min(py2, gy2)
    inter = (ix2 - ix1).clamp(min=0.0) * (iy2 - iy1).clamp(min=0.0)

    # Union
    area_p = (px2 - px1).clamp(min=0.0) * (py2 - py1).clamp(min=0.0)
    area_g = (gx2 - gx1).clamp(min=0.0) * (gy2 - gy1).clamp(min=0.0)
    union = area_p + area_g - inter

    iou = inter / (union + eps)

    if mode == "iou":
        return 1.0 - iou

    # Enclosing (smallest) box
    cx1 = torch.min(px1, gx1)
    cy1 = torch.min(py1, gy1)
    cx2 = torch.max(px2, gx2)
    cy2 = torch.max(py2, gy2)

    if mode == "giou":
        area_c = (cx2 - cx1).clamp(min=0.0) * (cy2 - cy1).clamp(min=0.0)
        giou = iou - (area_c - union) / (area_c + eps)
        return 1.0 - giou

    # Center distance squared
    pcx = (px1 + px2) * 0.5
    pcy = (py1 + py2) * 0.5
    gcx = (gx1 + gx2) * 0.5
    gcy = (gy1 + gy2) * 0.5
    rho2 = (pcx - gcx) ** 2 + (pcy - gcy) ** 2

    # Diagonal of enclosing box squared
    c2 = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2 + eps

    if mode == "diou":
        diou = iou - rho2 / c2
        return 1.0 - diou

    # CIoU: IoU - distance_penalty - aspect_ratio_penalty
    pw = (px2 - px1).clamp(min=eps)
    ph = (py2 - py1).clamp(min=eps)
    gw = (gx2 - gx1).clamp(min=eps)
    gh = (gy2 - gy1).clamp(min=eps)

    v = (4.0 / (math.pi ** 2)) * (torch.atan(gw / gh) - torch.atan(pw / ph)) ** 2
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)

    ciou = iou - rho2 / c2 - alpha * v
    return 1.0 - ciou


@register_loss("bbox_detector")
class BBoxDetectorLoss(nn.Module):
    """Loss for TinyFaceDetector-style single-box detection.

    Targets are expected as (B,5): [conf, x1, y1, x2, y2] with bbox coords normalized to [0,1].

    Model outputs are expected as (B,5): [conf_logit, x1_raw, y1_raw, x2_raw, y2_raw].
    The bbox raws are passed through sigmoid before regression.

    Supported bbox_loss modes:
        - 'smooth_l1', 'l1', 'mse': element-wise coordinate losses (original)
        - 'ciou': Complete-IoU loss (best for stability & smoothness)
        - 'diou': Distance-IoU loss
        - 'giou': Generalized-IoU loss
        - 'iou': vanilla IoU loss

    conf_label_smoothing > 0 replaces hard 1.0 targets with (1 - eps) to
    improve calibration and regularize the confidence head.
    """

    _IOU_MODES = {"iou", "giou", "diou", "ciou"}

    def __init__(
        self,
        conf_weight: float = 1.0,
        bbox_weight: float = 1.0,
        bbox_loss: str = "smooth_l1",
        conf_label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.conf_weight = float(conf_weight)
        self.bbox_weight = float(bbox_weight)
        self.bbox_loss = str(bbox_loss).lower()
        self.conf_label_smoothing = float(max(0.0, min(conf_label_smoothing, 0.5)))

        valid = {"smooth_l1", "l1", "mse"} | self._IOU_MODES
        if self.bbox_loss not in valid:
            raise ValueError(f"bbox_loss must be one of: {sorted(valid)}")

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

        # Optional label smoothing for positive confidence targets
        if self.conf_label_smoothing > 0:
            eps = self.conf_label_smoothing
            conf_t = conf_t * (1.0 - eps) + 0.5 * eps  # smooth towards 0.5

        bbox_t = targets[:, 1:5].clamp(0.0, 1.0)

        conf_logit = preds[:, 0]
        conf_loss_per = F.binary_cross_entropy_with_logits(conf_logit, conf_t, reduction="none")

        # bbox regression only for positive samples
        pos = (targets[:, 0] > 0.5).float()  # use original (un-smoothed) conf for masking
        bbox_p = torch.sigmoid(preds[:, 1:5])

        if self.bbox_loss in self._IOU_MODES:
            bbox_loss_per = _compute_iou_variants(bbox_p, bbox_t, mode=self.bbox_loss) * pos
        elif self.bbox_loss == "smooth_l1":
            bbox_err = F.smooth_l1_loss(bbox_p, bbox_t, reduction="none")
            bbox_loss_per = bbox_err.mean(dim=1) * pos
        elif self.bbox_loss == "l1":
            bbox_err = (bbox_p - bbox_t).abs()
            bbox_loss_per = bbox_err.mean(dim=1) * pos
        else:  # mse
            bbox_err = (bbox_p - bbox_t) ** 2
            bbox_loss_per = bbox_err.mean(dim=1) * pos

        total_per = self.conf_weight * conf_loss_per + self.bbox_weight * bbox_loss_per

        if sample_weight is None:
            return total_per.mean()

        sw = sample_weight.to(device=total_per.device, dtype=total_per.dtype).view(-1)
        denom = sw.sum().clamp(min=1e-6)
        return (total_per * sw).sum() / denom
