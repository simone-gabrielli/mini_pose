import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pose.registry import register_loss


def _gaussian_radius(det_size: tuple[torch.Tensor, torch.Tensor], min_overlap: float = 0.7) -> torch.Tensor:
    """Compute gaussian radius (vectorized) as in CenterNet.

    det_size: (h, w) tensors in heatmap units
    returns: radius tensor
    """

    h, w = det_size

    a1 = torch.ones_like(h)
    b1 = h + w
    c1 = w * h * (1.0 - min_overlap) / (1.0 + min_overlap)
    sq1 = torch.sqrt(torch.clamp(b1 * b1 - 4.0 * a1 * c1, min=0.0))
    r1 = (b1 + sq1) / 2.0

    a2 = 4.0 * torch.ones_like(h)
    b2 = 2.0 * (h + w)
    c2 = (1.0 - min_overlap) * w * h
    sq2 = torch.sqrt(torch.clamp(b2 * b2 - 4.0 * a2 * c2, min=0.0))
    r2 = (b2 + sq2) / 2.0

    a3 = 4.0 * min_overlap * torch.ones_like(h)
    b3 = -2.0 * min_overlap * (h + w)
    c3 = (min_overlap - 1.0) * w * h
    sq3 = torch.sqrt(torch.clamp(b3 * b3 - 4.0 * a3 * c3, min=0.0))
    r3 = (b3 + sq3) / 2.0

    return torch.min(torch.min(r1, r2), r3)


def _gaussian2d(diameter: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sigma = max(diameter / 6.0, 1e-6)
    radius = diameter // 2
    xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    ys = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    g = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    return g


def _draw_gaussian(hm: torch.Tensor, center_x: int, center_y: int, radius: int) -> None:
    """In-place max with a 2D gaussian. hm is (H,W)."""
    if radius <= 0:
        if 0 <= center_x < hm.shape[1] and 0 <= center_y < hm.shape[0]:
            hm[center_y, center_x] = torch.maximum(hm[center_y, center_x], hm.new_tensor(1.0))
        return

    diameter = 2 * radius + 1
    g = _gaussian2d(diameter, device=hm.device, dtype=hm.dtype)

    H, W = int(hm.shape[0]), int(hm.shape[1])
    left = min(center_x, radius)
    right = min(W - 1 - center_x, radius)
    top = min(center_y, radius)
    bottom = min(H - 1 - center_y, radius)

    if left < 0 or right < 0 or top < 0 or bottom < 0:
        return

    hm_patch = hm[center_y - top : center_y + bottom + 1, center_x - left : center_x + right + 1]
    g_patch = g[radius - top : radius + bottom + 1, radius - left : radius + right + 1]
    hm_patch.copy_(torch.maximum(hm_patch, g_patch))


def _centernet_focal_loss(pred: torch.Tensor, gt: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """Per-image focal loss (returns shape (B,)).

    pred: sigmoid(hm) in [0,1]
    gt: target heatmap in [0,1]
    """
    pred = pred.clamp(1e-6, 1.0 - 1e-6)

    pos_inds = (gt == 1.0).to(dtype=pred.dtype)
    neg_inds = (gt < 1.0).to(dtype=pred.dtype)
    neg_weights = (1.0 - gt).pow(beta) * neg_inds

    pos_loss = -(pred.log() * (1.0 - pred).pow(alpha) * pos_inds)
    neg_loss = -((1.0 - pred).log() * pred.pow(alpha) * neg_weights)

    # Sum over spatial dims
    pos_loss = pos_loss.flatten(1).sum(dim=1)
    neg_loss = neg_loss.flatten(1).sum(dim=1)

    num_pos = pos_inds.flatten(1).sum(dim=1).clamp(min=0.0)
    has_pos = num_pos > 0

    out = neg_loss
    out = torch.where(has_pos, (pos_loss + neg_loss) / num_pos.clamp(min=1.0), out)
    return out


def _gather_feat(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    """Gather (B,C,H,W) at flattened indices ind (B,) -> (B,C)."""
    B, C, H, W = feat.shape
    feat_flat = feat.view(B, C, H * W)
    ind = ind.view(B, 1, 1).expand(B, C, 1)
    return feat_flat.gather(2, ind).squeeze(-1)


@register_loss("centernet_detector")
class CenterNetDetectorLoss(nn.Module):
    """Loss for `centernet_single`.

    Targets are (B,5): [conf, x1, y1, x2, y2] normalized to [0,1].
    """

    def __init__(
        self,
        hm_weight: float = 1.0,
        wh_weight: float = 0.1,
        off_weight: float = 1.0,
        min_overlap: float = 0.7,
        min_radius: int = 0,
        max_radius: int = 8,
        focal_alpha: float = 2.0,
        focal_beta: float = 4.0,
    ):
        super().__init__()
        self.hm_weight = float(hm_weight)
        self.wh_weight = float(wh_weight)
        self.off_weight = float(off_weight)
        self.min_overlap = float(min_overlap)
        self.min_radius = int(min_radius)
        self.max_radius = int(max_radius)
        self.focal_alpha = float(focal_alpha)
        self.focal_beta = float(focal_beta)

    def forward(self, preds, targets: torch.Tensor, sample_weight=None):
        if preds is None:
            raise ValueError("preds is None")

        if isinstance(preds, (tuple, list)):
            preds = preds[0]

        if not isinstance(preds, dict):
            raise ValueError("CenterNetDetectorLoss expects preds as a dict with keys: hm, wh, off")

        for k in ("hm", "wh", "off"):
            if k not in preds:
                raise KeyError(f"Missing key in preds: {k}")

        hm_p = preds["hm"].float()
        wh_p = preds["wh"].float()
        off_p = preds["off"].float()

        if hm_p.dim() != 4 or hm_p.size(1) != 1:
            raise ValueError(f"Expected hm preds (B,1,H,W), got {tuple(hm_p.shape)}")
        if wh_p.dim() != 4 or wh_p.size(1) != 2:
            raise ValueError(f"Expected wh preds (B,2,H,W), got {tuple(wh_p.shape)}")
        if off_p.dim() != 4 or off_p.size(1) != 2:
            raise ValueError(f"Expected off preds (B,2,H,W), got {tuple(off_p.shape)}")
        if targets.dim() != 2 or targets.size(-1) != 5:
            raise ValueError(f"Expected targets shape (B,5), got {tuple(targets.shape)}")

        device = hm_p.device
        B, _, Hh, Wh = hm_p.shape

        targets = targets.to(device=device, dtype=hm_p.dtype)
        conf_t = targets[:, 0].clamp(0.0, 1.0)
        x1 = targets[:, 1].clamp(0.0, 1.0)
        y1 = targets[:, 2].clamp(0.0, 1.0)
        x2 = targets[:, 3].clamp(0.0, 1.0)
        y2 = targets[:, 4].clamp(0.0, 1.0)

        # Convert to heatmap units directly (avoids needing input_size/down_ratio)
        x1h = x1 * float(Wh)
        x2h = x2 * float(Wh)
        y1h = y1 * float(Hh)
        y2h = y2 * float(Hh)

        w = (x2h - x1h).clamp(min=0.0)
        h = (y2h - y1h).clamp(min=0.0)
        cx = (x1h + x2h) * 0.5
        cy = (y1h + y2h) * 0.5

        cx_int = torch.floor(cx).to(torch.int64)
        cy_int = torch.floor(cy).to(torch.int64)
        cx_int = cx_int.clamp(0, Wh - 1)
        cy_int = cy_int.clamp(0, Hh - 1)

        ind = (cy_int * Wh + cx_int).to(torch.int64)  # (B,)
        off_t = torch.stack([cx - cx_int.to(cx.dtype), cy - cy_int.to(cy.dtype)], dim=1).clamp(0.0, 1.0)  # (B,2)
        wh_t = torch.stack([w, h], dim=1)  # (B,2)

        # Build target heatmap
        hm_t = torch.zeros((B, 1, Hh, Wh), device=device, dtype=hm_p.dtype)
        pos = conf_t > 0.5
        if pos.any():
            # radius based on object size in heatmap units
            rad = _gaussian_radius((h[pos], w[pos]), min_overlap=self.min_overlap)
            rad = torch.floor(rad).to(torch.int64)
            rad = rad.clamp(self.min_radius, self.max_radius)
            pos_idx = torch.nonzero(pos, as_tuple=False).view(-1)
            for j, bi in enumerate(pos_idx.tolist()):
                r = int(rad[j].item())
                _draw_gaussian(hm_t[bi, 0], int(cx_int[bi].item()), int(cy_int[bi].item()), r)

        hm_pred = torch.sigmoid(hm_p)
        hm_loss_per = _centernet_focal_loss(hm_pred, hm_t, alpha=self.focal_alpha, beta=self.focal_beta)  # (B,)

        # Regression losses only for positives
        # IMPORTANT: do NOT ReLU/clip wh before loss. If we clamp to 0 here,
        # negative predictions get zero gradient and the head can get stuck
        # outputting ~0 widths (shows up as x1==x2 in visualizations).
        wh_pred = wh_p
        off_pred = torch.sigmoid(off_p)
        wh_g = _gather_feat(wh_pred, ind)  # (B,2)
        off_g = _gather_feat(off_pred, ind)  # (B,2)

        pos_f = pos.to(dtype=hm_p.dtype).view(B, 1)
        wh_loss_per = (F.l1_loss(wh_g, wh_t, reduction="none") * pos_f).mean(dim=1)  # (B,)
        off_loss_per = (F.l1_loss(off_g, off_t, reduction="none") * pos_f).mean(dim=1)  # (B,)

        total_per = self.hm_weight * hm_loss_per + self.wh_weight * wh_loss_per + self.off_weight * off_loss_per

        if sample_weight is None:
            return total_per.mean()

        sw = sample_weight.to(device=device, dtype=total_per.dtype).view(-1)
        denom = sw.sum().clamp(min=1e-6)
        return (total_per * sw).sum() / denom
