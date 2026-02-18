import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from pose.registry import register_model


def _infer_out_channels(features: nn.Module) -> int:
    for m in reversed(list(features.modules())):
        if isinstance(m, nn.Conv2d):
            return int(m.out_channels)
    return 256


def _make_head(in_ch: int, head_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, head_ch, kernel_size=3, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(head_ch, out_ch, kernel_size=1, bias=True),
    )


@register_model("centernet_single")
class CenterNetSingleDetector(nn.Module):
    """CenterNet-style *single-object* detector.

    Intended as a drop-in replacement for `tiny_face` in this repo's bbox-only
    training path (targets stay as (B,5): [conf,x1,y1,x2,y2] normalized [0,1]).

    Forward returns a dict:
      - hm: (B,1,Hh,Wh) heatmap logits
      - wh: (B,2,Hh,Wh) size (w,h) in heatmap units (feature space)
      - off: (B,2,Hh,Wh) center offset in [0,1] (after sigmoid during decode/loss)
    """

    def __init__(
        self,
        pretrained: bool = True,
        backbone: str = "mobilenet_v3_small",
        width_mult: float = 1.0,
        neck_ch: int = 256,
        head_ch: int = 64,
        num_deconv: int = 3,
        down_ratio: int = 4,
        num_keypoints=None,
        **kwargs: Any,
    ):
        super().__init__()

        self.down_ratio = int(down_ratio)
        if self.down_ratio != 4:
            # This implementation assumes backbone stride 32 and 3x upsample -> stride 4.
            # If you want other ratios, adjust `num_deconv` and target generation.
            raise ValueError("CenterNetSingleDetector currently supports down_ratio=4 only")

        bb = str(backbone).lower().strip()
        tv = torchvision.models

        if bb in {"mobilenet_v3_small", "mnetv3_small", "mbv3_small"}:
            try:
                weights = tv.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
                try:
                    net = tv.mobilenet_v3_small(weights=weights, width_mult=float(width_mult))
                except TypeError:
                    net = tv.mobilenet_v3_small(weights=weights)
            except Exception:
                try:
                    net = tv.mobilenet_v3_small(pretrained=pretrained, width_mult=float(width_mult))
                except TypeError:
                    net = tv.mobilenet_v3_small(pretrained=pretrained)
        else:
            raise ValueError("centernet_single currently supports backbone=mobilenet_v3_small only")

        self.features = net.features
        in_ch = _infer_out_channels(self.features)

        self.neck = nn.Sequential(
            nn.Conv2d(in_ch, int(neck_ch), kernel_size=1, bias=False),
            nn.BatchNorm2d(int(neck_ch)),
            nn.ReLU(inplace=True),
        )

        deconvs = []
        for _ in range(int(num_deconv)):
            deconvs.extend(
                [
                    nn.ConvTranspose2d(int(neck_ch), int(neck_ch), kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(int(neck_ch)),
                    nn.ReLU(inplace=True),
                ]
            )
        self.deconv = nn.Sequential(*deconvs)

        self.hm_head = _make_head(int(neck_ch), int(head_ch), 1)
        self.wh_head = _make_head(int(neck_ch), int(head_ch), 2)
        self.off_head = _make_head(int(neck_ch), int(head_ch), 2)

        # Heatmap bias init so sigmoid(hm) starts near 0.
        with torch.no_grad():
            last = self.hm_head[-1]
            if isinstance(last, nn.Conv2d) and last.bias is not None:
                last.bias.fill_(-2.19)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.features(x)
        feat = self.neck(feat)
        feat = self.deconv(feat)
        return {
            "hm": self.hm_head(feat),
            "wh": self.wh_head(feat),
            "off": self.off_head(feat),
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor, conf_th: float = 0.3):
        """Decode a single bbox per image in *input-pixel* coordinates."""
        self.eval()
        device = next(self.parameters()).device
        x = x.to(device)

        out = self.forward(x)
        hm = torch.sigmoid(out["hm"])  # (B,1,H,W)
        # Softplus keeps sizes positive without hard-clamping to 0.
        # This avoids "x1==x2" vertical-line boxes in visualizations when the
        # raw head output is still negative early in training.
        wh = F.softplus(out["wh"])  # (B,2,H,W) in heatmap units
        off = torch.sigmoid(out["off"])  # (B,2,H,W) in [0,1]

        B, _, Hh, Wh = hm.shape
        scores, inds = torch.max(hm.view(B, -1), dim=1)  # (B,)
        xs = (inds % Wh).float()
        ys = (inds // Wh).float()

        ind_exp = inds.view(B, 1, 1).expand(B, 2, 1)
        wh_g = wh.view(B, 2, -1).gather(2, ind_exp).squeeze(-1)  # (B,2)
        off_g = off.view(B, 2, -1).gather(2, ind_exp).squeeze(-1)  # (B,2)

        cx_hm = xs + off_g[:, 0]
        cy_hm = ys + off_g[:, 1]
        w_hm = wh_g[:, 0].clamp(min=1e-3)
        h_hm = wh_g[:, 1].clamp(min=1e-3)

        # Convert heatmap units -> input pixels
        stride = float(self.down_ratio)
        cx = cx_hm * stride
        cy = cy_hm * stride
        w = w_hm * stride
        h = h_hm * stride

        results = []
        for i in range(B):
            c = float(scores[i].detach().cpu().item())
            if c < float(conf_th):
                results.append({"conf": c, "bbox": None})
                continue
            x1 = float((cx[i] - 0.5 * w[i]).detach().cpu().item())
            y1 = float((cy[i] - 0.5 * h[i]).detach().cpu().item())
            x2 = float((cx[i] + 0.5 * w[i]).detach().cpu().item())
            y2 = float((cy[i] + 0.5 * h[i]).detach().cpu().item())
            results.append({"conf": c, "bbox": [x1, y1, x2, y2]})
        return results

    @torch.no_grad()
    def generate_sample_visualization(self, sample, out_path: str, device: torch.device):
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        img_t = sample.get("image", None)
        if img_t is None:
            return

        self.eval()
        x = img_t.unsqueeze(0).to(device)
        pred = self.predict(x, conf_th=0.0)[0]
        conf_pred = float(pred.get("conf", 0.0))

        # De-normalize image to uint8 BGR for cv2 drawing
        img_np = img_t.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np * std) + mean
        img_np = np.clip(img_np * 255.0, 0.0, 255.0).astype(np.uint8)
        img_bgr = img_np[..., ::-1].copy()
        H, W = img_bgr.shape[:2]

        cv2 = __import__("cv2")

        # GT bbox (green)
        tgt = sample.get("bbox", None)
        if tgt is not None:
            try:
                tgt_np = tgt.detach().cpu().numpy().astype(np.float32)
                if float(tgt_np[0]) > 0.5:
                    x1g = int(np.clip(tgt_np[1] * W, 0, W - 1))
                    y1g = int(np.clip(tgt_np[2] * H, 0, H - 1))
                    x2g = int(np.clip(tgt_np[3] * W, 0, W - 1))
                    y2g = int(np.clip(tgt_np[4] * H, 0, H - 1))
                    cv2.rectangle(img_bgr, (x1g, y1g), (x2g, y2g), (0, 255, 0), 2)
                    cv2.putText(img_bgr, "GT", (x1g, max(0, y1g - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except Exception:
                pass

        # Pred bbox (red)
        bb = pred.get("bbox")
        if bb is not None:
            try:
                x1 = int(np.clip(bb[0], 0, W - 1))
                y1 = int(np.clip(bb[1], 0, H - 1))
                x2 = int(np.clip(bb[2], 0, W - 1))
                y2 = int(np.clip(bb[3], 0, H - 1))
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img_bgr, f"pred: {conf_pred:.2f}", (x1, min(H - 1, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            except Exception:
                pass

        try:
            cv2.imwrite(out_path, img_bgr)
        except Exception:
            return
