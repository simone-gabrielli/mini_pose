import torch
import torch.nn as nn
import torchvision
import numpy as np
import os
from pose.registry import register_model


@register_model("tiny_face")
class TinyFaceDetector(nn.Module):
    """A lightweight face detector that regresses a single bounding box + confidence

    - Input: RGB image tensor, shape (B,3,H,W) expected in [0,1]
    - Output: tensor (B,5) -> [conf_logit, x1, y1, x2, y2] with bbox coords normalized [0..1]

    Supported backbones:
        - mobilenet_v2 (default, ~3.4M params)
        - mobilenet_v3_small (~2.5M params, fastest)
        - efficientnet_b0 (~5.3M params, best quality/speed for stability)

    Set ``deep_head=True`` for a 2-hidden-layer MLP with LayerNorm, which
    substantially reduces output jitter at negligible speed cost.
    """

    def __init__(
        self,
        pretrained: bool = True,
        embed_dim: int = 128,
        dropout: float = 0.1,
        backbone: str = "mobilenet_v2",
        width_mult: float = 1.0,
        deep_head: bool = False,
        num_keypoints=None,
        **kwargs,
    ):
        super().__init__()
        bb = str(backbone).lower().strip()
        self.backbone_name = bb

        # Lightweight backbone
        tv = torchvision.models
        if bb in {"efficientnet_b0", "effnet_b0", "enet_b0"}:
            try:
                weights = tv.EfficientNet_B0_Weights.DEFAULT if pretrained else None
                net = tv.efficientnet_b0(weights=weights)
            except Exception:
                net = tv.efficientnet_b0(pretrained=pretrained)
            self.features = net.features
            in_ch = 1280  # EfficientNet-B0 last channel

        elif bb in {"mobilenet_v3_small", "mnetv3_small", "mbv3_small"}:
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
            self.features = net.features
            in_ch = getattr(net, "last_channel", None)
            if in_ch is None:
                try:
                    for m in getattr(net, "classifier", []):
                        if isinstance(m, nn.Linear):
                            in_ch = int(m.in_features)
                            break
                except Exception:
                    in_ch = None
            if in_ch is None:
                in_ch = 576

        else:
            # Default: MobileNetV2
            try:
                weights = tv.MobileNet_V2_Weights.DEFAULT if pretrained else None
                try:
                    net = tv.mobilenet_v2(weights=weights, width_mult=float(width_mult))
                except TypeError:
                    net = tv.mobilenet_v2(weights=weights)
            except Exception:
                try:
                    net = tv.mobilenet_v2(pretrained=pretrained, width_mult=float(width_mult))
                except TypeError:
                    net = tv.mobilenet_v2(pretrained=pretrained)
            self.features = net.features
            in_ch = getattr(net, "last_channel", None)
            if in_ch is None:
                try:
                    for m in getattr(net, "classifier", []):
                        if isinstance(m, nn.Linear):
                            in_ch = int(m.in_features)
                            break
                except Exception:
                    in_ch = None
            if in_ch is None:
                in_ch = 1280

        # global pooling + regression head
        self.pool = nn.AdaptiveAvgPool2d(1)

        embed_dim = int(embed_dim)
        if deep_head:
            # 2-hidden-layer MLP with LayerNorm for stable, low-jitter output.
            # LayerNorm normalizes feature magnitudes, preventing scale drift
            # that otherwise causes frame-to-frame bbox jitter.
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_ch, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(float(dropout)),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.LayerNorm(embed_dim // 2),
                nn.SiLU(inplace=True),
                nn.Dropout(float(dropout) * 0.5),
                nn.Linear(embed_dim // 2, 5),
            )
        else:
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_ch, embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(float(dropout)),
                nn.Linear(embed_dim, 5),
            )

    def forward(self, x):
        # x assumed normalized already
        feat = self.features(x)
        pooled = self.pool(feat)
        out = self.fc(pooled)
        # out[:,0] is raw logit for confidence
        # out[:,1:5] are raw values; we'll sigmoid those during loss/prediction
        return out

    @torch.no_grad()
    def predict(self, x, conf_th=0.3):
        """Run forward and return (conf, bbox) in absolute coords for each image.

        Returns list of dicts: {'conf': float, 'bbox': [x1,y1,x2,y2]} with absolute pixel coords
        """
        self.eval()
        device = next(self.parameters()).device
        x = x.to(device)
        out = self.forward(x)
        logits = out[:, 0]
        conf = torch.sigmoid(logits)
        bbox = torch.sigmoid(out[:, 1:5])

        results = []
        _, _, H, W = x.shape
        for c, b in zip(conf.cpu(), bbox.cpu()):
            c_val = float(c.item())
            b = b.numpy().astype(np.float32)

            # Ensure x1<x2, y1<y2 and clamp to [0,1].
            x1n, y1n, x2n, y2n = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            if x2n < x1n:
                x1n, x2n = x2n, x1n
            if y2n < y1n:
                y1n, y2n = y2n, y1n
            x1n = float(np.clip(x1n, 0.0, 1.0))
            y1n = float(np.clip(y1n, 0.0, 1.0))
            x2n = float(np.clip(x2n, 0.0, 1.0))
            y2n = float(np.clip(y2n, 0.0, 1.0))

            # convert normalized x1,y1,x2,y2 to pixel coords
            x1 = float(x1n * W)
            y1 = float(y1n * H)
            x2 = float(x2n * W)
            y2 = float(y2n * H)
            if c_val >= conf_th:
                results.append({"conf": c_val, "bbox": [x1, y1, x2, y2]})
            else:
                results.append({"conf": c_val, "bbox": None})
        return results

    @torch.no_grad()
    def generate_sample_visualization(self, sample, out_path: str, device: torch.device):
        """Save a qualitative visualization for a single validation sample.

        This mirrors the landmark models' qualitative viz pipeline:
        - run a forward pass
        - de-normalize the input image
        - draw GT bbox (green) if present
        - draw predicted bbox (red) with predicted confidence

        Expects `sample` like CocoFaceDataset yields:
          sample['image']: Tensor (3,H,W) normalized by ImageNet stats
          sample['bbox']: Tensor (5,) -> [conf, x1, y1, x2, y2] normalized [0..1]
        """
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        img_t = sample.get("image", None)
        if img_t is None:
            return

        # Forward
        self.eval()
        x = img_t.unsqueeze(0).to(device)
        out = self.forward(x)

        # Decode prediction
        try:
            conf_pred = float(torch.sigmoid(out[0, 0]).detach().cpu().item())
        except Exception:
            conf_pred = 0.0

        try:
            bb = torch.sigmoid(out[0, 1:5]).detach().cpu().numpy().astype(np.float32)
        except Exception:
            bb = np.zeros((4,), dtype=np.float32)

        # Ensure x1<x2, y1<y2
        x1p, y1p, x2p, y2p = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
        if x2p < x1p:
            x1p, x2p = x2p, x1p
        if y2p < y1p:
            y1p, y2p = y2p, y1p
        x1p = float(np.clip(x1p, 0.0, 1.0))
        y1p = float(np.clip(y1p, 0.0, 1.0))
        x2p = float(np.clip(x2p, 0.0, 1.0))
        y2p = float(np.clip(y2p, 0.0, 1.0))

        # De-normalize image to uint8 BGR for cv2 drawing
        img_np = img_t.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np * std) + mean
        img_np = np.clip(img_np * 255.0, 0.0, 255.0).astype(np.uint8)
        img_bgr = img_np[..., ::-1].copy()

        H, W = img_bgr.shape[:2]

        # Draw GT bbox (green) if present
        tgt = sample.get("bbox", None)
        if tgt is not None:
            try:
                tgt_np = tgt.detach().cpu().numpy().astype(np.float32)
                conf_gt = float(tgt_np[0])
                if conf_gt > 0.5:
                    x1g = int(np.clip(tgt_np[1] * W, 0, W - 1))
                    y1g = int(np.clip(tgt_np[2] * H, 0, H - 1))
                    x2g = int(np.clip(tgt_np[3] * W, 0, W - 1))
                    y2g = int(np.clip(tgt_np[4] * H, 0, H - 1))
                    cv2 = __import__("cv2")
                    cv2.rectangle(img_bgr, (x1g, y1g), (x2g, y2g), (0, 255, 0), 2)
                    cv2.putText(img_bgr, "GT", (x1g, max(0, y1g - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except Exception:
                pass

        # Draw prediction (red)
        try:
            x1 = int(np.clip(x1p * W, 0, W - 1))
            y1 = int(np.clip(y1p * H, 0, H - 1))
            x2 = int(np.clip(x2p * W, 0, W - 1))
            y2 = int(np.clip(y2p * H, 0, H - 1))
            cv2 = __import__("cv2")
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                img_bgr,
                f"pred: {conf_pred:.2f}",
                (x1, min(H - 1, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        except Exception:
            pass

        # Save
        try:
            cv2 = __import__("cv2")
            cv2.imwrite(out_path, img_bgr)
        except Exception:
            # last resort: no-op if cv2 isn't available
            return


def load_tiny_face_detector(checkpoint_path, device=None):
    import re

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _strip_module_prefix(sd: dict) -> dict:
        out = {}
        for k, v in sd.items():
            nk = k[len("module.") :] if isinstance(k, str) and k.startswith("module.") else k
            out[nk] = v
        return out

    def _get_state_dict(ckpt_obj: object) -> dict:
        if isinstance(ckpt_obj, dict):
            for k in ("model", "model_state", "state_dict", "weights"):
                sd = ckpt_obj.get(k)
                if isinstance(sd, dict):
                    return sd
            # maybe the dict itself is already a state_dict
            if ckpt_obj and all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
                return ckpt_obj
        raise ValueError("Unsupported checkpoint format")

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    sd = _strip_module_prefix(_get_state_dict(ckpt))
    keys = list(sd.keys())

    inferred_backbone = "mobilenet_v2"
    if any(re.match(r"^features\.\d+\.\d+\.block\.", k) for k in keys):
        inferred_backbone = "efficientnet_b0"
    elif any(re.match(r"^features\.\d+\.block\.", k) for k in keys):
        inferred_backbone = "mobilenet_v3_small"

    inferred_embed_dim = None
    fc1 = sd.get("fc.1.weight")
    if isinstance(fc1, torch.Tensor) and fc1.dim() == 2:
        inferred_embed_dim = int(fc1.shape[0])

    inferred_deep_head = any(k.startswith("fc.9.") for k in keys)

    model_kwargs = {
        "pretrained": False,
        "backbone": str(inferred_backbone),
    }
    if inferred_embed_dim is not None:
        model_kwargs["embed_dim"] = int(inferred_embed_dim)
    if inferred_deep_head:
        model_kwargs["deep_head"] = True

    model = TinyFaceDetector(**model_kwargs)
    model.load_state_dict(sd, strict=True)

    model.to(device)
    model.eval()

    # Apply EMA shadow weights if present.
    ema_shadow = None
    if isinstance(ckpt, dict):
        ema = ckpt.get("ema")
        if isinstance(ema, dict):
            shadow = ema.get("shadow")
            if isinstance(shadow, dict) and shadow:
                ema_shadow = _strip_module_prefix(shadow)
    if isinstance(ema_shadow, dict) and ema_shadow:
        applied = 0
        for name, p in model.named_parameters():
            if name not in ema_shadow:
                continue
            t = ema_shadow.get(name)
            if not isinstance(t, torch.Tensor):
                continue
            if not torch.is_floating_point(p.data):
                continue
            if t.device != p.device:
                t = t.to(device=p.device)
            if t.dtype != p.dtype:
                t = t.to(dtype=p.dtype)
            try:
                p.data.copy_(t)
                applied += 1
            except Exception:
                continue
        if applied > 0:
            # Keep output minimal; loader may be used in training scripts too.
            pass

    return model
