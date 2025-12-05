import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image, ImageDraw
from torchvision.models import mobilenet_v2

from pose.models.base import PoseModel
from pose.registry import register_model

from pose.models.mobilenet_pose import LandmarkChannelAttention

@register_model("mobilenet_pose_3d")
class MobileNetPose3D(PoseModel):
    """MobileNetV2 backbone with a 3D heatmap head.

    Produces a volumetric heatmap per keypoint:
        (B, K * depth_bins, H_hm, W_hm)

    which is reshaped to (B, K, D, H, W) for loss computation.

    The 2D spatial head (deconv stack) is identical to the 2D model;
    only the last conv changes its number of output channels.
    """

    def __init__(
        self,
        num_keypoints: int,
        pretrained: bool = True,
        deconv_channels=(256, 256, 256),
        deconv_kernel=4,
        depth_bins: int = 8,
        depth_range: tuple | None = None,
        depth_mean: float | None = None,
        attention_landmarks=None,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.depth_bins = depth_bins
        # depth_range is (z_min, z_max) in the same units as keypoints' z
        self.depth_range = depth_range
        # depth_mean (optional) from dataset to initialize depth bias
        self.depth_mean = depth_mean

        backbone = mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
        self.backbone = backbone.features
        backbone_out_channels = 1280

        self.deconv_layers = self._make_deconv_layers(
            backbone_out_channels, deconv_channels, deconv_kernel
        )

        # 2D heatmap head (per-keypoint spatial heatmap)
        self.heatmap_head = nn.Conv2d(
            in_channels=deconv_channels[-1],
            out_channels=num_keypoints,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # depth regression head: predict a scalar depth per keypoint
        # implemented as a per-keypoint map followed by spatial pooling
        self.depth_head = nn.Conv2d(
            in_channels=deconv_channels[-1],
            out_channels=num_keypoints,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if attention_landmarks is not None:
            # attention still operates per keypoint channel; we apply it
            # after collapsing depth with a softmax-based expectation.
            self.attention = LandmarkChannelAttention(
                num_kpts=num_keypoints,
                important_idx=attention_landmarks,
            )
        else:
            self.attention = None

        self._init_weights()

        # Initialize depth_head bias to dataset mean z if provided
        if hasattr(self, "depth_head") and hasattr(self, "depth_range"):
            # If a depth_mean was passed via constructor kwargs, set it
            depth_mean = getattr(self, "depth_mean", None)
            if depth_mean is not None:
                try:
                    with torch.no_grad():
                        if self.depth_head.bias is not None:
                            self.depth_head.bias.data.fill_(float(depth_mean))
                except Exception:
                    pass

    def _make_deconv_layers(self, in_channels, num_filters, kernel_size):
        layers = []
        for f in num_filters:
            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    f,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(f))
            layers.append(nn.ReLU(inplace=True))
            in_channels = f
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.deconv_layers.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.heatmap_head.weight, std=0.001)
        if self.heatmap_head.bias is not None:
            nn.init.constant_(self.heatmap_head.bias, 0)
        nn.init.normal_(self.depth_head.weight, std=0.001)
        if self.depth_head.bias is not None:
            nn.init.constant_(self.depth_head.bias, 0)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.deconv_layers(feat)

        # 2D heatmaps
        hm = self.heatmap_head(feat)  # (B, K, H, W)

        # optional attention still operates on 2D heatmaps
        if self.attention is not None:
            hm = self.attention(hm)

        B, K, H_hm, W_hm = hm.shape

        # spatial softmax to get per-keypoint probability map
        hm_flat = hm.view(B, K, -1)
        prob = torch.softmax(hm_flat, dim=2).view(B, K, H_hm, W_hm)

        # expected 2D coordinates in heatmap pixels
        xs = torch.linspace(0, W_hm - 1, W_hm, device=prob.device, dtype=prob.dtype)
        ys = torch.linspace(0, H_hm - 1, H_hm, device=prob.device, dtype=prob.dtype)
        xs = xs.view(1, 1, 1, W_hm)
        ys = ys.view(1, 1, H_hm, 1)
        exp_x = (prob * xs).sum(dim=(2, 3))  # (B, K)
        exp_y = (prob * ys).sum(dim=(2, 3))  # (B, K)

        # depth regression: per-keypoint map pooled to scalar, predict z in the
        # same units as the dataset keypoints (not normalized). This treats z
        # exactly like x,y as requested.
        dz_map = self.depth_head(feat)  # (B, K, H, W)
        dz = dz_map.mean(dim=(2, 3))  # (B, K)
        # dz is raw and should correspond to z units (e.g., image pixels or dataset z)

        # assemble predicted 3D coords in image/heatmap pixel space + normalized depth
        coords = torch.stack([exp_x, exp_y, dz], dim=2)  # (B, K, 3)

        # Build predicted 3D gaussian volumes from coords to match dataset-generated targets
        # This allows use of an MSE loss against precomputed volumetric targets while
        # the model itself does not produce discrete depth bins.
        D = self.depth_bins
        device = coords.device
        dtype = coords.dtype

        # create grids for x,y on heatmap resolution and z using dataset units
        if hasattr(self, "depth_range") and self.depth_range is not None:
            zmin, zmax = float(self.depth_range[0]), float(self.depth_range[1])
            z_lin = torch.linspace(zmin, zmax, D, device=device, dtype=dtype)
        else:
            # fallback to normalized [0,1]
            z_lin = torch.linspace(0.0, 1.0, D, device=device, dtype=dtype)
        x_lin = torch.linspace(0, W_hm - 1, W_hm, device=device, dtype=dtype)
        y_lin = torch.linspace(0, H_hm - 1, H_hm, device=device, dtype=dtype)
        zz = z_lin.view(1, 1, D, 1, 1)
        yy = y_lin.view(1, 1, 1, H_hm, 1)
        xx = x_lin.view(1, 1, 1, 1, W_hm)

        # coords: exp_x, exp_y in heatmap pixel coords; dz in [0,1]
        cx = exp_x.view(B, K, 1, 1, 1)
        cy = exp_y.view(B, K, 1, 1, 1)
        cz = dz.view(B, K, 1, 1, 1)

        # gaussian widths (in pixels for x,y and in same z units for z)
        sigma_x = 1.5
        sigma_y = 1.5
        # choose a reasonable sigma in z units; if depth_range provided, use a small fraction
        if hasattr(self, "depth_range") and self.depth_range is not None:
            sigma_z = max(1.0, (float(self.depth_range[1]) - float(self.depth_range[0])) * 0.02)
        else:
            sigma_z = 0.08

        gx = ((xx - cx) ** 2) / (2 * (sigma_x ** 2))
        gy = ((yy - cy) ** 2) / (2 * (sigma_y ** 2))
        gz = ((zz - cz) ** 2) / (2 * (sigma_z ** 2))

        pred_vol = torch.exp(-(gx + gy + gz))  # (B, K, D, H, W)

        # Normalize per-keypoint volume to max 1 to match dataset target normalization
        max_vals = pred_vol.view(B, K, -1).max(dim=2)[0].view(B, K, 1, 1, 1)
        max_vals = torch.clamp(max_vals, min=1e-6)
        pred_vol = pred_vol / max_vals

        # For trainer compatibility: return a 2D heatmap as preds_last and a list
        # containing volumetric prediction for loss computation.
        hm_out = prob  # probabilistic 2D heatmaps
        return hm_out, [pred_vol]

    def generate_sample_visualization(self, sample, out_path: str, device: torch.device):
        """Model-level visualization hook for Trainer.

        Signature matches what `Trainer._save_qualitative_examples` expects:
            (sample: Dict, out_path: str, device: torch.device) -> None

        This runs a forward pass on the single-sample image, extracts
        per-keypoint (x,y) from the marginalised 2D heatmaps and an
        expected depth in normalized [0,1] from the volumetric heatmap,
        then delegates drawing to the module-level `generate_sample_visualization`.
        """
        self.eval()

        img = sample["image"]  # tensor C,H,W or numpy
        # preserve CPU image for drawing; we pass a copy to the helper
        img_np = img.cpu().numpy().transpose(1, 2, 0) if isinstance(img, torch.Tensor) else np.array(img)

        # If dataset normalized images (ImageNet mean/std), undo it for visualization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        try:
            img_np = (img_np * std) + mean
        except Exception:
            pass

        # Forward
        with torch.no_grad():
            inp = (
                img.unsqueeze(0).to(device)
                if isinstance(img, torch.Tensor)
                else torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
            )
            out = self.forward(inp)

        # Extract the last preds (support both 2-tuple and 4-tuple model outputs)
        if isinstance(out, tuple) and len(out) == 4:
            preds_all = out[1]
        else:
            preds_all = out[1]

        preds_last = preds_all[-1]

        # preds_last could be (B, K, H, W) for 2D models or (B, K, D, H, W)
        preds_last = preds_last.cpu()

        if preds_last.dim() == 4:
            # 2D: fallback to existing visualization (draw all keypoints in one color)
            hm2d = preds_last[0]  # (K, H, W)
            K, H_hm, W_hm = hm2d.shape
            # take argmax spatially
            coords = []
            for k in range(K):
                h = hm2d[k].view(-1)
                idx = int(h.argmax())
                y = idx // W_hm
                x = idx % W_hm
                coords.append((x, y, 0.0))
        elif preds_last.dim() == 5:
            # 3D volumetric heatmap: (B, K, D, H, W)
            vol = preds_last[0].numpy()  # (K, D, H, W)
            K, D, H_hm, W_hm = vol.shape

            # Marginalize over depth to get 2D heatmaps
            hm2d = vol.sum(axis=1)  # (K, H, W)

            coords = []
            depth_bins = D
            if hasattr(self, "depth_range") and self.depth_range is not None:
                depth_idxs = np.linspace(float(self.depth_range[0]), float(self.depth_range[1]), depth_bins)
            else:
                depth_idxs = np.linspace(0.0, 1.0, depth_bins)

            # For each keypoint, compute spatial argmax and expected depth
            for k in range(K):
                h = hm2d[k].reshape(-1)
                idx = int(h.argmax())
                y = idx // W_hm
                x = idx % W_hm

                # depth marginal: sum spatially to get p(d)
                p_d = vol[k].reshape(D, -1).sum(axis=1)
                # normalize
                s = p_d.sum()
                if s > 0:
                    p_d = p_d / s
                expected_z = float((p_d * depth_idxs).sum())
                coords.append((x * (sample["image"].shape[2] / W_hm), y * (sample["image"].shape[1] / H_hm), expected_z))
        else:
            # unknown format
            return

        keypoints_3d = np.array(coords)

        # Prepare ground-truth keypoints (image pixel coords)
        gt_kp = None
        try:
            if "keypoints" in sample:
                kp = sample["keypoints"]
                if isinstance(kp, torch.Tensor):
                    kp = kp.cpu().numpy()
                # take x,y columns
                if kp.ndim == 2 and kp.shape[1] >= 2:
                    gt_kp = kp[:, :2]
        except Exception:
            gt_kp = None

        # Use the module-level helper to draw colored keypoints by depth and show GT
        viz_img = generate_sample_visualization(img_np, keypoints_3d, gt_keypoints=gt_kp, return_pil=True)
        try:
            viz_img.save(out_path)
        except Exception as e:
            print(f"Warning: could not save visualization to {out_path}: {e}")


def generate_sample_visualization(image, keypoints_3d, cmap_name="viridis", radius=3, depth_clip=None, gt_keypoints=None, gt_color=(50, 205, 50), return_pil=True):
    """Draw landmarks on `image` colored by their depth.

    Args:
        image: HxW[x3] numpy array or torch Tensor (uint8 or float0-1).
        keypoints_3d: (K,3) or (K,4) array-like with [x, y, z (, v)].
        gt_keypoints: optional (K,2) or (K,3) array-like of ground-truth x,y(,v) in image pixels.
        cmap_name: matplotlib colormap name to map depth -> color.
        radius: circle radius in pixels.
        depth_clip: optional (min, max) tuple to clip depths before normalization.
        return_pil: if True return a PIL.Image, else return numpy array (H,W,3 uint8).

    Returns:
        PIL.Image or numpy.ndarray with colored landmark overlay.
    """
    # Convert image to numpy uint8 HxWx3
    if isinstance(image, torch.Tensor):
        img = image.detach().cpu().numpy()
    else:
        img = np.array(image)

    # Handle grayscale or single-channel images
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.dtype != np.uint8:
        # assume float in [0,1]
        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)

    H, W = img.shape[:2]

    kp = np.array(keypoints_3d)
    if kp.ndim == 1:
        kp = kp.reshape(-1, kp.shape[0])

    if kp.shape[1] < 3:
        raise ValueError("keypoints_3d must have at least 3 columns (x,y,z)")

    xs = kp[:, 0]
    ys = kp[:, 1]
    zs = kp[:, 2]

    # Filter out NaNs
    valid = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
    if not valid.any():
        # nothing to draw
        return Image.fromarray(img) if return_pil else img

    xs = xs[valid]
    ys = ys[valid]
    zs = zs[valid]

    # Clip or normalize depths
    if depth_clip is not None:
        zmin, zmax = depth_clip
        zs = np.clip(zs, zmin, zmax)
    else:
        zmin, zmax = float(np.nanmin(zs)), float(np.nanmax(zs))
        if zmax - zmin < 1e-6:
            zmax = zmin + 1e-6

    z_norm = (zs - zmin) / (zmax - zmin)
    cmap = cm.get_cmap(cmap_name)
    colors = (cmap(z_norm)[:, :3] * 255).astype(np.uint8)

    pil_img = Image.fromarray(img).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for x, y, col in zip(xs, ys, colors):
        # Round coordinates and clip to image bounds
        cx = int(round(x))
        cy = int(round(y))
        if cx < -radius or cy < -radius or cx > W + radius or cy > H + radius:
            continue
        left_up = (cx - radius, cy - radius)
        right_down = (cx + radius, cy + radius)
        draw.ellipse([left_up, right_down], fill=(int(col[0]), int(col[1]), int(col[2]), 200))

    # Draw ground-truth keypoints if provided (as small solid circles)
    if gt_keypoints is not None:
        gk = np.array(gt_keypoints)
        if gk.ndim == 1:
            gk = gk.reshape(-1, gk.shape[0])
        if gk.shape[1] >= 2:
            for gx, gy in gk[:, :2]:
                if not np.isfinite(gx) or not np.isfinite(gy):
                    continue
                gcx = int(round(gx))
                gcy = int(round(gy))
                if gcx < -radius or gcy < -radius or gcx > W + radius or gcy > H + radius:
                    continue
                left_up = (gcx - max(1, radius//2), gcy - max(1, radius//2))
                right_down = (gcx + max(1, radius//2), gcy + max(1, radius//2))
                draw.ellipse([left_up, right_down], fill=(int(gt_color[0]), int(gt_color[1]), int(gt_color[2]), 255))

    out = Image.alpha_composite(pil_img, overlay).convert("RGB")
    if return_pil:
        return out
    else:
        return np.array(out)
