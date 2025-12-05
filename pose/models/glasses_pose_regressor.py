import os
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

from pose.registry import register_model


def rotation_6d_to_matrix(r6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrix.

    Follow Zhou et al., "On the Continuity of Rotation Representations in Neural Networks".

    Args:
        r6: (..., 6) tensor

    Returns:
        R: (..., 3, 3) tensor of valid rotation matrices.
    """
    a1 = r6[..., 0:3]
    a2 = r6[..., 3:6]

    b1 = F.normalize(a1, dim=-1)
    a2_proj_b1 = (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(a2 - a2_proj_b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    R = torch.stack([b1, b2, b3], dim=-1)
    return R


def load_cad_points(cad_model_path: str, device: torch.device) -> torch.Tensor:
    """Load 3D CAD model points from the XReal XML or a .npy file.

    Supported formats:
        - .xml: expects the XReal-style landmark XML, where each
          <landmark id="i" x="..." y="..." z="..."/> entry defines
          one 3D point. Points are returned in ascending id order.
        - .npy: plain NumPy array of shape (N, 3).
    """
    if not os.path.isfile(cad_model_path):
        raise FileNotFoundError(f"CAD model file not found: {cad_model_path}")

    ext = os.path.splitext(cad_model_path)[1].lower()

    if ext == ".xml":
        tree = ET.parse(cad_model_path)
        root = tree.getroot()
        # Find all <landmark> elements under <landmarks>
        pts_dict = {}
        for lm in root.findall(".//landmark"):
            lm_id = lm.get("id")
            if lm_id is None:
                continue
            idx = int(lm_id)
            x = float(lm.get("x", "0.0"))
            y = float(lm.get("y", "0.0"))
            z = float(lm.get("z", "0.0"))
            pts_dict[idx] = (x, y, z)

        if not pts_dict:
            raise ValueError(f"No <landmark> entries found in XML: {cad_model_path}")

        # Sort by landmark id to get a deterministic (N,3) array
        max_id = max(pts_dict.keys())
        pts = []
        for i in range(max_id + 1):
            if i not in pts_dict:
                raise ValueError(
                    f"Missing landmark id={i} in XML model; ensure IDs are contiguous from 0."
                )
            pts.append(pts_dict[i])
        pts_arr = np.asarray(pts, dtype=np.float32)  # (N, 3)

    elif ext == ".npy":
        pts_arr = np.load(cad_model_path)
        if pts_arr.ndim != 2 or pts_arr.shape[1] != 3:
            raise ValueError("CAD model points must have shape (N, 3) for .npy files")
        pts_arr = pts_arr.astype(np.float32)

    else:
        raise ValueError(f"Unsupported CAD model file extension: {ext}. Use .xml or .npy")

    X = torch.from_numpy(pts_arr).to(device)
    return X  # (N, 3)


@register_model("glasses_pose_regressor")
class GlassesPoseRegressor(nn.Module):
    """Direct pose regressor for glasses from a single RGB image.

    - Backbone: MobileNetV2 encoder (no decoder).
    - Pose head: global pooled features -> 6D rotation + 3D translation.
    - Differentiable reprojection of a known 3D CAD model using given intrinsics.

    Forward outputs a dict with:
        {
            "R": (B, 3, 3),
            "t": (B, 3),
            "proj": (B, N, 2),   # projected 2D points in pixel coords
        }
    """

    def __init__(
        self,
        num_keypoints: int,  # kept for Trainer/API consistency, not used directly
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        cad_model_path: str,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)

        # Backbone: MobileNetV2 up to the last conv features
        backbone = mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
        self.backbone = backbone.features
        self.backbone_out_channels = 1280

        # Global average pooling + small MLP pose head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pose_head = nn.Sequential(
            nn.Linear(self.backbone_out_channels, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 9),  # 6D rotation + 3D translation
        )

        # CAD model will be registered as a buffer once we know the device
        self.register_buffer("cad_points", torch.empty(0, 3), persistent=False)
        self._cad_model_path = cad_model_path

    def _ensure_cad_loaded(self, device: torch.device) -> None:
        if self.cad_points.numel() == 0 or self.cad_points.device != device:
            X = load_cad_points(self._cad_model_path, device=device)
            # register as non-trainable buffer so it moves with .to(device)
            self.cad_points = X

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """Compute pose and reproject CAD model points.

        Args:
            x: (B, 3, H, W) input images.

        Returns:
            dict with keys: "R", "t", "proj".
        """
        device = x.device
        self._ensure_cad_loaded(device)
        B = x.shape[0]

        feat = self.backbone(x)  # (B, C, Hf, Wf)
        feat = self.global_pool(feat).view(B, -1)  # (B, C)

        pose_vec = self.pose_head(feat)  # (B, 9)
        r6 = pose_vec[:, 0:6]
        t = pose_vec[:, 6:9]
        R = rotation_6d_to_matrix(r6)  # (B, 3, 3)

        # Reproject CAD model points
        X_obj = self.cad_points  # (N, 3)
        X_cam = (R @ X_obj.t()).permute(0, 2, 1) + t[:, None, :]  # (B, N, 3)
        z = X_cam[..., 2].clamp(min=1e-6)
        x_norm = X_cam[..., 0] / z
        y_norm = X_cam[..., 1] / z

        u = self.fx * x_norm + self.cx
        v = self.fy * y_norm + self.cy
        proj = torch.stack([u, v], dim=-1)  # (B, N, 2)

        return {"R": R, "t": t, "proj": proj}

    def generate_sample_visualization(self, sample: dict, out_path: str, device: torch.device) -> None:
        """Generate a visualization for a single sample.

        Draws ground-truth landmarks in green and projected CAD points in red
        on top of the input image crop and saves to `out_path`.
        """
        self.eval()

        img = sample["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            out = self(img)

        if isinstance(out, dict):
            proj = out.get("proj")
        else:
            proj = out

        if proj is None:
            raise RuntimeError("Model did not return 'proj' for visualization")

        proj_np = proj[0].cpu().numpy()  # (N, 2)

        # Ground-truth: prefer explicit pose_keypoints_2d, fallback to 'keypoints'
        gt = sample.get("pose_keypoints_2d", None)
        if gt is None:
            k = sample.get("keypoints", None)
            if k is not None:
                gt = k[:, :2]

        if isinstance(gt, torch.Tensor):
            gt_np = gt.cpu().numpy()
        elif gt is None:
            gt_np = None
        else:
            gt_np = np.asarray(gt)

        img_np = sample["image"].detach().cpu().numpy()
        if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
            img_np = np.transpose(img_np, (1, 2, 0))

        # If normalization was applied, values may be outside [0,1]; clip for display
        img_np = np.clip(img_np, 0.0, 1.0)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img_np)

        if gt_np is not None:
            try:
                ax.scatter(gt_np[:, 0], gt_np[:, 1], c="lime", s=10, label="gt")
            except Exception:
                pass

        try:
            ax.scatter(proj_np[:, 0], proj_np[:, 1], c="red", s=10, label="proj")
        except Exception:
            pass

        ax.legend(loc="upper right", fontsize=6)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
