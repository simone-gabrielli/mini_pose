import json
import os
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from pose.data.heatmap import generate_heatmaps, generate_heatmaps_3d
from pose.data.albu_aug import AlbumentationsKeypointPipeline
from pose.registry import register_dataset


@register_dataset("coco_keypoints_3d")
class CocoKeypoints3DDataset(Dataset):
    """COCO-like dataset that returns 2D + 3D volumetric heatmaps.

    Expects per-annotation fields:
      - keypoints:     flattened list of K * 3 -> [x,y,v]
      - keypoints_3d:  flattened list of K * 4 -> [x,y,z,v]
    """

    def __init__(
        self,
        json_path: str,
        image_root: str,
        input_size: Tuple[int, int] = (256, 256),
        heatmap_size: Tuple[int, int] = (64, 64),
        sigma: float = 3.0,
        depth_bins: int = 8,
        depth_range: Tuple[float, float] | None = None,
        aug_cfg: dict | None = None,
    ):
        super().__init__()
        self.json_path = json_path
        self.image_root = image_root
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.depth_bins = depth_bins
        self.depth_range = depth_range

        with open(json_path, "r") as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.annotations = coco["annotations"]
        self.num_keypoints = len(coco["categories"][0]["keypoints"])

        flip_pairs = [
            (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
            (31, 35), (32, 34),
            (48, 54), (49, 53), (50, 52), (51, 51),
        ]

        aug_cfg = aug_cfg or {}
        self.use_bbox_crop = bool(aug_cfg.get("use_bbox_crop", False))
        self.face_margin = float(aug_cfg.get("face_margin", 0.25))
        self.transform = AlbumentationsKeypointPipeline(
            input_size=input_size,
            heatmap_size=heatmap_size,
            flip_pairs=flip_pairs,
            rotation=aug_cfg.get("rotation", 15),
            scale=aug_cfg.get("scale", 0.10),
            color_jitter=aug_cfg.get("color_jitter", 0.15),
            hflip_prob=aug_cfg.get("hflip_prob", 0.5),
        )

        # If depth_range not provided, compute a global range over the dataset
        if self.depth_range is None:
            zs = []
            for ann in self.annotations:
                k3 = np.array(ann.get("keypoints_3d", []), dtype=np.float32)
                if k3.size == self.num_keypoints * 4:
                    k3 = k3.reshape(-1, 4)
                    vis = k3[:, 3] > 0
                    if vis.any():
                        zs.append(k3[vis, 2])
            if len(zs) > 0:
                zs_all = np.concatenate(zs)
                zmin = float(zs_all.min())
                zmax = float(zs_all.max())
                zmean = float(zs_all.mean())
                if zmin == zmax:
                    zmin -= 0.5
                    zmax += 0.5
                self.depth_range = (zmin, zmax)
                self.depth_mean = zmean
            else:
                # fallback
                self.depth_range = (0.0, 1.0)
                self.depth_mean = float(0.5)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_info = self.images[ann["image_id"]]
        file_name = image_info["file_name"]
        img_path = os.path.join(self.image_root, file_name)

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(img_path)

        # 2D keypoints in original image coords
        kpts_2d = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
        kpts_3d_flat = np.array(ann["keypoints_3d"], dtype=np.float32)

        if kpts_3d_flat.size != self.num_keypoints * 4:
            raise ValueError(
                f"Expected keypoints_3d of length {self.num_keypoints * 4}, "
                f"got {kpts_3d_flat.size}"
            )
        kpts_3d = kpts_3d_flat.reshape(-1, 4)  # [x,y,z,v]

        bbox = ann.get("bbox", None)

        # Optional bbox-based crop
        if self.use_bbox_crop and bbox is not None:
            x, y, w, h = bbox
            cx = x + w / 2.0
            cy = y + h / 2.0
            size = max(w, h) * (1.0 + self.face_margin)
            x1 = int(cx - size / 2.0)
            y1 = int(cy - size / 2.0)
            x2 = int(cx + size / 2.0)
            y2 = int(cy + size / 2.0)

            H_orig, W_orig = img_bgr.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W_orig - 1, x2)
            y2 = min(H_orig - 1, y2)

            img_bgr = img_bgr[y1:y2, x1:x2]
            kpts_2d[:, 0] -= x1
            kpts_2d[:, 1] -= y1
            kpts_3d[:, 0] -= x1
            kpts_3d[:, 1] -= y1

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Apply Albumentations pipeline on 2D keypoints only; depth is invariant
        img_t, kpts_t = self.transform(img, kpts_2d, bbox)

        # 2D heatmaps for compatibility / visualization
        heatmaps_2d = generate_heatmaps(
            kpts_t.numpy(),
            heatmap_size=self.heatmap_size,
            image_size=self.input_size,
            sigma=self.sigma,
        )

        # Build keypoints3d in transformed 2D coords but same z,vis
        kpts3d_trans = np.zeros((self.num_keypoints, 4), dtype=np.float32)
        kpts3d_trans[:, 0] = kpts_t.numpy()[:, 0]
        kpts3d_trans[:, 1] = kpts_t.numpy()[:, 1]
        kpts3d_trans[:, 2] = kpts_3d[:, 2]
        kpts3d_trans[:, 3] = kpts_t.numpy()[:, 2]

        heatmaps_3d = generate_heatmaps_3d(
            kpts3d_trans,
            heatmap_size=self.heatmap_size,
            image_size=self.input_size,
            depth_bins=self.depth_bins,
            depth_range=self.depth_range,
            sigma_spatial=self.sigma,
            sigma_depth=1.0,
        )

        heatmaps_2d_t = torch.from_numpy(heatmaps_2d)
        heatmaps_3d_t = torch.from_numpy(heatmaps_3d)
        visible = (kpts_t[:, 2] > 0).float()

        return {
            "image": img_t,
            "keypoints": kpts_t,
            "heatmaps": heatmaps_2d_t,
            "heatmaps_3d": heatmaps_3d_t,
            "visible": visible,
            "meta": {"image_path": img_path, "id": ann["image_id"]},
        }
