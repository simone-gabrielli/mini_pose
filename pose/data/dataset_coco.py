# pose/data/dataset_coco.py

import json
import os
from typing import Tuple
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from pose.data.heatmap import generate_heatmaps
from pose.registry import register_dataset
from pose.data.albu_aug import AlbumentationsKeypointPipeline


@register_dataset("coco_keypoints")
class CocoKeypointsDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        image_root: str,
        input_size: Tuple[int, int] = (256, 256),
        heatmap_size: Tuple[int, int] = (64, 64),
        sigma: float = 3.0,
        aug_cfg: dict | None = None,
    ):
        super().__init__()
        self.json_path = json_path
        self.image_root = image_root
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma

        with open(json_path, "r") as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.annotations = coco["annotations"]
        self.num_keypoints = len(coco["categories"][0]["keypoints"])

        flip_pairs = [
            (36,45),(37,44),(38,43),(39,42),(40,47),(41,46),
            (31,35),(32,34),
            (48,54),(49,53),(50,52),(51,51),
        ]

        aug_cfg = aug_cfg or {}
        # Optional: enable bbox-based cropping (FAN-style) during training.
        # If enabled in the config (data.aug.use_bbox_crop: true), we will use
        # the annotation bbox and expand it with a margin before applying
        # augmentations.
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

        # Base keypoints in original image coords
        kpts = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
        bbox = ann.get("bbox", None)

        # Optional bbox-based crop: use COCO bbox as face box and expand.
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
            kpts[:, 0] -= x1
            kpts[:, 1] -= y1

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        img_t, kpts_t = self.transform(img, kpts, bbox)

        heatmaps = generate_heatmaps(
            kpts_t.numpy(),
            heatmap_size=self.heatmap_size,
            image_size=self.input_size,
            sigma=self.sigma,
        )
        heatmaps_t = torch.from_numpy(heatmaps)

        visible = (kpts_t[:, 2] > 0).float()

        return {
            "image": img_t,
            "keypoints": kpts_t,
            "heatmaps": heatmaps_t,
            "visible": visible,
            "meta": {
                "image_path": img_path,
                "id": ann["image_id"],
            }
        }
