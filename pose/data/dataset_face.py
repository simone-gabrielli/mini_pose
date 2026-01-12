import json
import os
from typing import Tuple
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from pose.registry import register_dataset
from pose.data.albu_aug import AlbumentationsKeypointPipeline
import albumentations as A
from albumentations.pytorch import ToTensorV2


@register_dataset("coco_face")
class CocoFaceDataset(Dataset):
    """COCO-style dataset for a single-face bbox regression task.

    Returns dicts with keys: 'image' (Tensor), 'bbox' (Tensor [5]: conf, x1,y1,x2,y2)
    where bbox coords are normalized [0,1].
    """

    def __init__(
        self,
        json_path: str,
        image_root: str,
        input_size: Tuple[int, int] = (256, 256),
        heatmap_size: Tuple[int, int] = (64, 64),
        aug_cfg: dict | None = None,
    ):
        super().__init__()
        self.json_path = json_path
        self.image_root = image_root
        self.input_size = tuple(input_size)
        self.heatmap_size = tuple(heatmap_size)

        with open(json_path, "r") as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco.get("images", [])}
        # choose largest bbox per image as face target
        ann_by_img = {}
        for ann in coco.get("annotations", []):
            img_id = ann["image_id"]
            bbox = ann.get("bbox")
            if bbox is None:
                continue
            area = bbox[2] * bbox[3]
            if img_id not in ann_by_img or area > ann_by_img[img_id]["area"]:
                ann_by_img[img_id] = {"bbox": bbox, "area": area}

        self.items = []
        # Optionally include negative images (no face annotation). By default include them
        # so the detector can learn background scoring. We will include all images; if an
        # image has no annotation, `bbox` will be None and the dataset will emit a
        # confidence target of 0.
        for img_id, info in self.images.items():
            fname = info.get("file_name")
            path = os.path.join(image_root, fname)
            if not os.path.exists(path):
                continue

            if img_id in ann_by_img:
                bbox = ann_by_img[img_id]["bbox"]
                x, y, w, h = bbox
                x1, y1, x2, y2 = x, y, x + w, y + h
                self.items.append({"path": path, "bbox": [x1, y1, x2, y2], "width": info.get("width"), "height": info.get("height")})
            else:
                # negative sample: no face
                self.items.append({"path": path, "bbox": None, "width": info.get("width"), "height": info.get("height")})

        aug_cfg = aug_cfg or {}
        # Use stronger augmentation pipeline (bbox-aware). For face-only dataset
        # we don't have keypoints, so pass None for keypoints when calling.
        self.transform = AlbumentationsKeypointPipeline(
            input_size=input_size,
            rotation=aug_cfg.get("rotation", 15),
            scale=aug_cfg.get("scale", 0.10),
            color_jitter=aug_cfg.get("color_jitter", 0.15),
            content_cfg=aug_cfg.get("content", None),
        )

        # Augmentation factor: replicate items to enlarge training set
        aug_factor = int(aug_cfg.get("aug_factor", 1))
        if aug_factor > 1:
            orig = list(self.items)
            self.items = []
            for i in range(aug_factor):
                self.items.extend([dict(it) for it in orig])

        # expose same attributes Trainer expects (input_size, heatmap_size)
        self.num_keypoints = 0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if not hasattr(self, "_missing_image_warned"):
            self._missing_image_warned = set()

        max_tries = 20
        for attempt in range(max_tries):
            it = self.items[idx]
            img = cv2.imread(it["path"])
            if img is None:
                p = it.get("path")
                if p and p not in self._missing_image_warned:
                    print(f"[mini_pose][WARN] Missing/unreadable image: {p}. Skipping.")
                    self._missing_image_warned.add(p)
                idx = (idx + 1) % len(self.items)
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            break
        else:
            raise RuntimeError(
                f"Too many missing/unreadable images (>{max_tries}) while sampling from {self.json_path}."
            )
        H, W = img.shape[:2]
        bbox = it["bbox"]
        if bbox is None:
            # negative example: confidence 0, dummy box
            x1 = y1 = x2 = y2 = 0.0
            conf = 0.0
        else:
            x1 = bbox[0] / W
            y1 = bbox[1] / H
            x2 = bbox[2] / W
            y2 = bbox[3] / H
            conf = 1.0

        # Pass bbox through augmentation pipeline so flares/occlusions/flip affect it
        if bbox is None:
            img_t, _, bbox_t = self.transform(img, keypoints=None, bbox=None)
            # negative sample: keep zero bbox
            target = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        else:
            img_t, _, bbox_t = self.transform(img, keypoints=None, bbox=[bbox[0], bbox[1], bbox[2], bbox[3]])
            if bbox_t is None:
                # fallback to normalized coords from original size
                target = torch.tensor([conf, x1, y1, x2, y2], dtype=torch.float32)
            else:
                # bbox_t is normalized relative to input_size
                bx = bbox_t.numpy()
                target = torch.tensor([conf, bx[0], bx[1], bx[2], bx[3]], dtype=torch.float32)

        return {"image": img_t, "bbox": target, "meta": {"image_path": it["path"], "orig_size": (W, H)}}
