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
                # Keep COCO bbox format (x, y, w, h) in pixels.
                # The augmentation pipeline expects xywh when values are > 1.
                x, y, w, h = bbox
                self.items.append({"path": path, "bbox": [x, y, w, h], "width": info.get("width"), "height": info.get("height")})
            else:
                # negative sample: no face
                self.items.append({"path": path, "bbox": None, "width": info.get("width"), "height": info.get("height")})

        aug_cfg = aug_cfg or {}
        # Use stronger augmentation pipeline (bbox-aware). For face-only dataset
        # we don't have keypoints, so pass None for keypoints when calling.
        self.transform = AlbumentationsKeypointPipeline(
            input_size=input_size,
            enabled=bool(aug_cfg.get("enabled", True)),
            hflip_p=aug_cfg.get("hflip_p", 0.0),
            rotation=aug_cfg.get("rotation", 15),
            scale=aug_cfg.get("scale", 0.10),
            color_jitter=aug_cfg.get("color_jitter", 0.15),
            content_cfg=aug_cfg.get("content", None),
            occlusion_cfg=aug_cfg.get("occlusion", None),
            keep_oob_visible=bool(aug_cfg.get("keep_oob_visible", False)),
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
            # bbox is COCO xywh in pixels
            x, y, w, h = bbox
            x1 = float(x) / float(W)
            y1 = float(y) / float(H)
            x2 = float(x + w) / float(W)
            y2 = float(y + h) / float(H)
            conf = 1.0

        # Pass bbox through augmentation pipeline so flares/occlusions/flip affect it
        if bbox is None:
            img_t, _, bbox_t = self.transform(img, keypoints=None, bbox=None)
            # negative sample: keep zero bbox
            target = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        else:
            # pass bbox as COCO xywh in pixels
            x, y, w, h = bbox
            img_t, _, bbox_t = self.transform(img, keypoints=None, bbox=[x, y, w, h])
            if bbox_t is None:
                # fallback to normalized coords from original size
                target = torch.tensor([conf, x1, y1, x2, y2], dtype=torch.float32)
            else:
                # bbox_t is normalized relative to input_size
                bx = bbox_t.numpy()
                target = torch.tensor([conf, bx[0], bx[1], bx[2], bx[3]], dtype=torch.float32)

        return {"image": img_t, "bbox": target, "meta": {"image_path": it["path"], "orig_size": (W, H)}}
