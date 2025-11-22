# pose/data/dataset_coco.py

import json
import os
from typing import Tuple
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from pose.data.heatmap import generate_heatmaps
from pose.data.transforms import BasicTransform
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
        sigma: float = 2.0,
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

        self.transform = AlbumentationsKeypointPipeline(
            input_size=input_size,
            heatmap_size=heatmap_size,
            flip_pairs=flip_pairs,
            rotation=40,
            scale=0.30,
        )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_info = self.images[ann["image_id"]]
        file_name = image_info["file_name"]
        img_path = os.path.join(self.image_root, file_name)

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        kpts = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
        bbox = ann.get("bbox", None)

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
