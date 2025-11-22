# pose/data/transforms.py

import cv2
import numpy as np
import torch


class BasicTransform:
    def __init__(self, input_size=(256, 256)):
        self.input_size = input_size

    def __call__(self, img, keypoints, bbox=None):
        """
        img: HxWx3 BGR or RGB
        keypoints: (K,3) [x,y,vis] in original image coords
        bbox: [x,y,w,h] or None
        """

        h, w = img.shape[:2]
        target_w, target_h = self.input_size

        # for now: simple resize (you can later implement affine crop around bbox)
        scale_x = target_w / w
        scale_y = target_h / h

        img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        kpts = keypoints.copy()
        kpts[:, 0] *= scale_x
        kpts[:, 1] *= scale_y

        img_resized = img_resized.astype(np.float32) / 255.0
        img_resized = (img_resized - 0.5) / 0.5  # simple normalization to [-1,1]
        img_resized = img_resized.transpose(2, 0, 1)  # C,H,W

        return torch.from_numpy(img_resized), torch.from_numpy(kpts)
