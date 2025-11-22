# pose/engine/inference.py

import torch
import cv2
import numpy as np
from typing import Tuple
from pose.registry import MODEL_REGISTRY
from pose.data.transforms import BasicTransform


def load_model(checkpoint_path, model_name, num_keypoints, device="cuda"):
    ModelCls = MODEL_REGISTRY[model_name]
    model = ModelCls(
        num_stacks=2,
        num_blocks=1,
        num_feats=256,
        num_keypoints=num_keypoints,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def decode_heatmaps(heatmaps: torch.Tensor) -> np.ndarray:
    """
    heatmaps: (K,H,W) tensor on CPU
    returns: (K,2) array with (x,y)
    """
    K, H, W = heatmaps.shape
    heatmaps = heatmaps.view(K, -1)
    idx = torch.argmax(heatmaps, dim=1)
    ys = (idx // W).float()
    xs = (idx % W).float()
    coords = torch.stack([xs, ys], dim=1)
    return coords.numpy()
