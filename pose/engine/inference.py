# pose/engine/inference.py

import torch
import cv2
import numpy as np
from typing import Tuple
from pose.registry import MODEL_REGISTRY
# Ensure model modules are imported so they register themselves in MODEL_REGISTRY
import pose.models  # noqa: F401
from pose.data.transforms import BasicTransform
import inspect


def load_model(checkpoint_path, model_name, num_keypoints, device="cuda", weights_only=True):
    try:
        ModelCls = MODEL_REGISTRY[model_name]
    except KeyError:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(f"Model '{model_name}' not found in MODEL_REGISTRY. Available models: {available}")

    # Load checkpoint first so we can attempt to infer model constructor args
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=weights_only)
    # checkpoint may either be a mapping with key 'model' or already be a state_dict
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

    # Try to infer number of hourglass stacks from checkpoint keys (e.g. 'hourglasses.2.')
    num_stacks_in_ckpt = None
    try:
        import re

        keys = list(state_dict.keys())
        hourglass_idxs = [int(m.group(1)) for k in keys for m in [re.search(r"hourglasses\.(\d+)\.", k)] if m]
        if hourglass_idxs:
            num_stacks_in_ckpt = max(hourglass_idxs) + 1
            print(f"Detected {num_stacks_in_ckpt} hourglass stacks from checkpoint keys.")
    except Exception:
        num_stacks_in_ckpt = None

    # Build a kwargs dict and pass only parameters that the model's constructor accepts.
    candidate_kwargs = {
        "num_stacks": num_stacks_in_ckpt if num_stacks_in_ckpt is not None else 2,
        "num_blocks": 1,
        "num_modules": 1,
        "num_feats": 256,
        "num_keypoints": num_keypoints,
    }

    sig = inspect.signature(ModelCls)
    filtered = {k: v for k, v in candidate_kwargs.items() if k in sig.parameters}
    print(f"Instantiating {ModelCls.__name__} with kwargs: {filtered}")

    model = ModelCls(**filtered)

    model.load_state_dict(state_dict)
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
