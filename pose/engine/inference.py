"""Small inference helpers.

This module is intentionally *minimal*: it provides just enough utilities for the
CLI scripts in `scripts/` to:
    - instantiate a registered model
    - load weights
    - decode heatmaps into (x,y) keypoints

Important limitations / gotchas:
    - `load_model()` is designed for *heatmap-style* models. It uses a small
        heuristic to infer hourglass stack counts from checkpoint keys.
    - For coordinate-regression models (e.g. LOTR) prefer the config-driven loader
        implemented in `scripts/infer.py` / `scripts/infer_video.py`.
"""

from __future__ import annotations

import inspect
from typing import Tuple

import numpy as np
import torch

from pose.registry import MODEL_REGISTRY

# Ensure model modules are imported so they register themselves in MODEL_REGISTRY.
# (The registry is populated by decorators that run at import time.)
import pose.models  # noqa: F401


def load_model(checkpoint_path, model_name, num_keypoints, device="cuda", weights_only=True):
    """Instantiate a registered heatmap model and load weights.

    Args:
        checkpoint_path: Path to a checkpoint. Expected formats:
          - dict with key "model" -> state_dict
          - raw state_dict
        model_name: Key in `MODEL_REGISTRY`.
        num_keypoints: Number of output keypoint channels (K).
        device: Target device string ("cuda" or "cpu").
        weights_only: If supported by your PyTorch version, avoids loading
          non-tensor pickled objects.

    Returns:
        Model in eval() mode on the requested device.
    """
    try:
        ModelCls = MODEL_REGISTRY[model_name]
    except KeyError:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(f"Model '{model_name}' not found in MODEL_REGISTRY. Available models: {available}")

    # Load checkpoint first so we can attempt to infer model constructor args.
    # For stacked hourglass variants, the checkpoint keys often include patterns like
    #   hourglasses.0., hourglasses.1., ...
    # which lets us infer `num_stacks`.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=weights_only)
    # checkpoint may either be a mapping with key 'model' or already be a state_dict
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

    # Try to infer number of hourglass stacks from checkpoint keys.
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
    # This keeps the helper usable across multiple model families.
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

    # NOTE: strict loading here is intentional: if you want to tolerate missing keys
    # (e.g. architectural tweaks), use the config-driven inference path.
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def decode_heatmaps(heatmaps: torch.Tensor) -> np.ndarray:
    """Argmax-decode heatmaps into keypoint coordinates.

    Args:
        heatmaps: CPU tensor shaped `(K, H, W)`.

    Returns:
        `(K, 2)` float array of `(x, y)` in *heatmap pixel coordinates*.

    Notes:
        - This is a simple argmax decoder (no subpixel refinement).
        - To map to original image pixels, scale by the resize ratio used
          for preprocessing.
    """
    K, H, W = heatmaps.shape
    heatmaps = heatmaps.view(K, -1)
    idx = torch.argmax(heatmaps, dim=1)
    ys = (idx // W).float()
    xs = (idx % W).float()
    coords = torch.stack([xs, ys], dim=1)
    return coords.numpy()
