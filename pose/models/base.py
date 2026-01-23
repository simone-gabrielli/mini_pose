"""Model interface notes.

mini-pose supports multiple *output styles* depending on the task:

1) Heatmap-based keypoints (StackedHourglass / FAN / MobileNetPose)
   Typical forward output:
     - `(pred_last, preds_all)` where
         - `pred_last` is a tensor shaped `(B, K, Hh, Wh)`
         - `preds_all` is a list of intermediate tensors for deep supervision

2) Coordinate regression (LOTR-style)
   Typical forward output:
     - `(coords_norm, coords_pixel)` where `coords_pixel` is `(B, K, 2)` (or `(B, K, 3)`)
     - or a single tensor of coordinates

3) Special tasks (e.g. pose reprojection, bbox detector)
   Models may return a dict with well-known keys used by the losses/scripts.

The Trainer/inference scripts contain adapter logic to handle these variants.
This base class is intentionally lightweight: it documents the convention but
does not enforce a single rigid return type.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class PoseModel(nn.Module, ABC):
    """Common base class for models used in mini-pose.

    Subclasses should implement :meth:`forward` and follow one of the output
    conventions described in the module docstring.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """Run a forward pass.

        Args:
            x: Input images shaped `(B, C, H, W)`.

        Returns:
            One of:
              - Heatmap tensor `(B, K, Hh, Wh)`
              - `(heatmaps_last, heatmaps_all)`
              - Coordinate tensor(s) `(B, K, 2/3)`
              - A dict containing task-specific outputs
        """
        raise NotImplementedError
