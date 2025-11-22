# pose/models/base.py

import torch.nn as nn
from abc import ABC, abstractmethod


class PoseModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        """Return heatmaps of shape (B, num_keypoints, H, W)."""
        pass
