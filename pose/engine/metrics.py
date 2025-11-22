import numpy as np
import torch

def compute_pck(preds, targets, thresh=0.05):
    """
    preds, targets: (B, K, 2)
    thresh: relative threshold (percentage of image size)
    """
    B, K, _ = preds.shape
    # Distance normalization using torso size or image diag
    diag = np.linalg.norm(np.array([256, 256]))  # approximate input resolution

    dists = np.linalg.norm(preds - targets, axis=2) / diag
    correct = (dists < thresh).sum()
    total = B * K
    return correct / total


def compute_nme(preds, targets):
    """
    Normalized Mean Error using inter-ocular distance.
    preds, targets: (B, K, 2)
    """
    B, K, _ = preds.shape

    # Inter-ocular distance = distance between keypoints 36 and 45 (68pt)
    interocular = np.linalg.norm(targets[:, 36] - targets[:, 45], axis=1)

    dists = np.linalg.norm(preds - targets, axis=2).mean(axis=1)
    nme = (dists / interocular).mean()
    return nme
