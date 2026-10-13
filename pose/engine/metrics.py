"""Keypoint evaluation metrics.

All coordinates and normalizers must be expressed in the same coordinate space.
For example, heatmap predictions should be paired with the heatmap size, while
direct coordinate predictions should be paired with the model input size.
"""

import numpy as np


def _arrays(preds, targets, visible=None):
    preds = np.asarray(preds, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if preds.shape != targets.shape or preds.ndim != 3 or preds.shape[-1] != 2:
        raise ValueError("preds and targets must have matching shape (B, K, 2)")

    if visible is None:
        mask = np.ones(preds.shape[:2], dtype=bool)
    else:
        mask = np.asarray(visible) > 0
        if mask.shape != preds.shape[:2]:
            raise ValueError("visible must have shape (B, K)")

    mask &= np.isfinite(preds).all(axis=2) & np.isfinite(targets).all(axis=2)
    return preds, targets, mask


def compute_pck(preds, targets, thresh=0.05, image_size=(256, 256), visible=None):
    """Percentage of Correct Keypoints using an image-diagonal threshold.

    ``image_size`` is ``(width, height)`` in the coordinate space used by
    ``preds`` and ``targets``. Invisible/non-finite landmarks are excluded.
    """
    preds, targets, mask = _arrays(preds, targets, visible)
    width, height = (float(image_size[0]), float(image_size[1]))
    diagonal = float(np.hypot(width, height))
    if diagonal <= 0:
        raise ValueError("image_size must have positive width and height")

    distances = np.linalg.norm(preds - targets, axis=2)
    valid_count = int(mask.sum())
    if valid_count == 0:
        return float("nan")
    return float(((distances / diagonal < float(thresh)) & mask).sum() / valid_count)


def compute_nme(preds, targets, visible=None, interocular_indices=(36, 45)):
    """Normalized Mean Error over visible landmarks.

    For 68-point layouts the default normalizer is the distance between points
    36 and 45. If those indices do not exist, each sample's visible-landmark
    bounding-box diagonal is used instead so the metric remains defined for
    custom landmark layouts.
    """
    preds, targets, mask = _arrays(preds, targets, visible)
    batch_size, num_keypoints, _ = preds.shape
    i, j = int(interocular_indices[0]), int(interocular_indices[1])
    use_interocular = 0 <= i < num_keypoints and 0 <= j < num_keypoints

    sample_values = []
    for b in range(batch_size):
        valid = mask[b]
        if not valid.any():
            continue

        if use_interocular:
            if not (valid[i] and valid[j]):
                continue
            normalizer = float(np.linalg.norm(targets[b, i] - targets[b, j]))
        else:
            visible_targets = targets[b, valid]
            span = visible_targets.max(axis=0) - visible_targets.min(axis=0)
            normalizer = float(np.linalg.norm(span))

        if not np.isfinite(normalizer) or normalizer <= 1e-8:
            continue
        mean_error = float(np.linalg.norm(preds[b, valid] - targets[b, valid], axis=1).mean())
        sample_values.append(mean_error / normalizer)

    if not sample_values:
        return float("nan")
    return float(np.mean(sample_values))
