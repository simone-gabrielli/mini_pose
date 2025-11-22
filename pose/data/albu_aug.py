import albumentations as A
import numpy as np
import torch
import cv2


class AlbumentationsKeypointPipeline:
    def __init__(self,
                 input_size=(256, 256),
                 heatmap_size=(64, 64),
                 flip_pairs=None,
                 rotation=40,
                 scale=0.30,
                 bbox_safe=True):
        """
        flip_pairs: list of tuples for horizontal flip keypoint swapping
        """
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.flip_pairs = flip_pairs or []

        # Define Albumentations pipeline
        self.transform = A.Compose(
            [
                A.Affine(
                    scale=(1 - scale, 1 + scale),
                    rotate=(-rotation, rotation),
                    translate_percent=None,
                    fit_output=False,
                    interpolation=cv2.INTER_LINEAR,
                    mode=cv2.BORDER_CONSTANT,
                    p=1.0
                ),
                A.HorizontalFlip(
                    p=0.5
                ),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.6
                ),
                A.Resize(height=input_size[1], width=input_size[0]),
            ],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            additional_targets={"image2": "image"},
        )

    def _swap_keypoints(self, kpts):
        """
        Swap keypoints after horizontal flip.
        """
        kpts = kpts.copy()
        for a, b in self.flip_pairs:
            kpts[[a, b]] = kpts[[b, a]]
        return kpts

    def __call__(self, img, keypoints, bbox=None):
        # Albumentations uses (x,y); your keypoints array is (K,3)
        kpts_xy = keypoints[:, :2].tolist()

        # Apply transformation
        augmented = self.transform(
            image=img,
            keypoints=kpts_xy,
        )
        new_img = augmented["image"]
        new_kpts = np.array(augmented["keypoints"], dtype=np.float32)

        # Detect flip
        if augmented.get("replay", None):
            # Not used here, but available
            pass

        # HorizontalFlip in Albumentations implicitly flips coords,
        # but we must also swap L/R keypoints
        if augmented.get("flipped", False):
            new_kpts = self._swap_keypoints(new_kpts)

        # Preserve visibility in keypoints
        vis = keypoints[:, 2:3]
        new_kpts = np.concatenate([new_kpts, vis], axis=1)

        # Normalize and convert to tensor
        new_img = new_img.astype(np.float32) / 255.0
        new_img = (new_img - 0.5) / 0.5
        new_img = new_img.transpose(2, 0, 1)

        return (
            torch.from_numpy(new_img).float(),
            torch.from_numpy(new_kpts).float(),
        )
