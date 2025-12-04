import albumentations as A
import numpy as np
import torch
import cv2


class AlbumentationsKeypointPipeline:
    def __init__(self,
                 input_size=(256, 256),
                 flip_pairs=None,
                 rotation=15,
                 scale=0.10,
                 color_jitter=0.15,
                 bbox_safe=True):
        """
        flip_pairs: list of tuples for horizontal flip keypoint swapping
        """
        self.input_size = input_size
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
                    p=0.7,
                ),
                A.ColorJitter(
                    brightness=color_jitter,
                    contrast=color_jitter,
                    saturation=color_jitter,
                    hue=color_jitter * 0.5,
                    p=0.4,
                ),
                A.Resize(height=input_size[1], width=input_size[0]),
            ],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            additional_targets={"image2": "image"},
        )

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

        # Preserve visibility in keypoints
        vis = keypoints[:, 2:3]
        new_kpts = np.concatenate([new_kpts, vis], axis=1)

        # Normalize and convert to tensor
        new_img = new_img.astype(np.float32) / 255.0
        new_img = new_img.transpose(2, 0, 1)

        return (
            torch.from_numpy(new_img).float(),
            torch.from_numpy(new_kpts).float(),
        )
