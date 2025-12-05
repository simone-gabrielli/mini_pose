import albumentations as A
import numpy as np
import torch
import cv2


class AlbumentationsKeypointPipeline:
    """A stronger albumentations pipeline for keypoints and optional bboxes.

    - Supports sun flares, coarse dropouts (occlusion), motion blur, jpeg
    - Uses ReplayCompose to detect if a HorizontalFlip was applied so we can
      swap keypoint indices according to `flip_pairs`.
    - If `aug_factor` is used at the dataset level, calling this pipeline
      repeatedly will produce different augmented samples.

    Returns tuple `(image_tensor, keypoints_tensor, bbox_tensor_or_None)`.
    Keypoints tensor shape: (K,3) -> (x,y,vis)
    BBox tensor: (4,) normalized [x1,y1,x2,y2] relative to `input_size`.
    """

    def __init__(
        self,
        input_size=(256, 256),
        flip_pairs=None,
        rotation=15,
        scale=0.10,
        color_jitter=0.15,
        bbox_safe=True,
    ):
        self.input_size = input_size
        self.flip_pairs = flip_pairs or []

        # Compose two pipelines:
        # - full_transform: includes geometric transforms and is used when no bbox
        # - content_transform: only content-changing transforms (flares, noise,
        #   compression, color) + resize. Used when a bbox is present to avoid
        #   changing bounding box geometry.

        # Build content-only transforms
        content_transforms = [
            A.RandomBrightnessContrast(brightness_limit=color_jitter, contrast_limit=color_jitter, p=0.5),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.3, p=0.15),
            A.MotionBlur(blur_limit=7, p=0.15),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        ]

        # Image compression transform: prefer JpegCompression, fallback to ImageCompression
        comp = None
        if hasattr(A, 'JpegCompression'):
            comp = A.JpegCompression(quality_lower=60, quality_upper=100, p=0.2)
        elif hasattr(A, 'ImageCompression'):
            comp = A.ImageCompression(quality_lower=60, quality_upper=100, p=0.2)

        if comp is not None:
            # place compression among content transforms
            content_transforms.insert(1, comp)

        # content pipeline: content-only + resize
        content_transforms.append(A.Resize(height=input_size[1], width=input_size[0]))

        self.content_transform = A.ReplayCompose(
            content_transforms,
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            additional_targets={"image2": "image"},
        )

        # full pipeline: geometry + content + resize
        geom_transforms = [
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.Affine(scale=(1 - scale, 1 + scale), rotate=(-rotation, rotation), interpolation=cv2.INTER_LINEAR, mode=cv2.BORDER_CONSTANT),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=scale, rotate_limit=rotation, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT),
            ], p=0.8),
        ]
        geom_transforms.extend(content_transforms)

        self.full_transform = A.ReplayCompose(
            geom_transforms,
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=[]),
            additional_targets={"image2": "image"},
        )

        # ImageNet normalization constants (matches typical A.Normalize default)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(self, img, keypoints=None, bbox=None):
        # img: HxWxC (RGB)
        kpts_xy = []
        vis = None
        if keypoints is not None and len(keypoints) > 0:
            # keypoints expected as (K,3)
            kpts_xy = keypoints[:, :2].tolist()
            vis = keypoints[:, 2:3]
        else:
            kpts_xy = []

        bboxes = []
        new_bbox = None
        if bbox is not None:
            # Accept several incoming bbox formats and convert to pascal_voc
            # in pixel coordinates: (x1,y1,x2,y2). We'll NOT pass bboxes to
            # albumentations to avoid geometric transforms affecting them.
            H_img, W_img = img.shape[:2]
            b = list(bbox)
            # If values are normalized in [0,1], treat as x1,y1,x2,y2 normalized
            if max(b) <= 1.0:
                x1 = float(b[0]) * W_img
                y1 = float(b[1]) * H_img
                x2 = float(b[2]) * W_img
                y2 = float(b[3]) * H_img
            else:
                # Otherwise assume COCO-style (x,y,w,h) in pixels OR pascal_voc pixels
                x = float(b[0])
                y = float(b[1])
                w = float(b[2])
                h = float(b[3])
                if w <= 1.0 and h <= 1.0:
                    x1 = x * W_img
                    y1 = y * H_img
                    x2 = (x + w) * W_img
                    y2 = (y + h) * H_img
                else:
                    x1 = x
                    y1 = y
                    x2 = x + w
                    y2 = y + h

            # Clamp to image bounds
            x1 = max(0.0, min(x1, W_img - 1.0))
            y1 = max(0.0, min(y1, H_img - 1.0))
            x2 = max(0.0, min(x2, W_img - 1.0))
            y2 = max(0.0, min(y2, H_img - 1.0))

            # Normalized bbox relative to original image size. Since we will only
            # apply content transforms + resize, normalized coords remain valid.
            new_bbox = np.array([x1 / W_img, y1 / H_img, x2 / W_img, y2 / H_img], dtype=np.float32)

        # Choose content-only pipeline when bbox is present to avoid geometry
        if bbox is not None:
            augmented = self.content_transform(image=img, keypoints=kpts_xy)
        else:
            # No bbox: safe to run geometric transforms which also update keypoints
            augmented = self.full_transform(image=img, keypoints=kpts_xy, bboxes=bboxes)
        new_img = augmented["image"]
        new_kpts = np.array(augmented.get("keypoints", []), dtype=np.float32)

        # If augmented keypoint count doesn't match original visibility length
        # (some albumentations transforms can behave oddly in replay mode),
        # fall back to a safe rescaling of the original keypoints to the
        # pipeline `input_size`. This preserves keypoint count and avoids
        # crashes while keeping content-only augmentation effects.
        H_img, W_img = img.shape[:2]
        if vis is not None and new_kpts.size != 0 and new_kpts.shape[0] != vis.shape[0]:
            try:
                scale_x = float(self.input_size[0]) / float(W_img)
                scale_y = float(self.input_size[1]) / float(H_img)
                if len(kpts_xy) > 0:
                    resized = [[x * scale_x, y * scale_y] for (x, y) in kpts_xy]
                    new_kpts = np.array(resized, dtype=np.float32)
                else:
                    new_kpts = np.zeros((vis.shape[0], 2), dtype=np.float32)
            except Exception:
                # Last resort: create zeros matching visibility length
                new_kpts = np.zeros((vis.shape[0], 2), dtype=np.float32)

        # If a bbox was passed, get the transformed bbox (in pixels of resized image)
        new_bbox = None
        if "bboxes" in augmented and len(augmented["bboxes"]) > 0:
            bx = augmented["bboxes"][0]
            # bx is (x1,y1,x2,y2) in pixel coords of the resized image (input_size)
            ih, iw = self.input_size[1], self.input_size[0]
            # normalize to [0,1]
            new_bbox = np.array([bx[0] / iw, bx[1] / ih, bx[2] / iw, bx[3] / ih], dtype=np.float32)

        # If a horizontal flip was applied, swap keypoint indices according to flip_pairs
        replay = augmented.get("replay", None)
        flipped = False
        if replay is not None:
            for t in replay.get("transforms", []):
                if t.get("name", "") == "HorizontalFlip" and t.get("applied", False):
                    flipped = True
                    break

        if flipped and new_kpts.size != 0 and len(self.flip_pairs) > 0:
            # swap x,y pairs
            for a, b in self.flip_pairs:
                if a < new_kpts.shape[0] and b < new_kpts.shape[0]:
                    tmp = new_kpts[a].copy()
                    new_kpts[a] = new_kpts[b]
                    new_kpts[b] = tmp

        # Restore visibility channel if provided
        if vis is not None and new_kpts.size != 0:
            new_kpts = np.concatenate([new_kpts, vis], axis=1)
        elif new_kpts.size != 0:
            # If no visibility information, append ones
            ones = np.ones((new_kpts.shape[0], 1), dtype=np.float32)
            new_kpts = np.concatenate([new_kpts, ones], axis=1)

        # Normalize image: to [0,1], then standardize by mean/std
        new_img = new_img.astype(np.float32) / 255.0
        new_img = (new_img - self.mean) / self.std
        new_img = new_img.transpose(2, 0, 1)

        img_t = torch.from_numpy(new_img).float()
        kpts_t = torch.from_numpy(new_kpts).float() if new_kpts.size != 0 else None
        bbox_t = torch.from_numpy(new_bbox).float() if new_bbox is not None else None

        return img_t, kpts_t, bbox_t
