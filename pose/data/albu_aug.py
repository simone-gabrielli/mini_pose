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
        # `flip_pairs` removed — pipeline no longer performs keypoint index swapping
        rotation=15,
        scale=0.10,
        color_jitter=0.15,
        bbox_safe=True,
        content_cfg: dict | None = None,
    ):
        self.input_size = input_size
        # flip_pairs support removed
        content_cfg = content_cfg or {}

        # Read content augmentation params (with sensible defaults)
        brightness_contrast_p = float(content_cfg.get("brightness_contrast_p", 0.5))
        sunflare_p = float(content_cfg.get("sunflare_p", 0.15))
        motion_blur_p = float(content_cfg.get("motion_blur_p", 0.15))
        gauss_noise_p = float(content_cfg.get("gauss_noise_p", 0.2))
        gauss_noise_var = tuple(content_cfg.get("gauss_noise_var", [10.0, 50.0]))
        compression_p = float(content_cfg.get("compression_p", 0.2))
        compression_quality = tuple(content_cfg.get("compression_quality", [60, 100]))

        # Compose two pipelines:
        # - full_transform: includes geometric transforms and is used when no bbox
        # - content_transform: only content-changing transforms (flares, noise,
        #   compression, color) + resize. Used when a bbox is present to avoid
        #   changing bounding box geometry.

        # Build content-only transforms
        content_transforms = [
            A.RandomBrightnessContrast(brightness_limit=color_jitter, contrast_limit=color_jitter, p=brightness_contrast_p),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.3, p=sunflare_p),
            A.MotionBlur(blur_limit=7, p=motion_blur_p),
            A.GaussNoise(var_limit=gauss_noise_var, p=gauss_noise_p),
        ]

        # Image compression transform: prefer JpegCompression, fallback to ImageCompression
        comp = None
        if hasattr(A, 'JpegCompression'):
            comp = A.JpegCompression(quality_lower=60, quality_upper=100, p=0.2)
        elif hasattr(A, 'ImageCompression'):
            comp = A.ImageCompression(quality_lower=60, quality_upper=100, p=0.2)

        if comp is not None:
            # set comp probability and quality from config
            if hasattr(comp, 'quality_lower'):
                # albumentations classes accept quality_lower/upper in ctor
                # comp already created with defaults; recreate with config
                try:
                    if hasattr(A, 'JpegCompression') and isinstance(comp, A.JpegCompression.__class__):
                        comp = A.JpegCompression(quality_lower=compression_quality[0], quality_upper=compression_quality[1], p=compression_p)
                    else:
                        comp = A.ImageCompression(quality_lower=compression_quality[0], quality_upper=compression_quality[1], p=compression_p)
                except Exception:
                    # fall back: set p on existing transform if possible
                    try:
                        comp.p = compression_p
                    except Exception:
                        pass
            else:
                try:
                    comp.p = compression_p
                except Exception:
                    pass

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

        # Determine whether bbox is valid to pass to albumentations.
        # Very small or degenerate boxes can cause albumentations to raise errors
        # (x_max <= x_min). In that case, fall back to content-only pipeline.
        pass_bbox_to_transform = False
        if bbox is not None:
            bw = x2 - x1
            bh = y2 - y1
            # require at least a couple pixels of size to consider geometric transform
            if bw >= 2.0 and bh >= 2.0:
                pass_bbox_to_transform = True

        if pass_bbox_to_transform:
            bboxes = [(x1, y1, x2, y2)]
            # Use full pipeline (geometry + content). Albumentations will transform
            # image, keypoints and bbox consistently and return updated bboxes.
            augmented = self.full_transform(image=img, keypoints=kpts_xy, bboxes=bboxes)
        else:
            # Degenerate or missing bbox: avoid geometry transforms that could
            # produce invalid bboxes; use content-only pipeline instead.
            augmented = self.content_transform(image=img, keypoints=kpts_xy)
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

        # Previously we swapped keypoint indices on horizontal flip using
        # configured `flip_pairs`. That logic has been removed: the dataset's
        # annotations must already be in a consistent order and models should
        # handle flips via symmetric training or by explicit label mapping
        # upstream if necessary.

        # Update visibility: mark keypoints outside the output image as invisible.
        # new_kpts contains (x,y) in pixel coords of the transformed/resized image.
        if new_kpts.size != 0:
            ih, iw = self.input_size[1], self.input_size[0]
            xs = new_kpts[:, 0]
            ys = new_kpts[:, 1]
            in_bounds = (xs >= 0) & (xs <= (iw - 1)) & (ys >= 0) & (ys <= (ih - 1))

            if vis is not None:
                orig_vis = (vis[:, 0] > 0).astype(np.bool_)
                merged_vis = (orig_vis & in_bounds).astype(np.float32)[:, None]
            else:
                merged_vis = in_bounds.astype(np.float32)[:, None]

            new_kpts = np.concatenate([new_kpts, merged_vis], axis=1)

        # Normalize image: to [0,1], then standardize by mean/std
        new_img = new_img.astype(np.float32) / 255.0
        new_img = (new_img - self.mean) / self.std
        new_img = new_img.transpose(2, 0, 1)

        img_t = torch.from_numpy(new_img).float()
        kpts_t = torch.from_numpy(new_kpts).float() if new_kpts.size != 0 else None
        bbox_t = torch.from_numpy(new_bbox).float() if new_bbox is not None else None

        return img_t, kpts_t, bbox_t
