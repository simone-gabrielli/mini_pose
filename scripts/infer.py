"""Inference helper.

Supports:
- Single image inference via --image (legacy behavior)
- COCO-style JSON inference via --coco-json, sampling a percentage of entries

Saves visualizations with predicted landmarks + ground-truth landmarks.
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch

from pose.engine.inference import decode_heatmaps, load_model
from pose.detectors.haar_face import HaarFaceDetector


@dataclass(frozen=True)
class CocoSample:
    image_id: int
    ann_id: Optional[int]
    image_path: str
    bbox_xywh: Optional[Tuple[float, float, float, float]]
    keypoints: Optional[np.ndarray]  # (K,3) in original image coords


def _safe_mkdir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _expand_bbox_xywh(
    bbox_xywh: Tuple[float, float, float, float],
    margin: float,
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox_xywh
    cx = x + w / 2.0
    cy = y + h / 2.0
    size = max(w, h) * (1.0 + margin)
    # Match training dataset cropping exactly (see pose/data/dataset_coco.py)
    x1 = int(cx - size / 2.0)
    y1 = int(cy - size / 2.0)
    x2 = int(cx + size / 2.0)
    y2 = int(cy + size / 2.0)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w - 1, x2)
    y2 = min(img_h - 1, y2)
    return x1, y1, x2, y2


def _crop_from_xyxy(img_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    # keep Python slicing semantics predictable
    if x2 <= x1 or y2 <= y1:
        return img_bgr
    return img_bgr[y1:y2, x1:x2]


def _detect_face_crop_xyxy(
    detector: HaarFaceDetector,
    img_bgr: np.ndarray,
    margin: float,
) -> Optional[Tuple[int, int, int, int]]:
    boxes = detector.detect(img_bgr)
    if not boxes:
        return None

    x, y, w, h = boxes[0]
    H, W = img_bgr.shape[:2]
    return _expand_bbox_xywh((float(x), float(y), float(w), float(h)), margin=margin, img_w=W, img_h=H)


def _infer_landmarks_on_crop(
    model,
    device: torch.device,
    face_bgr: np.ndarray,
    num_keypoints: int,
    input_size: Tuple[int, int],
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Returns predicted coords in resized crop pixel space + heatmap size.

    Important: preprocessing here mirrors the training/val pipeline:
    - RGB input
    - resize to input_size
    - ImageNet mean/std normalization
    """

    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    W_face, H_face = input_size
    face_rgb = cv2.resize(face_rgb, (W_face, H_face), interpolation=cv2.INTER_LINEAR)

    img = face_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    img_t = torch.from_numpy(img).float().unsqueeze(0).to(device)

    def _extract_preds_last(model_out: object) -> torch.Tensor:
        # Match Trainer._visualize_model_outputs selection logic
        if isinstance(model_out, tuple):
            if len(model_out) >= 1 and isinstance(model_out[0], torch.Tensor):
                return model_out[0]
            if len(model_out) >= 2 and isinstance(model_out[1], (list, tuple)):
                return model_out[1][-1]
        if isinstance(model_out, torch.Tensor):
            return model_out
        raise TypeError(f"Unsupported model output type for heatmaps: {type(model_out)}")

    with torch.no_grad():
        out = model(img_t)
        preds_last = _extract_preds_last(out)
        heatmaps = preds_last[0].detach().cpu()  # (K,Hh,Wh)

    coords_hm = decode_heatmaps(heatmaps)  # (K,2) in heatmap space
    H_hm, W_hm = heatmaps.shape[1:]

    # scale hm coords to resized crop coords
    coords_px = coords_hm.copy()
    coords_px[:, 0] *= (W_face / float(W_hm))
    coords_px[:, 1] *= (H_face / float(H_hm))

    return coords_px, (H_hm, W_hm)


def _draw_keypoints(img_bgr: np.ndarray, kpts: np.ndarray, color_bgr: Tuple[int, int, int], radius: int) -> None:
    for xk, yk, vis in kpts:
        if vis <= 0:
            continue
        cv2.circle(img_bgr, (int(round(xk)), int(round(yk))), radius, color_bgr, -1)


def _load_coco_samples(
    coco_json_path: str,
    images_root: str,
    sample_percent: float,
    seed: int,
    max_samples: Optional[int],
) -> List[CocoSample]:
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images_by_id = {int(img["id"]): img for img in coco.get("images", [])}
    annotations = coco.get("annotations", [])
    if not annotations:
        raise ValueError(f"No annotations found in {coco_json_path}")

    # Build one sample per annotation (common for face datasets).
    samples: List[CocoSample] = []
    coco_dir = Path(coco_json_path).parent

    for ann in annotations:
        image_id = int(ann["image_id"])
        image_info = images_by_id.get(image_id)
        if image_info is None:
            continue

        file_name = image_info.get("file_name")
        if not file_name:
            continue

        # Support both absolute file_name and relative file_name
        img_path = Path(file_name)
        if not img_path.is_absolute():
            # Prefer explicit images_root; fallback to JSON folder
            root = Path(images_root) if images_root else coco_dir
            img_path = root / file_name

        bbox = ann.get("bbox", None)
        bbox_xywh = None
        if bbox is not None and len(bbox) == 4:
            bbox_xywh = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

        kpts = ann.get("keypoints", None)
        kpts_arr = None
        if kpts is not None:
            kpts_arr = np.array(kpts, dtype=np.float32).reshape(-1, 3)

        samples.append(
            CocoSample(
                image_id=image_id,
                ann_id=int(ann.get("id")) if "id" in ann else None,
                image_path=str(img_path),
                bbox_xywh=bbox_xywh,
                keypoints=kpts_arr,
            )
        )

    if not samples:
        raise ValueError(f"No usable samples found in {coco_json_path}")

    pct = float(sample_percent)
    if pct <= 0:
        return []
    pct = min(100.0, pct)

    rng = random.Random(seed)
    n = max(1, int(round(len(samples) * (pct / 100.0))))
    n = min(n, len(samples))
    picked = rng.sample(samples, n)
    if max_samples is not None:
        picked = picked[: int(max_samples)]
    return picked

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-name", default="stacked_hourglass")
    parser.add_argument("--num-keypoints", type=int, default=68)
    parser.add_argument("--device", default="cuda")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Single image path")
    src.add_argument("--coco-json", help="COCO style annotation JSON (e.g. val.json)")

    parser.add_argument(
        "--images-root",
        default="",
        help="Root folder for COCO images (joined with images[].file_name if relative).",
    )
    parser.add_argument("--out-dir", default="outputs/infer", help="Output folder for visualizations")
    parser.add_argument("--sample-percent", type=float, default=10.0, help="Percent of COCO annotations to run")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of samples")

    # Default to the training config used by xreal_mobilenet (configs/xreal_mobilenet.yaml)
    parser.add_argument("--input-size", type=int, default=256, help="Model input size (square)")
    parser.add_argument("--face-margin", type=float, default=0.5, help="Margin around bbox/face detector box")
    parser.add_argument(
        "--use-detector-if-no-bbox",
        action="store_true",
        help="If COCO annotation has no bbox, try Haar face detector",
    )
    parser.add_argument("--draw-bbox", action="store_true", help="Draw the crop bbox used for inference")
    parser.add_argument("--no-draw-gt", action="store_true", help="Do not draw ground-truth keypoints")
    parser.add_argument("--no-draw-pred", action="store_true", help="Do not draw predicted keypoints")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = load_model(
        args.checkpoint,
        model_name=args.model_name,
        num_keypoints=args.num_keypoints,
        device=device,
        weights_only=True,
    )

    _safe_mkdir(args.out_dir)
    input_size = (int(args.input_size), int(args.input_size))
    detector = HaarFaceDetector()

    def process_one(
        *,
        img_path: str,
        out_path: str,
        bbox_xywh: Optional[Tuple[float, float, float, float]],
        gt_kpts: Optional[np.ndarray],
    ) -> bool:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[WARN] Missing/unreadable image: {img_path}")
            return False

        H_orig, W_orig = img_bgr.shape[:2]
        crop_xyxy = None
        if bbox_xywh is not None:
            crop_xyxy = _expand_bbox_xywh(bbox_xywh, margin=float(args.face_margin), img_w=W_orig, img_h=H_orig)
        elif args.use_detector_if_no_bbox:
            crop_xyxy = _detect_face_crop_xyxy(detector, img_bgr, margin=float(args.face_margin))

        if crop_xyxy is None:
            # Can't infer without a crop region; still save GT-only viz if present.
            viz = img_bgr.copy()
            if gt_kpts is not None and not args.no_draw_gt:
                _draw_keypoints(viz, gt_kpts, color_bgr=(0, 255, 0), radius=2)
            cv2.imwrite(out_path, viz)
            return True

        x1, y1, x2, y2 = crop_xyxy
        face_bgr = _crop_from_xyxy(img_bgr, x1, y1, x2, y2)

        pred_px, _ = _infer_landmarks_on_crop(
            model,
            device=device,
            face_bgr=face_bgr,
            num_keypoints=int(args.num_keypoints),
            input_size=input_size,
        )

        # map from resized crop coords -> original image coords
        W_face, H_face = input_size
        scale_x = (x2 - x1) / float(W_face)
        scale_y = (y2 - y1) / float(H_face)
        pred_orig = pred_px.copy()
        pred_orig[:, 0] = pred_orig[:, 0] * scale_x + x1
        pred_orig[:, 1] = pred_orig[:, 1] * scale_y + y1
        pred_kpts = np.concatenate([pred_orig, np.ones((pred_orig.shape[0], 1), dtype=np.float32)], axis=1)

        viz = img_bgr.copy()
        if not args.no_draw_gt and gt_kpts is not None:
            _draw_keypoints(viz, gt_kpts, color_bgr=(0, 255, 0), radius=2)  # GT green
        if not args.no_draw_pred:
            _draw_keypoints(viz, pred_kpts, color_bgr=(0, 0, 255), radius=2)  # pred red
        if args.draw_bbox:
            cv2.rectangle(viz, (x1, y1), (x2, y2), (255, 0, 0), 1)

        cv2.imwrite(out_path, viz)
        return True

    if args.image:
        # Single-image mode: run detector crop, save into out-dir
        img_path = args.image
        out_path = str(Path(args.out_dir) / (Path(img_path).stem + "_viz.png"))
        ok = process_one(img_path=img_path, out_path=out_path, bbox_xywh=None, gt_kpts=None)
        if ok:
            print(f"Saved {out_path}")
        return

    # COCO-json mode
    samples = _load_coco_samples(
        args.coco_json,
        images_root=args.images_root,
        sample_percent=float(args.sample_percent),
        seed=int(args.seed),
        max_samples=args.max_samples,
    )
    if not samples:
        print("No samples selected (sample-percent <= 0).")
        return

    num_ok = 0
    for i, s in enumerate(samples):
        stem = Path(s.image_path).stem
        ann_part = f"_ann{s.ann_id}" if s.ann_id is not None else ""
        out_path = str(Path(args.out_dir) / f"{stem}_img{s.image_id}{ann_part}.png")
        if process_one(img_path=s.image_path, out_path=out_path, bbox_xywh=s.bbox_xywh, gt_kpts=s.keypoints):
            num_ok += 1
        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(samples)}")

    print(f"Done. Wrote {num_ok}/{len(samples)} visualizations to {args.out_dir}")

if __name__ == "__main__":
    main()
