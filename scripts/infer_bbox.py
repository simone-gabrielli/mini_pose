"""BBox inference helper (TinyFace).

Follows the structure of scripts/infer.py, but targets bbox-only models.

Supports:
- COCO-style inference via --dataset/--coco-json
- Plain folder inference via --images-dir

Outputs:
- Per-image visualizations with predicted bbox (and GT bbox if present)
- predictions.json with one entry per processed image

Typical usage:
  python scripts/infer_bbox.py --config configs/face_mobilenet.yaml \
    --checkpoint work_dirs/face_mobilenet/best.pth \
    --dataset datasets/300W-xreal_air2 --split val \
    --out-dir outputs/infer_bbox

"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

try:
    import cv2
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "OpenCV (cv2) is required for infer_bbox visualization. "
        "Install it in your environment (e.g. pip install opencv-python)."
    ) from e

from pose.config import Config
from pose.detectors.face_detector import TinyFaceDetector


@dataclass(frozen=True)
class CocoBBoxSample:
    image_id: Optional[int]
    ann_id: Optional[int]
    image_path: str
    gt_bbox_xywh: Optional[Tuple[float, float, float, float]]


def _safe_mkdir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _iter_images_in_dir(images_dir: str) -> Iterable[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    root = Path(images_dir)
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield str(p)


def _find_coco_json(dataset_root: str, split: str) -> Optional[str]:
    ds = Path(dataset_root)
    split = str(split).lower().strip()

    candidates = [
        ds / "annotations" / f"{split}.json",
        ds / f"{split}.json",
        ds / "annotations" / "val.json" if split == "valid" else None,
    ]
    for c in candidates:
        if c is None:
            continue
        if c.exists() and c.is_file():
            return str(c)
    return None


def _load_coco_bbox_samples(
    coco_json_path: str,
    images_root: str,
    sample_percent: float,
    seed: int,
    max_samples: Optional[int],
) -> List[CocoBBoxSample]:
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images_by_id = {int(img["id"]): img for img in coco.get("images", []) if "id" in img}
    annotations = coco.get("annotations", [])
    if not annotations:
        raise ValueError(f"No annotations found in {coco_json_path}")

    coco_dir = Path(coco_json_path).parent
    samples: List[CocoBBoxSample] = []

    for ann in annotations:
        image_id = int(ann.get("image_id")) if ann.get("image_id") is not None else None
        image_info = images_by_id.get(image_id) if image_id is not None else None
        if not image_info:
            continue

        file_name = image_info.get("file_name")
        if not file_name:
            continue

        img_path = Path(file_name)
        if not img_path.is_absolute():
            root = Path(images_root) if images_root else coco_dir
            img_path = root / file_name

        bbox_xywh = None
        bbox = ann.get("bbox")
        if bbox is not None and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            bbox_xywh = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

        samples.append(
            CocoBBoxSample(
                image_id=image_id,
                ann_id=int(ann.get("id")) if ann.get("id") is not None else None,
                image_path=str(img_path),
                gt_bbox_xywh=bbox_xywh,
            )
        )

    if not samples:
        raise ValueError(f"No usable samples found in {coco_json_path}")

    pct = float(sample_percent)
    pct = min(100.0, max(0.0, pct))
    if pct <= 0:
        return []

    rng = random.Random(seed)
    n = max(1, int(round(len(samples) * (pct / 100.0))))
    n = min(n, len(samples))
    picked = rng.sample(samples, n)
    if max_samples is not None:
        picked = picked[: int(max_samples)]
    return picked


def _preprocess_rgb_to_tensor(
    rgb: np.ndarray,
    size_hw: Tuple[int, int],
    *,
    imagenet_norm: bool,
) -> torch.Tensor:
    """Resize to size_hw and convert to (1,3,H,W) float tensor.

    If imagenet_norm=True: apply ImageNet mean/std after scaling to [0,1].
    """

    H, W = int(size_hw[0]), int(size_hw[1])
    rgb_rs = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    img = rgb_rs.astype(np.float32) / 255.0

    if imagenet_norm:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img).unsqueeze(0).float()


def _extract_state_dict(ckpt: object) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("model", "model_state", "state_dict"):
            v = ckpt.get(key)
            if isinstance(v, dict) and v and all(isinstance(k, str) for k in v.keys()):
                return dict(v)  # type: ignore[return-value]

        # Fall back: maybe the dict itself is a state dict
        if ckpt and all(isinstance(k, str) for k in ckpt.keys()):
            return dict(ckpt)  # type: ignore[return-value]

    raise ValueError("Unsupported checkpoint format; expected a dict-like state.")


def _select_ema_shadow_state_dict(ckpt: object) -> Optional[Dict[str, torch.Tensor]]:
    if not isinstance(ckpt, dict):
        return None
    ema = ckpt.get("ema")
    if not isinstance(ema, dict):
        return None
    shadow = ema.get("shadow")
    if isinstance(shadow, dict) and shadow and all(isinstance(k, str) for k in shadow.keys()):
        return dict(shadow)  # type: ignore[return-value]
    return None


def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not sd:
        return sd
    if not any(k.startswith("module.") for k in sd.keys()):
        return sd
    return {k.replace("module.", "", 1): v for k, v in sd.items()}


def _load_tinyface_model(
    *,
    checkpoint_path: str,
    device: torch.device,
    backbone: str,
    width_mult: float,
    embed_dim: int,
    dropout: float,
    pretrained: bool,
    strict: bool,
    ema_mode: str,
    config_raw: Optional[dict],
) -> TinyFaceDetector:
    model = TinyFaceDetector(
        pretrained=bool(pretrained),
        backbone=str(backbone),
        width_mult=float(width_mult),
        embed_dim=int(embed_dim),
        dropout=float(dropout),
    )

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")

    ema_mode_n = str(ema_mode).lower().strip() if ema_mode is not None else "auto"
    use_ema = False
    if ema_mode_n == "on":
        use_ema = True
    elif ema_mode_n == "off":
        use_ema = False
    else:
        # auto: follow config train.ema.eval when available; else default to using non-EMA model weights.
        train_cfg = (config_raw or {}).get("train", {}) if isinstance(config_raw, dict) else {}
        ema_cfg = (train_cfg or {}).get("ema", {}) if isinstance(train_cfg, dict) else {}
        if isinstance(ema_cfg, dict) and "eval" in ema_cfg:
            use_ema = bool(ema_cfg.get("eval", True))
        else:
            use_ema = False

    state_dict = None
    if use_ema:
        state_dict = _select_ema_shadow_state_dict(ckpt)
        if state_dict is None:
            print("[WARN] --ema requested but checkpoint has no EMA shadow; falling back to ckpt['model']")

    if state_dict is None:
        state_dict = _extract_state_dict(ckpt)

    state_dict = _strip_module_prefix(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=bool(strict))
    if (missing or unexpected) and use_ema and ema_mode_n != "on":
        # Some checkpoints store an EMA shadow that doesn't include all buffers/params.
        # In auto mode, prefer the full non-EMA model weights in that case.
        print(
            f"[WARN] EMA shadow load incomplete (missing={len(missing)} unexpected={len(unexpected)}); "
            "falling back to ckpt['model'] weights. Use --ema on to force EMA."
        )
        state_dict_full = _strip_module_prefix(_extract_state_dict(ckpt))
        missing, unexpected = model.load_state_dict(state_dict_full, strict=bool(strict))

    if missing or unexpected:
        print(f"[WARN] TinyFace load missing={len(missing)} unexpected={len(unexpected)}")

    model.to(device)
    model.eval()
    return model


def _xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    x = float(x1)
    y = float(y1)
    w = float(max(0.0, x2 - x1))
    h = float(max(0.0, y2 - y1))
    return (x, y, w, h)


def _clip_xyxy(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[float, float, float, float]:
    x1 = float(np.clip(x1, 0.0, float(W - 1)))
    y1 = float(np.clip(y1, 0.0, float(H - 1)))
    x2 = float(np.clip(x2, 0.0, float(W - 1)))
    y2 = float(np.clip(y2, 0.0, float(H - 1)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _draw_bbox_xywh(img_bgr: np.ndarray, bbox_xywh: Tuple[float, float, float, float], color: Tuple[int, int, int], label: str) -> None:
    x, y, w, h = bbox_xywh
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(img_bgr, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--checkpoint", required=True, help="Trained bbox model checkpoint (e.g. best.pth)")
    p.add_argument(
        "--config",
        default="",
        help="Optional YAML config. If provided, TinyFace backbone/embed_dim/dropout/width_mult will default from it.",
    )
    p.add_argument("--device", default="cuda")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", help="Dataset folder (expects COCO JSON like annotations/val.json)")
    src.add_argument("--coco-json", help="Explicit COCO JSON path")
    src.add_argument("--images-dir", help="Plain folder of images (no GT)")

    p.add_argument(
        "--images-root",
        default="",
        help="Root folder for COCO images (joined with images[].file_name if relative). Default: dataset folder (or JSON folder).",
    )
    p.add_argument("--split", default="val", help="If using --dataset, which split JSON to pick (val/train/test)")

    p.add_argument("--out-dir", default="outputs/infer_bbox", help="Output folder")
    p.add_argument("--sample-percent", type=float, default=100.0, help="Percent of dataset entries to run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=None)

    p.add_argument("--input-size", type=int, default=256, help="Resize size for detector input (square)")
    p.add_argument(
        "--preprocess",
        choices=["imagenet", "none"],
        default="imagenet",
        help="Input normalization. imagenet matches training pipeline in pose/data/albu_aug.py.",
    )

    p.add_argument("--conf-th", type=float, default=0.3, help="Detection confidence threshold")
    p.add_argument("--draw-gt", action="store_true", help="Draw GT bbox (if present in COCO annotations)")
    p.add_argument("--save-failures", action="store_true", help="Also save images when confidence below threshold")

    # TinyFace model params (can be overridden even when --config is supplied)
    p.add_argument("--backbone", default="mobilenet_v3_small")
    p.add_argument("--width-mult", type=float, default=1.0)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained backbone weights")

    p.add_argument("--strict", action="store_true", help="Strict checkpoint loading")
    p.add_argument(
        "--ema",
        choices=["auto", "on", "off"],
        default="auto",
        help="EMA weights usage if checkpoint contains EMA state (auto follows config train.ema.eval when set).",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    config_raw = None

    # Defaults from config, if provided.
    if args.config:
        config_raw = Config.from_yaml(args.config).raw
        model_cfg = (config_raw or {}).get("model", {}) or {}
        # Only apply defaults if user didn't explicitly override (argparse already has defaults).
        args.backbone = str(model_cfg.get("backbone", args.backbone))
        args.width_mult = float(model_cfg.get("width_mult", args.width_mult))
        args.embed_dim = int(model_cfg.get("embed_dim", args.embed_dim))
        args.dropout = float(model_cfg.get("dropout", args.dropout))
        if "pretrained" in model_cfg and not args.no_pretrained:
            # keep CLI --no-pretrained as the strongest override
            args.no_pretrained = not bool(model_cfg.get("pretrained", True))

    model = _load_tinyface_model(
        checkpoint_path=str(args.checkpoint),
        device=device,
        backbone=str(args.backbone),
        width_mult=float(args.width_mult),
        embed_dim=int(args.embed_dim),
        dropout=float(args.dropout),
        pretrained=not bool(args.no_pretrained),
        strict=bool(args.strict),
        ema_mode=str(args.ema),
        config_raw=config_raw,
    )

    _safe_mkdir(args.out_dir)
    viz_dir = str(Path(args.out_dir) / "viz")
    _safe_mkdir(viz_dir)

    # Build samples
    coco_json_path = None
    images_root = str(args.images_root) if args.images_root else ""
    samples: List[CocoBBoxSample] = []

    if args.images_dir:
        for p in _iter_images_in_dir(args.images_dir):
            samples.append(CocoBBoxSample(image_id=None, ann_id=None, image_path=p, gt_bbox_xywh=None))
    else:
        if args.coco_json:
            coco_json_path = str(args.coco_json)
        else:
            coco_json_path = _find_coco_json(str(args.dataset), split=str(args.split))
            if coco_json_path is None:
                raise FileNotFoundError(
                    f"Could not find COCO JSON for split='{args.split}' under dataset '{args.dataset}'. "
                    "Try --coco-json explicitly."
                )

        if not images_root and args.dataset:
            images_root = str(args.dataset)

        samples = _load_coco_bbox_samples(
            coco_json_path=coco_json_path,
            images_root=images_root,
            sample_percent=float(args.sample_percent),
            seed=int(args.seed),
            max_samples=args.max_samples,
        )

    print(f"Running bbox inference on {len(samples)} images")

    preds: List[dict] = []
    input_size = int(args.input_size)
    imagenet_norm = str(args.preprocess).lower() == "imagenet"

    for i, s in enumerate(samples):
        img_bgr = cv2.imread(s.image_path)
        if img_bgr is None:
            print(f"[WARN] Could not read: {s.image_path}")
            continue

        H0, W0 = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x_t = _preprocess_rgb_to_tensor(rgb, (input_size, input_size), imagenet_norm=imagenet_norm)

        with torch.no_grad():
            out = model.predict(x_t.to(device), conf_th=float(args.conf_th))

        # model.predict returns per-image list with bbox in resized-tensor coords
        info = out[0]
        conf = float(info.get("conf", 0.0))
        bbox_resized = info.get("bbox")

        pred_bbox_xywh = None
        if bbox_resized is not None:
            x1r, y1r, x2r, y2r = [float(v) for v in bbox_resized]
            # Map back from resized square to original image
            sx = float(W0) / float(input_size)
            sy = float(H0) / float(input_size)
            x1 = x1r * sx
            y1 = y1r * sy
            x2 = x2r * sx
            y2 = y2r * sy
            x1, y1, x2, y2 = _clip_xyxy(x1, y1, x2, y2, W=W0, H=H0)
            pred_bbox_xywh = _xyxy_to_xywh(x1, y1, x2, y2)

        preds.append(
            {
                "image_id": s.image_id,
                "ann_id": s.ann_id,
                "image_path": s.image_path,
                "score": conf,
                "bbox_xywh": pred_bbox_xywh,
            }
        )

        should_save = pred_bbox_xywh is not None or bool(args.save_failures)
        if should_save:
            vis = img_bgr.copy()
            if bool(args.draw_gt) and s.gt_bbox_xywh is not None:
                _draw_bbox_xywh(vis, s.gt_bbox_xywh, (0, 255, 0), "gt")
            if pred_bbox_xywh is not None:
                _draw_bbox_xywh(vis, pred_bbox_xywh, (0, 0, 255), f"pred {conf:.2f}")
            else:
                cv2.putText(vis, f"pred < conf_th ({conf:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            stem = Path(s.image_path).stem
            suffix = ""
            if s.ann_id is not None:
                suffix = f"_ann{s.ann_id}"
            out_path = str(Path(viz_dir) / f"{stem}{suffix}.jpg")
            cv2.imwrite(out_path, vis)

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(samples)}")

    with open(str(Path(args.out_dir) / "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)

    print(f"Wrote visualizations to: {viz_dir}")
    print(f"Wrote predictions to: {Path(args.out_dir) / 'predictions.json'}")


if __name__ == "__main__":
    main()
