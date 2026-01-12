"""Run landmark inference on a video and save annotated video.

Features:
- Multiple detector backends: MTCNN (facenet-pytorch), dlib, composite Haar, haar
- Batch face crops per frame for faster GPU throughput
- Mixed precision with `--fp16` on CUDA
- Graceful shutdown: Ctrl+C saves the output before exiting

Examples:
python scripts/infer_video.py --checkpoint work_dirs/xreal_fan2d/best.pth --input-video input.mp4 --fp16
"""
import argparse
import os
import time
import signal
from time import perf_counter
from typing import Tuple

import cv2
import torch
import numpy as np

from pose.engine.inference import load_model, decode_heatmaps


def draw_landmarks(frame: np.ndarray, coords: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255)):
    for (xk, yk) in coords:
        cv2.circle(frame, (int(round(float(xk))), int(round(float(yk)))), 2, color, -1)


def _preprocess_rgb_to_tensor(img_rgb: np.ndarray, input_size: int) -> torch.Tensor:
    """Match training pipeline normalization (ImageNet mean/std) and resize."""
    img_rgb = cv2.resize(img_rgb, (int(input_size), int(input_size)), interpolation=cv2.INTER_LINEAR)
    img = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img).float()


def _extract_preds_last(model_out: object) -> torch.Tensor:
    """Match Trainer._visualize_model_outputs selection logic."""
    if isinstance(model_out, tuple):
        if len(model_out) >= 1 and isinstance(model_out[0], torch.Tensor):
            return model_out[0]
        if len(model_out) >= 2 and isinstance(model_out[1], (list, tuple)):
            return model_out[1][-1]
    if isinstance(model_out, torch.Tensor):
        return model_out
    raise TypeError(f"Unsupported model output type for heatmaps: {type(model_out)}")


def _try_extract_coords_batch(model_out: object) -> torch.Tensor | None:
    """Try to extract a (B, K, 2) coordinate tensor from a model output.

    Supports LOTR-style outputs:
      - (landmarks_norm, landmarks_pixel) where landmarks_pixel is (B, K, 2) or (B, K, 3)
    Returns:
      - coords_px: (B, K, 2) float tensor in crop pixel space, or None if not applicable.
    """
    if not isinstance(model_out, tuple):
        return None
    if len(model_out) < 2:
        return None
    coords = model_out[1]
    if not isinstance(coords, torch.Tensor):
        return None
    if coords.dim() != 3:
        return None
    if coords.size(-1) < 2:
        return None
    return coords[..., :2]


def _load_coco_bboxes_in_annotation_order(coco_json_path: str) -> list[tuple[float, float, float, float]]:
    import json

    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    anns = coco.get("annotations", [])
    out: list[tuple[float, float, float, float]] = []
    for a in anns:
        b = a.get("bbox", None)
        if b is None or len(b) != 4:
            out.append((0.0, 0.0, 0.0, 0.0))
            continue
        out.append((float(b[0]), float(b[1]), float(b[2]), float(b[3])))
    if not out:
        raise ValueError(f"No annotations/bboxes found in {coco_json_path}")
    return out


def build_detector(
    choice: str,
    device,
    *,
    yunet_model: str | None = None,
    yunet_score_thresh: float = 0.8,
    yunet_nms_thresh: float = 0.3,
    yunet_top_k: int = 5000,
):
    # MTCNN
    def make_mtcnn(device_str="cpu"):
        try:
            from facenet_pytorch import MTCNN

            mt = MTCNN(keep_all=True, device=device_str)

            class MTCNNDetector:
                def __init__(self, mt):
                    self.mt = mt

                def detect(self, img_bgr):
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    boxes, _ = self.mt.detect(img_rgb)
                    if boxes is None:
                        return []
                    out = []
                    for (x1, y1, x2, y2) in boxes:
                        w = int(max(1, x2 - x1))
                        h = int(max(1, y2 - y1))
                        out.append((int(x1), int(y1), w, h))
                    return out

            return MTCNNDetector(mt)
        except Exception:
            return None

    # dlib
    def make_dlib():
        try:
            import dlib

            detector = dlib.get_frontal_face_detector()

            class DlibDetector:
                def __init__(self, det):
                    self.det = det

                def detect(self, img_bgr):
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    dets = self.det(img_rgb, 1)
                    out = []
                    for d in dets:
                        x1 = d.left()
                        y1 = d.top()
                        x2 = d.right()
                        y2 = d.bottom()
                        out.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
                    return out

            return DlibDetector(detector)
        except Exception:
            return None

    # composite Haar (frontal + profile)
    class CompositeHaarDetector:
        def __init__(self):
            from pathlib import Path
            base = cv2.data.haarcascades
            self.frontal = cv2.CascadeClassifier(str(Path(base) / "haarcascade_frontalface_default.xml"))
            self.profile = cv2.CascadeClassifier(str(Path(base) / "haarcascade_profileface.xml"))

        def detect(self, img_bgr):
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            boxes = []
            if not self.frontal.empty():
                boxes += list(self.frontal.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3))
            if not self.profile.empty():
                boxes += list(self.profile.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3))
            out = []
            for (x, y, w, h) in boxes:
                out.append((int(x), int(y), int(w), int(h)))
            return out

    class HaarWrapper:
        def __init__(self):
            from pose.detectors.haar_face import HaarFaceDetector

            self.det = HaarFaceDetector()

        def detect(self, img_bgr):
            return self.det.detect(img_bgr)

    # OpenCV YuNet (FaceDetectorYN)
    def make_yunet(model_path: str, score_thresh: float = 0.8, nms_thresh: float = 0.3, top_k: int = 5000):
        if not model_path or not os.path.exists(model_path):
            return None

        creator = None
        # OpenCV exposes either FaceDetectorYN.create or FaceDetectorYN_create depending on version/build.
        if hasattr(cv2, "FaceDetectorYN") and hasattr(cv2.FaceDetectorYN, "create"):
            creator = cv2.FaceDetectorYN.create
        elif hasattr(cv2, "FaceDetectorYN_create"):
            creator = cv2.FaceDetectorYN_create

        if creator is None:
            return None

        try:
            # config is unused for YuNet ONNX
            det = creator(model_path, "", (320, 320), float(score_thresh), float(nms_thresh), int(top_k))

            class YuNetDetector:
                def __init__(self, det):
                    self.det = det

                def detect(self, img_bgr):
                    h, w = img_bgr.shape[:2]
                    # must match current image size
                    try:
                        self.det.setInputSize((int(w), int(h)))
                    except Exception:
                        # some builds require recreate; if setInputSize missing, fallback to no-op
                        pass

                    ok, faces = self.det.detect(img_bgr)
                    if faces is None or len(faces) == 0:
                        return []
                    out = []
                    for f in faces:
                        # YuNet output: [x, y, w, h, score, ...]
                        x, y, bw, bh = f[:4]
                        out.append((int(x), int(y), int(bw), int(bh)))
                    return out

            return YuNetDetector(det)
        except Exception:
            return None

    if choice == "mtcnn":
        det = make_mtcnn(device_str=("cuda" if device.type == "cuda" else "cpu"))
        if det is None:
            raise RuntimeError("MTCNN requested but not available (install facenet-pytorch)")
        return det
    if choice == "yunet":
        det = make_yunet(
            yunet_model or "",
            score_thresh=yunet_score_thresh,
            nms_thresh=yunet_nms_thresh,
            top_k=yunet_top_k,
        )
        if det is None:
            raise RuntimeError(
                "YuNet requested but not available. Ensure: (1) opencv-contrib-python installed, "
                "(2) your OpenCV build exposes FaceDetectorYN, (3) --yunet-model points to a valid .onnx file."
            )
        return det
    if choice == "dlib":
        det = make_dlib()
        if det is None:
            raise RuntimeError("dlib requested but not available")
        return det
    if choice == "composite":
        return CompositeHaarDetector()
    if choice == "haar":
        return HaarWrapper()

    # auto: prefer mtcnn -> dlib -> composite -> haar
    det = make_mtcnn(device_str=("cuda" if device.type == "cuda" else "cpu"))
    if det is not None:
        print("Using MTCNN detector")
        return det
    det = make_dlib()
    if det is not None:
        print("Using dlib detector")
        return det
    print("Using composite Haar detector")
    return CompositeHaarDetector()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--output-video", default=None)
    parser.add_argument("--model-name", default="fan_2d")
    parser.add_argument("--num-keypoints", type=int, default=68)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--face-margin", type=float, default=0.5, help="Margin around detected face box (training default)")

    # If your input video is built from a COCO dataset (frames are the dataset images
    # in the same order as COCO annotations), you can bypass detection entirely and
    # use the dataset bbox for cropping. This matches training/val much closer.
    parser.add_argument("--coco-json", default=None, help="Optional COCO json; if set, uses bboxes in annotation order per frame")
    parser.add_argument("--detector", choices=["auto", "mtcnn", "yunet", "dlib", "composite", "haar"], default="auto")
    parser.add_argument("--yunet-model", default=None, help="Path to YuNet ONNX model (required if --detector yunet)")
    parser.add_argument("--yunet-score", type=float, default=0.4, help="YuNet score threshold")
    parser.add_argument("--yunet-nms", type=float, default=0.3, help="YuNet NMS threshold")
    parser.add_argument("--yunet-topk", type=int, default=5000, help="YuNet top_k")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max-faces", type=int, default=4)
    parser.add_argument("--skip-frames", type=int, default=1)
    parser.add_argument("--draw-face-box", action="store_true")
    # Default to 0 because training bbox-crop doesn't add arbitrary padding.
    parser.add_argument("--box-pad", type=int, default=0, help="Pixel padding added around detected face box")
    parser.add_argument("--display", action="store_true", help="Show real-time preview window")
    parser.add_argument("--display-scale", type=float, default=0.5, help="Scale preview window (e.g. 0.5)")
    parser.add_argument(
        "--detect-every",
        type=int,
        default=1,
        help="Run face detector every N processed frames; reuse last boxes in-between",
    )
    parser.add_argument(
        "--detector-scale",
        type=float,
        default=1.0,
        help="Downscale factor for face detection only (e.g. 0.5). Boxes are scaled back.",
    )
    parser.add_argument(
        "--infer-every",
        type=int,
        default=1,
        help="Run keypoint model every N processed frames; reuse last keypoints in-between",
    )
    parser.add_argument("--profile", action="store_true", help="Log per-stage timing breakdown")
    parser.add_argument("--profile-every", type=int, default=60, help="Print timing every N processed frames")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Device requested: {args.device} | torch.cuda.is_available={torch.cuda.is_available()} | using: {device}")
    if device.type == "cuda":
        try:
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    if device.type == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    model = load_model(args.checkpoint, model_name=args.model_name, num_keypoints=args.num_keypoints, device=device)
    # Ensure coordinate-regression models (e.g., LOTR) scale outputs to the same
    # crop size we use in this script.
    if hasattr(model, "input_size"):
        try:
            model.input_size = (int(args.input_size), int(args.input_size))
        except Exception:
            pass
    model.eval()
    try:
        p0 = next(model.parameters())
        print(f"Model parameter device: {p0.device}")
    except Exception:
        pass

    input_path = args.input_video
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    out_path = args.output_video
    if out_path is None:
        base, _ = os.path.splitext(os.path.basename(input_path))
        out_path = f"{base}_landmarks.mp4"

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    detector = build_detector(
        args.detector if args.detector != "auto" else "auto",
        device,
        yunet_model=args.yunet_model,
        yunet_score_thresh=args.yunet_score,
        yunet_nms_thresh=args.yunet_nms,
        yunet_top_k=args.yunet_topk,
    )

    coco_bboxes = None
    if args.coco_json:
        coco_bboxes = _load_coco_bboxes_in_annotation_order(args.coco_json)
        print(f"Loaded {len(coco_bboxes)} COCO bboxes from {args.coco_json}. Using bbox-per-frame mode (annotation order).")
    frame_idx = 0
    t_start = time.time()
    processed = 0

    # reused buffers
    kpts_dummy = np.zeros((args.num_keypoints, 3), dtype=np.float32)

    # detector caching
    last_boxes = []
    last_detect_frame = -10**9

    # keypoint caching
    last_kpts = []  # list[np.ndarray] in full-frame coords
    last_kpts_frame = -10**9

    # profiling accumulators (seconds)
    prof = {
        "read": 0.0,
        "detect": 0.0,
        "preprocess": 0.0,
        "to_device": 0.0,
        "forward": 0.0,
        "to_cpu": 0.0,
        "decode_draw": 0.0,
        "write": 0.0,
        "display": 0.0,
        "total": 0.0,
    }
    prof_n = 0

    window_name = "mini_pose"
    paused = False

    stop_requested = {"stop": False}

    def _handle_sigint(sig, frame):
        print("\nReceived interrupt (SIGINT). Finishing current work and saving output...")
        stop_requested["stop"] = True

    try:
        signal.signal(signal.SIGINT, _handle_sigint)
    except Exception:
        pass

    try:
        while True:
            t_total0 = perf_counter() if args.profile else None

            t0 = perf_counter() if args.profile else None
            ret, frame_bgr = cap.read()
            if args.profile:
                prof["read"] += perf_counter() - t0

            if not ret:
                break

            if stop_requested["stop"]:
                break

            if (frame_idx % args.skip_frames) != 0:
                t0 = perf_counter() if args.profile else None
                writer.write(frame_bgr)
                if args.profile:
                    prof["write"] += perf_counter() - t0

                if args.display:
                    t0 = perf_counter() if args.profile else None
                    disp = frame_bgr
                    if args.display_scale and args.display_scale != 1.0:
                        disp = cv2.resize(
                            disp,
                            (int(W * args.display_scale), int(H * args.display_scale)),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    cv2.imshow(window_name, disp)
                    key = cv2.waitKey(1 if not paused else 30) & 0xFF
                    if key in (ord("q"), 27):
                        stop_requested["stop"] = True
                    elif key == ord("p"):
                        paused = not paused
                    if args.profile:
                        prof["display"] += perf_counter() - t0

                frame_idx += 1
                continue

            # Face detection OR COCO bbox-per-frame mode
            t0 = perf_counter() if args.profile else None
            do_detect = (processed - last_detect_frame) >= max(1, args.detect_every)
            boxes = last_boxes

            if coco_bboxes is not None:
                # Bbox-per-frame: assumes each processed frame corresponds to the next COCO annotation.
                idx = processed
                if idx >= len(coco_bboxes):
                    break
                x, y, bw, bh = coco_bboxes[idx]
                boxes = [(int(x), int(y), int(bw), int(bh))]
                last_boxes = boxes
                last_detect_frame = processed
                do_detect = True
            elif do_detect:
                det_frame = frame_bgr
                scale = float(args.detector_scale)
                if scale != 1.0 and scale > 0:
                    det_frame = cv2.resize(
                        frame_bgr,
                        (max(1, int(W * scale)), max(1, int(H * scale))),
                        interpolation=cv2.INTER_LINEAR,
                    )
                boxes = detector.detect(det_frame)
                if scale != 1.0 and scale > 0 and boxes:
                    inv = 1.0 / scale
                    scaled = []
                    for (x, y, w, h) in boxes:
                        scaled.append((int(x * inv), int(y * inv), int(w * inv), int(h * inv)))
                    boxes = scaled
                last_boxes = boxes
                last_detect_frame = processed

            if args.profile:
                prof["detect"] += perf_counter() - t0

            # Keypoint inference (optionally cached)
            do_infer = (processed - last_kpts_frame) >= max(1, args.infer_every)
            if do_detect:
                # if boxes just changed, ensure we refresh keypoints
                do_infer = True

            if boxes and do_infer:
                t0 = perf_counter() if args.profile else None
                faces = []
                boxes_clamped = []
                for box in boxes[: args.max_faces]:
                    x, y, w, h = box
                    margin = float(args.face_margin)
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    size = max(w, h) * (1.0 + margin)
                    pad = int(max(0, args.box_pad))
                    x1 = int(cx - size / 2.0) - pad
                    y1 = int(cy - size / 2.0) - pad
                    x2 = int(cx + size / 2.0) + pad
                    y2 = int(cy + size / 2.0) + pad

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(W - 1, x2)
                    y2 = min(H - 1, y2)

                    face_bgr = frame_bgr[y1:y2, x1:x2]
                    if face_bgr.size == 0:
                        continue

                    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                    img_t = _preprocess_rgb_to_tensor(face_rgb, input_size=args.input_size)
                    faces.append(img_t)
                    boxes_clamped.append((x1, y1, x2, y2))

                if args.profile:
                    prof["preprocess"] += perf_counter() - t0

                if faces:
                    t0 = perf_counter() if args.profile else None
                    batch = torch.stack(faces, dim=0).to(device)
                    if args.profile:
                        prof["to_device"] += perf_counter() - t0

                    with torch.inference_mode():
                        t0 = perf_counter() if args.profile else None
                        if args.profile and device.type == "cuda":
                            torch.cuda.synchronize()
                        if args.fp16 and device.type == "cuda":
                            from torch.cuda.amp import autocast

                            with autocast():
                                preds = model(batch)
                        else:
                            preds = model(batch)
                        if args.profile and device.type == "cuda":
                            torch.cuda.synchronize()
                        if args.profile:
                            prof["forward"] += perf_counter() - t0

                    # Decode either coordinate-regression outputs (LOTR) or heatmaps.
                    t0 = perf_counter() if args.profile else None
                    coords_batch = _try_extract_coords_batch(preds)

                    if coords_batch is not None:
                        # (B, K, 2) in resized crop pixel space
                        coords_batch = coords_batch.detach().float().cpu().numpy()
                        if args.profile:
                            prof["to_cpu"] += perf_counter() - t0

                        t1 = perf_counter() if args.profile else None
                        kpts_out = []
                        W_face = float(args.input_size)
                        H_face = float(args.input_size)
                        for i in range(coords_batch.shape[0]):
                            coords = coords_batch[i]
                            x1, y1, x2, y2 = boxes_clamped[i]
                            scale_x = (x2 - x1) / W_face
                            scale_y = (y2 - y1) / H_face
                            coords[:, 0] = coords[:, 0] * scale_x + x1
                            coords[:, 1] = coords[:, 1] * scale_y + y1
                            kpts_out.append(coords)

                        last_kpts = kpts_out
                        last_kpts_frame = processed
                        if args.profile:
                            prof["decode_draw"] += perf_counter() - t1

                    else:
                        # Heatmap-based models
                        preds_last = _extract_preds_last(preds)
                        preds_last = preds_last.detach().cpu()
                        if args.profile:
                            prof["to_cpu"] += perf_counter() - t0

                        t1 = perf_counter() if args.profile else None
                        kpts_out = []
                        for i in range(preds_last.shape[0]):
                            heatmaps = preds_last[i]
                            coords = decode_heatmaps(heatmaps)

                            H_hm, W_hm = heatmaps.shape[1:]
                            W_face = args.input_size
                            H_face = args.input_size
                            coords[:, 0] *= (W_face / W_hm)
                            coords[:, 1] *= (H_face / H_hm)

                            x1, y1, x2, y2 = boxes_clamped[i]
                            scale_x = (x2 - x1) / float(W_face)
                            scale_y = (y2 - y1) / float(H_face)
                            coords[:, 0] = coords[:, 0] * scale_x + x1
                            coords[:, 1] = coords[:, 1] * scale_y + y1

                            kpts_out.append(coords)

                        last_kpts = kpts_out
                        last_kpts_frame = processed

                        if args.profile:
                            prof["decode_draw"] += perf_counter() - t1

            # Draw cached keypoints + current boxes
            if boxes:
                t0 = perf_counter() if args.profile else None
                for i, box in enumerate(boxes[: args.max_faces]):
                    x, y, w, h = box
                    if args.draw_face_box:
                        pad = int(max(0, args.box_pad))
                        x1 = max(0, int(x) - pad)
                        y1 = max(0, int(y) - pad)
                        x2 = min(W - 1, int(x + w) + pad)
                        y2 = min(H - 1, int(y + h) + pad)
                        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    if i < len(last_kpts):
                        draw_landmarks(frame_bgr, last_kpts[i], color=(0, 0, 255))
                if args.profile:
                    prof["decode_draw"] += perf_counter() - t0

            t0 = perf_counter() if args.profile else None
            writer.write(frame_bgr)
            if args.profile:
                prof["write"] += perf_counter() - t0

            if args.display:
                t0 = perf_counter() if args.profile else None
                disp = frame_bgr
                if args.display_scale and args.display_scale != 1.0:
                    disp = cv2.resize(
                        disp,
                        (int(W * args.display_scale), int(H * args.display_scale)),
                        interpolation=cv2.INTER_LINEAR,
                    )
                cv2.imshow(window_name, disp)
                key = cv2.waitKey(1 if not paused else 30) & 0xFF
                if key in (ord("q"), 27):
                    stop_requested["stop"] = True
                elif key == ord("p"):
                    paused = not paused
                if args.profile:
                    prof["display"] += perf_counter() - t0

            frame_idx += 1
            processed += 1

            if args.profile:
                prof["total"] += perf_counter() - t_total0
                prof_n += 1
                if args.profile_every > 0 and (processed % args.profile_every) == 0 and prof_n > 0:
                    avg = {k: (v / prof_n) * 1000.0 for k, v in prof.items()}
                    fps_est = 1000.0 / max(1e-6, avg["total"])
                    det_mode = f"every={args.detect_every}, scale={args.detector_scale}" \
                        if args.detect_every != 1 or args.detector_scale != 1.0 else "default"
                    print(
                        "Timing (ms/frame): "
                        f"total={avg['total']:.1f}, read={avg['read']:.1f}, detect={avg['detect']:.1f}, "
                        f"prep={avg['preprocess']:.1f}, to_dev={avg['to_device']:.1f}, "
                        f"fwd={avg['forward']:.1f}, to_cpu={avg['to_cpu']:.1f}, decode/draw={avg['decode_draw']:.1f}, "
                        f"write={avg['write']:.1f}, display={avg['display']:.1f} | "
                        f"~{fps_est:.1f} fps | det({det_mode}) | infer(every={args.infer_every})"
                    )

            if processed % 100 == 0:
                elapsed = time.time() - t_start
                print(f"Processed {processed} frames, avg fps: {processed / max(1e-6, elapsed):.1f}")

    finally:
        cap.release()
        writer.release()
        if args.display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    total_t = time.time() - t_start
    print(f"Done. Wrote annotated video to: {out_path} (processed {processed} frames in {total_t:.1f}s)")


if __name__ == "__main__":
    main()
