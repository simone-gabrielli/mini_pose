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
from typing import Tuple

import cv2
import torch
import numpy as np

from pose.engine.inference import load_model, decode_heatmaps
from pose.data.transforms import BasicTransform


def draw_landmarks(frame: np.ndarray, coords: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255)):
    for (xk, yk) in coords:
        cv2.circle(frame, (int(round(float(xk))), int(round(float(yk)))), 2, color, -1)


def build_detector(choice: str, device):
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

    if choice == "mtcnn":
        det = make_mtcnn(device_str=("cuda" if device.type == "cuda" else "cpu"))
        if det is None:
            raise RuntimeError("MTCNN requested but not available (install facenet-pytorch)")
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
    parser.add_argument("--detector", choices=["auto", "mtcnn", "dlib", "composite", "haar"], default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max-faces", type=int, default=4)
    parser.add_argument("--skip-frames", type=int, default=1)
    parser.add_argument("--draw-face-box", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = load_model(args.checkpoint, model_name=args.model_name, num_keypoints=args.num_keypoints, device=device)
    model.eval()

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

    detector = build_detector(args.detector if args.detector != "auto" else "auto", device)
    tfm = BasicTransform(input_size=(args.input_size, args.input_size))

    frame_idx = 0
    t_start = time.time()
    processed = 0

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
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if stop_requested["stop"]:
                break

            if (frame_idx % args.skip_frames) != 0:
                writer.write(frame_bgr)
                frame_idx += 1
                continue

            boxes = detector.detect(frame_bgr)
            if boxes:
                faces = []
                boxes_clamped = []
                for box in boxes[: args.max_faces]:
                    x, y, w, h = box
                    margin = 0.25
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    size = max(w, h) * (1.0 + margin)
                    x1 = int(cx - size / 2.0)
                    y1 = int(cy - size / 2.0)
                    x2 = int(cx + size / 2.0)
                    y2 = int(cy + size / 2.0)

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(W - 1, x2)
                    y2 = min(H - 1, y2)

                    face_bgr = frame_bgr[y1:y2, x1:x2]
                    if face_bgr.size == 0:
                        continue

                    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                    kpts_dummy = np.zeros((args.num_keypoints, 3), dtype=np.float32)
                    img_t, _ = tfm(face_rgb, keypoints=kpts_dummy)
                    faces.append(img_t)
                    boxes_clamped.append((x1, y1, x2, y2))

                if faces:
                    batch = torch.stack(faces, dim=0).to(device)
                    with torch.no_grad():
                        if args.fp16 and device.type == "cuda":
                            from torch.cuda.amp import autocast

                            with autocast():
                                preds = model(batch)
                        else:
                            preds = model(batch)

                    if isinstance(preds, (tuple, list)):
                        preds_last = preds[0]
                    elif isinstance(preds, dict) and "preds_last" in preds:
                        preds_last = preds["preds_last"]
                    else:
                        preds_last = preds

                    preds_last = preds_last.cpu()
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

                        draw_landmarks(frame_bgr, coords, color=(0, 0, 255))
                        if args.draw_face_box:
                            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 1)

            writer.write(frame_bgr)
            frame_idx += 1
            processed += 1

            if processed % 100 == 0:
                elapsed = time.time() - t_start
                print(f"Processed {processed} frames, avg fps: {processed / max(1e-6, elapsed):.1f}")

    finally:
        cap.release()
        writer.release()

    total_t = time.time() - t_start
    print(f"Done. Wrote annotated video to: {out_path} (processed {processed} frames in {total_t:.1f}s)")


if __name__ == "__main__":
    main()
