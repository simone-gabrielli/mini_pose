"""Run landmark inference on a video and save annotated video.

Features:
- Face detection backends: OpenCV YuNet, TinyFace (single-face)
- Batch face crops per frame for faster GPU throughput
- Mixed precision with `--fp16` on CUDA
- Graceful shutdown: Ctrl+C saves the output before exiting

Examples:
python scripts/infer_video.py --checkpoint work_dirs/xreal_lotr_light/best.pth --input-video input.mp4 --fp16
"""
import argparse
import math
import os
import time
import signal
from time import perf_counter
from typing import Tuple
import xml.etree.ElementTree as ET

import cv2
import torch
import numpy as np

from pose.engine.inference import load_model as load_heatmap_model, decode_heatmaps
from pose.config import Config
from pose.registry import MODEL_REGISTRY

# Ensure model modules are imported so they register themselves in MODEL_REGISTRY
import pose.models  # noqa: F401


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--output-video", default=None)
    parser.add_argument("--model-name", default="lotr_light")
    parser.add_argument("--num-keypoints", type=int, default=68)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument(
        "--config",
        default="",
        help="Optional YAML config (recommended for LOTR). If set, preprocessing size defaults to cfg.data.input_size.",
    )
    parser.add_argument(
        "--override-input-size",
        action="store_true",
        help="In --config mode, force using --input-size instead of cfg.data.input_size for preprocessing.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict checkpoint loading (only applies when --config is set).",
    )
    parser.add_argument(
        "--ema",
        choices=["auto", "on", "off"],
        default="auto",
        help="EMA weights usage when checkpoint contains EMA state (auto follows config train.ema.eval).",
    )
    parser.add_argument(
        "--face-margin",
        type=float,
        default=50.0,
        help="Extra margin around detected face box, as a percentage (e.g. 50 -> +50%%).",
    )

    # Detection backend for face/glasses boxes.
    parser.add_argument("--detector", choices=["auto", "yunet", "tinyface", "yolov8"], default="auto")
    parser.add_argument(
        "--detector-score",
        type=float,
        default=0.4,
        help="Score threshold for detectors that output a score/confidence.",
    )
    parser.add_argument(
        "--yunet-model",
        default=None,
        help="Path to YuNet ONNX model (used when --detector is yunet/auto).",
    )

    parser.add_argument(
        "--tinyface-checkpoint",
        default=None,
        help="Path to TinyFace detector checkpoint (.pth). If omitted, uses ImageNet-pretrained backbone weights.",
    )
    parser.add_argument(
        "--tinyface-input",
        type=int,
        default=192,
        help="TinyFace detector square input size (defaults to 192 to match training configs)",
    )

    parser.add_argument("--yolo-weights", default=None, help="YOLOv8 weights (.pt). Required for --detector yolov8")
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="YOLOv8 inference image size")
    parser.add_argument("--yolo-iou", type=float, default=0.45, help="YOLOv8 NMS IoU threshold")
    parser.add_argument(
        "--yolo-classes",
        default=None,
        help="Optional YOLO class filter (comma-separated ids/names), e.g. '0' or 'glasses'.",
    )

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max-faces", type=int, default=4)
    parser.add_argument("--skip-frames", type=int, default=1)
    parser.add_argument("--draw-face-box", action="store_true")
    parser.add_argument("--draw-crop-box", action="store_true", help="Draw the square crop box actually used for inference")
    parser.add_argument(
        "--draw-detected-landmarks",
        dest="draw_detected_landmarks",
        action="store_true",
        help="Draw detected/predicted 2D landmarks",
    )
    parser.set_defaults(draw_detected_landmarks=False)
    # Default to 0 because training bbox-crop doesn't add arbitrary padding.
    parser.add_argument("--box-pad", type=int, default=0, help="Pixel padding added around detected face box")
    parser.add_argument("--display", action="store_true", help="Show real-time preview window")
    parser.add_argument(
        "--show-input-crops",
        action="store_true",
        help="Show a debug window with the per-face 256x256 model inputs and landmarks overlaid",
    )
    parser.add_argument("--debug-crops-cols", type=int, default=4, help="Columns for the debug crops tiling window")
    parser.add_argument("--model-3d-xml", default=None, help="Optional 3D landmark template XML; enables PnP + projection overlay")
    parser.add_argument("--draw-pnp", action="store_true", help="Draw reprojected 3D landmarks (from solvePnP) over the frame")
    parser.add_argument("--draw-axes", action="store_true", help="Draw pose axes (requires --model-3d-xml)")
    parser.add_argument("--axes-len", type=float, default=0.05, help="Axes length in 3D units (same units as XML)")
    parser.add_argument("--pnp-method", choices=["iterative", "epnp", "p3p", "ap3p"], default="iterative")
    parser.add_argument(
        "--pnp-max-reproj-err",
        type=float,
        default=8.0,
        help="Discard PnP result if mean reprojection error (pixels) is above this threshold",
    )
    parser.add_argument("--cam-fx", type=float, default=None, help="Camera fx in pixels (default: derived)")
    parser.add_argument("--cam-fy", type=float, default=None, help="Camera fy in pixels (default: derived)")
    parser.add_argument("--cam-cx", type=float, default=None, help="Camera cx in pixels (default: W/2)")
    parser.add_argument("--cam-cy", type=float, default=None, help="Camera cy in pixels (default: H/2)")
    parser.add_argument("--cam-fov-deg", type=float, default=None, help="Optional horizontal FOV in degrees (used to estimate fx/fy)")
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
    return parser


def draw_landmarks(frame: np.ndarray, coords: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255)):
    """Draw 2D landmark points (x,y) onto a BGR frame."""
    for (xk, yk) in coords:
        cv2.circle(frame, (int(round(float(xk))), int(round(float(yk)))), 2, color, -1)


def _load_3d_landmarks_xml(xml_path: str) -> np.ndarray:
    """Load 3D landmarks from the provided XML format.

    Expected structure:
      <model><landmarks><landmark id="0" x="..." y="..." z="..." /> ...

    Returns:
      (N,3) float32 array ordered by increasing landmark id.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lms = root.find("landmarks")
    if lms is None:
        # allow nesting: <model><landmarks>...
        lms = root.find("./model/landmarks")
    if lms is None:
        raise ValueError(f"No <landmarks> node found in XML: {xml_path}")

    items: list[tuple[int, float, float, float]] = []
    for el in lms.findall("landmark"):
        try:
            idx = int(el.attrib["id"])
            x = float(el.attrib["x"])
            y = float(el.attrib["y"])
            z = float(el.attrib["z"])
        except Exception:
            continue
        items.append((idx, x, y, z))

    if not items:
        raise ValueError(f"No <landmark .../> entries found in XML: {xml_path}")

    items.sort(key=lambda t: t[0])
    pts = np.array([[x, y, z] for (_, x, y, z) in items], dtype=np.float32)
    return pts


def _camera_matrix_from_args(W: int, H: int, args) -> np.ndarray:
    """Build an OpenCV pinhole camera matrix K from CLI args.

    If fx/fy aren't provided, use a simple heuristic (max(W,H)). Optionally
    derive focal length from a horizontal FOV.
    """
    cx = float(args.cam_cx) if args.cam_cx is not None else (W / 2.0)
    cy = float(args.cam_cy) if args.cam_cy is not None else (H / 2.0)

    fx = args.cam_fx
    fy = args.cam_fy
    if args.cam_fov_deg is not None and (fx is None or fy is None):
        fov = float(args.cam_fov_deg)
        # assume horizontal FOV
        fx_est = (W / 2.0) / max(1e-9, math.tan(math.radians(fov) / 2.0))
        fx = fx if fx is not None else fx_est
        fy = fy if fy is not None else fx_est

    if fx is None:
        fx = float(max(W, H))
    if fy is None:
        fy = float(max(W, H))

    K = np.array(
        [
            [float(fx), 0.0, float(cx)],
            [0.0, float(fy), float(cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return K


def _solve_pnp(
    obj_pts_3d: np.ndarray,
    img_pts_2d: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    if obj_pts_3d is None or img_pts_2d is None:
        return None
    n = min(int(obj_pts_3d.shape[0]), int(img_pts_2d.shape[0]))
    if n < 4:
        return None

    obj = obj_pts_3d[:n].astype(np.float64).reshape(-1, 1, 3)
    img = img_pts_2d[:n].astype(np.float64).reshape(-1, 1, 2)

    flag_map = {
        "iterative": cv2.SOLVEPNP_ITERATIVE,
        "epnp": cv2.SOLVEPNP_EPNP,
        "p3p": cv2.SOLVEPNP_P3P,
        "ap3p": cv2.SOLVEPNP_AP3P,
    }
    flags = flag_map.get(str(method).lower(), cv2.SOLVEPNP_ITERATIVE)
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=flags)
    except Exception:
        return None
    if not ok:
        return None
    return rvec, tvec


def _project_points(obj_pts_3d: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    obj = obj_pts_3d.astype(np.float64).reshape(-1, 1, 3)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return proj.reshape(-1, 2).astype(np.float32)


def _mean_reprojection_error(img_pts_2d: np.ndarray, proj_pts_2d: np.ndarray) -> float:
    n = min(int(img_pts_2d.shape[0]), int(proj_pts_2d.shape[0]))
    if n <= 0:
        return float("inf")
    diff = proj_pts_2d[:n].astype(np.float32) - img_pts_2d[:n].astype(np.float32)
    err = np.linalg.norm(diff, axis=1)
    return float(np.mean(err))


def _run_pnp_batch(
    obj_pts_3d: np.ndarray,
    kpts_out: list[np.ndarray],
    K_cam: np.ndarray,
    dist: np.ndarray,
    *,
    method: str,
    max_reproj_err: float,
) -> list[dict]:
    pnp_out = []
    for i in range(len(kpts_out)):
        sol = _solve_pnp(obj_pts_3d, kpts_out[i], K_cam, dist, method=method)
        if sol is None:
            pnp_out.append({"rvec": None, "tvec": None, "proj": None, "err": None})
            continue
        rvec, tvec = sol
        proj = _project_points(obj_pts_3d, rvec, tvec, K_cam, dist)
        err = _mean_reprojection_error(kpts_out[i], proj)
        if err > float(max_reproj_err):
            pnp_out.append({"rvec": None, "tvec": None, "proj": None, "err": float(err)})
        else:
            pnp_out.append({"rvec": rvec, "tvec": tvec, "proj": proj, "err": float(err)})
    return pnp_out


def _tile_same_size(images: list[np.ndarray], cols: int = 4, pad: int = 6, bg: Tuple[int, int, int] = (16, 16, 16)) -> np.ndarray:
    """Tile images (all same HxW) into a grid."""
    if not images:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    cols = max(1, int(cols))
    pad = max(0, int(pad))
    h, w = images[0].shape[:2]
    rows = int(np.ceil(len(images) / float(cols)))

    out_h = rows * h + (rows + 1) * pad
    out_w = cols * w + (cols + 1) * pad
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[:, :] = np.array(bg, dtype=np.uint8)

    for idx, im in enumerate(images):
        r = idx // cols
        c = idx % cols
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        canvas[y : y + h, x : x + w] = im
    return canvas


def _preprocess_rgb_to_tensor(img_rgb: np.ndarray, input_size: Tuple[int, int]) -> torch.Tensor:
    """Match training pipeline normalization (ImageNet mean/std) and resize."""
    w, h = int(input_size[0]), int(input_size[1])
    img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    img = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img).float()


def _load_model_from_config(
    *,
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
    fallback_num_keypoints: int,
    weights_only: bool,
    strict: bool,
    ema_mode: str = "auto",
):
    """Load a model using the same generic config wiring as Trainer.

    Also handles LOTR positional encoding size mismatches by rebuilding the model
    with an inferred input_size (from checkpoint pos_encoding.pe) if needed.

    Returns: (model, cfg_raw, num_keypoints)
    """

    cfg = Config.from_yaml(config_path).raw
    model_cfg = cfg.get("model", {})
    if not model_cfg or "name" not in model_cfg:
        raise ValueError(f"Config {config_path} missing model.name")

    model_name = str(model_cfg["name"])
    try:
        ModelCls = MODEL_REGISTRY[model_name]
    except KeyError:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(f"Model '{model_name}' not found in MODEL_REGISTRY. Available models: {available}")

    data_cfg = cfg.get("data", {})
    num_keypoints = int(data_cfg.get("num_keypoints", fallback_num_keypoints))

    model_kwargs = {k: v for k, v in model_cfg.items() if k != "name"}
    model_kwargs["num_keypoints"] = num_keypoints
    if "input_size" in data_cfg:
        model_kwargs["input_size"] = tuple(data_cfg["input_size"])

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=weights_only)
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

    def _infer_input_size_from_pe() -> Tuple[int, int] | None:
        pe = state_dict.get("pos_encoding.pe")
        if not isinstance(pe, torch.Tensor) or pe.dim() != 3:
            return None
        pe_h, pe_w = int(pe.shape[0]), int(pe.shape[1])

        use_upsampling = bool(model_kwargs.get("use_upsampling", True))
        up_layers = int(model_kwargs.get("upsampling_layers", 2)) if use_upsampling else 0
        scale = 2 ** max(0, up_layers)
        input_h = int(round(pe_h * 32 / float(scale)))
        input_w = int(round(pe_w * 32 / float(scale)))
        if input_h <= 0 or input_w <= 0:
            return None
        return (input_h, input_w)

    def _build_model(overrides: dict | None = None):
        kw = dict(model_kwargs)
        if overrides:
            kw.update(overrides)
        return ModelCls(**kw)

    model = _build_model()
    try:
        model.load_state_dict(state_dict, strict=bool(strict))
    except RuntimeError as e:
        inferred = _infer_input_size_from_pe()
        if inferred is not None:
            try:
                model = _build_model({"input_size": inferred})
                model.load_state_dict(state_dict, strict=bool(strict))
            except RuntimeError:
                # Last resort: drop positional encoding buffer
                model = _build_model({"input_size": inferred})
                sd = dict(state_dict)
                sd.pop("pos_encoding.pe", None)
                model.load_state_dict(sd, strict=False)
        else:
            sd = dict(state_dict)
            if "pos_encoding.pe" in sd:
                sd.pop("pos_encoding.pe", None)
                model.load_state_dict(sd, strict=False)
            else:
                raise e

    # Optional: apply EMA weights (common for 'best.pth' checkpoints)
    # Trainer saves EMA state under ckpt['ema']['shadow'] and (optionally) evaluates using EMA.
    ema_shadow = None
    if isinstance(ckpt, dict):
        ema_state = ckpt.get("ema")
        if isinstance(ema_state, dict):
            shadow = ema_state.get("shadow")
            if isinstance(shadow, dict):
                ema_shadow = shadow

    def _should_use_ema() -> bool:
        mode = str(ema_mode).lower().strip() if ema_mode is not None else "auto"
        if mode == "off":
            return False
        if mode == "on":
            return True
        # auto: follow config if present, else use EMA when available
        ema_cfg = (cfg.get("train", {}) or {}).get("ema", {}) or {}
        return bool(ema_cfg.get("eval", True))

    if ema_shadow is not None and _should_use_ema():
        # `ema_shadow` is a dict of parameter tensors only (Trainer.ModelEMA).
        # Apply it the same way as Trainer does: copy matching parameters by name.
        applied = 0
        try:
            for name, p in model.named_parameters():
                if name not in ema_shadow:
                    continue
                if not torch.is_floating_point(p.data):
                    continue
                src = ema_shadow[name]
                if not isinstance(src, torch.Tensor):
                    continue
                if src.device != p.device:
                    src = src.to(device=p.device)
                p.data.copy_(src)
                applied += 1
        except Exception as e:
            print(f"[WARN] Failed to apply EMA shadow weights: {e}")
        else:
            if applied > 0:
                print(f"Using EMA weights for inference (applied {applied} params)")

    model.to(device)
    model.eval()
    return model, cfg, num_keypoints


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


def _try_extract_norm_coords_batch(model_out: object) -> torch.Tensor | None:
    """Try to extract NORMALIZED (B, K, 2) coordinate tensor from a LOTR-style model output.

    Supports LOTR-style outputs:
      - (landmarks_norm, landmarks_pixel) where landmarks_norm is (B, K, 2)

    Using the normalized coordinates (element [0]) avoids any dependency on
    model.input_size and is consistent with how the Trainer computes the loss:
        preds_2d = landmarks_norm * [W_in, H_in]

    The caller must scale by the actual preprocessing size to get pixel coords.

    Returns:
      - coords_norm: (B, K, 2) float tensor in [0,1] normalized space, or None.
    """
    if not isinstance(model_out, tuple):
        return None
    if len(model_out) < 2:
        return None
    coords_norm = model_out[0]
    if not isinstance(coords_norm, torch.Tensor):
        return None
    if coords_norm.dim() != 3:
        return None
    if coords_norm.size(-1) < 2:
        return None
    return coords_norm[..., :2]


def build_detector(
    choice: str,
    device,
    *,
    score_thresh: float = 0.4,
    yunet_model: str | None = None,
    tinyface_checkpoint: str | None = None,
    tinyface_input: int = 192,
    yolo_weights: str | None = None,
    yolo_imgsz: int = 640,
    yolo_iou: float = 0.45,
    yolo_classes: str | None = None,
    fp16: bool = False,
    ema_mode: str = "auto",
):
    def _default_yunet_path() -> str | None:
        # shipped in this repo under scripts/video_test/yunet_detector/
        try:
            base = os.path.dirname(os.path.abspath(__file__))
            p = os.path.join(base, "video_test", "yunet_detector", "face_detection_yunet_2023mar.onnx")
            return p if os.path.exists(p) else None
        except Exception:
            return None

    # OpenCV YuNet (FaceDetectorYN)
    def make_yunet(model_path: str, score_thresh: float = 0.4):
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
            try:
                det = creator(model_path, "", (320, 320), float(score_thresh), 0.3, 5000)
            except TypeError:
                det = creator(model_path, "", (320, 320), float(score_thresh))

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
                        score = float(f[4]) if len(f) > 4 else 1.0
                        if score < float(score_thresh):
                            continue
                        out.append((int(x), int(y), int(bw), int(bh), float(score)))
                    return out

            return YuNetDetector(det)
        except Exception:
            return None

    # TinyFace single-box regressor
    def make_tinyface(
        checkpoint_path: str | None,
        *,
        input_size: int = 256,
        conf_th: float = 0.4,
        fp16: bool = False,
        ema_mode: str = "auto",
    ):
        try:
            from pose.detectors.face_detector import TinyFaceDetector
        except Exception:
            return None

        def _get_state_dict(ckpt_obj: object) -> dict:
            if isinstance(ckpt_obj, dict):
                for k in ("model", "model_state", "state_dict", "weights"):
                    sd = ckpt_obj.get(k)
                    if isinstance(sd, dict):
                        return sd
                # maybe the dict itself is already a state_dict
                if ckpt_obj and all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
                    return ckpt_obj
            if isinstance(ckpt_obj, dict) and "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
                return ckpt_obj["model"]
            raise ValueError("Unsupported checkpoint format")

        def _strip_module_prefix(sd: dict) -> dict:
            out = {}
            for k, v in sd.items():
                nk = k
                if nk.startswith("module."):
                    nk = nk[len("module.") :]
                out[nk] = v
            return out

        # Instantiate with a matching backbone/head. We infer these from the
        # checkpoint (when provided) so we don't need a long list of CLI args.
        inferred_backbone = "mobilenet_v2"
        inferred_embed_dim: int | None = None
        inferred_deep_head: bool | None = None
        sd = None
        ema_shadow = None
        if checkpoint_path:
            try:
                ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            except TypeError:
                ckpt = torch.load(checkpoint_path, map_location="cpu")
            sd = _strip_module_prefix(_get_state_dict(ckpt))

            # Many checkpoints are trained/evaluated with EMA. The 'best.pth'
            # is usually selected using EMA weights, while ckpt['model'] holds
            # the raw (non-EMA) weights. Use EMA shadow when available.
            if isinstance(ckpt, dict):
                ema = ckpt.get("ema")
                if isinstance(ema, dict):
                    shadow = ema.get("shadow")
                    if isinstance(shadow, dict) and shadow:
                        ema_shadow = _strip_module_prefix(shadow)

            try:
                import re

                keys = list(sd.keys())

                # Infer head type: deep_head checkpoints have fc.9.* (final Linear at index 9).
                inferred_deep_head = any(k.startswith("fc.9.") for k in keys) or any(k.startswith("fc.6.") for k in keys)

                # Infer embed dim from the first Linear in the head.
                fc1 = sd.get("fc.1.weight")
                if isinstance(fc1, torch.Tensor) and fc1.dim() == 2:
                    inferred_embed_dim = int(fc1.shape[0])

                # Infer backbone from feature key structure.
                # - EfficientNet-B0 (torchvision) has nested blocks: features.<stage>.<block>.block.*
                # - MobileNetV3 has: features.<idx>.block.*
                # - MobileNetV2 has: features.<idx>.conv.*
                if any(re.match(r"^features\.\d+\.\d+\.block\.", k) for k in keys):
                    inferred_backbone = "efficientnet_b0"
                elif any(re.match(r"^features\.\d+\.block\.", k) for k in keys):
                    inferred_backbone = "mobilenet_v3_small"
                else:
                    inferred_backbone = "mobilenet_v2"
            except Exception:
                pass

        model_kwargs = {
            "pretrained": bool(not checkpoint_path),
            "backbone": str(inferred_backbone),
        }
        if inferred_embed_dim is not None:
            model_kwargs["embed_dim"] = int(inferred_embed_dim)
        if inferred_deep_head is not None:
            model_kwargs["deep_head"] = bool(inferred_deep_head)

        model = TinyFaceDetector(**model_kwargs)

        if checkpoint_path and sd is not None:
            try:
                model.load_state_dict(sd, strict=True)
            except RuntimeError as e:
                raise RuntimeError(
                    "TinyFace checkpoint is not compatible with the TinyFaceDetector architecture. "
                    f"Inferred backbone='{inferred_backbone}'. "
                    "If this checkpoint is from a different model, use a proper TinyFace detector checkpoint "
                    "or switch to '--detector yunet'.\n\nOriginal error:\n" + str(e)
                )

        model.to(device)
        model.eval()

        if fp16 and device.type == "cuda":
            try:
                model = model.half()
            except Exception:
                pass

        # Optionally apply EMA shadow weights.
        try:
            use_ema = str(ema_mode).lower().strip() in {"auto", "on", "true", "1", "yes"}
        except Exception:
            use_ema = True

        if use_ema and isinstance(ema_shadow, dict) and ema_shadow:
            applied = 0
            for name, p in model.named_parameters():
                if name not in ema_shadow:
                    continue
                t = ema_shadow.get(name)
                if not isinstance(t, torch.Tensor):
                    continue
                if not torch.is_floating_point(p.data):
                    continue
                if t.device != p.device:
                    t = t.to(device=p.device)
                if t.dtype != p.dtype:
                    t = t.to(dtype=p.dtype)
                try:
                    p.data.copy_(t)
                    applied += 1
                except Exception:
                    continue
            if applied > 0:
                print(f"Using TinyFace EMA weights (applied {applied} params)")

        class TinyFaceWrapper:
            def __init__(self, model, input_size: int, conf_th: float):
                self.model = model
                self.input_size = int(input_size)
                self.conf_th = float(conf_th)

            def detect(self, img_bgr):
                H0, W0 = img_bgr.shape[:2]
                if H0 <= 0 or W0 <= 0:
                    return []

                # Match the TinyFace training pipeline (AlbumentationsKeypointPipeline):
                # resize to input_size and ImageNet mean/std normalization.
                rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                s = max(32, int(self.input_size))
                x_t = _preprocess_rgb_to_tensor(rgb, input_size=(s, s)).unsqueeze(0)
                if fp16 and device.type == "cuda":
                    x_t = x_t.half()
                else:
                    x_t = x_t.float()

                with torch.inference_mode():
                    res = self.model.predict(x_t, conf_th=self.conf_th)

                if not res or res[0].get("bbox") is None:
                    return []

                conf = float(res[0].get("conf", 1.0))
                x1, y1, x2, y2 = res[0]["bbox"]

                # Map from square resized coords back to original frame coords.
                sx = float(W0) / float(s)
                sy = float(H0) / float(s)
                x1 = int(round(float(x1) * sx))
                y1 = int(round(float(y1) * sy))
                x2 = int(round(float(x2) * sx))
                y2 = int(round(float(y2) * sy))

                # clamp
                x1 = max(0, min(W0 - 1, x1))
                y1 = max(0, min(H0 - 1, y1))
                x2 = max(0, min(W0 - 1, x2))
                y2 = max(0, min(H0 - 1, y2))

                if x2 <= x1 or y2 <= y1:
                    return []
                return [(int(x1), int(y1), int(x2 - x1), int(y2 - y1), float(conf))]

        return TinyFaceWrapper(model, input_size=input_size, conf_th=conf_th)

    def make_yolov8(
        weights_path: str | None,
        *,
        imgsz: int = 640,
        iou: float = 0.45,
        conf: float = 0.4,
        classes: str | None = None,
    ):
        if not weights_path:
            return None
        try:
            from pose.detectors.yolov8_detector import YOLOv8Detector
        except Exception:
            return None

        cls_list = None
        if classes:
            cls_list = [c.strip() for c in str(classes).split(",") if c.strip()]

        try:
            det = YOLOv8Detector(
                weights_path=str(weights_path),
                device=str(device.type if hasattr(device, "type") else device),
                imgsz=int(imgsz),
                conf=float(conf),
                iou=float(iou),
                classes=cls_list,
            )
            return det
        except Exception as e:
            print(f"[WARN] Failed to init YOLOv8 detector: {e}")
            return None

    if choice == "yunet":
        model_path = yunet_model or _default_yunet_path() or ""
        det = make_yunet(model_path, score_thresh=float(score_thresh))
        if det is None:
            raise RuntimeError(
                "YuNet requested but not available. Ensure: (1) opencv-contrib-python installed, "
                "(2) your OpenCV build exposes FaceDetectorYN, (3) --yunet-model points to a valid .onnx file."
            )
        return det
    if choice == "tinyface":
        det = make_tinyface(
            tinyface_checkpoint,
            input_size=tinyface_input,
            conf_th=float(score_thresh),
            fp16=fp16,
            ema_mode=str(ema_mode),
        )
        if det is None:
            raise RuntimeError("TinyFace requested but not available (check pose.detectors.face_detector and dependencies)")
        return det

    if choice == "yolov8":
        det = make_yolov8(
            yolo_weights,
            imgsz=int(yolo_imgsz),
            iou=float(yolo_iou),
            conf=float(score_thresh),
            classes=yolo_classes,
        )
        if det is None:
            raise RuntimeError(
                "YOLOv8 requested but not available. Ensure ultralytics is installed (pip install ultralytics) "
                "and --yolo-weights points to a valid .pt file."
            )
        return det

    # auto: prefer YOLOv8 (if weights provided) -> YuNet (if model available) -> TinyFace
    if yolo_weights:
        det = make_yolov8(
            yolo_weights,
            imgsz=int(yolo_imgsz),
            iou=float(yolo_iou),
            conf=float(score_thresh),
            classes=yolo_classes,
        )
        if det is not None:
            print("Using YOLOv8 detector")
            return det
    det = make_yunet((yunet_model or _default_yunet_path() or ""), score_thresh=float(score_thresh))
    if det is not None:
        print("Using YuNet detector")
        return det
    det = make_tinyface(
        tinyface_checkpoint,
        input_size=tinyface_input,
        conf_th=float(score_thresh),
        fp16=fp16,
        ema_mode=str(ema_mode),
    )
    if det is not None:
        print("Using TinyFace detector")
        return det
    raise RuntimeError(
        "No detector available. For YOLOv8: install ultralytics and provide --yolo-weights. "
        "For YuNet: install opencv-contrib-python and provide --yunet-model. "
        "For TinyFace: ensure pose.detectors.face_detector dependencies are installed."
    )


def main():
    parser = build_arg_parser()
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

    weights_only = True
    cfg = None
    if args.config:
        model, cfg, cfg_num_keypoints = _load_model_from_config(
            config_path=str(args.config),
            checkpoint_path=str(args.checkpoint),
            device=device,
            fallback_num_keypoints=int(args.num_keypoints),
            weights_only=weights_only,
            strict=bool(args.strict),
            ema_mode=str(args.ema),
        )
        args.num_keypoints = int(cfg_num_keypoints)
    else:
        model = load_heatmap_model(args.checkpoint, model_name=args.model_name, num_keypoints=args.num_keypoints, device=device)

    # Choose preprocessing input size (W,H). In config mode, prefer cfg.data.input_size
    # unless user forces --override-input-size.
    if args.config and cfg is not None and not args.override_input_size:
        data_cfg = (cfg or {}).get("data", {})
        if isinstance(data_cfg, dict) and "input_size" in data_cfg and len(data_cfg["input_size"]) == 2:
            preprocess_size = (int(data_cfg["input_size"][0]), int(data_cfg["input_size"][1]))
        else:
            preprocess_size = (int(args.input_size), int(args.input_size))
    else:
        preprocess_size = (int(args.input_size), int(args.input_size))

    if int(preprocess_size[0]) < 32 or int(preprocess_size[1]) < 32:
        print(f"[WARN] Very small preprocessing size {preprocess_size}. Did you mean e.g. 192 or 256?")

    try:
        print(f"Loaded model: {type(model).__name__} | preprocess_size={preprocess_size}")
        if hasattr(model, "input_size"):
            print(f"Model input_size attribute: {getattr(model, 'input_size')}")
    except Exception:
        pass
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
        out_path = f"outputs/{base}_landmarks.mp4"

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    obj_pts_3d = None
    if args.model_3d_xml:
        obj_pts_3d = _load_3d_landmarks_xml(args.model_3d_xml)
        print(f"Loaded 3D template points: {obj_pts_3d.shape[0]} from {args.model_3d_xml}")

    K_cam = _camera_matrix_from_args(W, H, args)
    dist = np.zeros((4, 1), dtype=np.float64)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    detector = build_detector(
        args.detector if args.detector != "auto" else "auto",
        device,
        score_thresh=float(args.detector_score),
        yunet_model=args.yunet_model,
        tinyface_checkpoint=args.tinyface_checkpoint,
        tinyface_input=args.tinyface_input,
        yolo_weights=args.yolo_weights,
        yolo_imgsz=args.yolo_imgsz,
        yolo_iou=args.yolo_iou,
        yolo_classes=args.yolo_classes,
        fp16=bool(args.fp16),
        ema_mode=str(args.ema),
    )

    frame_idx = 0
    t_start = time.time()
    processed = 0

    # ---- Per-stream caches (avoid redundant work when detect/infer cadence > 1) ----
    # last_boxes: list of detector boxes in full-frame pixel space.
    # Each box is either (x, y, w, h) or (x, y, w, h, score).
    last_boxes = []
    last_detect_frame = -10**9  # counter in "processed" frames

    # last_kpts: list of (K,2) landmarks in full-frame pixel space.
    last_kpts = []
    last_kpts_frame = -10**9

    # last_pnp: list aligned with last_kpts, each item contains rvec/tvec/proj/err.
    last_pnp = []
    last_pnp_frame = -10**9

    # Debug visualization cache for the tiled crop window.
    last_debug_tile = None

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

            # ---- 1) Read next frame ----
            t0 = perf_counter() if args.profile else None
            ret, frame_bgr = cap.read()
            if args.profile:
                prof["read"] += perf_counter() - t0

            if not ret:
                break

            if stop_requested["stop"]:
                break

            if (frame_idx % args.skip_frames) != 0:
                # Skip this frame entirely (write original frame, optionally preview).
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

            # ---- 2) Face detection (optionally cached) ----
            t0 = perf_counter() if args.profile else None
            do_detect = (processed - last_detect_frame) >= max(1, args.detect_every)
            boxes = last_boxes

            if do_detect:
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
                    for b in boxes:
                        if b is None or len(b) < 4:
                            continue
                        x, y, w, h = b[0], b[1], b[2], b[3]
                        if len(b) >= 5:
                            scaled.append((int(x * inv), int(y * inv), int(w * inv), int(h * inv), float(b[4])))
                        else:
                            scaled.append((int(x * inv), int(y * inv), int(w * inv), int(h * inv)))
                    boxes = scaled
                last_boxes = boxes
                last_detect_frame = processed

            if args.profile:
                prof["detect"] += perf_counter() - t0

            # ---- 3) Landmark inference (optionally cached) ----
            do_infer = (processed - last_kpts_frame) >= max(1, args.infer_every)
            if do_detect:
                # if boxes just changed, ensure we refresh keypoints
                do_infer = True

            if boxes and do_infer:
                t0 = perf_counter() if args.profile else None
                faces = []
                boxes_clamped = []
                face_inputs_bgr = []  # resized (stretched) model inputs, BGR
                for box in boxes[: args.max_faces]:
                    if box is None or len(box) < 4:
                        continue
                    x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    # --face-margin is percent (e.g. 50 -> +50%). For backward compat,
                    # treat values in [0..1.5] as already-a-fraction.
                    m = float(args.face_margin)
                    margin = (m / 100.0) if m > 1.5 else m
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
                    # x2/y2 are treated as EXCLUSIVE (Python slicing semantics)
                    x2 = min(W, x2)
                    y2 = min(H, y2)

                    face_bgr = frame_bgr[y1:y2, x1:x2]
                    if face_bgr.size == 0:
                        continue

                    if args.draw_crop_box:
                        cv2.rectangle(frame_bgr, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), (255, 0, 0), 1)

                    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                    img_t = _preprocess_rgb_to_tensor(face_rgb, input_size=preprocess_size)
                    faces.append(img_t)
                    boxes_clamped.append((x1, y1, x2, y2))

                    if args.show_input_crops:
                        face_in = cv2.resize(face_bgr, (int(preprocess_size[0]), int(preprocess_size[1])), interpolation=cv2.INTER_LINEAR)
                        face_inputs_bgr.append(face_in)

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
                    coords_norm_batch = _try_extract_norm_coords_batch(preds)

                    if coords_norm_batch is not None:
                        # (B, K, 2) in NORMALIZED [0,1] space. Scale by the
                        # actual preprocessing size to get crop-pixel coords,
                        # matching how the Trainer computes preds_2d during
                        # training (landmarks_norm * [W_in, H_in]).
                        coords_norm_batch = coords_norm_batch.detach().float().cpu().numpy()
                        if args.profile:
                            prof["to_cpu"] += perf_counter() - t0

                        t1 = perf_counter() if args.profile else None
                        kpts_out = []
                        W_face = float(preprocess_size[0])
                        H_face = float(preprocess_size[1])
                        debug_inputs = []
                        for i in range(coords_norm_batch.shape[0]):
                            coords = coords_norm_batch[i].copy()
                            # Normalized -> crop pixel space
                            coords[:, 0] *= W_face
                            coords[:, 1] *= H_face
                            if args.show_input_crops and i < len(face_inputs_bgr):
                                dbg = face_inputs_bgr[i].copy()
                                draw_landmarks(dbg, coords, color=(0, 0, 255))
                                cv2.putText(dbg, f"face {i}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                debug_inputs.append(dbg)
                            # Crop pixel space -> full frame pixel space
                            x1, y1, x2, y2 = boxes_clamped[i]
                            scale_x = (x2 - x1) / W_face
                            scale_y = (y2 - y1) / H_face
                            coords[:, 0] = coords[:, 0] * scale_x + x1
                            coords[:, 1] = coords[:, 1] * scale_y + y1
                            kpts_out.append(coords)

                        if args.show_input_crops and debug_inputs:
                            last_debug_tile = _tile_same_size(debug_inputs, cols=args.debug_crops_cols)

                        last_kpts = kpts_out
                        last_kpts_frame = processed

                        if obj_pts_3d is not None:
                            last_pnp = _run_pnp_batch(
                                obj_pts_3d,
                                kpts_out,
                                K_cam,
                                dist,
                                method=str(args.pnp_method),
                                max_reproj_err=float(args.pnp_max_reproj_err),
                            )
                            last_pnp_frame = processed
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
                        debug_inputs = []
                        for i in range(preds_last.shape[0]):
                            heatmaps = preds_last[i]
                            coords = decode_heatmaps(heatmaps)

                            H_hm, W_hm = heatmaps.shape[1:]
                            W_face = float(preprocess_size[0])
                            H_face = float(preprocess_size[1])
                            coords[:, 0] *= (W_face / W_hm)
                            coords[:, 1] *= (H_face / H_hm)

                            if args.show_input_crops and i < len(face_inputs_bgr):
                                dbg = face_inputs_bgr[i].copy()
                                draw_landmarks(dbg, coords, color=(0, 0, 255))
                                cv2.putText(dbg, f"face {i}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                debug_inputs.append(dbg)

                            x1, y1, x2, y2 = boxes_clamped[i]
                            scale_x = (x2 - x1) / float(W_face)
                            scale_y = (y2 - y1) / float(H_face)
                            coords[:, 0] = coords[:, 0] * scale_x + x1
                            coords[:, 1] = coords[:, 1] * scale_y + y1

                            kpts_out.append(coords)

                        if args.show_input_crops and debug_inputs:
                            last_debug_tile = _tile_same_size(debug_inputs, cols=args.debug_crops_cols)

                        last_kpts = kpts_out
                        last_kpts_frame = processed

                        if obj_pts_3d is not None:
                            last_pnp = _run_pnp_batch(
                                obj_pts_3d,
                                kpts_out,
                                K_cam,
                                dist,
                                method=str(args.pnp_method),
                                max_reproj_err=float(args.pnp_max_reproj_err),
                            )
                            last_pnp_frame = processed

                        if args.profile:
                            prof["decode_draw"] += perf_counter() - t1

            # ---- 4) Draw overlays ----
            # We draw the *latest cached* keypoints, which may be from a previous
            # frame if --infer-every > 1.
            if boxes:
                t0 = perf_counter() if args.profile else None
                for i, box in enumerate(boxes[: args.max_faces]):
                    if box is None or len(box) < 4:
                        continue
                    x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    if args.draw_face_box:
                        pad = int(max(0, args.box_pad))
                        x1 = max(0, int(x) - pad)
                        y1 = max(0, int(y) - pad)
                        x2 = min(W - 1, int(x + w) + pad)
                        y2 = min(H - 1, int(y + h) + pad)
                        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    if i < len(last_kpts):
                        if args.draw_detected_landmarks:
                            draw_landmarks(frame_bgr, last_kpts[i], color=(0, 0, 255))

                        if obj_pts_3d is not None and i < len(last_pnp):
                            proj = last_pnp[i].get("proj") if isinstance(last_pnp[i], dict) else None
                            rvec = last_pnp[i].get("rvec") if isinstance(last_pnp[i], dict) else None
                            tvec = last_pnp[i].get("tvec") if isinstance(last_pnp[i], dict) else None

                            if args.draw_pnp and proj is not None:
                                draw_landmarks(frame_bgr, proj, color=(0, 255, 0))

                            if args.draw_axes and rvec is not None and tvec is not None:
                                try:
                                    cv2.drawFrameAxes(frame_bgr, K_cam, dist, rvec, tvec, float(args.axes_len), 2)
                                except Exception:
                                    pass
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
                if args.show_input_crops and last_debug_tile is not None:
                    cv2.imshow(f"{window_name}_inputs", last_debug_tile)
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
