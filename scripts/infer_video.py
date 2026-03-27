"""Run face detection + landmark inference on a video and save an annotated video.

This script intentionally keeps the CLI small. Most settings are inferred from
the provided YAML configs.

Required:
    --detector-config   YAML config for face detector
    --detector-chkpt    detector checkpoint (.pth)
    --landmarks-config  YAML config for landmark model
    --landmarks-chkpt   landmarks checkpoint (.pth)
    --input-video       input video path

Example:
python scripts/infer_video.py \
    --detector-config configs/face_efficientnet.yaml --detector-chkpt work_dirs/face_efficientnet/best.pth \
    --landmarks-config configs/xreal_lotr_light.yaml --landmarks-chkpt work_dirs/xreal_lotr_light/best.pth \
    --input-video input.mp4
"""
import argparse
import math
import os
import time
import signal
from typing import Tuple
import xml.etree.ElementTree as ET

try:
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None
import torch
import numpy as np

from pose.engine.inference import decode_heatmaps
from pose.config import Config
from pose.registry import MODEL_REGISTRY

# Ensure model modules are imported so they register themselves in MODEL_REGISTRY
import pose.models  # noqa: F401
import pose.detectors  # noqa: F401


LANDMARK_LABEL_INDICES = (0, 9, 26, 57, 63)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument("--detector-chkpt", required=True, help="Face detector checkpoint (.pth)")
    parser.add_argument("--detector-config", required=True, help="Face detector YAML config")
    parser.add_argument("--landmarks-chkpt", required=True, help="Landmarks checkpoint (.pth)")
    parser.add_argument("--landmarks-config", required=True, help="Landmarks YAML config")
    parser.add_argument("--input-video", required=True)

    # Optional
    parser.add_argument("--output-video", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--detector-margin",
        type=float,
        default=None,
        help="Extra margin around detected face box for the landmark crop. Fraction (0.5 = +50%%). Default: inferred from landmarks config.",
    )
    parser.add_argument("--draw-face-box", action="store_true")
    parser.add_argument("--draw-crop-box", action="store_true")
    parser.add_argument("--draw-landmarks", action="store_true")
    parser.add_argument("--draw-axes", action="store_true", help="Draw pose axes (requires --model-3d-xml)")

    # True by default; use --no-display to disable.
    try:
        parser.add_argument("--display", action=argparse.BooleanOptionalAction, default=True)
    except Exception:
        parser.add_argument("--display", action="store_true", default=True)

    parser.add_argument("--model-3d-xml", default=None)
    parser.add_argument("--pnp-max-repr-err", type=float, default=8.0)
    parser.add_argument("--pnp-method", choices=["iterative", "epnp", "p3p", "ap3p"], default="iterative")
    parser.add_argument("--cam_fx", type=float, default=None)
    parser.add_argument("--cam_fy", type=float, default=None)
    parser.add_argument("--cam_cx", type=float, default=None)
    parser.add_argument("--cam_cy", type=float, default=None)
    parser.add_argument("--detect-every", type=int, default=1)
    return parser


def draw_landmarks(
    frame: np.ndarray,
    coords: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    confidences: np.ndarray | None = None,
):
    """Draw 2D landmark points (x,y) onto a BGR frame."""
    for i, (xk, yk) in enumerate(coords):
        draw_color = color
        if confidences is not None and color == (0, 0, 255):
            c = float(np.clip(confidences[i], 0.0, 1.0)) if i < len(confidences) else 0.0
            red = int(round(64.0 + 191.0 * c))
            draw_color = (0, 0, red)
        cv2.circle(frame, (int(round(float(xk))), int(round(float(yk)))), 2, draw_color, -1)


def draw_landmark_labels(
    frame: np.ndarray,
    coords: np.ndarray,
    indices: tuple[int, ...],
    confidences: np.ndarray | None = None,
) -> None:
    """Draw small labels near selected landmarks."""
    if coords is None or len(coords) == 0:
        return
    h, w = frame.shape[:2]
    for idx in indices:
        if idx < 0 or idx >= len(coords):
            continue
        xk, yk = coords[idx]
        x = int(round(float(xk)))
        y = int(round(float(yk)))

        if confidences is not None and idx < len(confidences):
            c = float(np.clip(confidences[idx], 0.0, 1.0))
            text = f"{idx}:{c:.2f}"
        else:
            text = f"{idx}"

        tx = max(0, min(w - 1, x + 4))
        ty = max(0, min(h - 1, y - 4))
        cv2.putText(
            frame,
            text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


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

    If fx/fy aren't provided, use a simple heuristic (max(W,H)).
    """
    cx = float(args.cam_cx) if args.cam_cx is not None else (W / 2.0)
    cy = float(args.cam_cy) if args.cam_cy is not None else (H / 2.0)

    fx = args.cam_fx
    fy = args.cam_fy

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
        load_res = model.load_state_dict(state_dict, strict=bool(strict))
        if not bool(strict):
            missing = getattr(load_res, "missing_keys", [])
            unexpected = getattr(load_res, "unexpected_keys", [])
            if missing or unexpected:
                print(f"[WARN] Base checkpoint load had missing={len(missing)} unexpected={len(unexpected)}")
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
    coords_norm = None
    if isinstance(model_out, tuple) and len(model_out) >= 2:
        coords_norm = model_out[0]
    elif isinstance(model_out, dict):
        coords_norm = model_out.get("landmarks_norm")
    else:
        return None
    if not isinstance(coords_norm, torch.Tensor):
        return None
    if coords_norm.dim() != 3:
        return None
    if coords_norm.size(-1) < 2:
        return None
    return coords_norm[..., :2]


def _try_extract_pixel_coords_batch(model_out: object) -> torch.Tensor | None:
    """Try to extract pixel-space (B, K, 2) coordinates from LOTR-style outputs."""
    coords_px = None
    if isinstance(model_out, tuple) and len(model_out) >= 2:
        coords_px = model_out[1]
    elif isinstance(model_out, dict):
        coords_px = model_out.get("landmarks_pixel")
    if not isinstance(coords_px, torch.Tensor):
        return None
    if coords_px.dim() != 3:
        return None
    if coords_px.size(-1) < 2:
        return None
    return coords_px[..., :2]


def _try_extract_confidence_batch(model_out: object) -> torch.Tensor | None:
    """Try to extract LOTR landmark confidence tensor as (B, K)."""
    conf = None
    if isinstance(model_out, tuple) and len(model_out) >= 3:
        conf = model_out[2]
    elif isinstance(model_out, dict):
        conf = model_out.get("landmark_confidence")

    # New channel format support: confidence packed as the 3rd channel in
    # landmark tensors, e.g. (B, K, 3) where [:, :, :2] are coords.
    if conf is None:
        packed = None
        if isinstance(model_out, tuple) and len(model_out) >= 1 and isinstance(model_out[0], torch.Tensor):
            packed = model_out[0]
        elif isinstance(model_out, dict):
            packed = model_out.get("landmarks_norm")
            if not isinstance(packed, torch.Tensor):
                packed = model_out.get("landmarks_pixel")
        if isinstance(packed, torch.Tensor) and packed.dim() == 3 and packed.size(-1) >= 3:
            conf = packed[..., 2]

    if not isinstance(conf, torch.Tensor):
        return None
    if conf.dim() == 3 and conf.size(-1) == 1:
        conf = conf.squeeze(-1)
    if conf.dim() != 2:
        return None
    return conf


def _get_cfg_input_size(cfg_raw: dict, *, fallback: Tuple[int, int]) -> Tuple[int, int]:
    data_cfg = (cfg_raw or {}).get("data", {})
    if isinstance(data_cfg, dict) and "input_size" in data_cfg and len(data_cfg["input_size"]) == 2:
        try:
            return (int(data_cfg["input_size"][0]), int(data_cfg["input_size"][1]))
        except Exception:
            return fallback
    return fallback


def _get_model_input_size(model: torch.nn.Module) -> Tuple[int, int] | None:
    """Return model.input_size as (W, H) when available and valid."""
    sz = getattr(model, "input_size", None)
    if isinstance(sz, (tuple, list)) and len(sz) == 2:
        try:
            w = int(sz[0])
            h = int(sz[1])
            if w > 0 and h > 0:
                return (w, h)
        except Exception:
            return None
    return None


def _infer_default_margin_from_landmarks_cfg(cfg_raw: dict) -> float:
    try:
        data_cfg = (cfg_raw or {}).get("data", {}) or {}
        aug = (data_cfg.get("aug", {}) if isinstance(data_cfg, dict) else {}) or {}
        m = aug.get("face_margin")
        if m is None:
            return 0.5
        return float(m)
    except Exception:
        return 0.5


def load_detector_from_config(*, config_path: str, checkpoint_path: str, device: torch.device):
    """Load detector model from config+checkpoint.

    Expected detector model type: registered as cfg.model.name and implements
    a `.predict(x, conf_th=...)` method returning a list of dicts with
    {'conf': float, 'bbox': [x1,y1,x2,y2]} in input-image pixel space.

    Returns: (detector_wrapper, cfg_raw, input_size)
    """

    det_model, det_cfg, _ = _load_model_from_config(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        device=device,
        fallback_num_keypoints=0,
        weights_only=True,
        strict=False,
        # Use non-EMA weights for detector, matching scripts/infer_bbox.py.
        # The EMA shadow for bbox models is typically incomplete (missing BN
        # buffers), so infer_bbox falls back to ckpt['model'].  Using the
        # partial EMA overlay produces a tighter / shifted box.
        ema_mode="off",
    )

    det_input_size = _get_cfg_input_size(det_cfg, fallback=(192, 192))

    class DetectorWrapper:
        def __init__(self, model, input_size: Tuple[int, int]):
            self.model = model
            self.input_size = (int(input_size[0]), int(input_size[1]))

        def detect(self, img_bgr: np.ndarray, *, conf_th: float = 0.4):
            H0, W0 = img_bgr.shape[:2]
            if H0 <= 0 or W0 <= 0:
                return []

            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            W_in, H_in = int(self.input_size[0]), int(self.input_size[1])
            # Match scripts/infer_bbox.py behavior: direct resize to model input size.
            x_t = _preprocess_rgb_to_tensor(rgb, input_size=self.input_size).unsqueeze(0)
            x_t = x_t.to(device=device)

            with torch.inference_mode():
                if hasattr(self.model, "predict"):
                    res = self.model.predict(x_t, conf_th=float(conf_th))
                else:
                    raise TypeError(f"Detector model {type(self.model).__name__} has no .predict()")

            if not res or not isinstance(res, list):
                return []
            bbox = res[0].get("bbox") if isinstance(res[0], dict) else None
            if bbox is None:
                return []

            conf = float(res[0].get("conf", 1.0)) if isinstance(res[0], dict) else 1.0
            x1, y1, x2, y2 = bbox

            sx = float(W0) / max(1e-9, float(W_in))
            sy = float(H0) / max(1e-9, float(H_in))
            x1 = int(round(float(x1) * sx))
            y1 = int(round(float(y1) * sy))
            x2 = int(round(float(x2) * sx))
            y2 = int(round(float(y2) * sy))

            x1 = max(0, min(W0 - 1, x1))
            y1 = max(0, min(H0 - 1, y1))
            x2 = max(0, min(W0 - 1, x2))
            y2 = max(0, min(H0 - 1, y2))
            if x2 <= x1 or y2 <= y1:
                return []
            return [(int(x1), int(y1), int(x2 - x1), int(y2 - y1), float(conf))]

    return DetectorWrapper(det_model, det_input_size), det_cfg, det_input_size


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if cv2 is None:
        raise ModuleNotFoundError(
            "OpenCV (cv2) is required for video I/O. Install e.g. 'opencv-python' or 'opencv-contrib-python'."
        )

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

    # Landmarks model is always loaded from config.
    model, lm_cfg, _ = _load_model_from_config(
        config_path=str(args.landmarks_config),
        checkpoint_path=str(args.landmarks_chkpt),
        device=device,
        fallback_num_keypoints=68,
        weights_only=True,
        strict=False,
        ema_mode="auto",
    )
    preprocess_size_cfg = _get_cfg_input_size(lm_cfg, fallback=(256, 256))
    preprocess_size_model = _get_model_input_size(model)
    if preprocess_size_model is not None:
        preprocess_size = preprocess_size_model
        if preprocess_size != preprocess_size_cfg:
            print(
                f"[WARN] landmarks config input_size={preprocess_size_cfg} differs from loaded model input_size={preprocess_size}; "
                f"using model input_size for preprocessing."
            )
    else:
        preprocess_size = preprocess_size_cfg

    if int(preprocess_size[0]) < 32 or int(preprocess_size[1]) < 32:
        print(f"[WARN] Very small preprocessing size {preprocess_size}. Did you mean e.g. 192 or 256?")

    print(f"Loaded landmarks model: {type(model).__name__} | preprocess_size={preprocess_size}")
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

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    detector, det_cfg, det_input_size = load_detector_from_config(
        config_path=str(args.detector_config),
        checkpoint_path=str(args.detector_chkpt),
        device=device,
    )
    print(f"Loaded detector model from {args.detector_config} | input_size={det_input_size}")

    if args.detector_margin is None:
        # Detector boxes are typically looser than GT annotation boxes used
        # during training. Reusing training face_margin (often ~0.75) here can
        # produce oversized crops and make landmarks appear too small/unstable.
        margin_default = 0.25
        margin_arg = float(margin_default)
        print(
            f"Using detector margin default: {margin_arg:.2f}. "
            "Use --detector-margin to override."
        )
    else:
        margin_arg = float(args.detector_margin)

    # Backward-compat convenience: treat large values as percent.
    margin_frac = (margin_arg / 100.0) if margin_arg > 1.5 else margin_arg

    t_start = time.time()
    processed = 0

    # Cache the last detector box so --detect-every can skip detector work.
    last_box = None
    last_detect_frame = -10**9

    window_name = "mini_pose"

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

            # ---- 1) Detection (optionally cached) ----
            do_detect = (processed - last_detect_frame) >= max(1, int(args.detect_every))
            if do_detect:
                dets = detector.detect(frame_bgr, conf_th=0.4)
                last_box = dets[0] if dets else None
                last_detect_frame = processed

            # ---- 2) Landmark inference (always runs; uses last_box when detector is skipped) ----
            if last_box is not None:
                x, y, w, h = int(last_box[0]), int(last_box[1]), int(last_box[2]), int(last_box[3])

                if args.draw_face_box:
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(W - 1, x + w)
                    y2 = min(H - 1, y + h)
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 1)

                # Build square crop around detection, expanded by margin_frac.
                cx = x + w / 2.0
                cy = y + h / 2.0
                size = max(w, h) * (1.0 + float(margin_frac))
                x1 = int(cx - size / 2.0)
                y1 = int(cy - size / 2.0)
                x2 = int(cx + size / 2.0)
                y2 = int(cy + size / 2.0)

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(W, x2)
                y2 = min(H, y2)
                face_bgr = frame_bgr[y1:y2, x1:x2]

                kpts_frame = None
                kpts_conf = None
                pnp_sol = None
                if face_bgr.size != 0:
                    if args.draw_crop_box:
                        cv2.rectangle(frame_bgr, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), (255, 0, 0), 1)

                    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                    img_t = _preprocess_rgb_to_tensor(face_rgb, input_size=preprocess_size)
                    batch = img_t.unsqueeze(0).to(device)

                    with torch.inference_mode():
                        preds = model(batch)

                    # Decode either coordinate-regression outputs (LOTR) or heatmaps.
                    # Prefer explicit pixel outputs when available to avoid
                    # ambiguity between normalized/pixel channels.
                    coords_px_batch = _try_extract_pixel_coords_batch(preds)
                    coords_norm_batch = _try_extract_norm_coords_batch(preds)
                    conf_batch = _try_extract_confidence_batch(preds)
                    if coords_px_batch is not None:
                        coords = coords_px_batch[0].detach().float().cpu().numpy().astype(np.float32)
                        if conf_batch is not None:
                            kpts_conf = conf_batch[0].detach().float().cpu().numpy().astype(np.float32)
                            kpts_conf = np.clip(kpts_conf, 0.0, 1.0)
                    elif coords_norm_batch is not None:
                        coords_norm = coords_norm_batch[0].detach().float().cpu().numpy().astype(np.float32)
                        coords = coords_norm.copy()
                        coords[:, 0] *= float(preprocess_size[0])
                        coords[:, 1] *= float(preprocess_size[1])

                        if conf_batch is not None:
                            kpts_conf = conf_batch[0].detach().float().cpu().numpy().astype(np.float32)
                            kpts_conf = np.clip(kpts_conf, 0.0, 1.0)
                    else:
                        preds_last = _extract_preds_last(preds)[0].detach().cpu()
                        coords = decode_heatmaps(preds_last)
                        H_hm, W_hm = preds_last.shape[1:]
                        coords[:, 0] *= float(preprocess_size[0]) / float(W_hm)
                        coords[:, 1] *= float(preprocess_size[1]) / float(H_hm)

                    # Crop pixel space -> full-frame pixel space.
                    scale_x = (x2 - x1) / float(preprocess_size[0])
                    scale_y = (y2 - y1) / float(preprocess_size[1])
                    coords[:, 0] = coords[:, 0] * scale_x + x1
                    coords[:, 1] = coords[:, 1] * scale_y + y1
                    kpts_frame = coords

                    if obj_pts_3d is not None:
                        pnp = _run_pnp_batch(
                            obj_pts_3d,
                            [kpts_frame],
                            K_cam,
                            dist,
                            method=str(args.pnp_method),
                            max_reproj_err=float(args.pnp_max_repr_err),
                        )
                        if pnp and isinstance(pnp[0], dict):
                            pnp_sol = pnp[0]

                if args.draw_landmarks and kpts_frame is not None:
                    draw_landmarks(frame_bgr, kpts_frame, color=(0, 0, 255), confidences=kpts_conf)
                    draw_landmark_labels(
                        frame_bgr,
                        kpts_frame,
                        indices=LANDMARK_LABEL_INDICES,
                        confidences=kpts_conf,
                    )

                # If 3D template is provided, draw projected points and axes by default.
                if obj_pts_3d is not None and pnp_sol is not None:
                    proj = pnp_sol.get("proj")
                    rvec = pnp_sol.get("rvec")
                    tvec = pnp_sol.get("tvec")
                    if proj is not None:
                        draw_landmarks(frame_bgr, proj, color=(0, 255, 0))
                    if args.draw_axes and rvec is not None and tvec is not None:
                        try:
                            cv2.drawFrameAxes(frame_bgr, K_cam, dist, rvec, tvec, 0.05, 2)
                        except Exception:
                            pass

            writer.write(frame_bgr)

            if args.display:
                cv2.imshow(window_name, frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    stop_requested["stop"] = True

            processed += 1

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
