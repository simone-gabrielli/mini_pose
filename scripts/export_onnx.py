"""Export a trained mini-pose model to ONNX.

This script mirrors the project's config-driven model construction used by
`Trainer` and `scripts/infer.py`:
  - Loads YAML via `pose.config.Config`
  - Builds model from `pose.registry.MODEL_REGISTRY`
  - Loads weights from a `.pth` checkpoint (supports Trainer checkpoints)
  - Optionally swaps in EMA weights when present

Examples:
  python scripts/export_onnx.py --config configs/xreal_mobilenet.yaml --checkpoint work_dirs/xreal_mobilenet/best.pth --output work_dirs/xreal_mobilenet/model.onnx

  # Export LOTR pixel landmark outputs
  python scripts/export_onnx.py --config configs/xreal_lotr.yaml --checkpoint work_dirs/xreal_lotr/best.pth --output work_dirs/xreal_lotr/lotr_pixel.onnx --output-type coords_pixel

    # Export LOTR per-landmark confidence head
    python scripts/export_onnx.py --config configs/xreal_lotr.yaml --checkpoint work_dirs/xreal_lotr/best.pth --output work_dirs/xreal_lotr/lotr_confidence.onnx --output-type confidence

    # Export LOTR coords_pixel + confidence in one ONNX
    python scripts/export_onnx.py --config configs/xreal_lotr.yaml --checkpoint work_dirs/xreal_lotr/best.pth --output work_dirs/xreal_lotr/lotr_pixel_confidence.onnx --output-type coords_pixel_confidence

Notes:
  - ONNX export typically requires the `onnx` Python package.
"""

from __future__ import annotations

import argparse
import inspect
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from pose.config import Config
from pose.registry import MODEL_REGISTRY

# Ensure model modules are imported so they register themselves in MODEL_REGISTRY
import pose.models  # noqa: F401


@dataclass(frozen=True)
class ExportSpec:
    input_size_hw: Tuple[int, int]
    num_keypoints: int
    model_name: str


def _parse_input_size(value: str) -> Tuple[int, int]:
    v = str(value).lower().strip()
    if "x" in v:
        a, b = v.split("x", 1)
        return int(a), int(b)
    n = int(v)
    return n, n


def _safe_torch_load(path: str, *, weights_only: bool) -> Any:
    # weights_only is supported in newer PyTorch; fall back if unavailable.
    try:
        return torch.load(path, map_location="cpu", weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _build_model_from_cfg(cfg: Dict[str, Any], *, num_keypoints: int) -> torch.nn.Module:
    model_cfg = cfg.get("model", {}) or {}
    if "name" not in model_cfg:
        raise ValueError("Config missing required key: model.name")

    model_name = str(model_cfg["name"])
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(f"Model '{model_name}' not found in MODEL_REGISTRY. Available models: {available}")

    ModelCls = MODEL_REGISTRY[model_name]

    model_kwargs: Dict[str, Any] = {"num_keypoints": int(num_keypoints)}
    for k, v in model_cfg.items():
        if k == "name":
            continue
        model_kwargs[k] = v

    # If config has data.input_size, prefer passing it when the model accepts it.
    data_cfg = cfg.get("data", {}) or {}
    if "input_size" in data_cfg:
        try:
            # Keep it as (H, W)
            input_size = tuple(data_cfg["input_size"])
            if len(input_size) == 2:
                model_kwargs["input_size"] = (int(input_size[0]), int(input_size[1]))
        except Exception:
            pass

    # Only pass kwargs the model accepts (matches Trainer behavior).
    try:
        sig = inspect.signature(ModelCls.__init__)
        params = sig.parameters
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if accepts_var_kw:
            kwargs_to_pass = model_kwargs
        else:
            allowed = {name for name in params.keys() if name != "self"}
            kwargs_to_pass = {k: v for k, v in model_kwargs.items() if k in allowed}
    except Exception:
        kwargs_to_pass = model_kwargs

    return ModelCls(**kwargs_to_pass)


def _infer_input_size_from_lotr_pe(state_dict: Dict[str, Any], model_cfg: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    pe = state_dict.get("pos_encoding.pe")
    if not isinstance(pe, torch.Tensor) or pe.dim() != 3:
        return None

    pe_h, pe_w = int(pe.shape[0]), int(pe.shape[1])

    # LOTR feature map starts at stride 32, then optional upsampling (2^upsampling_layers)
    use_upsampling = bool(model_cfg.get("use_upsampling", True))
    up_layers = int(model_cfg.get("upsampling_layers", 2)) if use_upsampling else 0
    scale = 2 ** max(0, up_layers)

    input_h = int(round(pe_h * 32 / float(scale)))
    input_w = int(round(pe_w * 32 / float(scale)))
    if input_h <= 0 or input_w <= 0:
        return None
    return (input_h, input_w)


class _OnnxOutputWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, output_type: str):
        super().__init__()
        self.model = model
        self.output_type = str(output_type).lower().strip()

    def forward(self, x: torch.Tensor) -> Any:
        out = self.model(x)

        # Dict outputs
        if isinstance(out, dict):
            if self.output_type in ("heatmaps", "auto") and isinstance(out.get("heatmaps"), torch.Tensor):
                return out["heatmaps"]
            if self.output_type in ("coords_pixel", "auto") and isinstance(out.get("landmarks_pixel"), torch.Tensor):
                return out["landmarks_pixel"]
            if self.output_type in ("coords_norm", "auto") and isinstance(out.get("landmarks_norm"), torch.Tensor):
                return out["landmarks_norm"]
            if self.output_type == "confidence":
                for k in ("landmark_confidence", "confidence", "pred_confidence"):
                    v = out.get(k)
                    if isinstance(v, torch.Tensor):
                        return v
            if self.output_type == "coords_pixel_confidence":
                px = out.get("landmarks_pixel")
                conf = None
                for k in ("landmark_confidence", "confidence", "pred_confidence"):
                    v = out.get(k)
                    if isinstance(v, torch.Tensor):
                        conf = v
                        break
                if isinstance(px, torch.Tensor) and isinstance(conf, torch.Tensor):
                    return px, conf
            if self.output_type == "log_var":
                for k in ("landmark_log_var", "log_var", "uncertainty", "pred_log_var"):
                    v = out.get(k)
                    if isinstance(v, torch.Tensor):
                        return v
            # fallback: first tensor value
            for v in out.values():
                if isinstance(v, torch.Tensor):
                    return v
            raise TypeError("Model returned a dict without any tensor values")

        # Tuple/list outputs (common for hourglass / FAN / MobileNetPose / LOTR)
        if isinstance(out, (tuple, list)):
            if len(out) == 0:
                raise TypeError("Model returned an empty tuple/list")

            # LOTR returns (norm, pixel)
            if self.output_type == "coords_norm" and isinstance(out[0], torch.Tensor):
                return out[0]
            if self.output_type == "coords_pixel" and len(out) >= 2 and isinstance(out[1], torch.Tensor):
                return out[1]
            if self.output_type == "confidence" and len(out) >= 3 and isinstance(out[2], torch.Tensor):
                return out[2]
            if self.output_type == "coords_pixel_confidence" and len(out) >= 3:
                if isinstance(out[1], torch.Tensor) and isinstance(out[2], torch.Tensor):
                    return out[1], out[2]
            if self.output_type == "log_var" and len(out) >= 4 and isinstance(out[3], torch.Tensor):
                return out[3]

            # Heatmap-style models return (pred_last, preds_all)
            if self.output_type in ("heatmaps", "auto") and isinstance(out[0], torch.Tensor):
                return out[0]

            # fallback: first tensor element
            for v in out:
                if isinstance(v, torch.Tensor):
                    return v
            raise TypeError("Model returned tuple/list without any tensor elements")

        # Plain tensor
        if isinstance(out, torch.Tensor):
            return out

        raise TypeError(f"Unsupported model output type for ONNX export: {type(out)}")


def _has_confidence_output(out: Any) -> bool:
    if isinstance(out, dict):
        for k in ("landmark_confidence", "confidence", "pred_confidence"):
            if isinstance(out.get(k), torch.Tensor):
                return True
        return False
    if isinstance(out, (tuple, list)):
        return len(out) >= 3 and isinstance(out[2], torch.Tensor)
    return False


def _resolve_export_spec(cfg: Dict[str, Any], args: argparse.Namespace) -> ExportSpec:
    data_cfg = cfg.get("data", {}) or {}
    model_cfg = cfg.get("model", {}) or {}

    model_name = str(args.model_name or model_cfg.get("name") or "")
    if not model_name:
        raise ValueError("Unable to determine model name. Provide --model-name or set model.name in config.")

    num_keypoints = int(args.num_keypoints if args.num_keypoints is not None else data_cfg.get("num_keypoints", 68))

    if args.input_size is not None:
        h, w = _parse_input_size(args.input_size)
    elif "input_size" in data_cfg:
        h, w = int(data_cfg["input_size"][0]), int(data_cfg["input_size"][1])
    else:
        h, w = 256, 256

    return ExportSpec(input_size_hw=(h, w), num_keypoints=num_keypoints, model_name=model_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output", required=True, help="Path to output .onnx")

    parser.add_argument("--model-name", default=None, help="Optional override for config model.name")
    parser.add_argument("--num-keypoints", type=int, default=None, help="Fallback if config data.num_keypoints missing")

    parser.add_argument(
        "--input-size",
        default=None,
        help="Export input size: either N (square) or HxW (e.g. 256 or 192x192). Defaults to config data.input_size.",
    )
    parser.add_argument("--batch", type=int, default=1, help="Dummy batch size for export")

    parser.add_argument(
        "--output-type",
        default="auto",
        choices=["auto", "heatmaps", "coords_norm", "coords_pixel", "confidence", "log_var", "coords_pixel_confidence"],
        help="Which tensor to export when the model returns multiple outputs.",
    )

    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--weights-only", action="store_true", help="Use torch.load(..., weights_only=True) if supported")
    parser.add_argument("--strict", action="store_true", help="Strict state_dict loading")
    parser.add_argument("--ema", default="off", choices=["auto", "on", "off"], help="Use EMA weights if present")

    parser.add_argument("--dynamic-batch", action="store_true", help="Export with dynamic batch dimension")
    parser.add_argument(
        "--dynamic-spatial",
        action="store_true",
        help="Also export with dynamic H/W (may not work for models with fixed positional encodings).",
    )

    args = parser.parse_args()

    # Friendly error if onnx package is missing.
    try:
        import onnx  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "ONNX export requires the `onnx` package. Install it with: pip install onnx"
        ) from e

    cfg = Config.from_yaml(args.config).raw

    # Optionally override model name in cfg for construction.
    if args.model_name is not None:
        cfg = dict(cfg)
        cfg_model = dict(cfg.get("model", {}) or {})
        cfg_model["name"] = str(args.model_name)
        cfg["model"] = cfg_model

    spec = _resolve_export_spec(cfg, args)

    # Build model
    model = _build_model_from_cfg(cfg, num_keypoints=spec.num_keypoints)

    # Load checkpoint
    ckpt = _safe_torch_load(args.checkpoint, weights_only=bool(args.weights_only))
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint did not contain a state_dict dict")

    # Handle positional encoding buffer mismatches (common for LOTR)
    try:
        model.load_state_dict(state_dict, strict=bool(args.strict))
    except RuntimeError as e:
        sd = dict(state_dict)
        model_cfg = cfg.get("model", {}) or {}
        inferred = _infer_input_size_from_lotr_pe(sd, model_cfg)
        if inferred is not None:
            # Update config's data.input_size so the model gets rebuilt with matching scaling.
            cfg2 = dict(cfg)
            data2 = dict(cfg2.get("data", {}) or {})
            data2["input_size"] = [int(inferred[0]), int(inferred[1])]
            cfg2["data"] = data2
            model = _build_model_from_cfg(cfg2, num_keypoints=spec.num_keypoints)
            try:
                model.load_state_dict(sd, strict=bool(args.strict))
            except RuntimeError:
                # Drop PE as last resort.
                sd.pop("pos_encoding.pe", None)
                model.load_state_dict(sd, strict=False)
        else:
            # Drop PE as last resort.
            if "pos_encoding.pe" in sd:
                sd.pop("pos_encoding.pe", None)
                model.load_state_dict(sd, strict=False)
            else:
                raise e

    # Optional EMA swap
    ema_shadow = None
    if isinstance(ckpt, dict):
        ema_state = ckpt.get("ema")
        if isinstance(ema_state, dict) and isinstance(ema_state.get("shadow"), dict):
            ema_shadow = ema_state["shadow"]

    def _should_use_ema() -> bool:
        mode = str(args.ema).lower().strip() if args.ema is not None else "auto"
        if mode == "off":
            return False
        if mode == "on":
            return True
        # auto: follow config if present, else use EMA when available
        ema_cfg = (cfg.get("train", {}) or {}).get("ema", {}) or {}
        return bool(ema_cfg.get("eval", True))

    if ema_shadow is not None and _should_use_ema():
        model.load_state_dict(ema_shadow, strict=False)

    model.eval()
    model.cpu()

    # Wrap output selection for stable, explicit export outputs.
    export_model = _OnnxOutputWrapper(model, output_type=args.output_type).eval()

    h, w = spec.input_size_hw
    dummy = torch.randn(int(args.batch), 3, int(h), int(w), dtype=torch.float32)

    # Quick dry-run to make export errors easier to interpret.
    with torch.no_grad():
        raw_out = model(dummy)
        if args.output_type == "auto" and _has_confidence_output(raw_out):
            print(
                "[WARN] Model exposes a confidence head, but --output-type=auto exports the first tensor "
                "(usually coordinates). Use --output-type confidence to export confidence scores."
            )
        out = export_model(dummy)
    if isinstance(out, torch.Tensor):
        output_names = ["output"]
        output_shapes = [tuple(out.shape)]
    elif isinstance(out, (tuple, list)) and len(out) > 0 and all(isinstance(v, torch.Tensor) for v in out):
        if args.output_type == "coords_pixel_confidence" and len(out) == 2:
            output_names = ["coords_pixel", "confidence"]
        else:
            output_names = [f"output_{i}" for i in range(len(out))]
        output_shapes = [tuple(v.shape) for v in out]
    else:
        raise TypeError("Wrapped model output is not a Tensor or tuple/list of Tensors; cannot export")

    # Build dynamic axes mapping.
    dynamic_axes = None
    if args.dynamic_batch or args.dynamic_spatial:
        dynamic_axes = {"image": {0: "batch"}}
        for name in output_names:
            dynamic_axes[name] = {0: "batch"}
        if args.dynamic_spatial:
            dynamic_axes["image"].update({2: "height", 3: "width"})

    if len(output_shapes) == 1:
        shape_msg = str(output_shapes[0])
    else:
        shape_msg = ", ".join(f"{n}:{s}" for n, s in zip(output_names, output_shapes))
    print(f"Exporting model='{spec.model_name}' input=({args.batch},3,{h},{w}) -> output_shape={shape_msg}")

    torch.onnx.export(
        export_model,
        (dummy,),
        args.output,
        export_params=True,
        opset_version=int(args.opset),
        do_constant_folding=True,
        input_names=["image"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    print(f"Wrote ONNX to: {args.output}")


if __name__ == "__main__":
    main()
