"""Evaluate a trained checkpoint on YAML-configured test datasets.

Example:
    python scripts/test.py \
        --config configs/xreal_mobilenet.yaml \
        --checkpoint work_dirs/xreal_mobilenet/best.pth

The config must contain ``data.test_datasets``. Its entries use the same schema
as ``data.val_datasets`` and should point at each dataset's ``test.json``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from pose.config import Config
from pose.engine.ema import ema_shadow_issues


def _deterministic_test_aug(data_cfg: dict[str, Any]) -> dict[str, Any]:
    """Keep configured crop geometry while disabling random augmentation."""
    configured = data_cfg.get("test_aug")
    if configured is None:
        configured = data_cfg.get("val_aug")
    if configured is None:
        configured = data_cfg.get("aug", {})
    result = copy.deepcopy(configured) if isinstance(configured, dict) else {}
    result["enabled"] = False
    result["bbox_offset_p"] = 0.0
    result["hflip_p"] = 0.0
    result["aug_factor"] = 1
    return result


def _evaluation_config(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    """Adapt test datasets to Trainer's generic validation implementation."""
    cfg = copy.deepcopy(raw_cfg)
    data_cfg = cfg.get("data")
    if not isinstance(data_cfg, dict):
        raise KeyError("Config is missing required section: data")

    test_items = data_cfg.get("test_datasets")
    if not isinstance(test_items, list) or not test_items:
        raise KeyError(
            "Config is missing data.test_datasets. Add entries with name, "
            "json_path (pointing to test.json), image_root, and optional weight."
        )

    for index, item in enumerate(test_items):
        if not isinstance(item, dict):
            raise ValueError(f"data.test_datasets[{index}] must be a mapping")
        json_path = item.get("json_path")
        if not isinstance(json_path, str) or not json_path:
            raise ValueError(f"data.test_datasets[{index}].json_path is required")
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Test annotations not found: {json_path}")

    # Trainer owns the project's model/loss/dataset branching. Point its
    # validation side at the test split and avoid requiring train.json to exist.
    data_cfg["train_datasets"] = copy.deepcopy(test_items)
    data_cfg["val_datasets"] = copy.deepcopy(test_items)
    data_cfg["primary_val"] = data_cfg.get("primary_test", test_items[0].get("name", "test0"))
    data_cfg["val_aug"] = _deterministic_test_aug(data_cfg)
    data_cfg["aug"] = copy.deepcopy(data_cfg["val_aug"])

    # A complete checkpoint replaces the backbone, so evaluation should never
    # need a network download for ImageNet initialization.
    model_cfg = cfg.get("model")
    if isinstance(model_cfg, dict):
        model_cfg["pretrained"] = False

    cfg.pop("_config_path", None)
    return cfg


def _checkpoint_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint: expected a dictionary")
    for key in ("model", "state_dict", "model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint
    raise ValueError(
        "Unsupported checkpoint: expected a raw state_dict or a 'model', "
        "'state_dict', or 'model_state_dict' entry"
    )


def _without_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state and all(key.startswith("module.") for key in state):
        return {key[len("module."):]: value for key, value in state.items()}
    return state


def _load_weights(trainer, checkpoint_path: str, ema_mode: str, strict: bool) -> str:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = _without_module_prefix(_checkpoint_state(checkpoint))
    incompatible = trainer.model.load_state_dict(state, strict=strict)
    if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        print(
            f"[WARN] Base weights: missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )

    ema_shadow = None
    if isinstance(checkpoint, dict):
        ema = checkpoint.get("ema")
        if isinstance(ema, dict) and isinstance(ema.get("shadow"), dict):
            ema_shadow = _without_module_prefix(ema["shadow"])

    configured = bool(
        ((trainer.cfg.get("train", {}) or {}).get("ema", {}) or {}).get("eval", True)
    )
    use_ema = ema_mode == "on" or (ema_mode == "auto" and configured and ema_shadow is not None)
    if ema_mode == "on" and ema_shadow is None:
        raise ValueError("--ema on was requested, but the checkpoint has no EMA weights")
    if use_ema and ema_shadow is not None:
        missing, unexpected, incompatible = ema_shadow_issues(trainer.model, ema_shadow)
        if missing or unexpected or incompatible:
            message = (
                "EMA state is incomplete "
                f"(missing={len(missing)}, unexpected={len(unexpected)}, "
                f"incompatible={len(incompatible)})"
            )
            if ema_mode == "on":
                raise ValueError(f"{message}; refusing forced EMA evaluation")
            print(f"[WARN] {message}; falling back to raw model weights")
            use_ema = False
    if use_ema and ema_shadow is not None:
        trainer.model.load_state_dict(ema_shadow, strict=True)
        weight_source = "ema"
    else:
        weight_source = "model"

    # EMA is already applied directly above; prevent _validate_one from swapping
    # in the Trainer's freshly initialized EMA shadow.
    trainer.ema = None
    trainer.model.eval()
    return weight_source


def _clean_metrics(metrics: dict[str, float]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key, value in metrics.items():
        if key.startswith("_"):
            continue
        number = float(value)
        result[key] = number if math.isfinite(number) else None
    return result


def _evaluate(trainer) -> tuple[dict[str, dict[str, float | None]], dict[str, float | None]]:
    per_dataset: dict[str, dict[str, float | None]] = {}
    raw_results: dict[str, dict[str, float]] = {}
    for name, loader in trainer.val_loaders.items():
        raw = trainer._validate_one(0, name, loader, trainer.val_datasets[name])
        raw_results[name] = raw
        per_dataset[name] = _clean_metrics(raw)

    aggregate: dict[str, float | None] = {}
    for metric in ("loss", "pck", "nme", "iou"):
        weighted_sum = 0.0
        total_weight = 0.0
        for name, raw in raw_results.items():
            value = raw.get(metric)
            if value is None or not math.isfinite(float(value)):
                continue
            dataset_weight = max(0.0, float(trainer.val_weights.get(name, 1.0)))
            count = float(len(trainer.val_datasets[name]))
            weight = dataset_weight * count
            weighted_sum += weight * float(value)
            total_weight += weight
        if total_weight > 0:
            aggregate[metric] = weighted_sum / total_weight
    return per_dataset, aggregate


def _print_results(per_dataset, aggregate, pck_threshold: float) -> None:
    print(f"\nTest metrics (PCK@{pck_threshold:g} of image diagonal)")
    for name, metrics in per_dataset.items():
        values = ", ".join(
            f"{key.upper()}={value:.6f}" if value is not None else f"{key.upper()}=n/a"
            for key, value in metrics.items()
        )
        print(f"  {name}: {values}")
    if len(per_dataset) > 1 and aggregate:
        values = ", ".join(f"{key.upper()}={value:.6f}" for key, value in aggregate.items())
        print(f"  aggregate: {values}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config containing data.test_datasets")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint (.pth)")
    parser.add_argument("--device", default="cuda", help="cuda, cpu, or another torch device")
    parser.add_argument(
        "--ema",
        choices=("auto", "on", "off"),
        default="auto",
        help="EMA weights: auto follows train.ema.eval (default)",
    )
    parser.add_argument("--strict", action="store_true", help="Require an exact model state_dict match")
    parser.add_argument("--output", help="Optional path for the metrics JSON")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    raw_cfg = Config.from_yaml(args.config).raw
    eval_cfg = _evaluation_config(raw_cfg)

    # Import after argument/config validation so --help stays lightweight.
    from pose.engine.trainer import Trainer

    trainer = Trainer(eval_cfg, device=args.device)
    weight_source = _load_weights(trainer, args.checkpoint, args.ema, args.strict)
    per_dataset, aggregate = _evaluate(trainer)

    metric_cfg = eval_cfg.get("data", {}).get("metrics", {}) or {}
    pck_threshold = float(metric_cfg.get("pck_threshold", 0.05))
    nme_indices = list(metric_cfg.get("nme_interocular_indices", [36, 45]))
    _print_results(per_dataset, aggregate, pck_threshold)

    report = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "device": str(trainer.device),
        "weights": weight_source,
        "pck_threshold": pck_threshold,
        "nme_interocular_indices": nme_indices,
        "nme_normalization": (
            f"interocular keypoints {nme_indices[0]} and {nme_indices[1]}; "
            "visible bbox diagonal fallback"
        ),
        "datasets": per_dataset,
        "aggregate": aggregate,
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
