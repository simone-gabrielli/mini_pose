"""Training engine.

`Trainer` is the central orchestrator of mini-pose training. It wires together:
    - dataset(s) via `DATASET_REGISTRY`
    - model via `MODEL_REGISTRY`
    - loss via `LOSS_REGISTRY`
    - optimizer/scheduler/regularization (AMP, EMA, grad clipping)

The design goal is to keep the training loop *model-agnostic* while supporting
multiple task styles:
    - heatmap keypoints (common)
    - coordinate regression (LOTR-style)
    - bbox-only detector heads
    - pose reprojection losses (e.g. projecting a known 3D CAD model)

If you are trying to understand the project end-to-end, start at:
    - `scripts/train.py` (CLI)
    - `Trainer.__init__` (builds datasets/model/loss)
    - `Trainer.train_epoch` (forward/loss branches)
    - `Trainer.validate_epoch` (metrics + checkpoint selection)
"""

import os
import shutil
from contextlib import contextmanager, nullcontext
from typing import Dict, Any
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


class ModelEMA:
    """Exponential Moving Average (EMA) of model parameters.

    Keeps a shadow copy of parameters updated as:
        ema = decay * ema + (1 - decay) * param

    Designed for easy evaluation via a temporary weight swap.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999, device: torch.device | None = None):
        self.decay = float(decay)
        self._shadow: dict[str, torch.Tensor] = {}
        self._device = device

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if not torch.is_floating_point(p.data):
                continue
            t = p.detach().clone()
            if device is not None:
                t = t.to(device=device)
            self._shadow[name] = t

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        d = self.decay
        for name, p in model.named_parameters():
            if name not in self._shadow:
                continue
            if not torch.is_floating_point(p.data):
                continue
            src = p.detach()
            if self._device is not None:
                src = src.to(device=self._device)
            self._shadow[name].mul_(d).add_(src, alpha=(1.0 - d))

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "shadow": {k: v.detach().cpu() for k, v in self._shadow.items()},
        }

    def load_state_dict(self, state: dict):
        if not isinstance(state, dict):
            return
        if "decay" in state:
            self.decay = float(state["decay"])
        shadow = state.get("shadow")
        if not isinstance(shadow, dict):
            return
        for k, v in shadow.items():
            if k not in self._shadow:
                continue
            try:
                t = v
                if not isinstance(t, torch.Tensor):
                    continue
                if self._device is not None:
                    t = t.to(device=self._device)
                else:
                    # keep on current model device if available
                    t = t.to(device=self._shadow[k].device)
                self._shadow[k].copy_(t)
            except Exception:
                continue

    @contextmanager
    def apply_to(self, model: torch.nn.Module):
        """Temporarily copy EMA weights into model parameters."""
        backup: dict[str, torch.Tensor] = {}
        try:
            for name, p in model.named_parameters():
                if name not in self._shadow:
                    continue
                if not torch.is_floating_point(p.data):
                    continue
                backup[name] = p.detach().clone()
                src = self._shadow[name]
                # If EMA is stored on CPU, copy to param device.
                if src.device != p.device:
                    src = src.to(device=p.device)
                p.data.copy_(src)
            yield
        finally:
            for name, p in model.named_parameters():
                if name not in backup:
                    continue
                p.data.copy_(backup[name].to(device=p.device))

import pose.data      # noqa: F401  # ensures datasets register
import pose.models    # noqa: F401  # ensures models register
import pose.detectors # noqa: F401  # ensure detectors register (e.g. tiny_face)
import pose.losses    # noqa: F401  # ensures losses register
import inspect
from pose.registry import MODEL_REGISTRY, LOSS_REGISTRY, DATASET_REGISTRY
from pose.engine.metrics import compute_pck, compute_nme
from pose.data import DatasetSpec, WeightedConcatDataset

class Trainer:
    """Train/evaluate a model defined by a YAML config.

        Expected config sections:
            - cfg['data']  : dataset type + paths + preprocessing sizes
            - cfg['model'] : model registry key + constructor kwargs
            - cfg['loss']  : loss registry key + constructor kwargs
            - cfg['train'] : epochs/lr/optimizer/scheduler/output_dir + optional AMP/EMA

        Expected dataset batch format (dict):
            - 'image'   : (B,3,H,W) float tensor, ImageNet-normalized
            - 'visible' : (B,K) visibility mask for keypoints (when applicable)
            - optional task-specific keys:
                - 'heatmaps' : supervision targets for heatmap models
                - 'keypoints': (B,K,3) with (x,y,vis/score) in *input crop pixel space*
                - 'bbox' (bbox detector)
                - 'pose_weights' (reprojection loss)
                - 'dataset_weight' (for multi-dataset weighted sampling)
    """

    def __init__(self, cfg: Dict[str, Any], device: str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Dataset
        ds_cfg = cfg["data"]
        DatasetCls = DATASET_REGISTRY[ds_cfg["type"]]
        if "input_size" not in ds_cfg:
            raise KeyError("Config is missing required key: data.input_size")

        # Only pass dataset kwargs that are explicitly specified.
        base_kwargs: dict[str, Any] = {
            "input_size": tuple(ds_cfg["input_size"]),
        }

        # Optional: only for models that use heatmaps
        if "heatmap_size" in ds_cfg:
            base_kwargs["heatmap_size"] = tuple(ds_cfg["heatmap_size"])

        # Optional: forward sigma from config into dataset so GT heatmaps
        # are generated at the requested spatial resolution with matching sigma.
        if "sigma" in ds_cfg:
            base_kwargs["sigma"] = ds_cfg["sigma"]


        # Multi-dataset training (optional)
        #
        # Backward-compatible (single dataset):
        #   data: { train_json, val_json, image_root, ... }
        # Multi-dataset (train only):
        #   data:
        #     train_datasets:
        #       - name: hmd_xreal
        #         train_json: ...
        #         image_root: ...
        #         loss_weight: 1.0
        #       - name: air2
        #         train_json: ...
        #         image_root: ...
        #         loss_weight: 0.5
        #     val_json: ...
        #     val_image_root: ...
        train_specs = []
        if isinstance(ds_cfg.get("train_datasets"), list) and len(ds_cfg.get("train_datasets")) > 0:
            for item in ds_cfg["train_datasets"]:
                if not isinstance(item, dict):
                    raise ValueError("Each entry in data.train_datasets must be a dict")

                train_json = item.get("train_json") or item.get("json_path")
                if not isinstance(train_json, str):
                    raise ValueError("Each train dataset must provide train_json (or json_path) as a string")

                image_root = item.get("image_root")
                if not isinstance(image_root, str):
                    raise ValueError("Each train dataset must provide image_root as a string")

                aug_cfg = item.get("aug", ds_cfg.get("aug", None))

                # Per-dataset overrides (optional). This lets you omit keys at the
                # top-level and only specify them where they matter.
                ds_kwargs = dict(base_kwargs)
                ds_kwargs["input_size"] = tuple(item["input_size"]) # Required
                if "heatmap_size" in item:
                    ds_kwargs["heatmap_size"] = tuple(item["heatmap_size"]) # Optional
                if "sigma" in item:
                    ds_kwargs["sigma"] = item["sigma"] # Optional

                ds = DatasetCls(json_path=train_json, image_root=image_root, aug_cfg=aug_cfg, **ds_kwargs)

                loss_weight = float(item.get("loss_weight", 1.0))
                name = item.get("name")
                train_specs.append(DatasetSpec(dataset=ds, loss_weight=loss_weight, name=name))

            self.train_ds = WeightedConcatDataset(train_specs)
        else:
            # single dataset
            self.train_ds = DatasetCls(
                json_path=ds_cfg["train_json"],
                image_root=ds_cfg["image_root"],
                aug_cfg=ds_cfg.get("aug", None),
                **base_kwargs,
            )

        # Validation dataset(s)
        #
        # Backward-compatible (single val dataset):
        #   data: { val_json, val_image_root?, ... }
        # Multi-dataset validation:
        #   data:
        #     val_datasets:
        #       - name: air2
        #         val_json: ...
        #         image_root: ...
        #         weight: 1.0
        #       - name: hmd
        #         val_json: ...
        #         image_root: ...
        #         weight: 1.0
        #     primary_val: air2
        self.val_loaders: dict[str, DataLoader] = {}
        self.val_datasets: dict[str, Any] = {}
        self.val_weights: dict[str, float] = {}
        self.primary_val_name: str = "val"

        if isinstance(ds_cfg.get("val_datasets"), list) and len(ds_cfg.get("val_datasets")) > 0:
            primary = ds_cfg.get("primary_val")
            for i, item in enumerate(ds_cfg["val_datasets"]):
                if not isinstance(item, dict):
                    raise ValueError("Each entry in data.val_datasets must be a dict")

                name = item.get("name")
                if not isinstance(name, str) or not name:
                    name = f"val{i}"

                val_json = item.get("val_json") or item.get("json_path")
                if not isinstance(val_json, str):
                    raise ValueError("Each val dataset must provide val_json (or json_path) as a string")

                image_root = item.get("val_image_root") or item.get("image_root")
                if not isinstance(image_root, str):
                    raise ValueError("Each val dataset must provide image_root (or val_image_root) as a string")

                aug_cfg = item.get("val_aug", item.get("aug", ds_cfg.get("val_aug", ds_cfg.get("aug", None))))

                ds_kwargs = dict(base_kwargs)
                if "input_size" in item:
                    ds_kwargs["input_size"] = tuple(item["input_size"])
                if "heatmap_size" in item:
                    ds_kwargs["heatmap_size"] = tuple(item["heatmap_size"])
                if "sigma" in item:
                    ds_kwargs["sigma"] = item["sigma"]

                ds = DatasetCls(json_path=val_json, image_root=image_root, aug_cfg=aug_cfg, **ds_kwargs)

                # allow either `weight` or `loss_weight` for convenience
                w = item.get("weight", item.get("loss_weight", 1.0))
                w = float(w)
                if w < 0:
                    raise ValueError("Validation dataset weight must be >= 0")

                bs = int(item.get("batch_size", ds_cfg.get("val_batch_size", ds_cfg["batch_size"])))
                if bs < 1:
                    raise ValueError("Validation batch_size must be >= 1")

                num_workers = ds_cfg.get("num_workers", 0)
                loader = DataLoader(
                    ds,
                    batch_size=bs,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=(num_workers > 0),
                )

                self.val_datasets[name] = ds
                self.val_loaders[name] = loader
                self.val_weights[name] = w

            if isinstance(primary, str) and primary in self.val_loaders:
                self.primary_val_name = primary
            else:
                # default to the first entry
                self.primary_val_name = next(iter(self.val_loaders.keys()))

            # Keep legacy attributes pointing at the primary val set.
            self.val_ds = self.val_datasets[self.primary_val_name]
            self.val_loader = self.val_loaders[self.primary_val_name]
        else:
            # single dataset (current behavior)
            val_json = ds_cfg["val_json"]
            val_image_root = ds_cfg.get("val_image_root")
            if val_image_root is None:
                # fall back to legacy key, or the first train dataset's root
                val_image_root = ds_cfg.get("image_root")
                if val_image_root is None and train_specs:
                    val_image_root = getattr(train_specs[0].dataset, "image_root", None)
            if not isinstance(val_image_root, str):
                raise ValueError(
                    "Validation requires data.val_image_root (or data.image_root for single-dataset configs)."
                )

            self.val_ds = DatasetCls(
                json_path=val_json,
                image_root=val_image_root,
                aug_cfg=ds_cfg.get("val_aug", ds_cfg.get("aug", None)),
                **base_kwargs,
            )

            num_workers = ds_cfg.get("num_workers", 0)
            self.val_loader = DataLoader(
                self.val_ds,
                batch_size=int(ds_cfg.get("val_batch_size", ds_cfg["batch_size"])),
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=(num_workers > 0),
            )
            self.val_datasets = {"val": self.val_ds}
            self.val_loaders = {"val": self.val_loader}
            self.val_weights = {"val": 1.0}
            self.primary_val_name = "val"

        num_workers = ds_cfg.get("num_workers", 0)
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=ds_cfg["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
        # self.val_loader is created above (single or multi-dataset)

        num_keypoints = self.train_ds.num_keypoints

        # Model: fully generic wiring.
        # - `name` selects the model class from the registry.
        # - all other keys are passed verbatim as kwargs, plus
        #   `num_keypoints` injected by the trainer.
        model_cfg = cfg["model"]
        ModelCls = MODEL_REGISTRY[model_cfg["name"]]

        model_kwargs = {"num_keypoints": num_keypoints}
        for k, v in model_cfg.items():
            if k == "name":
                continue
            model_kwargs[k] = v

        # Only pass kwargs that the model constructor actually accepts.
        # Some registered classes (e.g. detector heads) do not accept
        # `num_keypoints` or other pose-specific kwargs; guard against
        # passing unexpected arguments by inspecting the __init__
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

        self.model = ModelCls(**kwargs_to_pass).to(self.device)

        # Loss
        loss_cfg = cfg.get("loss", {})
        if not isinstance(loss_cfg, dict) or "name" not in loss_cfg:
            raise KeyError("Config is missing required key: loss.name")

        LossCls = LOSS_REGISTRY[loss_cfg["name"]]
        # Pass loss_cfg kwargs through (excluding 'name') so losses can be configured.
        loss_kwargs = {k: v for k, v in loss_cfg.items() if k != "name"}
        self.criterion = LossCls(**loss_kwargs).to(self.device)


        # Optimizer
        train_cfg = cfg["train"]
        opt_name = str(train_cfg.get("optimizer", "adam")).lower()
        lr = float(train_cfg["lr"])
        weight_decay = float(train_cfg.get("weight_decay", 0.0))
        betas = tuple(train_cfg.get("betas", [0.9, 0.999]))

        if opt_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
            )
        elif opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        # Scheduler (backward compatible default: MultiStepLR)
        sched_name = str(train_cfg.get("lr_schedule", "multistep")).lower()
        if sched_name == "multistep":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=train_cfg.get("lr_steps", [30, 50]),
                gamma=train_cfg.get("lr_gamma", 0.1),
            )
        elif sched_name in ("cosine_warmup", "cosine", "cosine_with_warmup"):
            warmup_epochs = int(train_cfg.get("warmup_epochs", 5))
            min_lr = float(train_cfg.get("min_lr", 0.0))
            warmup_start_factor = float(train_cfg.get("warmup_start_factor", 0.1))
            if warmup_epochs < 0:
                warmup_epochs = 0
            if warmup_start_factor <= 0:
                warmup_start_factor = 1e-4

            total_epochs = int(train_cfg["epochs"])
            cosine_epochs = max(1, total_epochs - warmup_epochs)

            if warmup_epochs > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=warmup_start_factor,
                    total_iters=warmup_epochs,
                )
                cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=cosine_epochs,
                    eta_min=min_lr,
                )
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[warmup_epochs],
                )
            else:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=cosine_epochs,
                    eta_min=min_lr,
                )
        else:
            raise ValueError(f"Unsupported lr_schedule: {sched_name}")

        # Regularization bundle flags
        self.grad_clip_norm = float(train_cfg.get("grad_clip_norm", 0.0))
        self.grad_clip_norm = max(0.0, self.grad_clip_norm)

        amp_cfg = train_cfg.get("amp", {}) or {}
        self.use_amp = bool(amp_cfg.get("enabled", False)) and self.device.type == "cuda"
        amp_dtype = str(amp_cfg.get("dtype", "fp16")).lower()
        if amp_dtype in ("bf16", "bfloat16"):
            self.amp_dtype = torch.bfloat16
            # GradScaler is typically not needed for bf16; still safe to keep enabled=False.
            self.scaler = torch.amp.GradScaler(device='cuda', enabled=False)
        else:
            self.amp_dtype = torch.float16
            self.scaler = torch.amp.GradScaler(device='cuda', enabled=self.use_amp)

        ema_cfg = train_cfg.get("ema", {}) or {}
        self.use_ema = bool(ema_cfg.get("enabled", False))
        self.ema_eval = bool(ema_cfg.get("eval", True))
        self.ema_decay = float(ema_cfg.get("decay", 0.999))
        self.ema_device = ema_cfg.get("device", None)
        if isinstance(self.ema_device, str):
            self.ema_device = torch.device(self.ema_device)
        else:
            self.ema_device = None
        self.ema_update_every = int(ema_cfg.get("update_every", 1))
        if self.ema_update_every < 1:
            self.ema_update_every = 1

        self.ema = ModelEMA(self.model, decay=self.ema_decay, device=self.ema_device) if self.use_ema else None

        self.epochs = train_cfg["epochs"]
        self.output_dir = train_cfg.get("output_dir", "work_dirs")
        os.makedirs(self.output_dir, exist_ok=True)

        # Optional: persist the config used for this run into output_dir
        # If the original config path is provided in cfg["_config_path"], copy it
        cfg_path = cfg.get("_config_path")
        if isinstance(cfg_path, str) and os.path.isfile(cfg_path):
            try:
                shutil.copy2(cfg_path, os.path.join(self.output_dir, os.path.basename(cfg_path)))
            except Exception:
                # best-effort; ignore failures so training is not blocked
                pass

        # generic history: split -> metric_name -> list[float]
        self.history = {"train": {}, "val": {}}
        self.start_epoch = 1  # default

    def _visualize_model_outputs(self, sample, out_path: str):
        # Runs forward pass and saves overlay images with predicted heatmaps
        self.model.eval()
        imgs = sample["image"].unsqueeze(0).to(self.device)
        keypts_gt = sample["keypoints"].cpu().numpy()

        with torch.no_grad():
            out = self.model(imgs)

        # prefer the model's primary output (out[0]) if present (2D heatmaps),
        # otherwise fall back to preds_all[-1]
        if isinstance(out, tuple):
            if isinstance(out[0], torch.Tensor):
                preds_last = out[0]
            else:
                preds_last = out[1][-1]
        else:
            preds_last = out

        hm = preds_last[0].cpu().numpy()  # shape (K, H_hm, W_hm)
        img_np = sample["image"].cpu().numpy().transpose(1,2,0)

        # Undo dataset normalization (ImageNet mean/std used in pipeline)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        try:
            img_np = (img_np * std) + mean
        except Exception:
            pass

        H_img, W_img = img_np.shape[0], img_np.shape[1]
        H_hm, W_hm = hm.shape[1], hm.shape[2]
        scale_x = W_img / W_hm
        scale_y = H_img / H_hm

        plt.figure(figsize=(4,4))
        plt.imshow(img_np)
        for k in range(hm.shape[0]):
            hm_k = hm[k]
            hm_k = hm_k / hm_k.max() if hm_k.max() > 0 else hm_k
            hm_resized = plt.cm.jet(hm_k)
            plt.imshow(hm_resized[..., :3], alpha=0.3)
        xs = keypts_gt[:,0]
        ys = keypts_gt[:,1]
        plt.scatter(xs, ys, c='lime', s=5, label='gt')
        # Predicted points:
        coords_pred = []
        for k in range(hm.shape[0]):
            idx_flat = hm[k].reshape(-1).argmax()
            y = idx_flat // W_hm
            x = idx_flat % W_hm
            coords_pred.append((x*scale_x, y*scale_y))
        coords_pred = np.array(coords_pred)
        plt.scatter(coords_pred[:,0], coords_pred[:,1], c='red', s=5, label='pred')
        plt.legend(loc='upper right', fontsize=6)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

    def log_metric(self, split: str, name: str, value: float):
        """Log any scalar metric for train/val in a generic way."""
        if split not in self.history:
            self.history[split] = {}
        if name not in self.history[split]:
            self.history[split][name] = []
        self.history[split][name].append(float(value))

    def train_epoch(self, epoch: int):
        """Run one training epoch.

        The forward/loss path depends on both the *loss name* and the presence of
        specific batch keys:
          - if batch contains 'bbox' -> bbox detector path
          - elif loss == 'pose_reprojection' -> dict output with 'proj'
          - elif loss in coord_regression_losses -> direct coordinate regression
          - else -> heatmap training (2D or 3D) with (possibly) intermediate supervision

        Returns:
            Mean training loss over the epoch.
        """
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [train]")
        running_loss = 0.0
        step_idx = 0
        for batch in pbar:
            imgs = batch["image"].to(self.device)
            sample_weight = batch.get("dataset_weight")
            if sample_weight is not None:
                sample_weight = sample_weight.to(self.device)

            loss_name = self.cfg.get("loss", {}).get("name", "")

            # Coordinate regression losses (LOTR-style direct coordinate prediction)
            coord_regression_losses = [
                "smooth_wing", "wing", "adaptive_wing", 
                "landmark_l1", "landmark_mse", "landmark_smooth_l1",
                "nme", "combined_landmark"
            ]

            self.optimizer.zero_grad(set_to_none=True)

            # Mixed precision: wrap forward+loss; choose dtype.
            autocast_ctx = (
                torch.amp.autocast(self.device.type, dtype=self.amp_dtype)
                if self.use_amp and self.device.type == "cuda"
                else nullcontext()
            )

            with autocast_ctx:
                # Branch: bbox detector (single box + confidence)
                if "bbox" in batch:
                    targets_bbox = batch["bbox"].to(self.device)
                    preds_bbox = self.model(imgs)
                    loss = self.criterion(preds_bbox, targets_bbox, sample_weight=sample_weight)

                # Branch: direct pose regression with reprojection loss
                elif loss_name == "pose_reprojection":
                    out = self.model(imgs)
                    if isinstance(out, dict):
                        preds_2d = out.get("proj")
                    else:
                        preds_2d = out

                    if preds_2d is None:
                        raise RuntimeError("Model must return projected 2D points 'proj' for pose_reprojection loss")

                    targets_2d = batch["keypoints"][:, :, :2].to(self.device)
                    weights = batch.get("pose_weights")
                    if weights is not None:
                        weights = weights.to(self.device)

                    loss = self.criterion(preds_2d, targets_2d, weights, sample_weight=sample_weight)

                # Branch: coordinate regression losses (LOTR, etc.)
                elif loss_name in coord_regression_losses:
                    targets_2d = batch["keypoints"][:, :, :2].to(self.device)
                    visible = batch["visible"].to(self.device)

                    out = self.model(imgs)

                    # LOTR returns (normalized_coords, pixel_coords)
                    if isinstance(out, tuple) and len(out) == 2:
                        _, landmarks_pixel = out
                        # Use pixel coordinates for loss computation
                        preds_2d = landmarks_pixel[..., :2]  # (B, N, 2)
                    else:
                        preds_2d = out
                        if preds_2d.dim() == 2:
                            # If flat output, reshape to (B, N, 2)
                            B = imgs.size(0)
                            preds_2d = preds_2d.view(B, -1, 2)

                    loss = self.criterion(preds_2d, targets_2d, visible, sample_weight=sample_weight)

                else:
                    # Heatmap-based training
                    targets = batch["heatmaps"].to(self.device)
                    visible = batch["visible"].to(self.device)
                    out = self.model(imgs)
                    if isinstance(out, tuple) and len(out) == 2:
                        preds_last, preds_all = out
                    else:
                        preds_last = out
                        preds_all = [out]

                    loss = 0.0
                    if isinstance(preds_all, (list, tuple)):
                        for p in preds_all:
                            loss += self.criterion(p, targets, visible, sample_weight=sample_weight)
                    else:
                        loss = self.criterion(preds_all, targets, visible, sample_weight=sample_weight)

            # Backward + step
            if self.use_amp and self.device.type == "cuda" and self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                if self.grad_clip_norm > 0:
                    # unscale before clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                self.optimizer.step()

            # EMA update
            if self.ema is not None and (step_idx % self.ema_update_every == 0):
                self.ema.update(self.model)
            step_idx += 1

            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(self.train_loader)
        self.log_metric("train", "loss", epoch_loss)
        return epoch_loss

    @torch.no_grad()
    def _validate_one(self, epoch: int, name: str, loader: DataLoader, dataset) -> dict[str, float]:
        """Validate on a single loader and return scalar metrics."""
        self.model.eval()
        pbar = tqdm(loader, desc=f"Epoch {epoch} [val:{name}]")
        running_loss = 0.0
        loss_name = self.cfg.get("loss", {}).get("name", "")

        coord_regression_losses = [
            "smooth_wing", "wing", "adaptive_wing",
            "landmark_l1", "landmark_mse", "landmark_smooth_l1",
            "nme", "combined_landmark",
        ]

        coord_preds: list[np.ndarray] = []
        coord_targets: list[np.ndarray] = []
        is_coord_regression = loss_name in coord_regression_losses

        det_iou_sum = 0.0
        det_iou_n = 0

        # Optionally evaluate EMA weights.
        eval_ctx = self.ema.apply_to(self.model) if (self.ema is not None and self.ema_eval) else nullcontext()

        with eval_ctx:
            for batch in pbar:
                imgs = batch["image"].to(self.device)
                sample_weight = batch.get("dataset_weight")
                if sample_weight is not None:
                    sample_weight = sample_weight.to(self.device)

                # Branch: bbox detector
                if "bbox" in batch:
                    targets_bbox = batch["bbox"].to(self.device)
                    preds_bbox = self.model(imgs)
                    loss = self.criterion(preds_bbox, targets_bbox, sample_weight=sample_weight)
                    preds_last = None

                    # Basic detector metrics (only for positives)
                    try:
                        conf_t = targets_bbox[:, 0]
                        pos = conf_t > 0.5
                        if pos.any():
                            # IoU between predicted and gt bboxes in normalized coords
                            pb = torch.sigmoid(preds_bbox[:, 1:5])
                            gb = targets_bbox[:, 1:5].clamp(0.0, 1.0)
                            pb = pb[pos]
                            gb = gb[pos]

                            ix1 = torch.max(pb[:, 0], gb[:, 0])
                            iy1 = torch.max(pb[:, 1], gb[:, 1])
                            ix2 = torch.min(pb[:, 2], gb[:, 2])
                            iy2 = torch.min(pb[:, 3], gb[:, 3])

                            iw = (ix2 - ix1).clamp(min=0.0)
                            ih = (iy2 - iy1).clamp(min=0.0)
                            inter = iw * ih
                            a_p = (pb[:, 2] - pb[:, 0]).clamp(min=0.0) * (pb[:, 3] - pb[:, 1]).clamp(min=0.0)
                            a_g = (gb[:, 2] - gb[:, 0]).clamp(min=0.0) * (gb[:, 3] - gb[:, 1]).clamp(min=0.0)
                            iou = inter / (a_p + a_g - inter).clamp(min=1e-6)
                            det_iou_sum += float(iou.mean().item())
                            det_iou_n += 1
                    except Exception:
                        pass

                elif loss_name == "pose_reprojection":
                    out = self.model(imgs)
                    if isinstance(out, dict):
                        preds_2d = out.get("proj")
                    else:
                        preds_2d = out

                    if preds_2d is None:
                        raise RuntimeError("Model must return projected 2D points 'proj' for pose_reprojection loss")

                    targets_2d = batch["keypoints"][:, :, :2].to(self.device)
                    weights = batch.get("pose_weights")
                    if weights is not None:
                        weights = weights.to(self.device)

                    loss = self.criterion(preds_2d, targets_2d, weights, sample_weight=sample_weight)
                    preds_last = None

                elif is_coord_regression:
                    targets_2d = batch["keypoints"][:, :, :2].to(self.device)
                    visible = batch["visible"].to(self.device)
                    out = self.model(imgs)

                    if isinstance(out, tuple) and len(out) == 2:
                        _, landmarks_pixel = out
                        preds_2d = landmarks_pixel[..., :2]
                    else:
                        preds_2d = out
                        if preds_2d.dim() == 2:
                            B = imgs.size(0)
                            preds_2d = preds_2d.view(B, -1, 2)

                    loss = self.criterion(preds_2d, targets_2d, visible, sample_weight=sample_weight)
                    preds_last = None

                    for b in range(preds_2d.size(0)):
                        coord_preds.append(preds_2d[b].cpu().numpy())
                        coord_targets.append(targets_2d[b].cpu().numpy())

                else:
                    targets = batch["heatmaps"].to(self.device)

                    visible = batch["visible"].to(self.device)
                    out = self.model(imgs)
                    if isinstance(out, tuple) and len(out) == 2:
                        preds_last, preds_all = out
                    else:
                        preds_last = out
                        preds_all = [out]

                    loss = 0.0
                    if isinstance(preds_all, (list, tuple)):
                        for p in preds_all:
                            loss += self.criterion(p, targets, visible, sample_weight=sample_weight)
                    else:
                        loss = self.criterion(preds_all, targets, visible, sample_weight=sample_weight)

                if (
                    preds_last is not None
                    and isinstance(preds_last, torch.Tensor)
                    and preds_last.dim() == 4
                    and hasattr(dataset, "heatmap_size")
                    and dataset.heatmap_size is not None
                ):
                    # image and heatmap sizes from the *validation* dataset
                    H_img, W_img = dataset.input_size[1], dataset.input_size[0]
                    H_hm, W_hm = dataset.heatmap_size[1], dataset.heatmap_size[0]
                    sx = W_hm / float(W_img)
                    sy = H_hm / float(H_img)

                    for b in range(preds_last.size(0)):
                        hm = preds_last[b].cpu()
                        K, H, W = hm.shape
                        h_flat = hm.view(K, -1)
                        idx = torch.argmax(h_flat, dim=1)
                        y = (idx // W).float().numpy()
                        x = (idx % W).float().numpy()
                        coord_preds.append(np.stack([x, y], axis=1))

                        t = batch["keypoints"][b].cpu().numpy()[:, :2]
                        coord_targets.append(np.stack([t[:, 0] * sx, t[:, 1] * sy], axis=1))

                running_loss += float(loss.item())
                pbar.set_postfix(loss=float(loss.item()))

        num_batches = int(len(loader))
        val_loss = running_loss / max(1, num_batches)

        # Expose both mean loss and the raw sums so the caller can aggregate
        # across datasets without biasing toward small/large val sets.
        metrics: dict[str, float] = {
            "loss": float(val_loss),
            "_loss_sum": float(running_loss),
            "_num_batches": float(num_batches),
        }
        if det_iou_n > 0:
            metrics["iou"] = float(det_iou_sum / float(det_iou_n))
        if coord_preds:
            try:
                coord_preds_arr = np.stack(coord_preds)
                coord_targets_arr = np.stack(coord_targets)
                metrics["pck"] = float(compute_pck(coord_preds_arr, coord_targets_arr))
                metrics["nme"] = float(compute_nme(coord_preds_arr, coord_targets_arr))
            except Exception:
                pass
        return metrics

    @torch.no_grad()
    def validate_epoch(self, epoch: int):
        """Validate on the configured val dataset(s) and return aggregate loss.

        - If multiple val datasets are configured, we evaluate all and compute a
          batch-count-aware weighted average loss for checkpoint selection.
        - PCK/NME are reported for the *primary* val dataset when available.

        Returns:
            Aggregate validation loss (lower is better).
        """
        # Validate each configured val dataset, then aggregate into a single scalar
        # for checkpoint selection.
        per_ds: dict[str, dict[str, float]] = {}

        for name, loader in self.val_loaders.items():
            ds = self.val_datasets[name]
            per_ds[name] = self._validate_one(epoch=epoch, name=name, loader=loader, dataset=ds)

            # Log per-dataset metrics with stable names (no path separators).
            for k, v in per_ds[name].items():
                if str(k).startswith("_"):
                    continue
                self.log_metric("val", f"{k}_{name}", float(v))

        # Aggregate loss across datasets.
        # Batch-count-aware weighted average:
        #   Val(agg) = sum_i weight_i * sum_b loss_{i,b} / sum_i weight_i * num_batches_i
        # This prevents small validation sets from dominating the aggregate.
        total_w_batches = 0.0
        agg_loss_sum = 0.0
        for name, m in per_ds.items():
            w = float(self.val_weights.get(name, 1.0))
            if w <= 0:
                continue
            loss_sum = float(m.get("_loss_sum", 0.0))
            n_batches = float(m.get("_num_batches", 0.0))
            if n_batches <= 0:
                continue
            agg_loss_sum += w * loss_sum
            total_w_batches += w * n_batches

        val_loss = agg_loss_sum / max(1e-12, total_w_batches)
        self.log_metric("val", "loss", float(val_loss))

        # Prefer reporting PCK/NME for the primary dataset (if present).
        primary = per_ds.get(self.primary_val_name, {})
        if "pck" in primary:
            self.log_metric("val", "pck", float(primary["pck"]))
        if "nme" in primary:
            self.log_metric("val", "nme", float(primary["nme"]))

        # Console summary
        msg = f"Val(agg): loss={val_loss:.4f}"
        if "pck" in primary:
            msg += f", PCK[{self.primary_val_name}]={primary['pck']:.4f}"
        if "nme" in primary:
            msg += f", NME[{self.primary_val_name}]={primary['nme']:.4f}"
        print(msg)
        for name, m in per_ds.items():
            extra = []
            if "pck" in m:
                extra.append(f"PCK={m['pck']:.4f}")
            if "nme" in m:
                extra.append(f"NME={m['nme']:.4f}")
            extra_s = (", " + ", ".join(extra)) if extra else ""
            print(f"  - val[{name}]: loss={m['loss']:.4f}{extra_s}")

        return float(val_loss)

    def save_checkpoint(self, epoch, best=False):
        name = "best.pth" if best else f"epoch_{epoch}.pth"
        path = os.path.join(self.output_dir, name)
        payload = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
        }
        if self.scaler is not None and getattr(self.scaler, "is_enabled", None) and self.scaler.is_enabled():
            payload["scaler"] = self.scaler.state_dict()
        if self.ema is not None:
            payload["ema"] = self.ema.state_dict()

        torch.save(payload, path)
        return path

    def _save_training_report(self):
        """Save a minimal, model-agnostic training report.

        - Plots all scalar metrics logged in self.history as <metric>_curve.png.
        - Optionally calls a task-specific visualize_sample hook for qualitative examples.
        """
        # 1) Plot all scalar metrics present in history
        train_metrics = self.history.get("train", {})
        val_metrics = self.history.get("val", {})
        metric_names = sorted(set(train_metrics.keys()) | set(val_metrics.keys()))

        for metric_name in metric_names:
            fig, ax = plt.subplots(figsize=(6, 4))

            if metric_name in train_metrics:
                ax.plot(train_metrics[metric_name], label=f"train/{metric_name}")
            if metric_name in val_metrics:
                ax.plot(val_metrics[metric_name], label=f"val/{metric_name}")

            ax.set_xlabel("epoch")
            ax.set_ylabel(metric_name)
            ax.set_title(metric_name)
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(self.output_dir, f"{metric_name}_curve.png"))
            plt.close(fig)

        # 2) Optional qualitative examples via a model-specific hook
        self._save_qualitative_examples()

    def _save_qualitative_examples(self):
        """Ask the model to generate qualitative visualizations, if supported.

        Models that want to participate should implement a method with
        the following signature on the model instance:

            def generate_sample_visualization(
                self,
                sample: Dict[str, Any],
                out_path: str,
                device: torch.device,
            ) -> None:
                ...  # run forward + draw/save visualization

        Trainer remains model-agnostic and only orchestrates sampling
        from the validation dataset and file naming.
        """
        generate_fn = getattr(self.model, "generate_sample_visualization", None)
        if generate_fn is None:
            return

        if not getattr(self, "val_datasets", None):
            return

        # for i in range(num_vis):
        #     sample = self.val_ds[i]
        #     out_path = os.path.join(self.output_dir, "viz", f"val_example_{i}.png")
        #     if hasattr(self.model, "generate_sample_visualization"):
        #         generate_fn = self.model.generate_sample_visualization
        #         generate_fn(sample, out_path, self.device)
        #     else:
        #         self._visualize_model_outputs(sample, out_path)

        self.model.eval()
        viz_root = os.path.join(self.output_dir, "viz")
        os.makedirs(viz_root, exist_ok=True)

        num_vis_cfg = int(self.cfg.get("train", {}).get("num_val_viz", 4))
        if num_vis_cfg < 0:
            num_vis_cfg = 0

        def _safe_name(s: str) -> str:
            s = str(s)
            return s.replace("/", "_").replace("\\\\", "_")

        with torch.no_grad():
            for ds_name, ds in self.val_datasets.items():
                if len(ds) == 0:
                    continue
                num_vis = min(num_vis_cfg, len(ds))
                ds_dir = os.path.join(viz_root, _safe_name(ds_name))
                os.makedirs(ds_dir, exist_ok=True)
                for i in range(num_vis):
                    sample = ds[i]
                    out_path = os.path.join(ds_dir, f"val_example_{i}.png")
                    generate_fn(sample, out_path, self.device)

    def _load_checkpoint(self, ckpt_path: str):
        # weights_only=True keeps the load safe and fast; our checkpoint payload
        # contains only tensors + dicts of tensors.
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)

        # 1) Model weights: allow missing / extra keys (e.g., new attention params)
        self.model.load_state_dict(ckpt["model"], strict=False)

        # 2) Optimizer: try to load, but fall back to fresh optimizer on mismatch
        if "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError as e:
                print(f"Warning: could not load optimizer state ({e}); "
                      f"reinitializing optimizer with current parameters.")

        # 2.5) Scheduler state
        if "scheduler" in ckpt and ckpt["scheduler"] is not None and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e:
                print(f"Warning: could not load scheduler state ({e}); using fresh scheduler.")

        # 2.6) AMP scaler state
        if "scaler" in ckpt and self.scaler is not None and getattr(self.scaler, "is_enabled", None) and self.scaler.is_enabled():
            try:
                self.scaler.load_state_dict(ckpt["scaler"])
            except Exception as e:
                print(f"Warning: could not load GradScaler state ({e}); using fresh scaler.")

        # 2.7) EMA state
        if "ema" in ckpt and self.ema is not None:
            try:
                self.ema.load_state_dict(ckpt["ema"])
            except Exception as e:
                print(f"Warning: could not load EMA state ({e}); using fresh EMA.")

        # 3) Resume epoch counter if present
        if "epoch" in ckpt:
            self.start_epoch = ckpt["epoch"] + 1

    def run(self, resume_path: str = None):
        """Main training loop.

        Responsibilities:
          - optionally resume model/optimizer/scheduler/scaler/EMA
          - run epoch loop (train -> validate -> scheduler.step)
          - keep track of best validation loss and write `best.pth`
          - periodically save qualitative examples + final training report

        This method also handles Ctrl+C gracefully by saving a last checkpoint.
        """
        if resume_path is not None:
            self._load_checkpoint(resume_path)

        best_val = float("inf")
        current_epoch = self.start_epoch  # Track current epoch for interrupt handling
        interrupted = False
        try:
            for epoch in range(self.start_epoch, self.epochs + 1):
                current_epoch = epoch
                train_loss = self.train_epoch(epoch)
                val_loss = self.validate_epoch(epoch)
                self.scheduler.step()

                print(f"[Epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

                is_best = val_loss < best_val
                if is_best:
                    best_val = val_loss
                    self.save_checkpoint(epoch, best=True)

                # periodically refresh qualitative visualizations
                self._save_qualitative_examples()
        except KeyboardInterrupt:
            interrupted = True
            print("\nTraining interrupted by user (Ctrl+C). Saving last checkpoint and report...")
        finally:
            # Save checkpoint on interruption (outside of except to avoid nested exceptions)
            if interrupted:
                try:
                    self.save_checkpoint(current_epoch, best=False)
                    print(f"Saved interrupt checkpoint at epoch {current_epoch}.")
                except Exception as e:
                    print(f"Warning: could not save interrupt checkpoint: {e}")
            # always try to save a small report (curves + a few images)
            try:
                self._save_training_report()
            except Exception as e:
                print(f"Warning: could not save training report: {e}")
