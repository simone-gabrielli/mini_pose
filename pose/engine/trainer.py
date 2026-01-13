# pose/engine/trainer.py

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
import pose.losses    # noqa: F401  # ensures losses register
from pose.registry import MODEL_REGISTRY, LOSS_REGISTRY, DATASET_REGISTRY
from pose.engine.metrics import compute_pck, compute_nme
from pose.data import DatasetSpec, WeightedConcatDataset

class Trainer:
    def __init__(self, cfg: Dict[str, Any], device: str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Dataset
        ds_cfg = cfg["data"]
        DatasetCls = DATASET_REGISTRY[ds_cfg["type"]]
        base_kwargs = dict(
            input_size=tuple(ds_cfg["input_size"]),
            heatmap_size=tuple(ds_cfg["heatmap_size"]),
        )

        # Optional: forward sigma from config into dataset so GT heatmaps
        # are generated at the requested spatial resolution with matching sigma.
        if "sigma" in ds_cfg:
            base_kwargs["sigma"] = ds_cfg["sigma"]


        # Pass optional 3D-related kwargs if the dataset supports them
        if "depth_bins" in ds_cfg:
            base_kwargs["depth_bins"] = ds_cfg["depth_bins"]

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

                ds = DatasetCls(
                    json_path=train_json,
                    image_root=image_root,
                    aug_cfg=aug_cfg,
                    **base_kwargs,
                )

                loss_weight = float(item.get("loss_weight", 1.0))
                name = item.get("name")
                train_specs.append(DatasetSpec(dataset=ds, loss_weight=loss_weight, name=name))

            self.train_ds = WeightedConcatDataset(train_specs)
        else:
            # single dataset (current behavior)
            self.train_ds = DatasetCls(
                json_path=ds_cfg["train_json"],
                image_root=ds_cfg["image_root"],
                aug_cfg=ds_cfg.get("aug", None),
                **base_kwargs,
            )

        # Validation dataset (single dataset)
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

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=ds_cfg["batch_size"],
            shuffle=True,
            num_workers=ds_cfg.get("num_workers", 4),
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=ds_cfg["batch_size"],
            shuffle=False,
            num_workers=ds_cfg.get("num_workers", 4),
            pin_memory=True,
        )

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

        # Provide dataset depth range to the model so z is in the same units as annotations
        if hasattr(self.train_ds, "depth_range") and self.train_ds.depth_range is not None:
            model_kwargs["depth_range"] = self.train_ds.depth_range
        if hasattr(self.train_ds, "depth_mean") and self.train_ds.depth_mean is not None:
            model_kwargs["depth_mean"] = self.train_ds.depth_mean

        self.model = ModelCls(**model_kwargs).to(self.device)

        # Loss
        LossCls = LOSS_REGISTRY[cfg["loss"]["name"]]
        self.criterion = LossCls().to(self.device)

        # auxiliary 2D MSE for spatial heatmap supervision when using 3D volumetric loss
        self._mse2d = torch.nn.MSELoss(reduction="mean")
        # weight for 2D auxiliary loss (can be overridden in config)
        self.aux_2d_weight = cfg.get("loss", {}).get("aux_2d_weight", 1.0)

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
            self.scaler = torch.cuda.amp.GradScaler(enabled=False)
        else:
            self.amp_dtype = torch.float16
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

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
                torch.cuda.amp.autocast(dtype=self.amp_dtype)
                if self.use_amp and self.device.type == "cuda"
                else nullcontext()
            )

            with autocast_ctx:
                # Branch: direct pose regression with reprojection loss
                if loss_name == "pose_reprojection":
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
                    # Heatmap-based training (existing behaviour)
                    if loss_name == "heatmap_3d_mse":
                        targets = batch["heatmaps_3d"].to(self.device)
                    else:
                        targets = batch["heatmaps"].to(self.device)

                    visible = batch["visible"].to(self.device)
                    depth_targets = batch.get("depth", None)
                    if depth_targets is not None:
                        depth_targets = depth_targets.to(self.device)

                    out = self.model(imgs)
                    # FAN3D returns: last_heatmap, all_heatmaps, last_depth, all_depths
                    if isinstance(out, tuple) and len(out) == 4:
                        preds_last, preds_all, depth_last, depth_all = out
                        loss = 0.0
                        for i in range(len(preds_all)):
                            p = preds_all[i]
                            loss += self.criterion(p, targets, visible, sample_weight=sample_weight)
                            if depth_targets is not None and hasattr(self, 'depth_criterion'):
                                d = depth_all[i]
                                loss += self.depth_criterion(d, depth_targets)
                    else:
                        preds_last, preds_all = out
                        loss = 0.0
                        for p in preds_all:
                            loss += self.criterion(p, targets, visible, sample_weight=sample_weight)

                        if loss_name == "heatmap_3d_mse":
                            hm_targets = batch["heatmaps"].to(self.device)
                            if sample_weight is None:
                                aux_loss = self._mse2d(preds_last, hm_targets)
                            else:
                                err2d = (preds_last - hm_targets) ** 2
                                per_sample = err2d.view(err2d.size(0), -1).mean(dim=1)
                                sw = sample_weight.to(dtype=per_sample.dtype).view(-1)
                                aux_loss = (per_sample * sw).sum() / sw.sum().clamp(min=1e-6)
                            loss = loss + self.aux_2d_weight * aux_loss

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
    def validate_epoch(self, epoch: int):
        self.model.eval()
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [val]")
        running_loss = 0.0
        loss_name = self.cfg.get("loss", {}).get("name", "")

        # Coordinate regression losses (LOTR-style direct coordinate prediction)
        coord_regression_losses = [
            "smooth_wing", "wing", "adaptive_wing", 
            "landmark_l1", "landmark_mse", "landmark_smooth_l1",
            "nme", "combined_landmark"
        ]

        # For heatmap-based models we also track PCK/NME; for pose
        # regression we only report the scalar loss.
        coord_preds = []
        coord_targets = []
        is_coord_regression = loss_name in coord_regression_losses

        # Optionally evaluate EMA weights.
        eval_ctx = self.ema.apply_to(self.model) if (self.ema is not None and self.ema_eval) else nullcontext()

        with eval_ctx:
            for batch in pbar:
                imgs = batch["image"].to(self.device)
                sample_weight = batch.get("dataset_weight")
                if sample_weight is not None:
                    sample_weight = sample_weight.to(self.device)

                if loss_name == "pose_reprojection":
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
                    preds_last = None  # no heatmaps here

                elif is_coord_regression:
                    # Coordinate regression validation (LOTR-style)
                    targets_2d = batch["keypoints"][:, :, :2].to(self.device)
                    visible = batch["visible"].to(self.device)

                    out = self.model(imgs)

                    # LOTR returns (normalized_coords, pixel_coords)
                    if isinstance(out, tuple) and len(out) == 2:
                        _, landmarks_pixel = out
                        preds_2d = landmarks_pixel[..., :2]  # (B, N, 2)
                    else:
                        preds_2d = out
                        if preds_2d.dim() == 2:
                            B = imgs.size(0)
                            preds_2d = preds_2d.view(B, -1, 2)

                    loss = self.criterion(preds_2d, targets_2d, visible, sample_weight=sample_weight)
                    preds_last = None  # no heatmaps

                    # Collect predictions and targets for NME computation
                    for b in range(preds_2d.size(0)):
                        pred_coords = preds_2d[b].cpu().numpy()
                        target_coords = targets_2d[b].cpu().numpy()
                        coord_preds.append(pred_coords)
                        coord_targets.append(target_coords)

                else:
                    if loss_name == "heatmap_3d_mse":
                        targets = batch["heatmaps_3d"].to(self.device)
                    else:
                        targets = batch["heatmaps"].to(self.device)

                    visible = batch["visible"].to(self.device)
                    depth_targets = batch.get("depth", None)
                    if depth_targets is not None:
                        depth_targets = depth_targets.to(self.device)

                    out = self.model(imgs)
                    if isinstance(out, tuple) and len(out) == 4:
                        preds_last, preds_all, depth_last, depth_all = out
                        loss = 0.0
                        for i in range(len(preds_all)):
                            p = preds_all[i]
                            loss += self.criterion(p, targets, visible, sample_weight=sample_weight)
                            if depth_targets is not None and hasattr(self, 'depth_criterion'):
                                d = depth_all[i]
                                loss += self.depth_criterion(d, depth_targets)
                    else:
                        preds_last, preds_all = out
                        loss = 0.0
                        for p in preds_all:
                            loss += self.criterion(p, targets, visible, sample_weight=sample_weight)

                # Only compute PCK/NME if we actually have heatmaps
                if preds_last is not None and preds_last.dim() == 4:
                    # image and heatmap sizes from dataset config
                    H_img, W_img = self.train_ds.input_size[1], self.train_ds.input_size[0]
                    H_hm, W_hm = self.train_ds.heatmap_size[1], self.train_ds.heatmap_size[0]
                    sx = W_hm / float(W_img)
                    sy = H_hm / float(H_img)

                    for b in range(preds_last.size(0)):
                        hm = preds_last[b].cpu()  # (K,H,W)
                        K, H, W = hm.shape
                        h_flat = hm.view(K, -1)
                        idx = torch.argmax(h_flat, dim=1)
                        y = (idx // W).float().numpy()
                        x = (idx % W).float().numpy()
                        coords = np.stack([x, y], axis=1)
                        coord_preds.append(coords)

                        # targets: from batch['keypoints'], scaled into heatmap space
                        t = batch["keypoints"][b].cpu().numpy()[:, :2]
                        t_hm = np.stack([t[:, 0] * sx, t[:, 1] * sy], axis=1)
                        coord_targets.append(t_hm)

                running_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

        val_loss = running_loss / len(self.val_loader)
        self.log_metric("val", "loss", val_loss)

        # If we computed keypoint coords, also log PCK/NME
        if coord_preds:
            try:
                coord_preds_arr = np.stack(coord_preds)
                coord_targets_arr = np.stack(coord_targets)
                pck = compute_pck(coord_preds_arr, coord_targets_arr)
                nme = compute_nme(coord_preds_arr, coord_targets_arr)
                self.log_metric("val", "pck", float(pck))
                self.log_metric("val", "nme", float(nme))
                print(f"Val: loss={val_loss:.4f}, PCK={pck:.4f}, NME={nme:.4f}")
            except Exception:
                print(f"Val: loss={val_loss:.4f}")
        else:
            # pose-reprojection case (no heatmaps)
            print(f"Val: loss={val_loss:.4f}")

        return val_loss

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

        if len(self.val_ds) == 0:
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
        os.makedirs(os.path.join(self.output_dir, "viz"), exist_ok=True)

        num_vis = min(4, len(self.val_ds))
        with torch.no_grad():
            for i in range(num_vis):
                sample = self.val_ds[i]
                out_path = os.path.join(self.output_dir, "viz", f"val_example_{i}.png")
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
        if resume_path is not None:
            self._load_checkpoint(resume_path)

        best_val = float("inf")
        try:
            for epoch in range(self.start_epoch, self.epochs + 1):
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
            print("\nTraining interrupted by user (Ctrl+C). Saving last checkpoint and report...")
            self.save_checkpoint(epoch, best=False)
        finally:
            # always try to save a small report (curves + a few images)
            self._save_training_report()
