# pose/engine/trainer.py

import os
import shutil
from typing import Dict, Any
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import pose.data      # noqa: F401  # ensures datasets register
import pose.models    # noqa: F401  # ensures models register
import pose.losses    # noqa: F401  # ensures losses register
from pose.registry import MODEL_REGISTRY, LOSS_REGISTRY, DATASET_REGISTRY
from pose.engine.metrics import compute_pck, compute_nme

class Trainer:
    def __init__(self, cfg: Dict[str, Any], device: str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Dataset
        ds_cfg = cfg["data"]
        DatasetCls = DATASET_REGISTRY[ds_cfg["type"]]
        common_kwargs = dict(
            image_root=ds_cfg["image_root"],
            input_size=tuple(ds_cfg["input_size"]),
            heatmap_size=tuple(ds_cfg["heatmap_size"]),
            aug_cfg=ds_cfg.get("aug", None),
        )

        # Optional: forward sigma from config into dataset so GT heatmaps
        # are generated at the requested spatial resolution with matching sigma.
        if "sigma" in ds_cfg:
            common_kwargs["sigma"] = ds_cfg["sigma"]


        # Pass optional 3D-related kwargs if the dataset supports them
        if "depth_bins" in ds_cfg:
            common_kwargs["depth_bins"] = ds_cfg["depth_bins"]

        self.train_ds = DatasetCls(
            json_path=ds_cfg["train_json"],
            **common_kwargs,
        )
        self.val_ds = DatasetCls(
            json_path=ds_cfg["val_json"],
            **common_kwargs,
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
        if train_cfg["optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=train_cfg["lr"],
                weight_decay=train_cfg.get("weight_decay", 0.0),
            )
        else:
            raise ValueError("Unsupported optimizer")

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=train_cfg.get("lr_steps", [30, 50]),
            gamma=train_cfg.get("lr_gamma", 0.1),
        )

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
        for batch in pbar:
            imgs = batch["image"].to(self.device)

            # Use volumetric 3D heatmaps when training with 3D loss
            if self.cfg.get("loss", {}).get("name", "") == "heatmap_3d_mse":
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
                    loss += self.criterion(p, targets, visible)
                    if depth_targets is not None and hasattr(self, 'depth_criterion'):
                        d = depth_all[i]
                        loss += self.depth_criterion(d, depth_targets)
            else:
                preds_last, preds_all = out
                loss = 0.0
                for p in preds_all:
                    loss += self.criterion(p, targets, visible)

                if self.cfg.get("loss", {}).get("name", "") == "heatmap_3d_mse":
                    hm_targets = batch["heatmaps"].to(self.device)
                    aux_loss = self._mse2d(preds_last, hm_targets)
                    loss = loss + self.aux_2d_weight * aux_loss

                # If we're using volumetric 3D loss, add an auxiliary 2D MSE
                if self.cfg.get("loss", {}).get("name", "") == "heatmap_3d_mse":
                    # preds_last is a 2D probabilistic heatmap (B,K,H,W)
                    hm_targets = batch["heatmaps"].to(self.device)
                    aux_loss = self._mse2d(preds_last, hm_targets)
                    loss = loss + self.aux_2d_weight * aux_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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
        for batch in pbar:
            imgs = batch["image"].to(self.device)

            if self.cfg.get("loss", {}).get("name", "") == "heatmap_3d_mse":
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
                    loss += self.criterion(p, targets, visible)
                    if depth_targets is not None and hasattr(self, 'depth_criterion'):
                        d = depth_all[i]
                        loss += self.depth_criterion(d, depth_targets)
            else:
                preds_last, preds_all = out
                loss = 0.0
                for p in preds_all:
                    loss += self.criterion(p, targets, visible)

            # Optionally compute metrics (e.g., PCK, NME) if model outputs heatmaps
            if preds_last.dim() == 4:  # (B, K, H, W)
                coord_preds = []
                coord_targets = []

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

        return val_loss

    def save_checkpoint(self, epoch, best=False):
        name = "best.pth" if best else f"epoch_{epoch}.pth"
        path = os.path.join(self.output_dir, name)
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
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
        ckpt = torch.load(ckpt_path, map_location=self.device)

        # 1) Model weights: allow missing / extra keys (e.g., new attention params)
        self.model.load_state_dict(ckpt["model"], strict=False)

        # 2) Optimizer: try to load, but fall back to fresh optimizer on mismatch
        if "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError as e:
                print(f"Warning: could not load optimizer state ({e}); "
                      f"reinitializing optimizer with current parameters.")

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
