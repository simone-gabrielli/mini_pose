# pose/engine/trainer.py

import os
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
        self.train_ds = DatasetCls(
            json_path=ds_cfg["train_json"],
            image_root=ds_cfg["image_root"],
            input_size=tuple(ds_cfg["input_size"]),
            heatmap_size=tuple(ds_cfg["heatmap_size"]),
            aug_cfg=ds_cfg.get("aug", None),
        )
        self.val_ds = DatasetCls(
            json_path=ds_cfg["val_json"],
            image_root=ds_cfg["image_root"],
            input_size=tuple(ds_cfg["input_size"]),
            heatmap_size=tuple(ds_cfg["heatmap_size"]),
            aug_cfg=ds_cfg.get("aug", None),
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

        # Model
        model_cfg = cfg["model"]
        ModelCls = MODEL_REGISTRY[model_cfg["name"]]

        model_kwargs = {
            "num_keypoints": num_keypoints,
        }

        if "num_stacks" in model_cfg:
            model_kwargs["num_stacks"] = model_cfg["num_stacks"]
        if "num_feats" in model_cfg:
            model_kwargs["num_feats"] = model_cfg["num_feats"]
        # support both stacked_hourglass (num_blocks) and FAN2D (num_modules)
        if "num_blocks" in model_cfg:
            model_kwargs["num_blocks"] = model_cfg["num_blocks"]
        if "num_modules" in model_cfg:
            model_kwargs["num_modules"] = model_cfg["num_modules"]

        self.model = ModelCls(**model_kwargs).to(self.device)

        # Loss
        LossCls = LOSS_REGISTRY[cfg["loss"]["name"]]
        self.criterion = LossCls().to(self.device)

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

        # generic history: split -> metric_name -> list[float]
        self.history = {"train": {}, "val": {}}

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
            targets = batch["heatmaps"].to(self.device)
            visible = batch["visible"].to(self.device)

            preds_last, preds_all = self.model(imgs)
            loss = 0.0
            # sum loss over stacks
            for p in preds_all:
                loss = loss + self.criterion(p, targets, visible)

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
            targets = batch["heatmaps"].to(self.device)
            visible = batch["visible"].to(self.device)

            preds_last, preds_all = self.model(imgs)
            loss = 0.0
            for p in preds_all:
                loss = loss + self.criterion(p, targets, visible)

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

        self.model.eval()
        os.makedirs(os.path.join(self.output_dir, "viz"), exist_ok=True)

        num_vis = min(4, len(self.val_ds))
        with torch.no_grad():
            for i in range(num_vis):
                sample = self.val_ds[i]
                out_path = os.path.join(self.output_dir, "viz", f"val_example_{i}.png")
                generate_fn(sample, out_path, self.device)

    def run(self):
        best_val = float("inf")
        try:
            for epoch in range(1, self.epochs + 1):
                train_loss = self.train_epoch(epoch)
                val_loss = self.validate_epoch(epoch)
                self.scheduler.step()

                print(f"[Epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

                is_best = val_loss < best_val
                if is_best:
                    best_val = val_loss
                    self.save_checkpoint(epoch, best=True)
        except KeyboardInterrupt:
            print("\nTraining interrupted by user (Ctrl+C). Saving last checkpoint and report...")
            self.save_checkpoint(epoch, best=False)
        finally:
            # always try to save a small report (curves + a few images)
            self._save_training_report()
