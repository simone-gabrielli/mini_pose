# pose/engine/trainer.py

import os
from typing import Dict, Any
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

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
        )
        self.val_ds = DatasetCls(
            json_path=ds_cfg["val_json"],
            image_root=ds_cfg["image_root"],
            input_size=tuple(ds_cfg["input_size"]),
            heatmap_size=tuple(ds_cfg["heatmap_size"]),
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
        self.model = ModelCls(
            num_stacks=model_cfg.get("num_stacks", 2),
            num_blocks=model_cfg.get("num_blocks", 1),
            num_feats=model_cfg.get("num_feats", 256),
            num_keypoints=num_keypoints,
        ).to(self.device)

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

        return running_loss / len(self.train_loader)

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

            # Compute metrics (e.g., PCK, NME) here if needed
            coord_preds = []
            coord_targets = []

            # Decode preds to keypoints
            for b in range(preds_last.size(0)):
                hm = preds_last[b].cpu()  # (K,H,W)
                K, H, W = hm.shape
                h_flat = hm.view(K, -1)
                idx = torch.argmax(h_flat, dim=1)
                y = (idx // W).float().numpy()
                x = (idx % W).float().numpy()
                coords = np.stack([x, y], axis=1)
                coord_preds.append(coords)

                # targets: from batch['keypoints']
                t = batch["keypoints"][b].cpu().numpy()[:, :2] / (256 / 64)  # adjust scaling
                coord_targets.append(t)

            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        coord_preds = np.stack(coord_preds)
        coord_targets = np.stack(coord_targets)

        pck = compute_pck(coord_preds, coord_targets)
        nme = compute_nme(coord_preds, coord_targets)

        val_loss = running_loss / len(self.val_loader)
        print(f"Val: loss={val_loss:.4f}, PCK={pck:.4f}, NME={nme:.4f}")

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
            print("\nTraining interrupted by user (Ctrl+C). Saving last checkpoint...")
            self.save_checkpoint(epoch, best=False)
