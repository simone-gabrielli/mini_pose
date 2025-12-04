"""Train a lightweight face detector that regresses a single bounding box per image.

This script supports two modes:
  - Legacy CLI: pass `--ann-file` and `--images-dir` (same as before)
  - Config mode: pass `--config configs/face_mobilenet.yaml` and the script will
    instantiate dataset and model using the project's registries.

When using a config, the file should follow the project's config conventions
(`configs/face_mobilenet.yaml` was added). Check the `configs/` folder for examples.
"""
import os
import json
import argparse
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from pose.config import Config
from pose.registry import MODEL_REGISTRY, DATASET_REGISTRY

from pose.detectors.face_detector import TinyFaceDetector
import pose.data  # ensure dataset modules are imported and registered


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        prob = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = prob * targets + (1 - prob) * (1 - targets)
        mod = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * mod * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def giou_loss(preds, targets):
    """Compute mean GIoU loss between preds and targets.

    preds/targets: (N,4) in normalized x1,y1,x2,y2
    Returns scalar tensor.
    """
    # ensure tensors
    p = preds
    t = targets
    # corner coords
    x1 = torch.min(p[:, 0], p[:, 2])
    y1 = torch.min(p[:, 1], p[:, 3])
    x2 = torch.max(p[:, 0], p[:, 2])
    y2 = torch.max(p[:, 1], p[:, 3])

    tx1 = torch.min(t[:, 0], t[:, 2])
    ty1 = torch.min(t[:, 1], t[:, 3])
    tx2 = torch.max(t[:, 0], t[:, 2])
    ty2 = torch.max(t[:, 1], t[:, 3])

    inter_x1 = torch.max(x1, tx1)
    inter_y1 = torch.max(y1, ty1)
    inter_x2 = torch.min(x2, tx2)
    inter_y2 = torch.min(y2, ty2)

    iw = (inter_x2 - inter_x1).clamp(min=0)
    ih = (inter_y2 - inter_y1).clamp(min=0)
    inter = iw * ih

    area_p = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_t = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
    union = area_p + area_t - inter
    iou = torch.zeros_like(inter)
    valid = union > 0
    iou[valid] = inter[valid] / union[valid]

    # enclosure
    enc_x1 = torch.min(x1, tx1)
    enc_y1 = torch.min(y1, ty1)
    enc_x2 = torch.max(x2, tx2)
    enc_y2 = torch.max(y2, ty2)
    enc_w = (enc_x2 - enc_x1).clamp(min=0)
    enc_h = (enc_y2 - enc_y1).clamp(min=0)
    enc_area = enc_w * enc_h
    giou = iou - (enc_area - union) / enc_area.clamp(min=1e-6)
    loss = 1.0 - giou
    return loss.mean()


def train_epoch(model, dataloader, optimizer, device, epoch, weight_bbox=10.0, use_giou=False):
    model.train()
    reg_loss = nn.SmoothL1Loss()
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    total_loss = 0.0
    n_samples = 0
    for batch in dataloader:
        # support dataset returning either (img, target) or dict {image, bbox}
        if isinstance(batch, dict) or (isinstance(batch, list) and isinstance(batch[0], dict)):
            imgs = batch["image"].to(device) if isinstance(batch, dict) else batch[0]["image"].to(device)
            targets = batch["bbox"].to(device) if isinstance(batch, dict) else batch[0]["bbox"].to(device)
        else:
            imgs, targets = batch
            imgs = imgs.to(device)
            targets = targets.to(device)

        preds = model(imgs)

        conf_logits = preds[:, 0]
        bbox_preds = torch.sigmoid(preds[:, 1:5])
        conf_targets = targets[:, 0]
        bbox_targets = targets[:, 1:5]

        # classification loss (focal)
        loss_conf = focal_loss(conf_logits, conf_targets)

        # regression loss only on positive samples
        pos_mask = (conf_targets > 0.5)
        if pos_mask.any():
            if use_giou:
                loss_bbox = giou_loss(bbox_preds[pos_mask], bbox_targets[pos_mask])
            else:
                loss_bbox = reg_loss(bbox_preds[pos_mask], bbox_targets[pos_mask])
        else:
            loss_bbox = torch.tensor(0.0, device=device)

        loss = loss_conf + weight_bbox * loss_bbox

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = imgs.size(0)
        total_loss += float(loss.item()) * batch_size
        n_samples += batch_size

    avg = total_loss / float(n_samples)
    print(f"Epoch {epoch} train loss: {avg:.6f}")
    return avg


def validate(model, dataloader, device, use_giou=False, weight_bbox=10.0):
    model.eval()
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    reg_loss = nn.SmoothL1Loss()
    total_loss = 0.0
    n_samples = 0
    total_iou = 0.0
    iou_count = 0
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict) or (isinstance(batch, list) and isinstance(batch[0], dict)):
                imgs = batch["image"].to(device) if isinstance(batch, dict) else batch[0]["image"].to(device)
                targets = batch["bbox"].to(device) if isinstance(batch, dict) else batch[0]["bbox"].to(device)
            else:
                imgs, targets = batch
                imgs = imgs.to(device)
                targets = targets.to(device)

            preds = model(imgs)

            conf_logits = preds[:, 0]
            bbox_preds = torch.sigmoid(preds[:, 1:5])
            conf_targets = targets[:, 0]
            bbox_targets = targets[:, 1:5]

            # classification (focal) + masked regression
            loss_conf = focal_loss(conf_logits, conf_targets)
            pos_mask = (conf_targets > 0.5)
            if pos_mask.any():
                if use_giou:
                    loss_bbox = giou_loss(bbox_preds[pos_mask], bbox_targets[pos_mask])
                else:
                    loss_bbox = reg_loss(bbox_preds[pos_mask], bbox_targets[pos_mask])
            else:
                loss_bbox = torch.tensor(0.0, device=device)
            loss = loss_conf + weight_bbox * loss_bbox

            # compute IoU for reporting when ground-truth present (conf_targets==1)
            try:
                # bbox_preds are normalized x1,y1,x2,y2
                preds_np = bbox_preds.cpu().numpy()
                targets_np = bbox_targets.cpu().numpy()
                conf_t_np = conf_targets.cpu().numpy()
                for p, t, cval in zip(preds_np, targets_np, conf_t_np):
                    if float(cval) > 0.5:
                        # compute iou
                        x1 = max(0.0, min(p[0], p[2]))
                        y1 = max(0.0, min(p[1], p[3]))
                        x2 = max(0.0, max(p[0], p[2]))
                        y2 = max(0.0, max(p[1], p[3]))
                        tx1 = max(0.0, min(t[0], t[2]))
                        ty1 = max(0.0, min(t[1], t[3]))
                        tx2 = max(0.0, max(t[0], t[2]))
                        ty2 = max(0.0, max(t[1], t[3]))
                        inter_x1 = max(x1, tx1)
                        inter_y1 = max(y1, ty1)
                        inter_x2 = min(x2, tx2)
                        inter_y2 = min(y2, ty2)
                        iw = max(0.0, inter_x2 - inter_x1)
                        ih = max(0.0, inter_y2 - inter_y1)
                        inter = iw * ih
                        area_p = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
                        area_t = max(0.0, (tx2 - tx1)) * max(0.0, (ty2 - ty1))
                        union = area_p + area_t - inter
                        iou = inter / union if union > 0 else 0.0
                        total_iou += float(iou)
                        iou_count += 1
            except Exception:
                pass

            batch_size = imgs.size(0)
            total_loss += float(loss.item()) * batch_size
            n_samples += batch_size

    avg = total_loss / float(n_samples)
    print(f"Validation loss: {avg:.6f}")
    if iou_count > 0:
        print(f"Validation IoU (mean over positives): {total_iou / iou_count:.4f} (N={iou_count})")
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-file", help="COCO-style annotations JSON")
    parser.add_argument("--images-dir", help="Images directory")
    parser.add_argument("--out-dir", help="Output directory (overrides config)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--config", help="Optional config YAML to use (preferred)")
    args = parser.parse_args()

    # If config provided, prefer its values and use registry to instantiate
    if args.config:
        cfg = Config.from_yaml(args.config).raw

        ds_cfg = cfg["data"]
        DatasetCls = DATASET_REGISTRY[ds_cfg["type"]]
        common_kwargs = dict(
            json_path=ds_cfg["train_json"],
            image_root=ds_cfg["image_root"],
            input_size=tuple(ds_cfg.get("input_size", [256, 256])),
            heatmap_size=tuple(ds_cfg.get("heatmap_size", [64, 64])),
            aug_cfg=ds_cfg.get("aug", None),
        )

        train_ds = DatasetCls(**common_kwargs)
        val_common = common_kwargs.copy()
        val_common["json_path"] = ds_cfg["val_json"]
        val_ds = DatasetCls(**val_common)

        train_batch = ds_cfg.get("batch_size", 32)
        num_workers = ds_cfg.get("num_workers", 4)
        model_cfg = cfg.get("model", {})
        ModelCls = MODEL_REGISTRY[model_cfg["name"]]
        model_kwargs = {}
        # pass keys like pretrained/embed_dim if present
        for k, v in model_cfg.items():
            if k == "name":
                continue
            model_kwargs[k] = v

        model = ModelCls(**model_kwargs)

        train_loader = DataLoader(train_ds, batch_size=train_batch, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=train_batch, shuffle=False, num_workers=max(1, num_workers//2), pin_memory=True)

        out_dir = cfg.get("train", {}).get("output_dir", "work_dirs/face_detector")
        epochs = cfg.get("train", {}).get("epochs", 10)
        lr = cfg.get("train", {}).get("lr", 1e-4)
        use_giou = cfg.get("train", {}).get("use_giou", False)
        weight_bbox = cfg.get("train", {}).get("weight_bbox", 10.0)
    else:
        # legacy CLI mode
        if not args.ann_file or not args.images_dir:
            raise ValueError("Either provide --config or both --ann-file and --images-dir")
        # build a simple Coco face dataset in-place (reuse the class from pose.data if available)
        from pose.data.dataset_face import CocoFaceDataset

        train_ds = CocoFaceDataset(args.ann_file, args.images_dir, input_size=(args.img_size or 256, args.img_size or 256))
        # tiny split: 90/10
        n = len(train_ds)
        split = int(0.9 * n)
        train_set, val_set = torch.utils.data.random_split(train_ds, [split, n - split])
        train_loader = DataLoader(train_set, batch_size=args.batch_size or 32, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size or 32, shuffle=False, num_workers=2, pin_memory=True)

        model = TinyFaceDetector(pretrained=True)
        out_dir = args.out_dir or "work_dirs/face_detector"
        epochs = args.epochs or 10
        lr = args.lr or 1e-4
        use_giou = False
        weight_bbox = 10.0

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        train_epoch(model, train_loader, optimizer, device, epoch, weight_bbox=weight_bbox, use_giou=use_giou)
        val = validate(model, val_loader, device, use_giou=use_giou, weight_bbox=weight_bbox)
        # save a few qualitative visualizations on validation set
        try:
            from pathlib import Path as _Path

            def _unnormalize_image(tensor_img):
                # tensor_img: C,H,W normalized with ImageNet mean/std
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                img = tensor_img.cpu().numpy().transpose(1, 2, 0)
                img = (img * std) + mean
                img = (img * 255.0).clip(0, 255).astype('uint8')
                return img

            def save_visualizations(model, val_dataset, out_dir, device, num_vis=4, conf_th=0.25):
                model.eval()
                viz_dir = _Path(out_dir) / "viz"
                viz_dir.mkdir(parents=True, exist_ok=True)
                n = min(num_vis, len(val_dataset))
                with torch.no_grad():
                    for i in range(n):
                        sample = val_dataset[i]
                        # sample may be dict or tuple
                        if isinstance(sample, dict):
                            img_t = sample["image"].unsqueeze(0).to(device)
                            meta = sample.get("meta", {})
                        else:
                            img_t, target = sample
                            img_t = img_t.unsqueeze(0).to(device)
                            meta = {}

                        res = model.predict(img_t, conf_th=conf_th)
                        # res is a list (per batch item)
                        out = res[0]

                        img_vis = _unnormalize_image(img_t[0].cpu())
                        # draw bbox if present
                        import cv2 as _cv2
                        h, w = img_vis.shape[:2]
                        gt_iou_str = None
                        if out.get("bbox") is not None:
                            x1, y1, x2, y2 = map(int, out["bbox"])
                            _cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            _cv2.putText(img_vis, f"pred:{out['conf']:.2f}", (max(5, x1), max(15, y1-5)), _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
                        else:
                            _cv2.putText(img_vis, f"pred:{out['conf']:.2f}", (5,15), _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                        # if GT available in sample, draw GT box and compute IoU
                        try:
                            if isinstance(sample, dict) and "bbox" in sample:
                                gt = sample["bbox"]
                                if isinstance(gt, torch.Tensor):
                                    gt = gt.cpu().numpy()
                                if float(gt[0]) > 0.5:
                                    gx1 = int(gt[1] * w)
                                    gy1 = int(gt[2] * h)
                                    gx2 = int(gt[3] * w)
                                    gy2 = int(gt[4] * h)
                                    _cv2.rectangle(img_vis, (gx1, gy1), (gx2, gy2), (0, 255, 0), 1)
                                    # compute IoU if prediction exists
                                    if out.get("bbox") is not None:
                                        px1, py1, px2, py2 = x1, y1, x2, y2
                                        inter_x1 = max(px1, gx1)
                                        inter_y1 = max(py1, gy1)
                                        inter_x2 = min(px2, gx2)
                                        inter_y2 = min(py2, gy2)
                                        iw = max(0, inter_x2 - inter_x1)
                                        ih = max(0, inter_y2 - inter_y1)
                                        inter = iw * ih
                                        area_p = max(0, (px2 - px1)) * max(0, (py2 - py1))
                                        area_g = max(0, (gx2 - gx1)) * max(0, (gy2 - gy1))
                                        union = area_p + area_g - inter
                                        iou = inter / union if union > 0 else 0.0
                                        gt_iou_str = f"IoU:{iou:.2f}"
                                        _cv2.putText(img_vis, gt_iou_str, (5, h-8), _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                        except Exception:
                            pass

                        out_path = viz_dir / f"val_example_{i}_pred.png"
                        _cv2.imwrite(str(out_path), cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))

            # choose dataset object for visualization: prefer val_ds if available
            try:
                ds_for_viz = val_ds
            except NameError:
                # if val_loader is a DataLoader built from subset, try to access dataset
                ds_for_viz = getattr(val_loader, 'dataset', None)

            if ds_for_viz is not None:
                save_visualizations(model, ds_for_viz, out_dir, device, num_vis=4, conf_th=0.25)
        except Exception as e:
            print(f"Warning: could not save visualizations ({e})")

        ckpt = {"model": model.state_dict(), "epoch": epoch, "optimizer": optimizer.state_dict()}
        torch.save(ckpt, os.path.join(out_dir, f"epoch_{epoch}.pth"))
        if val < best_val:
            best_val = val
            torch.save(ckpt, os.path.join(out_dir, "best.pth"))

    print("Training finished. Best val:", best_val)


if __name__ == "__main__":
    main()
