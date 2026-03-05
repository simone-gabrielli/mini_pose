"""Train a YOLOv8 detector (Ultralytics) from within this repo.

This intentionally does NOT use mini-pose's Trainer; YOLOv8 has its own
well-tested training loop. We integrate it as a thin wrapper so the repo has a
single place for training + inference entrypoints.

Example:
  python scripts/train_yolov8.py \
    --data datasets/glasses_yolo/data.yaml \
    --model yolov8n.pt \
        --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --device 0 \
    --project work_dirs/yolov8_glasses \
    --name exp

The resulting best weights are typically at:
  work_dirs/yolov8_glasses/exp/weights/best.pt
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Ultralytics dataset YAML (e.g. created by tools/coco_to_yolo.py)")
    ap.add_argument("--model", default="yolov8n.pt", help="Base model or checkpoint (.pt)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="Ultralytics device (e.g. 'cpu', '0', '0,1')")
    ap.add_argument("--project", default="work_dirs/yolov8", help="Output project directory")
    ap.add_argument("--name", default="exp", help="Run name under project/")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise SystemExit("Ultralytics not installed. Install it with: pip install ultralytics") from e

    model = YOLO(str(args.model))

    model.train(
        data=str(args.data),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=str(args.device),
        project=str(args.project),
        name=str(args.name),
        workers=int(args.workers),
        patience=int(args.patience),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
