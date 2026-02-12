# scripts/benchmark_tiny_face.py

import argparse
import time

import torch

from pose.detectors.face_detector import TinyFaceDetector


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--input", type=int, default=256, help="Square input size")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--fp16", action="store_true", help="Use fp16 for model+input (CUDA only)")

    p.add_argument("--backbone", default="mobilenet_v3_small", choices=["mobilenet_v2", "mobilenet_v3_small"])
    p.add_argument("--width-mult", type=float, default=1.0)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--no-pretrained", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True

    model = TinyFaceDetector(
        pretrained=(not args.no_pretrained),
        backbone=args.backbone,
        width_mult=args.width_mult,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
    ).to(device)
    model.eval()

    x = torch.randn(1, 3, args.input, args.input, device=device)

    if args.fp16 and device.type == "cuda":
        model = model.half()
        x = x.half()

    # Warmup
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Timed
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.iters):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    t1 = time.perf_counter()

    ms = (t1 - t0) * 1000.0 / float(args.iters)
    print(f"avg forward: {ms:.3f} ms   (device={device}, fp16={args.fp16 and device.type=='cuda'})")


if __name__ == "__main__":
    main()
