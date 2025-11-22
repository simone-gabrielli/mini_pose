# scripts/infer.py

import argparse
import cv2
import torch
from pose.engine.inference import load_model, decode_heatmaps
from pose.data.transforms import BasicTransform

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model-name", default="stacked_hourglass")
    parser.add_argument("--num-keypoints", type=int, default=68)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = load_model(
        args.checkpoint,
        model_name=args.model_name,
        num_keypoints=args.num_keypoints,
        device=device,
    )

    img_bgr = cv2.imread(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    tfm = BasicTransform(input_size=(256, 256))
    img_t, _ = tfm(img_rgb, keypoints=torch.zeros(args.num_keypoints, 3))  # dummy
    img_t = img_t.unsqueeze(0).to(device)

    with torch.no_grad():
        preds_last, _ = model(img_t)
        # preds_last: (B,K,H,W)
        heatmaps = preds_last[0].cpu()
    coords = decode_heatmaps(heatmaps)  # (K,2) in heatmap space

    # upscale to image space
    H_hm, W_hm = heatmaps.shape[1:]
    H_img, W_img = 256, 256  # because we resized
    coords[:, 0] *= (W_img / W_hm)
    coords[:, 1] *= (H_img / H_hm)

    # draw points
    for x, y in coords:
        cv2.circle(img_bgr, (int(x), int(y)), 2, (0, 0, 255), -1)

    cv2.imwrite("debug_out.png", img_bgr)
    print("Saved debug_out.png")

if __name__ == "__main__":
    main()
