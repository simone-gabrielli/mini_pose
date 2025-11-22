# scripts/infer.py

import argparse
import cv2
import torch
from pose.engine.inference import load_model, decode_heatmaps
from pose.data.transforms import BasicTransform
from pose.detectors.haar_face import HaarFaceDetector

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
    if img_bgr is None:
        raise FileNotFoundError(args.image)

    # 1) Detect faces (FAN-style pre-processing)
    detector = HaarFaceDetector()
    boxes = detector.detect(img_bgr)
    if not boxes:
        print("No faces detected; saving original image as debug_out.png")
        cv2.imwrite("debug_out.png", img_bgr)
        return

    # For now, handle the first detected face; can be extended to multi-face
    x, y, w, h = boxes[0]

    # Add some margin around the face, similar to FAN-style loose crop
    margin = 0.25
    cx = x + w / 2.0
    cy = y + h / 2.0
    size = max(w, h) * (1.0 + margin)
    x1 = int(cx - size / 2.0)
    y1 = int(cy - size / 2.0)
    x2 = int(cx + size / 2.0)
    y2 = int(cy + size / 2.0)

    # clamp to image bounds
    H_orig, W_orig = img_bgr.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W_orig - 1, x2)
    y2 = min(H_orig - 1, y2)

    face_bgr = img_bgr[y1:y2, x1:x2]
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    # 2) Resize/crop to model input (use 512x512 as requested)
    input_size = (512, 512)
    tfm = BasicTransform(input_size=input_size)
    img_t, _ = tfm(face_rgb, keypoints=torch.zeros(args.num_keypoints, 3))  # dummy kpts
    img_t = img_t.unsqueeze(0).to(device)

    # 3) Forward through model
    with torch.no_grad():
        preds_last, _ = model(img_t)
        heatmaps = preds_last[0].cpu()  # (K,Hh,Wh)

    coords = decode_heatmaps(heatmaps)  # (K,2) in heatmap space

    # 4) Map keypoints from heatmap space -> face crop -> original image
    H_hm, W_hm = heatmaps.shape[1:]
    H_face, W_face = input_size[1], input_size[0]
    coords[:, 0] *= (W_face / W_hm)
    coords[:, 1] *= (H_face / H_hm)

    # now scale from [0, W_face/H_face] in the resized crop back to original image coords
    scale_x = (x2 - x1) / float(W_face)
    scale_y = (y2 - y1) / float(H_face)
    coords[:, 0] = coords[:, 0] * scale_x + x1
    coords[:, 1] = coords[:, 1] * scale_y + y1

    # 5) Draw points on original image
    for xk, yk in coords:
        cv2.circle(img_bgr, (int(xk), int(yk)), 2, (0, 0, 255), -1)

    # Optionally, draw the detected face box
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.imwrite("debug_out.png", img_bgr)
    print("Saved debug_out.png")

if __name__ == "__main__":
    main()
