# scripts/infer_lotr.py
"""
Inference script for LOTR (Localization Transformer) models.

Usage:
    python scripts/infer_lotr.py --config configs/xreal_lotr.yaml --checkpoint work_dirs/xreal_lotr/best.pth --image path/to/image.jpg
    python scripts/infer_lotr.py --config configs/xreal_lotr.yaml --checkpoint work_dirs/xreal_lotr/best.pth --video path/to/video.mp4
"""

import argparse
import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pose.config import Config
from pose.registry import MODEL_REGISTRY
import pose.models  # Register models


def load_model(config_path: str, checkpoint_path: str, device: str = "cuda"):
    """Load LOTR model from config and checkpoint."""
    cfg = Config.from_yaml(config_path).raw
    
    model_cfg = cfg["model"]
    ModelCls = MODEL_REGISTRY[model_cfg["name"]]
    
    # Get number of keypoints from dataset config (or default to 68)
    num_keypoints = cfg.get("data", {}).get("num_keypoints", 68)
    
    model_kwargs = {"num_keypoints": num_keypoints}
    for k, v in model_cfg.items():
        if k == "name":
            continue
        model_kwargs[k] = v
    
    # Add input_size from config
    if "input_size" in cfg.get("data", {}):
        model_kwargs["input_size"] = tuple(cfg["data"]["input_size"])
    
    model = ModelCls(**model_kwargs)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    
    model = model.to(device)
    model.eval()
    
    return model, cfg


def preprocess_image(image: np.ndarray, input_size: tuple) -> torch.Tensor:
    """Preprocess image for LOTR model."""
    # Resize
    img = cv2.resize(image, input_size)
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize with ImageNet stats
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    
    # Convert to tensor (C, H, W)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0)  # Add batch dim
    
    return img


def draw_landmarks(image: np.ndarray, landmarks: np.ndarray, 
                   color: tuple = (0, 255, 0), radius: int = 2) -> np.ndarray:
    """Draw landmarks on image."""
    img = image.copy()
    
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(img, (int(x), int(y)), radius, color, -1)
        
        # Optionally add landmark index for debugging
        # cv2.putText(img, str(i), (int(x)+2, int(y)+2), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # Draw face contour connections (68-point)
    # Jaw line
    for i in range(16):
        pt1 = tuple(landmarks[i].astype(int))
        pt2 = tuple(landmarks[i+1].astype(int))
        cv2.line(img, pt1, pt2, color, 1)
    
    # Eyebrows
    for i in range(17, 21):
        pt1 = tuple(landmarks[i].astype(int))
        pt2 = tuple(landmarks[i+1].astype(int))
        cv2.line(img, pt1, pt2, color, 1)
    for i in range(22, 26):
        pt1 = tuple(landmarks[i].astype(int))
        pt2 = tuple(landmarks[i+1].astype(int))
        cv2.line(img, pt1, pt2, color, 1)
    
    # Nose
    for i in range(27, 30):
        pt1 = tuple(landmarks[i].astype(int))
        pt2 = tuple(landmarks[i+1].astype(int))
        cv2.line(img, pt1, pt2, color, 1)
    for i in range(31, 35):
        pt1 = tuple(landmarks[i].astype(int))
        pt2 = tuple(landmarks[i+1].astype(int))
        cv2.line(img, pt1, pt2, color, 1)
    
    # Eyes
    for i in range(36, 41):
        pt1 = tuple(landmarks[i].astype(int))
        pt2 = tuple(landmarks[i+1].astype(int))
        cv2.line(img, pt1, pt2, color, 1)
    cv2.line(img, tuple(landmarks[36].astype(int)), 
             tuple(landmarks[41].astype(int)), color, 1)
    
    for i in range(42, 47):
        pt1 = tuple(landmarks[i].astype(int))
        pt2 = tuple(landmarks[i+1].astype(int))
        cv2.line(img, pt1, pt2, color, 1)
    cv2.line(img, tuple(landmarks[42].astype(int)), 
             tuple(landmarks[47].astype(int)), color, 1)
    
    # Mouth outer
    for i in range(48, 59):
        pt1 = tuple(landmarks[i].astype(int))
        pt2 = tuple(landmarks[i+1].astype(int))
        cv2.line(img, pt1, pt2, color, 1)
    cv2.line(img, tuple(landmarks[48].astype(int)), 
             tuple(landmarks[59].astype(int)), color, 1)
    
    # Mouth inner
    for i in range(60, 67):
        pt1 = tuple(landmarks[i].astype(int))
        pt2 = tuple(landmarks[i+1].astype(int))
        cv2.line(img, pt1, pt2, color, 1)
    cv2.line(img, tuple(landmarks[60].astype(int)), 
             tuple(landmarks[67].astype(int)), color, 1)
    
    return img


def infer_image(model, image: np.ndarray, input_size: tuple, 
                device: str = "cuda") -> np.ndarray:
    """Run inference on a single image."""
    H_orig, W_orig = image.shape[:2]
    
    # Preprocess
    img_tensor = preprocess_image(image, input_size).to(device)
    
    # Run model
    with torch.no_grad():
        landmarks_norm, landmarks_pixel = model(img_tensor)
    
    # Get pixel coordinates
    landmarks = landmarks_pixel[0].cpu().numpy()  # (N, 2)
    
    # Scale to original image size
    scale_x = W_orig / input_size[0]
    scale_y = H_orig / input_size[1]
    landmarks[:, 0] *= scale_x
    landmarks[:, 1] *= scale_y
    
    return landmarks


def process_image(args):
    """Process a single image."""
    model, cfg = load_model(args.config, args.checkpoint, args.device)
    input_size = tuple(cfg["data"]["input_size"])
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image {args.image}")
        return
    
    # Infer
    landmarks = infer_image(model, image, input_size, args.device)
    
    # Draw results
    result = draw_landmarks(image, landmarks)
    
    # Save or display
    if args.output:
        cv2.imwrite(args.output, result)
        print(f"Saved result to {args.output}")
    else:
        cv2.imshow("LOTR Landmarks", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(args):
    """Process video file."""
    model, cfg = load_model(args.config, args.checkpoint, args.device)
    input_size = tuple(cfg["data"]["input_size"])
    
    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup output
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Infer
        landmarks = infer_image(model, frame, input_size, args.device)
        
        # Draw
        result = draw_landmarks(frame, landmarks)
        
        # Show FPS
        cv2.putText(result, f"Frame: {frame_idx}/{total_frames}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if args.output:
            out.write(result)
        else:
            cv2.imshow("LOTR Landmarks", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
    
    cap.release()
    if args.output:
        out.release()
        print(f"Saved result to {args.output}")
    else:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="LOTR Landmark Detection Inference")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--image", help="Path to input image")
    parser.add_argument("--video", help="Path to input video")
    parser.add_argument("--output", "-o", help="Path to save output")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    if args.image:
        process_image(args)
    elif args.video:
        process_video(args)
    else:
        print("Please specify --image or --video")
        

if __name__ == "__main__":
    main()
