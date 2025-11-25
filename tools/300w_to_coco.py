import os
import json
import random
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat

def find_image_for_mat(mat_path):
    """Given path/to/xxx.mat, try to find xxx.jpg / xxx.png / xxx.jpeg in same dir."""
    base = os.path.splitext(mat_path)[0]
    for ext in [".jpg", ".png", ".jpeg", ".bmp"]:
        img_path = base + ext
        if os.path.exists(img_path):
            return img_path
    return None

def convert_300w3d(root, out_dir, val_ratio=0.1, seed=42):
    """
    root: path to 300W-3D (folder containing AFW, HELEN, LFPW, IBUG, Code)
    out_dir: where to save annotations/*.json
    val_ratio: fraction of images for validation
    """
    os.makedirs(out_dir, exist_ok=True)
    annotations = []
    images = []
    categories = [{
        "id": 1,
        "name": "face",
        "supercategory": "person",
        "keypoints": [f"kpt_{i}" for i in range(68)],
        "skeleton": []
    }]

    img_id = 1
    ann_id = 1

    # Walk through AFW, HELEN, LFPW, IBUG (skip Code)
    for sub in ["AFW", "HELEN", "LFPW", "IBUG"]:
        sub_dir = os.path.join(root, sub)
        if not os.path.isdir(sub_dir):
            continue

        for dirpath, dirnames, filenames in os.walk(sub_dir):
            for fname in filenames:
                if not fname.lower().endswith(".mat"):
                    continue

                mat_path = os.path.join(dirpath, fname)
                img_path = find_image_for_mat(mat_path)
                if img_path is None:
                    # No matching image file
                    continue

                # Load image for dimensions
                img = cv2.imread(img_path)
                if img is None:
                    continue
                h, w = img.shape[:2]

                # Relative path from root (so image_root can be root)
                rel_img_path = os.path.relpath(img_path, root).replace("\\", "/")

                # Load .mat and extract pt2d (2 x 68)
                mat = loadmat(mat_path)
                if "pt2d" not in mat:
                    continue

                pt2d = mat["pt2d"]  # shape (2, 68): [x; y]
                if pt2d.shape != (2, 68):
                    # Unexpected shape, skip
                    continue

                xs = pt2d[0, :]
                ys = pt2d[1, :]

                keypoints = []
                for x, y in zip(xs, ys):
                    v = 2  # visible
                    keypoints.extend([float(x), float(y), v])

                x_min, x_max = float(xs.min()), float(xs.max())
                y_min, y_max = float(ys.min()), float(ys.max())
                bbox_w = x_max - x_min
                bbox_h = y_max - y_min
                if bbox_w <= 0 or bbox_h <= 0:
                    continue

                images.append({
                    "id": img_id,
                    "file_name": rel_img_path,
                    "width": int(w),
                    "height": int(h)
                })

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "keypoints": keypoints,
                    "num_keypoints": 68,
                    "bbox": [x_min, y_min, bbox_w, bbox_h],
                    "area": bbox_w * bbox_h,
                    "iscrowd": 0
                })

                img_id += 1
                ann_id += 1

    # Shuffle and split into train/val
    random.seed(seed)
    idxs = list(range(len(images)))
    random.shuffle(idxs)

    n_val = int(len(images) * val_ratio)
    val_idxs = set(idxs[:n_val])

    def build_coco(selected_idxs):
        img_map = {}
        anns = []
        new_images = []
        new_img_id = 1
        new_ann_id = 1

        # Build mapping from old img_id to new img_id
        for i in selected_idxs:
            old_img = images[i]
            old_id = old_img["id"]
            img_map[old_id] = new_img_id
            new_images.append({
                "id": new_img_id,
                "file_name": old_img["file_name"],
                "width": old_img["width"],
                "height": old_img["height"]
            })
            new_img_id += 1

        for ann in annotations:
            old_image_id = ann["image_id"]
            if old_image_id in img_map:
                anns.append({
                    "id": new_ann_id,
                    "image_id": img_map[old_image_id],
                    "category_id": ann["category_id"],
                    "keypoints": ann["keypoints"],
                    "num_keypoints": ann["num_keypoints"],
                    "bbox": ann["bbox"],
                    "area": ann["area"],
                    "iscrowd": ann["iscrowd"]
                })
                new_ann_id += 1

        return {
            "images": new_images,
            "annotations": anns,
            "categories": categories
        }

    all_idxs = list(range(len(images)))
    val_indices = sorted(list(val_idxs))
    train_indices = sorted(list(set(all_idxs) - val_idxs))

    coco_train = build_coco(train_indices)
    coco_val = build_coco(val_indices)
    coco_all = build_coco(all_idxs)

    with open(os.path.join(out_dir, "train.json"), "w") as f:
        json.dump(coco_train, f, indent=2)
    with open(os.path.join(out_dir, "val.json"), "w") as f:
        json.dump(coco_val, f, indent=2)
    with open(os.path.join(out_dir, "all.json"), "w") as f:
        json.dump(coco_all, f, indent=2)

    print(f"Total images: {len(images)}")
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")
    print("Saved:", os.path.join(out_dir, "train.json"))
    print("Saved:", os.path.join(out_dir, "val.json"))
    print("Saved:", os.path.join(out_dir, "all.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Path to 300W-3D root (contains AFW/HELEN/LFPW/IBUG)")
    parser.add_argument("--out", type=str, required=True,
                        help="Output directory for annotations json")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fraction of data used for validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    convert_300w3d(
        root=args.root,
        out_dir=args.out,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
