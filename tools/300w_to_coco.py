import os
import json
import random
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat


# -------------------------------------------------------
# Utilities
# -------------------------------------------------------

def find_image_for_mat(mat_path):
    """
    Given path/to/xxx.mat, try to find xxx.jpg / xxx.png / xxx.jpeg in same dir.
    """
    base = os.path.splitext(mat_path)[0]
    for ext in [".jpg", ".png", ".jpeg", ".bmp"]:
        img_path = base + ext
        if os.path.exists(img_path):
            return img_path
    return None


# -------------------------------------------------------
# 3DMM loading + reconstruction
# -------------------------------------------------------

def load_3dmm_model(shape_model_path, exp_model_path):
    """
    Loads the 3DMM used by 300W-3D from:
      - Model_Shape_Sim.mat
      - Model_Exp.mat

    Expected contents (from your files):
      Model_Shape_Sim.mat:
        - mu_shape  : (159645, 1)  = 3 * N vertices
        - tri       : (3, 105840)  (unused here)
        - keypoints : (1, 68)      landmark vertex indices (MATLAB 1-based)

      Model_Exp.mat:
        - mu_exp    : (159645, 1)  (unused here)
        - w_exp     : (159645, 29) expression basis
        - sigma_exp : (29, 1)      standard deviations of PCA components
    """
    shape_data = loadmat(shape_model_path)
    exp_data = loadmat(exp_model_path)

    mu_shape = np.asarray(shape_data["mu_shape"])        # (159645, 1)
    keypoints = np.squeeze(shape_data["keypoints"])      # (68,) MATLAB 1-based

    # Convert keypoints to 0-based indexing for Python
    landmark_indices = keypoints.astype(np.int64) - 1    # (68,)

    w_exp = np.asarray(exp_data["w_exp"])                # (159645, 29)
    sigma_exp = np.asarray(exp_data["sigma_exp"])        # (29, 1)

    # Basic sanity checks (optional)
    assert mu_shape.ndim == 2 and mu_shape.shape[1] == 1, "mu_shape unexpected shape"
    assert w_exp.shape[0] == mu_shape.shape[0], "w_exp and mu_shape dimension mismatch"
    assert sigma_exp.shape[0] == w_exp.shape[1], "sigma_exp and w_exp dimension mismatch"

    model = {
        "mu_shape": mu_shape,            # (3N, 1)
        "w_exp": w_exp,                  # (3N, 29)
        "sigma_exp": sigma_exp,          # (29, 1)
        "landmark_indices": landmark_indices
    }
    return model


def reconstruct_3d_landmarks(exp_para, model):
    """
    Reconstruct 3D landmarks (68 x 3) from 3DMM expression parameters.

    exp_para : Exp_Para from 300W-3D .mat, shape (29, 1) or (29,)
    model    : dict from load_3dmm_model()

    Generation model used:
        shape_vec = mu_shape + w_exp @ (exp_para * sigma_exp)
        where all are in vectorized (3N, 1) form.
    """

    mu_shape = model["mu_shape"]          # (3N, 1)
    w_exp = model["w_exp"]                # (3N, 29)
    sigma_exp = model["sigma_exp"]        # (29, 1)
    landmark_indices = model["landmark_indices"]  # (68,)

    exp_para = np.asarray(exp_para, dtype=np.float64)
    exp_para = np.squeeze(exp_para)       # (29,)
    if exp_para.ndim == 0:
        exp_para = np.array([exp_para], dtype=np.float64)
    assert exp_para.shape[0] == sigma_exp.shape[0], "Exp_Para dim mismatch"

    # Scale by PCA std
    coeff = exp_para * np.squeeze(sigma_exp)  # (29,)

    # Full 3N x 1 displacement
    disp = w_exp @ coeff.reshape(-1, 1)       # (3N, 1)

    # Add to mean shape
    shape_vec = mu_shape + disp               # (3N, 1)

    # Reshape to (N, 3)
    vertices = shape_vec.reshape(-1, 3)       # (N, 3)

    # Extract 68 landmarks
    lm = vertices[landmark_indices]           # (68, 3)
    return lm


# -------------------------------------------------------
# Main conversion function
# -------------------------------------------------------

def convert_300w3d(root, out_dir,
                   shape_model_path, exp_model_path,
                   val_ratio=0.1, seed=42):
    """
    root: path to 300W-3D (contains AFW, HELEN, LFPW, IBUG, Code)
    out_dir: where to save annotations/*.json
    shape_model_path: path to Model_Shape_Sim.mat
    exp_model_path: path to Model_Exp.mat
    val_ratio: fraction of images for validation
    seed: random seed for train/val split
    """

    os.makedirs(out_dir, exist_ok=True)

    # Load 3DMM model once
    model_3dmm = load_3dmm_model(shape_model_path, exp_model_path)

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
                    continue

                # Load image to get width/height
                img = cv2.imread(img_path)
                if img is None:
                    continue
                h, w = img.shape[:2]

                rel_img_path = os.path.relpath(img_path, root).replace("\\", "/")

                # Load landmark + parameter .mat
                mat = loadmat(mat_path)

                # ----------------- 2D keypoints -----------------
                if "pt2d" not in mat:
                    continue
                pt2d = mat["pt2d"]   # (2, 68)

                if pt2d.shape != (2, 68):
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

                # ----------------- Store image entry -----------------
                images.append({
                    "id": img_id,
                    "file_name": rel_img_path,
                    "width": int(w),
                    "height": int(h)
                })

                ann = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "keypoints": keypoints,
                    "num_keypoints": 68,
                    "bbox": [x_min, y_min, bbox_w, bbox_h],
                    "area": bbox_w * bbox_h,
                    "iscrowd": 0,
                }

                annotations.append(ann)

                img_id += 1
                ann_id += 1

    # -------------------------------------------------------
    # Shuffle & split into train / val and rebuild IDs
    # -------------------------------------------------------

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

        # Map old image ids to new
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

        # Remap annotations
        for ann in annotations:
            old_image_id = ann["image_id"]
            if old_image_id in img_map:
                new_ann = dict(ann)
                new_ann["id"] = new_ann_id
                new_ann["image_id"] = img_map[old_image_id]
                anns.append(new_ann)
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


# -------------------------------------------------------
# CLI
# -------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, required=True,
        help="Path to 300W-3D root (contains AFW/HELEN/LFPW/IBUG)"
    )
    parser.add_argument(
        "--out", type=str, required=True,
        help="Output directory for annotations json"
    )
    parser.add_argument(
        "--shape-model", type=str, required=True,
        help="Path to Model_Shape_Sim.mat"
    )
    parser.add_argument(
        "--exp-model", type=str, required=True,
        help="Path to Model_Exp.mat"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1,
        help="Fraction of data used for validation"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val split"
    )
    args = parser.parse_args()

    convert_300w3d(
        root=args.root,
        out_dir=args.out,
        shape_model_path=args.shape_model,
        exp_model_path=args.exp_model,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
