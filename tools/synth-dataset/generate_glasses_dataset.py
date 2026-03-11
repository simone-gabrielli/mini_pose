"""
generate_glasses_dataset.py

Synthetic augmentation: render 3D glasses on top of faces (e.g. 300W-LP)
using a weak-perspective (scaled orthographic) projection.

Pipeline:
    1. Read COCO-style annotations with face keypoints.
    2. For each face:
        - build 3D↔2D correspondences using GLASSES_TO_FACE_IDXS
        - estimate weak-perspective pose (R, s, t) via Procrustes
        - project all 3D glasses keypoints to 2D
        - call Blender to render glasses RGBA roughly aligned
        - composite onto the original RGB image
        - write a new image + annotation with extra keypoints
"""

import argparse
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


# -------------------------------------------------------------------------
# CONFIG: 3D GLASSES LANDMARKS + MAPPING
# -------------------------------------------------------------------------

import xml.etree.ElementTree as ET

# Mapping: glasses landmark id -> face landmark id
GLASSES_TO_FACE_IDXS = {
    17: 19,
    18: 20,
    19: 21,
    20: 22,
    21: 23,
    22: 28,
    23: 29,
    24: 30,
    25: 31,
    26: 32,
    0: 45,
    16: 51,
}

def load_glasses_3d_landmarks_from_xml(path):
    """
    Reads the glasses 3D CAD keypoints from the provided XML file.
    Returns:
        arr: (N,3) float32 array where arr[i] stores xyz for landmark i
    """

    tree = ET.parse(path)
    root = tree.getroot()

    landmarks_node = root.find("landmarks")
    if landmarks_node is None:
        raise RuntimeError("XML missing <landmarks> root node")

    # First determine how many landmarks exist (we expect ids 0..67)
    ids = []
    for lm in landmarks_node.findall("landmark"):
        idx = int(lm.attrib["id"])
        ids.append(idx)

    max_id = max(ids)
    arr = np.zeros((max_id + 1, 3), dtype=np.float32)

    for lm in landmarks_node.findall("landmark"):
        idx = int(lm.attrib["id"])
        x = float(lm.attrib["x"])
        y = float(lm.attrib["y"])
        z = float(lm.attrib["z"])
        arr[idx] = np.array([x, y, z], dtype=np.float32)

    return arr

# python generate_glasses_dataset.py   --images_root "C:\Users\simog\hmdrive\head_tracking/mini_pose/datasets/300W-3D/"   --annotations "C:\Users\simog\hmdrive\head_tracking/mini_pose/datasets/300W-3D/annotations/val.json"  --out_images "C:\Users\simog\hmdrive\head_tracking\dlib_custom\synth-300w\imgs_out\"   --out_annotations "C:\Users\simog\hmdrive\head_tracking\dlib_custom\synth-300w\annotations_glasses.json"  --blender_exec "C:/Program Files/Blender Foundation/Blender 4.4/blender.exe"   --blender_script blender_render_glasses.py   --glasses_obj "C:\Users\simog\hmdrive\head_tracking\dlib_custom/data/glasses_models/glasses.fbx" --glasses_xml "C:\Users\simog\hmdrive\head_tracking\dlib_custom\data/glasses_models/xreal_one_68_v3.xml"  --num_face_keypoints 68   --variants_per_image 

# Load the 3D CAD model
GLASSES_3D_LANDMARKS = None

# These are the glasses KP we want to output in the final dataset
GLASSES_KEYPOINT_IDXS = sorted(GLASSES_TO_FACE_IDXS.keys())


def _check_glasses_3d():
    # If all points are zero, you forgot to set real CAD coordinates.
    if np.allclose(GLASSES_3D_LANDMARKS, 0):
        raise RuntimeError(
            "GLASSES_3D_LANDMARKS is all zeros. "
            "Fill it with your real CAD 3D keypoints before using this script."
        )


# -------------------------------------------------------------------------
# ARGUMENTS
# -------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_root", type=str, required=True)
    parser.add_argument("--annotations", type=str, required=True)
    parser.add_argument("--out_images", type=str, required=True)
    parser.add_argument("--out_annotations", type=str, required=True)
    parser.add_argument("--blender_exec", type=str, required=True)
    parser.add_argument("--blender_script", type=str, required=True)
    parser.add_argument("--glasses_obj", type=str, required=True) # Glasses 3D model
    parser.add_argument("--glasses_xml", type=str, required=True) #Keypoints

    # Kept for backwards compatibility, NOT used in weak-perspective math.
    parser.add_argument("--focal_length_mm", type=float, default=35.0)
    parser.add_argument("--sensor_width_mm", type=float, default=36.0)

    # Size of the glasses render (we keep it equal to original image size).
    parser.add_argument("--render_width", type=int, default=256)
    parser.add_argument("--render_height", type=int, default=256)

    # Number of synthetic variants per original image
    parser.add_argument("--variants_per_image", type=int, default=1)

    # Number of face keypoints in the original dataset (e.g. 68 for 300W-LP).
    parser.add_argument("--num_face_keypoints", type=int, default=68)

    return parser.parse_args()


# -------------------------------------------------------------------------
# GEOMETRY / LANDMARK UTILS
# -------------------------------------------------------------------------


def get_face_landmarks_from_annotation(ann, num_kpts_face=None):
    """
    Extract 2D face landmarks from COCO-style annotation.
    ann["keypoints"] = [x1,y1,v1, x2,y2,v2, ...]
    Returns:
        landmarks: (K,2) float32 array in image coords.
    """
    kpts = ann["keypoints"]
    kpts = np.asarray(kpts, dtype=np.float32)
    if kpts.size % 3 != 0:
        raise ValueError("Annotation keypoints length is not multiple of 3")

    kpts = kpts.reshape(-1, 3)  # (K,3)
    if num_kpts_face is not None:
        kpts = kpts[:num_kpts_face]

    return kpts[:, :2]


def estimate_glasses_pose_weak(landmarks_2d):
    """
    Estimate weak-perspective parameters (R, s, t) aligning CAD 3D glasses
    keypoints to 2D face landmarks.

    Model:
        x_i = s * (R_2x3 X_i) + t
    where:
        X_i ∈ R^3  (CAD frame)
        R ∈ SO(3)
        s ∈ R+
        t ∈ R^2

    We use Procrustes / Umeyama on (3D, 2D-embedded-in-3D).

    Returns:
        R: (3,3) float32
        s: float
        t: (2,) float32
    """
    _check_glasses_3d()

    obj_pts = []
    img_pts = []

    for g_idx, face_idx in GLASSES_TO_FACE_IDXS.items():
        if face_idx >= landmarks_2d.shape[0]:
            continue
        obj_pts.append(GLASSES_3D_LANDMARKS[g_idx])
        img_pts.append(landmarks_2d[face_idx])

    if len(obj_pts) < 3:
        # Not enough constraints; fall back to identity.
        R = np.eye(3, dtype=np.float32)
        s = 1.0
        t = landmarks_2d.mean(axis=0).astype(np.float32)
        return R, s, t

    X = np.asarray(obj_pts, dtype=np.float32)  # (N,3)
    Y = np.asarray(img_pts, dtype=np.float32)  # (N,2)

    # Center
    mu_X = X.mean(axis=0, keepdims=True)
    mu_Y = Y.mean(axis=0, keepdims=True)
    Xc = X - mu_X
    Yc = Y - mu_Y

    # Embed Y into 3D for SVD (add zero z)
    Y3 = np.hstack([Yc, np.zeros((Yc.shape[0], 1), dtype=np.float32)])  # (N,3)

    H = Xc.T @ Y3  # (3,3)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure det(R) = +1
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    var_X = (Xc ** 2).sum()
    s = float(S.sum() / var_X)

    # Translation in 2D
    # R * mu_X is 3D; we take first two components.
    t_2d = (mu_Y[0] - s * (R @ mu_X[0])[:2]).astype(np.float32)

    return R.astype(np.float32), s, t_2d


def project_glasses_keypoints_weak(R, s, t, kp_indices=None, visibility=2):
    """
    Project 3D CAD glasses keypoints via weak-perspective to 2D.

    Args:
        R: (3,3)
        s: float
        t: (2,)
        kp_indices: iterable of indices to export; default = GLASSES_KEYPOINT_IDXS.
        visibility: COCO-style visibility flag (0,1,2). 2 = visible.

    Returns:
        list of [x, y, v] in the order of kp_indices.
    """
    if kp_indices is None:
        kp_indices = GLASSES_KEYPOINT_IDXS

    kpts = []
    for idx in kp_indices:
        X = GLASSES_3D_LANDMARKS[idx]  # (3,)
        RX = R[:2, :] @ X              # (2,)
        x2d = s * RX + t               # (2,)
        kpts.append([float(x2d[0]), float(x2d[1]), visibility])

    return kpts


def rotation_matrix_to_euler_xyz(R):
    """
    Convert 3x3 rotation matrix to XYZ Euler angles (radians).
    We use a standard convention; for Blender we only need consistency
    between here and apply_pose() in the Blender script.
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(-R[2, 0], sy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(-R[2, 0], sy)
        rz = 0.0

    return rx, ry, rz


# -------------------------------------------------------------------------
# BLENDER CALL + COMPOSITING
# -------------------------------------------------------------------------


def call_blender_render(
    blender_exec,
    blender_script,
    glasses_obj,
    tmp_out_path,
    image_width,
    image_height,
    center_xy,
    scale_px,
    roll,
    pitch,
    yaw,
):
    cmd = [
        blender_exec,
        "--background",
        "--python", blender_script,
        "--",
        "--model", glasses_obj,
        "--out", tmp_out_path,
        "--width", str(image_width),
        "--height", str(image_height),
        "--pos_x", str(center_xy[0]),
        "--pos_y", str(center_xy[1]),
        "--glasses_scale", str(scale_px),
        "--rot_roll", str(roll),
        "--rot_pitch", str(pitch),
        "--rot_yaw", str(yaw),
    ]

    print("Running Blender command:")
    print(cmd)

    subprocess.run(cmd, check=True)


def composite_glasses_on_image(bg_img_path, glasses_rgba_path, out_img_path):
    """
    Alpha-composite RGBA glasses render onto RGB background image.
    """
    bg = Image.open(bg_img_path).convert("RGB")
    gl = Image.open(glasses_rgba_path).convert("RGBA")

    if gl.size != bg.size:
        gl = gl.resize(bg.size, Image.BICUBIC)

    bg_rgba = bg.convert("RGBA")
    bg_rgba.alpha_composite(gl)
    out = bg_rgba.convert("RGB")
    out.save(out_img_path, quality=95)


# -------------------------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------------------------


def main():
    args = parse_args()
    global GLASSES_3D_LANDMARKS
    GLASSES_3D_LANDMARKS = load_glasses_3d_landmarks_from_xml(args.glasses_xml)

    _check_glasses_3d()

    images_root = Path(args.images_root)
    out_images_root = Path(args.out_images)
    out_images_root.mkdir(parents=True, exist_ok=True)
    tmp_rgba_dir = out_images_root / "_tmp_glasses_rgba"
    tmp_rgba_dir.mkdir(parents=True, exist_ok=True)

    with open(args.annotations, "r") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco.get("categories", [])

    # Optional: extend categories to encode new number of keypoints.
    # Here we assume a single face category and we just bump num_keypoints.
    num_extra_kpts = len(GLASSES_KEYPOINT_IDXS)
    for cat in categories:
        if "keypoints" in cat and "num_keypoints" in cat:
            cat["num_keypoints"] = cat.get("num_keypoints", args.num_face_keypoints) + num_extra_kpts

    image_by_id = {img["id"]: img for img in images}

    next_image_id = max(img["id"] for img in images) + 1 if images else 1
    next_ann_id = max(ann["id"] for ann in annotations) + 1 if annotations else 1

    new_images = []
    new_annotations = []

    for ann in annotations:
        img = image_by_id.get(ann["image_id"])
        if img is None:
            continue

        img_path = images_root / img["file_name"]
        if not img_path.is_file():
            print(f"Warning: image not found: {img_path}")
            continue

        # landmarks_2d: only original face keypoints
        landmarks_2d = get_face_landmarks_from_annotation(
            ann, num_kpts_face=args.num_face_keypoints
        )

        try:
            R, s, t = estimate_glasses_pose_weak(landmarks_2d)
        except Exception as e:
            print(f"Pose estimation failed for image {img_path}: {e}")
            continue

        # Project all glasses keypoints
        glasses_kpts = project_glasses_keypoints_weak(R, s, t)

        # Approximate glasses center and scale in pixels from projected kpts
        g_arr = np.asarray(glasses_kpts, dtype=np.float32)  # (G,3)
        center_xy = g_arr[:, :2].mean(axis=0)
        xs = g_arr[:, 0]
        ys = g_arr[:, 1]
        width = float(xs.max() - xs.min())
        height = float(ys.max() - ys.min())
        scale_px = max(width, height)

        # Euler angles from R; map to Blender's (roll,pitch,yaw) convention
        rx, ry, rz = rotation_matrix_to_euler_xyz(R)
        pitch = rx
        yaw = ry
        roll = rz

        # Repeat with slight jitter if requested
        for v in range(args.variants_per_image):
            if args.variants_per_image > 1:
                # Small random jitter in rotation and scale
                jitter_r = (np.random.randn(3) * np.deg2rad(3.0)).astype(np.float32)
                jitter_s = float(1.0 + 0.05 * np.random.randn())
                roll_j = roll + float(jitter_r[2])
                pitch_j = pitch + float(jitter_r[0])
                yaw_j = yaw + float(jitter_r[1])
                scale_j = scale_px * jitter_s
            else:
                roll_j, pitch_j, yaw_j, scale_j = roll, pitch, yaw, scale_px

            tmp_rgba_path = tmp_rgba_dir / f"glasses_{ann['id']}_v{v}.png"
            out_img_name = f"{Path(img['file_name']).stem}_glasses_{ann['id']}_v{v}.jpg"
            out_img_path = out_images_root / out_img_name

            # 1) Render glasses only (RGBA)
            call_blender_render(
                blender_exec=args.blender_exec,
                blender_script=args.blender_script,
                glasses_obj=args.glasses_obj,
                tmp_out_path=str(tmp_rgba_path),
                image_width=img["width"],
                image_height=img["height"],
                center_xy=center_xy,
                scale_px=scale_j,
                roll=roll_j,
                pitch=pitch_j,
                yaw=yaw_j,
            )

            # 2) Composite onto background
            composite_glasses_on_image(
                bg_img_path=str(img_path),
                glasses_rgba_path=str(tmp_rgba_path),
                out_img_path=str(out_img_path),
            )

            # 3) Create new COCO image entry
            new_img = {
                "id": next_image_id,
                "file_name": out_img_name,
                "width": img["width"],
                "height": img["height"],
            }
            new_images.append(new_img)

            # 4) Merge original + glasses kpts into COCO style
            orig_kpts = np.asarray(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
            all_kpts = orig_kpts.tolist() + glasses_kpts
            all_kpts_flat = [coord for kp in all_kpts for coord in kp]

            new_ann = {
                "id": next_ann_id,
                "image_id": next_image_id,
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "iscrowd": ann.get("iscrowd", 0),
                "num_keypoints": int(orig_kpts.shape[0] + len(glasses_kpts)),
                "keypoints": all_kpts_flat,
            }
            new_annotations.append(new_ann)

            next_image_id += 1
            next_ann_id += 1

    # Append synthetic entries to the original COCO dict
    coco["images"].extend(new_images)
    coco["annotations"].extend(new_annotations)

    with open(args.out_annotations, "w") as f:
        json.dump(coco, f)

    print(f"Saved augmented annotations to: {args.out_annotations}")
    print(f"Synthetic images written under: {out_images_root}")


if __name__ == "__main__":
    main()
