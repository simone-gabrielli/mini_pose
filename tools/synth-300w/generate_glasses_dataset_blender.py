#!/usr/bin/env python3
"""
generate_glasses_dataset_blender.py

Blender 4.4 version of the pyrender-based glasses overlay generator.

This script keeps the SAME math and conventions as the working pyrender script:
- weak-perspective pose estimation (R, s, t) and projection
- OpenCV solvePnP full-perspective pose and projection

The ONLY change is the renderer:
- instead of pyrender.OffscreenRenderer, we render the 3D model with Blender (4.4)
  in background mode (-b) and save an RGBA PNG, then composite it on the face image.

To keep Blender equivalent to pyrender:
- ORTHO (weak) rendering uses a fixed orthographic camera and an object matrix that
  reproduces pixel mapping; we include a Blender-specific orthographic-aspect fix.
- PERSP (PnP) rendering uses a perspective camera whose lens + shift parameters are
  solved so the principal point matches K exactly (via a numeric derivative inside Blender).
  Camera pose uses the same OpenCV->OpenGL/Blender axis conversion as your pyrender path.

Files needed:
- this script
- blender_render_jobs_44.py (must be in the same folder, unless you pass --blender_script)

Usage example:

  python generate_glasses_dataset_blender.py \
    --images_root /path/to/images \
    --annotations /path/to/coco.json \
    --out_images /path/to/out_images \
    --out_annotations /path/to/out_coco.json \
    --glasses_obj /path/to/glasses.obj \
    --glasses_xml /path/to/glasses_landmarks.xml \
    --blender_bin /path/to/blender \
    --ssaa 2 \
    --engine BLENDER_EEVEE_NEXT \
    --samples 64 \
    --debug_landmarks

Dependencies (outside Blender):
  pip install numpy pillow opencv-python trimesh

Blender must be 4.4+ (you said 4.4). Run `blender --version` to confirm.
"""

import argparse
import json
import math
import os
import subprocess
import tempfile
import uuid
import zlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import trimesh


# -------------------------------------------------------------------------
# CONFIG: 3D GLASSES LANDMARKS + MAPPING
# -------------------------------------------------------------------------

# Mapping: glasses landmark id -> face landmark id
GLASSES_TO_FACE_IDXS = {
    19: 26,
    20: 25,
    21: 24,
    22: 23,
    23: 22,
    28: 21,
    29: 20,
    30: 19,
    31: 18,
    32: 17,
    45: 16,
    51: 0,
}

GLASSES_3D_LANDMARKS = None
GLASSES_3D_LANDMARKS_DEFINED = None
GLASSES_KEYPOINT_IDXS = sorted(GLASSES_TO_FACE_IDXS.keys())


def load_glasses_3d_landmarks_from_xml(path):
    """Read glasses 3D CAD keypoints from XML.

    Returns:
      arr: (N,3) float32, arr[i] is xyz for landmark i
      defined: (N,) bool, True where that id exists in the XML
    """
    tree = ET.parse(path)
    root = tree.getroot()
    landmarks_node = root.find("landmarks")
    if landmarks_node is None:
        raise RuntimeError("XML missing <landmarks> root node")

    ids = []
    for lm in landmarks_node.findall("landmark"):
        ids.append(int(lm.attrib["id"]))
    if not ids:
        raise RuntimeError("No <landmark> nodes found in XML")

    max_id = max(ids)
    arr = np.zeros((max_id + 1, 3), dtype=np.float32)
    defined = np.zeros((max_id + 1,), dtype=bool)

    for lm in landmarks_node.findall("landmark"):
        idx = int(lm.attrib["id"])
        x = float(lm.attrib["x"])
        y = float(lm.attrib["y"])
        z = float(lm.attrib["z"])
        arr[idx] = np.array([x, y, z], dtype=np.float32)
        defined[idx] = True

    return arr, defined


def _check_glasses_3d():
    global GLASSES_3D_LANDMARKS, GLASSES_3D_LANDMARKS_DEFINED
    if GLASSES_3D_LANDMARKS is None or GLASSES_3D_LANDMARKS_DEFINED is None:
        raise RuntimeError("GLASSES_3D_LANDMARKS not loaded; call load_glasses_3d_landmarks_from_xml first")
    if np.allclose(GLASSES_3D_LANDMARKS, 0):
        raise RuntimeError("GLASSES_3D_LANDMARKS is all zeros; fill with real CAD keypoints")


# -------------------------------------------------------------------------
# ARGUMENTS
# -------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--images_root", type=str, required=True)
    p.add_argument("--annotations", type=str, required=True)
    p.add_argument("--out_images", type=str, required=True)
    p.add_argument("--out_annotations", type=str, required=True)

    # 3D resources
    p.add_argument("--glasses_obj", type=str, required=True, help="OBJ or FBX model path")
    p.add_argument("--glasses_xml", type=str, required=True)

    # Blender
    p.add_argument("--blender_bin", type=str, default="blender", help="Path to Blender 4.4 executable")
    p.add_argument("--blender_script", type=str, default=None, help="Path to blender_render_jobs_44.py (default: next to this script)")
    p.add_argument("--engine", type=str, default="BLENDER_EEVEE_NEXT", help="BLENDER_EEVEE_NEXT or CYCLES")
    p.add_argument("--samples", type=int, default=64)

    # Quality controls
    p.add_argument("--ssaa", type=int, default=2, help="Render at N× resolution and downsample. 1 disables.")
    p.add_argument("--ambient", type=float, default=0.35)
    p.add_argument("--key_light", type=float, default=0.2)
    p.add_argument("--fill_light", type=float, default=0.5)

    # Lighting randomization (Blender-only augmentation)
    p.add_argument(
        "--randomize_lights",
        action="store_true",
        default=False,
        help="If set, randomize light direction/position and intensity per variant (written into Blender jobs JSON).",
    )
    p.add_argument(
        "--light_rng_seed",
        type=int,
        default=0,
        help="Base RNG seed for randomized lighting. Use -1 for non-deterministic seeding.",
    )
    p.add_argument(
        "--light_angle_jitter_deg",
        type=float,
        default=35.0,
        help="Max absolute Euler jitter (degrees) applied to each light rotation component.",
    )
    p.add_argument(
        "--light_energy_jitter",
        type=float,
        default=0.6,
        help="Energy multiplier jitter: sampled uniformly in [1-j, 1+j] for each light.",
    )

    # Dataset controls
    p.add_argument("--variants_per_image", type=int, default=1)
    p.add_argument("--num_face_keypoints", type=int, default=68)

    # Annotation controls
    p.add_argument(
        "--bbox_from_glasses_landmarks",
        action="store_true",
        default=False,
        help="If set, recompute COCO bbox/area from the projected glasses keypoints (visible ones only). "
             "If no visible glasses keypoints exist for an annotation, falls back to the original bbox/area.",
    )

    # Debugging
    p.add_argument("--debug_landmarks", action="store_true", default=False,
                   help="Save debug images with projected glasses landmarks overlaid")
    p.add_argument("--pnp_only", action="store_true", default=False,
                   help="Ignore Blender and only run OpenCV solvePnP + projectPoints for debugging")
    p.add_argument("--verbose_blender", action="store_true", default=False,
                   help="Print per-job Blender logs and reprojection checks")
    p.add_argument("--keep_ssaa_rgba", action="store_true", default=False,
                   help="Keep the large SSAA RGBA outputs from Blender (otherwise overwritten with downsampled)")

    # Mesh/CAD alignment options (same semantics as pyrender script PnP renderer)
    p.add_argument("--model_prescale", type=float, default=1.0)
    p.add_argument("--center_model", action="store_true", default=False)
    p.add_argument("--model_rotate", type=float, nargs=3, default=None,
                   help="Pre-rotation to apply to the model (degrees): rx ry rz")
    p.add_argument("--pnp_mesh_auto_align", action="store_true", default=False)
    p.add_argument("--pnp_mesh_align_translate_z", action="store_true", default=False)

    return p.parse_args()


def _stable_u32_from_str(s: str) -> int:
    return int(zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF)


def _make_random_lights_spec(
    *,
    base_key: float,
    base_fill: float,
    seed_u32: int,
    angle_jitter_deg: float,
    energy_jitter: float,
):
    """Return a Blender jobs 'lights' list.

    Notes:
      - Uses SUN lights for scale-robustness (direction via rotation_euler).
      - Also emits a 'location' field for completeness, though SUN is direction-based.
    """
    rng = np.random.RandomState(int(seed_u32))

    # Base rotations in radians (matching blender_render_jobs_44.py defaults)
    base = [
        ("Key", (0.0, 0.0, 0.0), float(base_key)),
        ("Fill", (0.5, 0.0, 1.0), float(base_fill)),
        ("Rim", (-0.5, 0.0, -1.0), float(base_fill) * 0.75),
    ]

    ang = float(angle_jitter_deg) * (math.pi / 180.0)
    j = float(max(0.0, energy_jitter))

    lights = []
    for name, rot0, e0 in base:
        d_rot = rng.uniform(-ang, ang, size=(3,)).astype(np.float64)
        rot = (float(rot0[0] + d_rot[0]), float(rot0[1] + d_rot[1]), float(rot0[2] + d_rot[2]))

        mult = float(rng.uniform(1.0 - j, 1.0 + j)) if j > 0 else 1.0
        energy = max(0.001, float(e0) * mult)

        # Random location on a sphere (mostly decorative for SUN)
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        phi = float(rng.uniform(0.25 * math.pi, 0.75 * math.pi))
        r = float(rng.uniform(3.0, 8.0))
        loc = (
            float(r * math.cos(theta) * math.sin(phi)),
            float(r * math.sin(theta) * math.sin(phi)),
            float(r * math.cos(phi)),
        )

        lights.append({
            "name": name,
            "type": "SUN",
            "rotation_euler": [rot[0], rot[1], rot[2]],
            "location": [loc[0], loc[1], loc[2]],
            "energy": energy,
            "color": [1.0, 1.0, 1.0],
        })

    return lights


# -------------------------------------------------------------------------
# COCO / LANDMARK UTILITIES
# -------------------------------------------------------------------------

def get_face_landmarks_from_annotation(ann, num_kpts_face=None):
    kpts = np.asarray(ann["keypoints"], dtype=np.float32)
    if kpts.size % 3 != 0:
        raise ValueError("Annotation keypoints length is not multiple of 3")
    kpts = kpts.reshape(-1, 3)
    if num_kpts_face is not None:
        kpts = kpts[: int(num_kpts_face)]
    return kpts[:, :2]


def draw_landmarks_with_indices(img, kpts, color=(0, 255, 0), idxs=None):
    im = img.copy()
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    r = max(1, int(max(im.size) * 0.005))
    for i, kp in enumerate(kpts):
        try:
            x, y, v = kp
        except Exception:
            continue
        if v <= 0:
            continue
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
        label = str(idxs[i]) if (idxs is not None and i < len(idxs)) else str(i)
        draw.text((x + r + 1, max(0, y - r - 1)), label, fill=color, font=font)
    return im


def coco_bbox_from_visible_keypoints(kpts, image_width: int, image_height: int, min_size: float = 1.0):
    """Compute a COCO bbox [x,y,w,h] from visible keypoints.

    Expects keypoints as an iterable of (x, y, v). Uses only points with v > 0.
    Returns None if no visible points.
    """
    xs = []
    ys = []
    for kp in kpts:
        if kp is None or len(kp) < 3:
            continue
        x, y, v = kp[0], kp[1], kp[2]
        if float(v) <= 0:
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xs.append(float(x))
        ys.append(float(y))

    if not xs:
        return None

    x0 = max(0.0, min(xs))
    y0 = max(0.0, min(ys))
    x1 = min(float(image_width), max(xs))
    y1 = min(float(image_height), max(ys))

    w = max(float(min_size), float(x1 - x0))
    h = max(float(min_size), float(y1 - y0))
    return [float(x0), float(y0), float(w), float(h)]


# -------------------------------------------------------------------------
# WEAK-PERSPECTIVE POSE (copied from your pyrender script)
# -------------------------------------------------------------------------

def estimate_glasses_pose_weak(landmarks_2d):
    _check_glasses_3d()

    obj_pts, img_pts = [], []
    for g_idx, face_idx in GLASSES_TO_FACE_IDXS.items():
        if face_idx >= landmarks_2d.shape[0]:
            continue
        if GLASSES_3D_LANDMARKS_DEFINED is not None and not bool(GLASSES_3D_LANDMARKS_DEFINED[g_idx]):
            continue
        obj_pts.append(GLASSES_3D_LANDMARKS[g_idx])
        img_pts.append(landmarks_2d[face_idx])

    if len(obj_pts) < 3:
        R = np.eye(3, dtype=np.float32)
        s = 1.0
        t = landmarks_2d.mean(axis=0).astype(np.float32)
        return R, s, t

    X = np.asarray(obj_pts, dtype=np.float32)
    Y = np.asarray(img_pts, dtype=np.float32)

    mu_X = X.mean(axis=0, keepdims=True)
    mu_Y = Y.mean(axis=0, keepdims=True)
    Xc = X - mu_X
    Yc = Y - mu_Y

    Y3 = np.hstack([Yc, np.zeros((Yc.shape[0], 1), dtype=np.float32)])

    H = Xc.T @ Y3
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    var_X = (Xc ** 2).sum()
    s = float(S.sum() / var_X)
    t_2d = (mu_Y[0] - s * (R @ mu_X[0])[:2]).astype(np.float32)

    return R.astype(np.float32), float(s), t_2d


def project_glasses_keypoints_weak(R, s, t, kp_indices=None, visibility=2):
    _check_glasses_3d()
    if kp_indices is None:
        kp_indices = GLASSES_KEYPOINT_IDXS
    kpts = []
    for idx in kp_indices:
        if GLASSES_3D_LANDMARKS_DEFINED is not None and not bool(GLASSES_3D_LANDMARKS_DEFINED[idx]):
            kpts.append([0.0, 0.0, 0])
            continue
        X = GLASSES_3D_LANDMARKS[idx]
        RX = R[:2, :] @ X
        x2d = s * RX + t
        kpts.append([float(x2d[0]), float(x2d[1]), visibility])
    return kpts


# -------------------------------------------------------------------------
# PnP POSE (copied from your pyrender script)
# -------------------------------------------------------------------------

def _get_pnp_correspondences(landmarks_2d):
    _check_glasses_3d()
    obj_pts, img_pts = [], []
    for g_idx, face_idx in GLASSES_TO_FACE_IDXS.items():
        if face_idx >= landmarks_2d.shape[0]:
            continue
        if GLASSES_3D_LANDMARKS_DEFINED is not None and not bool(GLASSES_3D_LANDMARKS_DEFINED[g_idx]):
            continue
        obj_pts.append(GLASSES_3D_LANDMARKS[g_idx])
        img_pts.append(landmarks_2d[face_idx])

    if len(obj_pts) < 4:
        return None, None

    return np.asarray(obj_pts, dtype=np.float32), np.asarray(img_pts, dtype=np.float32)


def estimate_glasses_pose_pnp(landmarks_2d, image_width, image_height):
    obj_pts, img_pts = _get_pnp_correspondences(landmarks_2d)
    if obj_pts is None:
        return False, None, None, None, None, None

    f = float(max(image_width, image_height))
    cx = float(image_width) / 2.0
    cy = float(image_height) / 2.0
    K = np.array([[f, 0.0, cx],
                  [0.0, f, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)

    dist = np.zeros((4, 1), dtype=np.float32)

    n_pts = obj_pts.shape[0]
    flag = cv2.SOLVEPNP_ITERATIVE if n_pts >= 6 else cv2.SOLVEPNP_AP3P

    try:
        success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=flag)
    except cv2.error:
        return False, None, None, None, None, None

    # Reprojection error helper
    pts_3d = obj_pts.reshape(-1, 3)
    img_pts_2d = img_pts.reshape(-1, 2)

    def reproj_err(rvec_c, tvec_c):
        proj, _ = cv2.projectPoints(pts_3d, rvec_c, tvec_c, K, None)
        proj = proj.reshape(-1, 2)
        return float(np.mean(np.linalg.norm(proj - img_pts_2d, axis=1)))

    err_orig = reproj_err(rvec, tvec)

    # Flip ambiguity: rotate 180° about X
    R_mat, _ = cv2.Rodrigues(rvec)
    t_vec = tvec.reshape(3,)
    Rx_pi = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    R2 = Rx_pi @ R_mat
    rvec2, _ = cv2.Rodrigues(R2)
    tvec2 = (Rx_pi @ t_vec).reshape(3, 1)
    err_flip = reproj_err(rvec2, tvec2)

    if err_flip + 1e-6 < err_orig:
        rvec, tvec = rvec2, tvec2

    return True, rvec.astype(np.float32), tvec.astype(np.float32), K.astype(np.float32), obj_pts, img_pts


def project_glasses_keypoints_pnp(rvec, tvec, K, image_width, image_height, kp_indices=None, visibility=2):
    _check_glasses_3d()
    if kp_indices is None:
        kp_indices = list(range(GLASSES_3D_LANDMARKS.shape[0]))

    pts_3d = np.asarray([GLASSES_3D_LANDMARKS[idx] for idx in kp_indices], dtype=np.float32).reshape(-1, 3)
    dist = np.zeros((4, 1), dtype=np.float32)
    img_pts, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, dist)
    img_pts = img_pts.reshape(-1, 2)

    # camera-space Z check
    R_cam, _ = cv2.Rodrigues(rvec)
    t_cam = tvec.reshape(3, 1)
    pts_cam = (R_cam @ pts_3d.T + t_cam).T
    zs = pts_cam[:, 2]

    margin_x0 = -0.5 * float(image_width)
    margin_x1 = 1.5 * float(image_width)
    margin_y0 = -0.5 * float(image_height)
    margin_y1 = 1.5 * float(image_height)

    proj = []
    for i, idx in enumerate(kp_indices):
        if GLASSES_3D_LANDMARKS_DEFINED is not None and not bool(GLASSES_3D_LANDMARKS_DEFINED[idx]):
            proj.append([0.0, 0.0, 0])
            continue
        x, y = float(img_pts[i, 0]), float(img_pts[i, 1])
        z = float(zs[i])
        if (not np.isfinite(x)) or (not np.isfinite(y)) or (not np.isfinite(z)) or (z <= 1e-6):
            proj.append([0.0, 0.0, 0])
            continue
        if x < margin_x0 or x > margin_x1 or y < margin_y0 or y > margin_y1:
            proj.append([0.0, 0.0, 0])
            continue
        proj.append([x, y, visibility])
    return proj


# -------------------------------------------------------------------------
# Blender job generation helpers
# -------------------------------------------------------------------------

def _mat4_to_list(M: np.ndarray):
    M = np.asarray(M, dtype=np.float32).reshape(4, 4)
    return [[float(M[r, c]) for c in range(4)] for r in range(4)]


def blender_camera_pose_from_pnp(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """OpenCV -> Blender/OpenGL camera pose, identical to pyrender conversion."""
    R_cw_cv, _ = cv2.Rodrigues(rvec)
    t_cv = tvec.reshape(3,)

    C = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    R_cw_gl = C @ R_cw_cv.astype(np.float32)
    t_gl = C @ t_cv.astype(np.float32)

    R_wc = R_cw_gl.T
    t_wc = -R_wc @ t_gl

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R_wc.astype(np.float32)
    T[:3, 3] = t_wc.astype(np.float32)
    return T


def blender_object_matrix_from_weak_pose(rot_mat: np.ndarray, pose_s: float, pose_t: np.ndarray,
                                        out_w: int, out_h: int, ssaa: int) -> np.ndarray:
    """
    Object matrix that reproduces the same weak-perspective pixel mapping as the pyrender path.

    Blender ORTHO camera has a single ortho_scale (width), and height depends on aspect ratio.
    To match pyrender's independent xmag/ymag, we scale world Y by a = H/W.
    """
    ssaa_i = int(max(1, ssaa))
    render_w = int(out_w * ssaa_i)
    render_h = int(out_h * ssaa_i)
    a = float(render_h) / float(render_w)  # aspect correction for Blender ortho

    R = np.asarray(rot_mat, dtype=np.float32)
    s_eff = float(pose_s) * float(ssaa_i)
    t_eff = np.asarray(pose_t, dtype=np.float32).reshape(2,) * float(ssaa_i)

    RS = np.zeros((3, 3), dtype=np.float32)
    RS[0, :] = (s_eff / float(render_w)) * R[0, :]
    RS[1, :] = a * (-(s_eff / float(render_h)) * R[1, :])
    RS[2, 2] = 1.0

    tx = (t_eff[0] / float(render_w)) - 0.5
    ty = a * (0.5 - (t_eff[1] / float(render_h)))

    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = RS
    M[0, 3] = tx
    M[1, 3] = ty
    M[2, 3] = 0.0
    return M


def load_glasses_mesh(glasses_path: str) -> trimesh.Trimesh:
    if not os.path.isfile(glasses_path):
        raise FileNotFoundError(glasses_path)
    mesh = trimesh.load(glasses_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    return mesh


def compute_mesh_adjust_transform(mesh_trimesh: trimesh.Trimesh,
                                 model_prescale: float,
                                 center_model: bool,
                                 model_rotate_deg,
                                 pnp_mesh_auto_align: bool,
                                 pnp_mesh_align_translate_z: bool) -> np.ndarray:
    """
    Reproduce the exact transform chain used in your pyrender PnP renderer, but as a 4x4 matrix
    applied to the Blender object (equivalent to baking into vertices).
    """
    _check_glasses_3d()

    mesh_copy = mesh_trimesh.copy()
    A = np.eye(4, dtype=np.float32)

    # pre-rotation
    if model_rotate_deg is not None:
        rx_d, ry_d, rz_d = model_rotate_deg
        rx, ry, rz = math.radians(rx_d), math.radians(ry_d), math.radians(rz_d)
        Rx = np.array([[1, 0, 0, 0],
                       [0, math.cos(rx), -math.sin(rx), 0],
                       [0, math.sin(rx),  math.cos(rx), 0],
                       [0, 0, 0, 1]], dtype=np.float32)
        Ry = np.array([[ math.cos(ry), 0, math.sin(ry), 0],
                       [0, 1, 0, 0],
                       [-math.sin(ry), 0, math.cos(ry), 0],
                       [0, 0, 0, 1]], dtype=np.float32)
        Rz = np.array([[math.cos(rz), -math.sin(rz), 0, 0],
                       [math.sin(rz),  math.cos(rz), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]], dtype=np.float32)
        pre = (Rz @ Ry @ Rx).astype(np.float32)
        mesh_copy.apply_transform(pre)
        A = pre @ A

    # center_model
    if center_model:
        bmin, bmax = mesh_copy.bounds
        center = (bmin + bmax) / 2.0
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = (-center).astype(np.float32)
        mesh_copy.apply_translation(-center)
        A = T @ A

    # prescale
    prescale_f = float(model_prescale) if model_prescale is not None else 1.0
    if abs(prescale_f - 1.0) > 1e-9:
        S = np.eye(4, dtype=np.float32)
        S[:3, :3] *= prescale_f
        mesh_copy.apply_transform(S)
        A = S @ A

    # auto-align
    if pnp_mesh_auto_align:
        mesh_bounds_min, mesh_bounds_max = mesh_copy.bounds
        mesh_center = mesh_copy.centroid
        mesh_size = mesh_bounds_max - mesh_bounds_min

        lm = GLASSES_3D_LANDMARKS
        if GLASSES_3D_LANDMARKS_DEFINED is not None:
            valid_mask = GLASSES_3D_LANDMARKS_DEFINED.astype(bool)
        else:
            valid_mask = np.any(lm != 0.0, axis=1)

        if np.any(valid_mask):
            lm_nz = lm[valid_mask]
            lm_min = lm_nz.min(axis=0)
            lm_max = lm_nz.max(axis=0)
            lm_center = lm_nz.mean(axis=0)
        else:
            lm_min = lm.min(axis=0)
            lm_max = lm.max(axis=0)
            lm_center = lm.mean(axis=0)

        lm_size = lm_max - lm_min

        mesh_xy = np.asarray(mesh_size[:2], dtype=np.float32)
        lm_xy = np.asarray(lm_size[:2], dtype=np.float32)
        mean_mesh_xy = float(np.maximum(mesh_xy, 1e-9).mean())
        mean_lm_xy = float(np.maximum(lm_xy, 1e-9).mean())
        scale_factor = mean_lm_xy / mean_mesh_xy if (mean_mesh_xy > 1e-9 and mean_lm_xy > 1e-9) else 1.0

        delta = (lm_center - mesh_center * scale_factor).astype(np.float32)
        if not pnp_mesh_align_translate_z:
            delta[2] = 0.0

        T_align = np.eye(4, dtype=np.float32)
        T_align[:3, :3] *= float(scale_factor)
        T_align[:3, 3] = delta
        mesh_copy.apply_transform(T_align)
        A = T_align @ A

        print(f"[mesh auto-align] scale={scale_factor:.6g} translate={delta.tolist()} translate_z={bool(pnp_mesh_align_translate_z)}")

    return A


def run_blender_jobs(blender_bin: str, blender_script: str, jobs_json: str, verbose: bool):
    cmd = [blender_bin, "-b", "--python", blender_script, "--", "--jobs", jobs_json]
    if verbose:
        cmd.append("--verbose")

    print("Running Blender:")
    print("  " + " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            "Blender failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{res.stdout}\n"
            f"STDERR:\n{res.stderr}\n"
        )
    if verbose:
        print(res.stdout)
        if res.stderr.strip():
            print(res.stderr)


def composite_glasses_on_image(bg_img_path: str, glasses_rgba_img: Image.Image, out_img_path: str):
    if not os.path.isfile(bg_img_path):
        print(f"Warning: background not found: {bg_img_path}. Saving RGBA directly.")
        glasses_rgba_img.save(out_img_path)
        return
    try:
        bg = Image.open(bg_img_path).convert("RGB")
    except Exception as e:
        print(f"Warning: failed to open bg {bg_img_path}: {e}. Saving RGBA directly.")
        glasses_rgba_img.save(out_img_path)
        return

    if glasses_rgba_img.size != bg.size:
        glasses_rgba_img = glasses_rgba_img.resize(bg.size, Image.BICUBIC)

    out = bg.convert("RGBA")
    out.alpha_composite(glasses_rgba_img.convert("RGBA"))
    out.convert("RGB").save(out_img_path, quality=95)


def downsample_rgba(img: Image.Image, out_w: int, out_h: int) -> Image.Image:
    if img.size == (int(out_w), int(out_h)):
        return img
    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)
    return img.resize((int(out_w), int(out_h)), resampling)


def debug_pnp_only(coco, image_by_id, images_root: Path, out_images_root: Path, num_face_kpts: int):
    annots = coco.get("annotations", [])
    if not annots:
        print("No annotations found; nothing to run PnP on.")
        return

    for ann in annots:
        img_info = image_by_id.get(ann["image_id"])
        if img_info is None:
            continue
        img_path = images_root / img_info["file_name"]
        if not img_path.is_file():
            print(f"[PnP-only] Missing image: {img_path}")
            continue

        try:
            landmarks_2d = get_face_landmarks_from_annotation(ann, num_kpts_face=num_face_kpts)
        except Exception as e:
            print(f"[PnP-only] Failed landmarks for {img_path}: {e}")
            continue

        w, h = int(img_info["width"]), int(img_info["height"])
        ok, rvec, tvec, K, _, _ = estimate_glasses_pose_pnp(landmarks_2d, w, h)
        if not ok:
            print(f"[PnP-only] PnP failed: {img_path}")
            continue

        proj_kpts = project_glasses_keypoints_pnp(rvec, tvec, K, w, h)

        img = Image.open(img_path).convert("RGB")
        orig_kpts = np.asarray(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
        img_dbg = draw_landmarks_with_indices(img, orig_kpts.tolist(), color=(0, 255, 0))
        full_idxs = list(range(GLASSES_3D_LANDMARKS.shape[0]))
        img_dbg = draw_landmarks_with_indices(img_dbg, proj_kpts, color=(255, 0, 0), idxs=full_idxs)

        out_name = f"{Path(img_info['file_name']).stem}_pnp_debug.jpg"
        out_path = out_images_root / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img_dbg.save(out_path, quality=95)
        print(f"[PnP-only] wrote {out_path}")


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

def main():
    args = parse_args()

    # Resolve blender script path
    if args.blender_script is None:
        here = Path(__file__).resolve().parent
        args.blender_script = str(here / "blender_render_jobs_44.py")

    if not os.path.isfile(args.blender_script):
        raise FileNotFoundError(f"Blender helper script not found: {args.blender_script}")

    global GLASSES_3D_LANDMARKS, GLASSES_3D_LANDMARKS_DEFINED
    GLASSES_3D_LANDMARKS, GLASSES_3D_LANDMARKS_DEFINED = load_glasses_3d_landmarks_from_xml(args.glasses_xml)
    _check_glasses_3d()
    glasses_output_idxs = list(range(int(GLASSES_3D_LANDMARKS.shape[0])))

    images_root = Path(args.images_root)
    out_images_root = Path(args.out_images)
    out_images_root.mkdir(parents=True, exist_ok=True)

    tmp_rgba_dir = out_images_root / "_tmp_glasses_rgba"
    tmp_rgba_dir.mkdir(parents=True, exist_ok=True)

    with open(args.annotations, "r") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    image_by_id = {img["id"]: img for img in images}

    # If requested, run PnP-only debug mode and exit.
    if args.pnp_only:
        print("Running PnP-only debug mode (no Blender rendering)...")
        debug_pnp_only(coco, image_by_id, images_root, out_images_root, args.num_face_keypoints)
        return

    # Prepare COCO output category keypoints
    used_category_ids = {a.get("category_id") for a in annotations if "category_id" in a}
    n_out_kpts = int(len(glasses_output_idxs))
    new_kpt_names = [f"kpt_{i}" for i in range(n_out_kpts)]
    if categories:
        for cat in categories:
            if cat.get("id") in used_category_ids or len(used_category_ids) == 0:
                cat["keypoints"] = list(new_kpt_names)
                cat["skeleton"] = []
    else:
        categories = [{
            "id": 1,
            "name": "face",
            "supercategory": "person",
            "keypoints": list(new_kpt_names),
            "skeleton": [],
        }]

    next_image_id = (max(img["id"] for img in images) + 1) if images else 1
    next_ann_id = (max(ann["id"] for ann in annotations) + 1) if annotations else 1

    # Load mesh once for optional alignment transform
    glasses_mesh = load_glasses_mesh(args.glasses_obj)
    mesh_adjust = compute_mesh_adjust_transform(
        glasses_mesh,
        model_prescale=args.model_prescale,
        center_model=args.center_model,
        model_rotate_deg=args.model_rotate,
        pnp_mesh_auto_align=args.pnp_mesh_auto_align,
        pnp_mesh_align_translate_z=args.pnp_mesh_align_translate_z,
    )

    # Collect render tasks for Blender and post-processing
    blender_jobs = []
    tasks = []  # holds everything needed to composite + write COCO

    pose_log = []

    for ann in annotations:
        img = image_by_id.get(ann["image_id"])
        if img is None:
            continue

        img_path = images_root / img["file_name"]
        if not img_path.is_file():
            print(f"Warning: missing image: {img_path}")
            continue

        w, h = int(img["width"]), int(img["height"])
        try:
            landmarks_2d = get_face_landmarks_from_annotation(ann, num_kpts_face=args.num_face_keypoints)
        except Exception as e:
            print(f"Warning: failed to parse landmarks for {img_path}: {e}")
            continue

        # Pose estimation (PnP preferred, weak fallback)
        pnp_ok = False
        rvec = tvec = K = None
        obj_pts = img_pts = None
        Rw = None
        s_w = 1.0
        t_w = np.zeros(2, dtype=np.float32)

        ok, rvec_pnp, tvec_pnp, K_pnp, obj_pts_pnp, img_pts_pnp = estimate_glasses_pose_pnp(landmarks_2d, w, h)
        if ok:
            pnp_ok = True
            rvec, tvec, K = rvec_pnp, tvec_pnp, K_pnp
            obj_pts, img_pts = obj_pts_pnp, img_pts_pnp

            # log euler + degenerate roll discard similar to your script
            try:
                R_mat, _ = cv2.Rodrigues(rvec)
                sy = math.sqrt(R_mat[0, 0] ** 2 + R_mat[1, 0] ** 2)
                if sy < 1e-6:
                    rx = math.degrees(math.atan2(-R_mat[1, 2], R_mat[1, 1]))
                    ry = math.degrees(math.atan2(-R_mat[2, 0], sy))
                    rz = 0.0
                else:
                    rx = math.degrees(math.atan2(R_mat[2, 1], R_mat[2, 2]))
                    ry = math.degrees(math.atan2(-R_mat[2, 0], sy))
                    rz = math.degrees(math.atan2(R_mat[1, 0], R_mat[0, 0]))

                threshold_deg = 30.0
                if abs(rx) < threshold_deg:
                    print(f"[PnP] Discarding {img['file_name']} due to degenerate roll={rx:.1f}°")
                    pose_log.append({
                        "image": img["file_name"],
                        "method": "pnp",
                        "roll": float(rx), "pitch": float(ry), "yaw": float(rz),
                        "tvec": tvec.flatten().tolist(),
                        "discarded": True,
                    })
                    continue

                print(f"[PnP] {img['file_name']}: roll={rx:.1f} pitch={ry:.1f} yaw={rz:.1f} tvec={tvec.flatten().tolist()}")
                pose_log.append({
                    "image": img["file_name"],
                    "method": "pnp",
                    "roll": float(rx), "pitch": float(ry), "yaw": float(rz),
                    "tvec": tvec.flatten().tolist(),
                    "discarded": False,
                })
            except Exception:
                pose_log.append({
                    "image": img["file_name"],
                    "method": "pnp",
                    "tvec": tvec.flatten().tolist(),
                    "discarded": False,
                })

            glasses_kpts = project_glasses_keypoints_pnp(
                rvec, tvec, K, w, h, kp_indices=glasses_output_idxs
            )
        else:
            # weak fallback
            try:
                Rw, s_w, t_w = estimate_glasses_pose_weak(landmarks_2d)
            except Exception as e:
                print(f"Warning: weak pose failed for {img_path}: {e}")
                continue
            glasses_kpts = project_glasses_keypoints_weak(Rw, s_w, t_w, kp_indices=glasses_output_idxs)
            pose_log.append({
                "image": img["file_name"],
                "method": "weak",
                "s": float(s_w),
                "t": [float(x) for x in t_w.tolist()],
                "discarded": False,
            })

        # Create variants
        for v in range(int(max(1, args.variants_per_image))):
            orig_rel = Path(img["file_name"])
            if orig_rel.is_absolute():
                orig_rel = Path(orig_rel.name)

            out_rel = (Path(f"v{v}") / orig_rel) if args.variants_per_image > 1 else orig_rel
            out_img_path = out_images_root / out_rel
            out_img_path.parent.mkdir(parents=True, exist_ok=True)

            # RGBA output from Blender (temporary)
            job_id = f"ann{ann['id']}_v{v}_{uuid.uuid4().hex[:8]}"
            out_rgba_path = tmp_rgba_dir / f"{job_id}_ssaa.png"

            # Optional per-variant lighting randomization
            lights_spec = None
            if bool(getattr(args, "randomize_lights", False)):
                if int(getattr(args, "light_rng_seed", 0)) == -1:
                    # non-deterministic seed, but still distinct per job
                    base_seed = int.from_bytes(os.urandom(4), "little", signed=False)
                else:
                    base_seed = int(getattr(args, "light_rng_seed", 0))
                seed_u32 = (base_seed ^ _stable_u32_from_str(job_id)) & 0xFFFFFFFF
                lights_spec = _make_random_lights_spec(
                    base_key=float(args.key_light),
                    base_fill=float(args.fill_light),
                    seed_u32=seed_u32,
                    angle_jitter_deg=float(getattr(args, "light_angle_jitter_deg", 35.0)),
                    energy_jitter=float(getattr(args, "light_energy_jitter", 0.6)),
                )

            ssaa_i = int(max(1, args.ssaa))
            render_w = int(w * ssaa_i)
            render_h = int(h * ssaa_i)

            if pnp_ok:
                cam_pose = blender_camera_pose_from_pnp(rvec, tvec)

                # Scale intrinsics + correspondences to render resolution
                fx = float(K[0, 0]) * ssaa_i
                fy = float(K[1, 1]) * ssaa_i
                cx = float(K[0, 2]) * ssaa_i
                cy = float(K[1, 2]) * ssaa_i

                dbg_corr = None
                if obj_pts is not None and img_pts is not None:
                    dbg_corr = {
                        "obj_pts": obj_pts.tolist(),
                        "img_pts": (img_pts * float(ssaa_i)).tolist(),
                    }

                job = {
                    "model_path": str(Path(args.glasses_obj).resolve()),
                    "out_rgba_png": str(out_rgba_path),
                    "render_width": render_w,
                    "render_height": render_h,
                    "engine": args.engine,
                    "samples": int(args.samples),
                    "ambient": float(args.ambient),
                    "key_light": float(args.key_light),
                    "fill_light": float(args.fill_light),
                    "lights": lights_spec,
                    "object_matrix_world": _mat4_to_list(mesh_adjust),
                    "camera": {
                        "mode": "PERSP",
                        "matrix_world": _mat4_to_list(cam_pose),
                        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                        "sensor_width_mm": 36.0,
                    },
                }
                if dbg_corr is not None and args.verbose_blender:
                    job["debug_correspondences"] = dbg_corr
            else:
                # Weak/ortho render: pose encoded in object matrix; fixed camera
                obj_pose = blender_object_matrix_from_weak_pose(Rw, s_w, t_w, w, h, ssaa_i)

                cam_pose = np.eye(4, dtype=np.float32)
                cam_pose[2, 3] = 10.0

                job = {
                    "model_path": str(Path(args.glasses_obj).resolve()),
                    "out_rgba_png": str(out_rgba_path),
                    "render_width": render_w,
                    "render_height": render_h,
                    "engine": args.engine,
                    "samples": int(args.samples),
                    "ambient": float(args.ambient),
                    "key_light": float(args.key_light),
                    "fill_light": float(args.fill_light),
                    "lights": lights_spec,
                    "object_matrix_world": _mat4_to_list(obj_pose),
                    "camera": {
                        "mode": "ORTHO",
                        "matrix_world": _mat4_to_list(cam_pose),
                        "ortho_scale": 1.0,
                    },
                }

            blender_jobs.append(job)
            tasks.append({
                "job": job,
                "bg_path": str(img_path),
                "out_img_path": str(out_img_path),
                "out_rel": out_rel.as_posix(),
                "w": w, "h": h,
                "ann": ann,
                "image_info": img,
                "proj_kpts": glasses_kpts,  # (x,y,v) in output resolution
                "pnp_ok": pnp_ok,
            })

    # Write jobs file for Blender
    jobs_path = out_images_root / "blender_render_jobs.json"
    with open(jobs_path, "w") as f:
        json.dump({"jobs": blender_jobs}, f)
    print(f"Wrote Blender jobs: {jobs_path} ({len(blender_jobs)} renders)")

    # Run Blender
    run_blender_jobs(args.blender_bin, args.blender_script, str(jobs_path), verbose=args.verbose_blender)

    # Postprocess: composite and build COCO outputs
    new_images = []
    new_annotations = []

    for t in tasks:
        job = t["job"]
        rgba_path = Path(job["out_rgba_png"])
        if not rgba_path.is_file():
            print(f"Warning: missing Blender output RGBA: {rgba_path}")
            continue

        rgba = Image.open(rgba_path).convert("RGBA")
        rgba_ds = downsample_rgba(rgba, t["w"], t["h"])

        # Optionally overwrite SSAA file with downsampled RGBA
        if (not args.keep_ssaa_rgba) and rgba_ds.size != rgba.size:
            rgba_ds.save(rgba_path)  # overwrite

        # Debug overlay on RGBA (optional)
        if args.debug_landmarks:
            try:
                full_idxs = list(range(GLASSES_3D_LANDMARKS.shape[0]))
                rgba_dbg = draw_landmarks_with_indices(rgba_ds, t["proj_kpts"], color=(255, 0, 0), idxs=full_idxs)
                rgba_dbg.save(str(rgba_path).replace("_ssaa.png", "_kpts.png"))
            except Exception as e:
                print(f"Warning: failed to draw landmarks on RGBA: {e}")

        # Composite onto background and save
        composite_glasses_on_image(t["bg_path"], rgba_ds, t["out_img_path"])

        # Debug composite overlay
        if args.debug_landmarks:
            try:
                comp = Image.open(t["out_img_path"]).convert("RGB")
                orig_kpts_local = np.asarray(t["ann"]["keypoints"], dtype=np.float32).reshape(-1, 3)
                comp_dbg = draw_landmarks_with_indices(comp, orig_kpts_local.tolist(), color=(0, 255, 0))
                comp_dbg = draw_landmarks_with_indices(comp_dbg, t["proj_kpts"], color=(255, 0, 0), idxs=GLASSES_KEYPOINT_IDXS)
                comp_dbg.save(t["out_img_path"] + ".kpts.jpg", quality=90)
            except Exception as e:
                print(f"Warning: failed to save composite debug overlay: {e}")

        # COCO: new image entry
        img_info = t["image_info"]
        new_img = {
            "id": next_image_id,
            "file_name": t["out_rel"],
            "width": int(img_info["width"]),
            "height": int(img_info["height"]),
        }
        new_images.append(new_img)

        proj_kpts_flat = [float(c) for kp in t["proj_kpts"] for c in kp]
        new_ann = dict(t["ann"])
        new_ann["id"] = next_ann_id
        new_ann["image_id"] = next_image_id
        new_ann["keypoints"] = proj_kpts_flat
        new_ann["num_keypoints"] = int(sum(1 for kp in t["proj_kpts"] if float(kp[2]) > 0))
        if "keypoints_3d" in new_ann:
            new_ann["keypoints_3d"] = []
        if "parameters_3d" in new_ann:
            new_ann["parameters_3d"] = {}
        new_ann["iscrowd"] = int(new_ann.get("iscrowd", 0))

        # bbox/area: optionally recompute from projected glasses landmarks
        if args.bbox_from_glasses_landmarks:
            bbox = coco_bbox_from_visible_keypoints(t["proj_kpts"], t["w"], t["h"], min_size=1.0)
            if bbox is not None:
                new_ann["bbox"] = bbox
                new_ann["area"] = float(bbox[2] * bbox[3])
            else:
                if "bbox" in t["ann"]:
                    new_ann["bbox"] = t["ann"]["bbox"]
                if "area" in t["ann"]:
                    new_ann["area"] = t["ann"]["area"]
        else:
            # preserve bbox/area if present
            if "bbox" in t["ann"]:
                new_ann["bbox"] = t["ann"]["bbox"]
            if "area" in t["ann"]:
                new_ann["area"] = t["ann"]["area"]

        new_annotations.append(new_ann)

        next_image_id += 1
        next_ann_id += 1

    out_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": new_images,
        "annotations": new_annotations,
        "categories": categories,
    }

    with open(args.out_annotations, "w") as f:
        json.dump(out_coco, f)
    print(f"Saved synthetic annotations to: {args.out_annotations}")
    print(f"Synthetic images written under: {out_images_root}")

    # Also write <out_images>/annotations/<input_basename>.json
    try:
        out_ann_dir = out_images_root / "annotations"
        out_ann_dir.mkdir(parents=True, exist_ok=True)
        out_ann_path = out_ann_dir / Path(args.annotations).name
        with open(out_ann_path, "w") as f:
            json.dump(out_coco, f)
        print(f"Saved per-run annotations copy to: {out_ann_path}")
    except Exception as e:
        print(f"Warning: failed to write per-run annotations copy: {e}")

    # Pose log
    try:
        log_path = out_images_root / "pnp_poses_log.json"
        with open(log_path, "w") as lf:
            json.dump(pose_log, lf, indent=2)
        print(f"Wrote pose log: {log_path}")
    except Exception as e:
        print(f"Warning: failed to write pose log: {e}")


if __name__ == "__main__":
    main()
