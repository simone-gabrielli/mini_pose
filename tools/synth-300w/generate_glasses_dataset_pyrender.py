"""generate_glasses_dataset_pyrender.py

Blender-free version of `generate_glasses_dataset.py` using pyrender.

Instead of launching Blender to render an FBX model, this script
loads the same 3D glasses model (OBJ/FBX supported via trimesh) and
renders it directly with pyrender into an RGBA image, then composites
it on top of the original face image.

The overall pipeline and math (weak-perspective pose estimation,
keypoint projection, COCO augmentation) are copied from the original
script so that behaviour stays as close as possible, but we:

  - remove all Blender-specific arguments and calls
  - add pyrender + trimesh-based rendering instead
  - render glasses directly in memory to a PIL Image

Requirements (install via pip):

  pip install numpy pillow trimesh pyrender PyOpenGL pyglet

On some systems you may also need an EGL / OSMesa compatible
OpenGL context; by default this script uses pyrender's OffscreenRenderer
which works well with pyglet on many desktops.
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import xml.etree.ElementTree as ET

import trimesh
import pyrender
import cv2
import tempfile
import uuid


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


def load_glasses_3d_landmarks_from_xml(path):
	"""Read glasses 3D CAD keypoints from XML (same format as original).

	Returns
	-------
	arr : (N, 3) float32
		arr[i] stores xyz for landmark i
	"""

	tree = ET.parse(path)
	root = tree.getroot()

	landmarks_node = root.find("landmarks")
	if landmarks_node is None:
		raise RuntimeError("XML missing <landmarks> root node")

	ids = []
	for lm in landmarks_node.findall("landmark"):
		idx = int(lm.attrib["id"])
		ids.append(idx)

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


GLASSES_3D_LANDMARKS = None
GLASSES_3D_LANDMARKS_DEFINED = None
GLASSES_KEYPOINT_IDXS = sorted(GLASSES_TO_FACE_IDXS.keys())


def _check_glasses_3d():
	global GLASSES_3D_LANDMARKS
	global GLASSES_3D_LANDMARKS_DEFINED
	if GLASSES_3D_LANDMARKS is None:
		raise RuntimeError("GLASSES_3D_LANDMARKS not loaded")
	if GLASSES_3D_LANDMARKS_DEFINED is None:
		raise RuntimeError("GLASSES_3D_LANDMARKS_DEFINED not loaded")
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

	# 3D resources
	parser.add_argument("--glasses_obj", type=str, required=True)
	parser.add_argument("--glasses_xml", type=str, required=True)

	# Rendering quality controls (pyrender).
	# NOTE: This script renders at the *input image* resolution by default.
	# Use SSAA to improve edge quality (at the cost of speed/VRAM).
	parser.add_argument("--render_width", type=int, default=None,
						help="(Deprecated/no-op) Kept for compatibility; rendering uses the input image width.")
	parser.add_argument("--render_height", type=int, default=None,
						help="(Deprecated/no-op) Kept for compatibility; rendering uses the input image height.")
	parser.add_argument("--ssaa", type=int, default=2,
						help="Supersampling factor for pyrender (render at N× resolution and downsample). 1 disables.")
	parser.add_argument("--render_shadows", action="store_true", default=False,
						help="Enable pyrender shadow rendering (slower; can improve realism).")
	parser.add_argument("--ambient", type=float, default=0.35,
						help="Ambient light intensity for pyrender scene (0..1 typical).")
	parser.add_argument("--key_light", type=float, default=3.0,
						help="Key light intensity for pyrender.")
	parser.add_argument("--fill_light", type=float, default=1.5,
						help="Fill light intensity for pyrender.")

	# Debugging helpers
	parser.add_argument("--debug_alpha", action="store_true", default=False,
						help="Print renderer alpha/RGB stats and attempt premultiplied-alpha handling.")

	# Test helper: force a simple baseColorFactor (RGBA) to validate transparency/lighting
	parser.add_argument("--force_basecolor", type=float, nargs=4, default=None,
						help="Override mesh material baseColorFactor as r g b a (0..1) for debugging).")

	# Number of synthetic variants per original image
	parser.add_argument("--variants_per_image", type=int, default=1)

	# Number of face keypoints in the original dataset (e.g. 68 for 300W-LP).
	parser.add_argument("--num_face_keypoints", type=int, default=68)
	parser.add_argument("--debug_no_projection", action="store_true", default=False,
						help="Render glasses without applying weak-perspective scale (for debugging scale issues)")
	parser.add_argument("--model_prescale", type=float, default=1.0,
						help="Pre-scale applied to the imported model before other transforms (default=1.0)")
	parser.add_argument("--center_model", action="store_true", default=False,
						help="Translate the model so its bounding-box center is at the origin before rendering")
	parser.add_argument("--model_rotate", type=float, nargs=3, default=None,
						help="Pre-rotation to apply to the model (degrees) as three values: rx ry rz")
	parser.add_argument("--auto_test", action="store_true", default=False,
						help="Render the first annotation with several candidate pre-rotations and prescales for debugging")
	parser.add_argument("--debug_landmarks", action="store_true", default=False,
						help="Save debug images with reprojected glasses landmarks overlaid")
	parser.add_argument("--identity_projection", action="store_true", default=False,
						help="Bypass pose estimation and use identity rotation + unit scale; center model mean at image center")
	parser.add_argument("--pnp_only", action="store_true", default=False,
						help="Ignore pyrender and only run OpenCV solvePnP + projectPoints for debugging the 3D-2D projection")
	parser.add_argument("--pnp_mesh_auto_align", action="store_true", default=False,
						help="(PnP render) Auto-align mesh to CAD landmarks bbox/center. Can introduce perspective size mismatch if your mesh+XML are already aligned.")
	parser.add_argument("--pnp_mesh_align_translate_z", action="store_true", default=False,
						help="(PnP render) If auto-align is enabled, also translate along Z. Off by default to avoid changing apparent size in perspective.")

	return parser.parse_args()


def _render_scene_rgba_pyrender(scene, viewport_width, viewport_height, render_shadows=False):
	"""Render a pyrender scene to (color, depth), requesting RGBA when available."""
	flags = pyrender.RenderFlags.RGBA
	# Some pyrender versions may not have the SHADOWS flag; use it only when present.
	if render_shadows:
		if hasattr(pyrender.RenderFlags, "SHADOWS"):
			flags = flags | pyrender.RenderFlags.SHADOWS
		else:
			print("Warning: pyrender.RenderFlags.SHADOWS not available in this pyrender build; continuing without shadows.")
	r = pyrender.OffscreenRenderer(viewport_width=int(viewport_width), viewport_height=int(viewport_height))
	try:
		color, depth = r.render(scene, flags=flags)
	finally:
		r.delete()
	return color, depth


def _downsample_rgba_to_target(color_rgba_u8, out_w, out_h):
	"""Downsample an RGBA uint8 numpy array using high-quality filtering."""
	img = Image.fromarray(color_rgba_u8, mode="RGBA")
	if img.size != (int(out_w), int(out_h)):
		resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)
		img = img.resize((int(out_w), int(out_h)), resampling)
	return img


# -------------------------------------------------------------------------
# GEOMETRY / LANDMARK UTILS (copied from original script)
# -------------------------------------------------------------------------


def get_face_landmarks_from_annotation(ann, num_kpts_face=None):
	"""Extract 2D face landmarks from COCO-style annotation.

	ann["keypoints"] = [x1,y1,v1, x2,y2,v2, ...]

	Returns
	-------
	landmarks : (K, 2) float32 array in image coords.
	"""

	kpts = ann["keypoints"]
	kpts = np.asarray(kpts, dtype=np.float32)
	if kpts.size % 3 != 0:
		raise ValueError("Annotation keypoints length is not multiple of 3")

	kpts = kpts.reshape(-1, 3)
	if num_kpts_face is not None:
		kpts = kpts[:num_kpts_face]

	return kpts[:, :2]


def compute_face_bbox_from_annotation(ann, image_width, image_height, num_kpts_face=None):
	"""Compute a tight face bbox from the original face keypoints.

	Uses visible points (v > 0) among the first ``num_kpts_face`` keypoints.
	Returns COCO bbox format [x, y, w, h] and the bbox area.
	"""
	kpts = np.asarray(ann.get("keypoints", []), dtype=np.float32)
	if kpts.size == 0 or (kpts.size % 3) != 0:
		bbox = [0.0, 0.0, float(image_width), float(image_height)]
		return bbox, float(image_width * image_height)

	kpts = kpts.reshape(-1, 3)
	if num_kpts_face is not None:
		kpts = kpts[: int(num_kpts_face)]

	vis = kpts[:, 2] > 0
	pts = kpts[vis, :2]
	if pts.size == 0:
		bbox = [0.0, 0.0, float(image_width), float(image_height)]
		return bbox, float(image_width * image_height)

	x0, y0 = pts.min(axis=0)
	x1, y1 = pts.max(axis=0)

	# Clamp to image bounds
	x0 = float(np.clip(x0, 0.0, float(image_width - 1)))
	y0 = float(np.clip(y0, 0.0, float(image_height - 1)))
	x1 = float(np.clip(x1, 0.0, float(image_width - 1)))
	y1 = float(np.clip(y1, 0.0, float(image_height - 1)))

	w = max(0.0, x1 - x0)
	h = max(0.0, y1 - y0)
	bbox = [x0, y0, float(w), float(h)]
	return bbox, float(w * h)


def estimate_glasses_pose_weak(landmarks_2d):
	"""Estimate weak-perspective parameters (R, s, t).

	Model::

		x_i = s * (R_2x3 X_i) + t

	where X_i is CAD 3D point.

	Returns
	-------
	R : (3, 3) float32
	s : float
	t : (2,) float32
	"""

	_check_glasses_3d()

	obj_pts = []
	img_pts = []

	for g_idx, face_idx in GLASSES_TO_FACE_IDXS.items():
		if face_idx >= landmarks_2d.shape[0]:
			continue
		obj_pt = GLASSES_3D_LANDMARKS[g_idx]
		# Skip undefined (missing ID in the XML). NOTE: a valid landmark can be (0,0,0).
		if GLASSES_3D_LANDMARKS_DEFINED is not None and not bool(GLASSES_3D_LANDMARKS_DEFINED[g_idx]):
			continue
		obj_pts.append(obj_pt)
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

	return R.astype(np.float32), s, t_2d


def project_glasses_keypoints_weak(R, s, t, kp_indices=None, visibility=2):
	"""Project 3D CAD glasses keypoints via weak-perspective to 2D."""

	if kp_indices is None:
		kp_indices = GLASSES_KEYPOINT_IDXS

	kpts = []
	for idx in kp_indices:
		X = GLASSES_3D_LANDMARKS[idx]
		# If the CAD keypoint is undefined (missing ID in the XML), mark it invisible.
		# NOTE: A valid keypoint can legitimately be (0,0,0) in model space.
		if GLASSES_3D_LANDMARKS_DEFINED is not None and not bool(GLASSES_3D_LANDMARKS_DEFINED[idx]):
			kpts.append([0.0, 0.0, 0])
			continue
		RX = R[:2, :] @ X
		x2d = s * RX + t
		kpts.append([float(x2d[0]), float(x2d[1]), visibility])

	return kpts


def _get_pnp_correspondences(landmarks_2d):
	"""Build 3D-2D correspondence lists for PnP from current mapping.

	Returns
	-------
	obj_pts : (N, 3) float32
	img_pts : (N, 2) float32

	Only glasses keypoints with defined (non-zero) 3D CAD coordinates
	and valid 2D facial landmarks are included. At least 3 are required
	for a non-degenerate PnP solve.
	"""
	_check_glasses_3d()

	obj_pts = []
	img_pts = []

	for g_idx, face_idx in GLASSES_TO_FACE_IDXS.items():
		if face_idx >= landmarks_2d.shape[0]:
			continue
		X = GLASSES_3D_LANDMARKS[g_idx]
		if GLASSES_3D_LANDMARKS_DEFINED is not None and not bool(GLASSES_3D_LANDMARKS_DEFINED[g_idx]):
			continue
		obj_pts.append(X)
		img_pts.append(landmarks_2d[face_idx])

	# OpenCV's PnP implementations used here require at least 4
	# correspondences (we'll use AP3P for 4–5, ITERATIVE for 6+).
	if len(obj_pts) < 4:
		return None, None

	return (
		np.asarray(obj_pts, dtype=np.float32),
		np.asarray(img_pts, dtype=np.float32),
	)


def estimate_glasses_pose_pnp(landmarks_2d, image_width, image_height):
	"""Estimate full-perspective pose using OpenCV solvePnP.

	This is a "simple & explicit" PnP formulation using the available
	3D CAD keypoints and their 2D facial correspondences.

	We assume a basic pinhole camera with:

		fx = fy = max(image_width, image_height)
		cx = image_width  / 2
		cy = image_height / 2

	and zero distortion. This is enough to debug whether the 3D-2D
	correspondence and pose are correct, independent of pyrender.

	Returns
	-------
	success : bool
	rvec   : (3, 1) float32
	tvec   : (3, 1) float32
	K      : (3, 3) float32 camera matrix used
	"""
	obj_pts, img_pts = _get_pnp_correspondences(landmarks_2d)
	if obj_pts is None:
		# Not enough valid correspondences for a stable PnP solve
		return False, None, None, None

	f = float(max(image_width, image_height))
	cx = float(image_width) / 2.0
	cy = float(image_height) / 2.0
	K = np.array([
		[f, 0.0, cx],
		[0.0, f, cy],
		[0.0, 0.0, 1.0],
	], dtype=np.float32)

	dist_coeffs = np.zeros((4, 1), dtype=np.float32)

	# Choose a PnP algorithm based on the number of points.
	n_pts = obj_pts.shape[0]
	if n_pts >= 6:
		flag = cv2.SOLVEPNP_ITERATIVE
	else:
		# For 4–5 points, use an AP3P-style solver which is designed
		# for the minimal PnP case; this avoids the DLT initialisation
		# that requires 6+ points.
		flag = cv2.SOLVEPNP_AP3P

	try:
		success, rvec, tvec = cv2.solvePnP(
			objectPoints=obj_pts,
			imagePoints=img_pts,
			cameraMatrix=K,
			distCoeffs=dist_coeffs,
			flags=flag,
		)
	except cv2.error as e:
		print(f"[PnP-only] cv2.solvePnP failed with {n_pts} points and flag={flag}: {e}")
		return False, None, None, None

	# Compute reprojection error on the actual correspondence points
	# returned by `_get_pnp_correspondences` (obj_pts, img_pts).
	pts_3d = obj_pts.reshape(-1, 3)
	img_pts_2d = img_pts.reshape(-1, 2)

	def reproj_err(rvec_c, tvec_c):
		proj, _ = cv2.projectPoints(pts_3d.astype(np.float32), rvec_c, tvec_c, K, None)
		proj = proj.reshape(-1, 2)
		return float(np.mean(np.linalg.norm(proj - img_pts_2d, axis=1)))

	# Original reprojection error
	err_orig = reproj_err(rvec, tvec)

	# Candidate: rotate 180° about X (flip Y,Z) to handle flipped pose ambiguity
	R_mat, _ = cv2.Rodrigues(rvec)
	t_vec = tvec.reshape(3,)
	Rx_pi = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
	R2 = Rx_pi @ R_mat
	rvec2, _ = cv2.Rodrigues(R2)
	tvec2 = (Rx_pi @ t_vec).reshape(3, 1)
	err_flip = reproj_err(rvec2, tvec2)

	if err_flip + 1e-6 < err_orig:
		# adopt flipped pose
		rvec = rvec2
		tvec = tvec2

	return bool(success), rvec.astype(np.float32), tvec.astype(np.float32), K


def project_glasses_keypoints_pnp(rvec, tvec, K, image_width, image_height, kp_indices=None, visibility=2):
	"""Project 3D CAD glasses keypoints using the PnP pose.

	This uses cv2.projectPoints with the (rvec, tvec, K) estimated by
	:func:`estimate_glasses_pose_pnp`. It is a direct way to check the
	pose quality without involving pyrender.
	"""
	_check_glasses_3d()

	# By default project every CAD landmark slot so debug overlays can
	# show all available glasses keypoints (including those without
	# a 2D correspondence used for the PnP fit).
	if kp_indices is None:
		kp_indices = list(range(GLASSES_3D_LANDMARKS.shape[0]))

	pts_3d = np.asarray([GLASSES_3D_LANDMARKS[idx] for idx in kp_indices], dtype=np.float32).reshape(-1, 3)
	dist_coeffs = np.zeros((4, 1), dtype=np.float32)
	img_pts, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, dist_coeffs)
	img_pts = img_pts.reshape(-1, 2)

	# Camera-space Z test to avoid unstable projections for points behind the camera.
	R_cam, _ = cv2.Rodrigues(rvec)
	t_cam = tvec.reshape(3, 1)
	pts_cam = (R_cam @ pts_3d.T + t_cam).T  # (N,3)
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
# PYRENDER RENDERING
# -------------------------------------------------------------------------


def load_glasses_mesh(glasses_obj_path):
	"""Load the glasses 3D mesh with trimesh and return a Trimesh object.

	We return the trimesh.Trimesh so the renderer can apply correct
	pre-scaling, centering and per-render transforms before converting
	to a pyrender.Mesh.
	"""

	if not os.path.isfile(glasses_obj_path):
		raise FileNotFoundError(f"Glasses model not found: {glasses_obj_path}")

	mesh_trimesh = trimesh.load(glasses_obj_path, force='mesh')

	if isinstance(mesh_trimesh, trimesh.Scene):
		mesh_trimesh = mesh_trimesh.dump(concatenate=True)

	if not isinstance(mesh_trimesh, trimesh.Trimesh):
		# Ensure we have a Trimesh
		mesh_trimesh = trimesh.Trimesh(vertices=mesh_trimesh.vertices, faces=mesh_trimesh.faces)

	return mesh_trimesh


def render_glasses_rgba_pyrender(
	glasses_trimesh,
	image_width,
	image_height,
	center_xy,
	scale_px,
	roll=None,
	pitch=None,
	yaw=None,
	rot_mat: np.ndarray = None,
	pose_s: float | None = None,
	pose_t: np.ndarray | None = None,
	debug_no_projection=False,
	model_prescale=1.0,
	center_model=False,
	model_rotate=None,
	project_all_kpts=False,
	ssaa: int = 2,
	render_shadows: bool = False,
	ambient: float = 0.35,
	key_light: float = 3.0,
	fill_light: float = 1.5,
	debug_alpha: bool = False,
	force_basecolor: list | None = None,
):
	"""Render the glasses mesh with pyrender into an RGBA image.

	There are two code paths here:

	1) **PnP / weak-perspective path** (recommended):
	   - Caller passes ``rot_mat`` (3x3), ``pose_s`` (scalar), ``pose_t`` (2,)
	     coming from :func:`estimate_glasses_pose_weak`.
	   - We build a world transform (RS, T) and an orthographic camera such
	     that the rasterized pixels satisfy the same weak-perspective model
	
	         x_px = pose_s * (R[0] · X) + pose_t[0]
	         y_px = pose_s * (R[1] · X) + pose_t[1]

	     for any 3D point X in the same CAD coordinate system as
	     ``GLASSES_3D_LANDMARKS``. This mirrors ``project_glasses_keypoints_weak``.

	2) **Legacy Blender-like path** (fallback / debugging):
	   - Only uses ``center_xy`` and ``scale_px`` derived from a 2D box and
	     computes its own scale from the mesh bounding box and ``S_world``.
	   - This approximates the old Blender orthographic setup and is mainly
	     kept for ``--debug_no_projection`` / ``--auto_test``.

	The branch is selected automatically based on whether ``rot_mat``,
	``pose_s`` and ``pose_t`` are provided and whether ``debug_no_projection``
	is disabled.
	"""
	# Supersampling: do all math in render pixel space, then downsample.
	out_w = int(image_width)
	out_h = int(image_height)
	ssaa_i = int(max(1, ssaa))
	render_w = int(out_w * ssaa_i)
	render_h = int(out_h * ssaa_i)

	# Decide whether to use the weak-perspective pose (R, s, t) directly.
	# When rot_mat, pose_s and pose_t are provided (and we're not in
	# debug_no_projection mode), we build a world transform that exactly
	# reproduces x = s * (R_2x3 X) + t in pixel space via the ortho camera.
	use_pnp = (
		rot_mat is not None
		and pose_s is not None
		and pose_t is not None
		and not debug_no_projection
	)

	# Compute mesh bounds once (used by the legacy/non-PnP path)
	bounds_min, bounds_max = glasses_trimesh.bounds

	if use_pnp:
		# --- PnP-based branch: match weak-perspective exactly ---
		R = rot_mat.astype(np.float32)
		# pose_s, pose_t are in *output* pixel coords; scale to render pixels.
		s_eff = float(pose_s) * float(ssaa_i)
		t_vec = (np.asarray(pose_t, dtype=np.float32).reshape(2,) * float(ssaa_i))

		# Use S_world = 1.0 so that world coords in [-0.5,0.5] map to the image.
		S_world = 1.0

		# Build RS so that:
		#   px = image_width  * (Xw_x + 0.5) = s_eff * (R0 X) + t_x
		#   py = image_height * (0.5 - Xw_y) = s_eff * (R1 X) + t_y
		RS = np.zeros((3, 3), dtype=np.float32)
		RS[0, :] = (s_eff / float(render_w)) * R[0, :]
		RS[1, :] = -(s_eff / float(render_h)) * R[1, :]
		RS[2, 2] = 1.0

		tx = t_vec[0] / float(render_w) - 0.5
		ty = 0.5 - t_vec[1] / float(render_h)

		T = np.eye(4, dtype=np.float32)
		T[:3, :3] = RS
		T[0, 3] = tx
		T[1, 3] = ty
		T[2, 3] = 0.0
	else:
		# --- Legacy Blender-like branch: use center/scale in pixels ---
		# Determine rotation matrix: either use provided rot_mat (preferred),
		# or build from Euler angles.
		if rot_mat is not None:
			R = rot_mat.astype(np.float32)
		else:
			if roll is None or pitch is None or yaw is None:
				raise ValueError("Either rot_mat or roll/pitch/yaw must be provided")
			# Convert Euler angles (XYZ in radians) into a rotation matrix
			Rx = np.array(
				[
					[1, 0, 0],
					[0, math.cos(pitch), -math.sin(pitch)],
					[0, math.sin(pitch), math.cos(pitch)],
				],
				dtype=np.float32,
			)
			Ry = np.array(
				[
					[math.cos(yaw), 0, math.sin(yaw)],
					[0, 1, 0],
					[-math.sin(yaw), 0, math.cos(yaw)],
				],
				dtype=np.float32,
			)
			Rz = np.array(
				[
					[math.cos(roll), -math.sin(roll), 0],
					[math.sin(roll), math.cos(roll), 0],
					[0, 0, 1],
				],
				dtype=np.float32,
			)

			R = Rz @ Ry @ Rx

		# We'll follow the Blender logic for orthographic world units:
		# S_world = max(render_width, render_height)
		S_world = float(max(render_w, render_h))

		# Compute mesh local width (max extent in x) in model units
		size_local = bounds_max - bounds_min
		width_local = max(size_local[0], 1e-6)

		# Target world width (Blender): target_world_width = glasses_scale * S_world / W
		# scale_px is in output pixels; scale to render pixels.
		scale_px_r = float(scale_px) * float(ssaa_i)
		target_world_width = float(scale_px_r) * S_world / float(render_w)

		# Compute scale factor to convert model width to target world width.
		scale_factor = target_world_width / width_local

		if debug_no_projection:
			applied_scale = model_prescale
		else:
			applied_scale = model_prescale * float(scale_factor)

		if debug_no_projection:
			print(f"Model scale debug: width_local={width_local:.6f}, target_world_width={target_world_width:.6f}, scale_factor={scale_factor:.6f}, model_prescale={model_prescale}, applied_scale={applied_scale}")

		# Build RS (rotation * scale)
		S_mat = np.eye(3, dtype=np.float32) * applied_scale
		RS = R @ S_mat

		# Center in image: convert pixel center to Blender-like world coordinates
		# where X = (pos_x / W - 0.5) * S_world and Y = (0.5 - pos_y / H) * S_world
		cx, cy = center_xy
		cx_r = float(cx) * float(ssaa_i)
		cy_r = float(cy) * float(ssaa_i)
		tx = (cx_r / float(render_w) - 0.5) * S_world
		ty = (0.5 - cy_r / float(render_h)) * S_world

		T = np.eye(4, dtype=np.float32)
		T[:3, :3] = RS
		T[0, 3] = tx
		T[1, 3] = ty
		T[2, 3] = 0.0

	# Prepare trimesh copy, optionally center it, apply transforms and convert to pyrender.Mesh
	mesh_copy = glasses_trimesh.copy()

	# Optional pre-rotation (model coordinate frame adjustments)
	if model_rotate is not None and not use_pnp:
		# model_rotate supplied as degrees (rx,ry,rz)
		rx_d, ry_d, rz_d = model_rotate
		rx = math.radians(rx_d)
		ry = math.radians(ry_d)
		rz = math.radians(rz_d)
		Rx_pre = np.array([
			[1, 0, 0, 0],
			[0, math.cos(rx), -math.sin(rx), 0],
			[0, math.sin(rx), math.cos(rx), 0],
			[0, 0, 0, 1],
		], dtype=np.float32)
		Ry_pre = np.array([
			[math.cos(ry), 0, math.sin(ry), 0],
			[0, 1, 0, 0],
			[-math.sin(ry), 0, math.cos(ry), 0],
			[0, 0, 0, 1],
		], dtype=np.float32)
		Rz_pre = np.array([
			[math.cos(rz), -math.sin(rz), 0, 0],
			[math.sin(rz), math.cos(rz), 0, 0],
			[0, 0, 1, 0],
			[0, 0, 0, 1],
		], dtype=np.float32)
		pre_rot = Rz_pre @ Ry_pre @ Rx_pre
		mesh_copy.apply_transform(pre_rot)

	if center_model and not use_pnp:
		center = (bounds_min + bounds_max) / 2.0
		mesh_copy.apply_translation(-center)

	# Apply scale and rotation/translation as a single transform
	transform = np.eye(4, dtype=np.float32)
	transform[:3, :3] = RS
	transform[:3, 3] = [T[0, 3], T[1, 3], T[2, 3]]
	mesh_copy.apply_transform(transform)

	# Prefer the mesh's own materials if present; otherwise use a reasonable default.
	# Allow forcing a base color for debugging transparency/lighting issues.
	default_material = None
	try:
		vk = getattr(mesh_copy, "visual", None)
		kind = getattr(vk, "kind", None)
		if force_basecolor is not None:
			fc = [float(x) for x in force_basecolor]
			default_material = pyrender.MetallicRoughnessMaterial(
				baseColorFactor=[fc[0], fc[1], fc[2], fc[3]],
				metallicFactor=0.0,
				roughnessFactor=0.35,
				alphaMode="BLEND",
			)
		else:
			if kind is None or str(kind).lower() in {"none", "face"}:
				default_material = pyrender.MetallicRoughnessMaterial(
					baseColorFactor=[1.0, 1.0, 1.0, 1.0],
					metallicFactor=0.0,
					roughnessFactor=0.35,
					alphaMode="BLEND",
				)
	except Exception:
		default_material = None

	# If a forced basecolor is supplied, pass it to pyrender to override mesh materials.
	pyr_mesh = pyrender.Mesh.from_trimesh(mesh_copy, smooth=True, material=default_material)

	amb = float(max(0.0, ambient))
	scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[amb, amb, amb])
	scene.add(pyr_mesh, pose=np.eye(4, dtype=np.float32))

	# Orthographic camera roughly matching -1..1 plane to full image
	# Orthographic camera: set xmag/ymag to half the Blender ortho_scale so
	# that world coords in [-S_world/2, S_world/2] map to the image.
	cam = pyrender.OrthographicCamera(xmag=S_world / 2.0, ymag=S_world / 2.0)
	cam_pose = np.eye(4, dtype=np.float32)
	cam_pose[2, 3] = 10.0
	scene.add(cam, pose=cam_pose)

	# A simple 3-light rig tends to look much better than a single head-on sun.
	# Key light (camera-ish), plus two fills.
	key_i = float(max(0.0, key_light))
	fill_i = float(max(0.0, fill_light))
	light_key = pyrender.DirectionalLight(color=np.ones(3), intensity=key_i)
	light_fill = pyrender.DirectionalLight(color=np.ones(3), intensity=fill_i)
	light_rim = pyrender.DirectionalLight(color=np.ones(3), intensity=fill_i * 0.75)

	pose_key = cam_pose.copy()
	pose_fill = cam_pose.copy()
	pose_rim = cam_pose.copy()
	# Rotate fill/rim around Z and X to change direction.
	pose_fill[:3, :3] = (np.array([
		[0.8660254, -0.5, 0.0],
		[0.5, 0.8660254, 0.0],
		[0.0, 0.0, 1.0],
	], dtype=np.float32) @ pose_fill[:3, :3])
	pose_rim[:3, :3] = (np.array([
		[0.5, 0.0, 0.8660254],
		[0.0, 1.0, 0.0],
		[-0.8660254, 0.0, 0.5],
	], dtype=np.float32) @ pose_rim[:3, :3])

	scene.add(light_key, pose=pose_key)
	scene.add(light_fill, pose=pose_fill)
	scene.add(light_rim, pose=pose_rim)

	# Request color + depth so we can compute correct transparency
	color, depth = _render_scene_rgba_pyrender(scene, render_w, render_h, render_shadows=render_shadows)

	# If depth is available, build an alpha mask: where depth is finite there
	# is geometry, elsewhere it's background and should be transparent.
	h, w = render_h, render_w
	if depth is not None:
		# pyrender may return 0 for background pixels or np.inf; treat
		# non-positive or non-finite depths as background (transparent).
		finite = np.isfinite(depth)
		positive = depth > 0
		mask = finite & positive
		alpha_mask = (mask.astype(np.uint8) * 255).reshape(h, w, 1)
		if debug_no_projection:
			print(f"Depth stats: min={np.nanmin(depth):.6f}, max={np.nanmax(depth):.6f}, finite_pixels={mask.sum()}/{mask.size}")
	else:
		alpha_mask = np.full((h, w, 1), 255, dtype=np.uint8)

	# Normalize/convert the returned color array to uint8 RGB(A).
	if color.dtype == np.float32 or color.dtype == np.float64:
		color = np.clip(color, 0.0, 1.0)
		color = (color * 255.0).astype(np.uint8)
	else:
		color = color.astype(np.uint8)

	# Detect and fix premultiplied-alpha outputs from some pyrender builds.
	if color.ndim == 3 and color.shape[2] == 4:
		alpha_ch = color[..., 3]
		rgb = color[..., :3]
		if debug_alpha:
			print(f"[debug_alpha] color.shape={color.shape} alpha.min={int(alpha_ch.min())} alpha.max={int(alpha_ch.max())} rgb.min={int(rgb.min())} rgb.max={int(rgb.max())}")
			# Save raw premultiplied image for inspection
			tmp = Path(tempfile.gettempdir()) / f"pyrender_raw_{uuid.uuid4().hex}.png"
			try:
				Image.fromarray(color).save(tmp)
				print(f"[debug_alpha] wrote raw pyrender output to: {tmp}")
			except Exception as e:
				print(f"[debug_alpha] failed to write raw pyrender output: {e}")
		# Only consider pixels where alpha>0
		alpha_pos = alpha_ch > 0
		if np.any(alpha_pos):
			# premultiplied if for most alpha>0 pixels, rgb channels <= alpha
			check = (rgb <= alpha_ch[..., None]).all(axis=2)
			prem_frac = float(np.sum(check & alpha_pos) / np.sum(alpha_pos))
			if debug_alpha:
				print(f"[debug_alpha] premultiplied fraction={prem_frac:.3f}")
			if prem_frac > 0.85:
				# Un-premultiply into straight alpha space
				alpha_f = alpha_ch.astype(np.float32) / 255.0
				nz = alpha_f > 0
				scale = np.zeros_like(alpha_f, dtype=np.float32)
				scale[nz] = 1.0 / alpha_f[nz]
				rgb_un = (rgb.astype(np.float32) * scale[..., None])
				rgb_un = np.clip(rgb_un, 0.0, 255.0).astype(np.uint8)
				color[..., :3] = rgb_un
				# Save unpremultiplied result for inspection
				tmp2 = Path(tempfile.gettempdir()) / f"pyrender_unpremult_{uuid.uuid4().hex}.png"
				try:
					Image.fromarray(color).save(tmp2)
					print(f"[debug_alpha] wrote unpremultiplied pyrender output to: {tmp2}")
				except Exception as e:
					print(f"[debug_alpha] failed to write unpremultiplied pyrender output: {e}")

	# If only RGB, append the computed alpha mask.
	if color.ndim == 3 and color.shape[2] == 3:
		color = np.concatenate([color, alpha_mask], axis=2)
	elif color.ndim == 3 and color.shape[2] == 4:
		# Combine renderer alpha with our depth-based mask so background is transparent
		# Preserve material transparency: multiply existing alpha by depth mask.
		existing_a = color[..., 3].astype(np.float32) / 255.0
		mask_a = alpha_mask[:, :, 0].astype(np.float32) / 255.0
		new_a = (existing_a * mask_a * 255.0).astype(np.uint8)
		color[..., 3] = new_a
	else:
		# Fallback: reshape and apply mask
		color = color.reshape((h, w, -1))
		if color.shape[2] == 3:
			color = np.concatenate([color, alpha_mask], axis=2)
		elif color.shape[2] == 4:
			existing_a = color[..., 3].astype(np.float32) / 255.0
			mask_a = alpha_mask[:, :, 0].astype(np.float32) / 255.0
			color[..., 3] = (existing_a * mask_a * 255.0).astype(np.uint8)

	# Downsample to target resolution for antialiasing.
	img = _downsample_rgba_to_target(color, out_w, out_h)

	# Compute projected keypoints in pixel coords using the exact same
	# transforms we applied to the mesh so overlays are pixel-accurate.
	# Steps: take GLASSES_3D_LANDMARKS, apply pre-rotation (if any),
	# optional centering, then apply scale/rotation RS and translation tx,ty,
	# then map world X/Y to pixel coords.
	# Build pre-rotation matrix (4x4) for legacy path; PnP branch ignores
	# model_rotate/center_model to keep the pose consistent with the CAD
	# coordinates used for R,s,t.
	pre_rot_mat = np.eye(4, dtype=np.float32)
	if model_rotate is not None and not use_pnp:
		rx_d, ry_d, rz_d = model_rotate
		rx = math.radians(rx_d)
		ry = math.radians(ry_d)
		rz = math.radians(rz_d)
		Rx_pre = np.array([
			[1, 0, 0, 0],
			[0, math.cos(rx), -math.sin(rx), 0],
			[0, math.sin(rx), math.cos(rx), 0],
			[0, 0, 0, 1],
		], dtype=np.float32)
		Ry_pre = np.array([
			[math.cos(ry), 0, math.sin(ry), 0],
			[0, 1, 0, 0],
			[-math.sin(ry), 0, math.cos(ry), 0],
			[0, 0, 0, 1],
		], dtype=np.float32)
		Rz_pre = np.array([
			[math.cos(rz), -math.sin(rz), 0, 0],
			[math.sin(rz), math.cos(rz), 0, 0],
			[0, 0, 1, 0],
			[0, 0, 0, 1],
		], dtype=np.float32)
		pre_rot_mat = (Rz_pre @ Ry_pre @ Rx_pre)

	if use_pnp:
		center_vec = np.zeros(3, dtype=np.float32)
		full_transform = T.copy()
	else:
		center_vec = (bounds_min + bounds_max) / 2.0
		full_transform = np.eye(4, dtype=np.float32)
		full_transform[:3, :3] = RS
		full_transform[:3, 3] = [T[0, 3], T[1, 3], T[2, 3]]

	proj_kpts = []
	if project_all_kpts:
		kp_indices = list(range(GLASSES_3D_LANDMARKS.shape[0]))
	else:
		kp_indices = GLASSES_KEYPOINT_IDXS
    
	for idx in kp_indices:
		X = GLASSES_3D_LANDMARKS[idx]
		# If CAD keypoint undefined (missing ID in the XML), return invisible marker
		if GLASSES_3D_LANDMARKS_DEFINED is not None and not bool(GLASSES_3D_LANDMARKS_DEFINED[idx]):
			proj_kpts.append([0.0, 0.0, 0])
			continue
		X = X.astype(np.float32)
		Xh = np.array([X[0], X[1], X[2], 1.0], dtype=np.float32)
		Xh = pre_rot_mat @ Xh
		if center_model:
			Xh[:3] = Xh[:3] - center_vec
		Xw = full_transform @ Xh
		Xw_x = float(Xw[0])
		Xw_y = float(Xw[1])
		px_r = (Xw_x / S_world + 0.5) * render_w
		py_r = (0.5 - Xw_y / S_world) * render_h
		proj_kpts.append([px_r / float(ssaa_i), py_r / float(ssaa_i), 2])

	return img, proj_kpts


def render_glasses_rgba_pyrender_pnp(
	glasses_trimesh,
	image_width,
	image_height,
	rvec,
	tvec,
	K,
	model_prescale=1.0,
	center_model=False,
	model_rotate=None,
	pnp_mesh_auto_align=False,
	pnp_mesh_align_translate_z=False,
	ssaa: int = 2,
	render_shadows: bool = False,
	ambient: float = 0.35,
	key_light: float = 3.0,
	fill_light: float = 1.5,
	debug_alpha: bool = False,
	force_basecolor: list | None = None,
):
	"""Render glasses using a perspective camera driven by OpenCV PnP pose.

	This path ignores the weak-perspective parameters entirely and instead
	uses the full-perspective pose (rvec, tvec) and camera intrinsics K as
	estimated by :func:`estimate_glasses_pose_pnp`.

	The goal is that the rasterized model aligns with the projections from
	:func:`project_glasses_keypoints_pnp`, i.e. the same OpenCV pinhole
	camera model.
	"""
	out_w = int(image_width)
	out_h = int(image_height)
	ssaa_i = int(max(1, ssaa))
	render_w = int(out_w * ssaa_i)
	render_h = int(out_h * ssaa_i)

	mesh_copy = glasses_trimesh.copy()

	# Apply the same user-specified mesh-space adjustments used elsewhere.
	# These are intended to reconcile differences between the mesh file's
	# local coordinate frame/units and the CAD landmark coordinate system.
	if model_rotate is not None:
		# model_rotate supplied as degrees (rx,ry,rz)
		rx_d, ry_d, rz_d = model_rotate
		rx = math.radians(rx_d)
		ry = math.radians(ry_d)
		rz = math.radians(rz_d)
		Rx_pre = np.array([
			[1, 0, 0, 0],
			[0, math.cos(rx), -math.sin(rx), 0],
			[0, math.sin(rx), math.cos(rx), 0],
			[0, 0, 0, 1],
		], dtype=np.float32)
		Ry_pre = np.array([
			[math.cos(ry), 0, math.sin(ry), 0],
			[0, 1, 0, 0],
			[-math.sin(ry), 0, math.cos(ry), 0],
			[0, 0, 0, 1],
		], dtype=np.float32)
		Rz_pre = np.array([
			[math.cos(rz), -math.sin(rz), 0, 0],
			[math.sin(rz), math.cos(rz), 0, 0],
			[0, 0, 1, 0],
			[0, 0, 0, 1],
		], dtype=np.float32)
		pre_rot = (Rz_pre @ Ry_pre @ Rx_pre)
		mesh_copy.apply_transform(pre_rot)

	if center_model:
		bmin, bmax = mesh_copy.bounds
		center = (bmin + bmax) / 2.0
		mesh_copy.apply_translation(-center)

	# Uniform mesh prescale (units / general sizing)
	try:
		prescale_f = float(model_prescale)
	except Exception:
		prescale_f = 1.0
	if abs(prescale_f - 1.0) > 1e-9:
		S_pre = np.eye(4, dtype=np.float32)
		S_pre[:3, :3] *= prescale_f
		mesh_copy.apply_transform(S_pre)

	# Optional: auto-align mesh to CAD landmarks. This is OFF by default because
	# it applies a mesh-only 3D transform; in perspective, any Z translation
	# changes apparent size and can make XML landmarks look "bigger" than the mesh.
	if pnp_mesh_auto_align:
		try:
			# Mesh bounds/center
			mesh_bounds_min, mesh_bounds_max = mesh_copy.bounds
			mesh_center = mesh_copy.centroid
			mesh_size = mesh_bounds_max - mesh_bounds_min

			# Landmarks bounds/center (ignore all-zero placeholder rows)
			lm = GLASSES_3D_LANDMARKS
			if GLASSES_3D_LANDMARKS_DEFINED is not None:
				valid_mask = GLASSES_3D_LANDMARKS_DEFINED.astype(bool)
			else:
				valid_mask = np.any(lm != 0.0, axis=1)
			if np.any(valid_mask):
				lm_nz = lm[valid_mask]
				lm_min = lm_nz.min(axis=0)
				lm_max = lm_nz.max(axis=0)
				landmarks_center = lm_nz.mean(axis=0)
			else:
				lm_min = lm.min(axis=0)
				lm_max = lm.max(axis=0)
				landmarks_center = lm.mean(axis=0)
			landmarks_size = lm_max - lm_min

			# Scale using frontal (XY) extents only.
			mesh_xy = np.asarray(mesh_size[:2], dtype=np.float32)
			lm_xy = np.asarray(landmarks_size[:2], dtype=np.float32)
			mean_mesh_xy = float(np.maximum(mesh_xy, 1e-9).mean())
			mean_lm_xy = float(np.maximum(lm_xy, 1e-9).mean())
			if mean_mesh_xy > 1e-9 and mean_lm_xy > 1e-9:
				scale_factor = mean_lm_xy / mean_mesh_xy
			else:
				scale_factor = 1.0

			# Translate mostly in XY; Z translation is optional.
			delta = (landmarks_center - mesh_center * scale_factor).astype(np.float32)
			if not pnp_mesh_align_translate_z:
				delta[2] = 0.0

			T_align = np.eye(4, dtype=np.float32)
			T_align[:3, :3] *= float(scale_factor)
			T_align[:3, 3] = delta
			mesh_copy.apply_transform(T_align)

			print(
				f"[PnP-render] mesh auto-align: scale={float(scale_factor):.6g}, "
				f"translate={delta.tolist()}, mesh_xy={mesh_xy.tolist()}, lm_xy={lm_xy.tolist()}, "
				f"translate_z={bool(pnp_mesh_align_translate_z)}"
			)
		except Exception as e:
			print(f"[PnP-render] mesh/CAD auto-align failed: {e}")

	# Convert to a pyrender mesh
	default_material = None
	try:
		vk = getattr(mesh_copy, "visual", None)
		kind = getattr(vk, "kind", None)
		if kind is None or str(kind).lower() in {"none", "face"}:
			default_material = pyrender.MetallicRoughnessMaterial(
				baseColorFactor=[1.0, 1.0, 1.0, 1.0],
				metallicFactor=0.0,
				roughnessFactor=0.35,
				alphaMode="BLEND",
			)
	except Exception:
		default_material = None

	# If a forced basecolor is supplied, pass it to pyrender to override mesh materials.
	pyr_mesh = pyrender.Mesh.from_trimesh(mesh_copy, smooth=True, material=default_material)

	amb = float(max(0.0, ambient))
	scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[amb, amb, amb])
	scene.add(pyr_mesh, pose=np.eye(4, dtype=np.float32))

	# Use an intrinsics camera so (fx, fy, cx, cy) match cv2.projectPoints.
	# If we supersample, scale intrinsics so the rendered pixels match.
	fx = float(K[0, 0]) * float(ssaa_i)
	fy = float(K[1, 1]) * float(ssaa_i)
	cx = float(K[0, 2]) * float(ssaa_i)
	cy = float(K[1, 2]) * float(ssaa_i)
	camera = pyrender.IntrinsicsCamera(
		fx=fx,
		fy=fy,
		cx=cx,
		cy=cy,
		znear=0.01,
		zfar=10000.0,
	)

	# OpenCV camera coords: x right, y down, z forward.
	# OpenGL/pyrender camera coords: x right, y up, z backward (camera looks down -Z).
	# Convert pose by flipping Y and Z axes.
	R_cw_cv, _ = cv2.Rodrigues(rvec)
	t_cv = tvec.reshape(3, )
	C = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
	R_cw_gl = C @ R_cw_cv
	t_gl = C @ t_cv

	# Build camera-to-world pose from world-to-camera extrinsics.
	R_wc_gl = R_cw_gl.T
	t_wc_gl = -R_wc_gl @ t_gl
	T_cw_gl = np.eye(4, dtype=np.float32)
	T_cw_gl[:3, :3] = R_wc_gl.astype(np.float32)
	T_cw_gl[:3, 3] = t_wc_gl.astype(np.float32)

	scene.add(camera, pose=T_cw_gl)

	# Light rig: key at camera, plus a couple of fills.
	key_i = float(max(0.0, key_light))
	fill_i = float(max(0.0, fill_light))
	light_key = pyrender.DirectionalLight(color=np.ones(3), intensity=key_i)
	light_fill = pyrender.DirectionalLight(color=np.ones(3), intensity=fill_i)
	light_rim = pyrender.DirectionalLight(color=np.ones(3), intensity=fill_i * 0.75)

	pose_key = T_cw_gl.copy()
	pose_fill = T_cw_gl.copy()
	pose_rim = T_cw_gl.copy()
	pose_fill[:3, :3] = (np.array([
		[0.8660254, -0.5, 0.0],
		[0.5, 0.8660254, 0.0],
		[0.0, 0.0, 1.0],
	], dtype=np.float32) @ pose_fill[:3, :3])
	pose_rim[:3, :3] = (np.array([
		[0.5, 0.0, 0.8660254],
		[0.0, 1.0, 0.0],
		[-0.8660254, 0.0, 0.5],
	], dtype=np.float32) @ pose_rim[:3, :3])

	scene.add(light_key, pose=pose_key)
	scene.add(light_fill, pose=pose_fill)
	scene.add(light_rim, pose=pose_rim)

	color, depth = _render_scene_rgba_pyrender(scene, render_w, render_h, render_shadows=render_shadows)

	h, w = render_h, render_w
	if depth is not None:
		finite = np.isfinite(depth)
		positive = depth > 0
		mask = finite & positive
		alpha_mask = (mask.astype(np.uint8) * 255).reshape(h, w, 1)
	else:
		alpha_mask = np.full((h, w, 1), 255, dtype=np.uint8)

	if color.dtype == np.float32 or color.dtype == np.float64:
		color = np.clip(color, 0.0, 1.0)
		color = (color * 255.0).astype(np.uint8)
	else:
		color = color.astype(np.uint8)

	# Detect and fix premultiplied-alpha outputs from some pyrender builds.
	if color.ndim == 3 and color.shape[2] == 4:
		alpha_ch = color[..., 3]
		rgb = color[..., :3]
		if debug_alpha:
			print(f"[debug_alpha] PnP color.shape={color.shape} alpha.min={int(alpha_ch.min())} alpha.max={int(alpha_ch.max())} rgb.min={int(rgb.min())} rgb.max={int(rgb.max())}")
			# Save raw premultiplied image for inspection
			tmp = Path(tempfile.gettempdir()) / f"pyrender_pnp_raw_{uuid.uuid4().hex}.png"
			try:
				Image.fromarray(color).save(tmp)
				print(f"[debug_alpha] wrote raw PnP pyrender output to: {tmp}")
			except Exception as e:
				print(f"[debug_alpha] failed to write raw PnP pyrender output: {e}")
		alpha_pos = alpha_ch > 0
		if np.any(alpha_pos):
			check = (rgb <= alpha_ch[..., None]).all(axis=2)
			prem_frac = float(np.sum(check & alpha_pos) / np.sum(alpha_pos))
			if debug_alpha:
				print(f"[debug_alpha] PnP premultiplied fraction={prem_frac:.3f}")
			if prem_frac > 0.85:
				alpha_f = alpha_ch.astype(np.float32) / 255.0
				nz = alpha_f > 0
				scale = np.zeros_like(alpha_f, dtype=np.float32)
				scale[nz] = 1.0 / alpha_f[nz]
				rgb_un = (rgb.astype(np.float32) * scale[..., None])
				rgb_un = np.clip(rgb_un, 0.0, 255.0).astype(np.uint8)
				color[..., :3] = rgb_un
				# Save unpremultiplied result for inspection
				tmp2 = Path(tempfile.gettempdir()) / f"pyrender_pnp_unpremult_{uuid.uuid4().hex}.png"
				try:
					Image.fromarray(color).save(tmp2)
					print(f"[debug_alpha] wrote unpremultiplied PnP pyrender output to: {tmp2}")
				except Exception as e:
					print(f"[debug_alpha] failed to write unpremultiplied PnP pyrender output: {e}")
				rgb_un = np.clip(rgb_un, 0.0, 255.0).astype(np.uint8)
				color[..., :3] = rgb_un

	if color.ndim == 3 and color.shape[2] == 3:
		color = np.concatenate([color, alpha_mask], axis=2)
	elif color.ndim == 3 and color.shape[2] == 4:
		# Combine renderer alpha with our depth-based mask so background is transparent
		existing_a = color[..., 3].astype(np.float32) / 255.0
		mask_a = alpha_mask[:, :, 0].astype(np.float32) / 255.0
		new_a = (existing_a * mask_a * 255.0).astype(np.uint8)
		color[..., 3] = new_a
	else:
		color = color.reshape((h, w, -1))
		if color.shape[2] == 3:
			color = np.concatenate([color, alpha_mask], axis=2)
		elif color.shape[2] == 4:
			color[..., 3] = alpha_mask[:, :, 0]

	img = _downsample_rgba_to_target(color, out_w, out_h)

	return img


def composite_glasses_on_image(bg_img_path, glasses_rgba_img, out_img_path):
	"""Alpha-composite RGBA glasses Image onto background file path."""
	if not os.path.isfile(bg_img_path):
		print(f"Warning: background image not found: {bg_img_path}. Saving glasses RGBA directly to output.")
		# Save the glasses image (with transparency) as a fallback so user can inspect it
		glasses_rgba_img.save(out_img_path)
		return

	try:
		bg = Image.open(bg_img_path)
	except Exception as e:
		print(f"Error: failed to open background image {bg_img_path}: {e}. Saving glasses RGBA directly to output.")
		glasses_rgba_img.save(out_img_path)
		return

	# Log debug info about sizes/modes
	try:
		bg_mode = bg.mode
		bg_size = bg.size
		gl_size = glasses_rgba_img.size
		gl_mode = glasses_rgba_img.mode
		print(f"Compositing: bg='{bg_img_path}' mode={bg_mode} size={bg_size}; glasses mode={gl_mode} size={gl_size}")
	except Exception:
		pass

	bg = bg.convert("RGB")

	if glasses_rgba_img.size != bg.size:
		glasses_rgba_img = glasses_rgba_img.resize(bg.size, Image.BICUBIC)

	bg_rgba = bg.convert("RGBA")
	try:
		bg_rgba.alpha_composite(glasses_rgba_img)
	except ValueError as e:
		# Sometimes PIL raises when modes/sizes mismatch; save debug files and rethrow
		print(f"Error during alpha_composite: {e}. Saving debug images to help investigation.")
		debug_bg = out_img_path + ".debug_bg.png"
		debug_gl = out_img_path + ".debug_glasses.png"
		bg_rgba.save(debug_bg)
		glasses_rgba_img.save(debug_gl)
		print(f"Wrote debug files: {debug_bg}, {debug_gl}")
		# As a fallback, paste on top using simple composite
		out = Image.alpha_composite(bg_rgba, glasses_rgba_img.convert("RGBA"))
		out = out.convert("RGB")
		out.save(out_img_path, quality=95)
		return

	out = bg_rgba.convert("RGB")
	out.save(out_img_path, quality=95)



def draw_landmarks_with_indices(img, kpts, color=(0, 255, 0), idxs=None):
	"""Draw keypoints and their indices onto a PIL Image.

	kpts: iterable of [x,y,v] or Nx3 array-like. Only v>0 are drawn.
	idxs: optional list of labels to use for each keypoint (length should match kpts).
	"""
	im = img.copy()
	draw = ImageDraw.Draw(im)
	font = ImageFont.load_default()
	r = max(1, int(max(im.size) * 0.005))
	for i, kp in enumerate(kpts):
		# accept list/tuple or numpy row
		try:
			x, y, v = kp
		except Exception:
			continue
		if v <= 0:
			continue
		leftUp = (x - r, y - r)
		rightDown = (x + r, y + r)
		draw.ellipse([leftUp, rightDown], fill=color)
		# choose label from idxs if provided, otherwise use sequential index
		label = None
		if idxs is not None and i < len(idxs):
			label = str(idxs[i])
		else:
			label = str(i)
		# offset text slightly so it doesn't overlap the marker
		tx = x + r + 1
		ty = max(0, y - r - 1)
		draw.text((tx, ty), label, fill=color, font=font)
	return im


def debug_pnp_only(coco, image_by_id, images_root, out_images_root, num_face_kpts):
	"""Run a simple PnP-only debug: project CAD kpts onto faces and save overlays.

	For each annotation, we:
	- load its image
	- estimate pose via cv2.solvePnP on the 3D-2D correspondences
	- project all GLASSES_KEYPOINT_IDXS with cv2.projectPoints
	- draw face landmarks (green, indexed) and projected glasses kpts
	  (red, labeled with CAD indices) and save under out_images_root.

	No pyrender is involved here; this is just to verify that the PnP
	math and correspondences are correct.
	"""
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
			print(f"[PnP-only] Failed to get landmarks for {img_path}: {e}")
			continue

		w = img_info["width"]
		h = img_info["height"]
		ok, rvec, tvec, K = estimate_glasses_pose_pnp(landmarks_2d, w, h)
		if not ok:
			print(f"[PnP-only] PnP failed or insufficient correspondences for {img_path}")
			continue

		# Print found rotation (Rodrigues) and translation for debugging.
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
			# Discard degenerate solutions where roll (X Euler) is near 0
			threshold_deg = 40.0
			if abs(rx) < threshold_deg:
				print(f"[PnP-only] Discarding {img_info['file_name']} due to degenerate roll={rx:.1f}°")
				continue
			# Print roll/pitch/yaw and translation
			print(f"[PnP-only] {img_info['file_name']}: roll={rx:.1f} pitch={ry:.1f} yaw={rz:.1f} tvec={tvec.flatten().tolist()}")
		except Exception:
			print(f"[PnP-only] {img_info['file_name']}: rvec and tvec computed; skipping degenerate check")

		proj_kpts = project_glasses_keypoints_pnp(rvec, tvec, K, w, h)

		try:
			img = Image.open(img_path).convert("RGB")
		except Exception as e:
			print(f"[PnP-only] Failed to open image {img_path}: {e}")
			continue

		# Draw face landmarks (green) and projected glasses kpts (red with CAD ids)
		orig_kpts = np.asarray(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
		img_dbg = draw_landmarks_with_indices(img, orig_kpts.tolist(), color=(0, 255, 0))
		# Label projected landmarks with their CAD indices (0..N-1)
		full_idxs = list(range(GLASSES_3D_LANDMARKS.shape[0]))
		img_dbg = draw_landmarks_with_indices(img_dbg, proj_kpts, color=(255, 0, 0), idxs=full_idxs)

		out_name = f"{Path(img_info['file_name']).stem}_pnp_debug.jpg"
		out_path = out_images_root / out_name
		img_dbg.save(out_path, quality=95)
		print(f"[PnP-only] Wrote PnP debug overlay: {out_path}")

	# Only process first few images? For now we run all; user can filter
	# annotations externally if needed.


# -------------------------------------------------------------------------
# MAIN LOOP (mirrors original generate_glasses_dataset.main)
# -------------------------------------------------------------------------


def main():
	args = parse_args()

	global GLASSES_3D_LANDMARKS
	global GLASSES_3D_LANDMARKS_DEFINED
	GLASSES_3D_LANDMARKS, GLASSES_3D_LANDMARKS_DEFINED = load_glasses_3d_landmarks_from_xml(args.glasses_xml)
	_check_glasses_3d()
	# We output *all* glasses landmarks that exist in the XML/CAD array,
	# not only those used for pose correspondences.
	glasses_output_idxs = list(range(int(GLASSES_3D_LANDMARKS.shape[0])))

	images_root = Path(args.images_root)
	out_images_root = Path(args.out_images)
	out_images_root.mkdir(parents=True, exist_ok=True)

	# tmp directory no longer used for RGBA files on disk, but keep in
	# case caller expects it or wants to debug per-face renders.
	tmp_rgba_dir = out_images_root / "_tmp_glasses_rgba"
	tmp_rgba_dir.mkdir(parents=True, exist_ok=True)

	with open(args.annotations, "r") as f:
		coco = json.load(f)

	images = coco["images"]
	annotations = coco["annotations"]
	categories = coco.get("categories", [])

	# IMPORTANT: output annotations contain ONLY the newly generated projected
	# glasses keypoints. Ensure COCO categories reflect the new keypoint count.
	used_category_ids = {a.get("category_id") for a in annotations if "category_id" in a}
	n_out_kpts = int(len(glasses_output_idxs))
	new_kpt_names = [f"kpt_{i}" for i in range(n_out_kpts)]
	if categories:
		for cat in categories:
			if cat.get("id") in used_category_ids or len(used_category_ids) == 0:
				cat["keypoints"] = list(new_kpt_names)
				cat["skeleton"] = []
	else:
		# Minimal valid COCO category for keypoints tasks.
		categories = [
			{
				"id": 1,
				"name": "face",
				"supercategory": "person",
				"keypoints": list(new_kpt_names),
				"skeleton": [],
			}
		]

	image_by_id = {img["id"]: img for img in images}

	# Collect pose logs for all processed frames
	pose_log = []

	# If requested, run the simple PnP-only debug path and exit.
	if args.pnp_only:
		print("Running PnP-only debug mode (no pyrender rendering)...")
		debug_pnp_only(coco, image_by_id, images_root, out_images_root, args.num_face_keypoints)
		print(f"PnP-only debug images written under: {out_images_root}")
		return

	next_image_id = max(img["id"] for img in images) + 1 if images else 1
	next_ann_id = max(ann["id"] for ann in annotations) + 1 if annotations else 1

	new_images = []
	new_annotations = []

	# Preload glasses mesh once
	glasses_trimesh = load_glasses_mesh(args.glasses_obj)

	# If auto_test is requested, only render the first annotation with
	# a set of candidate rotations/prescales and exit.
	if args.auto_test:
		if len(annotations) == 0:
			print("No annotations to test on.")
			return
		ann = annotations[0]
		img = image_by_id.get(ann["image_id"])
		if img is None:
			print("First annotation has no image; cannot run auto_test.")
			return

		img_path = images_root / img["file_name"]
		if not img_path.is_file():
			print(f"Image for auto_test not found: {img_path}")
			return

		landmarks_2d = get_face_landmarks_from_annotation(ann, num_kpts_face=args.num_face_keypoints)
		try:
			R, s, t = estimate_glasses_pose_weak(landmarks_2d)
		except Exception as e:
			print(f"Pose estimation failed for auto_test image {img_path}: {e}")
			return

		glasses_kpts = project_glasses_keypoints_weak(R, s, t)
		g_arr = np.asarray(glasses_kpts, dtype=np.float32)
		center_xy = g_arr[:, :2].mean(axis=0)
		xs = g_arr[:, 0]
		ys = g_arr[:, 1]
		width = float(xs.max() - xs.min())
		height = float(ys.max() - ys.min())
		scale_px = max(width, height)

		candidate_rots = [(-90, 0, 0), (90, 0, 0), (0, -90, 0), (0, 90, 0), (180, 0, 0), (0,0,0)]
		candidate_prescales = [1.0, 10.0, 100.0]

		print(f"Auto-test: rendering {len(candidate_rots)*len(candidate_prescales)} combos to {tmp_rgba_dir}")
		for pr in candidate_prescales:
			for rr in candidate_rots:
				img_rgba, _ = render_glasses_rgba_pyrender(
					glasses_trimesh=glasses_trimesh,
					image_width=img["width"],
					image_height=img["height"],
					center_xy=center_xy,
					scale_px=scale_px,
					roll=0.0,
					pitch=0.0,
					yaw=0.0,
					debug_no_projection=True,
					model_prescale=pr,
					center_model=args.center_model,
					model_rotate=rr,
					project_all_kpts=True,
					debug_alpha=args.debug_alpha,
					force_basecolor=args.force_basecolor,
				)
				out_name = tmp_rgba_dir / f"auto_rot_{int(rr[0])}_{int(rr[1])}_{int(rr[2])}_prescale_{pr:.1f}.png"
				img_rgba.save(out_name)
				print(f"Wrote {out_name}")

		print("Auto-test complete — inspect the files in the tmp dir and pick a good rotation/prescale to pass as --model_rotate and --model_prescale")
		return

	for ann in annotations:
		img = image_by_id.get(ann["image_id"])
		if img is None:
			continue

		img_path = images_root / img["file_name"]
		if not img_path.is_file():
			print(f"Warning: image not found: {img_path}")
			continue

		landmarks_2d = get_face_landmarks_from_annotation(
			ann, num_kpts_face=args.num_face_keypoints
		)

		# Choose pose estimation strategy.
		use_pnp_pose = not args.identity_projection
		pnp_ok = False
		R = None
		s = 1.0
		t = np.zeros(2, dtype=np.float32)
		rvec = None
		tvec = None
		K = None

		if args.identity_projection:
			# Use identity rotation and unit scale; translate model mean to image center.
			# Compute mean only over the defined CAD keypoints (ignore zero placeholders).
			kp_idxs = GLASSES_KEYPOINT_IDXS
			try:
				kp_pts = GLASSES_3D_LANDMARKS[kp_idxs]
			except Exception:
				kp_pts = GLASSES_3D_LANDMARKS
			# Filter out any all-zero rows in case the CAD array has empty slots
			nz_mask = np.any(kp_pts != 0.0, axis=1)
			if np.any(nz_mask):
				mu_X = kp_pts[nz_mask].mean(axis=0)
			else:
				mu_X = GLASSES_3D_LANDMARKS.mean(axis=0)
			R = np.eye(3, dtype=np.float32)
			s = 1.0
			img_center = np.array([img["width"] / 2.0, img["height"] / 2.0], dtype=np.float32)
			t = (img_center - (R @ mu_X)[:2]).astype(np.float32)
			print(f"Using identity projection: R=I, s=1, t={t}")
			glasses_kpts = project_glasses_keypoints_weak(R, s, t, kp_indices=glasses_output_idxs)
		else:
			# Prefer full-perspective OpenCV PnP pose; fall back to weak-perspective
			# if PnP is not available or fails.
			ok, rvec_pnp, tvec_pnp, K_pnp = estimate_glasses_pose_pnp(
				landmarks_2d,
				img["width"],
				img["height"],
			)
			if ok:
				pnp_ok = True
				rvec = rvec_pnp
				tvec = tvec_pnp
				K = K_pnp

				# Print PnP solution for this image (Rodrigues, translation, Euler degrees)
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
					# Discard degenerate solutions where roll (X Euler) is near 0
					threshold_deg = 30.0
					if abs(rx) < threshold_deg:
						print(f"[PnP] Discarding {img['file_name']} due to degenerate roll={rx:.1f}°")
						pose_log.append({
							"image": img["file_name"],
							"method": "pnp",
							"roll": float(rx),
							"pitch": float(ry),
							"yaw": float(rz),
							"tvec": tvec.flatten().tolist(),
							"discarded": True,
						})
						continue
					print(f"[PnP] {img['file_name']}: roll={rx:.1f} pitch={ry:.1f} yaw={rz:.1f} tvec={tvec.flatten().tolist()}")
					pose_log.append({
						"image": img["file_name"],
						"method": "pnp",
						"roll": float(rx),
						"pitch": float(ry),
						"yaw": float(rz),
						"tvec": tvec.flatten().tolist(),
						"discarded": False,
					})
				except Exception:
					print(f"[PnP] {img['file_name']}: rvec={rvec.flatten().tolist()} tvec={tvec.flatten().tolist()}")
				glasses_kpts = project_glasses_keypoints_pnp(
					rvec,
					tvec,
					K,
					img["width"],
					img["height"],
					kp_indices=glasses_output_idxs,
				)
			else:
				try:
					R, s, t = estimate_glasses_pose_weak(landmarks_2d)
				except Exception as e:
					print(f"Pose estimation failed for image {img_path}: {e}")
					continue
				glasses_kpts = project_glasses_keypoints_weak(R, s, t, kp_indices=glasses_output_idxs)
				# Log weak-perspective pose (no euler conversion needed here)
				pose_log.append({
					"image": img["file_name"],
					"method": "weak",
					"s": float(s),
					"t": [float(x) for x in t.tolist()],
					"discarded": False,
				})

		g_arr = np.asarray(glasses_kpts, dtype=np.float32)
		center_xy = g_arr[:, :2].mean(axis=0)
		xs = g_arr[:, 0]
		ys = g_arr[:, 1]
		width = float(xs.max() - xs.min())
		height = float(ys.max() - ys.min())
		scale_px = max(width, height)

		for v in range(args.variants_per_image):
			# For now we do not jitter the PnP pose; each variant reuses the
			# same rvec/tvec (or weak-perspective parameters if PnP is not used).
			jitter_s = 1.0
			scale_j = scale_px

			# Effective weak-perspective scale only matters for the legacy
			# weak-perspective rendering path.
			jitter_factor = float(scale_j / scale_px) if scale_px > 1e-6 else 1.0
			s_eff = float(s * jitter_factor)

			# Preserve the original dataset relative path/name for output images.
			# If multiple variants are requested, avoid name collisions by placing
			# each variant under a separate subfolder while keeping the same basename.
			orig_rel = Path(img["file_name"])
			if orig_rel.is_absolute():
				# COCO file_name should be relative; if not, strip to basename as a fallback.
				orig_rel = Path(orig_rel.name)

			if args.variants_per_image > 1:
				out_rel = Path(f"v{v}") / orig_rel
			else:
				out_rel = orig_rel

			out_img_path = out_images_root / out_rel
			out_img_path.parent.mkdir(parents=True, exist_ok=True)

			# 1) Render glasses in memory (RGBA).
			if pnp_ok and rvec is not None and tvec is not None and K is not None:
				glasses_rgba_img = render_glasses_rgba_pyrender_pnp(
					glasses_trimesh=glasses_trimesh,
					image_width=img["width"],
					image_height=img["height"],
					rvec=rvec,
					tvec=tvec,
					K=K,
					model_prescale=args.model_prescale,
					center_model=args.center_model,
					model_rotate=args.model_rotate,
					pnp_mesh_auto_align=args.pnp_mesh_auto_align,
					pnp_mesh_align_translate_z=args.pnp_mesh_align_translate_z,
					ssaa=args.ssaa,
					render_shadows=args.render_shadows,
					ambient=args.ambient,
					key_light=args.key_light,
					fill_light=args.fill_light,
					debug_alpha=args.debug_alpha,
					force_basecolor=args.force_basecolor,
				)
				proj_kpts = glasses_kpts
			else:
				# Fall back to the weak-perspective pyrender path.
				glasses_rgba_img, proj_kpts = render_glasses_rgba_pyrender(
					glasses_trimesh=glasses_trimesh,
					image_width=img["width"],
					image_height=img["height"],
					center_xy=center_xy,
					scale_px=scale_j,
					rot_mat=R,
					pose_s=s_eff,
					pose_t=t,
					debug_no_projection=args.debug_no_projection,
					model_prescale=args.model_prescale,
					center_model=args.center_model,
					model_rotate=args.model_rotate,
					# Always project all keypoints because they are exported to COCO.
					project_all_kpts=True,
					ssaa=args.ssaa,
					render_shadows=args.render_shadows,
					ambient=args.ambient,
					key_light=args.key_light,
					fill_light=args.fill_light,
					debug_alpha=args.debug_alpha,
					force_basecolor=args.force_basecolor,
				)

			# Optional: save per-face RGBA for debugging (with landmarks overlay if requested)
			debug_rgba_path = tmp_rgba_dir / f"glasses_{ann['id']}_v{v}.png"
			if args.debug_landmarks:
				try:
					# Draw glasses keypoints (red) and face landmarks with indices (green)
					debug_rg = glasses_rgba_img.convert("RGBA")
					# Draw glasses keypoints with their CAD indices (red)
					full_idxs = list(range(GLASSES_3D_LANDMARKS.shape[0]))
					debug_rg = draw_landmarks_with_indices(debug_rg, proj_kpts, color=(255, 0, 0), idxs=full_idxs)
					# original face keypoints with visibility
					orig_kpts_local = np.asarray(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
					debug_rg = draw_landmarks_with_indices(debug_rg, orig_kpts_local.tolist(), color=(0, 255, 0))
					debug_rg.save(debug_rgba_path)
				except Exception as e:
					print(f"Warning: failed to draw landmarks on debug RGBA: {e}")
					glasses_rgba_img.save(debug_rgba_path)
			else:
				glasses_rgba_img.save(debug_rgba_path)

			# 2) Composite onto background
			composite_glasses_on_image(
				bg_img_path=str(img_path),
				glasses_rgba_img=glasses_rgba_img,
				out_img_path=str(out_img_path),
			)

			# If requested, draw projected glasses keypoints on the final composite for debug
			if args.debug_landmarks:
				try:
					comp = Image.open(out_img_path).convert("RGB")
					# draw face landmarks with indices (green) then glasses keypoints with CAD indices (red)
					orig_kpts_local = np.asarray(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
					comp_debug = draw_landmarks_with_indices(comp, orig_kpts_local.tolist(), color=(0, 255, 0))
					comp_debug = draw_landmarks_with_indices(comp_debug, proj_kpts, color=(255, 0, 0), idxs=GLASSES_KEYPOINT_IDXS)
					comp_debug_path = str(out_img_path) + ".kpts.jpg"
					comp_debug.save(comp_debug_path, quality=90)
				except Exception as e:
					print(f"Warning: failed to save debug composite with landmarks: {e}")

			# 3) Create new COCO image entry
			new_img = {
				"id": next_image_id,
				"file_name": out_rel.as_posix(),
				"width": img["width"],
				"height": img["height"],
			}
			new_images.append(new_img)

			# 4) Write ONLY the projected keypoints (2D) into the new COCO annotation
			proj_kpts_flat = [float(coord) for kp in proj_kpts for coord in kp]

			# Preserve the original dataset face bbox (do not recompute).
			bbox = ann.get("bbox")
			area = ann.get("area")
			if area is None and bbox is not None and len(bbox) == 4:
				try:
					area = float(bbox[2]) * float(bbox[3])
				except Exception:
					area = None

			# Preserve the input annotation schema as much as possible.
			# Only the keypoint-related fields are updated; 3D-related fields
			# (if present in the input) are intentionally left empty.
			new_ann = dict(ann)
			new_ann["id"] = next_ann_id
			new_ann["image_id"] = next_image_id
			new_ann["keypoints"] = proj_kpts_flat
			# COCO num_keypoints counts labeled points (v>0).
			new_ann["num_keypoints"] = int(sum(1 for kp in proj_kpts if float(kp[2]) > 0))
			if bbox is not None:
				new_ann["bbox"] = bbox
			if area is not None:
				new_ann["area"] = float(area)
			new_ann["iscrowd"] = int(new_ann.get("iscrowd", 0))
			if "keypoints_3d" in new_ann:
				new_ann["keypoints_3d"] = []
			if "parameters_3d" in new_ann:
				new_ann["parameters_3d"] = {}
			new_annotations.append(new_ann)

			next_image_id += 1
			next_ann_id += 1

	# Output COCO contains ONLY the synthetic images/annotations with projected keypoints.
	out_coco = {
		"info": coco.get("info", {}),
		"licenses": coco.get("licenses", []),
		"images": new_images,
		"annotations": new_annotations,
		"categories": categories,
	}

	with open(args.out_annotations, "w") as f:
		json.dump(out_coco, f)

	# Also write a copy of the final COCO JSON under <out_images>/annotations/
	# using the same basename as the input annotations file.
	try:
		out_ann_dir = out_images_root / "annotations"
		out_ann_dir.mkdir(parents=True, exist_ok=True)
		out_ann_path = out_ann_dir / Path(args.annotations).name
		with open(out_ann_path, "w") as f:
			json.dump(out_coco, f)
		print(f"Saved per-run annotations copy to: {out_ann_path}")
	except Exception as e:
		print(f"Warning: failed to write <out_images>/annotations copy: {e}")

	print(f"Saved synthetic annotations to: {args.out_annotations}")
	print(f"Synthetic images written under: {out_images_root}")

	# Write pose log
	try:
		log_path = out_images_root / "pnp_poses_log.json"
		with open(log_path, "w") as lf:
			json.dump(pose_log, lf, indent=2)
		print(f"Wrote pose log: {log_path}")
	except Exception as e:
		print(f"Warning: failed to write pose log: {e}")


if __name__ == "__main__":
	main()

