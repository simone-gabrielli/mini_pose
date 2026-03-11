# blender_render_jobs_44.py
# Blender 4.4 batch renderer for glasses RGBA overlays.
#
# Run:
#   blender -b --python blender_render_jobs_44.py -- --jobs /path/to/render_jobs.json
#
# The jobs JSON is produced by generate_glasses_dataset_blender.py

import bpy
import json
import os
import sys
from mathutils import Matrix, Vector

def _argv_after_dashes():
    argv = sys.argv
    if "--" not in argv:
        return []
    return argv[argv.index("--") + 1 :]

def _parse_args():
    argv = _argv_after_dashes()
    if "--jobs" not in argv:
        raise RuntimeError("Missing --jobs <path>")
    jobs_path = argv[argv.index("--jobs") + 1]
    verbose = ("--verbose" in argv)
    return jobs_path, verbose

def _reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def _engine_available(name: str) -> bool:
    try:
        enum_items = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
        return name in [e.identifier for e in enum_items]
    except Exception:
        return False

def _set_engine(scene, requested: str):
    # Blender 4.x typically: BLENDER_EEVEE_NEXT, CYCLES, WORKBENCH
    if requested and _engine_available(requested):
        scene.render.engine = requested
        return requested
    # Fallbacks
    if _engine_available("BLENDER_EEVEE_NEXT"):
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        return "BLENDER_EEVEE_NEXT"
    if _engine_available("BLENDER_EEVEE"):
        scene.render.engine = "BLENDER_EEVEE"
        return "BLENDER_EEVEE"
    scene.render.engine = "CYCLES"
    return "CYCLES"

def _import_model(model_path: str, verbose: bool = False):
    ext = os.path.splitext(model_path)[1].lower()

    if not os.path.isfile(model_path):
        raise RuntimeError(f"Model file not found: {model_path}")

    # Import WITHOUT applying axis conversion (identity mapping) by choosing
    # Blender-native forward/up where supported.
    if ext == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            try:
                bpy.ops.wm.obj_import(filepath=model_path, forward_axis='NEGATIVE_Y', up_axis='Z')
            except TypeError:
                bpy.ops.wm.obj_import(filepath=model_path)
        else:
            bpy.ops.import_scene.obj(filepath=model_path, axis_forward='-Y', axis_up='Z')
    elif ext == ".fbx":
        kwargs = dict(filepath=model_path, axis_forward='-Y', axis_up='Z')
        try:
            kwargs["apply_unit_scale"] = False
        except Exception:
            pass
        bpy.ops.import_scene.fbx(**kwargs)
    else:
        raise RuntimeError(f"Unsupported model extension: {ext}")

    mesh_objs = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    if not mesh_objs:
        raise RuntimeError(f"No mesh objects imported from {model_path}")

    # Join into one object so we can set matrix_world once
    if len(mesh_objs) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for o in mesh_objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = mesh_objs[0]
        bpy.ops.object.join()
        obj = bpy.context.view_layer.objects.active
    else:
        obj = mesh_objs[0]

    if verbose:
        print(f"[import] imported mesh object: {obj.name} from {model_path}")

    return obj

def _force_material_alpha(obj):
    # Try to ensure imported materials render with transparency.
    for slot in getattr(obj, "material_slots", []):
        mat = slot.material
        if mat is None:
            continue
        try:
            if hasattr(mat, "blend_method"):
                mat.blend_method = 'BLEND'
            if hasattr(mat, "shadow_method"):
                mat.shadow_method = 'HASHED'
        except Exception:
            pass

def _ensure_camera():
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    return cam_obj

def _ensure_lights(key_i: float, fill_i: float):
    # Remove existing lights
    for o in list(bpy.context.scene.objects):
        if o.type == "LIGHT":
            bpy.data.objects.remove(o, do_unlink=True)

    def add_sun(name, rot_xyz, energy):
        ldat = bpy.data.lights.new(name, type='SUN')
        ldat.energy = float(energy)
        lob = bpy.data.objects.new(name, ldat)
        bpy.context.collection.objects.link(lob)
        lob.rotation_euler = rot_xyz

    add_sun("Key",  (0.0, 0.0, 0.0), key_i)
    add_sun("Fill", (0.5, 0.0, 1.0), fill_i)
    add_sun("Rim",  (-0.5, 0.0, -1.0), fill_i * 0.75)


def _apply_lights_from_spec(lights_spec, fallback_key_i: float, fallback_fill_i: float, verbose: bool = False):
    """Apply per-job lighting.

    If lights_spec is None, falls back to legacy 3-sun setup using fallback intensities.
    lights_spec format (list of dicts):
      {name, type: SUN|POINT|AREA, location:[x,y,z], rotation_euler:[rx,ry,rz], energy, color:[r,g,b], size}
    """
    # Remove existing lights
    for o in list(bpy.context.scene.objects):
        if o.type == "LIGHT":
            bpy.data.objects.remove(o, do_unlink=True)

    if not lights_spec:
        _ensure_lights(float(fallback_key_i), float(fallback_fill_i))
        return

    if not isinstance(lights_spec, list):
        # Defensive fallback
        _ensure_lights(float(fallback_key_i), float(fallback_fill_i))
        return

    for idx, spec in enumerate(lights_spec):
        if not isinstance(spec, dict):
            continue

        name = str(spec.get("name") or f"Light_{idx}")
        ltype = str(spec.get("type") or "SUN").upper()
        if ltype not in {"SUN", "POINT", "AREA"}:
            ltype = "SUN"

        ldat = bpy.data.lights.new(name, type=ltype)
        try:
            ldat.energy = float(spec.get("energy", 1.0))
        except Exception:
            ldat.energy = 1.0

        col = spec.get("color", None)
        if isinstance(col, (list, tuple)) and len(col) >= 3:
            try:
                ldat.color = (float(col[0]), float(col[1]), float(col[2]))
            except Exception:
                pass

        if ltype == "AREA":
            try:
                ldat.size = float(spec.get("size", 1.0))
            except Exception:
                pass

        lob = bpy.data.objects.new(name, ldat)
        bpy.context.collection.objects.link(lob)

        loc = spec.get("location", None)
        if isinstance(loc, (list, tuple)) and len(loc) >= 3:
            try:
                lob.location = (float(loc[0]), float(loc[1]), float(loc[2]))
            except Exception:
                pass

        rot = spec.get("rotation_euler", None)
        if isinstance(rot, (list, tuple)) and len(rot) >= 3:
            try:
                lob.rotation_euler = (float(rot[0]), float(rot[1]), float(rot[2]))
            except Exception:
                pass

    if verbose:
        print(f"[lights] applied {len(lights_spec)} custom lights")

def _setup_render_base(scene):
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.resolution_percentage = 100
    # Keep colors predictable (optional)
    try:
        scene.view_settings.view_transform = 'Standard'
    except Exception:
        pass

def _set_resolution(scene, w: int, h: int, pixel_aspect_x: float = 1.0, pixel_aspect_y: float = 1.0):
    scene.render.resolution_x = int(w)
    scene.render.resolution_y = int(h)
    scene.render.pixel_aspect_x = float(pixel_aspect_x)
    scene.render.pixel_aspect_y = float(pixel_aspect_y)

def _world_to_pixel_cv(scene, cam_obj, p_world: Vector):
    # Blender normalized coordinates: x,y in [0,1], origin bottom-left.
    from bpy_extras.object_utils import world_to_camera_view
    co = world_to_camera_view(scene, cam_obj, p_world)
    u = float(co.x) * float(scene.render.resolution_x)
    v_bl = float(co.y) * float(scene.render.resolution_y)
    v = float(scene.render.resolution_y) - v_bl  # to OpenCV-like top-left origin
    return u, v, float(co.z)

def _configure_camera_ortho(scene, cam_obj, cam_mw_4x4, ortho_scale: float):
    cam = cam_obj.data
    cam.type = 'ORTHO'
    cam.ortho_scale = float(ortho_scale)
    try:
        cam.sensor_fit = 'HORIZONTAL'
    except Exception:
        pass
    cam_obj.matrix_world = Matrix(cam_mw_4x4)

def _configure_camera_persp_from_K(scene, cam_obj, cam_mw_4x4, fx, fy, cx, cy, sensor_width_mm=36.0):
    """
    Configure Blender camera to match OpenCV intrinsics K for the current render resolution.
    We:
      - set lens from fx (horizontal fit)
      - set pixel aspect ratio to match fy
      - solve shift_x/shift_y by numerical derivative so principal point matches exactly

    fx, fy, cx, cy MUST be specified in pixels for the *current render resolution*
    (i.e., already SSAA-scaled if you render at SSAA resolution).
    """
    W = float(scene.render.resolution_x)
    H = float(scene.render.resolution_y)

    cam = cam_obj.data
    cam.type = 'PERSP'
    cam.lens_unit = 'MILLIMETERS'
    cam.sensor_width = float(sensor_width_mm)
    try:
        cam.sensor_fit = 'HORIZONTAL'
    except Exception:
        pass

    # Lens from fx (horizontal fit)
    cam.lens = float(fx) * float(sensor_width_mm) / float(W)

    # Pixel aspect to reconcile fy (Blender uses horizontal fit as reference)
    pa_x = 1.0
    pa_y = float(fx) / float(fy) if float(fy) != 0.0 else 1.0
    scene.render.pixel_aspect_x = pa_x
    scene.render.pixel_aspect_y = pa_y

    cam_obj.matrix_world = Matrix(cam_mw_4x4)

    # Solve principal point shifts in normalized coords (origin bottom-left)
    # Desired normalized:
    x_des = float(cx) / float(W)
    y_des = 1.0 - (float(cy) / float(H))

    # Use a point on optical axis in front of camera
    p_axis_world = cam_obj.matrix_world @ Vector((0.0, 0.0, -1.0))

    # baseline
    cam.shift_x = 0.0
    cam.shift_y = 0.0
    x0, y0, _ = _world_to_pixel_cv(scene, cam_obj, p_axis_world)
    # Convert baseline pixel to normalized bottom-left
    x0n = x0 / W
    y0n = 1.0 - (y0 / H)

    eps = 1e-4

    # derivative wrt shift_x
    cam.shift_x = eps
    cam.shift_y = 0.0
    x1, y1, _ = _world_to_pixel_cv(scene, cam_obj, p_axis_world)
    x1n = x1 / W
    cam.shift_x = 0.0

    dx = (x1n - x0n) / eps if eps != 0.0 else 0.0
    if abs(dx) < 1e-9:
        dx = 1.0  # fallback

    # derivative wrt shift_y
    cam.shift_x = 0.0
    cam.shift_y = eps
    x2, y2, _ = _world_to_pixel_cv(scene, cam_obj, p_axis_world)
    y2n = 1.0 - (y2 / H)
    cam.shift_y = 0.0

    dy = (y2n - y0n) / eps if eps != 0.0 else 0.0
    if abs(dy) < 1e-9:
        dy = 1.0  # fallback

    # Required shifts
    cam.shift_x = (x_des - x0n) / dx
    cam.shift_y = (y_des - y0n) / dy

def _maybe_set_samples(scene, engine: str, samples: int):
    s = int(max(1, samples))
    if engine == "CYCLES":
        scene.cycles.samples = s
        try:
            scene.cycles.use_denoising = False
        except Exception:
            pass
    else:
        # Eevee / Eevee Next
        if hasattr(scene, "eevee"):
            try:
                scene.eevee.taa_render_samples = s
            except Exception:
                pass

def _render_to_png(scene, out_png: str):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    scene.render.filepath = out_png
    bpy.ops.render.render(write_still=True)

def main():
    jobs_path, verbose = _parse_args()
    with open(jobs_path, "r") as f:
        jobs_doc = json.load(f)

    jobs = jobs_doc.get("jobs", jobs_doc if isinstance(jobs_doc, list) else [])
    if not isinstance(jobs, list) or len(jobs) == 0:
        raise RuntimeError(f"No jobs found in: {jobs_path}")

    # One-time scene setup
    _reset_scene()
    scene = bpy.context.scene
    _setup_render_base(scene)

    # Import model once (assume a single model_path for all jobs)
    model_path = jobs[0]["model_path"]
    obj = _import_model(model_path, verbose=verbose)
    _force_material_alpha(obj)

    cam_obj = _ensure_camera()

    # Render loop
    for i, job in enumerate(jobs):
        out_png = job["out_rgba_png"]
        w = int(job["render_width"])
        h = int(job["render_height"])
        engine_req = job.get("engine", "")
        engine_used = _set_engine(scene, engine_req)
        _maybe_set_samples(scene, engine_used, int(job.get("samples", 64)))

        # Per-job lighting (randomized or fixed)
        _apply_lights_from_spec(
            job.get("lights", None),
            fallback_key_i=float(job.get("key_light", 3.0)),
            fallback_fill_i=float(job.get("fill_light", 1.5)),
            verbose=verbose,
        )

        # Ambient/world background strength
        try:
            world = scene.world
            if world is None:
                world = bpy.data.worlds.new("World")
                scene.world = world
            world.use_nodes = True
            bg = None
            for n in world.node_tree.nodes:
                if n.type == "BACKGROUND":
                    bg = n
                    break
            if bg is not None:
                bg.inputs[1].default_value = float(job.get("ambient", 0.35))
        except Exception:
            pass

        # Resolution/pixel aspect - may be overwritten in PnP camera config
        _set_resolution(scene, w, h, 1.0, 1.0)

        # Set object transform
        obj.matrix_world = Matrix(job["object_matrix_world"])

        cam_mode = job["camera"]["mode"]
        cam_mw = job["camera"]["matrix_world"]

        if cam_mode == "ORTHO":
            _configure_camera_ortho(scene, cam_obj, cam_mw, float(job["camera"].get("ortho_scale", 1.0)))
        elif cam_mode == "PERSP":
            fx = float(job["camera"]["fx"])
            fy = float(job["camera"]["fy"])
            cx = float(job["camera"]["cx"])
            cy = float(job["camera"]["cy"])
            _configure_camera_persp_from_K(scene, cam_obj, cam_mw, fx, fy, cx, cy, sensor_width_mm=float(job["camera"].get("sensor_width_mm", 36.0)))
        else:
            raise RuntimeError(f"Unknown camera mode: {cam_mode}")

        if verbose:
            print(f"[job {i+1}/{len(jobs)}] engine={engine_used} {w}x{h} cam={cam_mode} -> {out_png}")

        _render_to_png(scene, out_png)

        # Optional reprojection validation for PnP jobs
        if verbose and cam_mode == "PERSP" and "debug_correspondences" in job:
            corr = job["debug_correspondences"]
            obj_pts = corr.get("obj_pts", [])
            img_pts = corr.get("img_pts", [])
            if obj_pts and img_pts and len(obj_pts) == len(img_pts):
                errs = []
                for p3, p2 in zip(obj_pts, img_pts):
                    u, v, z = _world_to_pixel_cv(scene, cam_obj, Vector((float(p3[0]), float(p3[1]), float(p3[2]))))
                    du = u - float(p2[0])
                    dv = v - float(p2[1])
                    errs.append((du*du + dv*dv) ** 0.5)
                if errs:
                    mean_err = sum(errs) / len(errs)
                    print(f"  [PnP reproj check] mean_err_px={mean_err:.3f} over {len(errs)} points")

    if verbose:
        print("All jobs rendered successfully.")

if __name__ == "__main__":
    main()
