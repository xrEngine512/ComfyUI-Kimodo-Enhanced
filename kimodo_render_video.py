"""Render Kimodo motion as a ComfyUI VIDEO value.

The ``mesh`` style retargets motion onto a bundled CC0 rigged human. Other
styles render camera-projected joint/bone proxies directly from Kimodo motion.
"""
from __future__ import annotations

import math
from fractions import Fraction


_STYLE_OPTIONS = ["mesh", "silhouette", "openpose_2d", "openpose_3d", "scail"]
_OPENPOSE_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170), (255, 0, 85),
]
_SCAIL_COLORS = [
    (230, 72, 50), (240, 128, 48), (246, 190, 66), (204, 214, 74),
    (118, 196, 89), (62, 183, 128), (55, 174, 174), (61, 146, 202),
    (92, 112, 214), (139, 94, 206), (184, 82, 175), (216, 73, 124),
]
_MAX_VIDEO_PIXELS = 140_000_000


def _as_list(value):
    """Convert numpy/torch arrays to nested Python lists without requiring either."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _finite_vec(v):
    try:
        return len(v) >= 3 and all(math.isfinite(float(v[i])) for i in range(3))
    except (TypeError, ValueError, OverflowError):
        return False


def _vec3(value, default=(0.0, 0.0, 0.0)):
    if isinstance(value, dict):
        raw = (value.get("x", default[0]), value.get("y", default[1]), value.get("z", default[2]))
    elif isinstance(value, (list, tuple)) and len(value) >= 3:
        raw = value[:3]
    else:
        raw = default
    try:
        result = tuple(float(x) for x in raw)
        return result if all(math.isfinite(x) for x in result) else tuple(default)
    except (TypeError, ValueError, OverflowError):
        return tuple(default)


def _normalize(v, fallback):
    n = math.sqrt(sum(float(x) * float(x) for x in v))
    if n < 1e-8 or not math.isfinite(n):
        return tuple(fallback)
    return tuple(float(x) / n for x in v)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _flip_threejs(v):
    # Match ComfyUI's SAM3D camera convention for external Y-up skeletons.
    return (float(v[0]), -float(v[1]), -float(v[2]))


def _quaternion_camera_axes(q):
    if isinstance(q, dict):
        w, x, y, z = (float(q.get(k, d)) for k, d in (("w", 1.0), ("x", 0.0), ("y", 0.0), ("z", 0.0)))
    elif isinstance(q, (list, tuple)) and len(q) >= 4:
        w, x, y, z = (float(v) for v in q[:4])
    else:
        return None
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n < 1e-8 or not math.isfinite(n):
        return None
    w, x, y, z = w/n, x/n, y/n, z/n
    # Rotation matrix columns are the camera's local axes in world space.
    col_x = (1 - 2*(y*y + z*z), 2*(x*y + z*w), 2*(x*z - y*w))
    col_y = (2*(x*y - z*w), 1 - 2*(x*x + z*z), 2*(y*z + x*w))
    col_z = (2*(x*z + y*w), 2*(y*z - x*w), 1 - 2*(x*x + y*y))
    right = _normalize(_flip_threejs(col_x), (1.0, 0.0, 0.0))
    # _project uses screen_y = center_y - camera_up_y * focal / depth, so this
    # must be camera-local +Y (up), not the image-down row used by some Comfy
    # render helpers. Negating it here made quaternion cameras appear rolled 180°.
    up = _normalize(_flip_threejs(col_y), (0.0, -1.0, 0.0))
    forward = _normalize(_flip_threejs(tuple(-v for v in col_z)), (0.0, 0.0, 1.0))
    return right, up, forward


def _bounds_center(frames):
    mins = [float("inf")] * 3
    maxs = [float("-inf")] * 3
    for frame in frames:
        for point in frame:
            if not _finite_vec(point):
                continue
            for axis in range(3):
                value = float(point[axis])
                mins[axis] = min(mins[axis], value)
                maxs[axis] = max(maxs[axis], value)
    if not math.isfinite(mins[0]):
        raise ValueError("Kimodo motion contains no finite joint positions to render")
    center = tuple((mins[i] + maxs[i]) * 0.5 for i in range(3))
    extent = max(maxs[i] - mins[i] for i in range(3))
    return center, max(extent, 0.1)


def _camera_for_frames(frames, width, height, camera_info):
    camera_info = camera_info if isinstance(camera_info, dict) else None
    points = frames
    transform_camera = camera_info is not None
    if transform_camera:
        points = [[_flip_threejs(p) if _finite_vec(p) else p for p in frame] for frame in frames]
    center, extent = _bounds_center(points)
    fov = 50.0
    if camera_info is not None:
        try:
            fov = float(camera_info.get("fov", 0.0) or 50.0)
        except (TypeError, ValueError):
            fov = 50.0
    fov = min(120.0, max(10.0, fov))
    zoom = 1.0
    if camera_info is not None:
        try:
            zoom = max(0.01, float(camera_info.get("zoom", 1.0) or 1.0))
        except (TypeError, ValueError):
            zoom = 1.0
    focal = height / (2.0 * math.tan(math.radians(fov) * 0.5)) * zoom

    if camera_info is None:
        target = center
        distance = max(1.0, extent / (2.0 * math.tan(math.radians(fov) * 0.5) * 0.78))
        eye = (center[0], center[1], center[2] + distance)
        forward = _normalize(tuple(target[i] - eye[i] for i in range(3)), (0.0, 0.0, -1.0))
        right = _normalize(_cross(forward, (0.0, 1.0, 0.0)), (1.0, 0.0, 0.0))
        up = _normalize(_cross(right, forward), (0.0, 1.0, 0.0))
    else:
        pos_raw = _vec3(camera_info.get("position"))
        target_raw = _vec3(camera_info.get("target"))
        eye_base = _flip_threejs(pos_raw)
        target = _flip_threejs(target_raw)
        offset = tuple(eye_base[i] - target[i] for i in range(3))
        offset_len = math.sqrt(sum(x*x for x in offset))
        axes = _quaternion_camera_axes(camera_info.get("quaternion"))
        auto_distance = max(
            1.0,
            extent * zoom / (2.0 * math.tan(math.radians(fov) * 0.5) * 0.78),
        )
        eye = eye_base if offset_len >= 1e-6 else None
        if axes is not None:
            right, up, forward = axes
            if eye is None:
                eye = tuple(target[i] - forward[i] * auto_distance for i in range(3))
        else:
            if eye is None:
                eye = (target[0], target[1], target[2] + auto_distance)
            forward = _normalize(tuple(target[i] - eye[i] for i in range(3)), (0.0, 0.0, -1.0))
            world_up = (0.0, -1.0, 0.0)
            right = _normalize(_cross(forward, world_up), (1.0, 0.0, 0.0))
            up = _normalize(_cross(right, forward), world_up)

    return points, {
        "eye": eye, "right": right, "up": up, "forward": forward,
        "focal": focal, "transform_camera": transform_camera,
    }


def _project(point, camera, width, height):
    if not _finite_vec(point):
        return None
    rel = tuple(float(point[i]) - camera["eye"][i] for i in range(3))
    x = _dot(rel, camera["right"])
    y = _dot(rel, camera["up"])
    z = _dot(rel, camera["forward"])
    if z <= 0.02:
        return None
    return (width * 0.5 + camera["focal"] * x / z,
            height * 0.5 - camera["focal"] * y / z,
            z)


def _background_image(background, frame_index, width, height):
    from PIL import Image
    if background is None:
        return Image.new("RGB", (width, height), (24, 27, 32))
    import numpy as np
    image = background
    if hasattr(image, "detach"):
        image = image.detach().cpu()
    if hasattr(image, "shape") and len(image.shape) >= 4:
        image = image[min(frame_index, int(image.shape[0]) - 1)]
    if hasattr(image, "detach"):
        image = image.detach().cpu()
    if hasattr(image, "numpy"):
        array = image.numpy()
    else:
        array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError("background must be a ComfyUI IMAGE batch [B,H,W,C]")
    array = array[..., :3]
    if array.dtype != np.uint8:
        scale = 255.0 if array.size == 0 or float(np.nanmax(array)) <= 1.01 else 1.0
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
        array = np.clip(array * scale, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB").resize((width, height), Image.Resampling.LANCZOS)


def _render_frame(points, parents, camera, width, height, style, background):
    from PIL import Image, ImageDraw
    aa = 2 if max(width, height) <= 768 else 1
    canvas = background.resize((width * aa, height * aa), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(canvas)
    projected = [_project(point, camera, width, height) for point in points]

    if style == "silhouette":
        base_width, palette = 14, [(224, 229, 235)]
    elif style == "mesh":
        raise RuntimeError("The rigged mesh renderer must handle render_style='mesh'")
    elif style == "scail":
        base_width, palette = 7, _SCAIL_COLORS
    elif style == "openpose_2d":
        base_width, palette = 3, _OPENPOSE_COLORS
    else:
        base_width, palette = 5, _OPENPOSE_COLORS

    edges = []
    for joint_index, parent in enumerate(parents):
        if parent is None:
            continue
        try:
            parent = int(parent)
        except (TypeError, ValueError):
            continue
        if parent < 0 or parent >= len(points) or parent == joint_index:
            continue
        a, b = projected[parent], projected[joint_index]
        if a is None or b is None:
            continue
        edges.append((-(a[2] + b[2]) * 0.5, joint_index, a, b))
    edges.sort(key=lambda item: item[0])  # far limbs first

    for _, joint_index, a, b in edges:
        depth_scale = max(0.55, min(1.5, 3.0 / max(0.1, (a[2] + b[2]) * 0.5)))
        line_width = max(1, int(round(base_width * depth_scale * aa)))
        color = palette[(joint_index - 1) % len(palette)]
        xy = (int(round(a[0] * aa)), int(round(a[1] * aa)),
              int(round(b[0] * aa)), int(round(b[1] * aa)))
        draw.line(xy, fill=color, width=line_width, joint="curve")
        radius = max(1, line_width // 2)
        for x, y in ((xy[0], xy[1]), (xy[2], xy[3])):
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)

    if style in ("openpose_2d", "openpose_3d", "scail"):
        for joint_index, point in enumerate(projected):
            if point is None:
                continue
            r = max(2, int(round((2 if style == "openpose_2d" else 3) * aa)))
            x, y = int(round(point[0] * aa)), int(round(point[1] * aa))
            draw.ellipse((x-r, y-r, x+r, y+r), fill=(245, 245, 245), outline=(30, 30, 30), width=max(1, aa))
    if aa > 1:
        canvas = canvas.resize((width, height), Image.Resampling.LANCZOS)
    return canvas.convert("RGB")


def _motion_video_frames(motion, width, height, background, camera_info, render_style):
    import numpy as np
    import torch

    output = getattr(motion, "output_dict", None)
    if not isinstance(output, dict) or "posed_joints" not in output:
        raise ValueError("motion input is not a Kimodo Sampler motion (missing posed_joints)")
    all_joints = _as_list(output["posed_joints"])
    if not isinstance(all_joints, list) or not all_joints:
        raise ValueError("Kimodo motion contains no samples")
    # A VIDEO output represents one clip; choose the first Kimodo sample.
    frames = all_joints[0]
    if not isinstance(frames, list) or not frames:
        raise ValueError("Selected Kimodo sample contains no frames")
    joint_count = len(frames[0]) if isinstance(frames[0], list) else 0
    if joint_count < 2:
        raise ValueError("Kimodo motion must contain at least two joints per frame")
    if any(not isinstance(frame, list) or len(frame) != joint_count for frame in frames):
        raise ValueError("Kimodo motion has inconsistent joint counts between frames")

    parents = _as_list(getattr(motion, "joint_parents", None))
    if not parents or len(parents) != joint_count:
        parents = [-1] + list(range(joint_count - 1))
    if isinstance(render_style, dict):
        style = str(render_style.get("render_style", "openpose_3d"))
    else:
        style = str(render_style or "openpose_3d")
    if style not in _STYLE_OPTIONS:
        style = "openpose_3d"

    bg_shape = getattr(background, "shape", None) if background is not None else None
    bg_h = int(bg_shape[-3]) if bg_shape is not None and len(bg_shape) >= 3 else 0
    bg_w = int(bg_shape[-2]) if bg_shape is not None and len(bg_shape) >= 3 else 0
    if width == 0 and height == 0:
        width, height = (bg_w, bg_h) if bg_w > 0 and bg_h > 0 else (512, 512)
    elif width == 0:
        aspect = (bg_w / bg_h) if bg_w > 0 and bg_h > 0 else 1.0
        width = max(1, int(round(height * aspect)))
    elif height == 0:
        aspect = (bg_h / bg_w) if bg_w > 0 and bg_h > 0 else 1.0
        height = max(1, int(round(width * aspect)))
    width, height = int(width), int(height)
    if width < 1 or height < 1 or width > 8192 or height > 8192:
        raise ValueError("render width and height must be between 1 and 8192 pixels")
    if len(frames) * width * height > _MAX_VIDEO_PIXELS:
        raise ValueError(
            f"Requested video is too large ({len(frames)} x {width} x {height} pixels). "
            "Reduce width/height or motion duration; the renderer limit is 140 million pixels."
        )

    camera_points, camera = _camera_for_frames(frames, width, height, camera_info)
    if style == "mesh":
        from kimodo_mesh_renderer import render_quaternius_mesh_video
        return render_quaternius_mesh_video(
            motion=motion,
            frame_count=len(frames),
            camera=camera,
            width=width,
            height=height,
            background=background,
            background_image_fn=_background_image,
        )

    result = torch.empty((len(frames), height, width, 3), dtype=torch.float32, device="cpu")
    for frame_index, points in enumerate(camera_points):
        bg = _background_image(background, frame_index, width, height)
        image = _render_frame(points, parents, camera, width, height, style, bg)
        array = np.asarray(image, dtype=np.uint8).copy()
        result[frame_index] = torch.from_numpy(array).to(dtype=torch.float32).div_(255.0)
    return result


class Kimodo_Render3DVideo:
    """Render Kimodo motion as one ComfyUI VIDEO clip."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion": ("KIMODO_MOTION", {"tooltip": "Motion output from Kimodo Sampler. The first sample is rendered."}),
                "width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8,
                                   "tooltip": "Output width. 0 uses the background size, or 512 when no background is connected."}),
                "height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8,
                                    "tooltip": "Output height. 0 uses the background size, or 512 when no background is connected."}),
                "render_style": (_STYLE_OPTIONS, {"default": "openpose_3d",
                                                  "tooltip": "mesh uses a bundled Quaternius CC0 rigged human retargeted to the motion; other styles visualize Kimodo joints."}),
            },
            "optional": {
                "background": ("IMAGE", {"tooltip": "Optional per-frame background; omitted means a dark canvas."}),
                "camera_info": ("LOAD3D_CAMERA", {"tooltip": "Optional camera from ComfyUI Create Camera Info or a Load3D camera node."}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "render"
    CATEGORY = "Kimodo/Render"

    def render(self, motion, width=0, height=0, render_style="openpose_3d", background=None, camera_info=None):
        from comfy_api.latest import InputImpl, Types
        frames = _motion_video_frames(motion, width, height, background, camera_info, render_style)
        fps = float(getattr(motion, "fps", 30.0) or 30.0)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        fps = min(120.0, max(1.0, fps))
        video = InputImpl.VideoFromComponents(
            Types.VideoComponents(images=frames, audio=None, frame_rate=Fraction(str(fps)))
        )
        return (video,)
