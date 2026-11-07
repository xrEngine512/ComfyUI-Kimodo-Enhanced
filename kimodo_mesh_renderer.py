"""Retarget Kimodo motion onto the bundled CC0 Quaternius humanoid and rasterize it.

All runtime imports (NumPy, SciPy, pygltflib, torch and ComfyUI's rasterizer) are
provided by the ComfyUI/Kimodo environment. This module is imported lazily only
when the renderer's ``mesh`` style is selected.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
from pygltflib import GLTF2
from scipy.spatial.transform import Rotation


# The bundled Quaternius Superhero Male has a UE-style 65-joint skeleton. These
# are explicit SOMA -> Quaternius aliases; Mixamo name-prefix matching alone
# cannot retarget this rig (pelvis/spine_01/upperarm_l/etc.). Fingers stay in the
# model's neutral pose, matching Kimodo Export GLB's safe default.
QUATERNIUS_BODY_MAPPING = {
    "hips": "pelvis",
    "spine1": "spine_01",
    "spine2": "spine_02",
    "chest": "spine_03",
    "neck1": "neck_01",
    "head": "Head",
    "leftshoulder": "clavicle_l",
    "leftarm": "upperarm_l",
    "leftforearm": "lowerarm_l",
    "lefthand": "hand_l",
    "rightshoulder": "clavicle_r",
    "rightarm": "upperarm_r",
    "rightforearm": "lowerarm_r",
    "righthand": "hand_r",
    "leftleg": "thigh_l",
    "leftshin": "calf_l",
    "leftfoot": "foot_l",
    "lefttoebase": "ball_l",
    "rightleg": "thigh_r",
    "rightshin": "calf_r",
    "rightfoot": "foot_r",
    "righttoebase": "ball_r",
}

# Warm skin tone with high-contrast facial cues so video models can identify
# the front of the character. These colors are rendered through ComfyUI's
# per-vertex semantic-color path, retaining a single shared depth buffer.
MESH_SKIN_COLOR = (0.76, 0.53, 0.42)
MESH_EYE_COLOR = (0.96, 0.93, 0.84)
MESH_BROW_COLOR = (0.15, 0.08, 0.055)

_COMPONENT_DTYPES = {
    5120: np.dtype("i1"),
    5121: np.dtype("u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
_TYPE_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


def _read_accessor(gltf, accessor_index: int, blob: bytes) -> np.ndarray:
    """Read a GLB accessor, honoring offsets/strides and normalized integer data."""
    accessor = gltf.accessors[int(accessor_index)]
    if accessor.sparse is not None:
        raise ValueError("The bundled rig uses no sparse accessors; cannot decode this GLB accessor")
    if accessor.bufferView is None:
        return np.zeros((int(accessor.count), _TYPE_WIDTHS[accessor.type]), dtype=np.float32)

    view = gltf.bufferViews[int(accessor.bufferView)]
    dtype = _COMPONENT_DTYPES.get(int(accessor.componentType))
    if dtype is None:
        raise ValueError(f"Unsupported glTF accessor component type: {accessor.componentType}")
    width = _TYPE_WIDTHS.get(accessor.type)
    if width is None:
        raise ValueError(f"Unsupported glTF accessor type: {accessor.type}")

    count = int(accessor.count)
    item_bytes = dtype.itemsize * width
    stride = int(view.byteStride or item_bytes)
    start = int(view.byteOffset or 0) + int(accessor.byteOffset or 0)
    if stride < item_bytes or start + max(0, count - 1) * stride + item_bytes > len(blob):
        raise ValueError(f"Invalid/out-of-range GLB buffer view for accessor {accessor_index}")
    data = np.ndarray(
        shape=(count, width), dtype=dtype, buffer=blob, offset=start,
        strides=(stride, dtype.itemsize),
    ).copy()

    if bool(accessor.normalized) and dtype.kind in "iu":
        data = data.astype(np.float32)
        if dtype.kind == "u":
            data /= float(np.iinfo(dtype).max)
        else:
            data = np.maximum(data / float(np.iinfo(dtype).max), -1.0)
    return data


def _node_local_matrix(node, overrides: dict | None = None) -> np.ndarray:
    """Build a glTF local matrix, applying sampled animation values if present."""
    overrides = overrides or {}
    if node.matrix is not None and len(node.matrix) == 16:
        if overrides:
            raise ValueError("Animated glTF nodes must use TRS, not a matrix transform")
        return np.asarray(node.matrix, dtype=np.float64).reshape(4, 4).T

    translation = np.asarray(node.translation or [0.0, 0.0, 0.0], dtype=np.float64)
    rotation = np.asarray(node.rotation or [0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    scale = np.asarray(node.scale or [1.0, 1.0, 1.0], dtype=np.float64)
    if "translation" in overrides:
        translation = np.asarray(overrides["translation"], dtype=np.float64)
    if "rotation" in overrides:
        rotation = np.asarray(overrides["rotation"], dtype=np.float64)
    if "scale" in overrides:
        scale = np.asarray(overrides["scale"], dtype=np.float64)

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(rotation).as_matrix() @ np.diag(scale)
    matrix[:3, 3] = translation
    return matrix


def _world_matrices(gltf, animated_values: dict[int, dict[str, np.ndarray]]) -> list[np.ndarray]:
    parents: dict[int, int] = {}
    for parent_index, node in enumerate(gltf.nodes):
        for child_index in node.children or []:
            parents[int(child_index)] = parent_index

    cache: dict[int, np.ndarray] = {}
    visiting: set[int] = set()

    def world(index: int) -> np.ndarray:
        if index in cache:
            return cache[index]
        if index in visiting:
            raise ValueError("Cycle detected in glTF node hierarchy")
        visiting.add(index)
        local = _node_local_matrix(gltf.nodes[index], animated_values.get(index))
        if index in parents:
            local = world(parents[index]) @ local
        visiting.remove(index)
        cache[index] = local
        return local

    return [world(i) for i in range(len(gltf.nodes))]


def _read_animation(gltf, blob: bytes, frame_count: int):
    if not gltf.animations:
        raise RuntimeError("Kimodo retargeting produced no animation for the bundled rig")
    animation = gltf.animations[0]
    channels: dict[int, dict[str, np.ndarray]] = {}
    for channel in animation.channels or []:
        target = channel.target
        if target is None or target.node is None or target.path not in ("translation", "rotation", "scale"):
            continue
        sampler = animation.samplers[int(channel.sampler)]
        if (sampler.interpolation or "LINEAR") not in ("LINEAR", "STEP"):
            raise ValueError(f"Unsupported GLB animation interpolation: {sampler.interpolation}")
        values = _read_accessor(gltf, int(sampler.output), blob)
        if values.shape[0] != frame_count:
            raise ValueError(
                f"Retargeted GLB has {values.shape[0]} keys for {frame_count} motion frames"
            )
        channels.setdefault(int(target.node), {})[target.path] = values
    if not channels:
        raise RuntimeError(
            "Kimodo retargeting matched no bones on the bundled Quaternius rig; "
            "check the Quaternius bone-name mapping"
        )
    return channels


def _triangles(mode: int, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if mode == 4:  # TRIANGLES
        usable = indices.size - (indices.size % 3)
        return indices[:usable].reshape(-1, 3)
    if mode == 5:  # TRIANGLE_STRIP
        return np.asarray(
            [(indices[i - 2], indices[i - 1], indices[i]) if i % 2 == 0
             else (indices[i - 1], indices[i - 2], indices[i])
             for i in range(2, indices.size)], dtype=np.int64,
        ).reshape(-1, 3)
    if mode == 6:  # TRIANGLE_FAN
        return np.asarray(
            [(indices[0], indices[i - 1], indices[i]) for i in range(2, indices.size)],
            dtype=np.int64,
        ).reshape(-1, 3)
    return np.empty((0, 3), dtype=np.int64)


def _material_color(gltf, material_index: int | None) -> tuple[float, float, float]:
    fallback = (0.72, 0.60, 0.50)
    if material_index is None or material_index < 0 or material_index >= len(gltf.materials):
        return fallback
    material = gltf.materials[material_index]
    pbr = material.pbrMetallicRoughness
    color = getattr(pbr, "baseColorFactor", None) if pbr is not None else None
    if color is None or len(color) < 3:
        return fallback
    return tuple(max(0.0, min(1.0, float(v))) for v in color[:3])


def _mesh_parts(gltf, blob: bytes):
    skins = gltf.skins or []
    inverse_binds: list[np.ndarray] = []
    for skin in skins:
        if skin.inverseBindMatrices is None:
            matrices = np.repeat(np.eye(4, dtype=np.float64)[None], len(skin.joints), axis=0)
        else:
            flat = _read_accessor(gltf, int(skin.inverseBindMatrices), blob)
            matrices = np.stack([row.reshape(4, 4).T for row in flat], axis=0)
        if len(matrices) != len(skin.joints):
            raise ValueError("glTF inverse-bind matrix count does not match skin joint count")
        inverse_binds.append(matrices)

    parts = []
    face_parts = []
    vertex_offset = 0
    colors = []
    for node_index, node in enumerate(gltf.nodes):
        if node.mesh is None:
            continue
        if node.skin is not None and int(node.skin) >= len(skins):
            raise ValueError(f"glTF mesh node {node_index} references a missing skin")
        mesh = gltf.meshes[int(node.mesh)]
        for primitive in mesh.primitives or []:
            attributes = primitive.attributes
            position_accessor = getattr(attributes, "POSITION", None)
            if position_accessor is None:
                continue
            positions = _read_accessor(gltf, int(position_accessor), blob).astype(np.float64)
            if primitive.indices is None:
                raw_indices = np.arange(len(positions), dtype=np.int64)
            else:
                raw_indices = _read_accessor(gltf, int(primitive.indices), blob).reshape(-1).astype(np.int64)
            triangles = _triangles(int(primitive.mode if primitive.mode is not None else 4), raw_indices)
            if triangles.size == 0:
                continue
            if triangles.min() < 0 or triangles.max() >= len(positions):
                raise ValueError("glTF triangle index is outside its mesh primitive")

            joints_accessor = getattr(attributes, "JOINTS_0", None)
            weights_accessor = getattr(attributes, "WEIGHTS_0", None)
            joints = weights = None
            if joints_accessor is not None and weights_accessor is not None:
                if node.skin is None:
                    raise ValueError("Skinned glTF vertex attributes have no skin on their mesh node")
                joints = _read_accessor(gltf, int(joints_accessor), blob).astype(np.int64)
                weights = _read_accessor(gltf, int(weights_accessor), blob).astype(np.float64)
                if joints.shape != weights.shape or joints.shape[0] != len(positions):
                    raise ValueError("glTF joint/weight arrays do not match the mesh vertex count")
                if joints.shape[1] > 4:
                    joints, weights = joints[:, :4], weights[:, :4]
                total = weights.sum(axis=1, keepdims=True)
                weights = np.divide(weights, total, out=np.zeros_like(weights), where=total > 1e-8)

            parts.append({
                "node": node_index,
                "node_name": str(node.name or ""),
                "skin": int(node.skin) if node.skin is not None else None,
                "positions": positions,
                "joints": joints,
                "weights": weights,
                "vertex_offset": vertex_offset,
            })
            face_parts.append(triangles + vertex_offset)
            vertex_offset += len(positions)
            colors.append(_material_color(gltf, primitive.material))

    if not parts or not face_parts:
        raise RuntimeError("The bundled GLB contains no triangle mesh to render")
    faces = np.concatenate(face_parts, axis=0).astype(np.int64, copy=False)
    base_color = tuple(float(v) for v in np.mean(np.asarray(colors, dtype=np.float64), axis=0))
    return parts, faces, inverse_binds, base_color


def _canonical_material_colors(parts, vertex_count: int) -> dict:
    """Build per-vertex warm skin and contrasting face colors for the GLB parts."""
    positions = np.empty((vertex_count, 3), dtype=np.float32)
    colors = np.tile(np.asarray(MESH_SKIN_COLOR, dtype=np.float32), (vertex_count, 1))
    for part in parts:
        start = int(part["vertex_offset"])
        count = len(part["positions"])
        stop = start + count
        positions[start:stop] = np.asarray(part["positions"], dtype=np.float32)
        name = str(part.get("node_name", "")).lower()
        if "eyebrow" in name:
            colors[start:stop] = MESH_BROW_COLOR
        elif "eye" in name:
            colors[start:stop] = MESH_EYE_COLOR
    return {
        # The semantic shader uses these positions to seed its canonical palette;
        # face_region_rgb assigns every vertex explicitly, so no rainbow remains.
        "positions": positions,
        "face_mask": np.ones(vertex_count, dtype=np.bool_),
        "face_region_rgb": colors,
    }


def _posed_vertices(gltf, parts, inverse_binds, world, vertex_count: int) -> np.ndarray:
    output = np.empty((vertex_count, 3), dtype=np.float32)
    skins = gltf.skins or []
    for part in parts:
        positions = part["positions"]
        homogeneous = np.concatenate(
            [positions, np.ones((len(positions), 1), dtype=np.float64)], axis=1
        )
        skin_index = part["skin"]
        if skin_index is None or part["joints"] is None or part["weights"] is None:
            transformed = (world[part["node"]] @ homogeneous.T).T[:, :3]
        else:
            joint_nodes = skins[skin_index].joints
            matrices = np.stack(
                [world[int(node_index)] @ inverse_binds[skin_index][joint_index]
                 for joint_index, node_index in enumerate(joint_nodes)],
                axis=0,
            )
            joint_ids = np.clip(part["joints"], 0, len(matrices) - 1)
            weights = part["weights"]
            transformed = np.zeros((len(positions), 4), dtype=np.float64)
            for influence in range(joint_ids.shape[1]):
                w = weights[:, influence]
                active = w > 0.0
                if not np.any(active):
                    continue
                selected = matrices[joint_ids[active, influence]]
                p = np.einsum("nij,nj->ni", selected, homogeneous[active])
                transformed[active] += p * w[active, None]
            no_weight = weights.sum(axis=1) <= 1e-8
            if np.any(no_weight):
                transformed[no_weight] = (
                    world[part["node"]] @ homogeneous[no_weight].T
                ).T
            transformed = transformed[:, :3]

        start = int(part["vertex_offset"])
        output[start:start + len(positions)] = transformed.astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Skinned Quaternius mesh produced non-finite vertex positions")
    return output


def _camera_vertices(vertices: np.ndarray, camera: dict) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float64)
    if camera.get("transform_camera"):
        points = points.copy()
        points[:, 1] *= -1.0
        points[:, 2] *= -1.0
    relative = points - np.asarray(camera["eye"], dtype=np.float64)[None, :]
    right = np.asarray(camera["right"], dtype=np.float64)
    up = np.asarray(camera["up"], dtype=np.float64)
    forward = np.asarray(camera["forward"], dtype=np.float64)
    x = relative @ right
    y_up = relative @ up
    z_forward = relative @ forward
    # ComfyUI's CPU/GPU rasterizer consumes image-frame coordinates (+Y down,
    # +Z forward); it performs this row-sign conversion itself for shading.
    return np.column_stack((x, -y_up, z_forward)).astype(np.float32)


def render_quaternius_mesh_video(
    motion,
    frame_count: int,
    camera: dict,
    width: int,
    height: int,
    background,
    background_image_fn,
):
    """Retarget one Kimodo sample to the bundled GLB, then render it as VIDEO frames."""
    import torch
    from comfy_extras.sam3d_body.rasterizer import render_pose_data_torch
    from kimodo_retarget_glb import export_kimodo_glb

    skeleton_name = str(getattr(motion, "skeleton_name", "") or "").lower()
    if skeleton_name and "soma" not in skeleton_name:
        raise ValueError(
            f"The bundled Quaternius mesh is retargeted for Kimodo SOMA human motion; "
            f"received skeleton '{skeleton_name}'. Select a SOMA model in Kimodo Load Model."
        )
    source_joint_names = [str(name).lower() for name in (getattr(motion, "joint_names", None) or [])]
    joint_names = set(source_joint_names)
    required_joints = {
        "hips", "spine1", "spine2", "chest", "neck1", "head",
        "leftarm", "leftforearm", "lefthand", "rightarm", "rightforearm",
        "righthand", "leftleg", "leftshin", "leftfoot", "rightleg",
        "rightshin", "rightfoot",
    }
    missing_joints = sorted(required_joints - joint_names)
    if missing_joints:
        raise ValueError(
            "The bundled Quaternius mesh needs a Kimodo SOMA body skeleton; "
            "missing joints: " + ", ".join(missing_joints)
        )

    neutral_joints = getattr(motion, "neutral_joints", None)
    if neutral_joints is None:
        raise ValueError("Kimodo motion has no neutral_joints; cannot align the rig origins")
    if hasattr(neutral_joints, "detach"):
        neutral_joints = neutral_joints.detach().to(device="cpu").numpy()
    neutral_joints = np.asarray(neutral_joints, dtype=np.float64)
    hips_index = source_joint_names.index("hips")
    if neutral_joints.ndim != 2 or neutral_joints.shape[0] != len(source_joint_names) or neutral_joints.shape[1] < 3:
        raise ValueError("Kimodo neutral_joints do not match the motion joint names")
    source_rest_hips = neutral_joints[hips_index, :3]

    asset_path = os.path.join(
        os.path.dirname(__file__), "assets", "quaternius", "superhero_male.glb"
    )
    if not os.path.isfile(asset_path):
        raise FileNotFoundError(
            "Bundled Quaternius rigged character is missing: " + asset_path
        )

    with tempfile.TemporaryDirectory(prefix="kimodo-rigged-mesh-") as temp_dir:
        animated_path = os.path.join(temp_dir, "kimodo_quaternius.glb")
        export_kimodo_glb(
            motion_data=motion,
            target_glb_path=asset_path,
            output_path=animated_path,
            sample_index=0,
            force_scale=0.0,
            map_fingers=False,
            auto_fix_input_pose=False,  # This bundled rig is T-posed.
            fix_orientation=False,     # Its glTF is already Y-up.
            animate_in_place=False,
            target_mapping=QUATERNIUS_BODY_MAPPING,
            root_origin_aligned=True,
        )

        gltf = GLTF2().load_binary(animated_path)
        blob = gltf.binary_blob()
        if not blob:
            raise RuntimeError("Retargeted Quaternius GLB has no binary buffer")
        channels = _read_animation(gltf, blob, int(frame_count))
        parts, faces, inverse_binds, _asset_color = _mesh_parts(gltf, blob)
        vertex_count = sum(len(part["positions"]) for part in parts)
        base_color = MESH_SKIN_COLOR
        canonical_colors = _canonical_material_colors(parts, vertex_count)

        pelvis_index = next(
            (i for i, node in enumerate(gltf.nodes)
             if (node.name or "").split(":")[-1].lower() == "pelvis"),
            None,
        )
        if pelvis_index is None:
            raise RuntimeError("Bundled Quaternius GLB has no mapped pelvis node")
        # Kimodo SOMA is Hips-rooted (neutral Hips is at its origin), while the
        # Quaternius rig is floor-rooted (its bind-pose pelvis is ~0.95 m above
        # the root). Retargeting preserves the target bind translation, so shift
        # the posed mesh by this constant rest-origin delta. This aligns the
        # character to Kimodo's world origin without altering camera target or
        # per-frame root motion.
        rest_world = _world_matrices(gltf, {})
        target_rest_hips = rest_world[pelvis_index][:3, 3]
        origin_offset = source_rest_hips - target_rest_hips
        print(
            "[Kimodo mesh] Rig-origin offset (m): "
            f"({origin_offset[0]:.4f}, {origin_offset[1]:.4f}, {origin_offset[2]:.4f})",
            flush=True,
        )

        pose_data = {"faces": faces, "frames": [], "canonical_colors": canonical_colors}
        cache = {}
        result = torch.empty((frame_count, height, width, 3), dtype=torch.float32, device="cpu")
        zero_translation = np.zeros(3, dtype=np.float32)
        for frame_index in range(frame_count):
            animated_values = {
                node_index: {
                    path: values[frame_index]
                    for path, values in paths.items()
                }
                for node_index, paths in channels.items()
            }
            world = _world_matrices(gltf, animated_values)
            vertices = _posed_vertices(gltf, parts, inverse_binds, world, vertex_count)
            vertices += origin_offset.astype(np.float32, copy=False)[None, :]
            camera_space = _camera_vertices(vertices, camera)
            pose_data["frames"] = [[{
                "pred_vertices": camera_space,
                "pred_cam_t": zero_translation,
                "focal_length": float(camera["focal"]),
            }]]

            bg_image = background_image_fn(background, frame_index, width, height)
            bg_array = np.asarray(bg_image, dtype=np.float32) / 255.0
            rendered = render_pose_data_torch(
                pose_data,
                frame_idx=0,
                W=width,
                H=height,
                background=bg_array,
                composite="over",
                shader_preset="rainbow_face_semantic",
                base_color=base_color,
                cache=cache,
            )
            result[frame_index] = rendered.detach().to(device="cpu", dtype=torch.float32)
        return result
