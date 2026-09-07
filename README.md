# ComfyUI-Kimodo-Enhanced

A ComfyUI plugin that wraps [Kimodo](https://github.com/nv-tlabs/kimodo) — NVIDIA's kinematic motion-diffusion model — to generate high-quality 3D human and humanoid-robot motion from text prompts, and **retarget it onto your own rigged characters**.

This is an enhanced fork of [ComfyUI-Kimodo](https://github.com/jtydhr88/ComfyUI-Kimodo). It keeps the original text-to-motion pipeline and adds a rebuilt **retarget/export** stack: a new **glb export** path alongside FBX, plus fixes for the real-world rig problems (A-pose arms, fingers, orientation, and travelling root motion) that the original T-pose-only retarget could not handle.
## What's new in this fork

Everything below is added on top of upstream ComfyUI-Kimodo:

- **GLB retarget + export** — a new **Kimodo Export GLB** node retargets motion onto a rigged `.glb` character (loaded via ComfyUI's Load3D input) and writes the animation back as glTF samplers/channels. The FBX-Mixamo path is retained.
- **A-pose retarget fix (arm "flapping")** — a rest-**direction**-aware correction drives each target bone to point where the source bone points every frame, so A-pose rigs no longer over-rotate their arms. Strictly reduces to the original behaviour for T-pose rigs. Exposed as the **`auto_fix_input_pose`** toggle (default OFF).
- **Travelling root motion** — root/hip displacement is now converted through the parent's full inverse world transform (rotation **and** scale), so Mixamo-cm rigs (Armature scale 0.01) travel correctly instead of "walking on the spot." **`animate_in_place`** toggle (default OFF) pins horizontal travel while keeping vertical bob/jump.
- **Orientation fix (GLB)** — **`fix_orientation`** toggle (default OFF) rescues rigs that animate face-down, snapping the character upright by a clean 90° tilt without changing its facing.
- **Sensible finger handling** — **`map_fingers`** defaults OFF to avoid the "claw / spastic hands" from mismatched SOMA↔rig finger rest poses; the body animates, hands stay in their natural rest.
- **Robust bone matching** — retarget matches bones across skeleton name-prefix mismatches.
- **Stale-clip cleanup** — GLB export strips any pre-existing bind/rest clip so viewers play the generated motion instead of a frozen 2-keyframe placeholder.

See [CLAUDE.md](CLAUDE.md) for the full technical rationale behind each of these.

## Features

- **Text-to-Motion Generation** — Describe a motion in natural language, get 3D joint positions and rotations
- **Multiple Skeleton Types** — SOMA human body, SMPLX, and Unitree G1 humanoid robot
- **Kinematic Constraints** — Optional JSON constraints for pose keyframes, end-effector positions, 2D paths
- **Multi-Prompt Segments** — Chain multiple motion descriptions with smooth transitions
- **Multiple Samples** — Generate a batch of motion variations from the same prompt
- **NPZ Export** — Save motion data (joint positions, rotations, foot contacts, trajectories)
- **BVH Export** — Export to BVH format for animation software (SOMA skeletons)
- **FBX Export (Mixamo)** — Retarget motion onto Mixamo-rigged FBX characters and export animated FBX
- **GLB Export (Mixamo)** — Retarget motion onto a rigged glb character and export animated glTF
- **2D / 3D Preview** — Skeleton visualization as ComfyUI outputs
- **HuggingFace Auto-Download** — Models download automatically on first use (~17GB VRAM)

## Nodes

### Modular Workflow (Recommended)

| Node | Category | Description |
|------|----------|-------------|
| **Kimodo Load Model** | Loaders | Load a Kimodo model variant (auto-downloads from HuggingFace) |
| **Kimodo Text Encode** | Conditioning | Encode text prompt → reusable conditioning (swap seeds without re-encoding) |
| **Kimodo Sampler** | Sampling | Diffusion sampling with conditioning + optional constraints → motion |
| **Kimodo Post Process** | Post-processing | Foot-skate cleanup (optional, requires motion_correction module) |

### Preview & Export

| Node | Description |
|------|-------------|
| **Kimodo Preview (2D)** | Render 2D skeleton stick-figure for a specific frame |
| **Kimodo Preview 3D** | Interactive 3D skeleton visualization |
| **Kimodo Save NPZ** | Save motion data as NPZ files |
| **Kimodo Export BVH** | Export motion to BVH format (SOMA skeletons only) |
| **Kimodo Export FBX (Mixamo)** | Retarget and export motion to a Mixamo-rigged FBX character |
| **Kimodo Export GLB (Mixamo)** | Retarget and export motion to a rigged glb character |

### Retarget toggles (Export FBX / GLB)

The export nodes retarget Kimodo's SOMA motion onto your character's skeleton. Real rigs are rarely T-posed, so a few opt-in corrections are provided:

| Toggle | Nodes | Default | What it does |
|--------|-------|---------|--------------|
| `auto_fix_input_pose` | FBX, GLB | OFF | Rest-direction-aware correction for **A-pose** rigs (fixes arm flapping). No-op / exact legacy behaviour on a T-pose rig, so it never alters a T-posed mesh. Turn ON for an A-pose rig. |
| `map_fingers` | FBX, GLB | OFF | Retarget finger bones. Leave OFF unless the rig's fingers align well with SOMA — otherwise the hands claw up. |
| `animate_in_place` | FBX, GLB | OFF | Zero the character's horizontal travel (keeps vertical bob/jump) so it animates on the spot. |
| `fix_orientation` | GLB | OFF | Snap a rig that animates face-down back upright (clean 90° tilt, facing unchanged). No-op for an already-upright rig. |

By default (all OFF) the retarget matches upstream behaviour and is aimed at T-posed rigs; turn on `auto_fix_input_pose` for A-pose characters.

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/stevelittlefish/ComfyUI-Kimodo-Enhanced.git
```

Install dependencies:

```bash
cd ComfyUI-Kimodo-Enhanced
pip install -r requirements.txt
```

The kimodo package itself will be auto-installed on first launch if needed.

Restart ComfyUI. The **Kimodo** nodes will appear under the `Kimodo` category.

### Models

Models download automatically from HuggingFace on first use:

| Model | Skeleton | Dataset | Description |
|-------|----------|---------|-------------|
| Kimodo-SOMA-RP-v1 | SOMA (30 joints) | Rigplay (700h) | Human body, recommended |
| Kimodo-SMPLX-RP-v1 | SMPLX (22 joints) | Rigplay (700h) | SMPLX human body |
| Kimodo-G1-RP-v1 | G1 (34 joints) | Rigplay (700h) | Unitree G1 robot |
| Kimodo-SOMA-SEED-v1 | SOMA | SEED (288h) | Human body, SEED dataset |
| Kimodo-G1-SEED-v1 | G1 | SEED (288h) | G1 robot, SEED dataset |

### Manual Model Download

Kimodo's text encoder uses Meta Llama 3 8B, which is a **gated model** on HuggingFace. You need to:

1. Visit https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct and request access
2. Create a token at https://huggingface.co/settings/tokens
3. Log in and download all required models:

```bash
# Log in to HuggingFace
huggingface-cli login

# Text encoder: Llama 3 base model (gated, requires access approval)
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct

# Text encoder: LLM2Vec adapters
huggingface-cli download McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp
huggingface-cli download McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised

# Kimodo model (pick the one you want to use)
huggingface-cli download nvidia/Kimodo-SOMA-RP-v1
```

### Motion Correction (Optional)

The `motion_correction` C++ module provides foot-skate cleanup post-processing. You have two options:

**Option A: Use prebuilt binary (Windows + Python 3.11 only)**

```bash
# Copy the prebuilt files into your Python environment
cp -r prebuilt/win_amd64_cp311 <your-python-env>/Lib/site-packages/motion_correction
```

Or add the `prebuilt/win_amd64_cp311` directory to your Python path.

**Option B: Build from source (any platform)**

Requires CMake 3.15+ and a C++17 compiler (MSVC / GCC / Clang).

```bash
cd kimodo/MotionCorrection
pip install -e .
```

Verify: `python -c "import motion_correction; print('OK')"`

> Without this module, set `post_processing = False` in the Sampler/Post Process node. The motion will still work but may have foot-sliding artifacts.

### FBX Export (Optional)

To use the **Kimodo Export FBX (Mixamo)** node, install the FBX SDK Python bindings:

```bash
pip install fbxsdkpy --extra-index-url https://gitlab.inria.fr/api/v4/projects/18692/packages/pypi/simple
```

You also need a Mixamo-rigged FBX character file. Download one from [Mixamo](https://www.mixamo.com/) (select "Without Skin" or "T-Pose" for best results).

The **Kimodo Export GLB** node needs no extra SDK — it reads and writes glTF directly.

## Usage

### Modular Workflow (Recommended)

```
Load Model → Text Encode → Sampler → Post Process → Export/Preview
                                ↑
                         (constraints_json)
```

1. Add **Kimodo Load Model** — select a model variant
2. Add **Kimodo Text Encode** — enter text prompt (reusable across different seeds)
3. Add **Kimodo Sampler** — set duration, seed, diffusion steps
4. Add **Kimodo Post Process** — optional foot-skate cleanup
5. Add **Kimodo Preview** / **Export BVH** / **Export FBX** / **Export GLB** — visualize or save

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prompt` | — | Text description of the motion |
| `duration` | 5.0 | Duration in seconds |
| `seed` | 42 | Random seed for reproducibility |
| `num_samples` | 1 | Number of motion variations to generate |
| `diffusion_steps` | 100 | Denoising steps (more = better quality, slower) |
| `post_processing` | true | Foot-skate cleanup (recommended, ignored for G1) |
| `constraints_json` | — | Optional path to kinematic constraints JSON |

### Multi-Prompt

Separate motion segments with periods in the prompt:

```
A person walks forward. They stop and wave hello. They turn around and sit down.
```

Each segment gets the specified duration.

### Output Format

The NPZ output contains:
- `posed_joints` — Joint positions `[T, J, 3]`
- `global_rot_mats` — Joint rotation matrices `[T, J, 3, 3]`
- `root_positions` — Root trajectory `[T, 3]`
- `foot_contacts` — Foot contact labels `[T, 4]`
- `global_root_heading` — Root heading angle `[T]`

## Credits

This project is an enhanced fork of **[ComfyUI-Kimodo](https://github.com/jtydhr88/ComfyUI-Kimodo)** by **[jtydhr88](https://github.com/jtydhr88)** — huge thanks to them for the original ComfyUI integration, which does all the heavy lifting of wrapping the model into nodes. This fork simply builds a better retarget/export stack on top of that foundation.

It in turn wraps [Kimodo](https://github.com/nv-tlabs/kimodo), NVIDIA's kinematic motion-diffusion model.

## License

Apache-2.0
</content>
</invoke>
