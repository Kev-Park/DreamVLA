# Setup and external dependencies

Use the revision inventory in `configs/dependencies.json`; a new upstream branch
head is not an interchangeable dependency lock. The native online GR00T launcher
targets the N1.7 API at that revision, not N1.5/N1.6. No installers run on import.

## Separate environments, one simulator

- Isaac: the DreamControl/DreamVLA Isaac Lab backend, Isaac Sim 4.5, RSL-RL 2.3
  and the matching CUDA-enabled ONNX Runtime. The source extraction used Python
  3.10 / Torch 2.7.0+cu128. Install the backend packages using its documented
  editable setup. Simulator and asset licenses remain separate requirements.
- IK: Torch, JAX/JAXLIB, jaxlie, pyroki, pytorch_kinematics, yourdfpy, viser and
  DreamControl's `isaac_utils`. Use the upstream reference-generation environment;
  do not force these packages into GR00T's environment.
- OmniControl: legacy Python 3.7 sampling environment, separate from Isaac. The
  optional XYZ-only sampler reuses the upstream model/diffusion path, not SMPL
  inference. Retarget its output with the upstream task-specific retargeter to
  obtain a trusted G1 base motion. Model and normalization data are not included.
- Export: SONIC's official data-collection environment with its compatible
  LeRobot v2.1 exporter, PyAV, OpenCV and PyArrow, rather than an arbitrary latest
  LeRobot installation.
- GR00T: NVIDIA's N1.7 environment (the extraction used Python 3.12). Obtain model
  weights and gated backbone access yourself according to NVIDIA/Hugging Face
  requirements. Never place tokens in committed configs or shell scripts.

Install only this extension, not a second simulator, in the appropriate existing
environments. If a separate DreamControl checkout supplies the backend, point to
it explicitly. These variables are paths, not model-download credentials:

```bash
export DREAMVLA_ROOT=/path/to/DreamVLA
export DREAMVLA_BACKEND_ROOT=/path/to/backend       # defaults to DREAMVLA_ROOT
export DREAMVLA_SONIC_REPO=/path/to/GR00T-WholeBodyControl
export DREAMVLA_GROOT_REPO=/path/to/Isaac-GR00T
export DREAMVLA_ASSET_ROOT=/path/to/private/runtime/assets
# Optional: absolute ffmpeg executable instead of PATH lookup
export DREAMVLA_FFMPEG=/path/to/ffmpeg
```

Run `doctor --stage isaac`, `--stage reference`, `--stage data` and
`--stage groot` in their respective environments. It is read-only and does not
start Kit or allocate a GPU. It detects missing packages/files and LFS pointers,
not every ABI, mesh, material or network-asset problem.

## Required assets (not in this contribution)

The configured asset root should contain:

```text
assets/
  sonic_release/model_encoder.onnx
  sonic_release/model_decoder.onnx
  sonic_g1/g1_29dof_with_hand.usd
  sonic_g1/configuration/...             # converter-authored references
  button_press/source/...               # separately obtained panel and texture
  button_press/source_manifest.json
  button_press/button_panel_visual.usdc
  button_press/dreamvla_button_visual_v1.usda
  button_press/visual_manifest.json
```

Obtain the released SONIC model pair from NVIDIA under its model license. Do not
mix decoder versions or joint/gain conventions. The native action implementation
expects encoder input 1762 and decoder input 994, including 64 token dimensions;
the decoder returns 29 normalized body targets. Startup validates tensor shapes.

The backend must have the resolved G1 URDF and meshes under
`Sim2Real/resources/robots/g1_description/`, plus its original 27-DoF reference
URDF under `Training/HumanoidVerse/humanoidverse/data/robots/g1/`. Resolve actual
LFS objects; pointer text is not a robot asset.

In the Isaac environment, `prepare-robot-asset` converts the 29-body + 14-hand
URDF using the extracted settings: floating base, fixed joints retained,
instanceable geometry, convex-hull colliders, no self-collision, zero imported
drive gains (the task supplies SONIC gains). It refuses an existing output.
Perform conversion with an allocated GPU and inspect the resulting 43 joints.

```bash
python -m dreamvla_drawer_button prepare-robot-asset --device cuda:0
```

Drawer uses NVIDIA's stock Sektion cabinet via `ISAAC_NUCLEUS_DIR`. Network/cache
access must resolve `Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd`.
The task applies the corrected placement in both PPO and recording; it does not
apply a cosmetic mesh replacement only when rendering a video.

Button uses DreamVLA's original elevator button meshes/materials, not a newly
invented procedural substitute. Obtain
`Training/assets/Collected_ElevatorLobby/buttons.usd` and `textures/UpDown.jpg`
and keep their upstream licensing. Then, in a USD/Isaac-capable environment:

```bash
python -m dreamvla_drawer_button prepare-button-asset \
  --source "$DREAMVLA_BACKEND_ROOT/Training/assets/Collected_ElevatorLobby" \
  --source-commit ACTUAL_SOURCE_COMMIT --build
```

This copies into a new private asset directory, fixes texture references, strips
physics from the visual panel, preserves mesh/UV signatures, aligns the press
face to the task target, and writes hashes. Cosmetic travel depends on the
actual fingertip position, not reference timing. Exactly this derived panel is
used in both new training RGB capture and camera-based evaluation. Existing
datasets are not rewritten or cosmetically cleaned.

## Before claiming an end-to-end reproduction

Run two PPO iterations with two environments; qualify one expert rollout; collect
one accepted full episode; convert and independently count decoded RGB frames;
run one native GR00T update; serve that checkpoint; run one rollout per evaluation
track; inspect rendered images, active action masks and joint orders. Keep the
reference/data/model hashes, dependency revisions and command lines with the
private outputs. These GPU integration checks are not replaced by CPU unit tests.
