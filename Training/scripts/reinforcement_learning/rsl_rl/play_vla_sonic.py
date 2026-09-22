"""VLA (unitree_g1_sonic) + SONIC decoder closed-loop *recording* script in Isaac Lab.

Records a video of the VLA rolling out its policy (third-person + ego). Qualitative
sibling of ``eval_vla_sonic.py`` (which reports the physical-hold statistics headlessly);
this one renders one (or a few) rollouts so you can WATCH the policy. Same closed-loop
wiring as the eval, so the videos reflect exactly what is measured.

This VLA predicts the SONIC latent token DIRECTLY (``motion_token``, 64-D) plus the finger
joints. The token is fed through the SAME frozen decoder + proprio-history wrapper the residual
policy was trained and collected with (``TokenActionDecoderVecEnvWrapper``), so the only thing
that changes between data collection and this playback is WHO produces the token:

    env obs ─▶ ObsToPolicyAdapter ─▶ Gr00tPolicy.get_action ─▶ action_dict
                        motion_token (64) + right_hand_joints (7 -> binary scalar)
                                        │
                     TokenActionDecoderVecEnvWrapper.step([token | finger])
                       FSQ snap ─▶ [token | 10-frame history] ─▶ frozen decoder (.pt or ONNX)
                       ─▶ 29 body targets + binary finger slots ─▶ env.step

Run (from DreamVLA/Training, same env vars + recipe flags as the collection):

    HS_REWORK=1 HS_REWORK_OBJ_PC=1 HS_REWORK_OBJ_GATE=1 HS_REWORK_CONTACT_LEAD=2 \\
    python scripts/reinforcement_learning/rsl_rl/play_vla_sonic.py --headless \\
        --vla-checkpoint ~/kevin/checkpoints/run-01 --sonic-pt <native .pt dir> \\
        --ref-motions-path ../TrajGen/sample/Holosoma_Pick_29_fixH60 \\
        --motion-ids 1 4 --record-video ~/kevin/eval_videos/run01

Produces ``<prefix>_ep<k>_m<motion>_third_person.mp4`` and ``..._ego.mp4`` per episode, plus the
per-episode [PHYS] line the eval prints.
"""

from __future__ import annotations

import argparse

from vla_sonic.repo_paths import gear_sonic_deploy  # sibling-repo ONNX defaults (worktree-safe)
import builtins
import os
import sys
import time
from functools import partial
from pathlib import Path

print = partial(builtins.print, flush=True)

# This VLA embodiment predicts the SONIC token directly. Its tag is baked into
# the checkpoint as ``unitree_g1_sonic`` (NOT ``new_embodiment`` — that was the
# older vr_3pt formulation). Overridable via --embodiment-tag.
DEFAULT_EMBODIMENT_TAG = "unitree_g1_sonic"


# =========================================================================
# Phase 1: Isaac Lab AppLauncher MUST come first (before any gym/torch that
# might touch omniverse).
# =========================================================================

def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLA(unitree_g1_sonic)+SONIC closed-loop video recorder")
    parser.add_argument("--task", default="Isaac-Motion-Tracking-Pick-Cam-HOI-v0",
                        help="Kitchen-visual + camera variant of the HOI pick env (identical physics/obs/"
                             "actions/rewards/terminations to the residual training env, and the env "
                             "collect_sonic_adapter.py records in).")
    parser.add_argument("--sonic-pt", type=str, default=None,
                        help="Directory of the native SONIC .pt checkpoint whose DECODER turns the VLA's "
                             "motion_token into joint targets. MUST be the model the training data's "
                             "tokens were recorded with (collect_sonic_adapter.py --sonic-pt). Overrides "
                             "--decoder-onnx.")
    parser.add_argument("--ref-motions-path", type=str, default=None,
                        help="Override the env's ref_motions_path (dir of reference .pkl files) -- the set the "
                             "training data was collected on.")
    parser.add_argument("--waist-dof", type=int, default=29, choices=[27, 29],
                        help="29 = waist-actuated 29-DOF articulation (current pipeline).")
    parser.add_argument("--motion-ids", type=int, nargs="*", default=None,
                        help="Reference motion per episode (cycled). Default: episode k plays motion k mod N.")
    parser.add_argument("--finger-close-thres", type=float, default=0.6,
                        help="Binary-fingers env: the VLA's 7 predicted right-hand joint angles are mapped "
                             "to CLOSE when mean|q| exceeds this (rad); closed pose mean|q| ~ 1.27, open = 0.")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Keep at 1 to avoid camera OOM (the VLA pipeline is single-env).")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps-per-episode", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=8,
                        help="Execute first N of the VLA's predicted steps before replanning.")
    parser.add_argument("--vla-checkpoint", default=None,
                        help="Path to the unitree_g1_sonic fine-tuned GR00T checkpoint dir "
                             "(the VLA that emits motion_token + hand joints).")
    parser.add_argument("--embodiment-tag", default=DEFAULT_EMBODIMENT_TAG)
    parser.add_argument("--language", default="pick up the mustard bottle")
    # Default paths assume DreamVLA/ and GR00T-WholeBodyControl/ are sibling repos,
    # and you run this script from DreamVLA/Training/. Override if your layout differs.
    parser.add_argument("--decoder-onnx",
                        default=gear_sonic_deploy("policy/release/model_decoder.onnx"))
    parser.add_argument("--record-video", default=os.path.expanduser("~/kevin/eval_videos/vla_rollout"),
                        help="Output prefix; per episode writes <prefix>_ep<k>_m<motion>_{third_person,ego}.mp4.")
    parser.add_argument("--video-fps", type=int, default=50)
    parser.add_argument("--traj-dump", default=None,
                        help="Write <prefix>_ep<k>_m<motion>_traj.npz with the object and robot poses "
                             "ACTUALLY SIMULATED in this rollout. --replay-hdf5 re-simulates the object "
                             "from the recorded actions, so the rendered episode can diverge from the "
                             "recorded one; dump this to score exactly what the video shows. The frame "
                             "after a terminal reset is excluded (the object is re-placed on the table).")
    parser.add_argument("--phys-radius", type=float, default=0.15,
                        help="[PHYS] object must be within this (m) of the SIM right palm ...")
    parser.add_argument("--phys-steps", type=int, default=25,
                        help="... for >= this many consecutive steps, while lifted >= 2 cm / 5 cm above rest.")
    parser.add_argument("--skip-start-frames", type=int, default=None,
                        help="Start episodes N frames into the motion. MUST match the value used to COLLECT "
                             "the training data, or the VLA starts out-of-distribution.")
    parser.add_argument("--start-pregrab-margin", type=float, default=None,
                        help="Start episodes this many seconds before the grab. MUST match the training "
                             "collection setting for an in-distribution playback.")
    parser.add_argument("--raw-visuals", action="store_true", default=False,
                        help="Skip matching the ego view to the collection scene (non-Cam tasks only; the "
                             "Cam-* envs already carry the collection env's visual hooks).")
    parser.add_argument("--episode-scale", type=float, default=1.0,
                        help="Run episodes this many times longer than the reference clip (episode_length_s and "
                             "--max-steps-per-episode are scaled; the reference-exhausted time_out is replaced by "
                             "the flat cap and the reference holds its last frame past the clip end).")
    parser.add_argument("--motions-from", type=str, default=None,
                        help="collect_sonic_adapter.py HDF5 output root: sweep ONLY motion ids that have a "
                             "collected demonstration there (filtered reference init states).")
    parser.add_argument("--replay-hdf5", type=str, default=None,
                        help="DIAGNOSTIC: instead of the VLA, feed this demo's RECORDED executed token + finger "
                             "scalar through the same decoder wrapper (its motion id is used). The VLA checkpoint "
                             "is not loaded.")
    parser.add_argument("--seed", type=int, default=0)
    # AppLauncher args get appended below.
    return parser


_parser = _parse_cli()

# Lazy import AppLauncher so --help works even without isaaclab installed.
from isaaclab.app import AppLauncher  # noqa: E402
AppLauncher.add_app_launcher_args(_parser)
_ARGS = _parser.parse_args()

# Cameras always needed for the VLA's ego view + recording — override if not set.
_ARGS.enable_cameras = True

app_launcher = AppLauncher(_ARGS)
_APP = app_launcher.app


# =========================================================================
# Phase 2: Heavy imports (safe now that AppLauncher is up).
# =========================================================================

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401  # registers tasks
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.sim import PinholeCameraCfg  # noqa: E402
from isaaclab.managers import EventTermCfg  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402

# Ensure vla_sonic package is importable from its parent dir.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from vla_sonic.obs_to_policy import ObsAdapterConfig, ObsToPolicyAdapter  # noqa: E402
from vla_sonic.physics_overrides import apply_sonic_physics_overrides, apply_object_mass_override, apply_hand_gain_override  # noqa: E402
from vla_sonic.simple_robot_model import SimpleG1RobotModel  # noqa: E402
from vla_sonic.token_action_wrapper import TokenActionDecoderVecEnvWrapper, load_frozen_decoder  # noqa: E402
from vla_sonic.sonic_pt import load_sonic_pt  # noqa: E402
from vla_sonic.robot_29dof import apply_29dof_waist_override  # noqa: E402
from vla_sonic.eval_helpers import ReplaySource, apply_episode_scale, demo_motion_ids  # noqa: E402

from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: E402


# =========================================================================
# VLA SONIC-token extraction (identical to eval_vla_sonic.py).
# =========================================================================

def extract_motion_token(
    vla_action: dict, *, t_index: int = 0, batch_index: int = 0
) -> np.ndarray:
    """Return the (64,) SONIC latent token for one VLA step ((B, T, 64) in the action dict)."""
    if "motion_token" not in vla_action:
        raise KeyError(
            f"vla_action has no 'motion_token'; got {sorted(vla_action.keys())}. "
            f"Is this a unitree_g1_sonic checkpoint? (--embodiment-tag)"
        )
    tok = np.asarray(vla_action["motion_token"], dtype=np.float32)
    if tok.ndim != 3 or tok.shape[-1] != 64:
        raise ValueError(f"motion_token must be (B,T,64); got {tok.shape}")
    return tok[batch_index, t_index].copy()


# =========================================================================
# Video writer wrapping imageio-ffmpeg.
# =========================================================================

class VideoWriter:
    def __init__(self, path: Path, fps: int):
        import imageio
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7)
        self.path = path
        self.frames = 0

    def write(self, frame_rgb: np.ndarray) -> None:
        self._writer.append_data(frame_rgb)
        self.frames += 1

    def close(self) -> None:
        self._writer.close()


def _read_camera_rgb(env, key: str, *, verbose: bool = False) -> np.ndarray | None:
    try:
        cam = env.unwrapped.scene[key]
    except KeyError:
        if verbose:
            print(f"[video] scene has no '{key}'")
        return None
    try:
        output = cam.data.output
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"[video] {key}: cam.data.output failed: {e}")
        return None
    if "rgb" not in output:
        if verbose:
            print(f"[video] {key}: no 'rgb' key in output, have {list(output.keys())}")
        return None
    rgb = output["rgb"][0, ..., :3]
    if rgb.dtype != torch.uint8:
        rgb = rgb.clamp(0.0, 255.0).to(torch.uint8)
    return rgb.cpu().numpy()


# =========================================================================
# Camera injection — only for envs WITHOUT their own cameras (the Cam-* envs
# define both the 3rd-person `camera` and the torso-mounted `camera_robot`).
# Offsets from motion_tracking_pick_env.py (3rd-person) and
# collect_pick_cam.py:683-699 (robot-mounted d435).
# =========================================================================

def _inject_cameras(env_cfg) -> None:
    """Force-set 3rd-person `camera` and robot-mounted `camera_robot` on the scene cfg."""
    rot = np.array([0.7538, 0.61221, -0.1505, -0.1853])
    rot_mat = np.array(math_utils.matrix_from_quat(torch.tensor(rot)))
    theta = -np.pi * 0.75
    rot_z = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    rot_mat = rot_z @ rot_mat
    rot_quat = tuple(math_utils.quat_from_matrix(torch.tensor(rot_mat)).tolist())
    if getattr(env_cfg.scene, "camera", None) is None:
        env_cfg.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera_new",
            spawn=PinholeCameraCfg(
                focal_length=18.1476,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 10000.0),
            ),
            data_types=["rgb"],
            height=1920, width=2560,
            offset=CameraCfg.OffsetCfg(
                pos=(-1.03 + 2.1 - 0.034, 4.05 - 0.9, 1.31),
                rot=rot_quat,
                convention="opengl",
            ),
        )
    if getattr(env_cfg.scene, "camera_robot", None) is None:
        env_cfg.scene.camera_robot = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/Camera_robot",
            spawn=PinholeCameraCfg(
                focal_length=7.6,
                focus_distance=400.0,
                horizontal_aperture=20.0,
                clipping_range=(0.01, 100.0),
            ),
            data_types=["rgb"],
            height=480, width=640,
            offset=CameraCfg.OffsetCfg(
                pos=(0.05, 0.0, 0.36),
                rot=(0.568, 0.421, -0.421, -0.568),
                convention="opengl",
            ),
        )


# =========================================================================
# Match the ego view to the collection (training) scene — non-Cam tasks only.
# =========================================================================

def _hide_eval_clutter(env, env_ids=None) -> None:
    """Startup event: hide the ground plane + the kitchen USD's glass-bottle prop (visibility only)."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    hidden = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        name = prim.GetName().lower()
        is_ground = path == "/World/ground" or path.startswith("/World/ground/")
        is_kitchen_bottle = ("/Kitchen" in path) and ("/Object" not in path) and ("bottle" in name)
        if is_ground or is_kitchen_bottle:
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden += 1
    print(f"[match-visuals] hid {hidden} clutter prim(s) (ground plane + kitchen glass bottle)")


def _match_collection_visuals(env_cfg) -> None:
    """Disable the red grab marker and schedule the clutter-hide startup event."""
    pol = getattr(env_cfg.observations, "policy", None)
    if pol is not None:
        for _t in ("target_ref_curr", "target_ref_next", "target_ref_next_next"):
            term = getattr(pol, _t, None)
            if term is not None and isinstance(getattr(term, "params", None), dict):
                term.params = {**term.params, "visualize_markers": False}
    env_cfg.events.match_visuals_hide = EventTermCfg(func=_hide_eval_clutter, mode="startup")
    print("[match-visuals] red grab marker disabled; ground + glass-bottle hide scheduled "
          "(matches collect_sonic_adapter.py ego view)")


# =========================================================================
# Main rollout.
# =========================================================================

def _quat_to_R_t(q):
    """(w,x,y,z) -> 3x3 rotation, torch, matching vla_sonic.grasp_success.quat_to_R."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
    ])


def main() -> int:
    args = _ARGS

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- 1. Build env (same setup as eval_vla_sonic.py) --------------------------------
    env_cfg = parse_env_cfg(
        args.task,
        device="cuda:0",
        num_envs=args.num_envs,
        enable_cameras=True,
    )
    env_cfg.seed = args.seed
    # Match the SONIC decoder's training-time physics (500 Hz substep, fixed friction,
    # self-collisions, solver iters) — same call the residual training/eval scripts use.
    apply_sonic_physics_overrides(env_cfg)
    apply_object_mass_override(env_cfg)   # no-op unless HS_OBJ_MASS is set
    apply_hand_gain_override(env_cfg)     # no-op unless HS_HAND_STIFFNESS/DAMPING set
    if args.ref_motions_path is not None:
        env_cfg.ref_motions_path = args.ref_motions_path
        print(f"[play_vla_sonic] ref_motions_path override -> {args.ref_motions_path}")
    if args.waist_dof == 29:
        apply_29dof_waist_override(env_cfg)
    if os.environ.get("HS_EVAL_NO_EE_TERM", "0") == "1" and getattr(env_cfg.terminations, "ee_body_pos", None) is not None:
        env_cfg.terminations.ee_body_pos = None
        print("[play_vla_sonic] HS_EVAL_NO_EE_TERM=1: ee_body_pos termination removed")
    apply_episode_scale(env_cfg, args.episode_scale, "play_vla_sonic")
    if args.episode_scale != 1.0:
        args.max_steps_per_episode = int(round(args.max_steps_per_episode * args.episode_scale))
    # The Cam-* envs carry their own 3rd-person `camera` + 1280x960 ego `camera_robot`
    # (identical to the collection render); inject fallbacks only for envs without them.
    _inject_cameras(env_cfg)
    # Episode-start offset — MUST match the training-data collection.
    if args.start_pregrab_margin is not None:
        env_cfg.motion_start_pregrab_margin_s = args.start_pregrab_margin
        print(f"[play_vla_sonic] start_pregrab_margin = {args.start_pregrab_margin}s")
    if args.skip_start_frames is not None:
        env_cfg.motion_skip_start_frames = args.skip_start_frames
        print(f"[play_vla_sonic] skip_start_frames = {args.skip_start_frames}")
    if not args.raw_visuals and "-Cam-" not in args.task:
        _match_collection_visuals(env_cfg)
    # render_mode="rgb_array" activates the RTX render product so the camera annotators
    # actually receive frames.
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    print(f"[env] {args.task}  action_space={env.action_space}")
    # Pump the Omniverse event loop so RTX shaders/materials/texture streaming finish
    # loading before the first camera read.
    for _ in range(60):
        _APP.update()

    # --- 2. Build VLA policy -----------------------------------------------------------
    replay = ReplaySource(args.replay_hdf5) if args.replay_hdf5 else None
    if replay is not None:
        policy = None
        print(f"[replay] {replay.path.name}: {replay.num_steps} recorded steps, motion {replay.motion_id} -- "
              f"feeding the RECORDED token + finger scalar (VLA not loaded)")
    else:
        if not args.vla_checkpoint:
            raise SystemExit("--vla-checkpoint is required unless --replay-hdf5 is given")
        print(f"[vla] loading {args.vla_checkpoint}  (embodiment={args.embodiment_tag})")
        policy = Gr00tPolicy(
            embodiment_tag=args.embodiment_tag,
            model_path=args.vla_checkpoint,
            device="cuda:0",
        )

    # --- 3. SONIC decoder: the same frozen decoder + proprio-history wrapper the residual
    # policy was trained/collected with. The VLA's token replaces "encoder token + residual".
    if args.sonic_pt:
        _enc_unused, decoder = load_sonic_pt(args.sonic_pt, "cuda:0")
        print(f"[sonic] native .pt decoder from {args.sonic_pt}")
    else:
        decoder = load_frozen_decoder(args.decoder_onnx, "cuda:0")
        print(f"[sonic] ONNX decoder {args.decoder_onnx} (v1.0 latent space -- only valid if the "
              f"training tokens were recorded with the ONNX model)")
    env = TokenActionDecoderVecEnvWrapper(env, decoder, "cuda:0", clip_actions=None)
    unw = env.unwrapped
    total_motions = int(unw.total_motions)

    # --- 4. Obs adapter (reads env.unwrapped.scene: state + ego camera) -----------------
    robot_model = SimpleG1RobotModel.build()
    obs_adapter = ObsToPolicyAdapter(
        env,
        ObsAdapterConfig(
            language_instruction=args.language,
            robot_model=robot_model,
            camera_scene_key="camera_robot",
        ),
    )
    robot = unw.scene["robot"]
    _rw_bid = robot.find_bodies("right_wrist_yaw_link")[0][0]
    _tm = unw.termination_manager
    term_names = list(_tm.active_terms)
    scene_keys = list(unw.scene.keys()) if hasattr(unw.scene, "keys") else []
    cam_keys = {"camera": "third_person", "camera_robot": "ego"}
    for k in cam_keys:
        if k not in scene_keys:
            print(f"[video] scene has no '{k}' -- skipping its mp4")

    def _finger_scalar(vla_chunk: dict, t_idx: int) -> float:
        """VLA right_hand_joints (7) -> binary env scalar: <0 closes, >=0 opens."""
        rh = np.asarray(vla_chunk["right_hand_joints"], dtype=np.float32)
        q = rh[0, t_idx] if rh.ndim == 3 else rh.reshape(-1)
        return -1.0 if float(np.abs(q).mean()) > args.finger_close_thres else 1.0

    # --- 5. Rollout ----------------------------------------------------------------------
    PHYS_DZS = (0.02, 0.05)
    prefix = Path(args.record_video) if args.record_video else None
    written: list[Path] = []
    t_start = time.time()
    sweep_ids = demo_motion_ids(args.motions_from) if args.motions_from else list(range(total_motions))
    if args.motions_from:
        print(f"[play_vla_sonic] --motions-from: default sweep restricted to the {len(sweep_ids)} motions with a demonstration")
    for ep in range(args.num_episodes):
        if replay is not None:
            mid = replay.motion_id
        elif args.motion_ids:
            mid = int(args.motion_ids[ep % len(args.motion_ids)]) % total_motions
        else:
            mid = sweep_ids[ep % len(sweep_ids)]
        unw._forced_motion_id = mid
        print(f"\n[episode {ep}] motion_id={mid}")
        with torch.inference_mode():
            obs, _ = env.reset()                      # wrapper.reset() also seeds the decoder history
        _APP.update(); _APP.update()

        writers: dict[str, VideoWriter] = {}
        if prefix is not None:
            for k, tag in cam_keys.items():
                if k in scene_keys:
                    writers[k] = VideoWriter(prefix.with_name(f"{prefix.name}_ep{ep}_m{mid}_{tag}.mp4"), args.video_fps)
                    print(f"[video] writing {writers[k].path}")

        vla_chunk = None; chunk_step = 0
        obj_rest_z = None; phys_run = {dz: 0 for dz in PHYS_DZS}; phys_ok = {dz: False for dz in PHYS_DZS}
        max_lift = 0.0; toppled = False; touched = False
        grab_step = -1; fired = []
        step = 0
        traj = {k: [] for k in ("obj_pos", "obj_quat", "root_pos", "root_quat", "wrist")}
        for step in range(args.max_steps_per_episode):
            if replay is not None:
                token, _fs = replay.latent(step)
                vla_chunk = {"right_hand_joints": np.zeros((1, 1, 7), np.float32)}  # keeps the grasp log shape
                t_idx = 0
            else:
                if vla_chunk is None or chunk_step >= args.chunk_size:
                    vla_obs = obs_adapter()
                    vla_out = policy.get_action(vla_obs)
                    vla_chunk = vla_out[0] if isinstance(vla_out, tuple) else vla_out
                    chunk_step = 0
                    if ep == 0 and step == 0:
                        print("\n[VLA @ ep0 step0] action-dict dump (t=0 slice, batch=0):")
                        for k in sorted(vla_chunk.keys()):
                            arr = np.asarray(vla_chunk[k])
                            slice_ = arr[0, 0] if arr.ndim == 3 else arr.reshape(-1)
                            print(f"  {k} [shape {tuple(arr.shape)}] = {slice_.reshape(-1)[:8].round(4).tolist()}"
                                  f"{' ...' if slice_.size > 8 else ''}")
                t_idx = chunk_step
                token = extract_motion_token(vla_chunk, t_index=t_idx)                # (64,) continuous
                if ep == 0 and step == 0 and float(np.abs(token).max()) < 1e-6:
                    print("\n[WARN] motion_token is ALL ZERO -- the VLA is emitting a null SONIC token; "
                          "the robot is NOT VLA-controlled. Check action.motion_token in the dataset.\n")
            latent = torch.zeros((1, 65), device="cuda:0", dtype=torch.float32)
            latent[0, :64] = torch.as_tensor(token, device="cuda:0")
            latent[0, 64] = _fs if replay is not None else _finger_scalar(vla_chunk, t_idx)
            if step < 3:
                print(f"[step {step}] motion_token[:8] = {token[:8].round(3).tolist()}  finger={latent[0, 64].item():+.0f}")
            with torch.inference_mode():
                obs, _rew, dones, _extras = env.step(latent)                   # FSQ snap + decoder inside
            # Flush the RTX render pipeline so the camera annotator delivers THIS step's frame —
            # both for the video and for the ego view the next chunk's obs_adapter() reads.
            _APP.update(); _APP.update()

            for k, w in writers.items():
                frame = _read_camera_rgb(env, k, verbose=(step == 0))
                if frame is not None:
                    w.write(frame)

            # ---- metrics (env-local frame), read AFTER the step; identical to the eval ----
            org = unw.scene.env_origins[0]
            obj_p = unw.scene["object"].data.root_pos_w[0] - org
            oq = unw.scene["object"].data.root_quat_w[0]
            hq = robot.data.body_quat_w[0, _rw_bid]
            w_, x_, y_, z_ = hq[0], hq[1], hq[2], hq[3]
            hand_x = torch.stack([1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ + w_ * z_), 2 * (x_ * z_ - w_ * y_)])
            palm = (robot.data.body_pos_w[0, _rw_bid] - org) + 0.12 * hand_x
            d_palm = float(torch.norm(palm - obj_p).item())
            done_now = bool(torch.as_tensor(dones).reshape(-1)[0].item())
            if not done_now:                       # after a done the env has already reset -> skip
                if args.traj_dump:
                    rp = robot.data.root_pos_w[0] - org
                    rq = robot.data.root_quat_w[0]
                    wp = robot.data.body_pos_w[0, _rw_bid] - org
                    wq = robot.data.body_quat_w[0, _rw_bid]
                    Rr = _quat_to_R_t(rq); Rw = _quat_to_R_t(wq)
                    T = torch.eye(4, device=Rr.device, dtype=Rr.dtype)
                    T[:3, :3] = Rr.T @ Rw                       # wrist expressed in the root frame,
                    T[:3, 3] = Rr.T @ (wp - rp)                 # i.e. the same 4x4 as teleop/right_wrist
                    traj["obj_pos"].append(obj_p.cpu().numpy().copy())
                    traj["obj_quat"].append(oq.cpu().numpy().copy())
                    traj["root_pos"].append(rp.cpu().numpy().copy())
                    traj["root_quat"].append(rq.cpu().numpy().copy())
                    traj["wrist"].append(T.cpu().numpy().copy())
                if obj_rest_z is None or step <= 50:          # running min over the first 1 s (settle)
                    obj_rest_z = float(obj_p[2].item()) if obj_rest_z is None else min(obj_rest_z, float(obj_p[2].item()))
                dz = float(obj_p[2].item()) - obj_rest_z
                max_lift = max(max_lift, dz)
                for dzt in PHYS_DZS:
                    phys_run[dzt] = phys_run[dzt] + 1 if (d_palm < args.phys_radius and dz >= dzt) else 0
                    if phys_run[dzt] >= args.phys_steps:
                        phys_ok[dzt] = True
                if d_palm < 0.12:
                    touched = True
                if float((1.0 - 2.0 * (oq[1] ** 2 + oq[2] ** 2)).item()) < 0.7071:
                    toppled = True
                mt = unw.episode_length_buf * unw.step_dt + unw.start_motion_times.to("cuda:0", dtype=torch.float32)
                is_closed = bool(unw.motion_lib.get_motion_state(unw.motion_ids, mt)["is_closed"][0].item() > 0.5)
                if is_closed and grab_step < 0:
                    grab_step = step
            chunk_step += 1
            if done_now:
                try:
                    fired = [n for n in term_names if bool(_tm.get_term(n)[0].item())]
                except Exception:
                    fired = []
                break

        if args.traj_dump and traj["obj_pos"]:
            import numpy as _np
            _tp = Path(f"{args.traj_dump}_ep{ep}_m{mid}_traj.npz")
            _tp.parent.mkdir(parents=True, exist_ok=True)
            _np.savez(_tp, **{k: _np.stack(v) for k, v in traj.items()})
            print(f"[traj-dump] {_tp}  ({len(traj['obj_pos'])} frames, terminal reset excluded)")

        for w in writers.values():
            w.close()
            written.append(w.path)
            print(f"[video] closed {w.path} ({w.frames} frames @ {args.video_fps} fps)")
        print(f"[episode {ep}] motion {mid} ended at step {step+1} ({','.join(fired) if fired else 'max_steps'})  "
              f"phys_held(2cm/5cm)={phys_ok[0.02]}/{phys_ok[0.05]}  max_lift={max_lift*100:.1f}cm  "
              f"touched={touched} toppled={toppled}  grab_step={grab_step}")

    print(f"\n[play] recorded {args.num_episodes} episode(s) in {time.time() - t_start:.1f}s")
    for p in written:
        print(f"[play]   {p}")
    env.close()
    _APP.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
