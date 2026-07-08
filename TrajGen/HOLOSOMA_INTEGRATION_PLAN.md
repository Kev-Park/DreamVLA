# Holosoma Retargeting Integration — Plan & Progress Tracker

**Goal:** Replace the PyRoki retargeting + refinement in DreamVLA/TrajGen with **holosoma**'s
interaction-mesh retargeting, to improve reference-motion quality (accurate reach/grasp geometry,
29-DOF natural motion, fix the frozen left arm) for the G1 pick task, and to enable downstream
contact-based RL rewards.

This doc is the **authoritative cross-session reference**. Update the **Status** checklist as steps land.

---

## Locked design decisions (from design dialogue)

- **29-DOF references** — keep holosoma's waist roll/pitch (upgrade downstream from the old 27-DOF).
- **`object_interaction` mode** — feed the mustard-bottle mesh; holosoma's interaction mesh injects
  the wrist↔object geometry, replacing PyRoki refine's hand-coded cylinder/capsule contact model.
- **Object placement + height derived from the wrist** — the bottle is placed at the pick location
  from the wrist position at grab (OmniControl hint / retargeted wrist), including its **z/height
  from the wrist**.
- **Lift comes from OmniControl** — the post-grab lift IS preserved in the source (VERIFIED: neither
  `FREEZE_FOR` nor the refine height-floor removes it). `object_poses` rigidly attaches the bottle to
  the wrist from grab onward so holosoma carries it up → lift-inclusive reference.
- **Replicate lift-delaying + bounding** in the converter: re-insert the **10-frame grab-hold**
  (`FREEZE_FOR=10`) and the **post-grab height floor** (`max(z, offset_z)`).
- **Grasp synthesis (Phase 2)** — physics *close-until-firm-grip* on the dex hand (lift-capable, not a
  light touch); this drops `grab_idx` (contact-triggered).
- **PyRoki fully removed** (retarget + refine → holosoma's single optimizer). Training-side
  `pytorch_kinematics` in `motion_lib_base.py` STAYS (different library, not PyRoki).

---

## Target pipeline

```
OmniControl (geometry-blind body: SMPL kpts (F,22,3) + pick-hint)
  │  Adapter A (export): SMPL joints -> holosoma world-joints (.npz);  pick-hint -> bottle placement (xy + z from wrist)
  │  Fabricate object_poses(F,7): bottle STATIC at pick_point pre-grab -> rigidly follows wrist from grab (rides OmniControl's lift)
  ▼
holosoma object_interaction  (mustard bottle mesh, CPU: cvxpy+MuJoCo)  ->  .npz qpos (T,36): base(7)+29 joints, MuJoCo order, 30fps
  │  Adapter B (convert): qpos[0:7]->SE3;  qpos[7:36] permute MuJoCo->TrajGen(29);  detect grab_pos/grab_idx from wrist;
  │                       re-insert 10-frame grab-hold;  apply post-grab height floor
  ▼
.pkl reference (29-DOF)  ->  [Phase 2: grasp-synth: physics close-until-firm-grip]  ->  Training (+ contact rewards)
```

---

## Output `.pkl` contract (Adapter B must produce; downstream consumes)

```python
{
  "global_pose":      jaxlie.SE3,          # root pose (rot + trans)
  "joints":           torch.Tensor (F,29), # NEW: 29-DOF (was 27) — see 29-DOF order below
  "global_position":  torch.Tensor (F,3),  # root xyz (z min-shifted so lowest link ~0)
  "grab_pos":         torch.Tensor (3,),   # right-hand pos at grab frame
  "grab_idx":         int,                 # grab frame (Phase 2: removed, contact-triggered)
}
```

**Old 27-DOF joint order (TrajGen)** = legs L(6)+R(6), waist_yaw, arm L(7), arm R(7).
**New 29-DOF order** must insert `waist_roll`,`waist_pitch` after `waist_yaw` (indices 13,14) → shifts arms to 15–28. FINALIZE against `g1_29dof.urdf` joint order + the Training `JointNamesOrder`. **This permutation (holosoma MuJoCo order → this order) is the #1 correctness risk — validate explicitly.**

---

## Holosoma facts (from code map)

- **Entry:** `holosoma/src/holosoma_retargeting/.../examples/robot_retarget.py` (single),
  `parallel_robot_retarget.py` (batch).
- **Input:** world joints `(T,J,3)` Z-up (custom `.npz`: `global_joint_positions` + `height`), +
  `object_poses (F,7)` (wxyz quat + trans) per frame, + object mesh (`.obj`/urdf → surface points).
- **Output:** `.npz` with `qpos (T,36)` = `[base_pos(3), base_quat(4 wxyz), 29 joints]` MuJoCo order,
  `human_joints`, `fps=30`, `cost`.
- **Optimizer:** per-frame SQP; each iter a convex QP (cvxpy); cost = Laplacian deformation of the
  interaction mesh in object frame (human joints ∪ object surface pts); constraints (linearized) =
  object/ground non-penetration + foot-sticking + joint-limits + trust-region (`step_size=0.2`).
  Warm-started frame→frame (50 iters frame 1, 10 after). **CPU-bound (cvxpy+MuJoCo) — no GPU.**
- **G1 model:** `models/g1/g1_29dof.urdf`/`.xml` (body only, **no finger joints**), sphere-hand variant.
- **Interaction mesh uses the wrist/hand LINK (not fingers)** → preserves wrist↔object geometry =
  the grasp pose. Fingers = Phase-2 grasp-synth.

## PyRoki removal targets (Phase 4)
- `sample/*/retarget.py` (~16) — PyRoki solve.
- `sample/*/refine_motions*.py` (~28) — PyRoki FK + AL refine.
- `retarget_helpers/_utils.py` — PyRoki cost funcs.
- Then drop the `pyroki` package (only `visualize_motions*.py` still import it — viz-only).
- **KEEP:** `Training/.../motion_lib_base.py` `pytorch_kinematics` (Training FK).

---

## Implementation phases + VALIDATION CHECKPOINTS (run frequently for review)

- **Phase 0 — Setup.** Clone holosoma on the box; new conda env (cvxpy/mujoco/trimesh/smplx/tyro/…).
  ✅ **CHECKPOINT 0:** run holosoma's OWN demo (`robot_retarget.py` on `demo_data`) → a G1 `.npz qpos`
  + a headless render (MuJoCo offscreen MP4) proving holosoma works standalone on the box.
- **Phase 1 — Adapter A (in → holosoma).** OmniControl `results.npy` → holosoma `.npz` input +
  fabricated `object_poses` (bottle at wrist-derived pick_point, z from wrist).
  ✅ **CHECKPOINT 1:** holosoma retargets a REAL TrajGen `Pick` motion → `.npz`; headless render →
  MP4 to `~/kevin/eval_videos` for user review (robot reaches bottle? natural pose? left arm moves?).
- **Phase 2 — Adapter B (holosoma → .pkl).** Convert `.npz qpos` → `.pkl` schema: SE3, 29-DOF joint
  permute, grab detection from wrist, re-insert 10-frame hold + height floor.
  ✅ **CHECKPOINT 2:** `.pkl` matches schema exactly; render the `.pkl` (reuse a viz path) → MP4:
  reach + 10-frame hold + lift all present.
- **Phase 3 — Downstream 29-DOF (Training).** `motion_lib.joint_names` 27→29, FK URDF g1_27→g1_29,
  `JointNamesOrder`, remove scatter-27→29-with-zeros; check `KEYPTS_MASK`/keypoint FK alignment.
  ✅ **CHECKPOINT 3:** new `.pkl` plays in Training via `--reference-playback` → MP4 matches the
  Adapter-B render (proves the whole in→out chain is consistent).
- **Phase 4 — PyRoki removal.** Delete retarget/refine PyRoki; drop `pyroki` dep.
  ✅ **CHECKPOINT 4:** full pipeline (synth → holosoma → .pkl → Training) runs end-to-end with no PyRoki.
- **Phase 5 (LATER) — Grasp synth + contact rewards.** Physics close-until-firm-grip; contact-triggered
  grasp (drop `grab_idx`); contact-based residual rewards.

---

## Status (update each session)

- [~] **Phase 0 — IN PROGRESS.**
  - [x] holosoma cloned on box: `~/kevin/holosoma` (fresh, github amazon-far/holosoma).
  - [x] env built in `/kevin`: `~/kevin/.holosoma_deps/miniconda3/envs/hsretargeting` (py3.11.15).
        Activate: `source ~/kevin/.holosoma_deps/miniconda3/bin/activate hsretargeting`.
        Imports OK: holosoma_retargeting(editable→fresh clone), numpy 2.3.5, cvxpy 1.9.2, mujoco 3.10.0, torch 2.12.1, trimesh/smplx/viser/yourdfpy/tyro.
        GOTCHAS: (1) holosoma installer's `pip install -e` dep-resolver chokes on numpy==2.3.5 (transient/backtrack) — install deps separately + `pip install -e ... --no-deps --no-build-isolation`; (2) build isolation left no editable finder → use `--no-build-isolation`; (3) `source_common.sh` line1 WORKSPACE_DIR hardcoded to `$HOME/.holosoma_deps` — edited to `$HOME/kevin/.holosoma_deps`. Left the pre-existing (Feb-10, NOT in /kevin) hsretargeting env untouched.
  - [x] **CHECKPOINT 0 PASSED** — OMOMO object_interaction demo retargeted 196 frames →
        `~/kevin/hs_demo_out/sub3_largebox_003_original.npz`; skeleton render published:
        https://thiskevin.com/videos/2026-07-07_2215_holosoma_CKPT0_omomo_demo.mp4
    - **VERSION PINS REQUIRED** (else `CVXPY solve failed: infeasible` on frame 0): match the
      known-good Feb-10 env — `cvxpy==1.8.1 mujoco==3.4.0 osqp==1.1.0 highspy==1.13.0 scipy==1.17.0`
      (latest cvxpy 1.9.2 / mujoco 3.10 → infeasible). numpy 2.3.5 OK (intended pin).
    - **object_interaction qpos = (F, 43)** = base(7) + 29 joints(7:36) + **object pose(36:43)** (3 trans + 4 quat wxyz). human_joints (F,52,3). fps=30.
    - **HEADLESS RENDER:** MuJoCo EGL + osmesa both BROKEN on box (no working GL libs). Use CPU
      skeleton render: `~/kevin/render_skel.py <npz> <out.mp4>` — mj_forward → d.xpos link positions
      → matplotlib 3D skeleton + red box marker → imageio-ffmpeg MP4. Needs matplotlib + imageio-ffmpeg (installed).
    - **NOTE:** demo loops through object-pose AUGMENTATION passes after the base retarget — Ctrl-C after
      `*_original.npz` is saved; we'll disable augmentation for our single-motion use.
- [~] **Phase 1 — IN PROGRESS.**
  - [x] **Adapter A** written: `~/kevin/holosoma_adapters/export_to_holosoma.py` (results.npy `motion`
        (k,22,3,N) -> transpose -> rot_mat[[0,0,1],[1,0,0],[0,1,0]] Y-up→Z-up -> drop frame0 ->
        .npz {global_joint_positions (N,22,3), height}). KEY FINDING: **OmniControl 22 joints ==
        holosoma SMPLX_DEMO_JOINTS order exactly** → use `--data-format smplx` directly, no custom format.
  - [x] **CHECKPOINT 1a PASSED** (robot_only): real Pick motion 20 -> `~/kevin/hs_pick_out/pick_20.npz`;
        render https://thiskevin.com/videos/2026-07-07_2254_holosoma_CKPT1a_pick20_robotonly.mp4
        Cmd: `robot_retarget.py --data_path ~/kevin/hs_input --task-type robot_only --task-name pick_20 --data_format smplx --task-config.object-name ground --save_dir ~/kevin/hs_pick_out`
  - [x] **CHECKPOINT 1b PASSED** (object_interaction + real Isaac YCB mustard mesh): motion 20 →
        `~/kevin/hs_pick_out/pick_20_original.npz` (qpos (195,43)), cost 0.611→0.256.
        Render: `~/kevin/eval_videos/pick_20_mustard_oi.mp4` (local `out/pick_20_mustard_oi.mp4`) —
        red box = mustard bbox tracking the hand. **Non-subjective co-location check:** object↔right_rubber_hand_link
        dist = 0.43 m pre-grab (hand away) → 0.077 m at grab (f≈23) → ~0.06 m through the lift (0.06 = wrist↔hand-link offset). Awaiting user's subjective grasp/lift review.
        Cmd: `robot_retarget.py --task-type object_interaction --robot g1 --data-format smplx --task-name pick_20 --data-path ~/kevin/hs_input --save-dir ~/kevin/hs_pick_out --task-config.object-name mustard`
    - **MUSTARD MESH provenance:** the DreamVLA `Training/assets/mustard_bottle.usd` is a thin wrapper that
      only *references* the remote Isaac asset `.../Isaac/4.5/Isaac/Props/YCB/Axis_Aligned/006_mustard_bottle.usd`
      (pxr can't resolve `https://` refs → empty local traverse). FIX: `curl` that self-contained USD to
      `~/kevin/hs_input/006_mustard_bottle.usd`, extract with `~/kevin/holosoma_adapters/usd_to_obj.py`
      (8377 v / 16382 f, extents [0.096,0.191,0.058] Y-up). Built holosoma object assets by mirroring `largebox`:
      `models/mustard/mustard.obj` (Y-up→Z-up baked via R_x(+90): (x,y,z)→(x,-z,y), extents [0.096,0.058,0.191]),
      `models/mustard/mustard.urdf`, and combined `models/g1/g1_29dof_w_mustard.xml` (adds free-jointed
      `mustard_link` → nq 36→43 → `has_dynamic_object=True`). These live in `~/kevin/holosoma` (box-only; not in WBCBenchmark git).
    - **object_interaction is hardwired to InterMimic .pt (52-joint SMPL-H)**, but the `(smplh,g1)` mapping only
      uses 15 body joints (fingers ignored), and our validated CKPT1a path uses the `(smplx,g1)` 22-joint mapping.
      So instead of faking a 325-col InterMimic vector, **patched holosoma** (`examples/robot_retarget.py`, backup `.bak`):
      (1) `validate_config` allows `smplx` for object_interaction; (2) object_interaction `load_motion_data` branch
      prefers our `.npz` (`global_joint_positions` + `height` + `object_poses`) over `.pt`. Reuses the working smplx mapping + gains the interaction mesh.
    - **object_poses fabrication (Adapter A):** bottle static at grab loc until reached (`grab_t=argmin wrist z`), then
      rigidly tracks the right wrist (idx 21). **GOTCHA — preprocess asymmetry:** `preprocess_motion_data` scales human
      joints fully (`z-=z_min` then `*scale`) but object as `out_xy=scale*in_xy, out_z=in_z0+scale*(in_z-in_z0)` (z0 NOT
      scaled). Naively fabricating in raw frame left the object 0.14 m above the hand. Adapter A now **pre-inverts** the
      object z (`object_poses[:,6]=scale*A + (obj_z-obj_z0)`, `A=obj_z0-z_min`, `scale=1.32/height`) so the OUTPUT object lands on the retargeted hand.
  - **ENV CORRECTION (important):** the standard env is **`~/kevin/.holosoma_deps/miniconda3/envs/hsretargeting`**
    (in /kevin, mine; cvxpy 1.8.1, mujoco 3.4.0 — the required pins; has imageio_ffmpeg). Activate:
    `source ~/kevin/.holosoma_deps/miniconda3/bin/activate hsretargeting`. There is ALSO a pre-existing
    Feb-10 venv `~/.holosoma_deps` (cvxpy 1.7.5, mujoco 3.10.0, no imageio_ffmpeg) that imports the same
    /kevin/holosoma clone and happens to solve our (well-conditioned) pick data without infeasibility — but it is
    NOT the standard env; use the /kevin hsretargeting env. CKPT1b was reproduced bit-identically in both (cost 0.2563).
  - **Render:** `~/kevin/render_skel2.py <npz> <out.mp4>` (enhanced): pipes matplotlib frames to system
    `/usr/bin/ffmpeg` (robust, no imageio backend quirks) + draws the object as an ORIENTED BBOX (extents
    [0.096,0.058,0.191]) instead of a point. Original `render_skel.py` uses imageio-ffmpeg (only works in the /kevin env).
  - NOTE: robot_only qpos=(F,36); object_interaction=(F,43). augmentation defaults OFF (`RetargetingConfig.augmentation=False`) → only `*_original.npz` written, no Ctrl-C needed.
- [x] Phase 1 — Adapter A + real-motion retarget render (CHECKPOINT 1a robot_only + 1b object_interaction) — PASSED (awaiting user's subjective review of 1b)
- [ ] Phase 2 — Adapter B (.pkl + hold/floor) + render (CHECKPOINT 2)  ← NEXT
- [ ] Phase 3 — downstream 29-DOF + reference-playback (CHECKPOINT 3)
- [ ] Phase 4 — PyRoki removal (CHECKPOINT 4)
- [ ] Phase 5 — grasp synth + contact rewards (later)

## Open items / risks
- **Joint-order permutation** MuJoCo(29) → TrajGen(29): the top correctness risk — validate by
  round-tripping a known pose.
- **OmniControl → holosoma input:** use the custom `.npz` (3D joints) path — no SMPL body-model refit.
  Confirm the 22 SMPL joints map to holosoma's expected joint set (JOINTS_MAPPINGS).
- **`.usd` → `.obj`** for the mustard bottle (YCB mustard has standard meshes).
- **Headless validation render:** MuJoCo offscreen (EGL/osmesa) → frames/MP4 (no GPU/Isaac needed).
- **Box disk near-full** — check before cloning holosoma + demo data; FLAG user if low.
- **Cluster GPUs occupied by nima** — irrelevant for holosoma (CPU), but Training validation
  (Phase 3 playback) needs a GPU; gate on availability.

## Key paths
- TrajGen retarget/refine: `WBCBenchmark/TrajGen/sample/Pick_sim/retarget.py`,
  `sample/Pick_sim1/refine_motions_al.py`, `sample/retarget_helpers/_utils.py`
- Reference `.pkl` output dir (current): `WBCBenchmark/TrajGen/sample/Pick_sim2/*.pkl`
  (box: `~/kevin/DreamVLA/TrajGen/sample/Pick_sim2`; Training reads `../TrajGen/sample/Pick_sim2`)
- Holosoma (local): `EMBER/holosoma/src/holosoma_retargeting/holosoma_retargeting/`
- Adapters to write: `TrajGen/holosoma_adapters/export_to_holosoma.py`, `holosoma_to_pkl.py`
- Training downstream: `Training/source/.../motion_lib/motion_lib_base.py`, `.../manager_based/motion_tracking/g1/motion_tracking_env.py`
