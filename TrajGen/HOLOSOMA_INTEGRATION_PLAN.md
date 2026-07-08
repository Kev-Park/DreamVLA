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

- [ ] Phase 0 — holosoma cloned on box + env + demo render (CHECKPOINT 0)
- [ ] Phase 1 — Adapter A + real-motion retarget render (CHECKPOINT 1)
- [ ] Phase 2 — Adapter B (.pkl + hold/floor) + render (CHECKPOINT 2)
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
