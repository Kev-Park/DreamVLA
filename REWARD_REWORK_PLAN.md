# Reward + Termination Rework — Plan (living doc)

Iterative rework of the SONIC-adapter pick task's rewards, terminations, obs, and residual
formulation. **Do NOT one-shot** — resolve the OPEN decisions below, build in small steps, render
after each. Authoritative cross-session reference; update the Status/Decisions as they land.

Context files (current implementation):
- Rewards/terminations/scene: `Training/source/.../interactive_motion_tracking/g1/motion_tracking_pick_env.py`
- Tracking rewards + term functions + masks: `Training/source/.../motion_tracking/g1/motion_tracking_env.py`
- Object placement (`reset_object_state`, height=1.0): `.../interactive_motion_tracking/g1/motion_tracking_interactive_base.py`
- Motion load / grab_pos / spawn offset: `Training/source/.../utils/motion_lib/motion_lib_base.py`
- Train entry (reward flags, `--residual-scale`, `--tracking-only`, `--skip-start-frames`): `Training/scripts/reinforcement_learning/rsl_rl/train_sonic_adapter.py`
- Eval (`Episodes with any lift` = success %): `.../eval_sonic_adapter.py`
- Render/playback (terminations disabled by default; `--keep-terms`): `.../play_sonic_adapter.py`
- Residual + SONIC token/latent wiring: `vla_sonic/token_action_wrapper.py`, `vla_sonic/token_adapter_wrapper.py`, `vla_sonic/adapter_actor_critic.py`
- Dataset gen: `TrajGen/holosoma_adapters/{export_to_holosoma.py (Adapter A), holosoma_to_pkl.py (Adapter B, HS_NO_LEADIN), gen_dataset.sh}`
- Filtered dataset: `TrajGen/sample/Holosoma_Pick_29_filt60` (60 motions).

Residual sweep result (rs05 task cfg, filt60, 5000 iters): 0.50=0.65%, 0.45=0.95%, 0.35=1.75%
(lower residual -> higher success + better lift-fraction). rs040 re-run was CANCELLED to start this rework.

Proposed build order (they interlock): **#1 -> #2 -> #4 -> #7 -> #5 -> #6 -> #3.**
(#4 produces the object reference #6/#7 consume; #5 adds the right-hand ContactSensor that #6's contact-loss
termination needs. #5 is DISTINCT from #4 — ResMimic has BOTH an object point-cloud reward AND a separate
contact reward `Σ ĉ·exp(-λ/f)` using measured contact force; see #5.)

---

## 1. Frame-0 initialization (render + train)  — FRAME ACCOUNTING RESOLVED
Goal: init rollouts (render + train) at the motion's TRUE initial condition, no lead-in / skip.
- Render: works by omitting `--skip-start-frames` (confirmed true frame-0 spawn). Make it the render default.
- Train: regenerate filt60 with `HS_NO_LEADIN=1`, train with skip=0.
- **FRAME ACCOUNTING (traced source -> .pkl):**
  - `export_to_holosoma.py` (Adapter A) **drops source frame 0** (`smpl = smpl[:, 1:]`, line ~65).
  - Grounding / freeze_left_arm / arm-refine / `freeze_hold` are all LENGTH-PRESERVING (freeze_hold shifts, no drop).
  - `HS_NO_LEADIN=1` -> no 20-frame prepend, grab_idx unchanged.
  - `motion_lib_base.py` load TRUNCATES with `[:200]` (transl/quats/dof_pos) -> cuts the END; long motions lose their tail.
  - **=> no-lead-in `.pkl` frame 0 == SOURCE FRAME 1** (source frame 0 is lost in Adapter A). With lead-in, frame 0 = fabricated INIT pose; source frame 1 sits at .pkl frame 20.
- **Decisions:** (a) 1-frame offset — accept (negligible) OR remove Adapter A's `[1:]` to keep source frame 0 as the true IC.
  (b) `[:200]` end-truncation — check if any filt60 motion exceeds 200 frames (no-lead-in ~195, so likely fine; raise the cap if needed).
  (c) decoder-history warmup now UNBLOCKED: frame-0 init leaves the SONIC decoder cold — candidate: hold reference-frame-0 for ~10 pre-steps to seed history without advancing the motion.
- Files: `holosoma_to_pkl.py`, `export_to_holosoma.py`, `motion_lib_base.py` ([:200]), env reset, `play_sonic_adapter.py`.

## 2. Equal whole-body tracking  — APPROVED
Goal: track all limbs/joints/links EQUALLY; revert to base SONIC reward weightings.
- All-ones `KEYPTS_MASK` on `tracking_relative_body_pos/ori` (drop right-arm masking asymmetry).
- Delete per-limb task terms (`tracking_right_arm_pos/ori`, `tracking_hand_precise`) — folded into equal whole-body.
- Keep `anchor_pos/ori` (root) and `body_linvel` as separate global terms (not per-limb). Use base SONIC weights.
- **NOTE for later:** the root anchor DEADBAND (eps) may be worth reimplementing at a later point (removed for now).
- Files: `motion_tracking_pick_env.py` (RewardsCfg), `motion_tracking_env.py` (mask consts).

## 3. Residual clamping — RESOLVED WHERE; TEST additive vs multiplicative
**WHERE (traced `token_adapter_wrapper.py`):** residual is applied to the SONIC **TOKEN**, NOT a raw
latent and NOT the decoded 29-DOF cmd. `residual = residual_scale * tanh(latent[:, :64])`;
`body = self._base_token + residual`; then LATTICE-SNAP:
`token = clamp(round(body*16)/16, -1.0, 15/16)` (FSQ: 32 levels, TOKEN_DIM=64, step 1/16).
- **So the token IS naturally bounded** — the FSQ grid snap clamps to `[-1.0, 15/16]` regardless of
  residual magnitude. Unclamped-additive-on-token would NOT produce off-manifold garbage (it always
  re-snaps onto the codebook). This matches ResMimic (their unclamped residual is also on the ACTION/token,
  not a continuous latent). **Correction to earlier note:** ours is token-space, same family as ResMimic —
  the earlier "ours is on a latent, riskier" framing was wrong.
- **What `residual_scale * tanh()` actually buys:** a per-dimension ANCHOR keeping the token within a
  `±residual_scale` neighborhood of the base token BEFORE snapping. Sweep shows tighter anchor -> better
  (0.35 > 0.45 > 0.50), i.e. staying near the base policy helps. It's a regularizer, not a safety clamp.
- **Natural bound question:** the grid itself (`[-1, 15/16]`, width ~2) is the only truly "natural" bound.
  An unclamped residual just lets the token reach any codebook cell in one step (loses the near-base anchor).
- **TESTS to run (mark):**
  - (a) additive residual (current) at a few scales — baseline.
  - (b) **multiplicative / gated** token transform: `token = base_token * (1 + scale*tanh(g))` or a learned
    per-dim gate — keeps it "in-family" multiplicatively; compare to additive.
  - (c) optionally unclamped-additive-on-token (safe due to snap) to measure the anchor's value directly.
- Files: `token_adapter_wrapper.py` (residual op), `train_sonic_adapter.py` (transform flag).

## 4. Object-as-hand reference tracking (ResMimic-style) — TEST BOTH VISUALLY
Goal: track a reference OBJECT trajectory; **replaces `object_lift`**. No external GT object pose needed.
- Two candidate sources for the reference `obj_traj (F,7)` — **evaluate BOTH visually**:
  - (i) holosoma `object_poses` directly (static at grab pre-grab, rigidly on wrist post-grab).
  - (ii) recompute in Adapter B: `grab_pos` pre-grab, right-hand FK pose post-grab (frame-consistent, no dep on object_poses).
- **REQUIRED FIRST:** thorough VISUAL evaluation of holosoma object + robot kinematic trajectories (do they
  correspond; does the object sit on the hand through the grasp) before committing (i) vs (ii).
- Store `obj_traj` in the .pkl; add sim-object-vs-`obj_traj` pos+ori exp-kernel reward.
- Files: `holosoma_to_pkl.py`, `motion_lib_base.py` (load), new reward in the env.

## 5. Contact reward instead of wrist-pointing — DISTINCT FROM #4; NEEDS A CONTACT SENSOR
**IMPORTANT RECONCILIATION (paper vs released code):**
- The released `g1_hoi.py` implements ONLY the object point-cloud tracking reward
  (`exp(-10*object_point_cloud_dist)`, sim-object vs ref-object points, hand never referenced) plus a
  contact-FORCE TERMINATION (>1N on illegal bodies). It does NOT implement a contact reward.
- **But the ResMimic PAPER (arxiv 2510.05070v2) DOES define a separate CONTACT reward**, additive with the
  object reward: `r^c_t = Σ_i ĉ_t[i] · exp(-λ / f_t[i])`, where `ĉ_t[i]` = ORACLE contact indicator (binary,
  from the REFERENCE human-object trajectory) and `f_t[i]` = MEASURED contact force on link i. Object reward
  `r^o` and contact reward `r^c` are SEPARATE additive terms; the object reward is NOT contact-gated.
  => the paper's contact reward REQUIRES measured contact force = a **ContactSensor**. The released code just
  omits `r^c`. (User was right that contact sensing is integral to ResMimic.)
- **=> #5 is DISTINCT from #4 (un-merged).** #4 = object-trajectory tracking (kinematic, no sensor). #5 =
  ResMimic contact reward `Σ ĉ·exp(-λ/f)` rewarding actual hand-object contact where the reference says
  contact should exist. #5 NEEDS a right-hand ContactSensor.
- **Plan:** remove `target_orientation_error` (wrist pointing). Add a right-hand `ContactSensorCfg` filtered
  vs the object. Oracle `ĉ_t` from the reference = `is_closed`/post-grab hold phase (hand should be in contact
  after grab). Reward = `ĉ_t · exp(-λ / f_measured)` (or a simpler `ĉ_t · (contact>0)` bonus to start).
- Files: env scene (right-hand ContactSensor filtered vs object), RewardsCfg (remove wrist-pointing, add r^c).

## 6. New terminations — height backstop kept but SMALLER  (ResMimic terms confirmed)
**ResMimic terminations (`g1_hoi.py`):** object-far (point-cloud dist > 0.3) + pose-fail (body_pos_dist >
threshold) + height + roll/pitch + contact-force>1N on illegal bodies. NOTE: ResMimic has NO explicit
"contact-loss-for-N-frames" termination — its object-far term (object drifts from reference) already fires
when the grasp is lost. Their contact-force term is a fall/self-collision guard, not a grasp guard.
Replace fall-based (angle/height/contact) with:
- (a) reference-motion deviation (whole-body tracking error > tau)   [= ResMimic pose-fail].
- (b) object-tracking deviation (sim object vs `obj_traj` > tau)     [= ResMimic object-far; ALSO covers grasp-loss].
- (c) loss-of-contact > 10 consecutive frames — **CONFIRMED literal ResMimic** (paper: "if any required
  body-object contact is lost for more than 10 consecutive frames"). KEEP as a distinct termination; uses the
  #5 right-hand ContactSensor (fires when oracle says contact-required but measured contact absent >10 frames).
  ResMimic's object-far (pt-cloud) termination = our (b); both coexist there.
- KEEP a loose root-height backstop as a safety, but make the threshold **SMALLER** than current 0.3
  (it triggered too often in the past). Drop the tilt-angle and base-contact terminations.
- Files: `TerminationsCfg` + new term functions in `motion_tracking_env.py`.

## 7. Privileged object state in obs — lean BOTH (actor + critic)  (ResMimic confirmed)
Add to obs: current sim object pose (pos+ori) + reference object target (`obj_traj`, #4).
- **ResMimic:** the released `g1_hoi.py` supports asymmetry (object 7-DOF root state to ACTOR only if
  `nonblind=True`; critic always privileged). The PAPER, however, describes SYMMETRIC obs (both actor and
  critic get robot proprio + object state + reference). Either way object state is available to the actor.
- **DECISION (user: "give whichever obs combo offers highest chance of success"):** provide object pose +
  reference `obj_traj` to BOTH actor and critic (symmetric / nonblind). Max info to the policy; this pipeline
  is for DATASET GATHERING, not object-free deployment, so no reason to blind the actor.
- Files: `ObservationsCfg`, token-adapter wrapper obs plumbing, `adapter_actor_critic.py` (critic obs).

---

## Open items to resolve before building
- [x] #1 frame accounting source->.pkl: Adapter A drops source frame 0; motion_lib `[:200]` end-truncates.
      **DECIDED: accept the 1-frame offset** (keep Adapter A's `[1:]`). Decoder-warmup still to design.
- [x] #3 residual is on the FSQ **TOKEN** (naturally grid-bounded, same family as ResMimic).
      **DECIDED: run all 3 transform tests** — additive vs multiplicative/gated vs unclamped-on-token.
- [~] #4 visual eval of holosoma object + robot kinematic trajectories; choose obj_traj source (i) vs (ii).
      **IN PROGRESS — building the diagnostic render.** BLOCKER: the full object trajectory is NOT in the .pkl
      (only `grab_pos`, a single point). Adding `object_poses` (F,7) to Adapter B + motion_lib load + an
      `--overlay-obj-candidates` render flag; regen a few filt motions and render.
- [x] #5 RECONCILED (paper via arxiv HTML): released `g1_hoi.py` has ONLY the object pt-cloud reward, but the
      PAPER adds a SEPARATE contact reward `Σ ĉ·exp(-λ/f)` (oracle ref contact × measured contact force) +
      contact-loss>10f termination. => #5 is DISTINCT from #4 and NEEDS a right-hand ContactSensor. (Corrects
      the earlier "merge into #4"; user was right that contact sensing is integral.)
- [x] #7 obs: released code asymmetric (object to actor iff nonblind), paper symmetric.
      **DECIDED: object pose + obj_traj to BOTH actor and critic (max success).**

Revised build order: **#1 -> #2 -> #4 -> #7 -> #5 -> #6 -> #3.**
(#4 provides obj_traj that #7 obs + #6 object-deviation termination consume; #5 adds the ContactSensor #6's
contact-loss termination needs; #3 residual-transform tests last.)

## Status
- [x] rs040 re-run cancelled; this plan created.
- [x] Sweep trainings finished (rs035/rs040/rs045 all exit 0); rs045 eval = 0.95% / 61.7% lift-frac. Superseded by this rework.
- [x] All 3 ResMimic/residual/frame investigations folded in + paper reconciliation (contact reward exists).
- [~] #4 diagnostic render being built (object_poses in pkl + overlay flag). Nothing in the reward path built yet.
