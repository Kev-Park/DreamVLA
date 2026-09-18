"""Teacher-forced VLA grasp check (no simulator).

Feed a fine-tuned ``unitree_g1_sonic`` GR00T checkpoint the observations of its OWN training
demonstrations (ego frame + state, straight from the LeRobot set) at the same cadence the closed-loop
eval uses (re-plan every ``--chunk-size`` frames, execute the first ``chunk-size`` steps of each 40-step
chunk), and log what it would do with the right hand at every frame.

Answers: does the model close the hand at the right moment when it is shown the states the
demonstration passed through?  Compared with the closed-loop eval's ``[grasp]`` lines (which show the
prediction on the states the VLA reaches by itself), this separates "never learned to close" from
"never reaches the closing regime in closed loop".

Runs in the Isaac-GR00T venv (needs gr00t + torchcodec), e.g. on biped:

    cd ~/kevin/Isaac-GR00T && CUDA_VISIBLE_DEVICES=1 .venv/bin/python \\
        ~/kevin/DreamVLA/Training/scripts/reinforcement_learning/rsl_rl/teacher_force_vla.py \\
        --vla-checkpoint ~/kevin/checkpoints/<run>/checkpoint-10000 \\
        --dataset ~/kevin/datasets/<set> --episodes 0 1 2 3 4 --out ~/kevin/eval_videos/teacher_force_<run>.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STATE_GROUPS = ["left_leg", "right_leg", "waist", "left_arm", "right_arm", "left_hand", "right_hand", "projected_gravity"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vla-checkpoint", required=True)
    ap.add_argument("--dataset", required=True, help="LeRobot set root (the fine-tune's --dataset-path)")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--chunk-size", type=int, default=8, help="re-plan cadence, as in eval_vla_sonic.py")
    ap.add_argument("--close-thres", type=float, default=0.6, help="mean|q| above which the eval commands CLOSE")
    ap.add_argument("--language", default="pick up the mustard bottle")
    ap.add_argument("--embodiment-tag", default="unitree_g1_sonic")
    ap.add_argument("--out", type=str, default=None, help="npz with per-frame predicted/GT finger series")
    args = ap.parse_args()

    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    loader = LeRobotEpisodeLoader(args.dataset, MODALITY_CONFIGS[args.embodiment_tag], video_backend="torchcodec")
    policy = Gr00tPolicy(embodiment_tag=args.embodiment_tag, model_path=args.vla_checkpoint, device="cuda:0")
    print(f"[teacher-force] {args.vla_checkpoint}  dataset={args.dataset}  episodes={args.episodes}  "
          f"cadence={args.chunk_size}  close_thres={args.close_thres}")

    results = {}
    for ep in args.episodes:
        df = loader[ep]
        T = len(df)
        gt_rh = np.stack(df["action.right_hand_joints"].values).astype(np.float32)          # (T, 7)
        gt_tok = np.stack(df["action.motion_token"].values).astype(np.float32)             # (T, 64)
        pred_rh = np.full((T, 7), np.nan, np.float32)
        pred_tok = np.full((T, 64), np.nan, np.float32)
        for t in range(0, T, args.chunk_size):
            row = df.iloc[t]
            state = {g: np.asarray(row[f"state.{g}"], np.float32)[None, None, :] for g in STATE_GROUPS}
            video = {"ego_view": np.asarray(row["video.ego_view"], np.uint8)[None, None]}      # (1,1,H,W,3)
            obs = {"video": video, "state": state, "language": {"annotation.human.task_description": [[args.language]]}}
            out = policy.get_action(obs)
            act = out[0] if isinstance(out, tuple) else out
            n = min(args.chunk_size, T - t)
            pred_rh[t:t + n] = np.asarray(act["right_hand_joints"], np.float32)[0, :n]
            pred_tok[t:t + n] = np.asarray(act["motion_token"], np.float32)[0, :n]
        gm, pm = np.abs(gt_rh).mean(1), np.abs(pred_rh).mean(1)
        gt_close, pr_close = gm > args.close_thres, pm > args.close_thres
        gt_first = int(np.argmax(gt_close)) if gt_close.any() else -1
        pr_first = int(np.argmax(pr_close)) if pr_close.any() else -1
        agree = float((gt_close == pr_close).mean())
        tok_mse = float(np.nanmean((pred_tok - gt_tok) ** 2))
        print(f"[ep {ep}] T={T}  GT first close {gt_first}  PRED first close {pr_first}  "
              f"pred mean|q| max {np.nanmax(pm):.2f}  pred closed frames {int(pr_close.sum())}/{int(gt_close.sum())} (GT)  "
              f"open/close agreement {100*agree:.1f}%  |  motion_token MSE {tok_mse:.4f}  "
              f"|  pred mean|q| at GT grab +-10: {np.nanmean(pm[max(0, gt_first-10):gt_first+10]):.2f}" if gt_first >= 0 else
              f"[ep {ep}] T={T}  no GT close in this episode")
        results[ep] = dict(gt_first=gt_first, pr_first=pr_first, agree=agree, pred_max=float(np.nanmax(pm)), tok_mse=tok_mse,
                           gt_meanq=gm, pred_meanq=pm)
    n_ok = sum(1 for r in results.values() if r["pr_first"] >= 0)
    print(f"[teacher-force] episodes where the VLA would close at least once: {n_ok}/{len(results)}; "
          f"mean open/close agreement {100*np.mean([r['agree'] for r in results.values()]):.1f}%; "
          f"first-close offset (pred - GT) frames: {[r['pr_first'] - r['gt_first'] for r in results.values() if r['pr_first'] >= 0 and r['gt_first'] >= 0]}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, **{f"ep{ep}_{k}": v for ep, r in results.items() for k, v in r.items()},
                 meta=json.dumps({"checkpoint": args.vla_checkpoint, "dataset": args.dataset, "chunk": args.chunk_size, "thres": args.close_thres}))
        print(f"[teacher-force] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
