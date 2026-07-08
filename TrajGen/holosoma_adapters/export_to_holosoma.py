"""Adapter A: OmniControl results.npy (SMPL 22-joint kpts) -> holosoma smplx .npz input.

OmniControl's 22 joints == holosoma SMPLX_DEMO_JOINTS order (same joints, same order), so the
output feeds holosoma's existing `--data-format smplx` directly (no custom format needed).
rot_mat replicates sample/*/retarget.py (Y-up -> Z-up).

Usage: python export_to_holosoma.py <motion_idx> [results.npy] [out_dir]
  -> writes <out_dir>/pick_<motion_idx>.npz with {global_joint_positions (N,22,3), height}
"""
import numpy as np, sys, os

motion_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 20
results = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/kevin/DreamVLA/TrajGen/sample/Pick_sim/results.npy")
out_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser("~/kevin/hs_input")
os.makedirs(out_dir, exist_ok=True)

DATA = np.load(results, allow_pickle=True).item()
smpl = DATA["motion"].transpose((0, 3, 1, 2))                       # (k, N, 22, 3)
rot = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
smpl = np.einsum("ij,ankj->anki", rot, smpl.copy())                 # Y-up -> Z-up
smpl = smpl[:, 1:]                                                  # drop frame 0 (match retarget.py)
gj = smpl[motion_idx].astype(np.float32)                           # (N, 22, 3)

h = float(gj[..., 2].max() - gj[..., 2].min())                     # rough standing height
if not (1.3 < h < 2.2):
    h = 1.78
np.savez(os.path.join(out_dir, f"pick_{motion_idx}.npz"),
         global_joint_positions=gj, height=np.float32(h))
print(f"wrote pick_{motion_idx}.npz  frames={gj.shape[0]} joints={gj.shape[1]} "
      f"height={h:.2f}  z[{gj[...,2].min():.2f},{gj[...,2].max():.2f}]")
