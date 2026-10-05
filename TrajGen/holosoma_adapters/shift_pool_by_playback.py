"""Per-clip rigid XY shift of a reference pool, calibrated from a frozen-SONIC playback error.

For each row of the table (clip, dx, dy in metres, WORLD xy of the pkl) the whole robot trajectory
(global_pose translation + global_position, every frame) is moved by (dx, dy). Joints, heading,
object_poses, grab_idx and grab_pos are untouched, so the robot spawns and walks shifted relative to
the bottle. Use: measure (playback wrist - reference wrist) at the grab per clip, shift by its negative,
so the playback lands where the reference (or a chosen target) puts the hand.

usage: shift_pool_by_playback.py <in_pool> <table.csv> <out_pool>
"""
import os, sys, csv, pickle
import numpy as np, torch, jaxlie, jax.numpy as jnp

def pool_dir(p):
    if os.path.isdir(p):
        return p
    root = os.environ.get("REF_MOTIONS_DIR", os.path.expanduser("~/kevin/ref_motions"))
    return os.path.join(root, os.path.basename(p.rstrip("/")))

src, table, dst = pool_dir(sys.argv[1]), sys.argv[2], pool_dir(sys.argv[3])
os.makedirs(dst, exist_ok=True)
rows = {r["clip"]: (float(r["dx"]), float(r["dy"])) for r in csv.DictReader(open(table))}
n = 0
for cid, (dx, dy) in rows.items():
    d = pickle.load(open(os.path.join(src, cid + ".pkl"), "rb"))
    P = np.asarray(d["global_pose"].translation(), dtype=np.float64).copy()
    rot = d["global_pose"].rotation()
    Q = np.asarray(rot.wxyz if not callable(getattr(rot, "wxyz", None)) else rot.wxyz(), dtype=np.float64)
    P[:, 0] += dx; P[:, 1] += dy
    d["global_pose"] = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(jnp.array(Q)), jnp.array(P))
    d["global_position"] = torch.tensor(P, dtype=torch.float32)
    with open(os.path.join(dst, cid + ".pkl"), "wb") as f:
        pickle.dump(d, f)
    n += 1
    print("%-8s shift (%+.3f, %+.3f) m" % (cid, dx, dy))
print("wrote %d clips -> %s" % (n, dst))
