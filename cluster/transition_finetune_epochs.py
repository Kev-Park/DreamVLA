"""Apply an authorized epoch policy at the next safe DAgger fine-tune boundary.

Reserve only the named, not-yet-issued requests. Keep collection/conversion/eval
running. Once each dataset is built and its previous eval is finished, update
the unstarted request, then resume its idle driver with the original environment.
No training or Isaac process is terminated. Deployment is local git -> push ->
isolated cluster checkout -> this helper installs the versioned chain atomically.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

from watch_dagger import ENV_KEYS, LINES, atomic, processes, read, run


def step_budget(frames, passes, minimum=10000):
    if frames <= 0 or passes <= 0:
        raise ValueError("frames and passes must be positive")
    return max(minimum, math.ceil(passes * frames / 16 / 1000) * 1000)


def driver(root, tag):
    pane = run(["tmux", "display-message", "-pt", LINES[tag], "#{pane_pid}"])
    if pane.returncode:
        raise RuntimeError(pane.stderr)
    pane_pid = int(pane.stdout.strip())
    for p in processes():
        if p["args"].strip() != f"bash {root}/dline/_dline.sh":
            continue
        directory = Path(f"/proc/{p['pid']}")
        stat = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        if int(stat[1]) != pane_pid:
            continue
        env = dict(s.decode().split("=", 1) for s in (directory / "environ").read_bytes().split(b"\0") if b"=" in s)
        if env.get("TAG") == tag:
            return p["pid"], {k: env[k] for k in ENV_KEYS if k in env}
    raise RuntimeError(f"cannot uniquely identify {tag}'s main driver")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passes", type=float, required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="tag:run pairs")
    args = parser.parse_args()
    if not math.isfinite(args.passes) or args.passes <= 0:
        parser.error("passes must be finite and positive")
    root = Path.home() / "kevin"
    pending = dict(pair.split(":", 1) for pair in args.runs)
    token = f"epoch-policy-{os.getpid()}"
    reservations = []
    # Validate everything before reserving anything.
    for tag, name in pending.items():
        if tag not in LINES or not name.startswith(tag + "_dagger") or "/" in name:
            parser.error("unsupported line or run")
        q = root / "dline" / tag / "ftq"
        if (q / f"{name}.req").exists() or (q / f"{name}.claimed").exists():
            raise RuntimeError(f"{name} already issued; refusing to alter it")
        driver(root, tag)
    try:
        for tag, name in pending.items():
            q = root / "dline" / tag / "ftq"
            q.mkdir(exist_ok=True)
            claim = q / f"{name}.claimed"
            with claim.open("x") as f:
                f.write(token + "\n")
            reservations.append(claim)
            atomic(root / "dline" / tag / "finetune_policy.json", {"passes": args.passes, "max_steps": 0})
        # Only this explicitly owned runtime script is replaced, from committed code.
        source = Path(__file__).with_name("dagger_line.sh")
        target = root / "dline/_dline.sh"
        temporary = target.with_suffix(".epoch-policy.tmp")
        temporary.write_bytes(source.read_bytes())
        temporary.chmod(0o755)
        temporary.replace(target)
        while pending:
            for tag, name in list(pending.items()):
                w = root / "dline" / tag
                q = w / "ftq"
                req, claim = q / f"{name}.req", q / f"{name}.claimed"
                if (w / "abort").exists():
                    raise RuntimeError(f"{tag} aborted; transition needs diagnosis")
                if not req.exists():
                    continue
                if read(claim) != token:
                    raise RuntimeError(f"{name} reservation changed; refusing to interfere")
                # At this boundary there must be no converter, collector, eval or trainer
                # from this line. Drivers are asleep waiting on our reserved request.
                workers = [p for p in processes() if tag in p["args"] and
                           ("python" in p["args"] or "ffmpeg" in p["args"]) and
                           "transition_finetune_epochs.py" not in p["args"]]
                if workers:
                    continue
                pid, env = driver(root, tag)
                if not all(env.get(k) for k in ("TAG", "CK", "REFS", "EVREFS", "BASE", "NR")):
                    raise RuntimeError("incomplete original launch environment")
                dataset, _ = read(req).rsplit(" ", 1)
                info = json.loads((Path(dataset) / "meta/info.json").read_text())
                if not (Path(dataset).parents[1] / "built.ok").exists():
                    raise RuntimeError(f"{dataset} missing built.ok; unsafe to resume")
                steps = step_budget(info["total_frames"], args.passes, int(env.get("STEPS", 10000)))
                # Only the waiting main shell is stopped. Its children finished above.
                os.kill(pid, signal.SIGTERM)
                deadline = time.monotonic() + 20
                while Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
                    time.sleep(0.25)
                if Path(f"/proc/{pid}").exists():
                    raise RuntimeError("driver did not exit; refusing duplicate launch")
                (w / f"steps_{name}").write_text(str(steps) + "\n")
                req.write_text(f"{dataset} {steps}\n")
                env.update(FT_PASSES=str(args.passes), FT_MAX_STEPS="0", MONTAGE="0")
                command = "export XLA_PYTHON_CLIENT_PREALLOCATE=false; " + " ".join(
                    f"{k}={shlex.quote(v)}" for k, v in env.items()) + " bash ~/kevin/dline/_dline.sh"
                result = run(["tmux", "send-keys", "-t", LINES[tag], command, "Enter"])
                if result.returncode:
                    raise RuntimeError(result.stderr)
                claim.unlink()
                with (w / "status").open("a") as f:
                    f.write(f"{time.strftime('%H:%M', time.gmtime())} POLICY: {name} and future fine-tunes target {args.passes} passes, no cap; {info['total_frames']} frames -> {steps} steps\n")
                print(f"APPLIED {name}: {steps} steps; {steps * 16 / info['total_frames']:.4f} effective passes", flush=True)
                del pending[tag]
            time.sleep(20 if pending else 0)
    finally:
        for claim in reservations:
            if read(claim) == token:
                claim.unlink()


if __name__ == "__main__":
    main()
