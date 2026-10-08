"""Persistent, stdlib-only supervision for the two active DAgger chains.

Run on bluesclues with --cluster, or on Windows with --local. Never kills jobs.
Cluster mode resumes a dead chain only with its captured launch environment,
after two observations with no dependent process and no abort marker.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

LINES = {"af60v8f": "dline_v8_kevin", "T20maxf": "dline_t20_kevin"}
ENV_KEYS = "TAG CK REFS EVREFS EVIDS EVN BASE BOX FILTER NR NSH CODE BRANCH VENV ROLL XARGS ENVX FT_EXTERNAL FT_PASSES FT_MAX_STEPS STEPS DAGGER_N MONTAGE".split()


def read(path):
    try:
        return Path(path).read_text(errors="replace").strip()
    except OSError:
        return ""


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def run(args, **kw):
    return subprocess.run(args, capture_output=True, text=True, timeout=45, **kw)


def processes():
    result = []
    for directory in Path("/proc").glob("[0-9]*"):
        try:
            args = (directory / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if "/kevin/" not in args:
                continue
            stat = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            io = read(directory / "io")
            counters = dict(line.split(": ", 1) for line in io.splitlines())
            try:
                log = (directory / "fd/1").resolve()
                log_stat = log.stat() if "/kevin/" in str(log) else None
                output = [str(log), log_stat.st_size, log_stat.st_mtime] if log_stat else []
            except OSError:
                output = []
            result.append({"pid": int(directory.name), "args": args,
                           "ticks": int(stat[11]) + int(stat[12]),
                           "io": sum(int(counters.get(k, 0)) for k in ("rchar", "wchar")),
                           "start": stat[19], "output": output})
        except (OSError, ValueError, IndexError):
            continue
    return result


def checkpoint_complete(directory):
    """Require every indexed shard, with real tensor data, before delivery."""
    try:
        index = json.loads((directory / "model.safetensors.index.json").read_text())
        names = set(index["weight_map"].values())
        return bool(names) and all((directory / name).stat().st_size > 8 for name in names)
    except (OSError, ValueError, KeyError):
        return False


def cluster_tick(root, state, now):
    ps = processes()
    report = {"time": now, "lines": {}, "events": []}
    hb = read(root / "dline/.orch_heartbeat")
    age = now - int(hb) if hb.isdigit() else None
    report["orchestrator_heartbeat_age"] = age
    if age is None or age > 600:
        report["events"].append("Orchestrator heartbeat stale; local supervisor should inspect it")
    for tag, session in LINES.items():
        w = root / "dline" / tag
        old = state.setdefault(tag, {})
        relevant = [p for p in ps if tag in p["args"]]
        drivers = []
        for p in ps:
            if "dline/_dline.sh" not in p["args"]:
                continue
            raw = Path(f"/proc/{p['pid']}/environ")
            try:
                env = dict(s.decode(errors="replace").split("=", 1) for s in raw.read_bytes().split(b"\0") if b"=" in s)
            except OSError:
                continue
            if env.get("TAG") == tag:
                drivers.append(p)
                old["env"] = {k: env[k] for k in ENV_KEYS if k in env}
        status = read(w / "status").splitlines()
        # Exclude shell drivers: their CPU is not evidence of useful work.
        workers = [p for p in relevant if "python" in p["args"] or "ffmpeg" in p["args"]]
        activity = [(p["pid"], p["start"], p["ticks"], p["io"]) for p in workers]
        signature = [status[-1:] , activity]
        if signature != old.get("signature"):
            old["changed"] = now
            old["signature"] = signature
        idle = now - old.get("changed", now)
        outputs = [status[-1:], [(p["pid"], p.get("output", [])) for p in workers]]
        if outputs != old.get("outputs"):
            old["output_changed"] = now
            old["outputs"] = outputs
        output_idle = now - old.get("output_changed", now)
        done, abort = (w / "done").exists(), (w / "abort").exists()
        item = {"driver_pids": [p["pid"] for p in drivers], "worker_count": len(workers),
                "last_status": status[-5:], "done": done, "abort": abort,
                "inactive_seconds": idle, "output_inactive_seconds": output_idle}
        if abort:
            report["events"].append(f"{tag}: ABORT requires diagnosis; no blind restart")
        if idle > 1200 and not done:
            report["events"].append(f"{tag}: no CPU/I/O or stage progress for {idle:.0f}s")
        elif output_idle > 1800 and not done:
            report["events"].append(f"{tag}: CPU/I/O active but no log/stage output for {output_idle:.0f}s; inspect throughput")
        old["missing"] = old.get("missing", 0) + 1 if not drivers and not workers and not done else 0
        if old["missing"] >= 2 and not abort and old.get("env"):
            env = old["env"]
            if env.get("TAG") != tag or env.get("NR") != "5" or not all(
                    env.get(k) for k in ("CK", "REFS", "EVREFS", "BASE")):
                report["events"].append(f"{tag}: incomplete captured environment; refusing restart")
            else:
                command = "export XLA_PYTHON_CLIENT_PREALLOCATE=false; " + " ".join(
                    f"{k}={shlex.quote(v)}" for k, v in env.items()) + " bash ~/kevin/dline/_dline.sh"
                # An idle original pane is required. Never replace a busy pane.
                pane = run(["tmux", "display-message", "-pt", session, "#{pane_current_command}"])
                if pane.returncode == 0 and pane.stdout.strip() in ("bash", "zsh", "sh"):
                    result = run(["tmux", "send-keys", "-t", session, command, "Enter"])
                    report["events"].append(f"{tag}: safe resume requested, exit={result.returncode}")
                    old["missing"] = 0
        report["lines"][tag] = item
    return report


def local_tick(stage, output, state):
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
    # A network error stops this supervisor rather than retrying blindly.
    result = run(ssh + ["sastrygrp-dvij@bluesclues.ist.berkeley.edu",
                        "cat ~/kevin/dline/watch/status.json"])
    if result.returncode:
        reason = "Berkeley SSH failed; check VPN" if result.returncode == 255 else "Remote watcher check failed"
        raise RuntimeError(reason + ": " + result.stderr.strip())
    report = json.loads(result.stdout)
    if time.time() - report["time"] > 180:
        report["events"].append("Cluster watcher heartbeat stale; refusing to assume chains are healthy")
    atomic(output / "status.json", report)
    bash = r"C:/Program Files/Git/bin/bash.exe"
    check = run([bash, "-lc", 'p=$(cat /k/_ftstage/orch.pid 2>/dev/null); '
                 'test -n "$p" && ps -p "$p" -f'])
    # Verify PID identity from Windows, not merely kill -0 on a potentially reused PID.
    pid = read(stage / "orch.pid")
    identity = run(["powershell.exe", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'bash.exe' } | "
                    "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"])
    try:
        procs = json.loads(identity.stdout or "[]")
        if isinstance(procs, dict):
            procs = [procs]
        alive = any("_ftorch.sh" in (p.get("CommandLine") or "") for p in procs)
    except ValueError:
        alive = check.returncode == 0  # Refuse duplicate launches on an inspection failure.
    # MSYS may rewrite argv to plain bash: its live PID is also conservatively honored.
    alive = alive or check.returncode == 0
    if not alive and not (stage / "stop").exists():
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
        subprocess.Popen([bash, "-lc", 'cd /k/_ftstage && '
                          'LINES="af60v8f T20maxf" bash ./_ftorch.sh'],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, creationflags=flags)
        report["events"].append("Restarted absent local orchestrator for the two active lines")
    if not state.get("adroit_unavailable") and time.time() - state.get("adroit_checked", 0) > 600:
        state["adroit_checked"] = time.time()
        try:
            ad = run(["wsl.exe", "-e", "bash", "-lc",
                      "ssh -o ControlPath=~/.ssh/adroit.sock -o ControlMaster=no -o BatchMode=yes "
                      "-o ConnectTimeout=20 adroit hostname"])
            available = ad.returncode == 0
        except subprocess.TimeoutExpired:
            available = False
        if not available:
            report["events"].append("Adroit unavailable; user must reopen WSL connection")
            state["adroit_unavailable"] = True
    if state.get("adroit_unavailable"):
        report["events"].append("Adroit connection needs reopening; no further probes attempted")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.cluster == args.local:
        parser.error("choose --cluster or --local")
    root = Path.home() / "kevin"
    output = root / "dline/watch" if args.cluster else Path("K:/Coding Projects/Labs/EMBER/out/dagger_watch")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "watch.lock"
    handle = lock.open("a+")
    if os.name == "nt":
        import msvcrt
        handle.seek(0)
        handle.write("0")
        handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {}
    previous = None
    while not (output / "stop").exists():
        try:
            report = cluster_tick(root, state, time.time()) if args.cluster else local_tick(Path("K:/_ftstage"), output, state)
            atomic(output / "status.json", report)
            (output / "error.json").unlink(missing_ok=True)
            if args.cluster:
                atomic(output / "state.json", state)
            signature = json.dumps({"lines": report["lines"], "events": report["events"]}, sort_keys=True)
            if signature != previous:
                with (output / "events.jsonl").open("a") as f:
                    f.write(json.dumps(report) + "\n")
                previous = signature
            print(json.dumps(report), flush=True)
        except Exception as exc:
            atomic(output / "error.json", {"time": time.time(), "error": str(exc)})
            raise
        if args.once:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
