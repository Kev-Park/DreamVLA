"""Stage the working Linux GR00T environment without modifying the source host."""
from pathlib import Path
import json
import subprocess
import time
import threading
import shutil

ROOT = Path(__file__).resolve().parents[2] / "out" / "runpod"
HOST = "sastrygrp-dvij@bluesclues.ist.berkeley.edu"
FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    jobs = {
        "environment.tar": "tar -C ~/kevin -cf - Isaac-GR00T/.venv",
        "models.tar": "tar -C ~/.cache/huggingface/hub -cf - models--nvidia--GR00T-N1.7-3B models--nvidia--Cosmos-Reason2-2B",
        "datasets.tar": "tar -C ~/kevin/datasets -cf - af60v8f_base/lerobot/ds af60v8f_ds_dagger1/lerobot/ds af60v8f_ds_dagger2/lerobot/ds T20maxf_base/lerobot/ds T20maxf_ds_dagger1/lerobot/ds T20maxf_ds_dagger2/lerobot/ds",
        "code.tar": "git -C ~/kevin/Isaac-GR00T archive HEAD",
    }
    active = []
    for name, command in jobs.items():
        target = ROOT / name
        if target.exists():
            continue
        output = target.with_suffix(".partial").open("wb")
        error = (ROOT / (name + ".stderr")).open("ab")
        process = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", HOST, command], stdout=subprocess.PIPE, stderr=error, stdin=subprocess.DEVNULL, creationflags=FLAGS)
        copier = threading.Thread(target=shutil.copyfileobj, args=(process.stdout, output, 1024 * 1024))
        copier.start()
        active.append((process, target, output, error, copier))
    while active:
        for item in active[:]:
            process, target, output, error, copier = item
            if process.poll() is None:
                continue
            copier.join()
            output.close()
            error.close()
            if process.returncode:
                raise RuntimeError(f"Staging {target.name} failed; inspect its stderr; do not retry SSH blindly")
            target.with_suffix(".partial").replace(target)
            active.remove(item)
            print(json.dumps({"ready": target.name, "bytes": target.stat().st_size}), flush=True)
        time.sleep(2)
    print("STAGING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
