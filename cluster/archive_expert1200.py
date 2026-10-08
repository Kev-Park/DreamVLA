"""Archive completed expert passes to Adroit; delete source only after SHA256 verification."""
import json
from pathlib import Path
import shlex
import subprocess
import time

BL = "sastrygrp-dvij@bluesclues.ist.berkeley.edu"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
AD = ["wsl.exe", "-e", "bash", "-lc"]
LOG = Path("K:/Coding Projects/Labs/EMBER/out/expert1200_archive.log")


def log(message):
    with LOG.open("a") as stream:
        stream.write(time.strftime("%Y-%m-%d %H:%M:%S ") + message + "\n")


def ad(command):
    return AD + ["ssh -o ControlPath=~/.ssh/adroit.sock -o ControlMaster=no -o BatchMode=yes -o ConnectTimeout=20 adroit " + shlex.quote(command)]


def call(args, timeout=1800):
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(p.stderr[-3000:] or p.stdout[-3000:])
    return p.stdout


def hashes(path):
    return "python3 -c " + shlex.quote("import hashlib,json; from pathlib import Path; p=Path(" + repr(path) +
        ").expanduser(); out={}; exec(\"for f in sorted(p.rglob('*')):\\n if f.is_file():\\n  h=hashlib.sha256()\\n  with f.open('rb') as s:\\n   for b in iter(lambda:s.read(8*1024*1024),b''): h.update(b)\\n  out[str(f.relative_to(p))]=h.hexdigest()\"); print(json.dumps(out,sort_keys=True))")


def archive(tag, number):
    name = f"{tag}_pass{number}"
    source = "~/kevin/datasets/" + name
    target = "~/kevin/archive/expert1200/" + name
    log("Hashing " + name)
    before = json.loads(call(SSH + [BL, hashes(source)]))
    if not before or "keep.txt" not in before:
        raise RuntimeError("Incomplete source manifest")
    call(ad("mkdir -p ~/kevin/archive/expert1200"))
    log("Copying " + name)
    send = subprocess.Popen(SSH + [BL, "tar -C ~/kevin/datasets -cf - " + name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    receive = subprocess.Popen(ad("tar -C ~/kevin/archive/expert1200 -xf -"), stdin=send.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    send.stdout.close()
    send.stdout = None
    _, receive_error = receive.communicate()
    _, send_error = send.communicate()
    if receive.returncode or send.returncode:
        raise RuntimeError("Archive transfer failed: " + repr((send_error[-1000:], receive_error[-1000:])))
    after = json.loads(call(ad(hashes(target))))
    if before != after:
        raise RuntimeError("Archive checksum mismatch; source preserved")
    log("Verified every file SHA256: " + name)
    finalize = "import json,shutil; from pathlib import Path; root=Path.home()/'kevin'; p=root/'datasets'/" + repr(name) + "; assert p.resolve().parent==(root/'datasets').resolve(); w=root/'dline'/" + repr(tag) + "; assert (w/" + repr(f"pass{number}.ready") + ").is_file(); (w/" + repr(f"pass{number}.archive_manifest.json") + ").write_text(" + repr(json.dumps(before)) + "); (w/" + repr(f"pass{number}.archived") + ").write_text(" + repr(target) + "); shutil.rmtree(p)"
    call(SSH + [BL, "python3 -c " + shlex.quote(finalize)])
    log("Released verified source copy: " + name)


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    while True:
        command = "for tag in expertpick1200 expertwalk1200; do for f in ~/kevin/dline/$tag/pass*.ready; do test -f \"$f\" || continue; test -f \"${f%.ready}.archived\" || printf '%s %s\\n' \"$tag\" \"$(basename \"$f\" .ready)\"; done; done"
        pending = call(SSH + [BL, command], timeout=45)
        for row in pending.splitlines():
            tag, number = row.split()
            archive(tag, int(number.removeprefix("pass")))
        time.sleep(60)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log("STOPPED: " + str(exc))
        raise
