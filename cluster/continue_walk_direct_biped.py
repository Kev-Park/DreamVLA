"""Continue the walking direct-token policy on Biped, preserving its PPO recipe.

The original Bluesclues training remains a fallback. This continuation restores
the policy, optimizer, learning rate and global exploration schedule. Simulation
episodes restart at the checkpoint boundary; the continuation is labeled clearly.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

import watch_t20max_v3 as common

ROOT, REPO = common.ROOT, common.REPO
RUN = "T20_max_v3_direct_env2048_biped_resume3500"
OUT = ROOT/"eval_videos/walk_v3_study/biped_direct"
SOURCE = REPO/"Training/logs/rsl_rl/g1_sonic_adapter/_walk_direct_import3500/model_3500.pt"
MILESTONES = list(range(4000, 9000, 500)) + [8999]


def command(script, pool):
    cmd = common.command(script, pool)
    cmd[cmd.index("--residual-transform")+1] = "direct"
    return cmd


def gpu_room(index, minimum, allow_busy=False):
    text = subprocess.check_output(["nvidia-smi", f"--id={index}",
        "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    memory, utilization = map(int, text.strip().split(","))
    return memory > minimum and (allow_busy or utilization < 20)


class AttachedTrain:
    """Observe an existing detached trainer without interrupting its GPU work.

    Its exit status belongs to the previous controller. An observed exit is
    accepted only with the final update in the log and a nonempty final weight
    file; this evidence is recorded separately from a waitpid exit status.
    """
    def __init__(self, pid, directory):
        self.pid, self.directory = pid, directory
        self.returncode = None

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        command = Path(f'/proc/{self.pid}/cmdline')
        data = command.read_bytes() if command.exists() else b''
        if data:
            assert b'train_sonic_adapter.py' in data and RUN.encode() in data, 'Trainer PID was reused'
            return None
        text = common.tail(OUT/'train.log')
        iterations = re.findall(r'Learning iteration\s+(\d+)', text)
        final = self.directory/'model_8999.pt'
        verified = bool(iterations and int(iterations[-1]) == 8999
                        and final.is_file() and final.stat().st_size > 0
                        and 'Traceback (most recent call last)' not in text)
        self.returncode = 0 if verified else 1
        common.write(OUT/'attached_training_exit.json', {'observed_at': time.time(),
            'pid': self.pid, 'final_checkpoint': str(final), 'final_update_verified': verified,
            'exit_status_available': False})
        return self.returncode


def launch(cmd, name, gpu):
    log = OUT/(name+".log")
    assert not log.exists(), str(log)
    with log.open("w") as stream:
        proc = subprocess.Popen(cmd, cwd=REPO/"Training", env=common.environment(gpu),
                                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    common.write(OUT/(name+".launch.json"), {"pid": proc.pid, "gpu": gpu,
        "started": time.time(), "command": cmd})
    return proc, log


def main(resume_controller=False):
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/"controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = json.loads((OUT/'workflow.json').read_text()) if (OUT/'workflow.json').exists() else None
    assert resume_controller == bool(previous), 'Inspect previous continuation before restarting'
    assert SOURCE.is_file()
    state = {"status": "waiting_for_gpu", "controller_pid": os.getpid(), "started": time.time(),
             "source_checkpoint": str(SOURCE), "source_completed_updates": 3501,
             "target_updates": 9000, "train_gpu": 0, "eval_gpu": 1, "errors": [],
             "note": "Checkpoint continuation; simulator episodes reset at migration; Bluesclues remains a fallback"}
    train = active = None
    directory = None
    completed, rows = set(), []
    if resume_controller:
        old_command = Path(f"/proc/{previous['controller_pid']}/cmdline")
        assert not old_command.exists() or not old_command.read_bytes(), 'Previous controller is still live'
        assert previous['status'] == 'training' and not previous.get('active_eval'), 'Only idle eval-controller handoff is supported'
        assert previous['target_updates'] == 9000 and previous['source_completed_updates'] == 3501
        assert previous['source_checkpoint'] == str(SOURCE)
        pid = previous['train_pid']
        command = Path(f'/proc/{pid}/cmdline').read_bytes()
        assert b'train_sonic_adapter.py' in command and RUN.encode() in command
        # The trainer launched with start_new_session=True, so replacing its
        # controller cannot send it tmux's terminal hangup signal.
        assert os.getsid(pid) == pid
        directory = Path(previous['checkpoint_dir'])
        assert directory.is_relative_to(REPO/'Training/logs/rsl_rl/g1_sonic_adapter')
        train, log = AttachedTrain(pid, directory), OUT/'train.log'
        rows = json.loads((OUT/'results.json').read_text())
        completed = {r['iteration'] for r in rows}
        assert len(completed) == len(rows) and completed.issubset(MILESTONES)
        state = dict(previous, controller_pid=os.getpid(), controller_resumed_at=time.time(),
                     note='Detached PPO trainer retained; evals may share its GPU when the second GPU is occupied')
    try:
        while True:
            now = time.time()
            state["alerts"] = []
            if train is None and gpu_room(0, 16000):
                cmd = command("train_sonic_adapter.py", "StitchMAX400v3") + [
                    "--num_envs", "2048", "--max_iterations", "9000", "--seed", "42",
                    "--run_name", RUN, "--tracking-scale", "2.0", "--resume",
                    "--load_run", "_walk_direct_import3500", "--checkpoint", "model_3500.pt"]
                train, log = launch(cmd, "train", 0)
                state.update(status="training", train_pid=train.pid)
            if train:
                text = common.tail(log)
                iterations = re.findall(r"Learning iteration\s+(\d+)", text)
                if iterations:
                    state["latest_iteration"] = int(iterations[-1])
                if "Traceback (most recent call last)" in text:
                    state["alerts"].append("Training traceback; inspect train.log")
                if train.poll() is not None and train.returncode != 0:
                    raise RuntimeError(f"Biped training exited {train.returncode}; inspect {log}")
                dirs = list((REPO/"Training/logs/rsl_rl/g1_sonic_adapter").glob("*_"+RUN))
                assert len(dirs) <= 1
                if dirs:
                    directory = dirs[0]
                    state["checkpoint_dir"] = str(directory)
            if active and active[1].poll() is not None:
                iteration, proc, logfile, started = active
                assert proc.returncode == 0, f"Evaluation {iteration} failed; inspect {logfile}"
                row = {"iteration": iteration, "host": "biped", "run": RUN,
                       **common.metrics(common.tail(logfile, 2000000))}
                rows.append(row)
                common.write(OUT/"results.json", rows)
                completed.add(iteration)
                active = None
            eval_gpu = None
            if directory and active is None:
                if gpu_room(1, 8000):
                    eval_gpu = 1
                elif gpu_room(0, 8000, allow_busy=True):
                    eval_gpu = 0
            if eval_gpu is not None:
                for iteration in MILESTONES:
                    ck = directory/f"model_{iteration}.pt"
                    if iteration in completed or not ck.is_file() or now-ck.stat().st_mtime < 10:
                        continue
                    cmd = command("eval_sonic_adapter.py", "StitchMAX100v3_eval") + [
                        "--num_envs", "1000", "--num_episodes", "2000", "--seed", "0", "--path", str(ck)]
                    proc, logfile = launch(cmd, f"eval_{iteration}", eval_gpu)
                    state['eval_gpu'] = eval_gpu
                    active = (iteration, proc, logfile, now)
                    break
            state.update(heartbeat=now, completed_evals=len(rows),
                         active_eval={"iteration": active[0], "pid": active[1].pid} if active else None)
            common.write(OUT/"workflow.json", state)
            if train and train.poll() is not None and active is None and len(completed) == len(MILESTONES):
                assert directory and (directory/"model_8999.pt").is_file()
                state.update(status="complete", completed=time.time())
                common.write(OUT/"workflow.json", state)
                return
            time.sleep(30)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), heartbeat=time.time())
        common.write(OUT/"workflow.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-controller', action='store_true')
    main(parser.parse_args().resume_controller)
