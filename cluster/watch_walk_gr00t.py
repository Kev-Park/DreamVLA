"""Read-only health and completion checks for all three walking GR00T chains."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from gr00t_budget_chain import atomic, checkpoint_ok, read

ROOT = Path.home()/"kevin"
OUT = ROOT/"eval_videos/walk_v3_study/gr00t"


def process_argv(pid, proc_root=Path('/proc')):
    p = proc_root/str(pid)/'cmdline'
    try:
        raw = p.read_bytes().decode().strip('\0')
        return raw.split('\0') if raw else []
    except OSError:
        return []


def process(pid):
    args = process_argv(pid)
    return ' '.join(args) if args else None


def stage_command_matches(expected, actual):
    expected = list(expected)
    if expected and Path(expected[0]).name == 'env':
        expected.pop(0)
        while expected and '=' in expected[0]:
            expected.pop(0)
    return bool(actual) and actual == expected


def active_output_logs(pid, wrapper, proc_root=Path('/proc')):
    """Inspect outputs of this live stage's descendants, including shard logs."""
    pending, seen, logs = [pid], set(), set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        proc = proc_root/str(current)
        try:
            pending.extend(map(int, (proc/'task'/str(current)/'children').read_text().split()))
        except (OSError, ValueError):
            pass  # Children may exit between the process and descriptor reads.
        for descriptor in ('1', '2'):
            try:
                target = (proc/'fd'/descriptor).resolve(strict=True)
                if target != wrapper.resolve() and target.is_relative_to(ROOT.resolve()) and target.is_file():
                    logs.add(target)
            except OSError:
                pass
    return sorted(logs)


def output_activity(pid, wrapper, now):
    child_logs = []
    for log in active_output_logs(pid, wrapper):
        try:
            child_logs.append({'path': str(log), 'seconds_since_log_update': now-log.stat().st_mtime})
        except OSError:
            pass
    wrapper_age = now-wrapper.stat().st_mtime if wrapper.exists() else None
    age = min(r['seconds_since_log_update'] for r in child_logs) if child_logs else wrapper_age
    # Each live shard must advance: an active sibling cannot hide a stalled one.
    stalled = [Path(r['path']).name for r in child_logs if r['seconds_since_log_update'] > 1800]
    if not child_logs and wrapper_age is not None and wrapper_age > 1800:
        stalled = [wrapper.name]
    return {'seconds_since_log_update': age, 'wrapper_log_age': wrapper_age,
            'active_child_logs': child_logs}, stalled


def stage_only_controllers(work, proc_root=Path('/proc'), now=None):
    """Check the additional bookkeeper even while the original chain waits.

    This corroborates terminal receipts; the existing checkpoint checks and
    full completion audits still verify the actual training outputs.
    """
    now = time.time() if now is None else now
    rows = []
    for review_path in sorted(work.glob('round*_single_stage_operator_review.json')):
        prefix = review_path.name.removesuffix('_operator_review.json')
        launch_path = work/(prefix+'.launch.json')
        row = {'stage': prefix, 'alerts': [], 'controller_live': False}
        rows.append(row)
        if not launch_path.exists():
            row['status'] = 'starting'
            if now-read(review_path)['at'] > 120:
                row['alerts'].append('Reviewed launch did not register its controller')
            continue
        launch = read(launch_path)
        row.update(controller_pid=launch['pid'], run=launch['run'])
        phase = launch.get('phase', 'finetune')
        row['phase'] = phase
        if phase not in ('finetune', 'evaluate'):
            row['status'] = 'needs_attention'
            row['alerts'].append('Unknown stage-only controller phase')
            continue
        failed = work/(prefix+'.abort.json')
        done = work/(prefix+'.done.json')
        if failed.exists():
            row.update(status='failed', failure=read(failed))
            row['alerts'].append('Stage-only controller recorded a failure')
            continue
        if done.exists():
            final = read(done)
            trained_path = work/(launch['run']+'.trained.json')
            trained = read(trained_path) if trained_path.exists() else {}
            if (final.get('run') != launch['run'] or final.get('steps') != launch['steps']
                    or trained.get('run') != launch['run']
                    or trained.get('steps') != launch['steps']
                    or final.get('checkpoint') != trained.get('checkpoint')):
                row['alerts'].append('Stage-only completion lacks matching canonical training receipt')
            if phase == 'evaluate':
                evaluated_path = work/(launch['run']+'.eval.json')
                report = read(evaluated_path) if evaluated_path.exists() else {}
                if (report.get('pool') != 'StitchMAX100v3_eval'
                        or report.get('episodes') != 100
                        or sorted(r.get('motion_id', -1) for r in report.get('per_episode', [])) != list(range(100))):
                    row['alerts'].append('Stage-only evaluation completion lacks the canonical 100-reference report')
            row['status'] = 'needs_attention' if row['alerts'] else 'completed'
            continue
        actual = process_argv(launch['pid'], proc_root)
        review = launch['operator_review']
        try:
            owned = (proc_root/str(launch['pid'])).stat().st_uid == os.getuid()
            same_boot = ((proc_root/'sys/kernel/random/boot_id').read_text().strip()
                         == review['boot_id'])
        except OSError:
            owned = same_boot = False
        matches = (len(actual) == 4 and Path(actual[0]).name.startswith('python')
                   and actual[1:3] == ['-u', '-c']
                   and hashlib.sha256(actual[3].encode()).hexdigest() == review['payload_sha256'])
        row['controller_live'] = bool(owned and same_boot and matches)
        row['status'] = 'running' if row['controller_live'] else 'controller_missing_or_changed'
        if not row['controller_live']:
            row['alerts'].append('Stage-only controller exited or its process identity changed')
    return rows


def inspect(budget):
    tag = f"walk{budget}_v3_dagger57_20261013"
    work = ROOT/"dline"/tag
    row = {"budget": budget, "tag": tag, "alerts": [], "rounds": []}
    launch = work/"controller.launch.json"
    if not launch.exists():
        row["status"] = "not_launched"
        return row
    handle = read(launch)
    cmd = process(handle["pid"])
    row.update(controller_pid=handle["pid"], controller_command=cmd)
    stages = stage_only_controllers(work)
    if stages:
        row['stage_only_controllers'] = stages
        row['alerts'].extend(stage['stage']+': '+alert
                             for stage in stages for alert in stage['alerts'])
    repairs = []
    for path in work.glob('*.startup_repair.json'):
        repair = read(path)
        if repair['status'] == 'complete':
            continue
        repair['command_live'] = process(repair['pid'])
        repairs.append(repair)
        if not repair['command_live'] or repair['status'] == 'failed_for_review':
            row['alerts'].append('Startup repair stopped or failed: '+repair['run'])
    if repairs:
        row['startup_repairs'] = repairs
    recovering = any(r['command_live'] for r in repairs)
    if (work/"abort.json").exists():
        failure = read(work/'abort.json')
        expected = any(r['command_live'] and failure['pid'] == r['controller_pid']
                       and f"eval_{r['run']}_shard{r['shard']} exited -6" in failure['error']
                       for r in repairs)
        if not expected:
            row.update(status="failed", failure=failure)
            return row
        row['recovery_boundary_failure'] = failure
    if cmd is None and not recovering and not (work/"done.json").exists():
        row["alerts"].append("Controller exited")
    for stage in work.glob("*.process.json"):
        job = read(stage)
        log = stage.with_name(stage.name.replace(".process.json", ".log"))
        actual = process_argv(job['pid'])
        if stage_command_matches(job['command'], actual):
            live = ' '.join(actual)
            activity, stalled = output_activity(job['pid'], log, time.time())
            row.setdefault("active_workers", []).append(dict(job, command_live=live,
                **activity))
            row['alerts'].extend('Stage log stalled: '+name for name in stalled)
    for request in sorted((work/'external').glob('*.request.json')):
        reply = request.with_name(request.name.replace('.request.json', '.reply.json'))
        if reply.exists():
            continue
        spec = read(request)
        waiting = request.with_name(request.name.replace('.request.json', '.waiting.json'))
        heartbeat = read(waiting) if waiting.exists() else {}
        age = time.time()-heartbeat['time'] if 'time' in heartbeat else None
        current = bool(cmd) and heartbeat.get('controller_pid') == handle['pid']
        row.setdefault('pending_external_stages', []).append({
            'phase': spec['phase'], 'run': spec['run'], 'host': spec['host'],
            'request': str(request), 'controller_wait_live': current,
            'waiting_heartbeat_age': age,
        })
        if current and age is not None and age > 180:
            row['alerts'].append('External-stage waiting heartbeat stalled: '+spec['run'])
    for index in range(4):
        run = f"{tag}_r{index}"
        trained, evaluated = work/(run+".trained.json"), work/(run+".eval.json")
        item = {"round": index, "run": run, "trained": False, "evaluated": False}
        if trained.exists():
            item.update(training=read(trained), trained=True)
            if not checkpoint_ok(Path(item["training"]["checkpoint"])):
                row["alerts"].append("Checkpoint shards missing: "+run)
        if evaluated.exists():
            item.update(evaluation=read(evaluated), evaluated=True)
            assert item["evaluation"]["episodes"] == 100
            assert sorted(r["motion_id"] for r in item["evaluation"]["per_episode"]) == list(range(100))
        row["rounds"].append(item)
    resumed = any(r['command_live'] and r['pid'] == handle['pid']
                  and r['status'] == 'resuming_from_verified_completed_stages' for r in repairs)
    row["status"] = ("recovering_startup_failure" if recovering and not resumed
                     else "running" if cmd else "stopped_without_completion")
    if (work/"done.json").exists():
        if not all(r['trained'] and r['evaluated'] for r in row['rounds']):
            row['alerts'].append('Completion marker lacks verified training/evaluation stages')
        row['status'] = 'needs_attention' if row['alerts'] else 'complete'
    elif cmd and (work/"waiting_for_base.json").exists() and not (work/"recipe.json").exists():
        base = ROOT/f"datasets/walk_v3_study/base{budget}"
        if not (base/"ready").exists():
            row["status"] = "waiting_for_exact_expert_base"
        elif any("convert_sharded.sh" in p["command_live"] for p in row.get("active_workers", [])):
            row["status"] = "converting_base"
        else:
            row["status"] = "preparing_base_training"
    elif cmd and not row.get('active_workers') and any(
            stage['controller_wait_live'] for stage in row.get('pending_external_stages', [])):
        row['status'] = 'waiting_for_external_stage'
    return row


def inspect_all(budgets):
    rows = []
    for budget in budgets:
        try:
            rows.append(inspect(budget))
        except Exception as exc:
            rows.append({'budget': budget, 'tag': f'walk{budget}_v3_dagger57_20261013',
                         'status': 'inspection_failed', 'rounds': [],
                         'alerts': ['Health inspection failed: '+repr(exc)]})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/"health.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        try:
            rows = inspect_all([10, 50, 200])
            state = {"heartbeat": time.time(), "watcher_pid": os.getpid(), "chains": rows,
                     "stage_only_evaluation_checks": True,
                     "status": "complete" if all(r["status"] == "complete" for r in rows) else "monitoring"}
            atomic(OUT/"workflow.json", state)
            if state["status"] == "complete":
                return
        except BaseException as exc:
            atomic(OUT/"health_failure.json", {"time": time.time(), "error": repr(exc), "pid": os.getpid()})
            raise
        time.sleep(30)


if __name__ == "__main__":
    main()
