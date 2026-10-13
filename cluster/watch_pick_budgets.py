"""Read-only health watcher for the two fixed-reference pick ablation chains."""
import argparse
import json
import os
from pathlib import Path
import re
import time


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def tail(path, count=24000):
    try:
        with path.open('rb') as f:
            f.seek(max(0, path.stat().st_size-count))
            return f.read().decode(errors='replace')
    except OSError:
        return ''


def live_args(pid):
    try:
        return (Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    except OSError:
        return ''


def observe(work, now):
    status=tail(work/'status',100000)
    done=read_json(work/'done.json')
    abort=read_json(work/'abort.json')
    controller_text=tail(work/'controller.lock').strip()
    controller_pid=int(controller_text) if controller_text.isdigit() else None
    controller_args=live_args(controller_pid) if controller_pid else ''
    controller_live='pick_budget_chain.py' in controller_args
    result={'path':str(work),'controller_pid':controller_pid,'controller_live':controller_live,
            'done':done,'abort':abort,'active_workers':[],'alerts':[],'latest_status':status.splitlines()[-8:]}
    if abort:
        result['alerts'].append('Controller recorded an abort: '+str(abort.get('error')))
    if not controller_live and not done:
        result['alerts'].append('Controller is not alive and completion is unverified')
    for procfile in sorted(work.glob('*.process.json')):
        record=read_json(procfile)
        if not record:
            result['alerts'].append('Unreadable process record: '+str(procfile))
            continue
        label=procfile.name.removesuffix('.process.json')
        args=live_args(record['pid'])
        expected=record['command']
        # Long-lived workers must match the recorded script and run-specific paths,
        # rather than merely occupying a recycled process ID.
        command_matches=bool(args) and all(str(x) in args for x in expected[1:] if str(x) not in ['env'])
        exits=re.findall(r'EXIT '+re.escape(label)+r' code=(-?\d+)',status)
        if exits and int(exits[-1]):
            result['alerts'].append(f'{label}: exited {exits[-1]}')
        if not command_matches:
            if not exits and now-record['started']>120:
                result['alerts'].append(f'{label}: process disappeared without a recorded exit')
            continue
        logfile=work/(label+'.log')
        age=now-logfile.stat().st_mtime if logfile.exists() else now-record['started']
        content=tail(logfile)
        counts=re.findall(r'(\d+)/(\d+) \[[^\r\n]*?([\d.]+)(it/s|s/it)',content)
        progress=None
        if label.startswith('ft_') and counts:
            current,total,rate,unit=counts[-1]
            seconds_per_step=1/float(rate) if unit=='it/s' else float(rate)
            progress={'step':int(current),'total_steps':int(total),'seconds_per_step':seconds_per_step,
                      'remaining_train_seconds':(int(total)-int(current))*seconds_per_step}
        result['active_workers'].append({'label':label,'pid':record['pid'],'gpu':record.get('gpu'),
                                        'output_age_seconds':age,'training_progress':progress})
        if age>1800:
            result['alerts'].append(f'{label}: live process has no new log output for {age:.0f}s')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    root=Path.home()/'kevin/dline'
    output=root/'pick_budget_health.json'
    while True:
        now=time.time()
        report={'time':now,'watcher_pid':os.getpid(),'chains':{}}
        for budget in [10,50]:
            tag=f'pick{budget}_dagger57_20261013'
            report['chains'][tag]=observe(root/tag,now)
        temp=output.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(output)
        summary={tag:{'done':bool(r['done']),'workers':r['active_workers'],'alerts':r['alerts']} for tag,r in report['chains'].items()}
        print(json.dumps({'time':now,**summary}),flush=True)
        if args.once or all(r['done'] for r in report['chains'].values()):break
        time.sleep(60)


if __name__=='__main__':main()
