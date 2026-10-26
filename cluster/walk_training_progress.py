"""Read-only GR00T progress checks; never changes or restarts a job."""
import math
from pathlib import Path
import re
import time


BAR=re.compile(r'(\d+)/(\d+)\s+\[([\d:]+)<([\d:]+),\s*([\d.]+)(it/s|s/it)\]')


def seconds(value):
    result=0
    for part in value.split(':'):
        result=result*60+int(part)
    return result


def parse_progress(text,expected_total):
    matches=[m for m in BAR.finditer(text) if int(m[2])==expected_total]
    if not matches:return {}
    match=matches[-1];step=int(match[1]);rate=float(match[5])
    assert 0<=step<=expected_total and math.isfinite(rate) and rate>0
    return {'step':step,'total_steps':expected_total,
            'updates_per_second':rate if match[6]=='it/s' else 1/rate,
            'reported_elapsed_seconds':seconds(match[3]),
            'reported_remaining_seconds':seconds(match[4])}


def read_progress(log,expected_total):
    path=Path(log);result={'observed_at':time.time(),'log_exists':path.is_file()}
    if not result['log_exists']:return result
    stat=path.stat();result.update(log_bytes=stat.st_size,log_mtime=stat.st_mtime)
    with path.open('rb') as stream:
        stream.seek(max(0,stat.st_size-65536))
        text=stream.read().decode(errors='replace')
    result.update(parse_progress(text,expected_total))
    return result


def watch_progress(previous,progress,now,grace_seconds=1800):
    """Return alert metadata; a stalled observation never authorizes a restart."""
    watch=dict(previous or {'first_running_observed_at':now,'last_advanced_at':now})
    watch['observed_at']=now;watch.pop('attention',None)
    step=progress.get('step')
    if step is not None:
        previous_step=watch.get('step')
        if previous_step is None or step>previous_step:
            watch.update(step=step,last_advanced_at=now)
        elif step<previous_step:
            watch['attention']='Observed optimizer step decreased;inspect the existing job'
    if now-watch['last_advanced_at']>grace_seconds:
        watch['attention']='No advancing optimizer step observed for30minutes;inspect the existing job'
    if progress.get('observation_error'):
        watch['observation_error']=progress['observation_error']
    else:watch.pop('observation_error',None)
    return watch
