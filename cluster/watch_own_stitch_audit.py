"""Aggregate the latest attempt of each replay shard; never launch training."""
import json
import os
from pathlib import Path
import re
import time

import watch_t20max_v2 as shared

OUT = shared.ROOT/'eval_videos/StitchOwn_v3'


def main():
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:26])
    while True:
        latest = {}
        for path in OUT.glob('replay_full*_status.json'):
            state = json.loads(path.read_text())
            for job in state['jobs']:
                gpu = job['gpu']
                if gpu not in latest or state['started'] > latest[gpu]['attempt_started']:
                    latest[gpu] = {**job, 'attempt_started':state['started'], 'controller_status':str(path)}
        alerts = []
        for gpu, job in latest.items():
            log = Path(job['log'])
            txt = shared.tail(log)
            segments = re.findall(r'\[montage\] segment (\d+)/(\d+)',txt)
            job['segments_started'] = int(segments[-1][0]) if segments else 0
            if job['status'] == 'running':
                if 'Traceback (most recent call last)' in txt:
                    alerts.append(f'GPU {gpu}: exception in log; process shutdown may be hung')
                if time.time()-log.stat().st_mtime > 900:
                    alerts.append(f'GPU {gpu}: no log progress for 15 minutes')
                if not Path('/proc',str(job['pid'])).exists():
                    alerts.append(f'GPU {gpu}: process gone; awaiting controller receipt')
            elif job['status'] == 'failed':
                alerts.append(f'GPU {gpu}: replay failed; inspect receipt')
        complete = len(latest)==8 and all(j['status']=='complete' for j in latest.values())
        result = {'updated':time.time(),'watcher_pid':os.getpid(),
                  'phase':'replay_complete_pending_user_review' if complete else 'replay_in_progress',
                  'alerts':alerts,'training_authorized_to_start':False,
                  'expected_references':500,'completed_references':sum(j['count'] for j in latest.values() if j['status']=='complete'),
                  'segments_started':sum(j['segments_started'] for j in latest.values()),
                  'jobs':list(latest.values())}
        shared.write(OUT/'audit_workflow.json',result)
        if complete:
            assert result['completed_references']==500
            for pool in ['StitchMAX200v3','StitchMAX200v3_aug','StitchMAX400v3','StitchMAX100v3_eval']:
                p=shared.ROOT/'ref_motions'/pool/'_PROVENANCE.json'
                d=json.loads(p.read_text())
                d.update(phase=result['phase'], numerical_validation=str(OUT/'numerical_validation.json'),
                         replay_validation=str(OUT/'audit_workflow.json'), training_authorized_to_start=False)
                shared.write(p,d)
            return
        time.sleep(15)


if __name__=='__main__':
    main()
