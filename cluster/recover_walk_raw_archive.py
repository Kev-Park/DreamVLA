"""Reconcile a known interrupted walking archive; preserve old target fragments."""
import argparse
import json
from pathlib import Path
import time

import walk_late_takeover_bridge as bridge
from walk_adroit_bridge import inventory, save
from walk_archive_recovery import plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', required=True)
    parser.add_argument('--batch', choices=('batch_004', 'batch_005'), required=True)
    args = parser.parse_args()
    directory = Path(args.state_dir).resolve()
    bridge.configure(directory)
    label = args.batch + '_raw_adroit'
    marker = directory / (label + '.transfer.json')
    previous = json.loads(marker.read_text())
    expected = json.loads((directory / (label + '.source_inventory.json')).read_text())
    relative = bridge.WORK + '/raw/' + args.batch
    assert previous['source'] == 'bluesclues' and previous['destination'] == 'adroit'
    assert previous['relative'] == relative and previous.get('failed')
    assert not previous.get('verified'), 'Already repaired; inspect before another action'
    stamp = str(time.time_ns())
    record = directory / (label + '.recovery_' + stamp + '.json')
    state = {'pid': __import__('os').getpid(), 'started': time.time(),
             'batch': args.batch, 'status': 'preflight'}

    def update(**values):
        state.update(heartbeat=time.time(), **values)
        save(record, state)

    try:
        # These are per-operation admission reviews, not account-wide enforcement.
        cpu = bridge.resources('bluesclues')
        collector = bridge.collector_health()
        cap = 30 if collector['completed_attempts'] < 200 else 20
        assert cpu['cpu_measurement_complete']
        assert cpu['account_cpu_pct'] + 2 / cpu['logical_cpus'] * 100 <= cap, {'account_cpu_pct':cpu['account_cpu_pct'],'cap':cap}
        update(resource_review={'account_cpu_pct': cpu['account_cpu_pct'], 'cap': cap,
                                'additional_cpu_upper_bound': 2 / cpu['logical_cpus'] * 100})
        assert inventory('bluesclues', relative) == expected, 'Source changed; retain for review'
        inspected = inventory('adroit', relative)
        needed = plan(expected, inspected)
        update(status='planned', repaired_files=len(needed),
               repair_bytes=sum(row['bytes'] for row in needed),
               original_bytes=previous['bytes'])
        if needed:
            stage = bridge.WORK + '/recovery/' + args.batch + '_' + stamp
            backup = bridge.WORK + '/recovery/' + args.batch + '_old_' + stamp
            # Hard links add no raw-data payload to Bluesclues and preserve the
            # original source. Both endpoints use the ordinary verified relay.
            bridge.call('bluesclues', r'''
import json,os,pathlib
b=(pathlib.Path.home()/'kevin').resolve();source=b/REL;stage=b/STAGE
assert stage.resolve().is_relative_to(b/'dline'/TAG/'recovery')
assert not stage.exists();stage.mkdir(parents=True)
for row in NEEDED:
 p=source/row['path'];out=stage/row['path']
 assert not p.is_symlink() and p.resolve().is_relative_to(source.resolve())
 assert p.stat().st_size==row['bytes'];out.parent.mkdir(parents=True,exist_ok=True)
 os.link(p,out)
print(json.dumps({'linked_files':len(NEEDED)}))
'''.replace('REL', repr(relative)).replace('STAGE', repr(stage))
                .replace('TAG', repr(bridge.TAG)).replace('NEEDED', repr(needed)))
            update(status='transferring_selected_files', staging=stage, backup=backup)
            bridge.checked_transfer('bluesclues', 'adroit', stage, directory,
                                    label + '_repair_' + stamp)
            journal = bridge.call('adroit', r'''
import json,pathlib,sys
b=(pathlib.Path.home()/'kevin').resolve()
sys.path.insert(0,str(b/'wt/locomotion/DreamVLA/cluster'))
from walk_archive_recovery import promote
destination=b/REL;staging=b/STAGE;backup=b/BACKUP
assert destination.resolve().is_relative_to(b/'dline'/TAG/'raw')
assert all(p.resolve().is_relative_to(b/'dline'/TAG/'recovery') for p in (staging,backup))
print(json.dumps(promote(destination,staging,backup,EXPECTED,INSPECTED)))
'''.replace('REL', repr(relative)).replace('STAGE', repr(stage))
                .replace('BACKUP', repr(backup)).replace('TAG', repr(bridge.TAG))
                .replace('EXPECTED', repr(expected)).replace('INSPECTED', repr(inspected)),
                timeout=1800)
            update(status='promoted_and_verified', promotion=journal)
        assert inventory('adroit', relative) == expected
        save(directory / (label + '.failed_before_repair_' + stamp + '.json'), previous)
        repaired = dict(previous, verified=True, completed=time.time(), files=len(expected),
                        reconciliation=str(record), repaired_files=len(needed),
                        repair_bytes=sum(row['bytes'] for row in needed))
        save(marker, repaired)
        if needed:
            bridge.remove_relocated_copy('bluesclues', stage, needed,
                                         {'host': 'adroit', 'relative': relative})
        update(status='complete', completed=time.time())
        print(json.dumps(state), flush=True)
    except BaseException as exc:
        update(status='stopped_for_review', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
