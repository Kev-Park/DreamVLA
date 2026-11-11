"""Run one diagnostic reference with the accepted collector recipe.

Outputs are excluded from the training chain. The original stalled process and
its recordings remain untouched. Thread tracebacks expose any recurrent stall.
"""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    import gr00t_budget_chain as base
    from walk_skipstart_chain import FreshChain, TAG

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=(0, 4), default=4)
    args_cli = parser.parse_args()
    assert len(os.sched_getaffinity(0)) == 1
    chain = FreshChain()
    label = f'stall_reference53_gpu{args_cli.gpu}'
    state_file = chain.work/(label+'.probe.json')
    assert not state_file.exists()
    assert not (chain.work/(label+'.log')).exists()
    source = base.read(chain.work/'r2_batch005.process.json')
    assert source['pid'] == 172639
    old = Path('/proc/172639')
    fields = (old/'stat').read_text().rsplit(')', 1)[1].split()
    assert old.stat().st_uid == os.getuid()
    assert int(fields[19]) == 51971564 and fields[0] in ('T', 't')
    command = list(source['command'])
    pos = command.index('--skip-start-frames')
    assert command[pos+1] == '0'
    del command[pos:pos+2]
    command = [a for a in command if not str(a).startswith('--kit_args=')]
    ids = chain.work/(label+'.ids.txt')
    ids.write_text('53\n')
    command[command.index('--dagger-motion-ids')+1] = str(ids)
    raw = chain.data/'diagnostics'/label
    raw.mkdir(parents=True, exist_ok=False)
    command[command.index('--output-directory')+1] = str(raw)
    assert command[command.index('--seed')+1] == '8'
    state = dict(status='starting', pid=os.getpid(), at=time.time(), gpu=args_cli.gpu,
        motion_id=53, original_collector_pid=172639, cpu_mask=sorted(os.sched_getaffinity(0)),
        source_command=source['command'], training_data=False,
        comparison_limit='Fresh process and GPU differ; matching reference and recipe do not recreate prior RNG history')
    base.atomic(state_file, state)
    gpu = chain.reserve(args_cli.gpu, 20000)[0]
    try:
        state.update(status='collecting', at=time.time())
        base.atomic(state_file, state)
        chain.execute(command, label, gpu, chain.training_cwd)
        text = (chain.work/(label+'.log')).read_text()
        coverage = base.verify_collection_logs([text], [53])
        state.update(status='completed', at=time.time(), coverage=coverage,
            raw_files=[str(p) for p in raw.rglob('*.hdf5')], raw_root=str(raw))
        result = base.ROOT/'eval_videos'/(TAG+'_'+label+'.json')
        base.atomic(result, state)
        state['result'] = str(result)
        base.atomic(state_file, state)
    except BaseException as exc:
        state.update(status='failed', at=time.time(), error=repr(exc))
        base.atomic(state_file, state)
        raise
    finally:
        chain.release(gpu)


if __name__ == '__main__':
    main()
