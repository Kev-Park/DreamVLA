"""Select the smallest measured batch that preserves a deadline eval window."""
import argparse
import datetime
import json
import math
from pathlib import Path
import time


def choose(frames, policy, now):
    passes = float(policy['passes'])
    deadline = datetime.datetime.fromisoformat(policy['deadline'].replace('Z', '+00:00')).timestamp()
    reserve = policy.get('evaluation_transfer_reserve_seconds', 5400)
    available = deadline-now-reserve
    choices = []
    for result in policy['benchmarks']:
        if result.get('status') != 'completed':
            continue
        batch = int(result['batch'])
        assert batch in (32, 64, 128)
        steps = math.ceil(frames*passes/batch)
        # Short-run timing omits checkpoint saves and long-run variability.
        rate = float(result['frames_per_second'])*0.8
        seconds = steps*batch/rate
        choices.append(dict(global_batch_size=batch, steps=steps,
            estimated_training_seconds=seconds, deadline_feasible=seconds<=available))
    if not choices:
        raise ValueError('No completed batch benchmark available')
    feasible = [c for c in choices if c['deadline_feasible']]
    selected = min(feasible, key=lambda c:c['global_batch_size']) if feasible else min(
        choices, key=lambda c:c['estimated_training_seconds'])
    return dict(selected, frames=frames, effective_passes=selected['steps']*selected['global_batch_size']/frames,
        target_passes=passes, selected_at=now, available_training_seconds=available,
        evaluation_transfer_reserve_seconds=reserve, candidates=choices,
        precision='bf16', learning_rate=1e-4, optimizer='adamw_torch',
        throughput_discount=0.8, cloud_only=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('policy')
    parser.add_argument('dataset_info')
    parser.add_argument('output')
    args = parser.parse_args()
    recipe = choose(json.loads(Path(args.dataset_info).read_text())['total_frames'],
                    json.loads(Path(args.policy).read_text()), time.time())
    Path(args.output).write_text(json.dumps(recipe, indent=2))
    print(recipe['global_batch_size'], recipe['steps'])
