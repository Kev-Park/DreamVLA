"""Read account scratch quota; global filesystem free space is insufficient."""
import subprocess


# Two full model/optimizer saves coexist before Trainer rotates checkpoints.
CHECKPOINT_RESERVE_BYTES = 55_000_000_000


def parse_quota(text):
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[0].endswith(':/scratch/network'):
            used, soft, hard = [int(value.rstrip('*')) * 1024 for value in fields[1:4]]
            if hard <= 0 or used < 0:
                raise ValueError('Scratch quota must report a finite positive hard limit')
            return {'used_bytes': used, 'soft_limit_bytes': soft,
                    'hard_limit_bytes': hard, 'headroom_bytes': hard - used}
    raise ValueError('No unwrapped /scratch/network account quota row found')


def read_quota(required_reserve_bytes=CHECKPOINT_RESERVE_BYTES):
    # quota returns 1 when a limit is exceeded; its report remains authoritative.
    observed = subprocess.run(['quota', '-w', '-p'], capture_output=True,
                              text=True, timeout=15)
    result = parse_quota(observed.stdout)
    result['command_returncode'] = observed.returncode
    result['required_reserve_bytes'] = required_reserve_bytes
    if result['headroom_bytes'] < required_reserve_bytes:
        result['attention'] = 'Account scratch headroom below checkpoint reserve; hold archive writes and review storage'
    return result


if __name__ == '__main__':
    import json
    value = read_quota()
    print(json.dumps(value))
    if value.get('attention'):
        raise SystemExit('Insufficient account quota for a safe fine-tune start')
