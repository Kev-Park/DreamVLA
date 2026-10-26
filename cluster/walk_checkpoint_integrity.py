"""Bounded structural checks for completed GR00T checkpoints, without loading tensors."""
import json
from pathlib import Path
import struct
import zipfile


def file_info(path):
    stat = path.stat()
    assert stat.st_size > 8, f'Missing or empty checkpoint file: {path}'
    allocated = getattr(stat, 'st_blocks', None)
    if allocated is not None:
        # These saves write full tensor payloads. Unexpected holes need review:
        # quota failures can leave the expected logical length but unwritten data.
        allocated *= 512
        assert allocated + 4096 >= stat.st_size, f'Unexpected sparse checkpoint file; inspect save: {path}'
    return {'bytes': stat.st_size, 'allocated_bytes': allocated}


def validate_checkpoint(checkpoint, expected_step):
    checkpoint = Path(checkpoint)
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    assert state['global_step'] == expected_step, 'Incorrect checkpoint update count'
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())
    mapping = index['weight_map']
    assert mapping, 'Empty tensor index'
    seen = set()
    total = 0
    shards = []
    for name in sorted(set(mapping.values())):
        assert Path(name).name == name and name.endswith('.safetensors'), 'Unsafe shard name'
        path = checkpoint / name
        info = file_info(path)
        with path.open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            assert 0 < length < min(info['bytes'] - 7, 100_000_000), f'Invalid tensor header: {name}'
            header = json.loads(stream.read(length))
        offsets = []
        for tensor, spec in header.items():
            if tensor == '__metadata__':
                continue
            assert tensor not in seen and mapping.get(tensor) == name, 'Tensor index mismatch'
            seen.add(tensor)
            start, end = spec['data_offsets']
            assert isinstance(start, int) and isinstance(end, int) and 0 <= start <= end
            offsets.append((start, end))
        cursor = 0
        for start, end in sorted(offsets):
            assert start == cursor, 'Tensor offsets overlap or leave gaps'
            cursor = end
        assert cursor + length + 8 == info['bytes'], f'Incomplete tensor shard: {name}'
        total += cursor
        shards.append(dict(info, name=name, tensors=len(offsets), payload_bytes=cursor))
    assert seen == set(mapping), 'Index references missing tensors'
    assert total == index['metadata']['total_size'], 'Tensor byte count differs from index'
    archives = {}
    for name in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
        path = checkpoint / name
        info = file_info(path)
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            assert entries and any(item.filename.endswith('/data.pkl') for item in entries), f'Invalid training archive: {name}'
            assert all(0 <= item.header_offset and item.header_offset + 30 + item.compress_size <= info['bytes']
                       for item in entries), f'Truncated training archive: {name}'
        archives[name] = dict(info, entries=len(entries))
    return {'global_step': expected_step, 'shards': shards, 'tensor_payload_bytes': total,
            'training_archives': archives,
            'validation': 'Trainer step, tensor index/header/offsets/length, file allocation and training ZIP directories',
            'limits': 'No tensor materialization, payload CRC/hash checks, numerical checks or execution test'}
