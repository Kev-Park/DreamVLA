"""Compare decoder threading on real clips; require identical decoded pixels.

CPU-only probe on a cluster, bounded to eight CPUs and eight training clips.
It neither trains a model nor writes decoded images to disk.
"""
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time


def decode(item):
    import torch
    from torchcodec.decoders import VideoDecoder
    torch.set_num_threads(1)
    index, path = item
    rows = {}
    for threads in ([0, 1] if index % 2 == 0 else [1, 0]):
        started = time.perf_counter()
        decoder = VideoDecoder(str(path), device='cpu', dimension_order='NHWC',
                               num_ffmpeg_threads=threads)
        frames = decoder.get_frames_at(indices=list(range(len(decoder)))).data
        elapsed = time.perf_counter() - started
        rows[str(threads)] = {'decode_seconds': elapsed, 'frames': len(frames),
                             'shape': list(frames.shape),
                             'sha256': hashlib.sha256(frames.contiguous().numpy()).hexdigest()}
        del frames, decoder
    assert rows['0']['sha256'] == rows['1']['sha256']
    assert rows['0']['shape'] == rows['1']['shape']
    return {'file': str(path), 'measurements': rows}


def main():
    dataset = Path(sys.argv[1]).expanduser()
    clips = sorted(dataset.glob('videos/**/*.mp4'))[:8]
    assert len(clips) == 8
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
        rows = list(pool.map(decode, enumerate(clips)))
    totals = {n: sum(row['measurements'][n]['decode_seconds'] for row in rows) for n in ['0', '1']}
    report = {'host': __import__('socket').gethostname(), 'time': time.time(),
              'cpus': sorted(os.sched_getaffinity(0)), 'workers': 4,
              'clips': rows, 'total_decode_seconds': totals,
              'automatic_over_single_thread_ratio': totals['0'] / totals['1'],
              'pixels_identical': True}
    out = Path.home() / 'kevin/eval_videos/pick_video_decoder_threads.json'
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
