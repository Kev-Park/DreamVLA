"""Compare reused R2 + eight newly converted R3 episodes against prior full R3.

Run only on the cluster. Every reused/new parquet and video is checked, followed
by whole-dataset GR00T statistics on both independently assembled datasets.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

root = Path.home()/'kevin'
os.sched_setaffinity(0,set(json.loads((root/'dline/cpu_share.json').read_text())['shared_mask']))
work = root/'conversion_incremental_validation'
work.mkdir(exist_ok=False)
here = Path(__file__).resolve().parent
prefix_root = root/'datasets/af60v8f_ds_dagger2/hdf5'
prefix = root/'datasets/af60v8f_ds_dagger2/lerobot/ds'
reference = root/'datasets/af60v8f_ds_dagger3/lerobot/ds'
all_hdf5 = sorted((root/'datasets/af60v8f_ds_dagger3/hdf5').glob('*.hdf5'))
count = json.loads((prefix/'meta/info.json').read_text())['total_episodes']+8
inputs = work/'hdf5'; inputs.mkdir()
for i, source in enumerate(all_hdf5[:count]):
    (inputs/f'{i:06d}__{source.name}').symlink_to(source.resolve(strict=True))

started = time.monotonic()
subprocess.run([sys.executable,str(here/'convert_incremental.py'),'--hdf5-root',str(inputs),
    '--prefix-hdf5-root',str(prefix_root),'--prefix-dataset',str(prefix),
    '--out',str(work/'incremental/ds'),'--task-prompt','pick up the mustard bottle',
    '--shards','4'],check=True)

# Build an independent prefix view of the previously fully converted R3.
# Hardlinks preserve the source and avoid recopying a gigabyte of videos.
view = work/'reference/ds'; (view/'meta').mkdir(parents=True)
info = json.loads((reference/'meta/info.json').read_text())
episode_lines = (reference/'meta/episodes.jsonl').read_text().splitlines(keepends=True)[:count]
stats_lines = (reference/'meta/episodes_stats.jsonl').read_text().splitlines(keepends=True)[:count]
episodes = [json.loads(x) for x in episode_lines]
frames = sum(e['length'] for e in episodes)
videos = [k for k,v in info['features'].items() if v['dtype']=='video']
import os
for ep in episodes:
    eid = ep['episode_index']; fields=dict(episode_chunk=eid//info['chunks_size'],episode_index=eid)
    for relative in [info['data_path'].format(**fields)]+[
        info['video_path'].format(video_key=k,**fields) for k in videos]:
        dst = view/relative; dst.parent.mkdir(parents=True,exist_ok=True)
        os.link(reference/relative,dst)
for name in ['tasks.jsonl','modality.json']:
    if (reference/'meta'/name).exists():
        shutil.copy2(reference/'meta'/name,view/'meta'/name)
(view/'meta/episodes.jsonl').write_text(''.join(episode_lines))
(view/'meta/episodes_stats.jsonl').write_text(''.join(stats_lines))
from lerobot.common.datasets.utils import write_json
info.update(total_episodes=count,total_frames=frames,total_videos=count*len(videos),
    total_chunks=(count-1)//info['chunks_size']+1,splits={'train':f'0:{count}'})
write_json(info,view/'meta/info.json')
subprocess.run([sys.executable,str(here/'verify_merge.py'),
    '--merged',str(work/'incremental/ds'),'--reference',str(view)],check=True)
code = 'from pathlib import Path; from gr00t.data.stats import generate_stats,generate_rel_stats; from gr00t.data.types import EmbodimentTag; import sys; p=Path(sys.argv[1]); generate_stats(p); generate_rel_stats(p,EmbodimentTag("unitree_g1_sonic"))'
for dataset in [work/'incremental/ds',view]:
    subprocess.run([str(root/'Isaac-GR00T/.venv/bin/python'),'-c',code,str(dataset)],check=True)
for name in ['stats.json','relative_stats.json']:
    assert json.loads((work/'incremental/ds/meta'/name).read_text()) == json.loads((view/'meta'/name).read_text()), name+' differs'
report=dict(status='passed',reused_episodes=count-8,newly_converted_episodes=8,
    checked_episodes=count,checked_frames=frames,all_parquet_video_metadata_equal=True,
    whole_dataset_stats_equal=True,relative_action_stats_equal=True,seconds=time.monotonic()-started)
(work/'validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
