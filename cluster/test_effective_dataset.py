"""Verify effective expansion preserves source data and balanced sampling."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from build_effective_dataset import build, schedule, validate


class EffectiveDatasetTests(unittest.TestCase):
    def test_sampling_is_balanced_and_reproducible(self):
        from collections import Counter
        for n in (342,117):
            ids=schedule(list(range(n)))
            self.assertEqual(ids,schedule(list(range(n))))
            counts=Counter(ids)
            self.assertEqual(len(ids),1200)
            self.assertEqual(len(counts),n)
            self.assertLessEqual(max(counts.values())-min(counts.values()),1)

    def test_preserves_columns_and_video_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source';out=Path(directory)/'effective'
            (source/'meta').mkdir(parents=True)
            info=dict(codebase_version='v2.1',chunks_size=1000,
                data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
                video_path='videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
                features={'observation.images.ego_view':{'dtype':'video'},'action.motion_token':{'dtype':'float64'}},
                total_episodes=2,total_frames=7,total_videos=2,total_chunks=1,splits={'train':'0:2'})
            (source/'meta/info.json').write_text(json.dumps(info))
            (source/'meta/tasks.jsonl').write_text('{"task_index":0,"task":"pick"}\n')
            episodes=[];stats=[];offset=0
            for i,n in enumerate((3,4)):
                episodes.append(dict(episode_index=i,length=n,tasks=['pick']))
                stats.append(dict(episode_index=i,stats={'episode_index':{},'index':{}}))
                path=source/info['data_path'].format(episode_chunk=0,episode_index=i);path.parent.mkdir(parents=True,exist_ok=True)
                table=pa.table({'episode_index':np.full(n,i),'index':np.arange(offset,offset+n),
                    'action.motion_token':[[float(i),float(j)] for j in range(n)],'timestamp':np.arange(n)/50})
                pq.write_table(table,path)
                video=source/info['video_path'].format(episode_chunk=0,video_key='observation.images.ego_view',episode_index=i)
                video.parent.mkdir(parents=True,exist_ok=True);video.write_bytes(b'video-byte-fixture'+bytes([i]))
                offset+=n
            for name,rows in [('episodes.jsonl',episodes),('episodes_stats.jsonl',stats)]:
                (source/'meta'/name).write_text(''.join(json.dumps(row)+'\n' for row in rows))
            before=(source/'meta/info.json').read_bytes()
            result=build(source,out,count=5)
            self.assertEqual(result['episodes'],5)
            self.assertEqual(result['unique_source_episodes'],2)
            self.assertEqual(before,(source/'meta/info.json').read_bytes())
            self.assertTrue((out/'READY').exists())
            path=out/info['data_path'].format(episode_chunk=0,episode_index=0)
            table=pq.read_table(path)
            field=table.schema.field('timestamp');table=table.set_column(table.schema.get_field_index('timestamp'),field,pa.array(np.ones(len(table)),type=field.type))
            pq.write_table(table,path)
            with self.assertRaisesRegex(ValueError,'Original training column changed'):
                validate(source,out)


if __name__=='__main__':unittest.main()
