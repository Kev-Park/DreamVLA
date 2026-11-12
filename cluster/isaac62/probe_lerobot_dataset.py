"""Read real saved episodes through GR00T's loader and default video decoder.

Run remotely in the existing GR00T environment with CUDA disabled. This checks
data reuse and decoded-image validity, not task success or new Isaac rendering.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, action='append', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(started=time.time(), completed=False, gpu_used=False, datasets=[],
                  scope='Saved episode loader/parquet/codec only; no simulator rendering or task assessment')
    try:
        import numpy as np
        import pandas as pd
        from PIL import Image
        import torch
        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
        from gr00t.data.types import ModalityConfig

        torch.set_num_threads(1)
        config_path = args.checkpoint/'processor_config.json'
        raw_config = json.loads(config_path.read_text())['processor_kwargs']['modality_configs']['unitree_g1_sonic']
        config = {key: ModalityConfig(**value) for key, value in raw_config.items()}
        report['versions'] = {name: importlib.metadata.version(name)
                              for name in ['torch', 'numpy', 'torchcodec', 'pandas', 'pyarrow']}
        for dataset in args.dataset:
            row = dict(path=str(dataset), completed=False)
            report['datasets'].append(row)
            try:
                loader = LeRobotEpisodeLoader(dataset_path=dataset, modality_configs=config,
                                             video_backend='torchcodec')
                row['episodes'] = []
                for episode_offset, meta in enumerate(loader.episodes_metadata):
                    episode = int(meta['episode_index'])
                    raw_path = dataset/loader.data_path_pattern.format(
                        episode_chunk=episode//loader.chunk_size, episode_index=episode)
                    raw = pd.read_parquet(raw_path)
                    decoded = loader[episode_offset]
                    assert len(raw) == len(decoded) == int(meta['length']), 'Episode/video/parquet length mismatch'
                    comparisons = {}
                    for modality in ['state', 'action']:
                        for key in config[modality].modality_keys:
                            group = loader.modality_meta[modality][key]
                            column = group.get('original_key', 'observation.state' if modality == 'state' else 'action')
                            expected = np.stack(raw[column].to_numpy())[:, group['start']:group['end']]
                            actual = np.stack(decoded[f'{modality}.{key}'].to_numpy())
                            assert np.isfinite(actual).all(), (modality, key)
                            np.testing.assert_array_equal(actual, expected)
                            comparisons[f'{modality}.{key}'] = list(actual.shape)
                    images = []
                    for key in config['video'].modality_keys:
                        for index in sorted({0, len(decoded)//2, len(decoded)-1}):
                            pixels = np.asarray(decoded[f'video.{key}'].iloc[index])
                            assert pixels.ndim == 3 and pixels.shape[-1] == 3 and pixels.dtype == np.uint8
                            assert pixels.max() > 0 and pixels.std() > 0, 'Black/constant decoded image'
                            filename = f'{len(report["datasets"])-1}_ep{episode:06d}_{key}_{index:05d}.png'
                            Image.fromarray(pixels).save(args.output/filename)
                            images.append(dict(file=filename, shape=list(pixels.shape), mean=float(pixels.mean())))
                    row['episodes'].append(dict(episode=episode, episode_frames=len(decoded),
                               dataset_episodes=len(loader), exact_numeric_columns=comparisons,
                               decoded_images=images, backend='torchcodec',
                               task_success_assessed=False))
                row.update(completed=True, dataset_episodes=len(loader),
                           checked_frames=sum(ep['episode_frames'] for ep in row['episodes']))
                del raw, decoded, loader
            except Exception:
                row.update(error=traceback.format_exc())
        report['completed'] = all(row['completed'] for row in report['datasets'])
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['finished'] = time.time()
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if not report['completed']:
        raise SystemExit('At least one real dataset/codec check failed; inspect report')


if __name__ == '__main__':
    main()
