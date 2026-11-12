#!/usr/bin/env python3
"""Original OmniControl Button Press sampling, without unused SMPL/dataset loading.

Run with envs/phase1 (Python 3.7). This is strictly the HumanML3D hml_vec/XYZ
inference path: no SMPL rotations, meshes, training or evaluation are supported.
The upstream model, 1000-step diffusion, prompt/hint generator and normalization
are reused, not replaced with IK or a synthetic motion generator.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

SCRIPT = Path(__file__).resolve()
from .paths import BACKEND_ROOT, ASSET_ROOT

TRAJGEN = BACKEND_ROOT / "TrajGen"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.count <= 100:
        parser.error("A batch must contain 1..100 fresh samples")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(str(TRAJGEN))
    sys.path.insert(0, str(TRAJGEN))
    import numpy as np
    import torch
    import clip
    from utils.fixseed import fixseed
    from utils.parser_util import (
        add_data_options,
        add_model_options,
        add_diffusion_options,
    )
    from utils.model_util import (
        get_model_args,
        create_gaussian_diffusion,
        load_model_wo_clip,
    )
    from model import cmdm
    from model.cfg_sampler import ClassifierFreeSampleModel
    from data_loaders.tensors import collate
    from data_loaders.humanml.scripts.motion_process import recover_from_ric
    from utils.text_control_example import collate_all

    class XYZOnly:
        """The upstream XYZ branch is identity. Reject every other representation."""

        def __init__(self, device, dataset):
            if dataset != "humanml":
                raise ValueError("Only HumanML3D XYZ inference is supported")
            self.smpl_model = torch.nn.Identity()  # CMDM._apply/train delegate here.

        def __call__(self, x, pose_rep, **kwargs):
            if pose_rep != "xyz":
                raise ValueError("SMPL assets are required for non-XYZ conversion")
            return x

    # Process-local adapter only. Do not modify the shared upstream implementation.
    cmdm.Rotation2xyz = XYZOnly
    cache = ASSET_ROOT / "phase1/clip"
    cache.mkdir(parents=True, exist_ok=True)

    def load_clip(self, version):
        model, _ = clip.load(version, device="cpu", jit=False, download_root=str(cache))
        clip.model.convert_weights(model)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        return model

    cmdm.CMDM.load_and_freeze_clip = load_clip
    torch.set_num_threads(2)
    fixseed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("A single allocated CUDA GPU is required for sampling")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda:%d" % args.device)
    defaults = argparse.ArgumentParser()
    add_data_options(defaults)
    add_model_options(defaults)
    add_diffusion_options(defaults)
    config = defaults.parse_args([])
    config.cond_mode = "both_text_spatial"
    checkpoint = TRAJGEN / "save/omnicontrol_ckpt/model_humanml3d.pt"
    norms = [
        TRAJGEN / "dataset" / name
        for name in (
            "HumanML3D/Mean.npy",
            "HumanML3D/Std.npy",
            "humanml_spatial_norm/Mean_raw.npy",
            "humanml_spatial_norm/Std_raw.npy",
        )
    ]
    arrays = [np.load(str(path)) for path in norms]
    mean, std = arrays[:2]
    if mean.shape != (263,) or std.shape != (263,) or not np.all(std > 0):
        raise ValueError("Invalid HumanML3D normalization")
    # Same model-default override behavior as parse_and_load_from_model.
    config_path = checkpoint.parent / "args.json"
    if config_path.is_file():
        for name, value in json.loads(config_path.read_text()).items():
            if hasattr(config, name) and name not in ("control_joint", "density"):
                setattr(config, name, value)
    model_args = get_model_args(config, None)
    if model_args["data_rep"] != "hml_vec" or model_args["njoints"] != 263:
        raise ValueError("Unexpected motion representation")
    texts, hints = collate_all(196, "humanml", "button_press")
    texts, hints = texts[: args.count], hints[: args.count]
    _, model_kwargs = collate(
        [
            dict(inp=torch.zeros(196), tokens=None, lengths=196, text=text, hint=hint)
            for text, hint in zip(texts, hints)
        ]
    )
    model = cmdm.CMDM(**model_args)
    diffusion = create_gaussian_diffusion(config)
    load_model_wo_clip(model, torch.load(str(checkpoint), map_location="cpu"))
    model = ClassifierFreeSampleModel(model).to(device).eval()
    for name, value in model_kwargs["y"].items():
        if torch.is_tensor(value):
            model_kwargs["y"][name] = value.to(device)
    model_kwargs["y"]["scale"] = torch.ones(args.count, device=device) * 2.5
    start = time.monotonic()
    sample = diffusion.p_sample_loop(
        model,
        (args.count, 263, 1, 196),
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )
    sample = sample[:, :263].cpu().permute(0, 2, 3, 1)
    sample = (sample * std + mean).float()
    sample = recover_from_ric(sample, 22)
    sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
    motion = model.rot2xyz(x=sample, pose_rep="xyz").detach().numpy()
    raw_mean, raw_std = arrays[2:]
    raw_hint = np.asarray(hints).reshape(args.count, 196, 22, 3)
    mask = raw_hint.sum(-1) != 0
    raw_hint = (raw_hint * raw_std.reshape(22, 3) + raw_mean.reshape(22, 3)) * mask[
        ..., None
    ]
    if motion.shape != (args.count, 22, 3, 196) or not np.isfinite(motion).all():
        raise ValueError("Invalid generated XYZ motion")
    np.save(
        str(output / "results.npy"),
        dict(
            motion=motion,
            text=texts,
            lengths=np.full(args.count, 196),
            hint=raw_hint,
            num_samples=args.count,
            num_repetitions=1,
        ),
    )
    (output / "generation.json").write_text(
        json.dumps(
            dict(
                status="complete",
                seed=args.seed,
                count=args.count,
                source_fps=20,
                model_sha256=digest(checkpoint),
                normalization_sha256={str(p): digest(p) for p in norms},
                sampler="upstream OmniControl p_sample_loop; 1000 diffusion steps; CFG 2.5",
                task="button_press",
                smpl_used=False,
                input_dataset_used=False,
                seconds=time.monotonic() - start,
                generator_sha256=digest(SCRIPT),
            ),
            indent=2,
        )
        + "\n"
    )
    print("BUTTON_HUMAN_MOTIONS_COMPLETE", str(output), flush=True)


if __name__ == "__main__":
    main()
