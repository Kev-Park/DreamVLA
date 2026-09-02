"""Load a native SONIC PyTorch checkpoint (groot.rl-era) as plain torch MLPs.

Why this exists
---------------
The v1.0 release ships ONNX (``model_encoder.onnx`` / ``model_decoder.onnx``) which we
run through onnx2torch. Newer checkpoints (e.g. ``sonic_release_3pt_heading_wrist_81``)
ship only a training ``.pt``. Rather than depend on the ``groot.rl`` package to
re-export ONNX, we rebuild the two MLPs we need directly from the state_dict — the
architecture is a plain SiLU MLP stack, fully determined by the weight shapes:

    encoders.g1        640 -> 2048 -> 1024 -> 512 -> 512 -> 64     (64 = 2 tokens x 32 FSQ)
    decoders.g1_dyn    994 -> 4096 -> 4096 -> 2048 -> 2048
                           -> 1024 -> 1024 -> 512 -> 512 -> 29

Loading the checkpoint requires unpickling trainer metadata that references HuggingFace
TRL (``trl.experimental.ppo.ppo_trainer.OnlineTrainerState``). We do NOT install trl —
a throwaway meta-path stub satisfies the unpickler, and the tensors we want are plain.

Encoder input layout (g1 mode), reproduced from gear_sonic source:
    command_multi_future = cat([joint_pos_multi_future(290), joint_vel_multi_future(290)])
    -> reshape (N, 10, 58)                       [nonflat; the temporal split is nominal]
    motion_anchor_ori_heading_mf -> reshape (N, 10, 6)
    encoder_in = cat([cmd_nonflat, anchor_nonflat], dim=-1).view(N, 640)

NOTE the anchor is HEADING-normalized (yaw-only canonicalization) for these checkpoints,
unlike the v1.0 release's body-frame anchor. See commands.py::root_rot_dif_heading_multi_future.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
import types

import torch
import torch.nn as nn

ENCODER_G1_IN_DIM = 640
DECODER_G1_IN_DIM = 994
TOKEN_DIM = 64


def _install_trl_stub() -> None:
    """Satisfy the unpickler for trainer metadata without installing HuggingFace trl."""
    if "trl" in sys.modules:
        return

    def _getattr(name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return type(name, (object,), {"__setstate__": lambda self, s: None,
                                      "__init__": lambda self, *a, **k: None})

    class _Loader(importlib.abc.Loader):
        def create_module(self, spec):
            m = types.ModuleType(spec.name)
            m.__file__ = "<trl-stub>"
            m.__path__ = []
            m.__getattr__ = _getattr
            return m

        def exec_module(self, module):
            pass

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == "trl" or fullname.startswith("trl."):
                return importlib.machinery.ModuleSpec(fullname, _Loader(), is_package=True)
            return None

    sys.meta_path.insert(0, _Finder())


def _mlp_from_state_dict(sd: dict, prefix: str, device) -> nn.Sequential:
    """Rebuild an nn.Sequential(Linear, SiLU, ...) from ``<prefix>.module.<i>.{weight,bias}``."""
    idxs = sorted({int(k.split(".")[len(prefix.split(".")) + 1])
                   for k in sd if k.startswith(prefix + ".module.") and k.endswith(".weight")})
    layers: list[nn.Module] = []
    for n, i in enumerate(idxs):
        w = sd[f"{prefix}.module.{i}.weight"]
        b = sd[f"{prefix}.module.{i}.bias"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        with torch.no_grad():
            lin.weight.copy_(w)
            lin.bias.copy_(b)
        layers.append(lin)
        if n < len(idxs) - 1:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers).to(device).eval()


def load_sonic_pt(ckpt_dir: str, device) -> tuple[nn.Sequential, nn.Sequential]:
    """Return (encoder_g1, decoder_g1_dyn) as frozen torch modules.

    ``ckpt_dir`` may be the checkpoint directory or the .pt file itself.
    """
    path = ckpt_dir
    if os.path.isdir(path):
        pts = [f for f in os.listdir(path) if f.endswith(".pt")]
        if not pts:
            raise FileNotFoundError(f"no .pt in {path}")
        path = os.path.join(path, sorted(pts)[-1])

    _install_trl_stub()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["policy_state_dict"]

    enc = _mlp_from_state_dict(sd, "actor_module.encoders.g1", device)
    dec = _mlp_from_state_dict(sd, "actor_module.decoders.g1_dyn", device)

    in_e = enc[0].in_features
    in_d = dec[0].in_features
    out_e = enc[-1].out_features
    out_d = dec[-1].out_features
    if (in_e, out_e) != (ENCODER_G1_IN_DIM, TOKEN_DIM):
        raise ValueError(f"g1 encoder shape {in_e}->{out_e}, expected {ENCODER_G1_IN_DIM}->{TOKEN_DIM}")
    if in_d != DECODER_G1_IN_DIM:
        raise ValueError(f"g1_dyn decoder in {in_d}, expected {DECODER_G1_IN_DIM}")
    for p in list(enc.parameters()) + list(dec.parameters()):
        p.requires_grad_(False)
    print(f"[sonic-pt] loaded {os.path.basename(path)}: "
          f"encoder {in_e}->{out_e}, decoder {in_d}->{out_d}")
    return enc, dec
