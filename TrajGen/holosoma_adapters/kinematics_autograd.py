"""Preserve ordinary FK autograd used by the original reference optimizers."""
import inspect

import torch
from pytorch_kinematics import transforms


def autograd_forward_kinematics(chain):
    """Bind the chain's public FK interface with its original backward behavior.

    Newer pytorch-kinematics defaults to an analytical backward. It is close
    numerically, but changes the results of long Adam reference refinements.
    Keep ordinary autograd without patching the dependency or chain instance.
    """
    if "analytical_grad" not in inspect.signature(chain.forward_kinematics_tensor).parameters:
        return chain.forward_kinematics

    def forward(th, frame_indices=None):
        if frame_indices is None:
            frame_indices = chain.get_all_frame_indices()
        th = torch.atleast_2d(chain.ensure_tensor(th))
        matrices = chain.forward_kinematics_tensor(th, analytical_grad=False)
        return {chain.idx_to_frame[fi.item()]: transforms.Transform3d(matrix=matrices[fi])
                for fi in frame_indices}

    return forward
