"""Use the shared reference optimizer's original FK backward behavior."""
import sys

from .paths import REPOSITORY_ROOT


def autograd_forward_kinematics(chain):
    # Direct ``python -m dreamvla_drawer_button`` adds the contribution directory,
    # not necessarily the backend repository containing the shared reference code.
    root = str(REPOSITORY_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from TrajGen.holosoma_adapters.kinematics_autograd import autograd_forward_kinematics as bind

    return bind(chain)
