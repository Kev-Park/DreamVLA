"""Select a local policy or an explicitly identified inference service."""
from __future__ import annotations

import atexit


def load_gr00t_policy(*, embodiment_tag, model_path, device, service_descriptor=None):
    if service_descriptor is None:
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        return Gr00tPolicy(embodiment_tag=embodiment_tag, model_path=model_path, device=device)
    from .gr00t_service import PolicyServiceClient

    client = PolicyServiceClient(service_descriptor, model_path, device, embodiment_tag)
    atexit.register(client.close)
    return client
