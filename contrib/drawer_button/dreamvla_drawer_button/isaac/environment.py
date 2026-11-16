"""Environment contract for the contributed ten-second drawer/button protocol."""

from isaaclab.envs import ManagerBasedRLEnv


class DrawerButtonRLEnv(ManagerBasedRLEnv):
    """Retain 500 controls even when the reference includes a longer tail.

    Izat's native recorder intentionally omits the auto-reset boundary and saves
    499 observations. This subclass keeps training, collection and evaluation on
    that declared ten-second contract without changing other registered tasks.
    """

    def _resolve_reference_episode_length(self, longest_reference_s: float) -> float:
        if self.cfg.episode_length_s != 10.0:
            raise ValueError("Drawer/button protocol requires a ten-second episode")
        return self.cfg.episode_length_s
