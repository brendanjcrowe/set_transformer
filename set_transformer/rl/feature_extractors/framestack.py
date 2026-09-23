"""The no-belief control: the policy sees the last k base observations and nothing else.

Added 2026-09-22 (``change_mds/framestack_arm_2026-09-22.md``; design in
``change_mds/framestack_arm_plan_2026-09-22.md``, section 7). Like
:class:`~set_transformer.rl.feature_extractors.gaussian.WeightedGaussianFeaturesExtractor`
this extractor has no learnable parameters, but unlike every other arm it reads NO belief at
all: ``particles`` and ``weights`` are ignored.

The stacking happens in
:class:`~set_transformer.rl.wrappers.obs_history.ObsHistoryDictWrapper`, not here. An
extractor sees shuffled minibatches from the rollout buffer and has no temporal order to
accumulate over.
"""

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FrameStackFeaturesExtractor(BaseFeaturesExtractor):
    """Pass the (already stacked) base observation through; ignore the belief.

    ``n_stack`` is a constructor argument PURELY so it is recorded in the saved zip's
    ``policy_kwargs`` (``train.py`` builds them from ``Encoder.extractor_kwargs``). That is the
    only place the value travels INSIDE the zip, so it is what the standalone eval and the
    policy collector read back to rebuild the same env (plan section 6c), and what the width
    check below can be made against.
    """

    def __init__(self, observation_space: gym.spaces.Dict, n_stack: int = 1):
        width = int(observation_space["obs"].shape[0])
        n_stack = int(n_stack)
        if n_stack < 1 or width % n_stack:
            raise ValueError(
                f"obs width {width} is not a multiple of n_stack={n_stack}: this checkpoint "
                f"was trained against a differently stacked env.")
        super().__init__(observation_space, features_dim=width)
        self.n_stack = n_stack
        self._geometry = dict(encoder="framestack", n_stack=self.n_stack,
                              frame_dim=width // n_stack)

    def forward(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        # A fresh tensor, like every other arm's torch.cat: returning obs_dict["obs"] would
        # hand the policy a view into the rollout buffer's batch.
        return obs_dict["obs"].clone()
