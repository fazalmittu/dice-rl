"""
DICE-RL model variant for SIA DiT-based pretrained policies.

Extends DistillResidualRLImgModel to load a SIA DiT checkpoint
instead of the default hydra-based flow matching model.

Everything else (critic ensemble, residual actor, losses, exploration)
is inherited unchanged from the parent classes.
"""

import logging
import os
from typing import Optional

import torch

from model.rl.distill_residual_rl_img import DistillResidualRLImgModel
from model.rl.sia_policy_wrapper import SIAPolicyWrapper

log = logging.getLogger(__name__)


class DistillResidualRLSiaModel(DistillResidualRLImgModel):
    """
    Image-based distilled RL model using a SIA DiT as the frozen BC prior.

    The only override is ``_load_pretrained_policy`` which creates a
    ``SIAPolicyWrapper`` instead of hydra-instantiating a flow matching model.

    All other behaviour (get_action, actor_loss, critic_loss, exploration)
    is inherited from DistillResidualRLImgModel / DistillResidualRLModel.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        pretrained_flow_policy_path: str,
        # SIA-specific configuration
        task_prompt: str = "lift the cube",
        diffusion_steps: int = 5,
        sia_root: str = None,
        # All other parameters forwarded to parent
        **kwargs,
    ):
        self._task_prompt = task_prompt
        self._diffusion_steps = diffusion_steps
        self._sia_root = sia_root

        # Parent __init__ will call _load_pretrained_policy internally
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            pretrained_flow_policy_path=pretrained_flow_policy_path,
            **kwargs,
        )

        log.info(
            f"DistillResidualRLSiaModel initialized: obs_dim={obs_dim}, "
            f"action_dim={action_dim}, diffusion_steps={diffusion_steps}"
        )

    def _load_pretrained_policy(self, checkpoint_path: str, device: str):
        """
        Load a SIA DiT policy instead of a hydra-based flow matching model.

        Returns a frozen ``SIAPolicyWrapper`` that exposes:
        - ``forward_from_features(features, init_noise)``
        - ``network.extract_visual_features(cond)``
        """
        wrapper = SIAPolicyWrapper(
            checkpoint_path=checkpoint_path,
            device=device,
            task_prompt=self._task_prompt,
            diffusion_steps=self._diffusion_steps,
            sia_root=self._sia_root,
        )

        log.info("SIA DiT policy loaded and frozen as pretrained prior")
        return wrapper
