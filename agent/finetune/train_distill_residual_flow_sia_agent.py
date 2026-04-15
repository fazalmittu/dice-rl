"""
Training agent for DICE-RL finetuning of SIA DiT-based BC policies.

Extends TrainDistillResidualFlowImgAgent with:
- CLIP-based visual feature extraction (replacing ResNet spatial softmax)
- Automatic normalization file generation from SIA checkpoint norm_stats
- SIA-specific expert dataset preprocessing
- Proper obs_dim augmentation (state_dim + num_cameras * clip_dim)
"""

import os
import tempfile
import numpy as np
import torch
import logging

from omegaconf import OmegaConf

log = logging.getLogger(__name__)

from agent.finetune.train_distill_residual_flow_img_agent import (
    TrainDistillResidualFlowImgAgent,
)


class TrainDistillResidualFlowSiaAgent(TrainDistillResidualFlowImgAgent):
    """
    DICE-RL training agent for SIA DiT pretrained policies.

    Differences from the ResNet-based image agent:
    1. Visual features come from CLIP ViT-B/16 (512D per camera) instead of
       ResNet-18 spatial softmax (128D total).
    2. Normalization uses z-score stats from the SIA checkpoint, automatically
       converted to the min-max format that RobomimicImageWrapper expects.
    3. obs_dim = state_dim + num_cameras * clip_dim (e.g. 9 + 2*512 = 1033).
    """

    def __init__(self, cfg):
        # ── Load SIA checkpoint early to get dimensions & norm stats ──────
        self.original_obs_dim = cfg.obs_dim  # typically 9
        sia_ckpt_path = cfg.base_policy_path
        sia_ckpt = torch.load(sia_ckpt_path, map_location="cpu", weights_only=False)
        sia_config = sia_ckpt.get("config", {})
        self.sia_norm_stats = sia_ckpt.get("norm_stats", None)

        # Determine visual feature dimensions from the SIA model config
        num_cameras = len(sia_config.get("camera_keys", ["agentview", "robot0_eye_in_hand"]))
        # CLIP ViT-B/16 output dim. The SIA model discovers this at runtime,
        # but it's always 512 for ViT-B/16. We store it and verify later.
        clip_dim = 512
        self.visual_feature_dim = num_cameras * clip_dim

        # Update obs_dim BEFORE parent init (parent uses it to create model & buffer)
        self.obs_dim = self.original_obs_dim + self.visual_feature_dim
        cfg.obs_dim = self.obs_dim

        log.info(f"SIA agent: original_obs_dim={self.original_obs_dim}, "
                 f"visual_feature_dim={self.visual_feature_dim}, "
                 f"augmented obs_dim={self.obs_dim}")

        # ── Generate normalization file from SIA z-score stats ────────────
        # The mathematical trick: setting min=mean-std, max=mean+std makes
        # the wrapper's min-max formula equivalent to z-score normalization.
        if self.sia_norm_stats is not None:
            self._norm_dir = tempfile.mkdtemp(prefix="dice_sia_norm_")
            norm_path = os.path.join(self._norm_dir, "normalization.npz")
            self._generate_normalization_file(norm_path)

            # Point the wrapper config at our generated file
            if OmegaConf.is_missing(cfg, "normalization_path") or cfg.normalization_path is None:
                cfg.normalization_path = norm_path
            else:
                cfg.normalization_path = norm_path

            # Also update the nested wrapper config
            if hasattr(cfg, "env") and hasattr(cfg.env, "wrappers"):
                wrappers = cfg.env.wrappers
                if hasattr(wrappers, "robomimic_image"):
                    wrappers.robomimic_image.normalization_path = norm_path

            log.info(f"Generated normalization file at: {norm_path}")
        else:
            log.warning("No norm_stats in SIA checkpoint — running without normalization")

        # ── Skip the img agent's __init__ (call grandparent directly) ─────
        # TrainDistillResidualFlowImgAgent.__init__ probes a .hydra/config.yaml
        # from the BC checkpoint directory to extract visual_feature_dim from
        # the ResNet-based pretrained model. SIA checkpoints don't have this
        # hydra config. We already computed visual_feature_dim and updated
        # cfg.obs_dim above, so we bypass the img agent's init and call the
        # grandparent (TrainDistillResidualFlowAgent) directly.
        self._feature_cache = {}

        from agent.finetune.train_distill_residual_flow_agent import (
            TrainDistillResidualFlowAgent,
        )
        TrainDistillResidualFlowAgent.__init__(self, cfg)

        # Store reference to the pretrained SIA policy for feature extraction
        self.pretrained_model = self.model.pretrained_flow_policy
        self.pretrained_model.eval()

        # Exploration strategy settings (normally set by parent)
        self.online_explore_strategy = cfg.get("online_explore_strategy", "standard")
        self.evaluate_strategy = cfg.get("evaluate_strategy", "standard")
        self.num_exploration_samples = cfg.get("num_exploration_samples", 10)
        self.current_training_step = 0

        # Verify clip_dim matches what the model actually loaded
        actual_clip_dim = self.pretrained_model.clip_dim
        if actual_clip_dim != clip_dim:
            log.warning(
                f"CLIP dim mismatch: expected {clip_dim}, got {actual_clip_dim}. "
                f"Updating visual_feature_dim."
            )
            self.visual_feature_dim = num_cameras * actual_clip_dim
            self.obs_dim = self.original_obs_dim + self.visual_feature_dim

        log.info(f"TrainDistillResidualFlowSiaAgent initialized successfully")
        log.info(f"  Model obs_dim: {self.model.obs_dim}")
        log.info(f"  Visual features: {num_cameras} cameras × {actual_clip_dim}D CLIP = {self.visual_feature_dim}D")

    # ── Normalization helpers ─────────────────────────────────────────────

    def _generate_normalization_file(self, path: str):
        """Create a normalization.npz in dice-rl's min-max format from SIA z-score stats."""
        def _to_np(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy().astype(np.float64)
            return np.asarray(x, dtype=np.float64)

        s_mean = _to_np(self.sia_norm_stats["state"]["mean"])
        s_std = _to_np(self.sia_norm_stats["state"]["std"])
        a_mean = _to_np(self.sia_norm_stats["action"]["mean"])
        a_std = _to_np(self.sia_norm_stats["action"]["std"])

        np.savez(
            path,
            obs_min=s_mean - s_std,
            obs_max=s_mean + s_std,
            action_min=a_mean - a_std,
            action_max=a_mean + a_std,
        )

    # ── Visual feature extraction (overrides ResNet version) ──────────────

    def _extract_visual_features_from_obs(self, obs_venv):
        """
        Extract CLIP visual features from raw observations.

        Uses the SIA policy wrapper's extract_visual_features method.
        Returns augmented state: [normalized_state || clip_features].

        Args:
            obs_venv: Dict with 'state' and 'rgb' keys.

        Returns:
            augmented_state: (n_envs, cond_steps, augmented_obs_dim) tensor.
        """
        state = obs_venv["state"]
        rgb = obs_venv["rgb"]

        # Convert to tensors
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float().to(self.device)
        if not isinstance(rgb, torch.Tensor):
            rgb = torch.from_numpy(rgb).to(self.device)

        # Ensure cond_steps dimension
        if state.ndim == 2:
            state = state.unsqueeze(1)
        if rgb.ndim == 4:
            rgb = rgb.unsqueeze(1)

        # Extract CLIP features via the SIA wrapper
        cond = {"state": state, "rgb": rgb}
        with torch.no_grad():
            visual_features = self.pretrained_model.network.extract_visual_features(cond)
            # (n_envs, num_cameras * clip_dim)

        # Expand visual features to match cond_steps
        T = state.shape[1]
        if T > 1:
            visual_features = visual_features.unsqueeze(1).expand(-1, T, -1)
        else:
            visual_features = visual_features.unsqueeze(1)

        # Concatenate: [state || visual_features]
        augmented_state = torch.cat([state, visual_features], dim=-1)
        return augmented_state

    # ── Expert dataset preprocessing (overrides ResNet version) ───────────

    def _preprocess_expert_dataset(self, expert_dataset):
        """
        Pre-process expert dataset by extracting CLIP features and merging
        them with state observations.

        Args:
            expert_dataset: StitchedSequenceQLearningDataset with image data.

        Returns:
            List of Transition objects with augmented state.
        """
        log.info("Extracting CLIP features for expert dataset...")

        from agent.dataset.sequence import Transition

        pretrained_model = self.model.pretrained_flow_policy
        pretrained_model.eval()

        processed_transitions = []
        batch_size = 32

        for i in range(0, len(expert_dataset), batch_size):
            batch_end = min(i + batch_size, len(expert_dataset))
            batch_transitions = [expert_dataset[j] for j in range(i, batch_end)]

            # Collect states and images from transitions
            states_batch = []
            rgb_batch = []
            next_states_batch = []
            next_rgb_batch = []

            for transition in batch_transitions:
                condition = transition.conditions
                states_batch.append(condition["state"])
                rgb_batch.append(condition["rgb"])
                next_states_batch.append(condition["next_state"])
                next_rgb_batch.append(condition["next_rgb"])

            # Stack into tensors
            states_tensor = torch.stack(
                [torch.from_numpy(s) if isinstance(s, np.ndarray) else s for s in states_batch]
            ).float().to(self.device)
            rgb_tensor = torch.stack(
                [torch.from_numpy(r) if isinstance(r, np.ndarray) else r for r in rgb_batch]
            ).float().to(self.device)
            next_states_tensor = torch.stack(
                [torch.from_numpy(s) if isinstance(s, np.ndarray) else s for s in next_states_batch]
            ).float().to(self.device)
            next_rgb_tensor = torch.stack(
                [torch.from_numpy(r) if isinstance(r, np.ndarray) else r for r in next_rgb_batch]
            ).float().to(self.device)

            # Extract CLIP features
            with torch.no_grad():
                cond = {"state": states_tensor, "rgb": rgb_tensor}
                features = pretrained_model.network.extract_visual_features(cond)

                next_cond = {"state": next_states_tensor, "rgb": next_rgb_tensor}
                next_features = pretrained_model.network.extract_visual_features(next_cond)

            # Build augmented transitions
            for j, transition in enumerate(batch_transitions):
                original_state = transition.conditions["state"]
                original_next_state = transition.conditions.get("next_state", original_state)

                visual_feat = features[j]  # (visual_feature_dim,)
                next_visual_feat = next_features[j]

                augmented_state = torch.cat(
                    [original_state, visual_feat.unsqueeze(0).expand(self.cond_steps, -1)],
                    dim=-1,
                )
                augmented_next_state = torch.cat(
                    [original_next_state, next_visual_feat.unsqueeze(0).expand(self.cond_steps, -1)],
                    dim=-1,
                )

                processed_transitions.append(
                    Transition(
                        actions=transition.actions,
                        conditions={
                            "state": augmented_state,
                            "next_state": augmented_next_state,
                        },
                        rewards=transition.rewards,
                        dones=transition.dones,
                        mc_return=(
                            transition.mc_return
                            if hasattr(transition, "mc_return")
                            else transition.rewards
                        ),
                    )
                )

            if (i + batch_size) % (batch_size * 50) == 0:
                log.info(
                    f"Processed {min(i + batch_size, len(expert_dataset))}"
                    f"/{len(expert_dataset)} expert transitions"
                )

        log.info(f"Expert dataset preprocessing complete: {len(processed_transitions)} transitions")
        return processed_transitions

    # ── Evaluation with the raw SIA model ─────────────────────────────────

    def _get_flow_action_for_eval(self, obs):
        """
        Evaluate pretrained SIA policy on raw observations (without residual).

        This uses the full CLIP→DiT pipeline, matching the BC training setup
        exactly so we can compare pretrained vs finetuned performance.
        """
        with torch.no_grad():
            # Extract features and create augmented state
            augmented_state = self._extract_visual_features_from_obs(obs)

            # Generate noise and run through the frozen BC policy only
            B = augmented_state.shape[0]
            noise = torch.randn(
                B, self.horizon_steps, self.action_dim, device=self.device
            )
            output = self.pretrained_model.forward_from_features(
                features=augmented_state, init_noise=noise
            )
            action_venv = output.trajectories.cpu().numpy()
            return action_venv[:, : self.act_steps]
