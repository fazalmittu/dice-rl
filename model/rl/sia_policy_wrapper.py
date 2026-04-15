"""
Wrapper around SIA's DiTPolicy for use as the frozen BC prior in DICE-RL.

Implements the interfaces expected by DistillResidualRLImgModel:
- forward_from_features(features, init_noise) -> object with .trajectories
- network.extract_visual_features(cond) -> (B, visual_feature_dim)

The wrapper handles:
- Loading the SIA DiT checkpoint and creating the DiTPolicy
- CLIP feature extraction from raw images
- Image preprocessing (resize, normalize) for CLIP
- Task embedding caching
- Routing pre-extracted features through the DiT's conditioning pipeline
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from types import SimpleNamespace
from typing import Optional

log = logging.getLogger(__name__)

# Add sia to path so we can import DiTPolicy
SIA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "sia")
if os.path.exists(SIA_ROOT):
    SIA_ROOT = os.path.normpath(SIA_ROOT)
    if SIA_ROOT not in sys.path:
        sys.path.insert(0, SIA_ROOT)


class SIAPolicyWrapper(nn.Module):
    """
    Wraps SIA's DiTPolicy for integration with DICE-RL's residual RL framework.

    This wrapper:
    1. Loads the SIA DiT model from a BC checkpoint
    2. Provides CLIP feature extraction via extract_visual_features()
    3. Provides forward_from_features() that routes pre-extracted features
       through the DiT without re-running CLIP
    4. Caches the task embedding (computed once at init)
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        task_prompt: str = "lift the cube",
        diffusion_steps: int = 5,
        sia_root: str = None,
    ):
        super().__init__()

        self.device_str = device
        self.diffusion_steps = diffusion_steps
        self.task_prompt = task_prompt

        # Resolve sia root for imports
        if sia_root is not None:
            resolved = os.path.normpath(sia_root)
            if resolved not in sys.path:
                sys.path.insert(0, resolved)

        from models.lbm import DiTPolicy
        from models.transforms import resize_with_pad_torch

        self._resize_fn = resize_with_pad_torch

        # ── Load checkpoint ──────────────────────────────────────────────
        log.info(f"Loading SIA DiT checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        config = ckpt.get("config", {})
        self.norm_stats = ckpt.get("norm_stats", None)
        self._config = config

        # Core dimensions
        self.state_dim = config.get("state_dim", 9)
        self.action_dim = config.get("action_dim", 7)
        self.chunk_length = config.get("chunk_length", 4)
        self.img_emb_dim = config.get("img_emb_dim", 512)
        self.camera_keys = config.get("camera_keys", ["agentview", "robot0_eye_in_hand"])
        self.num_cameras = len(self.camera_keys)

        # CLIP embed dim is what the CLIP model outputs (may differ from img_emb_dim
        # if img_emb_dim was a user-facing config that gets overridden by the actual
        # CLIP output dim). We'll discover the true dim after model creation.
        self.clip_dim: int = None  # set after model is built

        # Visual feature dim exposed to dice-rl (CLIP features flattened across cameras)
        # Will be set after we know clip_dim
        self._visual_feature_dim: int = None

        # ── Create DiTPolicy ─────────────────────────────────────────────
        self.model = DiTPolicy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            chunk_length=self.chunk_length,
            img_emb_dim=self.img_emb_dim,
            hidden_size=config.get("hidden_size", 384),
            depth=config.get("depth", 12),
            num_heads=config.get("num_heads", 6),
            camera_keys=list(self.camera_keys),
            task_encoder=config.get("task_encoder", "clip"),
            num_diffusion_timesteps=config.get("num_diffusion_timesteps", 1000),
            post_cond_layer_norm=config.get("post_cond_layer_norm", False),
            pre_vizemb_norm=config.get("pre_vizemb_norm", False),
            normalize_clip_for_cond=config.get("normalize_clip_for_cond", False),
            pre_img_layer_norm=config.get("pre_img_layer_norm", False),
            device=device,
        )

        # ── Load weights ─────────────────────────────────────────────────
        state_dict = ckpt.get("model", ckpt.get("model_state_dict", None))
        if state_dict is None:
            raise ValueError("Checkpoint has no 'model' or 'model_state_dict' key")

        # Handle EMA state dict keys
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k.replace("ema_model.", "")] = v
        self.model.load_state_dict(cleaned, strict=False)
        self.model.to(device)
        self.model.eval()

        # Discover actual CLIP embed dim from the loaded model
        self.clip_dim = self.model.clip_embed_dim
        self._visual_feature_dim = self.num_cameras * self.clip_dim
        log.info(
            f"SIA DiT loaded: state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"chunk_length={self.chunk_length}, clip_dim={self.clip_dim}, "
            f"num_cameras={self.num_cameras}, visual_feature_dim={self._visual_feature_dim}"
        )

        # ── Freeze everything ────────────────────────────────────────────
        for p in self.model.parameters():
            p.requires_grad = False

        # ── Cache task embedding ─────────────────────────────────────────
        with torch.no_grad():
            self._task_vec_h = self.model.encode_task_to_hidden(
                [task_prompt], device=device
            )  # (1, hidden_size)
        log.info(f"Task embedding cached for prompt: '{task_prompt}'")

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def visual_feature_dim(self) -> int:
        """Total dimension of visual features (num_cameras * clip_dim)."""
        return self._visual_feature_dim

    @property
    def network(self):
        """Compatibility shim: dice-rl accesses pretrained_model.network."""
        return self

    # ── CLIP feature extraction ───────────────────────────────────────────

    def extract_visual_features(self, cond: dict) -> torch.Tensor:
        """
        Extract CLIP visual features from raw images.

        This follows the dice-rl convention where the training agent calls
        ``pretrained_model.network.extract_visual_features(cond)``
        during online collection and expert-data preprocessing.

        Args:
            cond: dict with key ``'rgb'`` containing raw images.
                  Shape: ``(B, [cond_steps,] H, W, C*num_cameras)`` uint8 or
                         ``(B, [cond_steps,] C*num_cameras, H, W)`` float.
                  The ``'state'`` key is ignored here.

        Returns:
            features: ``(B, num_cameras * clip_dim)`` float tensor.
        """
        rgb = cond["rgb"]

        # Collapse cond_steps if present: (B, T, ...) -> (B, ...)
        if rgb.dim() == 5:
            rgb = rgb[:, -1]  # take last timestep

        # Ensure tensor
        if isinstance(rgb, np.ndarray):
            rgb = torch.from_numpy(rgb)
        rgb = rgb.to(self.device_str)

        # Determine format and split per camera
        # dice-rl stacks cameras along the channel dim:
        #   HWC format: (B, H, W, 3*num_cameras)  uint8
        #   CHW format: (B, 3*num_cameras, H, W)   float
        if rgb.dim() == 4 and rgb.shape[-1] == 3 * self.num_cameras:
            # HWC format (from RobomimicImageWrapper)
            # Split along last dim and convert to CHW float
            per_cam = torch.chunk(rgb.float() / 255.0, self.num_cameras, dim=-1)
            per_cam = [img.permute(0, 3, 1, 2) for img in per_cam]  # -> (B, 3, H, W)
        elif rgb.dim() == 4 and rgb.shape[1] == 3 * self.num_cameras:
            # CHW format
            if rgb.dtype == torch.uint8:
                rgb = rgb.float() / 255.0
            per_cam = torch.chunk(rgb, self.num_cameras, dim=1)  # each (B, 3, H, W)
        else:
            raise ValueError(
                f"Unexpected rgb shape {rgb.shape} for {self.num_cameras} cameras"
            )

        # Preprocess each camera for CLIP: resize to 224×224, normalize
        images_dict = {}
        for i, cam_name in enumerate(self.camera_keys):
            img = per_cam[i]  # (B, 3, H, W) float in [0, 1]
            img = self._resize_fn(img, 224, 224)  # letterbox resize
            # CLIP normalization (registered buffers on the model)
            img = (img - self.model.clip_mean.to(img.device)) / self.model.clip_std.to(img.device)
            images_dict[cam_name] = img

        # Run CLIP encoder
        with torch.no_grad():
            clip_embs = self.model.encode_images_to_clip(images_dict)
            # (B, num_cameras, clip_dim)

        # Flatten cameras: (B, num_cameras * clip_dim)
        B = clip_embs.shape[0]
        return clip_embs.reshape(B, -1).float()

    # ── Forward from pre-extracted features ───────────────────────────────

    def forward_from_features(
        self,
        features: torch.Tensor,
        init_noise: torch.Tensor,
    ):
        """
        Run the DiT denoising from pre-extracted features (no CLIP or raw images).

        This is the interface ``DistillResidualRLImgModel.get_action`` calls on
        the frozen pretrained policy.

        Args:
            features: ``(B, cond_steps, augmented_obs_dim)`` where
                      ``augmented_obs_dim = state_dim + visual_feature_dim``.
                      The state portion is **already normalized** by the env wrapper.
            init_noise: ``(B, horizon_steps, action_dim)`` Gaussian noise to denoise from.

        Returns:
            ``SimpleNamespace(trajectories=actions)`` where
            ``actions`` has shape ``(B, horizon_steps, action_dim)``, in normalized
            action space (matching the env wrapper's normalization).
        """
        B = features.shape[0]
        model_dtype = self.model.x_embedder.weight.dtype

        # Collapse cond_steps: (B, T, D) -> (B, D)
        if features.dim() == 3:
            features = features[:, -1, :]

        features = features.to(self.device_str)
        init_noise = init_noise.to(self.device_str)

        # ── Unpack augmented state ────────────────────────────────────────
        # Layout: [state(state_dim) | clip_cam1(clip_dim) | ... | clip_camN(clip_dim)]
        state = features[:, : self.state_dim]  # (B, state_dim) — already normalized
        clip_flat = features[:, self.state_dim :]  # (B, num_cameras * clip_dim)
        clip_embs = clip_flat.reshape(B, self.num_cameras, self.clip_dim)

        # ── Build static conditioning manually (bypass model.compute_static_condition) ──
        state = state.to(model_dtype)
        clip_embs = clip_embs.to(model_dtype)

        # State embedding
        st_emb = self.model.x_embedder(state)  # (B, hidden_size)
        if st_emb.dim() == 3:
            st_vec = st_emb.squeeze(1)
        else:
            st_vec = st_emb
        st_vec = st_vec.to(model_dtype)

        # Image projection (skip CLIP encoding, feed pre-extracted embeddings)
        vision_emb = self.model.img_proj(clip_embs.to(self.model.img_dtype))
        img_vec = vision_emb.reshape(B, -1).to(model_dtype)

        # Task embedding (cached)
        task_vec = self._task_vec_h.expand(B, -1).to(
            device=st_vec.device, dtype=model_dtype
        )

        # Optional CLIP-for-cond normalization (matches training if it was enabled)
        if self.model.normalize_clip_for_cond:
            img_vec = F.normalize(img_vec, dim=-1, eps=1e-4)
            task_vec = F.normalize(task_vec, dim=-1, eps=1e-4)

        cond_static = torch.cat([st_vec, img_vec, task_vec], dim=-1)

        # ── Denoising loop ────────────────────────────────────────────────
        x_t = init_noise.to(model_dtype)
        dt = -1.0 / self.diffusion_steps
        t = torch.ones(B, device=features.device, dtype=model_dtype)

        for _ in range(self.diffusion_steps):
            # batch=None is safe here because cond_static is always provided;
            # denoise_step only accesses batch when cond_static is None.
            x_t = self.model.denoise_step(
                batch=None, x_t=x_t, t=t, dt=dt,
                cond_static=cond_static, tau=1.0,
            )
            t = t + dt

        return SimpleNamespace(trajectories=x_t.float())

    # ── Utilities ─────────────────────────────────────────────────────────

    def get_normalization_arrays(self):
        """
        Convert SIA z-score norm_stats into the min-max arrays that
        ``RobomimicImageWrapper`` expects.

        The mathematical equivalence:
            z-score: x_norm = (x - mean) / std
            min-max with min=mean-std, max=mean+std:
                x_norm = 2*((x - min) / (max - min) - 0.5) = (x - mean) / std

        Returns:
            dict with keys obs_min, obs_max, action_min, action_max (numpy arrays),
            or None if norm_stats are unavailable.
        """
        if self.norm_stats is None:
            return None

        def _to_np(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy()
            return np.asarray(x, dtype=np.float64)

        s_mean = _to_np(self.norm_stats["state"]["mean"])
        s_std = _to_np(self.norm_stats["state"]["std"])
        a_mean = _to_np(self.norm_stats["action"]["mean"])
        a_std = _to_np(self.norm_stats["action"]["std"])

        return {
            "obs_min": s_mean - s_std,
            "obs_max": s_mean + s_std,
            "action_min": a_mean - a_std,
            "action_max": a_mean + a_std,
        }
