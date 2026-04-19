# SIA DiT → DICE-RL Integration: Handoff Document

This document contains everything a new engineer needs to understand, run, and extend the integration between the **SIA** DiT-based behavior cloning (BC) policy and the **DICE-RL** residual reinforcement learning framework. Read this end-to-end before touching code.

---

## 1. Project Context

### 1.1 High-level goal

We are researchers investigating what makes DICE-RL effective as a residual RL finetuning method for BC policies. We want to validate DICE-RL by using it to finetune a **custom DiT-based BC policy** (from the SIA codebase) rather than the simpler flow-matching MLP that the DICE-RL authors used.

The pipeline we built does:

1. Train a BC policy in SIA (DiT architecture with CLIP vision features).
2. Freeze it and use it as the "pretrained prior" inside DICE-RL.
3. Run DICE-RL's residual RL finetuning on top.

### 1.2 The two codebases

Both repos live side-by-side on disk:

```
/Users/fazal/Desktop/dice-rl    # DICE-RL (this repo)
/Users/fazal/Desktop/sia        # SIA repo (read-only reference, imported from)
```

- **dice-rl** is a fork of `real-stanford/dice-rl`. Origin remote points to upstream; the personal fork is at `fazalmittu/dice-rl`. Our work lives on the `sia-integration` branch, which we push to the fork remote named `fazal`.
- **sia** is read-only from our perspective. We import a couple of modules from it (`models.lbm`, `models.transforms`) at runtime via `sys.path` manipulation.

### 1.3 Important non-goals and constraints

- **No GPU access during development.** All code was written but never actually trained or run end-to-end. Expect bugs when you first hit CUDA.
- **No expert dataset downloaded yet.** The expert data path in the config points at a file that doesn't exist locally.
- **No trained SIA BC checkpoint yet.** The user will train one separately; our config expects it at `checkpoints/lbm_bc_lift/best.pt`.

---

## 2. Background You Must Understand

### 2.1 What DICE-RL actually does (the paper: arXiv:2603.10263)

Full title: "From Prior to Pro: Efficient Skill Mastery via Distribution Contractive RL Finetuning" (Sun & Song, Stanford, 2026).

**Core idea**: given a frozen BC policy `π_pre(s, z)` (flow matching or diffusion; deterministic given state `s` and noise `z`), learn a **residual actor** `r_θ(s, z)` whose output is added to the base:

```
a_total = π_pre(s, z) + r_θ(s, z)
```

Both receive the **same noise vector** `z`, so the residual can learn noise-conditional corrections ("when the BC policy makes this particular kind of mistake from this noise draw, fix it this way").

**Key design choices that make DICE-RL special** (vs. plain residual RL or other post-training methods):

1. **Distilled residual architecture**: residual actor is a simple MLP `[1024, 1024, 1024]` with LayerNorm. No iterative denoising in the residual path.
2. **Critic ensemble of 10 networks** with min-Q for conservative estimates (extension of TD3's twin critics).
3. **Q-normalization** in the actor loss (divide by mean |Q| — from FQL).
4. **Multi-sample expectation training** (K=16 noise samples per state in the actor loss, each producing a different base action; loss averaged over samples).
5. **BC-loss filter** (a.k.a. soft Q-filtering): conditionally disable the `||r_θ||²` regularizer when the critic confirms the residual is improving *and* the Q-value isn't suspiciously high compared to MC returns.
6. **RLPD hybrid replay buffer** with adaptive expert ratio (starts 0.5 expert → decays to 0.1 over 10k steps).
7. **Best-of-N action selection during rollouts**: sample N (=16) noise vectors, pick the candidate with the highest min-Q across the ensemble.
8. **Action chunking** (horizon_steps=4 for short tasks, 8 for long). Critic evaluates the full chunk.
9. **N-step returns** (n=3 typically) for better credit assignment.

**Algorithm family**: TD3 (NOT SAC). The paper sometimes calls it "SAC-style" but there's no entropy term; the actor is deterministic given `(s, z)`; gradient flows through Q directly. `actor_update_freq=2` is TD3's delayed update.

**Actor loss** (full form):

```
L_actor(θ) = L_RL(θ) + β * L_BC(θ)

L_RL(θ)   = -(1/K) Σ_k Q(s_t, a_k)        where a_k = π_pre(s_t, z_k) + r_θ(s_t, z_k)
L_BC(θ)   = (1/K) Σ_k (1 - FILTER(s_t, z_k; ε)) * ||r_θ(s_t, z_k)||²
FILTER    = 1[ Q(s, a_cur) > Q(s, a_pre)  ∧  Q(s, a_cur) - Ĝ(s) ≤ ε ]
```

Where `Ĝ(s)` is a Monte Carlo return estimate computed from the replay buffer, `ε` is a small negative threshold (~ -0.25), and `β` is `bc_loss_weight`.

**Critic loss**: standard TD backup with target networks (Polyak averaging, τ=0.01), ensemble minimum for the target Q, n-step returns.

### 2.2 The SIA DiT BC policy

Located at `/Users/fazal/Desktop/sia`. The relevant class is `DiTPolicy` in `models/lbm.py`.

**Architecture** (for `dit_B` preset — the default):
- Backbone: Diffusion Transformer with 12 layers × 384 hidden × 6 heads
- Vision: frozen CLIP ViT-B/16 (outputs 512D per camera)
- Text: frozen CLIP text encoder for task prompts (e.g., "lift the cube")
- Action head: flow matching, 5 denoising steps by default
- Conditioning: adaLN-Zero with concatenated [state, per-camera images, task, timestep] embeddings

**State & action dimensions** (for Lift):
- `state_dim = 9` (eef_pos 3 + eef_quat 4 + gripper_qpos 2)
- `action_dim = 7` (3D pos delta + 3D axis-angle + 1D gripper)
- `chunk_length = 4` (we use 4 — not SIA's default 20 — to match DICE-RL's credit assignment assumptions)

**Training** (handled separately in SIA, not in this repo): supervised flow matching loss on demonstration data. See `sia/scripts/train_lbm_bc.py`.

**Checkpoint format** (what our wrapper consumes):
```python
{
    "model": state_dict,              # DiTPolicy weights
    "optimizer": state_dict,          # (we ignore)
    "norm_stats": {
        "state":  {"mean": [...], "std": [...]},  # z-score stats, 9D
        "action": {"mean": [...], "std": [...]},  # z-score stats, 7D
    },
    "config": {
        "state_dim": 9, "action_dim": 7, "chunk_length": 4,
        "img_emb_dim": 512, "camera_keys": ["agentview", "robot0_eye_in_hand"],
        "task_encoder": "clip", "use_quantile_normalization": False,
        "hidden_size": 384, "depth": 12, "num_heads": 6,
    },
    "step": int,
}
```

### 2.3 Key differences between the two codebases

| Aspect | DICE-RL (vanilla image pipeline) | SIA DiT | What we had to reconcile |
|---|---|---|---|
| BC architecture | Flow matching MLP / UNet | DiT (transformer) | Wrote `SIAPolicyWrapper` that exposes DICE-RL's expected interface |
| Vision encoder | ResNet-18 + spatial softmax (128D total) | CLIP ViT-B/16 (512D per camera) | New CLIP-based `extract_visual_features` method |
| Obs normalization | min-max to [-1, 1] | z-score | Converted SIA z-score stats to equivalent min-max at runtime (see §4.4) |
| Action chunk size | 4 | Normally 20, we use 4 | Train SIA BC with `chunk_length=4` to match |
| Number of cameras (Lift) | 1 | 2 | New env meta JSON with both cameras |
| Obs key naming | "rgb", "state" | "observation.images.{cam}", "observation.state" | We kept DICE-RL's naming; the wrapper handles the mapping internally |

---

## 3. Design Decisions (and why)

### 3.1 Standardize on DICE-RL's wrapper pattern, plug SIA into it

DICE-RL's existing image pipeline already has the exact right pattern: extract vision features in the training agent, stash them alongside proprioceptive state in a flat "augmented state" vector, and have the pretrained policy expose a `forward_from_features(features, init_noise)` method that skips vision encoding. All downstream components (critic, residual actor, replay buffer) see just the augmented state.

We reuse this pattern unchanged. The only things that change vs. DICE-RL's vanilla ResNet+flow-matching pipeline:
- Vision encoder: ResNet → CLIP
- BC architecture: FlowMatchingMLP → DiT
- `obs_dim`: 137 (9 proprio + 128 visual) → 1033 (9 proprio + 2×512 visual)

Everything else — critic, actor, losses, replay buffer, RLPD, exploration, evaluation — is inherited unchanged from the parent classes.

### 3.2 CLIP features cached in the augmented state (big efficiency win)

CLIP ViT-B/16 is ~80% of inference cost. During DICE-RL's 16-sample exploration, we'd naively run CLIP 16 times per observation. Instead, we extract CLIP features **once per observation** in the training agent, store them in the augmented state, and then the SIA wrapper's `forward_from_features` uses the cached features directly (skipping CLIP). Only the lightweight DiT denoising runs 16 times.

Same caching applies during actor-loss multi-z sampling (K=16 different noise vectors per state).

The task embedding is also cached once at `SIAPolicyWrapper.__init__` time, since the text prompt doesn't change during training.

### 3.3 The normalization trick

DICE-RL's `RobomimicImageWrapper` normalizes proprio state with min-max: `obs_norm = 2*((obs - min) / (max - min) - 0.5)`. SIA's DiT expects z-score normalized state: `obs_norm = (obs - mean) / std`.

Mathematical fact: if you set `min = mean - std` and `max = mean + std`, the min-max formula **reduces exactly to z-score normalization**. Same for unnormalization of actions.

So our agent converts SIA's `norm_stats` on-the-fly into a `normalization.npz` file in DICE-RL's expected format. The env wrapper uses this file unchanged. The SIA wrapper's `forward_from_features` therefore receives already-z-score-normalized state (no double normalization, no wrapper modifications needed).

See `TrainDistillResidualFlowSiaAgent._generate_normalization_file`.

### 3.4 Action chunk size decision: 4

SIA's default chunk size is 20 (captures 1 second of motion). DICE-RL's default is 4. We chose 4 because:
- DICE-RL's critic evaluates `Q(s, a_chunk)` — longer chunks are harder to credit-assign
- The replay buffer and n-step returns are designed around small chunks
- 20-step chunks with SAC/TD3 are very difficult to learn from in practice

**The BC policy must be trained in SIA with `chunk_length=4` for this integration to work.** Positional embeddings are baked at the chunk length; you cannot retrofit.

### 3.5 Why we skip the img agent's `__init__`

The parent class `TrainDistillResidualFlowImgAgent.__init__` probes a `.hydra/config.yaml` file in the BC checkpoint directory to extract `visual_feature_dim`. SIA checkpoints don't have this file. Our agent therefore bypasses the img agent's init and calls the grandparent `TrainDistillResidualFlowAgent.__init__` directly, after manually computing `visual_feature_dim` and updating `cfg.obs_dim`. See the agent constructor comments.

---

## 4. Files we created (everything is new — no modifications to upstream code)

### 4.1 `model/rl/sia_policy_wrapper.py` — The SIA policy wrapper

`SIAPolicyWrapper(nn.Module)` — wraps a `DiTPolicy` loaded from a SIA checkpoint. Implements the two methods DICE-RL expects on the frozen pretrained policy:

- **`extract_visual_features(cond)`**
  - Input: `cond` dict with `'rgb'` key (raw images from the env wrapper, HWC stacked across cameras)
  - Preprocesses per camera: splits stacked channels, converts to CHW float [0,1], resize-with-padding to 224×224, applies CLIP ImageNet-style normalization (using `clip_mean`/`clip_std` buffers from the model)
  - Runs `DiTPolicy.encode_images_to_clip` → `(B, num_cameras, clip_dim)`
  - Flattens → `(B, num_cameras * clip_dim)` — shape = `(B, 1024)` for 2 cameras × 512D CLIP
  - Returns a float tensor on `device`

- **`forward_from_features(features, init_noise)`**
  - Input: `features` of shape `(B, cond_steps, augmented_obs_dim)` where the first `state_dim` entries are already normalized proprio and the rest are the flattened CLIP features
  - Unpacks to `(state, clip_embs)` and runs the DiT's static-conditioning pipeline directly (skipping CLIP encoding and the prompt encoder):
    ```
    st_vec  = x_embedder(state)
    img_vec = img_proj(clip_embs).reshape(B, -1)
    task    = self._task_vec_h.expand(B, -1)  # cached
    cond_static = cat([st_vec, img_vec, task], dim=-1)
    ```
  - Loops `diffusion_steps` (default 5) times through `DiTPolicy.denoise_step(batch=None, x_t, t, dt, cond_static=cond_static)`
    - **IMPORTANT**: we pass `batch=None` because `cond_static` is always provided. `denoise_step` only dereferences `batch` when `cond_static is None`.
  - Returns `SimpleNamespace(trajectories=x_t)` — the `.trajectories` attribute is what `DistillResidualRLImgModel.get_action` reads

Other methods on the wrapper:
- `network` property returns `self` (DICE-RL accesses `pretrained_model.network.extract_visual_features`)
- `visual_feature_dim` property returns `num_cameras * clip_dim`
- `get_normalization_arrays()` converts SIA z-score `norm_stats` into `{obs_min, obs_max, action_min, action_max}` arrays — used by the agent to write the normalization.npz

**Import handling**: the wrapper adds `/Users/fazal/Desktop/sia` to `sys.path` (computed relative to its own file location) before importing `from models.lbm import DiTPolicy` and `from models.transforms import resize_with_pad_torch`. The agent/config also expose a `sia_root` kwarg to override this if the sia repo lives elsewhere.

**Freezes everything on load**: sets all params to `requires_grad=False` and calls `.eval()`. The DICE-RL training loop never updates the DiT weights.

### 4.2 `model/rl/distill_residual_rl_sia.py` — Model subclass

Tiny class `DistillResidualRLSiaModel(DistillResidualRLImgModel)` that overrides only one method: `_load_pretrained_policy`. Instead of DICE-RL's default behaviour (load a hydra config from `.hydra/config.yaml`, hydra-instantiate a flow matching model, load weights), it constructs a `SIAPolicyWrapper`.

Accepts three extra constructor kwargs that it plumbs through:
- `task_prompt: str` (default `"lift the cube"`) — used for the cached task embedding
- `diffusion_steps: int` (default `5`) — DiT denoising steps
- `sia_root: str | None` — optional override of the SIA repo path

Everything else (residual actor, critic ensemble, target network, loss functions, exploration) is inherited from `DistillResidualRLImgModel` / `DistillResidualRLModel` and works unchanged.

### 4.3 `agent/finetune/train_distill_residual_flow_sia_agent.py` — Training agent

`TrainDistillResidualFlowSiaAgent(TrainDistillResidualFlowImgAgent)` — overrides the methods that do vision-encoder-specific work:

- **`__init__`**:
  1. Loads the SIA checkpoint from `cfg.base_policy_path` (on CPU, weights_only=False) to peek at its config
  2. Computes `visual_feature_dim = num_cameras * 512` (assumes CLIP ViT-B/16 output dim; verifies after model creation)
  3. Sets `self.obs_dim = original_obs_dim + visual_feature_dim` and overwrites `cfg.obs_dim` in place so downstream components (model, replay buffer) see the augmented dimension
  4. Calls `_generate_normalization_file` to write a temp `.npz` from the checkpoint's `norm_stats` in DICE-RL's format
  5. Patches `cfg.normalization_path` and `cfg.env.wrappers.robomimic_image.normalization_path` to point at the generated file
  6. **Skips** the img agent's `__init__` (see §3.5); calls the grandparent directly
  7. Verifies that the actual CLIP dim matches what we assumed; updates `visual_feature_dim` if not

- **`_extract_visual_features_from_obs(obs_venv)`**: takes an env observation dict with `'state'` and `'rgb'`, passes through `SIAPolicyWrapper.extract_visual_features`, concatenates with state → augmented tensor `(n_envs, cond_steps, augmented_obs_dim)`.

- **`_preprocess_expert_dataset(expert_dataset)`**: same pattern but batched over the offline demonstration dataset. Produces augmented `Transition` objects for the replay buffer.

- **`_get_flow_action_for_eval(obs)`**: runs *only* the frozen SIA policy (no residual) on evaluation observations, for a pretrained-policy baseline curve during training. Uses `forward_from_features`, same as online rollouts.

- Inherits everything else (`get_action`, `collect_transition`, the full training loop) from the img agent / parent.

### 4.4 `cfg/robomimic/finetune/lift/ft_distill_residual_flow_sia_img.yaml` — Training config

All hyperparameters match DICE-RL's reference image config (`ft_distill_residual_flow_unet_img` for Square) **exactly**. We verified 18 key hyperparameters programmatically. The ONLY differences are:

- `_target_` points to our agent and model classes
- `env_name: lift` (with env meta below)
- `horizon_steps: 4`, `act_steps: 4` (DICE-RL defaults for short tasks)
- SIA-specific top-level params: `diffusion_steps: 5`, `task_prompt: "lift the cube"`, `sia_root: null` (nulled; wrapper auto-detects)
- `obs_dim: 9` at the top level (augmented to 1033 in-place by the agent)
- `base_policy_path: checkpoints/lbm_bc_lift/best.pt` (update this)
- `robomimic_env_cfg_path: cfg/robomimic/env_meta/lift-sia-img.json`
- `normalization_path: null` (will be auto-generated)
- `image_keys: ['agentview_image', 'robot0_eye_in_hand_image']` (2 cameras)
- `shape_meta.rgb.shape: [96, 96, 6]` (2 cameras × 3 channels)

Matched hyperparameters (critical; do not drift from these):
```
batch_size:                256
gradient_steps:            20
gamma:                     0.99
tau:                       0.01
actor_lr / critic_lr:      1e-4
bc_loss_weight:            50.0
critic_ensemble_size:      10
num_exploration_samples:   16
horizon_steps / act_steps: 4
n_step:                    3
num_train_steps:           150000
eval_freq:                 1000
actor_update_freq:         2
critic_target_update_freq: 2
adaptive_expert_ratio:     0.5 → 0.1 over 10000 steps
online_explore_strategy:   max_q_min
evaluate_strategy:         max_q_min
```

### 4.5 `cfg/robomimic/env_meta/lift-sia-img.json` — Robosuite env meta

Same as DICE-RL's `lift-img.json` with one change: `camera_names` now has both `"agentview"` and `"robot0_eye_in_hand"` (instead of just the wrist camera). Everything else matches: Panda robot, OSC_POSE controller (7D action space, control_delta=true), 20 Hz control freq, 96×96 camera resolution.

---

## 5. End-to-end data flow (memorize this)

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. Env step: robosuite Lift (2 cameras, OSC_POSE, Panda)            │
│    produces raw obs: state(9D, unnormalized), rgb(96,96,6)          │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. RobomimicImageWrapper                                            │
│    - normalizes state with min-max (≡ z-score via our trick)        │
│    - stacks cameras, returns {"state": (9,), "rgb": (96,96,6)}      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. MultiStepFull wrapper                                            │
│    - adds cond_steps / horizon_steps dimensions                     │
│    - handles reward aggregation across chunked actions              │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. TrainDistillResidualFlowSiaAgent._extract_visual_features_from_obs│
│    - takes state + rgb                                              │
│    - calls SIAPolicyWrapper.extract_visual_features(cond)           │
│         → splits stacked RGB per camera                             │
│         → resize_with_pad to 224×224, CLIP normalize                │
│         → CLIP encode → (B, 2, 512)                                 │
│         → flatten to (B, 1024)                                      │
│    - concatenates [state(9) || clip_features(1024)] = (B, 1, 1033) │
└─────────────────────────────────────────────────────────────────────┘
                                │ augmented_state
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. DistillResidualRLSiaModel.get_action(state, noise)               │
│    - noise ~ N(0, I), shape (B, 4, 7)                               │
│    - For each of 16 noise samples:                                  │
│        pretrained_actions = pretrained_flow_policy                  │
│                              .forward_from_features(state, noise)   │
│                              .trajectories   # (B, 4, 7)            │
│        residual_actions   = actor(state, noise)    # (B, 4, 7)      │
│        total_action        = pretrained + residual                  │
│    - Best-of-N: pick candidate with max min-Q over critic ensemble  │
└─────────────────────────────────────────────────────────────────────┘
                                │ action_chunk (B, 4, 7)
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 6. Env step with chunk                                              │
│    - MultiStepFull unrolls 4 env steps                              │
│    - RobomimicImageWrapper unnormalizes actions via min-max         │
│      (≡ z-score unnormalization via our trick)                     │
│    - robosuite applies OSC_POSE controller                          │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
   collect_transition → replay buffer (stores augmented states, actions)
                                │
                                ▼
   every gradient step: sample batch → compute loss → update actor/critic
```

---

## 6. How to actually run this

### 6.1 Prerequisites

On whatever machine you're on:

1. **Clone both repos side-by-side**:
   ```bash
   cd ~/Desktop     # or wherever; the wrapper infers ../sia from its own path
   git clone git@github.com:fazalmittu/dice-rl.git
   cd dice-rl && git checkout sia-integration
   cd ..
   git clone <sia-repo-url> sia
   ```

2. **Install dependencies**: DICE-RL's environment + SIA's dependencies (robosuite, diffusers, transformers for CLIP, timm, etc.). Check both repos' `requirements.txt` / `pyproject.toml`. Robosuite version must be v1.4.1 to match DICE-RL's env meta.

3. **Train the SIA BC policy** separately on the Lift task with `chunk_length=4`. The default SIA training script uses `chunk_length=20`; you must override. See `sia/scripts/train_lbm_bc.py`. Save the checkpoint at `dice-rl/checkpoints/lbm_bc_lift/best.pt` (or update `base_policy_path` in the config).

4. **Prepare the expert dataset** for RLPD. The config points at `data_dir/robomimic/lift-img/ph_finetune/train.npz`. This is a robomimic image dataset. You can follow the DICE-RL repo's data preparation instructions. The expert data must use the same 2-camera setup.

### 6.2 Launch command

```bash
cd /Users/fazal/Desktop/dice-rl
python script/run.py \
    --config-path ../cfg/robomimic/finetune/lift \
    --config-name ft_distill_residual_flow_sia_img
```

Or from the repo root, however DICE-RL's hydra entry point expects you to invoke it — check `script/run.py` to confirm.

### 6.3 What to watch in the logs

- **At startup**: "SIA DiT loaded: state_dim=9, action_dim=7, chunk_length=4, clip_dim=512, num_cameras=2, visual_feature_dim=1024". If any of these are wrong, the BC policy was trained with mismatched config.
- "Task embedding cached for prompt: 'lift the cube'".
- "TrainDistillResidualFlowSiaAgent initialized successfully ... Model obs_dim: 1033".
- Generated normalization file path — verify it's readable.
- During training: `actor_q_loss`, `actor_residual_loss`, `critic_loss`, `residual_norm` (should start near zero and grow), `current_q_mean`, `pretrained_q_mean`, `q_advantage_mean`.
- Eval runs every 1000 steps: look for `eval/success_rate` climbing over the base-policy baseline.

### 6.4 Common failure modes to expect (untested code)

1. **CUDA OOM**: 10-critic ensemble + DiT inference + 16-sample exploration is heavy. If OOM, reduce `critic_ensemble_size`, `num_exploration_samples`, `num_multi_z_for_actor_loss`, or `env.n_envs`.
2. **NaN losses early in training**: likely a normalization mismatch. Verify `norm_stats` made it into the generated npz correctly (load it and check the arrays are finite).
3. **`KeyError: 'norm_stats'` when loading the SIA checkpoint**: the BC training script didn't save norm_stats. Add it, or handle the fallback path in the agent (currently we log a warning and skip normalization — likely fatal).
4. **Shape mismatch in `extract_visual_features`**: the env wrapper's stacked RGB layout might differ from what we assume. Check the code path around lines 180-210 of `sia_policy_wrapper.py` — there's a heuristic for HWC vs CHW detection.
5. **Import error from `models.lbm`**: the SIA repo isn't at the expected relative path. Set `sia_root:` explicitly in the YAML config.
6. **Frozen DiT model outputs are the wrong dtype**: the DiT's internal dtype (set by `x_embedder.weight.dtype`) may be bf16 or fp16 when loaded on GPU. The wrapper converts inputs appropriately, but if you see dtype errors in `denoise_step`, that's where to look.

### 6.5 Verifying correctness (no-GPU smoke test)

We already did the following; confirm they still pass after a clone:

```bash
# 1. Syntactic compilation of all new files
python -c "import py_compile; [py_compile.compile(f, doraise=True) for f in [
    'model/rl/sia_policy_wrapper.py',
    'model/rl/distill_residual_rl_sia.py',
    'agent/finetune/train_distill_residual_flow_sia_agent.py']]"

# 2. YAML / JSON validity
python -c "import json; json.load(open('cfg/robomimic/env_meta/lift-sia-img.json'))"

# 3. SIA import path resolution
python -c "
import os
sia_root = os.path.normpath(os.path.join('model/rl', '..', '..', '..', 'sia'))
assert os.path.exists(os.path.join(sia_root, 'models', 'lbm.py'))
print('SIA path OK:', sia_root)
"
```

---

## 7. Git status

- Upstream: `origin` → `https://github.com/real-stanford/dice-rl.git` (do not push here)
- Personal fork: `fazal` → `https://github.com/fazalmittu/dice-rl.git`
- Working branch: `sia-integration`
- Committed files (5, all new — no upstream files modified):
  ```
  agent/finetune/train_distill_residual_flow_sia_agent.py
  cfg/robomimic/env_meta/lift-sia-img.json
  cfg/robomimic/finetune/lift/ft_distill_residual_flow_sia_img.yaml
  model/rl/distill_residual_rl_sia.py
  model/rl/sia_policy_wrapper.py
  ```

To pick up on a new machine:
```bash
git clone https://github.com/fazalmittu/dice-rl.git
cd dice-rl
git checkout sia-integration
git remote add origin https://github.com/real-stanford/dice-rl.git
git fetch origin    # optional, to track upstream
```

---

## 8. Known gaps and pending work

1. **Nothing has actually been run end-to-end.** All code was written on a machine without GPU access. Expect bugs on first execution.

2. **We only have a config for Lift.** The plan is to support four tasks: `Lift`, `Can`, `ToolHang`, and a bimanual task (`TwoArmTransport` or `TwoArmBoxCleanup`). Each needs:
   - A per-task config YAML (copy `ft_distill_residual_flow_sia_img.yaml` and change `env_name`, `base_policy_path`, `task_prompt`, camera keys, `max_episode_steps`)
   - A per-task env meta JSON (copy `lift-sia-img.json`, change `env_name` and camera names to match what SIA's BC was trained on)
   - A SIA BC checkpoint trained on that task with `chunk_length=4`

   Per-task differences to be aware of:
   - **ToolHang** uses larger images (240×240) and longer action chunks (8) in DICE-RL's reference config. Our `horizon_steps=4` choice may need revisiting.
   - **2-arm tasks** have `obs_dim=18` and `action_dim=14`. Our code reads these from the SIA checkpoint's config dict, so it should just work — but the camera list grows to 3 or more (see DICE-RL's transport env meta for reference).

3. **No expert dataset preprocessing has been tested.** The `_preprocess_expert_dataset` method extracts CLIP features from every expert transition at startup. On large datasets, this could take significant time and memory. If slow, add caching (save the preprocessed dataset to disk after the first run).

4. **Task prompt is hardcoded in the YAML.** If you change `env_name` to something other than Lift, **remember to also change `task_prompt`** — CLIP text conditioning is task-specific.

5. **Evaluation with the base policy alone** (no residual) uses `_get_flow_action_for_eval`, which is implemented but untested. It's important for reporting baseline numbers.

6. **We have not verified that the SIA repo's `models.transforms.resize_with_pad_torch` handles batched input correctly across all image sizes.** Double-check on first run.

7. **torch.compile on CLIP** — SIA has `enable_clip_compile()` on the DiT model. We do not call it. Enabling it may give a speed boost; add `self.model.enable_clip_compile()` in the wrapper's `__init__` after freezing, and test.

---

## 9. Quick reference: where things live

```
dice-rl/
├── SIA_INTEGRATION_HANDOFF.md                    # this file
├── agent/finetune/
│   ├── train_distill_residual_flow_agent.py      # upstream parent
│   ├── train_distill_residual_flow_img_agent.py  # upstream img parent
│   └── train_distill_residual_flow_sia_agent.py  # OURS
├── model/rl/
│   ├── distill_residual_rl.py                    # upstream base model
│   ├── distill_residual_rl_img.py                # upstream img model
│   ├── distill_residual_rl_sia.py                # OURS
│   └── sia_policy_wrapper.py                     # OURS
├── cfg/robomimic/
│   ├── env_meta/
│   │   ├── lift-img.json                         # upstream (1 camera)
│   │   └── lift-sia-img.json                     # OURS (2 cameras)
│   └── finetune/lift/
│       └── ft_distill_residual_flow_sia_img.yaml # OURS
└── util/hybrid_replay_buffer.py                  # upstream, unchanged

sia/                                              # separate repo, not modified
├── models/
│   ├── lbm.py                                    # DiTPolicy class (imported)
│   ├── transforms.py                             # resize_with_pad_torch (imported)
│   ├── task_encoder.py                           # used internally by DiTPolicy
│   └── clip/                                     # local CLIP (imported)
└── scripts/
    └── train_lbm_bc.py                           # BC training script (run separately)
```

---

## 10. If you get stuck

- The design decisions are all documented in §3 with rationale. If something seems weird, check there first.
- The canonical reference for what the residual RL pipeline is *supposed* to do is `DistillResidualRLImgModel.get_action` and `.loss` in `model/rl/distill_residual_rl_img.py` / `distill_residual_rl.py`. Our `DistillResidualRLSiaModel` only changes how the pretrained policy is loaded; the rest is identical.
- The canonical reference for how SIA's DiT works is `DiTPolicy.infer` and `compute_static_condition` in `sia/models/lbm.py`. Our wrapper deliberately bypasses `infer` and calls `denoise_step` directly, because we want to inject pre-extracted CLIP features.
- The DICE-RL paper (arXiv:2603.10263) has extensive ablations in Appendix A that you should consult before changing any hyperparameters. The method is claimed to be robust to most choices, with the exception of `bc_loss_weight` (50 for short tasks, 100 for long/precise) and `n_step` (3 typical, 5 for precision tasks).

Good luck.
