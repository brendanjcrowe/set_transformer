"""
RL training with a pretrained Set Transformer as feature extractor for Ant-Tag.

Follows the pattern of train_odd_even_pretrained.py:
  1) Load a pretrained PFSetTransformer (encoder only) as a fixed feature processor
  2) Wrap AntTag env with PF + ST feature extraction
  3) Train PPO / SAC on the augmented observation [base_obs + st_features]

Usage:
    python training/train_ant_tag_pretrained.py \
        --pretrained_st_model_path models/ant_tag_st_pretrained.pt \
        --algorithm PPO --total_timesteps 2000000 --n_envs 4
"""

import argparse

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

import pdomains  # noqa: F401 — registers pdomains-ant-tag-v0
from set_transformer.models import PFSetTransformer, SetTransformer
from set_transformer.rl.particle_filters.ant_tag import AntTagParticleFilter
from set_transformer.rl.wrappers.particle_filter import PFPlusFeaturesObservationWrapper


# ---------------------------------------------------------------------------
# Processor that loads a PFSetTransformer checkpoint and exposes the encoder
# as a feature extractor.  Implements the interface expected by
# PFPlusFeaturesObservationWrapper (dim_particle_input, st_output_dim,
# process_particles_numpy).
# ---------------------------------------------------------------------------

class AntTagPretrainedProcessor:
    """
    Loads a PFSetTransformer checkpoint (either a raw state_dict or a Trainer
    checkpoint dict) and exposes the SetTransformer encoder for feature
    extraction.

    Attributes:
        dim_particle_input: int  — expected last dim of input particles (2 for x,y)
        st_output_dim: int       — flattened output size of the encoder
    """

    def __init__(self, model_path: str, device: str = "auto",
                 # Model architecture params (must match pretraining):
                 num_particles: int = 100,
                 dim_particles: int = 2,
                 num_encodings: int = 8,
                 dim_encoder: int = 2,
                 num_inds: int = 32,
                 dim_hidden: int = 128,
                 num_heads: int = 4,
                 ln: bool = True):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Reconstruct the full PFSetTransformer so we can load the state_dict
        self._pf_st = PFSetTransformer(
            num_particles=num_particles,
            dim_particles=dim_particles,
            num_encodings=num_encodings,
            dim_encoder=dim_encoder,
            num_inds=num_inds,
            dim_hidden=dim_hidden,
            num_heads=num_heads,
            ln=ln,
        )

        # Load checkpoint
        loaded = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(loaded, dict) and "model_state_dict" in loaded:
            state_dict = loaded["model_state_dict"]
        elif isinstance(loaded, dict):
            state_dict = loaded
        else:
            raise ValueError(
                f"Expected a state_dict or trainer checkpoint, got {type(loaded)}"
            )

        self._pf_st.load_state_dict(state_dict)
        self._pf_st.eval()
        self._pf_st.to(self.device)

        # We only use the SetTransformer encoder part for feature extraction
        self._encoder: SetTransformer = self._pf_st.set_transformer

        self.dim_particle_input: int = dim_particles
        # Encoder output: [batch, num_encodings, dim_encoder] → flattened
        self.st_output_dim: int = num_encodings * dim_encoder

        print(
            f"AntTagPretrainedProcessor loaded from {model_path}\n"
            f"  dim_particle_input={self.dim_particle_input}, "
            f"st_output_dim={self.st_output_dim}"
        )

    def process_particles_numpy(
        self, particles_np: np.ndarray, weights_np: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Process a single particle set → feature vector.

        Args:
            particles_np: [num_particles, 2] array of particle positions.
            weights_np: Ignored (the encoder doesn't use weights).

        Returns:
            features: [st_output_dim,] numpy array.
        """
        particle_tensor = (
            torch.tensor(particles_np, dtype=torch.float32)
            .unsqueeze(0)  # add batch dim → [1, N, 2]
            .to(self.device)
        )
        with torch.no_grad():
            # Encoder: [1, N, 2] → [1, num_encodings, dim_encoder]
            enc_out = self._encoder(particle_tensor)
            # Flatten: [1, num_encodings * dim_encoder]
            features = enc_out.view(1, -1)
        return features.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# PF-based reward shaping
# ---------------------------------------------------------------------------

class PFRewardShapingWrapper(gym.Wrapper):
    """
    Dense reward shaping using the particle filter belief state.

    Three components:
      1. Distance: -dist_to_pf_mean (closer to belief mean = higher reward)
      2. Entropy: -sum(w_i * log(w_i)) of PF weights (lower entropy = higher reward)
      3. Tag bonus: large reward on successful tag (terminated=True)

    Coefficients are mutable — the CurriculumCallback can phase distance out
    and tag bonus in over the course of training.
    """

    def __init__(self, env: gym.Env, distance_coeff: float = 1.0,
                 entropy_coeff: float = 0.0, tag_bonus_coeff: float = 0.0):
        super().__init__(env)
        self.distance_coeff = distance_coeff
        self.entropy_coeff = entropy_coeff
        self.tag_bonus_coeff = tag_bonus_coeff

    def set_reward_coeffs(self, distance_coeff: float, entropy_coeff: float,
                          tag_bonus_coeff: float):
        """Called by CurriculumCallback to phase reward components."""
        self.distance_coeff = distance_coeff
        self.entropy_coeff = entropy_coeff
        self.tag_bonus_coeff = tag_bonus_coeff

    def _get_pf_wrapper(self):
        """Walk the wrapper stack to find a wrapper exposing a particle_filter."""
        e = self.env
        while e is not None:
            if getattr(e, "particle_filter", None) is not None:
                return e
            e = getattr(e, "env", None)
        raise RuntimeError("PFRewardShapingWrapper requires a PF wrapper (with .particle_filter) in the stack")

    @staticmethod
    def _pf_entropy(weights: np.ndarray) -> float:
        """Shannon entropy: -sum(w_i * log(w_i)), with 0*log(0) = 0."""
        w = weights[weights > 0]
        return -float(np.sum(w * np.log(w)))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pf_wrapper = self._get_pf_wrapper()
        pf = pf_wrapper.particle_filter

        # Support both flat (concatenated) obs and Dict obs ({"obs": ..., "particles": ...})
        base_obs = obs["obs"] if isinstance(obs, dict) else obs
        ant_pos = base_obs[:2]
        pf_mean = np.average(pf.particles, weights=pf.weights, axis=0)
        dist_to_mean = float(np.linalg.norm(ant_pos - pf_mean))
        entropy = self._pf_entropy(pf.weights)

        distance_reward = self.distance_coeff * (-dist_to_mean)
        entropy_reward = self.entropy_coeff * (-entropy)
        tag_bonus = self.tag_bonus_coeff if terminated else 0.0

        shaped_reward = reward + distance_reward + entropy_reward + tag_bonus
        info["distance_reward"] = distance_reward
        info["entropy_reward"] = entropy_reward
        info["tag_bonus"] = tag_bonus
        info["pf_entropy"] = entropy
        info["dist_to_pf_mean"] = dist_to_mean
        return obs, shaped_reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Curriculum: controllable visibility wrapper + annealing callback
# ---------------------------------------------------------------------------

class CurriculumVisibilityWrapper(gym.Wrapper):
    """
    Controls target visibility via a mutable visibility_radius.

    Sits between the base env and the PF wrapper. Overrides obs[-2:] based
    on the current curriculum radius rather than the env's fixed 3.0.

    During warm-start (visibility_radius=large), the target is always
    revealed so the PF collapses to the true position and the agent
    effectively trains with full observability through the ST pipeline.
    """

    def __init__(self, env: gym.Env, initial_visibility_radius: float = 100.0):
        super().__init__(env)
        self.visibility_radius = initial_visibility_radius

    def set_curriculum_radius(self, radius: float):
        """Called by CurriculumCallback via env_method."""
        self.visibility_radius = radius

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ant_pos = obs[:2]
        true_target = self.env.unwrapped.get_target_pos()
        dist = float(np.linalg.norm(ant_pos - true_target))

        if dist < self.visibility_radius:
            obs = obs.copy()
            obs[-2:] = true_target
        else:
            obs = obs.copy()
            obs[-2:] = np.zeros(2)

        return obs, reward, terminated, truncated, info


class CurriculumCallback(BaseCallback):
    """
    SB3 callback that anneals visibility_radius and reward coefficients.

    Visibility schedule: list of (progress_fraction, radius) waypoints.
    Reward schedule: list of (progress_fraction, distance_coeff, entropy_coeff).
    Between waypoints, values are linearly interpolated.

    Default visibility schedule:
      - 0-30%: radius=100 (fully observed)
      - 30-70%: linear decrease 100 → 3.0
      - 70-100%: radius=3.0 (real POMDP)

    Default reward schedule (frac, distance_coeff, entropy_coeff, tag_bonus):
      - 0-30%: distance=1.0, entropy=0.0, tag=0 (pure distance while fully visible)
      - 30-70%: distance 1.0→0.0, entropy 0→0, tag 0→50 (phase in tag bonus)
      - 70-100%: distance=0.0, entropy=0.0, tag=50 (sparse tag reward at real POMDP)
    """

    def __init__(
        self,
        total_timesteps: int,
        schedule: list[tuple[float, float]] | None = None,
        reward_schedule: list[tuple[float, ...]] | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        if schedule is None:
            schedule = [(0.0, 100.0), (0.3, 100.0), (0.7, 3.0), (1.0, 3.0)]
        self.schedule = sorted(schedule, key=lambda x: x[0])

        if reward_schedule is None:
            reward_schedule = [
                (0.0, 1.0, 0.0, 0.0),
                (0.3, 1.0, 0.0, 0.0),
                (0.7, 0.0, 0.0, 50.0),
                (1.0, 0.0, 0.0, 50.0),
            ]
        self.reward_schedule = sorted(reward_schedule, key=lambda x: x[0])

    def _interpolate_schedule(self, schedule, progress: float):
        """Linearly interpolate a schedule. Returns all values after the fraction."""
        if progress <= schedule[0][0]:
            return schedule[0][1:]
        if progress >= schedule[-1][0]:
            return schedule[-1][1:]
        for i in range(len(schedule) - 1):
            frac_lo = schedule[i][0]
            frac_hi = schedule[i + 1][0]
            if frac_lo <= progress <= frac_hi:
                t = (progress - frac_lo) / (frac_hi - frac_lo) if frac_hi > frac_lo else 1.0
                vals_lo = schedule[i][1:]
                vals_hi = schedule[i + 1][1:]
                return tuple(lo + t * (hi - lo) for lo, hi in zip(vals_lo, vals_hi))
        return schedule[-1][1:]

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps
        (radius,) = self._interpolate_schedule(self.schedule, progress)
        reward_vals = self._interpolate_schedule(self.reward_schedule, progress)
        dist_coeff, ent_coeff = reward_vals[0], reward_vals[1]
        tag_bonus_coeff = reward_vals[2] if len(reward_vals) > 2 else 0.0

        # Update all training envs (works through VecNormalize → SubprocVecEnv)
        vec_env = self.training_env
        # Unwrap VecNormalize if present
        while hasattr(vec_env, "venv"):
            vec_env = vec_env.venv

        # For both SubprocVecEnv and DummyVecEnv, walk wrapper stacks
        if hasattr(vec_env, "envs"):
            # DummyVecEnv — direct access
            for env in vec_env.envs:
                _set_radius_recursive(env, radius)
                _set_reward_coeffs_recursive(env, dist_coeff, ent_coeff, tag_bonus_coeff)
        elif hasattr(vec_env, "env_method"):
            # SubprocVecEnv — call into subprocesses
            vec_env.env_method("set_curriculum_radius", radius)
            vec_env.env_method("set_reward_coeffs", dist_coeff, ent_coeff, tag_bonus_coeff)

        if self.verbose > 0 and self.num_timesteps % 10000 < (self.training_env.num_envs if self.training_env else 1):
            print(
                f"[Curriculum] step={self.num_timesteps}, progress={progress:.2f}, "
                f"vis_radius={radius:.2f}, dist_coeff={dist_coeff:.3f}, "
                f"ent_coeff={ent_coeff:.3f}, tag_bonus={tag_bonus_coeff:.1f}"
            )

        return True


def _set_radius_recursive(env, radius: float):
    """Walk the wrapper stack and set visibility_radius on CurriculumVisibilityWrapper."""
    e = env
    while e is not None:
        if isinstance(e, CurriculumVisibilityWrapper):
            e.visibility_radius = radius
            return
        e = getattr(e, "env", None)


def _set_reward_coeffs_recursive(env, distance_coeff: float, entropy_coeff: float,
                                  tag_bonus_coeff: float):
    """Walk the wrapper stack and set reward coefficients on PFRewardShapingWrapper."""
    e = env
    while e is not None:
        if isinstance(e, PFRewardShapingWrapper):
            e.set_reward_coeffs(distance_coeff, entropy_coeff, tag_bonus_coeff)
            return
        e = getattr(e, "env", None)


class _CurriculumRouter(gym.Wrapper):
    """Thin outermost wrapper so SubprocVecEnv.env_method can reach inner wrappers."""

    def set_curriculum_radius(self, radius: float):
        _set_radius_recursive(self.env, radius)

    def set_reward_coeffs(self, distance_coeff: float, entropy_coeff: float,
                          tag_bonus_coeff: float):
        _set_reward_coeffs_recursive(self.env, distance_coeff, entropy_coeff,
                                     tag_bonus_coeff)


# ---------------------------------------------------------------------------
# PF interaction mapper for AntTag
# ---------------------------------------------------------------------------

def ant_tag_pf_interaction_mapper(
    base_env_obs: np.ndarray,
    base_env_info: dict,
    base_env_action: np.ndarray | None = None,
    unwrapped_env=None,
) -> dict:
    """
    Bridge between AntTag observations and the AntTagParticleFilter
    predict / update interface.

    Visibility is determined by obs[-2:]: the CurriculumVisibilityWrapper
    puts the true target there when within the curriculum radius, or zeros
    when not visible. So we check for non-zero obs[-2:] rather than
    hardcoding a radius.
    """
    ant_pos = base_env_obs[:2].copy()
    target_in_obs = base_env_obs[-2:].copy()

    # CurriculumVisibilityWrapper sets obs[-2:] to true target when visible,
    # zeros when not. A zero target at the origin is astronomically unlikely.
    visible = np.any(target_in_obs != 0.0)

    if visible:
        observed_target = target_in_obs
    else:
        observed_target = np.array([np.nan, np.nan])

    return {
        "predict_args": {"ant_current_pos_from_obs": ant_pos},
        "update_args": {
            "observed_target_pos": observed_target,
            "ant_current_pos_from_obs": ant_pos,
        },
    }


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_ant_tag_pretrained_env(
    pretrained_st_model_path: str,
    st_processor_device: str,
    num_particles: int,
    # Model architecture (must match pretraining):
    dim_particles: int = 2,
    num_encodings: int = 8,
    dim_encoder: int = 2,
    num_inds: int = 32,
    dim_hidden: int = 128,
    num_heads: int = 4,
    rank: int = 0,
    seed: int = 0,
    monitor_dir: str | None = None,
    distance_coeff: float = 1.0,
    entropy_coeff: float = 0.0,
    tag_bonus_coeff: float = 0.0,
    initial_visibility_radius: float = 100.0,
):
    """Return a callable that creates a wrapped AntTag env."""

    def _init():
        env = gym.make("pdomains-ant-tag-v0", rendering=False)
        env.reset(seed=seed + rank)

        # Curriculum wrapper: controls target visibility
        env = CurriculumVisibilityWrapper(env, initial_visibility_radius=initial_visibility_radius)

        processor = AntTagPretrainedProcessor(
            model_path=pretrained_st_model_path,
            device=st_processor_device,
            num_particles=num_particles,
            dim_particles=dim_particles,
            num_encodings=num_encodings,
            dim_encoder=dim_encoder,
            num_inds=num_inds,
            dim_hidden=dim_hidden,
            num_heads=num_heads,
        )

        env = PFPlusFeaturesObservationWrapper(
            env=env,
            particle_filter_class=AntTagParticleFilter,
            particle_filter_kwargs={},
            pretrained_st_processor=processor,
            num_particles=num_particles,
            pf_interaction_mapper=ant_tag_pf_interaction_mapper,
            # Mask obs[-2:] (target position) so the agent can only get
            # target info through ST features, never directly from obs.
            # The PF mapper still sees the unmasked obs for its updates.
            obs_mask_indices=[-2, -1],
        )

        env = PFRewardShapingWrapper(
            env, distance_coeff=distance_coeff, entropy_coeff=entropy_coeff,
            tag_bonus_coeff=tag_bonus_coeff,
        )

        if monitor_dir:
            env = Monitor(env, os.path.join(monitor_dir, str(rank)))
        else:
            env = Monitor(env)

        # Outermost wrapper so SubprocVecEnv.env_method can reach curriculum
        env = _CurriculumRouter(env)
        return env

    return _init


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_ant_tag_pretrained(
    pretrained_st_model_path: str,
    st_processor_device: str = "auto",
    algorithm: str = "PPO",
    total_timesteps: int = 2_000_000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    ppo_n_steps: int = 2048,
    num_particles: int = 100,
    dim_particles: int = 2,
    num_encodings: int = 8,
    dim_encoder: int = 2,
    num_inds: int = 32,
    dim_hidden: int = 128,
    num_heads: int = 4,
    seed: int = 0,
    log_dir: str = "./sb3_ant_tag_pretrained_logs/",
    model_save_path: str = "./sb3_ant_tag_pretrained_models/pretrained_agent.zip",
    eval_freq: int = 20_000,
    save_freq: int = 100_000,
    use_vec_normalize: bool = True,
    distance_coeff: float = 1.0,
    entropy_coeff: float = 0.0,
    curriculum_schedule: list[tuple[float, float]] | None = None,
    reward_schedule: list[tuple[float, ...]] | None = None,
    net_arch: list[int] | None = None,
):
    """Train PPO/SAC on AntTag with pretrained ST features and curriculum."""
    print(f"Training {algorithm} on AntTag with pretrained ST: {pretrained_st_model_path}")
    print(f"Initial reward: distance_coeff={distance_coeff}, entropy_coeff={entropy_coeff}")
    if net_arch:
        print(f"Policy net_arch: {net_arch}")
    if curriculum_schedule:
        print(f"Visibility schedule: {curriculum_schedule}")
    else:
        print("Visibility schedule: default [(0, 100), (0.3, 100), (0.7, 3), (1, 3)]")
    if reward_schedule:
        print(f"Reward schedule: {reward_schedule}")
    else:
        print("Reward schedule: default [(0, 1.0, 0.0), (0.3, 1.0, 0.0), (0.7, 0.0, 0.5), (1, 0.0, 0.5)]")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    monitor_dir = os.path.join(log_dir, "gym_monitor")
    os.makedirs(monitor_dir, exist_ok=True)

    # Determine initial visibility from schedule
    initial_vis = 100.0
    if curriculum_schedule:
        initial_vis = curriculum_schedule[0][1]

    # Common kwargs for env factory
    env_kw = dict(
        pretrained_st_model_path=pretrained_st_model_path,
        st_processor_device=st_processor_device,
        num_particles=num_particles,
        dim_particles=dim_particles,
        num_encodings=num_encodings,
        dim_encoder=dim_encoder,
        num_inds=num_inds,
        dim_hidden=dim_hidden,
        num_heads=num_heads,
        distance_coeff=distance_coeff,
        entropy_coeff=entropy_coeff,
        tag_bonus_coeff=0.0,  # starts at 0, curriculum phases it in
        initial_visibility_radius=initial_vis,
    )

    # Training envs
    env_fn = make_ant_tag_pretrained_env(
        **env_kw, seed=seed, monitor_dir=monitor_dir
    )
    vec_env_cls = SubprocVecEnv if n_envs > 1 else None
    vec_env = make_vec_env(env_fn, n_envs=n_envs, seed=seed, vec_env_cls=vec_env_cls)
    if use_vec_normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    # Eval envs — always evaluate at real POMDP difficulty (radius=3.0)
    eval_env_kw = dict(env_kw)
    eval_env_kw["initial_visibility_radius"] = 3.0
    eval_env_fn = make_ant_tag_pretrained_env(
        **eval_env_kw, rank=n_envs + 1, seed=seed
    )
    eval_vec_env = make_vec_env(eval_env_fn, n_envs=1)
    if use_vec_normalize:
        eval_vec_env = VecNormalize(
            eval_vec_env, training=False, norm_obs=True, norm_reward=False
        )

    # Create agent
    policy_kwargs = {}
    if net_arch is not None:
        policy_kwargs["net_arch"] = net_arch

    if algorithm.upper() == "PPO":
        model = PPO(
            "MlpPolicy", vec_env,
            learning_rate=learning_rate,
            n_steps=ppo_n_steps,
            batch_size=batch_size,
            verbose=1,
            tensorboard_log=log_dir,
            seed=seed,
            policy_kwargs=policy_kwargs if policy_kwargs else None,
        )
    elif algorithm.upper() == "SAC":
        model = SAC(
            "MlpPolicy", vec_env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            verbose=1,
            tensorboard_log=log_dir,
            seed=seed,
            policy_kwargs=policy_kwargs if policy_kwargs else None,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    print(f"Policy architecture:\n{model.policy}")

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(save_freq // n_envs, 1),
        save_path=os.path.join(os.path.dirname(model_save_path), "checkpoints"),
        name_prefix="ant_tag_pretrained",
    )
    eval_cb = EvalCallback(
        eval_vec_env,
        best_model_save_path=os.path.join(
            os.path.dirname(model_save_path), "best_model"
        ),
        log_path=log_dir,
        eval_freq=max(eval_freq // n_envs, 1),
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )
    curriculum_cb = CurriculumCallback(
        total_timesteps=total_timesteps,
        schedule=curriculum_schedule,
        reward_schedule=reward_schedule,
        verbose=1,
    )

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_cb, eval_cb, curriculum_cb],
            progress_bar=True,
        )
    finally:
        model.save(model_save_path)
        if use_vec_normalize and isinstance(vec_env, VecNormalize):
            vec_env.save(
                os.path.join(os.path.dirname(model_save_path), "vecnormalize.pkl")
            )
        print(f"Model saved to {model_save_path}")
        vec_env.close()
        eval_vec_env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RL with pretrained Set Transformer on Ant-Tag"
    )

    # Model checkpoint
    parser.add_argument(
        "--pretrained_st_model_path", type=str, required=True,
        help="Path to pretrained PFSetTransformer state_dict or trainer checkpoint"
    )
    parser.add_argument(
        "--st_processor_device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
    )

    # ST architecture (must match pretraining)
    parser.add_argument("--num_particles", type=int, default=100)
    parser.add_argument("--dim_particles", type=int, default=2)
    parser.add_argument("--num_encodings", type=int, default=8)
    parser.add_argument("--dim_encoder", type=int, default=2)
    parser.add_argument("--num_inds", type=int, default=32)
    parser.add_argument("--dim_hidden", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)

    # RL agent
    parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "SAC"])
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ppo_n_steps", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)

    # Logging / saving
    parser.add_argument("--log_dir", type=str, default="./sb3_ant_tag_pretrained_logs/")
    parser.add_argument(
        "--model_save_path", type=str,
        default="./sb3_ant_tag_pretrained_models/pretrained_agent.zip",
    )
    parser.add_argument("--eval_freq", type=int, default=20_000)
    parser.add_argument("--save_freq", type=int, default=100_000)
    parser.add_argument("--no_vec_normalize", action="store_true")
    parser.add_argument(
        "--net_arch", type=str, default=None,
        help="Policy network hidden layers, e.g. '256,256'. Default: SB3 default (64,64).",
    )

    # Reward shaping (initial values — overridden by reward_schedule if set)
    parser.add_argument(
        "--distance_coeff", type=float, default=1.0,
        help="Initial coefficient for distance reward (-dist_to_pf_mean)",
    )
    parser.add_argument(
        "--entropy_coeff", type=float, default=0.0,
        help="Initial coefficient for entropy reward (-pf_entropy)",
    )

    # Curriculum
    parser.add_argument(
        "--curriculum", type=str, default=None,
        help=(
            "Visibility curriculum schedule as comma-separated frac:radius pairs. "
            "E.g. '0:100,0.3:100,0.7:3,1:3' means fully visible for 30%%, "
            "linear decrease to 3.0 over next 40%%, then real task. "
            "Default: '0:100,0.3:100,0.7:3,1:3'"
        ),
    )
    parser.add_argument(
        "--reward_schedule", type=str, default=None,
        help=(
            "Reward coefficient schedule as frac:dist:ent:tag_bonus quads. "
            "E.g. '0:1:0:0,0.3:1:0:0,0.7:0:0:50,1:0:0:50' phases distance "
            "out and tag bonus in. Default: '0:1:0:0,0.3:1:0:0,0.7:0:0:50,1:0:0:50'"
        ),
    )

    args = parser.parse_args()

    # Parse curriculum schedule
    curriculum_schedule = None
    curriculum_str = args.curriculum or "0:100,0.3:100,0.7:3,1:3"
    try:
        curriculum_schedule = []
        for pair in curriculum_str.split(","):
            frac, rad = pair.strip().split(":")
            curriculum_schedule.append((float(frac), float(rad)))
    except Exception as e:
        parser.error(f"Invalid --curriculum format: {e}")

    # Parse reward schedule (frac:dist:ent:tag_bonus)
    reward_schedule = None
    reward_str = args.reward_schedule or "0:1:0:0,0.3:1:0:0,0.7:0:0:50,1:0:0:50"
    try:
        reward_schedule = []
        for entry in reward_str.split(","):
            parts = [float(p) for p in entry.strip().split(":")]
            if len(parts) == 3:
                parts.append(0.0)  # backward compat: no tag bonus
            reward_schedule.append(tuple(parts))
    except Exception as e:
        parser.error(f"Invalid --reward_schedule format: {e}")

    # Parse net_arch
    net_arch = None
    if args.net_arch:
        net_arch = [int(x) for x in args.net_arch.split(",")]

    train_ant_tag_pretrained(
        pretrained_st_model_path=args.pretrained_st_model_path,
        st_processor_device=args.st_processor_device,
        algorithm=args.algorithm,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        ppo_n_steps=args.ppo_n_steps,
        num_particles=args.num_particles,
        dim_particles=args.dim_particles,
        num_encodings=args.num_encodings,
        dim_encoder=args.dim_encoder,
        num_inds=args.num_inds,
        dim_hidden=args.dim_hidden,
        num_heads=args.num_heads,
        seed=args.seed,
        log_dir=args.log_dir,
        model_save_path=args.model_save_path,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        use_vec_normalize=not args.no_vec_normalize,
        distance_coeff=args.distance_coeff,
        entropy_coeff=args.entropy_coeff,
        curriculum_schedule=curriculum_schedule,
        reward_schedule=reward_schedule,
        net_arch=net_arch,
    )
