import gymnasium as gym
import inspect
import numpy as np

from set_transformer.rl.particle_filters.base import BaseParticleFilter


def _call_pf_interaction_mapper(
    mapper: callable,
    base_env_obs: np.ndarray,
    base_env_info: dict,
    base_env_action: np.ndarray | None,
    unwrapped_env,
    previous_base_env_obs: np.ndarray | None,
) -> dict:
    """Call env-specific PF mapper, passing previous obs when supported."""
    kwargs = {
        "base_env_obs": base_env_obs,
        "base_env_info": base_env_info,
        "base_env_action": base_env_action,
        "unwrapped_env": unwrapped_env,
    }
    signature = inspect.signature(mapper)
    if (
        "previous_base_env_obs" in signature.parameters
        or any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )
    ):
        kwargs["previous_base_env_obs"] = previous_base_env_obs
    return mapper(**kwargs)


class PFDictObservationWrapper(gym.Wrapper):
    """
    Wraps an environment to include particle filter beliefs in a Dict observation space.
    The observation space will be a gym.spaces.Dict with:
    - "obs": The original environment observation.
    - "particles": The particle set [num_particles, particle_dim] from the PF.

    Designed for end-to-end training where the Set Transformer is part of the policy.
    """
    def __init__(self,
                 env: gym.Env,
                 particle_filter_class: type[BaseParticleFilter],
                 particle_filter_kwargs: dict,
                 num_particles: int,
                 # This function will be responsible for extracting necessary info from
                 # base_env_obs for pf.update() and pf.predict()
                 # It should return a dict like {"predict": {kwargs for pf.predict}, "update": {kwargs for pf.update}}
                 pf_interaction_mapper: callable = None,
                 obs_mask_indices: list[int] | None = None,
                 particle_origin_fn: callable = None,
                ):
        super().__init__(env)
        self.particle_filter_class = particle_filter_class
        self.particle_filter_kwargs = particle_filter_kwargs
        self.num_particles = num_particles
        self.particle_filter: BaseParticleFilter | None = None
        self.obs_mask_indices = obs_mask_indices
        # Optional base-obs -> origin hook. When set, the particle set handed to the
        # policy is expressed RELATIVE to that origin (normally the agent's own
        # position). The filter itself keeps working in world coordinates -- only the
        # view the encoder sees is re-centred.
        #
        # This matters when the optimal policy is "move toward somewhere the belief
        # points at". In world coordinates the encoder produces a permutation-invariant
        # summary of absolute positions while the agent's own position arrives through a
        # separate MLP, so the network has to *learn* to subtract one from the other
        # before any relational quantity (which mode is nearest? in what direction?) is
        # available. Re-centring makes that geometry immediate for every method equally.
        self.particle_origin_fn = particle_origin_fn
        self._last_base_env_obs_float: np.ndarray | None = None
        
        # This mapper is crucial and env-specific. It defines how to get args for pf methods from env data.
        # Example for AntTag: 
        # def ant_tag_pf_mapper(base_env_obs, base_env_info, base_env_action=None, unwrapped_env=None):
        #     ant_pos = base_env_obs[:2]
        #     predict_args = {"ant_current_pos_from_obs": ant_pos} # action is passed directly by wrapper
        #     observed_target_pos = np.array([np.nan, np.nan]) # Default if not visible
        #     if unwrapped_env: # hasattr(unwrapped_env, 'get_target_pos'): 
        #            # This part is tricky as direct access to true opponent for visibility check might not always be clean
        #            # For simplicity, AntTagPF's update might need to take the full obs and deduce visibility itself.
        #            true_opponent_pos = unwrapped_env.get_target_pos() 
        #            if np.linalg.norm(ant_pos - true_opponent_pos) <= unwrapped_env.visibility_radius: 
        #                observed_target_pos = base_env_obs[unwrapped_env.observation_space.shape[0]-2:] # Assuming target is last 2
        #     update_args = {"observed_target_pos": observed_target_pos, "ant_current_pos_from_obs": ant_pos}
        #     return {"predict_args": predict_args, "update_args": update_args}
        self.pf_interaction_mapper = pf_interaction_mapper
        if self.pf_interaction_mapper is None:
            print("Warning: pf_interaction_mapper is not provided. PF predict/update calls will only receive action/obs_from_env directly.")

        # Determine particle_dim by instantiating a temporary PF
        # This is a bit of a hack. A better way would be if PF class had a static method or property.
        _temp_obs, _ = self.env.reset() # Need an initial obs for PF instantiation
        _temp_pf_kwargs = particle_filter_kwargs.copy()
        _temp_pf_kwargs['initial_env_obs'] = _temp_obs
        _temp_pf = particle_filter_class(num_particles=self.num_particles, **_temp_pf_kwargs)
        self.particle_dim = _temp_pf.particle_dim
        del _temp_pf, _temp_obs, _ # Clean up
        self.env.reset() # Reset again to ensure clean state for actual first reset

        self.observation_space = gym.spaces.Dict({
            "obs": self.env.observation_space,
            "particles": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_particles, self.particle_dim), 
                dtype=np.float32
            )
        })
        print(f"PFDictObservationWrapper: Original obs space: {self.env.observation_space}")
        print(f"PFDictObservationWrapper: New Dict obs space defined with particle_dim: {self.particle_dim}")

    def reset(self, **kwargs) -> tuple[dict, dict]:
        base_env_obs, info = self.env.reset(**kwargs)
        base_env_obs_float = base_env_obs.astype(np.float32)
        
        pf_init_kwargs = self.particle_filter_kwargs.copy()
        pf_init_kwargs['initial_env_obs'] = base_env_obs_float # Pass initial obs for PF internal init
        self.particle_filter = self.particle_filter_class(
            num_particles=self.num_particles, 
            **pf_init_kwargs
        )
        self._last_base_env_obs_float = base_env_obs_float.copy()
        return self._get_dict_obs(base_env_obs_float), info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        previous_base_env_obs = self._last_base_env_obs_float
        base_env_obs, reward, terminated, truncated, info = self.env.step(action)
        base_env_obs_float = base_env_obs.astype(np.float32)

        # Prepare arguments for PF predict and update using the mapper
        predict_call_kwargs = {}
        update_call_kwargs = {}
        if self.pf_interaction_mapper:
            mapped_args = _call_pf_interaction_mapper(
                self.pf_interaction_mapper,
                base_env_obs=base_env_obs_float, 
                base_env_info=info, 
                base_env_action=action, 
                unwrapped_env=self.env.unwrapped,
                previous_base_env_obs=previous_base_env_obs,
            )
            predict_call_kwargs = mapped_args.get("predict_args", {})
            update_call_kwargs = mapped_args.get("update_args", {})

        self.particle_filter.predict(action, **predict_call_kwargs)

        # When a mapper is provided, update_call_kwargs contains all needed args
        # (e.g. observed_target_pos, ant_current_pos_from_obs for AntTag).
        # Passing base_env_obs_float as a positional would conflict with those kwargs.
        if self.pf_interaction_mapper:
            self.particle_filter.update(**update_call_kwargs)
        else:
            self.particle_filter.update(base_env_obs_float)

        self._last_base_env_obs_float = base_env_obs_float.copy()
        return self._get_dict_obs(base_env_obs_float), reward, terminated, truncated, info

    def _get_dict_obs(self, base_env_obs: np.ndarray) -> dict:
        particles_state = self.particle_filter.particles.astype(np.float32)
        if self.particle_origin_fn is not None:
            origin = np.asarray(self.particle_origin_fn(base_env_obs), dtype=np.float32)
            particles_state = particles_state - origin[None, :]
        # If particles also include weights as the last dim, make sure ST network expects that.
        # The BaseParticleFilter defines particle_dim, which should be used by the ST network.
        agent_obs = base_env_obs
        if self.obs_mask_indices is not None:
            agent_obs = base_env_obs.copy()
            agent_obs[self.obs_mask_indices] = 0.0
        return {"obs": agent_obs, "particles": particles_state}


class PFDictWithWeightsObservationWrapper(gym.Wrapper):
    """Dict observation wrapper exposing PF particles and PF weights.

    The weighted counterpart of PFDictObservationWrapper above: the
    observation is {"obs", "particles", "weights"}, so an encoder can read
    the particle-filter weights instead of treating a near-dead particle as
    a full contributor. Adds deterministic per-episode PF seeding, derived
    from (worker seed, episode index) through a SeedSequence.

    Domain-independent: the env, the filter class and the optional
    pf_interaction_mapper are all supplied by the caller.

    Moved here from experiments/ant_tag/4_train_rl_cgf.py, which still
    re-exports the name. SB3 pickles a policy's features-extractor class by
    module path and the Ant-Tag scripts import each other by flat name, so
    the re-export is what keeps existing checkpoints and sibling scripts
    working.

    **Dead beliefs never reach the policy (PITFALLS.md section 8, item 8).**
    A filter whose weights come back all zero, or containing NaN / inf, has
    refuted every particle; it carries no information. The three encoders
    disagree about what to make of such a row -- CGF in K mode emits a
    sentinel, CGF in K' mode the tilted mean of a UNIFORM belief, and the
    Gaussian extractor mean 0 / covariance 0, i.e. a delta at the arena
    centre -- so in two of three modes a filter failure is indistinguishable
    from a real belief and the policy acts on it. This wrapper is the one
    place every domain's belief passes through, so the check lives here:
    ``on_dead_belief="raise"`` (default) stops the run with the filter's
    class name, and ``"uniform"`` substitutes uniform weights (the honest
    no-information belief over the current particles), sets
    ``info["pf_dead_belief"] = True`` and counts it in
    ``dead_belief_count``. Neither existing filter family can trigger it
    today -- the Odd-Even filters raise on zero mass themselves, and the
    Ant-Tag filters add 1e-300 before normalising so total refutation
    already comes out uniform -- which is why the default can afford to be
    loud.
    """

    ON_DEAD_BELIEF = ("raise", "uniform")

    def __init__(
        self,
        env: gym.Env,
        particle_filter_class,
        particle_filter_kwargs: dict,
        num_particles: int,
        pf_interaction_mapper=None,
        obs_mask_indices: list[int] | None = None,
        particle_filter_seed: int | None = None,
        on_dead_belief: str = "raise",
    ):
        super().__init__(env)
        if on_dead_belief not in self.ON_DEAD_BELIEF:
            raise ValueError(
                f"on_dead_belief must be one of {self.ON_DEAD_BELIEF}, got {on_dead_belief!r}")
        self.on_dead_belief = on_dead_belief
        self.dead_belief_count = 0
        self._warned_dead_belief = False
        self.particle_filter_class = particle_filter_class
        self.particle_filter_kwargs = particle_filter_kwargs
        self.num_particles = num_particles
        self.pf_interaction_mapper = pf_interaction_mapper
        self.obs_mask_indices = obs_mask_indices
        self.particle_filter_seed = (
            None if particle_filter_seed is None
            else int(particle_filter_seed)
        )
        self._pf_reset_count = 0
        self.particle_filter = None
        self._last_base_env_obs_float = None

        temp_obs, _ = self.env.reset()
        temp_pf_kwargs = particle_filter_kwargs.copy()
        temp_pf_kwargs["initial_env_obs"] = temp_obs
        if self.particle_filter_seed is not None:
            # Dimension probing must not consume the first real episode's PF
            # stream. Use a deterministic, separately derived seed.
            temp_pf_kwargs["rng_seed"] = self._derive_pf_seed(0xFFFFFFFF)
        temp_pf = particle_filter_class(num_particles=num_particles, **temp_pf_kwargs)
        self.particle_dim = temp_pf.particle_dim
        del temp_pf
        self.env.reset()

        self.observation_space = gym.spaces.Dict({
            "obs": self.env.observation_space,
            "particles": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_particles, self.particle_dim),
                dtype=np.float32,
            ),
            "weights": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.num_particles,),
                dtype=np.float32,
            ),
        })

    def _derive_pf_seed(self, episode_index: int) -> int:
        seed_sequence = np.random.SeedSequence(
            [self.particle_filter_seed, int(episode_index)])
        return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])

    def set_particle_filter_seed(self, seed: int) -> None:
        """Restart this worker's deterministic PF episode stream."""
        self.particle_filter_seed = int(seed)
        self._pf_reset_count = 0

    def reset(self, **kwargs):
        base_env_obs, info = self.env.reset(**kwargs)
        base_env_obs_float = base_env_obs.astype(np.float32)

        pf_init_kwargs = self.particle_filter_kwargs.copy()
        pf_init_kwargs["initial_env_obs"] = base_env_obs_float
        if self.particle_filter_seed is not None:
            pf_init_kwargs["rng_seed"] = self._derive_pf_seed(
                self._pf_reset_count)
            self._pf_reset_count += 1
        self.particle_filter = self.particle_filter_class(
            num_particles=self.num_particles,
            **pf_init_kwargs,
        )
        self._last_base_env_obs_float = base_env_obs_float.copy()
        return self._get_dict_obs(base_env_obs_float, info), info

    def step(self, action):
        previous_base_env_obs = self._last_base_env_obs_float
        base_env_obs, reward, terminated, truncated, info = self.env.step(action)
        base_env_obs_float = base_env_obs.astype(np.float32)

        predict_call_kwargs = {}
        update_call_kwargs = {}
        if self.pf_interaction_mapper is not None:
            mapped_args = _call_pf_interaction_mapper(
                self.pf_interaction_mapper,
                base_env_obs=base_env_obs_float,
                base_env_info=info,
                base_env_action=action,
                unwrapped_env=self.env.unwrapped,
                previous_base_env_obs=previous_base_env_obs,
            )
            predict_call_kwargs = mapped_args.get("predict_args", {})
            update_call_kwargs = mapped_args.get("update_args", {})

        self.particle_filter.predict(action, **predict_call_kwargs)
        if self.pf_interaction_mapper is not None:
            self.particle_filter.update(**update_call_kwargs)
        else:
            self.particle_filter.update(base_env_obs_float)

        self._last_base_env_obs_float = base_env_obs_float.copy()
        return (self._get_dict_obs(base_env_obs_float, info), reward, terminated,
                truncated, info)

    def _checked_weights(self, info: dict | None) -> np.ndarray:
        """The filter's weights, unless they carry no information.

        Dead = any non-finite entry, or a total mass that is not > 0. See the
        class docstring for why this is checked here and not in the encoders.
        """
        weights = np.asarray(self.particle_filter.weights, dtype=np.float64)
        finite = bool(np.all(np.isfinite(weights)))
        total = float(np.sum(weights)) if finite else float("nan")
        if finite and total > 0.0:
            return weights.astype(np.float32)

        self.dead_belief_count += 1
        what = ("non-finite weights" if not finite
                else f"total mass {total!r}")
        filter_name = type(self.particle_filter).__name__
        if self.on_dead_belief == "raise":
            raise RuntimeError(
                f"{filter_name} produced a dead belief ({what}) at PF reset "
                f"#{self._pf_reset_count}: every particle is refuted and the "
                "row carries no information. The encoders would read it as a "
                "plausible belief (CGF K': a uniform belief; Gaussian: a delta "
                "at the arena centre), so it is refused here. Fix the filter "
                "(floor the likelihood before normalising, as the Ant-Tag "
                "filters do with += 1e-300, or reset to uniform), or construct "
                "the wrapper with on_dead_belief='uniform' to substitute "
                "uniform weights and flag info['pf_dead_belief'].")
        if not self._warned_dead_belief:
            print(f"PFDictWithWeightsObservationWrapper: {filter_name} produced "
                  f"a dead belief ({what}); substituting uniform weights. "
                  "Counted in dead_belief_count; further occurrences are silent.",
                  flush=True)
            self._warned_dead_belief = True
        if info is not None:
            info["pf_dead_belief"] = True
        return np.full(weights.shape, 1.0 / weights.shape[0], dtype=np.float32)

    def _get_dict_obs(self, base_env_obs: np.ndarray, info: dict | None = None) -> dict:
        agent_obs = base_env_obs
        if self.obs_mask_indices is not None:
            agent_obs = base_env_obs.copy()
            agent_obs[self.obs_mask_indices] = 0.0
        return {
            "obs": agent_obs.astype(np.float32),
            "particles": self.particle_filter.particles.astype(np.float32),
            "weights": self._checked_weights(info),
        }
