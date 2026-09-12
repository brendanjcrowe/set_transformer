"""LEGACY (moved here 2026-09-11 from experiments/ant_tag/4_train_rl_frozen.py; NOT importable as-is).

The May-2026 runnable path of that script: a pretrained PFSetTransformer encoder run as a numpy
processor INSIDE the env wrapper (PFPlusFeaturesObservationWrapper, also removed), PPO/SAC on
[obs ++ features]. Superseded by 4_train_rl_st.py --pretrained_st_model_path [--st_frozen], which
puts the encoder inside the SB3 policy and reloads it after PPO construction. Kept for reference
only; see to_be_deleted/README.md.
"""


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
    apply_reward_shaping: bool = True,
):
    """Return a callable that creates a wrapped AntTag env.

    apply_reward_shaping=False skips PFRewardShapingWrapper entirely, so
    Monitor sees the env's true sparse reward (-1/step, 0-and-terminate on
    tag). Use this for the eval env: CurriculumCallback only ever updates
    reward coefficients on the training env (there's no handle to the eval
    env), so a shaped eval env would silently report reward numbers stuck
    at their initial (dense, distance-based) coefficients for the entire
    run — making EvalCallback's "best_model" selection meaningless. Eval
    envs already fix visibility at the real POMDP radius regardless of
    training progress; skipping shaping applies that same "always real
    difficulty" principle to the reward too.
    """

    def _init():
        env = gym.make("pdomains-ant-tag-v0", rendering=False)
        env.reset(seed=seed + rank)
        particle_filter_kwargs = get_ant_tag_pf_kwargs(env)

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
            particle_filter_kwargs=particle_filter_kwargs,
            pretrained_st_processor=processor,
            num_particles=num_particles,
            pf_interaction_mapper=ant_tag_pf_interaction_mapper,
            # Mask obs[-2:] (target position) so the agent can only get
            # target info through ST features, never directly from obs.
            # The PF mapper still sees the unmasked obs for its updates.
            obs_mask_indices=[-2, -1],
        )

        if apply_reward_shaping:
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

    # Eval envs — always evaluate at real POMDP difficulty (radius=3.0) and
    # on the true sparse reward (no shaping, so the metric doesn't depend
    # on where the training curriculum currently is).
    eval_env_kw = dict(env_kw)
    eval_env_kw["initial_visibility_radius"] = 3.0
    eval_env_kw["apply_reward_shaping"] = False
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
