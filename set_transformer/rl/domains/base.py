"""The ``Domain`` description: what the shared trainer needs to know about one problem.

Change 4 of the harness centralisation (``refactor_plans.md`` in the parent repo,
2026-09-12). Before it, each problem's knowledge -- how to build its belief env, which
quantities its curriculum anneals, which command-line flags are its own, what its defaults
are -- lived inside that problem's numbered training scripts, and a new problem meant a new
copy of the PPO loop. A :class:`Domain` puts that knowledge in one record, built at the
bottom of the problem's module in this package (``rl/domains/ant_tag.py`` ends with
``ANT_TAG = Domain(...)``, ``rl/domains/odd_even.py`` with ``ODD_EVEN = Domain(...)``), and
``rl/train.py`` reads the record instead of knowing either problem.

Every field is a plain value or a plain function, so the record can be inspected and tested
without an environment. The functions take the parsed command line (``args``, an
``argparse.Namespace``) where they need it, because the trainer's command line is assembled
from the domain's flags and the encoder's flags and only the domain knows what its flags mean.

Adding a problem: write ``rl/domains/<name>.py`` with its variant table, its env factory,
any wrappers, any :class:`~set_transformer.rl.curriculum.Schedule` declarations, and a
``Domain`` record at the bottom; register the name in ``rl/domains/__init__.py``. Nothing in
the record may refer to another domain.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from set_transformer.rl.curriculum import Schedule


@dataclass(frozen=True)
class Evaluation:
    """What the shared evaluation script (``rl/eval_true_reward.py``, change 5.1) lets a
    problem decide. Every field has a default, so a domain that only needs the generic
    protocol -- roll the checkpoint out, count an episode a success when its final ``info``
    says ``is_success`` or it ended before the cap -- leaves this record alone.

    The script owns everything else: the command line, the particle count read off the
    checkpoint, the cap from the registry, the env through :attr:`Domain.make_env` with
    ``training=False``, VecNormalize in inference mode, re-seeding after ``PPO.load`` and the
    rollout itself.
    """

    #: ``add_arguments(parser)``: evaluation-only flags of this problem (Ant-Tag: ``--no_mask``;
    #: Odd-Even: ``--collapse_step``, ``--baselines_only``).
    add_arguments: Callable = lambda parser: None
    #: ``options(args) -> dict``: the ``options`` :attr:`Domain.make_env` takes for the eval env.
    options: Callable = lambda args: {}
    #: Default ``--n_episodes``.
    default_n_episodes: int = 50
    #: ``False``: the vec env is seeded once, after ``PPO.load``, and the episodes run on.
    #: ``True``: ``env.seed(seed + episode)`` before EVERY reset (Odd-Even, where the hidden
    #: state is drawn at reset, so the seed IS the episode; PITFALLS.md section 2).
    reseed_per_episode: bool = False
    #: ``references(args, variant) -> object | None``: reference policies run BEFORE the
    #: checkpoint is loaded, on the episodes the policy will see (Odd-Even: the Bayes oracle
    #: and play-the-previous-observation). Whatever it returns is handed to :attr:`report`.
    references: Callable = lambda args, variant: None
    #: ``references_only(args) -> bool``: stop after the references; no checkpoint needed.
    references_only: Callable = lambda args: False
    #: ``report(episodes, references, args, variant, cap) -> None``: print the result.
    #: ``episodes`` is the list of :class:`~set_transformer.rl.eval_true_reward.Episode`
    #: records (per-step rewards and infos). ``None`` selects the script's success-rate report.
    report: Callable | None = None


@dataclass(frozen=True)
class Domain:
    """One problem, as the trainer sees it. See the module docstring."""

    #: Short name used in run directories, checkpoint prefixes and ``--domain``: ``ant_tag``.
    name: str
    #: Dimension of one particle in this problem's belief (Ant-Tag: 2, the target's x, y;
    #: Odd-Even: 1). Sizes encoder readouts before any env exists.
    particle_dim: int
    #: Variant key used when ``--variant`` is not given.
    default_variant: str

    # -- the registry -------------------------------------------------------------------------
    #: ``variant key -> Variant``; every Variant has at least ``env_id`` and ``particle_filter``.
    variants: Mapping[str, object]
    #: ``resolve(variant) -> Variant``, raising on an unknown key with the valid ones listed.
    resolve: Callable[[str], object]
    #: ``episode_cap(variant) -> int``, read from the gym registration.
    episode_cap: Callable[[str], int]
    #: ``run_subdir(encoder, variant) -> str``: the historical cwd-relative
    #: ``runs/<run_subdir>`` folder. Kept for the legacy layout and the run record.
    run_subdir: Callable[[str, str], str]
    #: ``add_variant_argument(parser, default=...)`` and ``print_variants()``.
    add_variant_argument: Callable
    print_variants: Callable[[], None]

    # -- the environment ---------------------------------------------------------------------
    #: ``make_env(variant, *, num_particles, particle_filter_class, seed, rank, monitor_dir,
    #: training, options) -> thunk``. ``training=False`` builds the EVAL env (Ant-Tag: the
    #: env's real visibility radius and no reward shaping); ``options`` is whatever
    #: :attr:`resolve_arguments` returned.
    make_env: Callable
    #: ``make_vec_env_from_fns(env_fns, n_envs)`` and ``make_vec_normalize(vec_env, training,
    #: norm_reward)``: SubprocVecEnv above one worker, VecNormalize on the ``obs`` key only.
    make_vec_env_from_fns: Callable
    make_vec_normalize: Callable
    #: ``particle_filter(args) -> type``: the filter class for this run (the variant's, or an
    #: override flag the domain offers).
    particle_filter: Callable

    # -- the domain's own command-line flags -----------------------------------------------------
    #: ``add_arguments(parser)``: flags that belong to the problem (Ant-Tag: reward shaping,
    #: curriculum strings, target masking, target speed; Odd-Even: the filter override).
    add_arguments: Callable
    #: ``resolve_arguments(parser, args) -> options``: fill the flags' defaults from the
    #: variant, refuse contradictions with ``parser.error``, and return the ``options`` dict
    #: :attr:`make_env` takes. Runs after the shared defaults (``num_particles``,
    #: ``arena_scale``) and before the encoder's resolution.
    resolve_arguments: Callable
    #: ``schedules(args) -> tuple[Schedule, ...]``: the annealed quantities for this run, with
    #: the waypoints the flags / variant resolved to. Empty for a domain without a curriculum.
    schedules: Callable[..., Sequence[Schedule]]
    #: ``run_config_extras(args) -> dict``: domain-derived values the run record should carry
    #: (Odd-Even: ``n_dist_size``, ``episode_cap``).
    run_config_extras: Callable

    # -- defaults --------------------------------------------------------------------------------
    default_num_particles: Callable[[str], int]
    default_arena_scale: Callable[[str], float]
    default_device: str
    default_total_timesteps: int
    #: ``encoder name -> {flag name: default}`` overrides for the encoder flag groups, plus the
    #: two CGF hooks ``t_bound_default(variant)`` and ``t_init_max_default(args)`` (see
    #: ``rl/encoders.py``). Encoders not listed use the generic defaults.
    encoder_defaults: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    #: ``encoder_callbacks(encoder_name) -> list``: callbacks the domain adds for one encoder
    #: (Odd-Even attaches its collapse sentinel to the ST arm). Empty by default.
    encoder_callbacks: Callable = lambda encoder_name: []
    #: What the shared evaluation script lets this problem decide (see :class:`Evaluation`).
    evaluation: Evaluation = field(default_factory=Evaluation)

    def variant_names(self) -> list[str]:
        return sorted(self.variants)
