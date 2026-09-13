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
from pathlib import Path
from typing import Any

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
class PretrainContext:
    """What the shared pretraining command hands an :class:`Objective` when it runs it
    (``rl/pretrain.py``, plan section 7, batch 7.2). Plain values only."""

    domain: "Domain"
    #: Registry key the run is filed under (``None`` when the dataset records none and the
    #: user placed the output with ``--base_dir``).
    variant: str | None
    #: The :class:`~set_transformer.rl.encoders.Encoder` record being pretrained.
    encoder: Any
    #: Torch device string, resolved (``cpu`` / ``cuda`` / ``cuda:1``).
    device: str
    #: ``<base_dir>/<experiment_name>/<run_name>``: created, ``run_config.json`` already in it.
    run_dir: Path
    base_dir: Path
    experiment_name: str
    run_name: str
    #: Whatever :attr:`Objective.prepare` returned (loaded data, validated matrix, ...).
    data: Any = None
    #: Where the objective writes its checkpoints (batch 7.5): ``run_dir / "checkpoints"`` under
    #: the root layout, ``run_dir`` itself under the legacy layouts the entry points keep.
    #: ``None`` means ``run_dir``.
    checkpoint_dir: Path | None = None


@dataclass(frozen=True)
class PretrainResult:
    """What an :class:`Objective` hands back: where it wrote, and which file the RL side
    loads (``--pretrained_path`` of ``rl/train.py``) so the command can prove it round-trips."""

    run_dir: Path
    #: The checkpoint ``rl/train.py --pretrained_path`` takes; ``None`` if the objective
    #: produced nothing loadable (then no round-trip check runs).
    rl_checkpoint: Path | None
    #: Every file worth naming, by role (``best``, ``latest``, ``best_export``, ...).
    checkpoints: Mapping[str, Path] = field(default_factory=dict)
    #: Small numbers for the log (best validation loss, epoch, ...).
    summary: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Objective:
    """One pretraining objective: what an encoder is trained to do before RL.

    Two kinds exist. A GENERIC objective needs nothing a problem must compute for it
    (``reconstruction``: any collected particle dataset) and lives once, in
    ``rl/pretrain_objectives/``; every domain gets it. A PROBLEM-SPECIFIC objective needs
    something only that problem's env can provide (Odd-Even's exact posterior) and is
    declared in that problem's own domain module through :class:`Pretraining`, nowhere
    else. ``--objective`` of ``rl/pretrain.py`` offers the union for the chosen domain.

    The command owns the command line around the objective (domain / encoder / objective
    selection, seed, device, placement under the root layout, ``run_config.json``, the
    round-trip check of the produced checkpoint into the RL extractor). The objective owns
    its data and its training loop. Call order: :attr:`add_arguments` (parser assembly),
    :attr:`locate` (before the domain is known, from the objective's own inputs),
    :attr:`resolve_arguments` (cheap checks), :attr:`prepare` (data; sets the dataset-derived
    ``num_particles`` / ``dim_particles`` / ``arena_scale`` on ``args`` so the encoder's own
    resolution can run), then the encoder's resolution, then :attr:`run` and :attr:`report`.
    """

    name: str
    description: str
    #: ``add_arguments(parser, domain | None)``: the objective's flags. ``domain`` is None
    #: during the first, selecting parse.
    add_arguments: Callable
    #: ``run(args, context) -> PretrainResult``: the training itself.
    run: Callable
    #: ``run_name(args, now: datetime) -> str``: the run folder's name under
    #: ``<base_dir>/<experiment_name>/``.
    run_name: Callable
    #: ``locate(args) -> {"variant": str | None, "env_id": str | None}``: where the
    #: objective's inputs say the run belongs (the collector records both in a dataset's
    #: metadata). Used only when ``--domain`` / ``--variant`` are not given.
    locate: Callable = lambda args: {"variant": None, "env_id": None}
    #: ``resolve_arguments(parser, args, domain, encoder)``: checks that need no data.
    resolve_arguments: Callable = lambda parser, args, domain, encoder: None
    #: ``prepare(parser, args, domain, encoder, device) -> data``: load and validate the
    #: inputs; MUST leave ``args.num_particles``, ``args.dim_particles`` and
    #: ``args.arena_scale`` set.
    prepare: Callable = lambda parser, args, domain, encoder, device: None
    #: ``report(args, context, result) -> None``: optional end-of-run analysis (Odd-Even's
    #: mode-readout probe).
    report: Callable | None = None
    #: Default ``--experiment_name`` when the user gives none: ``f(encoder_name) -> str``.
    default_experiment_name: Callable = lambda encoder_name: encoder_name


@dataclass(frozen=True)
class Pretraining:
    """The pretraining objectives a problem declares for itself (see :class:`Objective`).
    The default, an empty record, means "the generic objectives only"."""

    #: ``objective name -> Objective``. A name that collides with a generic objective is an
    #: error at command assembly.
    objectives: Mapping[str, Objective] = field(default_factory=dict)
    #: The objective ``rl/pretrain.py`` picks for this domain when ``--objective`` is not
    #: given; ``None`` means the generic default (``reconstruction``).
    default_objective: str | None = None


@dataclass(frozen=True)
class Collection:
    """What the shared dataset collector (``rl/collect.py``, plan section 7, batch 7.4) lets a
    problem decide. The collector owns the loop that is the same for every problem: build the
    belief env exactly as the RL step does, play episodes, store the particle cloud and its
    weights after the reset and after every step, stop at ``--max_snapshots``, rebalance, write
    one ``.npz`` (``particles [S,N,D]`` float32 raw coordinates, ``weights [S,N]``,
    ``particle_scale``, a ``metadata`` JSON) and print the distribution report. The problem
    owns what differs: how an episode is driven, how rows are rebalanced, what extra facts and
    arrays the file carries. Every hook takes the parsed command line ``args`` and the
    ``options`` dict :attr:`resolve_arguments` returned.
    """

    #: ``add_arguments(parser)``: the problem's own collection flags (Ant-Tag: the locomotion
    #: policy, the pursuit / fully-observed mix, the visibility radius range, the spread
    #: thresholds; Odd-Even: the filter override and the step / ESS rebalancing knobs).
    add_arguments: Callable = lambda parser: None
    #: Defaults for the SHARED flags where the problem's script had its own: ``seed``,
    #: ``num_particles``, ``timesteps`` (``None`` = resolved from the variant later).
    defaults: Mapping[str, object] = field(default_factory=dict)
    #: ``resolve_arguments(parser, args, domain) -> options``: resolve the env id and filter
    #: (overrides included), fill ``args.timesteps`` / ``args.num_particles`` when None, refuse
    #: contradictions. MUST return ``{"env_id": str, "particle_filter_class": type, ...}``.
    resolve_arguments: Callable = lambda parser, args, domain: {}
    #: ``particle_scale(args, options) -> float``: the scale recorded in the file (the arena
    #: half-width the RL extractors divide by).
    particle_scale: Callable = lambda args, options: 1.0
    #: ``particle_centre(args, options) -> float | None``: added back to the stored particles so
    #: the file holds RAW coordinates (Odd-Even undoes the env's centring); None = stored as is.
    particle_centre: Callable = lambda args, options: None
    #: ``prepare(args, options) -> state``: before the env is built (Ant-Tag: the episode RNG,
    #: the locomotion policy and its VecNormalize stats, the trajectory-type counts).
    prepare: Callable = lambda args, options: None
    #: ``make_env(args, options, state) -> env``: the collection env, built through the same
    #: factory the RL step uses.
    make_env: Callable = lambda args, options, state: None
    #: ``begin_episode(args, options, state, env, episode) -> (reset_kwargs, act)``: called
    #: right before ``env.reset(**reset_kwargs)``; ``act(obs) -> action`` drives the episode.
    begin_episode: Callable = lambda args, options, state, env, episode: ({}, None)
    #: ``finish(args, options, state, n_snapshots)``: after the loop (Ant-Tag prints its mix).
    finish: Callable = lambda args, options, state, n_snapshots: None
    #: ``report(args, options, particles, weights, steps, stage)``: the distribution report;
    #: ``stage`` is ``"raw"`` (before rebalancing) or ``"final"`` (what is written).
    report: Callable = lambda args, options, particles, weights, steps, stage: None
    #: ``rebalance(args, options, particles, weights, steps) -> (particles, weights, steps)``;
    #: ``steps`` may come back None when the problem does not keep the index. Skipped under
    #: ``--no_rebalance``.
    rebalance: Callable = lambda args, options, particles, weights, steps: (particles, weights, steps)
    #: ``metadata_extras(args, options, particles, weights, steps) -> dict``: facts added to the
    #: metadata JSON next to the shared ones (Odd-Even: n_dist_size, episode_cap, the centre,
    #: the step-index range).
    metadata_extras: Callable = lambda args, options, particles, weights, steps: {}
    #: ``extra_arrays(args, options, particles, weights, steps) -> dict``: further members of the
    #: ``.npz`` (Odd-Even: ``particle_centre``, ``steps``).
    extra_arrays: Callable = lambda args, options, particles, weights, steps: {}
    #: The progress bar's label.
    progress_desc: str = "Collecting episodes"


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
    #: The pretraining objectives this problem declares for itself (see :class:`Pretraining`);
    #: the generic ones (reconstruction) need no declaration.
    pretraining: Pretraining = field(default_factory=Pretraining)
    #: How this problem's particle-filter datasets are collected (see :class:`Collection`);
    #: ``None`` for a problem without a collector.
    collection: "Collection | None" = None

    def variant_names(self) -> list[str]:
        return sorted(self.variants)
