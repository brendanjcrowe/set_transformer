"""Fresh layouts for the generic ``reconstruction`` objective, and the command-line group every
objective that draws rows from an env's reset rules shares (2026-09-22;
``change_mds/fresh_layout_reconstruction_2026-09-22.md``; the plan is
``change_mds/fresh_layout_reconstruction_plan_2026-09-21.md``, sections 3, 11.4 and 12.3).

WHY. A collected file freezes one layout per episode: only the agent moves and the cloud is
redrawn, so most_var's 3,600 layouts were each seen ~100 times and the encoder memorised them.
Drawing the TASK objective's rows anew every epoch took its identify accuracy 0.371 -> 0.885
(``change_mds/fresh_layout_pretraining_2026-09-21.md``). The reconstruction objective trains on
the same files, so the same memorisation applies to its encoder; this module lets its training
clouds be regenerated every epoch too.

HOW, with an UNCHANGED ``training/trainer.py`` and ``data/dataset.py``. The Trainer touches its
loaders in five places only: ``for batch in train_loader`` once per epoch, ``for batch in
val_loader`` per evaluation, the alignment setup (unwraps ``.dataset`` down to a ``POMDPDataset``
and reads the validation ``Subset.indices`` as rows of that base), and the checkpoint writer (the
base's frame). Nothing measures a loader. So:

* :class:`GeneratedParticleDataset` is a ``POMDPDataset`` whose rows are ONE tensor laid out as
  ``[file rows (mixed only) | a held-out generated block, drawn once | a training block]``; the
  training block is REPLACED IN PLACE by :meth:`GeneratedParticleDataset.refresh_training_rows`.
  The parent's constructor does the frame mapping and the weight checks, so every property the
  objective and the Trainer read is the parent's.
* :class:`RefreshingSampler` regenerates that block at the top of its iterator and then yields a
  permutation from its OWN torch generator. A DataLoader asks its sampler for an iterator when an
  epoch starts, so the refresh lands before the first ``__getitem__`` of every epoch and the
  global torch stream is untouched.
* :func:`build_generated_loaders` returns the two loaders as ``Subset``s over that one base, so
  the alignment setup's index arithmetic keeps meaning what it means for a file: the validation
  rows are base rows, and the held-out block never moves. Under ``mixed`` the file rows are split
  with the same seeded ``random_split`` call the plain path makes, so the file half of a mixed
  run is the plain run's rows.

The command-line group (:func:`add_fresh_layout_arguments`) and its resolution
(:func:`resolve_fresh_arguments`, :func:`fresh_epoch_seed`, :func:`resolve_k_choices`,
:func:`data_source_of`) MOVED here from ``task_head.py`` on 2026-09-22 so the task and the
reconstruction doors cannot drift apart in spelling or defaults; ``task_head`` re-exports them and
passes its own help strings, so the task door's ``--help`` is byte-identical to before.

What a domain must give: ``Domain.fresh_layouts(variant) -> FreshLayouts | None``
(``rl/domains/base.py``), the variant's generator bound to its live env config plus the
collector's cluster-count draw and frame. A domain that declares nothing offers no
``--data_source`` on reconstruction and keeps its command line to the byte.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, Subset, random_split

from set_transformer.data.dataset import POMDPDataset

#: ``--data_source``: ``file`` is today's path and the default for every door.
DATA_SOURCES = ("file", "fresh", "mixed")
#: ``--val_select``: which held-out rows form the validation loss (``mean`` = both blocks).
VAL_SELECTIONS = ("file", "generated", "mean")
#: ``--fresh_rows_per_epoch`` when a ``fresh`` run leaves it out (a ``mixed`` run takes the file's
#: own training-row count instead, so its steps per epoch stay on the scale the schedules were
#: tuned at). Plan section 1.3; the same number for both doors (11.4 item 2: the driver's TASK
#: conditions spell 300,000 themselves).
DEFAULT_FRESH_ROWS_PER_EPOCH = 100_000


# ---------------------------------------------------------------------------
# The flag group and its resolution (moved from task_head.py, 2026-09-22)
# ---------------------------------------------------------------------------

def add_fresh_layout_arguments(group, *, rows_help: str, val_select_help: str) -> None:
    """The six flags, into an argument group the caller created (the title and the two
    objective-specific help strings are the caller's, so each door's ``--help`` says what the
    flag does THERE; the spellings, choices and defaults are shared)."""
    group.add_argument("--data_source", choices=DATA_SOURCES, default="file",
                       help="Where an epoch's TRAINING rows come from. file (the default): the "
                            "collected dataset, unchanged. fresh: layouts generated anew every "
                            "epoch from the variant's env rules (no dataset is read). mixed: the "
                            "file's training rows AND generated rows in every epoch.")
    group.add_argument("--fresh_rows_per_epoch", type=int, default=None, help=rows_help)
    group.add_argument("--fresh_val_rows", type=int, default=20000,
                       help="A FIXED generated validation set, drawn once from --fresh_seed + 1.")
    group.add_argument("--fresh_seed", type=int, default=None,
                       help="The generator's stream, separate from torch's. Default: --seed.")
    group.add_argument("--k_choices", type=str, default=None,
                       help="Comma list the number of live clusters is drawn from per generated "
                            "row, with repeats for weight. Default: the collector's own "
                            "(pick_target 2,3,4,5,5,5; collect_all 1,2,3,4,5,5,5).")
    group.add_argument("--val_select", choices=VAL_SELECTIONS, default=None, help=val_select_help)


def data_source_of(args) -> str:
    """``--data_source``, or ``file`` for a domain that does not offer the flag."""
    return str(getattr(args, "data_source", "file") or "file")


def resolve_fresh_arguments(parser, args, domain, source: str) -> None:
    """The generated sources' own checks and defaults (2026-09-21): a variant with a generator,
    the row budget, the generator's seed, the cluster-count draw and which loss selects."""
    if args.variant is None:
        parser.error(f"--data_source {source} needs --variant: the layouts are drawn from one "
                     f"env's rules, and there is no dataset to read the variant off "
                     f"({domain.name} variants: {domain.variant_names()})")
    variant = domain.resolve(args.variant)
    if getattr(variant, "layout_generator", None) is None:
        parser.error(f"--data_source {source}: variant {args.variant!r} of domain {domain.name} "
                     "declares no layout_generator, so its layouts cannot be generated; use "
                     "--data_source file")
    if args.fresh_seed is None:
        args.fresh_seed = int(args.seed)
    if args.val_select is None:
        args.val_select = "generated" if source == "fresh" else "mean"
    if source == "fresh" and args.val_select != "generated":
        parser.error(f"--val_select {args.val_select} needs the file's held-out rows, and "
                     "--data_source fresh reads no file; use --val_select generated")
    if args.fresh_rows_per_epoch is None and source == "fresh":
        args.fresh_rows_per_epoch = DEFAULT_FRESH_ROWS_PER_EPOCH
    if args.fresh_rows_per_epoch is not None and args.fresh_rows_per_epoch < 1:
        parser.error("--fresh_rows_per_epoch must be at least 1")
    if args.fresh_val_rows < 1:
        parser.error("--fresh_val_rows must be at least 1")


def fresh_epoch_seed(args) -> int:
    """The seed of the per-epoch generator stream: ``--fresh_seed + 2`` (the held-out set is
    drawn from ``--fresh_seed + 1``), or ``--seed + 2`` for a domain whose objective offers no
    ``--fresh_seed`` (Ant-Tag's task; its file-backed data ignores the stream anyway).

    2026-09-22 (plan 11.3 item 1): this was ``fresh_seed or seed``, which treats an explicit
    ``--fresh_seed 0`` as "not given" and seeds the epoch stream from ``--seed`` while the held-out set
    and the checkpoint's record use 0, so the run was not reproducible from its own record.
    """
    fresh_seed = getattr(args, "fresh_seed", None)
    base = fresh_seed if fresh_seed is not None else (getattr(args, "seed", None) or 0)
    return int(base) + 2


def resolve_k_choices(args, default: tuple[int, ...]) -> tuple[int, ...]:
    """``--k_choices 2,3,5,5`` -> (2, 3, 5, 5); not given -> the collector's own draw for the task.
    Repeats are meaningful (they weight the draw), so the list is not deduplicated."""
    text = getattr(args, "k_choices", None)
    if text is None or not str(text).strip():
        return tuple(int(k) for k in default)
    choices = tuple(int(part) for part in str(text).split(",") if part.strip())
    if not choices:
        raise ValueError(f"--k_choices {text!r} names no cluster count")
    return choices


# ---------------------------------------------------------------------------
# The dataset, the sampler and the loaders (the reconstruction objective's data)
# ---------------------------------------------------------------------------

class GeneratedParticleDataset(POMDPDataset):
    """A ``POMDPDataset`` whose training rows are drawn from a layout generator and replaced in
    place every epoch, while its held-out rows are drawn once.

    Row layout: ``[0, n_file)`` the file's rows (``mixed`` only; the file dataset's already-mapped
    tensors, so they are the plain run's rows bit for bit), then ``[n_file, n_file + n_val)`` the
    held-out generated block from ``default_rng(fresh_seed + 1)``, then the training block of
    ``n_train`` rows from the epoch stream ``default_rng(fresh_seed + 2)`` (the constructor draws
    the first block from it; :meth:`refresh_training_rows` draws the next one). ``generator`` is
    ``(k_choices, n, rng) -> dict`` with ``particles`` ``[n, N, D]`` in RAW env units and
    ``weights`` ``[n, N]``; every other array of a row (the task labels) is dropped here.

    Only ``particles`` / ``weights`` of the parent's state are extended. The frame mapping
    ``(x - centre) / scale`` is the parent's, applied by its constructor to the generated blocks and
    reproduced op for op by :meth:`_to_frame` for every later block.
    """

    def __init__(self, generator: Callable, k_choices, *, n_val: int, n_train: int,
                 fresh_seed: int, particle_scale: float, particle_centre: float, weighted: bool,
                 device: str = "cpu", file_dataset: POMDPDataset | None = None) -> None:
        self.generator = generator
        self.k_choices = tuple(int(k) for k in k_choices)
        self.n_val, self.n_train, self.fresh_seed = int(n_val), int(n_train), int(fresh_seed)
        if self.n_val < 1 or self.n_train < 1:
            raise ValueError(f"n_val and n_train must be positive, got {n_val} / {n_train}")
        self.epoch_rng = np.random.default_rng(self.fresh_seed + 2)
        self.refreshes = 0
        held_out = generator(self.k_choices, self.n_val, np.random.default_rng(self.fresh_seed + 1))
        first = generator(self.k_choices, self.n_train, self.epoch_rng)
        particles = np.concatenate([np.asarray(held_out["particles"], np.float32),
                                    np.asarray(first["particles"], np.float32)])
        weights = None
        if weighted:
            weights = np.concatenate([np.asarray(held_out["weights"], np.float32),
                                      np.asarray(first["weights"], np.float32)])
        super().__init__(particles, weights, device, particle_scale=particle_scale,
                         particle_centre=particle_centre)
        self.n_file = 0
        if file_dataset is not None:
            problems = []
            for key in ("particle_scale", "particle_centre"):
                if float(getattr(file_dataset, key)) != float(getattr(self, key)):
                    problems.append(f"{key}: file {getattr(file_dataset, key)}, generated {getattr(self, key)}")
            if file_dataset.is_weighted != self.is_weighted:
                problems.append(f"weighted: file {file_dataset.is_weighted}, generated {self.is_weighted}")
            if (file_dataset.num_particles, file_dataset.particle_dim) != (self.num_particles, self.particle_dim):
                problems.append(f"set geometry: file {file_dataset.num_particles}x{file_dataset.particle_dim}, "
                                f"generated {self.num_particles}x{self.particle_dim}")
            if problems:
                raise ValueError("the file's rows and the generated rows cannot share one dataset:\n  "
                                 + "\n  ".join(problems))
            self.n_file = len(file_dataset)
            self.data = torch.cat([file_dataset.data, self.data])
            if self.weights is not None:
                self.weights = torch.cat([file_dataset.weights, self.weights])
        self.val_positions = range(self.n_file, self.n_file + self.n_val)
        self.train_positions = range(self.n_file + self.n_val, self.n_file + self.n_val + self.n_train)

    def _to_frame(self, raw) -> torch.Tensor:
        """The parent constructor's mapping, op for op, so a refreshed block and the first one
        (mapped by the parent) are computed identically."""
        data = torch.from_numpy(np.ascontiguousarray(raw, dtype=np.float32)).float()
        if self.particle_centre != 0.0:
            data = data - self.particle_centre
        if self.particle_scale != 1.0:
            data = data / self.particle_scale
        return data

    def refresh_training_rows(self, rng=None) -> None:
        """Replace the training block with a new draw (from the epoch stream, or ``rng``). The
        held-out block and the file rows are never touched."""
        rows = self.generator(self.k_choices, self.n_train, self.epoch_rng if rng is None else rng)
        block = slice(self.train_positions.start, self.train_positions.stop)
        particles = self._to_frame(rows["particles"])
        if tuple(particles.shape) != tuple(self.data[block].shape):
            raise ValueError(f"the generator returned {tuple(particles.shape)} rows for a training "
                             f"block of {tuple(self.data[block].shape)}")
        self.data[block] = particles
        if self.weights is not None:
            self.weights[block] = torch.from_numpy(np.asarray(rows["weights"], np.float32)).float()
        self.refreshes += 1


class RefreshingSampler(Sampler):
    """Yields a permutation of ``range(n)`` per epoch, from its OWN torch generator, after asking
    the base dataset for a new training block. A DataLoader calls ``iter(sampler)`` (through its
    ``BatchSampler``) when an epoch's iteration starts and before the first index is used, so the
    refresh lands at the top of every epoch, before the first ``__getitem__``."""

    def __init__(self, base: GeneratedParticleDataset, n: int, seed: int) -> None:
        self.base, self.n = base, int(n)
        self.generator = torch.Generator().manual_seed(int(seed))

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        self.base.refresh_training_rows()
        return iter(torch.randperm(self.n, generator=self.generator).tolist())


def build_generated_loaders(*, generator: Callable, k_choices, source: str, val_select: str,
                            rows_per_epoch: int, val_rows: int, fresh_seed: int, seed: int,
                            particle_scale: float, particle_centre: float, weighted: bool,
                            device: str, batch_size: int, file_dataset: POMDPDataset | None = None,
                            train_split: float = 0.8):
    """``(train_loader, val_loader, train_size, val_size, base)`` for ``--data_source fresh`` (no
    ``file_dataset``) or ``mixed``. Both loaders are ``Subset``s over ONE :class:`GeneratedParticleDataset`,
    ``num_workers`` 0 (a worker would hold a stale pickled copy of the rows).

    ``mixed``: the file rows are split by the SAME seeded ``random_split`` call ``get_data_loader``
    makes (``train_split``, ``seed``), so the file half is the plain run's rows; the training
    positions are the file's training rows plus the generated training block; the validation rows
    follow ``val_select`` (``file`` / ``generated`` / ``mean`` = both blocks in one loader, which the
    Trainer averages per batch, so ``mean`` weights the two by their batch counts).
    """
    if source not in ("fresh", "mixed"):
        raise ValueError(f"source must be fresh or mixed, got {source!r}")
    if source == "mixed" and file_dataset is None:
        raise ValueError("--data_source mixed needs the file dataset")
    if source == "fresh":
        file_dataset, val_select = None, "generated"
    base = GeneratedParticleDataset(
        generator, k_choices, n_val=val_rows, n_train=rows_per_epoch, fresh_seed=fresh_seed,
        particle_scale=particle_scale, particle_centre=particle_centre, weighted=weighted,
        device=device, file_dataset=file_dataset)
    file_train, file_val = [], []
    if file_dataset is not None:
        train_size = int(train_split * len(file_dataset))
        eval_size = len(file_dataset) - train_size
        split_generator = torch.Generator().manual_seed(int(seed)) if seed is not None else None
        train_part, val_part = random_split(file_dataset, [train_size, eval_size], generator=split_generator)
        file_train, file_val = [int(i) for i in train_part.indices], [int(i) for i in val_part.indices]
    train_positions = file_train + list(base.train_positions)
    if val_select == "generated":
        val_positions = list(base.val_positions)
    elif val_select == "file":
        val_positions = file_val
    elif val_select == "mean":
        val_positions = file_val + list(base.val_positions)
    else:
        raise ValueError(f"val_select must be one of {VAL_SELECTIONS}, got {val_select!r}")
    if not val_positions:
        raise ValueError("the validation loader would be empty")
    train_loader = DataLoader(Subset(base, train_positions), batch_size=batch_size, shuffle=False,
                              sampler=RefreshingSampler(base, len(train_positions), seed=int(seed)),
                              num_workers=0)
    val_loader = DataLoader(Subset(base, val_positions), batch_size=batch_size, shuffle=False,
                            num_workers=0)
    return train_loader, val_loader, len(train_positions), len(val_positions), base
