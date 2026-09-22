"""The task-head pretraining machinery every domain's ``task`` objective shares (batch 10.9,
2026-09-14; MOVED from ``rl/domains/hunt.py``, where batch 9.2 had ported it from
``src/hunt_tasks/pretrain/pretrain.py``).

An encoder under pretraining, built as the RL arm's own SB3 extractor, feeds a 3-layer head
that predicts labels the collector read off the env's hidden state (``Collection.snapshot_extras``).
What differs per domain is data, not code: which label arrays the file must hold, which one is
the ``obs`` passthrough, the head's output size, the loss and the metrics. Those come in as
arguments; the loop, the split, the early stopping, the best-epoch snapshot of encoder AND head,
the per-source metrics and the checkpoint format are written once here.

This module is NOT a generic objective (it is not in ``GENERIC``): a domain declares its own
``task`` :class:`~set_transformer.rl.domains.base.Objective` with its labels and loss and calls
:func:`run_task_training`. It imports nothing from ``rl/domains`` at module level, so a domain
module may import it at the top (the domains package imports the domain modules, which import
this; a module-level import back into ``rl/domains`` would close the cycle).

LATENT METRIC ALIGNMENT (2026-09-19; ``change_mds/online_alignment_2026-09-19.md``). ``--align_lambda > 0``
adds ``lambda * (1 - pearson_r)`` between the pairwise distances of the encoder's features and the
debiased Sinkhorn divergences between the same clouds, computed per batch from the batch itself
(``latent_alignment.OnlineAlignment``; ``--sinkhorn_blur`` in the encoder's normalised frame, the raw
cloud divided by the dataset's scale as the extractor does). The validation LOSS that picks the best
epoch stays the task loss alone (the Trainer's rule); ``val_align_r`` is reported beside it on a fixed
pair sample of the held-out rows. At the default lambda 0 the loop is the record's, untouched.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from set_transformer.latent_alignment import (LambdaRamp, OnlineAlignment, add_alignment_arguments,
                                              resolve_sinkhorn_blur)


class TaskData:
    """The dataset on the device, split episode-disjoint: states arrive in rollout order and
    neighbouring states within an episode are near duplicates, so a random split leaks and the
    validation metric reads far higher than the encoder deserves (the record's rule: the last
    ``val_frac`` of rows are validation).

    ``labels`` are the arrays the file must hold next to ``particles`` / ``weights``; ``obs_key``
    names the one handed to the extractor as its ``obs`` passthrough (hunt: ``agent``, Ant-Tag:
    ``ant``); ``optional`` arrays are loaded when present (``source_round`` since 10.8a: an imported
    or merged file marks every row's source, and the report splits its metrics by it).
    """

    def __init__(self, path: str, val_frac: float, device: torch.device, *, labels: tuple[str, ...],
                 obs_key: str, scale_default: float, collect_hint: str,
                 optional: tuple[str, ...] = ("source_round",), kind: str = "task dataset",
                 val_sources: tuple[int, ...] | None = None):
        with np.load(path, allow_pickle=True) as z:
            missing = [k for k in ("particles", "weights", *labels) if k not in z.files]
            if missing:
                raise ValueError(f"{path} is not a {kind}: missing arrays {missing} ({collect_hint})")
            self.metadata = json.loads(str(z["metadata"])) if "metadata" in z.files else {}
            present_optional = [k for k in optional if k in z.files]
            arrays = {k: np.asarray(z[k]) for k in ("particles", "weights", *labels, *present_optional)}
        n = len(arrays["particles"])
        n_val = int(val_frac * n)
        if n_val < 1 or n - n_val < 1:
            raise ValueError(f"{n} rows cannot be split with val_frac={val_frac}")
        self.obs_key = obs_key
        self.particle_scale = float(self.metadata.get("particle_scale", scale_default))
        self.n_train, self.n_val = n - n_val, n_val
        self.device = device
        self.train = {k: torch.as_tensor(v[:n - n_val]).to(device) for k, v in arrays.items()}
        self.val = {k: torch.as_tensor(v[n - n_val:]).to(device) for k, v in arrays.items()}
        self.num_particles = int(arrays["particles"].shape[1])
        self.dim_particles = int(arrays["particles"].shape[2])
        # 2026-09-19 (src/scripts/least_mass_gap/NOTES.md, wave 2): the validation LOSS that drives
        # ReduceLROnPlateau and picks the best epoch can be restricted to the tail rows whose
        # `source_round` is in `val_sources` (a DAgger file: 0 = scripted collection, 1.. = states the
        # policy of that round visited). The record validated on policy rows only; validating on the
        # mixed tail (43 % easy scripted rows on least-mass) cost the record's own code ~10 points of
        # held-out policy-row identify accuracy. The TRAINING rows and the held-out tail are unchanged;
        # the reported metrics still cover the whole tail and every source. None = the whole tail.
        self.val_sources = tuple(int(s) for s in val_sources) if val_sources else None
        self.val_loss_rows: torch.Tensor | None = None
        if self.val_sources is not None:
            if "source_round" not in arrays:
                raise ValueError(f"--val_sources {self.val_sources} but {path} has no source_round array "
                                 "(a collected file has one source; only imported / merged DAgger files "
                                 "mark their rows)")
            rows = np.flatnonzero(np.isin(arrays["source_round"][n - n_val:], self.val_sources))
            if len(rows) == 0:
                raise ValueError(f"no validation row has source_round in {self.val_sources}")
            self.val_loss_rows = torch.as_tensor(rows, device=device)

    def val_loss_batches(self, batch_size: int):
        """The validation rows the loss is computed over, in chunks: the whole tail (slices) or the
        `val_sources` rows (index tensors)."""
        if self.val_loss_rows is None:
            return (slice(i, min(i + batch_size, self.n_val)) for i in range(0, self.n_val, batch_size))
        rows = self.val_loss_rows
        return (rows[i:i + batch_size] for i in range(0, len(rows), batch_size))

    def obs(self, split: dict, index) -> dict:
        """What the RL extractor is handed: the passthrough observation, the RAW cloud (the
        extractor divides by arena_scale) and the weights."""
        return {"obs": split[self.obs_key][index], "particles": split["particles"][index],
                "weights": split["weights"][index]}

    def add(self, key: str, train: torch.Tensor, val: torch.Tensor) -> None:
        """A derived per-row tensor (a scaled label, soft targets computed from the cloud)."""
        if len(train) != self.n_train or len(val) != self.n_val:
            raise ValueError(f"derived array {key!r}: {len(train)}/{len(val)} rows, "
                             f"expected {self.n_train}/{self.n_val}")
        self.train[key] = train.to(self.device)
        self.val[key] = val.to(self.device)

    # -- the two hooks the generated sources below implement (2026-09-21, fresh layouts) -------
    def refresh_epoch(self, rng) -> None:
        """Called once at the top of every epoch. File rows are fixed, so this does nothing; the
        generated sources overwrite their training rows in place here."""

    def selection_sets(self) -> tuple[tuple[str, dict, object], ...]:
        """``(name, split, rows)`` per held-out set whose loss picks the best epoch and drives
        ``ReduceLROnPlateau``; their losses are averaged. ``rows`` is None for the whole split or
        an index tensor. One entry here: the file's own tail, exactly the rows the loop validated
        on before this hook existed."""
        return (("file", self.val, self.val_loss_rows),)

    def metrics_sets(self) -> tuple[tuple[str, dict], ...]:
        """``(prefix, split)`` per held-out set the end-of-run metrics are reported on; an empty
        prefix leaves the metric names as they were. One entry here: the file's tail."""
        return (("", self.val),)

    def batches(self, split: dict, rows, batch_size: int):
        """Chunks of one held-out set: slices over the whole split, or index tensors of ``rows``."""
        if rows is None:
            n = len(split[self.obs_key])
            return (slice(i, min(i + batch_size, n)) for i in range(0, n, batch_size))
        return (rows[i:i + batch_size] for i in range(0, len(rows), batch_size))

    def provenance(self) -> dict:
        """What the run record and the checkpoint say this data object was (``rl/pretrain.py``'s
        ``pretraining_run`` and ``metrics.json``); the generated sources add their own facts."""
        return {"data_source": "file", "val_select": "file"}


class GeneratedTaskData:
    """Training rows drawn fresh every epoch from a variant's layout generator, held-out rows drawn
    ONCE (2026-09-21; ``change_mds/fresh_layout_pretraining_2026-09-21.md`` section 1.2).

    The same interface :func:`run_task_training` reads off :class:`TaskData` -- ``train`` / ``val``
    dicts of tensors, ``n_train`` / ``n_val``, ``obs``, ``particle_scale``, ``num_particles``,
    ``dim_particles``, ``metadata`` -- so the loop is the file loop with one call added. The rows
    are the collector's layout, so every label the domain's loss and metrics read is present.
    """

    def __init__(self, generator: Callable, cfg, *, variant: str, env_id: str, obs_key: str,
                 k_choices: tuple[int, ...], rows_per_epoch: int, val_rows: int, seed: int,
                 particle_scale: float, device: torch.device, label_arrays: tuple[str, ...],
                 val_select: str = "generated"):
        self.generator, self.cfg = generator, cfg
        self.obs_key, self.device = obs_key, device
        self.particle_scale = float(particle_scale)
        self.k_choices = tuple(int(k) for k in k_choices)
        self.rows_per_epoch, self.val_rows, self.seed = int(rows_per_epoch), int(val_rows), int(seed)
        self.val_select = val_select
        self.val_sources = None
        self.val_loss_rows = None
        # The held-out set is drawn ONCE, from a stream of its own (seed + 1), so it is the same
        # rows for every epoch of a run and reproducible from the recorded seed alone. The
        # TRAINING rows are not drawn here: the loop calls refresh_epoch before its first epoch,
        # and drawing them twice would cost a second pass over 100,000 layouts for nothing.
        self.val = self._to_device(generator(cfg, self.k_choices, self.val_rows,
                                             np.random.default_rng(self.seed + 1)))
        self.train: dict = {}
        self.n_train, self.n_val = self.rows_per_epoch, self.val_rows
        self.num_particles = int(self.val["particles"].shape[1])
        self.dim_particles = int(self.val["particles"].shape[2])
        self.metadata = {"variant": variant, "env_id": env_id, "particle_scale": self.particle_scale,
                         "label_arrays": list(label_arrays), **self.provenance()}

    def _to_device(self, arrays: dict) -> dict:
        return {k: torch.as_tensor(v).to(self.device) for k, v in arrays.items()}

    def refresh_epoch(self, rng) -> None:
        """A whole new set of layouts for this epoch, in place (the loop's ``randperm`` and every
        tensor reference are rebuilt per epoch, so replacing the dict is enough)."""
        self.train = self._to_device(self.generator(self.cfg, self.k_choices, self.rows_per_epoch, rng))

    def val_loss_batches(self, batch_size: int):
        return self.batches(self.val, None, batch_size)

    def batches(self, split: dict, rows, batch_size: int):
        if rows is None:
            n = len(split[self.obs_key])
            return (slice(i, min(i + batch_size, n)) for i in range(0, n, batch_size))
        return (rows[i:i + batch_size] for i in range(0, len(rows), batch_size))

    def selection_sets(self) -> tuple[tuple[str, dict, object], ...]:
        return (("generated", self.val, None),)

    def metrics_sets(self) -> tuple[tuple[str, dict], ...]:
        return (("", self.val),)

    def obs(self, split: dict, index) -> dict:
        return {"obs": split[self.obs_key][index], "particles": split["particles"][index],
                "weights": split["weights"][index]}

    def add(self, key: str, train: torch.Tensor, val: torch.Tensor) -> None:
        raise NotImplementedError(
            "a derived per-row array cannot be carried on generated data: the training rows are "
            "replaced every epoch, so the derived tensor would go stale after epoch 0. Compute it "
            "inside the objective's loss / metrics instead.")

    def provenance(self) -> dict:
        return {"data_source": "fresh", "val_select": self.val_select,
                "fresh_rows_per_epoch": self.rows_per_epoch, "fresh_val_rows": self.val_rows,
                "fresh_seed": self.seed, "k_choices": list(self.k_choices),
                "generator_constants": self.cfg.to_dict()}


class MixedTaskData(GeneratedTaskData):
    """A file's rows AND fresh layouts in every epoch (the standalone scripts' "Option A": the
    file's agent positions are where a policy actually goes, the generated ones are uniform).

    The file's episode-disjoint tail stays held out and is never trained on; the generated
    held-out set is drawn once as in :class:`GeneratedTaskData`. Rows per epoch = the file's
    training rows + ``--fresh_rows_per_epoch``.
    """

    def __init__(self, file_data: TaskData, generator: Callable, cfg, *, variant: str, env_id: str,
                 k_choices: tuple[int, ...], rows_per_epoch: int, val_rows: int, seed: int,
                 device: torch.device, label_arrays: tuple[str, ...], val_select: str = "mean"):
        self.file = file_data
        super().__init__(generator, cfg, variant=variant, env_id=env_id, obs_key=file_data.obs_key,
                         k_choices=k_choices, rows_per_epoch=rows_per_epoch, val_rows=val_rows,
                         seed=seed, particle_scale=file_data.particle_scale, device=device,
                         label_arrays=label_arrays, val_select=val_select)
        self.metadata = {**file_data.metadata, **self.metadata}
        self.val_sources = file_data.val_sources
        self.dropped_file_arrays = tuple(k for k in file_data.train if k not in self.val)
        self.n_train = self.rows_per_epoch + file_data.n_train

    def _merge_train(self, generated: dict) -> None:
        """The epoch's rows: the generated ones followed by the file's training rows. The keys are
        the intersection, so a file array the generator does not produce (``step``, ``episode``,
        ``source_round``) is dropped rather than left ragged."""
        self.train = {k: torch.cat([v, self.file.train[k]])
                      for k, v in generated.items() if k in self.file.train}
        self.n_train = self.rows_per_epoch + self.file.n_train

    def refresh_epoch(self, rng) -> None:
        self._merge_train(self._to_device(
            self.generator(self.cfg, self.k_choices, self.rows_per_epoch, rng)))

    def selection_sets(self) -> tuple[tuple[str, dict, object], ...]:
        """``mean`` (the default, and what both standalone scripts did): the average of the file
        tail's loss and the generated set's. ``file`` / ``generated``: that one alone."""
        file_set = ("file", self.file.val, self.file.val_loss_rows)
        generated_set = ("generated", self.val, None)
        if self.val_select == "file":
            return (file_set,)
        if self.val_select == "generated":
            return (generated_set,)
        return (file_set, generated_set)

    def metrics_sets(self) -> tuple[tuple[str, dict], ...]:
        return (("", self.val), ("file_", self.file.val))

    def provenance(self) -> dict:
        record = {**super().provenance(), "data_source": "mixed"}
        file_data = getattr(self, "file", None)
        if file_data is not None:
            record["file_rows_per_epoch"] = int(file_data.n_train)
        return record


def parse_val_sources(text: str | None) -> tuple[int, ...] | None:
    """``--val_sources 1,2`` -> (1, 2); None / empty -> None (validate on the whole tail). Shared by every
    domain's task objective so the flag means one thing everywhere (a domain that forwards it to a file
    without ``source_round`` gets :class:`TaskData`'s refusal, not a silently ignored flag)."""
    if text is None or not str(text).strip():
        return None
    return tuple(int(s) for s in str(text).split(",") if s.strip())


class TaskEncoderWithHead(nn.Module):
    """extractor -> belief features (the obs passthrough dropped) -> 3-layer head."""

    def __init__(self, extractor: nn.Module, obs_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.extractor = extractor
        self.obs_dim = obs_dim
        d = extractor.features_dim - obs_dim
        self.head = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim))

    def features(self, obs: dict) -> torch.Tensor:
        return self.extractor(obs)[:, self.obs_dim:]

    def forward(self, obs: dict) -> torch.Tensor:
        return self.head(self.features(obs))


def build_task_extractor(args, space: gym.spaces.Dict, scale: float):
    """The encoder under pretraining, as the RL arm's own SB3 extractor class, built from the
    flags THROUGH THE SHARED ENCODER TABLE (rl/encoders.py): the same construction
    `rl/train.py --encoder <name>` performs."""
    from set_transformer.rl import encoders as _encoders   # noqa: PLC0415 - see odd_even.build_extractor
    encoder = _encoders.get(args.encoder)
    kwargs = encoder.extractor_kwargs(args)
    kwargs["arena_scale"] = float(scale)
    kwargs[encoder.extractor_class.PRETRAINED_PATH_KWARG] = None
    return encoder.extractor_class(space, **kwargs)


def extractor_geometry(extractor) -> dict:
    """The geometry record the extractor's loader checks (the extractor's own
    `checkpoint_config`, batch 10.4)."""
    return extractor.checkpoint_config()


def save_task_checkpoint(model: TaskEncoderWithHead, path: Path, args, epoch: int, val: dict,
                         config: dict) -> None:
    """The format each extractor's own loader reads, assembled by
    :func:`~set_transformer.rl.pretrained_encoder.encoder_checkpoint` from the extractor's
    `checkpoint_state()` / `checkpoint_config()` (batch 10.4). ``config`` is the objective's
    own record (objective, task, variant, ...); the shared facts are added here."""
    from set_transformer.rl.pretrained_encoder import encoder_checkpoint   # noqa: PLC0415 - cycle
    extractor = model.extractor
    torch.save(encoder_checkpoint(
        extractor,
        config={**config,
                "arena_scale": float(extractor.arena_scale), "encoder": args.encoder,
                "encoder_params": int(getattr(args, "encoder_params", 0))},
        head_state_dict={k: v.detach().cpu() for k, v in model.head.state_dict().items()},
        epoch=epoch, val=val, args=vars(args)), path)


#: ``--data_source`` (2026-09-21, fresh layouts): where the TRAINING rows of one epoch come from.
DATA_SOURCES = ("file", "fresh", "mixed")


def add_task_arguments(parser, *, dataset_help: str, fresh_layouts: bool = False) -> None:
    """The flags every task objective has: the data and the record's training settings.

    ``fresh_layouts``: the domain's variants carry a ``layout_generator``, so the group below is
    offered (2026-09-21). A domain that leaves it False keeps exactly the command line it had.
    """
    g = parser.add_argument_group("task objective: data")
    g.add_argument("--data_path", type=str, default=None,
                   help=f"{dataset_help} Default: the variant's dataset under the output root.")
    g.add_argument("--val_frac", type=float, default=0.1,
                   help="Last fraction of rows (rollout order, so episode-disjoint) held out.")
    g.add_argument("--val_sources", type=str, default=None,
                   help="Comma-separated source_round values (e.g. 1,2 = policy-visited rows of a DAgger file). "
                        "The validation LOSS that halves the learning rate and picks the best epoch is then "
                        "computed on the held-out rows from these sources only, as the record did (it validated "
                        "on policy rows). Training rows, the held-out tail and the reported per-source metrics "
                        "are unchanged. Default: the whole tail.")
    if fresh_layouts:
        # 2026-09-21 (change_mds/fresh_layout_pretraining_2026-09-21.md section 1.3). `file` is the
        # default, so every recorded command means exactly what it meant before.
        f = parser.add_argument_group(
            "task objective: fresh layouts (rows drawn from the variant's own reset rules instead "
            "of a collected file; a collected file freezes one layout per episode and the encoder "
            "memorises them -- most_var 0.371 -> 0.885 identify on fresh rows)")
        f.add_argument("--data_source", choices=DATA_SOURCES, default="file",
                       help="Where an epoch's TRAINING rows come from. file (the default): the "
                            "collected dataset, unchanged. fresh: layouts generated anew every "
                            "epoch from the variant's env rules (no dataset is read). mixed: the "
                            "file's training rows AND generated rows in every epoch.")
        f.add_argument("--fresh_rows_per_epoch", type=int, default=None,
                       help="Generated rows per epoch. Default: 100,000 under --data_source fresh; "
                            "the file's own training-row count under mixed (which keeps the "
                            "epoch-counted LR plateau and early stopping on the scale they were "
                            "tuned at -- halving rows per epoch once collapsed the LR to 2e-06).")
        f.add_argument("--fresh_val_rows", type=int, default=20000,
                       help="A FIXED generated validation set, drawn once from --fresh_seed + 1.")
        f.add_argument("--fresh_seed", type=int, default=None,
                       help="The generator's stream, separate from torch's. Default: --seed.")
        f.add_argument("--k_choices", type=str, default=None,
                       help="Comma list the number of live clusters is drawn from per generated "
                            "row, with repeats for weight. Default: the collector's own "
                            "(pick_target 2,3,4,5,5,5; collect_all 1,2,3,4,5,5,5).")
        f.add_argument("--val_select", choices=("file", "generated", "mean"), default=None,
                       help="Which held-out loss halves the learning rate and picks the best "
                            "epoch. Default: file under --data_source file, generated under "
                            "fresh, mean (the average of the two, what the standalone scripts "
                            "did) under mixed.")
    t = parser.add_argument_group("task objective: training (the record's settings)")
    t.add_argument("--num_epochs", type=int, default=120)
    t.add_argument("--patience", type=int, default=15,
                   help="Stop after this many epochs without a validation improvement.")
    t.add_argument("--batch_size", type=int, default=256)
    t.add_argument("--learning_rate", type=float, default=1e-3)
    t.add_argument("--head_hidden", type=int, default=256)
    # 2026-09-19: the alignment term (online targets), off at the default lambda 0.
    al = parser.add_argument_group("task objective: latent metric alignment (optional; online "
                                   "Sinkhorn targets, read only with --align_lambda > 0)")
    add_alignment_arguments(al, sinkhorn=True)     # --sinkhorn_blur: None -> the domain's default


def locate_from_dataset(args) -> dict:
    """Where the dataset's metadata says the run belongs (``Objective.locate``)."""
    if not args.data_path:
        return {"variant": None, "env_id": None}
    from set_transformer.rl.pretrain_objectives.reconstruction import dataset_metadata  # noqa: PLC0415
    meta = dataset_metadata(args.data_path)
    return {"variant": meta.get("variant"), "env_id": meta.get("env_id")}


def data_source_of(args) -> str:
    """``--data_source``, or ``file`` for a domain that does not offer the flag."""
    return str(getattr(args, "data_source", "file") or "file")


#: ``--fresh_rows_per_epoch`` when a ``fresh`` run leaves it out (a ``mixed`` run takes the file's
#: own training-row count instead, so its steps per epoch stay on the scale the schedules were
#: tuned at). Section 1.3 of the plan.
DEFAULT_FRESH_ROWS_PER_EPOCH = 100_000


def resolve_task_arguments(parser, args, domain, encoder, *, collect_hint: str,
                           fresh_layouts: bool = False) -> None:
    """``Objective.resolve_arguments``: a learned encoder, and -- unless the rows are generated --
    a dataset that exists (the variant's file under the root when ``--data_path`` is not given).

    ``fresh_layouts`` (2026-09-21): the domain offers ``--data_source``, so the checks branch on
    it. Under ``fresh`` no dataset is read at all and ``--variant`` is REQUIRED (there is no file
    to read one off); under ``mixed`` both the file and the generator apply.
    """
    if not encoder.learned:
        parser.error(f"the task objective needs a learned encoder (st | cgf | deepset | pointnet); "
                     f"{encoder.name!r} has no parameters to train")
    resolve_sinkhorn_blur(args, domain)          # 2026-09-19: the domain's default when omitted
    source = data_source_of(args) if fresh_layouts else "file"
    if source != "file":
        _resolve_fresh_arguments(parser, args, domain, source)
    if source == "fresh":
        if args.data_path:
            parser.error("--data_source fresh generates every training row from the variant's env "
                         "rules and reads no dataset; drop --data_path, or pass --data_source mixed "
                         "to train on the file's rows as well")
        return
    if args.data_path is None:
        if args.variant is None:
            parser.error("--data_path or --variant is needed: the task objective reads the "
                         "collected dataset")
        from set_transformer.rl import run_records   # noqa: PLC0415
        args.data_path = str(run_records.dataset_path(domain.name, args.variant, root=args.output_root))
        print(f"--data_path not given; the variant's dataset under the root: {args.data_path}")
    if not os.path.isfile(args.data_path):
        parser.error(f"dataset {args.data_path} does not exist ({collect_hint})")


def _resolve_fresh_arguments(parser, args, domain, source: str) -> None:
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
    """The seed of the per-epoch generator stream in :func:`run_task_training`: ``--fresh_seed + 2``
    (the held-out set is drawn from ``--fresh_seed + 1``), or ``--seed + 2`` for a domain whose task
    objective offers no ``--fresh_seed`` (Ant-Tag; its file-backed data ignores the stream anyway).

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


def check_variant_and_geometry(parser, args, data: TaskData) -> None:
    """``Objective.prepare``'s shared tail: the file's variant against ``--variant``, and the
    dataset-derived geometry on ``args`` so the encoder's own resolution can run."""
    recorded = data.metadata.get("variant")
    if recorded is not None and args.variant is not None and recorded != args.variant:
        parser.error(f"--variant {args.variant} but the dataset was collected on {recorded!r}")
    if args.variant is None:
        args.variant = recorded
    args.num_particles = data.num_particles
    args.dim_particles = data.dim_particles
    args.arena_scale = data.particle_scale


def task_run_name(args, now: datetime) -> str:
    return f"{now.strftime('%Y%m%d_%H%M%S')}_task_{args.encoder}_seed{args.seed}"


def run_task_training(args, ctx, *, data: TaskData, obs_dim: int, out_dim: int,
                      loss_fn: Callable[[torch.Tensor, dict], torch.Tensor],
                      metrics_fn: Callable[[torch.Tensor, dict], dict],
                      checkpoint_config: dict, label: str):
    """The training loop of src/hunt_tasks/pretrain/pretrain.py::main, moved (9.2, then here):
    Adam, halve the rate on a plateau, early stop, keep the best encoder WITH its matching head.

    ``loss_fn(y, batch)`` / ``metrics_fn(y, batch)`` see the head output and the batch dict (every
    loaded and derived array); ``checkpoint_config`` is written into both checkpoints; ``label``
    names the run in the log (``<variant>/task/<encoder>``).
    """
    from set_transformer.rl.domains.base import PretrainResult   # noqa: PLC0415 - cycle (module doc)
    args.encoder = ctx.encoder.name
    device = torch.device(ctx.device)
    run_dir = Path(ctx.run_dir)
    checkpoint_dir = Path(ctx.checkpoint_dir) if ctx.checkpoint_dir else run_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))

    space = gym.spaces.Dict({
        "obs": gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32),
        "particles": gym.spaces.Box(-np.inf, np.inf, (data.num_particles, data.dim_particles), np.float32),
        "weights": gym.spaces.Box(0.0, 1.0, (data.num_particles,), np.float32),
    })
    extractor = build_task_extractor(args, space, data.particle_scale)
    model = TaskEncoderWithHead(extractor, obs_dim=obs_dim, out_dim=out_dim, hidden=args.head_hidden).to(device)
    n_encoder = sum(p.numel() for p in extractor.encoder_parameters())
    args.encoder_params = int(n_encoder)
    print(f"[{label}] train={data.n_train:,} val={data.n_val:,} "
          f"encoder params={n_encoder:,} head out={out_dim} device={device}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)

    n_val_loss = data.n_val if data.val_loss_rows is None else int(len(data.val_loss_rows))
    if data.val_sources is not None:
        print(f"[{label}] validation loss (LR schedule + best epoch) on the {n_val_loss:,} held-out rows "
              f"with source_round in {data.val_sources}; metrics still reported on all {data.n_val:,}", flush=True)
    # 2026-09-21 (fresh layouts): rows and optimiser steps per epoch, said out loud, because both
    # epoch-counted schedules below (ReduceLROnPlateau patience 5, early stopping --patience) are
    # sensitive to it -- an ablation that dropped rows/epoch 2.7x collapsed the LR to 2e-06 by
    # epoch 54 on a loss that was merely slower.
    provenance = data.provenance() if hasattr(data, "provenance") else {"data_source": "file"}
    if provenance.get("data_source", "file") != "file":
        steps = -(-data.n_train // args.batch_size)
        print(f"[{label}] data_source={provenance['data_source']}: {data.n_train:,} rows/epoch "
              f"({steps:,} steps at batch {args.batch_size}), generated "
              f"{provenance.get('fresh_rows_per_epoch', 0):,} of them per epoch from seed "
              f"{provenance.get('fresh_seed')}, k drawn from {provenance.get('k_choices')}; "
              f"selection on {', '.join(name for name, _, _ in data.selection_sets())} "
              f"(--val_select {provenance.get('val_select')})", flush=True)
        dropped = getattr(data, "dropped_file_arrays", ())
        if dropped:
            print(f"[{label}] the generator produces no {list(dropped)}, so those file arrays are "
                  "dropped from the training batches (the held-out sets keep theirs)", flush=True)

    # 2026-09-19 (online alignment): the latent metric-alignment term on the encoder's features, its
    # targets computed per batch from the batch's own clouds (divided by the dataset's scale, as the
    # extractor does). Off at the default lambda 0: the loop below is then the record's, untouched --
    # no Sinkhorn call, no pair draw, nothing added to the loss.
    align_lambda = float(getattr(args, "align_lambda", 0.0) or 0.0)
    align = ramp = align_record = None
    val_align_pairs = val_align_targets = None
    if align_lambda > 0:
        blur_source = resolve_sinkhorn_blur(args, getattr(ctx, "domain", None))   # a caller that skipped resolve
        align = OnlineAlignment(blur=float(args.sinkhorn_blur), scaling=float(args.sinkhorn_scaling),
                                metric=args.align_metric, pairs=args.align_pairs,
                                seed=int(getattr(args, "seed", 0) or 0))
        ramp = LambdaRamp(align_lambda, int(args.align_warmup_epochs), int(args.align_ramp_epochs))
        val_align_pairs = align.fixed_pairs(data.n_val, args.align_val_pairs)
        val_align_targets = align.targets(data.val["particles"] / data.particle_scale,
                                          data.val["weights"], val_align_pairs)
        align_record = {"lambda": align_lambda, "warmup_epochs": int(args.align_warmup_epochs),
                        "ramp_epochs": int(args.align_ramp_epochs), **align.record(),
                        "val_pairs": int(len(val_align_pairs[0]))}
        print(f"[{label}] latent alignment ON: lambda={align_lambda} ({args.align_metric}), warmup "
              f"{args.align_warmup_epochs} / ramp {args.align_ramp_epochs} epochs; online targets (blur "
              f"{args.sinkhorn_blur} [{blur_source}], scaling {args.sinkhorn_scaling}, {args.align_pairs} pairs "
              f"per batch); val_align_r over {len(val_align_pairs[0])} fixed pairs of the {data.n_val:,} held-out rows; "
              f"the validation loss stays the task loss alone", flush=True)
        if args.align_warmup_epochs + args.align_ramp_epochs >= args.num_epochs:
            print(f"[{label}] WARNING: warmup + ramp >= num_epochs: lambda never reaches its target, "
                  "so no checkpoint from this run is fully aligned", flush=True)

    def val_align_r() -> float:
        model.eval()
        feats = []
        with torch.no_grad():
            for s in range(0, data.n_val, 2048):
                feats.append(model.features(data.obs(data.val, slice(s, min(s + 2048, data.n_val)))))
        return align.correlation(torch.cat(feats), val_align_pairs, val_align_targets)

    def set_loss(split: dict, rows) -> float:
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for s in data.batches(split, rows, 2048):
                b = {k: v[s] for k, v in split.items()}
                count = (s.stop - s.start) if isinstance(s, slice) else int(len(s))
                tot += float(loss_fn(model(data.obs(split, s)), b)) * count
                n += count
        return tot / max(n, 1)

    def val_loss():
        """The loss that halves the learning rate and picks the best epoch: the average over the
        data object's selection sets. With a file's rows that is one set -- the held-out tail, or
        its `--val_sources` rows -- and the arithmetic is the loop's own, unchanged (2026-09-21)."""
        losses = {name: set_loss(split, rows) for name, split, rows in selection_sets}
        return float(np.mean(list(losses.values()))), losses

    selection_sets = data.selection_sets()

    best, best_epoch, best_state, best_head, bad, hist = float("inf"), None, None, None, 0, []
    t0 = time.time()
    # 2026-09-21 (fresh layouts): the generator's stream, separate from torch's, so a run's rows
    # are reproducible from --fresh_seed alone; a file-backed data object ignores it.
    epoch_rng = np.random.default_rng(fresh_epoch_seed(args))
    for ep in range(args.num_epochs):
        data.refresh_epoch(epoch_rng)
        model.train()
        perm = torch.randperm(data.n_train, device=device)
        run = 0.0
        lam_epoch = ramp(ep) if ramp is not None else 0.0
        run_align = run_r = 0.0
        n_r = 0
        for i in range(0, data.n_train, args.batch_size):
            j = perm[i:i + args.batch_size]
            b = {k: v[j] for k, v in data.train.items()}
            if align is None:
                loss = loss_fn(model(data.obs(data.train, j)), b)
            else:
                # the same forward, split so the features feed the alignment term too
                feats = model.features(data.obs(data.train, j))
                loss = loss_fn(model.head(feats), b)
                a_loss, r, _ = align.term(feats, b["particles"] / data.particle_scale, b["weights"])
                loss = loss + lam_epoch * a_loss
                run_align += float(a_loss.detach()) * len(j)
                if r is not None:
                    run_r += float(r.detach()) * len(j)
                    n_r += len(j)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += float(loss.detach()) * len(j)
        v, per_set = val_loss()
        sched.step(v)
        entry = {"epoch": ep, "train": run / data.n_train, "val": v}
        if len(per_set) > 1:
            # `val` is their mean (--val_select mean); each one is kept so the two can be read apart
            entry.update({f"val_{name}": value for name, value in per_set.items()})
        if align is not None:
            # `train` above includes lambda * align (the Trainer's epoch loss does the same)
            entry.update(align=run_align / data.n_train, align_r=(run_r / n_r if n_r else float("nan")),
                         align_lambda=lam_epoch, val_align_r=val_align_r())
        hist.append(entry)
        if v < best - 1e-6:
            best, best_epoch, bad = v, ep, 0
            # Snapshot the HEAD as well as the encoder: the metrics below are computed with
            # encoder + head, and a best-epoch encoder paired with a final-epoch head is a
            # combination that never existed during training (the record's ST dim_hidden=512
            # run reported a good best_val next to a near-chance identify_acc that way).
            best_state = {k: t.detach().clone() for k, t in extractor.state_dict().items()}
            best_head = {k: t.detach().clone() for k, t in model.head.state_dict().items()}
        else:
            bad += 1
        if ep % 10 == 0 or bad >= args.patience or ep == args.num_epochs - 1:
            print(f"  ep{ep:>3} train={run / data.n_train:.5f} val={v:.5f} best={best:.5f}", flush=True)
        (run_dir / "history.json").write_text(json.dumps(hist, indent=1))
        if bad >= args.patience:
            break
    if align is not None:
        # Model selection is by the task's validation loss, blind to alignment. If the best epoch
        # predates the end of the ramp, checkpoint_best.pt is only partially aligned; say so, and
        # record it (the Trainer's rule).
        align_record["lambda_at_best_epoch"] = float(ramp(best_epoch))
        align_record["val_align_r_at_best_epoch"] = float(hist[best_epoch]["val_align_r"])
        checkpoint_config = {**checkpoint_config, "alignment": align_record}
        if align_record["lambda_at_best_epoch"] < align_lambda:
            print(f"[{label}] WARNING: best-by-val-loss epoch {best_epoch} has align_lambda="
                  f"{align_record['lambda_at_best_epoch']:.3f} < target {align_lambda}; "
                  "checkpoint_best.pt is NOT fully aligned", flush=True)
    if provenance.get("data_source", "file") != "file":
        # 2026-09-21: a generated checkpoint carries what it was generated FROM (source, budget,
        # seed, k draw, the env constants). Added only when the rows were generated, so a
        # file-backed checkpoint is byte-for-byte the one this code produced before.
        checkpoint_config = {**checkpoint_config, "data": provenance}
    save_task_checkpoint(model, checkpoint_dir / "checkpoint_last.pt", args, len(hist) - 1,
                         {"loss": hist[-1]["val"]}, checkpoint_config)

    extractor.load_state_dict(best_state)
    model.head.load_state_dict(best_head)
    model.eval()

    def metrics_over(split: dict, rows_or_slices) -> dict:
        chunks = []
        with torch.no_grad():
            for idx in rows_or_slices:
                b = {k: v[idx] for k, v in split.items()}
                chunks.append(metrics_fn(model(data.obs(split, idx)), b))
        return {k: float(np.mean([c[k] for c in chunks])) for k in chunks[0]}

    def metrics_of(split: dict) -> dict:
        n = len(split[data.obs_key])
        return metrics_over(split, (slice(i, min(i + 4096, n)) for i in range(0, n, 4096)))

    # One entry per held-out set the data object reports on: the file's tail alone under
    # --data_source file (an empty prefix, so the metric names are the ones every recorded run
    # printed), the generated set plus the file's under mixed (2026-09-21).
    metrics = {}
    for prefix, split in data.metrics_sets():
        metrics.update({f"{prefix}{k}": v for k, v in metrics_of(split).items()})
    # 10.8a: the same metrics per source of the validation rows (a DAgger file mixes the scripted
    # collection with the states each round's policy visited; the record misread round 1 because
    # its validation set changed composition between rounds)
    by_source = {}
    source_split = next((split for _, split in data.metrics_sets() if "source_round" in split), None)
    if source_split is not None:
        src = source_split["source_round"].cpu().numpy()
        for r in np.unique(src):
            rows = np.flatnonzero(src == r)
            m = metrics_over(source_split, (torch.as_tensor(rows[i:i + 4096], device=data.device)
                                            for i in range(0, len(rows), 4096)))
            by_source[str(int(r))] = {"rows": int(len(rows)), **m}
        for r, m in by_source.items():
            print(f"  val source {r}: rows={m['rows']}  " + "  ".join(f"{k}={v:.4f}" for k, v in m.items() if k != "rows"))
    save_task_checkpoint(model, checkpoint_dir / "checkpoint_best.pt", args, best_epoch,
                         {"loss": best, **metrics}, checkpoint_config)
    record = dict(best_val=best, best_epoch=best_epoch, epochs=len(hist), minutes=(time.time() - t0) / 60,
                  val_sources=data.val_sources, val_loss_rows=n_val_loss,
                  val_metrics=metrics, val_metrics_by_source=by_source, alignment=align_record)
    if provenance.get("data_source", "file") != "file":
        # 2026-09-22 (plan 11.3 item 2): what the rows were generated from, for a generated run ONLY --
        # the rule the checkpoint record follows (rl/pretrain.py), so a file-backed run's metrics.json
        # keeps exactly the layout every recorded run has.
        record["data"] = provenance
    record["history"] = hist
    (run_dir / "metrics.json").write_text(json.dumps(record, indent=2))
    print(f"[{label}] best_val={best:.5f} (epoch {best_epoch})  "
          + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
          + f"  ({(time.time() - t0) / 60:.1f} min) -> {checkpoint_dir}", flush=True)
    checkpoints = {"best": checkpoint_dir / "checkpoint_best.pt", "last": checkpoint_dir / "checkpoint_last.pt"}
    return PretrainResult(run_dir=run_dir, rl_checkpoint=checkpoints["best"], checkpoints=checkpoints,
                          summary={"best_val_loss": best, "best_epoch": best_epoch,
                                   "epochs": len(hist), **metrics})
