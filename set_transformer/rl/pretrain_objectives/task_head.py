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


def add_task_arguments(parser, *, dataset_help: str) -> None:
    """The flags every task objective has: the data and the record's training settings."""
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
    t = parser.add_argument_group("task objective: training (the record's settings)")
    t.add_argument("--num_epochs", type=int, default=120)
    t.add_argument("--patience", type=int, default=15,
                   help="Stop after this many epochs without a validation improvement.")
    t.add_argument("--batch_size", type=int, default=256)
    t.add_argument("--learning_rate", type=float, default=1e-3)
    t.add_argument("--head_hidden", type=int, default=256)


def locate_from_dataset(args) -> dict:
    """Where the dataset's metadata says the run belongs (``Objective.locate``)."""
    if not args.data_path:
        return {"variant": None, "env_id": None}
    from set_transformer.rl.pretrain_objectives.reconstruction import dataset_metadata  # noqa: PLC0415
    meta = dataset_metadata(args.data_path)
    return {"variant": meta.get("variant"), "env_id": meta.get("env_id")}


def resolve_task_arguments(parser, args, domain, encoder, *, collect_hint: str) -> None:
    """``Objective.resolve_arguments``: a learned encoder, and a dataset that exists (the
    variant's file under the root when ``--data_path`` is not given)."""
    if not encoder.learned:
        parser.error(f"the task objective needs a learned encoder (st | cgf | deepset | pointnet); "
                     f"{encoder.name!r} has no parameters to train")
    if args.data_path is None:
        if args.variant is None:
            parser.error("--data_path or --variant is needed: the task objective reads the "
                         "collected dataset")
        from set_transformer.rl import run_records   # noqa: PLC0415
        args.data_path = str(run_records.dataset_path(domain.name, args.variant, root=args.output_root))
        print(f"--data_path not given; the variant's dataset under the root: {args.data_path}")
    if not os.path.isfile(args.data_path):
        parser.error(f"dataset {args.data_path} does not exist ({collect_hint})")


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

    def val_loss():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for s in data.val_loss_batches(2048):
                b = {k: v[s] for k, v in data.val.items()}
                count = (s.stop - s.start) if isinstance(s, slice) else int(len(s))
                tot += float(loss_fn(model(data.obs(data.val, s)), b)) * count
                n += count
        return tot / max(n, 1)

    best, best_epoch, best_state, best_head, bad, hist = float("inf"), None, None, None, 0, []
    t0 = time.time()
    for ep in range(args.num_epochs):
        model.train()
        perm = torch.randperm(data.n_train, device=device)
        run = 0.0
        for i in range(0, data.n_train, args.batch_size):
            j = perm[i:i + args.batch_size]
            b = {k: v[j] for k, v in data.train.items()}
            loss = loss_fn(model(data.obs(data.train, j)), b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += float(loss.detach()) * len(j)
        v = val_loss()
        sched.step(v)
        hist.append({"epoch": ep, "train": run / data.n_train, "val": v})
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
    save_task_checkpoint(model, checkpoint_dir / "checkpoint_last.pt", args, len(hist) - 1,
                         {"loss": hist[-1]["val"]}, checkpoint_config)

    extractor.load_state_dict(best_state)
    model.head.load_state_dict(best_head)
    model.eval()

    def metrics_over(rows_or_slices) -> dict:
        chunks = []
        with torch.no_grad():
            for idx in rows_or_slices:
                b = {k: v[idx] for k, v in data.val.items()}
                chunks.append(metrics_fn(model(data.obs(data.val, idx)), b))
        return {k: float(np.mean([c[k] for c in chunks])) for k in chunks[0]}

    metrics = metrics_over(slice(i, min(i + 4096, data.n_val)) for i in range(0, data.n_val, 4096))
    # 10.8a: the same metrics per source of the validation rows (a DAgger file mixes the scripted
    # collection with the states each round's policy visited; the record misread round 1 because
    # its validation set changed composition between rounds)
    by_source = {}
    if "source_round" in data.val:
        src = data.val["source_round"].cpu().numpy()
        for r in np.unique(src):
            rows = np.flatnonzero(src == r)
            m = metrics_over(torch.as_tensor(rows[i:i + 4096], device=data.device)
                             for i in range(0, len(rows), 4096))
            by_source[str(int(r))] = {"rows": int(len(rows)), **m}
        for r, m in by_source.items():
            print(f"  val source {r}: rows={m['rows']}  " + "  ".join(f"{k}={v:.4f}" for k, v in m.items() if k != "rows"))
    save_task_checkpoint(model, checkpoint_dir / "checkpoint_best.pt", args, best_epoch,
                         {"loss": best, **metrics}, checkpoint_config)
    (run_dir / "metrics.json").write_text(json.dumps(
        dict(best_val=best, best_epoch=best_epoch, epochs=len(hist), minutes=(time.time() - t0) / 60,
             val_sources=data.val_sources, val_loss_rows=n_val_loss,
             val_metrics=metrics, val_metrics_by_source=by_source, history=hist), indent=2))
    print(f"[{label}] best_val={best:.5f} (epoch {best_epoch})  "
          + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
          + f"  ({(time.time() - t0) / 60:.1f} min) -> {checkpoint_dir}", flush=True)
    checkpoints = {"best": checkpoint_dir / "checkpoint_best.pt", "last": checkpoint_dir / "checkpoint_last.pt"}
    return PretrainResult(run_dir=run_dir, rl_checkpoint=checkpoints["best"], checkpoints=checkpoints,
                          summary={"best_val_loss": best, "best_epoch": best_epoch,
                                   "epochs": len(hist), **metrics})
