"""One pretraining command for every (domain, encoder, objective).

Batch 7.2 of the harness centralisation (``refactor_plans.md`` section 7, 2026-09-13)::

    python3 -m set_transformer.rl.pretrain --domain ant_tag --encoder st \\
        --objective reconstruction --data_path data/smart_pf_dataset.npz ...

``experiments/ant_tag/3_train_st.py`` is an entry point of this command with the objective
fixed to ``reconstruction`` and its historical defaults filled in.

What this module owns (the same for every objective): the command line around the objective
(``--domain`` / ``--encoder`` / ``--objective`` selection, ``--variant``, ``--seed``,
``--device``, placement under the root layout with ``--output_root`` / ``--base_dir`` /
``--experiment_name`` / ``--run_tag``, ``--dry_run``); the encoder's flags and their
resolution FROM THE SAME ``Encoder`` TABLE the RL trainer reads (``rl/encoders.py``), so a
pretraining run and an RL run with the same flags always agree on the encoder's geometry;
seeding; ``run_config.json`` with git provenance and thread count; and, for a learned encoder,
the proof that the produced checkpoint loads into a fresh RL extractor with max|delta| == 0.
What the objective owns: its inputs and its training loop (``rl/domains/base.py::Objective``).

Placement (unchanged from change 5.2): ``--base_dir`` as given, else
``<output root>/<domain>/<variant>/pretrain/`` with ``<experiment_name>/<run name>`` below it;
the domain and variant come from the flags or from the objective's inputs (a collected dataset
records both).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from set_transformer.rl import domains as _domains
from set_transformer.rl import encoders as _encoders
from set_transformer.rl import pretrain_objectives as _objectives
from set_transformer.rl import run_records
from set_transformer.rl.domains.base import Domain, Objective, PretrainContext, PretrainResult
from set_transformer.rl.pretrained_encoder import verify_matches_checkpoint


# ---------------------------------------------------------------------------
# Objectives available to a domain
# ---------------------------------------------------------------------------

def objectives_for(domain: Domain | None) -> dict[str, Objective]:
    """The generic objectives plus the ones ``domain`` declares; a name clash is an error."""
    table = dict(_objectives.GENERIC)
    if domain is not None:
        for name, objective in domain.pretraining.objectives.items():
            if name in table:
                raise ValueError(
                    f"domain {domain.name!r} declares objective {name!r}, which is also a "
                    "generic objective; give the domain's one another name")
            table[name] = objective
    return table


def default_objective_name(domain: Domain | None) -> str:
    if domain is not None and domain.pretraining.default_objective:
        return domain.pretraining.default_objective
    return _objectives.DEFAULT_OBJECTIVE


def print_objectives(domain: Domain | None) -> None:
    for name, objective in objectives_for(domain).items():
        scope = "generic" if name in _objectives.GENERIC else f"{domain.name} only"
        print(f"{name:<18} {scope:<14} {objective.description}")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _add_common_arguments(parser: argparse.ArgumentParser, domain: Domain | None,
                          objective: Objective, encoder_name: str | None) -> None:
    g = parser.add_argument_group("run")
    g.add_argument("--seed", type=int, default=0,
                   help="Seeds model init, the shuffle order and the train/val split. "
                        "Without it every invocation gets a different val set, so "
                        "best_val_loss is not comparable across runs.")
    g.add_argument("--device", type=str, default=None,
                   help="Torch device (cpu, cuda, cuda:1). Default: cuda when available, else "
                        "cpu. Pass cpu for a bit-for-bit comparison between two checkouts.")
    g.add_argument("--dry_run", action="store_true",
                   help="Resolve everything, create the run folder and write run_config.json, "
                        "then stop before training.")
    p = parser.add_argument_group("placement")
    p.add_argument("--experiment_name", type=str, default=None,
                   help="Folder under <base_dir> holding this objective's runs. Default: "
                        f"{objective.default_experiment_name(encoder_name or '<encoder>')}.")
    p.add_argument("--run_tag", type=str, default="",
                   help="Appended to the run folder's name.")
    p.add_argument("--base_dir", type=str, default=None,
                   help="Default: <output root>/<domain>/<variant>/pretrain/ -- see --output_root.")
    p.add_argument("--output_root", type=str, default=None,
                   help="Root of the shared run layout when --base_dir is not given: "
                        "$RL_BMDP_RUNS, else <parent repo>/runs when this checkout is a "
                        "submodule, else <checkout>/runs.")


def build_parser(domain: Domain | None, encoder: _encoders.Encoder, objective: Objective, *,
                 prog: str | None = None, selectors: bool = True) -> argparse.ArgumentParser:
    """The full command line for one (domain, encoder, objective)."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description=f"Pretrain the {encoder.name} encoder by {objective.name}"
                    + (f" on {domain.name}" if domain else "") + ". "
                    f"Objective: {objective.description}.")
    if selectors:
        parser.add_argument("--domain", choices=sorted(_domains.DOMAIN_NAMES),
                            default=None if domain is None else domain.name,
                            help="Owner of the variant. Default: read off the objective's "
                                 "inputs (a dataset records its variant and env id).")
        parser.add_argument("--encoder", choices=sorted(_encoders.ENCODERS), default=encoder.name)
        parser.add_argument("--objective", default=objective.name,
                            help="--list_objectives shows the choices for the domain.")
        parser.add_argument("--list_encoders", action="store_true")
        parser.add_argument("--list_objectives", action="store_true")
    parser.add_argument("--variant", type=str, default=None,
                        help="Registry key of the env the inputs were collected on. Default: the "
                             "objective's inputs. Places the output and gives the encoder's "
                             "variant-dependent defaults (the CGF t_bound rule).")
    parser.add_argument("--list_variants", action="store_true")
    _add_common_arguments(parser, domain, objective, encoder.name)
    if domain is not None:
        encoder.add_arguments(parser, domain)
    objective.add_arguments(parser, domain)
    return parser


def _select(argv: list[str]) -> argparse.Namespace:
    """First pass: which (domain, encoder, objective) the rest of the flags belong to."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--domain", choices=sorted(_domains.DOMAIN_NAMES), default=None)
    pre.add_argument("--encoder", choices=sorted(_encoders.ENCODERS), default=None)
    pre.add_argument("--objective", default=None)
    pre.add_argument("--variant", default=None)
    pre.add_argument("--list_encoders", action="store_true")
    pre.add_argument("--list_objectives", action="store_true")
    pre.add_argument("--list_variants", action="store_true")
    known, _ = pre.parse_known_args(argv)
    return known


def _resolve_domain(pre, objective: Objective, argv: list[str], parser_for_errors):
    """``--domain`` if given, else the owner of the variant the objective's inputs record."""
    if pre.domain:
        return _domains.get(pre.domain)
    # The objective's inputs: parse its flags alone (unknown flags are the encoder's).
    mini = argparse.ArgumentParser(add_help=False)
    objective.add_arguments(mini, None)
    try:
        known, _ = mini.parse_known_args(argv)
    except SystemExit:
        return None
    hint = objective.locate(known)
    variant = pre.variant or hint.get("variant")
    if not variant:
        return None
    try:
        return _domains.domain_of_variant(variant, None if pre.variant else hint.get("env_id"))
    except ValueError as exc:
        parser_for_errors.error(f"{exc}; pass --domain")


def _resolve_device(args) -> str:
    return args.device or ("cuda" if torch.cuda.is_available() else "cpu")


def _round_trip(encoder: _encoders.Encoder, args, result: PretrainResult, weighted: bool) -> None:
    """Prove the produced checkpoint is what ``rl/train.py --pretrained_path`` will load: build
    a fresh RL extractor from the resolved flags, load the file, compare every encoder tensor."""
    path = result.rl_checkpoint
    if path is None or not encoder.learned:
        return
    kwargs = encoder.extractor_kwargs(args)
    kwargs[encoder.extractor_class.PRETRAINED_PATH_KWARG] = str(path)
    if "weight_channel" in kwargs:
        kwargs["weight_channel"] = bool(weighted)
    space = gym.spaces.Dict({
        "obs": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
        "particles": gym.spaces.Box(-np.inf, np.inf, (args.num_particles, args.dim_particles),
                                    np.float32),
        "weights": gym.spaces.Box(0.0, 1.0, (args.num_particles,), np.float32),
    })
    extractor = encoder.extractor_class(space, **kwargs)
    verify_matches_checkpoint(extractor.reference_state(str(path)), extractor.encoder_state_dict(),
                              str(path), label=f"{encoder.extractor_class.__name__} encoder")


def main(argv: Sequence[str] | None = None, *, domain: Domain | str | None = None,
         encoder: _encoders.Encoder | str | None = None, objective: Objective | str | None = None,
         prog: str | None = None, fallback_domain: str | None = None) -> PretrainResult | None:
    """Parse, resolve, record, pretrain, verify.

    ``objective`` / ``encoder`` / ``domain`` given: an entry point's fixed choices. Not given:
    read ``--objective`` / ``--encoder`` / ``--domain`` from ``argv`` (``python -m
    set_transformer.rl.pretrain``); a missing ``--domain`` is looked up from the objective's
    inputs. ``fallback_domain``: the domain whose encoder defaults apply when the inputs record
    no variant AND the output is placed with ``--base_dir`` (the ``3_train_st.py`` entry
    point passes ``ant_tag``: the script pretrained any dataset that way, with no env to ask).
    Returns the :class:`PretrainResult`, or None after a listing or ``--dry_run``.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    errors = argparse.ArgumentParser(prog=prog)
    pre = _select(argv)
    if pre.list_encoders:
        _encoders.print_encoders()
        return None

    # -- the objective, then the domain (which may come from the objective's inputs) ----------
    if domain is not None:
        domain = _domains.get(domain)
    chosen_objective = objective or pre.objective
    if chosen_objective is None:
        provisional = domain or (_domains.get(pre.domain) if pre.domain else None)
        chosen_objective = default_objective_name(provisional)
    if isinstance(chosen_objective, Objective):
        objective_record = chosen_objective
    else:
        # Resolve against the generic table first; a domain-declared one needs the domain.
        provisional = domain or (_domains.get(pre.domain) if pre.domain else None)
        table = objectives_for(provisional)
        if chosen_objective not in table:
            if provisional is None and pre.domain is None:
                errors.error(f"unknown objective {chosen_objective!r} among the generic ones "
                             f"{sorted(_objectives.GENERIC)}; a domain-declared objective needs "
                             "--domain")
            errors.error(f"unknown objective {chosen_objective!r} for domain "
                         f"{provisional.name}; choices: {sorted(table)}")
        objective_record = table[chosen_objective]
    if domain is None:
        domain = _resolve_domain(pre, objective_record, argv, errors)
    if pre.list_objectives:
        print_objectives(domain)
        return None
    if pre.list_variants:
        # Before the full parse: the objective's required inputs are not needed for a listing.
        if domain is None:
            errors.error("--list_variants needs --domain")
        domain.print_variants()
        return None

    # -- the encoder ---------------------------------------------------------------------------
    chosen_encoder = encoder or pre.encoder
    if chosen_encoder is None:
        if "-h" in argv or "--help" in argv:
            chosen_encoder = "st"
        else:
            errors.error("--encoder is required (e.g. --encoder st); --list_encoders lists them")
    encoder = _encoders.get(chosen_encoder)
    if not encoder.learned and not ("-h" in argv or "--help" in argv):
        errors.error(f"encoder {encoder.name!r} has no parameters to pretrain "
                     f"({encoder.extractor_class.__name__} is analytic); the learned encoders "
                     f"are {[e for e in _encoders.ENCODERS if _encoders.get(e).learned]}")
    if domain is None:
        if "-h" in argv or "--help" in argv:
            domain = _domains.get(sorted(_domains.DOMAIN_NAMES)[0])
        elif fallback_domain is not None and any(
                t == "--base_dir" or t.startswith("--base_dir=") for t in argv):
            domain = _domains.get(fallback_domain)
        else:
            errors.error("--base_dir not given and the dataset records no variant: pass "
                         "--variant <registry key> (and --domain) or --base_dir <folder>")

    parser = build_parser(domain, encoder, objective_record, prog=prog, selectors=True)
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        # 7.3: a domain may default to its own objective (Odd-Even: belief_kl), so a flag of
        # another objective (--data_path) is "unrecognized" for a reason worth stating.
        hint = ("" if (objective is not None or pre.objective) else
                f" (--objective was not given, so {domain.name}'s default "
                f"{objective_record.name!r} applies; --list_objectives shows the others)")
        parser.error(f"unrecognized arguments: {' '.join(unknown)}{hint}")
    if args.list_variants:
        domain.print_variants()
        return None

    # -- variant and placement -------------------------------------------------------------------
    hint = objective_record.locate(args)
    if args.variant is None:
        args.variant = hint.get("variant")
    if args.variant is not None and args.variant not in domain.variants:
        parser.error(f"--variant {args.variant!r} is not a {domain.name} variant "
                     f"({domain.variant_names()})")
    if args.experiment_name is None:
        args.experiment_name = objective_record.default_experiment_name(encoder.name)
    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        if not args.variant:
            # 7.3: an objective without inputs (the exact posterior rolls the env) has nothing
            # to read a variant from; say so instead of blaming a dataset it does not take.
            what = ("the dataset records no variant" if hasattr(args, "data_path")
                    else f"objective {objective_record.name!r} has no inputs to read one from")
            parser.error(f"--base_dir not given and {what}: pass "
                         "--variant <registry key> (and --domain) or --base_dir <folder>")
        experiment_dir = run_records.pretrain_dir(domain.name, args.variant, args.experiment_name,
                                                  root=args.output_root)
        print(f"Output root layout: {experiment_dir}/ (domain {domain.name}, variant {args.variant})")
        base_dir = experiment_dir.parent

    # -- the pretraining side PRODUCES the checkpoint: no start mode here -------------------------
    for dest, value in ((encoder.pretrained_dest, None), (encoder.frozen_dest, False),
                        (encoder.lr_scale_dest, 1.0), (encoder.unfreeze_dest, None)):
        if dest and not hasattr(args, dest):
            setattr(args, dest, value)
    # Seed before anything that draws: the split (via get_data_loader's generator), the
    # model init and the batch shuffle.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _resolve_device(args)

    # -- resolution: objective checks, objective inputs, then the encoder's own resolution -------
    objective_record.resolve_arguments(parser, args, domain, encoder)
    data = objective_record.prepare(parser, args, domain, encoder, device)
    for required in ("num_particles", "dim_particles", "arena_scale"):
        if getattr(args, required, None) is None:
            raise RuntimeError(f"objective {objective_record.name!r} left args.{required} unset")
    encoder.resolve_arguments(parser, args, domain)

    # -- the run folder and its record ------------------------------------------------------------
    now = datetime.now()
    run_name = objective_record.run_name(args, now)
    if args.run_tag:
        run_name = f"{run_name}_{re.sub(r'[^A-Za-z0-9._-]', '_', args.run_tag)}"
    run_dir = Path(base_dir) / args.experiment_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Selector and placement flags are written explicitly below (resolved), not as parsed.
    run_config = {k: v for k, v in vars(args).items()
                  if k not in ("domain", "encoder", "objective", "device", "list_variants",
                               "list_encoders", "list_objectives", "dry_run", "output_root",
                               "base_dir", "experiment_name")}
    run_records.write_run_config(
        str(run_dir),
        domain=domain.name, encoder=encoder.name, objective=objective_record.name,
        experiment_name=args.experiment_name, run_name=run_name,
        run_directory=os.path.abspath(run_dir), base_dir=str(Path(base_dir).resolve()),
        output_root=None if args.base_dir else str(run_records.output_root(args.output_root)),
        device=device, command=" ".join([prog or "set_transformer.rl.pretrain", *argv]),
        git=run_records.git_provenance(), threads=run_records.thread_settings(),
        **run_config,
    )
    if args.dry_run:
        print("--dry_run: run_config.json written; not training.")
        return None

    ctx = PretrainContext(domain=domain, variant=args.variant, encoder=encoder, device=device,
                          run_dir=run_dir, base_dir=Path(base_dir),
                          experiment_name=args.experiment_name, run_name=run_name, data=data)
    result = objective_record.run(args, ctx)
    weighted = bool(getattr(data, "weighted", getattr(args, "weight_channel", True)))
    _round_trip(encoder, args, result, weighted)
    if objective_record.report is not None:
        objective_record.report(args, ctx, result)
    if result.summary:
        print("Pretraining summary: " + ", ".join(f"{k}={v}" for k, v in result.summary.items()))
    return result


if __name__ == "__main__":
    main()
