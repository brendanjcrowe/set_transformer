"""Merge harness datasets into one (plan 10.8b, 2026-09-14): the DAgger step "old data + the beliefs the
trained policy visited". Generic over domains: the inputs must carry the same arrays (the collector's
``particles`` / ``weights`` / ``particle_scale`` / ``metadata`` plus the domain's label arrays), the same
``particle_scale`` and the same variant; they are concatenated in argument order.

Two per-row labels are added: ``source_round`` (kept where an input has it; an input without it gets the
next unused round number, in argument order -- the scripted collection is round 0, the states the round-0
policy visited are round 1, ...) and ``source_file`` (the input's index in the argument list). ``episode``
ids, where present, are offset per input so they stay unique. The metadata is the first input's, with
``n_samples``, ``inputs`` (path, sha256, rows, source rounds), ``merged`` and ``command`` written over it.

The rows are written with their UNITS in a random order (``--seed``; ``--no_shuffle`` keeps the input order): a unit
is an episode (the ``episode`` array where present, else the runs between ``step == 0`` resets, else one row), so
episodes stay contiguous and a tail validation split (``TaskData``: the last ``val_frac`` rows) has the composition
of the whole file. Without it the split is all the LAST input -- how the record misread its DAgger round 1
(PITFALLS.md section 13).

    python3 -m set_transformer.rl.merge_datasets a.npz b.npz --output_file <root>/hunt/least_mass/data/least_mass_pf_dataset_dagger1.npz
    python3 -m set_transformer.rl.merge_datasets a.npz b.npz --domain hunt --variant least_mass --run_tag dagger1 [--output_root ...]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from set_transformer.rl import run_records

#: Members that are not per-row arrays.
SCALARS = ("particle_scale", "particle_centre", "metadata")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load(path: Path) -> tuple[dict, dict, dict]:
    with np.load(path, allow_pickle=True) as z:
        arrays = {k: np.asarray(z[k]) for k in z.files if k not in SCALARS}
        scalars = {k: np.asarray(z[k]) for k in z.files if k in SCALARS and k != "metadata"}
        meta = json.loads(str(z["metadata"])) if "metadata" in z.files else {}
    return arrays, scalars, meta


def unit_ids(arrays: dict) -> np.ndarray:
    """One id per row naming its unit: the ``episode`` (where >= 0), else the run since the last
    ``step == 0`` reset, else the row itself."""
    n = len(arrays["particles"])
    if "episode" in arrays:
        ep = arrays["episode"].astype(np.int64)
        ids = np.where(ep >= 0, ep, -1 - np.arange(n))          # a collection row is its own unit
        _, ids = np.unique(ids, return_inverse=True)
        return ids
    if "step" in arrays:
        step = arrays["step"].astype(np.int64)
        starts = (step == 0) | (step < 0)
        starts[0] = True
        return np.cumsum(starts) - 1
    return np.arange(n)


def shuffle_units(arrays: dict, seed: int) -> dict:
    """Permute the rows unit by unit (rows of one unit stay contiguous and in order)."""
    ids = unit_ids(arrays)
    n_units = int(ids.max()) + 1
    rank = np.empty(n_units, np.int64)
    rank[np.random.default_rng(seed).permutation(n_units)] = np.arange(n_units)
    perm = np.lexsort((np.arange(len(ids)), rank[ids]))
    return {k: v[perm] for k, v in arrays.items()}


def merge(paths: Sequence[Path], command: str | None = None, *, shuffle: bool = True,
          seed: int = 0) -> tuple[dict, dict, dict]:
    """``(arrays, scalars, metadata)`` of the merged file. Raises ValueError on a mismatch."""
    if len(paths) < 2:
        raise ValueError("merge needs at least two datasets")
    loaded = [_load(Path(p)) for p in paths]
    base_arrays, base_scalars, base_meta = loaded[0]
    row_keys = sorted(k for k in base_arrays if k not in ("source_round", "source_file"))
    n_rows = [len(a["particles"]) for a, _, _ in loaded]
    for p, (arrays, scalars, meta) in zip(paths, loaded):
        keys = sorted(k for k in arrays if k not in ("source_round", "source_file"))
        if keys != row_keys:
            raise ValueError(f"{p}: arrays {keys} differ from the first input's {row_keys}")
        for k in row_keys:
            if arrays[k].shape[1:] != base_arrays[k].shape[1:]:
                raise ValueError(f"{p}: {k} has shape {arrays[k].shape[1:]} per row, the first input "
                                 f"{base_arrays[k].shape[1:]}")
            if len(arrays[k]) != len(arrays["particles"]):
                raise ValueError(f"{p}: {k} has {len(arrays[k])} rows, particles {len(arrays['particles'])}")
        if not np.isclose(float(scalars.get("particle_scale", meta.get("particle_scale", 1.0))),
                          float(base_scalars.get("particle_scale", base_meta.get("particle_scale", 1.0)))):
            raise ValueError(f"{p}: particle_scale differs from the first input's (PITFALLS.md section 4)")
        for field in ("variant", "env_id", "particle_filter_class"):
            if meta.get(field) is not None and base_meta.get(field) is not None and meta[field] != base_meta[field]:
                raise ValueError(f"{p}: {field}={meta[field]!r}, the first input has {base_meta[field]!r}")
    # source_round: keep, or assign the next unused number in argument order
    used = set()
    for arrays, _, _ in loaded:
        if "source_round" in arrays:
            used |= set(np.unique(arrays["source_round"]).astype(int).tolist())
    rounds, next_round = [], 0
    for arrays, _, _ in loaded:
        if "source_round" in arrays:
            rounds.append(arrays["source_round"].astype(np.int8))
        else:
            while next_round in used:
                next_round += 1
            rounds.append(np.full(len(arrays["particles"]), next_round, np.int8))
            used.add(next_round)
    out = {k: np.concatenate([a[k] for a, _, _ in loaded]) for k in row_keys}
    out["source_round"] = np.concatenate(rounds)
    out["source_file"] = np.concatenate([np.full(n, i, np.int16) for i, n in enumerate(n_rows)])
    if "episode" in out:
        offset, pieces = 0, []
        for arrays, _, _ in loaded:
            ep = arrays["episode"].astype(np.int64)
            has = ep >= 0
            pieces.append(np.where(has, ep + offset, -1))
            if has.any():
                offset += int(ep.max()) + 1
        out["episode"] = np.concatenate(pieces)
    if shuffle:
        out = shuffle_units(out, seed)
    meta = dict(base_meta)
    meta.update({
        "row_order": (f"units (episodes) shuffled with seed {seed}" if shuffle else "input order"),
        "n_samples": int(len(out["particles"])),
        "merged": True,
        "inputs": [{"path": str(Path(p).resolve()), "sha256": _sha256(Path(p)), "rows": int(n),
                    "source_rounds": sorted(set(np.unique(r).astype(int).tolist()))}
                   for p, n, r in zip(paths, n_rows, rounds)],
        "label_arrays": sorted(set(meta_labels(base_meta)) | {"source_round", "source_file"}),
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "command": command or " ".join(["set_transformer.rl.merge_datasets", *sys.argv[1:]]),
    })
    return out, base_scalars, meta


def meta_labels(meta: dict) -> list:
    labels = meta.get("label_arrays")
    return list(labels) if labels else []


def write(path: Path, arrays: dict, scalars: dict, meta: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays, **scalars, metadata=json.dumps(meta, default=str))
    return path


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    p.add_argument("inputs", nargs="+", help="Harness datasets (.npz), concatenated in this order.")
    p.add_argument("--output_file", type=str, default=None)
    p.add_argument("--domain", type=str, default=None)
    p.add_argument("--variant", type=str, default=None)
    p.add_argument("--run_tag", type=str, default="", help="Tag of the default output name under the root.")
    p.add_argument("--output_root", type=str, default=None)
    p.add_argument("--seed", type=int, default=0, help="Seed of the unit shuffle.")
    p.add_argument("--no_shuffle", action="store_true", help="Keep the input order (a tail split is then the last input).")
    return p


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> Path:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser(prog)
    args = parser.parse_args(argv)
    if args.output_file:
        out = Path(args.output_file)
    elif args.domain and args.variant:
        out = run_records.dataset_path(args.domain, args.variant, tag=args.run_tag, root=args.output_root)
    else:
        parser.error("--output_file, or --domain and --variant (+ --run_tag) for the default place under the root")
    if out.resolve() in {Path(p).resolve() for p in args.inputs}:
        parser.error("the output must not be one of the inputs")
    arrays, scalars, meta = merge([Path(p) for p in args.inputs],
                                  command=" ".join([prog or "set_transformer.rl.merge_datasets", *argv]),
                                  shuffle=not args.no_shuffle, seed=args.seed)
    write(out, arrays, scalars, meta)
    counts = {int(k): int(v) for k, v in zip(*np.unique(arrays["source_round"], return_counts=True))}
    print(f"Merged {len(args.inputs)} datasets -> {out} ({meta['n_samples']:,} rows; by source_round {counts})")
    return out


if __name__ == "__main__":
    main()
