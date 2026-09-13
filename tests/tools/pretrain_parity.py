"""Bit-for-bit parity of a PRETRAINING or COLLECTION run between two checkouts (plan section 7,
checks E-pretrain and E-collect).

Runs one command on side A (usually the master worktree's script) and one on side B (the
branch's entry point or package command), both on CPU with a fixed thread count, then compares
every tensor of every checkpoint the run wrote, the checkpoint's recorded config / losses, and
any JSON side files (history, probe results). Prints EQUAL or the first differences.

    python3 tests/tools/pretrain_parity.py \\
        --side_a "<cwd>::<command>" --side_b "<cwd>::<command>" \\
        --out_a <folder the A run writes into> --out_b <folder the B run writes into>

Each side's command is run as given (shell=False after shlex.split) from its cwd with
OMP/MKL/OPENBLAS_NUM_THREADS fixed, CUDA hidden and wandb offline; ``--out_*`` are globbed
recursively for ``*.pt`` and ``*.json`` files, matched by relative path ignoring timestamped
folder names.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}|\d{8}_\d{6}")


def _run(spec: str, threads: int) -> None:
    cwd, command = spec.split("::", 1)
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
           "OPENBLAS_NUM_THREADS": str(threads), "CUDA_VISIBLE_DEVICES": "", "WANDB_MODE": "offline",
           "PYTHONWARNINGS": "ignore"}
    proc = subprocess.run(shlex.split(command), cwd=cwd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:] + "\n" + proc.stderr[-4000:])
        raise SystemExit(f"command failed on {cwd}: {command}")


def _files(root: Path) -> dict[str, Path]:
    out = {}
    for path in sorted(root.rglob("*")):
        # run_config.json is the command's own record (paths, git, threads); args.json is the
        # parsed namespace, whose flag spellings changed by decision 2 (--epochs -> --num_epochs,
        # --lr -> --learning_rate) and which now carries the encoder table's extra flags. The
        # checkpoint's `args` key is ignored for the same reason; every VALUE that matters is
        # compared through the checkpoint's `config` block and the history / probe files.
        if path.suffix in (".pt", ".json", ".npz") and path.name not in ("run_config.json", "args.json"):
            rel = _TIMESTAMP.sub("<ts>", str(path.relative_to(root)))
            out[rel] = path
    return out


def _flatten(obj, prefix="") -> dict:
    """Every tensor in a nested checkpoint, keyed by its path; non-tensors kept as values.
    Dataclasses (a TrainingConfig stored as an object) flatten like the dict of their fields,
    so a side that stores the same config as a plain dict compares equal."""
    flat = {}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    if torch.is_tensor(obj):
        flat[prefix] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            flat.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            flat.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        flat[prefix] = obj
    return flat


_IGNORED = re.compile(r"(^|\.)(args|command|run_directory|base_dir|output_root|git|threads|"
                      r"data_path|emd_matrix_path|pretraining|seconds|device|experiment_name|"
                      r"run_name|out_dir|init_from|source_checkpoint|"
                      # 7.2: TrainingConfig gained num_post_sab (default 2, the old fixed count);
                      # a master checkpoint lacks the key, a branch one records 2.
                      r"num_post_sab)($|\.|\[)")


def _compare_checkpoint(a: Path, b: Path) -> list[str]:
    fa = _flatten(torch.load(a, map_location="cpu", weights_only=False))
    fb = _flatten(torch.load(b, map_location="cpu", weights_only=False))
    problems = []
    for key in sorted(set(fa) | set(fb)):
        if _IGNORED.search(key):
            continue
        if key not in fa or key not in fb:
            problems.append(f"  key only on one side: {key}")
            continue
        va, vb = fa[key], fb[key]
        if torch.is_tensor(va) and torch.is_tensor(vb):
            if va.shape != vb.shape:
                problems.append(f"  SHAPE {key}: {tuple(va.shape)} vs {tuple(vb.shape)}")
            elif not torch.equal(va, vb):
                delta = (va.double() - vb.double()).abs().max().item()
                problems.append(f"  DIFF {key}: max|delta|={delta:.3e}")
        elif str(va) != str(vb):
            problems.append(f"  VALUE {key}: {str(va)[:60]!r} vs {str(vb)[:60]!r}")
    return problems


def _compare_json(a: Path, b: Path) -> list[str]:
    da, db = json.loads(a.read_text()), json.loads(b.read_text())
    fa, fb = _flatten(da), _flatten(db)
    return [f"  JSON {k}: {str(fa.get(k))[:60]!r} vs {str(fb.get(k))[:60]!r}"
            for k in sorted(set(fa) | set(fb))
            if not _IGNORED.search(k) and str(fa.get(k)) != str(fb.get(k))]


def _compare_npz(a: Path, b: Path) -> list[str]:
    """Every array member exactly (dtype, shape, values); the metadata JSON member field by
    field with the same ignore list (args, git, command)."""
    problems = []
    with np.load(a, allow_pickle=True) as za, np.load(b, allow_pickle=True) as zb:
        for key in sorted(set(za.files) | set(zb.files)):
            if key not in za.files or key not in zb.files:
                problems.append(f"  member only on one side: {key}")
                continue
            if key == "metadata":
                fa, fb = _flatten(json.loads(str(za[key]))), _flatten(json.loads(str(zb[key])))
                problems += [f"  META {k}: {str(fa.get(k))[:60]!r} vs {str(fb.get(k))[:60]!r}"
                             for k in sorted(set(fa) | set(fb))
                             if not _IGNORED.search(k) and str(fa.get(k)) != str(fb.get(k))]
                continue
            va, vb = za[key], zb[key]
            if va.dtype != vb.dtype or va.shape != vb.shape:
                problems.append(f"  SHAPE/DTYPE {key}: {va.dtype}{tuple(va.shape)} vs {vb.dtype}{tuple(vb.shape)}")
            elif not np.array_equal(va, vb):
                delta = np.abs(va.astype(np.float64) - vb.astype(np.float64)).max()
                problems.append(f"  DIFF {key}: max|delta|={delta:.3e}")
    return problems


def compare(out_a: Path, out_b: Path) -> int:
    files_a, files_b = _files(out_a), _files(out_b)
    only_a = sorted(set(files_a) - set(files_b))
    only_b = sorted(set(files_b) - set(files_a))
    n_equal = 0
    problems = []
    for rel in sorted(set(files_a) & set(files_b)):
        compare_one = (_compare_checkpoint if rel.endswith(".pt") else
                       _compare_npz if rel.endswith(".npz") else _compare_json)
        found = compare_one(files_a[rel], files_b[rel])
        if found:
            problems.append(f"{rel}:")
            problems.extend(found[:12])
        else:
            n_equal += 1
    for rel in only_a:
        problems.append(f"only on side A: {rel}")
    for rel in only_b:
        problems.append(f"only on side B: {rel}")
    if not any(rel.endswith((".pt", ".npz")) for rel in set(files_a) & set(files_b)):
        problems.append("no checkpoint (.pt) or dataset (.npz) present on both sides -- nothing was compared")
    if problems:
        print(f"DIFFERENT: {n_equal} files identical, {len(problems)} problem lines")
        print("\n".join(problems))
        return 1
    print(f"EQUAL: {n_equal} files identical (checkpoint tensors, recorded config, JSON side files)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--side_a", required=True, help="'<cwd>::<command>' for side A")
    parser.add_argument("--side_b", required=True, help="'<cwd>::<command>' for side B")
    parser.add_argument("--out_a", required=True, type=Path)
    parser.add_argument("--out_b", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--compare_only", action="store_true", help="Skip running; compare the folders.")
    args = parser.parse_args(argv)
    if not args.compare_only:
        _run(args.side_a, args.threads)
        _run(args.side_b, args.threads)
    return compare(args.out_a, args.out_b)


if __name__ == "__main__":
    raise SystemExit(main())
