"""Bit-identity check between two checkouts for one RL training command (check E).

Not a pytest target (no ``test_`` prefix): a command-line tool for the harness
centralisation (``refactor_plans.md``, section 3, checks E and F). Given two checkouts of
``set_transformer`` -- normally a ``git worktree`` of master and the working branch -- it
runs the SAME numbered script with the SAME flags in each, with explicit log and model paths
under a scratch folder, and compares what a run leaves behind:

* ``evaluations.npz`` (every array, exactly);
* every tensor in the saved agent zip (``policy.pth`` and ``pytorch_variables.pth``, read
  straight out of the zip so no class has to be importable), for the final model and for
  every checkpoint zip;
* the VecNormalize statistics (observation means / variances / counts, return statistics);
* ``run_config.json`` key by key (check F in miniature), ignoring paths and git provenance.

Each side runs with the scratch folder as its working directory, so the cwd-relative
``runs/<subdir>/...`` record lands there and not beside the real runs. The script's own
directory is what the numbered scripts put first on ``sys.path``, so each side imports ITS
checkout's package.

    python3 tests/tools/rl_parity.py --a ../set_transformer_master --b . \\
        --domain odd_even --script 4_train_rl_gaussian.py --out /tmp/parity/oe_gaussian \\
        -- --variant oe50_short

    # the same checkout twice: is the run reproducible at all? (step 0 of check E)
    python3 tests/tools/rl_parity.py --a ../set_transformer_master --b ../set_transformer_master ...

The size flags of the short run (3,072 steps on the CPU, seed 0, rollout 512, eval every
1,024 steps with 2 episodes, a checkpoint at 2,048) are appended unless ``--no_preset``.
Exit status 0 when everything compared equal, 1 otherwise; ``--record_extra_ok`` lets the
B side's record carry extra keys (the shared trainer records more than the scripts did).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

PRESET = ["--total_timesteps", "3072", "--ppo_n_steps", "512", "--batch_size", "64",
          "--n_epochs", "2", "--eval_freq", "1024", "--n_eval_episodes", "2",
          "--save_freq", "2048", "--device", "cpu", "--seed", "0"]

RECORD_IGNORE = {"log_dir", "model_save_path", "git", "run_directory", "output_root"}


def run_side(label: str, checkout: Path, domain: str, script: str, flags: list[str],
             out: Path, encoder_stem: str) -> dict:
    side = out / label
    cwd = side / "cwd"
    cwd.mkdir(parents=True, exist_ok=True)
    log_dir = side / "logs"
    model_path = side / "models" / f"{encoder_stem}_agent.zip"
    script_path = checkout / "experiments" / domain / script
    cmd = [sys.executable, str(script_path), *flags,
           "--log_dir", str(log_dir) + "/", "--model_save_path", str(model_path)]
    env = dict(os.environ)
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    (side / "command.txt").write_text(" ".join(cmd) + f"\ncwd={cwd}\n")
    with open(side / "stdout.txt", "w") as handle:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=handle, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"[{label}] {script} exited {proc.returncode}; see {side / 'stdout.txt'}")
    records = sorted(cwd.glob("runs/*/*/run_config.json"))
    return dict(label=label, log_dir=log_dir, model_path=model_path,
                record=records[0] if records else None)


def zip_tensors(zip_path: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    with zipfile.ZipFile(zip_path) as zf:
        for member in ("policy.pth", "pytorch_variables.pth"):
            if member not in zf.namelist():
                continue
            state = torch.load(io.BytesIO(zf.read(member)), map_location="cpu", weights_only=False)
            if isinstance(state, dict):
                for key, value in state.items():
                    if torch.is_tensor(value):
                        tensors[f"{member}:{key}"] = value
    return tensors


def compare_tensors(a: dict, b: dict, name: str, report: list) -> None:
    if set(a) != set(b):
        report.append((name, False, f"different tensor names: {sorted(set(a) ^ set(b))[:5]}"))
        return
    worst = 0.0
    bad = 0
    for key in a:
        if a[key].shape != b[key].shape or not torch.equal(a[key], b[key]):
            bad += 1
            if a[key].shape == b[key].shape:
                worst = max(worst, float((a[key].double() - b[key].double()).abs().max()))
    report.append((name, bad == 0, f"{len(a)} tensors" if bad == 0
                   else f"{bad} of {len(a)} tensors differ, max|delta|={worst:.3e}"))


def compare_npz(a: Path, b: Path, report: list) -> None:
    if not a.exists() or not b.exists():
        report.append(("evaluations.npz", False, "missing on one side"))
        return
    with np.load(a) as za, np.load(b) as zb:
        if set(za.files) != set(zb.files):
            report.append(("evaluations.npz", False, "different keys"))
            return
        bad = [k for k in za.files if not np.array_equal(za[k], zb[k])]
    report.append(("evaluations.npz", not bad, ", ".join(za.files) if not bad else f"differ: {bad}"))


def compare_vecnormalize(a: Path, b: Path, report: list) -> None:
    if not a.exists() or not b.exists():
        report.append(("vecnormalize.pkl", False, "missing on one side"))
        return
    with open(a, "rb") as fa, open(b, "rb") as fb:
        va, vb = pickle.load(fa), pickle.load(fb)

    def stats(v):
        out = {}
        rms = v.obs_rms if isinstance(v.obs_rms, dict) else {"obs": v.obs_rms}
        for key, r in rms.items():
            out[f"obs[{key}].mean"], out[f"obs[{key}].var"], out[f"obs[{key}].count"] = r.mean, r.var, r.count
        out["ret.mean"], out["ret.var"], out["ret.count"] = v.ret_rms.mean, v.ret_rms.var, v.ret_rms.count
        return out
    sa, sb = stats(va), stats(vb)
    bad = [k for k in sa if not np.array_equal(np.asarray(sa[k]), np.asarray(sb[k]))]
    report.append(("vecnormalize.pkl", not bad, f"{len(sa)} statistics" if not bad else f"differ: {bad}"))


def parse_renames(specs: list[str]) -> dict:
    """``old=new`` or ``old=!new`` (B stores the negation): A's key -> (B's key, convert)."""
    renames = {}
    for spec in specs or ():
        old, _, new = spec.partition("=")
        negate = new.startswith("!")
        renames[old] = (new.lstrip("!"), (lambda v: not v) if negate else (lambda v: v))
    return renames


def compare_records(a: Path | None, b: Path | None, report: list, extra_ok: bool,
                    renames: dict | None = None) -> None:
    if a is None or b is None:
        report.append(("run_config.json", False, "missing on one side"))
        return
    ra, rb = json.loads(a.read_text()), json.loads(b.read_text())
    ra_full, rb_full = dict(ra), dict(rb)
    renames = renames or {}
    # A key B stores under another name (--record_rename): compare through the mapping.
    renamed_bad = [k for k, (new, conv) in renames.items()
                   if k in ra and (new not in rb or rb[new] != conv(ra[k]))]
    ra = {k: v for k, v in ra.items() if k not in renames}
    rb = {k: v for k, v in rb.items() if k not in {new for new, _ in renames.values()}}
    differ = renamed_bad + [k for k in ra if k not in RECORD_IGNORE and k in rb and ra[k] != rb[k]]
    missing = [k for k in ra if k not in RECORD_IGNORE and k not in rb]
    extra = sorted(k for k in rb if k not in RECORD_IGNORE and k not in ra)
    ok = not differ and not missing and (extra_ok or not extra)
    note = []
    if differ:
        note.append("differ: " + ", ".join(
            f"{k}={ra_full.get(k)!r}/{rb_full.get(renames.get(k, (k,))[0])!r}" for k in differ[:6]))
    if missing:
        note.append(f"missing in B: {missing}")
    if extra:
        note.append(f"extra in B: {extra}")
    report.append(("run_config.json", ok, "; ".join(note) or f"{len(ra)} keys equal"))


def compare(a: dict, b: dict, extra_ok: bool, renames: dict | None = None) -> list:
    report = []
    compare_npz(a["log_dir"] / "evaluations.npz", b["log_dir"] / "evaluations.npz", report)
    compare_tensors(zip_tensors(a["model_path"]), zip_tensors(b["model_path"]), "final model", report)
    ckpt_a = sorted((a["model_path"].parent / "checkpoints").glob("*_steps.zip"))
    ckpt_b = sorted((b["model_path"].parent / "checkpoints").glob("*_steps.zip"))
    if [p.name for p in ckpt_a] != [p.name for p in ckpt_b]:
        report.append(("checkpoints", False, f"{[p.name for p in ckpt_a]} vs {[p.name for p in ckpt_b]}"))
    else:
        for pa, pb in zip(ckpt_a, ckpt_b):
            compare_tensors(zip_tensors(pa), zip_tensors(pb), f"checkpoint {pa.name}", report)
    compare_vecnormalize(a["model_path"].parent / "vecnormalize.pkl",
                         b["model_path"].parent / "vecnormalize.pkl", report)
    compare_records(a["record"], b["record"], report, extra_ok, renames)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--a", required=True, type=Path, help="checkout A (e.g. the master worktree)")
    parser.add_argument("--b", required=True, type=Path, help="checkout B (e.g. the branch)")
    parser.add_argument("--domain", required=True, choices=["ant_tag", "odd_even"])
    parser.add_argument("--script", required=True, help="e.g. 4_train_rl_cgf.py")
    parser.add_argument("--out", required=True, type=Path, help="scratch folder for both runs")
    parser.add_argument("--encoder_stem", default=None,
                        help="<stem>_agent.zip model name; default from the script name")
    parser.add_argument("--no_preset", action="store_true", help="do not append the short-run size flags")
    parser.add_argument("--record_extra_ok", action="store_true",
                        help="B's run_config.json may carry keys A's does not")
    parser.add_argument("--record_rename", action="append", default=[], metavar="OLD=NEW",
                        help="A's record key OLD is B's key NEW; OLD=!NEW when B stores the "
                             "negation (e.g. no_layer_norm=!ln). Repeatable.")
    parser.add_argument("--reuse", action="store_true",
                        help="skip a side whose final model already exists under --out")
    parser.add_argument("flags", nargs=argparse.REMAINDER, help="script flags after --")
    args = parser.parse_args(argv)
    flags = [f for f in args.flags if f != "--"]
    if not args.no_preset:
        flags = flags + PRESET
    stem = args.encoder_stem or args.script.replace("4_train_rl_", "").replace(".py", "")
    args.out.mkdir(parents=True, exist_ok=True)

    sides = []
    for label, checkout in (("a", args.a.resolve()), ("b", args.b.resolve())):
        model = args.out / label / "models" / f"{stem}_agent.zip"
        if args.reuse and model.exists():
            records = sorted((args.out / label / "cwd").glob("runs/*/*/run_config.json"))
            sides.append(dict(label=label, log_dir=args.out / label / "logs", model_path=model,
                              record=records[0] if records else None))
            print(f"[{label}] reusing {model}")
            continue
        print(f"[{label}] running {args.script} from {checkout} ...", flush=True)
        sides.append(run_side(label, checkout, args.domain, args.script, flags, args.out, stem))

    report = compare(sides[0], sides[1], args.record_extra_ok, parse_renames(args.record_rename))
    width = max(len(name) for name, _, _ in report)
    for name, ok, note in report:
        print(f"  {name:<{width}}  {'EQUAL' if ok else 'DIFFER'}  {note}")
    all_ok = all(ok for _, ok, _ in report)
    print(f"RESULT {'EQUAL' if all_ok else 'DIFFER'}  {args.domain} {args.script} {' '.join(flags)}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
