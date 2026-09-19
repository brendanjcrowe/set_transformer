"""Run bookkeeping shared by every RL training script: run directory, run_config.json,
run_status.json, git provenance, stdout tee, and the root-level output layout.

Moved here 2026-09-12 (change 1d of the harness centralisation, ``refactor_plans.md`` in the
parent repo) from the copies in ``experiments/ant_tag/4_train_rl_{cgf,st,gaussian,pool}.py``,
``experiments/odd_even/4_train_rl_cgf.py`` (the run-status pair existed only there) and
``experiments/odd_even/2_collect_pf_dataset.py`` (git provenance). The bodies are unchanged;
the copies differed only in docstrings, one variable name, and the default ``run_subdir`` of
``default_run_dir``, which the Ant-Tag scripts now supply through a three-line wrapper each.
The scripts import these back under their old underscore names, so tests that patch or read
those names off a script still find the same objects.

``output_root`` / ``run_dir`` implement the root-level output layout (plan section 2b):
``<root>/<domain>/<variant>/rl/<encoder>/<timestamp>_seed<seed>[_<tag>]``. No script uses
them yet; the shared trainer (change 4) will. ``default_run_dir`` is today's cwd-relative
``runs/<run_subdir>/...`` and stays until then.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Run directory and run_config.json
# ---------------------------------------------------------------------------


def default_run_dir(seed: int, run_subdir: str, run_tag: str | None = None) -> str:
    """runs/<run_subdir>/<timestamp>_seed<seed>[_<run_tag>]/ -- RELATIVE to the cwd.

    Parallel runs with different seeds (and different env variants, via run_subdir) land
    in distinct, sortable, self-describing folders instead of overwriting a fixed path.
    run_tag is a free-form human label (e.g. "6M_vis0.2-0.5") for eyeballing
    `ls runs/<run_subdir>/` without opening any files. It is NOT the source of truth for
    what a run actually used -- that is run_config.json, written alongside with every CLI
    arg.

    No default for run_subdir here. The Ant-Tag scripts each keep a wrapper supplying
    theirs ("ant_tag_cgf", ...), because their train_* functions call this with only a
    seed when no --log_dir is given and then ignore the run_subdir they were passed -- a
    wart the Odd-Even trainer deliberately does not reproduce
    (tests/test_odd_even_pipeline.py::test_run_subdir_is_honoured).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{re.sub(r'[^A-Za-z0-9._-]', '_', run_tag)}" if run_tag else ""
    return os.path.join("runs", run_subdir, f"{timestamp}_seed{seed}{suffix}")


def write_run_config(run_dir: str, **config) -> None:
    """Dump every CLI arg for this run to run_dir/run_config.json -- the unambiguous
    source of truth for what a run used (total_timesteps, curriculum/reward/evasion
    schedules, env_id, etc.), since the run_tag in the directory name is just a
    human-readable hint, not a full record. Keys are sorted; values json cannot encode
    (classes, paths) are stringified."""
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "run_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=str, sort_keys=True)
    print(f"Run config saved to {path}")


# ---------------------------------------------------------------------------
# Git provenance
# ---------------------------------------------------------------------------


def git_provenance() -> dict:
    """Record WHICH CODE produced this run, for run_config.json (PITFALLS.md section 6).

    run_config.json pins every hyperparameter but not the source that read
    them, and that gap has already bitten this project: the
    ant_tag_cgf_cdens_terminal runs of 2026-08-26 finished at 17:30, and
    4_train_rl_cgf.py gained deterministic per-episode PF seeding at 18:39.
    Their run_config.json is byte-identical either side of that change, so
    nothing on disk says which behavior those checkpoints were trained with.

    HEAD alone would not close it -- both submodules are routinely dirty -- so
    the SHA-256 of `git diff HEAD` gives an uncommitted working tree a stable
    identity. Equal (head, diff_sha256) means the same code; a different
    diff_sha256 means something moved between two runs, even when both say
    "dirty". The porcelain status lines are kept as a human-readable hint of
    WHICH files were dirty.

    Never raises: a missing git, a detached worktree or a stripped checkout
    records an "error" string rather than killing a multi-hour training run.

    Repo paths are derived from THIS file: parents[2] is the set_transformer checkout
    (the same depth as experiments/<domain>/<script>.py had), parents[3]/pomdp-domains
    its sibling in the parent repo. In a standalone clone the second entry records an
    error, as before.
    """
    repos = {
        "set_transformer": Path(__file__).resolve().parents[2],
        "pomdp-domains": Path(__file__).resolve().parents[3] / "pomdp-domains",
    }

    def _git(repo: Path, *args: str) -> str:
        return subprocess.run(
            ("git", "-C", str(repo)) + args,
            capture_output=True, text=True, check=True, timeout=15,
        ).stdout

    provenance = {}
    for name, repo in repos.items():
        try:
            head = _git(repo, "rev-parse", "HEAD").strip()
            status = [line for line in
                      _git(repo, "status", "--porcelain").splitlines() if line]
            diff = _git(repo, "diff", "HEAD")
            provenance[name] = {
                "path": str(repo),
                "head": head,
                "dirty": bool(status),
                # Tracked-file modifications only; untracked content is not in
                # `git diff HEAD`, which is why the status lines are kept too.
                "diff_sha256": (hashlib.sha256(diff.encode()).hexdigest()
                                if diff else None),
                "status": status,
            }
        except Exception as exc:  # noqa: BLE001 - provenance must never abort a run
            provenance[name] = {"path": str(repo), "error": f"{type(exc).__name__}: {exc}"}
    return provenance


# ---------------------------------------------------------------------------
# stdout / stderr tee
# ---------------------------------------------------------------------------


class TeeStream:
    """Duplicates writes to multiple streams (e.g. the real stdout + a log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()


def thread_settings() -> dict:
    """Record HOW MANY THREADS this run computes with, for run_config.json (PITFALLS.md
    section 12, item 3).

    On CPU the thread count is part of the seed: BLAS reductions round differently per
    thread count, and PPO amplifies the last bit, so a recorded run is reproducible to the
    digit only at the thread count it ran with (checked 2026-09-12: the same Odd-Even CGF
    command matched a recorded run bit for bit at OMP_NUM_THREADS=8 and drifted after
    ~40k steps at 1, 2, 4 or 64). The three environment variables are recorded as the
    process saw them (None when unset); ``torch`` is the number PyTorch actually uses,
    which is what matters when none of them is set.
    """
    import torch  # local: keep importing this module cheap

    return {
        "omp": os.environ.get("OMP_NUM_THREADS"),
        "mkl": os.environ.get("MKL_NUM_THREADS"),
        "openblas": os.environ.get("OPENBLAS_NUM_THREADS"),
        "torch": int(torch.get_num_threads()),
    }


def tee_stdout_stderr(log_path: str) -> None:
    """Mirror stdout/stderr into log_path, in addition to the console.

    Lets a background (nohup) run's console output land in a log file that
    lives next to that same run's TensorBoard/model output (both under
    log_dir), instead of depending on the caller to redirect stdout by hand
    into a path that has to be matched back up to a run directory later.
    """
    log_file = open(log_path, "a", buffering=1)
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)


# ---------------------------------------------------------------------------
# run_status.json (written by the Odd-Even trainer since 2026-09-06; the shared trainer
# will write it for every domain)
# ---------------------------------------------------------------------------


RUN_STATUS_FILENAME = "run_status.json"


def write_run_status(model_save_path: str, *, completed: bool, error,
                     timesteps: int, total_timesteps: int) -> dict:
    """Write <model dir>/run_status.json and return its contents.

    "completed" means learn() returned. Anything else -- an exception, a
    KeyboardInterrupt, a SIGTERM-driven SystemExit -- is "failed", with the
    exception recorded, so no reader has to infer from stdout.log whether
    the sibling <encoder>_agent.zip is a result or a crash artefact.
    """
    status = {
        "status": "completed" if completed else "failed",
        "timesteps": int(timesteps),
        "total_timesteps": int(total_timesteps),
        "error": None if error is None else f"{type(error).__name__}: {error}",
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    model_dir = os.path.dirname(model_save_path)
    path = os.path.join(model_dir, RUN_STATUS_FILENAME) if model_dir else RUN_STATUS_FILENAME
    with open(path, "w") as handle:
        json.dump(status, handle, indent=2)
    return status


def resume_vecnormalize_path(resume_from: str) -> str:
    """The VecNormalize snapshot saved alongside a checkpoint zip.

    ``CheckpointCallback(save_vecnormalize=True)`` writes ``<prefix>_<N>_steps.zip`` next to
    ``<prefix>_vecnormalize_<N>_steps.pkl``; the final ``<encoder>_agent.zip`` sits next to
    ``vecnormalize.pkl``. Moved from ``experiments/ant_tag/4_train_rl_st.py``
    (``_default_resume_vecnormalize``, 2026-09-08) for the shared trainer (change 4,
    2026-09-12); generalised from ``st_agent.zip`` to any ``<encoder>_agent.zip``.
    """
    d, base = os.path.split(resume_from)
    m = re.fullmatch(r"(.+)_(\d+)_steps\.zip", base)
    if m:
        return os.path.join(d, f"{m.group(1)}_vecnormalize_{m.group(2)}_steps.pkl")
    if re.fullmatch(r"[A-Za-z0-9]+_agent\.zip", base):
        return os.path.join(d, "vecnormalize.pkl")
    raise ValueError(f"cannot derive the VecNormalize snapshot for {resume_from}; "
                     "pass --resume_vecnormalize")


def read_run_status(model_path: str) -> dict | None:
    """run_status.json for a saved agent, or None if the run predates it.

    Looks beside the model and one directory up, so best_model/best_model.zip
    and checkpoints/*.zip resolve to the run's status as well.
    """
    here = Path(model_path).resolve().parent
    for directory in (here, here.parent):
        candidate = directory / RUN_STATUS_FILENAME
        if candidate.exists():
            with open(candidate) as handle:
                return json.load(handle)
    return None


# ---------------------------------------------------------------------------
# Root-level output layout (plan section 2b). Not yet used by any script: the shared trainer
# (change 4) will call run_dir(); default_run_dir() above is what the scripts still use.
# ---------------------------------------------------------------------------


#: Environment variable that overrides where new runs write (resolution step 2 below).
OUTPUT_ROOT_ENV = "RL_BMDP_RUNS"


def checkout_root() -> Path:
    """The set_transformer checkout this package is imported from (parents[2] of this file:
    set_transformer/set_transformer/rl/run_records.py -> set_transformer/)."""
    return Path(__file__).resolve().parents[2]


def parent_repo(checkout: "Path | None" = None) -> "Path | None":
    """The repo that carries `checkout` as a git submodule, or None.

    Detected from the parent directory's `.gitmodules`: one of its `path = ...` entries must
    resolve to `checkout`. rl_for_beliefmdps lists set_transformer that way; a standalone
    clone (Brendan's) has no such parent and gets None.
    """
    checkout = checkout_root() if checkout is None else Path(checkout).resolve()
    gitmodules = checkout.parent / ".gitmodules"
    if not gitmodules.is_file():
        return None
    for line in gitmodules.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key.strip() == "path" and (checkout.parent / value.strip()).resolve() == checkout:
            return checkout.parent
    return None


def output_root(explicit: "str | os.PathLike | None" = None, *,
                environ: "Mapping[str, str] | None" = None,
                checkout: "Path | None" = None) -> Path:
    """The ONE folder new runs write under. Resolution order, first hit wins:

    1. `explicit` -- the --output_root flag;
    2. the RL_BMDP_RUNS environment variable;
    3. `<parent repo>/runs` when this checkout is a submodule of a parent repo
       (rl_for_beliefmdps/runs here);
    4. `<this checkout>/runs` (a standalone clone writes under itself).

    Never the current working directory: today's run dirs are cwd-relative, so a launch from
    another folder scatters a `runs/` there. `environ` and `checkout` exist for tests.
    """
    if explicit:
        return Path(explicit).expanduser().resolve()
    environ = os.environ if environ is None else environ
    if environ.get(OUTPUT_ROOT_ENV):
        return Path(environ[OUTPUT_ROOT_ENV]).expanduser().resolve()
    checkout = checkout_root() if checkout is None else Path(checkout).resolve()
    parent = parent_repo(checkout)
    return (parent if parent is not None else checkout) / "runs"


def run_leaf(seed: int, run_tag: "str | None" = None, timestamp: "str | None" = None) -> str:
    """`<timestamp>_seed<seed>[_<run_tag>]` -- the run-directory name every tool parses today,
    with the same tag sanitising as `default_run_dir`. `timestamp` exists for tests."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if timestamp is None else timestamp
    suffix = f"_{re.sub(r'[^A-Za-z0-9._-]', '_', run_tag)}" if run_tag else ""
    return f"{timestamp}_seed{seed}{suffix}"


def run_dir(domain: str, variant: str, encoder: str, seed: int, run_tag: "str | None" = None, *,
            kind: str = "rl", root: "str | os.PathLike | None" = None,
            timestamp: "str | None" = None) -> Path:
    """`<root>/<domain>/<variant>/<kind>/<encoder>/<timestamp>_seed<seed>[_<tag>]`.

    Variant first, because the experiment records (`domain_mds/<variant>_<domain>.md`) are
    per variant and that is how runs are looked up. `kind` is "rl" for training runs;
    pretraining and eval output go under "pretrain" and "eval" of the same variant. `root`
    defaults to `output_root()`. The leaf is unchanged from today's run dirs, so nothing that
    parses those names breaks.
    """
    root = output_root() if root is None else Path(root)
    return Path(root) / domain / variant / kind / encoder / run_leaf(seed, run_tag, timestamp)


def pretrain_dir(domain: str, variant: str, experiment_name: str, *,
                 root: "str | os.PathLike | None" = None) -> Path:
    """`<root>/<domain>/<variant>/pretrain/<experiment_name>`: where the pretraining scripts
    (`3_train_st.py`, `3_pretrain_st_belief.py`) put their run folders since change 5.2.
    The run folder inside keeps each script's own naming."""
    root = output_root() if root is None else Path(root)
    return Path(root) / domain / variant / "pretrain" / experiment_name


PRETRAIN_STATUS_FILENAME = RUN_STATUS_FILENAME


def write_pretrain_status(run_dir: "str | os.PathLike", *, completed: bool, error,
                          rl_checkpoint: "str | os.PathLike | None" = None,
                          checkpoints: "Mapping[str, object] | None" = None,
                          summary: "Mapping[str, object] | None" = None) -> dict:
    """Write <run_dir>/run_status.json for a PRETRAINING run (batch 7.5) and return it.

    Besides completed / failed it names the RL-loadable checkpoint (``rl_checkpoint``: what
    ``rl/train.py --pretrained_path`` takes -- the ST's checkpoint_best.pt, the reconstruction-
    pretrained CGF's exported checkpoint_best_cgf_arm.pt), so :func:`latest_pretrain_checkpoint`
    reads the answer instead of guessing a file name.
    """
    status = {
        "status": "completed" if completed else "failed",
        "error": None if error is None else f"{type(error).__name__}: {error}",
        "rl_checkpoint": None if rl_checkpoint is None else str(Path(rl_checkpoint).resolve()),
        "checkpoints": {k: str(Path(v).resolve()) for k, v in (checkpoints or {}).items()},
        "summary": dict(summary or {}),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = Path(run_dir) / PRETRAIN_STATUS_FILENAME
    with open(path, "w") as handle:
        json.dump(status, handle, indent=2, default=str)
    return status


#: Old run-folder shapes, recognised read-only by :func:`latest_pretrain_checkpoint`: the
#: reconstruction-pretrained CGF export, the Trainer's checkpoint, the Odd-Even script's.
_LEGACY_RL_CHECKPOINTS = ("checkpoints/checkpoint_best_cgf_arm.pt", "checkpoints/checkpoint_best.pt",
                          "checkpoint_best.pt")


def _sanitise_run_tag(run_tag: str) -> str:
    """The run-tag sanitising `run_leaf` / `default_run_dir` apply when they name a folder."""
    return re.sub(r'[^A-Za-z0-9._-]', '_', run_tag)


def _leaf_matches(name: str, seed: "int | None", run_tag: "str | None") -> bool:
    """Does the run-folder name `<timestamp>_seed<n>[_<run_tag>]` carry this seed and this run
    tag? Batch 8.0 (plan section 8): the sweep driver finds a cell's run by the exact
    `_seed<n>_<run_tag>` ending it chose itself, never by "newest". `None` = no constraint."""
    if run_tag is not None:
        tail = f"_{_sanitise_run_tag(run_tag)}"
        if not name.endswith(tail):
            return False
        name = name[: -len(tail)]
        if seed is None:
            return re.search(r"_seed\d+$", name) is not None
        return name.endswith(f"_seed{int(seed)}")
    if seed is not None:
        # `_seed<n>` followed by the end or by a tag; `_seed1` must not match `_seed10`.
        return re.search(rf"_seed{int(seed)}(?:_|$)", name) is not None
    return True


def run_folders(domain: str, variant: str, kind: str, encoder: str, *,
                seed: "int | None" = None, run_tag: "str | None" = None,
                objective: "str | None" = None,
                root: "str | os.PathLike | None" = None) -> "list[Path]":
    """The run folders of one cell, newest first, whatever their state (batch 10.1, 2026-09-14).

    ``<root>/<domain>/<variant>/<kind>/<encoder>[/<objective>]/<timestamp>_seed<n>[_<run_tag>]/``
    for ``kind`` ``"rl"`` (no objective) or ``"pretrain"`` (the objective names the folder), kept
    when the leaf carries ``seed`` and ``run_tag`` by the rule :func:`find_rl_run` and
    :func:`latest_pretrain_checkpoint` apply (``None`` = no constraint). Unlike those two this
    does NOT read status files: it is what a driver uses to tell "still being written" from
    "never started" (``run_config.json`` present, ``run_status.json`` absent, recent writes).
    Read-only.
    """
    root = output_root() if root is None else Path(root)
    folder = Path(root) / domain / variant / kind / encoder
    if kind == "pretrain":
        if objective is None:
            raise ValueError("run_folders: kind='pretrain' needs the objective (it names the folder)")
        folder = folder / objective
    if not folder.is_dir():
        return []
    return [d for d in sorted((d for d in folder.iterdir() if d.is_dir()), key=lambda d: d.name, reverse=True)
            if _leaf_matches(d.name, seed, run_tag)]


def latest_pretrain_checkpoint(domain: str, variant: str, encoder: str, objective: "str | None" = None, *,
                               root: "str | os.PathLike | None" = None,
                               experiment_name: "str | None" = None,
                               base_dir: "str | os.PathLike | None" = None,
                               seed: "int | None" = None,
                               run_tag: "str | None" = None) -> "Path | None":
    """The newest RL-loadable pretraining checkpoint for (domain, variant, encoder, objective),
    or None (batch 7.5; decision 4 of plan section 7).

    Default: the root layout, ``<root>/<domain>/<variant>/pretrain/<encoder>/<objective>/*/``,
    newest folder first (their names start with the timestamp), taking the first whose
    ``run_status.json`` says completed and names an existing ``rl_checkpoint``. With
    ``experiment_name`` (or ``base_dir`` + ``experiment_name``): an OLD experiment folder,
    ``<root>/<domain>/<variant>/pretrain/<experiment_name>/*/`` -- a run with a status file is
    read the same way, one without is recognised by its file names (``checkpoints/
    checkpoint_best_cgf_arm.pt``, ``checkpoints/checkpoint_best.pt``, ``checkpoint_best.pt``).
    Read-only: nothing is renamed or moved.

    ``seed`` / ``run_tag`` (batch 8.0): consider only run folders whose name ends in
    ``_seed<seed>`` / ``_seed<n>_<run_tag>``. Every arm pretrained with one objective shares the
    objective folder (all CGF arms under ``cgf/belief_kl/``), so a caller that wants ONE arm's run
    must say which; with both left at None the behaviour is unchanged.
    """
    if experiment_name is None:
        if objective is None:
            raise ValueError("latest_pretrain_checkpoint: the root layout needs the objective "
                             "(or pass experiment_name for an old experiment folder)")
        folder = pretrain_dir(domain, variant, encoder, root=root) / objective
    elif base_dir is not None:
        folder = Path(base_dir) / experiment_name
    else:
        folder = pretrain_dir(domain, variant, experiment_name, root=root)
    if not folder.is_dir():
        return None
    for run in sorted((d for d in folder.iterdir() if d.is_dir()), key=lambda d: d.name, reverse=True):
        if not _leaf_matches(run.name, seed, run_tag):
            continue
        status_path = run / PRETRAIN_STATUS_FILENAME
        if status_path.exists():
            with open(status_path) as handle:
                status = json.load(handle)
            if status.get("status") != "completed" or not status.get("rl_checkpoint"):
                continue
            candidate = Path(status["rl_checkpoint"])
            if candidate.exists():
                return candidate
            continue
        if experiment_name is not None:
            for rel in _LEGACY_RL_CHECKPOINTS:
                if (run / rel).exists():
                    return run / rel
    return None


def find_rl_run(domain: str, variant: str, encoder: str, seed: int, run_tag: str, *,
                root: "str | os.PathLike | None" = None) -> "Path | None":
    """The newest COMPLETED RL run folder for one sweep cell, or None (batch 8.0, plan section 8).

    Looks under ``<root>/<domain>/<variant>/rl/<encoder>/`` for folders named
    ``<timestamp>_seed<seed>_<run_tag>`` (the layout :func:`run_dir` writes) and returns the newest
    whose ``models/run_status.json`` says ``completed`` and whose ``models/<encoder>_agent.zip`` and
    ``models/vecnormalize.pkl`` exist. A folder whose status says ``failed``, has no status, or
    lacks either file is not a result and is skipped, so a driver re-runs that cell. Read-only.

    The VecNormalize file is required only when the run used VecNormalize: a run whose
    ``run_config.json`` records ``no_vec_normalize: true`` (``rl/train.py --no_vec_normalize``,
    the hunt ``fixed`` recipe since 2026-09-19) never writes one, and its zip alone is the result.
    Without that record (an older run, or no config file) the file is required as before.
    """
    root = output_root() if root is None else Path(root)
    folder = Path(root) / domain / variant / "rl" / encoder
    if not folder.is_dir():
        return None
    for run in sorted((d for d in folder.iterdir() if d.is_dir()), key=lambda d: d.name, reverse=True):
        if not _leaf_matches(run.name, seed, run_tag):
            continue
        models = run / "models"
        status_path = models / RUN_STATUS_FILENAME
        if not status_path.exists():
            continue
        with open(status_path) as handle:
            status = json.load(handle)
        if status.get("status") != "completed":
            continue
        if not (models / f"{encoder}_agent.zip").exists():
            continue
        if (models / "vecnormalize.pkl").exists() or not _run_used_vec_normalize(run):
            return run
    return None


def _run_used_vec_normalize(run: Path) -> bool:
    """False only when the run's ``run_config.json`` says ``no_vec_normalize`` is true."""
    config_path = run / "run_config.json"
    if not config_path.exists():
        return True
    try:
        with open(config_path) as handle:
            return not bool(json.load(handle).get("no_vec_normalize", False))
    except (OSError, ValueError):
        return True


def data_dir(domain: str, variant: str, *, root: "str | os.PathLike | None" = None) -> Path:
    """`<root>/<domain>/<variant>/data`: where the collectors put new datasets (plan section 7,
    decision 1, 2026-09-13). Recorded datasets stay in `experiments/<domain>/data/`."""
    root = output_root() if root is None else Path(root)
    return Path(root) / domain / variant / "data"


def dataset_path(domain: str, variant: str, *, tag: str = "",
                 root: "str | os.PathLike | None" = None) -> Path:
    """`<data_dir>/<variant>_pf_dataset[_<tag>].npz`: the collectors' default output file."""
    suffix = f"_{tag}" if tag else ""
    return data_dir(domain, variant, root=root) / f"{variant}_pf_dataset{suffix}.npz"


def eval_dir(domain: str, variant: str, *, root: "str | os.PathLike | None" = None) -> Path:
    """`<root>/<domain>/<variant>/eval`: the evaluation script's JSON summaries."""
    root = output_root() if root is None else Path(root)
    return Path(root) / domain / variant / "eval"


if __name__ == "__main__":
    # The wave drivers (bash) ask where runs go:  ROOT=$(python3 -m set_transformer.rl.run_records)
    print(output_root())
