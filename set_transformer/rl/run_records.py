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


def eval_dir(domain: str, variant: str, *, root: "str | os.PathLike | None" = None) -> Path:
    """`<root>/<domain>/<variant>/eval`: the evaluation script's JSON summaries."""
    root = output_root() if root is None else Path(root)
    return Path(root) / domain / variant / "eval"


if __name__ == "__main__":
    # The wave drivers (bash) ask where runs go:  ROOT=$(python3 -m set_transformer.rl.run_records)
    print(output_root())
