"""Supervised pretraining of an encoder on Odd-Even's EXACT POSTERIOR (pipeline step 3,
the Odd-Even path): teach the latent to carry the belief, then probe whether the frozen
latent identifies the posterior mode (target B) and the true state (target A).

Since batch 7.3 of the harness centralisation (2026-09-13) this file is an ENTRY POINT of the
shared pretraining command, ``set_transformer/rl/pretrain.py``, with the domain fixed to
``odd_even``::

    python3 -m set_transformer.rl.pretrain --domain odd_even --encoder st \\
        --objective belief_kl --variant oe50_short --num_epochs 40 ...

is the same run. The objectives, the data roller, the model, the loop, the checkpoint format
and the end-of-run probe live in ``set_transformer/rl/domains/odd_even.py`` (block "Supervised
pretraining on the exact posterior"); the encoder's flags come from the shared encoder table
(``rl/encoders.py``), so a checkpoint is built and later loaded by ``4_train_rl_{st,cgf}.py``
with one spelling of every flag.

What this file keeps: every flag this script ever took still works. Its historical spellings
are translated before parsing (``LEGACY_FLAGS`` / ``translate`` below):

    --epochs N        ->  --num_epochs N
    --lr X            ->  --learning_rate X
    --out_dir DIR     ->  --base_dir <DIR's parent> --experiment_name <DIR's name>
                          (so the run folder is DIR/<stamp>_<objective>_seed<n>, as before)

and ``--variant`` defaults to ``oe50_short``, ``--encoder`` to ``st``, ``--objective`` to
``belief_kl``, as they did. The RL-side flags ``--pretrained_cgf_model_path`` / ``--cgf_frozen``
are refused with the message the script always gave: this script PRODUCES the checkpoint.

OBJECTIVES (--objective):
    belief_kl   soft cross-entropy against the exact posterior. DEFAULT.
    mode_ce     hard cross-entropy against the posterior argmax (target B).
    state_ce    hard cross-entropy against the true state (target A).

WHY THIS EXISTS (domain_mds/oddeven.md, 2026-09-04/05): every encoder the pipeline had
produced was a reparameterisation of the posterior MEAN, because the training signal never
asked for the belief. This asks for it directly, with the RL arm's own extractor and a linear
head; the checkpoint is what ``4_train_rl_st.py --pretrained_st_model_path`` (or the cgf arm's
``--pretrained_cgf_model_path``) loads, geometry ``config`` included. The end-of-run probe
reuses the mode-readout protocol on its default seed (9000), so the rows join that table.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 3_pretrain_st_belief.py --variant oe50_short \\
        --n_train_episodes 4000 --epochs 40 --run_tag belief_kl_v1

    python3 3_pretrain_st_belief.py --encoder cgf --feature_mode K_grad \\
        --match_params 109448 --epochs 200 --run_tag kgrad_learnedt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent              # experiments/odd_even
_ST_ROOT = _HERE.parents[1]                          # set_transformer submodule root
_REPO_ROOT = _HERE.parents[2]                        # repo root
for _p in (str(_REPO_ROOT), str(_HERE), str(_ST_ROOT)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from set_transformer.rl import encoders as _encoders  # noqa: E402
from set_transformer.rl import pretrain as _pretrain  # noqa: E402
from set_transformer.rl.domains import get as _get_domain  # noqa: E402
from set_transformer.rl.domains.odd_even import (  # noqa: E402,F401 - the moved pieces, re-exported
    BeliefBatches,
    BeliefEncoderWithHead,
    belief_loss,
    build_extractor,
    collect_posterior_snapshots,
    evaluate_belief_head,
    extractor_geometry,
    save_belief_checkpoint,
)

PROG = "3_pretrain_st_belief.py"
#: This script's historical spelling -> the package command's.
LEGACY_FLAGS = {"--epochs": "--num_epochs", "--lr": "--learning_rate"}
#: What this script used when the flag was omitted.
LEGACY_DEFAULTS = {"--encoder": "st", "--variant": "oe50_short"}
#: Refused here, as the script always refused them: the pretraining side produces the checkpoint.
RL_SIDE_FLAGS = ("--pretrained_cgf_model_path", "--cgf_frozen")

LEGACY_HELP = """\
This script's historical spellings are translated before parsing:
  --epochs N      = --num_epochs N
  --lr X          = --learning_rate X
  --out_dir DIR   = --base_dir <DIR's parent> --experiment_name <DIR's name>
                    (so runs land in DIR/<stamp>_[<encoder>_]<objective>_seed<n>[_<tag>])
--variant defaults to oe50_short, --encoder to st, --objective to belief_kl; without --out_dir the
run lands in <root>/odd_even/<variant>/pretrain/<encoder>_belief_pretrain/ as it always did (the
package command's own default is pretrain/<encoder>/<objective>/<stamp>_seed<n>/checkpoints/).
--pretrained_cgf_model_path / --cgf_frozen are RL-side flags and are refused: this script
PRODUCES the checkpoint.
"""
CGF_HELP = """\
The CGF arm's flags (--num_cgf_features, --t_param, --t_bound, --t_init_mode, --t_frozen,
--feature_mode, --feature_norm, --match_params, --readout_*, --x_embed_*): --help --encoder cgf
"""


def _value(argv: list[str], flag: str):
    """The value given for ``flag`` on the command line, or None."""
    for i, token in enumerate(argv):
        if token == flag and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


def _given(argv: list[str], flag: str) -> bool:
    return any(token == flag or token.startswith(flag + "=") for token in argv)


def translate(argv: list[str]) -> list[str]:
    """``argv`` in the package command's spelling, with this script's defaults filled in."""
    errors = argparse.ArgumentParser(prog=PROG)
    out: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        name, eq, value = token.partition("=")
        if name in RL_SIDE_FLAGS:
            errors.error("--pretrained_cgf_model_path / --cgf_frozen are RL-side flags; "
                         "this script PRODUCES the checkpoint.")
        if name == "--out_dir":
            if not eq:
                if i + 1 >= len(argv):
                    errors.error("--out_dir expects a folder")
                value = argv[i + 1]
                i += 1
            out_dir = Path(value)
            out += ["--base_dir", str(out_dir.parent), "--experiment_name", out_dir.name]
        elif name in LEGACY_FLAGS:
            out.append(LEGACY_FLAGS[name] + (eq + value if eq else ""))
        else:
            out.append(token)
        i += 1
    for flag, value in LEGACY_DEFAULTS.items():
        if not _given(out, flag):
            out += [flag, value]
    if not any(_given(out, flag) for flag in ("--experiment_name", "--base_dir")):
        # 7.5: the package command files runs under pretrain/<encoder>/<objective>/; this
        # script keeps its historical folder pretrain/<encoder>_belief_pretrain/<stamp>_...
        out += ["--experiment_name", f"{_value(out, '--encoder')}_belief_pretrain"]
    return out


def _print_help(argv: list[str]) -> None:
    """The package command's help for the chosen encoder / objective, plus this script's
    translations as the epilogue."""
    domain = _get_domain("odd_even")
    encoder = _encoders.get(_value(argv, "--encoder") or LEGACY_DEFAULTS["--encoder"])
    objectives = _pretrain.objectives_for(domain)
    name = _value(argv, "--objective") or _pretrain.default_objective_name(domain)
    objective = objectives.get(name) or objectives[_pretrain.default_objective_name(domain)]
    parser = _pretrain.build_parser(domain, encoder, objective, prog=PROG)
    parser.formatter_class = argparse.RawDescriptionHelpFormatter   # keep the block's line breaks
    parser.epilog = LEGACY_HELP + (CGF_HELP if encoder.name != "cgf" else "")
    parser.print_help()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "-h" in argv or "--help" in argv:
        _print_help(argv)
        return None
    return _pretrain.main(translate(argv), domain="odd_even", prog=PROG)


if __name__ == "__main__":
    main()
