"""Supervised pretraining of the Set Transformer belief encoder on Odd-Even:
teach the 64-dim latent to carry the EXACT POSTERIOR, then probe whether the
frozen latent identifies the posterior mode (target B) and the true state
(target A).

WHY THIS SCRIPT EXISTS (domain_mds/oddeven.md, 2026-09-04/05). The mode
readout probe found that every encoder the pipeline had produced is a
reparameterisation of the posterior MEAN:

    encoder                         eff. rank   mode acc (MLP raw, tr / st)
    exact posterior (ceiling)          42.3          0.997 / 0.998
    Gaussian mean+var (floor)           1.00         0.580 / 0.735
    CGF, any t range, K or K'           1.0-1.5      0.50 - 0.65 / 0.53 - 0.69
    ST, Sinkhorn-pretrained             1.00         0.536 / 0.630
    ST, end-to-end under PPO            collapsed    chance

The two pretrained STs are rank 1 because their Sinkhorn decoders collapsed
to a point mass at the weighted mean, so reconstruction never asked the
encoder for anything but the mean. The end-to-end ST collapsed to features
with std 1e-6 under a sparse 0/1 reward. Neither says the architecture cannot
represent a 50-atom belief -- a Set Transformer is a universal approximator of
permutation-invariant functions and the target here is a 50-vector on a fixed
support. Both say the TRAINING SIGNAL never asked for it.

So this script asks for it directly. The encoder is the RL arm's own
`SetTransformerFeaturesExtractor` (same geometry, same weight channel, same
normalisation), a linear head maps its 64 features to 50 logits, and the loss
is the cross-entropy between those logits and the exact posterior from
`info['belief']` (= KL(posterior || model) up to a constant). If the latent
then probes at effective rank >> 1 and mode accuracy near the exact
posterior's, the encoder FAMILY can do what CGF provably cannot on this
domain, and the checkpoint is the one to hand to `4_train_rl_st.py
--pretrained_st_model_path` (it is saved in the format that loader reads,
geometry `config` included).

OBJECTIVES (--objective):
    belief_kl   soft cross-entropy against the exact posterior. DEFAULT.
                Asks for the whole belief, which subsumes both targets.
    mode_ce     hard cross-entropy against the posterior argmax (target B).
    state_ce    hard cross-entropy against the true state (target A). The
                Bayes-optimal predictor of s* IS the posterior mode, so this
                is a noisier version of mode_ce.

PROTOCOL. Data is rolled from the raw env with `seed + episode`, one snapshot
per decision, BEFORE that step's observation (the same timing as the readout
probe and the RL arms: the reset observation is step 1's snapshot). Particles
are the states centred on 25.5, weights are the exact posterior -- identical
to what the exact-support filter hands the RL arm (weights equal the env
posterior to 1e-15, oddeven.md Gap 5). Train and validation episodes come
from disjoint seed ranges. The probe at the end reuses
`diagnostics/probe_cgf_mode_readout.py`'s rollout, classifiers, split and
geometry code on ITS default seed (9000), so the rows join that table.

ENCODERS (--encoder, 2026-09-05). ``st`` is the Set Transformer above.
``cgf`` fits the CGF arm under the SAME objective, data and head: t (unless
--t_frozen), the running feature norm's statistics and a readout MLP sized
with --match_params to the ST's parameter count. This is the size- and
supervision-matched pretraining test the RL comparison needs: the frozen ST
result (0.877 steady, oddeven.md) came from an encoder with privileged
posterior supervision, so the CGF arm gets the same supervision here, and
`4_train_rl_cgf.py --pretrained_cgf_model_path ... --cgf_frozen` then runs
it frozen under PPO exactly as the ST arm is run. Six configurations:
{fixed, learned t} x {K, K_grad, both}.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 3_pretrain_st_belief.py --variant oe50_short \
        --n_train_episodes 4000 --epochs 40 --run_tag belief_kl_v1

    python3 3_pretrain_st_belief.py --encoder cgf --feature_mode K_grad \
        --match_params 109448 --epochs 200 --run_tag kgrad_learnedt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent              # experiments/odd_even
_ST_ROOT = _HERE.parents[1]                          # set_transformer submodule root
_REPO_ROOT = _HERE.parents[2]                        # repo root
for _p in (str(_REPO_ROOT), str(_HERE), str(_ST_ROOT)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import pdomains  # noqa: F401,E402 - registers the envs
import _sibling  # noqa: E402
variants = _sibling.load("variants")
from set_transformer.rl.feature_extractors.st import (  # noqa: E402
    SetTransformerFeaturesExtractor,
)
from set_transformer.rl.feature_extractors.cgf import (  # noqa: E402
    WeightedCGFFeaturesExtractor,
)

# The CGF arm's readout / pretrained flags and its geometry resolution
# (--match_params sizing) come from the RL script, so a checkpoint is built
# and later loaded with one spelling of every flag.
_train_rl_cgf = _sibling.load("4_train_rl_cgf")


def _load_probe_module():
    """diagnostics/probe_cgf_mode_readout.py, by path (Gap 12 convention)."""
    path = _HERE / "diagnostics" / "probe_cgf_mode_readout.py"
    spec = importlib.util.spec_from_file_location(
        "_odd_even_experiments.diagnostics.probe_cgf_mode_readout", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def collect(variant: str, n_episodes: int, seed: int) -> dict:
    """One snapshot per decision: exact posterior, true state, mode, step."""
    resolved = variants.resolve(variant)
    cap = variants.episode_cap(variant)
    env = gym.make(resolved.env_id)
    beliefs, true_states, modes, steps, groups = [], [], [], [], []
    for episode in range(n_episodes):
        _obs, info = env.reset(seed=seed + episode)
        for step in range(1, cap + 1):
            beliefs.append(np.asarray(info["belief"], dtype=np.float32))
            true_states.append(int(info["true_state"]))
            modes.append(int(info["optimal_prediction"]))
            steps.append(step)
            groups.append(episode)
            _obs, _r, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                break
    env.close()
    return {
        "weights": np.stack(beliefs),
        "true_state": np.asarray(true_states, dtype=np.int64),
        "mode": np.asarray(modes, dtype=np.int64),
        "step": np.asarray(steps, dtype=np.int64),
        "group": np.asarray(groups, dtype=np.int64),
        "n": int(resolved.n_dist_size),
        "cap": cap,
    }


class BeliefBatches:
    """Tensors on the device; particles are the centred states, shared."""

    def __init__(self, data: dict, centre: float, device: torch.device):
        n = data["n"]
        self.n_rows = len(data["weights"])
        self.weights = torch.from_numpy(data["weights"]).to(device)
        self.true_state = torch.from_numpy(data["true_state"] - 1).to(device)
        self.mode = torch.from_numpy(data["mode"] - 1).to(device)
        self.step = data["step"]
        self.group = data["group"]
        states = torch.arange(1, n + 1, dtype=torch.float32) - float(centre)
        self.particles = states.reshape(1, n, 1).to(device)      # [1, N, 1]
        self.device = device

    def obs(self, index: torch.Tensor) -> dict:
        b = len(index)
        return {
            "obs": torch.zeros(b, 1, device=self.device),
            "particles": self.particles.expand(b, -1, -1),
            "weights": self.weights[index],
        }


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def build_extractor(args, space, scale: float, pretrained_path: str | None = None):
    """The encoder under test, as the RL arm's own SB3 extractor class."""
    if args.encoder == "st":
        return SetTransformerFeaturesExtractor(
            space, num_encodings=args.num_encodings, dim_encoder=args.dim_encoder,
            num_inds=args.num_inds, dim_hidden=args.dim_hidden, num_heads=args.num_heads,
            ln=not args.no_layer_norm, arena_scale=scale, weight_channel=True,
            num_post_sab=args.num_post_sab,
            pretrained_st_model_path=pretrained_path)
    return WeightedCGFFeaturesExtractor(
        space, num_cgf_features=args.num_cgf_features, arena_scale=scale,
        t_init_mode=args.t_init_mode, t_init_scale=args.t_init_scale,
        t_clamp=args.t_clamp, t_frozen=args.t_frozen, t_param=args.t_param,
        t_bound=args.t_bound if args.t_param == "tanh" else None,
        t_init_max=args.t_init_max, feature_mode=args.feature_mode,
        feature_norm=args.feature_norm, readout_hidden=args.readout_hidden,
        readout_depth=args.readout_depth, readout_dim=args.readout_dim,
        x_embed_dim=args.x_embed_dim, x_embed_hidden=args.x_embed_hidden,
        x_embed_depth=args.x_embed_depth,
        pretrained_cgf_model_path=pretrained_path)


def extractor_geometry(extractor) -> dict:
    return dict(extractor._st_geometry if hasattr(extractor, "_st_geometry")
                else extractor._cgf_geometry)


class BeliefEncoderWithHead(nn.Module):
    def __init__(self, extractor: nn.Module, n_states: int):
        super().__init__()
        self.extractor = extractor
        st_dim = extractor.features_dim - 1                        # drop the obs passthrough
        self.head = nn.Linear(st_dim, n_states)

    def features(self, obs: dict) -> torch.Tensor:
        return self.extractor(obs)[:, 1:]

    def forward(self, obs: dict) -> torch.Tensor:
        return self.head(self.features(obs))


def loss_fn(logits: torch.Tensor, batches: BeliefBatches, index: torch.Tensor,
            objective: str) -> torch.Tensor:
    if objective == "belief_kl":
        target = batches.weights[index]
        return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    if objective == "mode_ce":
        return F.cross_entropy(logits, batches.mode[index])
    if objective == "state_ce":
        return F.cross_entropy(logits, batches.true_state[index])
    raise ValueError(objective)


@torch.no_grad()
def evaluate(model, batches: BeliefBatches, objective: str, batch_size: int,
             transient_max_step: int) -> dict:
    model.eval()
    losses, preds, feats = [], [], []
    for start in range(0, batches.n_rows, batch_size):
        index = torch.arange(start, min(start + batch_size, batches.n_rows),
                             device=batches.device)
        obs = batches.obs(index)
        f = model.features(obs)
        logits = model.head(f)
        losses.append(loss_fn(logits, batches, index, objective).item() * len(index))
        preds.append(logits.argmax(dim=-1))
        feats.append(f)
    preds = torch.cat(preds)
    feats = torch.cat(feats).double()
    transient = torch.from_numpy(batches.step <= transient_max_step).to(batches.device)

    def _split(hit):
        return (float(hit[transient].float().mean()),
                float(hit[~transient].float().mean()))

    mode_tr, mode_st = _split(preds == batches.mode)
    state_tr, state_st = _split(preds == batches.true_state)
    centred = feats - feats.mean(dim=0)
    singular = torch.linalg.svdvals(centred)
    spectrum = singular ** 2 / (singular ** 2).sum().clamp_min(1e-300)
    eff_rank = float(torch.exp(-(spectrum * torch.log(spectrum + 1e-300)).sum()))
    return {
        "loss": sum(losses) / batches.n_rows,
        "head_mode_acc": {"transient": mode_tr, "steady": mode_st},
        "head_true_state_acc": {"transient": state_tr, "steady": state_st},
        "feature_abs_std": float(feats.std(dim=0).mean()),
        "feature_eff_rank": eff_rank,
    }


def save_checkpoint(model: BeliefEncoderWithHead, path: Path, args, epoch: int,
                    val: dict, geometry: dict) -> None:
    """The format the RL arm's extractor loads: a dict with `model_state_dict`
    and a `config` carrying the geometry fields the loader checks.

    ST: encoder keys under `set_transformer.` (what SetTransformerFeaturesExtractor
    strips). CGF: the extractor's whole state_dict, unprefixed -- t, the norm
    statistics and the readout ARE the encoder."""
    if args.encoder == "st":
        encoder_state = {f"set_transformer.{k}": v.detach().cpu()
                         for k, v in model.extractor.encoder.state_dict().items()}
    else:
        encoder_state = {k: v.detach().cpu()
                         for k, v in model.extractor.state_dict().items()}
    torch.save({
        "model_state_dict": encoder_state,
        "head_state_dict": {k: v.detach().cpu() for k, v in model.head.state_dict().items()},
        "config": {**geometry, "objective": args.objective, "variant": args.variant,
                   "arena_scale": float(model.extractor.arena_scale),
                   "encoder": args.encoder,
                   "encoder_params": int(getattr(args, "encoder_params", 0)),
                   "pretraining": "3_pretrain_st_belief.py"},
        "epoch": epoch,
        "val": val,
        "args": vars(args),
    }, path)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    variants.add_variant_argument(parser, default="oe50_short")
    parser.add_argument("--encoder", default="st", choices=["st", "cgf"],
                        help="Which arm's extractor to pretrain. cgf takes the "
                             "CGF flags below (same spelling as 4_train_rl_cgf.py).")
    parser.add_argument("--objective", default="belief_kl",
                        choices=["belief_kl", "mode_ce", "state_ce"])
    parser.add_argument("--n_train_episodes", type=int, default=4000)
    parser.add_argument("--n_val_episodes", type=int, default=400)
    parser.add_argument("--data_seed", type=int, default=100000,
                        help="Train episodes use data_seed + e; validation "
                             "episodes data_seed + 10_000_000 + e. Both are far "
                             "from the probe's 9000 + e.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_encodings", type=int, default=8)
    parser.add_argument("--dim_encoder", type=int, default=8)
    parser.add_argument("--num_inds", type=int, default=16,
                        help="Default is the ClusterHunt-sized encoder (16 inducing "
                             "points, dim_hidden 64, ~109k params): on oe50_short it "
                             "reaches the same KL~0 / oracle-level readouts as the "
                             "128-wide, 32-point one at a quarter of the size "
                             "(oddeven.md, 2026-09-05). Pass 32 / 128 to reproduce "
                             "the v2_long and mode_ce checkpoints.")
    parser.add_argument("--dim_hidden", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_post_sab", type=int, default=2,
                        help="SAB blocks between the PMA and the output Linear. "
                             "0 is the ClusterHunt/LeastMass head (PMA -> Linear).")
    parser.add_argument("--no_layer_norm", action="store_true")
    # ---- CGF arm (ignored for --encoder st) ----
    parser.add_argument("--num_cgf_features", type=int, default=64)
    parser.add_argument("--t_init_mode", default="spread_1d",
                        choices=["spread_1d", "linspace_all_dims", "linspace_first_dim", "random"])
    parser.add_argument("--t_init_scale", type=float, default=0.1)
    parser.add_argument("--t_param", default="tanh", choices=["clamp", "tanh"])
    parser.add_argument("--t_bound", type=float, default=50.0)
    parser.add_argument("--t_init_max", type=float, default=None,
                        help="Default 40 in tanh mode, t_clamp in clamp mode "
                             "(4_train_rl_cgf.resolve_t_init_max).")
    parser.add_argument("--t_clamp", type=float, default=2.0)
    parser.add_argument("--t_frozen", action="store_true",
                        help="Fixed-t arm: t is a buffer, only the readout learns.")
    parser.add_argument("--feature_mode", default="K", choices=["K", "K_grad", "both"])
    parser.add_argument("--feature_norm", default="running", choices=["none", "running", "layernorm"],
                        help="running (default): per-feature z-score with running "
                             "statistics, used for fixed AND learned t. layernorm is "
                             "the ablation (normalises across features per sample).")
    _train_rl_cgf.add_readout_and_pretrained_arguments(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--probe_episodes", type=int, default=300)
    parser.add_argument("--probe_seed", type=int, default=9000,
                        help="probe_cgf_mode_readout.py's default, so rows join its table")
    parser.add_argument("--probe_splits", type=int, default=5)
    parser.add_argument("--skip_probe", action="store_true")
    parser.add_argument("--out_dir", default=None,
                        help="Default: <output root>/odd_even/<variant>/pretrain/"
                             "<encoder>_belief_pretrain/ (change 5.2; see --output_root). "
                             "Before 2026-09-12: experiments/<encoder>_belief_pretrain/ beside "
                             "this script, where the recorded checkpoints still are.")
    parser.add_argument("--output_root", default=None,
                        help="Root of the shared run layout when --out_dir is not given: "
                             "$RL_BMDP_RUNS, else <parent repo>/runs when this checkout is a "
                             "submodule, else <checkout>/runs.")
    parser.add_argument("--run_tag", default="")
    parser.add_argument("--init_from", default=None,
                        help="ST only: warm-start the encoder from a 3_train_st.py (Sinkhorn) "
                             "checkpoint instead of random init. Geometry must match the flags.")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Train only the linear head; the encoder (random or --init_from) "
                             "is held fixed. The linear-readout anchor for --init_from.")
    args = parser.parse_args(argv)
    if args.list_variants:
        variants.print_variants()
        return None
    if args.out_dir is None:
        from set_transformer.rl import run_records
        args.out_dir = str(run_records.pretrain_dir(
            "odd_even", args.variant, f"{args.encoder}_belief_pretrain", root=args.output_root))
    if args.encoder == "cgf":
        if args.pretrained_cgf_model_path or args.cgf_frozen:
            parser.error("--pretrained_cgf_model_path / --cgf_frozen are RL-side flags; "
                         "this script PRODUCES the checkpoint.")
        _train_rl_cgf.resolve_cgf_geometry(args, parser)
        _train_rl_cgf.resolve_t_init_max(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    resolved = variants.resolve(args.variant)
    centre, scale = variants.state_centre(args.variant), variants.state_scale(args.variant)
    n_states = resolved.n_dist_size
    transient_max_step = 21

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{stamp}_{args.objective}_seed{args.seed}"
    if args.encoder != "st":
        run_name = f"{stamp}_{args.encoder}_{args.objective}_seed{args.seed}"
    if args.run_tag:
        run_name += f"_{args.run_tag}"
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"run dir: {run_dir}")
    print(f"variant {args.variant} | env {resolved.env_id} | n={n_states} | "
          f"cap {variants.episode_cap(args.variant)} | normalisation (s - {centre}) / {scale} | "
          f"device {device}")

    # ---- data ----------------------------------------------------------------
    t0 = time.time()
    train = collect(args.variant, args.n_train_episodes, args.data_seed)
    val = collect(args.variant, args.n_val_episodes, args.data_seed + 10_000_000)
    print(f"collected {len(train['weights'])} train rows / {len(val['weights'])} val rows "
          f"in {time.time() - t0:.0f}s; train distinct s*: "
          f"{len(set(train['true_state'].tolist()))}")
    train_b = BeliefBatches(train, centre, device)
    val_b = BeliefBatches(val, centre, device)

    # ---- model ---------------------------------------------------------------
    space = gym.spaces.Dict({
        "obs": gym.spaces.Box(0.0, 1.0, (1,), np.float32),
        "particles": gym.spaces.Box(-np.inf, np.inf, (n_states, 1), np.float32),
        "weights": gym.spaces.Box(0.0, 1.0, (n_states,), np.float32),
    })
    if args.init_from and args.encoder != "st":
        raise SystemExit("--init_from is implemented for --encoder st only")
    extractor = build_extractor(args, space, scale, args.init_from)
    if args.init_from:
        # Verify the warm start landed (PITFALLS.md section 1 habit): compare every
        # encoder tensor against the checkpoint.
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        ref = {k[len("set_transformer."):]: v for k, v in ck["model_state_dict"].items()
               if k.startswith("set_transformer.")}
        cur = extractor.encoder.state_dict()
        delta = max(float((ref[k] - cur[k].cpu()).abs().max()) for k in ref)
        print(f"Verified: encoder warm-started from {args.init_from} "
              f"({len(ref)} tensors, max|delta| = {delta})")
        if delta != 0.0:
            raise SystemExit("warm start did not land exactly")
    if args.freeze_encoder:
        for p in extractor.parameters():
            p.requires_grad_(False)
        extractor.eval()
    geometry = extractor_geometry(extractor)
    model = BeliefEncoderWithHead(extractor, n_states).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_head = model.head.weight.numel() + n_states
    n_encoder_buffers = sum(b.numel() for b in extractor.buffers())
    print(f"encoder+head parameters: {n_params:,}  (head {n_head:,}; encoder "
          f"{n_params - n_head:,} trainable + {n_encoder_buffers:,} buffer values)")
    if args.encoder == "st":
        args.encoder_params = int(n_params - n_head)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in trainable):,}"
          + ("  (encoder FROZEN, head only)" if args.freeze_encoder else ""))
    optimizer = torch.optim.AdamW(trainable, lr=args.lr,
                                  weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(train_b.n_rows / args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * steps_per_epoch)

    # Reference: the entropy of the exact posterior is the floor of the
    # belief_kl loss (CE = H(posterior) + KL). Print it so the loss is readable.
    if args.objective == "belief_kl":
        w = val_b.weights.clamp_min(1e-30)
        entropy = float(-(val_b.weights * w.log()).sum(dim=-1).mean())
        print(f"val posterior entropy (loss floor for belief_kl): {entropy:.4f}")

    # ---- train ---------------------------------------------------------------
    history, best_val = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_encoder:
            model.extractor.eval()
        perm = torch.randperm(train_b.n_rows, device=device)
        running, t_epoch = 0.0, time.time()
        for start in range(0, train_b.n_rows, args.batch_size):
            index = perm[start:start + args.batch_size]
            logits = model(train_b.obs(index))
            loss = loss_fn(logits, train_b, index, args.objective)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item() * len(index)
        metrics = evaluate(model, val_b, args.objective, 2048, transient_max_step)
        metrics.update(epoch=epoch, train_loss=running / train_b.n_rows,
                       lr=scheduler.get_last_lr()[0], seconds=time.time() - t_epoch)
        history.append(metrics)
        flag = ""
        if metrics["loss"] < best_val:
            best_val = metrics["loss"]
            save_checkpoint(model, run_dir / "checkpoint_best.pt", args, epoch, metrics, geometry)
            flag = "  *best*"
        print(f"epoch {epoch:3d} | train {metrics['train_loss']:.4f} | val {metrics['loss']:.4f} | "
              f"head mode acc tr/st {metrics['head_mode_acc']['transient']:.3f}/"
              f"{metrics['head_mode_acc']['steady']:.3f} | "
              f"s* acc {metrics['head_true_state_acc']['transient']:.3f}/"
              f"{metrics['head_true_state_acc']['steady']:.3f} | "
              f"feat std {metrics['feature_abs_std']:.2e} | eff.rank {metrics['feature_eff_rank']:.1f} | "
              f"{metrics['seconds']:.0f}s{flag}", flush=True)
        (run_dir / "history.json").write_text(json.dumps(history, indent=1))
    save_checkpoint(model, run_dir / "checkpoint_last.pt", args, args.epochs, history[-1], geometry)
    print(f"best val loss {best_val:.4f}; checkpoints in {run_dir}")

    if args.skip_probe:
        return run_dir

    # ---- probe the FROZEN latent the way the readout probe does ----------------
    print("\nprobing the frozen best-checkpoint latent with the mode-readout protocol ...")
    probe = _load_probe_module()
    best = torch.load(run_dir / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    probe_extractor = build_extractor(args, space, scale,
                                      pretrained_path=str(run_dir / "checkpoint_best.pt"))
    probe_extractor.eval()
    head = nn.Linear(probe_extractor.features_dim - 1, n_states)
    head.load_state_dict(best["head_state_dict"])
    enc_label = f"{args.encoder.upper()}_BELIEF"

    data = probe.collect_rollouts(args.variant, args.probe_episodes, args.probe_seed)
    st_feats = probe._run_extractor(probe_extractor, data)                 # [R, 64] float64
    with torch.no_grad():
        head_pred = head(torch.from_numpy(st_feats).float()).argmax(dim=-1).numpy() + 1

    weights = data["weights"].astype(np.float64)
    weights /= weights.sum(axis=1, keepdims=True)
    x = data["particles"].astype(np.float64)[:, :, 0] / scale
    mean = (weights * x).sum(axis=1)
    var = (weights * (x - mean[:, None]) ** 2).sum(axis=1)
    feature_sets = {"EXACT": weights, "GAUSS2": np.c_[mean, var], enc_label: st_feats}

    results = {"geometry": {}, "head": {}, "targets": {}}
    print(f"\n{'encoding':<12} {'width':>6} {'abs.std':>10} {'eff.rank':>9} {'|r(PC1,mean)|':>14}")
    for name, feats in feature_sets.items():
        g = probe.geometry(feats, mean)
        results["geometry"][name] = g
        print(f"{name:<12} {feats.shape[1]:>6} {g['abs_std']:>10.2e} {g['eff_rank']:>9.2f} "
              f"{g['pc1_corr_mean']:>14.5f}")

    for target_label, truth in (("B_posterior_mode", data["mode"]),
                                ("A_true_state", data["true_state"])):
        acc = probe._split_accuracy(head_pred, truth, data["step"])
        results["head"][target_label] = acc
        print(f"\ntrained linear head on {enc_label} -> {target_label}: "
              f"tr {acc['transient']:.3f} / st {acc['steady']:.3f}")

    for target_label, truth in (("B_posterior_mode", data["mode"]),
                                ("A_true_state", data["true_state"])):
        results["targets"][target_label] = {}
        print(f"\n=== target {target_label}  (50-way, chance 0.020; GroupKFold {args.probe_splits}) ===")
        print(f"{'encoding':<12} {'logreg raw':>16} {'logreg z':>16} {'mlp raw':>16} {'mlp z':>16}")
        for name, feats in feature_sets.items():
            row, cells = {}, []
            for classifier in ("logreg", "mlp"):
                for standardise in (False, True):
                    preds = probe._fit_predict(feats, truth, data["group"], args.probe_splits,
                                               classifier, standardise, args.seed)
                    acc = probe._split_accuracy(preds, truth, data["step"])
                    row[f"{classifier}_{'z' if standardise else 'raw'}"] = acc
                    cells.append(f"{acc['transient']:.3f} / {acc['steady']:.3f}")
            results["targets"][target_label][name] = row
            print(f"{name:<12} " + " ".join(f"{c:>16}" for c in cells), flush=True)

    (run_dir / "probe_results.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {run_dir / 'probe_results.json'}")
    return run_dir


if __name__ == "__main__":
    main()
