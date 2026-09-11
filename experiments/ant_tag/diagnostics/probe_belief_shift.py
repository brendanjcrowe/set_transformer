"""Is the pretrained ST encoder in distribution on the beliefs the RL policy actually produces?

Rolls out a saved PPO agent in the belief env (real visible radius, true
sparse reward, deterministic), keeps every particle set + weight vector the
policy saw, and scores the PRETRAINING autoencoder on them with the exact
pretraining objective (weighted Sinkhorn, blur / scaling / frame from the
checkpoint). The same score on a sample of the pretraining dataset is the
reference. Beliefs are binned by weighted spread with the collector's own
classifier (collapsed < 0.5 <= intermediate < 2.5 <= diffuse, env units), so
the report says both WHAT the policy sees (spread mix vs the dataset's) and
HOW WELL the encoder handles each class.

    python3 diagnostics/probe_belief_shift.py --variant smart_mid_slow_v15 \
        --pretrained_ckpt experiments/st_pretrain_<v>_plain/<stamp>/checkpoints/checkpoint_best.pt \
        --dataset data/smart_mid_slow_v15_pf_dataset.npz \
        --runs runs/ant_tag_st_smart_mid_slow_v15/<run> [...] --n_episodes 30 --out_json <path>

Per run the latest models/checkpoints/ant_tag_st_<N>_steps.zip and its
VecNormalize snapshot are used (or --which final|best). CPU by default; the
encoder is small and the rollouts are the cost.
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ANT_TAG_DIR = Path(__file__).resolve().parents[1]
for p in (str(_REPO_ROOT), str(_ANT_TAG_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import gymnasium as gym  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

import variants  # noqa: E402

_cgf = importlib.import_module("4_train_rl_cgf")
importlib.import_module("4_train_rl_st")          # extractor class for unpickling
_collect = importlib.import_module("2_collect_pf_dataset")

CLASSES = ("collapsed", "intermediate", "diffuse")


def _make_belief_env(env_id, particle_filter_class, num_particles, seed):
    def _init():
        env = gym.make(env_id, rendering=False)
        env.reset(seed=seed)
        pf_kwargs = _cgf.get_ant_tag_pf_kwargs(env)
        env = _cgf.CurriculumVisibilityWrapper(
            env, initial_visibility_radius=float(env.unwrapped.visible_radius))
        env = _cgf.PFDictWithWeightsObservationWrapper(
            env=env, particle_filter_class=particle_filter_class,
            particle_filter_kwargs=pf_kwargs, num_particles=num_particles,
            pf_interaction_mapper=_cgf.ant_tag_pf_interaction_mapper,
            obs_mask_indices=[-2, -1])
        return _cgf._CurriculumRouter(env)
    return _init


def pick_checkpoint(run_dir: str, which: str) -> tuple[str, str, int | None]:
    models = os.path.join(run_dir, "models")
    if which == "final":
        return os.path.join(models, "st_agent.zip"), os.path.join(models, "vecnormalize.pkl"), None
    if which == "best":
        return (os.path.join(models, "best_model", "best_model.zip"),
                os.path.join(models, "vecnormalize.pkl"), None)
    zips = glob.glob(os.path.join(models, "checkpoints", "ant_tag_st_*_steps.zip"))
    if not zips:
        raise FileNotFoundError(f"no checkpoints under {models}")
    step_of = lambda p: int(re.search(r"_(\d+)_steps\.zip$", p).group(1))
    z = max(zips, key=step_of)
    n = step_of(z)
    vn = os.path.join(models, "checkpoints", f"ant_tag_st_vecnormalize_{n}_steps.pkl")
    return z, vn, n


def rollout_beliefs(run_dir, variant, cap, num_particles, n_episodes, seed, which):
    zip_path, vn_path, step = pick_checkpoint(run_dir, which)
    env = DummyVecEnv([_make_belief_env(variant.env_id, variant.particle_filter, num_particles, seed)])
    if os.path.exists(vn_path):
        env = VecNormalize.load(vn_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print(f"  WARNING: no VecNormalize at {vn_path}; obs unnormalized")
    model = PPO.load(zip_path, env=env, device="cpu")
    env.seed(seed)
    raw = (env.venv if isinstance(env, VecNormalize) else env).envs[0].unwrapped
    parts, wts, ants, tgts, lengths = [], [], [], [], []
    for _ in range(n_episodes):
        obs = env.reset()
        done, steps = False, 0
        while not done:
            parts.append(np.asarray(obs["particles"])[0].copy())
            wts.append(np.asarray(obs["weights"])[0].copy())
            # true simulator state at the moment of this belief (for readout probes)
            ants.append(np.asarray(raw.data.qpos[:2], dtype=np.float32).copy())
            tgts.append(np.asarray(raw.get_target_pos(), dtype=np.float32).copy())
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(action)
            steps += 1
            done = bool(dones[0])
        lengths.append(steps)
    tags = sum(int(l < cap) for l in lengths)   # ended before the cap = tagged
    return (np.stack(parts).astype(np.float32), np.stack(wts).astype(np.float32),
            dict(agent=zip_path, step=step, episodes=n_episodes, tags=tags,
                 tag_rate=tags / n_episodes, mean_len=float(np.mean(lengths)),
                 ant_xy=np.stack(ants), target_xy=np.stack(tgts)))


def load_autoencoder(ckpt_path: str):
    from set_transformer.models.pf_set_transformer import PFSetTransformer
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    weighted = bool(getattr(cfg, "weighted_particles", False))
    model = PFSetTransformer(
        num_particles=cfg.num_particles,
        dim_particles=cfg.dim_particles + (1 if weighted else 0),
        num_encodings=cfg.num_encodings, dim_encoder=cfg.dim_encoder,
        num_inds=cfg.num_inds, dim_hidden=cfg.dim_hidden, num_heads=cfg.num_heads,
        ln=cfg.use_layer_norm, dim_output_particles=cfg.dim_particles)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    frame = dict(scale=float(ck.get("particle_scale", 1.0)), centre=float(ck.get("particle_centre", 0.0)),
                 blur=float(cfg.sinkhorn_blur), scaling=float(cfg.sinkhorn_scaling), weighted=weighted,
                 val_loss=float(ck.get("best_val_loss", float("nan"))), epoch=int(ck.get("best_epoch", -1)))
    return model, frame


def model_input(particles_norm: torch.Tensor, weights: torch.Tensor, weighted: bool) -> torch.Tensor:
    """Trainer._model_input, verbatim: mass channel = clean weights x N."""
    if not weighted:
        return particles_norm
    clean = torch.clamp(torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0), min=0.0)
    clean = clean / (clean.sum(dim=-1, keepdim=True) + 1e-8)
    return torch.cat([particles_norm, (clean * clean.shape[-1]).unsqueeze(-1)], dim=-1)


@torch.no_grad()
def per_sample_loss(model, frame, particles: np.ndarray, weights: np.ndarray, batch=256) -> np.ndarray:
    from set_transformer.loss import SinkhornLoss
    loss_fn = SinkhornLoss(p=2, blur=frame["blur"], scaling=frame["scaling"], reduction="none")
    out = []
    for i in range(0, len(particles), batch):
        x = torch.from_numpy((particles[i:i + batch] - frame["centre"]) / frame["scale"]).float()
        w = torch.from_numpy(weights[i:i + batch]).float()
        recon = model(model_input(x, w, frame["weighted"]))
        if isinstance(recon, dict):
            recon = recon["recon"]
        l = loss_fn(recon, x, None, w) if frame["weighted"] else loss_fn(recon, x)
        out.append(l.detach().cpu().numpy().reshape(-1))
    return np.concatenate(out)


def classify(particles: np.ndarray, weights: np.ndarray, lo=0.5, hi=2.5) -> np.ndarray:
    s = _collect._weighted_spread(particles, weights)
    return np.where(s < lo, 0, np.where(s < hi, 1, 2)), s


def summarize(losses: np.ndarray, cls: np.ndarray, spread: np.ndarray) -> dict:
    n = len(losses)
    rec = dict(n=int(n), loss_mean=float(losses.mean()), loss_median=float(np.median(losses)),
               loss_p90=float(np.percentile(losses, 90)), spread_median=float(np.median(spread)))
    for k, name in enumerate(CLASSES):
        m = cls == k
        rec[f"{name}_frac"] = float(m.mean())
        rec[f"{name}_loss"] = float(losses[m].mean()) if m.any() else None
    return rec


def fmt(rec: dict, ref: dict | None = None) -> str:
    mix = " / ".join(f"{100 * rec[f'{c}_frac']:.0f}%" for c in CLASSES)
    per = " / ".join("-" if rec[f"{c}_loss"] is None else f"{rec[f'{c}_loss']:.4f}" for c in CLASSES)
    s = (f"n={rec['n']:6d}  mix C/I/D {mix:22s}  loss mean {rec['loss_mean']:.4f} "
         f"(median {rec['loss_median']:.4f}, p90 {rec['loss_p90']:.4f})  per class C/I/D {per}")
    if ref is not None:
        s += f"  x{rec['loss_mean'] / ref['loss_mean']:.2f} vs dataset"
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    variants.add_variant_argument(ap, default="smart_mid_slow_v15")
    ap.add_argument("--pretrained_ckpt", required=True)
    ap.add_argument("--dataset", required=True, help="the .npz the encoder was pretrained on")
    ap.add_argument("--dataset_samples", type=int, default=4000)
    ap.add_argument("--runs", nargs="*", default=[], help="run dirs to roll out")
    ap.add_argument("--which", choices=["latest", "final", "best"], default="latest")
    ap.add_argument("--n_episodes", type=int, default=30)
    ap.add_argument("--num_particles", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--save_beliefs_dir", default=None,
                    help="also dump each run's rolled-out beliefs as <dir>/<run>.npz")
    args = ap.parse_args()
    if args.list_variants:
        variants.print_variants(); return
    torch.set_num_threads(4)
    variant = variants.resolve(args.variant)
    cap = variants.episode_cap(args.variant)

    model, frame = load_autoencoder(args.pretrained_ckpt)
    print(f"encoder {args.pretrained_ckpt}\n  frame scale {frame['scale']} centre {frame['centre']}, "
          f"blur {frame['blur']} scaling {frame['scaling']}, weighted {frame['weighted']}, "
          f"recorded best val loss {frame['val_loss']:.5f} (epoch {frame['epoch']})")

    data = np.load(args.dataset, allow_pickle=True)
    P, W = data["particles"], data["weights"]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(P), size=min(args.dataset_samples, len(P)), replace=False)
    d_cls, d_spread = classify(P[idx], W[idx])
    d_loss = per_sample_loss(model, frame, P[idx], W[idx])
    ref = summarize(d_loss, d_cls, d_spread)
    print(f"DATASET ({os.path.basename(args.dataset)}, {len(idx)} random rows): {fmt(ref)}")
    results = dict(encoder=args.pretrained_ckpt, frame=frame, dataset=ref, runs={})

    for run in args.runs:
        name = os.path.basename(run.rstrip("/"))
        parts, wts, meta = rollout_beliefs(run, variant, cap, args.num_particles, args.n_episodes, args.seed, args.which)
        cls, spread = classify(parts, wts)
        losses = per_sample_loss(model, frame, parts, wts)
        rec = summarize(losses, cls, spread)
        ant_xy, target_xy = meta.pop("ant_xy"), meta.pop("target_xy")
        rec.update(meta)
        results["runs"][name] = rec
        print(f"RUN {name} @ {meta['step']} steps, rollout tag rate {meta['tag_rate']:.2f} "
              f"(mean len {meta['mean_len']:.0f}):\n  {fmt(rec, ref)}")
        if args.save_beliefs_dir:
            os.makedirs(args.save_beliefs_dir, exist_ok=True)
            np.savez_compressed(os.path.join(args.save_beliefs_dir, name + ".npz"),
                                particles=parts, weights=wts, losses=losses, spread=spread,
                                ant_xy=ant_xy, target_xy=target_xy)
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=1)
        print("wrote", args.out_json)


if __name__ == "__main__":
    main()
