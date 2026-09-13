"""Capacity probe on a TRAINED RL agent's feature extractor.

The question this answers, and why the other probe cannot:
`probe_odd_even_belief_separability.py` scores encodings at random init or
from a Sinkhorn autoencoder checkpoint. It says nothing about the extractor
that actually came out of a PPO run. When an arm scores at chance, two very
different things could be true --

    COLLAPSE      the features do not carry s* at all
    NO-USE        the features carry s* but the policy head never exploited it

-- and reward alone cannot tell them apart. PITFALLS.md #6: the probe measures
capacity, not use. So load the trained agent, freeze it, and ask whether a
linear readout of ITS features recovers the state.

Reported per arm:
    R^2(s*)        Ridge, GroupKFold by episode so no episode spans the split
    parity acc     balanced accuracy, chance = 0.500
    rel. spread    peak-to-peak / |mean| of the features -- is the signal at a
                   magnitude a policy head can use, or a 1e-4 ripple on an
                   O(1) offset (see the 2026-09-03 collapse finding)

Usage:
    python3 diagnostics/probe_trained_rl_features.py --variant oe50_short \
        --agent st:runs/.../st_agent.zip --agent cgf:runs/.../cgf_agent.zip
"""

from __future__ import annotations

import argparse
import importlib
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
_OE = _HERE.parent
_REPO = _OE.parents[2]
for _p in (str(_REPO), str(_REPO / "set_transformer"), str(_OE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# The registry and the belief env live in the package.
from set_transformer.rl.domains import odd_even as belief_env  # noqa: E402
variants = belief_env


def collect(agent_path: str, vecnorm: str | None, variant: str,
            n_episodes: int, seed: int):
    """Roll the frozen agent, recording (features, s*, episode id, step)."""
    v = variants.resolve(variant)
    cap = variants.episode_cap(variant)
    fn = belief_env.make_odd_even_belief_env(
        variant=variant, num_particles=v.n_dist_size, rank=0, seed=seed)
    env = DummyVecEnv([fn])
    if vecnorm and Path(vecnorm).exists():
        env = VecNormalize.load(vecnorm, env)
        env.training = False
        env.norm_reward = False

    # SB3 needs the arm's module importable to unpickle the extractor class when a zip
    # names the script (the Odd-Even zips name the package classes; harmless otherwise).
    for mod in ("4_train_rl_st", "4_train_rl_cgf", "4_train_rl_gaussian"):
        try:
            importlib.import_module(mod)
        except Exception:
            pass

    model = PPO.load(agent_path, env=env, device="cpu")
    extractor = model.policy.features_extractor
    extractor.eval()

    feats, labels, groups, steps = [], [], [], []
    for ep in range(n_episodes):
        env.seed(seed + ep)
        obs = env.reset()
        done, t = False, 0
        while not done and t < cap:
            with torch.no_grad():
                tens, _ = model.policy.obs_to_tensor(obs)
                f = extractor(tens).cpu().numpy().ravel()
            act, _ = model.predict(obs, deterministic=True)
            obs, _r, d, infos = env.step(act)
            ts = infos[0].get("true_state")
            if ts is not None:
                feats.append(f)
                labels.append(int(ts))
                groups.append(ep)
                steps.append(t)
            done = bool(d[0])
            t += 1
    env.close()
    return (np.asarray(feats), np.asarray(labels),
            np.asarray(groups), np.asarray(steps))


def probe(F, y, g, n_splits=4):
    """Ridge R^2 of s*, grouped by episode."""
    gkf = GroupKFold(n_splits=n_splits)
    preds = np.zeros_like(y, dtype=float)
    for tr, te in gkf.split(F, y, g):
        sc = StandardScaler().fit(F[tr])
        m = Ridge(alpha=1.0).fit(sc.transform(F[tr]), y[tr])
        preds[te] = m.predict(sc.transform(F[te]))
    return r2_score(y, preds)


def probe_parity(F, y, g, n_splits=4):
    """Balanced accuracy of s* parity, grouped by episode. Chance = 0.500."""
    par = (y % 2).astype(int)
    gkf = GroupKFold(n_splits=n_splits)
    preds = np.zeros_like(par)
    for tr, te in gkf.split(F, par, g):
        sc = StandardScaler().fit(F[tr])
        m = LogisticRegression(max_iter=2000).fit(sc.transform(F[tr]), par[tr])
        preds[te] = m.predict(sc.transform(F[te]))
    return balanced_accuracy_score(par, preds)


def relative_spread(F):
    """peak-to-peak / |mean|, the magnitude axis of the two-axis gate."""
    mu = np.abs(F.mean())
    return float(np.ptp(F) / mu) if mu > 0 else float("inf")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="oe50_short")
    ap.add_argument("--agent", action="append", required=True,
                    help="name:path/to/agent.zip (repeatable)")
    ap.add_argument("--n_episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--n_splits", type=int, default=4)
    a = ap.parse_args(argv)

    print(f"variant {a.variant} | {a.n_episodes} episodes | seed {a.seed}")
    print(f"{'arm':<12} {'R2(s*)':>9} {'parity':>8} {'rel.spread':>11} "
          f"{'feat_dim':>9} {'n':>7}")
    for spec in a.agent:
        name, path = spec.split(":", 1)
        vn = str(Path(path).with_name("vecnormalize.pkl"))
        try:
            F, y, g, _s = collect(path, vn, a.variant, a.n_episodes, a.seed)
            print(f"{name:<12} {probe(F, y, g, a.n_splits):>9.4f} "
                  f"{probe_parity(F, y, g, a.n_splits):>8.4f} "
                  f"{relative_spread(F):>11.3e} {F.shape[1]:>9} {len(y):>7}")
        except Exception as exc:
            print(f"{name:<12} FAILED {type(exc).__name__}: {exc}")
    print("\nchance: parity 0.500 | R^2 0.0 (predicting the mean)")
    print("collapse reading: high R^2 with rel.spread <~ 5e-3 means the signal "
          "is present but too small for a policy head -- capacity without use.")


if __name__ == "__main__":
    main()
